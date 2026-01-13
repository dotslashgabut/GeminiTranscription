import { GoogleGenAI, Type, Modality } from "@google/genai";
import { TranscriptionSegment } from "../types";

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || '' });

const SEGMENT_PROPERTIES_CORE = {
  startTime: {
    type: Type.STRING,
    description: "Absolute timestamp in MM:SS.mmm format (e.g. '01:05.300').",
  },
  endTime: {
    type: Type.STRING,
    description: "Absolute timestamp in MM:SS.mmm format.",
  },
  text: {
    type: Type.STRING,
    description: "Transcribed text. Exact words spoken. No hallucinations. Must include every single word.",
  },
};

const LINE_MODE_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    segments: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: SEGMENT_PROPERTIES_CORE,
        required: ["startTime", "endTime", "text"],
      },
    },
  },
  required: ["segments"],
};

const WORD_MODE_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    segments: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          ...SEGMENT_PROPERTIES_CORE,
          words: {
            type: Type.ARRAY,
            description: "Individual words.",
            items: {
              type: Type.OBJECT,
              properties: {
                text: { 
                  type: Type.STRING, 
                  description: "The single word, or character (for Chinese/Japanese/Korean), or linguistic unit." 
                },
                startTime: { 
                    type: Type.STRING, 
                    description: "Start time of the word in MM:SS.mmm format" 
                },
                endTime: { 
                    type: Type.STRING, 
                    description: "End time of the word in MM:SS.mmm format" 
                }
              },
              required: ["text", "startTime", "endTime"]
            }
          }
        },
        required: ["startTime", "endTime", "text", "words"],
      },
    },
  },
  required: ["segments"],
};

/**
 * Robustly parses various timestamp formats into total seconds.
 * Kept for App.tsx compatibility.
 */
export function timestampToSeconds(ts: string | number): number {
  if (ts === undefined || ts === null) return 0;
  let str = String(ts).trim();
  str = str.replace(/[０-９]/g, s => String.fromCharCode(s.charCodeAt(0) - 0xFEE0));
  str = str.replace(/：/g, ':');
  const clean = str.replace(/,/g, '.').replace(/[^\d:.]/g, '');
  
  if (!clean.includes(':')) return parseFloat(clean) || 0;

  const parts = clean.split(':').map(p => parseFloat(p) || 0);
  if (parts.length === 3) return (parts[0] * 3600) + (parts[1] * 60) + parts[2];
  if (parts.length === 2) return (parts[0] * 60) + parts[1];
  return parseFloat(clean) || 0;
}

/**
 * Robustly normalizes timestamp strings to HH:MM:SS.mmm
 * Logic derived from samples/geminiService.ts
 */
function normalizeTimestamp(ts: string): string {
  if (!ts) return "00:00.000";

  // Replace comma with dot for decimals, remove non-time chars
  let clean = ts.trim().replace(/,/g, '.').replace(/[^\d:.]/g, '');
  let totalSeconds = 0;

  // Handle if model returns raw seconds (e.g. "65.5") despite instructions
  if (!clean.includes(':') && /^[\d.]+$/.test(clean)) {
    totalSeconds = parseFloat(clean);
  } else {
    // Handle MM:SS.mmm or HH:MM:SS.mmm
    const parts = clean.split(':');

    if (parts.length === 3) {
      const h = parseInt(parts[0], 10) || 0;
      const m = parseInt(parts[1], 10) || 0;
      const secParts = parts[2].split('.');
      const s = parseInt(secParts[0], 10) || 0;
      let ms = 0;
      if (secParts[1]) ms = parseFloat("0." + secParts[1]);
      totalSeconds = h * 3600 + m * 60 + s + ms;

    } else if (parts.length === 2) {
      const m = parseInt(parts[0], 10) || 0;
      const secParts = parts[1].split('.');
      const s = parseInt(secParts[0], 10) || 0;
      let ms = 0;
      if (secParts[1]) ms = parseFloat("0." + secParts[1]);
      totalSeconds = m * 60 + s + ms;
    }
  }

  if (isNaN(totalSeconds) || totalSeconds < 0) return "00:00.000";

  const m = Math.floor(totalSeconds / 60);
  const s = Math.floor(totalSeconds % 60);
  const ms = Math.round((totalSeconds % 1) * 1000);

  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(ms).padStart(3, '0')}`;
}

/**
 * Attempts to repair truncated JSON strings commonly returned by LLMs.
 * Logic derived from samples/geminiService.ts
 */
function tryRepairJson(jsonString: string): any {
  const trimmed = jsonString.trim();

  // 1. Try simple parse
  try {
    const parsed = JSON.parse(trimmed);
    if (parsed.segments && Array.isArray(parsed.segments)) return parsed;
    if (Array.isArray(parsed)) return { segments: parsed };
  } catch (e) { /* continue */ }

  // 2. Try to fix unescaped quotes (common in "text" fields)
  // Look for: "text": "Something "quoted" here" -> "text": "Something \"quoted\" here"
  // This is a naive heuristic but helps with common Gemini 2.5 errors
  let fixedQuotes = trimmed.replace(/"text"\s*:\s*"((?:[^"\\]|\\.)*)"/g, (match, content) => {
    return match; 
  });

  // 3. Attempt to close truncated JSON
  const lastObjectEnd = trimmed.lastIndexOf('}');
  if (lastObjectEnd !== -1) {
    const sets = ["]}", "}", "]}"];
    for (const suffix of sets) {
      try {
        const repaired = trimmed + suffix;
        const parsed = JSON.parse(repaired);
        if (parsed.segments) return parsed;
      } catch (e) { }
    }
    try {
      const repaired = trimmed.substring(0, lastObjectEnd + 1) + "]}";
      const parsed = JSON.parse(repaired);
      if (parsed.segments) return parsed;
    } catch (e) { }
  }

  // 4. Fallback: Aggressive Regex Scraping
  // We extract anything that looks like a segment object.
  // This works for both flat and nested structures, though it flattens hierarchy.
  const segments: any[] = [];
  
  // Regex to capture: "startTime": "...", "endTime": "...", "text": "..."
  // Handles potential variation in whitespace and "start"/"end" vs "startTime"/"endTime"
  // Also handles single quotes for keys/values if model hallucinates JSON5
  const segmentRegex = /(?:["']start(?:Time)?["'])\s*:\s*["']([^"']+)["']\s*,\s*(?:["']end(?:Time)?["'])\s*:\s*["']([^"']+)["']\s*,\s*(?:["']text["'])\s*:\s*(?:["']((?:[^"'\\]|\\.)*)["'])/g;

  let match;
  // Use the original trimmed string for scraping to avoid corruption from partial fixes
  while ((match = segmentRegex.exec(trimmed)) !== null) {
    const rawText = match[3];
    let unescapedText = rawText;
    try {
      // Try to unescape using JSON.parse
      unescapedText = JSON.parse(`"${rawText.replace(/"/g, '\\"')}"`);
    } catch (e) {
      // Manual fallback unescape
      unescapedText = rawText
        .replace(/\\"/g, '"')
        .replace(/\\'/g, "'")
        .replace(/\\\\/g, "\\")
        .replace(/\\n/g, "\n");
    }

    segments.push({
      startTime: match[1],
      endTime: match[2],
      text: unescapedText
    });
  }

  if (segments.length > 0) {
    // If we scraped segments, we return them.
    // Note: In Word mode, this might return both the Line container and the Words.
    // This is acceptable as a fallback for a broken response.
    return { segments };
  }

  throw new Error("Response structure invalid and could not be repaired.");
}

export async function transcribeAudio(
  modelName: string,
  audioBase64: string,
  mimeType: string,
  signal?: AbortSignal,
  granularity: 'line' | 'word' = 'line',
  languageHint: string = ""
): Promise<TranscriptionSegment[]> {
  try {
    const isGemini3 = modelName.includes('gemini-3');
    const selectedSchema = granularity === 'word' ? WORD_MODE_SCHEMA : LINE_MODE_SCHEMA;

    const timingPolicy = `
    TIMING RULES (CRITICAL):
    1. FORMAT: strictly **MM:SS.mmm** (e.g., 01:23.450).
    2. CONTINUITY: Timestamps must be strictly chronological.
    3. ACCURACY: Sync text exactly to the audio.
    `;

    let segmentationPolicy = "";
    if (granularity === 'word') {
      segmentationPolicy = `
    SEGMENTATION: HIERARCHICAL WORD-LEVEL (TTML/KARAOKE)
    ---------------------------------------------------
    CRITICAL: You are generating data for rich TTML export.
    
    1. STRUCTURE: Group words into natural lines/phrases (this is the parent object).
    2. DETAILS: Inside each line object, you MUST provide a "words" array.
    3. WORDS: The "words" array must contain EVERY single word from that line with its own precise start/end time.
    4. CJK HANDLING: For Chinese, Japanese, or Korean scripts, treat each character (or logical block of characters) as a separate "word" for the purposes of karaoke timing.
      `;
    } else {
      segmentationPolicy = `
    SEGMENTATION: LINE-LEVEL (SUBTITLE/LRC MODE)
    ---------------------------------------------------
    CRITICAL: You are generating subtitles for a movie/music video.

    1. PHRASES: Group words into complete sentences or musical phrases.
    2. CLARITY: Do not break a sentence in the middle unless there is a pause.
    3. REPETITIONS: Separate repetitive vocalizations (e.g. "Oh oh oh") from the main lyrics into their own lines.
    4. LENGTH: Keep segments between 2 and 6 seconds for readability.
    5. WORDS ARRAY: You may omit the "words" array in this mode to save tokens.
      `;
    }

    const hints = languageHint 
      ? `IMPORTANT CONTEXT / LANGUAGE HINT: The audio likely contains: "${languageHint}". Use this to bias detection.` 
      : "";

    const systemInstruction = `
    You are an expert Audio Transcription AI specialized in generating timed lyrics.
    TASK: Transcribe the audio file into JSON segments.
    MODE: ${granularity.toUpperCase()} LEVEL.
    
    ${timingPolicy}
    ${segmentationPolicy}
    ${hints}

    LANGUAGE HANDLING (CRITICAL):
    1. RAPID CODE-SWITCHING: Audio often contains multiple languages mixed within the SAME sentence.
    2. MULTI-LINGUAL EQUALITY: The languages might NOT include English (e.g. Indonesian mixed with Japanese, Chinese mixed with Japanese). Treat all detected languages as equally probable.
    3. WORD-LEVEL DETECTION: Detect the language of every individual word.
    4. NATIVE SCRIPT STRICTNESS: Write EACH word in its native script.
       - Example: "Aku cinta kamu" (Indonesian) -> Latin.
       - Example: "愛してる" (Japanese) -> Kanji/Kana.
    5. MIXED SCRIPT PRESERVATION (IMPORTANT):
       - If specific English/Latin words are spoken amidst Japanese/Chinese/etc., KEEP them in LATIN script.
       - DO NOT transliterate English words into Katakana (e.g. if audio says "I love you", write "I love you", NOT "アイラブユー").
       - Maintain mixed text (Kanji/Kana + Latin) exactly as spoken.
    6. PROHIBITIONS:
       - DO NOT translate.
       - DO NOT romanize (unless explicitly spelled out).
       - DO NOT force English if it is not spoken.
    
    GENERAL RULES:
    - Verbatim: Transcribe exactly what is heard. Include fillers (um, ah).
    - Completeness: Transcribe from 00:00 to the very end.
    - JSON Only: Output pure JSON.
    `;

    const requestConfig: any = {
      responseMimeType: "application/json",
      responseSchema: selectedSchema,
      temperature: 0.1,
      systemInstruction: systemInstruction,
    };

    if (isGemini3) {
      requestConfig.thinkingConfig = { thinkingBudget: 1024 }; 
    }

    const abortPromise = new Promise<never>((_, reject) => {
      if (signal?.aborted) reject(new DOMException("Aborted", "AbortError"));
      signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
    });

    const response: any = await Promise.race([
      ai.models.generateContent({
        model: modelName,
        contents: [
          {
            parts: [
              { inlineData: { data: audioBase64, mimeType: mimeType } },
              { text: "Transcribe audio." },
            ],
          },
        ],
        config: requestConfig,
      }),
      abortPromise
    ]);

    let text = response.text || "";
    // Robust cleaning from sample
    text = text.trim();
    if (text.startsWith('```json')) {
      text = text.replace(/^```json\s*/, '').replace(/\s*```$/, '');
    } else if (text.startsWith('```')) {
      text = text.replace(/^```\s*/, '').replace(/\s*```$/, '');
    }
    // Remove JS comments if any
    text = text.replace(/\/\/.*$/gm, '');

    if (!text) throw new Error("Empty response");

    const parsed = tryRepairJson(text);
    let rawSegments = parsed.segments || [];

    // FLATTENING LOGIC: If Word mode, extract 'words' array to top level
    if (granularity === 'word') {
      const flattened: any[] = [];
      rawSegments.forEach((s: any) => {
        if (Array.isArray(s.words) && s.words.length > 0) {
           s.words.forEach((w: any) => flattened.push(w));
        } else {
           // Fallback: Use segment itself if no words found
           flattened.push(s);
        }
      });
      if (flattened.length > 0) rawSegments = flattened;
    }

    // Map segments ensuring normalized timestamps
    return rawSegments.map((s: any) => ({
      startTime: normalizeTimestamp(String(s.startTime)),
      endTime: normalizeTimestamp(String(s.endTime)),
      text: String(s.text),
      words: s.words ? s.words.map((w: any) => ({
        text: String(w.text),
        startTime: normalizeTimestamp(String(w.startTime)),
        endTime: normalizeTimestamp(String(w.endTime))
      })) : undefined
    }));

  } catch (error: any) {
    if (error.name === 'AbortError') throw error;
    console.error(`Transcription error (${modelName}):`, error);
    throw new Error(error.message || "Transcription failed");
  }
}

export async function translateSegments(
  segments: TranscriptionSegment[],
  targetLanguage: string
): Promise<TranscriptionSegment[]> {
  try {
    const schema = {
      type: Type.OBJECT,
      properties: {
        segments: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              startTime: { type: Type.STRING },
              endTime: { type: Type.STRING },
              text: { type: Type.STRING },
              translatedText: { type: Type.STRING },
            },
            required: ["startTime", "endTime", "text", "translatedText"],
          },
        },
      },
    };

    const systemInstruction = `
    You are a professional translator. 
    Translate the "text" field of the provided JSON segments into ${targetLanguage}.
    Populate the "translatedText" field.
    Keep "startTime", "endTime", and "text" EXACTLY unchanged.
    Output ONLY valid JSON.
    `;

    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: [{ parts: [{ text: JSON.stringify(segments) }] }],
      config: {
        responseMimeType: "application/json",
        responseSchema: schema,
        systemInstruction: systemInstruction,
        thinkingConfig: { thinkingBudget: 0 }, // Disable thinking for translation speed
      },
    });

    let text = response.text || "";
    text = text.replace(/```json|```/g, "").trim();
    
    // Repair potential JSON issues
    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch(e) {
      console.warn("Simple JSON parse failed for translation, trying repair", e);
      parsed = tryRepairJson(text);
    }
    
    return parsed.segments || [];
  } catch (error: any) {
    console.error("Translation error:", error);
    throw error;
  }
}

export async function generateSpeech(text: string, language: string = "English"): Promise<string | undefined> {
  try {
    if (!text || text.trim().length === 0) return undefined;

    const systemInstruction = `
    You are a native speaker of ${language}. 
    Read the following text clearly and naturally in ${language}.
    Even if the text is a single word, pronounce it fully.
    `;

    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash-preview-tts",
      contents: [{ parts: [{ text: text }] }],
      config: {
        responseModalities: [Modality.AUDIO],
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: { voiceName: 'Zephyr' },
          },
        },
        systemInstruction: systemInstruction,
      },
    });

    return response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
  } catch (error) {
    console.error("TTS error:", error);
    throw error;
  }
}

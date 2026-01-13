import { GoogleGenAI, Type, Modality } from "@google/genai";
import { TranscriptionSegment } from "../types";

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || '' });

const SEGMENT_PROPERTIES_CORE = {
  startTime: {
    type: Type.STRING,
    description: "Timestamp in 'MM:SS.mmm' format (e.g. '00:41.520').",
  },
  endTime: {
    type: Type.STRING,
    description: "Timestamp in 'MM:SS.mmm' format.",
  },
  text: {
    type: Type.STRING,
    description: "Transcribed text. VERBATIM. MANDATORY: Include all fillers (umm, uh, ah) and sounds.",
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
                startTime: { type: Type.STRING },
                endTime: { type: Type.STRING },
                text: { type: Type.STRING }
              },
              required: ["startTime", "endTime", "text"]
            }
          }
        },
        required: ["startTime", "endTime", "text"],
      },
    },
  },
  required: ["segments"],
};

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

function secondsToTimestamp(totalSeconds: number): string {
  const roundedMs = Math.round(totalSeconds * 1000);
  const m = Math.floor(roundedMs / 60000);
  const s = Math.floor((roundedMs % 60000) / 1000);
  const ms = roundedMs % 1000;
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(ms).padStart(3, '0')}`;
}

function postProcessSegments(segments: any[], granularity: 'line' | 'word'): TranscriptionSegment[] {
  if (!segments || segments.length === 0) return [];

  const processed: TranscriptionSegment[] = [];
  let lastStartTime = -1;
  let lastEndTime = 0;

  // Granularity settings
  // Word mode needs very tight gaps to prevent "karaoke lag"
  const MAX_GAP = granularity === 'word' ? 5.0 : 15.0; 
  const MAX_DURATION = granularity === 'word' ? 4.0 : 60.0;

  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i];
    let start = timestampToSeconds(seg.startTime);
    let end = timestampToSeconds(seg.endTime);

    // Allow timestamps to start whenever they actually start.
    // If the audio has 10s of silence, start at 10s.

    // Fix backward jumps (Model hallucination or reset)
    if (start < lastEndTime - 0.2) {
      start = lastEndTime;
    }

    if (start < lastStartTime) start = lastStartTime;
    if (end <= start) end = start + (granularity === 'word' ? 0.2 : 1.0);
    if ((end - start) > MAX_DURATION) end = start + (granularity === 'word' ? 0.8 : 5.0);

    lastStartTime = start;
    lastEndTime = end;

    processed.push({
      startTime: secondsToTimestamp(start),
      endTime: secondsToTimestamp(end),
      text: String(seg.text).trim()
    });
  }

  return processed;
}

function tryRepairJson(jsonString: string): any {
  const trimmed = jsonString.trim();
  try {
    const parsed = JSON.parse(trimmed);
    if (parsed.segments && Array.isArray(parsed.segments)) return parsed;
    if (Array.isArray(parsed)) return { segments: parsed };
  } catch (e) { /* ignore */ }

  // Try to find the last valid closing brace
  const lastObjectEnd = trimmed.lastIndexOf('}');
  if (lastObjectEnd !== -1) {
    try {
      // Attempt to close an open array
      const parsed = JSON.parse(trimmed.substring(0, lastObjectEnd + 1) + "]}");
      if (parsed.segments) return parsed;
    } catch (e) { /* ignore */ }
  }

  // Regex fallback
  const segments = [];
  const regex = /"startTime":\s*"([^"]+)",\s*"endTime":\s*"([^"]+)",\s*"text":\s*"((?:[^"\\]|\\.)*)"/g;
  let match;
  while ((match = regex.exec(trimmed)) !== null) {
    segments.push({ startTime: match[1], endTime: match[2], text: match[3] });
  }
  return { segments };
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

    // --- SYSTEM POLICIES ---
    const timingPolicy = `
    TIMING RULES (CRITICAL):
    1. PRECISION: The 'startTime' MUST match exactly when the sound begins in the audio.
    2. SILENCE HANDLING: If the audio starts with silence, noise, or untranscribed sounds, the first timestamp MUST reflect that time (e.g. 00:05.200). DO NOT force it to 00:00.000.
    3. MONOTONICITY: Ensure strictly monotonic time (startTime < endTime).
    `;

    const mixedLanguagePolicy = `
    TRANSCRIPTION CONTENT RULES:
    1. STRICT VERBATIM: Transcribe exactly what is spoken.
    2. FILLERS ARE MANDATORY: You MUST transcribe vocalizations like "huummm", "haaa", "umm", "uh", "ah", "hmm".
       - Treat "huummm" as a word.
       - Treat "haaa" as a word.
       - DO NOT SKIP THEM. They are crucial for timestamp alignment.
    3. SCRIPT: Write words in their NATIVE script (e.g. Kanji for Japanese, Latin for English). Do not transliterate.
    `;

    const segmentationPolicy = granularity === 'word' 
      ? `MODE: WORD-LEVEL (KARAOKE)
         - Output structure MUST be hierarchical: Lines -> Words.
         - Inside 'words' array: List EVERY single word (including fillers) with precise start/end time.
         - Accuracy is paramount.`
      : `MODE: LINE-LEVEL (SUBTITLE)
         - Group words into natural phrases or sentences.
         - Max duration per segment: ~5-7 seconds.
         - Include fillers in the line where they naturally occur.`;

    const hints = languageHint 
      ? `IMPORTANT CONTEXT / LANGUAGE HINT: The audio likely contains: "${languageHint}". Use this to bias detection.` 
      : "No specific language hint provided. Detect languages automatically.";

    const systemInstruction = `
    You are a professional, high-fidelity audio alignment and transcription engine.
    Your task is to transcribe the audio file provided by the user accurately and verbatim.

    ${timingPolicy}
    ${mixedLanguagePolicy}
    ${segmentationPolicy}
    ${hints}

    NEGATIVE CONSTRAINTS:
    - DO NOT describe the audio (e.g., "music playing", "silence").
    - DO NOT output conversational text like "Here is the transcription".
    - OUTPUT ONLY VALID JSON matching the schema.
    `;

    const requestConfig: any = {
      responseMimeType: "application/json",
      responseSchema: selectedSchema,
      temperature: 0,
      topP: 0.95,
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
              { text: "Transcribe audio." }, // Minimal prompt to trigger generation
            ],
          },
        ],
        config: requestConfig,
      }),
      abortPromise
    ]);

    let text = response.text || "";
    text = text.replace(/```json|```/g, "").trim();
    if (!text) throw new Error("Empty response");

    const parsed = tryRepairJson(text);
    let rawSegments = parsed.segments || [];

    // Flatten hierarchical words if needed
    if (granularity === 'word') {
      const flattened: any[] = [];
      rawSegments.forEach((seg: any) => {
        if (seg.words && Array.isArray(seg.words) && seg.words.length > 0) {
          seg.words.forEach((word: any) => flattened.push(word));
        } else {
          flattened.push(seg);
        }
      });
      if (flattened.length > 0) rawSegments = flattened;
    }

    return postProcessSegments(rawSegments, granularity);

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

    // Use System Instruction for translation to ensure valid JSON
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

    // We use systemInstruction to enforce the language accent.
    // This helps resolve issues where "word mode" segments (single words)
    // are ignored or pronounced with the wrong accent (e.g. reading 'Air' as English instead of Indonesian).
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

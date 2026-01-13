
import { GoogleGenAI, Type } from "@google/genai";
import { TranscriptionSegment } from "../types";

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || '' });

const TRANSCRIPTION_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    segments: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
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
            description: "Transcribed text. Exact words spoken in their native script.",
          },
        },
        required: ["startTime", "endTime", "text"],
      },
    },
  },
  required: ["segments"],
};

/**
 * Converts various timestamp strings to seconds with high robustness.
 * Handles:
 * - MM:SS.mmm (Preferred)
 * - HH:MM:SS.mmm (Fallback)
 * - Full-width characters (CJK)
 * - Numeric inputs (seconds)
 */
export function timestampToSeconds(ts: string | number): number {
  if (ts === undefined || ts === null) return 0;
  let str = String(ts).trim();

  // Normalize full-width characters (numbers ０-９ and colon ：) commonly found in CJK output
  str = str.replace(/[０-９]/g, s => String.fromCharCode(s.charCodeAt(0) - 0xFEE0));
  str = str.replace(/：/g, ':');

  // Normalize: replace comma with dot, remove everything else except digits, dots, colons
  const clean = str.replace(/,/g, '.').replace(/[^\d:.]/g, '');
  
  if (!clean.includes(':')) {
    return parseFloat(clean) || 0;
  }

  const parts = clean.split(':').map(p => parseFloat(p) || 0);
  
  // Handle HH:MM:SS:mmm (4 parts) - rare hallucination
  if (parts.length === 4) {
    return (parts[0] * 3600) + (parts[1] * 60) + parts[2] + (parts[3] / 1000);
  }
  
  // Handle HH:MM:SS.mmm (3 parts)
  if (parts.length === 3) {
    return (parts[0] * 3600) + (parts[1] * 60) + parts[2];
  }
  
  // Handle MM:SS.mmm (2 parts) - Preferred
  if (parts.length === 2) {
    return (parts[0] * 60) + parts[1];
  }
  
  return parseFloat(clean) || 0;
}

/**
 * Formats seconds to MM:SS.mmm correctly handling rounding.
 * Note: Minutes can exceed 59 (e.g. 90:00.000 for 1.5 hours) to strictly follow MM:SS format.
 */
function secondsToTimestamp(totalSeconds: number): string {
  const roundedMs = Math.round(totalSeconds * 1000);
  const m = Math.floor(roundedMs / 60000);
  const s = Math.floor((roundedMs % 60000) / 1000);
  const ms = roundedMs % 1000;
  
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(ms).padStart(3, '0')}`;
}

/**
 * Repairs non-monotonic timestamps (jumping backwards).
 */
function enforceMonotonicity(segments: any[]): TranscriptionSegment[] {
  if (!segments || segments.length === 0) return [];

  const processed: TranscriptionSegment[] = [];
  let lastStartTime = -1;
  let lastEndTime = 0;

  for (const seg of segments) {
    let start = timestampToSeconds(seg.startTime);
    let end = timestampToSeconds(seg.endTime);

    // 1. Fix backward jumps (Model hallucination or reset)
    // If start time jumps back significantly compared to last end, clamp it.
    // We use a small tolerance (0.1s) to allow minor overlaps which are natural in speech.
    if (start < lastEndTime - 0.1) {
      start = lastEndTime;
    }

    // 2. Ensure start doesn't drift before last start (Basic causality)
    if (start < lastStartTime) {
      start = lastStartTime;
    }

    // 3. Ensure duration is positive.
    if (end <= start) {
      end = start + 0.05; // Reduced minimum duration to 50ms for very fast speech
    }

    // Update trackers
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

/**
 * Attempts to repair truncated JSON strings.
 */
function tryRepairJson(jsonString: string): any {
  const trimmed = jsonString.trim();

  try {
    const parsed = JSON.parse(trimmed);
    if (parsed.segments && Array.isArray(parsed.segments)) {
      return parsed;
    }
  } catch (e) {
    // Continue
  }

  const lastObjectEnd = trimmed.lastIndexOf('}');
  if (lastObjectEnd !== -1) {
    const repaired = trimmed.substring(0, lastObjectEnd + 1) + "]}";
    try {
      const parsed = JSON.parse(repaired);
      if (parsed.segments && Array.isArray(parsed.segments)) {
        return parsed;
      }
    } catch (e) {
      // Continue
    }
  }

  // Fallback regex extraction
  const segments = [];
  const segmentRegex = /\{\s*"startTime"\s*"?\s*:\s*"?([^",]+)"?\s*,\s*"endTime"\s*"?\s*:\s*"?([^",]+)"?\s*,\s*"text"\s*"?\s*:\s*(?:"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)')/g;
  
  let match;
  while ((match = segmentRegex.exec(trimmed)) !== null) {
    const rawText = match[3] !== undefined ? match[3] : match[4];
    let unescapedText = rawText;
    try {
      unescapedText = JSON.parse(`"${rawText.replace(/"/g, '\\"')}"`); 
    } catch (e) {
      unescapedText = rawText.replace(/\\"/g, '"').replace(/\\'/g, "'").replace(/\\\\/g, "\\");
    }

    segments.push({
      startTime: match[1],
      endTime: match[2],
      text: unescapedText
    });
  }
  
  if (segments.length > 0) {
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
    
    // Updated policies to strictly enforce MM:SS.mmm and prevent CJK width issues
    const timingPolicy = `
    STRICT TIMING & MONOTONICITY POLICY:
    1. FORMAT: You MUST use **MM:SS.mmm** (e.g. 00:12.512, 05:30.900) for ALL timestamps.
    2. NO HOURS: Do NOT use HH:MM:SS format. If time > 60 mins, use 65:00.000.
    3. DIGITS: USE STANDARD ASCII DIGITS (0-9). DO NOT use full-width characters (e.g. ０, １).
    4. PRECISION: Do NOT round. Use PRECISE milliseconds (e.g., .042, .915) matching the exact audio waveform.
    5. CONTINUITY: Timestamps are relative to the start (00:00.000). Never reset.
    6. ACCURACY: Sync text exactly.
    `;

    const mixedLanguagePolicy = `
    RAPID CODE-SWITCHING (CRITICAL):
    - Audio may contain multiple languages mixed in the SAME sentence.
    - BEWARE: The languages might NOT include English. (e.g. Indonesian + Japanese, Chinese + Japanese).
    - DETECT LANGUAGE AT THE WORD LEVEL. Treat all languages as equally probable.
    - WRITE EACH WORD IN ITS NATIVE SCRIPT. (e.g. "Aku cinta kamu" -> Latin, "愛してる" -> Kanji).
    - DO NOT translate. DO NOT romanize. DO NOT force English if it's not spoken.
    `;

    const subtitlePolicy = `
    SUBTITLE (LINE MODE):
    1. SEGMENTATION: 1-5 seconds per segment.
    2. TEXT: Complete sentences or natural phrases.
    `;

    const wordLevelPolicy = `
    WORD-LEVEL (WORD MODE):
    1. SEGMENTATION: **ONE WORD PER SEGMENT**.
    2. CONTINUITY: Ensure no large gaps between words unless there is actual silence.
    3. STRICT ORDER: Do not output words out of order.
    `;

    // Updated few-shot examples to MM:SS.mmm
    const fewShotExamples = `
    EXAMPLE JSON OUTPUT (Word Mode):
    {
      "segments": [
        { "startTime": "00:00.042", "endTime": "00:00.589", "text": "The" },
        { "startTime": "00:00.589", "endTime": "00:01.102", "text": "quick" },
        { "startTime": "00:01.102", "endTime": "00:01.815", "text": "brown" }
      ]
    }
    `;

    const segmentationPolicy = granularity === 'word' ? wordLevelPolicy : subtitlePolicy;
    const contextPrompt = languageHint ? `CONTEXT & LANGUAGE HINT: The audio contains the following languages/content: "${languageHint}". Pay extra attention to these languages.` : "";

    const requestConfig: any = {
      responseMimeType: "application/json",
      responseSchema: TRANSCRIPTION_SCHEMA,
      temperature: 0,
      topP: 0.95, // Encourage focused probability mass
      topK: 64,   // Restrict vocabulary to likely candidates
    };

    if (isGemini3) {
      requestConfig.thinkingConfig = { thinkingBudget: 2048 }; 
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
              {
                inlineData: {
                  data: audioBase64,
                  mimeType: mimeType,
                },
              },
              {
                text: `You are a high-fidelity audio alignment and transcription engine expert.
                
                ${contextPrompt}

                ${timingPolicy}
                ${mixedLanguagePolicy}
                ${segmentationPolicy}
                
                ${fewShotExamples}

                Rules:
                - Transcribe VERBATIM (every stutter, false start).
                - Do not summarize.
                - Output valid JSON.
                - Ensure timestamps NEVER jump backwards.
                - Listen carefully for fast speech; preserve millisecond-level precision.
                
                Audio processing...`,
              },
            ],
          },
        ],
        config: requestConfig,
      }),
      abortPromise
    ]);

    let text = response.text;
    if (!text) throw new Error("Empty response from model");

    text = text.trim();
    if (text.startsWith('```json')) {
      text = text.replace(/^```json\s*/, '').replace(/\s*```$/, '');
    } else if (text.startsWith('```')) {
      text = text.replace(/^```\s*/, '').replace(/\s*```$/, '');
    }

    const parsed = tryRepairJson(text);
    const rawSegments = parsed.segments || [];

    // Apply strict post-processing
    return enforceMonotonicity(rawSegments);

  } catch (error: any) {
    if (error.name === 'AbortError') throw error;
    console.error(`Error with ${modelName}:`, error);
    throw new Error(error.message || "Transcription failed");
  }
}

export async function translateSegments(
  segments: TranscriptionSegment[],
  targetLanguage: string
): Promise<TranscriptionSegment[]> {
  try {
    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: [
        {
          parts: [
            {
              text: `Translate the following segments into ${targetLanguage}. 
              CRITICAL: Do NOT modify the timestamps. Keep the exact format provided.
              Data: ${JSON.stringify(segments)}`,
            },
          ],
        },
      ],
      config: {
        responseMimeType: "application/json",
        responseSchema: {
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
        },
      },
    });

    let text = response.text;
    if (!text) throw new Error("Empty translation response");
    
    text = text.trim();
    if (text.startsWith('```json')) {
      text = text.replace(/^```json\s*/, '').replace(/\s*```$/, '');
    } else if (text.startsWith('```')) {
      text = text.replace(/^```\s*/, '').replace(/\s*```$/, '');
    }

    const parsed = JSON.parse(text);
    return parsed.segments || [];
  } catch (error: any) {
    console.error("Translation error:", error);
    throw error;
  }
}

export async function generateSpeech(text: string): Promise<string | undefined> {
  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash-preview-tts",
      contents: [{ parts: [{ text: text }] }],
      config: {
        responseModalities: ["AUDIO"],
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: { voiceName: 'Zephyr' },
          },
        },
      },
    });

    return response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
  } catch (error) {
    console.error("TTS error:", error);
    throw error;
  }
}


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
            description: "Timestamp in 'MM:SS.mmm' format (e.g. '00:41.520'). Use 'HH:MM:SS.mmm' only if needed.",
          },
          endTime: {
            type: Type.STRING,
            description: "Timestamp in 'MM:SS.mmm' format.",
          },
          text: {
            type: Type.STRING,
            description: "Transcribed text. Exact words spoken. No hallucinations. Must include every single word.",
          },
        },
        required: ["startTime", "endTime", "text"],
      },
    },
  },
  required: ["segments"],
};

/**
 * Robustly normalizes timestamp strings to HH:MM:SS.mmm
 * Handles MM:SS.mmm, HH:MM:SS.mmm, and even MM:SS:mmm (colon errors).
 */
function normalizeTimestamp(ts: string): string {
  if (!ts) return "00:00:00.000";
  
  // Replace comma with dot for standardizing milliseconds if model uses comma
  let clean = ts.trim().replace(',', '.').replace(/[^\d:.]/g, '');
  
  // Handle if model returns raw seconds (e.g. "65.5") despite instructions
  if (!clean.includes(':') && /^[\d.]+$/.test(clean)) {
    const totalSeconds = parseFloat(clean);
    if (!isNaN(totalSeconds)) {
       const h = Math.floor(totalSeconds / 3600);
       const m = Math.floor((totalSeconds % 3600) / 60);
       const s = Math.floor(totalSeconds % 60);
       const ms = Math.round((totalSeconds % 1) * 1000);
       return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(ms).padStart(3, '0')}`;
    }
  }

  const parts = clean.split(':');
  let h = 0, m = 0, s = 0, ms = 0;

  if (parts.length === 4) {
    // Model hallucinated HH:MM:SS:mmm (colon for ms)
    h = parseInt(parts[0], 10) || 0;
    m = parseInt(parts[1], 10) || 0;
    s = parseInt(parts[2], 10) || 0;
    ms = parseInt(parts[3].substring(0, 3).padEnd(3, '0'), 10) || 0;
  } else if (parts.length === 3) {
    const p0 = parseInt(parts[0], 10) || 0;
    const p1 = parseInt(parts[1], 10) || 0;
    
    if (parts[2].includes('.')) {
      // Standard HH:MM:SS.mmm
      h = p0;
      m = p1;
      const secParts = parts[2].split('.');
      s = parseInt(secParts[0], 10) || 0;
      ms = parseInt(secParts[1].substring(0, 3).padEnd(3, '0'), 10) || 0;
    } else {
       // Ambiguous: HH:MM:SS or MM:SS:mmm
       // Check if 3rd part is clearly milliseconds (3 digits or > 59)
       const p2Raw = parts[2];
       const p2 = parseInt(p2Raw, 10) || 0;
       
       if (p2Raw.length === 3 || p2 > 59) {
           // Treat as MM:SS:mmm (Model error using colon for ms)
           m = p0; s = p1; ms = p2;
       } else {
           // Treat as HH:MM:SS
           h = p0; m = p1; s = p2;
       }
    }
  } else if (parts.length === 2) {
    // MM:SS.mmm or MM:SS
    m = parseInt(parts[0], 10) || 0;
    if (parts[1].includes('.')) {
      const secParts = parts[1].split('.');
      s = parseInt(secParts[0], 10) || 0;
      ms = parseInt(secParts[1].substring(0, 3).padEnd(3, '0'), 10) || 0;
    } else {
      s = parseInt(parts[1], 10) || 0;
    }
  }

  // Recalculate to normalize potential overflows (e.g. 65 seconds -> 1 min 5 sec)
  const totalSeconds = (h * 3600) + (m * 60) + s + (ms / 1000);
  const finalH = Math.floor(totalSeconds / 3600);
  const finalM = Math.floor((totalSeconds % 3600) / 60);
  const finalS = Math.floor(totalSeconds % 60);
  const finalMs = Math.round((totalSeconds % 1) * 1000);

  return `${String(finalH).padStart(2, '0')}:${String(finalM).padStart(2, '0')}:${String(finalS).padStart(2, '0')}.${String(finalMs).padStart(3, '0')}`;
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

  const segments = [];
  const segmentRegex = /\{\s*"startTime"\s*:\s*"?([^",]+)"?\s*,\s*"endTime"\s*:\s*"?([^",]+)"?\s*,\s*"text"\s*:\s*(?:"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)')/g;
  
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
  signal?: AbortSignal
): Promise<TranscriptionSegment[]> {
  try {
    const isGemini3 = modelName.includes('gemini-3');
    
    const timingPolicy = `
    STRICT TIMING POLICY:
    1. FORMAT: Use **MM:SS.mmm** (e.g. 05:30.500). If audio > 1 hour, use HH:MM:SS.mmm.
    2. SEPARATOR: Use a DOT (.) for milliseconds.
    3. ABSOLUTE & CUMULATIVE: Timestamps must be relative to the START of the file.
    4. MONOTONICITY: Time MUST always move forward. startTime[n] >= endTime[n-1].
    5. ACCURACY: Sync text exactly to when it is spoken.
    `;

    const verbatimPolicy = `
    VERBATIM & FIDELITY:
    1. STRICT VERBATIM: Transcribe EXACTLY what is spoken. Do not paraphrase, summarize, or "correct" grammar.
    2. REPETITIONS: Include all repetitions (e.g. "I... I... I don't know").
    3. NO CLEANUP: Do not remove filler words like "um", "ah", "uh".
    `;

    const completenessPolicy = `
    COMPLETENESS POLICY (CRITICAL):
    1. EXHAUSTIVE: You must transcribe the ENTIRE audio file from start to finish.
    2. NO SKIPPING: Do not skip any sentences or words, even if they are quiet or fast.
    3. NO DEDUPLICATION: If a speaker repeats the same sentence, you MUST transcribe it every time it is said.
    4. SEGMENTATION: Break segments at least every 5-7 seconds. Do NOT create long segments.
    `;

    const antiHallucinationPolicy = `
    ANTI-HALLUCINATION:
    1. NO INVENTED TEXT: Do NOT output text if no speech is present.
    2. NO GUESSING: If audio is absolutely unintelligible, skip it.
    3. NO LABELS: Do not add speaker labels (like "Speaker 1:").
    `;

    const jsonSafetyPolicy = `
    JSON FORMATTING SAFETY:
    1. TEXT ESCAPING: The 'text' field MUST be wrapped in DOUBLE QUOTES (").
    2. INTERNAL QUOTES: If the text contains a double quote, ESCAPE IT (e.g. \\"). 
    `;

    const requestConfig: any = {
      responseMimeType: "application/json",
      responseSchema: TRANSCRIPTION_SCHEMA,
      temperature: 0, 
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
                text: `You are a high-fidelity, verbatim audio transcription engine. Your output must be exhaustive and complete.
                
                ${timingPolicy}
                ${verbatimPolicy}
                ${completenessPolicy}
                ${antiHallucinationPolicy}
                ${jsonSafetyPolicy}
                
                REQUIRED FORMAT: JSON object with "segments" array. 
                Preferred timestamp format: 'MM:SS.mmm'.
                Do not stop until you have reached the end of the audio.`,
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
    const segments = parsed.segments || [];

    return segments.map((s: any) => ({
      startTime: normalizeTimestamp(String(s.startTime)),
      endTime: normalizeTimestamp(String(s.endTime)),
      text: String(s.text)
    }));
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

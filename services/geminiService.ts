
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
            description: "Timestamp mulai. WAJIB format HH:MM:SS.mmm (contoh: '00:00:01.234'). Jangan bulatkan.",
          },
          endTime: {
            type: Type.STRING,
            description: "Timestamp akhir. WAJIB format HH:MM:SS.mmm (contoh: '00:00:04.567').",
          },
          text: {
            type: Type.STRING,
            description: "Teks transkripsi.",
          },
        },
        required: ["startTime", "endTime", "text"],
      },
    },
  },
  required: ["segments"],
};

/**
 * Ensures a timestamp string strictly follows HH:MM:SS.mmm
 */
function normalizeTimestamp(ts: string): string {
  const parts = ts.split(':');
  if (parts.length < 3) return ts; // Return as is if format is unrecognizable

  let secondsPart = parts[2];
  if (!secondsPart.includes('.')) {
    // If milliseconds are missing, append .000
    parts[2] = secondsPart + ".000";
  } else {
    // Ensure 3 digits for milliseconds
    const [s, ms] = secondsPart.split('.');
    parts[2] = `${s}.${ms.padEnd(3, '0').substring(0, 3)}`;
  }
  return parts.map(p => p.padStart(2, '0')).join(':');
}

export async function transcribeAudio(
  modelName: string,
  audioBase64: string,
  mimeType: string
): Promise<TranscriptionSegment[]> {
  try {
    const isGemini3 = modelName.includes('gemini-3');
    
    // Instruksi yang sangat ketat untuk format milidetik
    const precisionInstruction = "ANALISIS gelombang suara secara mendetail. JANGAN MEMBULATKAN waktu. Gunakan presisi milidetik (mmm) secara eksplisit. Format WAJIB: HH:MM:SS.mmm.";

    const response = await ai.models.generateContent({
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
              text: `Transkripsikan audio ini. ${precisionInstruction} 
              Kembalikan JSON dengan properti 'segments'. 
              Sama seperti Gemini 3, format waktu harus HH:MM:SS.mmm. 
              Ketepatan milidetik sangat penting agar teks tidak melompat-lompat saat diputar.`,
            },
          ],
        },
      ],
      config: {
        responseMimeType: "application/json",
        responseSchema: TRANSCRIPTION_SCHEMA,
        temperature: 0.1,
      },
    });

    const text = response.text;
    if (!text) throw new Error("Empty response from model");

    const parsed = JSON.parse(text);
    const segments = parsed.segments || [];

    // Post-processing untuk menjamin kualitas format timestamp
    return segments.map((s: any) => ({
      startTime: normalizeTimestamp(String(s.startTime)),
      endTime: normalizeTimestamp(String(s.endTime)),
      text: String(s.text)
    }));
  } catch (error: any) {
    console.error(`Error transcribing with ${modelName}:`, error);
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
              text: `Translate these segments into ${targetLanguage}. 
              Keep the high-precision HH:MM:SS.mmm timestamps EXACTLY as they are.
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

    const text = response.text;
    if (!text) throw new Error("Empty response from translation model");
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

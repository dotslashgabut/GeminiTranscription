
export interface TranscriptionSegment {
  startTime: string; // Format like "00:00:00" or seconds
  endTime: string;
  text: string;
  translatedText?: string;
  words?: TranscriptionSegment[];
}

export interface TranscriptionResult {
  segments: TranscriptionSegment[];
  modelName: string;
  error?: string;
  loading: boolean;
  translating?: boolean;
}

export interface AudioFileData {
  base64: string;
  mimeType: string;
  fileName: string;
  previewUrl: string;
}

export type GeminiModel = string;
export type TranscriptionMode = 'line' | 'word';

export interface SubtitleSegment {
  start: number;
  end: number;
  text: string;
  words?: SubtitleSegment[];
}

export interface LyricWord {
  text: string;
  startTime: number;
  endTime: number;
}

export interface LyricLine {
  time: number;
  text: string;
  endTime?: number;
  words?: LyricWord[];
}

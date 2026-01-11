
# Gemini Dual-Engine Audio Transcriber

An advanced, responsive web application for high-precision audio transcription, translation, and comparative analysis using Google's Gemini models.

![App Screenshot](screenshot.jpg)

## 🚀 Features

- **Dual-Model Comparison**: Transcribe audio simultaneously using `gemini-2.5-flash` and `gemini-3-flash-preview` to compare accuracy and temporal precision.
- **Model Selection**: Choose to run **Both models**, **Gemini 2.5 Only**, or **Gemini 3 Only** to save resources or focus on specific outputs.
- **Microphone Recording**: Capture audio directly from your microphone for instant transcription.
- **Precise Timestamps**: Native support for `HH:MM:SS.mmm` format, ensuring synchronization with audio playback.
- **Interactive Transcript**: Click any segment to instantly seek the audio player to that specific moment.
- **Auto-Scroll**: The transcript automatically follows the audio playhead, highlighting the current active segment.
- **Process Control**: Abort transcription requests instantly with a dedicated **Stop** button for each model.
- **Lyric & Verbatim Mode**: Advanced prompting logic ensures repeated lines (choruses) and stuttering are transcribed exactly as spoken, preventing AI summarization or deduplication.
- **Multi-Language Translation**: Translate generated transcripts into over 100 supported languages with a single click.
- **Text-to-Speech (TTS)**: Listen to translated segments using high-quality neural voices.
- **Flexible Input**: Upload local audio files, load via URL, or record directly.
- **Pro Exports**: Export your transcripts in professional formats:
  - **SRT**: Standard SubRip format for video subtitles.
  - **LRC**: Lyric file format with precise timing (optimized for karaoke).
  - **TXT**: Clean, readable text logs.
  - **JSON**: Structured data for developers.

## 🛠 Technology Stack

- **Frontend**: React 19, TypeScript
- **Styling**: Tailwind CSS
- **AI Models**: 
  - `gemini-3-flash-preview` (Reasoning & Transcription)
  - `gemini-2.5-flash` (Transcription)
  - `gemini-2.5-flash-preview-tts` (Speech Generation)
- **APIs**: Google GenAI SDK (@google/genai)

## 📖 How to Use

1. **Load Audio**: Use the "Upload" button for local files, paste a link in the URL box, or click "Record" to capture your voice.
2. **Select Model**: Choose whether to run Gemini 2.5, Gemini 3, or Both using the dropdown next to the Transcribe button.
3. **Transcribe**: Click the "Transcribe" button. The panes will populate as the Gemini models process the audio. You can stop the process at any time if needed.
4. **Navigate**: Use the built-in audio player. The transcript will highlight segments in real-time. Click any text to jump the audio to that part.
5. **Translate & Listen**: Select a target language and hit "Translate". Once finished, click the speaker icon on translated segments to hear them.
6. **Download**: Use the format buttons (Orig/Tran) at the top of each pane to save your work.

## 📝 Notes

- **API Key**: This app requires a valid Google Gemini API key configured in the environment.
- **Precision**: Gemini 3 models generally provide superior temporal alignment for fast-paced speech. The app now includes specific instructions to enforce granular segmentation (< 7s) and prevent text grouping.
- **Mobile Friendly**: The interface is fully responsive, supporting both desktop and mobile browsers.

---
*Created by a Senior Frontend Engineer with ❤️ and Gemini.*


import React, { useState, useRef, useEffect, useMemo } from 'react';
import { AudioFileData, TranscriptionResult, TranscriptionSegment } from './types';
import { transcribeAudio, translateSegments, timestampToSeconds } from './services/geminiService';
import SegmentItem from './components/SegmentItem';
import * as Exporters from './utils/exporters';

const LANGUAGES = [
  "Afrikaans", "Albanian", "Amharic", "Arabic", "Armenian", "Azerbaijani",
  "Basque", "Belarusian", "Bengali", "Bosnian", "Bulgarian", "Catalan",
  "Cebuano", "Chichewa", "Chinese (Simplified)", "Chinese (Traditional)",
  "Corsican", "Croatian", "Czech", "Danish", "Dutch", "English", "Esperanto",
  "Estonian", "Filipino", "Finnish", "French", "Frisian", "Galician",
  "Georgian", "German", "Greek", "Gujarati", "Haitian Creole", "Hausa",
  "Hawaiian", "Hebrew", "Hindi", "Hmong", "Hungarian", "Icelandic", "Igbo",
  "Indonesian", "Irish", "Italian", "Japanese", "Javanese", "Kannada",
  "Kazakh", "Khmer", "Kinyarwanda", "Korean", "Kurdish (Kurmanji)", "Kyrgyz",
  "Lao", "Latin", "Latvian", "Lithuanian", "Luxembourgish", "Macedonian",
  "Malagasy", "Malay", "Malayalam", "Maltese", "Maori", "Marathi", "Mongolian",
  "Myanmar (Burmese)", "Nepali", "Norwegian", "Odia (Oriya)", "Pashto",
  "Persian", "Polish", "Portuguese", "Punjabi", "Romanian", "Russian",
  "Samoan", "Scots Gaelic", "Serbian", "Sesotho", "Shona", "Sindhi", "Sinhala",
  "Slovak", "Slovenian", "Somali", "Spanish", "Sundanese", "Swahili", "Swedish",
  "Tajik", "Tamil", "Tatar", "Telugu", "Thai", "Turkish", "Turkmen",
  "Ukrainian", "Urdu", "Uyghur", "Uzbek", "Vietnamese", "Welsh", "Xhosa",
  "Yiddish", "Yoruba", "Zulu"
].sort();

/**
 * Robustly parses various timestamp formats into total seconds.
 * Relies on the shared logic from geminiService to ensure consistency.
 */
export const parseTimestamp = (timestamp: string | number): number => {
  if (timestamp === undefined || timestamp === null) return 0;
  return timestampToSeconds(timestamp.toString());
};

/**
 * Ensures the correct MIME type is sent to Gemini based on file extension.
 */
const getCorrectMimeType = (filename: string, originalType: string): string => {
  const ext = filename.split('.').pop()?.toLowerCase();
  switch (ext) {
    case 'mp3': return 'audio/mp3';
    case 'wav': return 'audio/wav';
    case 'flac': return 'audio/flac';
    case 'aac': return 'audio/aac';
    case 'ogg': case 'oga': return 'audio/ogg';
    case 'm4a': return 'audio/mp4';
    case 'aiff': case 'aif': return 'audio/aiff';
    case 'webm': return 'audio/webm';
    case 'opus': return 'audio/ogg';
  }
  if (originalType && (originalType.startsWith('audio/') || originalType.startsWith('video/'))) {
    return originalType;
  }
  return 'audio/mp3';
};

type ModelSelection = 'both' | 'left' | 'right';

const App: React.FC = () => {
  const [audioFile, setAudioFile] = useState<AudioFileData | null>(null);
  const [urlInput, setUrlInput] = useState("");
  const [isFetchingUrl, setIsFetchingUrl] = useState(false);
  const [targetLang, setTargetLang] = useState("Indonesian");
  const [currentTime, setCurrentTime] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [granularity, setGranularity] = useState<'line' | 'word'>('line');
  const [modelSelection, setModelSelection] = useState<ModelSelection>('both');
  const [languageHint, setLanguageHint] = useState("");
  
  // Recording State
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);

  const [results, setResults] = useState<{
    left: TranscriptionResult;
    right: TranscriptionResult;
  }>({
    left: { modelName: 'gemini-2.5-flash', segments: [], loading: false },
    right: { modelName: 'gemini-3-flash-preview', segments: [], loading: false },
  });

  const [lastInteractedSide, setLastInteractedSide] = useState<'left' | 'right' | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const appContainerRef = useRef<HTMLDivElement>(null);
  const interactionTimeout = useRef<number | null>(null);
  
  // Recording Refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const recordingIntervalRef = useRef<number | null>(null);

  // Abort Controllers for Cancellation
  const abortControllersRef = useRef<{ [key in 'left' | 'right']: AbortController | null }>({ left: null, right: null });

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      appContainerRef.current?.requestFullscreen().catch((err) => {
        console.error(`Error attempting to enable fullscreen: ${err.message}`);
      });
    } else {
      document.exitFullscreen();
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const base64 = (e.target?.result as string).split(',')[1];
        const mimeType = getCorrectMimeType(file.name, file.type);
        setAudioFile({
          base64,
          mimeType,
          fileName: file.name,
          previewUrl: URL.createObjectURL(file),
        });
        setCurrentTime(0);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleUrlLoad = async () => {
    if (!urlInput.trim()) return;
    setIsFetchingUrl(true);
    try {
      const response = await fetch(urlInput);
      if (!response.ok) throw new Error('Failed to fetch audio from URL.');
      
      const blob = await response.blob();
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64 = (reader.result as string).split(',')[1];
        const fileName = urlInput.split('/').pop()?.split('?')[0] || 'remote-audio';
        const mimeType = getCorrectMimeType(fileName, blob.type);
        setAudioFile({
          base64,
          mimeType,
          fileName,
          previewUrl: URL.createObjectURL(blob),
        });
        setCurrentTime(0);
        setIsFetchingUrl(false);
      };
      reader.readAsDataURL(blob);
    } catch (error: any) {
      alert(`Error loading audio URL: ${error.message}`);
      setIsFetchingUrl(false);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64 = (reader.result as string).split(',')[1];
          setAudioFile({
            base64,
            mimeType: 'audio/webm',
            fileName: `recording-${new Date().toISOString()}.webm`,
            previewUrl: URL.createObjectURL(audioBlob),
          });
          setCurrentTime(0);
        };
        reader.readAsDataURL(audioBlob);
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setRecordingTime(0);
      recordingIntervalRef.current = window.setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
    } catch (err) {
      console.error("Error accessing microphone:", err);
      alert("Could not access microphone.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (recordingIntervalRef.current) {
        clearInterval(recordingIntervalRef.current);
        recordingIntervalRef.current = null;
      }
    }
  };

  const handleNewSession = () => {
    stopTranscription('left');
    stopTranscription('right');
    if (isRecording) stopRecording();

    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    if (audioFile?.previewUrl) {
      URL.revokeObjectURL(audioFile.previewUrl);
    }

    setAudioFile(null);
    setUrlInput("");
    setLanguageHint("");
    setCurrentTime(0);
    setResults({
      left: { modelName: 'gemini-2.5-flash', segments: [], loading: false },
      right: { modelName: 'gemini-3-flash-preview', segments: [], loading: false },
    });
    setLastInteractedSide(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleSegmentClick = (startTime: string, side: 'left' | 'right') => {
    setLastInteractedSide(side);
    if (interactionTimeout.current) window.clearTimeout(interactionTimeout.current);
    interactionTimeout.current = window.setTimeout(() => setLastInteractedSide(null), 3000);

    if (audioRef.current) {
      const seconds = parseTimestamp(startTime);
      audioRef.current.currentTime = seconds;
      audioRef.current.play().catch(console.error);
    }
  };

  const handleTimeUpdate = () => {
    if (audioRef.current) setCurrentTime(audioRef.current.currentTime);
  };

  const activeIndices = useMemo(() => {
    const EPSILON = 0.05;
    const findActive = (segments: TranscriptionSegment[]) => {
      if (!segments.length) return -1;
      const normalized = segments.map((s, idx) => ({
        idx,
        start: parseTimestamp(s.startTime),
        end: parseTimestamp(s.endTime)
      }));
      const exactMatches = normalized.filter(m => currentTime >= m.start && currentTime < m.end);
      if (exactMatches.length > 0) return exactMatches[0].idx;
      
      let candidateIdx = -1;
      for (let i = 0; i < normalized.length; i++) {
        if (currentTime >= normalized[i].start - EPSILON) candidateIdx = i;
        else break;
      }
      return candidateIdx;
    };
    return {
      left: findActive(results.left.segments),
      right: findActive(results.right.segments)
    };
  }, [currentTime, results.left.segments, results.right.segments]);

  const startTranscription = async () => {
    if (!audioFile) return;

    // Stop audio playback when transcription starts
    if (audioRef.current) {
      audioRef.current.pause();
    }

    // Reset only the selected sides
    setResults(prev => {
      const next = { ...prev };
      if (modelSelection === 'both' || modelSelection === 'left') {
        next.left = { ...prev.left, loading: true, segments: [], error: undefined };
      }
      if (modelSelection === 'both' || modelSelection === 'right') {
        next.right = { ...prev.right, loading: true, segments: [], error: undefined };
      }
      return next;
    });

    const runTranscription = async (side: 'left' | 'right') => {
      if (abortControllersRef.current[side]) abortControllersRef.current[side]?.abort();
      const controller = new AbortController();
      abortControllersRef.current[side] = controller;
      try {
        const segments = await transcribeAudio(
          results[side].modelName, 
          audioFile.base64, 
          audioFile.mimeType,
          controller.signal,
          granularity,
          languageHint
        );
        setResults(prev => ({ ...prev, [side]: { ...prev[side], segments, loading: false } }));
      } catch (err: any) {
        if (err.name === 'AbortError') {
          setResults(prev => ({ ...prev, [side]: { ...prev[side], loading: false, error: 'Canceled' } }));
        } else {
          setResults(prev => ({ ...prev, [side]: { ...prev[side], error: err.message, loading: false } }));
        }
      } finally {
        abortControllersRef.current[side] = null;
      }
    };

    const promises = [];
    if (modelSelection === 'both' || modelSelection === 'left') {
      promises.push(runTranscription('left'));
    }
    if (modelSelection === 'both' || modelSelection === 'right') {
      promises.push(runTranscription('right'));
    }

    await Promise.all(promises);
  };

  const stopTranscription = (side: 'left' | 'right') => {
    const controller = abortControllersRef.current[side];
    if (controller) {
      controller.abort();
      abortControllersRef.current[side] = null;
    }
  };

  const handleTranslate = async () => {
    setResults(prev => ({
      left: { ...prev.left, translating: true },
      right: { ...prev.right, translating: true }
    }));
    const runTranslate = async (side: 'left' | 'right') => {
      if (results[side].segments.length === 0) return;
      try {
        const translated = await translateSegments(results[side].segments, targetLang);
        setResults(prev => ({ ...prev, [side]: { ...prev[side], segments: translated, translating: false } }));
      } catch (err: any) {
        alert(`Translation failed: ${err.message}`);
        setResults(prev => ({ ...prev, [side]: { ...prev[side], translating: false } }));
      }
    };
    await Promise.all(['left', 'right'].map((s: any) => runTranslate(s as 'left' | 'right')));
  };

  const handleDownload = (side: 'left' | 'right', format: string, type: 'original' | 'translated') => {
    const segments = results[side].segments;
    if (segments.length === 0) return;
    const baseFileName = audioFile?.fileName.split('.').slice(0, -1).join('.') || 'audio';
    const filename = `${baseFileName}_${results[side].modelName}${type === 'translated' ? '_translated' : ''}.${format.toLowerCase()}`;
    const totalDuration = audioRef.current?.duration;

    let content = "";
    switch (format) {
      case 'TXT': content = Exporters.exportAsTXT(segments, type); break;
      case 'JSON': content = Exporters.exportAsJSON(segments, type); break;
      case 'SRT': content = Exporters.exportAsSRT(segments, type); break;
      case 'LRC': content = Exporters.exportAsLRC(segments, type, totalDuration); break;
      case 'TTML': content = Exporters.exportAsTTML(segments, type); break;
    }
    Exporters.downloadFile(content, filename);
  };

  const isTranscribing = results.left.loading || results.right.loading;
  const isTranslating = results.left.translating || results.right.translating;
  const hasResults = results.left.segments.length > 0 || results.right.segments.length > 0;
  const canClear = audioFile !== null || urlInput !== "" || hasResults;

  return (
    <div ref={appContainerRef} className="h-screen flex flex-col bg-slate-100 overflow-hidden font-sans">
      <header className="bg-white border-b border-slate-200 px-4 md:px-6 py-3 md:py-4 flex-shrink-0 z-30 shadow-sm overflow-y-auto max-h-[40vh] md:max-h-none">
        <div className="max-w-full mx-auto flex flex-col gap-3 md:gap-4">
          
          {/* Header Top Row: Title & App Actions */}
          <div className="flex items-center justify-between">
            <h1 className="text-xl md:text-2xl font-bold text-slate-900 flex items-center gap-2">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="text-blue-600 md:w-7 md:h-7"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" /><path d="M19 10v2a7 7 0 0 1-14 0v-2" /><line x1="12" x2="12" y1="19" y2="22" /></svg>
              <span className="truncate">Gemini Transcription</span>
            </h1>
            
            <div className="flex items-center gap-2">
              {canClear && (
                <button 
                  onClick={handleNewSession}
                  title="New Session"
                  className="p-2 md:px-4 md:py-2 text-slate-600 hover:text-red-600 hover:bg-red-50 rounded-xl border border-slate-200 bg-white shadow-sm transition-all flex items-center gap-2"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>
                  <span className="hidden md:inline text-xs font-semibold">New Session</span>
                </button>
              )}
              <button onClick={toggleFullscreen} title="Toggle Fullscreen" className="p-2 text-slate-600 hover:bg-slate-100 rounded-xl border border-slate-200 shadow-sm bg-white">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M8 3H5a2 2 0 0 0-2 2v3M21 8V5a2 2 0 0 0-2-2h-3M3 16v3a2 2 0 0 0 2 2h3M16 21h3a2 2 0 0 0 2-2v-3" /></svg>
              </button>
            </div>
          </div>

          {/* Controls Row: Flattened for optimal wrapping */}
          <div className="flex flex-wrap items-center gap-2 md:gap-3">
            
            {/* URL Input */}
            <div className="flex items-center bg-slate-100 rounded-xl border border-slate-200 p-1 flex-grow md:flex-grow-0 min-w-[150px] w-full md:w-auto">
              <input 
                type="text" 
                value={urlInput}
                onChange={(e) => setUrlInput(e.target.value)}
                placeholder="Paste URL (mp3, wav...)"
                disabled={isRecording}
                className="bg-transparent text-xs md:text-sm px-2 md:px-3 py-1 outline-none flex-1 w-full text-slate-700 disabled:opacity-50"
              />
              <button 
                onClick={handleUrlLoad}
                disabled={isFetchingUrl || !urlInput || isRecording}
                className={`px-2 md:px-3 py-1 text-[10px] md:text-xs font-bold rounded-lg transition-all whitespace-nowrap ${isFetchingUrl || isRecording ? 'bg-slate-200 text-slate-400' : 'bg-white text-blue-600 shadow-sm hover:bg-slate-50 border border-slate-200'}`}
              >
                {isFetchingUrl ? '...' : 'Load'}
              </button>
            </div>

            {/* File Upload */}
            <input 
              type="file" 
              accept="audio/*,.mp3,.wav,.aac,.ogg,.flac,.m4a,.aiff,.webm" 
              className="hidden" 
              ref={fileInputRef} 
              onChange={handleFileUpload} 
            />
            <button 
              onClick={() => fileInputRef.current?.click()} 
              disabled={isRecording}
              className={`px-3 md:px-4 py-2 text-xs md:text-sm font-semibold text-slate-700 bg-white border border-slate-300 rounded-xl hover:bg-slate-50 shadow-sm transition-all whitespace-nowrap ${isRecording ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              {audioFile ? 'Change File' : 'Upload File'}
            </button>
            
            {/* Record */}
            <button 
              onClick={isRecording ? stopRecording : startRecording}
              disabled={isTranscribing}
              className={`px-3 md:px-4 py-2 text-xs md:text-sm font-semibold rounded-xl shadow-sm transition-all flex items-center gap-2 whitespace-nowrap border ${
                isRecording 
                  ? 'bg-red-50 text-red-600 border-red-200 hover:bg-red-100' 
                  : 'bg-white text-slate-700 border-slate-300 hover:bg-slate-50'
              } ${isTranscribing ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              <span className={`w-2 h-2 rounded-full ${isRecording ? 'bg-red-600 animate-pulse' : 'bg-slate-400'}`}></span>
              {isRecording ? `Stop (${recordingTime}s)` : 'Record'}
            </button>

            {/* Granularity Switch */}
            <div className="flex items-center bg-white rounded-xl border border-slate-300 overflow-hidden shadow-sm">
              <button 
                onClick={() => setGranularity('line')}
                className={`px-3 py-2 text-xs font-semibold transition-colors ${granularity === 'line' ? 'bg-slate-100 text-slate-900' : 'text-slate-500 hover:bg-slate-50'}`}
                disabled={isTranscribing}
              >
                Line
              </button>
              <div className="w-px h-full bg-slate-200"></div>
              <button 
                onClick={() => setGranularity('word')}
                className={`px-3 py-2 text-xs font-semibold transition-colors ${granularity === 'word' ? 'bg-slate-100 text-slate-900' : 'text-slate-500 hover:bg-slate-50'}`}
                disabled={isTranscribing}
              >
                Word
              </button>
            </div>

            {/* Language Hint Input */}
            <div className="flex items-center bg-white rounded-xl border border-slate-300 px-2 py-1 shadow-sm w-full md:w-auto">
                <input 
                  type="text" 
                  value={languageHint}
                  onChange={(e) => setLanguageHint(e.target.value)}
                  placeholder="Mixed Lang Hint (e.g. Jpn/Eng)"
                  disabled={isTranscribing}
                  className="bg-transparent text-xs md:text-sm px-1 py-1 outline-none w-full md:w-48 text-slate-700 disabled:opacity-50"
                  title="Provide a hint for mixed languages (e.g., 'Japanese mixed with English')"
                />
            </div>

            <div className="w-px h-6 bg-slate-300 mx-1 hidden lg:block"></div>

            {/* Model Selection */}
            <select
              value={modelSelection}
              onChange={(e) => setModelSelection(e.target.value as ModelSelection)}
              disabled={isTranscribing || isRecording}
              className="text-xs md:text-sm font-medium border-slate-300 rounded-xl py-2 px-2 bg-white shadow-sm text-slate-900 focus:ring-2 focus:ring-blue-500 outline-none border hover:bg-slate-50 cursor-pointer disabled:opacity-50"
            >
              <option value="both">Both Models</option>
              <option value="left">Gemini 2.5 Only</option>
              <option value="right">Gemini 3 Only</option>
            </select>

            {/* Transcribe Button */}
            <button
              disabled={!audioFile || isTranscribing || isRecording}
              onClick={startTranscription}
              className={`px-4 md:px-6 py-2 text-xs md:text-sm font-bold text-white rounded-xl shadow-lg transition-all whitespace-nowrap flex items-center justify-center gap-2 min-w-[120px] ${
                !audioFile || isTranscribing || isRecording 
                ? 'bg-slate-300 cursor-not-allowed' 
                : 'bg-blue-600 hover:bg-blue-700 hover:shadow-xl active:scale-95'
              }`}
            >
              {isTranscribing ? (
                <>
                  <svg className="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  <span>Working...</span>
                </>
              ) : (
                'Transcribe'
              )}
            </button>

            {/* Translate Controls - Rendered inline to allow wrapping with other buttons */}
            {hasResults && (
              <>
                <div className="w-px h-6 bg-slate-300 mx-1 hidden lg:block"></div>
                <select
                  value={targetLang}
                  onChange={(e) => setTargetLang(e.target.value)}
                  disabled={isRecording}
                  className="text-xs md:text-sm font-medium border-slate-300 rounded-xl py-2 px-2 md:px-3 bg-white shadow-sm text-slate-900 focus:ring-2 focus:ring-blue-500 outline-none disabled:opacity-50 border"
                >
                  {LANGUAGES.map(l => <option key={l} value={l}>{l}</option>)}
                </select>
                <button
                  disabled={isTranslating || isRecording}
                  onClick={handleTranslate}
                  className={`px-3 md:px-5 py-2 text-xs md:text-sm font-bold text-white rounded-xl shadow-lg transition-all whitespace-nowrap flex items-center justify-center gap-2 min-w-[100px] ${
                    isTranslating || isRecording 
                    ? 'bg-indigo-400 cursor-not-allowed opacity-90' 
                    : 'bg-indigo-600 hover:bg-indigo-700 hover:shadow-xl active:scale-95'
                  }`}
                >
                  {isTranslating ? (
                    <>
                      <svg className="animate-spin h-3 w-3 md:h-4 md:w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      <span>Translating</span>
                    </>
                  ) : (
                    'Translate'
                  )}
                </button>
              </>
            )}

          </div>

          {/* Audio Player Row */}
          {audioFile && (
            <div className="pt-2 border-t border-slate-100 flex flex-col md:flex-row md:items-center gap-2 md:gap-4 w-full">
              <span className="text-[10px] md:text-xs font-bold text-slate-500 truncate max-w-full md:max-w-xs bg-slate-50 px-2 py-1 rounded">
                Playing: {audioFile.fileName}
              </span>
              <audio
                ref={audioRef}
                src={audioFile.previewUrl}
                onTimeUpdate={handleTimeUpdate}
                controls
                className="h-9 w-full md:flex-1"
              />
            </div>
          )}
        </div>
      </header>

      <div className="flex-1 overflow-hidden px-1 py-1 md:px-2 md:py-2 flex flex-col">
        <main className="flex-1 w-full max-w-full mx-auto bg-white rounded-xl shadow-2xl border border-slate-200 overflow-hidden grid grid-cols-1 md:grid-cols-2 gap-px relative">
          {(['left', 'right'] as const).map((side) => {
            const activeIdx = activeIndices[side];
            const isLoading = results[side].loading;
            const isTranslatingLocal = results[side].translating;
            const hasTranslated = results[side].segments.some(s => s.translatedText);
            const error = results[side].error;

            return (
              <div key={side} className="flex flex-col h-full min-h-0 bg-white">
                <div className={`px-4 py-2 border-b border-slate-200 flex-shrink-0 z-10 ${side === 'left' ? 'bg-slate-50/50' : 'bg-indigo-50/30'}`}>
                  <div className="flex items-center justify-between mb-1">
                    <h2 className="text-[10px] md:text-xs font-black text-slate-800 uppercase tracking-widest flex items-center gap-2">
                      <span className={`w-2 h-2 rounded-full ${side === 'left' ? 'bg-blue-500' : 'bg-indigo-500'}`}></span>
                      {results[side].modelName}
                    </h2>
                    {results[side].segments.length > 0 && (
                      <div className="flex items-center gap-2">
                        {isTranslatingLocal && <span className="animate-spin h-3 w-3 border-2 border-indigo-600 border-t-transparent rounded-full"></span>}
                        <span className="text-[9px] bg-green-100 text-green-700 px-2 py-0.5 rounded-full font-black">
                          {results[side].segments.length}
                        </span>
                      </div>
                    )}
                  </div>
                  {results[side].segments.length > 0 && (
                    <div className="flex flex-col gap-1.5 overflow-hidden">
                      <div className="flex items-center gap-2 overflow-x-auto pb-0.5 no-scrollbar">
                        <span className="text-[8px] font-bold text-slate-400 uppercase min-w-[32px]">Orig:</span>
                        {['TXT', 'SRT', 'LRC', 'TTML', 'JSON'].map(format => (
                          <button 
                            key={format} 
                            onClick={() => handleDownload(side, format, 'original')} 
                            className="px-2 py-0.5 text-[9px] font-black border border-slate-200 rounded bg-white hover:bg-slate-50 text-slate-600 shadow-sm transition-all whitespace-nowrap"
                          >
                            {format}
                          </button>
                        ))}
                      </div>
                      {hasTranslated && (
                        <div className="flex items-center gap-2 overflow-x-auto pb-0.5 no-scrollbar">
                          <span className="text-[8px] font-bold text-indigo-400 uppercase min-w-[32px]">Tran:</span>
                          {['TXT', 'SRT', 'LRC', 'TTML', 'JSON'].map(format => (
                            <button 
                              key={format} 
                              onClick={() => handleDownload(side, format, 'translated')} 
                              className="px-2 py-0.5 text-[9px] font-black border border-indigo-100 rounded bg-indigo-50 hover:bg-indigo-100 text-indigo-600 shadow-sm transition-all whitespace-nowrap"
                            >
                              {format}
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>

                <div className="flex-1 overflow-y-auto overflow-x-hidden relative bg-white scrolling-touch p-0 min-h-0">
                  {isLoading ? (
                    <div className="h-full flex flex-col items-center justify-center p-8">
                      <div className={`animate-spin rounded-full h-12 w-12 border-4 ${
                          side === 'left' ? 'border-blue-100 border-t-blue-500' : 'border-indigo-100 border-t-indigo-500'
                      }`}></div>
                      <button 
                        onClick={() => stopTranscription(side)}
                        className="mt-6 px-4 py-1.5 bg-white border border-red-100 text-red-500 text-[10px] font-bold rounded-full hover:bg-red-50 transition-colors uppercase tracking-wide"
                      >
                        Stop Generation
                      </button>
                    </div>
                  ) : results[side].segments.length > 0 ? (
                    <div className="space-y-0.5">
                      {results[side].segments.map((s, idx) => (
                        <SegmentItem
                          key={`${side}-${idx}-${s.startTime}`}
                          segment={s}
                          isActive={activeIdx === idx}
                          isManualSeek={lastInteractedSide === side}
                          onSelect={(ts) => handleSegmentClick(ts, side)}
                        />
                      ))}
                    </div>
                  ) : error ? (
                    <div className="h-full flex flex-col items-center justify-center p-8 text-center text-slate-400">
                      <div className="bg-red-50 p-3 rounded-full mb-3">
                         <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-red-500"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
                      </div>
                      <p className="text-xs font-bold text-red-600">{error}</p>
                    </div>
                  ) : (
                    <div className="h-full flex flex-col items-center justify-center p-8 text-center opacity-25">
                      <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round"><path d="M2 10v3"/><path d="M6 6v11"/><path d="M10 3v18"/><path d="M14 8v7"/><path d="M18 5v13"/><path d="M22 10v3"/></svg>
                      <p className="mt-4 text-[10px] md:text-xs font-black uppercase tracking-widest">Awaiting Audio</p>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </main>
      </div>

      <footer className="bg-white border-t border-slate-200 p-2 text-[10px] font-bold text-slate-400 flex-shrink-0">
        <div className="max-w-[98%] mx-auto flex justify-between items-center px-4">
          <div className="flex items-center gap-4">
            <span className="hidden md:inline bg-slate-100 text-slate-600 px-2 py-0.5 rounded uppercase">Dual Engine Comparative Analysis</span>
          </div>
          <div className="font-mono text-[9px] text-slate-500">PLAYHEAD: {currentTime.toFixed(3)}s</div>
        </div>
      </footer>
    </div>
  );
};

export default App;

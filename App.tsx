
import React, { useState, useRef, useEffect, useMemo } from 'react';
import { AudioFileData, TranscriptionResult, TranscriptionSegment } from './types';
import { transcribeAudio, translateSegments } from './services/geminiService';
import SegmentItem from './components/SegmentItem';
import * as Exporters from './utils/exporters';

const LANGUAGES = [
  "Indonesian", "English", "Japanese", "Korean", "Spanish", "French", "German",
  "Chinese (Simplified)", "Chinese (Traditional)", "Arabic", "Russian",
  "Portuguese", "Italian", "Dutch", "Turkish", "Vietnamese", "Thai", "Hindi",
  "Malay", "Filipino"
];

/**
 * Robustly parses various timestamp formats into total seconds.
 */
export const parseTimestamp = (timestamp: string | number): number => {
  if (timestamp === undefined || timestamp === null) return 0;
  
  let str = timestamp.toString().trim().toLowerCase();
  str = str.replace(/[ms]/g, '').replace(',', '.');

  if (str.includes(':')) {
    const parts = str.split(':').map(p => parseFloat(p) || 0);
    if (parts.length === 3) {
      return (parts[0] * 3600) + (parts[1] * 60) + parts[2];
    } else if (parts.length === 2) {
      return (parts[0] * 60) + parts[1];
    }
  }
  return parseFloat(str) || 0;
};

const App: React.FC = () => {
  const [audioFile, setAudioFile] = useState<AudioFileData | null>(null);
  const [targetLang, setTargetLang] = useState("Indonesian");
  const [currentTime, setCurrentTime] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);
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
        setAudioFile({
          base64,
          mimeType: file.type,
          fileName: file.name,
          previewUrl: URL.createObjectURL(file),
        });
        setCurrentTime(0);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSegmentClick = (startTime: string, side: 'left' | 'right') => {
    setLastInteractedSide(side);

    if (interactionTimeout.current) window.clearTimeout(interactionTimeout.current);
    interactionTimeout.current = window.setTimeout(() => {
      setLastInteractedSide(null);
    }, 3000);

    if (audioRef.current) {
      const seconds = parseTimestamp(startTime);
      audioRef.current.currentTime = seconds;
      audioRef.current.play().catch(console.error);
    }
  };

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
    }
  };

  const activeIndices = useMemo(() => {
    const EPSILON = 0.05; // 50ms buffer to prevent flickering

    const findActive = (segments: TranscriptionSegment[]) => {
      if (!segments.length) return -1;
      
      const normalized = segments.map((s, idx) => ({
        idx,
        start: parseTimestamp(s.startTime),
        end: parseTimestamp(s.endTime)
      }));

      // 1. Find the exact match first
      const exactMatches = normalized.filter(m => currentTime >= m.start && currentTime < m.end);
      if (exactMatches.length > 0) {
        return exactMatches[0].idx;
      }
      
      // 2. Fallback to closest previous segment with a small buffer
      let candidateIdx = -1;
      for (let i = 0; i < normalized.length; i++) {
        if (currentTime >= normalized[i].start - EPSILON) {
          candidateIdx = i;
        } else {
          break;
        }
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
    setResults({
      left: { ...results.left, loading: true, segments: [], error: undefined },
      right: { ...results.right, loading: true, segments: [], error: undefined },
    });
    const runTranscription = async (side: 'left' | 'right') => {
      try {
        const segments = await transcribeAudio(results[side].modelName, audioFile.base64, audioFile.mimeType);
        setResults(prev => ({ ...prev, [side]: { ...prev[side], segments, loading: false } }));
      } catch (err: any) {
        setResults(prev => ({ ...prev, [side]: { ...prev[side], error: err.message, loading: false } }));
      }
    };
    await Promise.all([runTranscription('left'), runTranscription('right')]);
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
    }
    Exporters.downloadFile(content, filename);
  };

  const isTranscribing = results.left.loading || results.right.loading;
  const isTranslating = results.left.translating || results.right.translating;
  const hasResults = results.left.segments.length > 0 || results.right.segments.length > 0;

  return (
    <div ref={appContainerRef} className="h-screen flex flex-col bg-slate-100 overflow-hidden font-sans">
      <header className="bg-white border-b border-slate-200 px-6 py-4 flex-shrink-0 z-30 shadow-sm">
        <div className="max-w-full mx-auto space-y-4">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
                <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="text-blue-600"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" /><path d="M19 10v2a7 7 0 0 1-14 0v-2" /><line x1="12" x2="12" y1="19" y2="22" /></svg>
                Gemini Audio comparison
              </h1>
            </div>

            <div className="flex items-center gap-3 flex-wrap">
              <input type="file" accept="audio/*" className="hidden" ref={fileInputRef} onChange={handleFileUpload} />
              <button onClick={() => fileInputRef.current?.click()} className="px-4 py-2 text-sm font-semibold text-slate-700 bg-white border border-slate-300 rounded-xl hover:bg-slate-50 shadow-sm transition-all">
                {audioFile ? 'Change File' : 'Upload Audio'}
              </button>

              <button
                disabled={!audioFile || isTranscribing}
                onClick={startTranscription}
                className={`px-6 py-2 text-sm font-bold text-white rounded-xl shadow-lg transition-all ${!audioFile || isTranscribing ? 'bg-slate-300' : 'bg-blue-600 hover:bg-blue-700'}`}
              >
                {isTranscribing ? 'Processing...' : 'Transcribe'}
              </button>

              {hasResults && (
                <div className="flex items-center gap-2 border-l pl-3 ml-1 border-slate-200">
                  <select
                    value={targetLang}
                    onChange={(e) => setTargetLang(e.target.value)}
                    className="text-sm font-medium border-slate-300 rounded-xl py-2 px-3 bg-white shadow-sm text-slate-900 focus:ring-2 focus:ring-blue-500 outline-none"
                  >
                    {LANGUAGES.map(l => <option key={l} value={l}>{l}</option>)}
                  </select>
                  <button
                    disabled={isTranslating}
                    onClick={handleTranslate}
                    className="px-5 py-2 text-sm font-bold text-white bg-indigo-600 rounded-xl shadow-lg hover:bg-indigo-700 disabled:bg-slate-300 transition-all"
                  >
                    {isTranslating ? 'Translating...' : 'Translate'}
                  </button>
                </div>
              )}

              <button onClick={toggleFullscreen} className="p-2.5 text-slate-600 hover:bg-slate-100 rounded-xl border border-slate-200 shadow-sm bg-white ml-auto">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M8 3H5a2 2 0 0 0-2 2v3M21 8V5a2 2 0 0 0-2-2h-3M3 16v3a2 2 0 0 0 2 2h3M16 21h3a2 2 0 0 0 2-2v-3" /></svg>
              </button>
            </div>
          </div>

          {audioFile && (
            <div className="pt-2 border-t border-slate-100 flex items-center gap-4">
              <audio
                ref={audioRef}
                src={audioFile.previewUrl}
                onTimeUpdate={handleTimeUpdate}
                controls
                className="h-9 flex-1 max-w-full"
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

            return (
              <div key={side} className="flex flex-col h-full min-h-0 bg-white">
                <div className={`px-4 py-2 border-b border-slate-200 flex-shrink-0 z-10 ${side === 'left' ? 'bg-slate-50/50' : 'bg-indigo-50/30'}`}>
                  <div className="flex items-center justify-between mb-1">
                    <h2 className="text-xs font-black text-slate-800 uppercase tracking-widest flex items-center gap-2">
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
                        {['TXT', 'SRT', 'LRC', 'JSON'].map(format => (
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
                          {['TXT', 'SRT', 'LRC', 'JSON'].map(format => (
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

                <div className="flex-1 overflow-y-auto overflow-x-hidden relative bg-white scrolling-touch p-1 scroll-smooth min-h-0">
                  {isLoading ? (
                    <div className="absolute inset-0 bg-white/80 backdrop-blur-sm z-20 flex flex-col items-center justify-center p-8">
                      <div className="w-10 h-10 border-4 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                      <p className="mt-4 text-xs font-black text-slate-700 tracking-widest animate-pulse uppercase">Transcribing...</p>
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
                  ) : (
                    <div className="h-full flex flex-col items-center justify-center p-8 text-center opacity-25">
                      <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round"><path d="M2 10v3"/><path d="M6 6v11"/><path d="M10 3v18"/><path d="M14 8v7"/><path d="M18 5v13"/><path d="M22 10v3"/></svg>
                      <p className="mt-4 text-xs font-black uppercase tracking-widest">Awaiting Audio</p>
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
            <span className="bg-slate-100 text-slate-600 px-2 py-0.5 rounded uppercase">Dual Engine Comparative Analysis</span>
          </div>
          <div className="font-mono text-[9px] text-slate-500">PLAYHEAD: {currentTime.toFixed(3)}s</div>
        </div>
      </footer>
    </div>
  );
};

export default App;

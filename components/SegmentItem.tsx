
import React, { useEffect, useRef, useState } from 'react';
import { TranscriptionSegment } from '../types';
import { generateSpeech } from '../services/geminiService';
import { decodeBase64, decodeAudioData } from '../utils/audio';

interface SegmentItemProps {
  segment: TranscriptionSegment;
  isActive?: boolean;
  isManualSeek?: boolean;
  onSelect: (startTime: string) => void;
}

const SegmentItem: React.FC<SegmentItemProps> = ({ segment, isActive, isManualSeek, onSelect }) => {
  const elementRef = useRef<HTMLButtonElement>(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const audioContextRef = useRef<AudioContext | null>(null);

  useEffect(() => {
    if (isActive && elementRef.current && !isManualSeek) {
      const element = elementRef.current;
      const container = element.closest('.overflow-y-auto');
      
      if (container) {
        // Use a faster timeout for snappier response in word-mode
        const scrollTimeout = window.setTimeout(() => {
          const containerHeight = container.clientHeight;
          const elementTop = element.offsetTop;
          const elementHeight = element.clientHeight;
          const scrollTop = container.scrollTop;

          // Reduced buffer to 20px to ensure better tracking on small screens
          const buffer = 20; 
          const isFullyVisible = (elementTop >= scrollTop + buffer) && 
                                (elementTop + elementHeight <= scrollTop + containerHeight - buffer);

          if (!isFullyVisible) {
            const targetScrollTop = elementTop - (containerHeight / 2) + (elementHeight / 2);
            container.scrollTo({
              top: targetScrollTop,
              behavior: 'smooth'
            });
          }
        }, 50); // Reduced delay to 50ms
        return () => window.clearTimeout(scrollTimeout);
      }
    }
  }, [isActive, isManualSeek]);

  const handleSpeak = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (isSpeaking) return;
    if (!segment.translatedText) return;

    setIsSpeaking(true);
    try {
      const audioData = await generateSpeech(segment.translatedText);
      if (!audioData) throw new Error("No audio data");

      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      }
      
      const ctx = audioContextRef.current;
      const decodedBytes = decodeBase64(audioData);
      const audioBuffer = await decodeAudioData(decodedBytes, ctx, 24000, 1);
      
      const source = ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(ctx.destination);
      source.onended = () => setIsSpeaking(false);
      source.start();
    } catch (error) {
      console.error("TTS error:", error);
      setIsSpeaking(false);
    }
  };

  return (
    <button
      ref={elementRef}
      type="button"
      onClick={(e) => {
        e.preventDefault();
        onSelect(segment.startTime);
      }}
      className={`w-full text-left flex flex-col px-4 py-3 transition-colors duration-200 focus:outline-none relative select-none border-l-[6px] group ${
        isActive 
          ? 'bg-blue-50 border-blue-600 z-10' 
          : 'hover:bg-slate-50 border-transparent'
      }`}
    >
      <div className="flex items-center gap-3 mb-1 pointer-events-none">
        <span className={`text-[11px] font-bold font-mono px-2 py-0.5 rounded-md tracking-tight transition-colors ${
          isActive ? 'text-blue-700 bg-blue-100' : 'text-slate-400 bg-slate-100 group-hover:text-slate-500'
        }`}>
          {segment.startTime}
        </span>
      </div>
      
      {/* Transcription Text - Removed font-weight change to prevent layout shift */}
      <p className={`text-lg md:text-xl leading-relaxed transition-colors duration-200 font-medium ${
        isActive ? 'text-slate-900' : 'text-slate-600'
      }`}>
        {segment.text}
      </p>

      {segment.translatedText && (
        <div className={`mt-2 p-2.5 rounded-lg border flex items-start gap-3 transition-all ${
          isActive ? 'bg-white border-indigo-200 shadow-sm' : 'bg-slate-50 border-slate-100'
        }`}>
          <p className={`text-base italic flex-1 leading-relaxed ${
            isActive ? 'text-indigo-800 font-medium' : 'text-slate-500'
          }`}>
            {segment.translatedText}
          </p>
          <button 
            type="button"
            onClick={handleSpeak}
            disabled={isSpeaking}
            className={`p-1.5 rounded-md transition-all flex-shrink-0 ${
              isSpeaking ? 'bg-indigo-600 text-white' : 'bg-indigo-50 text-indigo-600 hover:bg-indigo-100'
            }`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M11 5L6 9H2v6h4l5 4V5z"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/></svg>
          </button>
        </div>
      )}
    </button>
  );
};

export default SegmentItem;

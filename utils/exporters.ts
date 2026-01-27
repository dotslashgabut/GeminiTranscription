import { TranscriptionSegment } from "../types";

/**
 * Robustly parses various timestamp formats into total seconds.
 */
const parseTimestampToSeconds = (ts: string | number): number => {
  if (ts === undefined || ts === null) return 0;
  let str = ts.toString().trim().toLowerCase();
  str = str.replace(/[ms]/g, '').replace(',', '.');

  if (str.includes(':')) {
    const parts = str.split(':').map(p => parseFloat(p) || 0);
    // HH:MM:SS.mmm
    if (parts.length === 3) return (parts[0] * 3600) + (parts[1] * 60) + parts[2];
    // MM:SS.mmm
    if (parts.length === 2) return (parts[0] * 60) + parts[1];
  }
  return parseFloat(str) || 0;
};

const formatSecondsToSRT = (totalSeconds: number): string => {
  const roundedMs = Math.round(totalSeconds * 1000);
  const h = Math.floor(roundedMs / 3600000);
  const m = Math.floor((roundedMs % 3600000) / 60000);
  const s = Math.floor((roundedMs % 60000) / 1000);
  const ms = roundedMs % 1000;
  
  return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')},${ms.toString().padStart(3, '0')}`;
};

const formatSecondsToVTT = (totalSeconds: number): string => {
  const roundedMs = Math.round(totalSeconds * 1000);
  const h = Math.floor(roundedMs / 3600000);
  const m = Math.floor((roundedMs % 3600000) / 60000);
  const s = Math.floor((roundedMs % 60000) / 1000);
  const ms = roundedMs % 1000;
  
  return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${ms.toString().padStart(3, '0')}`;
};

const formatSecondsToLRC = (totalSeconds: number): string => {
  const m = Math.floor(totalSeconds / 60);
  const s = (totalSeconds % 60);
  const sInt = Math.floor(s);
  const ms = Math.round((s % 1) * 100); // LRC usually uses 2 digits for ms (hundredths)

  return `[${m.toString().padStart(2, '0')}:${sInt.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}]`;
};

/**
 * Ensures timestamp is in HH:MM:SS.mmm format for TTML.
 */
const formatTimestampForTTML = (ts: string): string => {
  let clean = ts.trim();
  if (!clean.includes('.')) clean += '.000';
  
  // Normalize colon count
  const colonCount = (clean.match(/:/g) || []).length;
  if (colonCount === 1) {
    // MM:SS.mmm -> 00:MM:SS.mmm
    clean = "00:" + clean;
  } else if (colonCount === 0) {
    // raw seconds -> HH:MM:SS.mmm
    const sec = parseFloat(clean);
    if (!isNaN(sec)) {
      const h = Math.floor(sec / 3600);
      const m = Math.floor((sec % 3600) / 60);
      const s = Math.floor(sec % 60);
      const ms = Math.round((sec % 1) * 1000);
      return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(ms).padStart(3, '0')}`;
    }
  }
  
  // Ensure 3 digits for milliseconds
  if (clean.includes('.')) {
      const [main, msPart] = clean.split('.');
      clean = `${main}.${msPart.padEnd(3, '0').substring(0, 3)}`;
  }
  
  return clean;
};

const escapeXml = (unsafe: string): string => {
  return unsafe.replace(/[<>&'"]/g, (c) => {
    switch (c) {
      case '<': return '&lt;';
      case '>': return '&gt;';
      case '&': return '&amp;';
      case '\'': return '&apos;';
      case '"': return '&quot;';
      default: return c;
    }
  });
};

const isCJKChar = (char: string): boolean => {
  return /[\u4e00-\u9fa5\u3040-\u30ff\uac00-\ud7af]/.test(char);
};

export const downloadFile = (content: string, filename: string) => {
  const blob = new Blob([content], { type: 'application/octet-stream' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
};

export const exportAsTXT = (segments: TranscriptionSegment[], type: 'original' | 'translated'): string => {
  return segments.map(s => {
    return type === 'translated' ? (s.translatedText || '') : s.text;
  }).join('\n\n');
};

export const exportAsSRT = (segments: TranscriptionSegment[], type: 'original' | 'translated'): string => {
  return segments.map((s, i) => {
    const text = type === 'translated' ? (s.translatedText || '') : s.text;
    const start = formatSecondsToSRT(parseTimestampToSeconds(s.startTime));
    const end = formatSecondsToSRT(parseTimestampToSeconds(s.endTime));
    return `${i + 1}\n${start} --> ${end}\n${text}\n`;
  }).join('\n');
};

export const exportAsVTT = (segments: TranscriptionSegment[], type: 'original' | 'translated'): string => {
  // Logic to group individual segments (words/phrases) into readable cues
  const groups: TranscriptionSegment[][] = [];
  let currentGroup: TranscriptionSegment[] = [];

  segments.forEach((s) => {
    if (currentGroup.length === 0) {
      currentGroup.push(s);
      return;
    }

    const prev = currentGroup[currentGroup.length - 1];
    const prevEnd = parseTimestampToSeconds(prev.endTime);
    const currStart = parseTimestampToSeconds(s.startTime);
    const prevText = type === 'translated' ? (prev.translatedText || prev.text) : prev.text;
    
    const isSentenceEnd = /[.!?。！？]$/.test(prevText.trim());
    const isPause = (currStart - prevEnd) > 0.8;
    const currentChars = currentGroup.reduce((acc, seg) => acc + (type === 'translated' ? (seg.translatedText || '').length : seg.text.length), 0);
    const isLong = currentChars > 45;
    const isModeratePause = (currStart - prevEnd) > 0.3;
    const hasComma = /[,，]$/.test(prevText.trim());

    if (isSentenceEnd || isPause || (isLong && (isModeratePause || hasComma))) {
      groups.push(currentGroup);
      currentGroup = [s];
    } else {
      currentGroup.push(s);
    }
  });
  
  if (currentGroup.length > 0) groups.push(currentGroup);

  const bodyContent = groups.map((group) => {
    const first = group[0];
    const last = group[group.length - 1];
    const start = formatSecondsToVTT(parseTimestampToSeconds(first.startTime));
    const end = formatSecondsToVTT(parseTimestampToSeconds(last.endTime));

    const content = group.map((s, index) => {
      const wordStart = formatSecondsToVTT(parseTimestampToSeconds(s.startTime));
      let text = type === 'translated' ? (s.translatedText || '') : s.text;
      
      // Auto-spacing for mixed language
      const isLast = index === group.length - 1;
      if (!isLast && !text.endsWith(' ')) {
        const nextContent = type === 'translated' ? (group[index + 1].translatedText || '') : group[index + 1].text;
        if (!isCJKChar(text.slice(-1)) || !isCJKChar(nextContent.trim().charAt(0))) {
          text += ' ';
        }
      }

      // Add intra-line timestamps for original word-mode results
      const isWordMode = segments.length > 0 && !segments[0].text.includes(' '); // Heuristic check
      if (type === 'original' && segments.length > 10) { // Assume word mode if many segments
         return `<${wordStart}>${text}`;
      }
      return text;
    }).join('');

    return `${start} --> ${end}\n${content}`;
  }).join('\n\n');

  return `WEBVTT\n\n${bodyContent}`;
};

export const exportAsLRC = (segments: TranscriptionSegment[], type: 'original' | 'translated', totalDuration?: number): string => {
  const lines: string[] = [];
  
  for (let i = 0; i < segments.length; i++) {
    const s = segments[i];
    const startTime = parseTimestampToSeconds(s.startTime);
    const text = type === 'translated' ? (s.translatedText || '') : s.text;

    const cleanText = text.replace(/[\r\n]+/g, ' ');
    lines.push(`${formatSecondsToLRC(startTime)}${cleanText}`);
  }
  return lines.join('\n');
};

export const exportAsTTML = (segments: TranscriptionSegment[], type: 'original' | 'translated'): string => {
  // Logic to group individual segments (words/phrases) into readable lines (paragraphs)
  const groups: TranscriptionSegment[][] = [];
  let currentGroup: TranscriptionSegment[] = [];

  segments.forEach((s) => {
    if (currentGroup.length === 0) {
      currentGroup.push(s);
      return;
    }

    const prev = currentGroup[currentGroup.length - 1];
    const prevEnd = parseTimestampToSeconds(prev.endTime);
    const currStart = parseTimestampToSeconds(s.startTime);
    
    const prevText = type === 'translated' ? (prev.translatedText || prev.text) : prev.text;
    const isSentenceEnd = /[.!?。！？]$/.test(prevText.trim());
    const isPause = (currStart - prevEnd) > 0.8;
    const currentChars = currentGroup.reduce((acc, seg) => acc + (type === 'translated' ? (seg.translatedText || '').length : seg.text.length), 0);
    const isLong = currentChars > 45;
    const isModeratePause = (currStart - prevEnd) > 0.3;
    const hasComma = /[,，]$/.test(prevText.trim());

    if (isSentenceEnd || isPause || (isLong && (isModeratePause || hasComma))) {
      groups.push(currentGroup);
      currentGroup = [s];
    } else {
      currentGroup.push(s);
    }
  });
  
  if (currentGroup.length > 0) {
    groups.push(currentGroup);
  }

  // Generate XML
  const bodyContent = groups.map((group) => {
    if (group.length === 0) return '';

    const first = group[0];
    const last = group[group.length - 1];
    
    const pStart = formatTimestampForTTML(first.startTime);
    const pEnd = formatTimestampForTTML(last.endTime);

    const spans = group.map((s, index) => {
      const start = formatTimestampForTTML(s.startTime);
      const end = formatTimestampForTTML(s.endTime);
      let content = type === 'translated' ? (s.translatedText || '') : s.text;
      
      const isLast = index === group.length - 1;
      
      if (!isLast && !content.endsWith(' ')) {
        const nextContent = type === 'translated' 
          ? (group[index + 1].translatedText || '') 
          : group[index + 1].text;
        
        const lastChar = content.slice(-1);
        const nextChar = nextContent.trim().charAt(0);
        const isBoundaryCJK = isCJKChar(lastChar) && isCJKChar(nextChar);
        
        if (!isBoundaryCJK) {
          content += ' ';
        }
      }
      
      return `        <span begin="${start}" end="${end}">${escapeXml(content)}</span>`;
    }).join('\n');

    return `      <p begin="${pStart}" end="${pEnd}">\n${spans}\n      </p>`;
  }).join('\n');

  return `<?xml version="1.0" encoding="UTF-8"?>
<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling" xml:lang="en">
  <head>
    <styling>
      <style xml:id="defaultCaption" tts:fontSize="10px" tts:fontFamily="SansSerif" tts:fontWeight="normal" tts:fontStyle="normal" tts:textDecoration="none" tts:color="white" tts:backgroundColor="black" tts:textAlign="center" />
    </styling>
  </head>
  <body>
    <div style="defaultCaption">
${bodyContent}
    </div>
  </body>
</tt>`;
};

export const exportAsJSON = (segments: TranscriptionSegment[], type: 'original' | 'translated'): string => {
  const data = segments.map(s => ({
    startTime: s.startTime,
    endTime: s.endTime,
    text: type === 'translated' ? (s.translatedText || '') : s.text
  }));
  return JSON.stringify(data, null, 2);
};
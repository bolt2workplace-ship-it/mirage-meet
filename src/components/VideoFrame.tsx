import { VideoOff, Mic, MicOff } from 'lucide-react';
import { useEffect, useRef } from 'react';
import type { Participant } from '../types';

interface VideoFrameProps {
  stream?: MediaStream | null;
  participant?: Participant;
  isLocal?: boolean;
  isSpeaking?: boolean;
}

export default function VideoFrame({ stream, participant, isLocal, isSpeaking }: VideoFrameProps) {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = isLocal ? null : stream;
      if (isLocal) {
        videoRef.current.srcObject = stream;
        videoRef.current.muted = true;
      }
    }
  }, [stream, isLocal]);

  const cameraEnabled = participant?.cameraEnabled !== false;
  const micEnabled = participant?.microphoneEnabled !== false;
  const hasVideo = !!stream;

  return (
    <div className="video-container relative rounded-xl overflow-hidden bg-dark-950 aspect-video">
      {hasVideo && cameraEnabled ? (
        <>
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted={isLocal}
            className="w-full h-full object-cover"
          />
          {isSpeaking && (
            <div className="absolute inset-0 ring-2 ring-primary-400 ring-opacity-60 rounded-xl pointer-events-none" />
          )}
        </>
      ) : (
        <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-dark-800 to-dark-900">
          <div className="w-16 h-16 md:w-20 md:h-20 rounded-full bg-dark-700 flex items-center justify-center">
            <span className="text-2xl md:text-3xl font-semibold text-primary-400">
              {participant?.displayName?.charAt(0).toUpperCase() || '?'}
            </span>
          </div>
        </div>
      )}

      {participant && (
        <div className="absolute bottom-2 left-2 right-2 flex items-center justify-between">
          <div className="glass px-2.5 py-1 rounded-lg">
            <span className="text-xs md:text-sm font-medium text-white truncate">
              {participant.displayName}
              {isLocal && ' (You)'}
            </span>
          </div>
          <div className="flex items-center gap-1">
            <div className={`w-7 h-7 md:w-8 md:h-8 rounded-lg flex items-center justify-center ${micEnabled ? 'glass' : 'bg-red-500/80'}`}>
              {micEnabled ? <Mic size={14} className="text-white" /> : <MicOff size={14} className="text-white" />}
            </div>
            {!cameraEnabled && hasVideo && (
              <div className="w-7 h-7 md:w-8 md:h-8 rounded-lg flex items-center justify-center bg-red-500/80">
                <VideoOff size={14} className="text-white" />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

import { useRef, useState, useCallback, useEffect } from 'react';
import type { TransformationSettings, BackgroundOption } from '../types';

interface UseFaceTransformReturn {
  processedStream: MediaStream | null;
  transformationSettings: TransformationSettings;
  setTransformationSettings: React.Dispatch<React.SetStateAction<TransformationSettings>>;
  referenceVideo: HTMLVideoElement | null;
  setReferenceVideo: (video: HTMLVideoElement | null) => void;
  backgroundOptions: BackgroundOption[];
  isProcessing: boolean;
  initializeTransform: (stream: MediaStream) => Promise<void>;
  updateBackground: (backgroundId: string) => void;
  cleanup: () => void;
}

export const backgroundOptions: BackgroundOption[] = [
  { id: 'none', name: 'None', thumbnail: '', value: '' },
  { id: 'office', name: 'Modern Office', thumbnail: 'https://images.pexels.com/photos/1170412/pexels-photo-1170412.jpeg?w=200', value: 'https://images.pexels.com/photos/1170412/pexels-photo-1170412.jpeg?w=1920' },
  { id: 'luxury', name: 'Luxury Office', thumbnail: 'https://images.pexels.com/photos/3801740/pexels-photo-3801740.jpeg?w=200', value: 'https://images.pexels.com/photos/3801740/pexels-photo-3801740.jpeg?w=1920' },
  { id: 'studio', name: 'Studio', thumbnail: 'https://images.pexels.com/photos/1595385/pexels-photo-1595385.jpeg?w=200', value: 'https://images.pexels.com/photos/1595385/pexels-photo-1595385.jpeg?w=1920' },
  { id: 'conference', name: 'Conference Room', thumbnail: 'https://images.pexels.com/photos/159711/books-coffee-students-room-159711.jpeg?w=200', value: 'https://images.pexels.com/photos/159711/books-coffee-students-room-159711.jpeg?w=1920' },
  { id: 'apartment', name: 'Modern Apartment', thumbnail: 'https://images.pexels.com/photos/1571460/pexels-photo-1571460.jpeg?w=200', value: 'https://images.pexels.com/photos/1571460/pexels-photo-1571460.jpeg?w=1920' },
];

export function useFaceTransform(): UseFaceTransformReturn {
  const [processedStream, setProcessedStream] = useState<MediaStream | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [transformationSettings, setTransformationSettings] = useState<TransformationSettings>({
    enabled: false,
    referenceVideo: null,
    background: '',
  });
  const [referenceVideo, setReferenceVideo] = useState<HTMLVideoElement | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const bgCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const backgroundImgRef = useRef<HTMLImageElement | null>(null);
  const inputStreamRef = useRef<MediaStream | null>(null);
  const selfieSegmentationRef = useRef<any>(null);
  const isInitializedRef = useRef(false);

  const loadMediaPipeScripts = useCallback(async () => {
    if (isInitializedRef.current) return true;

    const scripts = [
      { id: 'mediapipe-selfie', src: 'https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/selfie_segmentation.js' },
    ];

    const loadScript = (id: string, src: string): Promise<void> => {
      return new Promise((resolve, reject) => {
        if (document.getElementById(id)) {
          resolve();
          return;
        }
        const script = document.createElement('script');
        script.id = id;
        script.src = src;
        script.onload = () => resolve();
        script.onerror = reject;
        document.head.appendChild(script);
      });
    };

    await Promise.all(scripts.map(s => loadScript(s.id, s.src)));
    isInitializedRef.current = true;
    return true;
  }, []);

  const initializeTransform = useCallback(async (stream: MediaStream) => {
    inputStreamRef.current = stream;
    setIsProcessing(true);

    const video = document.createElement('video');
    video.srcObject = stream;
    video.playsInline = true;
    video.muted = true;
    await video.play().catch((err) => console.error('Video play error:', err));
    videoRef.current = video;

    const canvas = document.createElement('canvas');
    canvas.width = 1280;
    canvas.height = 720;
    canvasRef.current = canvas;

    const bgCanvas = document.createElement('canvas');
    bgCanvas.width = 1280;
    bgCanvas.height = 720;
    bgCanvasRef.current = bgCanvas;

    const outputStream = canvas.captureStream(30);
    stream.getAudioTracks().forEach(track => outputStream.addTrack(track));

    try {
      await loadMediaPipeScripts();

      if (window.SelfieSegmentation) {
        const selfieSegmentation = new window.SelfieSegmentation({
          locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${file}`,
        });

        selfieSegmentation.setOptions({
          modelSelection: 1,
          selfieMode: false,
        });

        selfieSegmentation.onResults((results: any) => {
          if (!canvasRef.current || !videoRef.current || !bgCanvasRef.current) return;

          const canvas = canvasRef.current;
          const ctx = canvas.getContext('2d', { willReadFrequently: true });
          const bgCanvas = bgCanvasRef.current;
          const bgCtx = bgCanvas.getContext('2d');

          if (!ctx || !bgCtx) return;

          ctx.clearRect(0, 0, canvas.width, canvas.height);

          if (referenceVideo && transformationSettings.enabled && referenceVideo.readyState >= 2) {
            bgCtx.drawImage(referenceVideo, 0, 0, bgCanvas.width, bgCanvas.height);
            ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

            if (results.segmentationMask) {
              ctx.globalCompositeOperation = 'destination-out';
              ctx.drawImage(results.segmentationMask, 0, 0, canvas.width, canvas.height);
              ctx.globalCompositeOperation = 'destination-over';
              ctx.drawImage(bgCanvas, 0, 0, canvas.width, canvas.height);
              ctx.globalCompositeOperation = 'source-over';
            }
          } else if (transformationSettings.background && backgroundImgRef.current) {
            ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

            if (results.segmentationMask) {
              ctx.globalCompositeOperation = 'destination-out';
              ctx.drawImage(results.segmentationMask, 0, 0, canvas.width, canvas.height);
              ctx.globalCompositeOperation = 'destination-over';
              ctx.drawImage(backgroundImgRef.current, 0, 0, canvas.width, canvas.height);
              ctx.globalCompositeOperation = 'source-over';
            }
          } else {
            ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);
          }
        });

        selfieSegmentationRef.current = selfieSegmentation;

        const processFrame = async () => {
          if (!selfieSegmentationRef.current || !videoRef.current || !canvasRef.current) {
            animationFrameRef.current = requestAnimationFrame(processFrame);
            return;
          }

          if (videoRef.current.readyState >= 2) {
            try {
              await selfieSegmentationRef.current.send({ image: videoRef.current });
            } catch (err: any) {
              const ctx = canvasRef.current.getContext('2d');
              if (ctx && videoRef.current) {
                ctx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);
              }
            }
          }
          animationFrameRef.current = requestAnimationFrame(processFrame);
        };

        processFrame();
      } else {
        const processFrame = () => {
          if (!canvasRef.current || !videoRef.current) {
            animationFrameRef.current = requestAnimationFrame(processFrame);
            return;
          }

          const canvas = canvasRef.current;
          const ctx = canvas.getContext('2d');
          if (!ctx || !videoRef.current) return;

          ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

          if (transformationSettings.background && backgroundImgRef.current) {
            ctx.globalCompositeOperation = 'destination-over';
            ctx.drawImage(backgroundImgRef.current, 0, 0, canvas.width, canvas.height);
            ctx.globalCompositeOperation = 'source-over';
          }

          animationFrameRef.current = requestAnimationFrame(processFrame);
        };

        processFrame();
      }
    } catch (error) {
      console.error('Error initializing segmentation:', error);
      const streamFallback = canvas.captureStream(30);
      stream.getAudioTracks().forEach(track => streamFallback.addTrack(track));
      setProcessedStream(streamFallback);
    }

    setProcessedStream(outputStream);
  }, [loadMediaPipeScripts, referenceVideo, transformationSettings.enabled, transformationSettings.background]);

  const updateBackground = useCallback((backgroundId: string) => {
    const bgOption = backgroundOptions.find(opt => opt.id === backgroundId);
    const backgroundValue = bgOption?.value || '';

    setTransformationSettings(prev => ({
      ...prev,
      background: backgroundValue,
    }));

    if (backgroundValue) {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.src = backgroundValue;
      img.onload = () => {
        backgroundImgRef.current = img;
      };
      img.onerror = () => {
        backgroundImgRef.current = null;
      };
    } else {
      backgroundImgRef.current = null;
    }
  }, []);

  const cleanup = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    if (selfieSegmentationRef.current && selfieSegmentationRef.current.close) {
      selfieSegmentationRef.current.close();
    }
    if (referenceVideo) {
      referenceVideo.pause();
      referenceVideo.src = '';
    }
    setProcessedStream(null);
    setIsProcessing(false);
  }, [referenceVideo]);

  useEffect(() => {
    return () => {
      cleanup();
    };
  }, [cleanup]);

  return {
    processedStream,
    transformationSettings,
    setTransformationSettings,
    referenceVideo,
    setReferenceVideo,
    backgroundOptions,
    isProcessing,
    initializeTransform,
    updateBackground,
    cleanup,
  };
}

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
  statusMessage: string;
  initializeTransform: (stream: MediaStream) => Promise<void>;
  updateBackground: (backgroundId: string) => void;
  cleanup: () => void;
}

export const backgroundOptions: BackgroundOption[] = [
  { id: 'none', name: 'None', thumbnail: '', value: '' },
  { id: 'office', name: 'Modern Office', thumbnail: 'https://images.pexels.com/photos/1170412/pexels-photo-1170412.jpeg?auto=compress&cs=tinysrgb&w=200', value: 'https://images.pexels.com/photos/1170412/pexels-photo-1170412.jpeg?auto=compress&cs=tinysrgb&w=1280' },
  { id: 'luxury', name: 'Luxury Office', thumbnail: 'https://images.pexels.com/photos/3801740/pexels-photo-3801740.jpeg?auto=compress&cs=tinysrgb&w=200', value: 'https://images.pexels.com/photos/3801740/pexels-photo-3801740.jpeg?auto=compress&cs=tinysrgb&w=1280' },
  { id: 'studio', name: 'Studio', thumbnail: 'https://images.pexels.com/photos/1595385/pexels-photo-1595385.jpeg?auto=compress&cs=tinysrgb&w=200', value: 'https://images.pexels.com/photos/1595385/pexels-photo-1595385.jpeg?auto=compress&cs=tinysrgb&w=1280' },
  { id: 'conference', name: 'Conference Room', thumbnail: 'https://images.pexels.com/photos/159711/books-coffee-students-room-159711.jpeg?auto=compress&cs=tinysrgb&w=200', value: 'https://images.pexels.com/photos/159711/books-coffee-students-room-159711.jpeg?auto=compress&cs=tinysrgb&w=1280' },
  { id: 'apartment', name: 'Modern Apartment', thumbnail: 'https://images.pexels.com/photos/1571460/pexels-photo-1571460.jpeg?auto=compress&cs=tinysrgb&w=200', value: 'https://images.pexels.com/photos/1571460/pexels-photo-1571460.jpeg?auto=compress&cs=tinysrgb&w=1280' },
];

export function useFaceTransform(): UseFaceTransformReturn {
  const [processedStream, setProcessedStream] = useState<MediaStream | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [statusMessage, setStatusMessage] = useState('Camera Ready');
  const [transformationSettings, setTransformationSettings] = useState<TransformationSettings>({
    enabled: false,
    referenceVideo: null,
    background: '',
  });
  const [referenceVideo, setReferenceVideo] = useState<HTMLVideoElement | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const backgroundImgRef = useRef<HTMLImageElement | null>(null);
  const inputStreamRef = useRef<MediaStream | null>(null);
  const selfieSegmentationRef = useRef<any>(null);
  const isInitializedRef = useRef(false);
  const isProcessingRef = useRef(false);

  const currentBackgroundRef = useRef<string>('');
  const settingsRef = useRef(transformationSettings);

  useEffect(() => {
    settingsRef.current = transformationSettings;
  }, [transformationSettings]);

  const loadMediaPipeScripts = useCallback(async () => {
    if (isInitializedRef.current) return true;

    const loadScript = (id: string, src: string): Promise<void> => {
      return new Promise((resolve, reject) => {
        if (document.getElementById(id)) {
          resolve();
          return;
        }
        const script = document.createElement('script');
        script.id = id;
        script.src = src;
        script.crossOrigin = 'anonymous';
        script.onload = () => resolve();
        script.onerror = reject;
        document.head.appendChild(script);
      });
    };

    await loadScript('mediapipe-selfie', 'https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/selfie_segmentation.js');
    await loadScript('mediapipe-selfie-utils', 'https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/selfie_segmentation_solution_utils.js');

    isInitializedRef.current = true;
    return true;
  }, []);

  const initializeTransform = useCallback(async (stream: MediaStream) => {
    if (isProcessingRef.current) return;
    isProcessingRef.current = true;
    inputStreamRef.current = stream;
    setIsProcessing(true);
    setStatusMessage('Initializing camera...');

    const video = document.createElement('video');
    video.srcObject = stream;
    video.playsInline = true;
    video.muted = true;

    try {
      await video.play();
      videoRef.current = video;
      setStatusMessage('Camera Ready');
    } catch (err) {
      console.error('Video play error:', err);
      setStatusMessage('Camera Error');
      isProcessingRef.current = false;
      return;
    }

    const canvas = document.createElement('canvas');
    canvas.width = 1280;
    canvas.height = 720;
    canvasRef.current = canvas;

    const outputStream = canvas.captureStream(30);
    stream.getAudioTracks().forEach(track => outputStream.addTrack(track));
    setProcessedStream(outputStream);

    try {
      setStatusMessage('Loading AI model...');
      await loadMediaPipeScripts();

      if (window.SelfieSegmentation) {
        const selfieSegmentation = new window.SelfieSegmentation({
          locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${file}`,
        });

        selfieSegmentation.setOptions({
          modelSelection: 1,
          selfieMode: false,
        });

        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        if (!ctx) throw new Error('Canvas context not available');

        selfieSegmentation.onResults((results: any) => {
          if (!canvasRef.current || !videoRef.current) return;

          const canvas = canvasRef.current;
          const ctx = canvas.getContext('2d', { willReadFrequently: true });
          if (!ctx) return;

          ctx.clearRect(0, 0, canvas.width, canvas.height);

          const settings = settingsRef.current;
          const bgValue = currentBackgroundRef.current;

          if (bgValue && backgroundImgRef.current && backgroundImgRef.current.complete) {
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
        setStatusMessage('Background Active');

        let lastProcessTime = 0;
        const targetFPS = 30;
        const frameInterval = 1000 / targetFPS;

        const processFrame = async (timestamp: number) => {
          if (!selfieSegmentationRef.current || !videoRef.current || !canvasRef.current) {
            animationFrameRef.current = requestAnimationFrame(processFrame);
            return;
          }

          const elapsed = timestamp - lastProcessTime;

          if (elapsed >= frameInterval && videoRef.current.readyState >= 2) {
            lastProcessTime = timestamp - (elapsed % frameInterval);
            try {
              await selfieSegmentationRef.current.send({ image: videoRef.current });
            } catch (err) {
              const ctx = canvasRef.current.getContext('2d');
              if (ctx && videoRef.current) {
                ctx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);
              }
            }
          }
          animationFrameRef.current = requestAnimationFrame(processFrame);
        };

        animationFrameRef.current = requestAnimationFrame(processFrame);
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

          if (currentBackgroundRef.current && backgroundImgRef.current && backgroundImgRef.current.complete) {
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
      setStatusMessage('AI module failed to load');

      const fallbackCanvas = document.createElement('canvas');
      fallbackCanvas.width = 1280;
      fallbackCanvas.height = 720;
      canvasRef.current = fallbackCanvas;

      const fallbackStream = fallbackCanvas.captureStream(30);
      stream.getAudioTracks().forEach(track => fallbackStream.addTrack(track));
      setProcessedStream(fallbackStream);

      const processFrame = () => {
        if (!canvasRef.current || !videoRef.current) {
          animationFrameRef.current = requestAnimationFrame(processFrame);
          return;
        }

        const ctx = canvasRef.current.getContext('2d');
        if (!ctx || !videoRef.current) return;

        ctx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);
        animationFrameRef.current = requestAnimationFrame(processFrame);
      };

      processFrame();
    }

    isProcessingRef.current = false;
  }, [loadMediaPipeScripts]);

  const updateBackground = useCallback((backgroundId: string) => {
    const bgOption = backgroundOptions.find(opt => opt.id === backgroundId);
    const bgValue = bgOption?.value || '';

    currentBackgroundRef.current = bgValue;

    setTransformationSettings(prev => ({
      ...prev,
      background: bgValue,
    }));

    if (bgValue) {
      setStatusMessage('Loading background...');
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.src = bgValue;

      img.onload = () => {
        backgroundImgRef.current = img;
        setStatusMessage('Background Active');
      };

      img.onerror = () => {
        backgroundImgRef.current = null;
        setStatusMessage('Background Load Failed');
      };
    } else {
      backgroundImgRef.current = null;
      setStatusMessage('Camera Ready');
    }
  }, []);

  const cleanup = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    if (selfieSegmentationRef.current) {
      try {
        selfieSegmentationRef.current.close();
      } catch (e) {}
      selfieSegmentationRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
      videoRef.current = null;
    }
    inputStreamRef.current = null;
    backgroundImgRef.current = null;
    currentBackgroundRef.current = '';
    setProcessedStream(null);
    setIsProcessing(false);
    setStatusMessage('Camera Ready');
    isProcessingRef.current = false;
  }, []);

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
    statusMessage,
    initializeTransform,
    updateBackground,
    cleanup,
  };
}

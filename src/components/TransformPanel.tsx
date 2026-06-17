import { Upload, X, ChevronLeft, ChevronRight } from 'lucide-react';
import { useState, useRef, useEffect } from 'react';
import type { TransformationSettings } from '../types';
import { backgroundOptions } from '../hooks/useFaceTransform';

interface TransformPanelProps {
  transformationSettings: TransformationSettings;
  setTransformationSettings: React.Dispatch<React.SetStateAction<TransformationSettings>>;
  referenceVideo: HTMLVideoElement | null;
  setReferenceVideo: (video: HTMLVideoElement | null) => void;
  onBackgroundChange: (backgroundId: string) => void;
  isCollapsed: boolean;
  onToggleCollapse: () => void;
}

export default function TransformPanel({
  transformationSettings,
  setTransformationSettings,
  referenceVideo: _referenceVideo,
  setReferenceVideo,
  onBackgroundChange,
  isCollapsed,
  onToggleCollapse,
}: TransformPanelProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const videoInputRef = useRef<HTMLInputElement>(null);
  const previewVideoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const handleVideoUpload = (file: File) => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    const video = document.createElement('video');
    video.src = url;
    video.loop = true;
    video.muted = true;
    video.playsInline = true;
    video.play();
    setReferenceVideo(video);

    setTransformationSettings(prev => ({
      ...prev,
      referenceVideo: url,
    }));
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) {
      handleVideoUpload(file);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleVideoUpload(file);
    }
  };

  const clearVideo = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setReferenceVideo(null);
    setTransformationSettings(prev => ({
      ...prev,
      referenceVideo: null,
      enabled: false,
    }));
  };

  const toggleTransformation = () => {
    setTransformationSettings(prev => ({
      ...prev,
      enabled: !prev.enabled,
    }));
  };

  return (
    <div
      className={`fixed top-16 right-0 bottom-16 w-80 bg-dark-900 border-l border-dark-700
        transition-transform duration-300 z-20
        ${isCollapsed ? 'translate-x-full' : 'translate-x-0'}`}
    >
      <button
        onClick={onToggleCollapse}
        className="absolute -left-10 top-1/2 -translate-y-1/2 w-10 h-20 bg-dark-800
          border border-dark-700 rounded-l-lg flex items-center justify-center
          hover:bg-dark-700 transition-colors"
      >
        {isCollapsed ? (
          <ChevronLeft size={20} className="text-white" />
        ) : (
          <ChevronRight size={20} className="text-white" />
        )}
      </button>

      <div className="h-full overflow-y-auto p-4">
        <h2 className="text-lg font-semibold text-white mb-4">Transformation Controls</h2>
        <p className="text-xs text-dark-400 mb-6">
          Upload a reference video to enable face transformation. Only visible to you.
        </p>

        <div className="space-y-6">
          <div>
            <label className="text-sm font-medium text-dark-300 block mb-2">
              Reference Video
            </label>
            <div
              className={`relative border-2 border-dashed rounded-lg p-4 transition-colors ${
                isDragging ? 'border-primary-400 bg-primary-400/10' : 'border-dark-600'
              } ${transformationSettings.referenceVideo ? 'border-primary-500' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              onClick={() => videoInputRef.current?.click()}
            >
              {previewUrl ? (
                <div className="relative">
                  <video
                    ref={previewVideoRef}
                    src={previewUrl}
                    className="w-full h-32 object-cover rounded-lg"
                    autoPlay
                    loop
                    muted
                    playsInline
                  />
                  <button
                    onClick={(e) => { e.stopPropagation(); clearVideo(); }}
                    className="absolute top-2 right-2 w-6 h-6 rounded-full bg-red-500 flex items-center justify-center hover:bg-red-600 transition-colors"
                  >
                    <X size={14} className="text-white" />
                  </button>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-32 text-center">
                  <Upload size={24} className="text-dark-400 mb-2" />
                  <p className="text-sm text-dark-400">
                    Drag & drop video or click to upload
                  </p>
                  <p className="text-xs text-dark-500 mt-1">
                    MP4, WebM supported
                  </p>
                </div>
              )}
              <input
                ref={videoInputRef}
                type="file"
                accept="video/*"
                className="hidden"
                onChange={handleFileSelect}
              />
            </div>
          </div>

          <div>
            <label className="text-sm font-medium text-dark-300 block mb-3">
              Virtual Background
            </label>
            <div className="grid grid-cols-2 gap-2">
              {backgroundOptions.map((bg) => (
                <button
                  key={bg.id}
                  className={`relative rounded-lg overflow-hidden aspect-video border-2 transition-all ${
                    transformationSettings.background === bg.value
                      ? 'border-primary-400 ring-2 ring-primary-400/30'
                      : 'border-dark-600 hover:border-dark-500'
                  }`}
                  onClick={() => onBackgroundChange(bg.id)}
                >
                  {bg.id === 'none' ? (
                    <div className="w-full h-full bg-dark-700 flex items-center justify-center">
                      <span className="text-xs text-dark-400">No Background</span>
                    </div>
                  ) : (
                    <img
                      src={bg.thumbnail}
                      alt={bg.name}
                      className="w-full h-full object-cover"
                    />
                  )}
                  <div className="absolute bottom-0 left-0 right-0 bg-dark-900/80 px-2 py-1">
                    <span className="text-xs text-white truncate">{bg.name}</span>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {transformationSettings.referenceVideo && (
            <div className="pt-4 border-t border-dark-700">
              <button
                onClick={toggleTransformation}
                className={`w-full py-3 px-4 rounded-lg font-medium transition-all ${
                  transformationSettings.enabled
                    ? 'bg-primary-500 text-white hover:bg-primary-600'
                    : 'bg-dark-700 text-dark-300 hover:bg-dark-600'
                }`}
              >
                {transformationSettings.enabled ? 'Transformation Enabled' : 'Enable Transformation'}
              </button>
              <p className="text-xs text-dark-400 mt-2 text-center">
                {transformationSettings.enabled
                  ? 'Your face is being transformed based on reference'
                  : 'Toggle to start face transformation'}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

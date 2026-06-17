/// <reference types="vite/client" />

interface Window {
  SelfieSegmentation: {
    new (config?: { locateFile?: (file: string) => string }): {
      setOptions: (options: { modelSelection?: number; selfieMode?: boolean }) => void;
      onResults: (callback: (results: any) => void) => void;
      send: (input: { image: HTMLVideoElement | HTMLCanvasElement }) => Promise<void>;
      close: () => void;
    };
  };
}

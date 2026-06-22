import { useRef, useState, useCallback, useEffect } from 'react';
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';
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
  { id: 'none',       name: 'None',             thumbnail: '', value: '' },
  {
    id: 'office',
    name: 'Modern Office',
    thumbnail: 'https://images.pexels.com/photos/667838/pexels-photo-667838.jpeg?auto=compress&cs=tinysrgb&w=200',
    value:     'https://images.pexels.com/photos/667838/pexels-photo-667838.jpeg?auto=compress&cs=tinysrgb&w=1280',
  },
  {
    id: 'luxury',
    name: 'Luxury Office',
    thumbnail: 'https://images.pexels.com/photos/1743555/pexels-photo-1743555.jpeg?auto=compress&cs=tinysrgb&w=200',
    value:     'https://images.pexels.com/photos/1743555/pexels-photo-1743555.jpeg?auto=compress&cs=tinysrgb&w=1280',
  },
  {
    id: 'studio',
    name: 'Studio',
    thumbnail: 'https://images.pexels.com/photos/1571463/pexels-photo-1571463.jpeg?auto=compress&cs=tinysrgb&w=200',
    value:     'https://images.pexels.com/photos/1571463/pexels-photo-1571463.jpeg?auto=compress&cs=tinysrgb&w=1280',
  },
  {
    id: 'conference',
    name: 'Conference Room',
    thumbnail: 'https://images.pexels.com/photos/416320/pexels-photo-416320.jpeg?auto=compress&cs=tinysrgb&w=200',
    value:     'https://images.pexels.com/photos/416320/pexels-photo-416320.jpeg?auto=compress&cs=tinysrgb&w=1280',
  },
  {
    id: 'apartment',
    name: 'Modern Apartment',
    thumbnail: 'https://images.pexels.com/photos/1571460/pexels-photo-1571460.jpeg?auto=compress&cs=tinysrgb&w=200',
    value:     'https://images.pexels.com/photos/1571460/pexels-photo-1571460.jpeg?auto=compress&cs=tinysrgb&w=1280',
  },
];

// ─── Types ────────────────────────────────────────────────────────────────────
interface Pt { x: number; y: number }
type Tri = [number, number, number];

// ─── MediaPipe Face Mesh: key landmark indices (out of 468) ──────────────────
// Face oval boundary
const FACE_OVAL: number[] = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
  397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
  172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
];
// Interior face landmarks for triangulation
const INTERIOR: number[] = [
  1, 2, 4, 5, 6, 8, 9, 11, 13, 14, 17, 18, 19, 20,
  33, 37, 40, 46, 52, 55, 57, 61, 65, 66, 70,
  78, 80, 82, 84, 87, 88, 91, 95, 96,
  105, 107, 109, 117, 118, 119, 121, 122, 123, 124, 125, 126, 127, 128,
  130, 132, 133, 136, 138, 139, 140, 141, 142, 143, 144, 145, 146,
  148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161,
  162, 163, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
  180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
  234, 246, 249, 251, 263, 267, 269, 270, 276,
  282, 283, 284, 285, 286, 288, 293, 295, 296, 297, 300,
  310, 311, 312, 314, 317, 318, 321, 323, 324, 325, 326, 327, 328, 329,
  330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343,
  344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357,
  358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371,
  372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385,
  386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399,
  400, 401, 402, 403, 404, 405,
];
const ALL_IDX = [...new Set([...FACE_OVAL, ...INTERIOR])].sort((a, b) => a - b);

// ─── Bowyer-Watson Delaunay triangulation ─────────────────────────────────────
function delaunay(pts: Pt[]): Tri[] {
  if (pts.length < 3) return [];
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of pts) { minX = Math.min(minX, p.x); minY = Math.min(minY, p.y); maxX = Math.max(maxX, p.x); maxY = Math.max(maxY, p.y); }
  const dx = (maxX - minX) * 10, dy = (maxY - minY) * 10;
  const n = pts.length;
  const aug = [...pts, { x: minX - dx, y: minY - dy * 3 }, { x: minX + (maxX - minX) / 2, y: maxY + dy * 3 }, { x: maxX + dx, y: minY - dy * 3 }];
  const sA = n, sB = n + 1, sC = n + 2;
  let tris: Tri[] = [[sA, sB, sC]];
  for (let i = 0; i < n; i++) {
    const bad: Tri[] = [];
    for (const t of tris) { if (inCC(aug[t[0]], aug[t[1]], aug[t[2]], aug[i])) bad.push(t); }
    const edgeCnt = new Map<string, number>();
    for (const t of bad) {
      for (const [a, b] of [[t[0],t[1]],[t[1],t[2]],[t[2],t[0]]]) {
        const k = a < b ? `${a},${b}` : `${b},${a}`;
        edgeCnt.set(k, (edgeCnt.get(k) ?? 0) + 1);
      }
    }
    tris = tris.filter(t => !bad.includes(t));
    for (const [k, c] of edgeCnt) if (c === 1) { const [a, b] = k.split(',').map(Number); tris.push([a, b, i]); }
  }
  return tris.filter(t => t[0] < n && t[1] < n && t[2] < n);
}

function inCC(a: Pt, b: Pt, c: Pt, p: Pt): boolean {
  const ax = a.x - p.x, ay = a.y - p.y;
  const bx = b.x - p.x, by = b.y - p.y;
  const cx = c.x - p.x, cy = c.y - p.y;
  return ax * (by * (cx*cx + cy*cy) - cy * (bx*bx + by*by))
       - ay * (bx * (cx*cx + cy*cy) - cx * (bx*bx + by*by))
       + (ax*ax + ay*ay) * (bx*cy - by*cx) > 0;
}

// ─── Affine from 3 src points → 3 dst points (maps src-coords to dst-canvas) ─
function affine3pt(
  src: [Pt, Pt, Pt],
  dst: [Pt, Pt, Pt],
): [number, number, number, number, number, number] | null {
  const [s0, s1, s2] = src;
  const [d0, d1, d2] = dst;
  const det = s0.x*(s1.y-s2.y) - s0.y*(s1.x-s2.x) + (s1.x*s2.y - s2.x*s1.y);
  if (Math.abs(det) < 1e-10) return null;
  const inv = 1 / det;
  const a = inv*(d0.x*(s1.y-s2.y) + d1.x*(s2.y-s0.y) + d2.x*(s0.y-s1.y));
  const c = inv*(d0.x*(s2.x-s1.x) + d1.x*(s0.x-s2.x) + d2.x*(s1.x-s0.x));
  const e = inv*(d0.x*(s1.x*s2.y-s2.x*s1.y) + d1.x*(s2.x*s0.y-s0.x*s2.y) + d2.x*(s0.x*s1.y-s1.x*s0.y));
  const b = inv*(d0.y*(s1.y-s2.y) + d1.y*(s2.y-s0.y) + d2.y*(s0.y-s1.y));
  const d = inv*(d0.y*(s2.x-s1.x) + d1.y*(s0.x-s2.x) + d2.y*(s1.x-s0.x));
  const f = inv*(d0.y*(s1.x*s2.y-s2.x*s1.y) + d1.y*(s2.x*s0.y-s0.x*s2.y) + d2.y*(s0.x*s1.y-s1.x*s0.y));
  return [a, b, c, d, e, f];
}

// ─── Suppress TF console noise ────────────────────────────────────────────────
const suppressTFLogs = () => {
  const orig = console.warn.bind(console);
  console.warn = (...args) => { if (String(args[0]).includes('tf')) return; orig(...args); };
};


export function useFaceTransform(): UseFaceTransformReturn {
  const [processedStream, setProcessedStream]               = useState<MediaStream | null>(null);
  const [isProcessing, setIsProcessing]                     = useState(false);
  const [statusMessage, setStatusMessage]                   = useState('Camera Ready');
  const [transformationSettings, setTransformationSettings] = useState<TransformationSettings>({
    enabled: false, referenceVideo: null, background: '',
  });
  const [referenceVideo, setReferenceVideo] = useState<HTMLVideoElement | null>(null);

  const outputCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const hostVideoRef    = useRef<HTMLVideoElement | null>(null);
  const animFrameRef    = useRef<number | null>(null);
  const bgImgRef        = useRef<HTMLImageElement | null>(null);

  // TF.js face landmark detectors
  const detectorRef     = useRef<faceLandmarksDetection.FaceLandmarksDetector | null>(null);

  // MediaPipe Selfie Segmentation (CDN)
  const selfieSegRef    = useRef<any>(null);
  const segResultRef    = useRef<any>(null);

  // Landmark caches (pixel coords, updated async)
  const hostLmsRef      = useRef<Pt[] | null>(null);
  const refLmsRef       = useRef<Pt[] | null>(null);
  // Cached triangulation of reference landmarks (rebuild when ref changes)
  const triCacheRef     = useRef<Tri[] | null>(null);
  const prevRefLmsRef   = useRef<Pt[] | null>(null);

  // Reference frame canvas
  const refCanvasRef    = useRef<HTMLCanvasElement | null>(null);

  // Busy flags for async detection
  const hostBusyRef     = useRef(false);
  const refBusyRef      = useRef(false);
  const segBusyRef      = useRef(false);
  const frameRef        = useRef(0);

  const settingsRef    = useRef(transformationSettings);
  const refVideoRef    = useRef<HTMLVideoElement | null>(null);
  const currentBgRef   = useRef('');
  const statusCacheRef = useRef('');

  // Only call setStatusMessage when text actually changes (avoids 30fps re-renders)
  const setStatus = useCallback((msg: string) => {
    if (statusCacheRef.current !== msg) {
      statusCacheRef.current = msg;
      setStatusMessage(msg);
    }
  }, []);

  useEffect(() => { settingsRef.current = transformationSettings; }, [transformationSettings]);
  useEffect(() => { refVideoRef.current = referenceVideo; },         [referenceVideo]);

  // ── Script loader for selfie segmentation ────────────────────────────────
  const loadScript = useCallback((id: string, src: string): Promise<void> =>
    new Promise((res, rej) => {
      if (document.getElementById(id)) { res(); return; }
      const s = Object.assign(document.createElement('script'), { id, src, crossOrigin: 'anonymous' });
      s.onload = () => res(); s.onerror = () => rej(new Error(`Failed: ${src}`));
      document.head.appendChild(s);
    }), []);

  // ── Init selfie segmentation ──────────────────────────────────────────────
  const initSelfie = useCallback(async () => {
    await loadScript('mp-selfie', 'https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/selfie_segmentation.js');
    const SS = (window as any).SelfieSegmentation;
    if (!SS) return;
    const seg = new SS({ locateFile: (f: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${f}` });
    seg.setOptions({ modelSelection: 1, selfieMode: false });
    seg.onResults((r: any) => { segResultRef.current = r; });
    selfieSegRef.current = seg;
  }, [loadScript]);

  // ── Init TF.js face landmark detector ────────────────────────────────────
  const initDetector = useCallback(async () => {
    suppressTFLogs();
    setStatus('Loading face-tracking model...');
    const detector = await faceLandmarksDetection.createDetector(
      faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
      { runtime: 'tfjs', maxFaces: 1, refineLandmarks: false },
    );
    detectorRef.current = detector;
    setStatus('Camera Ready');
  }, []);

  // ── Main init ─────────────────────────────────────────────────────────────
  const initializeTransform = useCallback(async (stream: MediaStream) => {
    if (hostVideoRef.current) return;
    setIsProcessing(true);
    setStatus('Starting camera...');

    const vid = document.createElement('video');
    vid.srcObject = stream; vid.playsInline = true; vid.muted = true;
    try { await vid.play(); } catch { setStatus('Camera Error'); setIsProcessing(false); return; }
    hostVideoRef.current = vid;

    const out = document.createElement('canvas');
    out.width = 1280; out.height = 720;
    outputCanvasRef.current = out;

    const refC = document.createElement('canvas');
    refC.width = 640; refC.height = 360;
    refCanvasRef.current = refC;

    const outStream = out.captureStream(30);
    stream.getAudioTracks().forEach(t => outStream.addTrack(t));
    setProcessedStream(outStream);

    // Start selfie segmentation first (fast CDN load)
    initSelfie().catch(err => console.warn('Selfie seg failed:', err));

    // Start face landmark detector (TF.js, slower)
    initDetector().catch(err => {
      console.warn('Face detector failed:', err);
      setStatus('Face tracking unavailable');
    });

    startRenderLoop();
    setIsProcessing(false);
  }, [initSelfie, initDetector]);

  // ── Detection helpers ─────────────────────────────────────────────────────
  const detectFace = useCallback(async (
    el: HTMLVideoElement | HTMLCanvasElement,
    canvasW: number,
    canvasH: number,
  ): Promise<Pt[] | null> => {
    const det = detectorRef.current;
    if (!det) return null;
    try {
      const faces = await det.estimateFaces(el, { flipHorizontal: false });
      if (!faces.length) return null;
      const kps = faces[0].keypoints;
      // Map the 468 keypoints; pick our ALL_IDX subset
      const all = kps.map(k => ({ x: k.x * (canvasW / (el instanceof HTMLVideoElement ? el.videoWidth || canvasW : el.width || canvasW)), y: k.y * (canvasH / (el instanceof HTMLVideoElement ? el.videoHeight || canvasH : el.height || canvasH)) }));
      return ALL_IDX.map(i => all[i] ?? { x: 0, y: 0 });
    } catch { return null; }
  }, []);

  // ── Render loop ───────────────────────────────────────────────────────────
  const startRenderLoop = useCallback(() => {
    const tick = async () => {
      const vid    = hostVideoRef.current;
      const refVid = refVideoRef.current;
      const out    = outputCanvasRef.current;
      const refC   = refCanvasRef.current;
      const s      = settingsRef.current;

      frameRef.current++;
      const frame = frameRef.current;

      if (vid && out && vid.readyState >= 2) {
        // Selfie segmentation every frame
        if (!segBusyRef.current && selfieSegRef.current) {
          segBusyRef.current = true;
          selfieSegRef.current.send({ image: vid })
            .then(() => { segBusyRef.current = false; })
            .catch(() => { segBusyRef.current = false; });
        }

        // Host face detection every 2 frames (when transformation active)
        if (s.enabled && frame % 2 === 0 && !hostBusyRef.current && detectorRef.current) {
          hostBusyRef.current = true;
          detectFace(vid, 1280, 720).then(pts => {
            if (pts) hostLmsRef.current = pts;
            hostBusyRef.current = false;
          });
        }

        // Reference face detection every 4 frames (when transformation active + ref video exists)
        if (s.enabled && frame % 4 === 0 && !refBusyRef.current && refVid && refVid.readyState >= 2 && detectorRef.current) {
          refBusyRef.current = true;
          // Capture ref frame to canvas (smaller for speed)
          const rCtx = refC?.getContext('2d');
          if (rCtx && refC) rCtx.drawImage(refVid, 0, 0, refC.width, refC.height);

          detectFace(refC!, refC!.width, refC!.height).then(pts => {
            if (pts) {
              // Scale from refC coords back to output canvas coords
              const scaleX = 1280 / refC!.width;
              const scaleY = 720  / refC!.height;
              refLmsRef.current = pts.map(p => ({ x: p.x * scaleX, y: p.y * scaleY }));
              triCacheRef.current = null; // invalidate triangulation cache
            }
            refBusyRef.current = false;
          });
        }

        renderFrame();
      }

      animFrameRef.current = requestAnimationFrame(tick);
    };

    animFrameRef.current = requestAnimationFrame(tick);
  }, [detectFace]);

  // ── Frame renderer ────────────────────────────────────────────────────────
  const renderFrame = useCallback(() => {
    const out     = outputCanvasRef.current;
    const vid     = hostVideoRef.current;
    const refC    = refCanvasRef.current;
    const seg     = segResultRef.current;
    const bgImg   = bgImgRef.current;
    const bgVal   = currentBgRef.current;
    const s       = settingsRef.current;
    const hLms    = hostLmsRef.current;
    const rLms    = refLmsRef.current;

    if (!out || !vid) return;
    const ctx = out.getContext('2d');
    if (!ctx) return;
    const W = out.width, H = out.height;

    ctx.clearRect(0, 0, W, H);

    // 1 ▸ Background
    if (bgVal && bgImg && bgImg.complete && bgImg.naturalWidth > 0) {
      ctx.drawImage(bgImg, 0, 0, W, H);
    } else {
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, W, H);
    }

    // 2 ▸ Person over background
    if (seg?.segmentationMask && vid.readyState >= 2) {
      const personOff = new OffscreenCanvas(W, H);
      const pCtx = personOff.getContext('2d') as OffscreenCanvasRenderingContext2D | null;
      if (pCtx) {
        // Draw live camera frame
        pCtx.drawImage(seg.image, 0, 0, W, H);

        // Face swap if enabled and we have both sets of landmarks
        if (s.enabled && hLms && rLms && refC) {
          applyFaceSwap(pCtx, refC, hLms, rLms, W, H);
        }

        // Knock out background → keep only person
        pCtx.globalCompositeOperation = 'destination-in';
        pCtx.drawImage(seg.segmentationMask, 0, 0, W, H);
        pCtx.globalCompositeOperation = 'source-over';

        ctx.drawImage(personOff, 0, 0);
      }
    } else if (vid.readyState >= 2) {
      ctx.drawImage(vid, 0, 0, W, H);
    }

    // Status (deduplicated via setStatus to avoid re-renders every frame)
    if (s.enabled && hLms && rLms)        setStatus('Face Tracking Active');
    else if (s.enabled && !detectorRef.current) setStatus('Loading face model...');
    else if (s.enabled)                   setStatus('Detecting face...');
    else if (bgVal && bgImg?.complete)    setStatus('Background Active');
    else                                  setStatus('Camera Ready');
  }, [setStatus]);

  // ── Triangulated face swap ────────────────────────────────────────────────
  const applyFaceSwap = (
    dstCtx: OffscreenCanvasRenderingContext2D,
    srcCanvas: HTMLCanvasElement,
    hLms: Pt[],
    rLms: Pt[],
    W: number,
    H: number,
  ) => {
    // Rebuild Delaunay triangulation only when ref landmarks change
    if (!triCacheRef.current || prevRefLmsRef.current !== rLms) {
      triCacheRef.current = delaunay(rLms);
      prevRefLmsRef.current = rLms;
    }
    const tris = triCacheRef.current;
    if (!tris.length) return;

    // ── Off-screen canvas: draw each warped triangle from ref → host ──────
    const faceOff = new OffscreenCanvas(W, H);
    const fCtx   = faceOff.getContext('2d') as OffscreenCanvasRenderingContext2D | null;
    if (!fCtx) return;

    // Scale ref canvas coords back to W×H for the drawImage call
    const refScaleX = W / srcCanvas.width;
    const refScaleY = H / srcCanvas.height;

    for (const [i0, i1, i2] of tris) {
      const rPts: [Pt, Pt, Pt] = [
        { x: rLms[i0].x / refScaleX, y: rLms[i0].y / refScaleY },
        { x: rLms[i1].x / refScaleX, y: rLms[i1].y / refScaleY },
        { x: rLms[i2].x / refScaleX, y: rLms[i2].y / refScaleY },
      ];
      const hPts: [Pt, Pt, Pt] = [hLms[i0], hLms[i1], hLms[i2]];

      // Skip tiny / degenerate triangles
      const area = Math.abs(
        (hPts[1].x - hPts[0].x) * (hPts[2].y - hPts[0].y) -
        (hPts[2].x - hPts[0].x) * (hPts[1].y - hPts[0].y),
      );
      if (area < 2) continue;

      const M = affine3pt(rPts, hPts);
      if (!M) continue;

      fCtx.save();
      fCtx.beginPath();
      fCtx.moveTo(hPts[0].x, hPts[0].y);
      fCtx.lineTo(hPts[1].x, hPts[1].y);
      fCtx.lineTo(hPts[2].x, hPts[2].y);
      fCtx.closePath();
      fCtx.clip();
      fCtx.setTransform(M[0], M[1], M[2], M[3], M[4], M[5]);
      fCtx.drawImage(srcCanvas, 0, 0, srcCanvas.width, srcCanvas.height);
      fCtx.restore();
    }

    // ── Soft face-oval mask → feathered edge ──────────────────────────────
    const maskOff = new OffscreenCanvas(W, H);
    const mCtx   = maskOff.getContext('2d') as OffscreenCanvasRenderingContext2D | null;
    if (mCtx) {
      const ovalPts = FACE_OVAL
        .map(mpIdx => { const pos = ALL_IDX.indexOf(mpIdx); return pos >= 0 ? hLms[pos] : null; })
        .filter((p): p is Pt => p !== null);

      if (ovalPts.length > 3) {
        const cx = ovalPts.reduce((s, p) => s + p.x, 0) / ovalPts.length;
        const cy = ovalPts.reduce((s, p) => s + p.y, 0) / ovalPts.length;
        const maxR = Math.max(...ovalPts.map(p => Math.hypot(p.x - cx, p.y - cy)));

        const grad = mCtx.createRadialGradient(cx, cy, maxR * 0.45, cx, cy, maxR * 1.05);
        grad.addColorStop(0,    'rgba(0,0,0,1)');
        grad.addColorStop(0.75, 'rgba(0,0,0,0.92)');
        grad.addColorStop(1,    'rgba(0,0,0,0)');

        mCtx.beginPath();
        mCtx.moveTo(ovalPts[0].x, ovalPts[0].y);
        for (let i = 1; i < ovalPts.length; i++) {
          const prev = ovalPts[i - 1], curr = ovalPts[i];
          mCtx.quadraticCurveTo(prev.x, prev.y, (prev.x + curr.x) / 2, (prev.y + curr.y) / 2);
        }
        mCtx.closePath();
        mCtx.fillStyle = grad;
        mCtx.fill();

        fCtx!.globalCompositeOperation = 'destination-in';
        fCtx!.drawImage(maskOff, 0, 0);
        fCtx!.globalCompositeOperation = 'source-over';
      }
    }

    // ── Composite warped face onto person canvas ──────────────────────────
    dstCtx.drawImage(faceOff, 0, 0);
  };

  // ── Background update ─────────────────────────────────────────────────────
  const updateBackground = useCallback((backgroundId: string) => {
    const opt   = backgroundOptions.find(o => o.id === backgroundId);
    const bgVal = opt?.value ?? '';
    currentBgRef.current = bgVal;
    setTransformationSettings(prev => ({ ...prev, background: bgVal }));
    if (!bgVal) { bgImgRef.current = null; setStatus('Camera Ready'); return; }
    setStatus('Loading background...');
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload  = () => { bgImgRef.current = img; setStatus('Background Active'); };
    img.onerror = () => { bgImgRef.current = null; setStatus('Background load failed'); };
    img.src = bgVal;
  }, [setStatus]);

  // ── Cleanup ───────────────────────────────────────────────────────────────
  const cleanup = useCallback(() => {
    if (animFrameRef.current) { cancelAnimationFrame(animFrameRef.current); animFrameRef.current = null; }
    try { selfieSegRef.current?.close(); } catch { /* noop */ }
    selfieSegRef.current = null;
    detectorRef.current?.dispose?.();
    detectorRef.current = null;
    if (hostVideoRef.current) { hostVideoRef.current.pause(); hostVideoRef.current.srcObject = null; hostVideoRef.current = null; }
    bgImgRef.current = null; outputCanvasRef.current = null; refCanvasRef.current = null;
    hostLmsRef.current = null; refLmsRef.current = null; triCacheRef.current = null;
    segResultRef.current = null; currentBgRef.current = ''; refVideoRef.current = null;
    hostBusyRef.current = false; refBusyRef.current = false; segBusyRef.current = false;
    frameRef.current = 0;
    statusCacheRef.current = '';
    setProcessedStream(null); setIsProcessing(false); setStatusMessage('Camera Ready');
  }, []);

  useEffect(() => () => { cleanup(); }, [cleanup]);

  return {
    processedStream, transformationSettings, setTransformationSettings,
    referenceVideo, setReferenceVideo, backgroundOptions, isProcessing,
    statusMessage, initializeTransform, updateBackground, cleanup,
  };
}

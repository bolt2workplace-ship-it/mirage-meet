import { createCanvas, createImageData, loadImage } from 'canvas';
import { InferenceSession, Tensor } from 'onnxruntime-node';
import { createWriteStream, existsSync, mkdirSync, statSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { pipeline } from 'stream/promises';
import https from 'https';
import http from 'http';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const MODELS_DIR = join(__dirname, 'models');
if (!existsSync(MODELS_DIR)) mkdirSync(MODELS_DIR, { recursive: true });

const MODEL_URLS = {
  det: 'https://huggingface.co/Aitrepreneur/insightface/resolve/main/models/buffalo_l/det_10g.onnx',
  arcface: 'https://huggingface.co/Aitrepreneur/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx',
  swapper: 'https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx',
};

const MODEL_PATHS = {
  det: join(MODELS_DIR, 'det_10g.onnx'),
  arcface: join(MODELS_DIR, 'w600k_r50.onnx'),
  swapper: join(MODELS_DIR, 'inswapper_128.onnx'),
};

const MODEL_SIZES = {
  det: 16_900_000,      // ~16MB
  arcface: 166_000_000, // ~166MB
  swapper: 554_000_000, // ~554MB
};

let sessions = { det: null, arcface: null, swapper: null };
let swapperEmap = null; // internal embedding map from inswapper ONNX
let initState = 'idle'; // idle | downloading | loading | ready | error
let initError = null;
let downloadProgress = {};

export function getInitState() {
  return { state: initState, error: initError, progress: downloadProgress };
}

// Follow redirects and stream to file
function downloadFile(url, dest, label) {
  return new Promise((resolve, reject) => {
    const attempt = (u) => {
      const mod = u.startsWith('https') ? https : http;
      const req = mod.get(u, { headers: { 'User-Agent': 'Mozilla/5.0' } }, (res) => {
        if (res.statusCode === 301 || res.statusCode === 302 || res.statusCode === 307) {
          return attempt(res.headers.location);
        }
        if (res.statusCode !== 200) {
          return reject(new Error(`HTTP ${res.statusCode} for ${label}`));
        }
        const total = parseInt(res.headers['content-length'] || '0', 10);
        let received = 0;
        const ws = createWriteStream(dest);
        res.on('data', (chunk) => {
          received += chunk.length;
          if (total > 0) {
            downloadProgress[label] = Math.round((received / total) * 100);
          }
        });
        res.pipe(ws);
        ws.on('finish', () => { downloadProgress[label] = 100; resolve(); });
        ws.on('error', reject);
      });
      req.on('error', reject);
    };
    attempt(url);
  });
}

async function ensureModel(key) {
  const path = MODEL_PATHS[key];
  if (existsSync(path)) {
    const stat = statSync(path);
    if (stat.size > MODEL_SIZES[key] * 0.9) {
      console.log(`[FaceSwap] Model ${key} already cached (${(stat.size / 1e6).toFixed(1)}MB)`);
      downloadProgress[key] = 100;
      return;
    }
    console.log(`[FaceSwap] Cached ${key} looks incomplete (${stat.size} bytes), re-downloading`);
  }
  console.log(`[FaceSwap] Downloading ${key} from HuggingFace...`);
  downloadProgress[key] = 0;
  await downloadFile(MODEL_URLS[key], path, key);
  console.log(`[FaceSwap] Downloaded ${key}`);
}

// ArcFace reference landmarks for 112x112 crop
const ARCFACE_DST = Float32Array.from([
  38.2946, 51.6963,
  73.5318, 51.5014,
  56.0252, 71.7366,
  41.5493, 92.3655,
  70.7299, 92.2041,
]);

// Compute similarity transform M (2x3) that maps src 5 pts → ARCFACE_DST
// Returns flat [a, b, tx, c, d, ty]
function estimateNorm(srcKps, imageSize = 112) {
  // Destination is ARCFACE_DST scaled for imageSize
  const ratio = imageSize / 112.0;
  const dx = imageSize % 128 === 0 ? 8.0 * ratio : 0;
  const dst = new Float64Array(10);
  for (let i = 0; i < 5; i++) {
    dst[i * 2]     = ARCFACE_DST[i * 2]     * ratio + dx;
    dst[i * 2 + 1] = ARCFACE_DST[i * 2 + 1] * ratio;
  }

  // src and dst are arrays of 5 [x,y] pairs
  // Similarity transform: dst = M * [src; 1]
  // Solve least-squares 2x3 using 5 correspondence points
  // Build design matrix A (10x4) and target b (10x1) for [a, b, tx, c, d, ty]
  // Using the Umeyama method for similarity transform

  // Mean src and dst
  let sxMean = 0, syMean = 0, dxMean = 0, dyMean = 0;
  for (let i = 0; i < 5; i++) {
    sxMean += srcKps[i * 2]; syMean += srcKps[i * 2 + 1];
    dxMean += dst[i * 2];    dyMean += dst[i * 2 + 1];
  }
  sxMean /= 5; syMean /= 5; dxMean /= 5; dyMean /= 5;

  let ss = 0, sxy = 0;
  for (let i = 0; i < 5; i++) {
    const sx = srcKps[i * 2] - sxMean, sy = srcKps[i * 2 + 1] - syMean;
    const dx2 = dst[i * 2] - dxMean, dy2 = dst[i * 2 + 1] - dyMean;
    ss += sx * sx + sy * sy;
    sxy += sx * dx2 + sy * dy2;
  }
  let syx = 0;
  for (let i = 0; i < 5; i++) {
    const sx = srcKps[i * 2] - sxMean, sy = srcKps[i * 2 + 1] - syMean;
    const dy2 = dst[i * 2 + 1] - dyMean;
    const dx2 = dst[i * 2] - dxMean;
    syx += sx * dy2 - sy * dx2;
  }

  const scale = ss > 0 ? Math.sqrt(sxy * sxy + syx * syx) / ss : 1;
  const cos_a = sxy / (ss * scale + 1e-10);
  const sin_a = syx / (ss * scale + 1e-10);

  const a = scale * cos_a;
  const b = -scale * sin_a;
  const tx = dxMean - a * sxMean - b * syMean;
  const c = scale * sin_a;
  const d = scale * cos_a;
  const ty = dyMean - c * sxMean - d * syMean;

  return [a, b, tx, c, d, ty];
}

// Warp image using 2x3 affine matrix, output is size x size RGBA pixel array
function warpAffine(pixelData, srcW, srcH, M, outSize) {
  // Invert M (2x3 affine) → M_inv
  const [a, b, tx, c, d, ty] = M;
  const det = a * d - b * c;
  if (Math.abs(det) < 1e-10) return new Uint8ClampedArray(outSize * outSize * 4);
  const ia = d / det, ib = -b / det, ic = -c / det, id = a / det;
  const itx = (b * ty - d * tx) / det;
  const ity = (c * tx - a * ty) / det;

  const out = new Uint8ClampedArray(outSize * outSize * 4);
  for (let y = 0; y < outSize; y++) {
    for (let x = 0; x < outSize; x++) {
      // Map output pixel back to source
      const sx = ia * x + ib * y + itx;
      const sy = ic * x + id * y + ity;
      const ix = Math.round(sx), iy = Math.round(sy);
      const di = (y * outSize + x) * 4;
      if (ix >= 0 && ix < srcW && iy >= 0 && iy < srcH) {
        const si = (iy * srcW + ix) * 4;
        out[di]     = pixelData[si];
        out[di + 1] = pixelData[si + 1];
        out[di + 2] = pixelData[si + 2];
        out[di + 3] = 255;
      }
    }
  }
  return out;
}

// Warp using inverse matrix (forward warp)
function warpAffineInverse(pixelData, srcW, srcH, M_inv, outW, outH) {
  const [a, b, tx, c, d, ty] = M_inv;
  const out = new Uint8ClampedArray(outW * outH * 4);
  for (let y = 0; y < outH; y++) {
    for (let x = 0; x < outW; x++) {
      const sx = Math.round(a * x + b * y + tx);
      const sy = Math.round(c * x + d * y + ty);
      const di = (y * outW + x) * 4;
      if (sx >= 0 && sx < srcW && sy >= 0 && sy < srcH) {
        const si = (sy * srcW + sx) * 4;
        out[di]     = pixelData[si];
        out[di + 1] = pixelData[si + 1];
        out[di + 2] = pixelData[si + 2];
        out[di + 3] = 255;
      }
    }
  }
  return out;
}

// Invert 2x3 affine transform
function invertAffine(M) {
  const [a, b, tx, c, d, ty] = M;
  const det = a * d - b * c;
  if (Math.abs(det) < 1e-10) return [1, 0, 0, 0, 1, 0];
  const ia = d / det, ib = -b / det, ic = -c / det, id2 = a / det;
  return [ia, ib, (b * ty - d * tx) / det, ic, id2, (c * tx - a * ty) / det];
}

// Convert RGBA pixel array (W x H) to CHW float32 blob for ONNX
// normalize = 'arcface': (x - 127.5) / 128, channels: RGB
// normalize = 'swapper': x / 255, channels: RGB (NCHW)
function pixelsToBlob(pixels, w, h, normalize) {
  const n = w * h;
  const blob = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    const r = pixels[i * 4], g = pixels[i * 4 + 1], b = pixels[i * 4 + 2];
    if (normalize === 'arcface') {
      blob[i]         = (r - 127.5) / 128.0;
      blob[n + i]     = (g - 127.5) / 128.0;
      blob[2 * n + i] = (b - 127.5) / 128.0;
    } else {
      blob[i]         = r / 255.0;
      blob[n + i]     = g / 255.0;
      blob[2 * n + i] = b / 255.0;
    }
  }
  return blob;
}

// 3xHxW float32 predictions → RGBA pixels
function predToPixels(pred, size) {
  const n = size * size;
  const out = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    out[i * 4]     = Math.max(0, Math.min(255, Math.round(pred[i]         * 255)));
    out[i * 4 + 1] = Math.max(0, Math.min(255, Math.round(pred[n + i]     * 255)));
    out[i * 4 + 2] = Math.max(0, Math.min(255, Math.round(pred[2 * n + i] * 255)));
    out[i * 4 + 3] = 255;
  }
  return out;
}

// Simple Gaussian blur on single-channel Float32Array
function gaussianBlur1ch(data, w, h, radius) {
  const k = radius * 2 + 1;
  const kernel = new Float32Array(k);
  const sigma = radius / 3;
  let sum = 0;
  for (let i = 0; i < k; i++) {
    const x = i - radius;
    kernel[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
    sum += kernel[i];
  }
  for (let i = 0; i < k; i++) kernel[i] /= sum;

  const tmp = new Float32Array(w * h);
  // horizontal
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v = 0;
      for (let ki = 0; ki < k; ki++) {
        const xi = Math.min(w - 1, Math.max(0, x + ki - radius));
        v += data[y * w + xi] * kernel[ki];
      }
      tmp[y * w + x] = v;
    }
  }
  const out = new Float32Array(w * h);
  // vertical
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v = 0;
      for (let ki = 0; ki < k; ki++) {
        const yi = Math.min(h - 1, Math.max(0, y + ki - radius));
        v += tmp[yi * w + x] * kernel[ki];
      }
      out[y * w + x] = v;
    }
  }
  return out;
}

// Decode SCRFD outputs to bounding boxes + 5 keypoints
// Following the original SCRFD decode logic
function decodeSCRFD(outputs, inputW, inputH, threshold = 0.5) {
  const strides = [8, 16, 32];
  const fmH = strides.map(s => Math.ceil(inputH / s));
  const fmW = strides.map(s => Math.ceil(inputW / s));
  const numAnchors = 2;

  let faces = [];
  let outIdx = 0;

  for (let si = 0; si < strides.length; si++) {
    const stride = strides[si];
    const fh = fmH[si], fw = fmW[si];
    const n = fh * fw * numAnchors;

    const scores = outputs[outIdx].data;       // [n]
    const bboxDeltas = outputs[outIdx + 1].data; // [n, 4]
    const kpsDeltas = outputs[outIdx + 2].data;  // [n, 10]
    outIdx += 3;

    // Build anchor centers
    for (let i = 0; i < n; i++) {
      const score = scores[i];
      if (score < threshold) continue;

      const ay = Math.floor(i / (fw * numAnchors));
      const ax = Math.floor((i % (fw * numAnchors)) / numAnchors);

      const cx = (ax + 0.5) * stride;
      const cy = (ay + 0.5) * stride;

      const x1 = cx - bboxDeltas[i * 4]     * stride;
      const y1 = cy - bboxDeltas[i * 4 + 1] * stride;
      const x2 = cx + bboxDeltas[i * 4 + 2] * stride;
      const y2 = cy + bboxDeltas[i * 4 + 3] * stride;

      const kps = new Float32Array(10);
      for (let k = 0; k < 5; k++) {
        kps[k * 2]     = cx + kpsDeltas[i * 10 + k * 2]     * stride;
        kps[k * 2 + 1] = cy + kpsDeltas[i * 10 + k * 2 + 1] * stride;
      }

      faces.push({ score, x1, y1, x2, y2, kps });
    }
  }

  // NMS
  faces.sort((a, b) => b.score - a.score);
  const keep = [];
  const used = new Set();
  for (let i = 0; i < faces.length; i++) {
    if (used.has(i)) continue;
    keep.push(faces[i]);
    for (let j = i + 1; j < faces.length; j++) {
      if (iou(faces[i], faces[j]) > 0.4) used.add(j);
    }
  }
  return keep;
}

function iou(a, b) {
  const ix1 = Math.max(a.x1, b.x1), iy1 = Math.max(a.y1, b.y1);
  const ix2 = Math.min(a.x2, b.x2), iy2 = Math.min(a.y2, b.y2);
  if (ix2 <= ix1 || iy2 <= iy1) return 0;
  const inter = (ix2 - ix1) * (iy2 - iy1);
  const aA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const bA = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (aA + bA - inter);
}

// Detect faces in an image (canvas ImageData → list of face objects)
async function detectFaces(imgData, W, H) {
  const det = sessions.det;
  if (!det) throw new Error('Detection model not loaded');

  const inputSize = 640;
  // Resize image to 640x640 with letterbox
  const scale = Math.min(inputSize / W, inputSize / H);
  const newW = Math.round(W * scale), newH = Math.round(H * scale);
  const padX = Math.floor((inputSize - newW) / 2), padY = Math.floor((inputSize - newH) / 2);

  const canvas = createCanvas(inputSize, inputSize);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = 'rgb(128,128,128)';
  ctx.fillRect(0, 0, inputSize, inputSize);

  const srcCanvas = createCanvas(W, H);
  const srcCtx = srcCanvas.getContext('2d');
  const id = createImageData(
    Uint8ClampedArray.from(imgData), W, H
  );
  srcCtx.putImageData(id, 0, 0);
  ctx.drawImage(srcCanvas, padX, padY, newW, newH);

  const padded = ctx.getImageData(0, 0, inputSize, inputSize).data;

  // BGR blob, mean subtract: (x - 127.5) / 128
  const blob = new Float32Array(3 * inputSize * inputSize);
  const n = inputSize * inputSize;
  for (let i = 0; i < n; i++) {
    blob[i]         = (padded[i * 4]     - 127.5) / 128.0; // R
    blob[n + i]     = (padded[i * 4 + 1] - 127.5) / 128.0; // G
    blob[2 * n + i] = (padded[i * 4 + 2] - 127.5) / 128.0; // B
  }

  const inputTensor = new Tensor('float32', blob, [1, 3, inputSize, inputSize]);
  const results = await det.run({ input: inputTensor });

  const outputValues = Object.values(results);
  const faces = decodeSCRFD(outputValues, inputSize, inputSize, 0.4);

  // Scale back to original image coordinates
  return faces.map(f => ({
    ...f,
    x1: (f.x1 - padX) / scale,
    y1: (f.y1 - padY) / scale,
    x2: (f.x2 - padX) / scale,
    y2: (f.y2 - padY) / scale,
    kps: f.kps.map((v, i) => i % 2 === 0 ? (v - padX) / scale : (v - padY) / scale),
  }));
}

// Get best (largest) face from detection results
function getBestFace(faces) {
  if (!faces.length) return null;
  return faces.reduce((best, f) => {
    const area = (f.x2 - f.x1) * (f.y2 - f.y1);
    const bestArea = (best.x2 - best.x1) * (best.y2 - best.y1);
    return area > bestArea ? f : best;
  });
}

// Extract ArcFace embedding from an aligned face crop
async function getEmbedding(pixelData, W, H, kps) {
  const arcfaceSize = 112;
  const M = estimateNorm(kps, arcfaceSize);
  const cropped = warpAffine(pixelData, W, H, M, arcfaceSize);
  const blob = pixelsToBlob(cropped, arcfaceSize, arcfaceSize, 'arcface');

  const inputTensor = new Tensor('float32', blob, [1, 3, arcfaceSize, arcfaceSize]);
  const arcInputName = sessions.arcface.inputNames[0];
  const arcOutputName = sessions.arcface.outputNames[0];
  const result = await sessions.arcface.run({ [arcInputName]: inputTensor });
  const emb = result[arcOutputName].data;

  // Normalize
  let norm = 0;
  for (let i = 0; i < emb.length; i++) norm += emb[i] * emb[i];
  norm = Math.sqrt(norm);
  const normed = new Float32Array(emb.length);
  for (let i = 0; i < emb.length; i++) normed[i] = emb[i] / norm;
  return normed;
}

// Load embedding map (emap) from inswapper ONNX model.
// The emap is a [512,512] initializer tensor; we parse it from raw protobuf.
async function loadSwapperEmap() {
  try {
    const { readFile } = await import('fs/promises');
    const buf = await readFile(MODEL_PATHS.swapper);

    const { Root } = await import('protobufjs');
    const root = Root.fromJSON({
      nested: {
        TensorProto: {
          fields: {
            dims:       { id: 1, rule: 'repeated', type: 'int64' },
            data_type:  { id: 2, rule: 'optional', type: 'int32' },
            float_data: { id: 4, rule: 'repeated', type: 'float', options: { packed: true } },
            name:       { id: 8, rule: 'optional', type: 'string' },
            raw_data:   { id: 9, rule: 'optional', type: 'bytes' },
          }
        },
        GraphProto: {
          fields: {
            node:        { id: 1, rule: 'repeated', type: 'NodeProto' },
            initializer: { id: 6, rule: 'repeated', type: 'TensorProto' },
          }
        },
        NodeProto: {
          fields: { output: { id: 2, rule: 'repeated', type: 'string' } }
        },
        ModelProto: {
          fields: {
            graph: { id: 7, rule: 'optional', type: 'GraphProto' },
          }
        },
      }
    });

    const ModelProto = root.lookupType('ModelProto');
    const model = ModelProto.decode(buf);
    const initializers = model.graph?.initializer ?? [];

    for (let i = initializers.length - 1; i >= 0; i--) {
      const t = initializers[i];
      const dims = Array.from(t.dims ?? []);
      if (dims.length === 2 && Number(dims[0]) === 512 && Number(dims[1]) === 512) {
        let emap = null;
        if (t.raw_data && t.raw_data.length >= 512 * 512 * 4) {
          const arr = new Float32Array(t.raw_data.buffer, t.raw_data.byteOffset, 512 * 512);
          emap = Float32Array.from(arr);
        } else if (t.float_data && t.float_data.length === 512 * 512) {
          emap = Float32Array.from(t.float_data);
        }
        if (emap) {
          console.log('[FaceSwap] emap loaded from initializer', i);
          return emap;
        }
      }
    }
    console.warn('[FaceSwap] emap not found in model, identity pass-through will be used');
    return null;
  } catch (e) {
    console.warn('[FaceSwap] Could not load emap:', e.message);
    return null;
  }
}

// Run face swap: source embedding + target crop → swapped face
async function runSwapper(targetPixels, targetW, targetH, targetKps, sourceEmbedding) {
  const swapSize = 128;
  const M = estimateNorm(targetKps, swapSize);
  const alignedPixels = warpAffine(targetPixels, targetW, targetH, M, swapSize);

  // Target blob: (pixel / 255) - 0.0, std 1.0 → just / 255
  const targetBlob = pixelsToBlob(alignedPixels, swapSize, swapSize, 'swapper');

  // Latent: normed_embedding @ emap, then re-normalize
  let latent;
  if (swapperEmap) {
    // Matrix multiply: [1,512] @ [512,512] → [1,512]
    const dim = 512;
    latent = new Float32Array(dim);
    for (let j = 0; j < dim; j++) {
      let sum = 0;
      for (let k = 0; k < dim; k++) {
        sum += sourceEmbedding[k] * swapperEmap[k * dim + j];
      }
      latent[j] = sum;
    }
    let norm = 0;
    for (let i = 0; i < dim; i++) norm += latent[i] * latent[i];
    norm = Math.sqrt(norm);
    for (let i = 0; i < dim; i++) latent[i] /= norm;
  } else {
    latent = sourceEmbedding;
  }

  const swapperInputNames = sessions.swapper.inputNames;
  const inputs = {
    [swapperInputNames[0]]: new Tensor('float32', targetBlob, [1, 3, swapSize, swapSize]),
    [swapperInputNames[1]]: new Tensor('float32', latent, [1, 512]),
  };

  const outName = sessions.swapper.outputNames[0];
  const result = await sessions.swapper.run(inputs);
  const pred = result[outName].data;

  return { pred, M, alignedPixels };
}

// Paste swapped face back onto full frame
function pasteBack(targetPixels, targetW, targetH, swappedPred, alignedPixels, M, swapSize) {
  const fakePx = predToPixels(swappedPred, swapSize);
  const M_inv = invertAffine(M);

  // Warp fake face back to full frame
  const warpedFake = warpAffineInverse(fakePx, swapSize, swapSize, M_inv, targetW, targetH);

  // Build mask: white where fake was, erode + blur edges
  const rawMask = new Float32Array(targetW * targetH);
  for (let i = 0; i < targetW * targetH; i++) {
    rawMask[i] = warpedFake[i * 4 + 3] > 0 ? 255 : 0;
  }

  // Erode by 5px
  const eroded = new Float32Array(targetW * targetH);
  const erR = 5;
  for (let y = 0; y < targetH; y++) {
    for (let x = 0; x < targetW; x++) {
      let min = 255;
      for (let dy = -erR; dy <= erR; dy++) {
        for (let dx = -erR; dx <= erR; dx++) {
          const nx = Math.min(targetW - 1, Math.max(0, x + dx));
          const ny = Math.min(targetH - 1, Math.max(0, y + dy));
          if (rawMask[ny * targetW + nx] < min) min = rawMask[ny * targetW + nx];
        }
      }
      eroded[y * targetW + x] = min;
    }
  }

  // Gaussian blur mask
  const blurR = Math.max(10, Math.floor(Math.sqrt(targetW * targetH / (640 * 480)) * 20));
  const blurredMask = gaussianBlur1ch(eroded, targetW, targetH, blurR);

  // Composite
  const out = new Uint8ClampedArray(targetPixels.length);
  for (let i = 0; i < targetW * targetH; i++) {
    const alpha = blurredMask[i] / 255;
    out[i * 4]     = Math.round(alpha * warpedFake[i * 4]     + (1 - alpha) * targetPixels[i * 4]);
    out[i * 4 + 1] = Math.round(alpha * warpedFake[i * 4 + 1] + (1 - alpha) * targetPixels[i * 4 + 1]);
    out[i * 4 + 2] = Math.round(alpha * warpedFake[i * 4 + 2] + (1 - alpha) * targetPixels[i * 4 + 2]);
    out[i * 4 + 3] = 255;
  }
  return out;
}

// ─── Public API ─────────────────────────────────────────────────────────────

let referenceEmbedding = null;

export async function initialize(onProgress) {
  if (initState === 'ready') return;
  if (initState === 'loading' || initState === 'downloading') return;

  try {
    initState = 'downloading';
    onProgress?.({ state: 'downloading', message: 'Downloading AI models (this takes a few minutes)...' });

    // Download models in parallel where possible
    await ensureModel('det');
    onProgress?.({ state: 'downloading', message: 'Downloading ArcFace identity model...', progress: downloadProgress });
    await ensureModel('arcface');
    onProgress?.({ state: 'downloading', message: 'Downloading InSwapper face synthesis model...', progress: downloadProgress });
    await ensureModel('swapper');

    initState = 'loading';
    onProgress?.({ state: 'loading', message: 'Loading ONNX sessions...' });

    const opts = { executionProviders: ['cpu'] };

    sessions.det     = await InferenceSession.create(MODEL_PATHS.det,     opts);
    onProgress?.({ state: 'loading', message: 'Loaded face detector...' });
    sessions.arcface = await InferenceSession.create(MODEL_PATHS.arcface,  opts);
    onProgress?.({ state: 'loading', message: 'Loaded ArcFace model...' });
    sessions.swapper = await InferenceSession.create(MODEL_PATHS.swapper,  opts);
    onProgress?.({ state: 'loading', message: 'Loaded InSwapper model...' });

    console.log('[FaceSwap] Swapper input names:', sessions.swapper.inputNames);
    console.log('[FaceSwap] Swapper output names:', sessions.swapper.outputNames);

    swapperEmap = await loadSwapperEmap();

    initState = 'ready';
    onProgress?.({ state: 'ready', message: 'AI transformation engine ready' });
    console.log('[FaceSwap] All models loaded and ready');
  } catch (err) {
    initState = 'error';
    initError = err.message;
    console.error('[FaceSwap] Initialization error:', err);
    onProgress?.({ state: 'error', message: err.message });
    throw err;
  }
}

export async function registerReferenceFace(imageBuffer) {
  if (initState !== 'ready') throw new Error(`Engine not ready: ${initState}`);

  const img = await loadImage(imageBuffer);
  const W = img.width, H = img.height;
  const canvas = createCanvas(W, H);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);
  const { data } = ctx.getImageData(0, 0, W, H);

  const faces = await detectFaces(data, W, H);
  const best = getBestFace(faces);
  if (!best) throw new Error('No face detected in reference image');

  referenceEmbedding = await getEmbedding(data, W, H, Array.from(best.kps));
  console.log('[FaceSwap] Reference face registered, embedding dim:', referenceEmbedding.length);
  return { success: true, bbox: { x1: best.x1, y1: best.y1, x2: best.x2, y2: best.y2 } };
}

export async function transformFrame(jpegBuffer) {
  if (initState !== 'ready') return null;
  if (!referenceEmbedding) return null;

  const img = await loadImage(jpegBuffer);
  const W = img.width, H = img.height;
  const canvas = createCanvas(W, H);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);
  const { data } = ctx.getImageData(0, 0, W, H);

  const faces = await detectFaces(data, W, H);
  const best = getBestFace(faces);
  if (!best) return null;

  const { pred, M, alignedPixels } = await runSwapper(data, W, H, Array.from(best.kps), referenceEmbedding);
  const resultPixels = pasteBack(data, W, H, pred, alignedPixels, M, 128);

  // Encode result to JPEG
  const outCanvas = createCanvas(W, H);
  const outCtx = outCanvas.getContext('2d');
  const outId = createImageData(resultPixels, W, H);
  outCtx.putImageData(outId, 0, 0);
  return outCanvas.toBuffer('image/jpeg', { quality: 0.85 });
}

export function clearReference() {
  referenceEmbedding = null;
}

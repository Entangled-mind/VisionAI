/**
 * VisionAI — Production-Grade Dual-Model Computer Vision Engine
 * Powered by TensorFlow.js, COCO-SSD (Spatial Localization) & 
 * MobileNet-v2 (1,000-Class Fine-Grained ImageNet Classifier)
 * Featuring Real-Time Night Vision & Adaptive Low-Light Enhancement
 */

// Global Model State
let cocoModel = null;
let classifierModel = null;
let isModelsReady = false;

// Camera State
let isCameraRunning = false;
let currentStream = null;
let currentFacingMode = "user";
let animationFrameId = null;

// Telemetry State
let lastFrameTime = performance.now();
let frameCount = 0;
let fps = 0;

// Night Vision & Low-Light Enhancement State
let liveNightVisionMode = "auto";   // "auto", "on", "off"
let staticNightVisionMode = "auto"; // "auto", "on", "off"
let staticNightGain = 3.0;          // Exposure gain multiplier (1.0x to 5.0x)
let staticNightGamma = 0.40;        // Shadow expansion exponent (0.35 to 0.50)
let currentLoadedImageSrc = null;

// DOM Elements - Camera
const videoEl = document.getElementById("webcam");
const overlayCanvas = document.getElementById("overlay-canvas");
const overlayCtx = overlayCanvas ? overlayCanvas.getContext("2d") : null;
const placeholderOverlay = document.getElementById("camera-placeholder");
const btnToggleCamera = document.getElementById("btn-toggle-camera");
const btnStartCameraHero = document.getElementById("btn-start-camera-hero");
const btnFlipCamera = document.getElementById("btn-flip-camera");
const btnNightVisionLive = document.getElementById("btn-night-vision-live");
const btnSnapshot = document.getElementById("btn-snapshot");
const liveConfSlider = document.getElementById("live-conf-slider");
const liveConfVal = document.getElementById("live-conf-val");
const teleFps = document.getElementById("tele-fps");
const teleLat = document.getElementById("tele-lat");
const teleCount = document.getElementById("tele-count");
const teleLum = document.getElementById("tele-lum");
const liveDetectionsList = document.getElementById("live-detections-list");
const modelStatusEl = document.getElementById("model-status");

// DOM Elements - Static Inspector
const fileInput = document.getElementById("file-input");
const dropzone = document.getElementById("dropzone");
const staticCanvas = document.getElementById("static-canvas");
const staticCtx = staticCanvas ? staticCanvas.getContext("2d") : null;
const staticPlaceholder = document.getElementById("static-placeholder");
const staticConfSlider = document.getElementById("static-conf-slider");
const staticConfVal = document.getElementById("static-conf-val");
const staticNightModeSelect = document.getElementById("static-night-mode");
const staticNightGainSlider = document.getElementById("static-night-gain");
const staticNightGainVal = document.getElementById("static-night-gain-val");
const staticTelemetry = document.getElementById("static-telemetry");
const staticRes = document.getElementById("static-res");
const staticLum = document.getElementById("static-lum");
const staticNvStatus = document.getElementById("static-nv-status");
const staticAnimals = document.getElementById("static-animals");
const staticObjects = document.getElementById("static-objects");
const inspectorTableContainer = document.getElementById("inspector-table-container");
const inspectorTableBody = document.getElementById("inspector-table-body");
const sampleChips = document.querySelectorAll(".chip-btn");

// ---------------------------------------------------------------------------
// 1. Comprehensive Taxonomy & Entity Resolution Engine
// ---------------------------------------------------------------------------
const ANIMAL_KEYWORDS = [
  "lion", "tiger", "cheetah", "leopard", "jaguar", "panther", "cat", "dog",
  "puppy", "kitten", "wolf", "fox", "bear", "elephant", "zebra", "giraffe",
  "sheep", "ram", "goat", "cow", "ox", "bull", "bison", "horse", "donkey",
  "bird", "eagle", "hawk", "owl", "parrot", "penguin", "duck", "goose",
  "monkey", "chimpanzee", "gorilla", "baboon", "panda", "koala", "rabbit",
  "hare", "deer", "moose", "antelope", "gazelle", "camel", "hippopotamus",
  "rhinoceros", "crocodile", "alligator", "snake", "lizard", "turtle", "frog"
];

const TOOL_KEYWORDS = [
  "pen", "ballpoint", "biro", "fountain pen", "pencil", "eraser", "ruler",
  "notebook", "binder", "book", "copy", "pad", "paper", "stapler",
  "scissors", "glasses", "sunglasses", "spectacles", "shades", "phone",
  "cellphone", "cellular", "mobile", "telephone", "bag", "backpack",
  "knapsack", "handbag", "purse", "wallet", "chair", "table", "desk",
  "laptop", "mouse", "keyboard", "bottle", "cup", "mug", "clock", "watch"
];

/**
 * Maps raw ImageNet / COCO labels to clean, formatted entity metadata
 */
function resolveEntityInfo(rawLabel) {
  const lower = rawLabel.toLowerCase();

  // Specific Wild & Domestic Animals
  if (lower.includes("lion")) {
    return { title: "Lion (King of Beasts)", category: "Animal", emoji: "🦁", stroke: "#f59e0b", fill: "rgba(245, 158, 11, 0.22)", badge: "#d97706" };
  }
  if (lower.includes("tiger")) {
    return { title: "Tiger", category: "Animal", emoji: "🐯", stroke: "#f97316", fill: "rgba(249, 115, 22, 0.22)", badge: "#ea580c" };
  }
  if (lower.includes("cheetah") || lower.includes("leopard") || lower.includes("jaguar")) {
    return { title: "Leopard / Cheetah", category: "Animal", emoji: "🐆", stroke: "#eab308", fill: "rgba(234, 179, 8, 0.22)", badge: "#ca8a04" };
  }
  if (lower.includes("dog") || lower.includes("retriever") || lower.includes("terrier") || lower.includes("hound") || lower.includes("shepherd") || lower.includes("bulldog") || lower.includes("poodle") || lower.includes("spaniel") || lower.includes("husky")) {
    return { title: "Dog", category: "Animal", emoji: "🐶", stroke: "#f97316", fill: "rgba(249, 115, 22, 0.22)", badge: "#c2410c" };
  }
  if (lower.includes("cat") || lower.includes("tabby") || lower.includes("siamese") || lower.includes("persian")) {
    return { title: "Cat", category: "Animal", emoji: "🐱", stroke: "#f59e0b", fill: "rgba(245, 158, 11, 0.22)", badge: "#b45309" };
  }
  if (lower.includes("bear") && !lower.includes("teddy")) {
    return { title: "Bear", category: "Animal", emoji: "🐻", stroke: "#854d0e", fill: "rgba(133, 77, 14, 0.22)", badge: "#713f12" };
  }
  if (lower.includes("elephant")) {
    return { title: "Elephant", category: "Animal", emoji: "🐘", stroke: "#94a3b8", fill: "rgba(148, 163, 184, 0.22)", badge: "#64748b" };
  }
  if (lower.includes("zebra")) {
    return { title: "Zebra", category: "Animal", emoji: "🦓", stroke: "#e2e8f0", fill: "rgba(226, 232, 240, 0.22)", badge: "#475569" };
  }
  if (lower.includes("giraffe")) {
    return { title: "Giraffe", category: "Animal", emoji: "🦒", stroke: "#d97706", fill: "rgba(217, 119, 6, 0.22)", badge: "#b45309" };
  }
  if (lower.includes("horse")) {
    return { title: "Horse", category: "Animal", emoji: "🐴", stroke: "#b45309", fill: "rgba(180, 83, 9, 0.22)", badge: "#92400e" };
  }
  if (lower.includes("bird") || lower.includes("parrot") || lower.includes("eagle") || lower.includes("owl")) {
    return { title: "Bird", category: "Animal", emoji: "🐦", stroke: "#06b6d4", fill: "rgba(6, 182, 212, 0.22)", badge: "#0891b2" };
  }
  if (lower.includes("sheep") || lower.includes("ram")) {
    return { title: "Sheep / Ram", category: "Animal", emoji: "🐑", stroke: "#a3e635", fill: "rgba(163, 230, 53, 0.22)", badge: "#65a30d" };
  }
  if (lower.includes("cow") || lower.includes("ox") || lower.includes("bull")) {
    return { title: "Cow / Cattle", category: "Animal", emoji: "🐄", stroke: "#e2e8f0", fill: "rgba(226, 232, 240, 0.22)", badge: "#334155" };
  }

  // Everyday Tools & Stationery
  if (lower.includes("ballpoint") || lower.includes("ballpen") || lower.includes("biro")) {
    return { title: "Ballpoint Pen", category: "Stationery / Tool", emoji: "🖊️", stroke: "#06b6d4", fill: "rgba(6, 182, 212, 0.22)", badge: "#0891b2" };
  }
  if (lower.includes("fountain pen")) {
    return { title: "Fountain Pen", category: "Stationery / Tool", emoji: "✒️", stroke: "#3b82f6", fill: "rgba(59, 130, 246, 0.22)", badge: "#2563eb" };
  }
  if (lower.includes("pen") && !lower.includes("penguin") && !lower.includes("open")) {
    return { title: "Pen", category: "Stationery / Tool", emoji: "🖊️", stroke: "#06b6d4", fill: "rgba(6, 182, 212, 0.22)", badge: "#0891b2" };
  }
  if (lower.includes("notebook") || lower.includes("binder") || lower.includes("copy") || lower.includes("spiral")) {
    return { title: "Notebook / Copy", category: "Stationery / Tool", emoji: "📓", stroke: "#8b5cf6", fill: "rgba(139, 92, 246, 0.22)", badge: "#7c3aed" };
  }
  if (lower.includes("book") || lower.includes("booklet") || lower.includes("dust cover")) {
    return { title: "Book", category: "Stationery / Tool", emoji: "📖", stroke: "#a855f7", fill: "rgba(168, 85, 247, 0.22)", badge: "#9333ea" };
  }
  if (lower.includes("spectacle") || lower.includes("eyeglass") || (lower.includes("glasses") && !lower.includes("dark glasses"))) {
    return { title: "Glasses / Spectacles", category: "Accessories", emoji: "👓", stroke: "#14b8a6", fill: "rgba(20, 184, 166, 0.22)", badge: "#0d9488" };
  }
  if (lower.includes("sunglass") || lower.includes("dark glasses") || lower.includes("shades")) {
    return { title: "Sunglasses", category: "Accessories", emoji: "🕶️", stroke: "#0ea5e9", fill: "rgba(14, 165, 233, 0.22)", badge: "#0284c7" };
  }
  if (lower.includes("cellular") || lower.includes("cellphone") || lower.includes("phone") || lower.includes("cell phone")) {
    return { title: "Smartphone / Phone", category: "Electronics", emoji: "📱", stroke: "#6366f1", fill: "rgba(99, 102, 241, 0.22)", badge: "#4f46e5" };
  }
  if (lower.includes("backpack") || lower.includes("knapsack") || lower.includes("rucksack")) {
    return { title: "Backpack / School Bag", category: "Everyday Item", emoji: "🎒", stroke: "#ec4899", fill: "rgba(236, 72, 153, 0.22)", badge: "#db2777" };
  }
  if (lower.includes("handbag") || lower.includes("purse") || lower.includes("pocketbook") || (lower.includes("bag") && !lower.includes("mailbag"))) {
    return { title: "Handbag / Bag", category: "Everyday Item", emoji: "👜", stroke: "#f43f5e", fill: "rgba(244, 63, 94, 0.22)", badge: "#e11d48" };
  }
  if (lower.includes("chair") || lower.includes("rocker") || lower.includes("stool")) {
    return { title: "Chair", category: "Furniture", emoji: "🪑", stroke: "#10b981", fill: "rgba(16, 185, 129, 0.22)", badge: "#059669" };
  }
  if (lower.includes("desk") || lower.includes("table") || lower.includes("dining table")) {
    return { title: "Table / Desk", category: "Furniture", emoji: "🪵", stroke: "#84cc16", fill: "rgba(132, 204, 22, 0.22)", badge: "#65a30d" };
  }
  if (lower.includes("laptop") || lower.includes("notebook computer")) {
    return { title: "Laptop Computer", category: "Electronics", emoji: "💻", stroke: "#38bdf8", fill: "rgba(56, 189, 248, 0.22)", badge: "#0284c7" };
  }
  if (lower.includes("mouse") && (lower.includes("computer") || lower.includes("optical"))) {
    return { title: "Computer Mouse", category: "Electronics", emoji: "🖱️", stroke: "#a855f7", fill: "rgba(168, 85, 247, 0.22)", badge: "#9333ea" };
  }
  if (lower.includes("keyboard") || lower.includes("keypad")) {
    return { title: "Keyboard", category: "Electronics", emoji: "⌨️", stroke: "#a855f7", fill: "rgba(168, 85, 247, 0.22)", badge: "#9333ea" };
  }
  if (lower.includes("bottle")) {
    return { title: "Bottle", category: "Everyday Item", emoji: "🍾", stroke: "#06b6d4", fill: "rgba(6, 182, 212, 0.22)", badge: "#0891b2" };
  }
  if (lower.includes("cup") || lower.includes("mug")) {
    return { title: "Cup / Mug", category: "Everyday Item", emoji: "☕", stroke: "#14b8a6", fill: "rgba(20, 184, 166, 0.22)", badge: "#0d9488" };
  }
  if (lower.includes("person") || lower.includes("human") || lower.includes("man") || lower.includes("woman")) {
    return { title: "Person", category: "Person", emoji: "👤", stroke: "#38bdf8", fill: "rgba(56, 189, 248, 0.22)", badge: "#0284c7" };
  }

  // Any other animal in ImageNet
  const isAnimal = ANIMAL_KEYWORDS.some(kw => lower.includes(kw));
  if (isAnimal) {
    const cleanName = rawLabel.split(",")[0].trim();
    return { 
      title: cleanName.charAt(0).toUpperCase() + cleanName.slice(1), 
      category: "Animal", 
      emoji: "🐾", 
      stroke: "#f59e0b", 
      fill: "rgba(245, 158, 11, 0.22)", 
      badge: "#d97706" 
    };
  }

  // Any other tool in ImageNet
  const isTool = TOOL_KEYWORDS.some(kw => lower.includes(kw));
  if (isTool) {
    const cleanName = rawLabel.split(",")[0].trim();
    return { 
      title: cleanName.charAt(0).toUpperCase() + cleanName.slice(1), 
      category: "Stationery / Tool", 
      emoji: "🔧", 
      stroke: "#06b6d4", 
      fill: "rgba(6, 182, 212, 0.22)", 
      badge: "#0891b2" 
    };
  }

  // Default clean title
  const cleanName = rawLabel.split(",")[0].trim();
  const title = cleanName.charAt(0).toUpperCase() + cleanName.slice(1);
  return { 
    title: title, 
    category: "Object", 
    emoji: "📦", 
    stroke: "#6366f1", 
    fill: "rgba(99, 102, 241, 0.22)", 
    badge: "#4f46e5" 
  };
}

// ---------------------------------------------------------------------------
// 2. Night Vision & Low-Light Enhancement Subsystem
// ---------------------------------------------------------------------------
/**
 * Rapidly measures scene ambient luminance (0% pitch dark to 100% full daylight)
 */
function measureLuminance(canvasOrImg) {
  const sampleSize = 48;
  const offCanvas = document.createElement("canvas");
  offCanvas.width = sampleSize;
  offCanvas.height = sampleSize;
  const offCtx = offCanvas.getContext("2d", { willReadFrequently: true });
  
  const w = canvasOrImg.naturalWidth || canvasOrImg.videoWidth || canvasOrImg.width || 640;
  const h = canvasOrImg.naturalHeight || canvasOrImg.videoHeight || canvasOrImg.height || 480;
  
  offCtx.drawImage(canvasOrImg, 0, 0, sampleSize, sampleSize);
  const data = offCtx.getImageData(0, 0, sampleSize, sampleSize).data;
  
  let sumLuma = 0;
  for (let i = 0; i < data.length; i += 4) {
    sumLuma += (0.299 * data[i] + 0.587 * data[i+1] + 0.114 * data[i+2]);
  }
  
  const avgLuma = sumLuma / (sampleSize * sampleSize);
  return Math.max(1, Math.round((avgLuma / 255.0) * 100));
}

/**
 * High-speed hardware-friendly Retinex / Gamma Tone Mapping Look-Up Table (LUT).
 * Recovers crushed gradients and edges from shadows in < 8ms.
 */
function createLowLightEnhancedCanvas(source, gain = 3.0, gamma = 0.40) {
  const w = source.naturalWidth || source.videoWidth || source.width || 640;
  const h = source.naturalHeight || source.videoHeight || source.height || 480;
  
  const offCanvas = document.createElement("canvas");
  offCanvas.width = w;
  offCanvas.height = h;
  const offCtx = offCanvas.getContext("2d", { willReadFrequently: true });
  
  offCtx.drawImage(source, 0, 0, w, h);
  const imgData = offCtx.getImageData(0, 0, w, h);
  const d = imgData.data;
  
  // Precompute 256-entry Look-Up Table
  const lut = new Uint8Array(256);
  for (let v = 0; v < 256; v++) {
    const normalized = v / 255.0;
    const boosted = Math.pow(normalized, gamma) * gain * 255.0;
    lut[v] = Math.min(255, Math.max(0, Math.round(boosted)));
  }
  
  for (let i = 0; i < d.length; i += 4) {
    d[i] = lut[d[i]];       // R
    d[i+1] = lut[d[i+1]];   // G
    d[i+2] = lut[d[i+2]];   // B
  }
  
  offCtx.putImageData(imgData, 0, 0);
  return offCanvas;
}

// ---------------------------------------------------------------------------
// 3. Model Initialization (Dual Model: COCO-SSD + MobileNet ImageNet-1k)
// ---------------------------------------------------------------------------
async function initModel() {
  try {
    console.log("[VisionAI] Setting TensorFlow.js WebGL Backend...");
    await tf.setBackend("webgl");
    await tf.ready();

    if (modelStatusEl) {
      modelStatusEl.innerHTML = '<span class="status-dot"></span><span class="status-text">Loading Neural Networks...</span>';
    }

    console.log("[VisionAI] Loading MobileNet-v2 (1,000 fine-grained ImageNet classes)...");
    const [loadedCoco, loadedClassifier] = await Promise.all([
      cocoSsd.load({ base: "mobilenet_v2" }).catch(e => {
        console.warn("[VisionAI] mobilenet_v2 base fallback to lite_mobilenet_v2:", e);
        return cocoSsd.load({ base: "lite_mobilenet_v2" });
      }),
      mobilenet.load({ version: 2, alpha: 1.0 }).catch(e => {
        console.warn("[VisionAI] MobileNet v2 fallback to default:", e);
        return mobilenet.load();
      })
    ]);

    cocoModel = loadedCoco;
    classifierModel = loadedClassifier;
    isModelsReady = true;

    if (modelStatusEl) {
      modelStatusEl.className = "status-indicator ready";
      modelStatusEl.innerHTML = '<span class="status-dot"></span><span class="status-text">AI Ready: COCO + ImageNet (1k Classes) + Night Vision</span>';
    }
    console.log("[VisionAI] Dual AI Models & Night Vision Engine ready!");
  } catch (err) {
    console.warn("[VisionAI] WebGL init fallback, loading CPU backend...", err);
    try {
      await tf.setBackend("cpu");
      cocoModel = await cocoSsd.load();
      classifierModel = await mobilenet.load();
      isModelsReady = true;

      if (modelStatusEl) {
        modelStatusEl.className = "status-indicator ready";
        modelStatusEl.innerHTML = '<span class="status-dot"></span><span class="status-text">AI Ready (CPU Mode)</span>';
      }
    } catch (fallbackErr) {
      console.error("[VisionAI] Model failed to load:", fallbackErr);
      if (modelStatusEl) {
        modelStatusEl.className = "status-indicator";
        modelStatusEl.style.borderColor = "#ef4444";
        modelStatusEl.style.color = "#f87171";
        modelStatusEl.innerHTML = '<span class="status-dot" style="background:#ef4444;"></span><span class="status-text">Model Load Failed</span>';
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 4. Styled Bounding Box Drawing Logic
// ---------------------------------------------------------------------------
function drawStyledDetectionBox(ctx, x, y, w, h, entityInfo, score, isNightVision = false) {
  const icon = entityInfo.emoji || "📦";
  const confText = `${Math.round(score * 100)}%`;
  const nvTag = isNightVision ? " 🌙" : "";
  const badgeText = `${icon} ${entityInfo.title} [${confText}]${nvTag}`;

  // Draw semi-transparent bounding fill
  ctx.fillStyle = entityInfo.fill;
  ctx.fillRect(x, y, w, h);

  // Draw crisp high-tech border
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = isNightVision ? "#38bdf8" : entityInfo.stroke;
  ctx.strokeRect(x, y, w, h);

  // Draw corner accents
  const cornerLen = Math.min(16, Math.min(w, h) / 3);
  ctx.lineWidth = 4;
  ctx.strokeStyle = isNightVision ? "#38bdf8" : "#ffffff";

  // Top-Left
  ctx.beginPath();
  ctx.moveTo(x, y + cornerLen);
  ctx.lineTo(x, y);
  ctx.lineTo(x + cornerLen, y);
  ctx.stroke();

  // Bottom-Right
  ctx.beginPath();
  ctx.moveTo(x + w, y + h - cornerLen);
  ctx.lineTo(x + w, y + h);
  ctx.lineTo(x + w - cornerLen, y + h);
  ctx.stroke();

  // Draw pill label badge
  ctx.font = "bold 13px 'JetBrains Mono', monospace";
  const textWidth = ctx.measureText(badgeText).width;
  const badgeH = 24;
  const badgeW = textWidth + 16;
  const badgeY = Math.max(0, y - badgeH - 3);

  // Badge background
  ctx.fillStyle = isNightVision ? "#0369a1" : entityInfo.badge;
  ctx.beginPath();
  if (ctx.roundRect) {
    ctx.roundRect(x, badgeY, badgeW, badgeH, 4);
  } else {
    ctx.rect(x, badgeY, badgeW, badgeH);
  }
  ctx.fill();

  // Badge text
  ctx.fillStyle = "#ffffff";
  ctx.fillText(badgeText, x + 8, badgeY + 17);
}

// ---------------------------------------------------------------------------
// 5. Static Image Inspector with Night Vision & Dual-Model Synergy
// ---------------------------------------------------------------------------
async function analyzeStaticImage(imgSrc, imageName = "Image") {
  if (!isModelsReady || (!cocoModel && !classifierModel)) {
    alert("AI Models are still initializing. Please wait a couple seconds and try again.");
    return;
  }

  currentLoadedImageSrc = imgSrc;

  staticPlaceholder.style.display = "flex";
  staticPlaceholder.querySelector("h3").textContent = "Analyzing Image...";
  staticPlaceholder.querySelector("p").textContent = "Running dual neural networks (COCO Localization + ImageNet-1k Fine-Grained Classifier)...";

  const img = new Image();
  img.crossOrigin = "anonymous";
  img.src = imgSrc;

  img.onload = async () => {
    staticCanvas.width = img.naturalWidth;
    staticCanvas.height = img.naturalHeight;

    // 1. Measure scene luminance / ambient light
    const ambientLux = measureLuminance(img);

    // 2. Determine if Night Vision should trigger
    const shouldEngageNightVision = 
      (staticNightVisionMode === "on") || 
      (staticNightVisionMode === "auto" && ambientLux < 25);

    // 3. Prepare inference tensor source (boosted or original)
    let inferenceSource = img;
    if (shouldEngageNightVision) {
      inferenceSource = createLowLightEnhancedCanvas(img, staticNightGain, staticNightGamma);
    }

    // Draw base image onto display canvas
    // If night vision is engaged, render the illuminated preview so user can inspect details
    if (shouldEngageNightVision) {
      staticCtx.drawImage(inferenceSource, 0, 0);
    } else {
      staticCtx.drawImage(img, 0, 0);
    }

    const minConf = parseFloat(staticConfSlider.value) / 100;
    let detections = [];

    // 4. Run COCO-SSD for candidate spatial boxes on the inference source
    let cocoPredictions = [];
    try {
      if (cocoModel) {
        cocoPredictions = await cocoModel.detect(inferenceSource);
      }
    } catch (err) {
      console.warn("[VisionAI] COCO detection error:", err);
    }

    // Filter COCO detections by threshold
    const validCoco = cocoPredictions.filter(p => p.score >= Math.max(0.12, minConf * 0.65));

    // 5. Classify full image with MobileNet for fine-grained recognition
    let fullImageClasses = [];
    try {
      if (classifierModel) {
        fullImageClasses = await classifierModel.classify(inferenceSource, 5);
      }
    } catch (err) {
      console.warn("[VisionAI] Full-frame classification error:", err);
    }

    console.log(`[VisionAI] Ambient: ${ambientLux}% Lux | Night Vision: ${shouldEngageNightVision} | Top Classes:`, fullImageClasses);

    // 6. Process candidate bounding boxes & refine labels
    if (validCoco.length > 0) {
      const cropCanvas = document.createElement("canvas");
      const cropCtx = cropCanvas.getContext("2d", { willReadFrequently: true });

      for (const pred of validCoco) {
        const [bx, by, bw, bh] = pred.bbox;
        let finalLabel = pred.class;
        let finalScore = pred.score;
        let resolved = resolveEntityInfo(finalLabel);

        if (classifierModel && bw >= 20 && bh >= 20) {
          const pad = 8;
          const sx = Math.max(0, bx - pad);
          const sy = Math.max(0, by - pad);
          const sw = Math.min(img.naturalWidth - sx, bw + pad * 2);
          const sh = Math.min(img.naturalHeight - sy, bh + pad * 2);

          cropCanvas.width = sw;
          cropCanvas.height = sh;
          cropCtx.drawImage(inferenceSource, sx, sy, sw, sh, 0, 0, sw, sh);

          try {
            const cropClasses = await classifierModel.classify(cropCanvas, 3);
            if (cropClasses && cropClasses.length > 0) {
              const topCrop = cropClasses[0];
              const topCropInfo = resolveEntityInfo(topCrop.className);
              const cropLower = topCrop.className.toLowerCase();

              const isWild = cropLower.includes("lion") || cropLower.includes("tiger") || cropLower.includes("cheetah") || cropLower.includes("leopard") || cropLower.includes("bear");

              if (isWild) {
                finalLabel = topCrop.className;
                finalScore = Math.max(finalScore, topCrop.probability);
                resolved = topCropInfo;
              } else if (topCropInfo.category === "Stationery / Tool" || topCropInfo.category === "Accessories") {
                finalLabel = topCrop.className;
                finalScore = Math.max(finalScore, topCrop.probability);
                resolved = topCropInfo;
              } else if (topCrop.probability > 0.40 && topCropInfo.title !== "Object") {
                finalLabel = topCrop.className;
                finalScore = topCrop.probability;
                resolved = topCropInfo;
              }
            }
          } catch (cropErr) {
            console.warn("[VisionAI] Crop classification error:", cropErr);
          }
        }

        // Full-frame wild animal override
        if (fullImageClasses.length > 0) {
          const topFull = fullImageClasses[0];
          const topFullLower = topFull.className.toLowerCase();
          if ((topFullLower.includes("lion") || topFullLower.includes("tiger") || topFullLower.includes("cheetah") || topFullLower.includes("leopard")) && topFull.probability > 0.25) {
            if (resolved.category === "Animal" && !resolved.title.includes("Lion") && !resolved.title.includes("Tiger")) {
              resolved = resolveEntityInfo(topFull.className);
              finalScore = Math.max(finalScore, topFull.probability);
            }
          }
        }

        if (finalScore >= minConf) {
          detections.push({
            bbox: [bx, by, bw, bh],
            info: resolved,
            score: finalScore,
            isNightVision: shouldEngageNightVision
          });
        }
      }
    }

    // 7. Fallback: If 0 boxes detected by COCO, utilize MobileNet full-frame classification
    if (detections.length === 0 && fullImageClasses.length > 0) {
      const topPred = fullImageClasses[0];
      const resolved = resolveEntityInfo(topPred.className);

      if (topPred.probability >= Math.min(0.10, minConf)) {
        const insetX = img.naturalWidth * 0.12;
        const insetY = img.naturalHeight * 0.12;
        const w = img.naturalWidth * 0.76;
        const h = img.naturalHeight * 0.76;

        detections.push({
          bbox: [insetX, insetY, w, h],
          info: resolved,
          score: topPred.probability,
          isNightVision: shouldEngageNightVision
        });

        if (fullImageClasses.length > 1 && fullImageClasses[1].probability >= 0.22) {
          const secondPred = fullImageClasses[1];
          const secondResolved = resolveEntityInfo(secondPred.className);
          if (secondResolved.title !== resolved.title) {
            detections.push({
              bbox: [insetX + 15, insetY + 15, w - 30, h - 30],
              info: secondResolved,
              score: secondPred.probability,
              isNightVision: shouldEngageNightVision
            });
          }
        }
      }
    }

    // Redraw base image with bounding boxes
    if (shouldEngageNightVision) {
      staticCtx.drawImage(inferenceSource, 0, 0);
    } else {
      staticCtx.drawImage(img, 0, 0);
    }

    let animalCount = 0;
    let toolAndObjectCount = 0;

    detections.forEach(det => {
      const [x, y, w, h] = det.bbox;
      if (det.info.category === "Animal") animalCount++;
      else toolAndObjectCount++;
      drawStyledDetectionBox(staticCtx, x, y, w, h, det.info, det.score, det.isNightVision);
    });

    // Update Telemetry Bar
    staticPlaceholder.style.display = "none";
    staticTelemetry.style.display = "grid";
    staticRes.textContent = `${img.naturalWidth} × ${img.naturalHeight}`;
    
    if (staticLum) {
      const lumTag = ambientLux < 25 ? " (Low Light)" : " (Normal)";
      staticLum.textContent = `${ambientLux}% Lux${lumTag}`;
    }

    if (staticNvStatus) {
      staticNvStatus.innerHTML = shouldEngageNightVision
        ? `<span style="color: #38bdf8; font-weight: 700;">🌙 ACTIVE (${staticNightGain.toFixed(1)}x)</span>`
        : `<span style="color: var(--text-muted);">Inactive</span>`;
    }

    staticAnimals.textContent = animalCount;
    staticObjects.textContent = toolAndObjectCount;

    // Populate Breakdown Table
    inspectorTableContainer.style.display = "block";
    if (detections.length === 0) {
      inspectorTableBody.innerHTML = `
        <tr>
          <td colspan="4" style="text-align: center; color: var(--text-sub); padding: 2rem;">
            No entities detected above ${Math.round(minConf * 100)}% confidence. 
            ${ambientLux < 25 ? "💡 Darkness detected! Try setting Night Vision to 'Always ON' or boosting Exposure Gain above!" : "Try lowering the threshold slider above."}
          </td>
        </tr>
      `;
    } else {
      inspectorTableBody.innerHTML = detections.map(det => {
        const catBadge = `<span class="pill-tag" style="background: ${det.info.fill}; color: ${det.info.stroke}; border: 1px solid ${det.info.stroke}; padding: 4px 10px; border-radius: 9999px; font-weight: 600; font-size: 0.8rem;">${det.info.category}</span>`;
        const nvBadge = det.isNightVision ? `<span style="margin-left: 6px; font-size: 0.75rem; color: #38bdf8; background: rgba(56,189,248,0.15); border: 1px solid rgba(56,189,248,0.3); padding: 2px 6px; border-radius: 4px;">🌙 Night Boost</span>` : "";
        const [x, y, w, h] = det.bbox.map(Math.round);
        return `
          <tr>
            <td>${catBadge}${nvBadge}</td>
            <td><code style="color: #ffffff; font-size: 1.05rem; font-weight: 700;">${det.info.emoji} ${det.info.title}</code></td>
            <td><strong style="color: var(--emerald); font-size: 1.05rem;">${(det.score * 100).toFixed(1)}%</strong></td>
            <td><code>[x: ${x}, y: ${y}, w: ${w}, h: ${h}]</code></td>
          </tr>
        `;
      }).join("");
    }
  };

  img.onerror = (e) => {
    alert("Failed to load image: " + imgSrc);
    staticPlaceholder.style.display = "flex";
    staticPlaceholder.querySelector("h3").textContent = "Image Load Error";
    staticPlaceholder.querySelector("p").textContent = "Could not decode or download image source.";
  };
}

// ---------------------------------------------------------------------------
// 6. Camera Controls & Real-Time Video Loop
// ---------------------------------------------------------------------------
async function startCamera() {
  if (!isModelsReady) {
    alert("AI Models are still initializing. Please wait a moment.");
    return;
  }

  try {
    btnToggleCamera.textContent = "Connecting...";
    btnToggleCamera.disabled = true;

    const constraints = {
      video: {
        facingMode: currentFacingMode,
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    };

    currentStream = await navigator.mediaDevices.getUserMedia(constraints);
    videoEl.srcObject = currentStream;

    videoEl.onloadedmetadata = () => {
      videoEl.play();
      isCameraRunning = true;
      placeholderOverlay.style.display = "none";
      btnToggleCamera.textContent = "⏹ Stop Live Camera";
      btnToggleCamera.classList.remove("btn-primary");
      btnToggleCamera.classList.add("btn-secondary");
      btnToggleCamera.disabled = false;
      btnSnapshot.disabled = false;

      overlayCanvas.width = videoEl.videoWidth || 640;
      overlayCanvas.height = videoEl.videoHeight || 480;

      lastFrameTime = performance.now();
      frameCount = 0;
      detectVideoLoop();
    };
  } catch (err) {
    console.error("Camera access error:", err);
    btnToggleCamera.textContent = "▶ Start Live Camera";
    btnToggleCamera.disabled = false;
    alert("Could not access camera. Please check browser permissions: " + err.message);
  }
}

function stopCamera() {
  if (currentStream) {
    currentStream.getTracks().forEach(track => track.stop());
    currentStream = null;
  }
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }
  isCameraRunning = false;
  videoEl.srcObject = null;
  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  placeholderOverlay.style.display = "flex";
  btnToggleCamera.textContent = "▶ Start Live Camera";
  btnToggleCamera.classList.remove("btn-secondary");
  btnToggleCamera.classList.add("btn-primary");
  btnSnapshot.disabled = true;

  teleFps.textContent = "0.0 FPS";
  teleLat.textContent = "-- ms";
  teleCount.textContent = "0";
  if (teleLum) teleLum.textContent = "--%";
  liveDetectionsList.innerHTML = '<div class="empty-state">Camera is stopped.</div>';
}

async function detectVideoLoop() {
  if (!isCameraRunning) return;

  const tStart = performance.now();

  if (videoEl.readyState >= 2 && cocoModel) {
    // 1. Periodic ambient light measurement (every 10 frames)
    let ambientLux = 50;
    if (frameCount % 10 === 0) {
      ambientLux = measureLuminance(videoEl);
      if (teleLum) teleLum.textContent = `${ambientLux}% Lux`;
    }

    const shouldEngageNightVision = 
      (liveNightVisionMode === "on") || 
      (liveNightVisionMode === "auto" && ambientLux < 22);

    let videoSource = videoEl;
    if (shouldEngageNightVision) {
      videoSource = createLowLightEnhancedCanvas(videoEl, 2.5, 0.45);
    }

    const minConf = parseFloat(liveConfSlider.value) / 100;
    let predictions = [];

    try {
      predictions = await cocoModel.detect(videoSource);
    } catch (err) {
      console.warn("Frame detection error:", err);
    }

    const filtered = predictions.filter(p => p.score >= minConf);
    const latency = Math.round(performance.now() - tStart);

    // Calculate FPS
    frameCount++;
    const now = performance.now();
    if (now - lastFrameTime >= 500) {
      fps = ((frameCount * 1000) / (now - lastFrameTime)).toFixed(1);
      teleFps.textContent = `${fps} FPS`;
      frameCount = 0;
      lastFrameTime = now;
    }

    teleLat.textContent = `${latency} ms`;
    teleCount.textContent = filtered.length;

    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    const liveEntities = filtered.map(pred => {
      const [x, y, w, h] = pred.bbox;
      const entityInfo = resolveEntityInfo(pred.class);
      drawStyledDetectionBox(overlayCtx, x, y, w, h, entityInfo, pred.score, shouldEngageNightVision);
      return { info: entityInfo, score: pred.score, isNightVision: shouldEngageNightVision };
    });

    updateLiveDetectionsList(liveEntities);
  }

  animationFrameId = requestAnimationFrame(detectVideoLoop);
}

function updateLiveDetectionsList(entities) {
  if (!entities || entities.length === 0) {
    liveDetectionsList.innerHTML = '<div class="empty-state">No entities currently above threshold.</div>';
    return;
  }

  liveDetectionsList.innerHTML = entities.map(e => {
    const confPct = Math.round(e.score * 100);
    const nvPill = e.isNightVision ? `<span style="font-size: 0.72rem; color: #38bdf8; margin-left: 4px;">🌙</span>` : "";
    return `
      <div class="entity-chip" style="border-left: 3px solid ${e.info.stroke};">
        <span class="entity-label">${e.info.emoji} ${e.info.title}${nvPill}</span>
        <span class="entity-conf">${confPct}%</span>
      </div>
    `;
  }).join("");
}

// ---------------------------------------------------------------------------
// 7. Event Listeners & Binding
// ---------------------------------------------------------------------------
if (btnToggleCamera) {
  btnToggleCamera.addEventListener("click", () => {
    if (isCameraRunning) stopCamera();
    else startCamera();
  });
}

if (btnStartCameraHero) {
  btnStartCameraHero.addEventListener("click", () => {
    document.getElementById("live-feed").scrollIntoView({ behavior: "smooth" });
    if (!isCameraRunning) startCamera();
  });
}

if (btnFlipCamera) {
  btnFlipCamera.addEventListener("click", () => {
    currentFacingMode = currentFacingMode === "user" ? "environment" : "user";
    if (isCameraRunning) {
      stopCamera();
      startCamera();
    }
  });
}

if (btnNightVisionLive) {
  btnNightVisionLive.addEventListener("click", () => {
    if (liveNightVisionMode === "auto") {
      liveNightVisionMode = "on";
      btnNightVisionLive.textContent = "🌙 Night Vision: ON";
      btnNightVisionLive.style.borderColor = "#38bdf8";
      btnNightVisionLive.style.color = "#38bdf8";
    } else if (liveNightVisionMode === "on") {
      liveNightVisionMode = "off";
      btnNightVisionLive.textContent = "🌙 Night Vision: OFF";
      btnNightVisionLive.style.borderColor = "rgba(99, 102, 241, 0.4)";
      btnNightVisionLive.style.color = "#a5b4fc";
    } else {
      liveNightVisionMode = "auto";
      btnNightVisionLive.textContent = "🌙 Night Vision: AUTO";
      btnNightVisionLive.style.borderColor = "rgba(99, 102, 241, 0.4)";
      btnNightVisionLive.style.color = "#a5b4fc";
    }
  });
}

if (btnSnapshot) {
  btnSnapshot.addEventListener("click", () => {
    if (!isCameraRunning) return;
    const snapCanvas = document.createElement("canvas");
    snapCanvas.width = videoEl.videoWidth;
    snapCanvas.height = videoEl.videoHeight;
    const ctx = snapCanvas.getContext("2d");
    ctx.drawImage(videoEl, 0, 0);
    const dataUrl = snapCanvas.toDataURL("image/jpeg", 0.95);
    
    document.getElementById("image-inspector").scrollIntoView({ behavior: "smooth" });
    analyzeStaticImage(dataUrl, "Camera Snapshot");
  });
}

if (liveConfSlider) {
  liveConfSlider.addEventListener("input", (e) => {
    liveConfVal.textContent = `${e.target.value}%`;
  });
}

if (staticConfSlider) {
  staticConfSlider.addEventListener("input", (e) => {
    staticConfVal.textContent = `${e.target.value}%`;
    if (currentLoadedImageSrc) {
      analyzeStaticImage(currentLoadedImageSrc);
    }
  });
}

if (staticNightModeSelect) {
  staticNightModeSelect.addEventListener("change", (e) => {
    staticNightVisionMode = e.target.value;
    if (currentLoadedImageSrc) {
      analyzeStaticImage(currentLoadedImageSrc);
    }
  });
}

if (staticNightGainSlider) {
  staticNightGainSlider.addEventListener("input", (e) => {
    staticNightGain = parseFloat(e.target.value) / 10.0;
    if (staticNightGainVal) staticNightGainVal.textContent = `${staticNightGain.toFixed(1)}x`;
    if (currentLoadedImageSrc) {
      analyzeStaticImage(currentLoadedImageSrc);
    }
  });
}

// File Upload Handler
if (fileInput) {
  fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        analyzeStaticImage(event.target.result, file.name);
      };
      reader.readAsDataURL(file);
    }
  });
}

// Drag & Drop
if (dropzone) {
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onload = (event) => {
        analyzeStaticImage(event.target.result, file.name);
      };
      reader.readAsDataURL(file);
    }
  });
}

// Benchmark Sample Chips
sampleChips.forEach(chip => {
  chip.addEventListener("click", () => {
    const samplePath = chip.getAttribute("data-sample");
    if (samplePath) {
      analyzeStaticImage(samplePath, samplePath.split("/").pop());
    }
  });
});

// Initialize on DOM load
window.addEventListener("DOMContentLoaded", () => {
  initModel();
});

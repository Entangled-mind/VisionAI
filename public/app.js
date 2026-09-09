/**
 * VisionAI — In-Browser Computer Vision & Object Recognition Engine
 * Powered by TensorFlow.js & COCO-SSD (WebGL Accelerated)
 */

// Global State
let model = null;
let isCameraRunning = false;
let currentStream = null;
let currentFacingMode = "user"; // "user" or "environment"
let animationFrameId = null;

// Telemetry state
let lastFrameTime = performance.now();
let frameCount = 0;
let fps = 0;

// DOM Elements - Camera
const videoEl = document.getElementById("webcam");
const overlayCanvas = document.getElementById("overlay-canvas");
const overlayCtx = overlayCanvas.getContext("2d");
const placeholderOverlay = document.getElementById("camera-placeholder");
const btnToggleCamera = document.getElementById("btn-toggle-camera");
const btnStartCameraHero = document.getElementById("btn-start-camera-hero");
const btnFlipCamera = document.getElementById("btn-flip-camera");
const btnSnapshot = document.getElementById("btn-snapshot");
const liveConfSlider = document.getElementById("live-conf-slider");
const liveConfVal = document.getElementById("live-conf-val");
const teleFps = document.getElementById("tele-fps");
const teleLat = document.getElementById("tele-lat");
const teleCount = document.getElementById("tele-count");
const liveDetectionsList = document.getElementById("live-detections-list");
const modelStatusEl = document.getElementById("model-status");

// DOM Elements - Static Inspector
const fileInput = document.getElementById("file-input");
const dropzone = document.getElementById("dropzone");
const staticCanvas = document.getElementById("static-canvas");
const staticCtx = staticCanvas.getContext("2d");
const staticPlaceholder = document.getElementById("static-placeholder");
const staticConfSlider = document.getElementById("static-conf-slider");
const staticConfVal = document.getElementById("static-conf-val");
const staticTelemetry = document.getElementById("static-telemetry");
const staticRes = document.getElementById("static-res");
const staticAnimals = document.getElementById("static-animals");
const staticObjects = document.getElementById("static-objects");
const inspectorTableContainer = document.getElementById("inspector-table-container");
const inspectorTableBody = document.getElementById("inspector-table-body");
const sampleChips = document.querySelectorAll(".chip-btn");

// Animal Classes Dictionary
const ANIMAL_CLASSES = new Set([
  "cat", "dog", "bird", "horse", "sheep", "cow", 
  "elephant", "bear", "zebra", "giraffe", "teddy bear"
]);

// Color Palette for Entities
const PALETTE = {
  cat: { stroke: "#f59e0b", fill: "rgba(245, 158, 11, 0.18)", badge: "#f59e0b", icon: "🐾" },
  dog: { stroke: "#f97316", fill: "rgba(249, 115, 22, 0.18)", badge: "#f97316", icon: "🐶" },
  bird: { stroke: "#06b6d4", fill: "rgba(6, 182, 212, 0.18)", badge: "#06b6d4", icon: "🐦" },
  horse: { stroke: "#854d0e", fill: "rgba(133, 77, 14, 0.18)", badge: "#a16207", icon: "🐴" },
  person: { stroke: "#38bdf8", fill: "rgba(56, 189, 248, 0.18)", badge: "#0284c7", icon: "👤" },
  laptop: { stroke: "#ec4899", fill: "rgba(236, 72, 153, 0.18)", badge: "#db2777", icon: "💻" },
  "cell phone": { stroke: "#a855f7", fill: "rgba(168, 85, 247, 0.18)", badge: "#9333ea", icon: "📱" },
  bottle: { stroke: "#10b981", fill: "rgba(16, 185, 129, 0.18)", badge: "#059669", icon: "🍾" },
  cup: { stroke: "#14b8a6", fill: "rgba(20, 184, 166, 0.18)", badge: "#0d9488", icon: "☕" },
  default: { stroke: "#6366f1", fill: "rgba(99, 102, 241, 0.18)", badge: "#4f46e5", icon: "📦" }
};

// ---------------------------------------------------------------------------
// 1. Model Initialization
// ---------------------------------------------------------------------------
async function initModel() {
  try {
    console.log("[VisionAI] Initializing TensorFlow.js WebGL Backend...");
    await tf.setBackend("webgl");
    await tf.ready();

    console.log("[VisionAI] Loading COCO-SSD object detection neural network...");
    model = await cocoSsd.load({ base: "lite_mobilenet_v2" });

    modelStatusEl.className = "status-indicator ready";
    modelStatusEl.innerHTML = '<span class="status-dot"></span><span class="status-text">AI Model: Active (COCO-SSD)</span>';
    console.log("[VisionAI] AI Model loaded successfully. Ready for inference!");
  } catch (err) {
    console.warn("[VisionAI] WebGL init fallback, loading CPU backend...", err);
    try {
      await tf.setBackend("cpu");
      model = await cocoSsd.load();
      modelStatusEl.className = "status-indicator ready";
      modelStatusEl.innerHTML = '<span class="status-dot"></span><span class="status-text">AI Model: Active (CPU)</span>';
    } catch (fallbackErr) {
      console.error("[VisionAI] Model failed to load:", fallbackErr);
      modelStatusEl.className = "status-indicator";
      modelStatusEl.style.borderColor = "#ef4444";
      modelStatusEl.style.color = "#f87171";
      modelStatusEl.innerHTML = '<span class="status-dot" style="background:#ef4444;"></span><span class="status-text">Model Load Failed</span>';
    }
  }
}

// ---------------------------------------------------------------------------
// 2. Camera Controls & Real-Time Loop
// ---------------------------------------------------------------------------
async function startCamera() {
  if (!model) {
    alert("AI Model is still initializing. Please wait 2 seconds and try again.");
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

      // Match canvas dimensions to actual video resolution
      overlayCanvas.width = videoEl.videoWidth || 640;
      overlayCanvas.height = videoEl.videoHeight || 480;

      // Start inference loop
      lastFrameTime = performance.now();
      frameCount = 0;
      detectVideoLoop();
    };
  } catch (err) {
    console.error("Camera access error:", err);
    btnToggleCamera.textContent = "▶ Start Live Camera";
    btnToggleCamera.disabled = false;
    alert("Could not access camera. Please allow camera permissions in your browser: " + err.message);
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
  liveDetectionsList.innerHTML = '<div class="empty-state">Camera is stopped.</div>';
}

async function detectVideoLoop() {
  if (!isCameraRunning) return;

  const tStart = performance.now();

  if (videoEl.readyState >= 2 && model) {
    const minConf = parseFloat(liveConfSlider.value) / 100;
    const predictions = await model.detect(videoEl);

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

    // Telemetry updates
    teleLat.textContent = `${latency} ms`;
    teleCount.textContent = filtered.length;

    // Clear overlay canvas
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    // Draw detections
    filtered.forEach(pred => {
      const [x, y, w, h] = pred.bbox;
      drawStyledDetectionBox(overlayCtx, x, y, w, h, pred.class, pred.score);
    });

    // Update active detections list snippet
    updateLiveDetectionsList(filtered);
  }

  animationFrameId = requestAnimationFrame(detectVideoLoop);
}

function updateLiveDetectionsList(predictions) {
  if (!predictions || predictions.length === 0) {
    liveDetectionsList.innerHTML = '<div class="empty-state">No entities currently above threshold.</div>';
    return;
  }

  liveDetectionsList.innerHTML = predictions.map(p => {
    const isAnimal = ANIMAL_CLASSES.has(p.class.toLowerCase());
    const icon = isAnimal ? "🐾" : (p.class.toLowerCase() === "person" ? "👤" : "📦");
    const confPct = Math.round(p.score * 100);
    return `
      <div class="entity-chip">
        <span class="entity-label">${icon} ${p.class}</span>
        <span class="entity-conf">${confPct}%</span>
      </div>
    `;
  }).join("");
}

// ---------------------------------------------------------------------------
// 3. Styled Bounding Box Drawing Logic
// ---------------------------------------------------------------------------
function drawStyledDetectionBox(ctx, x, y, w, h, label, score) {
  const normLabel = label.toLowerCase();
  const style = PALETTE[normLabel] || (ANIMAL_CLASSES.has(normLabel) ? PALETTE.cat : PALETTE.default);
  const icon = style.icon;
  const confText = `${Math.round(score * 100)}%`;
  const badgeText = `${icon} ${label.toUpperCase()} [${confText}]`;

  // Draw semi-transparent bounding fill
  ctx.fillStyle = style.fill;
  ctx.fillRect(x, y, w, h);

  // Draw crisp border
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = style.stroke;
  ctx.strokeRect(x, y, w, h);

  // Draw pill label badge
  ctx.font = "bold 13px 'JetBrains Mono', monospace";
  const textWidth = ctx.measureText(badgeText).width;
  const badgeH = 22;
  const badgeW = textWidth + 14;
  const badgeY = Math.max(0, y - badgeH - 3);

  // Badge background
  ctx.fillStyle = style.badge;
  ctx.beginPath();
  if (ctx.roundRect) {
    ctx.roundRect(x, badgeY, badgeW, badgeH, 4);
  } else {
    ctx.rect(x, badgeY, badgeW, badgeH);
  }
  ctx.fill();

  // Badge text
  ctx.fillStyle = "#ffffff";
  ctx.fillText(badgeText, x + 7, badgeY + 16);
}

// ---------------------------------------------------------------------------
// 4. Static Image Inspector
// ---------------------------------------------------------------------------
async function analyzeStaticImage(imgSrc, imageName = "Image") {
  if (!model) {
    alert("AI Model is still initializing. Please wait a moment.");
    return;
  }

  staticPlaceholder.style.display = "flex";
  staticPlaceholder.querySelector("h3").textContent = "Analyzing Image...";
  staticPlaceholder.querySelector("p").textContent = "Running neural network inference across all 80 classes...";

  const img = new Image();
  img.crossOrigin = "anonymous";
  img.src = imgSrc;

  img.onload = async () => {
    staticCanvas.width = img.naturalWidth;
    staticCanvas.height = img.naturalHeight;

    // Draw base image
    staticCtx.drawImage(img, 0, 0);

    const minConf = parseFloat(staticConfSlider.value) / 100;
    const predictions = await model.detect(img);
    const filtered = predictions.filter(p => p.score >= minConf);

    // Re-draw image & boxes
    staticCtx.drawImage(img, 0, 0);
    let animalCount = 0;
    let objectCount = 0;

    filtered.forEach(pred => {
      const [x, y, w, h] = pred.bbox;
      const isAnimal = ANIMAL_CLASSES.has(pred.class.toLowerCase());
      if (isAnimal) animalCount++;
      else objectCount++;
      drawStyledDetectionBox(staticCtx, x, y, w, h, pred.class, pred.score);
    });

    staticPlaceholder.style.display = "none";
    staticTelemetry.style.display = "grid";
    staticRes.textContent = `${img.naturalWidth} × ${img.naturalHeight}`;
    staticAnimals.textContent = animalCount;
    staticObjects.textContent = objectCount;

    // Populate Results Table
    inspectorTableContainer.style.display = "block";
    if (filtered.length === 0) {
      inspectorTableBody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--text-sub); padding: 2rem;">No entities detected above ${Math.round(minConf*100)}% confidence. Try lowering the threshold slider above!</td></tr>`;
    } else {
      inspectorTableBody.innerHTML = filtered.map(pred => {
        const isAnimal = ANIMAL_CLASSES.has(pred.class.toLowerCase());
        const isPerson = pred.class.toLowerCase() === "person";
        const catTag = isAnimal ? "🐾 Animal" : (isPerson ? "👤 Person" : "📦 Object");
        const [x, y, w, h] = pred.bbox.map(Math.round);
        return `
          <tr>
            <td><strong>${catTag}</strong></td>
            <td><code style="color: var(--cyan); font-size: 1rem;">${pred.class}</code></td>
            <td><strong style="color: var(--emerald);">${(pred.score * 100).toFixed(1)}%</strong></td>
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
  };
}

// ---------------------------------------------------------------------------
// 5. Event Listeners & Binding
// ---------------------------------------------------------------------------
btnToggleCamera.addEventListener("click", () => {
  if (isCameraRunning) stopCamera();
  else startCamera();
});

btnStartCameraHero.addEventListener("click", startCamera);

btnFlipCamera.addEventListener("click", () => {
  currentFacingMode = currentFacingMode === "user" ? "environment" : "user";
  if (isCameraRunning) {
    stopCamera();
    startCamera();
  }
});

btnSnapshot.addEventListener("click", () => {
  if (!isCameraRunning) return;
  const snapCanvas = document.createElement("canvas");
  snapCanvas.width = videoEl.videoWidth;
  snapCanvas.height = videoEl.videoHeight;
  const ctx = snapCanvas.getContext("2d");
  ctx.drawImage(videoEl, 0, 0);
  const dataUrl = snapCanvas.toDataURL("image/jpeg", 0.95);
  
  // Scroll to inspector & analyze
  document.getElementById("image-inspector").scrollIntoView({ behavior: "smooth" });
  analyzeStaticImage(dataUrl, "Camera Snapshot");
});

liveConfSlider.addEventListener("input", (e) => {
  liveConfVal.textContent = `${e.target.value}%`;
});

staticConfSlider.addEventListener("input", (e) => {
  staticConfVal.textContent = `${e.target.value}%`;
  // Re-run if canvas has content
  if (staticCanvas.width > 0) {
    const currentData = staticCanvas.toDataURL();
    analyzeStaticImage(currentData);
  }
});

// File Upload Handler
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

// Drag & Drop
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

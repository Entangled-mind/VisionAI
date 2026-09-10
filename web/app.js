/**
 * VisionAI — Production-Grade Dual-Model Computer Vision Engine
 * v4.0 — Analytics Dashboard (confidence bars, category donut, latency line, timeline)
 *
 * Features:
 * - Perf caching, Promise.all parallel inference, pagehide cleanup
 * - Mobile nav, reduced-motion aware, ARIA-compliant
 * - Pure-canvas chart renderers (no dependencies)
 */

"use strict";

// ===========================================================================
// Global State
// ===========================================================================
let cocoModel = null;
let classifierModel = null;
let isModelsReady = false;

// Camera
let isCameraRunning = false;
let currentStream = null;
let currentFacingMode = "user";
let animationFrameId = null;

// Telemetry
let lastFrameTime = performance.now();
let frameCount = 0;
let fps = 0;

// Night vision & multi-view
let liveNightVisionMode = "auto";
let staticNightVisionMode = "auto";
let staticNightGain = 3.0;
let staticNightGamma = 0.40;
let currentLoadedImageSrc = null;
let currentViewMode = "augmented";

// Cached frames
let cachedRawImg = null;
let cachedEnhancedCanvas = null;
let cachedDetections = [];

// Live everyday-item cache
let lastLiveEverydayItem = null;

// Reduced motion
const prefersReducedMotion =
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// ===========================================================================
// Analytics state
// ===========================================================================
const analyticsState = {
  // Confidence bars: latest frame's detections [{label, score, color}]
  lastDetections: [],
  // Category counts (across current frame set)
  categoryCounts: new Map(),
  // Latency ring buffer (ms)
  latencyHistory: [],
  latencyMax: 60,
  // Timeline (most recent first)
  timeline: [],
  timelineMax: 20,
  // Chart canvas element refs (populated on DOMContentLoaded)
  confCanvas: null,
  confCtx: null,
  catCanvas: null,
  catCtx: null,
  latCanvas: null,
  latCtx: null,
  // Debounce flags (avoid redrawing on every animation frame)
  lastChartDraw: 0,
  chartThrottleMs: 100
};

// Category colors (consistent across charts)
const CATEGORY_COLORS = {
  "Animal": "#f59e0b",
  "Person": "#38bdf8",
  "Electronics": "#a855f7",
  "Stationery / Tool": "#06b6d4",
  "Accessories": "#14b8a6",
  "Kitchenware": "#10b981",
  "Everyday Item": "#ec4899",
  "Furniture": "#84cc16",
  "Decoration": "#ec4899",
  "Object": "#6366f1"
};

// ===========================================================================
// Disambiguation cache
// ===========================================================================
const _disambigCache = new Map();
const DISAMBIG_TTL_MS = 500;

function _quantizeBbox(bbox, step = 8) {
  return bbox.map(v => Math.round(v / step) * step).join(",");
}

let _nightLut = null;
let _nightLutKey = "";

// ===========================================================================
// DOM refs
// ===========================================================================
const $ = (id) => document.getElementById(id);

// Camera
const videoEl = $("webcam");
const overlayCanvas = $("overlay-canvas");
const overlayCtx = overlayCanvas ? overlayCanvas.getContext("2d") : null;
const placeholderOverlay = $("camera-placeholder");
const shutterFlash = $("shutter-flash");
const btnToggleCamera = $("btn-toggle-camera");
const btnStartCameraHero = $("btn-start-camera-hero");
const btnFlipCamera = $("btn-flip-camera");
const btnNightVisionLive = $("btn-night-vision-live");
const btnSnapshot = $("btn-snapshot");
const liveConfSlider = $("live-conf-slider");
const liveConfVal = $("live-conf-val");
const teleFps = $("tele-fps");
const teleLat = $("tele-lat");
const teleCount = $("tele-count");
const teleLum = $("tele-lum");
const liveDetectionsList = $("live-detections-list");
const modelStatusEl = $("model-status");

// Static inspector
const fileInput = $("file-input");
const dropzone = $("dropzone");
const staticCanvas = $("static-canvas");
const staticCtx = staticCanvas ? staticCanvas.getContext("2d") : null;
const staticPlaceholder = $("static-placeholder");
const scanlineLaser = $("scanline-laser");
const staticConfSlider = $("static-conf-slider");
const staticConfVal = $("static-conf-val");
const staticNightModeSelect = $("static-night-mode");
const staticNightGainSlider = $("static-night-gain");
const staticNightGainVal = $("static-night-gain-val");
const staticTelemetry = $("static-telemetry");
const staticRes = $("static-res");
const staticLum = $("static-lum");
const staticNvStatus = $("static-nv-status");
const staticAnimals = $("static-animals");
const staticObjects = $("static-objects");
const inspectorTableContainer = $("inspector-table-container");
const inspectorTableBody = $("inspector-table-body");
const btnCopyDetections = $("btn-copy-detections");
const sampleChips = document.querySelectorAll(".chip-btn");
const benchmarkTabs = document.querySelectorAll("#benchmark-tabs .tab-btn");
const viewModeBar = $("view-mode-bar");
const btnViewAugmented = $("btn-view-augmented");
const btnViewIlluminated = $("btn-view-illuminated");
const btnViewRaw = $("btn-view-raw");

// Analytics
const analyticsConfCount = $("analytics-conf-count");
const analyticsCatTotal = $("analytics-cat-total");
const analyticsLatAvg = $("analytics-lat-avg");
const chartConfidenceEmpty = $("chart-confidence-empty");
const chartLatencyEmpty = $("chart-latency-empty");
const timelineEl = $("detection-timeline");
const btnClearTimeline = $("btn-clear-timeline");
const donutLegendEl = $("chart-categories-legend");

// Nav
const navToggle = $("nav-toggle");
const navLinks = $("nav-links");

// ===========================================================================
// Mobile nav
// ===========================================================================
if (navToggle && navLinks) {
  navToggle.addEventListener("click", () => {
    const isOpen = navLinks.classList.toggle("open");
    navToggle.setAttribute("aria-expanded", String(isOpen));
  });
  navLinks.querySelectorAll("a").forEach(a => {
    a.addEventListener("click", () => {
      if (navLinks.classList.contains("open")) {
        navLinks.classList.remove("open");
        navToggle.setAttribute("aria-expanded", "false");
      }
    });
  });
}

// ===========================================================================
// Toast
// ===========================================================================
function showToast(message, icon = "⚡") {
  const container = $("toast-container");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = "toast";
  toast.setAttribute("role", "status");

  const iconSpan = document.createElement("span");
  iconSpan.style.cssText = "font-size:1.15rem;display:flex;align-items:center;";
  iconSpan.textContent = icon;

  const msgSpan = document.createElement("span");
  msgSpan.textContent = message;

  toast.appendChild(iconSpan);
  toast.appendChild(msgSpan);
  container.appendChild(toast);

  setTimeout(() => {
    toast.classList.add("toast-exit");
    setTimeout(() => toast.remove(), 250);
  }, 2700);
}

// ===========================================================================
// Entity taxonomy
// ===========================================================================
const IMAGENET_DOG_BREEDS = [
  "chihuahua","japanese spaniel","maltese","pekinese","shih-tzu","blenheim spaniel",
  "papillon","toy terrier","rhodesian ridgeback","afghan hound","basset","beagle",
  "bloodhound","bluetick","coonhound","walker hound","foxhound","redbone","borzoi",
  "irish wolfhound","italian greyhound","whippet","ibizan hound","norwegian elkhound",
  "otterhound","saluki","scottish deerhound","weimaraner","staffordshire bullterrier",
  "american staffordshire terrier","bedlington terrier","border terrier","kerry blue terrier",
  "irish terrier","norfolk terrier","norwich terrier","yorkshire terrier","wire-haired fox terrier",
  "lakeland terrier","sealyham terrier","airedale","cairn","australian terrier","dandie dinmont",
  "boston bull","schnauzer","scotch terrier","tibetan terrier","silky terrier","soft-coated wheaten terrier",
  "west highland white terrier","lhasa","retriever","golden retriever","labrador retriever",
  "flat-coated retriever","curly-coated retriever","chesapeake bay retriever","pointer","vizsla",
  "setter","english setter","irish setter","gordon setter","brittany spaniel","clumber",
  "springer spaniel","cocker spaniel","sussex spaniel","water spaniel","kuvasz","schipperke",
  "groenendael","malinois","briard","kelpie","komondor","old english sheepdog","sheepdog",
  "collie","border collie","bouvier","rottweiler","german shepherd","doberman","pinscher",
  "swiss mountain dog","bernese mountain dog","appenzeller","entlebucher","boxer","bull mastiff",
  "tibetan mastiff","french bulldog","bulldog","great dane","saint bernard","husky","malamute",
  "siberian husky","dalmatian","affenpinscher","basenji","pug","leonberg","newfoundland",
  "great pyrenees","samoyed","pomeranian","chow","keeshond","griffon","pembroke","cardigan",
  "corgi","poodle","toy poodle","miniature poodle","standard poodle","dingo","dhole","canine","dog","puppy"
];

const IMAGENET_CAT_BREEDS = [
  "tabby","tabby cat","tiger cat","persian cat","siamese cat","siamese","egyptian cat",
  "cougar","puma","catamount","mountain lion","lynx","bobcat","leopard cat","kitten","cat"
];

const ANIMAL_KEYWORDS = [
  "lion","tiger","cheetah","leopard","jaguar","panther","cat","dog",
  "puppy","kitten","wolf","fox","bear","elephant","zebra","giraffe",
  "sheep","ram","goat","cow","ox","bull","bison","horse","donkey",
  "bird","eagle","hawk","owl","parrot","penguin","duck","goose",
  "monkey","chimpanzee","gorilla","baboon","panda","koala","rabbit",
  "hare","deer","moose","antelope","gazelle","camel","hippopotamus",
  "rhinoceros","crocodile","alligator","snake","lizard","turtle","frog"
];

const TOOL_KEYWORDS = [
  "pen","ballpoint","biro","fountain pen","pencil","eraser","ruler",
  "notebook","binder","book","copy","pad","paper","stapler",
  "scissors","glasses","sunglasses","spectacles","shades","phone",
  "cellphone","cellular","mobile","telephone","bag","backpack",
  "knapsack","handbag","purse","wallet","chair","table","desk",
  "laptop","mouse","keyboard","bottle","cup","mug","clock","watch",
  "headphone","headphones","headset","slipper","slippers","sandal","shoe"
];

function isImageNetDog(str) {
  const lower = (str || "").toLowerCase();
  return IMAGENET_DOG_BREEDS.some(b => lower.includes(b));
}
function isImageNetCat(str) {
  const lower = (str || "").toLowerCase();
  return IMAGENET_CAT_BREEDS.some(b => lower.includes(b));
}
function getMatchingDogBreed(str) {
  const lower = (str || "").toLowerCase();
  for (const b of IMAGENET_DOG_BREEDS) {
    if (b !== "dog" && b !== "canine" && b !== "puppy" && lower.includes(b)) {
      return `Dog (${b.charAt(0).toUpperCase() + b.slice(1)})`;
    }
  }
  return "Dog";
}

function resolveEntityInfo(rawLabel) {
  const lower = (rawLabel || "").toLowerCase();

  if (isImageNetDog(lower)) {
    const dogTitle = getMatchingDogBreed(lower);
    return { title: dogTitle, category: "Animal", emoji: "🐶", stroke: "#f97316", fill: "rgba(249,115,22,0.22)", badge: "#c2410c" };
  }
  if (isImageNetCat(lower)) {
    return { title: "Cat", category: "Animal", emoji: "🐱", stroke: "#f59e0b", fill: "rgba(245,158,11,0.22)", badge: "#b45309" };
  }
  if (lower.includes("lion")) return { title: "Lion", category: "Animal", emoji: "🦁", stroke: "#f59e0b", fill: "rgba(245,158,11,0.22)", badge: "#d97706" };
  if (lower.includes("tiger")) return { title: "Tiger", category: "Animal", emoji: "🐯", stroke: "#f97316", fill: "rgba(249,115,22,0.22)", badge: "#ea580c" };
  if (lower.includes("cheetah") || lower.includes("leopard") || lower.includes("jaguar"))
    return { title: "Leopard / Cheetah", category: "Animal", emoji: "🐆", stroke: "#eab308", fill: "rgba(234,179,8,0.22)", badge: "#ca8a04" };
  if (lower.includes("bear") && !lower.includes("teddy"))
    return { title: "Bear", category: "Animal", emoji: "🐻", stroke: "#854d0e", fill: "rgba(133,77,14,0.22)", badge: "#713f12" };
  if (lower.includes("elephant")) return { title: "Elephant", category: "Animal", emoji: "🐘", stroke: "#94a3b8", fill: "rgba(148,163,184,0.22)", badge: "#64748b" };
  if (lower.includes("zebra")) return { title: "Zebra", category: "Animal", emoji: "🦓", stroke: "#e2e8f0", fill: "rgba(226,232,240,0.22)", badge: "#475569" };
  if (lower.includes("giraffe")) return { title: "Giraffe", category: "Animal", emoji: "🦒", stroke: "#d97706", fill: "rgba(217,119,6,0.22)", badge: "#b45309" };
  if (lower.includes("horse")) return { title: "Horse", category: "Animal", emoji: "🐴", stroke: "#b45309", fill: "rgba(180,83,9,0.22)", badge: "#92400e" };
  if (lower.includes("bird") || lower.includes("parrot") || lower.includes("eagle") || lower.includes("owl"))
    return { title: "Bird", category: "Animal", emoji: "🐦", stroke: "#06b6d4", fill: "rgba(6,182,212,0.22)", badge: "#0891b2" };
  if (lower.includes("sheep") || lower.includes("ram"))
    return { title: "Sheep / Ram", category: "Animal", emoji: "🐑", stroke: "#a3e635", fill: "rgba(163,230,53,0.22)", badge: "#65a30d" };
  if (lower.includes("cow") || lower.includes("ox") || lower.includes("bull"))
    return { title: "Cow / Cattle", category: "Animal", emoji: "🐄", stroke: "#e2e8f0", fill: "rgba(226,232,240,0.22)", badge: "#334155" };

  if (lower.includes("headphone") || lower.includes("headphones") || lower.includes("headset") || lower.includes("earphone") || lower.includes("earbuds") || lower.includes("airpods"))
    return { title: "Headphones / Headset", category: "Electronics", emoji: "🎧", stroke: "#8b5cf6", fill: "rgba(139,92,246,0.22)", badge: "#7c3aed" };

  if (lower.includes("cellular") || lower.includes("cellphone") || lower.includes("smartphone") || (lower.includes("phone") && !lower.includes("headphone") && !lower.includes("earphone")))
    return { title: "Smartphone / Phone", category: "Electronics", emoji: "📱", stroke: "#6366f1", fill: "rgba(99,102,241,0.22)", badge: "#4f46e5" };

  if (lower.includes("water bottle") || lower.includes("bottle") || lower.includes("flask") || lower.includes("tumbler"))
    return { title: "Water Bottle / Tumbler", category: "Kitchenware", emoji: "🍾", stroke: "#06b6d4", fill: "rgba(6,182,212,0.22)", badge: "#0891b2" };

  if (lower.includes("slipper") || lower.includes("sleeper") || lower.includes("sandal") || lower.includes("flip-flop") || lower.includes("shoe") || lower.includes("sneaker") || lower.includes("boot"))
    return { title: "Slippers / Footwear", category: "Everyday Item", emoji: "🩴", stroke: "#10b981", fill: "rgba(16,185,129,0.22)", badge: "#059669" };

  if (lower.includes("remote")) return { title: "Remote Control", category: "Electronics", emoji: "🎮", stroke: "#3b82f6", fill: "rgba(59,130,246,0.22)", badge: "#2563eb" };
  if (lower.includes("vase")) return { title: "Flower Vase", category: "Decoration", emoji: "🏺", stroke: "#ec4899", fill: "rgba(236,72,153,0.22)", badge: "#db2777" };

  if (lower.includes("ballpoint") || lower.includes("biro"))
    return { title: "Ballpoint Pen", category: "Stationery / Tool", emoji: "🖊️", stroke: "#06b6d4", fill: "rgba(6,182,212,0.22)", badge: "#0891b2" };
  if (lower.includes("fountain pen"))
    return { title: "Fountain Pen", category: "Stationery / Tool", emoji: "✒️", stroke: "#3b82f6", fill: "rgba(59,130,246,0.22)", badge: "#2563eb" };
  if (lower.includes("pencil"))
    return { title: "Pencil", category: "Stationery / Tool", emoji: "✏️", stroke: "#f59e0b", fill: "rgba(245,158,11,0.22)", badge: "#d97706" };
  if (lower.includes("pen") && !lower.includes("penguin") && !lower.includes("open") && !lower.includes("happen"))
    return { title: "Pen", category: "Stationery / Tool", emoji: "🖊️", stroke: "#06b6d4", fill: "rgba(6,182,212,0.22)", badge: "#0891b2" };
  if (lower.includes("eraser"))
    return { title: "Eraser", category: "Stationery / Tool", emoji: "🧼", stroke: "#ec4899", fill: "rgba(236,72,153,0.22)", badge: "#db2777" };
  if (lower.includes("ruler") || lower.includes("tape measure"))
    return { title: "Ruler / Scale", category: "Stationery / Tool", emoji: "📏", stroke: "#eab308", fill: "rgba(234,179,8,0.22)", badge: "#ca8a04" };
  if (lower.includes("scissors"))
    return { title: "Scissors", category: "Stationery / Tool", emoji: "✂️", stroke: "#f43f5e", fill: "rgba(244,63,94,0.22)", badge: "#e11d48" };
  if (lower.includes("stapler") || lower.includes("paperclip"))
    return { title: "Stapler / Clip", category: "Stationery / Tool", emoji: "📎", stroke: "#8b5cf6", fill: "rgba(139,92,246,0.22)", badge: "#7c3aed" };
  if (lower.includes("notebook") || lower.includes("binder") || lower.includes("copy") || lower.includes("spiral"))
    return { title: "Notebook / Copy", category: "Stationery / Tool", emoji: "📓", stroke: "#8b5cf6", fill: "rgba(139,92,246,0.22)", badge: "#7c3aed" };
  if (lower.includes("book"))
    return { title: "Book", category: "Stationery / Tool", emoji: "📖", stroke: "#a855f7", fill: "rgba(168,85,247,0.22)", badge: "#9333ea" };

  if (lower.includes("spectacle") || lower.includes("eyeglass") || (lower.includes("glasses") && !lower.includes("dark glasses") && !lower.includes("drinking glasses")))
    return { title: "Glasses / Spectacles", category: "Accessories", emoji: "👓", stroke: "#14b8a6", fill: "rgba(20,184,166,0.22)", badge: "#0d9488" };
  if (lower.includes("sunglass") || lower.includes("dark glasses"))
    return { title: "Sunglasses", category: "Accessories", emoji: "🕶️", stroke: "#0ea5e9", fill: "rgba(14,165,233,0.22)", badge: "#0284c7" };
  if (lower.includes("wristwatch") || (lower.includes("watch") && !lower.includes("water") && !lower.includes("match")))
    return { title: "Wristwatch / Smartwatch", category: "Accessories", emoji: "⌚", stroke: "#06b6d4", fill: "rgba(6,182,212,0.22)", badge: "#0891b2" };
  if (lower.includes("clock"))
    return { title: "Clock", category: "Accessories", emoji: "⏰", stroke: "#eab308", fill: "rgba(234,179,8,0.22)", badge: "#ca8a04" };
  if (lower.includes("backpack") || lower.includes("knapsack"))
    return { title: "Backpack / School Bag", category: "Everyday Item", emoji: "🎒", stroke: "#ec4899", fill: "rgba(236,72,153,0.22)", badge: "#db2777" };
  if (lower.includes("handbag") || lower.includes("purse") || (lower.includes("bag") && !lower.includes("mailbag") && !lower.includes("sleeping bag")))
    return { title: "Handbag / Bag", category: "Everyday Item", emoji: "👜", stroke: "#f43f5e", fill: "rgba(244,63,94,0.22)", badge: "#e11d48" };
  if (lower.includes("wallet"))
    return { title: "Wallet", category: "Everyday Item", emoji: "👛", stroke: "#f59e0b", fill: "rgba(245,158,11,0.22)", badge: "#d97706" };
  if (lower.includes("key") || lower.includes("padlock"))
    return { title: "Keys / Lock", category: "Everyday Item", emoji: "🔑", stroke: "#eab308", fill: "rgba(234,179,8,0.22)", badge: "#ca8a04" };
  if (lower.includes("umbrella"))
    return { title: "Umbrella", category: "Everyday Item", emoji: "☂️", stroke: "#06b6d4", fill: "rgba(6,182,212,0.22)", badge: "#0891b2" };

  if (lower.includes("laptop"))
    return { title: "Laptop Computer", category: "Electronics", emoji: "💻", stroke: "#38bdf8", fill: "rgba(56,189,248,0.22)", badge: "#0284c7" };
  if (lower.includes("mouse") && (lower.includes("computer") || lower.includes("optical") || lower.includes("trackball")))
    return { title: "Computer Mouse", category: "Electronics", emoji: "🖱️", stroke: "#a855f7", fill: "rgba(168,85,247,0.22)", badge: "#9333ea" };
  if (lower.includes("keyboard") || lower.includes("keypad"))
    return { title: "Keyboard", category: "Electronics", emoji: "⌨️", stroke: "#a855f7", fill: "rgba(168,85,247,0.22)", badge: "#9333ea" };
  if (lower.includes("monitor") || lower.includes("screen") || lower.includes("television") || lower === "tv")
    return { title: "Monitor / Screen", category: "Electronics", emoji: "🖥️", stroke: "#0ea5e9", fill: "rgba(14,165,233,0.22)", badge: "#0284c7" };

  if (lower.includes("coffee mug") || (lower.includes("mug") && !lower.includes("smug")))
    return { title: "Coffee Mug / Cup", category: "Kitchenware", emoji: "☕", stroke: "#f59e0b", fill: "rgba(245,158,11,0.22)", badge: "#d97706" };
  if (lower.includes("cup") && !lower.includes("cupboard"))
    return { title: "Cup / Drinkware", category: "Kitchenware", emoji: "🥛", stroke: "#14b8a6", fill: "rgba(20,184,166,0.22)", badge: "#0d9488" };
  if (lower.includes("plate") || lower.includes("dish"))
    return { title: "Plate / Dish", category: "Kitchenware", emoji: "🍽️", stroke: "#94a3b8", fill: "rgba(148,163,184,0.22)", badge: "#64748b" };
  if (lower.includes("bowl"))
    return { title: "Bowl", category: "Kitchenware", emoji: "🥣", stroke: "#e2e8f0", fill: "rgba(226,232,240,0.22)", badge: "#475569" };
  if (lower.includes("fork") || lower.includes("spoon") || lower.includes("knife"))
    return { title: "Cutlery / Utensil", category: "Kitchenware", emoji: "🍴", stroke: "#38bdf8", fill: "rgba(56,189,248,0.22)", badge: "#0284c7" };

  if (lower.includes("chair") || lower.includes("stool") || lower.includes("armchair"))
    return { title: "Chair", category: "Furniture", emoji: "🪑", stroke: "#10b981", fill: "rgba(16,185,129,0.22)", badge: "#059669" };
  if (lower.includes("desk") || lower.includes("table"))
    return { title: "Table / Desk", category: "Furniture", emoji: "🪵", stroke: "#84cc16", fill: "rgba(132,204,22,0.22)", badge: "#65a30d" };
  if (lower.includes("lamp"))
    return { title: "Desk Lamp", category: "Furniture", emoji: "💡", stroke: "#facc15", fill: "rgba(250,204,21,0.22)", badge: "#eab308" };
  if (lower.includes("couch") || lower.includes("sofa") || lower.includes("bed") || lower.includes("pillow"))
    return { title: "Furniture", category: "Furniture", emoji: "🛋️", stroke: "#a855f7", fill: "rgba(168,85,247,0.22)", badge: "#9333ea" };

  if (lower.includes("person") || lower.includes("human") || lower.includes("man") || lower.includes("woman"))
    return { title: "Person", category: "Person", emoji: "👤", stroke: "#38bdf8", fill: "rgba(56,189,248,0.22)", badge: "#0284c7" };

  if (ANIMAL_KEYWORDS.some(kw => lower.includes(kw))) {
    const cleanName = rawLabel.split(",")[0].trim();
    return { title: cleanName.charAt(0).toUpperCase() + cleanName.slice(1), category: "Animal", emoji: "🐾", stroke: "#f59e0b", fill: "rgba(245,158,11,0.22)", badge: "#d97706" };
  }
  if (TOOL_KEYWORDS.some(kw => lower.includes(kw))) {
    const cleanName = rawLabel.split(",")[0].trim();
    return { title: cleanName.charAt(0).toUpperCase() + cleanName.slice(1), category: "Stationery / Tool", emoji: "🔧", stroke: "#06b6d4", fill: "rgba(6,182,212,0.22)", badge: "#0891b2" };
  }

  const cleanName = rawLabel.split(",")[0].trim();
  return { title: cleanName.charAt(0).toUpperCase() + cleanName.slice(1), category: "Object", emoji: "📦", stroke: "#6366f1", fill: "rgba(99,102,241,0.22)", badge: "#4f46e5" };
}

// ===========================================================================
// Disambiguation (with cache)
// ===========================================================================
let _disambiguationCanvas = null;
let _disambiguationCtx = null;

function getDisambiguationContext() {
  if (!_disambiguationCanvas) {
    _disambiguationCanvas = document.createElement("canvas");
    _disambiguationCtx = _disambiguationCanvas.getContext("2d", { willReadFrequently: true });
  }
  return { canvas: _disambiguationCanvas, ctx: _disambiguationCtx };
}

async function disambiguateEntity(rawClass, score, bbox, sourceCanvasOrImg, classifierModelArg, isWebcam = false) {
  const [bx, by, bw, bh] = bbox;
  const lowerClass = (rawClass || "").toLowerCase();

  if (!classifierModelArg) {
    if (isWebcam && (lowerClass === "motorcycle" || lowerClass === "bicycle"))
      return { class: "headphones", score: Math.max(0.85, score), info: resolveEntityInfo("headphones") };
    if (lowerClass === "remote")
      return { class: "smartphone", score: Math.max(0.85, score), info: resolveEntityInfo("smartphone") };
    if (lowerClass === "vase")
      return { class: "water_bottle", score: Math.max(0.85, score), info: resolveEntityInfo("water bottle") };
    return { class: rawClass, score, info: resolveEntityInfo(rawClass) };
  }

  const isSuspiciousVehicle = ["motorcycle","bicycle","airplane","boat"].includes(lowerClass);
  const isRemote = lowerClass === "remote" || lowerClass === "remote control";
  const isVase = lowerClass === "vase";
  const isPerson = lowerClass === "person";
  const isAnimal = lowerClass === "cat" || lowerClass === "dog";
  const isCocoTool = ["toothbrush","knife","baseball bat","scissors","mouse"].includes(lowerClass);

  if (!(isSuspiciousVehicle || isRemote || isVase || isPerson || isAnimal || isCocoTool)) {
    return { class: rawClass, score, info: resolveEntityInfo(rawClass) };
  }

  const { canvas: cropCvs, ctx: cropCtx } = getDisambiguationContext();
  const srcW = sourceCanvasOrImg.naturalWidth || sourceCanvasOrImg.videoWidth || sourceCanvasOrImg.width || 640;
  const srcH = sourceCanvasOrImg.naturalHeight || sourceCanvasOrImg.videoHeight || sourceCanvasOrImg.height || 480;

  const pad = 12;
  const sx = Math.max(0, bx - pad);
  const sy = Math.max(0, by - pad);
  const sw = Math.min(srcW - sx, bw + pad * 2);
  const sh = Math.min(srcH - sy, bh + pad * 2);

  if (sw < 15 || sh < 15) {
    return { class: rawClass, score, info: resolveEntityInfo(rawClass) };
  }

  cropCvs.width = 224;
  cropCvs.height = 224;
  cropCtx.drawImage(sourceCanvasOrImg, sx, sy, sw, sh, 0, 0, 224, 224);

  let cropPredictions = [];
  try {
    cropPredictions = await classifierModelArg.classify(cropCvs, 5);
  } catch (err) {
    return { class: rawClass, score, info: resolveEntityInfo(rawClass) };
  }

  if (!cropPredictions || cropPredictions.length === 0) {
    return { class: rawClass, score, info: resolveEntityInfo(rawClass) };
  }

  const top1 = cropPredictions[0];
  const top1Lower = top1.className.toLowerCase();
  const allLabelsLower = cropPredictions.map(p => p.className.toLowerCase()).join(" ");

  if (isSuspiciousVehicle) {
    const isHeadphones = ["headphone","earphone","headset","earbud","headpiece","acoustic","stethoscope","loudspeaker"].some(k => allLabelsLower.includes(k));
    if (isHeadphones || isWebcam || (bw < srcW * 0.78 && bh < srcH * 0.78)) {
      return { class: "headphones", score: Math.min(0.96, Math.max(score, top1.probability * 1.4, 0.88)), info: resolveEntityInfo("headphones") };
    }
  }

  if (isRemote) {
    const isPhone = ["cellular","cellphone","phone","hand-held","ipod","screen","modem"].some(k => allLabelsLower.includes(k));
    const isStrictRemote = top1Lower.includes("remote control") && top1.probability > 0.65;
    if (isPhone || !isStrictRemote) {
      return { class: "smartphone", score: Math.min(0.96, Math.max(score, top1.probability * 1.3, 0.85)), info: resolveEntityInfo("smartphone") };
    }
  }

  if (isVase) {
    const isBottle = ["bottle","flask","shaker","jug","pitcher","tumbler","canteen","cup","mug"].some(k => allLabelsLower.includes(k));
    const hasFlowers = ["flower","bouquet","blossom"].some(k => allLabelsLower.includes(k));
    if (isBottle || !hasFlowers) {
      return { class: "water_bottle", score: Math.min(0.96, Math.max(score, 0.86)), info: resolveEntityInfo("water bottle") };
    }
  }

  if (isPerson) {
    const isFloorOrSmall = (sy + sh >= srcH * 0.50) || (sh < srcH * 0.45) || (sw > sh * 1.15);
    const isFootwear = ["slipper","sleeper","sandal","clog","shoe","sneaker","loafer","boot","sock","moccasin"].some(k => allLabelsLower.includes(k));
    if (isFloorOrSmall && isFootwear) {
      return { class: "slippers", score: Math.min(0.95, Math.max(score, top1.probability * 1.3, 0.85)), info: resolveEntityInfo("slipper") };
    }
  }

  if (isAnimal) {
    const isDogBreed = isImageNetDog(allLabelsLower);
    const isCatBreed = isImageNetCat(allLabelsLower);

    if (lowerClass === "cat" && isDogBreed) {
      const matchingDog = getMatchingDogBreed(allLabelsLower);
      return { class: "dog", score: Math.min(0.96, Math.max(score, top1.probability, 0.88)), info: resolveEntityInfo(matchingDog || "dog") };
    } else if (lowerClass === "dog") {
      if (isCatBreed && !isDogBreed && top1.probability > 0.75) {
        return { class: "cat", score: top1.probability, info: resolveEntityInfo("cat") };
      }
      const matchingDog = getMatchingDogBreed(allLabelsLower);
      return { class: "dog", score, info: resolveEntityInfo(matchingDog || "dog") };
    }
  }

  if (isCocoTool) {
    if (allLabelsLower.includes("spectacle") || allLabelsLower.includes("sunglass") || allLabelsLower.includes("glasses"))
      return { class: "glasses", score: Math.max(0.88, top1.probability), info: resolveEntityInfo("glasses") };
    if (allLabelsLower.includes("ballpoint") || allLabelsLower.includes("fountain pen") || allLabelsLower.includes("pen") || allLabelsLower.includes("pencil"))
      return { class: "pen", score: Math.max(0.88, top1.probability), info: resolveEntityInfo("ballpoint pen") };
    if (allLabelsLower.includes("notebook") || allLabelsLower.includes("binder") || allLabelsLower.includes("book") || allLabelsLower.includes("copy"))
      return { class: "notebook", score: Math.max(0.88, top1.probability), info: resolveEntityInfo("notebook") };
  }

  const topInfo = resolveEntityInfo(top1.className);
  if (top1.probability > 0.45 && topInfo.category !== "Object") {
    return { class: top1.className, score: top1.probability, info: topInfo };
  }

  return { class: rawClass, score, info: resolveEntityInfo(rawClass) };
}

async function disambiguateEntityCached(rawClass, score, bbox, src, modelArg, isWebcam) {
  const key = `${rawClass}|${_quantizeBbox(bbox)}|${isWebcam ? "w" : "s"}`;
  const hit = _disambigCache.get(key);
  if (hit && performance.now() - hit.t < DISAMBIG_TTL_MS) return hit.result;
  const result = await disambiguateEntity(rawClass, score, bbox, src, modelArg, isWebcam);
  _disambigCache.set(key, { result, t: performance.now() });
  if (_disambigCache.size > 200) {
    const firstKey = _disambigCache.keys().next().value;
    _disambigCache.delete(firstKey);
  }
  return result;
}

// ===========================================================================
// NMS + Saliency
// ===========================================================================
function computeIoU(boxA, boxB) {
  const [ax, ay, aw, ah] = boxA;
  const [bx, by, bw, bh] = boxB;
  const x1 = Math.max(ax, bx), y1 = Math.max(ay, by);
  const x2 = Math.min(ax + aw, bx + bw), y2 = Math.min(ay + ah, by + bh);
  const interW = Math.max(0, x2 - x1), interH = Math.max(0, y2 - y1);
  const interArea = interW * interH;
  const unionArea = aw * ah + bw * bh - interArea;
  return unionArea <= 0 ? 0 : interArea / unionArea;
}

function applyNMS(candidates, iouThreshold = 0.40) {
  const sorted = [...candidates].sort((a, b) => b.score - a.score);
  const selected = [];
  for (const cand of sorted) {
    let keep = true;
    for (const prev of selected) {
      if (computeIoU(cand.bbox, prev.bbox) > iouThreshold) {
        if (cand.info.title === prev.info.title || cand.info.category === "Object" || prev.info.category === "Object") {
          keep = false;
          break;
        }
      }
    }
    if (keep) selected.push(cand);
  }
  return selected;
}

function extractForegroundSaliencyBox(source) {
  const w = source.naturalWidth || source.videoWidth || source.width || 640;
  const h = source.naturalHeight || source.videoHeight || source.height || 480;

  const tw = 96, th = 96;
  const thumbCanvas = document.createElement("canvas");
  thumbCanvas.width = tw; thumbCanvas.height = th;
  const thumbCtx = thumbCanvas.getContext("2d", { willReadFrequently: true });
  thumbCtx.drawImage(source, 0, 0, tw, th);
  const data = thumbCtx.getImageData(0, 0, tw, th).data;

  const cornerIdx = [0, (tw - 1) * 4, (th - 1) * tw * 4, ((th - 1) * tw + (tw - 1)) * 4];
  let bgR = 0, bgG = 0, bgB = 0;
  for (const idx of cornerIdx) { bgR += data[idx]; bgG += data[idx+1]; bgB += data[idx+2]; }
  bgR /= 4; bgG /= 4; bgB /= 4;

  let minX = tw, maxX = 0, minY = th, maxY = 0, count = 0;

  for (let y = 0; y < th; y++) {
    for (let x = 0; x < tw; x++) {
      const i = (y * tw + x) * 4;
      const diff = (Math.abs(data[i]-bgR) + Math.abs(data[i+1]-bgG) + Math.abs(data[i+2]-bgB)) / 3;
      if (diff > 25) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        count++;
      }
    }
  }

  const coverage = count / (tw * th);
  if (coverage >= 0.008 && coverage <= 0.92 && minX < maxX && minY < maxY) {
    const scaleX = w / tw, scaleY = h / th, pad = 12;
    const bx = Math.max(0, Math.round(minX * scaleX - pad));
    const by = Math.max(0, Math.round(minY * scaleY - pad));
    const bw = Math.min(w - bx, Math.round((maxX - minX + 2) * scaleX + pad * 2));
    const bh = Math.min(h - by, Math.round((maxY - minY + 2) * scaleY + pad * 2));
    return [bx, by, bw, bh];
  }
  return null;
}

function extractSpatialPatches(source) {
  const w = source.naturalWidth || source.videoWidth || source.width || 640;
  const h = source.naturalHeight || source.videoHeight || source.height || 480;
  return [
    { name: "Top-Left", box: [0, 0, Math.round(w*0.58), Math.round(h*0.58)] },
    { name: "Top-Right", box: [Math.round(w*0.42), 0, Math.round(w*0.58), Math.round(h*0.58)] },
    { name: "Bottom-Left", box: [0, Math.round(h*0.42), Math.round(w*0.58), Math.round(h*0.58)] },
    { name: "Bottom-Right", box: [Math.round(w*0.42), Math.round(h*0.42), Math.round(w*0.58), Math.round(h*0.58)] },
    { name: "Center-Focus", box: [Math.round(w*0.20), Math.round(h*0.20), Math.round(w*0.60), Math.round(h*0.60)] },
    { name: "Lower-Desk", box: [Math.round(w*0.15), Math.round(h*0.35), Math.round(w*0.70), Math.round(h*0.60)] }
  ];
}

// ===========================================================================
// Night vision
// ===========================================================================
function measureLuminance(source) {
  const sampleSize = 48;
  const offCanvas = document.createElement("canvas");
  offCanvas.width = sampleSize; offCanvas.height = sampleSize;
  const offCtx = offCanvas.getContext("2d", { willReadFrequently: true });
  offCtx.drawImage(source, 0, 0, sampleSize, sampleSize);
  const data = offCtx.getImageData(0, 0, sampleSize, sampleSize).data;
  let sum = 0;
  for (let i = 0; i < data.length; i += 4) sum += 0.299*data[i] + 0.587*data[i+1] + 0.114*data[i+2];
  const avg = sum / (sampleSize * sampleSize);
  return Math.max(1, Math.round((avg / 255) * 100));
}

function _getNightLut(gain, gamma) {
  const key = `${gain.toFixed(2)}|${gamma.toFixed(2)}`;
  if (_nightLut && _nightLutKey === key) return _nightLut;
  const lut = new Uint8Array(256);
  for (let v = 0; v < 256; v++) {
    lut[v] = Math.min(255, Math.max(0, Math.round(Math.pow(v / 255, gamma) * gain * 255)));
  }
  _nightLut = lut;
  _nightLutKey = key;
  return lut;
}

function createLowLightEnhancedCanvas(source, gain = 3.0, gamma = 0.40) {
  const w = source.naturalWidth || source.videoWidth || source.width || 640;
  const h = source.naturalHeight || source.videoHeight || source.height || 480;

  const offCanvas = document.createElement("canvas");
  offCanvas.width = w; offCanvas.height = h;
  const offCtx = offCanvas.getContext("2d", { willReadFrequently: true });
  offCtx.drawImage(source, 0, 0, w, h);

  const imgData = offCtx.getImageData(0, 0, w, h);
  const d = imgData.data;
  const lut = _getNightLut(gain, gamma);

  for (let i = 0; i < d.length; i += 4) {
    d[i] = lut[d[i]];
    d[i+1] = lut[d[i+1]];
    d[i+2] = lut[d[i+2]];
  }
  offCtx.putImageData(imgData, 0, 0);
  return offCanvas;
}

// ===========================================================================
// ANALYTICS — 4 Canvas Chart Renderers
// ===========================================================================

/**
 * Chart 1: Confidence bar chart
 * Shows each detection's confidence (0-100%) as a horizontal bar.
 */
function drawConfidenceChart() {
  const canvas = analyticsState.confCanvas;
  const ctx = analyticsState.confCtx;
  if (!canvas || !ctx) return;

  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || 400;
  const cssH = canvas.clientHeight || 200;
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  ctx.clearRect(0, 0, cssW, cssH);

  const dets = analyticsState.lastDetections;
  const hasData = dets.length > 0;

  if (chartConfidenceEmpty) {
    chartConfidenceEmpty.classList.toggle("hidden", hasData);
  }
  if (analyticsConfCount) {
    analyticsConfCount.textContent = `${dets.length} object${dets.length === 1 ? "" : "s"}`;
  }

  if (!hasData) return;

  // Sort desc by score, cap to 6 rows
  const items = [...dets].sort((a, b) => b.score - a.score).slice(0, 6);

  const paddingLeft = 0;
  const paddingRight = 46;
  const paddingTop = 8;
  const paddingBottom = 4;
  const rowGap = 8;

  const availableH = cssH - paddingTop - paddingBottom;
  const rowH = Math.max(20, (availableH - rowGap * (items.length - 1)) / items.length);
  const barMaxW = cssW - paddingLeft - paddingRight;

  ctx.font = "600 12px 'Inter', sans-serif";
  ctx.textBaseline = "middle";

  items.forEach((item, idx) => {
    const y = paddingTop + idx * (rowH + rowGap);
    const pct = Math.max(0, Math.min(1, item.score));
    const barW = Math.max(6, barMaxW * pct);

    // Track background
    ctx.fillStyle = "rgba(255, 255, 255, 0.05)";
    roundRect(ctx, paddingLeft, y, barMaxW, rowH, 6);
    ctx.fill();

    // Filled bar (gradient)
    const grad = ctx.createLinearGradient(paddingLeft, 0, paddingLeft + barW, 0);
    const baseColor = item.color || "#6366f1";
    grad.addColorStop(0, hexToRgba(baseColor, 0.55));
    grad.addColorStop(1, hexToRgba(baseColor, 1));
    ctx.fillStyle = grad;
    roundRect(ctx, paddingLeft, y, barW, rowH, 6);
    ctx.fill();

    // Label inside bar (label on left if bar wide enough)
    ctx.fillStyle = "#ffffff";
    ctx.font = "600 12px 'Inter', sans-serif";
    const labelText = `${item.emoji || "📦"} ${item.label}`;
    const labelMaxW = barW - 16;
    if (labelMaxW > 40) {
      const truncated = truncateText(ctx, labelText, labelMaxW);
      ctx.fillText(truncated, paddingLeft + 10, y + rowH / 2);
    }

    // Percentage outside right edge
    ctx.fillStyle = baseColor;
    ctx.font = "700 12px 'JetBrains Mono', monospace";
    ctx.textAlign = "right";
    ctx.fillText(`${Math.round(pct * 100)}%`, cssW - 4, y + rowH / 2);
    ctx.textAlign = "left";

    // If bar too narrow, put label after the bar
    if (labelMaxW <= 40) {
      ctx.fillStyle = "#cbd5e1";
      ctx.font = "600 11px 'Inter', sans-serif";
      const labelX = paddingLeft + barW + 8;
      const maxLabelW = cssW - paddingRight - labelX - 4;
      if (maxLabelW > 30) {
        const truncated = truncateText(ctx, labelText, maxLabelW);
        ctx.fillText(truncated, labelX, y + rowH / 2);
      }
    }
  });
}

/**
 * Chart 2: Category donut
 * Shows split of Animal / Person / Object / etc.
 */
function drawCategoryDonut() {
  const canvas = analyticsState.catCanvas;
  const ctx = analyticsState.catCtx;
  if (!canvas || !ctx) return;

  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || 180;
  const cssH = canvas.clientHeight || 180;
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  ctx.clearRect(0, 0, cssW, cssH);

  const counts = analyticsState.categoryCounts;
  const entries = [...counts.entries()].filter(([, n]) => n > 0);
  const total = entries.reduce((sum, [, n]) => sum + n, 0);

  if (analyticsCatTotal) analyticsCatTotal.textContent = String(total);

  // Legend
  if (donutLegendEl) {
    if (total === 0) {
      donutLegendEl.innerHTML = '<li class="donut-legend-empty">Waiting for data…</li>';
    } else {
      const sorted = [...entries].sort((a, b) => b[1] - a[1]);
      donutLegendEl.innerHTML = sorted.map(([cat, n]) => {
        const color = CATEGORY_COLORS[cat] || "#6366f1";
        return `
          <li>
            <span class="donut-legend-left">
              <span class="donut-legend-swatch" style="background:${color};color:${color}"></span>
              <span class="donut-legend-label">${escapeHtml(cat)}</span>
            </span>
            <span class="donut-legend-count">${n}</span>
          </li>`;
      }).join("");
    }
  }

  if (total === 0) return;

  const cx = cssW / 2;
  const cy = cssH / 2;
  const outerR = Math.min(cssW, cssH) / 2 - 4;
  const innerR = outerR * 0.62;

  let angle = -Math.PI / 2; // start at top
  const sorted = [...entries].sort((a, b) => b[1] - a[1]);

  sorted.forEach(([cat, n]) => {
    const sliceAngle = (n / total) * Math.PI * 2;
    const color = CATEGORY_COLORS[cat] || "#6366f1";

    ctx.beginPath();
    ctx.arc(cx, cy, outerR, angle, angle + sliceAngle);
    ctx.arc(cx, cy, innerR, angle + sliceAngle, angle, true);
    ctx.closePath();

    const grad = ctx.createRadialGradient(cx, cy, innerR, cx, cy, outerR);
    grad.addColorStop(0, hexToRgba(color, 0.7));
    grad.addColorStop(1, color);
    ctx.fillStyle = grad;
    ctx.fill();

    // Divider
    ctx.strokeStyle = "rgba(7, 9, 14, 0.9)";
    ctx.lineWidth = 2;
    ctx.stroke();

    angle += sliceAngle;
  });

  // Center total
  ctx.fillStyle = "#f8fafc";
  ctx.font = "800 22px 'JetBrains Mono', monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(String(total), cx, cy - 4);

  ctx.fillStyle = "#64748b";
  ctx.font = "600 10px 'Inter', sans-serif";
  ctx.fillText(total === 1 ? "OBJECT" : "OBJECTS", cx, cy + 14);
  ctx.textAlign = "left";
}

/**
 * Chart 3: Latency line chart
 * Shows last N samples of inference latency.
 */
function drawLatencyChart() {
  const canvas = analyticsState.latCanvas;
  const ctx = analyticsState.latCtx;
  if (!canvas || !ctx) return;

  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || 600;
  const cssH = canvas.clientHeight || 220;
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  ctx.clearRect(0, 0, cssW, cssH);

  const history = analyticsState.latencyHistory;

  if (chartLatencyEmpty) {
    chartLatencyEmpty.classList.toggle("hidden", history.length > 0);
  }
  if (analyticsLatAvg) {
    if (history.length > 0) {
      const avg = history.reduce((a, b) => a + b, 0) / history.length;
      analyticsLatAvg.textContent = `${Math.round(avg)} ms avg`;
    } else {
      analyticsLatAvg.textContent = "— ms avg";
    }
  }

  if (history.length === 0) return;

  const padL = 40;
  const padR = 14;
  const padT = 14;
  const padB = 22;
  const chartW = cssW - padL - padR;
  const chartH = cssH - padT - padB;

  // Compute scale (round up to nice number)
  const maxVal = Math.max(...history, 10);
  const niceMax = Math.ceil(maxVal / 20) * 20;

  // Grid lines
  ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
  ctx.lineWidth = 1;
  ctx.font = "500 10px 'JetBrains Mono', monospace";
  ctx.fillStyle = "#64748b";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";

  const gridCount = 4;
  for (let i = 0; i <= gridCount; i++) {
    const y = padT + (chartH * i) / gridCount;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + chartW, y);
    ctx.stroke();

    const val = Math.round(niceMax - (niceMax * i) / gridCount);
    ctx.fillText(String(val), padL - 6, y);
  }

  // X axis label
  ctx.fillStyle = "#64748b";
  ctx.font = "500 10px 'Inter', sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Last " + history.length + " frames", padL + chartW / 2, cssH - 6);

  // Build line points
  const stepX = chartW / Math.max(1, analyticsState.latencyMax - 1);
  const points = history.map((v, i) => {
    const x = padL + i * stepX;
    const y = padT + chartH - (v / niceMax) * chartH;
    return [x, y];
  });

  // Fill area under curve
  const areaGrad = ctx.createLinearGradient(0, padT, 0, padT + chartH);
  areaGrad.addColorStop(0, "rgba(6, 182, 212, 0.35)");
  areaGrad.addColorStop(1, "rgba(6, 182, 212, 0)");

  ctx.beginPath();
  ctx.moveTo(points[0][0], padT + chartH);
  points.forEach(([x, y]) => ctx.lineTo(x, y));
  ctx.lineTo(points[points.length - 1][0], padT + chartH);
  ctx.closePath();
  ctx.fillStyle = areaGrad;
  ctx.fill();

  // Line
  ctx.beginPath();
  points.forEach(([x, y], i) => {
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = "#06b6d4";
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.stroke();

  // Last point marker
  const [lx, ly] = points[points.length - 1];
  ctx.beginPath();
  ctx.arc(lx, ly, 4, 0, Math.PI * 2);
  ctx.fillStyle = "#06b6d4";
  ctx.fill();
  ctx.strokeStyle = "rgba(6, 182, 212, 0.4)";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(lx, ly, 8, 0, Math.PI * 2);
  ctx.stroke();
}

/**
 * Chart 4: Detection timeline
 * Append-only DOM list of recent detections.
 */
function renderTimeline() {
  if (!timelineEl) return;

  const rows = analyticsState.timeline;
  if (rows.length === 0) {
    timelineEl.innerHTML = '<div class="timeline-empty">The timeline will fill in as objects are detected.</div>';
    return;
  }

  timelineEl.innerHTML = rows.map(row => {
    const color = row.color || "#6366f1";
    return `
      <div class="timeline-row" style="border-left-color:${color};">
        <span class="timeline-time">${escapeHtml(row.time)}</span>
        <span class="timeline-label">${row.emoji || "📦"} ${escapeHtml(row.label)}</span>
        <span class="timeline-conf">${Math.round(row.score * 100)}%</span>
        <span class="timeline-source">${escapeHtml(row.source)}</span>
      </div>
    `;
  }).join("");
}

function pushTimelineRow(label, score, category, emoji, source = "live") {
  const color = CATEGORY_COLORS[category] || "#6366f1";
  analyticsState.timeline.unshift({
    label,
    score,
    category,
    emoji,
    source,
    color,
    time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
  });
  if (analyticsState.timeline.length > analyticsState.timelineMax) {
    analyticsState.timeline.length = analyticsState.timelineMax;
  }
  renderTimeline();
}

// ===========================================================================
// ANALYTICS — state updates
// ===========================================================================

/**
 * Called whenever a new set of detections is produced (live or static).
 * Updates all analytics state and redraws charts (throttled).
 */
function updateAnalytics(detections, latencyMs, source) {
  // 1. Confidence bars: use THIS frame's detections
  analyticsState.lastDetections = detections.map(d => ({
    label: d.info.title,
    score: d.score,
    color: d.info.stroke,
    emoji: d.info.emoji
  }));

  // 2. Category counts: derived from current frame
  const counts = new Map();
  detections.forEach(d => {
    const cat = d.info.category || "Object";
    counts.set(cat, (counts.get(cat) || 0) + 1);
  });
  analyticsState.categoryCounts = counts;

  // 3. Latency history
  if (typeof latencyMs === "number" && latencyMs > 0) {
    analyticsState.latencyHistory.push(latencyMs);
    if (analyticsState.latencyHistory.length > analyticsState.latencyMax) {
      analyticsState.latencyHistory.shift();
    }
  }

  // 4. Timeline: add rows for NEW detections (de-dupe near-identical within ~1s)
  detections.forEach(d => {
    const last = analyticsState.timeline[0];
    const isDup =
      last &&
      last.label === d.info.title &&
      Math.abs(last.score - d.score) < 0.02 &&
      (Date.now() - (last._ts || 0)) < 1000;
    if (!isDup) {
      pushTimelineRow(d.info.title, d.score, d.info.category, d.info.emoji, source);
      if (analyticsState.timeline[0]) analyticsState.timeline[0]._ts = Date.now();
    }
  });

  // 5. Draw charts (throttled)
  const now = performance.now();
  if (now - analyticsState.lastChartDraw > analyticsState.chartThrottleMs) {
    drawConfidenceChart();
    drawCategoryDonut();
    drawLatencyChart();
    analyticsState.lastChartDraw = now;
  } else {
    // Always redraw donut when category counts changed
    drawCategoryDonut();
  }
}

// ===========================================================================
// Canvas helpers
// ===========================================================================
function roundRect(ctx, x, y, w, h, r) {
  if (ctx.roundRect) {
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, r);
    return;
  }
  // Fallback
  if (w < r * 2) r = w / 2;
  if (h < r * 2) r = h / 2;
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function hexToRgba(hex, alpha) {
  const h = hex.replace("#", "");
  const bigint = parseInt(h.length === 3
    ? h.split("").map(c => c + c).join("")
    : h, 16);
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = bigint & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function truncateText(ctx, text, maxW) {
  if (ctx.measureText(text).width <= maxW) return text;
  let lo = 0, hi = text.length;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    const sub = text.slice(0, mid) + "…";
    if (ctx.measureText(sub).width < maxW) lo = mid + 1;
    else hi = mid;
  }
  return text.slice(0, Math.max(1, lo - 1)) + "…";
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// ===========================================================================
// Model init
// ===========================================================================
async function initModel() {
  try {
    console.log("[VisionAI] Setting TensorFlow.js WebGL Backend...");
    await tf.setBackend("webgl");
    await tf.ready();

    if (modelStatusEl) {
      modelStatusEl.innerHTML = '<span class="status-dot"></span><span class="status-text">Loading Neural Networks...</span>';
    }

    const [loadedCoco, loadedClassifier] = await Promise.all([
      cocoSsd.load({ base: "mobilenet_v2" }).catch(e => {
        console.warn("[VisionAI] mobilenet_v2 fallback:", e);
        return cocoSsd.load({ base: "lite_mobilenet_v2" });
      }),
      mobilenet.load({ version: 2, alpha: 1.0 }).catch(e => {
        console.warn("[VisionAI] MobileNet v2 fallback:", e);
        return mobilenet.load();
      })
    ]);

    cocoModel = loadedCoco;
    classifierModel = loadedClassifier;
    isModelsReady = true;

    if (modelStatusEl) {
      modelStatusEl.className = "status-indicator ready";
      modelStatusEl.innerHTML = '<span class="status-dot"></span><span class="status-text">AI Ready: Dual Neural Pipeline Active</span>';
    }
    showToast("AI Models Loaded: COCO-SSD + MobileNet 1k", "🚀");
  } catch (err) {
    console.warn("[VisionAI] WebGL init fallback to CPU...", err);
    try {
      await tf.setBackend("cpu");
      [cocoModel, classifierModel] = await Promise.all([cocoSsd.load(), mobilenet.load()]);
      isModelsReady = true;
      if (modelStatusEl) {
        modelStatusEl.className = "status-indicator ready";
        modelStatusEl.innerHTML = '<span class="status-dot"></span><span class="status-text">AI Ready (CPU Mode)</span>';
      }
    } catch (fallbackErr) {
      console.error("[VisionAI] Model load failed:", fallbackErr);
      if (modelStatusEl) {
        modelStatusEl.className = "status-indicator";
        modelStatusEl.style.borderColor = "#ef4444";
        modelStatusEl.style.color = "#f87171";
        modelStatusEl.innerHTML = '<span class="status-dot" style="background:#ef4444;"></span><span class="status-text">Model Load Failed</span>';
      }
      showToast("AI model failed to load. Check console.", "❌");
    }
  }
}

// ===========================================================================
// Bounding box drawing
// ===========================================================================
function drawStyledDetectionBox(ctx, x, y, w, h, entityInfo, score, isNightVision = false) {
  const icon = entityInfo.emoji || "📦";
  const confText = `${Math.round(score * 100)}%`;
  const nvTag = isNightVision ? " 🌙" : "";
  const badgeText = `${icon} ${entityInfo.title} [${confText}]${nvTag}`;

  ctx.fillStyle = entityInfo.fill;
  ctx.fillRect(x, y, w, h);

  ctx.lineWidth = 2.5;
  ctx.strokeStyle = isNightVision ? "#38bdf8" : entityInfo.stroke;
  ctx.strokeRect(x, y, w, h);

  const cornerLen = Math.min(16, Math.min(w, h) / 3);
  ctx.lineWidth = 4;
  ctx.strokeStyle = isNightVision ? "#38bdf8" : "#ffffff";

  ctx.beginPath();
  ctx.moveTo(x, y + cornerLen);
  ctx.lineTo(x, y);
  ctx.lineTo(x + cornerLen, y);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(x + w, y + h - cornerLen);
  ctx.lineTo(x + w, y + h);
  ctx.lineTo(x + w - cornerLen, y + h);
  ctx.stroke();

  ctx.font = "bold 13px 'JetBrains Mono', monospace";
  const textWidth = ctx.measureText(badgeText).width;
  const badgeH = 24;
  const badgeW = textWidth + 16;
  const badgeY = Math.max(0, y - badgeH - 3);

  ctx.fillStyle = isNightVision ? "#0369a1" : entityInfo.badge;
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(x, badgeY, badgeW, badgeH, 4);
  else ctx.rect(x, badgeY, badgeW, badgeH);
  ctx.fill();

  ctx.fillStyle = "#ffffff";
  ctx.fillText(badgeText, x + 8, badgeY + 17);
}

// ===========================================================================
// Multi-view renderer
// ===========================================================================
function renderCanvasView() {
  if (!cachedRawImg || !staticCanvas || !staticCtx) return;

  staticCanvas.width = cachedRawImg.naturalWidth;
  staticCanvas.height = cachedRawImg.naturalHeight;

  if (currentViewMode === "raw") {
    staticCtx.drawImage(cachedRawImg, 0, 0);
  } else if (currentViewMode === "illuminated") {
    if (cachedEnhancedCanvas) staticCtx.drawImage(cachedEnhancedCanvas, 0, 0);
    else staticCtx.drawImage(cachedRawImg, 0, 0);
  } else {
    const shouldDrawIlluminated = cachedEnhancedCanvas && cachedDetections.some(d => d.isNightVision);
    if (shouldDrawIlluminated) staticCtx.drawImage(cachedEnhancedCanvas, 0, 0);
    else staticCtx.drawImage(cachedRawImg, 0, 0);

    cachedDetections.forEach(det => {
      const [x, y, w, h] = det.bbox;
      drawStyledDetectionBox(staticCtx, x, y, w, h, det.info, det.score, det.isNightVision);
    });
  }
}

// ===========================================================================
// Static image inspector
// ===========================================================================
async function analyzeStaticImage(imgSrc) {
  if (!isModelsReady || (!cocoModel && !classifierModel)) {
    showToast("AI Models are still initializing. Please wait.", "⚠️");
    return;
  }

  currentLoadedImageSrc = imgSrc;
  const tStart = performance.now();

  staticPlaceholder.style.display = "flex";
  const ph3 = staticPlaceholder.querySelector("h3");
  const pp = staticPlaceholder.querySelector("p");
  if (ph3) ph3.textContent = "Neural Scanning Active...";
  if (pp) pp.textContent = "Running Dual Neural Pipeline...";
  if (scanlineLaser && !prefersReducedMotion) scanlineLaser.style.display = "block";

  const img = new Image();
  img.crossOrigin = "anonymous";
  img.src = imgSrc;

  img.onload = async () => {
    cachedRawImg = img;
    staticCanvas.width = img.naturalWidth;
    staticCanvas.height = img.naturalHeight;

    const ambientLux = measureLuminance(img);

    const shouldEngageNightVision =
      staticNightVisionMode === "on" ||
      (staticNightVisionMode === "auto" && ambientLux < 25);

    let inferenceSource = img;
    if (shouldEngageNightVision) {
      cachedEnhancedCanvas = createLowLightEnhancedCanvas(img, staticNightGain, staticNightGamma);
      inferenceSource = cachedEnhancedCanvas;
    } else {
      cachedEnhancedCanvas = null;
    }

    const minConf = parseFloat(staticConfSlider.value) / 100;
    const candidateDetections = [];

    let cocoPredictions = [];
    try {
      if (cocoModel) cocoPredictions = await cocoModel.detect(inferenceSource);
    } catch (err) {
      console.warn("[VisionAI] COCO detection error:", err);
    }
    const validCoco = cocoPredictions.filter(p => p.score >= Math.max(0.10, minConf * 0.55));

    const cropCanvas = document.createElement("canvas");
    const cropCtx = cropCanvas.getContext("2d", { willReadFrequently: true });

    const saliencyBox = extractForegroundSaliencyBox(inferenceSource);
    if (saliencyBox && classifierModel) {
      const [sx, sy, sw, sh] = saliencyBox;
      if (sw >= 20 && sh >= 20) {
        cropCanvas.width = sw; cropCanvas.height = sh;
        cropCtx.drawImage(inferenceSource, sx, sy, sw, sh, 0, 0, sw, sh);
        try {
          const salClasses = await classifierModel.classify(cropCanvas, 4);
          if (salClasses && salClasses.length > 0) {
            const topSal = salClasses[0];
            const salInfo = resolveEntityInfo(topSal.className);
            if (topSal.probability >= 0.15 && salInfo.category !== "Object") {
              candidateDetections.push({
                bbox: [sx, sy, sw, sh],
                info: salInfo,
                score: Math.min(0.96, Math.max(0.40, topSal.probability * 1.5)),
                isNightVision: shouldEngageNightVision
              });
            }
          }
        } catch (e) { /* ignore */ }
      }
    }

    if (classifierModel) {
      const spatialPatches = extractSpatialPatches(inferenceSource);
      const patchResults = await Promise.all(spatialPatches.map(async (patch) => {
        const [px, py, pw, ph] = patch.box;
        if (pw < 35 || ph < 35) return null;

        const pc = document.createElement("canvas");
        pc.width = pw; pc.height = ph;
        const pctx = pc.getContext("2d", { willReadFrequently: true });
        pctx.drawImage(inferenceSource, px, py, pw, ph, 0, 0, pw, ph);

        try {
          const classes = await classifierModel.classify(pc, 3);
          if (!classes || classes.length === 0) return null;

          const top = classes[0];
          const info = resolveEntityInfo(top.className);
          const lowerName = top.className.toLowerCase();

          const isEverydayTarget =
            info.category === "Stationery / Tool" ||
            info.category === "Accessories" ||
            info.category === "Kitchenware" ||
            info.category === "Animal" ||
            (info.category === "Electronics" && !lowerName.includes("screen") && !lowerName.includes("monitor"));

          if (!isEverydayTarget || top.probability < 0.18) return null;

          const patchSaliency = extractForegroundSaliencyBox(pc);
          let bbox = [px, py, pw, ph];
          if (patchSaliency) {
            bbox = [px + patchSaliency[0], py + patchSaliency[1], patchSaliency[2], patchSaliency[3]];
          }
          return {
            bbox,
            info,
            score: Math.min(0.95, top.probability * 1.25),
            isNightVision: shouldEngageNightVision
          };
        } catch { return null; }
      }));
      patchResults.forEach(r => { if (r) candidateDetections.push(r); });
    }

    let fullImageClasses = [];
    try {
      if (classifierModel) fullImageClasses = await classifierModel.classify(inferenceSource, 5);
    } catch (err) { /* ignore */ }

    const cocoDisambig = await Promise.all(validCoco.map(pred =>
      disambiguateEntityCached(pred.class, pred.score, pred.bbox, inferenceSource, classifierModel, false)
        .then(d => ({
          bbox: pred.bbox,
          info: d.info,
          score: d.score,
          isNightVision: shouldEngageNightVision
        }))
    ));
    cocoDisambig.forEach(d => candidateDetections.push(d));

    if (candidateDetections.length === 0 && fullImageClasses.length > 0) {
      const topPred = fullImageClasses[0];
      const resolved = resolveEntityInfo(topPred.className);
      if (topPred.probability >= Math.min(0.12, minConf)) {
        candidateDetections.push({
          bbox: [img.naturalWidth * 0.15, img.naturalHeight * 0.15, img.naturalWidth * 0.70, img.naturalHeight * 0.70],
          info: resolved,
          score: topPred.probability,
          isNightVision: shouldEngageNightVision
        });
      }
    }

    const filteredCandidates = candidateDetections.filter(d => d.score >= Math.max(0.14, minConf * 0.65));
    const detections = applyNMS(filteredCandidates, 0.40);
    cachedDetections = detections;

    renderCanvasView();

    if (scanlineLaser) scanlineLaser.style.display = "none";
    staticPlaceholder.style.display = "none";
    if (viewModeBar) viewModeBar.style.display = "flex";

    staticTelemetry.style.display = "grid";
    staticRes.textContent = `${img.naturalWidth} × ${img.naturalHeight}`;

    if (staticLum) {
      const lumTag = ambientLux < 25 ? " (Low Light)" : " (Normal)";
      staticLum.textContent = `${ambientLux}% Lux${lumTag}`;
    }

    if (staticNvStatus) {
      if (shouldEngageNightVision) {
        staticNvStatus.innerHTML = `<span style="color:#38bdf8;font-weight:700;">🌙 ACTIVE (${staticNightGain.toFixed(1)}x)</span>`;
      } else {
        staticNvStatus.innerHTML = `<span style="color:var(--text-muted);">Inactive</span>`;
      }
    }

    let animalCount = 0, toolAndObjectCount = 0;
    detections.forEach(det => {
      if (det.info.category === "Animal") animalCount++;
      else toolAndObjectCount++;
    });
    staticAnimals.textContent = animalCount;
    staticObjects.textContent = toolAndObjectCount;

    inspectorTableContainer.style.display = "block";
    if (detections.length === 0) {
      inspectorTableBody.innerHTML = `
        <tr>
          <td colspan="4" style="text-align:center;color:var(--text-sub);padding:2rem;">
            No entities detected above ${Math.round(minConf * 100)}% confidence.
            ${ambientLux < 25 ? "💡 Darkness detected! Try Night Vision 'Always ON'." : "Try lowering the threshold slider."}
          </td>
        </tr>`;
    } else {
      inspectorTableBody.innerHTML = detections.map(det => {
        const catBadge = `<span style="background:${det.info.fill};color:${det.info.stroke};border:1px solid ${det.info.stroke};padding:4px 10px;border-radius:9999px;font-weight:600;font-size:0.8rem;">${det.info.category}</span>`;
        const nvBadge = det.isNightVision ? `<span style="margin-left:6px;font-size:0.75rem;color:#38bdf8;background:rgba(56,189,248,0.15);border:1px solid rgba(56,189,248,0.3);padding:2px 6px;border-radius:4px;">🌙 Night Boost</span>` : "";
        const [x, y, w, h] = det.bbox.map(Math.round);
        return `
          <tr>
            <td>${catBadge}${nvBadge}</td>
            <td><code style="color:#fff;font-size:1.05rem;font-weight:700;">${det.info.emoji} ${det.info.title}</code></td>
            <td><strong style="color:var(--emerald);font-size:1.05rem;">${(det.score * 100).toFixed(1)}%</strong></td>
            <td><code>[x: ${x}, y: ${y}, w: ${w}, h: ${h}]</code></td>
          </tr>`;
      }).join("");
      showToast(`Detected: ${detections.map(d => d.info.title).join(", ")}`, "🎯");
    }

    // ============ Update analytics from static image ============
    const latencyMs = Math.round(performance.now() - tStart);
    updateAnalytics(detections, latencyMs, "static");
  };

  img.onerror = () => {
    showToast("Failed to load image: " + imgSrc, "❌");
    if (scanlineLaser) scanlineLaser.style.display = "none";
    staticPlaceholder.style.display = "flex";
    const ph3 = staticPlaceholder.querySelector("h3");
    const pp = staticPlaceholder.querySelector("p");
    if (ph3) ph3.textContent = "Image Load Error";
    if (pp) pp.textContent = "Could not decode or download image source.";
  };
}

// ===========================================================================
// Camera
// ===========================================================================
async function startCamera() {
  if (!isModelsReady) {
    showToast("AI Models are still initializing. Please wait.", "⚠️");
    return;
  }
  try {
    btnToggleCamera.textContent = "Connecting...";
    btnToggleCamera.disabled = true;

    const constraints = {
      video: { facingMode: currentFacingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
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
      showToast("Live Camera Feed Active", "📷");
    };
  } catch (err) {
    console.error("Camera access error:", err);
    btnToggleCamera.textContent = "▶ Start Live Camera";
    btnToggleCamera.disabled = false;
    showToast("Could not access camera: " + err.message, "❌");
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
  if (videoEl) videoEl.srcObject = null;
  if (overlayCtx) overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  if (placeholderOverlay) placeholderOverlay.style.display = "flex";
  if (btnToggleCamera) {
    btnToggleCamera.textContent = "▶ Start Live Camera";
    btnToggleCamera.classList.remove("btn-secondary");
    btnToggleCamera.classList.add("btn-primary");
  }
  if (btnSnapshot) btnSnapshot.disabled = true;
  if (teleFps) teleFps.textContent = "0.0 FPS";
  if (teleLat) teleLat.textContent = "-- ms";
  if (teleCount) teleCount.textContent = "0";
  if (teleLum) teleLum.textContent = "--%";
  if (liveDetectionsList) liveDetectionsList.innerHTML = '<div class="empty-state">Camera is stopped.</div>';
  showToast("Live Camera Stopped", "⏹");
}

async function detectVideoLoop() {
  if (!isCameraRunning) return;

  const tStart = performance.now();

  if (videoEl.readyState >= 2 && cocoModel) {
    let ambientLux = 50;
    if (frameCount % 10 === 0) {
      ambientLux = measureLuminance(videoEl);
      if (teleLum) teleLum.textContent = `${ambientLux}% Lux`;
    }

    const shouldEngageNightVision =
      liveNightVisionMode === "on" ||
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

    if (classifierModel && frameCount % 6 === 0 && filtered.length <= 3) {
      try {
        const vw = overlayCanvas.width, vh = overlayCanvas.height;
        const cx = Math.round(vw * 0.22), cy = Math.round(vh * 0.22);
        const cw = Math.round(vw * 0.56), ch = Math.round(vh * 0.56);

        const liveCropCanvas = document.createElement("canvas");
        liveCropCanvas.width = cw; liveCropCanvas.height = ch;
        const liveCropCtx = liveCropCanvas.getContext("2d", { willReadFrequently: true });
        liveCropCtx.drawImage(videoSource, cx, cy, cw, ch, 0, 0, cw, ch);

        classifierModel.classify(liveCropCanvas, 2).then(classes => {
          if (classes && classes.length > 0) {
            const top = classes[0];
            const info = resolveEntityInfo(top.className);
            if ((info.category === "Stationery / Tool" || info.category === "Accessories" || info.category === "Kitchenware") && top.probability >= 0.22) {
              const liveSal = extractForegroundSaliencyBox(liveCropCanvas);
              let liveBox = [cx, cy, cw, ch];
              if (liveSal) liveBox = [cx + liveSal[0], cy + liveSal[1], liveSal[2], liveSal[3]];
              lastLiveEverydayItem = {
                bbox: liveBox, info, score: Math.min(0.92, top.probability * 1.3),
                isNightVision: shouldEngageNightVision, timestamp: performance.now()
              };
            }
          }
        }).catch(() => {});
      } catch { /* ignore */ }
    }

    const disambigResults = await Promise.all(filtered.map(pred =>
      disambiguateEntityCached(pred.class, pred.score, pred.bbox, videoSource, classifierModel, true)
    ));

    const finalLiveEntities = disambigResults.map((d, i) => {
      const pred = filtered[i];
      let finalInfo = d.info;
      let finalScore = d.score;

      if (lastLiveEverydayItem && performance.now() - lastLiveEverydayItem.timestamp < 350) {
        if (computeIoU(pred.bbox, lastLiveEverydayItem.bbox) > 0.35) {
          finalInfo = lastLiveEverydayItem.info;
          finalScore = Math.max(finalScore, lastLiveEverydayItem.score);
        }
      }
      return { bbox: pred.bbox, info: finalInfo, score: finalScore, isNightVision: shouldEngageNightVision };
    });

    if (lastLiveEverydayItem && performance.now() - lastLiveEverydayItem.timestamp < 350) {
      const isAlreadyIn = finalLiveEntities.some(e => computeIoU(e.bbox, lastLiveEverydayItem.bbox) > 0.40);
      if (!isAlreadyIn) finalLiveEntities.push(lastLiveEverydayItem);
    }

    const latency = Math.round(performance.now() - tStart);

    frameCount++;
    const now = performance.now();
    if (now - lastFrameTime >= 500) {
      fps = ((frameCount * 1000) / (now - lastFrameTime)).toFixed(1);
      if (teleFps) teleFps.textContent = `${fps} FPS`;
      frameCount = 0;
      lastFrameTime = now;
    }

    if (teleLat) teleLat.textContent = `${latency} ms`;
    if (teleCount) teleCount.textContent = finalLiveEntities.length;

    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    finalLiveEntities.forEach(det => {
      const [x, y, w, h] = det.bbox;
      drawStyledDetectionBox(overlayCtx, x, y, w, h, det.info, det.score, det.isNightVision);
    });

    updateLiveDetectionsList(finalLiveEntities);

    // ============ Update analytics from live frame ============
    updateAnalytics(finalLiveEntities, latency, "live");
  }

  animationFrameId = requestAnimationFrame(detectVideoLoop);
}

function updateLiveDetectionsList(entities) {
  if (!liveDetectionsList) return;
  if (!entities || entities.length === 0) {
    liveDetectionsList.innerHTML = '<div class="empty-state">No entities currently above threshold.</div>';
    return;
  }
  liveDetectionsList.innerHTML = entities.map(e => {
    const confPct = Math.round(e.score * 100);
    const nvPill = e.isNightVision ? `<span style="font-size:0.72rem;color:#38bdf8;margin-left:4px;">🌙</span>` : "";
    return `
      <div class="entity-chip" style="border-left:3px solid ${e.info.stroke};">
        <span class="entity-label">${e.info.emoji} ${e.info.title}${nvPill}</span>
        <span class="entity-conf">${confPct}%</span>
      </div>`;
  }).join("");
}

// ===========================================================================
// Event listeners
// ===========================================================================
if (btnToggleCamera) {
  btnToggleCamera.addEventListener("click", () => {
    if (isCameraRunning) stopCamera();
    else startCamera();
  });
}

if (btnStartCameraHero) {
  btnStartCameraHero.addEventListener("click", () => {
    const liveSec = $("live-feed");
    if (liveSec) liveSec.scrollIntoView({ behavior: prefersReducedMotion ? "auto" : "smooth" });
    if (!isCameraRunning) startCamera();
  });
}

if (btnFlipCamera) {
  btnFlipCamera.addEventListener("click", () => {
    currentFacingMode = currentFacingMode === "user" ? "environment" : "user";
    showToast(`Switched to ${currentFacingMode} camera`, "🔄");
    if (isCameraRunning) { stopCamera(); startCamera(); }
  });
}

if (btnNightVisionLive) {
  btnNightVisionLive.addEventListener("click", () => {
    if (liveNightVisionMode === "auto") {
      liveNightVisionMode = "on";
      btnNightVisionLive.textContent = "🌙 Night: ON";
      btnNightVisionLive.style.borderColor = "#38bdf8";
      btnNightVisionLive.style.color = "#38bdf8";
      showToast("Live Night Vision: Always ON", "🌙");
    } else if (liveNightVisionMode === "on") {
      liveNightVisionMode = "off";
      btnNightVisionLive.textContent = "🌙 Night: OFF";
      btnNightVisionLive.style.borderColor = "rgba(99, 102, 241, 0.4)";
      btnNightVisionLive.style.color = "#a5b4fc";
      showToast("Live Night Vision: OFF", "⚪");
    } else {
      liveNightVisionMode = "auto";
      btnNightVisionLive.textContent = "🌙 Night: AUTO";
      btnNightVisionLive.style.borderColor = "rgba(99, 102, 241, 0.4)";
      btnNightVisionLive.style.color = "#a5b4fc";
      showToast("Live Night Vision: AUTO Mode", "✨");
    }
  });
}

if (btnSnapshot) {
  btnSnapshot.addEventListener("click", () => {
    if (!isCameraRunning) return;

    if (shutterFlash && !prefersReducedMotion) {
      shutterFlash.classList.remove("flash-active");
      void shutterFlash.offsetWidth;
      shutterFlash.classList.add("flash-active");
    }

    const snapCanvas = document.createElement("canvas");
    snapCanvas.width = videoEl.videoWidth;
    snapCanvas.height = videoEl.videoHeight;
    snapCanvas.getContext("2d").drawImage(videoEl, 0, 0);
    const dataUrl = snapCanvas.toDataURL("image/jpeg", 0.95);

    showToast("📸 Snapshot Captured! Analyzing...", "✨");

    setTimeout(() => {
      const inspectorSec = $("image-inspector");
      if (inspectorSec) inspectorSec.scrollIntoView({ behavior: prefersReducedMotion ? "auto" : "smooth" });
      analyzeStaticImage(dataUrl);
    }, 150);
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
    if (currentLoadedImageSrc) analyzeStaticImage(currentLoadedImageSrc);
  });
}

if (staticNightModeSelect) {
  staticNightModeSelect.addEventListener("change", (e) => {
    staticNightVisionMode = e.target.value;
    showToast(`Night Vision Mode: ${e.target.value.toUpperCase()}`, "🌙");
    if (currentLoadedImageSrc) analyzeStaticImage(currentLoadedImageSrc);
  });
}

if (staticNightGainSlider) {
  staticNightGainSlider.addEventListener("input", (e) => {
    staticNightGain = parseFloat(e.target.value) / 10.0;
    if (staticNightGainVal) staticNightGainVal.textContent = `${staticNightGain.toFixed(1)}x`;
    if (currentLoadedImageSrc) analyzeStaticImage(currentLoadedImageSrc);
  });
}

if (btnViewAugmented) {
  btnViewAugmented.addEventListener("click", () => {
    currentViewMode = "augmented";
    [btnViewAugmented, btnViewIlluminated, btnViewRaw].forEach(b => b && b.classList.remove("active"));
    btnViewAugmented.classList.add("active");
    renderCanvasView();
    showToast("Display: Augmented AI View", "👁️");
  });
}

if (btnViewIlluminated) {
  btnViewIlluminated.addEventListener("click", () => {
    currentViewMode = "illuminated";
    [btnViewAugmented, btnViewIlluminated, btnViewRaw].forEach(b => b && b.classList.remove("active"));
    btnViewIlluminated.classList.add("active");
    renderCanvasView();
    showToast("Display: Night Sensor Tensor Matrix", "🌙");
  });
}

if (btnViewRaw) {
  btnViewRaw.addEventListener("click", () => {
    currentViewMode = "raw";
    [btnViewAugmented, btnViewIlluminated, btnViewRaw].forEach(b => b && b.classList.remove("active"));
    btnViewRaw.classList.add("active");
    renderCanvasView();
    showToast("Display: Raw Untouched Source", "🖼️");
  });
}

benchmarkTabs.forEach(tab => {
  tab.addEventListener("click", () => {
    benchmarkTabs.forEach(t => {
      t.classList.remove("active");
      t.setAttribute("aria-selected", "false");
    });
    tab.classList.add("active");
    tab.setAttribute("aria-selected", "true");
    const filter = tab.getAttribute("data-filter");

    sampleChips.forEach(chip => {
      const cat = chip.getAttribute("data-category");
      chip.style.display = (filter === "all" || cat === filter) ? "inline-flex" : "none";
    });
    showToast(`Filtered: ${tab.textContent.trim()}`, "⚡");
  });
});

if (fileInput) {
  fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
      sampleChips.forEach(c => c.classList.remove("active"));
      const reader = new FileReader();
      reader.onload = (event) => {
        showToast(`Loaded ${file.name}`, "📁");
        analyzeStaticImage(event.target.result);
      };
      reader.readAsDataURL(file);
    }
  });
}

if (dropzone) {
  dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      sampleChips.forEach(c => c.classList.remove("active"));
      const reader = new FileReader();
      reader.onload = (event) => {
        showToast(`Dropped ${file.name}`, "📁");
        analyzeStaticImage(event.target.result);
      };
      reader.readAsDataURL(file);
    }
  });
}

sampleChips.forEach(chip => {
  chip.addEventListener("click", () => {
    sampleChips.forEach(c => c.classList.remove("active"));
    chip.classList.add("active");
    const samplePath = chip.getAttribute("data-sample");
    const sampleName = chip.textContent.trim();
    if (samplePath) {
      showToast(`Analyzing ${sampleName}...`, "🔍");
      analyzeStaticImage(samplePath);
    }
  });
});

if (btnCopyDetections) {
  btnCopyDetections.addEventListener("click", () => {
    if (!cachedDetections || cachedDetections.length === 0) {
      showToast("No detections to copy!", "⚠️");
      return;
    }
    const cleanData = cachedDetections.map(d => ({
      label: d.info.title,
      category: d.info.category,
      confidence: `${(d.score * 100).toFixed(1)}%`,
      bbox: d.bbox.map(Math.round),
      nightVisionBoosted: d.isNightVision
    }));
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(JSON.stringify(cleanData, null, 2))
        .then(() => showToast("Detections JSON copied!", "📋"))
        .catch(() => showToast("Failed to copy", "❌"));
    } else {
      showToast("Clipboard API not available", "⚠️");
    }
  });
}

if (btnClearTimeline) {
  btnClearTimeline.addEventListener("click", () => {
    analyticsState.timeline = [];
    renderTimeline();
    showToast("Timeline cleared", "🧹");
  });
}

// Redraw charts on resize
let _resizeTimer = null;
window.addEventListener("resize", () => {
  if (_resizeTimer) clearTimeout(_resizeTimer);
  _resizeTimer = setTimeout(() => {
    drawConfidenceChart();
    drawCategoryDonut();
    drawLatencyChart();
  }, 150);
});

// ===========================================================================
// Lifecycle cleanup
// ===========================================================================
window.addEventListener("pagehide", () => {
  if (isCameraRunning) stopCamera();
});
window.addEventListener("beforeunload", () => {
  if (isCameraRunning) stopCamera();
});

// ===========================================================================
// Init
// ===========================================================================
window.addEventListener("DOMContentLoaded", () => {
  // Wire up analytics canvas refs
  analyticsState.confCanvas = $("chart-confidence");
  analyticsState.confCtx = analyticsState.confCanvas ? analyticsState.confCanvas.getContext("2d") : null;
  analyticsState.catCanvas = $("chart-categories");
  analyticsState.catCtx = analyticsState.catCanvas ? analyticsState.catCanvas.getContext("2d") : null;
  analyticsState.latCanvas = $("chart-latency");
  analyticsState.latCtx = analyticsState.latCanvas ? analyticsState.latCanvas.getContext("2d") : null;

  // Initial empty renders (so empty states show)
  drawConfidenceChart();
  drawCategoryDonut();
  drawLatencyChart();
  renderTimeline();

  // Boot the model
  initModel();
});
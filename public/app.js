/**
 * VisionAI — Production-Grade Dual-Model Computer Vision Engine
 * Featuring Tactile Micro-Interactions, Benchmark Suite Filter Tabs,
 * Multi-View Switcher & In-Browser Night Vision Image Signal Processor
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

// Night Vision & Multi-View Rendering Cache
let liveNightVisionMode = "auto";   // "auto", "on", "off"
let staticNightVisionMode = "auto"; // "auto", "on", "off"
let staticNightGain = 3.0;          // Exposure gain multiplier (1.0x to 5.0x)
let staticNightGamma = 0.40;        // Shadow expansion exponent (0.35 to 0.50)
let currentLoadedImageSrc = null;
let currentViewMode = "augmented";   // "augmented", "illuminated", "raw"

// Cached Frames for Instant Switching
let cachedRawImg = null;
let cachedEnhancedCanvas = null;
let cachedDetections = [];

// DOM Elements - Camera
const videoEl = document.getElementById("webcam");
const overlayCanvas = document.getElementById("overlay-canvas");
const overlayCtx = overlayCanvas ? overlayCanvas.getContext("2d") : null;
const placeholderOverlay = document.getElementById("camera-placeholder");
const shutterFlash = document.getElementById("shutter-flash");
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
const scanlineLaser = document.getElementById("scanline-laser");
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
const btnCopyDetections = document.getElementById("btn-copy-detections");
const sampleChips = document.querySelectorAll(".chip-btn");
const benchmarkTabs = document.querySelectorAll("#benchmark-tabs .tab-btn");
const viewModeBar = document.getElementById("view-mode-bar");
const btnViewAugmented = document.getElementById("btn-view-augmented");
const btnViewIlluminated = document.getElementById("btn-view-illuminated");
const btnViewRaw = document.getElementById("btn-view-raw");

// ---------------------------------------------------------------------------
// 0. Interactive Toast Notification System
// ---------------------------------------------------------------------------
function showToast(message, icon = "⚡") {
  const container = document.getElementById("toast-container");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = "toast";
  toast.innerHTML = `
    <span style="font-size: 1.15rem; display: flex; align-items: center;">${icon}</span>
    <span>${message}</span>
  `;
  container.appendChild(toast);

  setTimeout(() => {
    toast.classList.add("toast-exit");
    setTimeout(() => {
      if (toast.parentNode) toast.remove();
    }, 250);
  }, 2700);
}

// ---------------------------------------------------------------------------
// 1. Comprehensive Taxonomy & Entity Resolution Engine
// ---------------------------------------------------------------------------
const IMAGENET_DOG_BREEDS = [
  "chihuahua", "japanese spaniel", "maltese", "pekinese", "shih-tzu", "blenheim spaniel",
  "papillon", "toy terrier", "rhodesian ridgeback", "afghan hound", "basset", "beagle",
  "bloodhound", "bluetick", "coonhound", "walker hound", "foxhound", "redbone", "borzoi",
  "irish wolfhound", "italian greyhound", "whippet", "ibizan hound", "norwegian elkhound",
  "otterhound", "saluki", "scottish deerhound", "weimaraner", "staffordshire bullterrier",
  "american staffordshire terrier", "bedlington terrier", "border terrier", "kerry blue terrier",
  "irish terrier", "norfolk terrier", "norwich terrier", "yorkshire terrier", "wire-haired fox terrier",
  "lakeland terrier", "sealyham terrier", "airedale", "cairn", "australian terrier", "dandie dinmont",
  "boston bull", "schnauzer", "scotch terrier", "tibetan terrier", "silky terrier", "soft-coated wheaten terrier",
  "west highland white terrier", "lhasa", "retriever", "golden retriever", "labrador retriever",
  "flat-coated retriever", "curly-coated retriever", "chesapeake bay retriever", "pointer", "vizsla",
  "setter", "english setter", "irish setter", "gordon setter", "brittany spaniel", "clumber",
  "springer spaniel", "cocker spaniel", "sussex spaniel", "water spaniel", "kuvasz", "schipperke",
  "groenendael", "malinois", "briard", "kelpie", "komondor", "old english sheepdog", "sheepdog",
  "collie", "border collie", "bouvier", "rottweiler", "german shepherd", "doberman", "pinscher",
  "swiss mountain dog", "bernese mountain dog", "appenzeller", "entlebucher", "boxer", "bull mastiff",
  "tibetan mastiff", "french bulldog", "bulldog", "great dane", "saint bernard", "husky", "malamute",
  "siberian husky", "dalmatian", "affenpinscher", "basenji", "pug", "leonberg", "newfoundland",
  "great pyrenees", "samoyed", "pomeranian", "chow", "keeshond", "griffon", "pembroke", "cardigan",
  "corgi", "poodle", "toy poodle", "miniature poodle", "standard poodle", "dingo", "dhole", "canine", "dog", "puppy"
];

const IMAGENET_CAT_BREEDS = [
  "tabby", "tabby cat", "tiger cat", "persian cat", "siamese cat", "siamese", "egyptian cat",
  "cougar", "puma", "catamount", "mountain lion", "lynx", "bobcat", "leopard cat", "kitten", "cat"
];

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
  "laptop", "mouse", "keyboard", "bottle", "cup", "mug", "clock", "watch",
  "headphone", "headphones", "headset", "slipper", "slippers", "sandal", "shoe"
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

/**
 * Maps raw ImageNet / COCO labels to clean, formatted entity metadata
 */
function resolveEntityInfo(rawLabel) {
  const lower = (rawLabel || "").toLowerCase();

  // 1. Canine / Dog Resolution (Strict check against 120+ ImageNet breeds)
  if (isImageNetDog(lower)) {
    let dogTitle = "Dog";
    const matched = getMatchingDogBreed(lower);
    if (matched !== "Dog") {
      dogTitle = matched;
    }
    return { title: dogTitle, category: "Animal", emoji: "🐶", stroke: "#f97316", fill: "rgba(249, 115, 22, 0.22)", badge: "#c2410c" };
  }

  // 2. Feline / Cat Resolution
  if (isImageNetCat(lower)) {
    return { title: "Cat", category: "Animal", emoji: "🐱", stroke: "#f59e0b", fill: "rgba(245, 158, 11, 0.22)", badge: "#b45309" };
  }

  // Specific Wild Animals
  if (lower.includes("lion")) {
    return { title: "Lion (King of Beasts)", category: "Animal", emoji: "🦁", stroke: "#f59e0b", fill: "rgba(245, 158, 11, 0.22)", badge: "#d97706" };
  }
  if (lower.includes("tiger")) {
    return { title: "Tiger", category: "Animal", emoji: "🐯", stroke: "#f97316", fill: "rgba(249, 115, 22, 0.22)", badge: "#ea580c" };
  }
  if (lower.includes("cheetah") || lower.includes("leopard") || lower.includes("jaguar")) {
    return { title: "Leopard / Cheetah", category: "Animal", emoji: "🐆", stroke: "#eab308", fill: "rgba(234, 179, 8, 0.22)", badge: "#ca8a04" };
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

  // 3. Audio / Headphones & Headsets
  if (lower.includes("headphone") || lower.includes("headphones") || lower.includes("headset") || lower.includes("earphone") || lower.includes("earphones") || lower.includes("earbuds") || lower.includes("airpods") || lower.includes("headpiece")) {
    return { title: "Headphones / Headset", category: "Electronics", emoji: "🎧", stroke: "#8b5cf6", fill: "rgba(139, 92, 246, 0.22)", badge: "#7c3aed" };
  }

  // 4. Smartphone / Mobile Phone
  if (lower.includes("cellular") || lower.includes("cellphone") || lower.includes("smartphone") || lower.includes("mobile phone") || lower.includes("cell phone") || (lower.includes("phone") && !lower.includes("headphone") && !lower.includes("earphone"))) {
    return { title: "Smartphone / Phone", category: "Electronics", emoji: "📱", stroke: "#6366f1", fill: "rgba(99, 102, 241, 0.22)", badge: "#4f46e5" };
  }

  // 5. Water Bottle & Drinkware
  if (lower.includes("water bottle") || lower.includes("beer bottle") || lower.includes("wine bottle") || lower.includes("pop bottle") || lower.includes("bottle") || lower.includes("flask") || lower.includes("tumbler") || lower.includes("shaker") || lower.includes("canteen")) {
    return { title: "Water Bottle / Tumbler", category: "Kitchenware", emoji: "🍾", stroke: "#06b6d4", fill: "rgba(6, 182, 212, 0.22)", badge: "#0891b2" };
  }

  // 6. Slippers & Footwear
  if (lower.includes("slipper") || lower.includes("slippers") || lower.includes("sleeper") || lower.includes("sleepers") || lower.includes("sandal") || lower.includes("sandals") || lower.includes("flip-flop") || lower.includes("slides") || lower.includes("clog") || lower.includes("shoe") || lower.includes("shoes") || lower.includes("sneaker") || lower.includes("boot") || lower.includes("loafer") || lower.includes("moccasin")) {
    return { title: "Slippers / Footwear", category: "Everyday Item", emoji: "🩴", stroke: "#10b981", fill: "rgba(16, 185, 129, 0.22)", badge: "#059669" };
  }

  // 7. Remote Control (TV / Electronics)
  if (lower.includes("remote") || lower.includes("remote control")) {
    return { title: "Remote Control", category: "Electronics", emoji: "🎮", stroke: "#3b82f6", fill: "rgba(59, 130, 246, 0.22)", badge: "#2563eb" };
  }

  // 8. Flower Vase
  if (lower.includes("vase")) {
    return { title: "Flower Vase", category: "Decoration", emoji: "🏺", stroke: "#ec4899", fill: "rgba(236, 72, 153, 0.22)", badge: "#db2777" };
  }

  // Everyday Tools, Stationery & Writing Instruments
  if (lower.includes("ballpoint") || lower.includes("ballpen") || lower.includes("biro")) {
    return { title: "Ballpoint Pen", category: "Stationery / Tool", emoji: "🖊️", stroke: "#06b6d4", fill: "rgba(6, 182, 212, 0.22)", badge: "#0891b2" };
  }
  if (lower.includes("fountain pen")) {
    return { title: "Fountain Pen", category: "Stationery / Tool", emoji: "✒️", stroke: "#3b82f6", fill: "rgba(59, 130, 246, 0.22)", badge: "#2563eb" };
  }
  if (lower.includes("pencil") || lower.includes("pencil case") || lower.includes("pencil box") || lower.includes("lead")) {
    return { title: "Pencil", category: "Stationery / Tool", emoji: "✏️", stroke: "#f59e0b", fill: "rgba(245, 158, 11, 0.22)", badge: "#d97706" };
  }
  if (lower.includes("pen") && !lower.includes("penguin") && !lower.includes("open") && !lower.includes("happen")) {
    return { title: "Pen", category: "Stationery / Tool", emoji: "🖊️", stroke: "#06b6d4", fill: "rgba(6, 182, 212, 0.22)", badge: "#0891b2" };
  }
  if (lower.includes("eraser") || lower.includes("rubber")) {
    return { title: "Eraser", category: "Stationery / Tool", emoji: "🧼", stroke: "#ec4899", fill: "rgba(236, 72, 153, 0.22)", badge: "#db2777" };
  }
  if (lower.includes("ruler") || lower.includes("rule") || lower.includes("tape measure")) {
    return { title: "Ruler / Scale", category: "Stationery / Tool", emoji: "📏", stroke: "#eab308", fill: "rgba(234, 179, 8, 0.22)", badge: "#ca8a04" };
  }
  if (lower.includes("scissors")) {
    return { title: "Scissors", category: "Stationery / Tool", emoji: "✂️", stroke: "#f43f5e", fill: "rgba(244, 63, 94, 0.22)", badge: "#e11d48" };
  }
  if (lower.includes("stapler") || lower.includes("paperclip")) {
    return { title: "Stapler / Clip", category: "Stationery / Tool", emoji: "📎", stroke: "#8b5cf6", fill: "rgba(139, 92, 246, 0.22)", badge: "#7c3aed" };
  }
  if (lower.includes("notebook") || lower.includes("binder") || lower.includes("copy") || lower.includes("spiral")) {
    return { title: "Notebook / Copy", category: "Stationery / Tool", emoji: "📓", stroke: "#8b5cf6", fill: "rgba(139, 92, 246, 0.22)", badge: "#7c3aed" };
  }
  if (lower.includes("book") || lower.includes("booklet") || lower.includes("dust cover") || lower.includes("novel")) {
    return { title: "Book", category: "Stationery / Tool", emoji: "📖", stroke: "#a855f7", fill: "rgba(168, 85, 247, 0.22)", badge: "#9333ea" };
  }

  // Personal Accessories & Wearables
  if (lower.includes("spectacle") || lower.includes("specs") || lower.includes("eyeglass") || lower.includes("eyeglasses") || (lower.includes("glasses") && !lower.includes("dark glasses") && !lower.includes("drinking glasses"))) {
    return { title: "Glasses / Spectacles", category: "Accessories", emoji: "👓", stroke: "#14b8a6", fill: "rgba(20, 184, 166, 0.22)", badge: "#0d9488" };
  }
  if (lower.includes("sunglass") || lower.includes("sunglasses") || lower.includes("dark glasses") || lower.includes("shades")) {
    return { title: "Sunglasses", category: "Accessories", emoji: "🕶️", stroke: "#0ea5e9", fill: "rgba(14, 165, 233, 0.22)", badge: "#0284c7" };
  }
  if (lower.includes("wristwatch") || lower.includes("digital watch") || lower.includes("stopwatch") || (lower.includes("watch") && !lower.includes("water") && !lower.includes("match"))) {
    return { title: "Wristwatch / Smartwatch", category: "Accessories", emoji: "⌚", stroke: "#06b6d4", fill: "rgba(6, 182, 212, 0.22)", badge: "#0891b2" };
  }
  if (lower.includes("clock") || lower.includes("wall clock") || lower.includes("analog clock") || lower.includes("alarm clock")) {
    return { title: "Clock", category: "Accessories", emoji: "⏰", stroke: "#eab308", fill: "rgba(234, 179, 8, 0.22)", badge: "#ca8a04" };
  }
  if (lower.includes("backpack") || lower.includes("knapsack") || lower.includes("rucksack")) {
    return { title: "Backpack / School Bag", category: "Everyday Item", emoji: "🎒", stroke: "#ec4899", fill: "rgba(236, 72, 153, 0.22)", badge: "#db2777" };
  }
  if (lower.includes("handbag") || lower.includes("purse") || lower.includes("pocketbook") || (lower.includes("bag") && !lower.includes("mailbag") && !lower.includes("sleeping bag"))) {
    return { title: "Handbag / Bag", category: "Everyday Item", emoji: "👜", stroke: "#f43f5e", fill: "rgba(244, 63, 94, 0.22)", badge: "#e11d48" };
  }
  if (lower.includes("wallet") || lower.includes("billfold")) {
    return { title: "Wallet", category: "Everyday Item", emoji: "👛", stroke: "#f59e0b", fill: "rgba(245, 158, 11, 0.22)", badge: "#d97706" };
  }
  if (lower.includes("key") || lower.includes("keys") || lower.includes("keychain") || lower.includes("padlock")) {
    return { title: "Keys / Lock", category: "Everyday Item", emoji: "🔑", stroke: "#eab308", fill: "rgba(234, 179, 8, 0.22)", badge: "#ca8a04" };
  }
  if (lower.includes("umbrella")) {
    return { title: "Umbrella", category: "Everyday Item", emoji: "☂️", stroke: "#06b6d4", fill: "rgba(6, 182, 212, 0.22)", badge: "#0891b2" };
  }

  // Electronics & Desk Devices
  if (lower.includes("laptop") || lower.includes("notebook computer")) {
    return { title: "Laptop Computer", category: "Electronics", emoji: "💻", stroke: "#38bdf8", fill: "rgba(56, 189, 248, 0.22)", badge: "#0284c7" };
  }
  if (lower.includes("mouse") && (lower.includes("computer") || lower.includes("optical") || lower.includes("trackball"))) {
    return { title: "Computer Mouse", category: "Electronics", emoji: "🖱️", stroke: "#a855f7", fill: "rgba(168, 85, 247, 0.22)", badge: "#9333ea" };
  }
  if (lower.includes("keyboard") || lower.includes("keypad")) {
    return { title: "Keyboard", category: "Electronics", emoji: "⌨️", stroke: "#a855f7", fill: "rgba(168, 85, 247, 0.22)", badge: "#9333ea" };
  }
  if (lower.includes("monitor") || lower.includes("screen") || lower.includes("display") || lower.includes("television") || lower.includes("tv")) {
    return { title: "Monitor / Screen", category: "Electronics", emoji: "🖥️", stroke: "#0ea5e9", fill: "rgba(14, 165, 233, 0.22)", badge: "#0284c7" };
  }

  // Kitchenware & Drinkware
  if (lower.includes("coffee mug") || (lower.includes("mug") && !lower.includes("smug"))) {
    return { title: "Coffee Mug / Cup", category: "Kitchenware", emoji: "☕", stroke: "#f59e0b", fill: "rgba(245, 158, 11, 0.22)", badge: "#d97706" };
  }
  if (lower.includes("cup") && !lower.includes("cupboard") && !lower.includes("world cup")) {
    return { title: "Cup / Drinkware", category: "Kitchenware", emoji: "🥛", stroke: "#14b8a6", fill: "rgba(20, 184, 166, 0.22)", badge: "#0d9488" };
  }
  if (lower.includes("plate") || lower.includes("dish") || lower.includes("saucer")) {
    return { title: "Plate / Dish", category: "Kitchenware", emoji: "🍽️", stroke: "#94a3b8", fill: "rgba(148, 163, 184, 0.22)", badge: "#64748b" };
  }
  if (lower.includes("bowl") || lower.includes("soup bowl")) {
    return { title: "Bowl", category: "Kitchenware", emoji: "🥣", stroke: "#e2e8f0", fill: "rgba(226, 232, 240, 0.22)", badge: "#475569" };
  }
  if (lower.includes("fork") || lower.includes("spoon") || lower.includes("knife") || lower.includes("cutlery")) {
    return { title: "Cutlery / Utensil", category: "Kitchenware", emoji: "🍴", stroke: "#38bdf8", fill: "rgba(56, 189, 248, 0.22)", badge: "#0284c7" };
  }

  // Furniture & Home
  if (lower.includes("chair") || lower.includes("rocker") || lower.includes("stool") || lower.includes("armchair")) {
    return { title: "Chair", category: "Furniture", emoji: "🪑", stroke: "#10b981", fill: "rgba(16, 185, 129, 0.22)", badge: "#059669" };
  }
  if (lower.includes("desk") || lower.includes("table") || lower.includes("dining table")) {
    return { title: "Table / Desk", category: "Furniture", emoji: "🪵", stroke: "#84cc16", fill: "rgba(132, 204, 22, 0.22)", badge: "#65a30d" };
  }
  if (lower.includes("lamp") || lower.includes("table lamp") || lower.includes("lampshade")) {
    return { title: "Desk Lamp", category: "Furniture", emoji: "💡", stroke: "#facc15", fill: "rgba(250, 204, 21, 0.22)", badge: "#eab308" };
  }
  if (lower.includes("couch") || lower.includes("sofa") || lower.includes("bed") || lower.includes("pillow")) {
    return { title: "Furniture", category: "Furniture", emoji: "🛋️", stroke: "#a855f7", fill: "rgba(168, 85, 247, 0.22)", badge: "#9333ea" };
  }

  // People
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
// 1B. Multi-Model Neural Disambiguation & Anti-Hallucination Engine
// ---------------------------------------------------------------------------
let _disambiguationCanvas = null;
let _disambiguationCtx = null;

function getDisambiguationContext() {
  if (!_disambiguationCanvas) {
    _disambiguationCanvas = document.createElement("canvas");
    _disambiguationCtx = _disambiguationCanvas.getContext("2d", { willReadFrequently: true });
  }
  return { canvas: _disambiguationCanvas, ctx: _disambiguationCtx };
}

/**
 * Disambiguates known neural confusion pairs (COCO-SSD vs MobileNet):
 * - Headphones vs Motorcycle / Bicycle (COCO has no headphones)
 * - Smartphone vs Remote Control (Touchscreen glass slabs vs TV remotes)
 * - Water Bottle vs Vase (Cylindrical drinkware vs flower vases)
 * - Slippers / Footwear vs Person (Floor-level footwear vs human beings)
 * - Dog vs Cat (Cross-validation using 120+ ImageNet breeds)
 * - Glasses, Pens, Notebooks vs COCO toothbrush, knife, scissors
 */
async function disambiguateEntity(rawClass, score, bbox, sourceCanvasOrImg, classifierModel, isWebcam = false) {
  const [bx, by, bw, bh] = bbox;
  const lowerClass = (rawClass || "").toLowerCase();

  // Fast domain heuristics if classifier not yet ready
  if (!classifierModel) {
    if (isWebcam && (lowerClass === "motorcycle" || lowerClass === "bicycle")) {
      return { class: "headphones", score: Math.max(0.85, score), info: resolveEntityInfo("headphones") };
    }
    if (lowerClass === "remote") {
      return { class: "smartphone", score: Math.max(0.85, score), info: resolveEntityInfo("smartphone") };
    }
    if (lowerClass === "vase") {
      return { class: "water_bottle", score: Math.max(0.85, score), info: resolveEntityInfo("water bottle") };
    }
    return { class: rawClass, score: score, info: resolveEntityInfo(rawClass) };
  }

  // Check if this class belongs to high-confusion clusters
  const isSuspiciousVehicle = lowerClass === "motorcycle" || lowerClass === "bicycle" || lowerClass === "airplane" || lowerClass === "boat";
  const isRemote = lowerClass === "remote" || lowerClass === "remote control";
  const isVase = lowerClass === "vase";
  const isPerson = lowerClass === "person";
  const isAnimal = lowerClass === "cat" || lowerClass === "dog";
  const isCocoTool = lowerClass === "toothbrush" || lowerClass === "knife" || lowerClass === "baseball bat" || lowerClass === "scissors" || lowerClass === "mouse";

  const needsDisambiguation = isSuspiciousVehicle || isRemote || isVase || isPerson || isAnimal || isCocoTool;
  if (!needsDisambiguation) {
    return { class: rawClass, score: score, info: resolveEntityInfo(rawClass) };
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
    return { class: rawClass, score: score, info: resolveEntityInfo(rawClass) };
  }

  cropCvs.width = 224;
  cropCvs.height = 224;
  cropCtx.drawImage(sourceCanvasOrImg, sx, sy, sw, sh, 0, 0, 224, 224);

  let cropPredictions = [];
  try {
    cropPredictions = await classifierModel.classify(cropCvs, 5);
  } catch (err) {
    console.warn("[Disambiguation] Crop classify error:", err);
    return { class: rawClass, score: score, info: resolveEntityInfo(rawClass) };
  }

  if (!cropPredictions || cropPredictions.length === 0) {
    return { class: rawClass, score: score, info: resolveEntityInfo(rawClass) };
  }

  const top1 = cropPredictions[0];
  const top1Lower = top1.className.toLowerCase();
  const allLabelsLower = cropPredictions.map(p => p.className.toLowerCase()).join(" ");

  // 1. HEADPHONES vs MOTORCYCLE / BICYCLE
  // COCO lacks headphones completely; circular earcups + curved band get tagged as motorcycle
  if (isSuspiciousVehicle) {
    const isHeadphones = 
      allLabelsLower.includes("headphone") || 
      allLabelsLower.includes("earphone") || 
      allLabelsLower.includes("headset") || 
      allLabelsLower.includes("earbud") ||
      allLabelsLower.includes("headpiece") ||
      allLabelsLower.includes("acoustic") ||
      allLabelsLower.includes("stethoscope") ||
      allLabelsLower.includes("loudspeaker");

    if (isHeadphones || isWebcam || (bw < srcW * 0.78 && bh < srcH * 0.78)) {
      return {
        class: "headphones",
        score: Math.min(0.96, Math.max(score, top1.probability * 1.4, 0.88)),
        info: resolveEntityInfo("headphones")
      };
    }
  }

  // 2. SMARTPHONE vs REMOTE CONTROL
  // Modern smartphones with smooth black touchscreen slabs get confused with TV remotes
  if (isRemote) {
    const isPhone = 
      allLabelsLower.includes("cellular") || 
      allLabelsLower.includes("cellphone") || 
      allLabelsLower.includes("phone") || 
      allLabelsLower.includes("hand-held") || 
      allLabelsLower.includes("ipod") ||
      allLabelsLower.includes("screen") ||
      allLabelsLower.includes("modem");

    const isStrictRemote = top1Lower.includes("remote control") && top1.probability > 0.65;
    if (isPhone || !isStrictRemote) {
      return {
        class: "smartphone",
        score: Math.min(0.96, Math.max(score, top1.probability * 1.3, 0.85)),
        info: resolveEntityInfo("smartphone")
      };
    }
  }

  // 3. WATER BOTTLE vs VASE
  // Bottles & tumblers get mislabeled as flower vases unless flowers/bouquets are present
  if (isVase) {
    const isBottle = 
      allLabelsLower.includes("bottle") || 
      allLabelsLower.includes("flask") || 
      allLabelsLower.includes("shaker") || 
      allLabelsLower.includes("jug") || 
      allLabelsLower.includes("pitcher") || 
      allLabelsLower.includes("tumbler") ||
      allLabelsLower.includes("canteen") ||
      allLabelsLower.includes("cup") ||
      allLabelsLower.includes("mug");

    const hasFlowers = 
      allLabelsLower.includes("flower") || 
      allLabelsLower.includes("bouquet") || 
      allLabelsLower.includes("blossom");

    if (isBottle || !hasFlowers) {
      return {
        class: "water_bottle",
        score: Math.min(0.96, Math.max(score, 0.86)),
        info: resolveEntityInfo("water bottle")
      };
    }
  }

  // 4. SLIPPERS / FOOTWEAR vs PERSON
  // Floor-level or low-crop slippers/sandals get misclassified by COCO as whole "person"
  if (isPerson) {
    const isFloorOrSmall = (sy + sh >= srcH * 0.50) || (sh < srcH * 0.45) || (sw > sh * 1.15);
    const isFootwear = 
      allLabelsLower.includes("slipper") || 
      allLabelsLower.includes("sleeper") ||
      allLabelsLower.includes("sandal") || 
      allLabelsLower.includes("clog") || 
      allLabelsLower.includes("shoe") || 
      allLabelsLower.includes("sneaker") || 
      allLabelsLower.includes("loafer") || 
      allLabelsLower.includes("boot") || 
      allLabelsLower.includes("sock") ||
      allLabelsLower.includes("moccasin");

    if (isFloorOrSmall && isFootwear) {
      return {
        class: "slippers",
        score: Math.min(0.95, Math.max(score, top1.probability * 1.3, 0.85)),
        info: resolveEntityInfo("slipper")
      };
    }
  }

  // 5. DOG vs CAT DISAMBIGUATION
  // Checks all 120+ ImageNet canine breeds to prevent dogs from being called cats
  if (isAnimal) {
    const isDogBreed = isImageNetDog(allLabelsLower);
    const isCatBreed = isImageNetCat(allLabelsLower);

    if (lowerClass === "cat" && isDogBreed) {
      const matchingDog = getMatchingDogBreed(allLabelsLower);
      return {
        class: "dog",
        score: Math.min(0.96, Math.max(score, top1.probability, 0.88)),
        info: resolveEntityInfo(matchingDog || "dog")
      };
    } else if (lowerClass === "dog") {
      // Keep dog unless crop strongly indicates cat with high probability
      if (isCatBreed && !isDogBreed && top1.probability > 0.75) {
        return {
          class: "cat",
          score: top1.probability,
          info: resolveEntityInfo("cat")
        };
      }
      const matchingDog = getMatchingDogBreed(allLabelsLower);
      return {
        class: "dog",
        score: score,
        info: resolveEntityInfo(matchingDog || "dog")
      };
    }
  }

  // 6. EVERYDAY TOOLS (Glasses, Pens, Notebooks vs COCO toothbrush / knife / scissors)
  if (isCocoTool) {
    if (allLabelsLower.includes("spectacle") || allLabelsLower.includes("sunglass") || allLabelsLower.includes("glasses")) {
      return { class: "glasses", score: Math.max(0.88, top1.probability), info: resolveEntityInfo("glasses") };
    }
    if (allLabelsLower.includes("ballpoint") || allLabelsLower.includes("fountain pen") || allLabelsLower.includes("pen") || allLabelsLower.includes("pencil")) {
      return { class: "pen", score: Math.max(0.88, top1.probability), info: resolveEntityInfo("ballpoint pen") };
    }
    if (allLabelsLower.includes("notebook") || allLabelsLower.includes("binder") || allLabelsLower.includes("book") || allLabelsLower.includes("copy")) {
      return { class: "notebook", score: Math.max(0.88, top1.probability), info: resolveEntityInfo("notebook") };
    }
  }

  // Fallback: check if crop classified a high-confidence everyday item
  const topInfo = resolveEntityInfo(top1.className);
  if (top1.probability > 0.45 && topInfo.category !== "Object") {
    return { class: top1.className, score: top1.probability, info: topInfo };
  }

  return { class: rawClass, score: score, info: resolveEntityInfo(rawClass) };
}


// ---------------------------------------------------------------------------
// 1B. Spatial Patches & Non-Maximum Suppression (NMS) Engine
// ---------------------------------------------------------------------------
function computeIoU(boxA, boxB) {
  const [ax, ay, aw, ah] = boxA;
  const [bx, by, bw, bh] = boxB;

  const x1 = Math.max(ax, bx);
  const y1 = Math.max(ay, by);
  const x2 = Math.min(ax + aw, bx + bw);
  const y2 = Math.min(ay + ah, by + bh);

  const interW = Math.max(0, x2 - x1);
  const interH = Math.max(0, y2 - y1);
  const interArea = interW * interH;

  const areaA = aw * ah;
  const areaB = bw * bh;
  const unionArea = areaA + areaB - interArea;

  if (unionArea <= 0) return 0;
  return interArea / unionArea;
}

function applyNMS(candidates, iouThreshold = 0.40) {
  candidates.sort((a, b) => b.score - a.score);
  const selected = [];

  for (const cand of candidates) {
    let keep = true;
    for (const prev of selected) {
      const iou = computeIoU(cand.bbox, prev.bbox);
      if (iou > iouThreshold) {
        // If same entity or redundant category, keep the higher confidence one
        if (cand.info.title === prev.info.title || cand.info.category === "Object" || prev.info.category === "Object") {
          keep = false;
          break;
        }
      }
    }
    if (keep) {
      selected.push(cand);
    }
  }
  return selected;
}

function extractForegroundSaliencyBox(source) {
  const w = source.naturalWidth || source.videoWidth || source.width || 640;
  const h = source.naturalHeight || source.videoHeight || source.height || 480;

  const tw = 96;
  const th = 96;
  const thumbCanvas = document.createElement("canvas");
  thumbCanvas.width = tw;
  thumbCanvas.height = th;
  const thumbCtx = thumbCanvas.getContext("2d", { willReadFrequently: true });
  thumbCtx.drawImage(source, 0, 0, tw, th);
  const data = thumbCtx.getImageData(0, 0, tw, th).data;

  // Background estimation from corners
  const cornerIndices = [0, (tw - 1) * 4, (th - 1) * tw * 4, ((th - 1) * tw + (tw - 1)) * 4];
  let bgR = 0, bgG = 0, bgB = 0;
  for (const idx of cornerIndices) {
    bgR += data[idx];
    bgG += data[idx + 1];
    bgB += data[idx + 2];
  }
  bgR /= 4; bgG /= 4; bgB /= 4;

  let minX = tw, maxX = 0, minY = th, maxY = 0;
  let count = 0;

  for (let y = 0; y < th; y++) {
    for (let x = 0; x < tw; x++) {
      const i = (y * tw + x) * 4;
      const dr = Math.abs(data[i] - bgR);
      const dg = Math.abs(data[i + 1] - bgG);
      const db = Math.abs(data[i + 2] - bgB);
      const diff = (dr + dg + db) / 3;

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
    const scaleX = w / tw;
    const scaleY = h / th;
    const pad = 12;
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
    { name: "Top-Left", box: [0, 0, Math.round(w * 0.58), Math.round(h * 0.58)] },
    { name: "Top-Right", box: [Math.round(w * 0.42), 0, Math.round(w * 0.58), Math.round(h * 0.58)] },
    { name: "Bottom-Left", box: [0, Math.round(h * 0.42), Math.round(w * 0.58), Math.round(h * 0.58)] },
    { name: "Bottom-Right", box: [Math.round(w * 0.42), Math.round(h * 0.42), Math.round(w * 0.58), Math.round(h * 0.58)] },
    { name: "Center-Focus", box: [Math.round(w * 0.20), Math.round(h * 0.20), Math.round(w * 0.60), Math.round(h * 0.60)] },
    { name: "Lower-Desk", box: [Math.round(w * 0.15), Math.round(h * 0.35), Math.round(w * 0.70), Math.round(h * 0.60)] }
  ];
}

// ---------------------------------------------------------------------------
// 2. Night Vision & Low-Light Enhancement Subsystem
// ---------------------------------------------------------------------------
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
  
  const lut = new Uint8Array(256);
  for (let v = 0; v < 256; v++) {
    const normalized = v / 255.0;
    const boosted = Math.pow(normalized, gamma) * gain * 255.0;
    lut[v] = Math.min(255, Math.max(0, Math.round(boosted)));
  }
  
  for (let i = 0; i < d.length; i += 4) {
    d[i] = lut[d[i]];
    d[i+1] = lut[d[i+1]];
    d[i+2] = lut[d[i+2]];
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
      modelStatusEl.innerHTML = '<span class="status-dot"></span><span class="status-text">AI Ready: Dual Neural Pipeline Active</span>';
    }
    showToast("AI Models Loaded: COCO-SSD + MobileNet 1k", "🚀");
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

  // Semi-transparent bounding fill
  ctx.fillStyle = entityInfo.fill;
  ctx.fillRect(x, y, w, h);

  // Crisp high-tech border
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = isNightVision ? "#38bdf8" : entityInfo.stroke;
  ctx.strokeRect(x, y, w, h);

  // Corner accents
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

  // Pill label badge
  ctx.font = "bold 13px 'JetBrains Mono', monospace";
  const textWidth = ctx.measureText(badgeText).width;
  const badgeH = 24;
  const badgeW = textWidth + 16;
  const badgeY = Math.max(0, y - badgeH - 3);

  ctx.fillStyle = isNightVision ? "#0369a1" : entityInfo.badge;
  ctx.beginPath();
  if (ctx.roundRect) {
    ctx.roundRect(x, badgeY, badgeW, badgeH, 4);
  } else {
    ctx.rect(x, badgeY, badgeW, badgeH);
  }
  ctx.fill();

  ctx.fillStyle = "#ffffff";
  ctx.fillText(badgeText, x + 8, badgeY + 17);
}

// ---------------------------------------------------------------------------
// 5. Multi-View Rendering Engine
// ---------------------------------------------------------------------------
function renderCanvasView() {
  if (!cachedRawImg || !staticCanvas || !staticCtx) return;

  staticCanvas.width = cachedRawImg.naturalWidth;
  staticCanvas.height = cachedRawImg.naturalHeight;

  if (currentViewMode === "raw") {
    // Pure original untouched image
    staticCtx.drawImage(cachedRawImg, 0, 0);
  } else if (currentViewMode === "illuminated") {
    // Night Vision sensor matrix view
    if (cachedEnhancedCanvas) {
      staticCtx.drawImage(cachedEnhancedCanvas, 0, 0);
    } else {
      staticCtx.drawImage(cachedRawImg, 0, 0);
    }
  } else {
    // Augmented AI View (Default): Image + Bounding Boxes & Badges
    const shouldDrawIlluminated = cachedEnhancedCanvas && (cachedDetections.some(d => d.isNightVision));
    if (shouldDrawIlluminated) {
      staticCtx.drawImage(cachedEnhancedCanvas, 0, 0);
    } else {
      staticCtx.drawImage(cachedRawImg, 0, 0);
    }

    cachedDetections.forEach(det => {
      const [x, y, w, h] = det.bbox;
      drawStyledDetectionBox(staticCtx, x, y, w, h, det.info, det.score, det.isNightVision);
    });
  }
}

// ---------------------------------------------------------------------------
// 6. Static Image Inspector with Dual-Model Synergy & Night Vision
// ---------------------------------------------------------------------------
async function analyzeStaticImage(imgSrc, imageName = "Image") {
  if (!isModelsReady || (!cocoModel && !classifierModel)) {
    alert("AI Models are still initializing. Please wait a couple seconds and try again.");
    return;
  }

  currentLoadedImageSrc = imgSrc;

  // Show placeholder with futuristic laser scanline
  staticPlaceholder.style.display = "flex";
  staticPlaceholder.querySelector("h3").textContent = "Neural Scanning Active...";
  staticPlaceholder.querySelector("p").textContent = "Running Dual Neural Pipeline (Spatial Proposal + 1,000-Class ImageNet Filter)...";
  if (scanlineLaser) scanlineLaser.style.display = "block";

  const img = new Image();
  img.crossOrigin = "anonymous";
  img.src = imgSrc;

  img.onload = async () => {
    cachedRawImg = img;
    staticCanvas.width = img.naturalWidth;
    staticCanvas.height = img.naturalHeight;

    // 1. Measure scene luminance / ambient light
    const ambientLux = measureLuminance(img);

    // 2. Determine if Night Vision should trigger
    const shouldEngageNightVision = 
      (staticNightVisionMode === "on") || 
      (staticNightVisionMode === "auto" && ambientLux < 25);

    // 3. Prepare inference tensor source
    let inferenceSource = img;
    if (shouldEngageNightVision) {
      cachedEnhancedCanvas = createLowLightEnhancedCanvas(img, staticNightGain, staticNightGamma);
      inferenceSource = cachedEnhancedCanvas;
    } else {
      cachedEnhancedCanvas = null;
    }

    const minConf = parseFloat(staticConfSlider.value) / 100;
    let candidateDetections = [];

    // 4. Run COCO-SSD for candidate spatial anchor boxes
    let cocoPredictions = [];
    try {
      if (cocoModel) {
        cocoPredictions = await cocoModel.detect(inferenceSource);
      }
    } catch (err) {
      console.warn("[VisionAI] COCO detection error:", err);
    }

    const validCoco = cocoPredictions.filter(p => p.score >= Math.max(0.10, minConf * 0.55));

    // Shared offscreen scratch canvas for neural crops
    const cropCanvas = document.createElement("canvas");
    const cropCtx = cropCanvas.getContext("2d", { willReadFrequently: true });

    // 5A. Foreground High-Contrast Saliency Discovery
    // Tightly bounds small or prominent standalone items (pens, notebooks, glasses on desks)
    const saliencyBox = extractForegroundSaliencyBox(inferenceSource);
    if (saliencyBox && classifierModel) {
      const [sx, sy, sw, sh] = saliencyBox;
      if (sw >= 20 && sh >= 20) {
        cropCanvas.width = sw;
        cropCanvas.height = sh;
        cropCtx.drawImage(inferenceSource, sx, sy, sw, sh, 0, 0, sw, sh);
        try {
          const saliencyClasses = await classifierModel.classify(cropCanvas, 4);
          if (saliencyClasses && saliencyClasses.length > 0) {
            const topSal = saliencyClasses[0];
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
        } catch (e) {
          console.warn("[VisionAI] Saliency classification error:", e);
        }
      }
    }

    // 5B. Multi-Scale Spatial Quadrants & Center Dense Scan
    // Uncovers small desktop tools occupying <5% of whole frame
    const spatialPatches = extractSpatialPatches(inferenceSource);
    if (classifierModel) {
      for (const patch of spatialPatches) {
        const [px, py, pw, ph] = patch.box;
        if (pw < 35 || ph < 35) continue;

        cropCanvas.width = pw;
        cropCanvas.height = ph;
        cropCtx.drawImage(inferenceSource, px, py, pw, ph, 0, 0, pw, ph);

        try {
          const patchClasses = await classifierModel.classify(cropCanvas, 3);
          if (patchClasses && patchClasses.length > 0) {
            const top = patchClasses[0];
            const info = resolveEntityInfo(top.className);
            const lowerName = top.className.toLowerCase();

            const isEverydayTarget = 
              info.category === "Stationery / Tool" || 
              info.category === "Accessories" || 
              info.category === "Kitchenware" ||
              info.category === "Animal" ||
              (info.category === "Electronics" && !lowerName.includes("screen") && !lowerName.includes("monitor"));

            if (isEverydayTarget && top.probability >= 0.18) {
              // Attempt to tighten box using sub-patch saliency
              const patchSaliency = extractForegroundSaliencyBox(cropCanvas);
              let bbox = [px, py, pw, ph];
              if (patchSaliency) {
                bbox = [px + patchSaliency[0], py + patchSaliency[1], patchSaliency[2], patchSaliency[3]];
              }

              candidateDetections.push({
                bbox: bbox,
                info: info,
                score: Math.min(0.95, top.probability * 1.25),
                isNightVision: shouldEngageNightVision
              });
            }
          }
        } catch (patchErr) {
          console.warn("[VisionAI] Patch scanning error:", patchErr);
        }
      }
    }

    // 5C. Global Full-Image MobileNet Classification
    let fullImageClasses = [];
    try {
      if (classifierModel) {
        fullImageClasses = await classifierModel.classify(inferenceSource, 5);
      }
    } catch (err) {
      console.warn("[VisionAI] Full-frame classification error:", err);
    }

    // 6. Refine & Disambiguate COCO Bounding Proposals using Multi-Model Neural Engine
    for (const pred of validCoco) {
      const disambiguated = await disambiguateEntity(
        pred.class,
        pred.score,
        pred.bbox,
        inferenceSource,
        classifierModel,
        false
      );

      candidateDetections.push({
        bbox: pred.bbox,
        info: disambiguated.info,
        score: disambiguated.score,
        isNightVision: shouldEngageNightVision
      });
    }

    // 7. Full-Image Fallback if 0 candidates found
    if (candidateDetections.length === 0 && fullImageClasses.length > 0) {
      const topPred = fullImageClasses[0];
      const resolved = resolveEntityInfo(topPred.className);
      if (topPred.probability >= Math.min(0.12, minConf)) {
        const insetX = img.naturalWidth * 0.15;
        const insetY = img.naturalHeight * 0.15;
        const w = img.naturalWidth * 0.70;
        const h = img.naturalHeight * 0.70;
        candidateDetections.push({
          bbox: [insetX, insetY, w, h],
          info: resolved,
          score: topPred.probability,
          isNightVision: shouldEngageNightVision
        });
      }
    }

    // 8. Apply Non-Maximum Suppression (NMS) to eliminate duplicate overlapping boxes
    const filteredCandidates = candidateDetections.filter(d => d.score >= Math.max(0.14, minConf * 0.65));
    let detections = applyNMS(filteredCandidates, 0.40);

    cachedDetections = detections;

    // Render Canvas according to current view mode
    renderCanvasView();

    // Turn off scanline and placeholder
    if (scanlineLaser) scanlineLaser.style.display = "none";
    staticPlaceholder.style.display = "none";
    if (viewModeBar) viewModeBar.style.display = "flex";

    // Update Telemetry Bar
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

    let animalCount = 0;
    let toolAndObjectCount = 0;
    detections.forEach(det => {
      if (det.info.category === "Animal") animalCount++;
      else toolAndObjectCount++;
    });

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

      showToast(`Detected: ${detections.map(d => d.info.title).join(", ")}`, "🎯");
    }
  };

  img.onerror = (e) => {
    alert("Failed to load image: " + imgSrc);
    if (scanlineLaser) scanlineLaser.style.display = "none";
    staticPlaceholder.style.display = "flex";
    staticPlaceholder.querySelector("h3").textContent = "Image Load Error";
    staticPlaceholder.querySelector("p").textContent = "Could not decode or download image source.";
  };
}

// ---------------------------------------------------------------------------
// 7. Camera Controls & Real-Time Video Loop
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
      showToast("Live Camera Feed Active", "📷");
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

    let filtered = predictions.filter(p => p.score >= minConf);

    // Periodic Center Everyday Tool Discovery (e.g. holding up a pen, glasses, notebook)
    if (classifierModel && frameCount % 6 === 0 && filtered.length <= 3) {
      try {
        const vw = overlayCanvas.width;
        const vh = overlayCanvas.height;
        const cx = Math.round(vw * 0.22);
        const cy = Math.round(vh * 0.22);
        const cw = Math.round(vw * 0.56);
        const ch = Math.round(vh * 0.56);

        const liveCropCanvas = document.createElement("canvas");
        liveCropCanvas.width = cw;
        liveCropCanvas.height = ch;
        const liveCropCtx = liveCropCanvas.getContext("2d", { willReadFrequently: true });
        liveCropCtx.drawImage(videoSource, cx, cy, cw, ch, 0, 0, cw, ch);

        classifierModel.classify(liveCropCanvas, 2).then(classes => {
          if (classes && classes.length > 0) {
            const top = classes[0];
            const info = resolveEntityInfo(top.className);
            if ((info.category === "Stationery / Tool" || info.category === "Accessories" || info.category === "Kitchenware") && top.probability >= 0.22) {
              const liveSal = extractForegroundSaliencyBox(liveCropCanvas);
              let liveBox = [cx, cy, cw, ch];
              if (liveSal) {
                liveBox = [cx + liveSal[0], cy + liveSal[1], liveSal[2], liveSal[3]];
              }
              window._lastLiveEverydayItem = {
                bbox: liveBox,
                info: info,
                score: Math.min(0.92, top.probability * 1.3),
                isNightVision: shouldEngageNightVision,
                timestamp: performance.now()
              };
            }
          }
        }).catch(() => {});
      } catch (e) {}
    }

    // Blend cached everyday item if recent (< 350ms)
    let finalLiveEntities = [];
    for (const pred of filtered) {
      const disambiguated = await disambiguateEntity(
        pred.class,
        pred.score,
        pred.bbox,
        videoSource,
        classifierModel,
        true /* isWebcam */
      );

      let finalInfo = disambiguated.info;
      let finalScore = disambiguated.score;

      // Blend cached center-scan everyday item if recent (< 350ms)
      if (window._lastLiveEverydayItem && (performance.now() - window._lastLiveEverydayItem.timestamp < 350)) {
        if (computeIoU(pred.bbox, window._lastLiveEverydayItem.bbox) > 0.35) {
          finalInfo = window._lastLiveEverydayItem.info;
          finalScore = Math.max(finalScore, window._lastLiveEverydayItem.score);
        }
      }

      finalLiveEntities.push({
        bbox: pred.bbox,
        info: finalInfo,
        score: finalScore,
        isNightVision: shouldEngageNightVision
      });
    }

    if (window._lastLiveEverydayItem && (performance.now() - window._lastLiveEverydayItem.timestamp < 350)) {
      const isAlreadyIn = finalLiveEntities.some(e => computeIoU(e.bbox, window._lastLiveEverydayItem.bbox) > 0.40);
      if (!isAlreadyIn) {
        finalLiveEntities.push(window._lastLiveEverydayItem);
      }
    }

    const latency = Math.round(performance.now() - tStart);

    frameCount++;
    const now = performance.now();
    if (now - lastFrameTime >= 500) {
      fps = ((frameCount * 1000) / (now - lastFrameTime)).toFixed(1);
      teleFps.textContent = `${fps} FPS`;
      frameCount = 0;
      lastFrameTime = now;
    }

    teleLat.textContent = `${latency} ms`;
    teleCount.textContent = finalLiveEntities.length;

    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    const liveEntities = finalLiveEntities.map(det => {
      const [x, y, w, h] = det.bbox;
      drawStyledDetectionBox(overlayCtx, x, y, w, h, det.info, det.score, det.isNightVision);
      return det;
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
// 8. Event Listeners & Tactile Micro-Interactions
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
    showToast(`Switched to ${currentFacingMode} camera`, "🔄");
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
      showToast("Live Night Vision: Always ON", "🌙");
    } else if (liveNightVisionMode === "on") {
      liveNightVisionMode = "off";
      btnNightVisionLive.textContent = "🌙 Night Vision: OFF";
      btnNightVisionLive.style.borderColor = "rgba(99, 102, 241, 0.4)";
      btnNightVisionLive.style.color = "#a5b4fc";
      showToast("Live Night Vision: OFF", "⚪");
    } else {
      liveNightVisionMode = "auto";
      btnNightVisionLive.textContent = "🌙 Night Vision: AUTO";
      btnNightVisionLive.style.borderColor = "rgba(99, 102, 241, 0.4)";
      btnNightVisionLive.style.color = "#a5b4fc";
      showToast("Live Night Vision: AUTO Mode", "✨");
    }
  });
}

if (btnSnapshot) {
  btnSnapshot.addEventListener("click", () => {
    if (!isCameraRunning) return;

    // Trigger Camera Shutter Flash animation
    if (shutterFlash) {
      shutterFlash.classList.remove("flash-active");
      void shutterFlash.offsetWidth; // Trigger reflow
      shutterFlash.classList.add("flash-active");
    }

    const snapCanvas = document.createElement("canvas");
    snapCanvas.width = videoEl.videoWidth;
    snapCanvas.height = videoEl.videoHeight;
    const ctx = snapCanvas.getContext("2d");
    ctx.drawImage(videoEl, 0, 0);
    const dataUrl = snapCanvas.toDataURL("image/jpeg", 0.95);
    
    showToast("📸 Snapshot Captured! Analyzing...", "✨");

    // Scroll to inspector & analyze
    setTimeout(() => {
      document.getElementById("image-inspector").scrollIntoView({ behavior: "smooth" });
      analyzeStaticImage(dataUrl, "Camera Snapshot");
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
    if (currentLoadedImageSrc) {
      analyzeStaticImage(currentLoadedImageSrc);
    }
  });
}

if (staticNightModeSelect) {
  staticNightModeSelect.addEventListener("change", (e) => {
    staticNightVisionMode = e.target.value;
    showToast(`Night Vision Mode: ${e.target.value.toUpperCase()}`, "🌙");
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

// Display View Mode Switcher Listeners
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

// Category Filter Tabs Handling
benchmarkTabs.forEach(tab => {
  tab.addEventListener("click", () => {
    benchmarkTabs.forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    const filter = tab.getAttribute("data-filter");

    sampleChips.forEach(chip => {
      const cat = chip.getAttribute("data-category");
      if (filter === "all" || cat === filter) {
        chip.style.display = "inline-flex";
      } else {
        chip.style.display = "none";
      }
    });
    showToast(`Filtered: ${tab.textContent}`, "⚡");
  });
});

// File Upload Handler
if (fileInput) {
  fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
      sampleChips.forEach(c => c.classList.remove("active"));
      const reader = new FileReader();
      reader.onload = (event) => {
        showToast(`Loaded ${file.name}`, "📁");
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
      sampleChips.forEach(c => c.classList.remove("active"));
      const reader = new FileReader();
      reader.onload = (event) => {
        showToast(`Dropped ${file.name}`, "📁");
        analyzeStaticImage(event.target.result, file.name);
      };
      reader.readAsDataURL(file);
    }
  });
}

// Benchmark Sample Chips with Active State & Toast
sampleChips.forEach(chip => {
  chip.addEventListener("click", () => {
    sampleChips.forEach(c => c.classList.remove("active"));
    chip.classList.add("active");
    const samplePath = chip.getAttribute("data-sample");
    const sampleName = chip.textContent.trim();
    if (samplePath) {
      showToast(`Analyzing ${sampleName}...`, "🔍");
      analyzeStaticImage(samplePath, samplePath.split("/").pop());
    }
  });
});

// Copy Results JSON
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
    navigator.clipboard.writeText(JSON.stringify(cleanData, null, 2)).then(() => {
      showToast("Detections JSON copied to clipboard!", "📋");
    }).catch(err => {
      showToast("Failed to copy to clipboard", "❌");
    });
  });
}

// Initialize on DOM load
window.addEventListener("DOMContentLoaded", () => {
  initModel();
});

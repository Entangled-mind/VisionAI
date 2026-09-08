"""VisionAI - Benchmark Dataset Loader & Multi-Identity Trainer

Loads the standard Olivetti Multi-Pose, Multi-Lighting Face Dataset (40 subjects, 10 photos each)
and enrolls selected benchmark identities into data/faces/ and the SFace embeddings registry.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from sklearn.datasets import fetch_olivetti_faces

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FACES_DIR
from src.face_recognition import FaceRecognizer


def import_benchmark_dataset(num_subjects: int = 8, samples_per_subject: int = 10) -> int:
    """Imports multi-pose face dataset and saves high-resolution samples into data/faces/."""
    print(f"\n[Dataset Loader] Fetching standard face dataset ({num_subjects} subjects, {samples_per_subject} photos each)...")
    dataset = fetch_olivetti_faces()

    images = dataset.images  # Shape: (400, 64, 64)
    targets = dataset.target # Labels: 0 to 39

    # Standard human subject names for intuitive display
    subject_names = [
        "Alex_Rivera", "Elena_Rostova", "Marcus_Vance", "Sophia_Chen",
        "David_Kim", "Amara_Okonkwo", "Lucas_Moretti", "Priya_Patel",
        "Tariq_Mansour", "Chloe_Dupont"
    ]

    FACES_DIR.mkdir(parents=True, exist_ok=True)
    saved_count = 0

    for s_idx in range(min(num_subjects, len(subject_names))):
        name = subject_names[s_idx]
        person_dir = FACES_DIR / name
        person_dir.mkdir(parents=True, exist_ok=True)

        # Find indices for this subject
        subject_indices = np.where(targets == s_idx)[0][:samples_per_subject]

        for p_idx, img_idx in enumerate(subject_indices):
            raw_img = (images[img_idx] * 255).astype(np.uint8)
            # Upscale 64x64 to 200x200 with bicubic interpolation for clean SFace alignment
            upscaled = cv2.resize(raw_img, (200, 200), interpolation=cv2.INTER_CUBIC)
            bgr_img = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR)

            out_path = person_dir / f"pose_{p_idx+1:02d}.jpg"
            cv2.imwrite(str(out_path), bgr_img)
            saved_count += 1

        print(f"  * Prepared dataset for '{name}': {len(subject_indices)} varied poses/lighting conditions.")

    print(f"\n[Dataset Ready] Saved {saved_count} benchmark photos across {num_subjects} identities in '{FACES_DIR}'.")

    # Now train and build the SFace embeddings database
    print("\n[Training] Generating 128-D SFace embedding clusters with data augmentation...")
    recognizer = FaceRecognizer()
    total_enrolled = recognizer.build_database_from_directory()

    print(f"\n[Training Complete] Model successfully trained on {len(recognizer.database)} identities with {sum(len(v) for v in recognizer.database.values())} total embedding representations!\n")
    return total_enrolled


if __name__ == "__main__":
    import_benchmark_dataset(num_subjects=6, samples_per_subject=8)

"""VisionAI - Milestone 1 Unit Tests

Validates:
1. Image loading functionality and error handling.
2. Metadata extraction and dimension calculations.
3. Grayscale conversion correctness (channel reduction).
4. Bilinear resizing dimensional precision.
5. Disk I/O saving and reloading.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import cv2

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TEST_IMAGES_DIR, PROCESSED_DIR
from src.image_basics import (
    load_image,
    get_image_info,
    convert_to_grayscale,
    resize_image,
    save_image,
)


class TestMilestone1Basics(unittest.TestCase):
    """Test suite for Computer Vision fundamentals in Milestone 1."""

    def setUp(self) -> None:
        """Create a synthetic test image in memory before each test."""
        self.height, self.width = 120, 160
        self.test_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Add color patches (BGR)
        self.test_img[0:60, 0:80] = [255, 0, 0]      # Blue
        self.test_img[60:120, 80:160] = [0, 0, 255]  # Red

    def test_load_valid_image(self) -> None:
        """Test loading an existing image from disk."""
        sample_path = TEST_IMAGES_DIR / "sample.jpg"
        self.assertTrue(sample_path.exists(), "Sample test image should exist.")
        img = load_image(sample_path)
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(len(img.shape), 3)

    def test_load_missing_image_raises_error(self) -> None:
        """Test that attempting to load a non-existent file raises FileNotFoundError."""
        non_existent = TEST_IMAGES_DIR / "does_not_exist_9999.jpg"
        with self.assertRaises(FileNotFoundError):
            load_image(non_existent)

    def test_image_info_extraction(self) -> None:
        """Verify image properties match actual NumPy attributes."""
        info = get_image_info(self.test_img, label="Synthetic Test")
        self.assertEqual(info["height"], self.height)
        self.assertEqual(info["width"], self.width)
        self.assertEqual(info["channels"], 3)
        self.assertEqual(info["dtype"], "uint8")
        self.assertEqual(info["min_pixel_value"], 0)
        self.assertEqual(info["max_pixel_value"], 255)

    def test_grayscale_conversion(self) -> None:
        """Verify grayscale conversion drops channels from 3 to 1."""
        gray = convert_to_grayscale(self.test_img)
        self.assertEqual(gray.shape, (self.height, self.width))
        self.assertEqual(len(gray.shape), 2)
        # Idempotency check: passing already grayscale image should return itself
        gray_again = convert_to_grayscale(gray)
        self.assertEqual(gray_again.shape, (self.height, self.width))

    def test_image_resizing(self) -> None:
        """Verify image resizing scales correctly to target (Width, Height)."""
        target_w, target_h = 80, 50
        resized = resize_image(self.test_img, target_width=target_w, target_height=target_h)
        # Note: NumPy shape is (Height, Width, Channels)
        self.assertEqual(resized.shape[0], target_h)
        self.assertEqual(resized.shape[1], target_w)
        self.assertEqual(resized.shape[2], 3)

    def test_image_saving(self) -> None:
        """Verify saving and re-loading an image from disk."""
        temp_out = PROCESSED_DIR / "unit_test_temp.jpg"
        try:
            saved = save_image(self.test_img, temp_out)
            self.assertTrue(saved)
            self.assertTrue(temp_out.exists())

            # Reload to verify integrity
            reloaded = cv2.imread(str(temp_out))
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.shape, self.test_img.shape)
        finally:
            if temp_out.exists():
                temp_out.unlink()  # Clean up


if __name__ == "__main__":
    unittest.main()

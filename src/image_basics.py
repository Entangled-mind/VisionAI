"""VisionAI - Computer Vision Basics: Image Processing Fundamentals

Demonstrates core OpenCV concepts:
1. Loading an image from disk.
2. Inspecting dimensions, data types, and pixel arrays numerically.
3. Displaying images in a GUI window.
4. Converting BGR images to Grayscale.
5. Resizing images (interpolation techniques).
6. Saving processed images back to disk.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np

# Add project root to sys.path so config import works smoothly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TEST_IMAGES_DIR, PROCESSED_DIR


def load_image(image_path: Path) -> np.ndarray:
    """Loads an image from disk into a NumPy ndarray.

    Args:
        image_path: Path to the image file.

    Returns:
        np.ndarray: Loaded image in BGR format.

    Raises:
        FileNotFoundError: If the file does not exist on disk.
        ValueError: If OpenCV fails to decode the image file (e.g., corrupted file).
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found at: {image_path}")

    # cv2.imread loads images in BGR format by default
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"OpenCV could not decode image at {image_path}. File may be corrupted or in an unsupported format."
        )

    return image


def get_image_info(image: np.ndarray, label: str = "Image") -> Dict[str, Any]:
    """Extracts numerical properties of an image array.

    Args:
        image: NumPy array representing the image.
        label: Descriptive label for printing.

    Returns:
        Dict with keys: label, shape, height, width, channels, dtype, min_val, max_val, memory_bytes.
    """
    # In NumPy, shape is (Height, Width) for 2D or (Height, Width, Channels) for 3D
    shape = image.shape
    height = shape[0]
    width = shape[1]
    channels = shape[2] if len(shape) == 3 else 1

    info = {
        "label": label,
        "shape": shape,
        "height": height,
        "width": width,
        "channels": channels,
        "dtype": str(image.dtype),
        "min_pixel_value": int(np.min(image)),
        "max_pixel_value": int(np.max(image)),
        "memory_bytes": image.nbytes,
    }
    return info


def print_image_info(info: Dict[str, Any]) -> None:
    """Prints formatted image metadata to stdout."""
    color_model = "BGR (3 channels)" if info["channels"] == 3 else "Grayscale (1 channel)"
    print(f"\n{'=' * 56}")
    print(f"  [Image Properties] {info['label']}")
    print(f"{'=' * 56}")
    print(f"  * NumPy Array Shape : {info['shape']}  (Height x Width x Channels)")
    print(f"  * Resolution        : {info['width']} px (W) x {info['height']} px (H)")
    print(f"  * Total Pixels      : {info['width'] * info['height']:,} pixels")
    print(f"  * Color Model       : {color_model}")
    print(f"  * Data Type (dtype) : {info['dtype']} (uint8 = 8-bit unsigned integer: 0 to 255)")
    print(f"  * Pixel Value Range : [{info['min_pixel_value']}, {info['max_pixel_value']}]")
    print(f"  * Memory in RAM     : {info['memory_bytes'] / 1024:.2f} KB ({info['memory_bytes']:,} bytes)")
    print(f"{'=' * 56}")


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Converts a 3-channel BGR image into a 1-channel Grayscale image.

    Formula used by OpenCV (standard luminance perception formula):
        Y = 0.299*R + 0.587*G + 0.114*B

    Args:
        image: Input BGR image array.

    Returns:
        np.ndarray: 2D array of grayscale intensities (shape: H x W).
    """
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        print("  [Notice] Image is already single-channel grayscale.")
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_image(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Resizes an image using bilinear interpolation.

    NOTE on coordinate ordering:
        OpenCV's cv2.resize expects dimensions as (Width, Height).
        NumPy's array indexing is [row, col] i.e. (Height, Width).

    Args:
        image: Input image array.
        target_width: New width in pixels.
        target_height: New height in pixels.

    Returns:
        np.ndarray: Resized image array.
    """
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def save_image(image: np.ndarray, output_path: Path) -> bool:
    """Saves an image to disk.

    Args:
        image: Image array to write.
        output_path: Target file path.

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), image)
    if success:
        print(f"  [Saved] File written successfully -> {output_path}")
    else:
        print(f"  [Error] Failed to save image -> {output_path}", file=sys.stderr)
    return success


def display_image(window_name: str, image: np.ndarray, delay_ms: int = 0) -> None:
    """Displays an image in an OpenCV GUI window.

    Args:
        window_name: Title of the display window.
        image: Image array to render.
        delay_ms: Milliseconds to wait for keypress (0 means wait indefinitely).
    """
    try:
        cv2.imshow(window_name, image)
        print(f"  [Display] Showing window: '{window_name}'")
        print("            -> Click on the window and press ANY KEY to continue...")
        cv2.waitKey(delay_ms)
        cv2.destroyAllWindows()
    except cv2.error as err:
        print(f"  [Display Warning] GUI window could not be opened (e.g. headless environment): {err}")


def inspect_numerical_pixels(image: np.ndarray, top_left_y: int = 100, top_left_x: int = 100, size: int = 3) -> None:
    """Demonstrates that an image is literally a NumPy matrix of numbers.

    Args:
        image: Image array to inspect.
        top_left_y: Row index to start slice.
        top_left_x: Column index to start slice.
        size: Size of square slice (size x size).
    """
    h, w = image.shape[:2]
    end_y = min(top_left_y + size, h)
    end_x = min(top_left_x + size, w)
    patch = image[top_left_y:end_y, top_left_x:end_x]

    print(f"\n  [Numerical Proof] Inspecting a {size}x{size} pixel patch at [Row {top_left_y}:{end_y}, Col {top_left_x}:{end_x}]:")
    print(f"  Matrix slice:\n{patch}")
    if len(patch.shape) == 3:
        print("  -> Each pixel is a vector of 3 numbers: [Blue, Green, Red] ranging from 0 to 255.")
    else:
        print("  -> Each pixel is a single scalar: Luminance/Intensity (0=Pure Black, 255=Pure White).")


def run_pipeline(image_path: Path, target_w: int = 300, target_h: int = 200, no_display: bool = False) -> None:
    """Executes the complete Milestone 1 image processing pipeline.

    1. Load image
    2. Print dimensions and numerical array inspection
    3. Display original image
    4. Convert to grayscale
    5. Resize image
    6. Save processed images
    """
    print("\n" + "=" * 60)
    print("  VisionAI - Milestone 1: Computer Vision Basics Pipeline")
    print("=" * 60)

    # 1. Load image
    print(f"\n[Step 1] Loading image from: {image_path}")
    original_img = load_image(image_path)
    print("  -> Image successfully loaded into memory as a NumPy ndarray.")

    # 2. Print image dimensions and inspect numerical matrix
    print("\n[Step 2] Analyzing image dimensions and numerical array...")
    original_info = get_image_info(original_img, label="Original Image")
    print_image_info(original_info)
    inspect_numerical_pixels(original_img, top_left_y=100, top_left_x=100, size=3)

    # 3. Display original image
    if not no_display:
        print("\n[Step 3] Displaying original image in GUI window...")
        display_image("VisionAI - Step 3: Original Image (BGR)", original_img)
    else:
        print("\n[Step 3] Display skipped (--no-display active).")

    # 4. Convert to grayscale
    print("\n[Step 4] Converting image to Grayscale...")
    gray_img = convert_to_grayscale(original_img)
    gray_info = get_image_info(gray_img, label="Grayscale Image")
    print_image_info(gray_info)
    inspect_numerical_pixels(gray_img, top_left_y=100, top_left_x=100, size=3)

    if not no_display:
        print("  -> Displaying grayscale image in GUI window...")
        display_image("VisionAI - Step 4: Grayscale Image", gray_img)

    # 5. Resize image
    print(f"\n[Step 5] Resizing image to {target_w} px (W) x {target_h} px (H)...")
    resized_img = resize_image(original_img, target_width=target_w, target_height=target_h)
    resized_info = get_image_info(resized_img, label=f"Resized Image ({target_w}x{target_h})")
    print_image_info(resized_info)

    if not no_display:
        print("  -> Displaying resized image in GUI window...")
        display_image("VisionAI - Step 5: Resized Image", resized_img)

    # 6. Save processed images
    print("\n[Step 6] Saving processed images to disk...")
    gray_path = PROCESSED_DIR / f"{image_path.stem}_grayscale.jpg"
    resized_path = PROCESSED_DIR / f"{image_path.stem}_resized.jpg"

    save_image(gray_img, gray_path)
    save_image(resized_img, resized_path)

    print("\n" + "=" * 60)
    print("  Milestone 1 Image Processing Finished Successfully!")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VisionAI - Milestone 1: Image Processing Fundamentals"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=str(TEST_IMAGES_DIR / "sample.jpg"),
        help="Path to input image file (defaults to data/test_images/sample.jpg)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without displaying GUI popup windows (ideal for headless testing)",
    )
    parser.add_argument(
        "--resize-w",
        type=int,
        default=300,
        help="Target resize width in pixels (default: 300)",
    )
    parser.add_argument(
        "--resize-h",
        type=int,
        default=200,
        help="Target resize height in pixels (default: 200)",
    )

    args = parser.parse_args()

    input_path = Path(args.image)
    try:
        run_pipeline(
            image_path=input_path,
            target_w=args.resize_w,
            target_h=args.resize_h,
            no_display=args.no_display,
        )
    except Exception as exc:
        print(f"\n[Execution Error] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

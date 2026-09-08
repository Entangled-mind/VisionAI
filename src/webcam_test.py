"""VisionAI - Computer Vision Basics: Webcam Live Stream Test

Demonstrates:
1. Connecting to a video capture device (webcam).
2. Streaming video frames continuously in a real-time loop.
3. Calculating and displaying real-time FPS (Frames Per Second).
4. Overlaying diagnostic text on live frames using cv2.putText.
5. Handling keyboard interrupts (clean exit on 'q' or ESC).
6. Graceful resource management (cap.release() and cv2.destroyAllWindows()).
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Add project root to sys.path so config can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DEFAULT_CAMERA_INDEX, DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT, DEFAULT_WINDOW_NAME


def initialize_camera(
    camera_index: int = DEFAULT_CAMERA_INDEX,
    frame_width: int = DEFAULT_FRAME_WIDTH,
    frame_height: int = DEFAULT_FRAME_HEIGHT,
) -> cv2.VideoCapture:
    """Initializes and configures the camera video capture device.

    Args:
        camera_index: Hardware index of the camera (0 is usually the built-in webcam).
        frame_width: Desired capture frame width.
        frame_height: Desired capture frame height.

    Returns:
        cv2.VideoCapture: Configured video capture object.

    Raises:
        RuntimeError: If the video capture device cannot be opened.
    """
    print(f"\n[Hardware] Attempting to open camera at index: {camera_index}...")

    # On Windows, cv2.CAP_DSHOW (DirectShow) provides fast initialization and eliminates lag
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_index)

    # If DirectShow fails or index doesn't respond, fall back to default backend
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera at index {camera_index}.\n"
            "Troubleshooting Tips:\n"
            "  1. Ensure your webcam is connected and not in use by another app (e.g. Teams, Zoom, Browser).\n"
            "  2. On Windows, check: Settings -> Privacy & Security -> Camera -> Allow apps to access your camera.\n"
            "  3. If you have multiple cameras (external/internal), try running with: --camera 1"
        )

    # Request resolution from the hardware/driver
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Hardware] Camera successfully initialized!")
    print(f"           Actual Capture Resolution: {actual_w} x {actual_h} pixels")

    return cap


def draw_hud(frame: np.ndarray, fps: float) -> np.ndarray:
    """Draws an informative Head-Up Display (HUD) overlay on the video frame.

    Args:
        frame: Live video frame (BGR).
        fps: Current measured frames-per-second.

    Returns:
        np.ndarray: Frame with diagnostic text overlay.
    """
    h, w = frame.shape[:2]

    # Semi-transparent top banner for readability
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 20), -1)
    # Blend overlay with original frame (alpha=0.6)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Title & Controls
    cv2.putText(
        frame,
        "VisionAI - Live Feed Test",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),  # Yellow-cyan in BGR
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Press 'q' or 'ESC' in this window to exit",
        (15, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    # Diagnostic badge (Resolution & FPS)
    info_text = f"{w}x{h} | {fps:.1f} FPS"
    cv2.putText(
        frame,
        info_text,
        (w - 180, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),  # Bright Green in BGR
        2,
        cv2.LINE_AA,
    )

    return frame


def run_webcam_stream(
    camera_index: int = DEFAULT_CAMERA_INDEX,
    frame_width: int = DEFAULT_FRAME_WIDTH,
    frame_height: int = DEFAULT_FRAME_HEIGHT,
    window_name: str = DEFAULT_WINDOW_NAME,
    max_frames: Optional[int] = None,
) -> None:
    """Runs the real-time webcam capture loop.

    Reads frames sequentially, measures FPS, renders the live feed,
    and terminates safely when 'q' or 'ESC' is pressed.

    Args:
        camera_index: Index of camera device.
        frame_width: Desired width in pixels.
        frame_height: Desired height in pixels.
        window_name: Title of the OpenCV display window.
        max_frames: Optional frame limit for testing or headless runs.
    """
    cap = None
    try:
        cap = initialize_camera(camera_index, frame_width, frame_height)

        print("\n" + "=" * 60)
        print("  VisionAI - Webcam Live Feed Active")
        print("=" * 60)
        print("  * Video stream running.")
        print("  * Focus on the video window and press 'q' or 'ESC' to exit cleanly.")
        print("=" * 60 + "\n")

        # Variables for real-time FPS calculation
        frame_count = 0
        prev_time = time.time()
        fps = 0.0

        while True:
            # Read single frame: ret is a boolean flag, frame is a NumPy ndarray
            ret, frame = cap.read()

            if not ret or frame is None:
                print("\n[Warning] Failed to grab frame from camera stream. Exiting loop.")
                break

            frame_count += 1

            # Calculate FPS smoothly over a moving window
            current_time = time.time()
            delta_time = current_time - prev_time
            if delta_time > 0.5:  # Update FPS readout every 0.5s for stability
                fps = frame_count / delta_time
                frame_count = 0
                prev_time = current_time

            # Draw status HUD onto the frame
            annotated_frame = draw_hud(frame, fps)

            # Display the frame in the interactive window
            cv2.imshow(window_name, annotated_frame)

            # Wait 1 millisecond for key event
            # 0xFF extracts lowest 8 bits for cross-platform ASCII matching
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 27 is ASCII for ESC
                print("\n[User Action] 'q' or 'ESC' pressed. Exiting cleanly...")
                break

            if max_frames and frame_count >= max_frames:
                print(f"\n[Test Limit] Reached max_frames limit ({max_frames}). Exiting...")
                break

    except KeyboardInterrupt:
        print("\n[User Action] KeyboardInterrupt (Ctrl+C) detected. Shutting down...")
    except Exception as exc:
        print(f"\n[Error] {exc}", file=sys.stderr)
    finally:
        # Crucial: Always release hardware and destroy GUI windows
        if cap is not None and cap.isOpened():
            cap.release()
            print("[Hardware] Camera device released successfully.")
        cv2.destroyAllWindows()
        print("[GUI] OpenCV display windows destroyed. Program terminated cleanly.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="VisionAI - Milestone 1: Webcam Live Stream Test")
    parser.add_argument(
        "--camera",
        type=int,
        default=DEFAULT_CAMERA_INDEX,
        help=f"Camera index to open (default: {DEFAULT_CAMERA_INDEX})",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_FRAME_WIDTH,
        help=f"Target frame width in pixels (default: {DEFAULT_FRAME_WIDTH})",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_FRAME_HEIGHT,
        help=f"Target frame height in pixels (default: {DEFAULT_FRAME_HEIGHT})",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to capture before auto-exiting (useful for automated testing)",
    )

    args = parser.parse_args()

    run_webcam_stream(
        camera_index=args.camera,
        frame_width=args.width,
        frame_height=args.height,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()

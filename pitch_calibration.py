import cv2
import numpy as np
from typing import List, Tuple, Dict
import json


class PitchCalibration:


    # Standard football pitch dimensions (meters)
    PITCH_LENGTH = 105.0  # meters
    PITCH_WIDTH = 68.0  # meters

    def __init__(self):
        self.homography_matrix = None
        self.pitch_corners = None
        self.image_corners = None

    def interactive_calibration(self, video_path: str, output_path: str = "calibration.json"):

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError("Could not read video")

        # Store clicked points
        points = []

        def click_event(event, x, y, flags, params):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append([x, y])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(frame, f"P{len(points)}", (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow("Calibration", frame)

        cv2.imshow("Calibration", frame)
        cv2.setMouseCallback("Calibration", click_event)

        print("Click the 4 corners of the pitch:")
        print("1. Top-left")
        print("2. Top-right")
        print("3. Bottom-right")
        print("4. Bottom-left")
        print("Press 'q' when done")

        while len(points) < 4:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

        if len(points) == 4:
            self.calibrate_from_corners(points)
            self.save_calibration(output_path)
            print(f"Calibration saved to {output_path}")
            return True
        else:
            print("Calibration failed - need 4 points")
            return False

    def calibrate_from_corners(self, image_corners: List[List[float]]):
        """
        Calculate homography matrix from pitch corners.

        Args:
            image_corners: List of [x, y] points in image (pixel coordinates)
        """
        self.image_corners = np.array(image_corners, dtype=np.float32)

        # Real-world pitch corners (in meters)
        self.pitch_corners = np.array([
            [0, 0],  # Top-left
            [self.PITCH_LENGTH, 0],  # Top-right
            [self.PITCH_LENGTH, self.PITCH_WIDTH],  # Bottom-right
            [0, self.PITCH_WIDTH]  # Bottom-left
        ], dtype=np.float32)

        # Calculate homography matrix
        self.homography_matrix, _ = cv2.findHomography(
            self.image_corners,
            self.pitch_corners
        )

    def pixel_to_meters(self, pixel_coords: np.ndarray) -> np.ndarray:
        """
        Convert pixel coordinates to real-world meters.

        Args:
            pixel_coords: Array of shape (N, 2) with [x, y] pixel coordinates

        Returns:
            Array of shape (N, 2) with [x, y] meter coordinates
        """
        if self.homography_matrix is None:
            raise ValueError("Calibration not done. Run calibrate_from_corners first.")

        # Ensure input is 2D array
        pixel_coords = np.array(pixel_coords)
        if pixel_coords.ndim == 1:
            pixel_coords = pixel_coords.reshape(1, -1)

        # Add homogeneous coordinate
        ones = np.ones((pixel_coords.shape[0], 1))
        pixel_coords_h = np.hstack([pixel_coords, ones])

        # Apply homography
        meter_coords_h = (self.homography_matrix @ pixel_coords_h.T).T

        # Convert back from homogeneous coordinates
        meter_coords = meter_coords_h[:, :2] / meter_coords_h[:, 2:]

        return meter_coords

    def meters_to_pixel(self, meter_coords: np.ndarray) -> np.ndarray:
        """Convert real-world meters back to pixel coordinates."""
        if self.homography_matrix is None:
            raise ValueError("Calibration not done.")

        inv_homography = np.linalg.inv(self.homography_matrix)

        meter_coords = np.array(meter_coords)
        if meter_coords.ndim == 1:
            meter_coords = meter_coords.reshape(1, -1)

        ones = np.ones((meter_coords.shape[0], 1))
        meter_coords_h = np.hstack([meter_coords, ones])

        pixel_coords_h = (inv_homography @ meter_coords_h.T).T
        pixel_coords = pixel_coords_h[:, :2] / pixel_coords_h[:, 2:]

        return pixel_coords

    def calculate_real_distance(self, pixel_pos1: np.ndarray, pixel_pos2: np.ndarray) -> float:
        """
        Calculate real-world distance in meters between two pixel positions.
        """
        meter_pos1 = self.pixel_to_meters(pixel_pos1)
        meter_pos2 = self.pixel_to_meters(pixel_pos2)

        distance = np.linalg.norm(meter_pos2 - meter_pos1)
        return float(distance)

    def calculate_real_speed(self, pixel_velocity: np.ndarray, fps: float) -> float:
        """
        Calculate real-world speed in m/s from pixel velocity.

        Args:
            pixel_velocity: Velocity in pixels per frame
            fps: Frames per second of video

        Returns:
            Speed in meters per second
        """
        # Convert to meters per frame
        meter_velocity = self.pixel_to_meters(pixel_velocity.reshape(1, -1))[0]

        # Convert to meters per second
        speed = np.linalg.norm(meter_velocity) * fps
        return float(speed)

    def save_calibration(self, filepath: str):
        """Save calibration data to JSON."""
        data = {
            'homography_matrix': self.homography_matrix.tolist() if self.homography_matrix is not None else None,
            'pitch_corners': self.pitch_corners.tolist() if self.pitch_corners is not None else None,
            'image_corners': self.image_corners.tolist() if self.image_corners is not None else None,
            'pitch_length': self.PITCH_LENGTH,
            'pitch_width': self.PITCH_WIDTH
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_calibration(self, filepath: str):
        """Load calibration data from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.homography_matrix = np.array(data['homography_matrix']) if data['homography_matrix'] else None
        self.pitch_corners = np.array(data['pitch_corners']) if data['pitch_corners'] else None
        self.image_corners = np.array(data['image_corners']) if data['image_corners'] else None
        self.PITCH_LENGTH = data.get('pitch_length', 105.0)
        self.PITCH_WIDTH = data.get('pitch_width', 68.0)

    def draw_pitch_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw pitch lines overlay on frame for visualization."""
        if self.homography_matrix is None:
            return frame

        overlay = frame.copy()

        # Define pitch lines in real coordinates (meters)
        lines = [
            # Outer boundary
            [(0, 0), (self.PITCH_LENGTH, 0)],
            [(self.PITCH_LENGTH, 0), (self.PITCH_LENGTH, self.PITCH_WIDTH)],
            [(self.PITCH_LENGTH, self.PITCH_WIDTH), (0, self.PITCH_WIDTH)],
            [(0, self.PITCH_WIDTH), (0, 0)],
            # Center line
            [(self.PITCH_LENGTH / 2, 0), (self.PITCH_LENGTH / 2, self.PITCH_WIDTH)],
            # Center circle (approximate)
        ]

        # Convert to pixel coordinates and draw
        for line in lines:
            pt1 = self.meters_to_pixel(np.array(line[0]))[0].astype(int)
            pt2 = self.meters_to_pixel(np.array(line[1]))[0].astype(int)
            cv2.line(overlay, tuple(pt1), tuple(pt2), (0, 255, 0), 2)

        # Blend with original frame
        result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        return result


class DataPreprocessor:
    """
    Preprocess video data before tracking to improve accuracy.
    """

    def __init__(self):
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2()

    def enhance_video(self, video_path: str, output_path: str,
                      brightness: float = 1.0,
                      contrast: float = 1.0,
                      denoise: bool = True) -> str:
        """
        Enhance video quality for better tracking.

        Args:
            video_path: Input video path
            output_path: Output video path
            brightness: Brightness multiplier (1.0 = no change)
            contrast: Contrast multiplier (1.0 = no change)
            denoise: Whether to apply denoising

        Returns:
            Path to enhanced video
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Adjust brightness and contrast
            frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=int((brightness - 1) * 255))

            # Denoise
            if denoise:
                frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

            out.write(frame)
            frame_count += 1

            if frame_count % 100 == 0:
                print(f"Enhanced {frame_count} frames...")

        cap.release()
        out.release()

        print(f"Enhanced video saved to: {output_path}")
        return output_path

    def extract_field_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract field/pitch mask to filter out crowd, ads, etc.
        Uses color-based segmentation to identify the green pitch.
        """
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define range for green color (grass)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])

        # Create mask
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    def stabilize_video(self, video_path: str, output_path: str) -> str:
        """
        Stabilize shaky camera footage.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Read first frame
        ret, prev_frame = cap.read()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        transforms = []

        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break

            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            # Detect feature points
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200,
                                               qualityLevel=0.01, minDistance=30)

            # Calculate optical flow
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

            # Filter valid points
            idx = np.where(status == 1)[0]
            prev_pts = prev_pts[idx]
            curr_pts = curr_pts[idx]

            # Estimate transform
            transform = cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]
            transforms.append(transform)

            prev_gray = curr_gray

        # Smooth transforms
        transforms = np.array(transforms)
        # Apply moving average
        smoothed = np.copy(transforms)
        for i in range(len(transforms)):
            start = max(0, i - 5)
            end = min(len(transforms), i + 6)
            smoothed[i] = transforms[start:end].mean(axis=0)

        # Apply smoothed transforms
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for transform in smoothed:
            ret, frame = cap.read()
            if not ret:
                break

            stabilized = cv2.warpAffine(frame, transform, (width, height))
            out.write(stabilized)

        cap.release()
        out.release()

        print(f"Stabilized video saved to: {output_path}")
        return output_path

    def split_by_half(self, video_path: str, output_dir: str) -> Tuple[str, str]:
        """
        Split video into first and second half for comparison.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        half_point = total_frames // 2

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        first_half_path = os.path.join(output_dir, "first_half.mp4")
        second_half_path = os.path.join(output_dir, "second_half.mp4")

        out1 = cv2.VideoWriter(first_half_path, fourcc, fps, (width, height))
        out2 = cv2.VideoWriter(second_half_path, fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count < half_point:
                out1.write(frame)
            else:
                out2.write(frame)

            frame_count += 1

        cap.release()
        out1.release()
        out2.release()

        print(f"Video split into:\n  {first_half_path}\n  {second_half_path}")
        return first_half_path, second_half_path


# Example usage
if __name__ == "__main__":
    # Calibrate pitch
    calibrator = PitchCalibration()
    calibrator.interactive_calibration("match.mp4", "calibration.json")

    # Convert pixel position to meters
    pixel_pos = np.array([640, 480])
    meter_pos = calibrator.pixel_to_meters(pixel_pos)
    print(f"Position in meters: {meter_pos}")

    # Preprocess video
    preprocessor = DataPreprocessor()
    enhanced = preprocessor.enhance_video("match.mp4", "enhanced.mp4",
                                          brightness=1.1, contrast=1.2)

    # Split into halves
    first, second = preprocessor.split_by_half("match.mp4", "split_videos")
import cv2
import numpy as np
import torch
from collections import defaultdict, deque
from scipy.spatial.distance import euclidean
from scipy.optimize import linear_sum_assignment
import json


class PlayerTracker:
    """
    Advanced player tracking system for football analysis.
    Tracks individual players across frames and analyzes their movements.
    """

    def __init__(self):
        # Load YOLO for player detection
        self.detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        # Tracking state
        self.tracks = {}  # player_id -> track history
        self.next_id = 0
        self.max_age = 30  # frames before track is deleted
        self.min_hits = 3  # minimum detections before track is confirmed

        # Track history for each player
        self.track_history = defaultdict(lambda: {
            'positions': [],
            'velocities': [],
            'accelerations': [],
            'distances': [],
            'timestamps': [],
            'active': True,
            'age': 0,
            'hits': 0
        })

    def detect_players(self, frame):
        """
        Detect all players in a frame.
        Returns list of bounding boxes [x1, y1, x2, y2, confidence]
        """
        results = self.detector(frame)
        detections = results.xyxy[0].cpu().numpy()

        # Filter for person class (0) with high confidence
        players = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if cls == 0 and conf > 0.5:  # person class with >50% confidence
                players.append([x1, y1, x2, y2, conf])

        return np.array(players) if players else np.empty((0, 5))

    def get_center(self, bbox):
        """Get center point of bounding box."""
        x1, y1, x2, y2 = bbox[:4]
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    def get_foot_position(self, bbox):
        """Get foot position (bottom center) of bounding box - more stable for tracking."""
        x1, y1, x2, y2 = bbox[:4]
        return np.array([(x1 + x2) / 2, y2])

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def associate_detections_to_tracks(self, detections, tracks):
        """
        Match current detections to existing tracks using Hungarian algorithm.
        Returns matched pairs, unmatched detections, and unmatched tracks.
        """
        if len(tracks) == 0:
            return [], list(range(len(detections))), []

        if len(detections) == 0:
            return [], [], list(tracks.keys())

        # Create cost matrix based on distance and IOU
        cost_matrix = np.zeros((len(detections), len(tracks)))
        track_ids = list(tracks.keys())

        for d_idx, det in enumerate(detections):
            det_pos = self.get_foot_position(det)

            for t_idx, track_id in enumerate(track_ids):
                track = tracks[track_id]
                if len(track['positions']) == 0:
                    cost_matrix[d_idx, t_idx] = 1e5
                    continue

                # Last known position
                last_pos = track['positions'][-1]

                # Distance cost
                distance = euclidean(det_pos, last_pos)

                # IOU cost (if we have bbox info)
                iou = self.calculate_iou(det, track.get('last_bbox', det))

                # Combined cost (lower is better)
                cost_matrix[d_idx, t_idx] = distance * (1 - iou * 0.5)

        # Hungarian algorithm for optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Filter out assignments with high cost
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(tracks.keys())

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 50:  # threshold for valid match
                matches.append((r, track_ids[c]))
                if r in unmatched_detections:
                    unmatched_detections.remove(r)
                if track_ids[c] in unmatched_tracks:
                    unmatched_tracks.remove(track_ids[c])

        return matches, unmatched_detections, unmatched_tracks

    def update_track(self, track_id, position, bbox, timestamp, fps):
        """Update track with new detection."""
        track = self.track_history[track_id]
        track['positions'].append(position)
        track['timestamps'].append(timestamp)
        track['last_bbox'] = bbox
        track['age'] = 0
        track['hits'] += 1

        # Calculate velocity if we have enough history
        if len(track['positions']) >= 2:
            dt = 1.0 / fps
            prev_pos = track['positions'][-2]
            velocity = (position - prev_pos) / dt
            track['velocities'].append(velocity)

            # Calculate acceleration
            if len(track['velocities']) >= 2:
                prev_vel = track['velocities'][-2]
                acceleration = (velocity - prev_vel) / dt
                track['accelerations'].append(acceleration)

            # Calculate distance traveled
            distance = euclidean(position, prev_pos)
            track['distances'].append(distance)

    def track_video(self, video_path, output_path=None):
        """
        Track all players throughout the video.

        Args:
            video_path: Path to input video
            output_path: Optional path to save annotated video

        Returns:
            Dictionary with all player tracks and movement data
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video writer if output requested
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0

        print(f"Tracking players in video...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect players in current frame
            detections = self.detect_players(frame)

            # Associate detections with existing tracks
            matches, unmatched_dets, unmatched_tracks = \
                self.associate_detections_to_tracks(detections, self.track_history)

            # Update matched tracks
            for det_idx, track_id in matches:
                detection = detections[det_idx]
                position = self.get_foot_position(detection)
                timestamp = frame_count / fps
                self.update_track(track_id, position, detection, timestamp, fps)

            # Create new tracks for unmatched detections
            for det_idx in unmatched_dets:
                detection = detections[det_idx]
                position = self.get_foot_position(detection)
                timestamp = frame_count / fps

                self.track_history[self.next_id]['positions'].append(position)
                self.track_history[self.next_id]['timestamps'].append(timestamp)
                self.track_history[self.next_id]['last_bbox'] = detection
                self.track_history[self.next_id]['hits'] = 1
                self.next_id += 1

            # Age unmatched tracks
            for track_id in unmatched_tracks:
                self.track_history[track_id]['age'] += 1
                if self.track_history[track_id]['age'] > self.max_age:
                    self.track_history[track_id]['active'] = False

            # Draw tracks on frame if output requested
            if writer:
                frame = self.draw_tracks(frame, detections, matches)
                writer.write(frame)

            frame_count += 1

            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")

        cap.release()
        if writer:
            writer.release()
            print(f"Annotated video saved to: {output_path}")

        print(f"Tracking complete! Found {len(self.track_history)} players")

        # Clean up tracks with too few detections
        confirmed_tracks = {
            tid: track for tid, track in self.track_history.items()
            if track['hits'] >= self.min_hits
        }

        return confirmed_tracks

    def draw_tracks(self, frame, detections, matches):
        """Draw bounding boxes and track IDs on frame."""
        # Draw all detections
        for detection in detections:
            x1, y1, x2, y2, conf = detection
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Draw track IDs for matched detections
        for det_idx, track_id in matches:
            detection = detections[det_idx]
            x1, y1, x2, y2, _ = detection

            # Draw track ID
            cv2.putText(frame, f"ID:{track_id}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Draw trajectory (last 30 positions)
            track = self.track_history[track_id]
            positions = track['positions'][-30:]
            for i in range(1, len(positions)):
                pt1 = tuple(positions[i - 1].astype(int))
                pt2 = tuple(positions[i].astype(int))
                cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

        return frame

    def analyze_player_movement(self, track_id):
        """
        Analyze individual player movement patterns.

        Returns comprehensive movement statistics for a player.
        """
        if track_id not in self.track_history:
            return None

        track = self.track_history[track_id]
        positions = np.array(track['positions'])

        if len(positions) < 2:
            return None

        # Calculate total distance
        total_distance = sum(track['distances']) if track['distances'] else 0

        # Calculate average speed
        velocities = np.array(track['velocities']) if track['velocities'] else np.array([])
        avg_speed = np.mean(np.linalg.norm(velocities, axis=1)) if len(velocities) > 0 else 0
        max_speed = np.max(np.linalg.norm(velocities, axis=1)) if len(velocities) > 0 else 0

        # Calculate area covered (convex hull)
        from scipy.spatial import ConvexHull
        if len(positions) >= 3:
            try:
                hull = ConvexHull(positions)
                area_covered = hull.volume  # volume in 2D is area
            except:
                area_covered = 0
        else:
            area_covered = 0

        # Sprint detection (high acceleration/velocity)
        sprints = []
        if len(velocities) > 0:
            speed_threshold = np.percentile(np.linalg.norm(velocities, axis=1), 90)
            for i, vel in enumerate(velocities):
                if np.linalg.norm(vel) > speed_threshold:
                    sprints.append({
                        'timestamp': track['timestamps'][i],
                        'speed': float(np.linalg.norm(vel)),
                        'position': positions[i].tolist()
                    })

        # Movement patterns (changes in direction)
        direction_changes = 0
        if len(velocities) > 1:
            for i in range(1, len(velocities)):
                angle_change = np.arccos(np.clip(
                    np.dot(velocities[i], velocities[i - 1]) /
                    (np.linalg.norm(velocities[i]) * np.linalg.norm(velocities[i - 1]) + 1e-6),
                    -1.0, 1.0
                ))
                if angle_change > np.pi / 4:  # 45 degrees
                    direction_changes += 1

        return {
            'player_id': track_id,
            'total_distance': float(total_distance),
            'avg_speed': float(avg_speed),
            'max_speed': float(max_speed),
            'area_covered': float(area_covered),
            'duration': track['timestamps'][-1] - track['timestamps'][0],
            'num_positions': len(positions),
            'sprints': sprints,
            'direction_changes': direction_changes,
            'positions': positions.tolist(),
            'timestamps': track['timestamps']
        }

    def analyze_team_movement(self):
        """
        Analyze overall team movement patterns.
        """
        all_analyses = []
        for track_id in self.track_history:
            analysis = self.analyze_player_movement(track_id)
            if analysis:
                all_analyses.append(analysis)

        if not all_analyses:
            return None

        # Aggregate statistics
        total_distances = [a['total_distance'] for a in all_analyses]
        avg_speeds = [a['avg_speed'] for a in all_analyses]

        return {
            'num_players_tracked': len(all_analyses),
            'total_team_distance': sum(total_distances),
            'avg_player_distance': np.mean(total_distances),
            'avg_team_speed': np.mean(avg_speeds),
            'max_team_speed': max([a['max_speed'] for a in all_analyses]),
            'player_analyses': all_analyses
        }

    def generate_heatmap(self, video_path, output_path, player_id=None):
        """
        Generate movement heatmap for all players or specific player.
        """
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        # Create heatmap
        heatmap = np.zeros(frame.shape[:2], dtype=np.float32)

        # Add positions to heatmap
        tracks_to_plot = [player_id] if player_id else self.track_history.keys()

        for tid in tracks_to_plot:
            if tid in self.track_history:
                positions = self.track_history[tid]['positions']
                for pos in positions:
                    x, y = int(pos[0]), int(pos[1])
                    if 0 <= y < heatmap.shape[0] and 0 <= x < heatmap.shape[1]:
                        cv2.circle(heatmap, (x, y), 15, 1, -1)

        # Apply gaussian blur and colormap
        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Overlay on frame
        overlay = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)
        cv2.imwrite(output_path, overlay)

        return output_path

    def export_tracking_data(self, output_json='tracking_data.json'):
        """Export all tracking data to JSON."""
        team_analysis = self.analyze_team_movement()

        data = {
            'summary': {
                'num_players': team_analysis['num_players_tracked'] if team_analysis else 0,
                'total_distance': team_analysis['total_team_distance'] if team_analysis else 0,
                'avg_speed': team_analysis['avg_team_speed'] if team_analysis else 0
            },
            'players': []
        }

        for track_id in self.track_history:
            analysis = self.analyze_player_movement(track_id)
            if analysis:
                data['players'].append(analysis)

        with open(output_json, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Tracking data exported to: {output_json}")
        return data


# Example usage
if __name__ == "__main__":
    tracker = PlayerTracker()

    # Track players in video
    tracks = tracker.track_video(
        "football_match.mp4",
        output_path="tracked_output.mp4"
    )

    # Analyze team movement
    team_stats = tracker.analyze_team_movement()
    print(f"\nTeam Statistics:")
    print(f"Players tracked: {team_stats['num_players_tracked']}")
    print(f"Total distance: {team_stats['total_team_distance']:.2f} pixels")
    print(f"Avg team speed: {team_stats['avg_team_speed']:.2f} px/s")

    # Analyze individual player
    player_stats = tracker.analyze_player_movement(0)
    if player_stats:
        print(f"\nPlayer 0 Statistics:")
        print(f"Distance: {player_stats['total_distance']:.2f} pixels")
        print(f"Avg speed: {player_stats['avg_speed']:.2f} px/s")
        print(f"Sprints detected: {len(player_stats['sprints'])}")

    # Generate heatmap
    tracker.generate_heatmap("football_match.mp4", "player_heatmap.jpg")

    # Export data
    tracker.export_tracking_data("tracking_data.json")
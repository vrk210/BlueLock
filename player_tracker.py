import cv2
import numpy as np
import torch
from collections import defaultdict
from player_tracker_base import PlayerTracker
from pitch_calibration import PitchCalibration
from sklearn.cluster import KMeans
import json


class EnhancedPlayerTracker(PlayerTracker):
    """
    Enhanced player tracker with real-world coordinates and advanced analytics.
    """

    def __init__(self, calibration_file: str = None):
        super().__init__()
        self.calibrator = PitchCalibration()

        if calibration_file:
            self.calibrator.load_calibration(calibration_file)
            self.calibrated = True
        else:
            self.calibrated = False

    def track_with_calibration(self, video_path: str, calibration_file: str = None,
                               output_path: str = None):
        """
        Track players with pitch calibration for real-world measurements.
        """
        # Load or create calibration
        if calibration_file:
            self.calibrator.load_calibration(calibration_file)
            self.calibrated = True
        elif not self.calibrated:
            print("No calibration provided. Measurements will be in pixels.")
            print("Use calibrator.interactive_calibration() to calibrate.")

        # Track normally
        tracks = self.track_video(video_path, output_path)

        return tracks

    def analyze_player_movement_real(self, track_id: int, fps: float):
        """
        Analyze player movement in real-world coordinates (meters, km/h).

        Returns:
            Movement statistics in real units
        """
        if track_id not in self.track_history:
            return None

        track = self.track_history[track_id]
        positions_pixels = np.array(track['positions'])

        # Get basic pixel analysis
        pixel_analysis = self.analyze_player_movement(track_id)
        if not pixel_analysis:
            return None

        # If calibrated, convert to real-world units
        if self.calibrated:
            # Convert positions to meters
            positions_meters = self.calibrator.pixel_to_meters(positions_pixels)

            # Calculate real distances
            total_distance_m = 0
            for i in range(1, len(positions_meters)):
                dist = np.linalg.norm(positions_meters[i] - positions_meters[i - 1])
                total_distance_m += dist

            # Calculate real velocities (m/s and km/h)
            velocities_real = []
            speeds_kmh = []

            if len(positions_meters) >= 2:
                dt = 1.0 / fps
                for i in range(1, len(positions_meters)):
                    vel = (positions_meters[i] - positions_meters[i - 1]) / dt
                    speed_ms = np.linalg.norm(vel)
                    speed_kmh = speed_ms * 3.6  # Convert m/s to km/h

                    velocities_real.append(vel)
                    speeds_kmh.append(speed_kmh)

            avg_speed_kmh = np.mean(speeds_kmh) if speeds_kmh else 0
            max_speed_kmh = np.max(speeds_kmh) if speeds_kmh else 0

            # Detect sprints (above 20 km/h typically considered sprinting)
            sprint_threshold = 20.0  # km/h
            sprints = []
            for i, speed in enumerate(speeds_kmh):
                if speed > sprint_threshold:
                    sprints.append({
                        'timestamp': track['timestamps'][i + 1],
                        'speed_kmh': float(speed),
                        'position_meters': positions_meters[i + 1].tolist()
                    })

            # Calculate area covered in square meters
            from scipy.spatial import ConvexHull
            if len(positions_meters) >= 3:
                try:
                    hull = ConvexHull(positions_meters)
                    area_m2 = hull.volume
                except:
                    area_m2 = 0
            else:
                area_m2 = 0

            return {
                'player_id': track_id,
                'total_distance_km': total_distance_m / 1000,
                'total_distance_m': total_distance_m,
                'avg_speed_kmh': avg_speed_kmh,
                'max_speed_kmh': max_speed_kmh,
                'avg_speed_ms': avg_speed_kmh / 3.6,
                'max_speed_ms': max_speed_kmh / 3.6,
                'duration_minutes': (track['timestamps'][-1] - track['timestamps'][0]) / 60,
                'sprints': sprints,
                'sprint_count': len(sprints),
                'area_covered_m2': area_m2,
                'positions_meters': positions_meters.tolist(),
                'timestamps': track['timestamps'],
                'calibrated': True
            }
        else:
            # Return pixel-based analysis with warning
            pixel_analysis['calibrated'] = False
            pixel_analysis['warning'] = 'Not calibrated - values are in pixels'
            return pixel_analysis

    def compare_halves(self, first_half_path: str, second_half_path: str, fps: float):
        """
        Compare player performance between first and second half.
        """
        print("Tracking first half...")
        tracker1 = EnhancedPlayerTracker()
        if self.calibrated:
            tracker1.calibrator = self.calibrator
            tracker1.calibrated = True

        tracks1 = tracker1.track_video(first_half_path)

        print("\nTracking second half...")
        tracker2 = EnhancedPlayerTracker()
        if self.calibrated:
            tracker2.calibrator = self.calibrator
            tracker2.calibrated = True

        tracks2 = tracker2.track_video(second_half_path)

        # Analyze each half
        print("\nAnalyzing first half...")
        first_half_stats = []
        for track_id in tracker1.track_history:
            stats = tracker1.analyze_player_movement_real(track_id, fps)
            if stats:
                first_half_stats.append(stats)

        print("Analyzing second half...")
        second_half_stats = []
        for track_id in tracker2.track_history:
            stats = tracker2.analyze_player_movement_real(track_id, fps)
            if stats:
                second_half_stats.append(stats)

        # Compare statistics
        comparison = {
            'first_half': {
                'num_players': len(first_half_stats),
                'avg_distance_km': np.mean([s['total_distance_km'] for s in first_half_stats]),
                'avg_speed_kmh': np.mean([s['avg_speed_kmh'] for s in first_half_stats]),
                'total_sprints': sum([s['sprint_count'] for s in first_half_stats]),
                'player_stats': first_half_stats
            },
            'second_half': {
                'num_players': len(second_half_stats),
                'avg_distance_km': np.mean([s['total_distance_km'] for s in second_half_stats]),
                'avg_speed_kmh': np.mean([s['avg_speed_kmh'] for s in second_half_stats]),
                'total_sprints': sum([s['sprint_count'] for s in second_half_stats]),
                'player_stats': second_half_stats
            }
        }

        # Calculate fatigue indicators
        if comparison['first_half']['avg_distance_km'] > 0:
            distance_decline = (
                    (comparison['first_half']['avg_distance_km'] -
                     comparison['second_half']['avg_distance_km']) /
                    comparison['first_half']['avg_distance_km'] * 100
            )
            speed_decline = (
                    (comparison['first_half']['avg_speed_kmh'] -
                     comparison['second_half']['avg_speed_kmh']) /
                    comparison['first_half']['avg_speed_kmh'] * 100
            )

            comparison['fatigue_indicators'] = {
                'distance_decline_percent': distance_decline,
                'speed_decline_percent': speed_decline,
                'sprint_decline': comparison['first_half']['total_sprints'] -
                                  comparison['second_half']['total_sprints']
            }

        return comparison

    def detect_formation(self, frame_number: int = None):
        """
        Detect team formation based on player positions.
        Returns likely formation (e.g., "4-4-2", "4-3-3").
        """
        if not self.track_history:
            return None

        # Get positions at specific frame or average positions
        positions = []

        if frame_number is not None:
            for track_id, track in self.track_history.items():
                if len(track['positions']) > frame_number:
                    positions.append(track['positions'][frame_number])
        else:
            # Use average positions
            for track_id, track in self.track_history.items():
                if len(track['positions']) > 0:
                    avg_pos = np.mean(track['positions'], axis=0)
                    positions.append(avg_pos)

        if len(positions) < 8:  # Need at least 8 players (excluding keeper and subs)
            return None

        positions = np.array(positions)

        # Convert to meters if calibrated
        if self.calibrated:
            positions = self.calibrator.pixel_to_meters(positions)

        # Sort by x-coordinate (depth on field)
        sorted_indices = np.argsort(positions[:, 0])
        sorted_positions = positions[sorted_indices]

        # Cluster into lines (defenders, midfielders, forwards)
        from sklearn.cluster import KMeans

        # Try different formation patterns
        formations_to_test = [
            {'name': '4-4-2', 'lines': [4, 4, 2]},
            {'name': '4-3-3', 'lines': [4, 3, 3]},
            {'name': '3-5-2', 'lines': [3, 5, 2]},
            {'name': '4-2-3-1', 'lines': [4, 2, 3, 1]},
            {'name': '3-4-3', 'lines': [3, 4, 3]}
        ]

        best_formation = None
        best_score = float('inf')

        for formation in formations_to_test:
            n_lines = len(formation['lines'])
            if len(positions) < sum(formation['lines']):
                continue

            # Cluster by x-coordinate (depth)
            kmeans = KMeans(n_clusters=n_lines, random_state=42)
            x_coords = sorted_positions[:, 0].reshape(-1, 1)
            clusters = kmeans.fit_predict(x_coords)

            # Check if cluster sizes match formation
            cluster_sizes = [np.sum(clusters == i) for i in range(n_lines)]
            cluster_sizes.sort(reverse=True)
            expected = sorted(formation['lines'], reverse=True)

            # Calculate difference score
            score = sum(abs(a - b) for a, b in zip(cluster_sizes, expected))

            if score < best_score:
                best_score = score
                best_formation = formation['name']

        return {
            'formation': best_formation,
            'confidence': 1.0 / (1.0 + best_score),
            'num_players': len(positions)
        }

    def export_to_sports_analytics_format(self, output_file: str, fps: float):
        """
        Export data in format compatible with sports analytics tools.
        CSV format: timestamp, player_id, x, y, speed, event
        """
        import csv

        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'player_id', 'x_meters', 'y_meters',
                             'speed_kmh', 'is_sprinting'])

            for track_id, track in self.track_history.items():
                analysis = self.analyze_player_movement_real(track_id, fps)
                if not analysis:
                    continue

                positions = np.array(track['positions'])
                timestamps = track['timestamps']

                # Convert to meters if calibrated
                if self.calibrated:
                    positions_meters = self.calibrator.pixel_to_meters(positions)
                else:
                    positions_meters = positions

                # Calculate speeds
                speeds = []
                for i in range(1, len(positions_meters)):
                    dt = timestamps[i] - timestamps[i - 1]
                    if dt > 0:
                        vel = np.linalg.norm(positions_meters[i] - positions_meters[i - 1]) / dt
                        speed_kmh = vel * 3.6 if self.calibrated else vel
                        speeds.append(speed_kmh)
                    else:
                        speeds.append(0)

                speeds = [0] + speeds  # Add initial speed

                # Write data
                for i, (pos, ts, speed) in enumerate(zip(positions_meters, timestamps, speeds)):
                    is_sprinting = 1 if speed > 20 else 0
                    writer.writerow([
                        f"{ts:.3f}",
                        track_id,
                        f"{pos[0]:.2f}",
                        f"{pos[1]:.2f}",
                        f"{speed:.2f}",
                        is_sprinting
                    ])

        print(f"Data exported to: {output_file}")
        return output_file

    def generate_comprehensive_report(self, video_path: str, fps: float,
                                      output_json: str = 'comprehensive_report.json'):
        """
        Generate a comprehensive analysis report with all metrics.
        """
        report = {
            'video': video_path,
            'calibrated': self.calibrated,
            'timestamp': str(np.datetime64('now')),
            'team_statistics': {},
            'player_statistics': [],
            'formation_analysis': None,
            'sprint_analysis': {},
            'distance_analysis': {}
        }

        # Team statistics
        all_player_stats = []
        for track_id in self.track_history:
            stats = self.analyze_player_movement_real(track_id, fps)
            if stats:
                all_player_stats.append(stats)
                report['player_statistics'].append(stats)

        if all_player_stats:
            report['team_statistics'] = {
                'num_players_tracked': len(all_player_stats),
                'total_distance_km': sum(s['total_distance_km'] for s in all_player_stats if 'total_distance_km' in s),
                'avg_player_distance_km': np.mean(
                    [s['total_distance_km'] for s in all_player_stats if 'total_distance_km' in s]),
                'avg_speed_kmh': np.mean([s['avg_speed_kmh'] for s in all_player_stats if 'avg_speed_kmh' in s]),
                'max_speed_kmh': max([s['max_speed_kmh'] for s in all_player_stats if 'max_speed_kmh' in s]),
                'total_sprints': sum(s['sprint_count'] for s in all_player_stats if 'sprint_count' in s)
            }

        # Formation analysis
        formation = self.detect_formation()
        if formation:
            report['formation_analysis'] = formation

        # Sprint analysis
        all_sprints = []
        for stats in all_player_stats:
            if 'sprints' in stats:
                all_sprints.extend(stats['sprints'])

        if all_sprints:
            sprint_speeds = [s['speed_kmh'] for s in all_sprints]
            report['sprint_analysis'] = {
                'total_sprints': len(all_sprints),
                'avg_sprint_speed_kmh': np.mean(sprint_speeds),
                'max_sprint_speed_kmh': max(sprint_speeds),
                'sprint_distribution': self._get_sprint_distribution(all_sprints)
            }

        # Distance analysis by player
        if all_player_stats:
            distances = sorted([(s['player_id'], s.get('total_distance_km', 0))
                                for s in all_player_stats],
                               key=lambda x: x[1], reverse=True)

            report['distance_analysis'] = {
                'top_distance_player': distances[0][0] if distances else None,
                'top_distance_km': distances[0][1] if distances else 0,
                'distance_ranking': [
                    {'player_id': pid, 'distance_km': dist}
                    for pid, dist in distances
                ]
            }

        # Save report
        with open(output_json, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Comprehensive report saved to: {output_json}")
        return report

    def _get_sprint_distribution(self, sprints):
        """Get distribution of sprints over time."""
        if not sprints:
            return {}

        timestamps = [s['timestamp'] for s in sprints]
        duration = max(timestamps) - min(timestamps)

        if duration == 0:
            return {}

        # Divide into quarters
        quarter_duration = duration / 4
        quarters = [0, 0, 0, 0]

        for ts in timestamps:
            quarter = int((ts - min(timestamps)) / quarter_duration)
            if quarter > 3:
                quarter = 3
            quarters[quarter] += 1

        return {
            'quarter_1': quarters[0],
            'quarter_2': quarters[1],
            'quarter_3': quarters[2],
            'quarter_4': quarters[3]
        }

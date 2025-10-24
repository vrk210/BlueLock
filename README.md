# Football Player Tracking and Analysis System

Computer vision-based platform for automated player tracking and performance analysis for football video. The system provides real-world measurements including distance covered, speed metrics, and tactical insights.

## Overview

This system uses YOLOv5 for player detection and the Hungarian algorithm for multi-object tracking across video frames. When calibrated, it transforms pixel coordinates to real-world measurements, providing analytics like distance covered in kilometers, speed in kph, sprint detection, and formation analysis.

### Core Features

- Multi-player tracking with unique IDs throughout video
- Real-world measurements (distance in km, speed in km/h)
- Sprint detection and analysis (>20 km/h threshold)
- Formation detection (4-4-2, 4-3-3, etc.)
- First vs. second half performance comparison
- Heatmap generation and trajectory visualization
- CSV and JSON export for external analysis

## Installation

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/football-player-tracking.git
cd football-player-tracking
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create required directories:
```bash
mkdir -p uploaded_videos calibrations preprocessed tracked_videos heatmaps reports exports
```

5. Start the server:
```bash
uvicorn backend:app --reload --host 0.0.0.0 --port 8000
```

## Workflow

### Quick Start (No Calibration)

```bash
curl -X POST -F "video=@match.mp4" \
     -F "create_annotated_video=true" \
     -F "create_heatmap=true" \
     -F "export_csv=true" \
     http://localhost:8000/track/full-analysis/
```

### Improved Workflow (With Calibration)

For real-world measurements in meters and km/h:

**Step 1: Upload video and get calibration frame**
```bash
curl -X POST -F "video=@match.mp4" \
     http://localhost:8000/calibrate/interactive/
```

**Step 2: Save calibration with pitch corner coordinates**

Identify the four corners of the pitch in the returned frame (top-left, top-right, bottom-right, bottom-left):

```bash
curl -X POST \
     -F "video_filename=match.mp4" \
     -F "corner1_x=120" -F "corner1_y=80" \
     -F "corner2_x=1800" -F "corner2_y=85" \
     -F "corner3_x=1750" -F "corner3_y=980" \
     -F "corner4_x=150" -F "corner4_y=975" \
     http://localhost:8000/calibrate/save/
```

**Step 3: Preprocess video**

Not strictly necessary, but does provide better results.

Enhance video quality:
```bash
curl -X POST -F "video=@match.mp4" \
     -F "brightness=1.1" \
     -F "contrast=1.2" \
     -F "denoise=true" \
     http://localhost:8000/preprocess/enhance/
```

**Step 4: Run full analysis**
```bash
curl -X POST -F "video=@match.mp4" \
     -F "calibration_file=cal_match.mp4.json" \
     -F "create_annotated_video=true" \
     -F "create_heatmap=true" \
     -F "export_csv=true" \
     http://localhost:8000/track/full-analysis/
```

**Step 5: Query specific player**
```bash
curl -X POST "http://localhost:8000/track/player/7?video_filename=match.mp4"
```

**Step 6: Download results**
```bash
# Tracked video
curl -O http://localhost:8000/download/video/tracked_match.mp4

# Heatmap
curl -O http://localhost:8000/download/heatmap/heatmap_match.mp4.jpg

# JSON report
curl -O http://localhost:8000/download/report/report_match.mp4.json

# CSV export
curl -O http://localhost:8000/download/export/tracking_match.mp4.csv
```

## API Endpoints

**Calibration**
- `POST /calibrate/interactive/` - Get calibration frame
- `POST /calibrate/save/` - Save pitch calibration
- `GET /calibrate/load/{filename}` - Load existing calibration

**Preprocessing**
- `POST /preprocess/enhance/` - Enhance video quality
- `POST /preprocess/stabilize/` - Stabilize shaky footage
- `POST /preprocess/split-halves/` - Split into halves

**Tracking & Analysis**
- `POST /track/full-analysis/` - Complete tracking analysis
- `POST /track/player/{player_id}` - Specific player statistics
- `POST /track/compare-halves/` - First vs. second half comparison
- `GET /track/formation/{video_filename}` - Formation detection

**Downloads**
- `GET /download/video/{filename}` - Download tracked video
- `GET /download/heatmap/{filename}` - Download heatmap
- `GET /download/report/{filename}` - Download JSON report
- `GET /download/export/{filename}` - Download CSV data

Full documentation: `http://localhost:8000/docs`

## Output Examples

Team Statistics JSON:
```json
{
  "num_players_tracked": 22,
  "total_distance_km": 127.8,
  "avg_player_distance_km": 5.81,
  "avg_speed_kmh": 6.2,
  "max_speed_kmh": 34.7,
  "total_sprints": 147
}
```

Individual Player JSON:
```json
{
  "player_id": 7,
  "total_distance_km": 11.2,
  "avg_speed_kmh": 7.8,
  "max_speed_kmh": 32.5,
  "sprint_count": 23,
  "area_covered_m2": 1847.3
}
```

Example CSV Format:
```csv
timestamp,player_id,x_meters,y_meters,speed_kmh,is_sprinting
0.033,0,52.3,34.1,12.4,0
0.066,0,52.5,34.3,13.2,0
0.099,0,52.8,34.6,24.8,1
```


How it works

1. **Detection**: YOLOv5 identifies all players in each frame using bounding boxes
2. **Tracking**: Hungarian algorithm matches players across consecutive frames using IoU metrics
3. **Calibration**: Homography matrix transforms pixel positions to real-world pitch coordinates
4. **Analysis**: Calculate distances, speeds, sprints, and formations from tracked positions
5. **Export**: Generate reports, visualizations, and data files

## License

MIT License - See LICENSE file for details.

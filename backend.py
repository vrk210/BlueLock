from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import shutil
import os
import cv2
from datetime import datetime

from player_tracker import EnhancedPlayerTracker
from pitch_calibration import PitchCalibration, DataPreprocessor

app = FastAPI(
    title="Football Analytics Platform",
    description="Complete player tracking and game analysis system",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = "uploaded_videos"
CALIBRATION_DIR = "calibrations"
OUTPUT_DIR = "tracked_videos"
HEATMAP_DIR = "heatmaps"
REPORT_DIR = "reports"
EXPORT_DIR = "exports"
PREPROCESSED_DIR = "preprocessed"

for directory in [UPLOAD_DIR, CALIBRATION_DIR, OUTPUT_DIR, HEATMAP_DIR,
                  REPORT_DIR, EXPORT_DIR, PREPROCESSED_DIR]:
    os.makedirs(directory, exist_ok=True)

# Global trackers cache
active_trackers = {}


# ============= CALIBRATION ENDPOINTS =============

@app.post("/calibrate/interactive/")
async def calibrate_pitch_interactive(video: UploadFile):
    """
    Interactive pitch calibration - returns frame for manual corner selection.
    User should then call /calibrate/save/ with corner coordinates.
    """
    video_path = os.path.join(UPLOAD_DIR, video.filename)
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # Extract first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(status_code=400, detail="Could not read video")

    # Save frame for calibration
    frame_path = os.path.join(CALIBRATION_DIR, f"frame_{video.filename}.jpg")
    cv2.imwrite(frame_path, frame)

    return {
        "status": "ready_for_calibration",
        "video": video.filename,
        "frame_path": f"frame_{video.filename}.jpg",
        "instructions": "Click 4 corners: top-left, top-right, bottom-right, bottom-left"
    }


@app.post("/calibrate/save/")
async def save_calibration(
        video_filename: str = Form(...),
        corner1_x: float = Form(...),
        corner1_y: float = Form(...),
        corner2_x: float = Form(...),
        corner2_y: float = Form(...),
        corner3_x: float = Form(...),
        corner3_y: float = Form(...),
        corner4_x: float = Form(...),
        corner4_y: float = Form(...)
):
    """
    Save pitch calibration with 4 corner points.
    """
    corners = [
        [corner1_x, corner1_y],
        [corner2_x, corner2_y],
        [corner3_x, corner3_y],
        [corner4_x, corner4_y]
    ]

    calibrator = PitchCalibration()
    calibrator.calibrate_from_corners(corners)

    calibration_file = os.path.join(CALIBRATION_DIR, f"cal_{video_filename}.json")
    calibrator.save_calibration(calibration_file)

    return {
        "status": "calibration_saved",
        "calibration_file": f"cal_{video_filename}.json",
        "corners": corners
    }


@app.get("/calibrate/load/{filename}")
async def load_calibration(filename: str):
    """Load existing calibration."""
    calibration_path = os.path.join(CALIBRATION_DIR, filename)

    if not os.path.exists(calibration_path):
        raise HTTPException(status_code=404, detail="Calibration not found")

    calibrator = PitchCalibration()
    calibrator.load_calibration(calibration_path)

    return {
        "status": "loaded",
        "calibration_file": filename,
        "pitch_dimensions": {
            "length": calibrator.PITCH_LENGTH,
            "width": calibrator.PITCH_WIDTH
        }
    }


# ============= PREPROCESSING ENDPOINTS =============

@app.post("/preprocess/enhance/")
async def enhance_video(
        video: UploadFile,
        brightness: float = Form(1.0),
        contrast: float = Form(1.0),
        denoise: bool = Form(True)
):
    """
    Enhance video quality before tracking.
    """
    video_path = os.path.join(UPLOAD_DIR, video.filename)
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    preprocessor = DataPreprocessor()
    output_path = os.path.join(PREPROCESSED_DIR, f"enhanced_{video.filename}")

    enhanced_path = preprocessor.enhance_video(
        video_path,
        output_path,
        brightness=brightness,
        contrast=contrast,
        denoise=denoise
    )

    return {
        "status": "enhanced",
        "original": video.filename,
        "enhanced": f"enhanced_{video.filename}",
        "parameters": {
            "brightness": brightness,
            "contrast": contrast,
            "denoise": denoise
        }
    }


@app.post("/preprocess/stabilize/")
async def stabilize_video(video: UploadFile):
    """
    Stabilize shaky video footage.
    """
    video_path = os.path.join(UPLOAD_DIR, video.filename)
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    preprocessor = DataPreprocessor()
    output_path = os.path.join(PREPROCESSED_DIR, f"stabilized_{video.filename}")

    stabilized_path = preprocessor.stabilize_video(video_path, output_path)

    return {
        "status": "stabilized",
        "original": video.filename,
        "stabilized": f"stabilized_{video.filename}"
    }


@app.post("/preprocess/split-halves/")
async def split_video_halves(video: UploadFile):
    """
    Split video into first and second half.
    """
    video_path = os.path.join(UPLOAD_DIR, video.filename)
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    preprocessor = DataPreprocessor()
    split_dir = os.path.join(PREPROCESSED_DIR, f"split_{video.filename}")

    first_half, second_half = preprocessor.split_by_half(video_path, split_dir)

    return {
        "status": "split",
        "original": video.filename,
        "first_half": os.path.basename(first_half),
        "second_half": os.path.basename(second_half)
    }


# ============= TRACKING ENDPOINTS =============

@app.post("/track/full-analysis/")
async def full_tracking_analysis(
        video: UploadFile,
        background_tasks: BackgroundTasks,
        calibration_file: Optional[str] = Form(None),
        create_annotated_video: bool = Form(True),
        create_heatmap: bool = Form(True),
        export_csv: bool = Form(True)
):
    """
    Complete tracking analysis with all features.
    This is the main endpoint for comprehensive analysis.
    """
    # Save video
    video_path = os.path.join(UPLOAD_DIR, video.filename)
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # Get video FPS
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Initialize tracker
    cal_path = os.path.join(CALIBRATION_DIR, calibration_file) if calibration_file else None
    tracker = EnhancedPlayerTracker(calibration_file=cal_path)

    # Track players
    output_video_path = None
    if create_annotated_video:
        output_video_path = os.path.join(OUTPUT_DIR, f"tracked_{video.filename}")

    print(f"Starting full analysis for {video.filename}...")
    tracks = tracker.track_with_calibration(
        video_path,
        calibration_file=cal_path,
        output_path=output_video_path
    )

    # Generate comprehensive report
    report_path = os.path.join(REPORT_DIR, f"report_{video.filename}.json")
    report = tracker.generate_comprehensive_report(video_path, fps, report_path)

    # Generate heatmap
    heatmap_path = None
    if create_heatmap:
        heatmap_path = os.path.join(HEATMAP_DIR, f"heatmap_{video.filename}.jpg")
        tracker.generate_heatmap(video_path, heatmap_path)

    # Export CSV
    csv_path = None
    if export_csv:
        csv_path = os.path.join(EXPORT_DIR, f"tracking_{video.filename}.csv")
        tracker.export_to_sports_analytics_format(csv_path, fps)

    # Cache tracker for future queries
    active_trackers[video.filename] = tracker

    return {
        "status": "analysis_complete",
        "video": video.filename,
        "summary": report['team_statistics'],
        "formation": report.get('formation_analysis'),
        "files": {
            "report": f"report_{video.filename}.json",
            "tracked_video": f"tracked_{video.filename}" if output_video_path else None,
            "heatmap": f"heatmap_{video.filename}.jpg" if heatmap_path else None,
            "csv_export": f"tracking_{video.filename}.csv" if csv_path else None
        }
    }


@app.post("/track/player/{player_id}")
async def track_specific_player(video_filename: str, player_id: int):
    """
    Get detailed analysis for a specific player.
    Video must have been analyzed first.
    """
    if video_filename not in active_trackers:
        raise HTTPException(
            status_code=404,
            detail="Video not analyzed yet. Run /track/full-analysis/ first"
        )

    tracker = active_trackers[video_filename]

    # Get video FPS
    video_path = os.path.join(UPLOAD_DIR, video_filename)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Analyze player
    player_stats = tracker.analyze_player_movement_real(player_id, fps)

    if not player_stats:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found")

    # Generate player heatmap
    heatmap_path = os.path.join(HEATMAP_DIR, f"player_{player_id}_{video_filename}.jpg")
    tracker.generate_heatmap(video_path, heatmap_path, player_id=player_id)

    return {
        "status": "success",
        "player_id": player_id,
        "statistics": player_stats,
        "heatmap": f"player_{player_id}_{video_filename}.jpg"
    }


@app.post("/track/compare-halves/")
async def compare_match_halves(
        first_half: UploadFile,
        second_half: UploadFile,
        calibration_file: Optional[str] = Form(None)
):
    """
    Compare player performance between first and second half.
    """
    # Save videos
    first_path = os.path.join(UPLOAD_DIR, first_half.filename)
    second_path = os.path.join(UPLOAD_DIR, second_half.filename)

    with open(first_path, "wb") as f:
        shutil.copyfileobj(first_half.file, f)
    with open(second_path, "wb") as f:
        shutil.copyfileobj(second_half.file, f)

    # Get FPS
    cap = cv2.VideoCapture(first_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Initialize tracker
    cal_path = os.path.join(CALIBRATION_DIR, calibration_file) if calibration_file else None
    tracker = EnhancedPlayerTracker(calibration_file=cal_path)

    # Compare halves
    comparison = tracker.compare_halves(first_path, second_path, fps)

    # Save comparison report
    report_path = os.path.join(REPORT_DIR, "half_comparison.json")
    import json
    with open(report_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    return {
        "status": "comparison_complete",
        "comparison": comparison,
        "report_file": "half_comparison.json"
    }


@app.get("/track/formation/{video_filename}")
async def detect_team_formation(video_filename: str, frame_number: Optional[int] = None):
    """
    Detect team formation from tracking data.
    """
    if video_filename not in active_trackers:
        raise HTTPException(
            status_code=404,
            detail="Video not analyzed yet"
        )

    tracker = active_trackers[video_filename]
    formation = tracker.detect_formation(frame_number)

    if not formation:
        raise HTTPException(status_code=404, detail="Could not detect formation")

    return {
        "status": "success",
        "formation": formation,
        "frame_number": frame_number
    }


# ============= FILE DOWNLOAD ENDPOINTS =============

@app.get("/download/video/{filename}")
async def download_video(filename: str):
    """Download tracked/processed video."""
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        file_path = os.path.join(PREPROCESSED_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, media_type="video/mp4", filename=filename)


@app.get("/download/heatmap/{filename}")
async def download_heatmap(filename: str):
    """Download heatmap image."""
    file_path = os.path.join(HEATMAP_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, media_type="image/jpeg", filename=filename)


@app.get("/download/report/{filename}")
async def download_report(filename: str):
    """Download analysis report JSON."""
    file_path = os.path.join(REPORT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, media_type="application/json", filename=filename)


@app.get("/download/export/{filename}")
async def download_export(filename: str):
    """Download CSV export."""
    file_path = os.path.join(EXPORT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, media_type="text/csv", filename=filename)


# ============= INFO ENDPOINTS =============

@app.get("/")
async def root():
    return {
        "name": "Football Analytics Platform",
        "version": "2.0.0",
        "description": "Complete player tracking and game analysis system with real-world measurements",
        "features": [
            "Pitch calibration for real-world measurements (meters, km/h)",
            "Video preprocessing (enhancement, stabilization)",
            "Player detection and tracking",
            "Movement analysis (distance, speed, sprints)",
            "Formation detection",
            "Half-by-half comparison",
            "Heatmap generation",
            "Export to CSV for external tools"
        ],
        "workflow": {
            "1": "Upload video",
            "2": "Calibrate pitch (optional but recommended)",
            "3": "Preprocess if needed (enhance/stabilize)",
            "4": "Run full analysis",
            "5": "Query specific players or formations",
            "6": "Download results"
        }
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Run with: uvicorn backend_complete_system:app --reload --host 0.0.0.0 --port 8000
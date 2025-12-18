# Cross-Camera Player Mapping

This project implements a system that maps players between different camera views (broadcast and tacticam) in sports footage. The system uses perspective-invariant graph features to match players across different camera angles and maintains stable associations over time.

## Features

- **Perspective-invariant matching**: Uses ordinal and topological features that remain consistent across different camera perspectives
- **Multi-camera tracking**: Tracks players in both broadcast and tacticam video feeds
- **Temporal smoothing**: Reduces flickering in player ID assignments using voting over recent frames
- **Stable matching**: Maintains consistent player associations over time
- **Visual output**: Combines both video streams with player boxes, trails, and matched IDs

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- Ultralytics YOLO
- TQDM
- SciPy

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Make sure you have the required YOLO model file (`yolov11_custom.pt`)
2. Place your broadcast and tacticam video files in the appropriate location
3. Run the script:
   ```bash
   python main.py
   ```

## Architecture

The system is modularized into several components:

- `constants.py`: Configuration values and constants
- `utils.py`: Core utility functions and classes:
  - `RobustTracker`: Handles player tracking across frames
  - `PerspectiveInvariantGraph`: Computes perspective-invariant features
  - `AdaptiveMatcher`: Matches players between camera views
  - `TemporalSmoother`: Smooths match assignments over time
  - Drawing and utility functions
- `main.py`: Main video processing loop with modular functions

## How It Works

1. **Detection**: Players are detected in both video streams using YOLO
2. **Tracking**: Each detected player is assigned a consistent ID across frames
3. **Graph Construction**: A perspective-invariant graph is constructed for each camera view
4. **Matching**: Players are matched between camera views using ordinal features
5. **Temporal Smoothing**: Matches are smoothed over time to reduce flickering
6. **Visualization**: The combined video stream is generated with player information
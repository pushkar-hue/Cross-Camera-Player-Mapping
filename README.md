# Cross-Camera Player Mapping

This project implements a system that maps players between different camera views (broadcast and tacticam) in sports footage. The system uses perspective-invariant graph features to match players across different camera angles and maintains stable associations over time.



## Features

- **Perspective-invariant matching**: Uses ordinal and topological features that remain consistent across different camera perspectives
- **Multi-camera tracking**: Tracks players in both broadcast and tacticam video feeds
- **Temporal smoothing**: Reduces flickering in player ID assignments using voting over recent frames
- **Stable matching**: Maintains consistent player associations over time
- **Visual output**: Combines both video streams with player boxes, trails, and matched IDs

## Approach

The core challenge in cross-camera player mapping is **perspective distortion**. Players that appear close together in a broadcast view might appear far apart in a tactical view, and absolute pixel distances are not preserved across different camera angles.

Instead of relying on a fixed homography matrix (which breaks when cameras pan or zoom), this system implements a **Perspective-Invariant Graph approach**:

### 1. Perspective-Invariant Feature Extraction
For every frame, we construct a graph where nodes represent players. Instead of using absolute $(x, y)$ coordinates, we compute **ordinal and topological features** which are robust to perspective changes:
* **Ordinal Ranks:** We calculate normalized $x$ and $y$ ranks (0.0 to 1.0). If a player is the "left-most" in the broadcast view, they will likely be the "left-most" in the tactical view, regardless of the camera angle.
* **Topological Neighbors:** We identify the $k$-nearest neighbors for every player. The identity of a player's neighbors is a topological property that tends to persist across views.
* **Relative Directionality:** We encode the number of neighbors strictly to the left, right, above, and below each player.

### 2. Adaptive Graph Matching
We treat the mapping problem as a **Linear Sum Assignment Problem**. The cost function between a Broadcast Player ($P_b$) and a Tactical Player ($P_t$) is calculated dynamically:
* **Base Cost:** Weighted difference in ordinal ranks (X-rank differences are penalized more heavily than Y-rank differences).
* **Topological Bonus:** If $P_b$'s neighbor is already matched to $P_t$'s neighbor, the cost is significantly reduced. This allows the algorithm to "grow" matches from confident pairs.
* **Momentum:** The cost matrix includes a "momentum" term that favors maintaining the match from the previous frame.

### 3. Temporal Stability & Smoothing
To prevent "flickering" IDs when detection confidence drops or players cross paths, we implement a two-stage stabilization pipeline:
1.  **Stable Sets:** Pairs that maintain high matching confidence over several frames are promoted to a "Stable Set." These matches are given priority in subsequent frames.
2.  **Voting Smoother:** A sliding window (size=7) maintains a history of raw matches. The final ID displayed is determined by a majority vote within this window, filtering out single-frame outliers.

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

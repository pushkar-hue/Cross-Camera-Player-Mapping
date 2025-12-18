# Model and video paths
MODEL_PATH = "yolov11_custom.pt"
BROADCAST_VIDEO = "Assignment Materials/broadcast.mp4"
TACTICAM_VIDEO = "Assignment Materials/tacticam.mp4"
OUTPUT_FILENAME = "output_perspective_aware.mp4"

# Target classes for detection (person, player, etc.)
TARGET_CLASSES = [1, 2]

# Tacticam video skip seconds
TACTICAM_SKIP_SECONDS = 2.5

# Tracker parameters
BROADCAST_TRACKER_MAX_DIST = 120
BROADCAST_TRACKER_MAX_MISSING = 20
TACTICAM_TRACKER_MAX_DIST = 80
TACTICAM_TRACKER_MAX_MISSING = 20

# Smoothing window size
SMOOTHING_WINDOW_SIZE = 7

# Colors for visualization
COLORS = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
    (170, 110, 40), (128, 0, 0), (170, 255, 195), (0, 0, 128)
]

# Matching thresholds
MATCH_ACCEPT_THRESHOLD = 80
STABILITY_CHECK_X_DIFF = 0.4
STABILITY_CHECK_Y_DIFF = 0.5

# Display parameters
GRAPH_CONNECTIONS_ALPHA = 0.4
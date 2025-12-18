import cv2
import numpy as np
from tqdm import tqdm
from constants import (
    MODEL_PATH, BROADCAST_VIDEO, TACTICAM_VIDEO, OUTPUT_FILENAME,
    TACTICAM_SKIP_SECONDS, BROADCAST_TRACKER_MAX_DIST, BROADCAST_TRACKER_MAX_MISSING,
    TACTICAM_TRACKER_MAX_DIST, TACTICAM_TRACKER_MAX_MISSING, SMOOTHING_WINDOW_SIZE,
    GRAPH_CONNECTIONS_ALPHA
)
from utils import (
    get_feet_points, RobustTracker, PerspectiveInvariantGraph,
    AdaptiveMatcher, TemporalSmoother, draw_player_box, draw_graph_connections, get_color
)
from ultralytics import YOLO


def initialize_components():
    """Initialize all required components for the player mapping system."""
    model = YOLO(MODEL_PATH)

    # Initialize trackers
    tracker_b = RobustTracker(
        max_dist=BROADCAST_TRACKER_MAX_DIST,
        max_missing=BROADCAST_TRACKER_MAX_MISSING
    )
    tracker_t = RobustTracker(
        max_dist=TACTICAM_TRACKER_MAX_DIST,
        max_missing=TACTICAM_TRACKER_MAX_MISSING
    )

    # Initialize matcher and smoother
    matcher = AdaptiveMatcher()
    smoother = TemporalSmoother(window_size=SMOOTHING_WINDOW_SIZE)

    return model, tracker_b, tracker_t, matcher, smoother


def initialize_video_capture():
    """Initialize video capture objects and output writer."""
    cap_b = cv2.VideoCapture(BROADCAST_VIDEO)
    cap_t = cv2.VideoCapture(TACTICAM_VIDEO)

    # Skip frames in the tacticam video
    fps_t = cap_t.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(TACTICAM_SKIP_SECONDS * fps_t)
    cap_t.set(cv2.CAP_PROP_POS_FRAMES, frames_to_skip)

    w_b, h_b = int(cap_b.get(3)), int(cap_b.get(4))
    w_t, h_t = int(cap_t.get(3)), int(cap_t.get(4))
    fps = int(cap_b.get(5))

    out = cv2.VideoWriter(
        OUTPUT_FILENAME,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w_b + w_t, max(h_b, h_t))
    )

    total_frames = int(min(cap_b.get(7), cap_t.get(7) - frames_to_skip))

    return cap_b, cap_t, out, total_frames


def process_frame_detection(model, frame_b, frame_t):
    """Run detection on both frames and extract feet points."""
    res_b = model(frame_b, verbose=False)[0]
    res_t = model(frame_t, verbose=False)[0]

    pts_b, idx_b = get_feet_points(
        res_b.boxes.xyxy.cpu().numpy(),
        res_b.boxes.cls.cpu().numpy()
    )
    pts_t, idx_t = get_feet_points(
        res_t.boxes.xyxy.cpu().numpy(),
        res_t.boxes.cls.cpu().numpy()
    )

    return res_b, res_t, pts_b, pts_t, idx_b, idx_t


def process_frame_tracking(tracker_b, tracker_t, pts_b, pts_t):
    """Update trackers and get confirmed tracks."""
    ids_b = tracker_b.update(pts_b.tolist() if len(pts_b) > 0 else [])
    ids_t = tracker_t.update(pts_t.tolist() if len(pts_t) > 0 else [])

    pos_b, conf_ids_b = tracker_b.get_confirmed(min_hits=2)
    pos_t, conf_ids_t = tracker_t.get_confirmed(min_hits=2)

    return ids_b, ids_t, pos_b, pos_t, conf_ids_b, conf_ids_t


def process_frame_matching(graph_b, graph_t, matcher, smoother):
    """Perform matching between graphs and update smoother."""
    raw_matches = matcher.match(graph_b, graph_t)
    matcher.verify_stable_matches(graph_b, graph_t)
    smoother.add_frame(raw_matches)
    matcher.cleanup(set(graph_b.ids))

    return raw_matches


def draw_tacticam_players(display_t, ids_t, res_t, idx_t, tracker_t):
    """Draw tacticam players with trails."""
    for i, t_id in enumerate(ids_t):
        if i >= len(idx_t):
            continue
        box = res_t.boxes.xyxy[idx_t[i]].cpu().numpy()
        color = get_color(t_id)
        draw_player_box(display_t, box, f"ID:{t_id}", color, is_stable=True)

        # Trail
        tr = tracker_t.tracks.get(t_id)
        if tr and len(tr['history']) > 1:
            pts = np.array(tr['history'][-12:], dtype=np.int32)
            cv2.polylines(display_t, [pts], False, color, 2)


def draw_broadcast_players(display_b, ids_b, res_b, idx_b, matcher, smoother, tracker_b, raw_matches):
    """Draw broadcast players with matched IDs and trails."""
    frame_matched = 0
    frame_stable = 0
    frame_smoothed = 0

    for i, b_id in enumerate(ids_b):
        if i >= len(idx_b):
            continue

        box = res_b.boxes.xyxy[idx_b[i]].cpu().numpy()

        display_id = None
        is_stable = False
        source = ""

        stable_id = matcher.stable_matches.get(b_id)
        if stable_id is not None:
            display_id = stable_id
            is_stable = True
            source = "stable"
            frame_stable += 1

        if display_id is None:
            smoothed_id, confidence = smoother.get_smoothed_match(b_id)
            if smoothed_id is not None and confidence >= 0.5:
                display_id = smoothed_id
                source = "smooth"
                frame_smoothed += 1

        if display_id is None and b_id in raw_matches:
            display_id = raw_matches[b_id]
            source = "raw"
            frame_matched += 1

        # Draw player box
        if display_id is not None:
            color = get_color(display_id)
            label = f"ID:{display_id}" if is_stable else f"ID:{display_id}?"
            draw_player_box(display_b, box, label, color, is_stable)
        else:
            draw_player_box(display_b, box, "?", (128, 128, 128), False)

        # Draw trail
        tr = tracker_b.tracks.get(b_id)
        if tr and len(tr['history']) > 1:
            pts = np.array(tr['history'][-8:], dtype=np.int32)
            col = get_color(display_id) if display_id else (128, 128, 128)
            cv2.polylines(display_b, [pts], False, col, 1)

    return frame_matched, frame_stable, frame_smoothed


def add_frame_info(display_b, display_t, frame_num, ids_b, ids_t, matcher,
                   frame_matched, frame_stable, frame_smoothed):
    """Add informational overlays to frames."""
    # Broadcast frame info
    cv2.rectangle(display_b, (5, 5), (200, 120), (0, 0, 0), -1)
    info_lines = [
        f"Frame: {frame_num}",
        f"Detected: {len(ids_b)}",
        f"Stable: {frame_stable}",
        f"Smoothed: {frame_smoothed}",
        f"Raw: {frame_matched}"
    ]
    for i, line in enumerate(info_lines):
        cv2.putText(display_b, line, (10, 22 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Tacticam frame info
    cv2.rectangle(display_t, (5, 5), (180, 80), (0, 0, 0), -1)
    info_t = [
        "TACTICAM (Master)",
        f"Tracked: {len(ids_t)}",
        f"Total stable: {len(matcher.stable_matches)}"
    ]
    for i, line in enumerate(info_t):
        cv2.putText(display_t, line, (10, 22 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def resize_frames_to_same_height(display_b, display_t):
    """Resize frames to have the same height."""
    if display_b.shape[0] != display_t.shape[0]:
        target_h = max(display_b.shape[0], display_t.shape[0])
        if display_b.shape[0] != target_h:
            scale = target_h / display_b.shape[0]
            display_b = cv2.resize(display_b, None, fx=scale, fy=scale)
        if display_t.shape[0] != target_h:
            scale = target_h / display_t.shape[0]
            display_t = cv2.resize(display_t, None, fx=scale, fy=scale)

    return display_b, display_t


def process_video():
    """Main video processing function."""
    # Initialize components
    model, tracker_b, tracker_t, matcher, smoother = initialize_components()

    # Initialize video capture and output
    cap_b, cap_t, out, total_frames = initialize_video_capture()

    print(f"📹 Processing {total_frames} frames...")

    # Statistics
    stats = {'total_matched': 0, 'total_stable': 0, 'total_smoothed': 0}

    for frame_num in tqdm(range(total_frames)):
        ret_b, frame_b = cap_b.read()
        ret_t, frame_t = cap_t.read()

        if not ret_b or not ret_t:
            break

        display_b = frame_b.copy()
        display_t = frame_t.copy()

        # Detection
        res_b, res_t, pts_b, pts_t, idx_b, idx_t = process_frame_detection(model, frame_b, frame_t)

        # Tracking
        ids_b, ids_t, pos_b, pos_t, conf_ids_b, conf_ids_t = process_frame_tracking(tracker_b, tracker_t, pts_b, pts_t)

        # Build perspective-invariant graphs
        graph_b = PerspectiveInvariantGraph(pos_b, conf_ids_b)
        graph_t = PerspectiveInvariantGraph(pos_t, conf_ids_t)

        # Match using ordinal features
        raw_matches = process_frame_matching(graph_b, graph_t, matcher, smoother)

        # Draw graph connections (faint)
        draw_graph_connections(display_b, graph_b, alpha=GRAPH_CONNECTIONS_ALPHA)
        draw_graph_connections(display_t, graph_t, alpha=GRAPH_CONNECTIONS_ALPHA)

        # Draw Tacticam players
        draw_tacticam_players(display_t, ids_t, res_t, idx_t, tracker_t)

        # Draw Broadcast players with matched IDs
        frame_matched, frame_stable, frame_smoothed = draw_broadcast_players(
            display_b, ids_b, res_b, idx_b, matcher, smoother, tracker_b, raw_matches
        )

        stats['total_matched'] += frame_matched
        stats['total_stable'] += frame_stable
        stats['total_smoothed'] += frame_smoothed

        # Add information overlays
        add_frame_info(display_b, display_t, frame_num, ids_b, ids_t, matcher,
                       frame_matched, frame_stable, frame_smoothed)

        # Resize frames to same height and combine
        display_b, display_t = resize_frames_to_same_height(display_b, display_t)
        combined_frame = np.hstack((display_b, display_t))

        out.write(combined_frame)

    # Clean up
    cap_b.release()
    cap_t.release()
    out.release()

    print(f"\n Done! Saved to {OUTPUT_FILENAME}")
    print(f"\n Final Statistics:")
    print(f"   Total frames: {total_frames}")
    print(f"   Stable matches established: {len(matcher.stable_matches)}")
    print(f"   Avg stable/frame: {stats['total_stable']/total_frames:.2f}")
    print(f"   Avg smoothed/frame: {stats['total_smoothed']/total_frames:.2f}")
    print(f"   Avg raw/frame: {stats['total_matched']/total_frames:.2f}")


if __name__ == "__main__":
    process_video()
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from constants import (
    MODEL_PATH, BROADCAST_VIDEO, TACTICAM_VIDEO, OUTPUT_FILENAME,
    TARGET_CLASSES, TACTICAM_SKIP_SECONDS, BROADCAST_TRACKER_MAX_DIST,
    BROADCAST_TRACKER_MAX_MISSING, TACTICAM_TRACKER_MAX_DIST,
    TACTICAM_TRACKER_MAX_MISSING, SMOOTHING_WINDOW_SIZE, MATCH_ACCEPT_THRESHOLD,
    STABILITY_CHECK_X_DIFF, STABILITY_CHECK_Y_DIFF, GRAPH_CONNECTIONS_ALPHA, COLORS
)

def get_color(pid):
    return COLORS[pid % len(COLORS)]


def get_feet_points(boxes, classes):
    """Extract feet positions"""
    points, indices = [], []
    for i, box in enumerate(boxes):
        if int(classes[i]) in TARGET_CLASSES:
            x1, y1, x2, y2 = box
            points.append([(x1 + x2) / 2, y2])
            indices.append(i)
    return np.array(points) if points else np.array([]), indices


class PerspectiveInvariantGraph:
    """
    Graph with features designed to be robust to perspective distortion.

    Key insight: While absolute distances change with perspective,
    certain ORDINAL and TOPOLOGICAL properties are preserved:
    - Who is leftmost/rightmost
    - Who is nearest neighbor to whom
    - Relative ordering along field axis
    """

    def __init__(self, positions, ids):
        self.positions = np.array(positions) if len(positions) > 0 else np.array([])
        self.ids = list(ids)
        self.n = len(ids)

        if self.n < 2:
            self.features = {}
            return

        # Compute robust features
        self.features = self._compute_perspective_invariant_features()

    def _compute_perspective_invariant_features(self):
        """
        Compute features that survive perspective transformation.
        Focus on ORDINAL properties rather than metric ones.
        """
        features = {}

        # Sort players by X and Y coordinates
        x_order = np.argsort(self.positions[:, 0])
        y_order = np.argsort(self.positions[:, 1])

        # Create rank arrays
        x_ranks = np.zeros(self.n)
        y_ranks = np.zeros(self.n)
        for rank, idx in enumerate(x_order):
            x_ranks[idx] = rank / (self.n - 1) if self.n > 1 else 0.5
        for rank, idx in enumerate(y_order):
            y_ranks[idx] = rank / (self.n - 1) if self.n > 1 else 0.5

        # Distance matrix
        dist_matrix = cdist(self.positions, self.positions)

        for i, pid in enumerate(self.ids):
            # This is preserved under perspective!
            x_rank = x_ranks[i]  # 0 = leftmost, 1 = rightmost
            y_rank = y_ranks[i]  # 0 = topmost, 1 = bottommost

            # 2. NEAREST NEIGHBOR IDENTITY
            # Who are my k nearest neighbors? (order preserved under perspective)
            distances = dist_matrix[i].copy()
            distances[i] = np.inf
            nn_order = np.argsort(distances)[:min(5, self.n - 1)]
            nn_ids = [self.ids[j] for j in nn_order]

            # 3. NEIGHBOR RANKS
            # What are the ordinal positions of my neighbors?
            nn_x_ranks = [x_ranks[j] for j in nn_order]
            nn_y_ranks = [y_ranks[j] for j in nn_order]

            # 4. DIRECTIONAL NEIGHBORS
            # Who is to my left/right/above/below?
            left_neighbors = [self.ids[j] for j in range(self.n)
                             if self.positions[j, 0] < self.positions[i, 0]]
            right_neighbors = [self.ids[j] for j in range(self.n)
                              if self.positions[j, 0] > self.positions[i, 0]]
            above_neighbors = [self.ids[j] for j in range(self.n)
                              if self.positions[j, 1] < self.positions[i, 1]]
            below_neighbors = [self.ids[j] for j in range(self.n)
                              if self.positions[j, 1] > self.positions[i, 1]]

            # 5. QUADRANT (robust to scale)
            centroid = np.mean(self.positions, axis=0)
            quadrant = (2 if self.positions[i, 0] > centroid[0] else 0) + \
                      (1 if self.positions[i, 1] > centroid[1] else 0)

            # 6. RELATIVE POSITION SIGNATURE
            # Angle to each neighbor (somewhat preserved)
            angles_to_neighbors = []
            for j in nn_order[:3]:
                dx = self.positions[j, 0] - self.positions[i, 0]
                dy = self.positions[j, 1] - self.positions[i, 1]
                angles_to_neighbors.append(np.arctan2(dy, dx))

            # 7. DISTANCE RANK SIGNATURE
            # Instead of actual distances, use distance ranks
            dist_ranks = np.zeros(self.n)
            sorted_dists = np.argsort(distances)
            for rank, j in enumerate(sorted_dists):
                if j != i:
                    dist_ranks[j] = rank / (self.n - 1)

            features[pid] = {
                'x_rank': x_rank,
                'y_rank': y_rank,
                'nn_ids': nn_ids,
                'nn_x_ranks': nn_x_ranks,
                'nn_y_ranks': nn_y_ranks,
                'n_left': len(left_neighbors),
                'n_right': len(right_neighbors),
                'n_above': len(above_neighbors),
                'n_below': len(below_neighbors),
                'quadrant': quadrant,
                'angles': angles_to_neighbors,
                'position': self.positions[i].copy()
            }

        return features


class AdaptiveMatcher:
    """
    Matcher that adapts to perspective differences between views.
    Uses ordinal/topological features that are perspective-invariant.
    """

    def __init__(self):
        # Match confidence tracking
        self.match_scores = defaultdict(lambda: defaultdict(float))
        self.stable_matches = {}

        # Adaptive thresholds
        self.base_confidence = 2.5  # Quick to establish
        self.stable_confidence = 4.0  # Higher bar for stability

        # Perspective model
        self.perspective_model = None
        self.model_confidence = 0

        # Recent match history for momentum
        self.recent_matches = []  # Last N frames of matches
        self.max_recent = 10

    def compute_ordinal_similarity(self, feat_b, feat_t, known_matches):
        """
        Compare features using ordinal/topological properties.
        These are more robust to perspective distortion.
        """
        score = 0

        # Weight by number of players (more players = more reliable ranks)
        x_diff = abs(feat_b['x_rank'] - feat_t['x_rank'])
        y_diff = abs(feat_b['y_rank'] - feat_t['y_rank'])

        # Allow more Y difference (perspective affects Y more)
        position_score = x_diff * 40 + y_diff * 25
        score += position_score

        # How many players to left/right/above/below
        count_diff = (
            abs(feat_b['n_left'] - feat_t['n_left']) +
            abs(feat_b['n_right'] - feat_t['n_right']) +
            abs(feat_b['n_above'] - feat_t['n_above']) * 0.5 +  # Less weight on vertical
            abs(feat_b['n_below'] - feat_t['n_below']) * 0.5
        )
        score += count_diff * 8

        if feat_b['quadrant'] != feat_t['quadrant']:
            score += 15

        # Do my neighbors have similar relative positions?
        if feat_b['nn_x_ranks'] and feat_t['nn_x_ranks']:
            min_nn = min(len(feat_b['nn_x_ranks']), len(feat_t['nn_x_ranks']))
            for k in range(min_nn):
                x_nn_diff = abs(feat_b['nn_x_ranks'][k] - feat_t['nn_x_ranks'][k])
                y_nn_diff = abs(feat_b['nn_y_ranks'][k] - feat_t['nn_y_ranks'][k])
                score += (x_nn_diff * 15 + y_nn_diff * 10)

        # If my neighbor is already matched, that's strong evidence
        for k, nn_b in enumerate(feat_b['nn_ids'][:3]):
            if nn_b in known_matches:
                matched_t = known_matches[nn_b]
                if matched_t in feat_t['nn_ids'][:3]:
                    # Neighbor relationship preserved!
                    score -= 20
                    # Bonus if same neighbor rank
                    if feat_t['nn_ids'].index(matched_t) == k:
                        score -= 10

        return score

    def match(self, graph_b, graph_t):
        """
        Match players between views using perspective-invariant features.
        """
        if graph_b.n == 0 or graph_t.n == 0:
            return {}

        # Build cost matrix
        cost_matrix = np.zeros((graph_b.n, graph_t.n))

        # Get current known matches for neighbor bonus
        known_matches = {**self.stable_matches}

        for i, b_id in enumerate(graph_b.ids):
            feat_b = graph_b.features.get(b_id, {})

            for j, t_id in enumerate(graph_t.ids):
                feat_t = graph_t.features.get(t_id, {})

                if not feat_b or not feat_t:
                    cost_matrix[i, j] = 100
                    continue

                # Ordinal similarity
                cost = self.compute_ordinal_similarity(feat_b, feat_t, known_matches)

                # Historical match bonus
                history_score = self.match_scores[b_id][t_id]
                cost -= min(history_score * 12, 50)

                # Stable match super bonus
                if b_id in self.stable_matches and self.stable_matches[b_id] == t_id:
                    cost -= 80

                # Recent momentum bonus
                recent_count = sum(1 for m in self.recent_matches if m.get(b_id) == t_id)
                cost -= recent_count * 5

                cost_matrix[i, j] = cost

        # Hungarian matching
        row_idx, col_idx = linear_sum_assignment(cost_matrix)

        matches = {}
        for r, c in zip(row_idx, col_idx):
            if cost_matrix[r, c] < 80:  # Accept threshold
                b_id = graph_b.ids[r]
                t_id = graph_t.ids[c]
                matches[b_id] = t_id

        # Update scores based on matches
        self._update_scores(matches, graph_b.ids, graph_t.ids)

        # Update recent history
        self.recent_matches.append(matches.copy())
        if len(self.recent_matches) > self.max_recent:
            self.recent_matches.pop(0)

        return matches

    def _update_scores(self, matches, all_b_ids, all_t_ids):
        """Update match confidence scores"""

        for b_id in all_b_ids:
            if b_id in matches:
                t_id = matches[b_id]
                # Boost matched pair
                self.match_scores[b_id][t_id] += 1.0

                # Check for stability
                if self.match_scores[b_id][t_id] >= self.stable_confidence:
                    self.stable_matches[b_id] = t_id

                # Decay alternatives (but slowly)
                for other_t in all_t_ids:
                    if other_t != t_id:
                        self.match_scores[b_id][other_t] *= 0.85
            else:
                # Decay all scores for unmatched broadcast player
                for t_id in all_t_ids:
                    self.match_scores[b_id][t_id] *= 0.95

    def get_display_id(self, b_id, current_match=None):
        """Get display ID with stability indicator"""
        if b_id in self.stable_matches:
            return self.stable_matches[b_id], True
        elif current_match is not None:
            return current_match, False
        return None, False

    def verify_stable_matches(self, graph_b, graph_t):
        """
        Verify that stable matches still make sense.
        Remove matches that have become inconsistent.
        """
        to_remove = []

        for b_id, t_id in self.stable_matches.items():
            if b_id not in graph_b.features or t_id not in graph_t.features:
                continue

            feat_b = graph_b.features[b_id]
            feat_t = graph_t.features[t_id]

            # Check if position ranks are wildly different
            x_diff = abs(feat_b['x_rank'] - feat_t['x_rank'])
            y_diff = abs(feat_b['y_rank'] - feat_t['y_rank'])

            if x_diff > 0.4 or y_diff > 0.5:  # Moved too much
                # Penalize this match
                self.match_scores[b_id][t_id] -= 0.5
                if self.match_scores[b_id][t_id] < self.base_confidence:
                    to_remove.append(b_id)

        for b_id in to_remove:
            del self.stable_matches[b_id]

    def cleanup(self, active_b_ids):
        """Remove stale data"""
        stale = [b for b in self.stable_matches if b not in active_b_ids]
        for b in stale:
            del self.stable_matches[b]
            if b in self.match_scores:
                del self.match_scores[b]


class TemporalSmoother:
    """
    Smooth match assignments over time to reduce flickering.
    Uses voting over recent frames.
    """

    def __init__(self, window_size=7):
        self.window_size = window_size
        self.history = []  # List of (b_id -> t_id) dicts

    def add_frame(self, matches):
        """Add current frame's matches to history"""
        self.history.append(matches.copy())
        if len(self.history) > self.window_size:
            self.history.pop(0)

    def get_smoothed_match(self, b_id):
        """
        Get the most consistent match for b_id over recent frames.
        Returns (t_id, confidence) or (None, 0)
        """
        if not self.history:
            return None, 0

        # Count votes for each t_id
        votes = defaultdict(int)
        for frame_matches in self.history:
            if b_id in frame_matches:
                votes[frame_matches[b_id]] += 1

        if not votes:
            return None, 0

        best_t_id = max(votes.keys(), key=lambda t: votes[t])
        confidence = votes[best_t_id] / len(self.history)

        return best_t_id, confidence


class RobustTracker:
    """Simple but robust tracker using Kalman-like prediction"""

    def __init__(self, max_dist=100, max_missing=20):
        self.next_id = 0
        self.tracks = {}
        self.max_dist = max_dist
        self.max_missing = max_missing

    def update(self, detections):
        detections = [np.array(d) for d in detections] if detections else []

        # Predict
        for tid, tr in self.tracks.items():
            tr['age'] += 1
            tr['time_since_update'] += 1
            if len(tr['history']) >= 2:
                vel = tr['history'][-1] - tr['history'][-2]
                tr['predicted'] = tr['position'] + vel * 0.8
            else:
                tr['predicted'] = tr['position']

        if len(detections) == 0:
            self._cleanup()
            return []

        if len(self.tracks) == 0:
            return self._init_tracks(detections)

        # Match
        track_ids = list(self.tracks.keys())
        cost = np.zeros((len(track_ids), len(detections)))

        for i, tid in enumerate(track_ids):
            pred = self.tracks[tid]['predicted']
            for j, det in enumerate(detections):
                cost[i, j] = np.linalg.norm(pred - det)

        row_idx, col_idx = linear_sum_assignment(cost)

        assigned = [-1] * len(detections)

        for r, c in zip(row_idx, col_idx):
            if cost[r, c] < self.max_dist:
                tid = track_ids[r]
                self._update_track(tid, detections[c])
                assigned[c] = tid

        # New tracks
        for j, det in enumerate(detections):
            if assigned[j] == -1:
                assigned[j] = self._create_track(det)

        self._cleanup()
        return assigned

    def _init_tracks(self, detections):
        ids = []
        for det in detections:
            ids.append(self._create_track(det))
        return ids

    def _create_track(self, pos):
        tid = self.next_id
        self.tracks[tid] = {
            'position': pos,
            'predicted': pos.copy(),
            'history': [pos.copy()],
            'age': 0,
            'time_since_update': 0,
            'hits': 1
        }
        self.next_id += 1
        return tid

    def _update_track(self, tid, pos):
        tr = self.tracks[tid]
        tr['position'] = pos
        tr['history'].append(pos.copy())
        if len(tr['history']) > 30:
            tr['history'].pop(0)
        tr['time_since_update'] = 0
        tr['hits'] += 1

    def _cleanup(self):
        to_del = [t for t, tr in self.tracks.items()
                  if tr['time_since_update'] > self.max_missing]
        for t in to_del:
            del self.tracks[t]

    def get_confirmed(self, min_hits=2):
        """Get positions and IDs of confirmed tracks"""
        pos, ids = [], []
        for tid, tr in self.tracks.items():
            if tr['hits'] >= min_hits:
                pos.append(tr['position'])
                ids.append(tid)
        return np.array(pos) if pos else np.array([]), ids


def draw_player_box(frame, box, label, color, is_stable=False):
    """Draw player with box and label"""
    x1, y1, x2, y2 = map(int, box)

    thickness = 3 if is_stable else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Stability indicator
    if is_stable:
        cv2.circle(frame, (x2 - 8, y1 + 8), 5, (0, 255, 0), -1)

    # Label background
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, y1 - h - 8), (x1 + w + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def draw_graph_connections(frame, graph, alpha=0.3):
    """Draw faint connections between nearby players"""
    if graph.n < 2:
        return

    overlay = frame.copy()

    # Draw connections to nearest 2 neighbors
    for i, pid in enumerate(graph.ids):
        feat = graph.features.get(pid, {})
        if 'nn_ids' not in feat:
            continue

        pt1 = tuple(graph.positions[i].astype(int))

        for nn_id in feat['nn_ids'][:2]:
            if nn_id in graph.ids:
                j = graph.ids.index(nn_id)
                pt2 = tuple(graph.positions[j].astype(int))
                cv2.line(overlay, pt1, pt2, (100, 100, 100), 1)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
"""Unit tests for anonymous face tracking using synthetic bounding boxes.

Ensures deterministic tracking, ID continuity, multi-face handling,
temporary occlusion recovery, and timeout pruning without requiring a webcam.
"""

import pytest

from attendance_agent.vision.detector import FaceBoundingBox, RawDetection
from attendance_agent.vision.tracker import AnonymousFaceTracker


def make_detection(
    x: int,
    y: int,
    w: int = 100,
    h: int = 100,
    confidence: float = 0.9,
    timestamp: float = 0.0,
) -> RawDetection:
    """Helper to construct synthetic face detection."""
    bbox = FaceBoundingBox(x=x, y=y, w=w, h=h, norm_x=x / 640.0, norm_y=y / 480.0, norm_w=w / 640.0, norm_h=h / 480.0)
    return RawDetection(bbox=bbox, confidence=confidence, timestamp=timestamp)


def test_iou_calculation() -> None:
    """Verify IoU calculation on identical, overlapping, and disjoint boxes."""
    box_a = FaceBoundingBox(x=100, y=100, w=100, h=100)
    box_identical = FaceBoundingBox(x=100, y=100, w=100, h=100)
    box_disjoint = FaceBoundingBox(x=300, y=300, w=100, h=100)
    box_half = FaceBoundingBox(x=150, y=100, w=100, h=100)

    # Identical overlap = 1.0
    assert box_a.iou(box_identical) == pytest.approx(1.0)
    # Disjoint overlap = 0.0
    assert box_a.iou(box_disjoint) == pytest.approx(0.0)
    # 50 pixel horizontal overlap: intersection = 50*100 = 5000, union = 10000 + 10000 - 5000 = 15000 -> 1/3
    assert box_a.iou(box_half) == pytest.approx(5000.0 / 15000.0, rel=1e-3)


def test_single_face_tracking_continuity() -> None:
    """Verify that a moving face retains its ID across consecutive frames."""
    tracker = AnonymousFaceTracker(iou_threshold=0.3, max_inactive_frames=5)

    # Frame 1: Face at (100, 100)
    res1 = tracker.update([make_detection(100, 100, timestamp=0.0)], timestamp=0.0)
    assert res1.active_count == 1
    track_id = res1.active_tracks[0].track_id
    assert track_id == 1
    assert res1.active_tracks[0].status == "active"

    # Frame 2: Face moves slightly to (105, 102)
    res2 = tracker.update([make_detection(105, 102, timestamp=0.1)], timestamp=0.1)
    assert res2.active_count == 1
    assert res2.active_tracks[0].track_id == track_id
    assert res2.active_tracks[0].total_detections == 2
    assert res2.active_tracks[0].movement_distance > 0.0

    # Frame 3: Face moves to (110, 105)
    res3 = tracker.update([make_detection(110, 105, timestamp=0.2)], timestamp=0.2)
    assert res3.active_count == 1
    assert res3.active_tracks[0].track_id == track_id
    assert res3.active_tracks[0].total_detections == 3


def test_multiple_faces_tracking() -> None:
    """Verify multiple faces are assigned distinct IDs and tracked simultaneously."""
    tracker = AnonymousFaceTracker(iou_threshold=0.3)

    # Frame 1: Three distinct faces in classroom
    dets_f1 = [
        make_detection(x=50, y=100),
        make_detection(x=250, y=100),
        make_detection(x=450, y=100),
    ]
    res1 = tracker.update(dets_f1, timestamp=1.0)
    assert res1.active_count == 3
    ids = {t.track_id for t in res1.active_tracks}
    assert len(ids) == 3

    # Frame 2: Faces undergo small independent movements
    dets_f2 = [
        make_detection(x=52, y=102),
        make_detection(x=248, y=98),
        make_detection(x=455, y=101),
    ]
    res2 = tracker.update(dets_f2, timestamp=1.1)
    assert res2.active_count == 3
    ids_f2 = {t.track_id for t in res2.active_tracks}
    assert ids == ids_f2  # Same IDs preserved


def test_temporary_disappearance_and_recovery() -> None:
    """Verify track switches to temporary_lost during occlusion and recovers same ID."""
    tracker = AnonymousFaceTracker(iou_threshold=0.3, max_inactive_frames=3, max_disappearance_seconds=2.0)

    # Frame 1: Face detected
    tracker.update([make_detection(100, 100, timestamp=0.0)], timestamp=0.0)

    # Frame 2: Face temporarily obscured (0 detections)
    res2 = tracker.update([], timestamp=0.5)
    assert res2.active_count == 0
    assert len(res2.lost_tracks) == 1
    assert res2.lost_tracks[0].track_id == 1
    assert res2.lost_tracks[0].status == "temporary_lost"

    # Frame 3: Face reappears at same position
    res3 = tracker.update([make_detection(102, 101, timestamp=1.0)], timestamp=1.0)
    assert res3.active_count == 1
    assert res3.active_tracks[0].track_id == 1  # Reclaimed original ID
    assert res3.active_tracks[0].status == "active"


def test_inactive_timeout_and_purging() -> None:
    """Verify tracks transition to inactive after timeout and are purged."""
    tracker = AnonymousFaceTracker(iou_threshold=0.3, max_inactive_frames=2, max_disappearance_seconds=1.0)

    # Frame 1: Face detected
    tracker.update([make_detection(100, 100, timestamp=0.0)], timestamp=0.0)

    # Frames 2 & 3: Missing (temporary_lost)
    tracker.update([], timestamp=0.5)
    tracker.update([], timestamp=1.0)

    # Frame 4: Missing past threshold -> marked inactive
    res4 = tracker.update([], timestamp=2.5)
    assert res4.active_count == 0
    inactive_tracks = [t for t in res4.all_tracks if t.status == "inactive"]
    assert len(inactive_tracks) == 1

    # Frames 5 to 10: Prolonged absence -> purged
    for i in range(5, 12):
        tracker.update([], timestamp=float(i))

    assert len(tracker.get_tracks()) == 0


def test_new_participant_entry() -> None:
    """Verify new participant entering scene receives a new sequential ID."""
    tracker = AnonymousFaceTracker()

    # Frame 1: One student
    res1 = tracker.update([make_detection(100, 100, timestamp=0.0)], timestamp=0.0)
    assert res1.active_count == 1
    assert res1.active_tracks[0].track_id == 1

    # Frame 2: Second student arrives
    res2 = tracker.update(
        [
            make_detection(102, 101, timestamp=0.1),
            make_detection(350, 150, timestamp=0.1),
        ],
        timestamp=0.1,
    )
    assert res2.active_count == 2
    track_ids = {t.track_id for t in res2.active_tracks}
    assert track_ids == {1, 2}


def test_tracker_reset() -> None:
    """Verify tracker reset completely clears active state."""
    tracker = AnonymousFaceTracker()
    tracker.update([make_detection(100, 100)], timestamp=0.0)
    assert len(tracker.get_tracks()) == 1

    tracker.reset()
    assert len(tracker.get_tracks()) == 0
    assert tracker.next_track_id == 1


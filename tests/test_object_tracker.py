from elec5308_autonomy.perception.object_tracker import ObjectTracker
from elec5308_autonomy.types import ObjectDetectionResult


def test_tracker_assigns_consistent_id_to_same_object_across_frames():
    """An object that moves slightly between frames should keep the same ID."""
    tracker = ObjectTracker()

    frame_1 = ObjectDetectionResult(
        objects=["car"],
        bounding_boxes=[[100, 100, 150, 150]],
        confidence=[0.9],
    )
    frame_2 = ObjectDetectionResult(
        objects=["car"],
        bounding_boxes=[[105, 102, 155, 152]],  # moved slightly
        confidence=[0.88],
    )

    ids_frame_1 = tracker.update(frame_1)
    ids_frame_2 = tracker.update(frame_2)

    assert ids_frame_1 == ids_frame_2  # same object, same ID


def test_tracker_assigns_new_id_to_new_object():
    """A newly appearing object should get a different ID than existing ones."""
    tracker = ObjectTracker()

    frame_1 = ObjectDetectionResult(
        objects=["car"],
        bounding_boxes=[[100, 100, 150, 150]],
        confidence=[0.9],
    )
    frame_2 = ObjectDetectionResult(
        objects=["car", "person"],
        bounding_boxes=[[102, 101, 152, 151], [400, 400, 420, 450]],  # same car + new person, far away
        confidence=[0.9, 0.85],
    )

    ids_frame_1 = tracker.update(frame_1)
    ids_frame_2 = tracker.update(frame_2)

    assert ids_frame_2[0] == ids_frame_1[0]  # the car kept its ID
    assert ids_frame_2[1] != ids_frame_1[0]  # the new person got a different ID


def test_tracker_keeps_id_through_brief_missed_detection():
    """An object that disappears for one frame (but reappears nearby) should
    keep its original ID, not get treated as a new object."""
    tracker = ObjectTracker(max_missed_frames=3)

    seen = ObjectDetectionResult(
        objects=["traffic light"],
        bounding_boxes=[[100, 100, 150, 150]],
        confidence=[0.77],
    )
    missed = ObjectDetectionResult(objects=[], bounding_boxes=[], confidence=[])
    seen_again = ObjectDetectionResult(
        objects=["traffic light"],
        bounding_boxes=[[102, 101, 151, 149]],  # roughly same spot
        confidence=[0.6],
    )

    id_1 = tracker.update(seen)
    tracker.update(missed)  # dropout frame, nothing detected
    id_2 = tracker.update(seen_again)

    assert id_1 == id_2  # same object, same ID despite the gap
def test_tracker_does_not_reassign_id_when_spurious_detection_appears():
    """A false-positive detection should get its own new ID rather than
    stealing the ID of an object that is still visible nearby."""
    tracker = ObjectTracker()

    frame_1 = ObjectDetectionResult(
        objects=["person"],
        bounding_boxes=[[330, 246, 342, 285]],
        confidence=[0.98],
    )
    frame_2 = ObjectDetectionResult(
        objects=["person", "boat"],
        bounding_boxes=[[331, 243, 342, 281], [361, 267, 369, 275]],
        confidence=[0.99, 0.74],
    )

    ids_1 = tracker.update(frame_1)
    ids_2 = tracker.update(frame_2)

    assert ids_2[0] == ids_1[0]   # person keeps its ID
    assert ids_2[1] != ids_1[0]   # spurious detection gets a different one

def test_tracker_does_not_match_across_different_classes():
    """A detection should never inherit the ID of a track with a different
    label, even if it appears in almost the same position."""
    tracker = ObjectTracker()

    frame_1 = ObjectDetectionResult(
        objects=["person"],
        bounding_boxes=[[330, 246, 342, 285]],
        confidence=[0.98],
    )
    frame_2 = ObjectDetectionResult(
        objects=["boat"],
        bounding_boxes=[[331, 247, 343, 286]],  # nearly identical position
        confidence=[0.81],
    )

    person_ids = tracker.update(frame_1)
    boat_ids = tracker.update(frame_2)

    assert boat_ids[0] != person_ids[0]
from scipy.optimize import linear_sum_assignment

from ..types import ObjectDetectionResult


class ObjectTracker:
    """Assigns consistent IDs to detected objects across consecutive frames.

    Matches this frame's detections to previously tracked objects by comparing
    the centre points of their bounding boxes. Matching is done with the
    Hungarian algorithm, which finds the assignment that minimises total
    distance across all pairs at once. A simpler greedy approach (take each
    detection in turn and grab its nearest free track) can swap IDs between
    two objects when they are similar distances apart, because the result
    then depends on the order the detector happened to return them in.

    A detection is only ever matched to a track of the same class, so a
    "person" track cannot be inherited by a "boat" detection that happens to
    appear nearby.

    Objects that go undetected for a few consecutive frames (a momentary
    confidence dip, or brief occlusion) are kept alive for up to
    `max_missed_frames` before being forgotten, rather than being treated as
    brand-new objects the instant they reappear.
    """

    def __init__(self, max_distance: float = 50.0, max_missed_frames: int = 3):
        """
        Args:
            max_distance: Maximum pixel distance between frames for two
                detections to be considered the same object.
            max_missed_frames: How many consecutive frames an object can go
                undetected before it is forgotten and its ID is retired.
        """
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.next_id = 0
        # obj_id -> {"centroid": (x, y), "label": str, "missed": int}
        self.tracked_objects = {}

    def _centroid(self, box):
        """Return the centre point (x, y) of a [x1, y1, x2, y2] bounding box."""
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _distance(self, p1, p2):
        """Straight-line distance between two (x, y) points."""
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def update(self, detection_result: ObjectDetectionResult) -> list[int]:
        """Given this frame's detections, return a track ID for each one.

        Args:
            detection_result: The current frame's ObjectDetectionResult.

        Returns:
            A list of track IDs, one per detected object, in the same order
            as detection_result.bounding_boxes.
        """
        new_centroids = [self._centroid(box) for box in detection_result.bounding_boxes]
        new_labels = list(detection_result.objects)

        # Nothing detected this frame: every existing track counts as missed.
        if not new_centroids:
            self._age_unmatched_tracks(set())
            return []

        existing_ids = list(self.tracked_objects.keys())

        # No existing tracks: everything in this frame is new.
        if not existing_ids:
            return self._assign_new_ids(new_centroids, new_labels)

        n_detections = len(new_centroids)
        n_tracks = len(existing_ids)

        # Cost matrix: rows are this frame's detections. The first n_tracks
        # columns are existing tracks; the remaining n_detections columns are
        # "no match" slots priced at max_distance.
        #
        # Those dummy columns matter. Without them linear_sum_assignment is
        # forced to hand out every track to some detection, so a spurious
        # detection can steal a track from the object that actually belongs to
        # it whenever that lowers the total cost across all pairs. Pairs that
        # are beyond max_distance, or that mismatch on class, are priced at
        # LARGE so the optimiser never prefers them over a dummy column.
        LARGE = 1e6
        cost_matrix = []
        for centroid, label in zip(new_centroids, new_labels):
            row = []
            for obj_id in existing_ids:
                track = self.tracked_objects[obj_id]
                dist = self._distance(centroid, track["centroid"])
                if track.get("label") != label or dist > self.max_distance:
                    row.append(LARGE)
                else:
                    row.append(dist)
            # One dummy column per detection, so every detection can opt out.
            row.extend([self.max_distance] * n_detections)
            cost_matrix.append(row)

        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Start by assuming every detection is new, then fill in the ones that
        # were matched to a real track within max_distance.
        assigned_ids = [None] * n_detections
        matched_ids = set()

        for row, col in zip(row_indices, col_indices):
            # Anything landing in a dummy column, or on a too-far or
            # differently-classed track, is new.
            if col < n_tracks and cost_matrix[row][col] <= self.max_distance:
                obj_id = existing_ids[col]
                assigned_ids[row] = obj_id
                matched_ids.add(obj_id)
                self.tracked_objects[obj_id] = {
                    "centroid": new_centroids[row],
                    "label": new_labels[row],
                    "missed": 0,
                }

        # Any detection not matched to an existing track is a new object.
        for i, obj_id in enumerate(assigned_ids):
            if obj_id is None:
                new_id = self.next_id
                self.next_id += 1
                assigned_ids[i] = new_id
                matched_ids.add(new_id)
                self.tracked_objects[new_id] = {
                    "centroid": new_centroids[i],
                    "label": new_labels[i],
                    "missed": 0,
                }

        self._age_unmatched_tracks(matched_ids)
        return assigned_ids

    def _assign_new_ids(self, centroids, labels):
        """Give every centroid a fresh ID (used when nothing is being tracked yet)."""
        assigned_ids = []
        for centroid, label in zip(centroids, labels):
            new_id = self.next_id
            self.next_id += 1
            assigned_ids.append(new_id)
            self.tracked_objects[new_id] = {
                "centroid": centroid,
                "label": label,
                "missed": 0,
            }
        return assigned_ids

    def _age_unmatched_tracks(self, matched_ids):
        """Increment the missed counter on tracks not seen this frame, and drop
        any that have been missing for longer than max_missed_frames."""
        for obj_id in list(self.tracked_objects.keys()):
            if obj_id not in matched_ids:
                self.tracked_objects[obj_id]["missed"] += 1
                if self.tracked_objects[obj_id]["missed"] > self.max_missed_frames:
                    del self.tracked_objects[obj_id]
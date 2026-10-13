from ..types import LaneDetectionResult, NavigationInfo


class PathFollower:
    """Generate a simple centre path from detected lane boundaries."""

    def follow(self, lane_result: LaneDetectionResult) -> NavigationInfo:
        """Generate navigation information from detected lane boundaries."""

        if not lane_result.left_lane or not lane_result.right_lane:
            raise ValueError("Both left and right lane boundaries are required.")

        left_lane = lane_result.left_lane[0]
        right_lane = lane_result.right_lane[0]

        if len(left_lane) != 4 or len(right_lane) != 4:
            raise ValueError("Each lane boundary must be [x1, y1, x2, y2].")

        lx1, ly1, lx2, ly2 = left_lane
        rx1, ry1, rx2, ry2 = right_lane

        left_points = [(lx1, ly1), (lx2, ly2)]
        right_points = [(rx1, ry1), (rx2, ry2)]

        left_far = min(left_points, key=lambda point: point[1])
        left_near = max(left_points, key=lambda point: point[1])

        right_far = min(right_points, key=lambda point: point[1])
        right_near = max(right_points, key=lambda point: point[1])

        near_centre = (
            (left_near[0] + right_near[0]) / 2,
            (left_near[1] + right_near[1]) / 2,
        )

        far_centre = (
            (left_far[0] + right_far[0]) / 2,
            (left_far[1] + right_far[1]) / 2,
        )

        centre_path = [near_centre, far_centre]
        target_point = far_centre

        return NavigationInfo(
            path=centre_path,
            target_point=target_point,
        )
    
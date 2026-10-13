import math

from ..types import NavigationInfo, SteeringCommand


class SteeringController:
    """Calculate steering from path heading and lateral lane position."""

    def __init__(
        self,
        image_width: float,
        gain: float = 0.35,
        lateral_gain: float = 0.14,
        smoothing: float = 0.7,
        max_steer: float = 0.5,
    ):
        """
        Args:
            image_width: Camera image width in pixels.
            gain: Scaling applied to the centre-path heading error.
            lateral_gain: Scaling applied to lateral lane-centre error.
            smoothing: Previous-command weight in the range [0.0, 1.0).
            max_steer: Maximum steering magnitude in radians.

        The navigation path is expected to use image coordinates:
        x increases to the right and y increases downward.
        """
        if image_width <= 0:
            raise ValueError("image_width must be positive.")

        if not (0.0 <= smoothing < 1.0):
            raise ValueError("smoothing must be in the range [0.0, 1.0).")

        if max_steer <= 0.0:
            raise ValueError("max_steer must be positive.")

        self.image_center_x = image_width / 2.0
        self.gain = gain
        self.lateral_gain = lateral_gain
        self.smoothing = smoothing
        self.max_steer = max_steer
        self.previous_angle = 0.0

    def reset(self) -> None:
        """Reset the steering filter state."""
        self.previous_angle = 0.0

    def calculate_steering(
        self, navigation_info: NavigationInfo
    ) -> SteeringCommand:
        """Calculate a smoothed steering command from heading and lateral error."""

        if not navigation_info.path or len(navigation_info.path) < 2:
            raise ValueError("At least two path points are required.")

        near_x, near_y = navigation_info.path[0]
        far_x, far_y = navigation_info.path[1]

        dx = far_x - near_x
        dy = near_y - far_y

        if dx == 0 and dy == 0:
            raise ValueError("Path points must not be identical.")

        # Direction the road centreline is pointing.
        heading_correction = math.atan2(dx, dy) * self.gain

        # Position of the road centre relative to the camera/vehicle centre.
        lateral_error = (
            near_x - self.image_center_x
        ) / self.image_center_x

        lateral_correction = self.lateral_gain * lateral_error

        raw_angle = heading_correction + lateral_correction

        filtered_angle = (
            self.smoothing * self.previous_angle
            + (1.0 - self.smoothing) * raw_angle
        )

        steering_angle = max(
            -self.max_steer,
            min(self.max_steer, filtered_angle),
        )

        self.previous_angle = steering_angle

        return SteeringCommand(
            steering_angle=steering_angle,
        )
from ..types import NavigationInfo, SpeedCommand


class SpeedController:
    """Calculate a speed command from navigation path information."""

    def calculate_speed(
        self, navigation_info: NavigationInfo
    ) -> SpeedCommand:
        """Reduce speed when the navigation path requires a sharper turn."""

        if not navigation_info.path or len(navigation_info.path) < 2:
            raise ValueError("At least two path points are required.")

        near_point = navigation_info.path[0]
        far_point = navigation_info.path[1]

        near_x, near_y = near_point
        far_x, far_y = far_point

        dx = abs(far_x - near_x)
        dy = abs(near_y - far_y)

        if dx == 0 and dy == 0:
            raise ValueError("Path points must not be identical.")

        turn_ratio = dx / max(dy, 1e-6)

        if turn_ratio < 0.15:
            speed = 1.0
        elif turn_ratio < 0.35:
            speed = 0.7
        else:
            speed = 0.4

        return SpeedCommand(
            speed=speed,
        )
import pytest

from elec5308_autonomy.navigation.path_following import PathFollower
from elec5308_autonomy.navigation.speed_control import SpeedController
from elec5308_autonomy.navigation.steering import SteeringController
from elec5308_autonomy.types import LaneDetectionResult, NavigationInfo


def test_path_follower_generates_lane_centre_path():
    """Path follower should generate the midpoint between both lane boundaries."""
    follower = PathFollower()

    lane_result = LaneDetectionResult(
        left_lane=[[100, 300, 150, 100]],
        right_lane=[[300, 300, 250, 100]],
        driving_area=None,
    )

    result = follower.follow(lane_result)

    assert result.path == [(200.0, 300.0), (200.0, 100.0)]
    assert result.target_point == (200.0, 100.0)


def test_path_follower_rejects_missing_lane_boundary():
    """Path follower should reject input when either lane boundary is missing."""
    follower = PathFollower()

    lane_result = LaneDetectionResult(
        left_lane=[[100, 300, 150, 100]],
        right_lane=[],
        driving_area=None,
    )

    with pytest.raises(ValueError):
        follower.follow(lane_result)


def test_straight_centred_path_produces_zero_steering():
    """A straight path through the image centre should require no steering."""
    controller = SteeringController(
        image_width=400,
        smoothing=0.0,
    )

    navigation_info = NavigationInfo(
        path=[(200.0, 300.0), (200.0, 100.0)],
        target_point=(200.0, 100.0),
    )

    command = controller.calculate_steering(navigation_info)

    assert command.steering_angle == pytest.approx(0.0)


def test_lateral_path_offset_changes_steering_direction():
    """Paths offset left and right should produce steering in opposite directions."""
    left_controller = SteeringController(
        image_width=400,
        smoothing=0.0,
    )
    right_controller = SteeringController(
        image_width=400,
        smoothing=0.0,
    )

    left_path = NavigationInfo(
        path=[(150.0, 300.0), (150.0, 100.0)],
        target_point=(150.0, 100.0),
    )

    right_path = NavigationInfo(
        path=[(250.0, 300.0), (250.0, 100.0)],
        target_point=(250.0, 100.0),
    )

    left_command = left_controller.calculate_steering(left_path)
    right_command = right_controller.calculate_steering(right_path)

    assert left_command.steering_angle < 0
    assert right_command.steering_angle > 0


def test_steering_respects_maximum_limit():
    """Steering output should never exceed the configured maximum magnitude."""
    controller = SteeringController(
        image_width=400,
        gain=10.0,
        smoothing=0.0,
        max_steer=0.5,
    )

    navigation_info = NavigationInfo(
        path=[(200.0, 300.0), (400.0, 100.0)],
        target_point=(400.0, 100.0),
    )

    command = controller.calculate_steering(navigation_info)

    assert abs(command.steering_angle) <= 0.5


def test_speed_decreases_as_turn_becomes_sharper():
    """Speed controller should command lower speeds for sharper paths."""
    controller = SpeedController()

    straight = NavigationInfo(
        path=[(200.0, 300.0), (200.0, 100.0)],
        target_point=(200.0, 100.0),
    )

    moderate_turn = NavigationInfo(
        path=[(200.0, 300.0), (250.0, 100.0)],
        target_point=(250.0, 100.0),
    )

    sharp_turn = NavigationInfo(
        path=[(200.0, 300.0), (300.0, 100.0)],
        target_point=(300.0, 100.0),
    )

    straight_speed = controller.calculate_speed(straight).speed
    moderate_speed = controller.calculate_speed(moderate_turn).speed
    sharp_speed = controller.calculate_speed(sharp_turn).speed

    assert straight_speed == 1.0
    assert moderate_speed == 0.7
    assert sharp_speed == 0.4
    
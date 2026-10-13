from ..types import (  # Import the control command data structures
    SpeedCommand,
    SteeringCommand,
)


class SimulatorAdapter:  # Define an interface for connecting the library to a simulator
    """Interface for connecting the autonomy library to a simulator."""  # Describe the purpose of this class

    def apply_control(
        self,
        steering: SteeringCommand,
        speed: SpeedCommand,
    ):  # Define the method for sending control commands to the simulator
        """Apply steering and speed commands to the simulator."""  # Describe what the method does
        raise NotImplementedError  # Indicate that the actual simulator implementation is not provided yet
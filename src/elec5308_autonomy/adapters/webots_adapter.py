from ..types import SpeedCommand, SteeringCommand
from .simulator import SimulatorAdapter


class WebotsAdapter(SimulatorAdapter):

    def __init__(self):
        from vehicle import Driver

        self.driver = Driver()

        self.camera = self.driver.getDevice("camera")
        self.timestep = int(self.driver.getBasicTimeStep())
        self.camera.enable(self.timestep)

    def apply_control(
        self,
        steering: SteeringCommand,
        speed: SpeedCommand,
    ):
        self.driver.setSteeringAngle(
            float(steering.steering_angle)
        )

        self.driver.setCruisingSpeed(
            float(speed.speed)
        )

    def get_camera_image(self):
        return self.camera.getImage()
    
    def step(self):
        return self.driver.step()
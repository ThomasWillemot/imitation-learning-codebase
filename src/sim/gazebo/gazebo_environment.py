
from src.sim.common.environment import EnvironmentConfig, Environment


class GazeboEnvironment(Environment):

    def __init__(self, config: EnvironmentConfig):
        super().__init__(config)
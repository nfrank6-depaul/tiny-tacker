from gym_sailing.envs.motorboat_env import MotorboatEnv
from gym_sailing.envs.sailboat_env import SailboatEnv
from gym_sailing.envs.sailboat_env import SailboatDiscreteEnv
from gym_sailing.envs.boat_env import BoatEnv
from gym_sailing.envs.boat_env import BoatDiscreteEnv
from gymnasium.envs.registration import register

register(
    id="SailboatOlympicTriangle-v0",
    entry_point="gym_sailing.envs:SailboatOlympicTriangleEnv",
)

register(
    id="SailboatOlympicTriangleDiscrete-v0",
    entry_point="gym_sailing.envs:SailboatOlympicTriangleDiscreteEnv",
)
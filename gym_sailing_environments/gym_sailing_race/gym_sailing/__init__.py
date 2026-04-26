from gymnasium.envs.registration import register

register(
    id="SailboatOlympicTriangle-v0",
    entry_point="gym_sailing.envs:SailboatOlympicTriangleEnv",
)

register(
    id="SailboatOlympicTriangleDiscrete-v0",
    entry_point="gym_sailing.envs:SailboatOlympicTriangleDiscreteEnv",
)
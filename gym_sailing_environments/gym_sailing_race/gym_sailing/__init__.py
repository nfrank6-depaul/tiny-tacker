from gymnasium.envs.registration import register



register(
    id="SailboatOlympicTriangle-v0",
    entry_point="gym_sailing.envs:SailboatOlympicTriangle",
    max_episode_steps=3000,
)

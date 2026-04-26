import os
import sys
import time
import builtins
import numpy as np
import pandas as pd
import gymnasium as gym
import pygame

builtins.quit = lambda *args, **kwargs: None




# Resolve repo root from this script location:
# tiny-tackers/tiny_tackers/human_race_tt.py -> tiny-tackers/
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(REPO_ROOT)

GYM_SAILING_PARENT = os.path.join(
    REPO_ROOT,
    "gym_sailing_environments",
    "gym_sailing_race",
)
GYM_SAILING_PACKAGE = os.path.join(GYM_SAILING_PARENT, "gym_sailing")

if not os.path.isdir(GYM_SAILING_PACKAGE):
    raise FileNotFoundError(
        "Could not find the local gym_sailing package.\n"
        f"Expected package folder at: {GYM_SAILING_PACKAGE}\n"
        f"Current working directory: {os.getcwd()}\n"
        "Confirm that the repo contains: "
        "gym_sailing_environments/gym_sailing_race/gym_sailing"
    )

sys.path.insert(0, GYM_SAILING_PARENT)
# type ignore below so I stop seeing import error in VS code. package is here, but is deined by path above.
import gym_sailing  # type: ignore
import gym_sailing.envs.sailboat_env as sailboat_module  # type: ignore
import gym_sailing.envs.boat_env as boat_module  # type: ignore
import gym_sailing.utils.renderer as renderer_module  # type: ignore

print("Loaded gym_sailing from:", gym_sailing.__file__)
print("Loaded sailboat_env from:", sailboat_module.__file__)
print("Loaded boat_env from:", boat_module.__file__)
print("Loaded renderer from:", renderer_module.__file__)
ENV_ID = "SailboatOlympicTriangle-v0"
MAX_STEPS = 10_000
RESULTS_DIR = "data/results"
RESULTS_PATH = os.path.join(RESULTS_DIR, "human_race_score.csv")

import sys

print("\n--- MODULE CACHE CHECK ---")

for name, module in sys.modules.items():

    if "gym_sailing" in name:

        print(name, "->", getattr(module, "__file__", "built-in"))


def get_human_action(env):
    pygame.event.pump()
    keys = pygame.key.get_pressed()

    steer = 0.0

    if keys[pygame.K_LEFT]:
        steer = -1.0
    elif keys[pygame.K_RIGHT]:
        steer = 1.0

    action = np.array([steer], dtype=np.float32)
    return np.clip(action, env.action_space.low, env.action_space.high)


def run_human_episode():
    pygame.init()
    env = gym.make(ENV_ID, render_mode="human")
    print("Created environment:", env.unwrapped.__class__)
    print("Reset method file:", env.unwrapped.reset.__code__.co_filename)
    if hasattr(env.unwrapped, "MARKS"):
        print("Race marks:", env.unwrapped.MARKS)
    else:
        print("WARNING: This environment does not expose MARKS; it may not be the race environment.")
    obs, info = env.reset()

    total_reward = 0.0
    timesteps = 0
    running = True

    print("Human controls:")
    print("LEFT arrow = steer left")
    print("RIGHT arrow = steer right")
    print("SPACE = start")
    print("ESC = quit episode")

    waiting = True
    time.sleep(0.5)

    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    waiting = False
                elif event.key == pygame.K_ESCAPE:
                    waiting = False
                    running = False

        time.sleep(0.01)

    try:
        while running and timesteps < MAX_STEPS:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            action = get_human_action(env)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            timesteps += 1

            if terminated or truncated:
                break

            time.sleep(0.03)

    finally:
        try:
            env.close()
        except Exception:
            pass

        try:
            pygame.display.quit()
            pygame.quit()
        except Exception:
            pass

    return {
        "agent": "Human Race",
        "env_id": ENV_ID,
        "total_reward": total_reward,
        "timesteps": timesteps,
    }


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    result = run_human_episode()
    pd.DataFrame([result]).to_csv(RESULTS_PATH, index=False)

    print("Saved human score:")
    print(result)
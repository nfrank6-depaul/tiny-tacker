import numpy as np

from gym_sailing.envs.boat_env import BoatDiscreteEnv, BoatEnv
from gym_sailing.physics.sailboat import SailBoat


OLYMPIC_TRIANGLE_MARKS = [
    (50 * 0.55, 50 * 0.10),  # start / leeward mark
    (50 * 0.50, 50 * 0.88),  # windward mark
    (50 * 0.18, 50 * 0.52),  # reach mark
    (50 * 0.55, 50 * 0.10),  # leeward mark
    (50 * 0.50, 50 * 0.88),  # windward mark
    (50 * 0.55, 50 * 0.10),  # downwind / leeward mark
    (50 * 0.50, 50 * 0.88),  # finish at windward mark
]


class SailboatEnv(BoatEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode)

    def reset(self, options=None, seed=None):
        self.boat = SailBoat(
            x=self.COURSE_SIZE * (0.5 + np.random.uniform(-0.2, 0.2)),
            y=self.COURSE_SIZE * 0.10,
            heading=self.np_random.random() * np.pi * 2,
            heading_dot=np.random.uniform(-0.03, 0.03),
            speed=np.random.uniform(-1, 0.5),
        )
        return super().reset(options, seed)


class SailboatOlympicTriangleEnv(BoatEnv):
    """Continuous-action sailboat environment using an Olympic triangle course.

    This variant keeps the original sailboat physics and single-target reward
    shaping, but advances the target through a sequence of race marks instead of
    terminating after one buoy. The renderer will show the currently active mark.
    """

    MARKS = OLYMPIC_TRIANGLE_MARKS
    MARK_REWARD = 50
    FINISH_REWARD = 100

    def __init__(self, render_mode=None):
        super().__init__(render_mode)
        self.mark_index = 0
        self.TARGET = self.MARKS[self.mark_index]

    def _advance_target(self):
        self.mark_index += 1
        self.TARGET = self.MARKS[self.mark_index]
        self.prev_distance2target = np.array([self.boat.x, self.boat.y]) - np.array(
            self.TARGET
        )

    def _get_reward(self, distance2target):
        terminated = False
        reward = -0.1  # Alive penalty

        if np.linalg.norm(distance2target) < self.TARGET_RAD:
            if self.mark_index == len(self.MARKS) - 1:
                reward = self.FINISH_REWARD
                terminated = True
            else:
                reward = self.MARK_REWARD
                self._advance_target()

        elif np.linalg.norm(distance2target) >= self.COURSE_SIZE:
            reward = -100
            terminated = True

        else:
            reward += 10 * (
                np.linalg.norm(self.prev_distance2target, 8)
                - np.linalg.norm(distance2target, 8)
            )
            self.prev_distance2target = distance2target

        self.last_reward = reward
        return terminated, reward

    def reset(self, options=None, seed=None):
        self.mark_index = 0
        self.TARGET = self.MARKS[self.mark_index]
        self.boat = SailBoat(
            x=self.COURSE_SIZE * 0.55,
            y=self.COURSE_SIZE * 0.10,
            heading=np.pi / 2,
            heading_dot=np.random.uniform(-0.03, 0.03),
            speed=np.random.uniform(-0.2, 0.2),
        )
        return super().reset(options, seed)


class SailboatDiscreteEnv(BoatDiscreteEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode)

    def reset(self, options=None, seed=None):
        self.boat = SailBoat(
            x=self.COURSE_SIZE * (0.5 + np.random.uniform(-0.2, 0.2)),
            y=self.COURSE_SIZE * 0.10,
            heading=self.np_random.random() * np.pi * 2,
            heading_dot=np.random.uniform(-0.03, 0.03),
            speed=np.random.uniform(-1, 0.5),
        )
        return super().reset(options, seed)


class SailboatOlympicTriangleDiscreteEnv(BoatDiscreteEnv):
    """Discrete-action sailboat environment using an Olympic triangle course."""

    MARKS = OLYMPIC_TRIANGLE_MARKS
    MARK_REWARD = 50
    FINISH_REWARD = 100

    def __init__(self, render_mode=None):
        super().__init__(render_mode)
        self.mark_index = 0
        self.TARGET = self.MARKS[self.mark_index]

    def _advance_target(self):
        self.mark_index += 1
        self.TARGET = self.MARKS[self.mark_index]
        self.prev_distance2target = np.array([self.boat.x, self.boat.y]) - np.array(
            self.TARGET
        )

    def _get_reward(self, distance2target):
        terminated = False
        reward = -0.1  # Alive penalty

        if np.linalg.norm(distance2target) < self.TARGET_RAD:
            if self.mark_index == len(self.MARKS) - 1:
                reward = self.FINISH_REWARD
                terminated = True
            else:
                reward = self.MARK_REWARD
                self._advance_target()

        elif np.linalg.norm(distance2target) >= self.COURSE_SIZE:
            reward = -100
            terminated = True

        else:
            reward += 10 * (
                np.linalg.norm(self.prev_distance2target, 8)
                - np.linalg.norm(distance2target, 8)
            )
            self.prev_distance2target = distance2target

        self.last_reward = reward
        return terminated, reward

    def reset(self, options=None, seed=None):
        self.mark_index = 0
        self.TARGET = self.MARKS[self.mark_index]
        self.boat = SailBoat(
            x=self.COURSE_SIZE * 0.55,
            y=self.COURSE_SIZE * 0.10,
            heading=np.pi / 2,
            heading_dot=np.random.uniform(-0.03, 0.03),
            speed=np.random.uniform(-0.2, 0.2),
        )
        return super().reset(options, seed)

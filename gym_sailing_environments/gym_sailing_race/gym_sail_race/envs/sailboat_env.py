import numpy as np

from gym_sail_race.envs.boat_env import BoatDiscreteEnv, BoatEnv
from gym_sail_race.physics.sailboat import SailBoat

# Olympic triangle race-course geometry.
# The windward/leeward base is vertical and positioned on the right side.
COURSE_CENTER = np.array([25.0, 25.0])
VERTICAL_BASE_X = 38.0
TRIANGLE_SIDE = 30.0
TRIANGLE_HEIGHT = TRIANGLE_SIDE * np.sqrt(3) / 2

WINDWARD_MARK = (
    VERTICAL_BASE_X,
    COURSE_CENTER[1] + TRIANGLE_SIDE / 2,
)

LEEWARD_MARK = (
    VERTICAL_BASE_X,
    COURSE_CENTER[1] - TRIANGLE_SIDE / 2,
)

REACH_MARK = (
    VERTICAL_BASE_X - TRIANGLE_HEIGHT,
    COURSE_CENTER[1],
)

RACE_MARKS = [
    LEEWARD_MARK,
    WINDWARD_MARK,
    REACH_MARK,
]

TARGET_SEQUENCE = [
    WINDWARD_MARK,
    REACH_MARK,
    LEEWARD_MARK,
    WINDWARD_MARK,
    LEEWARD_MARK,
    WINDWARD_MARK,
]

class SailboatRaceEnv(BoatEnv):
    MARKS = RACE_MARKS
    TARGET_SEQUENCE = TARGET_SEQUENCE
    MARK_REWARD = 50
    FINISH_REWARD = 100

    def __init__(self, render_mode=None):
        super().__init__(render_mode)
        self.mark_index = 0
        self.TARGET = self.TARGET_SEQUENCE[self.mark_index]

    @property
    def active_target_index(self):
        return self.MARKS.index(self.TARGET)

    def _advance_target(self):
        self.mark_index += 1
        self.TARGET = self.TARGET_SEQUENCE[self.mark_index]
        self.prev_distance2target = np.array([self.boat.x, self.boat.y]) - np.array(
            self.TARGET
        )

    def _get_reward(self, distance2target):
        terminated = False
        reward = -0.1

        if np.linalg.norm(distance2target) < self.TARGET_RAD:
            if self.mark_index == len(self.TARGET_SEQUENCE) - 1:
                reward = self.FINISH_REWARD
                terminated = True
            else:
                reward = self.MARK_REWARD
                self._advance_target()

        elif (
            self.boat.x < 0
            or self.boat.x > self.COURSE_SIZE
            or self.boat.y < 0
            or self.boat.y > self.COURSE_SIZE
        ):
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
        self.TARGET = self.TARGET_SEQUENCE[self.mark_index]

        start_x, start_y = LEEWARD_MARK
        self.boat = SailBoat(
            x=start_x + self.BOAT_LENGTH,
            y=start_y,
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

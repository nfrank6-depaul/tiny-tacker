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

# Starting line: slightly windward of the leeward mark, extending to the right.
START_LINE_Y = LEEWARD_MARK[1] + 2.0
START_LINE = (
    (LEEWARD_MARK[0], START_LINE_Y),
    (min(50.0, LEEWARD_MARK[0] + 10.0), START_LINE_Y),
)

# Finish line: slightly leeward of the windward mark, extending to the right.
FINISH_LINE_Y = WINDWARD_MARK[1] - 2.0
FINISH_LINE = (
    (WINDWARD_MARK[0], FINISH_LINE_Y),
    (min(50.0, WINDWARD_MARK[0] + 10.0), FINISH_LINE_Y),
)


class SailboatRaceEnv(BoatEnv):
    MARKS = RACE_MARKS
    TARGET_SEQUENCE = TARGET_SEQUENCE
    MARK_REWARD = 50
    FINISH_REWARD = 100
    ROUNDING_ZONE_MULTIPLIER = 2.5
    INVALID_ROUNDING_PENALTY = 1.0

    def __init__(self, render_mode=None):
        super().__init__(render_mode)
        self.mark_index = 0
        self.TARGET = self.TARGET_SEQUENCE[self.mark_index]
        self.start_line = START_LINE
        self.show_start_line = True
        self.finish_line = FINISH_LINE
        self.prev_boat_y = None
        self.valid_port_rounding_started = False

    @property
    def active_target_index(self):
        return self.MARKS.index(self.TARGET)

    @property
    def show_finish_line(self):
        return self.mark_index == len(self.TARGET_SEQUENCE) - 1

    @property
    def active_course_line(self):
        if self.show_start_line:
            return self.start_line
        if self.show_finish_line:
            return self.finish_line
        return None

    @property
    def active_course_line_label(self):
        if self.show_start_line:
            return "START"
        if self.show_finish_line:
            return "FINISH"
        return None

    def _target_vector(self):
        return np.array(self.TARGET) - np.array([self.boat.x, self.boat.y])

    def _heading_vector(self):
        return np.array([
            np.cos(self.boat.heading),
            np.sin(self.boat.heading),
        ])

    def _target_is_to_port(self):
        heading_vec = self._heading_vector()
        target_vec = self._target_vector()
        cross = heading_vec[0] * target_vec[1] - heading_vec[1] * target_vec[0]
        return cross > 0

    def _inside_rounding_zone(self, distance2target):
        return np.linalg.norm(distance2target) < self.TARGET_RAD * self.ROUNDING_ZONE_MULTIPLIER

    def _target_hit_by_valid_rounding(self, distance2target):
        inside_rounding_zone = self._inside_rounding_zone(distance2target)

        if inside_rounding_zone and self._target_is_to_port():
            self.valid_port_rounding_started = True

        if self.valid_port_rounding_started and not inside_rounding_zone:
            self.valid_port_rounding_started = False
            return True

        return False

    def _final_target_passed_to_port(self, distance2target):
        return (
            self.mark_index == len(self.TARGET_SEQUENCE) - 1
            and self._inside_rounding_zone(distance2target)
            and self._target_is_to_port()
        )

    def step(self, action):
        previous_y = self.boat.y
        obs, reward, terminated, truncated, info = super().step(action)

        if self.show_start_line:
            line_start, _ = self.start_line
            line_y = line_start[1]

            crossed_windward = previous_y < line_y <= self.boat.y

            if crossed_windward:
                self.show_start_line = False

        return obs, reward, terminated, truncated, info

    def _advance_target(self):
        self.mark_index += 1
        self.TARGET = self.TARGET_SEQUENCE[self.mark_index]
        self.valid_port_rounding_started = False
        self.prev_distance2target = np.array([self.boat.x, self.boat.y]) - np.array(
            self.TARGET
        )

    def _get_reward(self, distance2target):
        terminated = False
        reward = -0.1

        if self._final_target_passed_to_port(distance2target):
            reward = self.FINISH_REWARD
            terminated = True

        elif self._target_hit_by_valid_rounding(distance2target):
            reward = self.MARK_REWARD
            self._advance_target()

        elif self._inside_rounding_zone(distance2target) and not self._target_is_to_port():
            reward -= self.INVALID_ROUNDING_PENALTY

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
        self.start_line = START_LINE
        self.show_start_line = True
        self.finish_line = FINISH_LINE
        self.prev_boat_y = None
        self.valid_port_rounding_started = False
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

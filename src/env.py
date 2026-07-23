import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from dataclasses import dataclass
from src.renderer import SoccerRenderer

X_MIN = 0
Y_MIN = 0
X_MAX = 20
Y_MAX = 20
VX_MIN = -10
VY_MIN = -10
VX_MAX = 10
VY_MAX = 10
GOAL_MIN_WIDTH = 3
GOAL_MAX_WIDTH = 6
FIXED_GOAL_X = 10
FIXED_GOAL_WIDTH = 5
AGENT_RADIUS = 0.35
BALL_RADIUS = 0.20
COLLISION_DISTANCE = AGENT_RADIUS + BALL_RADIUS
TIME_STEP = 0.1
PHYSICS_SUBSTEPS = 4
PHYSICS_TIME_STEP = TIME_STEP / PHYSICS_SUBSTEPS
AGENT_ACCELERATION = 75.0
MAX_AGENT_SPEED = 10.0
MAX_BALL_SPEED = 9.0
AGENT_DRAG = 2.0
BALL_DRAG = 0.8
BALL_BOUNCE = 0.7
BALL_KICK_TRANSFER = 1.2
INITIAL_SEPARATION = 1.0
RESET_MARGIN = 1.0
MAX_STEPS = 50
STEP_PENALTY = -0.01
CONTACT_REWARD = 0.5
GOAL_REWARD = 50.0


@dataclass
class Vector2:
    x: float
    y: float


def distance(point_1, point_2):
    return math.sqrt((point_1.x - point_2.x) ** 2 + (point_1.y - point_2.y) ** 2)


def vector_length(vector):
    return math.sqrt(vector.x ** 2 + vector.y ** 2)


def normalize(vector):
    length = vector_length(vector)
    if length == 0:
        return Vector2(x=0, y=0)
    return Vector2(x=vector.x / length, y=vector.y / length)


def dot(vector_1, vector_2):
    return vector_1.x * vector_2.x + vector_1.y * vector_2.y


def limit_vector(vector, max_length):
    length = vector_length(vector)
    if length <= max_length:
        return vector
    factor = max_length / length
    return Vector2(x=vector.x * factor, y=vector.y * factor)


def apply_drag(vector, drag):
    factor = max(0, 1 - drag * PHYSICS_TIME_STEP)
    return Vector2(x=vector.x * factor, y=vector.y * factor)


class SoccerEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, randomize_goal=True, max_steps=MAX_STEPS):
        super().__init__()
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"Unsupported render mode: {render_mode}")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")

        self.render_mode = render_mode
        self.randomize_goal = randomize_goal
        self.max_steps = max_steps
        self.renderer = None
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([
                X_MIN, Y_MIN,  # agent
                VX_MIN, VY_MIN,  # agent speed
                X_MIN, Y_MIN,  # ball
                VX_MIN, VY_MIN,  # ball speed
                X_MIN, Y_MIN,  # goal position
                GOAL_MIN_WIDTH
            ], dtype=np.float32),

            high=np.array([
                X_MAX, Y_MAX,  # agent
                VX_MAX, VY_MAX,  # agent speed
                X_MAX, Y_MAX,  # ball
                VX_MAX, VY_MAX,  # ball speed
                X_MAX, Y_MAX,  # goal position
                GOAL_MAX_WIDTH
            ], dtype=np.float32),

            shape=(11,),
            dtype=np.float32)
        self.agent_pos = Vector2(x=5, y=5)
        self.agent_speed = Vector2(x=0, y=0)
        self.ball_pos = Vector2(x=3, y=3)
        self.ball_speed = Vector2(x=0, y=0)
        self.goal_pos = Vector2(x=FIXED_GOAL_X, y=Y_MIN)
        self.goal_width = FIXED_GOAL_WIDTH
        
        self.steps = 0
        self.last_reward = 0.0
        self.last_touched_ball = False
        self.last_terminated = False


    def get_obs(self):
        return np.array([
            self.agent_pos.x,
            self.agent_pos.y,
            self.agent_speed.x,
            self.agent_speed.y,
            self.ball_pos.x,
            self.ball_pos.y,
            self.ball_speed.x,
            self.ball_speed.y,
            self.goal_pos.x,
            self.goal_pos.y,
            self.goal_width,
        ], dtype=np.float32)
    
    def get_random_position(self):
        x_ = self.np_random.uniform(X_MIN + RESET_MARGIN, X_MAX - RESET_MARGIN)
        y_ = self.np_random.uniform(Y_MIN + RESET_MARGIN, Y_MAX - RESET_MARGIN)
        return Vector2(x=x_, y=y_)

    def get_random_position_away_from(self, another_position, min_distance=INITIAL_SEPARATION):
        random_pos = self.get_random_position()       
        while distance(another_position, random_pos) < min_distance:
            random_pos = self.get_random_position()       
        return random_pos

    def get_random_goal(self):
        goal_width = self.np_random.uniform(GOAL_MIN_WIDTH, GOAL_MAX_WIDTH)
        goal_x = self.np_random.uniform(
            X_MIN + goal_width / 2,
            X_MAX - goal_width / 2,
        )
        return Vector2(x=goal_x, y=Y_MIN), goal_width
 
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.randomize_goal:
            self.goal_pos, self.goal_width = self.get_random_goal()
        else:
            self.goal_pos = Vector2(x=FIXED_GOAL_X, y=Y_MIN)
            self.goal_width = FIXED_GOAL_WIDTH
        self.agent_pos = self.get_random_position()
        self.agent_speed = Vector2(x=0, y=0)
        self.ball_pos = self.get_random_position_away_from(self.agent_pos)
        self.ball_speed = Vector2(x=0, y=0)

        observation = self.get_obs()

        info = {}

        self.steps = 0
        self.last_reward = 0.0
        self.last_touched_ball = False
        self.last_terminated = False
        if self.render_mode == "human":
            self.render()
        return observation, info


    def update_agent_speed(self, action):
        self.agent_speed.x += action[0] * AGENT_ACCELERATION * PHYSICS_TIME_STEP
        self.agent_speed.y += action[1] * AGENT_ACCELERATION * PHYSICS_TIME_STEP
        self.agent_speed = limit_vector(self.agent_speed, MAX_AGENT_SPEED)
        self.agent_speed = apply_drag(self.agent_speed, AGENT_DRAG)

    def move_agent(self):
        self.agent_pos.x += self.agent_speed.x * PHYSICS_TIME_STEP
        self.agent_pos.y += self.agent_speed.y * PHYSICS_TIME_STEP

        if self.agent_pos.x < X_MIN + AGENT_RADIUS:
            self.agent_pos.x = X_MIN + AGENT_RADIUS
            self.agent_speed.x = max(0, self.agent_speed.x)
        elif self.agent_pos.x > X_MAX - AGENT_RADIUS:
            self.agent_pos.x = X_MAX - AGENT_RADIUS
            self.agent_speed.x = min(0, self.agent_speed.x)

        if self.agent_pos.y < Y_MIN + AGENT_RADIUS:
            self.agent_pos.y = Y_MIN + AGENT_RADIUS
            self.agent_speed.y = max(0, self.agent_speed.y)
        elif self.agent_pos.y > Y_MAX - AGENT_RADIUS:
            self.agent_pos.y = Y_MAX - AGENT_RADIUS
            self.agent_speed.y = min(0, self.agent_speed.y)

    def resolve_agent_ball_collision(self):
        offset = Vector2(
            x=self.ball_pos.x - self.agent_pos.x,
            y=self.ball_pos.y - self.agent_pos.y,
        )
        separation = vector_length(offset)
        if separation >= COLLISION_DISTANCE:
            return False

        if separation == 0:
            normal = normalize(
                Vector2(
                    x=self.agent_speed.x - self.ball_speed.x,
                    y=self.agent_speed.y - self.ball_speed.y,
                )
            )
            if vector_length(normal) == 0:
                normal = Vector2(x=1, y=0)
        else:
            normal = normalize(offset)

        self.ball_pos.x = self.agent_pos.x + normal.x * COLLISION_DISTANCE
        self.ball_pos.y = self.agent_pos.y + normal.y * COLLISION_DISTANCE

        relative_speed = Vector2(
            x=self.agent_speed.x - self.ball_speed.x,
            y=self.agent_speed.y - self.ball_speed.y,
        )
        closing_speed = dot(relative_speed, normal)
        if closing_speed > 0:
            self.ball_speed.x += normal.x * closing_speed * BALL_KICK_TRANSFER
            self.ball_speed.y += normal.y * closing_speed * BALL_KICK_TRANSFER
            self.ball_speed = limit_vector(self.ball_speed, MAX_BALL_SPEED)
            self.agent_speed.x -= normal.x * closing_speed
            self.agent_speed.y -= normal.y * closing_speed

        return True

    def crossed_goal_line(self, previous_ball_pos, new_ball_pos):
        if previous_ball_pos.y < BALL_RADIUS or new_ball_pos.y > BALL_RADIUS:
            return False

        vertical_distance = previous_ball_pos.y - new_ball_pos.y
        if vertical_distance == 0:
            return False

        crossing_fraction = (previous_ball_pos.y - BALL_RADIUS) / vertical_distance
        crossing_x = previous_ball_pos.x + (
            new_ball_pos.x - previous_ball_pos.x
        ) * crossing_fraction
        goal_start = self.goal_pos.x - self.goal_width / 2
        goal_end = self.goal_pos.x + self.goal_width / 2
        if goal_start <= crossing_x <= goal_end:
            self.ball_pos = Vector2(x=crossing_x, y=BALL_RADIUS)
            self.ball_speed = Vector2(x=0, y=0)
            return True
        return False

    def move_ball(self):
        previous_ball_pos = Vector2(x=self.ball_pos.x, y=self.ball_pos.y)
        new_ball_pos = Vector2(
            x=self.ball_pos.x + self.ball_speed.x * PHYSICS_TIME_STEP,
            y=self.ball_pos.y + self.ball_speed.y * PHYSICS_TIME_STEP,
        )
        if self.crossed_goal_line(previous_ball_pos, new_ball_pos):
            return True

        self.ball_pos = new_ball_pos
        if self.ball_pos.x < X_MIN + BALL_RADIUS:
            self.ball_pos.x = X_MIN + BALL_RADIUS
            self.ball_speed.x = abs(self.ball_speed.x) * BALL_BOUNCE
        elif self.ball_pos.x > X_MAX - BALL_RADIUS:
            self.ball_pos.x = X_MAX - BALL_RADIUS
            self.ball_speed.x = -abs(self.ball_speed.x) * BALL_BOUNCE

        if self.ball_pos.y < Y_MIN + BALL_RADIUS:
            self.ball_pos.y = Y_MIN + BALL_RADIUS
            self.ball_speed.y = abs(self.ball_speed.y) * BALL_BOUNCE
        elif self.ball_pos.y > Y_MAX - BALL_RADIUS:
            self.ball_pos.y = Y_MAX - BALL_RADIUS
            self.ball_speed.y = -abs(self.ball_speed.y) * BALL_BOUNCE

        self.ball_speed = apply_drag(self.ball_speed, BALL_DRAG)
        return False

    def render(self):
        if self.render_mode is None:
            return None

        if self.renderer is None:
            self.renderer = SoccerRenderer(
                X_MIN,
                Y_MIN,
                X_MAX,
                Y_MAX,
                AGENT_RADIUS,
                BALL_RADIUS,
                self.metadata["render_fps"],
            )

        return self.renderer.render(
            self.render_mode,
            self.agent_pos,
            self.ball_pos,
            self.goal_pos,
            self.goal_width,
            self.steps,
            self.last_reward,
            self.last_touched_ball,
            self.last_terminated,
        )

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
                
    def step(self, action):
        reward = STEP_PENALTY
        self.steps += 1

        action = np.clip(action, self.action_space.low, self.action_space.high)
        touched_ball = False
        terminated = False
        for _ in range(PHYSICS_SUBSTEPS):
            self.update_agent_speed(action)
            self.move_agent()
            touched_ball = self.resolve_agent_ball_collision() or touched_ball
            terminated = self.move_ball()
            if terminated:
                break
            touched_ball = self.resolve_agent_ball_collision() or touched_ball

        if touched_ball:
            reward += CONTACT_REWARD
        if terminated:
            reward += GOAL_REWARD

        obs = self.get_obs()
        
        truncated = self.steps >= self.max_steps

        info = {
                "touched_ball": touched_ball,
                "goal": terminated,
                "distance_agent_ball": distance(self.agent_pos, self.ball_pos),
        }
        self.last_reward = reward
        self.last_touched_ball = touched_ball
        self.last_terminated = terminated
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

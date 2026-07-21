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
GOAL_DEPTH = 1
TOUCH_DISTANCE = 1
MAX_STEPS = 200


@dataclass
class Vector2:
    x: float
    y: float


def distance(point_1, point_2):
    return math.sqrt((point_1.x - point_2.x) ** 2 + (point_1.y - point_2.y) ** 2)
        
def distance_x(point_1, point_2):
    return abs(point_1.x - point_2.x)


def is_in_x_limits(pos_x):
    return pos_x > X_MIN and pos_x < X_MAX

def is_in_y_limits(pos_y):
    return pos_y > Y_MIN and pos_y < Y_MAX
def is_in_limits(pos):
    return is_in_x_limits(pos.x) and is_in_y_limits(pos.y)
class SoccerEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None):
        super().__init__()
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"Unsupported render mode: {render_mode}")

        self.render_mode = render_mode
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
        self.agent_speed = Vector2(x=1, y=1)
        self.ball_pos = Vector2(x=3, y=3)
        self.ball_speed = Vector2(x=0, y=0)
        self.goal_pos = Vector2(x=10, y=0)
        self.goal_width = 5
        
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
        x_ = self.np_random.uniform(X_MIN + 1, X_MAX - 1)
        y_ = self.np_random.uniform(Y_MIN + GOAL_DEPTH, Y_MAX - 1.0)
        return Vector2(x=x_, y=y_)
    def get_random_position_away_from(self, another_position, min_distance=TOUCH_DISTANCE):
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

        self.goal_pos, self.goal_width = self.get_random_goal()
        self.agent_pos = self.get_random_position()
        self.agent_speed = Vector2(x=1, y=1)
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


    def inside_goal(self):
        return self.ball_pos.y >= 0 and self.ball_pos.y < GOAL_DEPTH and distance_x(self.ball_pos, self.goal_pos) < self.goal_width / 2

    def collisioning_with_ball(self, new_agent_pos):
        return distance(new_agent_pos, self.ball_pos) < 0.1
            
    def make_agent_move(self, action):
        new_agent_x_pos = self.agent_pos.x + action[0] * self.agent_speed.x
        new_agent_y_pos = self.agent_pos.y + action[1] * self.agent_speed.y
        new_agent_pos = Vector2(x=new_agent_x_pos, y=new_agent_y_pos)
        if not self.collisioning_with_ball(new_agent_pos) and is_in_limits(new_agent_pos):
            if is_in_x_limits(new_agent_x_pos):
                self.agent_pos.x = new_agent_x_pos
            if is_in_y_limits(new_agent_y_pos):
                self.agent_pos.y = new_agent_y_pos
            
    def make_ball_move(self):
        dx = self.ball_pos.x -  self.agent_pos.x
        dy = self.ball_pos.y -  self.agent_pos.y
        length = math.sqrt(dx ** 2 +  dy ** 2)
        if length > 0:
            new_ball_x_pos = self.ball_pos.x + (dx / length)
            new_ball_y_pos = self.ball_pos.y + (dy / length)
            if is_in_x_limits(new_ball_x_pos):
                self.ball_pos.x = new_ball_x_pos
            if is_in_y_limits(new_ball_y_pos):
                self.ball_pos.y = new_ball_y_pos

    def render(self):
        if self.render_mode is None:
            return None

        if self.renderer is None:
            self.renderer = SoccerRenderer(
                X_MIN,
                Y_MIN,
                X_MAX,
                Y_MAX,
                TOUCH_DISTANCE,
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
        reward = -0.01
        terminated = False
        self.steps += 1
        
        # for now, speed stay fix
        self.make_agent_move(action)
        touched_ball = distance(self.agent_pos, self.ball_pos) < TOUCH_DISTANCE
        if touched_ball:
                self.make_ball_move()
                reward += 0.1

        if self.inside_goal():
                reward += 10.0
                terminated = True

        obs = self.get_obs()
        
        truncated = self.steps >= MAX_STEPS

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

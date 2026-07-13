import gymnasium as gym
from gymnasium import spaces
import numpy as np
from types import SimpleNamespace
import math

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
TOUCH_DISTANCE = 1
MAX_STEPS = 200

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
    def __init__(self):
        super().__init__()
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
        self.agent_pos = SimpleNamespace(x=5, y=5)
        self.agent_speed = SimpleNamespace(vx=1, vy=1)
        self.ball_pos = SimpleNamespace(x=3, y=3)
        self.ball_speed = SimpleNamespace(vx=0, vy=0)
        self.goal_pos = SimpleNamespace(x=10, y=0)
        self.goal_width = 5
        
        self.steps = 0

    def get_obs(self):
        return np.array([
            self.agent_pos.x,
            self.agent_pos.y,
            self.agent_speed.vx,
            self.agent_speed.vy,
            self.ball_pos.x,
            self.ball_pos.y,
            self.ball_speed.vx,
            self.ball_speed.vy,
            self.goal_pos.x,
            self.goal_pos.y,
            self.goal_width,
        ], dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.agent_pos = SimpleNamespace(x=5, y=5)
        self.agent_speed = SimpleNamespace(vx=1, vy=1)
        self.ball_pos = SimpleNamespace(x=3, y=3)
        self.ball_speed = SimpleNamespace(vx=0, vy=0)
        self.goal_pos = SimpleNamespace(x=10, y=0)
        self.goal_width = 5

        observation = self.get_obs()

        info = {}

        self.steps = 0
        return observation, info


    def inside_goal(self):
        return self.ball_pos.y >= 0 and self.ball_pos.y < 0.5 and distance_x(self.ball_pos, self.goal_pos) < self.goal_width / 2

    def collisioning_with_ball(self, new_agent_pos):
        return distance(new_agent_pos, self.ball_pos) < 0.1
            
    def make_agent_move(self, action):
        new_agent_x_pos = self.agent_pos.x + action[0] * self.agent_speed.vx 
        new_agent_y_pos = self.agent_pos.y + action[1] * self.agent_speed.vy
        new_agent_pos = SimpleNamespace(x=new_agent_x_pos, y=new_agent_y_pos)
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
        return obs, reward, terminated, truncated, info

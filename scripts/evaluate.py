from stable_baselines3 import PPO
from src.env import SoccerEnv


env = SoccerEnv(render_mode="human")

model = PPO.load("models/ppo_soccer")

for i in range(10):
    obs, info = env.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
env.close()

from src.env import SoccerEnv
from stable_baselines3 import PPO
from pathlib import Path


env = SoccerEnv(render_mode=None)
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1
)

model.learn(total_timesteps=100_000)

Path("models").mkdir(exist_ok=True)
model.save("models/ppo_soccer")
env.close()

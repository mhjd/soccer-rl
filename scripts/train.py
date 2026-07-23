from pathlib import Path

from stable_baselines3 import PPO

from src.env import SoccerEnv

fixed_goal_env = SoccerEnv(render_mode=None, randomize_goal=False)
model = PPO(
    policy="MlpPolicy",
    env=fixed_goal_env,
    verbose=1,
)

model.learn(total_timesteps=400_000)

random_goal_env = SoccerEnv(render_mode=None, randomize_goal=True)
model.set_env(random_goal_env)
fixed_goal_env.close()
model.learn(total_timesteps=400_000, reset_num_timesteps=False)

Path("models").mkdir(exist_ok=True)
model.save("models/ppo_soccer_curriculum")
random_goal_env.close()

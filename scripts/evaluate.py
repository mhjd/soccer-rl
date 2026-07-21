from stable_baselines3 import PPO
from src.env import SoccerEnv

NUM_EPISODES = 100

env = SoccerEnv(render_mode="human")

model = PPO.load("models/ppo_soccer")

total_reward = 0
success = 0

steps = 0
for i in range(NUM_EPISODES):
    obs, info = env.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated:
            success += 1
        
env.close()

print(f"goals scored : {success}")
print(f"Success rate: {success/NUM_EPISODES*100}%")
print(f"mean episode length: {steps/NUM_EPISODES}")
print(f"mean reward by episode : {total_reward/NUM_EPISODES}")

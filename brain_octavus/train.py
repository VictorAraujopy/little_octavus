import gymnasium as gym
import torch
from brain_octavus.brain import SimpleBrain
from world_octavus.environment import OctopusEnv


env = OctopusEnv()

brain = SimpleBrain(env.observation_space.shape[0], env.action_space.shape[0])
n_steps = 2048 #steps before learn
memory_x, memory_y, memory_reward, memory_done = [], [], [], []
x, _ = env.reset(seed=0)

for step in range(n_steps): # for each action from model
    x = torch.tensor(x, dtype=torch.float32)
     
    with torch.no_grad():
        y = brain.act(x)
        next_x, reward, terminated, truncated, _ = env.step(y.numpy())
        memory_x.append(x)
        memory_y.append(y)
        memory_reward.append(reward)
        memory_done.append(terminated or truncated)

        x = next_x
        
        if terminated or truncated:
            x, _ = env.reset()
            
print(f"{n_steps} steps played, {sum(memory_done)} episodes finished")
print(f"total reward this round: {sum(memory_reward):.0f}")
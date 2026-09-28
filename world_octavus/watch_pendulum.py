"""
Shows the SimpleBrain controlling the InvertedPendulum (the "tuner" to tell training bugs from octopus difficulty).
The episode ends when the pole tilts more than ~11°; the pause at the end gives time to see that moment.

Run:                    uv run world_octavus/watch_pendulum.py
Without the 11° rule:   uv run world_octavus/watch_pendulum.py --let-fall   (the pole falls all the way, 3s per try)
"""

import sys
import time

import gymnasium as gym
import numpy as np
import torch

from brain_octavus.brain import SimpleBrain

let_fall = "--let-fall" in sys.argv
steps_when_letting_fall = 75

env = gym.make("InvertedPendulum-v5", render_mode="human", disable_env_checker=let_fall)
brain = SimpleBrain(env.observation_space.shape[0], env.action_space.shape[0])

observation, _ = env.reset(seed=0)
steps = 0
while True:
    # only watching, not training: no need to record the math for weight updates
    with torch.no_grad():
        action = brain.body(torch.tensor(observation, dtype=torch.float32)).numpy()
    observation, _, terminated, truncated, _ = env.step(action)
    steps += 1
    done = steps >= steps_when_letting_fall if let_fall else (terminated or truncated)
    if done:
        angle = np.degrees(observation[1])
        print(f"{steps} steps ({steps * env.unwrapped.dt:.1f}s), pole ended at {angle:+.0f}°")
        time.sleep(1.0)
        observation, _ = env.reset()
        steps = 0

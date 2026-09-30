"""
Watch the trained octopus: loads octavus.pt and lets the brain drive the body in the viewer.
The brain is reloaded at the start of every episode, so you can keep this open while training runs
and see it improve without restarting.

Run (on macOS the viewer needs mjpython):  uv run mjpython world_octavus/watch_octopus.py
With exploration (the random tries the trainer sees):  add --explore
"""

import sys
from pathlib import Path

import torch

from brain_octavus.brain import Octavus_arms_brain
from world_octavus.environment import OctopusEnv

CHECKPOINT = Path(__file__).resolve().parents[1] / "octavus.pt"
explore = "--explore" in sys.argv


def load_latest(brain):
    try:
        brain.load_state_dict(torch.load(CHECKPOINT))
        return True
    except RuntimeError as error:
        if "size mismatch" in str(error):
            raise SystemExit(f"{CHECKPOINT.name} was trained on a different body: train again with the current one")
        return False  # caught while the trainer was writing it: keep the previous weights
    except (FileNotFoundError, EOFError):
        return False


if not CHECKPOINT.exists():
    raise SystemExit(f"no {CHECKPOINT.name} yet: run the training first (uv run brain_octavus/train.py)")

env = OctopusEnv(render_mode="human")
brain = Octavus_arms_brain()
load_latest(brain)
x, _ = env.reset()
steps = 0

while env.viewer is None or env.viewer.is_running():
    with torch.no_grad():
        _, distribuition = brain.act(torch.tensor(x, dtype=torch.float32))
        # the center of the draw is the move the brain believes in most; --explore adds the random tries
        action = distribuition.sample() if explore else distribuition.mean
    x, _, terminated, truncated, info = env.step(action.numpy())
    steps += 1
    if terminated or truncated:
        result = "REACHED the target" if info["reached"] else "flipped over" if info["flipped"] else "time's up"
        print(f"{result} after {steps} steps ({steps * env.dt:.1f}s), {info['distance']:.2f} m from the target")
        load_latest(brain)
        x, _ = env.reset()
        steps = 0
env.close()

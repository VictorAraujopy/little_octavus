# little_octavus

An octopus that learns to walk across the sea floor — and later to hunt a crab — through reinforcement learning. The brain and the PPO training loop are written from scratch in PyTorch; the body and the physics run on MuJoCo.

## How it works

- **The world** (`world_octavus/`): an octopus with 8 arms on a sandy sea floor, simulated in MuJoCo with water drag. Each arm has 2 segments and 4 motors (shoulder swing, shoulder lift, elbow, sucker) plus a touch sensor on the tip — 32 motors in total. The environment follows the Gymnasium API and places a target on the floor in a random direction every episode.
- **The brain** (`brain_octavus/`): an actor-critic network. The actor is a single small network shared by all 8 arms — each arm feeds it its own sensors, the body's state and its position on the body, and gets back its 4 motor commands. What one arm learns, every arm knows. The critic looks at the whole octopus and estimates how much reward is still to come.
- **The reward** is kept out of the environment on purpose: the trainer passes a `reward_fn(info)`, and the environment only reports the facts of each step (distance to the target, whether it arrived, whether it flipped over, the action used).

## Status

Work in progress.

- [x] Body, sea floor and Gymnasium environment
- [x] Brain with a shared arm network
- [ ] Reward function
- [ ] PPO training loop
- [ ] Walking to a target
- [ ] Hunting a crab that learns to run away

## Running

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14.

```bash
uv sync

# watch the untrained octopus flail around (on macOS the viewer needs mjpython)
uv run mjpython world_octavus/environment.py

# open just the body in the MuJoCo viewer (the path must be absolute)
uv run python -m mujoco.viewer --mjcf="$PWD/world_octavus/octopus.xml"

# collect one round of experience (training loop in progress)
uv run brain_octavus/train.py
```

`world_octavus/watch_pendulum.py` runs the simple brain on Gymnasium's `InvertedPendulum-v5`, used as a quick sanity check: if training can't balance the pole, the bug is in the training code, not in the octopus.

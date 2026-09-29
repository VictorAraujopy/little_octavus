# little_octavus

An octopus that learns to walk across the sea floor — and later to hunt a crab — through reinforcement learning. The brain and the PPO training loop are written from scratch in PyTorch; the body and the physics run on MuJoCo.

## How it works

- **The world** (`world_octavus/`): an octopus with 8 arms on a sandy sea floor, simulated in MuJoCo with water drag. Each arm has 3 segments and 6 motors (shoulder swing, shoulder lift, elbow, tip bend, tip curl, sucker) plus a touch sensor on the tip — 48 motors in total. The tip joints have deliberately weak muscles, so the tips sway and give in the water like soft tentacles. The environment follows the Gymnasium API and places a target on the floor in a random direction every episode.
- **The brain** (`brain_octavus/brain.py`): an actor-critic network with **17,037 parameters**. The actor is a single small network (6,278 parameters) shared by all 8 arms — each arm feeds it its own sensors, the body's state and its position on the body, and gets back its 4 motor commands. What one arm learns, every arm knows. The critic (10,753 parameters) looks at the whole octopus and estimates how much reward is still to come.
- **The reward** (`brain_octavus/reward.py`) is kept out of the environment on purpose: the trainer passes a `reward_fn(info)`, and the environment only reports the facts of each step. Octavus earns points for getting closer to the target and loses points for jerky moves, for bouncing, and for flipping over.
- **The training** (`brain_octavus/train.py`): PPO — play 2048 steps, score every decision against what the critic expected, nudge the arms towards the better-than-expected moves (never more than 20% per round), repeat.

## Status

Work in progress — training started on day one.

- [x] Body, sea floor and Gymnasium environment
- [x] Brain with a shared arm network
- [x] Reward function
- [x] PPO training loop
- [x] Walking to a target — after about 10 minutes of training it reached the target in every test episode… by bouncing there
- [ ] Crawling like an actual octopus (reshaping the reward to make bouncing expensive)
- [ ] Hunting a crab that learns to run away

## Running

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14.

```bash
uv sync

# train (saves octavus.pt every 10 rounds; Ctrl+C to stop)
uv run brain_octavus/train.py

# watch the trained octopus (on macOS the viewer needs mjpython).
# It reloads octavus.pt every episode, so it can stay open while training runs.
uv run mjpython world_octavus/watch_octopus.py
uv run mjpython world_octavus/watch_octopus.py --explore   # with the random tries the trainer sees

# watch the untrained octopus flail around
uv run mjpython world_octavus/environment.py

# open just the body in the MuJoCo viewer (the path must be absolute)
uv run python -m mujoco.viewer --mjcf="$PWD/world_octavus/octopus.xml"
```

`world_octavus/watch_pendulum.py` runs a simpler brain on Gymnasium's `InvertedPendulum-v5`, used as a quick sanity check: if training can't balance the pole, the bug is in the training code, not in the octopus.

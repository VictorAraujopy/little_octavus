# little_octavus

An octopus that learns to walk across the sea floor — and later to hunt a crab — through reinforcement learning. The brain and the PPO training loop are written from scratch in PyTorch; the body and the physics run on MuJoCo.

## How it works

- **The world** (`world_octavus/`): an octopus with 8 arms on a sandy sea floor, simulated in MuJoCo with water drag. Each arm has 3 segments and 6 motors (shoulder swing, shoulder lift, elbow, tip bend, tip curl, sucker) plus a touch sensor on the tip — 48 motors in total. The tip joints have deliberately weak muscles, so the tips sway and give in the water like soft tentacles. The environment follows the Gymnasium API and places a target on the floor in a random direction every episode.
- **The brain** (`brain_octavus/brain.py`): an actor-critic network with **196,407 parameters**. The actor is a single network that sees the whole octopus (101 numbers) and drives all 48 motors at once, so the arms can coordinate with each other. The critic looks at the same 101 numbers and estimates how much reward is still to come. (An earlier version shared one small network across the 8 arms, each arm deciding on its own; it learned fast but couldn't coordinate a proper gait.)
- **The reward** (`brain_octavus/reward.py`) is kept out of the environment on purpose: the trainer passes a `reward_fn(info)`, and the environment only reports the facts of each step. Octavus earns points for getting closer to the target (capped at an octopus pace of 0.3 m/s, so rushing earns nothing) and loses points for wasted muscle energy, jerky moves, bouncing, flying (nothing touching the floor), keeping its arm tips off the floor, lifting its body like a spider, spinning, flipping over and not facing the target — it has a front now, where its eyes point. Every one of those penalties was added after the agent found a way to exploit the previous reward.
- **The training** (`brain_octavus/train.py`): PPO — play 2048 steps, score every decision against what the critic expected, nudge the arms towards the better-than-expected moves (never more than 20% per round), repeat.

## Status

Work in progress — training started on day one.

- [x] Body, sea floor and Gymnasium environment
- [x] Brain with a shared arm network
- [x] Reward function
- [x] PPO training loop
- [x] Walking to a target — after about 10 minutes of training it reached the target in every test episode… by bouncing there
- [ ] Crawling like an actual octopus — in progress: it no longer hops (0% of the time in the air, down from 43%) and crawls with most arm tips on the floor; now learning to turn towards the target and get there
- [ ] Hunting a crab that learns to run away

## Running

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14.

```bash
uv sync

# train (saves octavus.pt every 10 rounds and continues from it if it exists; Ctrl+C to stop)
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

"""
The exam: a brain (octavus.pt by default) drives the octopus in every scenario of the teacher's watch (close, middle,
far, and each after a messed-up start; see world_octavus/teacher.py), in "mean" mode, and it prints how many targets it
reached. The episodes are the same in every exam (the episode number is the seed), so two brains can be compared.
The teacher reaches all of them: this says how close the copy (brain_octavus/imitate.py) is to it.

Run (from the project root, a few minutes):  uv run world_octavus/exam_octopus.py
Fewer or more episodes per scenario:  add --episodes 5
Another brain:  add --brain path/to/brain.pt
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from brain_octavus.brain import Octavus_arms_brain
from world_octavus.environment import OctopusEnv
from world_octavus.teacher import SCENARIOS, SIPHON_TILT, mess_up

CHECKPOINT = Path(__file__).resolve().parents[1] / "octavus.pt"


def take_scenario(scenario, episodes, checkpoint):
    # one scenario, all its episodes: each scenario runs in its own process, so the six take about as long as one
    torch.set_num_threads(1)  # tiny matrices: one thread each beats six processes fighting over every core
    name, farthest, messy = scenario
    brain = Octavus_arms_brain()
    brain.load_state_dict(torch.load(checkpoint))
    env = OctopusEnv(curriculum=False)
    env.farthest = farthest
    reached, times, switches = 0, [], []
    for seed in range(episodes):
        rng = np.random.default_rng(seed)
        obs, _ = env.reset(seed=seed)
        if messy:
            obs, _ = mess_up(env, obs, rng)
        done, steps, swimming, changes = False, 0, False, 0
        while not done:
            with torch.no_grad():
                _, distribuition = brain.act(torch.tensor(obs, dtype=torch.float32))
            action = distribuition.mean.numpy()  # what it believes, without the exploration draw
            changes += (action[SIPHON_TILT] < -0.5) != swimming  # funnel tipped down = swimming
            swimming = action[SIPHON_TILT] < -0.5
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated
        reached += info["reached"]
        if info["reached"]:
            times.append(steps * env.dt)
        switches.append(changes)
    return name, reached, times, switches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--brain", type=Path, default=CHECKPOINT)
    args = parser.parse_args()
    if not args.brain.exists():
        raise SystemExit(f"no {args.brain.name} yet: copy the teacher first (uv run brain_octavus/imitate.py)")

    with Pool(len(SCENARIOS)) as pool:
        results = pool.starmap(take_scenario, [(scenario, args.episodes, args.brain) for scenario in SCENARIOS])

    total = 0
    for name, reached, times, switches in results:
        total += reached
        time = f"{np.mean(times):.0f} s" if times else "-"
        print(f"{name:34s} reached {reached:2d}/{args.episodes}   mean time {time:>5s}   "
              f"swim/crawl switches per episode {np.mean(switches):.1f}")
    print(f"total: {total}/{args.episodes * len(SCENARIOS)} (the teacher reaches all of them, switching 0.4-2 times per episode)")


if __name__ == "__main__":
    main()

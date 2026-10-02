"""
Records the teacher (teacher.py) for the brain to copy: every step, what the octopus felt (the observation) and what
the teacher did (its action), saved to teacher_demos.npz in the project root.

In half the episodes the action that is carried out gets a little noise, so the octopus drifts off the teacher's
perfect path, but the action saved is the teacher's clean one: the brain also sees how the teacher gets back on track.
Without this, the first small mistake of the copying brain would take it somewhere the teacher never went, and it
wouldn't know what to do there. The noise has to be tiny on the arms: with the soft arm springs a command of 0.07
already bends a joint to its limit, and a noise of 0.1 left the teacher flailing (0/8 targets at 3-6 m).

The noise alone wasn't enough: the brain copied from the teacher's path stood still from the first step, half swimming
and half crawling near the distance where the teacher switches, and the farther it drifted, the stranger the places it
was in. With --student, that brain drives instead and the teacher only says, at every step, what it would do there;
these steps are added to the recording, and the brain studies again (DAgger: Ross, Gordon & Bagnell 2011). The teacher
can do this because it decides only from the observation.

Run (from the project root, ~8 min for the default 100,000 steps):  uv run world_octavus/record_teacher.py
After a copy (brain_octavus/imitate.py):  uv run world_octavus/record_teacher.py --student octavus.pt --steps 50000
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from brain_octavus.brain import Octavus_arms_brain
from world_octavus.environment import OctopusEnv
from world_octavus.teacher import teacher_action

OUT = Path(__file__).resolve().parents[1] / "teacher_demos.npz"
FARTHEST = np.arange(1.0, 6.5, 0.5)  # each episode picks one of these: targets from 0.5 m up to the real 3-6 m


def student_action(student, obs):
    with torch.no_grad():
        _, distribuition = student.act(torch.tensor(obs, dtype=torch.float32))
    return distribuition.mean.numpy()  # what it believes, without the exploration draw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--arm-noise", type=float, default=0.02)  # nudge on the 160 arm commands
    parser.add_argument("--siphon-noise", type=float, default=0.1)  # nudge on the siphon's aim, tilt and jet
    parser.add_argument("--clean-share", type=float, default=0.5)  # share of the episodes recorded with no nudge at all
    parser.add_argument("--student", type=Path)  # a brain that drives while the teacher says what it would do; adds to --out
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    if args.student:
        student = Octavus_arms_brain()
        student.load_state_dict(torch.load(args.student))

    env = OctopusEnv(curriculum=False)  # curriculum off: the distance is picked here, and curriculum.txt stays the training's
    rng = np.random.default_rng(0)
    observations, actions = [], []
    episodes, reached = 0, 0
    noise = np.r_[np.full(160, args.arm_noise), np.full(3, args.siphon_noise)]
    while len(observations) < args.steps:
        env.farthest = rng.choice(FARTHEST)
        nudge = 0.0 if rng.random() < args.clean_share else 1.0
        obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
        done = False
        while not done and len(observations) < args.steps:
            clean = teacher_action(obs)
            observations.append(obs)
            actions.append(clean)
            if args.student:
                carried_out = student_action(student, obs)
            else:
                carried_out = np.clip(clean + nudge * rng.normal(0.0, 1.0, clean.shape) * noise, -1.0, 1.0)
            obs, _, terminated, truncated, info = env.step(carried_out)
            done = terminated or truncated
        episodes += 1
        reached += bool(done and info["reached"])
        if episodes % 10 == 0:
            print(f"{len(observations):,} steps, {episodes} episodes, reached {reached}", flush=True)

    observations, actions = np.array(observations, np.float32), np.array(actions, np.float32)
    if args.student and args.out.exists():
        before = np.load(args.out)
        observations = np.concatenate([before["observations"], observations])
        actions = np.concatenate([before["actions"], actions])
    np.savez_compressed(args.out, observations=observations, actions=actions)
    print(f"saved {len(observations):,} steps to {args.out} ({episodes} new episodes, {reached} reached)")


if __name__ == "__main__":
    main()

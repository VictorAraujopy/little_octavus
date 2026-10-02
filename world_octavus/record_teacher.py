"""
Records the teacher (teacher.py) for the brain to copy: every step, what the octopus felt (the observation) and what
the teacher did (its action), saved to teacher_demos.npz in the project root.

The teacher has to show every situation, not only its own perfect path: a brain copied from that path alone stood still
from the first step, half swimming and half crawling, and the farther it drifted the stranger the places it was in.
So half the episodes start messed up (0-3 s of random commands, not recorded: tilted, arms swept, funnel turned) and
the teacher shows how to get out of it. In half the episodes the action that is carried out also gets a little noise,
while the action saved is the teacher's clean one, so it shows how it gets back on track. The noise has to be tiny on
the arms: with the soft arm springs a command of 0.07 already bends a joint to its limit.

Only the episodes where the teacher reached the target are saved: an example that ends in failure would teach failing.

With --student, a copied brain drives instead and the teacher only says, at every step, what it would do there; these
steps are added to the recording, failures included (that is where the lesson is), and the brain studies again (DAgger:
Ross, Gordon & Bagnell 2011). The teacher can do this because it decides only from the observation. Keep these episodes
short (--seconds 20): a stuck student otherwise fills the recording with thousands of steps of the same place.

Run (from the project root, ~40 min for the default 500,000 steps):  uv run world_octavus/record_teacher.py
After a copy (brain_octavus/imitate.py):  uv run world_octavus/record_teacher.py --student octavus.pt --steps 100000 --seconds 20
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from brain_octavus.brain import Octavus_arms_brain
from world_octavus.environment import OctopusEnv
from world_octavus.teacher import mess_up, teacher_action

OUT = Path(__file__).resolve().parents[1] / "teacher_demos.npz"
FARTHEST = np.arange(1.0, 6.5, 0.5)  # each episode picks one of these: targets from 0.5 m up to the real 3-6 m


def student_action(student, obs):
    with torch.no_grad():
        _, distribuition = student.act(torch.tensor(obs, dtype=torch.float32))
    return distribuition.mean.numpy()  # what it believes, without the exploration draw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--mess-share", type=float, default=0.5)  # share of the episodes that start messed up
    parser.add_argument("--arm-noise", type=float, default=0.02)  # nudge on the 160 arm commands
    parser.add_argument("--siphon-noise", type=float, default=0.1)  # nudge on the siphon's aim, tilt and jet
    parser.add_argument("--clean-share", type=float, default=0.5)  # share of the episodes recorded with no nudge at all
    parser.add_argument("--student", type=Path)  # a brain that drives while the teacher says what it would do; adds to --out
    parser.add_argument("--seconds", type=float)  # cut every episode at this long (default: the whole episode)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    if args.student:
        student = Octavus_arms_brain()
        student.load_state_dict(torch.load(args.student))

    env = OctopusEnv(curriculum=False)  # curriculum off: the distance is picked here, and curriculum.txt stays the training's
    longest = int(args.seconds / env.dt) if args.seconds else env.max_steps
    rng = np.random.default_rng(0)
    observations, actions = [], []
    episodes, reached, discarded = 0, 0, 0
    noise = np.r_[np.full(160, args.arm_noise), np.full(3, args.siphon_noise)]
    while len(observations) < args.steps:
        env.farthest = rng.choice(FARTHEST)
        nudge = 0.0 if rng.random() < args.clean_share else 1.0
        obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
        if rng.random() < args.mess_share:
            obs, _ = mess_up(env, obs, rng)
        episode_observations, episode_actions = [], []
        done, steps = False, 0
        while not done and steps < longest:
            clean = teacher_action(obs)
            episode_observations.append(obs)
            episode_actions.append(clean)
            if args.student:
                carried_out = student_action(student, obs)
            else:
                carried_out = np.clip(clean + nudge * rng.normal(0.0, 1.0, clean.shape) * noise, -1.0, 1.0)
            obs, _, terminated, truncated, info = env.step(carried_out)
            done = terminated or truncated
            steps += 1
        episodes += 1
        reached += bool(info["reached"])
        if args.student or info["reached"]:
            observations += episode_observations
            actions += episode_actions
        else:
            discarded += 1
        if episodes % 10 == 0:
            print(f"{len(observations):,} steps, {episodes} episodes, reached {reached}, discarded {discarded}", flush=True)

    observations, actions = np.array(observations, np.float32), np.array(actions, np.float32)
    if args.student and args.out.exists():
        before = np.load(args.out)
        observations = np.concatenate([before["observations"], observations])
        actions = np.concatenate([before["actions"], actions])
    np.savez_compressed(args.out, observations=observations, actions=actions)
    print(f"saved {len(observations):,} steps to {args.out} ({episodes} new episodes, {reached} reached, {discarded} discarded)")


if __name__ == "__main__":
    main()

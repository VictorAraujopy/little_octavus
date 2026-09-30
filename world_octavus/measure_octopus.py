"""
Measures how the trained octopus is doing: loads octavus.pt, lets the brain drive the body in "mean" mode
(the move it believes in most, no random tries) and prints a report. Every run also appends one row to
measurements.csv in the project root, so the numbers can be compared over time (open it in any spreadsheet).

Run (training can keep running):  uv run world_octavus/measure_octopus.py
Fewer or more episodes:            add --episodes 8
"""

import argparse
import csv
import math
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import torch

from brain_octavus.brain import Octavus_arms_brain
from brain_octavus.reward import reward
from world_octavus.build_octopus import SECTIONS
from world_octavus.environment import OctopusEnv

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "octavus.pt"
LOG = ROOT / "measurements.csv"
MUSCLES = ["bend_up", "bend_side", "twist", "stretch"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=16)
    episodes = parser.parse_args().episodes
    if not CHECKPOINT.exists():
        raise SystemExit(f"no {CHECKPOINT.name} yet: run the training first (uv run brain_octavus/train.py)")

    env = OctopusEnv(reward_fn=reward)
    brain = Octavus_arms_brain()
    brain.load_state_dict(torch.load(CHECKPOINT))
    suckers = [i for i in range(env.model.nu) if env.model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_BODY]
    sucker_touch_sensors = [env.model.sensor_adr[env.model.sensor(env.model.actuator(i).name.replace("_sucker", "_touch")).id]
                            for i in suckers]

    # one entry per step
    flipped, off_floor, power, touching, grip, jetting = [], [], [], [], [], []
    step_rewards, energy_costs = [], []
    brain_outputs, pre_tanh_sizes = [], []
    # one entry per episode
    reached, covered, jet_pulses, lowest_stamina = [], [], [], []

    for episode in range(episodes):
        observation, _ = env.reset(seed=100 + episode)
        start_distance = env.distance
        jet_on_this_episode, stamina_this_episode = [], []

        while True:
            x = torch.tensor(observation)
            with torch.no_grad():
                pre_tanh = brain.pre_tanh(x)  # every number the brain has before its last Tanh
                _, distribution = brain.act(x)
            brain_output = distribution.mean.numpy()  # 163 numbers: the center of the brain's draw
            observation, step_reward, terminated, truncated, info = env.step(brain_output)

            pre_tanh_sizes.append(pre_tanh.abs().numpy())  # size only: -5 and +5 are equally jammed
            brain_outputs.append(brain_output)
            flipped.append(info["flipped"])
            off_floor.append(info["airborne"])
            power.append(info["power"])
            step_rewards.append(step_reward)
            energy_costs.append(math.log(1 + 0.6 * info["power"] / 1000))  # the energy line of reward.py
            segment_touching = env.data.sensordata[sucker_touch_sensors] > 0.005  # each segment's touch sensor
            touching.append(segment_touching.mean())
            sucker_pull = env.data.ctrl[suckers] * env.model.actuator_gainprm[suckers, 0]  # command x sucker strength
            grip.append(sucker_pull[segment_touching].sum())  # only a touching sucker actually pulls
            jet_on = env.data.ctrl[env.jet] > 0.05  # the jet motor, after the env's mantle and stamina limits
            jetting.append(jet_on)
            jet_on_this_episode.append(jet_on)
            stamina_this_episode.append(env.stamina)
            if terminated or truncated:
                break

        reached.append(info["reached"])
        covered.append(start_distance - info["distance"])
        lowest_stamina.append(min(stamina_this_episode))
        jet_on_this_episode = np.array(jet_on_this_episode)
        squirts_started = jet_on_this_episode[1:] & ~jet_on_this_episode[:-1]  # off -> on
        jet_pulses.append(squirts_started.sum() + jet_on_this_episode[0])

    # the brain's 163 numbers have no names; the ENVIRONMENT's action layout (environment.py docstring) gives them:
    # the first 160 are [arm, section, motor], and a section's 5 motors are bend_up, bend_side, twist, stretch, sucker
    env_layout = np.clip(np.array(brain_outputs)[:, :160], -1, 1).reshape(-1, env.n_arms, SECTIONS, 5)
    muscle_commands = env_layout[:, :, :, :4]
    sucker_commands = env_layout[:, :, :, 4]
    # how hard each muscle was told to pull, whatever the direction: 0 = relaxed, 1 = full force
    # (-0.97 bends one way and +0.97 the other, both almost at full force)
    muscle_strength = np.abs(muscle_commands)
    muscle_almost_full = muscle_strength > 0.95
    # suckers only grip with a positive command; 0 or below is off
    sucker_grip_command = np.clip(sucker_commands, 0, 1)
    pre_tanh_sizes = np.array(pre_tanh_sizes)
    tanh_slope = 1 - np.tanh(pre_tanh_sizes) ** 2  # how much the Tanh lets a change through: 1 = all, 0 = nothing

    # [env] = the environment (info, sensors, motors), [brain] = straight from the network, [reward] = reward.py,
    # [brain + env layout] = the brain's numbers, named muscle/sucker by the environment's action layout
    row = {
        "when": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "checkpoint_saved": datetime.fromtimestamp(CHECKPOINT.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        "episodes": episodes,
        # [env] episodes that ended on the target. Ideal: all of them
        "reached": int(sum(reached)),
        # [env] meters it got closer to the target. Targets are 3-6 m away: ~3+ means arriving, below 0 = moving away
        "covered_m": round(float(np.mean(covered)), 3),
        # [env] % of the time upside down (it doesn't know how to right itself yet). Ideal: ~0
        "flipped_pct": round(float(np.mean(flipped)) * 100, 1),
        # [env] % of the time touching nothing. No ideal by itself: crawling keeps it low, swimming raises it
        "air_pct": round(float(np.mean(off_floor)) * 100, 1),
        # [env] % of the time jetting. A real octopus mostly crawls and jets in short bursts; high = it relies on the jet
        "jet_pct": round(float(np.mean(jetting)) * 100, 1),
        # [env] squirts per episode. Many short ones is how the mantle works (1 s squirt, then refill)
        "jet_pulses": round(float(np.mean(jet_pulses)), 1),
        # [env] lowest jet stamina in each episode, 0 to 1. 0 = it ran out; above ~0.3 = it saves some
        "lowest_stamina": round(float(np.mean(lowest_stamina)), 2),
        # [env] metabolic watts. Lower is better for the same progress: efficient moves cost tens to ~150 W, flailing 1000-3000 W
        "power_w": round(float(np.mean(power))),
        # [reward] energy cost per step, log(1 + 0.6 x kW). Lower is better (~0.03 at 50 W, ~1 at 3000 W)
        "energy_per_step": round(float(np.mean(energy_costs)), 3),
        # [reward] total reward per step. Higher is better; negative while it doesn't reach the target
        "reward_per_step": round(float(np.mean(step_rewards)), 3),
        # [brain + env layout] % of muscle commands at almost full force (strength > 0.95). Ideal below ~20%; rising toward 50%+ = clenched/jammed
        "pinned_pct": round(float(muscle_almost_full.mean()) * 100, 1),
        # [brain] typical size of the number before the Tanh. Healthy below ~1.5; above 2 the Tanh is near its flat edge
        "tanh_median": round(float(np.median(pre_tanh_sizes)), 2),
        # [brain] % of those numbers above 2 (Tanh saturated). Ideal below ~20%; above ~40% and rising = jamming
        "tanh_over2_pct": round(float((pre_tanh_sizes > 2).mean()) * 100, 1),
        # [brain] share of the learning signal that gets through the Tanh. 1 = all of it; below ~0.1 = training is blind
        "tanh_push": round(float(np.median(tanh_slope)), 2),
        # [brain + env layout] average sucker command, 0 off to 1 full. No ideal by itself: what matters is gripping when it's touching
        "sucker_cmd": round(float(sucker_grip_command.mean()), 2),
        # [env] real sucker pull on what it touches, newtons. Crawling needs grip; above ~17 N = holding more than its weight
        "grip_n": round(float(np.mean(grip)), 1),
        # [env] % of arm segments touching something. High = crawling on its arms, low = swimming or floating
        "touching_pct": round(float(np.mean(touching)) * 100, 1),
        # [brain] exploration (size of the random tries). 1.0 at birth, dropping slowly is normal; ~0.03 = stopped trying
        "exploration": round(float(brain.exploration.detach().exp().mean()), 2),
    }

    almost_full_by_muscle = {}
    for index, name in enumerate(MUSCLES):
        almost_full_by_muscle[name] = round(float(muscle_almost_full[:, :, :, index].mean()) * 100)

    print(f"checkpoint saved {row['checkpoint_saved']}, {episodes} episodes\n")
    print(f"GOAL      reached {row['reached']}/{episodes} | covered {row['covered_m']:+.2f} m on average (targets 3-6 m away)")
    print(f"BODY      flipped {row['flipped_pct']}% of the time | off the floor {row['air_pct']}% | segments touching {row['touching_pct']}%")
    print(f"JET       on {row['jet_pct']}% of the time | {row['jet_pulses']} pulses per episode | lowest stamina {row['lowest_stamina']}")
    print(f"ENERGY    {row['power_w']} W | energy cost {row['energy_per_step']} per step | reward {row['reward_per_step']} per step")
    print(f"MUSCLES   commands at almost full force {row['pinned_pct']}% | by muscle:", almost_full_by_muscle)
    print(f"SUCKERS   command {row['sucker_cmd']} (0 off, 1 full) | real grip {row['grip_n']} N (it weighs ~17 N in water)")
    print(f"TANH      before-Tanh median {row['tanh_median']} | over 2: {row['tanh_over2_pct']}% | push that gets through {row['tanh_push']}"
          "  (jammed if over 2 keeps rising and push drops toward 0)")
    print(f"EXPLORE   {row['exploration']} (1.0 at birth; ~0.03 = stopped trying new things)")

    new_file = not LOG.exists()
    with LOG.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    print(f"\nsaved to {LOG.name}")


if __name__ == "__main__":
    main()

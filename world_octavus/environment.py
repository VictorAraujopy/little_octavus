"""
Octavus environment in the Gymnasium format: the octopus has to walk to a target on the sea floor.

Each episode the octopus starts at the center and the target appears in a random direction.

Each arm is a soft tentacle (see build_octopus.py): 16 segments in 4 sections, and each section has 4 muscles
(bend up/down, bend sideways, twist, stretch/shorten) plus the suckers of its segments.

Observation (304 numbers):
    obs[:288].reshape(8, 4, 9) -> per arm, per section: where its 4 muscles are (-1 to 1 of their reach),
                                  how fast they move, and how hard the section touches something
    obs[288:]                  -> body: target (3), up (3), velocity (3), spin (3), height (1),
                                  siphon angle (1), mantle water (1), jet stamina (1)

Action (162 numbers between -1 and 1), the arms first and the siphon last:
    action[:160].reshape(8, 4, 5) -> per arm, per section: bend_up, bend_side, twist, stretch, sucker
                                     (one sucker number drives the suckers of all the section's segments)
    action[160], action[161]      -> siphon: siphon_aim (where the funnel points), jet (how hard it squirts)

The jet squirts the water in the mantle: a full jet empties it in 1 s, and it refills in 2 s while the jet rests.
Jetting also stops the octopus's systemic heart, so it tires: 3 s of full jet in total, back after 30 s of rest.

power is the metabolic cost in watts, what the food pays for: muscles pushing cost 4x their work (25% efficient),
muscles braking 1/1.2 of it, holding suckers nothing, the jet its hydrodynamic power at 25%, plus 2.7 W just to be alive.

The reward is not decided here: the trainer passes a function reward_fn(info) -> float,
and the environment hands over the facts of each step in the info dict:
distance, previous_distance, reached, flipped, action, previous_action, vertical_speed, height, spin, power (metabolic watts), tips_touching (0 to 1: arms whose last section touches something), airborne (nothing touching the floor), facing (1 = eyes pointing at the target, -1 = back to it), dt.

Watch the octopus moving randomly (on macOS the viewer needs mjpython):
    uv run mjpython world_octavus/environment.py
"""

import time
from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np

from world_octavus.build_octopus import SECTIONS, SEGMENTS, section_of

XML = Path(__file__).resolve().parent / "octopus.xml"
MUSCLES = ("bend_up", "bend_side", "twist", "stretch")


class OctopusEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    physics_steps = 5
    max_steps = 2000
    target_radius = 0.15
    target_distance = (3.0, 6.0)
    initial_noise = 0.1  # fraction of each joint's range
    # keep every observation number close to 1, so the brain's Tanh layers don't saturate
    muscle_speed_time = 0.1  # a muscle crossing its whole reach in 0.1 s reads 1
    touch_scale = 100.0  # a section gripping with its suckers presses up to ~90 N
    tip_touch = 0.005  # newtons: a resting tip lies on the floor with ~0.01 N, so half of that counts as touching
    spin_scale = 5.0
    jet_empty_time = 1.0
    jet_refill_time = 2.0
    jet_relaxed = 0.05
    jet_stamina_time = 3.0
    jet_recovery_time = 30.0
    funnel_radius = 0.02  # same as the siphon capsule in octopus.xml
    muscle_efficiency = 0.25
    braking_efficiency = 1.2
    basal_power = 2.7  # resting O2 use of a 32 kg octopus

    def __init__(self, reward_fn=None, render_mode=None, xml=XML):
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.reward_fn = reward_fn
        self.render_mode = render_mode
        self.viewer = None
        m = self.model

        self.torso = m.body("torso").id
        self.n_arms = sum(m.body(i).name.startswith("arm") for i in range(m.nbody))
        motor = {m.actuator(i).name: i for i in range(m.nu)}
        tendon = lambda name: m.tendon(name).id
        sensor = lambda name: m.sensor_adr[m.sensor(name).id]

        # the brain speaks per section: 4 muscles, then one number for all the suckers of that section's segments
        drives = []
        for a in range(self.n_arms):
            for k in range(SECTIONS):
                drives += [[motor[f"sec{k}_{kind}{a}"]] for kind in MUSCLES]
                drives.append([motor[f"seg{i}_sucker{a}"] for i in range(SEGMENTS) if section_of(i) == k])
        drives += [[motor["siphon_aim"]], [motor["jet"]]]
        self.action_of_motor = np.zeros(m.nu, dtype=int)
        for number, motors in enumerate(drives):
            self.action_of_motor[motors] = number
        self.n_actions = len(drives)

        self.section_muscles = np.array([[[tendon(f"sec{k}_{kind}{a}") for kind in MUSCLES]
                                          for k in range(SECTIONS)] for a in range(self.n_arms)])
        # a muscle's reach: how far its tendon goes when all its joints hit their limit
        joints = [m.wrap_objid[m.tendon_adr[t]:m.tendon_adr[t] + m.tendon_num[t]] for t in range(m.ntendon)]
        reach = np.array([np.abs(m.jnt_range[j]).max(axis=1).sum() for j in joints])
        self.muscle_reach = reach[self.section_muscles]
        # sums each segment's touch sensor into its section
        self.touch_to_section = np.zeros((self.n_arms * SECTIONS, m.nsensordata))
        for a in range(self.n_arms):
            for i in range(SEGMENTS):
                self.touch_to_section[a * SECTIONS + section_of(i), sensor(f"seg{i}_touch{a}")] = 1

        # every joint but the free one gets a start nudge proportional to its range (a slide only moves centimeters)
        joint_ids = np.arange(1, m.njnt)
        self.joint_qpos = m.jnt_qposadr[joint_ids]
        self.joint_noise = self.initial_noise * (m.jnt_range[joint_ids, 1] - m.jnt_range[joint_ids, 0]) / 2

        self.muscles = np.isin(m.actuator_trntype, [mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_TENDON])
        self.jet = m.actuator("jet").id
        self.max_thrust = m.actuator_gear[self.jet][2]
        self.siphon_qpos = m.jnt_qposadr[m.joint("siphon_aim").id]
        self.mantle_water = 1.0
        self.stamina = 1.0
        self.dt = m.opt.timestep * self.physics_steps
        self.target = np.zeros(2)

        self.action_space = gym.spaces.Box(-1.0, 1.0, (self.n_actions,), np.float32)
        observation_size = self._observe().size
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (observation_size,), np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.joint_qpos] += self.np_random.uniform(-1, 1, self.joint_noise.size) * self.joint_noise

        angle = self.np_random.uniform(0, 2 * np.pi)
        distance = self.np_random.uniform(*self.target_distance)
        self.target = distance * np.array([np.cos(angle), np.sin(angle)])
        self.data.mocap_pos[0][:2] = self.target

        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        self.distance = self._distance_to_target()
        self.previous_action = np.zeros(self.n_actions)
        self.mantle_water = 1.0
        self.stamina = 1.0
        return self._observe(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        low, high = self.model.actuator_ctrlrange.T
        self.data.ctrl[:] = low + (action[self.action_of_motor] + 1) / 2 * (high - low)
        self._squirt()
        mujoco.mj_step(self.model, self.data, nstep=self.physics_steps)
        self.step_count += 1

        previous_distance, self.distance = self.distance, self._distance_to_target()
        reached = self.distance < self.target_radius
        flipped = self.data.xmat[self.torso][8] < 0
        info = {
            "distance": self.distance,
            "previous_distance": previous_distance,
            "reached": reached,
            "flipped": flipped,
            "action": action,
            "previous_action": self.previous_action,
            "vertical_speed": self.data.qvel[2],
            "height": self.data.xpos[self.torso][2],
            "spin": self.data.qvel[5],
            "power": self._metabolic_power(),
            "tips_touching": (self._section_touch()[:, -1] > self.tip_touch).mean(),
            "airborne": self.data.ncon == 0,
            "facing": self._facing_target(),
            "dt": self.dt,
        }
        reward = self.reward_fn(info) if self.reward_fn else 0.0
        self.previous_action = action
        terminated = reached or flipped
        truncated = self.step_count >= self.max_steps

        if self.render_mode == "human":
            self.render()
        return self._observe(), reward, terminated, truncated, info

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = self.torso
            self.viewer.cam.distance = 3.0
            self.viewer.cam.elevation = -35
        self.viewer.sync()
        time.sleep(self.dt)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _metabolic_power(self):
        # arm and siphon muscles: work done pushing costs 1/0.25, work absorbed braking costs 1/1.2 (suckers hold for free)
        work = (self.data.actuator_force * self.data.actuator_velocity)[self.muscles]
        muscles = work.clip(min=0).sum() / self.muscle_efficiency - work.clip(max=0).sum() / self.braking_efficiency
        return muscles + self._jet_power() / self.muscle_efficiency + self.basal_power

    def _squirt(self):
        # the mantle is a pump: it squirts while it has water and only refills while the jet is relaxed.
        # Jetting also stops the heart, so a stamina runs out and only comes back while resting
        requested = self.data.ctrl[self.jet]
        drain = self.dt / self.jet_empty_time
        tire = self.dt / self.jet_stamina_time
        jet = min(requested, self.mantle_water / drain, self.stamina / tire)
        self.data.ctrl[self.jet] = jet
        # max(0, ...): float rounding would leave them at -1e-16
        self.mantle_water = max(0.0, self.mantle_water - jet * drain)
        self.stamina = max(0.0, self.stamina - jet * tire)
        if requested < self.jet_relaxed:
            self.mantle_water = min(1.0, self.mantle_water + self.dt / self.jet_refill_time)
            self.stamina = min(1.0, self.stamina + self.dt / self.jet_recovery_time)

    def _jet_power(self):
        # pushing water out: thrust T needs the water leaving the funnel at u = sqrt(T / (density * area)), costing T * u / 2
        thrust = self.data.ctrl[self.jet] * self.max_thrust
        area = np.pi * self.funnel_radius ** 2
        return thrust * np.sqrt(thrust / (self.model.opt.density * area)) / 2

    def _section_touch(self):
        return (self.touch_to_section @ self.data.sensordata).reshape(self.n_arms, SECTIONS)

    def _facing_target(self):
        # cosine of the angle between where the eyes point (the body's +x) and the target, on the floor plane
        rotation = self.data.xmat[self.torso].reshape(3, 3)
        seen = rotation.T @ (np.array([*self.target, 0.0]) - self.data.xpos[self.torso])
        return seen[0] / (np.hypot(seen[0], seen[1]) + 1e-8)

    def _distance_to_target(self):
        return np.linalg.norm(self.data.xpos[self.torso][:2] - self.target)

    def _observe(self):
        rotation = self.data.xmat[self.torso].reshape(3, 3)
        position = self.data.xpos[self.torso]
        target_3d = np.array([*self.target, 0.0])

        per_section = np.concatenate([
            self.data.ten_length[self.section_muscles] / self.muscle_reach,
            self.data.ten_velocity[self.section_muscles] * self.muscle_speed_time / self.muscle_reach,
            self._section_touch()[..., None] / self.touch_scale,
        ], axis=2)

        target_seen = rotation.T @ (target_3d - position)
        up_seen = rotation.T @ [0.0, 0.0, 1.0]
        torso_velocity = rotation.T @ self.data.qvel[:3]
        torso_spin = self.data.qvel[3:6] / self.spin_scale
        height = [position[2]]
        siphon = [self.data.qpos[self.siphon_qpos] / np.pi, self.mantle_water, self.stamina]

        return np.concatenate([
            per_section.ravel(),
            target_seen,
            up_seen,
            torso_velocity,
            torso_spin,
            height,
            siphon,
        ]).astype(np.float32)


if __name__ == "__main__":
    env = OctopusEnv(render_mode="human")
    observation, _ = env.reset(seed=0)
    print(f"the brain receives {observation.size} numbers and returns {env.action_space.shape[0]}")

    while env.viewer is None or env.viewer.is_running():
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            print(f"episode over: reached={info['reached']} flipped={info['flipped']} distance={info['distance']:.2f} m")
            env.reset()
    env.close()

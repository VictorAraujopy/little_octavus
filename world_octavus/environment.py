"""
Octavus environment in the Gymnasium format: the octopus has to walk to a target on the sea floor.

Each episode the octopus starts at the center and the target appears in a random direction.

Observation (104 numbers), grouped by arm so one network can be shared by all arms:
    obs[:88].reshape(8, 11) -> one row per arm: 5 angles, 5 velocities, 1 touch
    obs[88:]                -> body: target (3), up (3), velocity (3), spin (3), height (1),
                               siphon angle (1), mantle water (1), jet stamina (1)

Action (50 numbers between -1 and 1), the arms first and the siphon last:
    action[:48].reshape(8, 6) -> one row per arm: shoulder_swing, shoulder_lift, elbow, tip_bend, tip_curl, sucker
    action[48], action[49]    -> siphon: siphon_aim (where the funnel points), jet (how hard it squirts)

The jet squirts the water in the mantle: a full jet empties it in 1 s, and it refills in 2 s while the jet rests.
Jetting also stops the octopus's systemic heart, so it tires: 3 s of full jet in total, back after 30 s of rest.

power is the metabolic cost in watts, what the food pays for: muscles pushing cost 4x their work (25% efficient),
muscles braking 1/1.2 of it, holding suckers nothing, the jet its hydrodynamic power at 25%, plus 2.7 W just to be alive.

The reward is not decided here: the trainer passes a function reward_fn(info) -> float,
and the environment hands over the facts of each step in the info dict:
distance, previous_distance, reached, flipped, action, previous_action, vertical_speed, height, spin, power (metabolic watts), tips_touching (0 to 1), airborne (nothing touching the floor), facing (1 = eyes pointing at the target, -1 = back to it), dt.

Watch the octopus moving randomly (on macOS the viewer needs mjpython):
    uv run mjpython world_octavus/environment.py
"""

import time
from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np

XML = Path(__file__).resolve().parent / "octopus.xml"


class OctopusEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    physics_steps = 5
    max_steps = 2000
    target_radius = 0.15
    target_distance = (3.0, 6.0)
    initial_noise = 0.1
    # keep every observation number close to 1, so the brain's Tanh layers don't saturate
    joint_velocity_scale = 10.0
    touch_scale = 100.0
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

    def __init__(self, reward_fn=None, render_mode=None):
        self.model = mujoco.MjModel.from_xml_path(str(XML))
        self.data = mujoco.MjData(self.model)
        self.reward_fn = reward_fn
        self.render_mode = render_mode
        self.viewer = None

        self.torso = self.model.body("torso").id
        self.n_arms = sum(self.model.body(i).name.startswith("arm") for i in range(self.model.nbody))
        # MuJoCo lists the siphon's motors before the replicated arms; the brain gets the arms first, siphon last
        siphon_motors = [self.model.actuator("siphon_aim").id, self.model.actuator("jet").id]
        self.motor_order = [i for i in range(self.model.nu) if i not in siphon_motors] + siphon_motors
        self.motor_limits = self.model.actuator_ctrlrange[self.motor_order]
        self.muscles = self.model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT
        self.jet = self.model.actuator("jet").id
        self.max_thrust = self.model.actuator_gear[self.jet][2]
        siphon_joint = self.model.joint("siphon_aim").id
        self.siphon_qpos = self.model.jnt_qposadr[siphon_joint]
        self.siphon_qvel = self.model.jnt_dofadr[siphon_joint]
        self.mantle_water = 1.0
        self.stamina = 1.0
        self.dt = self.model.opt.timestep * self.physics_steps
        self.target = np.zeros(2)

        self.action_space = gym.spaces.Box(-1.0, 1.0, (self.model.nu,), np.float32)
        observation_size = self._observe().size
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (observation_size,), np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        n_joints = self.model.nq - 7
        self.data.qpos[7:] += self.np_random.uniform(-self.initial_noise, self.initial_noise, n_joints)

        angle = self.np_random.uniform(0, 2 * np.pi)
        distance = self.np_random.uniform(*self.target_distance)
        self.target = distance * np.array([np.cos(angle), np.sin(angle)])
        self.data.mocap_pos[0][:2] = self.target

        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        self.distance = self._distance_to_target()
        self.previous_action = np.zeros(self.model.nu)
        self.mantle_water = 1.0
        self.stamina = 1.0
        return self._observe(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        low, high = self.motor_limits.T
        self.data.ctrl[self.motor_order] = low + (action + 1) / 2 * (high - low)
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
            "tips_touching": (self.data.sensordata > 1.0).mean(),
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
        # joint muscles: work done pushing costs 1/0.25, work absorbed braking costs 1/1.2 (suckers hold for free)
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

        per_arm = np.concatenate([
            self.data.qpos[7:self.siphon_qpos].reshape(self.n_arms, -1),
            self.data.qvel[6:self.siphon_qvel].reshape(self.n_arms, -1) / self.joint_velocity_scale,
            self.data.sensordata.reshape(self.n_arms, -1) / self.touch_scale,
        ], axis=1)

        target_seen = rotation.T @ (target_3d - position)
        up_seen = rotation.T @ [0.0, 0.0, 1.0]
        torso_velocity = rotation.T @ self.data.qvel[:3]
        torso_spin = self.data.qvel[3:6] / self.spin_scale
        height = [position[2]]
        siphon = [self.data.qpos[self.siphon_qpos] / np.pi, self.mantle_water, self.stamina]

        return np.concatenate([
            per_arm.ravel(),
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

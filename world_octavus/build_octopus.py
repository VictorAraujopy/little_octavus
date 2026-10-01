"""
Builds octopus.xml: Octavus's body and the sea floor, in MuJoCo's MJCF format.
Edit this file, not the XML, then run:  uv run python world_octavus/build_octopus.py

Each arm is a soft tentacle: SEGMENTS pieces tapering to the tip. Every segment can bend (up/down and sideways),
twist and stretch, held by springs, so the arm sways in the water on its own. Like the octopus's muscle bands,
a muscle doesn't move one joint: it pulls a whole section of the arm (a fixed tendon over that section's joints).
Per section, 4 muscles (bend up/down, bend sideways, twist, stretch/shorten) and the suckers of its segments.

Real numbers used: an octopus arm stretches up to 70%, shortens ~20% and pulls with ~40 N.
Twist and how far one segment bends are estimates (no measured value found).
"""

import argparse
from pathlib import Path

import numpy as np

ARMS = 8
SEGMENTS = 16
SECTIONS = 4
ARM_LENGTH = 0.6
BASE_RADIUS = 0.045
TIP_RADIUS = 0.008
BEND = 45  # degrees per segment, both ways: 16 segments can curl the tip into a spiral
TWIST = 15  # degrees per segment
SHORTEN, STRETCH = 0.2, 0.7  # fraction of the segment's length
PULL = 40.0  # newtons, the longitudinal muscles at the base; thinner sections pull less
TWIST_SHARE = 0.25  # oblique muscles are a thin layer: a quarter of the bending strength
GRIP = 120.0  # newtons of sucker adhesion per arm, spread by sucker size
BUOYANCY = 0.952  # an octopus is ~5% denser than sea water: the water holds 1/1.05 of its weight
# the head carries the mantle, a sac full of sea water (which weighs nothing in water), so it is almost neutral:
# only the organs pull it down a little. With the arms' 0.952, lying sideways it nosedived (27 deg in 2 s; now 16)
MANTLE_BUOYANCY = 0.99
DAMPING = 0.05  # seconds: each spring's damping is its stiffness times this, so the arm doesn't ring

XML = Path(__file__).resolve().parent / "octopus.xml"


def segment_radius(i):
    return BASE_RADIUS + (TIP_RADIUS - BASE_RADIUS) * i / (SEGMENTS - 1)


def section_of(i):
    return i * SECTIONS // SEGMENTS


def joint(name, kind, axis, low, high, stiffness):
    kind_attr = ' type="slide"' if kind == "slide" else ""
    return (f'<joint name="{name}"{kind_attr} axis="{axis}" range="{low:.4g} {high:.4g}" '
            f'stiffness="{stiffness:.4g}" damping="{stiffness * DAMPING:.4g}"/>')


def arm():
    length = ARM_LENGTH / SEGMENTS
    torque = lambda i: PULL * segment_radius(i)  # muscle force x lever arm (the arm's radius)
    grips = np.array([segment_radius(i) for i in range(SEGMENTS)])
    grips = GRIP * grips / grips.sum()

    xml, close = "", ""
    for i in range(SEGMENTS):
        r = segment_radius(i)
        pos = "0 0 0" if i == 0 else f"{length:.4f} 0 0"
        tilt = ' euler="0 25 0"' if i == 0 else ""  # the base slopes down so the arm reaches the floor
        material = "skin" if section_of(i) < 2 else "tip"
        bend_k = torque(i) / np.radians(BEND)
        twist_k = TWIST_SHARE * torque(i) / np.radians(TWIST)
        stretch_k = PULL * r / BASE_RADIUS / (STRETCH * length)
        xml += f"""
<body name="seg{i}_" pos="{pos}"{tilt} gravcomp="{BUOYANCY}">
  {joint(f"seg{i}_stretch", "slide", "1 0 0", -SHORTEN * length, STRETCH * length, stretch_k)}
  {joint(f"seg{i}_bend_up", "hinge", "0 1 0", -BEND, BEND, bend_k)}
  {joint(f"seg{i}_bend_side", "hinge", "0 0 1", -BEND, BEND, bend_k)}
  {joint(f"seg{i}_twist", "hinge", "1 0 0", -TWIST, TWIST, twist_k)}
  <geom material="{material}" type="capsule" fromto="0 0 0 {length:.4f} 0 0" size="{r:.4f}"/>
  <geom class="visual" material="sucker" type="sphere" size="{0.4 * r:.4f}" pos="{length / 2:.4f} 0 {-0.85 * r:.4f}"/>
  <site name="seg{i}_touch" type="capsule" fromto="0 0 0 {length:.4f} 0 0" size="{1.2 * r:.4f}" rgba="0 0 0 0"/>"""
        close += "</body>\n"

    tendons, muscles, suckers, sensors = "", "", "", ""
    for k in range(SECTIONS):
        segs = [i for i in range(SEGMENTS) if section_of(i) == k]
        lever = np.mean([segment_radius(i) for i in segs])
        strength = {"bend_up": PULL * lever, "bend_side": PULL * lever,
                    "twist": TWIST_SHARE * PULL * lever, "stretch": PULL * lever / BASE_RADIUS}
        for kind, gear in strength.items():
            joints = "".join(f'<joint joint="seg{i}_{kind}" coef="1"/>' for i in segs)
            tendons += f'<fixed name="sec{k}_{kind}">{joints}</fixed>\n'
            # shortening only gets the force its spring holds at -20%, or it would push past the limit
            low = -SHORTEN / STRETCH if kind == "stretch" else -1
            muscles += f'<motor name="sec{k}_{kind}" tendon="sec{k}_{kind}" ctrlrange="{low:.3g} 1" gear="{gear:.4g}"/>\n'
    for i in range(SEGMENTS):
        suckers += f'<adhesion name="seg{i}_sucker" body="seg{i}_" ctrlrange="0 1" gain="{grips[i]:.4g}"/>\n'
        sensors += f'<touch name="seg{i}_touch" site="seg{i}_touch"/>\n'
    return xml + close, tendons, muscles + suckers, sensors


def octopus():
    arm_bodies, tendons, actuators, sensors = arm()
    return f"""<!--
  Generated by build_octopus.py: edit that file, not this one.
  Octavus's body and the sea floor. {ARMS} soft arms of {SEGMENTS} segments in {SECTIONS} sections, and a siphon.
  View it (from the project root; the path must be absolute, hence $PWD):
    uv run python -m mujoco.viewer --mjcf="$PWD/world_octavus/octopus.xml"
-->
<mujoco model="octavus">
  <compiler angle="degree" autolimits="true"/>

  <!-- sea water: Earth's gravity, and each body's gravcomp is the buoyancy (MuJoCo's fluid only adds drag).
       Euler + sparse: the soft arms have ~500 joints, and this integrator is ~7x faster for a body shaped like a tree -->
  <option timestep="0.005" integrator="Euler" jacobian="sparse" gravity="0 0 -9.81" density="1025" viscosity="0.001"/>

  <visual>
    <rgba haze="0.05 0.25 0.35 1" fog="0.03 0.18 0.26 1"/>
    <map fogstart="1.5" fogend="7"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <texture name="water" type="skybox" builtin="gradient" rgb1="0.1 0.4 0.55" rgb2="0.01 0.06 0.1" width="512" height="512"/>
    <texture name="sand" type="2d" builtin="checker" rgb1="0.78 0.71 0.52" rgb2="0.72 0.65 0.47" width="512" height="512"/>
    <material name="sand" texture="sand" texrepeat="12 12"/>
    <!-- skin with darker specks; cube textures wrap around round shapes -->
    <texture name="skin" type="cube" builtin="flat" rgb1="0.8 0.32 0.24" rgb2="0.8 0.32 0.24" mark="random" markrgb="0.55 0.18 0.13" random="0.06" width="256" height="256"/>
    <material name="skin" texture="skin" rgba="1 1 1 1" specular="0.3" shininess="0.4"/>
    <texture name="tip" type="cube" builtin="flat" rgb1="0.88 0.42 0.32" rgb2="0.88 0.42 0.32" mark="random" markrgb="0.62 0.25 0.18" random="0.05" width="256" height="256"/>
    <material name="tip" texture="tip" rgba="1 1 1 1" specular="0.3" shininess="0.4"/>
    <material name="sucker" rgba="1 0.8 0.72 1"/>
    <material name="eye" rgba="0.86 0.68 0.3 1" specular="0.8" shininess="0.9"/>
    <material name="pupil" rgba="0.05 0.05 0.05 1"/>
  </asset>

  <default>
    <!-- slippery skin (mucus on wet sand): the suckers do the gripping. The floor inherits it: MuJoCo uses the larger of the two.
         fluidshape ellipsoid: water forces for long thin shapes (drag sideways vs lengthwise, lift, the water it drags along) -->
    <geom material="skin" contype="2" conaffinity="1" friction="0.3 0.02 0.01" fluidshape="ellipsoid"/>
    <joint armature="0.01"/>
    <default class="visual">
      <geom contype="0" conaffinity="0" density="0" fluidshape="none"/>
    </default>
  </default>

  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" directional="true" diffuse="0.55 0.7 0.8" castshadow="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="sand" contype="1" conaffinity="1" fluidshape="none"/>

    <!-- a ball where the head has to get: it rests on the floor at height 0 and floats up as the targets rise -->
    <body name="target" mocap="true" pos="2 0 0.125">
      <geom class="visual" type="sphere" size="0.12" rgba="1 0.25 0.2 0.7"/>
    </body>

    <body name="torso" pos="0 0 0.19" gravcomp="{MANTLE_BUOYANCY}">
      <freejoint name="root"/>
      <camera name="follow" mode="trackcom" pos="0 -1.8 1.1" xyaxes="1 0 0 0 0.53 0.848"/>
      <geom name="head" type="sphere" size="0.12"/>
      <!-- octopus mantle, overlapping the head sphere so both read as one smooth egg (not two stacked balls).
           Arms are 3-5x its length: ~20 cm for 60 cm arms (the old 40 cm one looked like a squid) -->
      <geom name="mantle" class="visual" type="ellipsoid" size="0.13 0.123 0.165" pos="-0.035 0 0.055" euler="0 -28 0"/>
      <!-- the web only fills between the arm roots: past them it showed as a plate under the arms, or a stub between them -->
      <geom name="web" class="visual" type="ellipsoid" size="0.12 0.12 0.035" pos="0 0 -0.06"/>
      <!-- amber eyes half sunk into the sides of the head, with the octopus's horizontal slit pupil -->
      <geom class="visual" material="eye" type="sphere" size="0.034" pos="0.01 0.1 0.05"/>
      <geom class="visual" material="eye" type="sphere" size="0.034" pos="0.01 -0.1 0.05"/>
      <geom class="visual" material="pupil" type="ellipsoid" size="0.019 0.003 0.006" pos="0.013 0.1294 0.0647" euler="26.6 0 0"/>
      <geom class="visual" material="pupil" type="ellipsoid" size="0.019 0.003 0.006" pos="0.013 -0.1294 0.0647" euler="-26.6 0 0"/>

      <replicate count="{ARMS}" euler="0 0 {360 / ARMS:g}">
        <body name="arm" pos="0.0924 0.0383 -0.05" euler="0 0 22.5">
{arm_bodies}
        </body>
      </replicate>

      <!-- the siphon (funnel): swivels around the head (aim) and tips up or down (tilt), and squirts the mantle's water.
           At rest it points 20 degrees down; the tilt adds up to 45 degrees either way, so like a real octopus it can
           steer the jet up or down (a rudder to keep the head from nosediving). No collision: it only aims the jet -->
      <body name="siphon" pos="0 0 -0.03" gravcomp="{BUOYANCY}">
        <joint name="siphon_aim" axis="0 0 1" range="-180 180" damping="0.5"/>
        <joint name="siphon_tilt" axis="0 1 0" range="-45 45" damping="0.5"/>
        <!-- the siphon's mass (what the aim and tilt muscles move), kept as it was but invisible -->
        <geom type="capsule" fromto="0.08 0 0 0.16 0 -0.029" size="0.02" contype="0" conaffinity="0" fluidshape="none" rgba="0 0 0 0"/>
        <!-- what shows: just the funnel's mouth, barely out from under the head -->
        <geom class="visual" material="skin" type="capsule" fromto="0.095 0 0 0.108 0 -0.005" size="0.009"/>
        <!-- the jet pushes the body opposite to where the water leaves: backwards and a bit up -->
        <site name="jet" zaxis="-0.94 0 0.342" size="0.01" rgba="0 0 0 0"/>
      </body>
    </body>
  </worldbody>

  <tendon>
{tendons}  </tendon>

  <actuator>
{actuators}    <position name="siphon_aim" joint="siphon_aim" kp="8" kv="0.3" forcerange="-4 4" inheritrange="1"/>
    <position name="siphon_tilt" joint="siphon_tilt" kp="8" kv="0.3" forcerange="-4 4" inheritrange="1"/>
    <!-- up to 60 N, 4x its weight in water: one full squirt lifts it off. The environment stops it when the mantle is empty -->
    <motor name="jet" site="jet" ctrlrange="0 1" gear="0 0 60 0 0 0"/>
  </actuator>

  <sensor>
{sensors}  </sensor>
</mujoco>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=XML)
    out = parser.parse_args().out
    out.write_text(octopus())
    print(f"wrote {out}")

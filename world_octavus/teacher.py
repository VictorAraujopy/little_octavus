"""
The teacher: an octopus driven by a few hand-written rules, no learning, that gets to the target the way a common
octopus does: it crawls, and swims by jet when the target is far.

Crawling, the way Levy, Flash & Hochner (2015) measured it in the common octopus: "the simple pushing-by-elongation
mechanism by which the arms create the crawling thrust"; the octopus "chooses in a moment-to-moment fashion which arms
to recruit for pushing", and the direction is "a vectorial summation of the pushing directions of the active arms"
(Levy & Hochner 2017). Every arm whose push has some component toward the target joins in (the back and the side
arms): it grips the floor with the suckers where it touches and stretches its base, pushing the body away from that
grip; once the base is nearly stretched out it lets go and shortens back, then grips again. There is no shared rhythm,
as in the octopus. Each arm decides only from what the octopus feels (how stretched its base is, and whether its
middle is pressing the floor, which it only does while gripping), so a brain that sees the same observation can copy it.
Meanwhile no arm is ever still, like a living octopus: each one holds its place around the body (the more its base
is bent or twisted away from it, the harder it comes back; after a swim this opens the arms out again), its middle sways gently, and its outer half sways and its tip curls,
each like a pendulum driven by what it feels: bend one way until a limit, then the other way.

Swimming: the funnel points straight away from the target, so the water leaves that way and pushes the body toward
it, tipped down so the jet pushes level instead of lifting it off, with the arms trailing behind and their tips curling. It swims while the target is
farther than SWIM_BEYOND and it still has breath, and crawls the rest of the way. If it tips over mid-swim, the jet
eases off (full up to ~32 degrees, none by ~50) while it rights itself: tilted, the jet would roll it further.

Righting, blended in while crawling and swimming: past ~32 degrees of tilt (it feels which way is up), righting takes over from
holding posture and pushing, fully by ~50 degrees: the arms on the side that is down straighten their base firmly,
sticking out on the side it is falling to, and the others rest, until it is upright. Pressing those bases down onto the
floor instead folded the arms under the body, which then sat on them, tilted: after a messed-up start it lay at 37-50
degrees for whole episodes (19 of 60 missed); straightening them, 4 of 60 missed, all of them only for lack of time.
Below ~32 degrees it doesn't right itself at all: on a slightly tilted body, pressing "down" folded those arms under it
(3 arms trapped there 97% of the time after a swim).

Watch it (on macOS the viewer needs mjpython):  uv run mjpython world_octavus/teacher.py
It goes through every scenario in turn (listed when it starts); add a scenario's number to repeat only that one.
A messed-up start is 0-3 s of random commands before the teacher takes over, so it starts tilted, with its arms swept
or its funnel turned. It reaches every target, from a normal start and after a messed-up one (20 of each per distance:
close, up to 3 m and 3-6 m). Far targets take up to ~63 s: one breath swims it ~3.5 m and it crawls the rest at ~5.5 cm/s
(a common octopus averages 9 cm/s, Huffard 2006 citing Wells; here the base stretches at most 0.36 of its length per
second, Zullo 2022, and pushing harder didn't make it faster), which is why an episode lasts 75 s.
"""

import itertools
import sys

import numpy as np

from world_octavus.environment import OctopusEnv

# in the observation (see environment.py)
ARMS = slice(0, 288)  # reshaped to (8 arms, 4 sections, 9): muscle positions, muscle speeds, touch
STRETCH, TOUCH = 3, 8  # per section: how stretched the stretch muscle is, and how hard it presses on something
TARGET = slice(288, 291)  # where the target is, seen from the body (forward, left, up)
UP = slice(291, 294)  # which way is up, felt in the body's own frame
FUNNEL_TILT, STAMINA = 302, 304  # how tipped the funnel is (-1 all the way down) and jet stamina, 0 to 1
# in the action
STRETCH_COMMAND, SUCKER_COMMAND = 3, 4  # per section, in action[:160].reshape(8, 4, 5)
SIPHON_AIM, SIPHON_TILT, JET = 160, 161, 162

ARM_ANGLES = np.radians(22.5 + 45 * np.arange(8))  # where each arm sits around the body
PUSHING_ARMS = 90.0  # degrees from "away from the target": every arm whose push helps at all
PUSH_SECTION, GRIP_SECTIONS = 0, [1, 2]  # the base stretches; the middle, which lies on the floor, grips
GRIPPING = 0.01  # touch reading (1 = 100 N) above which the middle is pressing the floor: it is mid-push
BEND_UP, BEND_SIDE, TWIST = 0, 1, 2  # the first three muscles of a section, in the observation and in the action
POSTURE = 1.5  # how hard the base bends and twists back toward its place around the body
MIDDLE_SWAY, OUTER_SWAY, TIP_CURL = 0.3, 0.5, 0.5  # how far each part bends before turning back (1 = every joint at 45 deg)
GENTLE, FIRMER = 0.018, 0.03  # how hard they bend (small: the springs are soft and the tips weigh grams); the outer half a bit firmer
SWAY_BRAKE = 0.8  # the faster a part is already moving, the less it is pushed: wide, slow curls instead of whipping
ARM_VARIETY = 0.03  # each arm turns back a little later than the one before, so they don't move in lockstep
STILL = 0.02  # muscle speed below which it counts as still: going by the sign of a tiny speed made the arms tremble
STRETCHED, SHORT = 0.6, 0.05  # how stretched the base is when a push ends, and when the recovery ends
PUSH_FORCE, RECOVER_FORCE = 0.6, 0.3  # gentle: at full force each stroke jerked the arm and its tip whipped (it looked like flickering)
SWIM_BEYOND = 1.0  # meters, in 3D (tilting the body doesn't change it): a swim goes on until the target is this close
TAKE_OFF_BEYOND = 1.3  # a crawling or resting octopus only takes off for a target farther than this: no back and forth
SWIMMING_TILT = -0.5  # a funnel tipped down past this means it is mid-swim (it relaxes back while crawling)
TAKE_OFF_BREATH, SWIM_BREATH = 0.3, 0.05  # stamina to start a swim, and to keep one going: tired, it doesn't twitch
TILT = -1.0  # funnel tipped all the way down: the jet pushes level
RIGHTING_FROM, RIGHTING_RAMP = 0.15, 0.2  # tilt (0 upright, 1 on its side) where righting starts (~32 deg) and how far on it is full (~50 deg)
STRAIGHTEN = 4.0  # how firmly the arms on the low side straighten their base while righting (POSTURE is the gentle everyday hold)


def crawl(obs):
    arms = obs[ARMS].reshape(8, 4, 9)
    target = obs[TARGET]
    away = np.arctan2(-target[1], -target[0])  # the direction opposite the target: where a pushing arm points
    action = np.zeros(163)  # everything relaxed (0 = no force) and the jet off
    action[JET] = -1.0
    commands = action[:160].reshape(8, 4, 5)  # a view: writing here writes into action
    for arm in range(8):
        if np.cos(ARM_ANGLES[arm] - away) <= np.cos(np.radians(PUSHING_ARMS)):
            continue  # its push would not help: stays relaxed
        stretch = arms[arm, PUSH_SECTION, STRETCH]
        gripping = arms[arm, GRIP_SECTIONS, TOUCH].max() > GRIPPING
        # grip and push until the base is nearly stretched out; let go and shorten until it is short, then grip again
        pushing = stretch < SHORT or (stretch < STRETCHED and gripping)
        commands[arm, GRIP_SECTIONS, SUCKER_COMMAND] = 1.0 if pushing else -1.0
        commands[arm, PUSH_SECTION, STRETCH_COMMAND] = PUSH_FORCE if pushing else -RECOVER_FORCE
    keep_alive(arms, commands, obs[UP])
    return action


def sway(arms, arm, section, muscle, limit, force):
    # a pendulum driven by what the arm feels: bend one way until the limit, then the other way
    bent = arms[arm, section, muscle]
    moving = arms[arm, section, 4 + muscle]  # the muscle speeds come right after the 4 positions
    if bent < -limit or moving > STILL:
        going_up = bent < limit
    elif bent > limit or moving < -STILL:
        going_up = False
    else:
        going_up = bent < 0  # nearly still (stuck on the floor, or turning): head for the far side, no flickering
    push = force if going_up else -force
    return np.clip(push - SWAY_BRAKE * moving, -abs(force), abs(force))


def keep_tips_alive(arms, commands):
    for arm in range(8):
        variety = ARM_VARIETY * arm
        commands[arm, 2, BEND_SIDE] = sway(arms, arm, 2, BEND_SIDE, 0.8 * OUTER_SWAY + variety, FIRMER)
        commands[arm, 3, BEND_SIDE] = sway(arms, arm, 3, BEND_SIDE, OUTER_SWAY + variety, FIRMER)
        commands[arm, 3, BEND_UP] = sway(arms, arm, 3, BEND_UP, TIP_CURL, GENTLE)


def how_much_righting(up):
    tilt = 1.0 - up[2]  # 0 upright, 1 on its side, 2 upside down
    return np.clip((tilt - RIGHTING_FROM) / RIGHTING_RAMP, 0.0, 1.0)  # how much righting takes over, 0 to 1


def straighten(arms, arm, up):
    # tilted, an arm on the side that is down straightens its base firmly (see the docstring: pressing it down folded it
    # under the body); the arms on the other side rest
    low_side = np.arctan2(-up[1], -up[0])  # the side of the body that is down
    if np.cos(ARM_ANGLES[arm] - low_side) <= 0.3:
        return 0.0
    return np.clip(-STRAIGHTEN * arms[arm, 0, BEND_UP], -1.0, 1.0)


def keep_alive(arms, commands, up):
    righting = how_much_righting(up)
    for arm in range(8):
        for muscle in (BEND_SIDE, TWIST):
            commands[arm, 0, muscle] = (1 - righting) * np.clip(-POSTURE * arms[arm, 0, muscle], -0.3, 0.3)
        # the base holds its place; tilted, an arm on the low side presses down instead (the others rest)
        hold = np.clip(-POSTURE * arms[arm, 0, BEND_UP], -0.3, 0.3)
        commands[arm, 0, BEND_UP] = (1 - righting) * hold + righting * straighten(arms, arm, up)
        commands[arm, 1, BEND_SIDE] = sway(arms, arm, 1, BEND_SIDE, MIDDLE_SWAY, GENTLE)
        # pushing fades out too: on its side a push gets it nowhere
        commands[arm, :, STRETCH_COMMAND] *= 1 - righting
        commands[arm, :, SUCKER_COMMAND] = np.where(righting > 0.5, -1.0, commands[arm, :, SUCKER_COMMAND])
    keep_tips_alive(arms, commands)


def swim(obs):
    target = obs[TARGET]
    action = np.zeros(163)  # arms trailing behind (holding them open would make a parachute of the web)
    arms, commands = obs[ARMS].reshape(8, 4, 9), action[:160].reshape(8, 4, 5)
    righting = how_much_righting(obs[UP])
    for arm in range(8):
        # only the tips curl in and out: swaying the outer half sideways while jetting slowed it down
        commands[arm, 3, BEND_UP] = sway(arms, arm, 3, BEND_UP, TIP_CURL, GENTLE)
        commands[arm, 0, BEND_UP] = righting * straighten(arms, arm, obs[UP])  # tipped over, it rights itself as when crawling
    # the funnel points the opposite way from the target, so the jet pushes the body toward it
    action[SIPHON_AIM] = np.arctan2(-target[1], -target[0]) / np.pi  # -1..1 is -180..180 degrees
    action[SIPHON_TILT] = TILT
    # the jet eases off as it tips over: tilted, it would roll it further. A tilt limit here flickered instead:
    # jet on, it tipped, jet off, it righted itself, jet on, every 0.3 s
    action[JET] = 1.0 - righting
    return action


def teacher_action(obs):
    # mid-swim the funnel stays tipped down; the mantle itself goes full for an instant between squeezes, which
    # looked like resting and made it land with breath to spare
    swimming = obs[FUNNEL_TILT] < SWIMMING_TILT
    far = np.linalg.norm(obs[TARGET]) > (SWIM_BEYOND if swimming else TAKE_OFF_BEYOND)
    breath = obs[STAMINA] > (SWIM_BREATH if swimming else TAKE_OFF_BREATH)
    if far and breath:
        return swim(obs)
    return crawl(obs)


MESS_SECONDS = 3.0  # a messed-up start: up to this long of random commands before the teacher takes over

SCENARIOS = [  # what it shows, the farthest target (m), and whether it starts messed up
    ("close: crawls", 1.5, False),
    ("middle: swims, then crawls", 3.0, False),
    ("far: the real task, 3-6 m", 6.0, False),
    ("close, after a messed-up start", 1.5, True),
    ("far, after a messed-up start", 6.0, True),
]


def mess_up(env, obs, rng):
    # random commands for 0 to 3 s: it ends up tilted, with its arms swept or its funnel turned, like a brain still learning
    steps = rng.integers(0, int(MESS_SECONDS / env.dt))
    for _ in range(steps):
        obs, *_ = env.step(rng.uniform(-1.0, 1.0, 163))
    return obs, steps * env.dt


def watching(env):
    return env.viewer is None or env.viewer.is_running()  # None: the window only opens on the first step


def run_scenario(env, rng, name, farthest, messy):
    env.farthest = farthest
    obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
    print(f"{name}: target {np.linalg.norm(obs[TARGET]):.1f} m away", flush=True)
    if messy:
        obs, seconds = mess_up(env, obs, rng)
        print(f"  messed up for {seconds:.1f} s, now the teacher drives", flush=True)
    steps, done = 0, False
    while not done and watching(env):
        obs, _, terminated, truncated, info = env.step(teacher_action(obs))
        steps += 1
        done = terminated or truncated
    if done:
        result = "REACHED the target" if info["reached"] else f"time's up, {info['distance']:.2f} m from the target"
        print(f"  {result} after {steps * env.dt:.1f} s", flush=True)


if __name__ == "__main__":
    for number, (name, _, _) in enumerate(SCENARIOS, start=1):
        print(f"{number}. {name}")
    if len(sys.argv) > 1:
        chosen = [SCENARIOS[int(sys.argv[1]) - 1]]  # only the scenario with that number
    else:
        chosen = SCENARIOS
    env = OctopusEnv(render_mode="human", curriculum=False)
    rng = np.random.default_rng()
    for name, farthest, messy in itertools.cycle(chosen):
        if not watching(env):
            break
        run_scenario(env, rng, name, farthest, messy)
    env.close()

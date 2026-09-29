import math

def reward(info):
    progress = (info["previous_distance"] - info["distance"]) / info["dt"]
    #dt = time for step
    #speed only pays up to an octopus pace (0.3 m/s): going faster earns nothing extra,
    #so the only way to score more is to get there spending less energy
    progress = min(progress, 0.3)
    
    jerk = ((info["action"] - info["previous_action"]) **2).mean()
    #bouncing up and down costs points (vertical speed squared)
    bounce = info["vertical_speed"] ** 2
    #lifting the body high like a spider costs points: an octopus crawls low
    #0.203 = 2 cm above the resting height in sea water (0.183), so small movements are free
    lift = max(0.0, info["height"] - 0.203)
    #spinning costs a little (it now needs to turn to face the target), capped so a flailing newborn
    #isn't fined into standing still
    turn = min(info["spin"] ** 2, 0.5)
    #the octopus has a front now (where the eyes point): not facing the target costs points
    #0 when looking straight at it, 1 when it's to the side, 2 when it's behind
    #by the angle, not the cosine: near 180 the cosine barely changes, so turning from behind earned nothing
    not_facing = math.acos(info["facing"]) / (math.pi / 2)
    #muscle effort costs points: flapping the tentacles fast or slamming them wastes energy
    #real animals move the way that spends the least energy, that's what makes them look natural
    energy = info["power"] / 1000  #watts -> kilowatts
    #tips in the air cost points: an octopus crawls by pushing and pulling with its tips on the floor
    #(a penalty, not a bonus: pressing the tips down while standing still earns nothing)
    tips_up = 1.0 - info["tips_touching"]
    score = progress - 0.05 * jerk - 1.0 * bounce - 0.5 * lift - 0.05 * turn - 0.1 * energy - 0.1 * tips_up - 0.05 * not_facing
    
    #flying costs points: when nothing touches the floor it is hopping, and an octopus never leaves the ground
    if info["airborne"]:
        score -= 0.3

    if info["flipped"]:
        score -= 10
        
    if info["reached"]:
        score += 10
    
    return score
    
    
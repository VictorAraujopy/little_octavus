import math

def reward(info):
    progress = (info["previous_distance"] - info["distance"]) / info["dt"]
    #dt = time for step
    #no speed cap: going faster is paid for with metabolic energy,
    #so a jet burst is only used if the time it saves is worth what it costs

    jerk = ((info["action"] - info["previous_action"]) **2).mean()
    #no rules for height, bouncing or leaving the floor: in water swimming is allowed,
    #and the metabolic energy (plus the jet's stamina) decides if it's worth it
    #spinning costs a little (it now needs to turn to face the target), capped so a flailing newborn
    #isn't fined into standing still
    turn = min(info["spin"] ** 2, 0.5)
    #the octopus has a front now (where the eyes point): not facing the target costs points
    #0 when looking straight at it, 1 when it's to the side, 2 when it's behind
    #by the angle, not the cosine: near 180 the cosine barely changes, so turning from behind earned nothing
    not_facing = math.acos(info["facing"]) / (math.pi / 2)
    #muscle effort costs points: flapping the tentacles fast or slamming them wastes energy
    #real animals move the way that spends the least energy, that's what makes them look natural
    #power is metabolic (what the food pays). 0.6 per kW: an efficient move (~55 W at 0.16 m/s) costs ~20% of its progress
    #capped at 1.0 (~1700 W): only a flailing newborn (~3300 W) hits it, so it isn't fined into standing still
    energy = min(0.6 * info["power"] / 1000, 1.0)
    #tips in the air cost points: an octopus crawls by pushing and pulling with its tips on the floor
    #(a penalty, not a bonus: pressing the tips down while standing still earns nothing)
    tips_up = 1.0 - info["tips_touching"]
    score = progress - 0.05 * jerk - 0.05 * turn - energy - 0.1 * tips_up - 0.05 * not_facing

    if info["flipped"]:
        score -= 10
        
    if info["reached"]:
        score += 10
    
    return score
    
    
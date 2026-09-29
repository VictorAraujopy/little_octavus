import math

def reward(info):
    progress = (info["previous_distance"] - info["distance"]) / info["dt"]
    #dt = time for step
    #no speed cap: going faster is paid for with metabolic energy,
    #so a jet burst is only used if the time it saves is worth what it costs

    jerk = ((info["action"] - info["previous_action"]) **2).mean()
    #no rules for height, bouncing or leaving the floor: in water swimming is allowed,
    #and the metabolic energy (plus the jet's stamina) decides if it's worth it
    #spinning costs a little (spinning in place was an old cheat), capped so a flailing newborn
    #isn't fined into standing still
    turn = min(info["spin"] ** 2, 0.5)
    #no rule about facing the target: a real octopus crawls any way and swims mantle-first,
    #and here it knows where the target is without looking at it
    #muscle effort costs points: flapping the tentacles fast or slamming them wastes energy
    #real animals move the way that spends the least energy, that's what makes them look natural
    #power is metabolic (what the food pays). 0.6 per kW: an efficient move (~55 W at 0.16 m/s) costs ~20% of its progress
    #log instead of a cap: a flailing newborn still pays only ~1 per step, but more power always costs more
    #(the old cap at 1.0 made everything above ~1700 W free, and it learned to burn 3000 W)
    energy = math.log(1 + 0.6 * info["power"] / 1000)
    #tips in the air cost points: an octopus crawls by pushing and pulling with its tips on the floor
    #(a penalty, not a bonus: pressing the tips down while standing still earns nothing)
    tips_up = 1.0 - info["tips_touching"]
    score = progress - 0.05 * jerk - 0.05 * turn - energy - 0.1 * tips_up

    #no extra cost for flipping over, and it no longer ends the episode: upside down it can't crawl,
    #so the normal costs keep running until it rights itself

    if info["reached"]:
        score += 10
    
    return score
    
    
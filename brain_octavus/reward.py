
def reward(info):
    progress = (info["previous_distance"] - info["distance"]) / info["dt"]
    #dt = time for step
    
    jerk = ((info["action"] - info["previous_action"]) **2).mean()
    #bouncing up and down costs points (vertical speed squared)
    bounce = info["vertical_speed"] ** 2
    #lifting the body high like a spider costs points: an octopus crawls low
    #0.19 = a bit above the resting height (0.173), so small movements are free
    lift = max(0.0, info["height"] - 0.19)
    score = progress - 0.05 * jerk - 0.1 * bounce - 0.5 * lift
    
    if info["flipped"]:
        score -= 10
        
    if info["reached"]:
        score += 10
    
    return score
    
    
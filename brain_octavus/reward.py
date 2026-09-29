
def reward(info):
    progress = (info["previous_distance"] - info["distance"]) / info["dt"]
    #dt = time for step
    
    jerk = ((info["action"] - info["previous_action"]) **2).mean()
    #bouncing up and down costs points (vertical speed squared)
    bounce = info["vertical_speed"] ** 2
    score = progress - 0.05 * jerk - 0.1 * bounce
    
    if info["flipped"]:
        score -= 10
        
    if info["reached"]:
        score += 10
    
    return score
    
    
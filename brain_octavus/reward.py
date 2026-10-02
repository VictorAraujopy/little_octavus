def reward(info):
    progress = (info["previous_distance"] - info["distance"]) / info["dt"]
    #dt = time for step
    #no speed cap: going faster is paid for with metabolic energy,
    #so a jet burst is only used if the time it saves is worth what it costs
    #x10: getting to food pays back far more energy than the trip costs, so moving toward it is worth spending on
    #(at x1 its progress was ~4% of its energy cost and it ignored it; moving back and forth still nets 0)
    progress = 10 * progress

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
    #power is metabolic (what the food pays), per kg of octopus: the same effort costs the same for any body size
    #straight line: every W/kg costs the same, like food paid per joule. The old log went flat at high power
    #(at 580 W/kg, saving 100 W/kg was worth only 0.16), so flailing light arms barely cost more than calm ones
    #0.013: jetting at 1 m/s (~190 W/kg) costs ~25% of its progress, crawling like a real octopus ~1%,
    #a flailing newborn (~420 W/kg) ~5.5 per step; saving 100 W/kg is worth 1.3
    #(the old cap at 1.0 made everything above ~1700 W free, and it learned to burn 3000 W)
    #the cost grows with the curriculum: 20% with the first targets, 100% at the real 3-6 m
    #paying it all from birth, it never learned to move: random moves get nowhere, so the only thing left
    #to improve was energy, and it curled into a ball with the jet off. First learn to get there, then to save
    energy_share = 0.2 + 0.8 * info["curriculum_level"]
    watts_per_kg = info["power"] / info["mass"]
    energy = energy_share * 0.013 * watts_per_kg
    #rigidity costs in full from birth: holding the same muscle command for a long time, not each contraction
    #(charging every contraction in full taught a newborn that moving its arms at all was too expensive)
    #0.9: rigid at 0.65 like it was costs ~0.38 per step, light tone at 0.3 ~0.08, contracting and releasing ~0.01
    rigidity = 0.9 * info["rigidity"]
    #no rule about keeping the tips on the floor: in water swimming is a healthy octopus move
    #(it swam all the time, so that cost was a fixed tax that never changed and taught nothing)
    score = progress - 0.05 * jerk - 0.05 * turn - energy - rigidity

    #no extra cost for flipping over, and it no longer ends the episode: upside down it can't crawl,
    #so the normal costs keep running until it rights itself

    if info["reached"]:
        score += 10
    
    return score
    
    
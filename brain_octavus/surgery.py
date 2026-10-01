import torch
from brain_octavus.brain import Octavus_arms_brain

old_weights = torch.load("octavus.pt")
brain = Octavus_arms_brain()
new_weights = brain.state_dict()

for name, old in old_weights.items():
    if new_weights[name].shape == old.shape:
        new_weights[name] = old
    else:
        print("opreatinf on", name, tuple(old.shape), "->", tuple(new_weights[name].shape))
        new_weights[name] = torch.zeros_like(new_weights[name])
        new_weights[name][:, :old.shape[1]] = old
        
brain.load_state_dict(new_weights)
torch.save(brain.state_dict(), "octavus.pt")
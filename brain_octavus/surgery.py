import torch
from brain_octavus.brain import Octavus_arms_brain

old_weights = torch.load("octavus.pt")
brain = Octavus_arms_brain()
new_weights = brain.state_dict()

for name, old_piece in old_weights.items():
    if new_weights[name].shape == old_piece.shape:
        new_weights[name] = old_piece
    else:
        print("opreatinf on", name, tuple(old_piece.shape), "->", tuple(new_weights[name].shape))
        new_weights[name] = torch.zeros_like(new_weights[name])
        new_weights[name][:, :old_piece.shape[1]] = old_piece
        
brain.load_state_dict(new_weights)
torch.save(brain.state_dict(), "octavus.pt")
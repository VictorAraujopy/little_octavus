import numpy as np
import torch
from brain_octavus.brain import Octavus_arms_brain

#learning by watching: the brain copies the teacher (world_octavus/teacher.py) before the PPO
#record the teacher first (uv run world_octavus/record_teacher.py), then run this from the project root

torch.manual_seed(0) #same shuffles every run, so two runs can be compared

data = np.load("teacher_demos.npz")
xs = torch.tensor(data["observations"]) #what the octopus felt, every step
ys = torch.tensor(data["actions"]).clamp(-0.9, 0.9) #what the teacher did; Tanh never reaches 1, asking for it would saturate

#the last 10% is the exam: other episodes, never studied
#(a random 10% would be too easy: each step looks almost like the one before it)
cut = len(xs) * 9 // 10
study_xs, study_ys = xs[:cut], ys[:cut]
exam_xs, exam_ys = xs[cut:], ys[cut:]

#each command's usual size: the gentle sways are ~0.02 and the suckers ~1. each error is measured in its own size,
#or the brain would copy only the big commands and leave the arms still
size = study_ys.std(0).clamp(min=0.01) #min: a command the teacher never uses must stay near 0, not divide by 0

brain = Octavus_arms_brain()
optimizer = torch.optim.Adam(brain.parameters(), lr=3e-4) #same as train.py
minibatch_size = 256 #bigger than train.py's 64: no luck here, every example is the right answer


def copy_error(x, y):
    _, distribuition = brain.act(x)
    return (((distribuition.mean - y) / size) ** 2).mean() #mean = what the brain believes, without the draw


for epoch in range(100): #how many times it studies the whole recording (with 20 the tip curls came out at a third)
    errors = []
    for chunk in torch.randperm(cut).split(minibatch_size):
        error = copy_error(study_xs[chunk], study_ys[chunk])
        pre_tanh = brain.pre_tanh(study_xs[chunk])
        saturation = ((pre_tanh.abs() - 2).clamp(min=0) ** 2).mean() #same rule as train.py, so the PPO doesn't yank it
        loss = error + 0.1 * saturation

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        errors.append(error.item())

    with torch.no_grad(): #the exam only looks, it doesn't learn
        exam = copy_error(exam_xs, exam_ys).item()
    print(f"epoch {epoch}: study {np.mean(errors):.3f} | exam {exam:.3f}", flush=True)

#small exploration for the PPO: with the soft arm springs a 0.07 command already bends a joint to its limit
brain.exploration.data[:20] = np.log(0.02) #arms
brain.exploration.data[20:] = np.log(0.1) #siphon: aim, tilt and jet
torch.save(brain.state_dict(), "octavus.pt")
print("saved octavus.pt")

import torch
from torch import nn

#model
class Octavus_brain(nn.Module):
    def __init__(self):
        super().__init__()
        
        n_arms, n_arm_x, n_body_x, n_arm_y, n_jet = 8, 11, 14, 6, 2
        
        self.n_arms = n_arms
        self.n_arm_x = n_arm_x
        
        #calculate each arms angle start at 22.5 and go foward with 45 for each arm
        angles = torch.deg2rad(22.5 + 45 * torch.arange(n_arms))
        self.arm_position = torch.stack([angles.cos(), angles.sin()], dim=-1)
        
        n_x_for_neural = n_jet + n_body_x + n_arm_x * n_arms
        
        self.arm = nn.Sequential(
            nn.Linear(n_x_for_neural, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, n_arm_y * n_arms + n_jet)
        
        )
        
        
        self.critic = nn.Sequential(
            nn.Linear(n_x_for_neural, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
            
        )
        n_exploration = n_arm_y + n_jet
        self.exploration = nn.Parameter(torch.zeros(n_exploration))
        
    def act(self, x: torch.Tensor):
   
        gross_y = self.arm(x)
        
        arms = self.exploration[:6].repeat(self.n_arms) #each arm recive the same draw
        jets = self.exploration[6:]
        exp_format = torch.cat([arms, jets])
        
        variation = exp_format.exp()
        distribuition = torch.distributions.Normal(gross_y, variation)
        draw_y = distribuition.sample()
        return draw_y, distribuition


class SimpleBrain(nn.Module):
    
    def __init__(self, n_x, n_y): #number of x and number of y (not the real values)
        #runs the nn.Module __init__
        #super refers the mother class(nn.Module)
        super().__init__()
        #neural network that decides the action
        self.body = nn.Sequential(#create the object
            nn.Linear(n_x, 64),#Neural layer, defines the layer size and create it
            #make a curve so the multiply dont stay only on a straight line
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n_y),
        )
        self.exploration = nn.Parameter(torch.zeros(n_y))#create a number for each y
        
        self.critic = nn.Sequential(
            nn.Linear(n_x, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),    
        )
    
    def act(self, x):
        gross_y = self.body(x)
        variation = self.exploration.exp()#read the exploration number 
        #exp catch the number and made 2 raised by it
        
        distribuition = torch.distributions.Normal(gross_y, variation)
        #keep de randomizer
        draw_y = distribuition.sample()#do the math and draw 
        
        return draw_y
    
if __name__ == "__main__":
    brain = SimpleBrain(n_x=3, n_y=1)
    x = torch.tensor([1.0, 0.0, 0.5])#the neural network just accept 
    #tensor values
    #same x three times: the draw should give a different y each time
    print("y:", brain.act(x))
    print("y:", brain.act(x))
    print("y:", brain.act(x))
    #critic's guess of how much reward comes from this situation
    print("critic:", brain.critic(x))
    print("weights:", sum(p.numel() for p in brain.parameters()))

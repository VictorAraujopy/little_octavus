import torch
from torch import nn

#model
class Octavus_brain(nn.Module):
    def __init__(self):
        super().__init__()
        
        n_arms, n_arm_x, n_body_x, n_arm_y, n_jet = 8, 36, 14, 20, 3
        
        self.n_arms = n_arms
        self.n_arm_x = n_arm_x
        
        #calculate each arms angle start at 22.5 and go foward with 45 for each arm
        angles = torch.deg2rad(22.5 + 45 * torch.arange(n_arms))
        self.arm_position = torch.stack([angles.cos(), angles.sin()], dim=-1)
        
        n_x_for_neural = n_jet + n_body_x + n_arm_x * n_arms
        
        self.arm = nn.Sequential(
            nn.Linear(n_x_for_neural, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, n_arm_y * n_arms + n_jet),
            nn.Tanh()
        )
        
        
        self.critic = nn.Sequential(
            nn.Linear(n_x_for_neural, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
            
        )
        n_exploration = n_arm_y + n_jet
        self.exploration = nn.Parameter(torch.zeros(n_exploration))
        
    def act(self, x: torch.Tensor):
   
        gross_y = self.arm(x)
        
        arms = self.exploration[:20].repeat(self.n_arms) #each arm recive the same draw
        jets = self.exploration[20:]
        exp_format = torch.cat([arms, jets])
        
        variation = exp_format.exp()
        distribuition = torch.distributions.Normal(gross_y, variation)
        draw_y = distribuition.sample()
        return draw_y, distribuition

    def pre_tanh(self, x: torch.Tensor):
        """
        The 163 numbers right before the output Tanh, for the saturation cost in train.py (it charges the ones past 2).
        Here the Tanh is the last layer of self.arm, so it's every layer but the last: self.arm[:-1].
        (Octavus_arms_brain.pre_tanh gives the same kind of numbers, but 227 of them, from three outputs.)
        """
        return self.arm[:-1](x)


class Octavus_arms_brain(nn.Module):
    #like a real octopus (2/3 of its neurons are in the arms): one small network shared by the 8 arms,
    #and a central brain that sees the whole body, sends each arm a short order and drives the siphon
    def __init__(self):
        super().__init__()

        n_arms, n_arm_x, n_body_x, n_arm_y, n_jet_y = 8, 36, 17, 20, 3
        #numbers the central brain sends to each arm: a short order, like "reach that way" or "grip"
        n_order = 8

        self.n_arms = n_arms
        self.n_arm_x = n_arm_x
        self.n_order = n_order

        #where each arm sits around the body (cos, sin): the shared arm network needs it to know which arm it is
        angles = torch.deg2rad(22.5 + 45 * torch.arange(n_arms))
        self.register_buffer("arm_position", torch.stack([angles.cos(), angles.sin()], dim=-1))

        n_x = n_arms * n_arm_x + n_body_x  #305: everything the octopus feels

        #central brain: sees everything
        self.central = nn.Sequential(
            nn.Linear(n_x, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )
        #from the central thought: one order per arm, and the siphon commands (their Tanh is in think)
        self.orders = nn.Linear(256, n_arms * n_order)
        self.siphon = nn.Linear(256, n_jet_y)

        #arm network, the same one for all 8 arms: what this arm feels + its order + where it is -> its 20 commands
        #8 arms use it every step, so every step gives it 8 examples to learn from
        self.arm = nn.Sequential(
            nn.Linear(n_arm_x + n_order + 2, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, n_arm_y),
        )

        self.critic = nn.Sequential(
            nn.Linear(n_x, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )
        #20 kinds of arm motor (shared by the 8 arms) + 3 for the siphon, same as Octavus_brain
        self.exploration = nn.Parameter(torch.zeros(n_arm_y + n_jet_y))

    def think(self, x: torch.Tensor):
        """
        One pass through the whole brain. Returns three outputs, each stopped right BEFORE its Tanh (raw numbers):
            arms_z   (160): the 8 arms' 20 commands, in a row
            siphon_z (3):   the siphon commands
            orders_z (64):  the central brain's 8-number order for each arm
        The orders do go through a Tanh inside (that's the version the arms receive), but come back raw.
        act() applies the Tanh to arms_z and siphon_z; pre_tanh() joins all three for the saturation cost.
        """
        thought = self.central(x)
        orders_z = self.orders(thought)
        orders = torch.tanh(orders_z).reshape(*x.shape[:-1], self.n_arms, self.n_order)
        arms_x = x[..., :self.n_arms * self.n_arm_x].reshape(*x.shape[:-1], self.n_arms, self.n_arm_x)
        position = self.arm_position.expand(*x.shape[:-1], self.n_arms, 2)
        arms_z = self.arm(torch.cat([arms_x, orders, position], dim=-1)).flatten(-2)  #8 arms x 20, in a row
        siphon_z = self.siphon(thought)
        return arms_z, siphon_z, orders_z

    def pre_tanh(self, x: torch.Tensor):
        """
        The numbers right before each output Tanh, for the saturation cost in train.py (it charges the ones past 2).
        DIFFERENT from Octavus_brain.pre_tanh: there the Tanh is one layer at the end, so it slices it off
        (self.arm[:-1]). Here there are three outputs with their own Tanh applied by hand in think(), so it
        just joins think()'s raw numbers: 160 arms + 3 siphon + 64 orders = 227 (not 163).
        The orders are included because they can jam at the edge too.
        """
        return torch.cat(self.think(x), dim=-1)

    def act(self, x: torch.Tensor):
        arms_z, siphon_z, _ = self.think(x)
        gross_y = torch.tanh(torch.cat([arms_z, siphon_z], dim=-1))  #163: arms first, siphon last (the env's order)

        arms = self.exploration[:20].repeat(self.n_arms)  #each arm kind shares one draw size
        jets = self.exploration[20:]
        variation = torch.cat([arms, jets]).exp()
        distribuition = torch.distributions.Normal(gross_y, variation)
        draw_y = distribuition.sample()
        return draw_y, distribuition



if __name__ == "__main__":
    #quick check: both brains take the 305 numbers and answer 163 commands
    x = torch.zeros(305)
    for brain in (Octavus_brain(), Octavus_arms_brain()):
        y, _ = brain.act(x)
        print(type(brain).__name__, "| commands:", y.shape[0], "| weights:", sum(p.numel() for p in brain.parameters()))

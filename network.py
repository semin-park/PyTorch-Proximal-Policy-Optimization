import torch

class Policy(torch.nn.Module):
    def __init__(self, c_in, actions_out):
        super(Policy, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(c_in, 32, 8, stride=4, padding=0, bias=False),
            torch.nn.ReLU(),

            torch.nn.Conv2d(32, 64, 4, stride=2, padding=0, bias=False),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(64, 64, 3, stride=1, padding=0, bias=False),
            torch.nn.ReLU()
        )

        self.fc = torch.nn.Linear(3136, 512)
        
        self.actor_fc = torch.nn.Linear(512, actions_out)
        self.critic_fc = torch.nn.Linear(512, 1)
        

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        x = self.fc(x)

        logit = self.actor_fc(x)
        value = self.critic_fc(x)

        return logit, value
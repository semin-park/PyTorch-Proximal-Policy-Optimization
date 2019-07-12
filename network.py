import torch

class Policy(torch.nn.Module):
    def __init__(self, c_in, actions_out):
        super(Policy, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(c_in, 32, 3, stride=2, padding=1, bias=False),
            torch.nn.ReLU(),

            torch.nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=False),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(32,32, 3, stride=2, padding=1, bias=False),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(32,32, 3, stride=2, padding=1, bias=False),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(32,32, 3, stride=2, padding=1, bias=False),
            torch.nn.ReLU()
        )

        # Assuming input size of 84x84, the output of the convolution becomes 32x3x3 = 288
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(288, 256),
            torch.nn.ReLU()
        )
        
        self.actor_fc = torch.nn.Linear(256, actions_out)
        self.critic_fc = torch.nn.Linear(256, 1)
        

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        x = self.fc(x)

        logit = self.actor_fc(x)
        value = self.critic_fc(x)

        return logit, value
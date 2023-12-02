import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class QNet(nn.Module):
    """
    Q network, to predict q values
    """
    def __init__(self, input_size, hidden_size, output_size, device):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, device=device)
        self.fc2 = nn.Linear(hidden_size, hidden_size, device=device)
        self.fc3 = nn.Linear(hidden_size, output_size, device=device)
        self.device = device

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def save(self, save_dir):
        torch.save(self.state_dict(), os.path.join(save_dir, "q_net.pth"))
    
    def load(self, load_dir):
        self.load_state_dict(torch.load(os.path.join(load_dir, "q_net.pth")))
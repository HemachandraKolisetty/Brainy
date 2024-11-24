import torch.nn as nn

class ANNModel(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = nn.ReLU()
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.lif2 = nn.ReLU()
        self.fc3 = nn.Linear(num_hidden, num_outputs)
        self.lif3 = nn.ReLU()

    def forward(self, x):
        for step in range(x.size(1)):
            input_step = x[:, step, :]
            cur1 = self.fc1(input_step)
            spk1 = self.lif1(cur1)
            cur2 = self.fc2(spk1)
            spk2 = self.lif2(cur2)
            cur3 = self.fc3(spk2)
            out = self.lif3(cur3)
        
        return out
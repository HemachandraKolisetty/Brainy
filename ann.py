import torch.nn as nn

class ANNModelSimple(nn.Module):
    def __init__(self, input_size, num_outputs, num_hidden=128):
        super(ANNModelSimple, self).__init__()
        self.fc1 = nn.Linear(input_size, num_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, num_outputs)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
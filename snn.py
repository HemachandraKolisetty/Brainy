import torch
import torch.nn as nn
import snntorch as snn

class SNNModelSimple(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden = 128, beta=0.9):
        super(SNNModelSimple, self).__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc3 = nn.Linear(num_hidden, num_outputs)
        self.lif3 = snn.Leaky(beta=beta)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk_rec = []
        num_steps = x.size(1)
        for step in range(num_steps):
            input_step = x[:, step, :]
            cur1 = self.fc1(input_step)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk_rec.append(spk3)
        
        out_spikes = torch.stack(spk_rec, dim=0)
        return out_spikes
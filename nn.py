import torch.nn as nn


class CostSensitiveLRLayer(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.layer1 = nn.Linear(num_inputs, 1)
        self.act_fun = nn.Sigmoid()

    def forward(self, input):
        res = self.layer1(input)
        res = self.act_fun(res)
        return res

    def regularize(self):
        loss_val = 0
        for param in self.parameters():
            loss_val += th.norm(param)
        return loss_val

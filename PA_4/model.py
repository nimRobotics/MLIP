import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.layer1 =nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid()
        ) 
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, num_classes),
            nn.Softmax(dim=0)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

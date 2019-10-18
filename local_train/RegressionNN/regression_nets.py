import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch


class RegressionNet(nn.Module):
    def __init__(self, out_dim, psi_dim, hidden_dim=100, x_dim=1, attention_net=None):
        super().__init__()

        self.fc1 = nn.Linear(x_dim + psi_dim, hidden_dim)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)

        self.fc3 = nn.Linear(hidden_dim, out_dim)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0.0)

        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.constant_(self.fc4.bias, 0.0)

        self.psi_dim = psi_dim
        self.attention_net = attention_net

    def forward(self, params):
        """
            Generator takes a vector of noise and produces sample
        """
        if self.attention_net:
            params = self.attention_net(params)
        h1 = torch.tanh(self.fc1(params))
        h4 = torch.tanh(self.fc4(h1))
        h2 = F.leaky_relu(self.fc2(h4))
        y_gen = self.fc3(h2)
        return y_gen

class RegressionLosses(object):
    def __init__(self):
        pass

    def loss(self, true_values, predicted_values):
        return F.mse_loss(predicted_values, true_values)
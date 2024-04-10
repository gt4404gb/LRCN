
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CNNClassifier, self).__init__()

        # LSTM Encoder
        self.conv1 = nn.Conv1d(1, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)

        # Estimation network
        self.estimation_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        # Use the last hidden state as the context
        x = x.unsqueeze(1)
        h = self.conv1(x)
        h = F.relu(h)
        h = self.conv2(h)
        h = F.relu(h)
        h = self.adaptive_pool(h)
        h = h.view(x.size(0), -1)

        # Use the estimation network for classification
        #y_hat = self.estimation_net(h)

        return h,h


def cnn_loss(y, y_hat):
    diff_loss = nn.CrossEntropyLoss()(y_hat, y)
    return diff_loss
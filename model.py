import torch
import torch.nn as nn
import torch.nn.functional as F
from non_local import NonLocalBlock


class scoreNet(nn.Module):
    def __init__(self):
        super(scoreNet, self).__init__()
        self.fc_embed1 = nn.Linear(in_features=6, out_features=256, bias=True)
        self.ln1 = nn.LayerNorm(256)
        self.fc_embed2 = nn.Linear(in_features=4096, out_features=512, bias=True)
        self.ln2 = nn.LayerNorm(512)

        self.nonlocalblock = NonLocalBlock(channel=768)

        # conv1d
        # self.conv1 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, groups=1)
        # self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, groups=1)
        # self.conv3 = nn.Conv1d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)

        # lstm
        self.lstm = nn.LSTM(input_size=768, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(in_features=1024, out_features=128, bias=True)
        self.fc2 = nn.Linear(in_features=128, out_features=1, bias=True)

    def forward(self, x):
        x1 = self.fc_embed1(x[:, :, :6])
        x1 = self.ln1(x1)
        x1 = F.relu(x1)
        x2 = self.fc_embed2(x[:, :, 6:])
        x2 = self.ln2(x2)
        x2 = F.relu(x2)

        x = torch.cat((x1, x2), 2)
        
        # conv1d
        # x = x.permute(0, 2, 1)
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = torch.sigmoid(self.conv3(x))
        # x = x.permute(0, 2, 1)


        # non-local
        x = self.nonlocalblock(x)
        # lstm
        x, _ = self.lstm(x)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        
        x = x.squeeze(2)

        return x

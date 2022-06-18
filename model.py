import torch
import torch.nn as nn


class C3D(nn.Module):

    def __init__(self, num_classes):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.batchnorm1 = nn.BatchNorm3d(64)

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.batchnorm2 = nn.BatchNorm3d(128)

        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.batchnorm3 = nn.BatchNorm3d(256)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.batchnorm4 = nn.BatchNorm3d(512)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.fc1 = nn.Linear(13824, 512)
        
        self.fc2 = nn.Linear(512, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()


        # Attention
        self.attention_conv1 = nn.Conv3d(256, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2))

        self.attention_conv2 = nn.Conv3d(64, 16, kernel_size=(1, 3, 3), stride=(1, 2, 2))

        self.attention_conv3 = nn.Conv3d(16, 1, kernel_size=(1, 2, 2), stride=(1, 1, 1))

        # SPP
        self.spp = SPP()


    def forward(self, x):
        # batch_size,channels,帧数,H,W
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.relu(self.batchnorm2(self.conv2(x)))
        x = self.relu(self.batchnorm3(self.conv3(x)))
        x = self.pool1(x)


        atte = self.attention_conv1(x)
        atte = self.attention_conv2(atte)
        atte = self.attention_conv3(atte)
        outputs2 = atte.view(-1, 8)

        x = atte * x

        x = self.relu(self.batchnorm4(self.conv4(x)))
        x = self.pool2(x)

        x = self.spp(x)
    
        x = x.view(-1, 13824)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        outputs1 = self.fc2(x)

        return outputs1, outputs2


class SPP(nn.Module):
    def __init__(self):
        super(SPP, self).__init__()
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 3, 3))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))

    def forward(self, x):
        x1 = self.pool1(x)
        x2 = self.pool2(x)

        x = x.view(x.size(0), -1)
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        outputs = torch.cat([x, x1, x2], dim=1)
        
        return outputs

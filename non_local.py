import torch
import torch.nn as nn


class NonLocalBlock(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.fc_phi = nn.Linear(in_features=channel, out_features=self.inter_channel)
        self.fc_theta = nn.Linear(in_features=channel, out_features=self.inter_channel)
        self.fc_g = nn.Linear(in_features=channel, out_features=self.inter_channel)
        self.softmax = nn.Softmax(dim=1)
        self.fc_mask = nn.Linear(in_features=self.inter_channel, out_features=channel)

    def forward(self, x):
        b, c, d = x.size()

        x_phi = self.fc_phi(x)
        x_theta = self.fc_theta(x).permute(0, 2, 1).contiguous()
        x_g = self.fc_g(x).permute(0, 2, 1).contiguous()

        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g).permute(0, 2, 1).contiguous()
        
        mask = self.fc_mask(mul_theta_phi_g)
        out = mask + x

        return out


if __name__=='__main__':
    model = NonLocalBlock(channel=512)

    input = torch.randn(32, 30, 512)
    
    out = model(input)
    print(out.shape)

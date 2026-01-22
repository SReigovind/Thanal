import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self):
        super(AttentionUNet, self).__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
        self.e1 = conv_block(3, 64)
        self.pool = nn.MaxPool2d(2)
        self.e2 = conv_block(64, 128)
        self.e3 = conv_block(128, 256)
        self.e4 = conv_block(256, 512)
        self.b = conv_block(512, 1024)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.d1 = conv_block(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.d2 = conv_block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.d3 = conv_block(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.d4 = conv_block(128, 64)
        self.out = nn.Conv2d(64, 1, kernel_size=1)
        
    def forward(self, x):
        c1 = self.e1(x)
        p1 = self.pool(c1)
        c2 = self.e2(p1)
        p2 = self.pool(c2)
        c3 = self.e3(p2)
        p3 = self.pool(c3)
        c4 = self.e4(p3)
        p4 = self.pool(c4)
        b = self.b(p4)
        u1 = self.up1(b)
        x4 = self.att1(g=u1, x=c4)
        cat1 = torch.cat((u1, x4), dim=1)
        ud1 = self.d1(cat1)
        u2 = self.up2(ud1)
        x3 = self.att2(g=u2, x=c3)
        cat2 = torch.cat((u2, x3), dim=1)
        ud2 = self.d2(cat2)
        u3 = self.up3(ud2)
        x2 = self.att3(g=u3, x=c2)
        cat3 = torch.cat((u3, x2), dim=1)
        ud3 = self.d3(cat3)
        u4 = self.up4(ud3)
        x1 = self.att4(g=u4, x=c1)
        cat4 = torch.cat((u4, x1), dim=1)
        ud4 = self.d4(cat4)
        return self.out(ud4)
import torch
import torch.nn as nn
import torch.nn.functional as F


# DOWN CONV
class double_conv(nn.Module):
    '''(conv-BN-ReLU)X2 :   in_ch  , in_ch , out_ch    '''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 3, padding=1),
            nn.GroupNorm(4, in_ch),
            nn.ReLU(inplace=True),  # True means cover the origin input
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class double_conv_in(nn.Module):
    '''(conv-BN-ReLU)X2 :   in_ch  , in_ch , out_ch    '''

    def __init__(self, in_ch, out_ch):
        super(double_conv_in, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=True),  # True means cover the origin input
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# UP CONV
class double_conv2(nn.Module):
    '''(conv-BN-ReLU)X2 :   in_ch  , out_ch , out_ch    '''

    def __init__(self, in_ch, out_ch):
        super(double_conv2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=True),  # True means cover the origin input
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class down_s(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_s, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d((1, 2, 2)),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up_s(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_s, self).__init__()
        if 0:
            self.up = nn.Sequential(
                nn.ConvTranspose3d(in_ch * 2 // 3, in_ch * 2 // 3, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm3d(in_ch * 2 // 3),
                nn.ReLU(inplace=True),
            )
        else:
            self.up = nn.Upsample(scale_factor=2)
        self.conv = double_conv2(in_ch, out_ch)

    def forward(self, x1, x2):  # x1--up , x2 ---down
        # print(x1.shape)
        x1 = self.up(x1)
        # print(x1.shape)
        # print(x2.shape)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, (diffZ // 2, diffZ - diffZ // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffX // 2, diffX - diffX // 2,))
        # print(x1.shape)
        # print(x2.shape)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        if 0:
            self.up = nn.Sequential(
                nn.ConvTranspose3d(in_ch * 2 // 3, in_ch * 2 // 3, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm3d(in_ch * 2 // 3),
                nn.ReLU(inplace=True),
            )
        else:
            self.up = nn.Upsample(scale_factor=(1, 2, 2))
        self.conv = double_conv2(in_ch, out_ch)

    def forward(self, x1, x2):  # x1--up , x2 ---down
        # print(x1.shape)
        x1 = self.up(x1)
        # print(x1.shape)
        # print(x2.shape)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, (diffZ // 2, diffZ - diffZ // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffX // 2, diffX - diffX // 2,))
        # print(x1.shape)
        # print(x2.shape)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        # x=F.sigmoid(x)
        # print(x.shape)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv_in(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


cc = 16  # 12

class Unet_3D(nn.Module):
    def __init__(self, n_channels, n_classes, init_weights=True):
        super(Unet_3D, self).__init__()
        self.inconv = inconv(n_channels, cc)
        self.down1 = down_s(cc, 2 * cc)
        self.down2 = down_s(2 * cc, 4 * cc)
        self.down3 = down(4 * cc, 8 * cc)
        self.down4 = down(8 * cc, 16 * cc)
        self.down5 = down(16 * cc, 16 * cc)
        self.down6 = down(16 * cc, 16 * cc)
        self.up1 = up(32 * cc, 16 * cc)
        self.up2 = up(32 * cc, 16 * cc)
        self.up3 = up(24 * cc, 8 * cc)
        self.up4 = up(12 * cc, 4 * cc)
        self.up5 = up(6 * cc, 2 * cc)
        self.up6 = up(3 * cc, cc)
        self.outconv = outconv(cc, n_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        # print(x7.shape)
        x = self.up1(x7, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)
        x = self.outconv(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


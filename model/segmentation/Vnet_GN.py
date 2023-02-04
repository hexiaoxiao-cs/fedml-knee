

import torch
from torch import nn
import torch.nn.functional as F

GN_CHANNEL = 8


def passthrough(x, **kwargs):
    return x


def ELUCons(elu='elu', nchan=1):
    if elu == 'elu':
        return nn.ELU(inplace=True)
    elif elu == 'relu':
        return nn.ReLU(inplace=True)
    else:
        return nn.PReLU(nchan)


class LUConv(nn.Module):
    def __init__(self, nchan, elu, norm):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(nchan) if norm == 'bn' else nn.GroupNorm(GN_CHANNEL, nchan)  # ContBatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu, norm):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu, norm))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, outChans, elu, norm):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, outChans, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(outChans) if norm == 'bn' else nn.GroupNorm(GN_CHANNEL,
                                                                              outChans)  # ContBatchNorm3d(outChans)
        self.relu1 = ELUCons(elu, outChans)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.bn1(self.conv1(x))
        # split input in to 16 channels
        # x16 = torch.cat((x, x, x, x), 0)
        out = self.relu1(out)  # torch.add(out, x16)
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, norm, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2 * inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=[2, 2, 2], stride=[2, 2, 2], padding=[0, 0, 0])
        self.bn1 = nn.BatchNorm3d(outChans) if norm == 'bn' else nn.GroupNorm(GN_CHANNEL,
                                                                              outChans)  # ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu, norm)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class DownTransition_ct(nn.Module):
    def __init__(self, inChans, nConvs, elu, norm, dropout=False):
        super(DownTransition_ct, self).__init__()
        outChans = 2 * inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=[3, 2, 2], stride=[1, 2, 2], padding=[1, 0, 0])
        self.bn1 = nn.BatchNorm3d(outChans) if norm == 'bn' else nn.GroupNorm(GN_CHANNEL,
                                                                              outChans)  # ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        self.conv_hm = nn.Sequential(
            nn.Conv3d(outChans, 256, kernel_size=5, padding=2),
            ELUCons(elu, 256),
            nn.Conv3d(256, 8, kernel_size=5, padding=2)
        )
        self.conv_wh = nn.Sequential(
            nn.Conv3d(outChans, 256, kernel_size=5, padding=2),
            ELUCons(elu, 256),
            nn.Conv3d(256, 3, kernel_size=5, padding=2)
        )
        self.conv_reg = nn.Sequential(
            nn.Conv3d(outChans, 256, kernel_size=5, padding=2),
            ELUCons(elu, 256),
            nn.Conv3d(256, 3, kernel_size=5, padding=2)
        )
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu, norm)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        ft_hm = self.conv_hm(out)
        ft_reg = self.conv_reg(out)
        ft_wh = self.conv_wh(out)
        out = self.relu2(torch.add(out, down))
        return out, {"hm": ft_hm, "reg": ft_reg, "wh": ft_wh}


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, norm, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=[2, 2, 2], stride=[2, 2, 2],
                                          padding=[0, 0, 0])
        self.bn1 = nn.BatchNorm3d(outChans // 2) if norm == 'bn' else nn.GroupNorm(GN_CHANNEL,
                                                                                   outChans // 2)  # ContBatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu, norm)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, out_class, elu, norm, nll):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, out_class, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(out_class) if norm == 'bn' else nn.GroupNorm(out_class,
                                                                               out_class)  # ContBatchNorm3d(7)
        self.conv2 = nn.Conv3d(out_class, out_class, kernel_size=1)
        self.relu1 = ELUCons(elu, out_class)
        if nll:
            self.softmax = F.log_softmax
        else:
            self.softmax = F.softmax

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)

        # make channels the last axis
        # out = out.permute(0, 2, 3, 4, 1).contiguous()
        # # flatten
        # out = out.view(out.numel() // 2, 2)
        # out = self.softmax(out)
        # treat channel 0 as the predicted output
        return out


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

    def _initialize_weights(self):
        # print('haha',' : %.2fMB' % (sum(p.numel() for p in self.parameters()) / (1024.0 * 1024) * 4))
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                # print('CONV: weight_init... ')
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                # print('NORM: weight_init... ')
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def __repr__(self):
        params = ' : %.2fMB' % (sum(p.numel() for p in self.parameters()) / (1024.0 * 1024) * 4)
        return self.__class__.__name__ + params


class VNet(BaseNet):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, model_channel=4, out_class=9, elu='elu', norm='gn', nll=False, dropout=True):
        super().__init__()
        self.model_channel = model_channel
        self.in_tr = InputTransition(self.model_channel, elu, norm)
        self.down_tr32 = DownTransition(self.model_channel, 1, elu, norm)
        self.down_tr64 = DownTransition(2 * self.model_channel, 2, elu, norm)
        self.down_tr128 = DownTransition(4 * self.model_channel, 3, elu, norm, dropout=dropout)
        self.down_tr256 = DownTransition(8 * self.model_channel, 2, elu, norm, dropout=dropout)
        self.up_tr256 = UpTransition(16 * self.model_channel, 16 * self.model_channel, 2, elu, norm, dropout=dropout)
        self.up_tr128 = UpTransition(16 * self.model_channel, 8 * self.model_channel, 2, elu, norm, dropout=dropout)
        self.up_tr64 = UpTransition(8 * self.model_channel, 4 * self.model_channel, 1, elu, norm)
        self.up_tr32 = UpTransition(4 * self.model_channel, 2 * self.model_channel, 1, elu, norm)
        self.out_tr = OutputTransition(2 * self.model_channel, out_class, elu, norm, nll)
        self._initialize_weights()

    def forward(self, x):
        # print(x.shape)
        out16 = self.in_tr(x)
        # print(out16.shape)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        self.out256=out256
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        # print(out.shape)
        out = self.out_tr(out)
        return out


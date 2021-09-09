import torch
import torch.nn as nn
import torch.nn.functional as F

INPLACE = False
BIAS = False

class Conv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, shape, affine=True ):
        super(Conv,self).__init__()
        self.shape = shape
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.affine = affine

        if isinstance(kernel_size, int):
            self.ops = nn.Sequential(
                nn.ReLU(inplace=INPLACE),
                nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride= stride,padding=padding, bias= BIAS),
                nn.BatchNorm2d(C_out,affine=affine),
            )
        else:
            assert isinstance(kernel_size,tuple)
            k1, k2 = kernel_size[0], kernel_size[1]
            self.ops = nn.Sequential(
                nn.ReLU(inplace=INPLACE),
                nn.Conv2d(C_in, C_out,kernel_size=(k1,k2),stride=(1,stride),padding=padding[0],bias=BIAS),
                nn.BatchNorm2d(C_out,affine=affine),
                nn.ReLU(inplace=INPLACE),
                nn.Conv2d(C_out,C_out,(k2,k1),stride=(stride,1),padding=padding[1],bias=BIAS),
                nn.BatchNorm2d(C_out,affine=affine),
            )
    def forward(self, x):
        x = self.ops(x)
        return x

class MaxPool(nn.Module):
    def __init__(self, C_in, C_out, shape, kernel_size, stride=1, padding=0):
        super(MaxPool,self).__init__()
        self.shape = shape
        self.C_in = C_in
        self.C_out = C_out

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        if isinstance(padding,tuple):
            padding = 0
        self.ops = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
    def forward(self, x):
        if isinstance(self.padding,tuple):
            x = F.pad(x,self.padding)
        x = self.ops(x)
        return x

class AvgPool(nn.Module):
    def __init__(self,C_in, C_out, shape, kernel_size, stride=1, padding=0, count_include_pad=False):
        super(AvgPool,self).__init__()
        self.shape = shape
        self.C_in = C_in
        self.C_out = C_out

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        if isinstance(padding,tuple):
            padding = 0
        self.ops = nn.AvgPool2d(kernel_size, stride=stride, padding=padding, count_include_pad=count_include_pad)
    def forward(self,x):
        if isinstance(self.padding,tuple):
            x = F.pad(x,self.padding)
        x = self.ops(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity,self).__init__()
    def forward(self,x):
        return x

#-------------------------------------------------------------------
class AuxHeadImageNet(nn.Module):
    def __init__(self, C_in, classes):
        """input should be in [B, C, 7, 7]"""
        super(AuxHeadImageNet, self).__init__()
        self.relu1 = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False)
        self.conv1 = nn.Conv2d(C_in, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 768, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(768)
        self.relu3 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(768, classes)

    def forward(self, x, bn_train=False):
        if bn_train:
            self.bn1.train()
            self.bn2.train()
        x = self.relu1(x)
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu3(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x



class AuxHeadCIFAR(nn.Module):
    def __init__(self, C_in, classes):
        """assuming input size 8x8"""
        super(AuxHeadCIFAR, self).__init__()

        self.relu1 = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False)
        self.conv1 = nn.Conv2d(C_in, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 768, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(768)
        self.relu3 = nn.ReLU(inplace=True)


        self.classifier = nn.Linear(768, classes)

    def forward(self, x, bn_train=False):
        if bn_train:
            self.bn1.train()
            self.bn2.train()
        x = self.relu1(x)
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu3(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x



class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, shape, affine=True):
        super(ReLUConvBN, self).__init__()
        self.relu = nn.ReLU(inplace=INPLACE)
        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x, bn_train=False):
        x = self.relu(x)
        x = self.conv(x)
        if bn_train:
            self.bn.train()
        x = self.bn(x)
        return x

# class ReLUConvBN(nn.Module):
#
#     def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
#         super(ReLUConvBN, self).__init__()
#         self.op = nn.Sequential(
#             nn.ReLU(inplace=False),
#             nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
#             nn.BatchNorm2d(C_out, affine=affine)
#         )
#
#     def forward(self, x):
#         return self.op(x)



class MaybeCalibrateSize(nn.Module):
    def __init__(self, layers, channels, affine=True):
        super(MaybeCalibrateSize, self).__init__()
        self.channels = channels
        hw = [layer[0] for layer in layers]
        c = [layer[-1] for layer in layers]

        x_out_shape = [hw[0], hw[0], c[0]]
        y_out_shape = [hw[1], hw[1], c[1]]
        # previous reduction cell
        if hw[0] != hw[1]:
            assert hw[0] == 2 * hw[1]
            self.relu = nn.ReLU(inplace=INPLACE)
            self.preprocess_x = FactorizedReduce(c[0], channels, [hw[0], hw[0], c[0]], affine)
            x_out_shape = [hw[1], hw[1], channels]
        elif c[0] != channels:
            self.preprocess_x = ReLUConvBN(c[0], channels, 1, 1, 0, [hw[0], hw[0]], affine)
            x_out_shape = [hw[0], hw[0], channels]
        if c[1] != channels:
            self.preprocess_y = ReLUConvBN(c[1], channels, 1, 1, 0, [hw[1], hw[1]], affine)
            y_out_shape = [hw[1], hw[1], channels]

        self.out_shape = [x_out_shape, y_out_shape]

    def forward(self, s0, s1, bn_train=False):
        if s0.size(2) != s1.size(2):
            s0 = self.relu(s0)
            s0 = self.preprocess_x(s0, bn_train=bn_train)
        elif s0.size(1) != self.channels:
            s0 = self.preprocess_x(s0, bn_train=bn_train)
        if s1.size(1) != self.channels:
            s1 = self.preprocess_y(s1, bn_train=bn_train)
        return [s0, s1]


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, shape, affine=True):
        super(FactorizedReduce, self).__init__()
        self.shape = shape
        self.C_in = C_in
        self.C_out = C_out

        assert C_out % 2 == 0
        self.path1 = nn.Sequential(nn.AvgPool2d(1, stride=2, padding=0, count_include_pad=False),
                                   nn.Conv2d(C_in, C_out // 2, 1, bias=False))
        self.path2 = nn.Sequential(nn.AvgPool2d(1, stride=2, padding=0, count_include_pad=False),
                                   nn.Conv2d(C_in, C_out // 2, 1, bias=False))
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x, bn_train=False):
        if bn_train:
            self.bn.train()
        path1 = x
        path2 = F.pad(x, (0, 1, 0, 1), "constant", 0)[:, :, 1:, 1:]
        out = torch.cat([self.path1(path1), self.path2(path2)], dim=1)
        out = self.bn(out)
        return out

def apply_drop_path(x, drop_path_keep_prob, layer_id, layers, step, steps):
    layer_ratio = float(layer_id+1) / (layers)
    drop_path_keep_prob = 1.0 - layer_ratio * (1.0 - drop_path_keep_prob)
    step_ratio = float(step + 1) / steps
    drop_path_keep_prob = 1.0 - step_ratio * (1.0 - drop_path_keep_prob)
    if drop_path_keep_prob < 1.:
        mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(drop_path_keep_prob).cuda()
        #x.div_(drop_path_keep_prob)
        #x.mul_(mask)
        x = x / drop_path_keep_prob * mask
    return x


Operations_11 ={
    0: lambda c_in, c_out, stride, shape, affine: Conv(c_in, c_out, 1, stride, 0, shape, affine=affine),
    1: lambda c_in, c_out, stride, shape, affine: Conv(c_in, c_out, 3, stride, 1, shape, affine=affine),
    2: lambda c_in, c_out, stride, shape, affine: Conv(c_in, c_out, (1,3), stride, ((0,1),(1,0)), shape, affine=affine),
    3: lambda c_in, c_out, stride, shape, affine: Conv(c_in, c_out, (1,7), stride, ((0,3),(3,0)), shape, affine=affine),

    4: lambda c_in, c_out, stride, shape, affine: MaxPool(c_in, c_out, shape, kernel_size=2, stride=stride, padding=(0,1,0,1)),
    5: lambda c_in, c_out, stride, shape, affine: MaxPool(c_in, c_out, shape, kernel_size=3, stride=stride, padding=1),
    6: lambda c_in, c_out, stride, shape, affine: MaxPool(c_in, c_out, shape, kernel_size=5, stride=stride, padding=2),

    7: lambda c_in, c_out, stride, shape, affine: AvgPool(c_in, c_out, shape, kernel_size=2, stride=stride, padding=(0,1,0,1)),
    8: lambda c_in, c_out, stride, shape, affine: AvgPool(c_in, c_out, shape, kernel_size=3, stride=stride, padding=1),
    9: lambda c_in, c_out, stride, shape, affine: AvgPool(c_in, c_out, shape, kernel_size=5, stride=stride, padding=2),

    10:lambda c_in, c_out, stride, shape, affine: Identity(),
    11:lambda c_in, c_out, stride, shape, affine: SELayer(c_in)
    }

Operations_11_name = [
    'Conv 1*1',
    'Conv 3*3',
    'Conv 1*3+3*1',
    'Conv 1*7+7*1',

    'MaxPool 2*2',
    'MaxPool 3*3',
    'MaxPool 5*5',

    'AvgPool 2*2',
    'AvgPool 3*3',
    'AvgPool 5*5',

    'Identity',
    'SELayer',
]


#-------------------------------------------

class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, shape, affine=True):
        super(SepConv, self).__init__()
        self.shape = shape
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.affine = affine

        self.op = nn.Sequential(
                nn.ReLU(inplace=INPLACE),
                nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
                nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(C_in, affine=affine),
                nn.ReLU(inplace=INPLACE),
                nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
                nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(C_out, affine=affine),
            )

    def forward(self, x):
        return self.op(x)

class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, shape, affine=True):
        super(DilConv, self).__init__()
        self.shape = shape
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.affine = affine

        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )
    def forward(self, x):
        return self.op(x)




Operations_7 ={
    0: lambda c_in, c_out, stride, shape, affine: SepConv(c_in, c_out, 3, stride, 1, shape, affine=affine),
    1: lambda c_in, c_out, stride, shape, affine: SepConv(c_in, c_out, 5, stride, 5, shape, affine=affine),
    2: lambda c_in, c_out, stride, shape, affine: DilConv(c_in, c_out, 3, stride, 2, dilation=2, shape=shape, affine=affine),
    3: lambda c_in, c_out, stride, shape, affine: DilConv(c_in, c_out, 5, stride, 4, dilation=2, shape=shape, affine=affine),

    4: lambda c_in, c_out, stride, shape, affine: MaxPool(c_in, c_out, shape, kernel_size=3, stride=stride, padding=1),
    5: lambda c_in, c_out, stride, shape, affine: AvgPool(c_in, c_out, shape, kernel_size=3, stride=stride, padding=1),
    6:lambda c_in, c_out, stride, shape, affine: Identity(),

}

Operations_7_name = [
    'SepConv 3*3',
    'SepConv 5*5',
    'DilConv 3*3',
    'DilConv 5*5',

    'MaxPool 3*3',
    'AvgPool 3*3',
    'Identity',
]
import torch
import torch.nn as nn
import numpy as np

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    net = nn.Sequential()
    net.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups))
    net.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return net

class DepthWiseConv(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1):
        super().__init__()
        padding = 1
        if kernel_size == 1:
            padding = 0
        self.conv = conv_bn(in_channels, in_channels, kernel_size, stride, padding, in_channels)
    
    def forward(self, x):
        return self.conv(x)

class PointWiseConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = conv_bn(in_channels, out_channels, 1, 1, 0)
    
    def forward(self, x):
        return self.conv(x)

class MobileOneBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k, stride=1, dilation=1,
                 padding_mode='zeros', deploy=False, use_se=False):
        super(MobileOneBlock, self).__init__()
        self.k = k
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        kernel_size = 3
        padding = 1
        assert kernel_size == 3
        assert padding == 1
        padding_11 = padding - kernel_size // 2
        
        self.nonlinear = nn.ReLU()

        if use_se:
            ...
        else:
            self.se = nn.Identity()

        if deploy:
            self.dw_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=True,
                                        padding_mode=padding_mode)
            self.pw_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=True)
        else:
            self.dw_bn_layer = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None
            for i in range(k):
                setattr(self, f'dw_3x3_{i}', DepthWiseConv(in_channels, 3, stride=stride))
            self.dw_1x1 = DepthWiseConv(in_channels, 1, stride=stride)
            self.pw_bn_layer = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None
            for i in range(k):
                setattr(self, f'pw_1x1_{i}', PointWiseConv(in_channels, out_channels))
    
    def forward(self, inputs):
        if self.deploy:
            x = self.dw_reparam(inputs)
            x = self.nonlinear(x)
            x = self.pw_reparam(x)
            x = self.nonlinear(x)
            return x
        
        # 3x3 conv
        if self.dw_bn_layer is None:
            id_out = 0
        else:
            id_out = self.dw_bn_layer(inputs)
        x_conv_3x3 = []
        for i in range(self.k):
            x = getattr(self, f'dw_3x3_{i}')(inputs)
            # print(x.shape)
            x_conv_3x3.append(x)
        x_conv_1x1 = self.dw_1x1(inputs)

        x = id_out + x_conv_1x1 + sum(x_conv_3x3)
        x = self.nonlinear(self.se(x))

        # 1x1 conv
        if self.pw_bn_layer in None:
            id_out = 0
        else:
            id_out = self.pw_bn_layer(x)
        x_conv_1x1 = []
        for i in range(self.k):
            x_conv_1x1.append(getattr(self, f'pw_1x1_{i}')(x))
        x = id_out + sum(x_conv_1x1)
        x = self.nonlinear(x)
        
        return x

    def get_custom_l2(self):
        ...

    def get_equivalent_kernel_bias(self):
        # dw
        dw_kernel_3x3 = []
        dw_bias_3x3 = []
        for i in range(self.k):
            k3, b3 = self._fuse_bn_tensor(getattr(self, f'dw_3x3_{i}').conv)
            # print(k3.shape, b3.shape)
            dw_kernel_3x3.append(k3)
            dw_bias_3x3.append(b3)
        dw_kernel_1x1, dw_bias_1x1 = self._fuse_bn_tensor(self.dw_1x1.conv)
        dw_kernel_id, dw_bias_id = self._fuse_bn_tensor(self.dw_bn_layer, self.in_channels)
        dw_kernel = sum(dw_bias_3x3) + self._pad_1x1_to_3x3_tensor(dw_kernel_1x1) + dw_kernel_id
        dw_bias = sum(dw_bias_3x3) + dw_bias_1x1 + dw_bias_id

        # pw
        pw_kernel = []
        pw_bias = []
        for i in range(self.k):
            k1, b1 = self._fuse_bn_tensor(getattr(self, f"pw_1x1_{i}").conv)
            # print(k1.shape)
            pw_kernel.append(k1)
            pw_bias.append(b1)
        pw_kernel_id, pw_bias_id = self._fuse_bn_tensor(self.pw_bn_layer, 1)
        pw_kernel_1x1 = sum(pw_kernel) + pw_kernel_id
        pw_bias_1x1 = sum(pw_bias) + pw_bias_id
        return dw_kernel, dw_bias, pw_kernel_1x1, pw_bias_1x1

    def _pad_1x1_to_3x3_tensor(self, kernel_1x1):
        if kernel_1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel_1x1, [1,1,1,1])
    
    def _fuse_bn_tensor(self, branch, groups=None):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            bias = branch.conv.bias
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            input_dim = self.in_channels // groups # self.groups
            if groups == 1:
                ks = 1
            else:
                ks = 3
            kernel_value = np.zeros((self.in_channels, input_dim, ks, ks), dtype=np.float32)
            for i in range(self.in_channels):
                if ks == 1:
                    kernel_value[i, i % input_dim, 0, 0] = 1
                else:
                    kernel_value[i, i % input_dim, 1, 1] = 1
            self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)

            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
            
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std
    
    def deploy(self):
        dw_kernel, dw_bias, pw_kernel, pw_bias = self.get_equivalent_kernel_bias()

        self.dw_reparam = nn.Conv2d(
            in_channels=self.pw_1x1_0.conv.conv.in_channels, 
            out_channels=self.pw_1x1_0.conv.conv.in_channels,                              
            kernel_size=self.dw_3x3_0.conv.conv.kernel_size, 
            stride=self.dw_3x3_0.conv.conv.stride,
            padding=self.dw_3x3_0.conv.conv.padding, 
            groups=self.dw_3x3_0.conv.conv.in_channels, 
            bias=True, 
        )
        self.pw_reparam = nn.Conv2d(
            in_channels=self.pw_1x1_0.conv.conv.in_channels,
            out_channels=self.pw_1x1_0.conv.conv.out_channels, 
            kernel_size=1, 
            stride=1, 
            bias=True
        )
        
        self.dw_reparam.weight.data = dw_kernel
        self.dw_reparam.bias.data = dw_bias
        self.pw_reparam.weight.data = pw_kernel
        self.pw_reparam.bias.data = pw_bias

        for param in self.parameters():
            param.detach_()
        self.__delattr__('dw_1x1')
        for i in range(self.k):
            self.__delattr__(f'dw_3x3_{i}')
            self.__delattr__(f'pw_1x1_{i}')
        if hasattr(self, 'dw_bn_layer'):
            self.__delattr__('dw_bn_layer')
        if hasattr(self, 'pw_bn_layer'):
            self.__delattr__('pw_bn_layer')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True
import copy
import torch
import torch.nn as nn

from mobileone_block import MobileOneBlock

class MobileOneNet(nn.Module):
    def __init__(self, blocks, ks, channels, strides, width_muls, num_classes, deploy=False):
        super().__init__()

        self.stage_num = len(blocks)
        self.stage0 = nn.Sequential(
            nn.Conv2d(3, int(channels[0] * width_muls[0]), 3, 2, 1, bias=True),
            nn.BatchNorm2d(int(channels[0] * width_muls[0])),
            nn.ReLU(),
        )
        in_channels = int(channels[0] * width_muls[0])
        for i, block_num in enumerate(blocks[1:]):
            i += 1
            module = []
            out_channels = int(channels[i] * width_muls[i])
            for j in range(block_num):
                stride = strides[i] if j == 0 else 1
                block = MobileOneBlock(in_channels, out_channels, ks[i], stride, deploy=deploy)
                in_channels = out_channels
                module.append(block)
            setattr(self, f'stage{i}', nn.Sequential(*module))
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(out_channels, num_classes,),
        )
    
    def forward(self, x):
        x0 = self.stage0(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        assert x5.shape[-1] == 7
        x = self.avg_pool(x5)
        x = torch.flatten(x, start_dim=1) # b, c
        x = self.fc1(x)
        return x

def make_mobileone_s0(deploy=False):
    blocks = [1, 2, 8, 5, 5, 1]
    strides = [2, 2, 2, 2, 1, 2]
    ks = [4, 4, 4, 4, 4, 4] if deploy is False else [1, 1, 1, 1, 1, 1]
    width_muls = [0.75, 0.75, 1, 1, 1, 2] # 261 M flops
    channels = [64, 64, 128, 256, 256, 512, 512]
    num_classes = 1000

    model = MobileOneNet(blocks, ks, channels, strides, width_muls, num_classes, deploy)
    return model

def repvgg_model_convert(model: torch.nn.Module, do_copy=True, input=None, output=None):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    print('swith done. Checking....')

    deploy_model = make_mobileone_s0(deploy=True)
    deploy_model.eval()
    deploy_model.load_state_dict(model.state_dict())
    
    if input is not None:
        o = deploy_model(x)
        # print(o)
        # print(output)
        print((output - o).sum())
    # if save_path is not None:
    #     torch.save(model.state_dict(), save_path)
    return deploy_model

if __name__ == '__main__':
    net = MobileOneNet()
    print(net)
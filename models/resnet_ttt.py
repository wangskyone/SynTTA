import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
# from robustbench.model_zoo.architectures.utils_architectures import normalize_model
import torchvision.models as models

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class ResNet_FATA(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet_FATA, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 1000)

        self.layer3_p = deepcopy(self.layer3)
        self.layer4_p = deepcopy(self.layer4)
        self.avgpool_p = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_p = deepcopy(self.fc)

        self.u_sig = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)

        u = torch.mean(out, dim=[2, 3])
        u_sig_temp = (torch.std(u, dim=0) / max(torch.std(u, dim=0))).to(u.device)
        b, c, _, _ = out.shape
        alpha, beta = torch.distributions.Normal(1,1).sample((b,c, 1, 1)).to(out.device), torch.distributions.Normal(1,1).sample((b,c, 1, 1)).to(out.device)        

        if self.u_sig == None:
            self.u_sig = u_sig_temp.detach()
        else:
            # EMA update
            self.u_sig = 0.95 * self.u_sig + 0.05 * u_sig_temp.detach()

        out_p = alpha * out + (beta - alpha) * self.u_sig.view(1, c, 1, 1)


        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        out_p = self.layer3_p(out_p)
        out_p = self.layer4_p(out_p)
        out_p = self.avgpool_p(out_p)
        out_p = torch.flatten(out_p, 1)
        out_p = self.fc_p(out_p)

        return out, out_p



def resnet18(num_classes=1000):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    try:
        # Load pre-trained weights from torchvision
        pretrained_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        pretrained_dict = pretrained_model.state_dict()
        model_dict = model.state_dict()
        
        # Filter out the fc layer
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'fc' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("Loaded ImageNet-1k pre-trained weights for resnet18 (without fc layer)")
    except:
        print("Failed to load pre-trained weights, using randomly initialized weights")
    
    return model


def resnet34(num_classes=1000):
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    try:
        # Load pre-trained weights from torchvision
        pretrained_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        pretrained_dict = pretrained_model.state_dict()
        model_dict = model.state_dict()
        
        # Filter out the fc layer
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'fc' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("Loaded ImageNet-1k pre-trained weights for resnet34 (without fc layer)")
    except:
        print("Failed to load pre-trained weights, using randomly initialized weights")
    
    return model


def resnet50(num_classes=1000):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    try:
        # Load pre-trained weights from torchvision
        pretrained_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        pretrained_dict = pretrained_model.state_dict()
        model_dict = model.state_dict()
        
        # Filter out the fc layer
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'fc' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("Loaded ImageNet-1k pre-trained weights for resnet50 (without fc layer)")
    except:
        print("Failed to load pre-trained weights, using randomly initialized weights")
    
    return model


def resnet101(num_classes=1000):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
    try:
        # Load pre-trained weights from torchvision
        pretrained_model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        pretrained_dict = pretrained_model.state_dict()
        model_dict = model.state_dict()
        
        # Filter out the fc layer
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'fc' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("Loaded ImageNet-1k pre-trained weights for resnet101 (without fc layer)")
    except:
        print("Failed to load pre-trained weights, using randomly initialized weights")


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
}

# def get_resnet50_fata():
#     model = ResNet_FATA(Bottleneck, [3, 4, 6, 3])
#     state_dict = torch.load('/home/server/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth')
#     new_stat_dict = deepcopy(state_dict)
#     for k, v in state_dict.items():
#         if 'layer3' in k:
#             new_stat_dict[k.replace('layer3', 'layer3_p')] = state_dict[k]
#         if 'layer4' in k:
#             new_stat_dict[k.replace('layer4', 'layer4_p')] = state_dict[k]
#         if 'fc' in k:
#             new_stat_dict[k.replace('fc', 'fc_p')] = state_dict[k]
        
#     model.load_state_dict(new_stat_dict)
#     model = normalize_model(model, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     return model



class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""
    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x


class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet50', head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)
    
class ExtractorHead(nn.Module):
	def __init__(self, ext, head):
		super(ExtractorHead, self).__init__()
		self.ext = ext
		self.head = head

	def forward(self, x):
		return self.head(self.ext(x))
    


if __name__ == "__main__":
    model = resnet50(7)
    # print(model)
    # model = get_resnet50_fata()
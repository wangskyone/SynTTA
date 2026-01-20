# coding=utf-8
import torch.nn as nn
import torch.nn.utils.weight_norm as weightNorm
from torchvision import models
import torchvision
import torch
import timm  #load ViT or MLP-mixer

vgg_dict = {"vgg11": models.vgg11, "vgg13": models.vgg13, "vgg16": models.vgg16, "vgg19": models.vgg19,
            "vgg11bn": models.vgg11_bn, "vgg13bn": models.vgg13_bn, "vgg16bn": models.vgg16_bn, "vgg19bn": models.vgg19_bn}






class VGGBase(nn.Module):
    def __init__(self, vgg_name):
        super(VGGBase, self).__init__()
        model_vgg = vgg_dict[vgg_name](pretrained=True)
        self.features = model_vgg.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module(
                "classifier"+str(i), model_vgg.classifier[i])
        self.in_features = model_vgg.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


res_dict = {"resnet18": models.resnet18, 
            "resnet34": models.resnet34, 
            "resnet50": models.resnet50,
            "resnet101": models.resnet101, 
            "resnet152": models.resnet152, 
            "resnext50": models.resnext50_32x4d, 
            "resnext101": models.resnext101_32x8d}

class ResBase(nn.Module):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        weights_name = res_name.replace('resnet', 'ResNet') + '_Weights'
        if hasattr(models, weights_name):
            weights = getattr(models, weights_name).IMAGENET1K_V1
        else:
            raise ValueError(f"Unsupported ResNet model: {res_name}")
    

        model_resnet = res_dict[res_name](weights=weights)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class ResNet(nn.Module):
    def __init__(self, res_name, num_classes, type="bn"):
        super(ResNet, self).__init__()
        self.feature_extractor = ResBase(res_name)
        self.classifier = feat_classifier(num_classes, bottleneck_dim=self.feature_extractor.in_features)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits


class DTNBase(nn.Module):
    def __init__(self):
        super(DTNBase, self).__init__()
        self.conv_params = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.ReLU()
        )
        self.in_features = 256*4*4

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x


class LeNetBase(nn.Module):
    def __init__(self):
        super(LeNetBase, self).__init__()
        self.conv_params = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.in_features = 50*4*4

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x


class ViTBase(nn.Module):
    def __init__(self,model_name):
        self.KNOWN_MODELS = {
        'ViT-B16': 'vit_base_patch16_224_in21k', 
        'ViT-B32': 'vit_base_patch32_224_in21k',
        'ViT-L16': 'vit_large_patch16_224_in21k',
        'ViT-L32': 'vit_large_patch32_224_in21k',
        'ViT-H14': 'vit_huge_patch14_224_in21k'
    }
    
        self.FEAT_DIM = {
        'ViT-B16': 768, 
        'ViT-B32': 768,
        'ViT-L16': 1024,
        'ViT-L32': 1024,
        'ViT-H14': 1280
    }    
        super().__init__()
        self.vit_backbone = timm.create_model(self.KNOWN_MODELS[model_name],pretrained=True,num_classes=0)
        self.in_features = self.FEAT_DIM[model_name]
    
    def forward(self,x):
        return self.vit_backbone(x)
        


effnet_dict = {"efficientnet_b0": models.efficientnet_b0, 
         "efficientnet_b1": models.efficientnet_b1,
         "efficientnet_b2": models.efficientnet_b2,
         "efficientnet_b3": models.efficientnet_b3,
         "efficientnet_b4": models.efficientnet_b4,
         "efficientnet_b5": models.efficientnet_b5,
         "efficientnet_b6": models.efficientnet_b6,
         "efficientnet_b7": models.efficientnet_b7}


class EfficientBase(nn.Module):
    def __init__(self,backbone="efficientnet_b4"):
        super().__init__()
        self.network = effnet_dict[backbone](pretrained=True)
        self.in_features = self.network.classifier[1].in_features
        self.network.classifier = Identity()
        
        
    def forward(self,x):
        return self.network(x)



# mlp_mixer_path = {'Mixer-B16':models.mlp_mixer_b16_path,
#                   'Mixer-L16':models.mlp_mixer_l16_path}

class MLPMixer(nn.Module):
    KNOWN_MODELS = {
        'Mixer-B16': timm.models.mlp_mixer.mixer_b16_224_in21k,
        'Mixer-L16': timm.models.mlp_mixer.mixer_l16_224_in21k,
    }
    def __init__(self,backbone="Mixer-L16"):
        super().__init__()
        func = self.KNOWN_MODELS[backbone]
        self.network = func(pretrained=True)
        self.in_features = self.network.norm.normalized_shape[0]
        self.network.head = Identity()

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.network(x)
    


class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        # self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x


class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(
                nn.Linear(bottleneck_dim, class_num), name="weight")
            # self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            # self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x


class feat_classifier_two(nn.Module):
    def __init__(self, class_num, input_dim, bottleneck_dim=256):
        super(feat_classifier_two, self).__init__()
        self.type = type
        self.fc0 = nn.Linear(input_dim, bottleneck_dim)
        # self.fc0.apply(init_weights)
        self.fc1 = nn.Linear(bottleneck_dim, class_num)
        # self.fc1.apply(init_weights)

    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        return x

        
class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import inception_v3


class Model(nn.Module):
    def _init_(self, num_classes=8):
        super(Model, self)._init_()
        self.seg = GCN(14, 512)

        # Load pretrained Inception v3 model
        inception = inception_v3(pretrained=True)
        # Modify input layer to accept 15 channels
        inception.Conv2d_1a_3x3.conv = nn.Conv2d(15, 32, kernel_size=3, stride=2, bias=False)

        # Replace the first layer of Inception v3 with the modified one
        self.features = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Linear(2048, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        seg = F.sigmoid(self.seg(torch.cat((x, x, x), dim=1)))
        y = torch.cat((seg, x), dim=1)
        out = self.features(y)
        out = self.classifier(out)
        return out, seg


class _GlobalConvModule(nn.Module):
    def _init_(self, in_dim, out_dim, kernel_size):
        super(GlobalConvModule, self).init_()
        pad0 = int((kernel_size[0] - 1) / 2)
        pad1 = int((kernel_size[1] - 1) / 2)
        # kernel size had better be odd number so as to avoid alignment error
        super(GlobalConvModule, self).init_()
        self.conv_l1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r
        return x


class _BoundaryRefineModule(nn.Module):
    def _init_(self, dim):
        super(BoundaryRefineModule, self).init_()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        out = x + residual
        return out


class GCN(nn.Module):
    def _init_(self, num_classes, input_size):
        super(GCN, self)._init_()
        self.input_size = input_size
        resnet = models.resnet152(pretrained=True)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.gcm1 = _GlobalConvModule(2048, num_classes, (7, 7))
        self.gcm2 = _GlobalConvModule(1024, num_classes, (7, 7))
        self.gcm3 = _GlobalConvModule(512, num_classes, (7, 7))
        self.gcm4 = _GlobalConvModule(256, num_classes, (7, 7))

        self.brm1 = _BoundaryRefineModule(num_classes)
        self.brm2 = _BoundaryRefineModule(num_classes)
        self.brm3 = _BoundaryRefineModule(num_classes)
        self.brm4 = _BoundaryRefineModule(num_classes)
        self.brm5 = _BoundaryRefineModule(num_classes)
        self.brm6 = _BoundaryRefineModule(num_classes)
        self.brm7 = _BoundaryRefineModule(num_classes)
        self.brm8 = _BoundaryRefineModule(num_classes)
        self.brm9 = _BoundaryRefineModule(num_classes)

    def forward(self, x):
        # if x: 512
        fm0 = self.layer0(x)  # 256
        fm1 = self.layer1(fm0)  # 128
        fm2 = self.layer2(fm1)  # 64
        fm3 = self.layer3(fm2)  # 32
        fm4 = self.layer4(fm3)  # 16

        gcfm1 = self.brm1(self.gcm1(fm4))  # 16
        gcfm2 = self.brm2(self.gcm2(fm3))  # 32
        gcfm3 = self.brm3(self.gcm3(fm2))  # 64
        gcfm4 = self.brm4(self.gcm4(fm1))  # 128

        fs1 = self.brm5(F.upsample(gcfm1, fm3.size()[2:], mode='bilinear') + gcfm2)  # 32
        fs2 = self.brm6(F.upsample(fs1, fm2.size()[2:], mode='bilinear') + gcfm3)  # 64
        fs3 = self.brm7(F.upsample(fs2, fm1.size()[2:], mode='bilinear') + gcfm4)  # 128
        fs4 = self.brm8(F.upsample(fs3, fm0.size()[2:], mode='bilinear'))  # 256
        out = self.brm9(F.upsample(fs4, self.input_size, mode='bilinear'))  # 512

        return out
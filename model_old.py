import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class FPN(nn.Module):
    def __init__(self, num_channels=256):
        super(FPN, self).__init__()

        self.lateral_layer_5 = nn.Conv2d(2048, num_channels, kernel_size=1, stride=1, padding=0)
        self.lateral_layer_4 = nn.Conv2d(1024, num_channels, kernel_size=1, stride=1, padding=0)
        self.lateral_layer_3 = nn.Conv2d(512, num_channels, kernel_size=1, stride=1, padding=0)
        self.lateral_layer_2 = nn.Conv2d(256, num_channels, kernel_size=1, stride=1, padding=0)

        self.output_layer_5 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.output_layer_4 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.output_layer_3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.output_layer_2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)

    def _upsample_like(self, source, target):
        _, _, height, width = target.shape
        return F.interpolate(source, size=(height, width), mode='nearest')

    def forward(self, C2, C3, C4, C5):
        # No ReLu???

        P5 = self.lateral_layer_5(C5)
        P5_upsampled = self._upsample_like(P5, C4)
        P5 = self.output_layer_5(P5)

        P4 = self.lateral_layer_4(C4)
        P4_upsampled = self._upsample_like(P4, C3)
        P3 = self.output_layer_3(P4 + P5_upsampled)

        P3= self.lateral_layer_3(C3)
        P3_upsampled = self._upsample_like(P3, C2)
        P3 = self.output_layer_3(P3 + P4_upsampled)

        P2 = self.lateral_layer_2(C2)
        P2 = self.output_layer_2(P2 + P3_upsampled)

        return P2, P3, P4, P5

class Map_Branch(nn.Module):
    def __init__(self, num_channels_output):
        super(Map_Branch, self).__init__()

        self.convs_on_P5 = self._two_conv()
        self.convs_on_P4 = self._two_conv()
        self.convs_on_P3 = self._two_conv()
        self.convs_on_P2 = self._two_conv()

        self.convs_on_feature_maps = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_channels_output, kernel_size=1, stride=1, padding=0)
        )

    def _upsample(self, x, scale_factor):
        _, _, height, width = target.shape
        return F.upsample(x, scale_factor=scale_factor, mode='bilinear')

    def _two_conv(self):
        return nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, P2, P3, P4, P5):
        # No ReLu???
        D5 = F.interpolate(self.convs_on_P5(P5), scale_factor=8, mode='bilinear')
        D4 = F.interpolate(self.convs_on_P4(P4), scale_factor=4, mode='bilinear')
        D3 = F.interpolate(self.convs_on_P3(P3), scale_factor=2, mode='bilinear')
        D2 = self.convs_on_P2(P2)

        feature_maps = torch.cat((D2, D3, D4, D5), dim=1)

        return self.convs_on_feature_maps(feature_maps)

class NADS_Net(torch.nn.Module):
    def __init__(self):
        super(NADS_Net, self).__init__()

        self.resnet50_modules = nn.ModuleList(list(torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True).children())[:-2]).eval()
        # self.resnet50_modules = nn.ModuleList(list(resnet50(pretrained=True).children())[:-2]).eval()
        self.FPN = self._make_FPN()
        # self.parts_heatmap_branch = self._make_map_branch(10)
        # self.PAF_branch = self._make_map_branch(16)
        self.parts_heatmap_branch = self._make_map_branch(19)
        self.PAF_branch = self._make_map_branch(38)

        def init_weights(m):
            if type(m) == nn.Conv2d:
                torch.nn.init.normal_(m.weight, mean=0, std=0.01)
                m.bias.data.fill_(0.0)

        self.FPN.apply(init_weights)
        self.parts_heatmap_branch.apply(init_weights)
        self.PAF_branch.apply(init_weights)

    def _make_FPN(self):
        return FPN()

    def _make_map_branch(self, num_output_channels):
        return Map_Branch(num_output_channels)

    def forward(self, x, part_heatmap_masks, PAF_masks):
        resnet50_outputs = []
        for i, model in enumerate(self.resnet50_modules):
            x = model(x)
            if i in [4, 5, 6, 7]:
                resnet50_outputs.append(x)
        C2, C3, C4, C5 = resnet50_outputs

        P2, P3, P4, P5 = self.FPN(C2, C3, C4, C5)

        part_heatmaps = self.parts_heatmap_branch(P2, P3, P4, P5)
        PAFs = self.PAF_branch(P2, P3, P4, P5)

        part_heatmaps *= part_heatmap_masks
        PAFs *= PAF_masks

        return part_heatmaps, PAFs
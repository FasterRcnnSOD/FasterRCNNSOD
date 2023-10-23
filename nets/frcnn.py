import torch.nn as nn

from nets.classifier import Resnet50RoIHead, SwinRoIHead
from nets.resnet50 import resnet50
from nets.swin_transformer import Swin_T
from nets.rpn import RegionProposalNetwork


class FasterRCNN(nn.Module):
    def __init__(self, num_classes,
                 mode="training",
                 feat_stride=16,
                 anchor_scales=[8, 16, 32],
                 ratios=[0.5, 1, 2],
                 backbone='vgg',
                 pretrained=False):
        super(FasterRCNN, self).__init__()
        self.feat_stride = feat_stride
        self.backbone = backbone
        if backbone == 'resnet50':
            self.extractor, classifier = resnet50(pretrained)
            self.rpn = RegionProposalNetwork(
                1024, 512,
                ratios=ratios,
                anchor_scales=anchor_scales,
                feat_stride=self.feat_stride,
                mode=mode
            )
            self.head = Resnet50RoIHead(
                n_class=num_classes + 1,
                roi_size=14,
                spatial_scale=1,
                classifier=classifier
            )
        elif backbone == 'swinT':
            self.extractor, classifier = Swin_T()
            self.rpn = RegionProposalNetwork(
                384, 512,
                ratios=ratios,
                anchor_scales=anchor_scales,
                feat_stride=self.feat_stride,
                mode=mode
            )
            self.head = SwinRoIHead(
                n_class=num_classes + 1,
                roi_size=14,
                spatial_scale=1,
                classifier=classifier,
                clsdim_in=768
            )

    def forward(self, x, scale=1., mode="forward"):
        if mode == "forward":
            img_size = x.shape[2:]
            base_feature = self.extractor.forward(x)
            if self.backbone == 'swinT':
                base_feature = base_feature.permute(0, 3, 1, 2)
            _, _, rois, roi_indices, _ = self.rpn.forward(base_feature, img_size, scale)
            roi_cls_locs, roi_scores = self.head.forward(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores, rois, roi_indices
        elif mode == "extractor":
            base_feature = self.extractor.forward(x)
            if self.backbone == 'swinT':
                base_feature = base_feature.permute(0, 3, 1, 2)
            return base_feature
        elif mode == "rpn":
            base_feature, img_size = x
            rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn.forward(base_feature, img_size, scale)
            return rpn_locs, rpn_scores, rois, roi_indices, anchor
        elif mode == "head":
            base_feature, rois, roi_indices, img_size = x
            roi_cls_locs, roi_scores = self.head.forward(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

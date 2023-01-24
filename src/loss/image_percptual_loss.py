import torch
from torch import nn
from torchvision.models.vgg import vgg16

from .image_loss import ImageLoss
from .percptual_loss import TVLoss


class ImagePercptualLoss(nn.Module):
    def __init__(self, gradient=True, loss_weight=[20, 1e-4]):
        super(ImagePercptualLoss, self).__init__()

        # ImageLoss:L2+Lgp
        self.gradient = gradient
        self.loss_weight = loss_weight
        self.image_loss = ImageLoss(gradient=self.gradient, loss_weight=self.loss_weight)

        # PercptualLoss
        vgg = vgg16(pretrained=True)
        
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_images, target_images):
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images[:, :3, :, :]), self.loss_network(target_images[:, :3, :, :]))
        # Image Loss
        image_loss = self.image_loss(out_images, target_images)
        # # TV Loss
        tv_loss = self.tv_loss(out_images)
        return image_loss + 0.006 * perception_loss + 2e-8 * tv_loss



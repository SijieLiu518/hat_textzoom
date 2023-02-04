import torch
from torch import nn
# from torchvision.models.vgg import vgg16
import sys
sys.path.append('/home/videt/lsj/hat_textzoom/src')
from model.crnn import CRNN

from .image_loss import ImageLoss
from .percptual_loss import TVLoss

class CRNNImagePercptualLoss(nn.Module):
    def __init__(self, gradient=True, loss_weight=[20, 1e-4]):
        super(CRNNImagePercptualLoss, self).__init__()

        # ImageLoss:L2+Lgp
        self.gradient = gradient
        self.loss_weight = loss_weight
        self.image_loss = ImageLoss(gradient=self.gradient, loss_weight=self.loss_weight)

        # PercptualLoss
        # vgg = vgg16(pretrained=True)
        crnn = CRNN(32, 1, 37, 256)
        # crnn = crnn.to(self.device)
        crnn_path = 'loss/crnn.pth'
        print('loading pretrained crnn model from %s' % crnn_path)
        crnn.load_state_dict(torch.load(crnn_path))
        
        loss_network = nn.Sequential(*list(crnn.cnn)).eval()
        # loss_network = nn.Sequential(*list(crnn.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_images, target_images):
        # Perception Loss
        perception_loss = self.mse_loss(
            self.loss_network(parse_crnn_data(out_images[:, :3, :, :])), 
            self.loss_network(parse_crnn_data(target_images[:, :3, :, :])))
        # Image Loss
        image_loss = self.image_loss(out_images, target_images)
        # # TV Loss
        tv_loss = self.tv_loss(out_images)
        return image_loss + 0.006 * perception_loss + 2e-8 * tv_loss


def parse_crnn_data(imgs_input):
        imgs_input = torch.nn.functional.interpolate(imgs_input, (32, 100), mode='bicubic')
        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        return tensor

if __name__ == '__main__':    
    loss = CRNNImagePercptualLoss()
    out_images = torch.zeros(7, 3, 32, 128)
    target_images = torch.zeros(7, 3, 32, 128)
    loss(out_images, target_images)
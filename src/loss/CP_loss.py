import torch
from torch import nn
# from torchvision.models.vgg import vgg16
import sys
sys.path.append('/home/videt/lsj/hat_textzoom/src')
from model.crnn import CRNN

from .image_loss import ImageLoss
from .percptual_loss import TVLoss


class ContentPercptualLoss(nn.Module):
    def __init__(self, gradient=True, loss_weight=[20, 1e-4]):
        super(ContentPercptualLoss, self).__init__()

        # ImageLoss:L2+Lgp
        self.gradient = gradient
        self.loss_weight = loss_weight
        self.image_loss = ImageLoss(gradient=self.gradient, loss_weight=self.loss_weight)

        # ContentPercptualLoss
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        crnn = CRNN(32, 1, 37, 256)
        crnn = crnn.to(self.device)
        crnn_path = 'loss/crnn.pth'
        print('loading pretrained crnn model from %s' % crnn_path)
        crnn.load_state_dict(torch.load(crnn_path))
        feature_map1 = nn.Sequential(crnn.cnn[:3]).eval()
        feature_map2 = nn.Sequential(crnn.cnn[3:6]).eval()
        feature_map3 = nn.Sequential(crnn.cnn[6:12]).eval()
        feature_map4 = nn.Sequential(crnn.cnn[12:18]).eval()
        feature_map5 = nn.Sequential(crnn.cnn[18:]).eval()
        
        # loss_network = nn.Sequential(*list(crnn.cnn)).eval()
        # for param in loss_network.parameters():
        #     param.requires_grad = False
        for feature_map in [feature_map1, feature_map2, feature_map3, feature_map4, feature_map5]:
            for param in feature_map.parameters():
                param.requires_grad = False
        self.feature_map1 = feature_map1
        self.feature_map2 = feature_map2
        self.feature_map3 = feature_map3
        self.feature_map4 = feature_map4
        self.feature_map5 = feature_map5
        
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_images, target_images):
        out_images = out_images.to(self.device)
        target_images = target_images.to(self.device)
        # ContentPercptualLoss
        out = self.feature_map1(parse_crnn_data(out_images[:, :3, :, :]))
        target = self.feature_map1(parse_crnn_data(target_images[:, :3, :, :]))
        CP_loss = self.mse_loss(out, target)
        
        out = self.feature_map2(out)
        target = self.feature_map2(target)
        CP_loss += self.mse_loss(out, target)

        out = self.feature_map3(out)
        target = self.feature_map3(target)
        CP_loss += self.mse_loss(out, target)

        out = self.feature_map4(out)
        target = self.feature_map4(target)
        CP_loss += self.mse_loss(out, target)

        out = self.feature_map5(out)
        target = self.feature_map5(target)
        CP_loss += self.mse_loss(out, target)

        # Image Loss
        image_loss = self.image_loss(out_images, target_images)
        # # TV Loss
        tv_loss = self.tv_loss(out_images)
        # return image_loss + 0.006 * CP_loss + 2e-8 * tv_loss
        return 0.1*image_loss + 0.0005 * CP_loss


def parse_crnn_data(imgs_input):
        imgs_input = torch.nn.functional.interpolate(imgs_input, (32, 100), mode='bicubic')
        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        return tensor

if __name__ == '__main__':    
    loss = ContentPercptualLoss()
    out_images = torch.zeros(7, 3, 32, 128)
    target_images = torch.zeros(7, 3, 32, 128)
    loss(out_images, target_images)
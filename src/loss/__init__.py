# __init__.py
from .image_loss import *
from .image_percptual_loss import *
from .percptual_loss import *
from .crnn_percptual_loss import *
from .CP_loss import *
__all__ = ['ImageLoss', 'ImagePercptualLoss', 'CRNNImagePercptualLoss', 'ContentPercptualLoss']

# print("loss init file is called")
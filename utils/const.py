from enum import Enum

from models.fc import FC
from models.unet import UNET


class Models_types(Enum):
    FC = 1
    UNET_V1 = 2


def choose_type(model_type):
    if model_type == Models_types.FC:
        return FC()
    if model_type == Models_types.UNET_V1:
        return UNET()

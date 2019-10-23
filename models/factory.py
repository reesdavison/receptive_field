from .resnet import get_model as get_resnet_model
from .mobile_net_v2 import get_model as get_mobile_net_v2

def get_model(name):
    if 'resnet' in name:
        return get_resnet_model(name)
    elif 'mobile_net' in name:
        return get_mobile_net_v2(name)
    else:
        raise Exception('Unknown model type')

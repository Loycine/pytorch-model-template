from .criterion import BCELoss, L1Loss, L2Loss
from .toy_net import ToyNet
from torch import Tensor


class NetBuilder():
    def weights_init(self, m):
        pass

    def build_toy_net(self, output_dim: int) -> ToyNet:
        toy_net = ToyNet(output_dim)
        return toy_net

    def build_criterion(self, arch: str) -> Tensor:
        if arch == 'bce':
            net = BCELoss()
        elif arch == 'l1':
            net = L1Loss()
        elif arch == 'l2':
            net = L2Loss()
        else:
            raise Exception('Architecture undefined!')
        return net

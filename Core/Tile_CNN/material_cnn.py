"""CNN on decoded PBR material maps, before texture sampling and shading."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MaterialCNN(nn.Module):
    version = 1

    def __init__(self, channels=10, hidden=16, rounds=2):
        super().__init__()
        if channels not in (10, 11) or hidden < 1 or rounds not in (1, 2):
            raise ValueError('MaterialCNN requires 10/11 canonical channels, positive hidden width and 1/2 rounds')
        self.channels, self.hidden, self.rounds = channels, hidden, rounds
        self.halo = rounds
        self.layers = nn.ModuleList([nn.ModuleList((nn.Conv2d(channels, hidden, 3),
                                                   nn.Conv2d(hidden, channels, 1))) for _ in range(rounds)])
        for _, last in self.layers:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def valid(self, image):
        """Consume a real halo of width rounds; retain encoded normal channels."""
        value = image
        for spatial, pointwise in self.layers:
            residual = pointwise(F.leaky_relu(spatial(value), negative_slope=.01))
            value = value[..., 1:-1, 1:-1] + residual
        return value.clamp(0, 1)

    def forward(self, image):
        return self.valid(F.pad(image, (self.halo,)*4, mode='circular'))

    def tiled(self, image, tile_size=128):
        if tile_size < 1:
            return self(image)
        height, width = image.shape[-2:]
        result = torch.empty_like(image)
        r = self.halo
        for y in range(0, height, tile_size):
            h = min(tile_size, height-y)
            iy = torch.arange(y-r, y+h+r, device=image.device) % height
            for x in range(0, width, tile_size):
                w = min(tile_size, width-x)
                ix = torch.arange(x-r, x+w+r, device=image.device) % width
                result[..., y:y+h, x:x+w] = self.valid(image.index_select(-2, iy).index_select(-1, ix))
        return result

    def export_arrays(self):
        arrays = dict(material_cnn_version=np.array(self.version, dtype=np.int32),
                      material_cnn_mode=np.array('native_material_residual_leaky_relu_0.01_repeat_clamp'),
                      material_cnn_shape=np.array([self.channels, self.hidden, self.rounds], dtype=np.int32))
        arrays.update({'material_cnn_'+k.replace('.', '_'):v.detach().cpu().numpy()
                       for k,v in self.state_dict().items()})
        return arrays

    @classmethod
    def from_arrays(cls, arrays):
        model = cls(*map(int, arrays['material_cnn_shape']))
        if (int(arrays['material_cnn_version']) != cls.version
                or str(arrays['material_cnn_mode']) != str(model.export_arrays()['material_cnn_mode'])):
            raise ValueError('Unsupported material CNN export')
        model.load_state_dict({k:torch.from_numpy(np.array(arrays['material_cnn_'+k.replace('.', '_')], copy=True))
                               for k in model.state_dict()}, strict=True)
        return model

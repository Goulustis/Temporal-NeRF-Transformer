from torch.utils.data import Dataset
import jax
import numpy as np
import os.path as osp
from absl import logging

from hypernerf import utils


def expand_dim(x):
    if x.ndim == 1 and x[0].ndim == 0:
        x = x[..., None]
    
    return x

def _flatten_fnc(x):
    x = x.squeeze()

    x = expand_dim(x)
    
    x = x.reshape(-1, x.shape[-1])

    return x


class ColcamDataset(Dataset):
    """
    camera dataset, Does nerfies things slightly better
    """
    def __init__(self, data_dict, shuffle=True, flatten=True):
        """
        inp:
            data_dict: dictionary of structure:{"orientations" : ndarray: (n,),
                                                "directions"   : ndarray: (n, h, w, 3),
                                                "rgb"          : ndarray: (n, h, w, 3),
                                                "origins"      : (n,),
                                                "pixels"       : np.ndarray: (h, w),
                                                "metadata":{"warp": ndarray: (n,),
                                                            ...   : ndarray: (n,),
                                                            }  
                                                }
        """
        self.data_dict = data_dict
        self.shuffle = shuffle
        self.flatten = flatten
        self.img_size = data_dict["pixels"].shape[:2]
        self.n_pix = np.prod(self.img_size)

        if self.flatten:
            self.data_dict = jax.tree_map(_flatten_fnc, self.data_dict)
            self.get_item_fnc = self._getitem_flatten
            self.rgb = self.data_dict.pop("rgb")
            self.dirs = self.data_dict.pop("directions")
            self.pixels = self.data_dict.pop("pixels")
        else:
            self.data_dict = jax.tree_map(expand_dim, self.data_dict)
            self.get_item_fnc = self._getitem_unflatten
    
    def __len__(self):
        size = 250000*10000
        return  size
    
    def __getitem__(self, idx):
        return self.get_item_fnc(idx)
    
    def _getitem_unflatten(self, idx):
        data = jax.tree_map(lambda x : x[idx], self.data_dict)
        assert "rgb" in data, "rgb should exist in batch, it got popped somewhere, fix the bug"

        data["rgb"] = data["rgb"].astype(np.float32)
        data["directions"] = data["directions"].astype(np.float32)
        data["origins"] = np.broadcast_to(data['origins'], data['directions'].shape)
        data["pixels"] = self.data_dict["pixels"]
        data["metadata"] = jax.tree_map(
                               lambda x : np.broadcast_to(x[None,None],
                                                       (self.img_size[0], self.img_size[1], x.shape[-1])),
                               data["metadata"])
       
        
        return data



    def _getitem_flatten(self, idx):
        """
        if self.{rgb, dirs, pixels} are None, the dataset is
        running in testing mode. get flatten items will work only
        during training
        """
        if self.shuffle:
            idx = np.random.randint(len(self.rgb))
        
        idx = idx%len(self.rgb)
        rgb = self.rgb[idx]
        dirs = self.dirs[idx]

        d1_idx = idx//self.n_pix
        _get_item_fnc = lambda x : x[d1_idx]
        data_dict = jax.tree_map(_get_item_fnc, self.data_dict)

        d2_idxs = idx%self.n_pix
        pixels = self.pixels[d2_idxs]

        data_dict["rgb"] = rgb.astype(np.float32)
        data_dict["directions"] = dirs.astype(np.float32)
        data_dict["pixels"] = pixels.astype(np.float32)

        return data_dict

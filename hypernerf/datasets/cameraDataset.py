from torch.utils.data import Dataset
import jax
import numpy as np
import os.path as osp
from absl import logging

from hypernerf import utils


# def _flatten_fnc(x):
#     x = x.squeeze()
    
#     if not isinstance(x, np.ndarray):
#         x = np.asarray(x)

#     if x.ndim == 1 and x[0].ndim == 0:
#         x = np.expand_dims(x, -1)
    
#     x = x.reshape(-1, x.shape[-1])

#     return x

def expand_dim(x):
    if x.ndim == 1 and x[0].ndim == 0:
        x = x[..., None]
    
    return x

def _flatten_fnc(x):
    x = x.squeeze()

    x = expand_dim(x)
    
    x = x.reshape(-1, x.shape[-1])

    return x

class EcamDataset(Dataset):
    """
    Dataset for event images    
    """
    def __init__(self, eimgs, prev_dict, next_dict, shuffle=True, flatten=True, **kwargs):
        """
        input:
            eimgs (np.ndarray): event image of shape (n, h, w)
            img_size (tupple): (w, h) of image
            prev_dict, next_dict: dictionary of struct: {"orientations" : ndarray: (n, 3),
                                                        "directions"   : ndarray: (n, h, w, 3),
                                                        "origins"      : (n, 3),
                                                        "pixels"       : np.ndarray: (h, w)
                                                        "metadata":{"warp": ndarray: (n,),
                                                                    ...   : ndarray: (n,),
                                                                    }  
                                                        }
        """
        self.eimgs = eimgs
        self.prev_dict = prev_dict
        self.next_dict = next_dict
        self.shuffle=shuffle
        self.flatten = flatten

        assert self.flatten, "event camera dataset only supports flatten"

        self.img_size = prev_dict["pixels"].shape[:2]
        self.n_pix = np.prod(self.img_size)

        # flatten everything
        if self.eimgs.shape[-1] > 6:
            self.eimgs = self.eimgs.reshape(-1,1)
        else:
            self.eimgs = eimgs.reshape(-1, self.eimgs.shape[-1])    
        
        
        self.prev_dict = jax.tree_map(_flatten_fnc, self.prev_dict)
        self.next_dict = jax.tree_map(_flatten_fnc, self.next_dict)

        # pop them out because their index is different (look at the docs above)
        self.prev_dir = self.prev_dict.pop("directions")
        self.next_dir = self.next_dict.pop("directions")
        self.pixels = self.next_dict.pop("pixels").astype(np.float32)

        self.non_zero_idxs = None
        self.sample_non_zeros = kwargs.get("sample_non_zeros")
        if kwargs.get("sample_non_zeros"):
            non_zero_idxs_path = osp.join(kwargs["data_dir"], "cache", f'{kwargs["image_scale"]}x', "non_zero_idxs.npy")
            if osp.exists(non_zero_idxs_path):
                self.non_zero_idxs = np.load(non_zero_idxs_path)
            else:
                logging.info("creating non zero idxs for event images...")
                idxs = np.arange(len(self.eimgs), dtype=np.uint32)
                self.non_zero_idxs = idxs[(self.eimgs != 0).squeeze()]
                np.save(non_zero_idxs_path, self.non_zero_idxs)
                del idxs


    
    def __len__(self):
        # set size to be infinity, because hypernerf just goes infinity
        # so choose a number that is effectively infinity
        size = 250000*500000
        assert size > len(self.eimgs), "event camera dataset size is smaller than event frames, be careful!"
        return  size
    

    def __getitem__(self, idx):
        idx = np.random.randint(len(self.eimgs)) if self.shuffle else idx
        if self.sample_non_zeros and (np.random.rand() < 0.8) and self.shuffle:
            idx = np.random.choice(self.non_zero_idxs)
                
        idx = idx%len(self.eimgs)
        ev_data = self.eimgs[idx].astype(np.float32)
        prev_dir = self.prev_dir[idx].astype(np.float32)
        next_dir = self.next_dir[idx].astype(np.float32)

        # indexes of 1d elements
        d1_idx = idx//self.n_pix
        _get_item_fnc = lambda x : x[d1_idx]
        prev_data = jax.tree_map(_get_item_fnc, self.prev_dict)
        next_data = jax.tree_map(_get_item_fnc, self.next_dict)

        # indexes of 2d elements (self.pixel)
        d2_idx = idx%self.n_pix
        pixel = self.pixels[d2_idx].astype(np.float32)

        prev_data["pixels"] = pixel
        next_data["pixels"] = pixel
        prev_data["directions"] = prev_dir
        next_data["directions"] = next_dir

        batch_data = {"evs": ev_data, "prev_data": prev_data, "next_data": next_data}
        return {"evs_data":batch_data}



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
        # self.rgb, self.dirs, self.pixels = None

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
        # assert size > len(self.rgb), "color camera dataset size is smaller than event frames, be careful!"
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

        return {"col_data":data_dict}


class MulticamDataset(Dataset):
    """
    small wrapper for the color and event camera dataset
    """

    def __init__(self, colcam_set:ColcamDataset, ecam_set:EcamDataset):
        self.colcam_set = colcam_set
        self.ecam_set = ecam_set

        self.id_source = self.colcam_set if colcam_set is not None else self.ecam_set
    
    def __len__(self):
        return len(self.id_source)
    
    def __getitem__(self, idx):
        data = {}
        if self.colcam_set is not None:
            data["col_data"] = self.colcam_set[idx]["col_data"]
        
        if self.ecam_set is not None:
            data["evs_data"] = self.ecam_set[idx]["evs_data"]

        return data #{"col_data": col_data, "evs_data": evs_data}
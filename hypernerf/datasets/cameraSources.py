import torch
import json

from absl import logging
import gin
import numpy as np
import jax

import json
import os
import os.path as osp
import glob
import copy
from flax import jax_utils
from torch.utils.data import DataLoader

from hypernerf import camera as cam
from hypernerf import utils
from hypernerf.datasets.nerfies import NerfiesDataSource, _load_image
from hypernerf.datasets.cameraDataset import EcamDataset, ColcamDataset


# TODO: plan:
# throw away the last event image
# on rgb side, keep last trigger img so that [r,e,r,e]
                                                   #^ that's thrown away 

@gin.configurable
class EcamDataSource(NerfiesDataSource):
    """
    Data loader for event camera dataset
    """
    def __init__(self,
               data_dir: str = gin.REQUIRED,  # scene/ecam_set
               image_scale: int = gin.REQUIRED,
               shuffle_pixels: bool = False,
               camera_type: str = 'json',
               test_camera_trajectory: str = 'orbit-mild',
               **kwargs):

        super().__init__(data_dir, image_scale, shuffle_pixels, camera_type, test_camera_trajectory, **kwargs)

        self.eimg_f = self.data_dir/"eimgs"/f"eimgs_{image_scale}x.npy"
        self.cache_dir = self.data_dir/"cache"/f"{image_scale}x"
        

    def load_rgb(self, item_id):
        return None
    
    def get_time(self, item_id):
        return self.metadata_dict[item_id]["t"]

    def get_item(self, item_id, scale_factor=1):
        data = super().get_item(item_id, scale_factor, do_print=False)
        if data.get("rgb") is None:
            del data["rgb"]

        if data['metadata'].get('t') is not None:
            data['metadata']['t'] = np.atleast_1d(self.get_time(item_id))

        logging.info(
          '\tLoaded item %s: scale_factor=%f, metadata=%s',
          item_id,
          scale_factor,
          str(data.get('metadata')))
        
        return data

    def _camera_to_rays_fn(self, item):
        """Converts camera params to rays."""
        camera_params = item.pop('camera_params')

        camera = cam.Camera(**camera_params)

        if not hasattr(self, "pixels"):
            self.pixels = camera.get_pixel_centers()

        directions = camera.pixels_to_rays(self.pixels).astype(np.float16)
        origins = camera.position[None, None, :] 
        item['origins'] = origins
        item['directions'] = directions
        return item
        

    def _save_loaded_data(self, data_dict):
        os.makedirs(self.cache_dir, exist_ok=True)

        keys = set(data_dict.keys()) - set(["metadata"])
        for d_name in keys:
            np.save(self.cache_dir/f"{d_name}.npy", data_dict[d_name])
        
        metadata = copy.deepcopy(data_dict["metadata"])
        metadata = jax.tree_map(lambda x : x.squeeze().tolist(), metadata)

        with (self.cache_dir/"metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)
    
    def _load_cache_data(self):
        get_f_name = lambda x : x.split("/")[-1].split(".")[0]

        np_fs = glob.glob(str(self.cache_dir/"*.npy")) 
        np_fs = [f for f in np_fs if not ("non_zero_idxs" in f)]   # not load non_zero_idx
        data_dict = {}
        for np_f in np_fs:
            data_dict[get_f_name(np_f)] = np.load(np_f)
        
        with (self.cache_dir/"metadata.json").open("r") as f:
            metadata = json.load(f)
        
        metadata = utils.dict_map(lambda x: np.array(x), metadata)
        data_dict["metadata"] = metadata
        self.pixels = data_dict["pixels"]

        return data_dict

    def _get_idx_cond(self, idxs):
        all_ids = np.array(self.all_ids)[:-1]
        cond = np.zeros(len(all_ids), dtype=np.bool)

        for idx in idxs:
            cond = cond | (all_ids == idx)
        
        return cond
    
    def _load_data_dict(self):
        all_ids = self.all_ids

        if not (self.cache_dir/"metadata.json").exists():
            data_list = utils.parallel_map(self.get_item, all_ids)
            data_list = utils.parallel_map(self._camera_to_rays_fn, data_list, show_pbar=True, desc="creating cameras")
            data_dict = utils.tree_collate(data_list)
            data_dict["metadata"] = jax.tree_map(lambda x : x.squeeze(), data_dict["metadata"])
            data_dict["pixels"] = self.pixels

            self._save_loaded_data(data_dict)
        else:
            logging.info(f"{type(self)} loading dataset cache")
            data_dict = self._load_cache_data()
        
        # data_dict = jax.tree_map(lambda x : torch.from_numpy(x), data_dict)
        return data_dict


    def _create_preloaded_dataset(self, item_ids, flatten=False, shuffle=True, ret_dict=False):
        """
            Create a pyTorch Dataset
        """
        data_dict = self._load_data_dict()
        pixels = data_dict.pop("pixels")

        prev_dict = jax.tree_map(lambda x : x[:-1], data_dict)
        next_dict = jax.tree_map(lambda x : x[1:], data_dict)
        eimgs = np.load(self.eimg_f)[:-1]  # drop the last one event image frame because the last frame happens at the last rgb frame
        # eimgs = torch.from_numpy(eimgs)

        keep_cond = self._get_idx_cond(item_ids)
        filter_fn = lambda x : x[keep_cond]
        prev_dict, next_dict = jax.tree_map(filter_fn, prev_dict), jax.tree_map(filter_fn, next_dict)
        eimgs = eimgs[keep_cond]

        prev_dict["pixels"] = pixels
        next_dict["pixels"] = pixels

        return EcamDataset(eimgs, prev_dict, next_dict, 
                           sample_non_zeros=True,
                           data_dir=self.data_dir, 
                           image_scale=self.image_scale, 
                           shuffle=shuffle,
                           flatten=flatten)

    
    def iterator_from_dataset(self, 
                          dataset,
                          batch_size: int,
                          repeat: bool = True,
                          prefetch_size: int = 0,
                          devices= None):
        

        # shuffle takes too long in dataloader, will random sample in dataset instead
        it = DataLoader(dataset, 
                        batch_size=max(1, batch_size), 
                        shuffle=False,
                        collate_fn = utils.tree_collate)

        def _prepare_data_batched(xs):
            devices = jax.local_devices()
            _prepare = lambda x : x.reshape((len(devices), -1) + x.shape[1:])

            return jax.tree_map(_prepare, xs)
        
        def _prepare_data_unbatched(xs):
            _prepare = lambda x : np.squeeze(x, axis=0)
            return jax.tree_map(_prepare, xs)

        if batch_size > 0:
            it = map(_prepare_data_batched, it)
        else:
            it = map(_prepare_data_unbatched, it)
            
        if prefetch_size > 0:
            it = jax_utils.prefetch_to_device(it, prefetch_size, devices)
        
        return it


@gin.configurable
class ColcamDataSource(EcamDataSource):

    def __init__(self,
               data_dir: str = gin.REQUIRED,  # scene/ecam_set
               image_scale: int = gin.REQUIRED,
               shuffle_pixels: bool = False,
               camera_type: str = 'json',
               test_camera_trajectory: str = 'orbit-mild',
               **kwargs):

        super().__init__(data_dir, image_scale, shuffle_pixels, camera_type, test_camera_trajectory, **kwargs)
        
    
    def _get_idx_cond(self, idxs):
        cond = np.zeros(len(self.all_ids), dtype=np.bool)
        all_ids = np.array(self.all_ids)

        for idx in idxs:
            cond = cond | (all_ids == idx)
        
        return cond

    def load_rgb(self, item_id: str) -> np.ndarray:
        return _load_image(self.rgb_dir / f'{item_id}.png').astype(np.float16)

    def _create_preloaded_dataset(self, item_ids, flatten=False, shuffle=True, ret_dict=False):
        
        data_dict = self._load_data_dict()
        pixels = data_dict.pop("pixels")

        keep_cond = self._get_idx_cond(item_ids)
        filter_fn = lambda x : x[keep_cond]
        data_dict = jax.tree_map(filter_fn, data_dict)
        
        data_dict["pixels"] = pixels

        return ColcamDataset(data_dict, shuffle=shuffle, flatten=flatten)
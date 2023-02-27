import json
from typing import Any, List, Tuple, Optional, Sequence, Union
import immutabledict
import functools

from absl import logging
import cv2
import gin
import numpy as np
import os.path as osp
from torch.utils.data import DataLoader

import jax
from flax import jax_utils
from hypernerf.utils import shard

from hypernerf.datasets.cameraSources import EcamDataSource, ColcamDataSource
from hypernerf.datasets.cameraDataset import MulticamDataset


# NOTE: only used for training



@gin.configurable
class EviesDataSource:
    """
        wrapper wrapping EcamDataset and ColcamDataset
    """    

    def __init__(self, data_dir: str = gin.REQUIRED, 
                       image_scale: int = gin.REQUIRED, 
                       shuffle_pixels: bool = False, 
                       camera_type: str = 'json', 
                       test_camera_trajectory: str = 'orbit-mild',
                       rgb_frac=0.7,
                       **kwargs):
        self.use_appearance_id = kwargs["use_appearance_id"]
        self.use_camera_id = kwargs["use_camera_id"]
        self.use_warp_id = kwargs["use_warp_id"]
        self.rgb_frac = rgb_frac

        self.colcam_source, self.ecam_source = None, None

        if rgb_frac > 0:
            self.colcam_source = ColcamDataSource(osp.join(data_dir, "colcam_set"),
                                                        image_scale=image_scale,
                                                        shuffle_pixels=shuffle_pixels,
                                                        camera_type=camera_type,
                                                        test_camera_trajectory=test_camera_trajectory,
                                                        **kwargs)

        if rgb_frac != 1:
            self.ecam_source = EcamDataSource(osp.join(data_dir, "ecam_set"),
                                                        image_scale=image_scale,
                                                        shuffle_pixels=shuffle_pixels,
                                                        camera_type=camera_type,
                                                        test_camera_trajectory=test_camera_trajectory,
                                                        **kwargs)

        self.id_source = self.ecam_source if (self.ecam_source is not None) else self.colcam_source

    #### for compatibility #####vvvvvvvvvvvvvvvv
    @property
    def all_ids(self):
        return None

    @property
    def train_ids(self):
        return None

    @property
    def val_ids(self):
        return None
    
    #### for compatibility #####^^^^^^^^^^^^^^^^
    
    @property
    def near(self) -> float:
        return self.id_source.near

    @property
    def far(self) -> float:
        return self.id_source.far
    
    def load_points(self, shuffle=False):
        return self.colcam_source.load_points(shuffle)

    @property
    @functools.lru_cache(maxsize=None)
    def appearance_ids(self) -> Sequence[int]:
        if not self.use_appearance_id:
            return tuple()
        return tuple(
            sorted(set([self.id_source.get_appearance_id(i) for i in self.train_ids])))

    @property
    @functools.lru_cache(maxsize=None)
    def camera_ids(self) -> Sequence[int]:
        if not self.use_camera_id:
            return tuple()
        return tuple(sorted(set([self.id_source.get_camera_id(i) for i in self.train_ids])))

    @property
    @functools.lru_cache(maxsize=None)
    def warp_ids(self) -> Sequence[int]:
        if not self.use_warp_id:
            return tuple()
        return tuple(sorted(set([self.id_source.get_warp_id(i) for i in self.train_ids])))

    @property
    def embeddings_dict(self):
        return immutabledict.immutabledict({
            'warp': self.id_source.warp_ids,
            'appearance': self.id_source.appearance_ids,
            'camera': self.id_source.camera_ids,
        })

    @property
    @functools.lru_cache(maxsize=None)
    def time_ids(self):
        if not self.use_time:
            return []
        return sorted(set([self.get_time_id(i) for i in self.train_ids]))

    def create_dataset(self,
                     item_ids=None,
                     flatten=False,
                     shuffle=False,
                     row_shuffle_buffer_size=1000000,
                     pixel_shuffle_buffer_size=1000000):

        colcam_set, ecam_set = None, None
        if self.colcam_source is not None:
            colcam_set = self.colcam_source._create_preloaded_dataset(
                self.colcam_source.train_ids,
                flatten=True,
                shuffle=True
            )
        
        if self.ecam_source is not None:
            ecam_set = self.ecam_source._create_preloaded_dataset(
                self.ecam_source.train_ids,
                flatten=True,
                shuffle=True
            )

        return colcam_set, ecam_set                     


    def create_iterator(self,
                        item_ids,
                        batch_size: int,
                        repeat: bool = True,
                        flatten: bool = False,
                        shuffle: bool = False,
                        prefetch_size: int = 0,
                        shuffle_buffer_size: int = 1000000,
                        devices = None):                                                                                                          
        
        colcam_set, ecam_set = self.create_dataset()
        multi_set = MulticamDataset(colcam_set, ecam_set)
        it = DataLoader(multi_set, batch_size=batch_size)

        n_rgb = int(batch_size*self.rgb_frac)
        n_evs = batch_size - n_rgb

        def _prepare_data(xs):
            devices = jax.local_devices()
            _prepare = lambda x : x.numpy().reshape((len(devices), -1) + x.shape[1:])

            if xs.get("col_data") is not None:
                xs["col_data"] = jax.tree_map(lambda x : x[:n_rgb], xs["col_data"])
            if xs.get("evs_data") is not None:
                xs["evs_data"] = jax.tree_map(lambda x : x[:n_evs], xs["evs_data"])

            return jax.tree_map(_prepare, xs)

        it = map(_prepare_data, it)
        if prefetch_size > 0:
            it = jax_utils.prefetch_to_device(it, prefetch_size, devices)
        
        return it
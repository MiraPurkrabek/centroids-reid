# encoding: utf-8
"""
@author: mikwieczorek
"""

import copy
import random
import torch
from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import Sampler


class FrameBasedSampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances, world_size=None, rank=None, frame_ids=None):
        # print(data_source)
        print("Initializing the FrameBasedSampler")

        if frame_ids is None:
            raise ValueError("Frame_ids are needed in the FrameBasedSampler")
        if batch_size > 8:
            raise ValueError("Batch size has to be lower or equal than 8 for FrameBasedSampler")

        self.pids = np.array([row[1] for row in data_source]).flatten()

        self.batch_size = batch_size  # This is the number of unique frames in reality

        self.frame_ids = np.array(frame_ids)
        self.unique_frames = np.unique(self.frame_ids)

        self.epoch = 0
        # self.return_size = self.batch_size * 4
        self.return_size = len(self.pids) // self.batch_size
        self.length = self.return_size


    def __iter__(self):
        # print("Iter of the sampler")
         # deterministically shuffle based on epoch
        np.random.seed(self.epoch)
        random.seed(self.epoch)  # Just in case ...
        
        selected_idxs = []
        while len(selected_idxs) < self.return_size:
            
            # Select random frame and IDs that are in the frame
            selected_frame = np.random.choice(self.unique_frames, size=1, replace=False)
            pids_in_frame = np.unique(self.pids[self.frame_ids == selected_frame])

            # If not enough IDs, skip
            # Otherwise sample IDs
            if self.batch_size > len(pids_in_frame):
                continue
            elif self.batch_size == len(pids_in_frame):
                selected_idxs.extend(list(pids_in_frame))
            else:
                random_pids_from_frame = list(np.random.choice(pids_in_frame, size=self.batch_size, replace=False))
                selected_idxs.extend(random_pids_from_frame)
        
        self.length = len(selected_idxs)
        return iter(selected_idxs)

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        self.epoch = epoch
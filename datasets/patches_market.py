# encoding: utf-8

import glob
import os.path as osp
import re
from collections import defaultdict

import pytorch_lightning as pl
from torch.utils.data import (DataLoader, Dataset, DistributedSampler,
                              SequentialSampler)

from .bases import (BaseDatasetLabelled, BaseDatasetLabelledPerPid,
                    ReidBaseDataModule, collate_fn_alternative, pil_loader)
from .samplers import get_sampler
from .transforms import ReidTransforms


class PatchesMarket(ReidBaseDataModule):
    """
    
    """
    dataset_dir = 'patches_market'

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.dataset_dir = osp.join(cfg.DATASETS.ROOT_DIR, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

    def setup(self):
        self._check_before_run()
        transforms_base = ReidTransforms(self.cfg)
        
        train, train_dict = self._process_dir(self.train_dir, relabel=True)
        self.train_dict = train_dict
        self.train_list = train
        self.train = BaseDatasetLabelledPerPid(train_dict, transforms_base.build_transforms(is_train=True), self.num_instances, self.cfg.DATALOADER.USE_RESAMPLING)

        query, query_dict = self._process_dir(self.query_dir, relabel=False)
        gallery, gallery_dict  = self._process_dir(self.gallery_dir, relabel=False)
        self.query_list = query
        self.gallery_list = gallery
        self.val = BaseDatasetLabelled(query+gallery, transforms_base.build_transforms(is_train=False))

        self._print_dataset_statistics(train, query, gallery)
        # For reid_metic to evaluate properly
        num_query_pids, num_query_imgs, num_query_cams = self._get_imagedata_info(query)
        num_train_pids, num_train_imgs, num_train_cams = self._get_imagedata_info(train)
        self.num_query = len(query)
        self.num_classes = num_train_pids
        
        # Used for inference with the original model
        # self.num_classes = 751

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)s(\d)_f([-\d]+)')

        pid_container = set()
        for img_path in img_paths:
            try:
                pid, _, _, _ = map(int, pattern.search(img_path).groups())
            except AttributeError:
                print(img_path)
                raise AttributeError
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset_dict = defaultdict(list)
        dataset = []

        for idx, img_path in enumerate(img_paths):
            pid, camid, _, frame_id = map(int, pattern.search(img_path).groups())
            camid += 1
            if pid == -1: continue  # junk images are just ignored
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid, idx))
            dataset_dict[pid].append((img_path, pid, camid, idx))

        return dataset, dataset_dict

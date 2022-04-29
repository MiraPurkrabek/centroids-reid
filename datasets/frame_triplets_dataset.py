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


class FrameTripletsDataset(ReidBaseDataModule):
    """
    
    """
    dataset_dir = 'frame_triplets_dataset'

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.dataset_dir = osp.join(cfg.DATASETS.ROOT_DIR, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

    def _get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid, *_ in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)

        if self.frame_ids is None:
            num_frames = -1
        else:
            frames = set(self.frame_ids)
            num_frames = len(frames)
        return num_pids, num_imgs, num_cams, num_frames

    def _print_dataset_statistics(self, train, query=None, gallery=None):
        num_train_pids, num_train_imgs, num_train_cams, num_train_frames = self._get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_query_frames = self._get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_gallery_frames = self._get_imagedata_info(
            gallery
        )

        print("Dataset statistics:")
        print("  -------------------------------------------------- ")
        print("  subset   | # ids | # images | # cameras | # frames ")
        print("  -------------------------------------------------- ")
        print(
            "  train    | {:5d} | {:8d} | {:9d} | {:8d} ".format(
                num_train_pids, num_train_imgs, num_train_cams, num_train_frames,
            )
        )
        print(
            "  query    | {:5d} | {:8d} | {:9d} | {:8d} ".format(
                num_query_pids, num_query_imgs, num_query_cams, num_query_frames,
            )
        )
        print(
            "  gallery  | {:5d} | {:8d} | {:9d} | {:8d} ".format(
                num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_gallery_frames,
            )
        )
        print("  -------------------------------------------------- ")
        
    def train_dataloader(
        self, cfg, trainer, sampler_name: str = "frame_based", **kwargs
    ):
        if trainer.distributed_backend == "ddp_spawn":
            rank = trainer.root_gpu
        else:
            rank = trainer.local_rank
        world_size = trainer.num_nodes * trainer.num_processes
        sampler = get_sampler(
            "frame_based",
            data_source=self.train_list,
            batch_size=self.cfg.SOLVER.IMS_PER_BATCH,
            num_instances=self.num_instances,
            world_size=world_size,
            rank=rank,
            frame_ids=self.frame_ids
        )

        return DataLoader(
            self.train,
            self.cfg.SOLVER.IMS_PER_BATCH,
            num_workers=self.num_workers,
            shuffle=False,
            sampler=sampler,
            collate_fn=collate_fn_alternative,
            **kwargs,
        )

    def setup(self):
        self._check_before_run()
        transforms_base = ReidTransforms(self.cfg)
        
        train, train_dict, frame_ids = self._process_dir(self.train_dir, relabel=True)
        self.train_dict = train_dict
        self.train_list = train
        self.frame_ids = frame_ids
        self.train = BaseDatasetLabelledPerPid(train_dict, transforms_base.build_transforms(is_train=True), self.num_instances, self.cfg.DATALOADER.USE_RESAMPLING)

        query, query_dict, _ = self._process_dir(self.query_dir, relabel=False)
        gallery, gallery_dict, _  = self._process_dir(self.gallery_dir, relabel=False)
        self.query_list = query
        self.gallery_list = gallery
        self.val = BaseDatasetLabelled(query+gallery, transforms_base.build_transforms(is_train=False))

        self._print_dataset_statistics(train, query, gallery)
        # For reid_metic to evaluate properly
        num_query_pids, num_query_imgs, num_query_cams, _ = self._get_imagedata_info(query)
        num_train_pids, num_train_imgs, num_train_cams, _ = self._get_imagedata_info(train)
        self.num_query = len(query)
        self.num_classes = num_train_pids

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)s(\d)_f([-\d]+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _, _, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset_dict = defaultdict(list)
        dataset = []
        frame_ids = []

        for idx, img_path in enumerate(img_paths):
            pid, camid, _, frame_id = map(int, pattern.search(img_path).groups())
            camid += 1
            if pid == -1: continue  # junk images are just ignored
            # assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid, idx))
            dataset_dict[pid].append((img_path, pid, camid, idx))
            frame_ids.append(frame_id)

        return dataset, dataset_dict, frame_ids

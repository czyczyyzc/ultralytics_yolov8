# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

from nanotrack.core.config import cfg
from nanotrack.datasets.augmentation import Augmentation
from nanotrack.datasets.point_target import PointTarget
from nanotrack.utils.bbox import Center, center2corner

logger = logging.getLogger("nanotrack")
cv2.ocl.setUseOpenCL(False)


class SubDataset(object):
    def __init__(self, name, root, anno, frame_range, num_use, start_idx):
        self.name = name
        self.root = Path(root).expanduser().resolve()
        self.anno = Path(anno).expanduser().resolve()
        self.frame_range = frame_range
        self.num_use = num_use
        self.start_idx = start_idx
        with open(self.anno, "r", encoding="utf-8") as handle:
            meta_data = self._filter_zero(json.load(handle))

        for video in list(meta_data.keys()):
            for track in list(meta_data[video].keys()):
                frames = list(map(int, filter(str.isdigit, meta_data[video][track].keys())))
                frames.sort()
                meta_data[video][track]["frames"] = frames
                if not frames:
                    del meta_data[video][track]
            if not meta_data[video]:
                del meta_data[video]

        self.labels = meta_data
        self.num = len(self.labels)
        self.num_use = self.num if self.num_use == -1 else self.num_use
        self.videos = list(meta_data.keys())
        self.path_format = "{}.{}.x.jpg"
        self.pick = self.shuffle()

    @staticmethod
    def _filter_zero(meta_data):
        filtered = {}
        for video, tracks in meta_data.items():
            new_tracks = {}
            for track_id, frames in tracks.items():
                new_frames = {}
                for frame_key, bbox in frames.items():
                    if len(bbox) == 4:
                        _, _, w, h = bbox
                        if w <= 0 or h <= 0:
                            continue
                    new_frames[frame_key] = bbox
                if new_frames:
                    new_tracks[track_id] = new_frames
            if new_tracks:
                filtered[video] = new_tracks
        return filtered

    def shuffle(self):
        picks = []
        candidates = list(range(self.start_idx, self.start_idx + self.num))
        while len(picks) < self.num_use:
            np.random.shuffle(candidates)
            picks.extend(candidates)
        return picks[:self.num_use]

    def get_image_anno(self, video, track, frame):
        frame = f"{frame:06d}"
        return self.root / video / self.path_format.format(frame, track), self.labels[video][track][frame]

    def get_positive_pair(self, index):
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        frames = video[track]["frames"]
        template_idx = np.random.randint(0, len(frames))
        left = max(template_idx - self.frame_range, 0)
        right = min(template_idx + self.frame_range, len(frames) - 1) + 1
        search_range = frames[left:right]
        template_frame = frames[template_idx]
        search_frame = np.random.choice(search_range)
        return self.get_image_anno(video_name, track, template_frame), self.get_image_anno(video_name, track, search_frame)

    def get_random_target(self, index=-1):
        if index == -1:
            index = np.random.randint(0, self.num)
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        frames = video[track]["frames"]
        frame = np.random.choice(frames)
        return self.get_image_anno(video_name, track, frame)


class BANDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.point_target = PointTarget()
        self.all_dataset = []
        start = 0
        total = 0
        for name in cfg.DATASET.NAMES:
            sub_cfg = getattr(cfg.DATASET, name)
            sub_dataset = SubDataset(name, sub_cfg.ROOT, sub_cfg.ANNO, sub_cfg.FRAME_RANGE, sub_cfg.NUM_USE, start)
            start += sub_dataset.num
            total += sub_dataset.num_use
            self.all_dataset.append(sub_dataset)
        self.template_aug = Augmentation(cfg.DATASET.TEMPLATE.SHIFT, cfg.DATASET.TEMPLATE.SCALE, cfg.DATASET.TEMPLATE.BLUR, cfg.DATASET.TEMPLATE.FLIP, cfg.DATASET.TEMPLATE.COLOR)
        self.search_aug = Augmentation(cfg.DATASET.SEARCH.SHIFT, cfg.DATASET.SEARCH.SCALE, cfg.DATASET.SEARCH.BLUR, cfg.DATASET.SEARCH.FLIP, cfg.DATASET.SEARCH.COLOR)
        self.num = cfg.DATASET.VIDEOS_PER_EPOCH if cfg.DATASET.VIDEOS_PER_EPOCH > 0 else total
        self.pick = self.shuffle()

    def shuffle(self):
        picks = []
        while len(picks) < self.num:
            current = []
            for sub_dataset in self.all_dataset:
                current.extend(sub_dataset.pick)
            np.random.shuffle(current)
            picks.extend(current)
        return picks[:self.num]

    def resample(self):
        self.pick = self.shuffle()

    def _find_dataset(self, index):
        for dataset in self.all_dataset:
            if dataset.start_idx + dataset.num > index:
                return dataset, index - dataset.start_idx
        raise IndexError(index)

    @staticmethod
    def _get_bbox(image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            _, _, w, h = shape
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w *= scale_z
        h *= scale_z
        cx, cy = imw // 2, imh // 2
        return center2corner(Center(cx, cy, w, h))

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        index = self.pick[index]
        dataset, index = self._find_dataset(index)
        gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()
        neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()
        if neg:
            template = dataset.get_random_target(index)
            search = np.random.choice(self.all_dataset).get_random_target()
        else:
            template, search = dataset.get_positive_pair(index)

        template_image = cv2.imread(str(template[0]))
        search_image = cv2.imread(str(search[0]))
        template_box = self._get_bbox(template_image, template[1])
        search_box = self._get_bbox(search_image, search[1])
        template_crop, _ = self.template_aug(template_image, template_box, cfg.TRAIN.EXEMPLAR_SIZE, gray=gray)
        search_crop, bbox = self.search_aug(search_image, search_box, cfg.TRAIN.SEARCH_SIZE, gray=gray)
        cls, delta = self.point_target(bbox, cfg.TRAIN.OUTPUT_SIZE, neg)
        return {
            "template": template_crop.transpose((2, 0, 1)).astype(np.float32),
            "search": search_crop.transpose((2, 0, 1)).astype(np.float32),
            "label_cls": cls,
            "label_loc": delta,
            "bbox": np.array(bbox, dtype=np.float32),
        }

import os
import torch
import pandas as pd

import slowfast.utils.logging as logging
from .build import DATASET_REGISTRY

from slowfast.datasets.sth_bbox import Sth_bbox
from .build import build_dataset
from tools.data_record.data_record import DataRecorder

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Sth_fusion(torch.utils.data.Dataset):

    def __init__(self, cfg, split):
        assert split in ['train', 'val', 'test']
        assert cfg.DATA.PATH_TO_BBOX_SAMPLE_DIR
        assert cfg.DATA.PATH_TO_BBOX_ANNO_DIR

        self.cfg = cfg
        self.split = split

        if cfg.FUSION.RGB_BACKBONE in ["uniformerv2_b16", "uniformerv2_l14", "videomaev2_giant"]:
            # SSV2 Dataset
            extra_data_dir_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')), "data_list/extra_data/")
            skip_txtpath = [os.path.join(extra_data_dir_path, "5.txt"),
                            os.path.join(extra_data_dir_path, "skipt.txt" if split == "train" else "skipv.txt")]
            sample_size_path = os.path.join(extra_data_dir_path, "sth_frame_size.json")

            sample_file_path = os.path.join(cfg.DATA.PATH_TO_BBOX_SAMPLE_DIR,
                                            "something-something-v2-train.json" if split == "train"
                                            else "something-something-v2-validation.json")
            label_file_path = os.path.join(cfg.DATA.PATH_TO_BBOX_SAMPLE_DIR,
                                           "something-something-v2-labels.json")

            self.bbox_dataset = Sth_bbox(
                dataset_sample_file_path=sample_file_path,
                anno_root_path=cfg.DATA.PATH_TO_BBOX_ANNO_DIR,
                label_file_path=label_file_path,
                num_segments=cfg.DATA.NUM_BBOX_FRAMES,
                num_boxs=cfg.DATA.NUM_BBOXES,
                skip_txtpath=skip_txtpath,
                sample_size_path=sample_size_path,
                train_or_val='train' if split == "train" else "val",
                sample_indices_mode='mode 1',
                resize_shape=(256, 340),
                max_words=cfg.FUSION.MAX_WORDS
            )
            cfg.DATA.SKIP_IDS = self.bbox_dataset.get_skip_ids()
        else:
            raise NotImplementedError

        self.rgb_dataset = DataRecorder(cfg.DATARECORDER.OUTPUT_FOLDER_PATH, split)
        self.rgb_dataset.init_dataset(cfg=cfg, dataset_or_feature='feature')

        # assert len(self.rgb_dataset) == len(self.bbox_dataset), f"rgb_dataset-{len(self.rgb_dataset)} != bbox_dataset-{len(self.bbox_dataset)}"

    def __len__(self):
        return len(self.rgb_dataset)

    def __getitem__(self, index):
        # get rgb feature
        rgb_feature_data, video_index, rgb_sample_index, rgb_label = self.rgb_dataset[index]

        # get bbox data
        box_tensors, box_categories, bbox_label, caption, text_id, frame_mask, video_id = self.bbox_dataset.getitem_based_video_index(
            video_index=video_index.item())

        assert rgb_label == bbox_label, f'BBox label {bbox_label}-{type(bbox_label)} does not match RGB label {rgb_label}-{type(rgb_label)}'
        assert video_index == video_id, f"BBox video index {video_id}-{type(video_id)} does not match RGB video index {video_index.item()}-{type(video_index.item())}"

        meta = {"box_tensors": box_tensors, "box_categories": box_categories, "label": bbox_label, "caption": caption, "text_id": text_id, "video_index": video_index}

        return rgb_feature_data, rgb_label, index, meta

    def get_label_id(self, label_str):
        return self.bbox_dataset.get_label(label_str)

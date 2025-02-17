import orjson
import torch
import numpy as np
from typing import Union
import re

from slowfast.datasets import transform

import slowfast.utils.logging as logging
from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)


def save_json(fp, data, is_indent=False):
    with open(fp, 'wb') as file:
        if is_indent:
            file.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
        else:
            file.write(orjson.dumps(data))


def read_json(fp):
    with open(fp, 'rb') as file:
        return orjson.loads(file.read())


def pre_caption(caption, max_words=50, extra_string_list=None):
    """
    预处理字符串 1、去除特殊符号
    """
    caption = re.sub(
        r"([.!\"()*#:;~\-])",
        ' ',
        caption.lower(),
    )

    # 去除方括号
    caption = re.sub(
        r"([\[\]])",
        '',
        caption,
    )

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        # caption = ' '.join(caption_words[:max_words])
        raise ValueError(f'words len error {len(caption_words)} > {max_words}')
    return caption


class Sth_Label(object):
    def __init__(self, label_file_path):
        self.__path = label_file_path
        self.label_dict = self.read_label()

    def read_label(self):
        # 对label的字符进行处理
        label_dict = read_json(self.__path)
        new_dict = {}
        for key in label_dict.keys():
            value = label_dict[key]
            key = pre_caption(key)
            new_dict[key] = value
        label_dict = new_dict
        return label_dict

    def get_label(self, label_str: str) -> int:
        if label_str in self.label_dict.keys():
            return int(self.label_dict[label_str])
        else:
            raise KeyError


class Sth_Sample(object):
    def __init__(self, sample: dict):
        self.__sample = sample
        self.size_w = None
        self.size_h = None

    def get_id(self) -> str:
        return self.__sample['id']

    def get_label(self) -> str:
        """
        完整的 label text: putting pen on a surface
        注意使用时使用 pre_caption函数 做字符串处理
        """
        # label = pre_caption(self.__sample['label'])
        label = self.__sample['label']
        # label = self.__sample['template'].replace('[', '').replace(']', '')
        return label

    def get_template(self) -> str:
        """
        带something的 label text: Putting [something] on a surface
        注意使用时使用 pre_caption函数 做字符串处理
        """
        # template = pre_caption(self.__sample['template'])
        template = self.__sample['template']
        return template

    def get_caption(self) -> str:
        """
        文本 caption 的使用
        """
        # return self.__sample['label']
        return self.__sample['template']

    def set_size(self, wh: Union[list, tuple]):
        self.size_w = wh[0]
        self.size_h = wh[1]

    def get_size_wh(self) -> tuple:
        assert self.size_w is not None and self.size_h is not None, "Used before assignment"
        return self.size_w, self.size_h


@DATASET_REGISTRY.register()
class Sth_bbox(torch.utils.data.Dataset):
    def __init__(self, dataset_sample_file_path, anno_root_path, label_file_path, num_segments, num_boxs, skip_txtpath,
                 sample_size_path='slowfast/datasets/extra data/sth_frame_size.json', train_or_val: str = 'train', sample_indices_mode='mode 1',
                 resize_shape=(256, 340), max_words=30, prompt=''):
        self.__dataset_samples = read_json(dataset_sample_file_path)  # train/val data
        self.__anno_root_path = anno_root_path
        self.__label = Sth_Label(label_file_path)
        self.__num_segments = num_segments
        self.__num_boxs = num_boxs
        self.__skip_ids = self.read_txt_2_int(skip_txtpath)
        self.data_fliter()
        self.__sample_size_dict = read_json(sample_size_path)

        self.__resize_shape = resize_shape
        self.__crop_size_wh = (224, 224)

        assert train_or_val in ['train', 'val'], 'parameter error'
        self.__train_or_val = train_or_val

        assert sample_indices_mode in ['mode 1', 'mode 2', 'mode 3'], 'parameter error'
        self.__sample_indices_mode = sample_indices_mode

        if self.__train_or_val == 'train':
            self.__random_crop = transform.GroupMultiScaleCrop(output_size=self.__crop_size_wh, scales=[1, .875, .75])
        else:
            self.__random_crop = transform.GroupMultiScaleCrop(output_size=self.__crop_size_wh, scales=[1], max_distort=0, center_crop_only=True)

        self.__max_words = max_words
        self.__prompt = prompt

        if self.__train_or_val == 'train':
            self.text_ids = {}
            n = 0

            for index in range(len(self.__dataset_samples)):
                sample = self.__dataset_samples[index]
                sample = Sth_Sample(sample)
                text = pre_caption(sample.get_caption(), max_words=self.__max_words)
                if text not in self.text_ids.keys():
                    label_id = self.__label.get_label(pre_caption(sample.get_template(), max_words=self.__max_words))
                    self.text_ids[text] = int(label_id)
                    n = n + 1

        else:
            self.all_text2id = {}
            self.all_vision = []
            self.vision2text = {}
            self.text2vision = {}
            self.text2label = {}

            text_id = 0
            for index in range(len(self.__dataset_samples)):
                sample = self.__dataset_samples[index]
                sample = Sth_Sample(sample)

                vision_id = index
                vision = sample.get_id()
                text = pre_caption(sample.get_caption(), max_words=self.__max_words)

                self.all_vision.append(vision)
                if text not in self.all_text2id.keys():
                    label_id = self.__label.get_label(pre_caption(sample.get_template(), max_words=self.__max_words))
                    self.all_text2id[text] = int(label_id)
                    # self.all_text2id[text] = text_id
                    self.text2label[self.all_text2id[text]] = int(label_id)
                    self.text2vision[self.all_text2id[text]] = []
                    text_id += 1

                self.text2vision[self.all_text2id[text]].append(vision_id)
                if vision_id not in self.vision2text.keys():
                    self.vision2text[vision_id] = []
                self.vision2text[vision_id].append(self.all_text2id[text])

            # for text in self.all_text2id.keys():
            #     self.all_text2label_id.append(self.get_label(text))

        # video index: sample
        self.video_index2_sample_dict = {}
        self.process_data_for_video_index()

    def process_data_for_video_index(self):
        for index in range(len(self.__dataset_samples)):
            sample = self.__dataset_samples[index]
            sample = Sth_Sample(sample)
            sample.set_size(self.__sample_size_dict[sample.get_id()])

            video_index = sample.get_id()
            if video_index not in self.video_index2_sample_dict.keys():
                self.video_index2_sample_dict[video_index] = sample
            else:
                raise KeyError(f"key: {video_index} has already been in the video_index2_sample_dict")

    def getitem_based_video_index(self, video_index: int):
        sample = self.video_index2_sample_dict[str(video_index)]
        anno_data = self.read_json(self.__anno_root_path + '/' + sample.get_id() + '.json')
        available_frames_list = self.get_available_frames(anno_data)
        frame_indices, frame_mask = self.sample_indices(sample, anno_data, available_frames_list)
        box_tensors, box_categories = self.sample_data_onefile(frame_indices, sample, anno_data)
        label = self.__label.get_label(pre_caption(sample.get_template(), self.__max_words))
        caption = self.__prompt + pre_caption(sample.get_caption(), max_words=self.__max_words)
        video_id = int(sample.get_id())

        # text id
        if self.__train_or_val == 'train':
            text_id = self.text_ids[pre_caption(sample.get_caption(), max_words=self.__max_words)]
            assert label == text_id, f"Label-{label} != Text ID-{text_id} video_id:{video_id} text:{pre_caption(sample.get_caption())}"
        else:
            text_id = 0

        return box_tensors, box_categories, label, caption, text_id, frame_mask, video_id

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.__dataset_samples)

    def get_label(self, label_str):
        """
        通过label的str 得到 label 对应的序号
        """
        return self.__label.get_label(label_str)

    def get_label_obj(self):
        return self.__label

    def get_available_frames(self, anno_data):
        """
        获取样本中所有符合筛选条件的帧的序号
        筛选条件：包含一个object 或者 hand
        """
        available_frames_list = []

        # for frame_index in range(len(anno_data)):
        #     if len(anno_data[frame_index]['labels']) > 0:  # 该帧有数据
        #         available_frames_list.append(frame_index)

        for frame_index in range(len(anno_data)):
            available_frames_list.append(frame_index)

        return available_frames_list

    def sample_data_onefile(self, frame_indices, sample: Sth_Sample, anno_data):
        """
        根据样本索引，该样本帧索引，样本
        得到 obj 和 hand 的 bbox，及对应类别
        """
        box_tensors = torch.zeros((len(frame_indices), self.__num_boxs, 4), dtype=torch.float32)
        box_categories = torch.zeros((len(frame_indices), self.__num_boxs), dtype=torch.int32)
        label_text2id = {'None': 0, 'hand': 6, '0000': 1, '0001': 2, '0002': 3, '0003': 4, '0004': 5}  # '0000', 'hand', '0001', '0002', '0003', '0004'

        sample_size = sample.get_size_wh()
        for index, frame_index in enumerate(frame_indices):
            if frame_index != -1:
                frame_data = anno_data[frame_index]

                objs_hand_bboxes_list = []
                objs_hand_labels_list = []
                objs_hand_categories_list = []

                # 顺序取
                for bbox in frame_data['labels']:
                    if len(objs_hand_bboxes_list) < self.__num_boxs:
                        x0, y0, x1, y1 = bbox['box2d']['x1'], bbox['box2d']['y1'], bbox['box2d']['x2'], bbox['box2d']['y2']
                        if x0 > x1:
                            raise ValueError(f"x0-{x0} > x1-{x1}")
                        objs_hand_bboxes_list.append(self.object_hand_coord_convert((x0, y0, x1, y1), w_max=float(sample_size[0]), h_max=float(sample_size[1])))
                        if bbox['standard_category'] == 'hand':
                            objs_hand_categories_list.append('hand')
                        else:
                            objs_hand_categories_list.append('object')
                        objs_hand_labels_list.append(label_text2id[bbox['standard_category']])
                    else:
                        break
                for j, coord in enumerate(objs_hand_bboxes_list):
                    box_tensors[index, j] = torch.tensor(coord, dtype=torch.float32)
                    box_categories[index, j] = objs_hand_labels_list[j]

            else:
                pass

        return box_tensors, box_categories

    def sample_indices(self, sample: Sth_Sample, anno_data, available_frames_list):
        if self.__sample_indices_mode == 'mode 1':
            if len(available_frames_list) > 0:
                num_frames = len(available_frames_list)
                average_duration = num_frames // self.__num_segments
                frame_mask = torch.zeros(self.__num_segments)

                if average_duration > 0:
                    frame_indexes = (np.multiply(list(range(self.__num_segments)), average_duration) +
                                     np.random.randint(average_duration, size=self.__num_segments))
                else:
                    frame_indexes = np.array([-1] * self.__num_segments)
                    frame_indexes[:num_frames] = np.array(range(num_frames))

                    assert self.__num_segments - num_frames >= 0, 'unanticipated error'
                    frame_mask[num_frames:] = 1
                frame_indexes = [available_frames_list[i] for i in frame_indexes]

            else:
                frame_indexes = np.array([-1] * self.__num_segments)
                frame_mask = torch.ones(self.__num_segments)
        else:
            raise ValueError('do not support yet')

        return frame_indexes, frame_mask

    def object_hand_coord_convert(self, obj_coord: Union[list, tuple], w_max, h_max) -> np.ndarray:
        x0, y0, x1, y1 = obj_coord
        cx = (x0 + x1) / 2.
        cy = (y0 + y1) / 2.
        w = np.abs(x1 - x0)
        h = np.abs(y1 - y0)

        coord_convert = np.array([cx / w_max, cy / h_max, w / w_max, h / h_max], dtype=np.float32)
        return coord_convert

    def data_fliter(self):
        new_samples = []
        for sample in self.__dataset_samples:
            if int(sample['id']) not in self.__skip_ids:
                new_samples.append(sample)
        self.__dataset_samples = new_samples

    def read_json(self, fp):
        with open(fp, 'rb') as file:
            return orjson.loads(file.read())

    def read_txt_2_int(self, skip_txtpath_list: Union[list, tuple]) -> list:
        all_txt_data = []
        for skip_txtpath in skip_txtpath_list:
            with open(skip_txtpath, 'r', encoding='UTF-8') as f:
                txt_data = f.readlines()
            for i in range(len(txt_data)):
                txt_data[i] = int(txt_data[i])
            all_txt_data = all_txt_data + txt_data
        return all_txt_data

    def get_skip_ids(self):
        return self.__skip_ids

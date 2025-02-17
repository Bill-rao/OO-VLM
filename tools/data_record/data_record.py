import numpy as np
import torch
from torch.utils.data import Dataset
import os
from glob import glob
import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)


class DataRecorder(Dataset):
    def __init__(self, path_to_record_folder, split: str):
        assert split in ['train', 'val', 'test'], "Invalid split"
        logger.info(f"Create DataRecorder-{split}")
        if os.path.exists(path_to_record_folder):
            logger.warning(f"Folder {path_to_record_folder} already exists. If Recorder is used as dataset, ignore this warning."
                           f"Otherwise make sure to duplicate the recorder creation to avoid data overwriting. ")

        os.makedirs(path_to_record_folder, exist_ok=True)

        self.path_to_record_folder = path_to_record_folder

        self.split = split

        # self.dataset_data_type = None
        self.current_dataset_epoch = 0
        self.current_dataset_iter = 0
        self.dataset_data = {}

        # self.feature_data_type = None
        self.current_feature_epoch = 0
        self.current_feature_iter = 0
        self.feature_data = {}

        self.total_epoch = 0
        self.skip_ids = None

        self.all_files = {}
        self.num_samples = 0
        self.num_videos = 0

        self.cfg = None

    def init_dataset(self, cfg, dataset_or_feature='feature'):
        self.cfg = cfg

        assert dataset_or_feature in ['dataset', 'feature']
        logger.info(f"Choose output {dataset_or_feature}")

        if os.path.exists(os.path.join(self.path_to_record_folder, f'{dataset_or_feature}_{self.split}')):
            if dataset_or_feature == 'dataset':
                raise NotImplementedError()
            else:  # 'feature'
                for file_path in glob(os.path.join(self.path_to_record_folder, f'{dataset_or_feature}_{self.split}', '*')):
                    file_name = os.path.basename(file_path).split('.')[0]
                    self.all_files[file_name] = file_path

                self.total_epoch = len(self.all_files)
                self.skip_ids = self.cfg.DATA.SKIP_IDS

            # update_epoch_data
            self.update_epoch_data(0, dataset_or_feature='feature')
        else:
            raise FileExistsError(f'No such files in {os.path.join(self.path_to_record_folder, f"{dataset_or_feature}_{self.split}")}')
        logger.info("Init DataRecorder Dataset Successfully")

    def skip_id_process(self):
        assert self.skip_ids is not None, "SKIP_IDS is empty"
        if len(self.skip_ids) > 0:
            ids = []
            for i in range(self.feature_data['feature_data'].shape[0]):
                if int(self.feature_data['video_index'][i].item()) not in self.skip_ids:
                    ids.append(i)
            assert len(ids) == len(set(ids)), 'ids has duplicate values'

            new_feature_data = {}
            for key in self.feature_data.keys():
                new_feature_data[key] = self.feature_data[key][ids]
            self.feature_data = new_feature_data

    def update_epoch_data(self, epoch, dataset_or_feature='feature'):
        assert dataset_or_feature in ['dataset', 'feature']
        if epoch > len(list(self.all_files.keys())) - 1:
            epoch = epoch % len(list(self.all_files.keys()))

        file_name = f"epoch_{epoch}"
        if dataset_or_feature == 'feature':
            if file_name in self.all_files.keys():
                logger.info(f"loading data... epoch({epoch})")
                self.feature_data = torch.load(self.all_files[file_name], map_location='cpu')
                self.skip_id_process()

                self.num_samples = self.feature_data['label'].shape[0]
                self.num_videos = torch.unique(self.feature_data['video_index']).shape[0]
            else:
                raise ValueError(f'No such file: {file_name}.pth')
        else:
            raise NotImplementedError()

    def __len__(self):
        if self.num_samples != 0:
            return self.num_samples
        else:
            raise RuntimeError("Please run update_epoch_data() to update the number of samples")

    def __getitem__(self, index):
        data = self.feature_data['feature_data'][index]
        video_index = self.feature_data['video_index'][index]
        sample_index = self.feature_data['sample_index'][index]
        label = self.feature_data['label'][index]
        return data, video_index, sample_index, label

    @staticmethod
    def transfer_tensor_device(transferred_tensor, dest_device):
        assert dest_device in ['cpu', 'cuda'], 'Invalid'
        if transferred_tensor.device.type != dest_device:
            return transferred_tensor.to(dest_device)
        else:
            return transferred_tensor

    def update_dataset_data(self, rgb_data, sample_index, label, epoch):
        assert isinstance(rgb_data, torch.Tensor) and isinstance(sample_index, torch.Tensor) and isinstance(label, torch.Tensor)
        assert isinstance(epoch, int)
        self.transfer_tensor_device(rgb_data, 'cpu')
        self.transfer_tensor_device(sample_index, 'cpu')
        self.transfer_tensor_device(label, 'cpu')

        # if self.current_dataset_epoch + 1 == epoch:
        #     # save
        #     logger.info(f"Epoch is changed ({self.current_feature_epoch} -> {epoch}), the dataset data will saved")
        #     self.save_dataset_data()
        #     self.current_dataset_epoch = epoch

        if self.current_dataset_epoch == epoch:  # 还在同一个epoch内或者进入新的epoch，则缓存dataset数据
            logger.info(f"caching dataset data......")
            if 'rgb_data' not in self.dataset_data.keys():
                self.dataset_data = {'rgb_data': rgb_data, 'sample_index': sample_index, 'label': label}
            else:
                self.dataset_data['rgb_data'] = torch.cat([self.dataset_data['rgb_data'], rgb_data], dim=0)
                self.dataset_data['sample_index'] = torch.cat([self.dataset_data['sample_index'], sample_index], dim=0)
                self.dataset_data['label'] = torch.cat([self.dataset_data['label'], label], dim=0)
        else:
            raise RuntimeError()

    def save_dataset_data(self):
        logger.info("Saving dataset data")
        path = os.path.join(self.path_to_record_folder, f'dataset_{self.split}')
        os.makedirs(path, exist_ok=True)

        torch.save(self.dataset_data, os.path.join(path, f'epoch_{self.current_dataset_epoch}.pth'))
        self.dataset_data.clear()

    def update_feature_data(self, feature_data: torch.Tensor, sample_index: torch.Tensor, label: torch.Tensor,
                            video_index: torch.Tensor, epoch: int):

        assert isinstance(feature_data, torch.Tensor) and isinstance(sample_index, torch.Tensor)
        assert isinstance(label, torch.Tensor) and isinstance(video_index, torch.Tensor)
        assert isinstance(epoch, int)

        feature_data = self.transfer_tensor_device(feature_data, 'cpu')
        sample_index = self.transfer_tensor_device(sample_index, 'cpu')
        label = self.transfer_tensor_device(label, 'cpu')
        video_index = self.transfer_tensor_device(video_index, 'cpu')

        if self.current_feature_epoch + 1 == epoch:
            # save
            logger.info(f"Epoch is changed ({self.current_feature_epoch} -> {epoch}), the feature data will saved")
            self.save_feature_data()
            self.current_feature_epoch = epoch

        if self.current_feature_epoch == epoch:
            logger.info(f"caching feature data---epoch:{epoch} feature_data: {feature_data.shape} sample_index:{sample_index.shape} label:{label.shape} video_index:{video_index.shape}")
            if 'feature_data' not in self.feature_data.keys():
                self.feature_data = {'feature_data': feature_data, 'sample_index': sample_index, 'label': label, 'video_index': video_index}
            else:
                self.feature_data['feature_data'] = torch.cat([self.feature_data['feature_data'], feature_data], dim=0)
                self.feature_data['sample_index'] = torch.cat([self.feature_data['sample_index'], sample_index], dim=0)
                self.feature_data['label'] = torch.cat([self.feature_data['label'], label], dim=0)
                self.feature_data['video_index'] = torch.cat([self.feature_data['video_index'], video_index], dim=0)
        else:
            raise RuntimeError(f"current_feature_epoch({self.current_feature_epoch}) does not match the input epoch({epoch})")

    def clear_duplicates_of_feature_data(self):
        logger.info(f"num of samples: {self.feature_data['sample_index'].shape[0]}")
        duplicates = self.find_duplicates_with_indices(self.feature_data['sample_index'].tolist())
        logger.info(f"duplicates(sample_index)[sample_index: [position_index]]: {duplicates}")
        all_remove_index = []
        for k, v in duplicates.items():
            all_remove_index.append(*v[1:])
        logger.info(f"all removed index: {all_remove_index}")
        if len(all_remove_index) > 0:
            all_index = list(range(self.feature_data['sample_index'].shape[0]))
            retained_element_index = self.remove_items_by_set_operation(all_index, all_remove_index)

            new_feature_data = {}
            for key, value in self.feature_data.items():
                new_feature_data[key] = value[retained_element_index]

            assert self.find_duplicates_with_indices(new_feature_data['sample_index'].tolist()) == {}, "still have duplicates"
            logger.info(f"num of samples {self.feature_data['feature_data'].shape[0]} -> {new_feature_data['feature_data'].shape[0]}")
            self.feature_data = new_feature_data
            logger.info(f"feature data was cleared successfully")
        else:
            logger.info(f"feature data does not have duplicates")

    def save_feature_data(self):
        logger.info("Saving feature data")
        if len(self.feature_data) > 0:
            path = os.path.join(self.path_to_record_folder, f'feature_{self.split}')
            self.clear_duplicates_of_feature_data()

            os.makedirs(path, exist_ok=True)
            torch.save(self.feature_data, os.path.join(path, f'epoch_{self.current_feature_epoch}.pth'))
            self.feature_data.clear()
        else:
            logger.info(f"feature_data is empty, skip save progress")

    def find_duplicates_with_indices(self, lst):
        element_indices = {}
        for index, element in enumerate(lst):
            if element in element_indices:
                element_indices[element].append(index)
            else:
                element_indices[element] = [index]

        duplicates = {
            element: indices
            for element, indices in element_indices.items()
            if len(indices) > 1
        }
        return duplicates

    def remove_items_by_set_operation(self, A, B):
        if not isinstance(A, list):
            A = list(A)
        if not isinstance(B, list):
            B = list(B)

        set_B = set(B)
        return [item for item in A if item not in set_B]

    def reset_epoch_count(self, dataset_or_feature: 'str'):
        assert dataset_or_feature in ['dataset', 'feature']
        if dataset_or_feature == 'dataset':
            self.current_dataset_epoch = 0
            self.current_dataset_iter = 0
        else:
            self.current_feature_epoch = 0
            self.current_feature_iter = 0

    def reset_iter_count(self, dataset_or_feature: 'str'):
        assert dataset_or_feature in ['dataset', 'feature']
        if dataset_or_feature == 'dataset':
            self.current_dataset_iter = 0
        else:
            self.current_feature_iter = 0

    def save_meta_data(self, dict_data: dict):
        logger.info('Saving meta data')
        path = os.path.join(self.path_to_record_folder, f'meta_{self.split}')
        os.makedirs(path, exist_ok=True)

        torch.save(dict_data, os.path.join(path, f'meta.pth'))


def construct_loader(datarecorder: DataRecorder, batch_size: int, sampler, shuffle: bool, num_workers: int, pin_memory: bool, drop_last: bool):
    loader = torch.utils.data.DataLoader(datarecorder,
                                         batch_size=batch_size,
                                         shuffle=(False if sampler else shuffle),
                                         sampler=sampler,
                                         num_workers=num_workers,
                                         pin_memory=pin_memory,
                                         drop_last=drop_last)
    return loader

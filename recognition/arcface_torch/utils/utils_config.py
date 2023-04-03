import importlib
import os.path as osp
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from easydict import EasyDict
from datasets.AIHubDataset import AIHubDataset


def get_config(config_file):
    assert config_file.startswith(
        "configs/"
    ), "config file setting must start with configs/"
    temp_config_name = osp.basename(config_file)
    temp_module_name = osp.splitext(temp_config_name)[0]
    config = importlib.import_module("configs.base")
    cfg = config.config
    config = importlib.import_module("configs.%s" % temp_module_name)
    job_cfg = config.config
    cfg.update(job_cfg)
    if cfg.output is None:
        cfg.output = osp.join("work_dirs", temp_module_name)
    return cfg


def return_pairs_path(config_data_split: dict) -> str:
    """
    주어진 split과 task에 해당하는 파일 경로를 반환합니다.

    :param config_data_split: 데이터셋의 분할 유형과 수행할 작업에 대한 설명을 포함한 data config dict
    :return: 파일 경로를 포함한 문자열
    :rtype: str
    :raises FileNotFoundError: 파일이 존재하지 않을 때 발생하는 예외
    """

    def _cap(s: str) -> str:
        return s[0].upper() + s[1:]

    file_path = f"/home/jupyter/insightface/data/pairs/{config_data_split.split}/pairs_{_cap(config_data_split.task)}.txt"

    if not osp.exists(file_path):
        raise FileNotFoundError(f"파일이 존재하지 않습니다: {file_path}")

    return file_path


def return_aihub_dataloader(
    config_data_split: EasyDict, transform: Compose
) -> DataLoader:
    """
    주어진 설정과 데이터 변환을 사용하여 DataLoader를 생성합니다.

    :param config_data_split: 데이터 분할에 대한 설정이 담긴 EasyDict 객체
    :param transform: 이미지 데이터에 적용할 데이터 변환 객체 (예: torchvision.transforms.Compose)
    :return: 생성된 DataLoader 객체
    :rtype: torch.utils.data.DataLoader
    """
    aihub_dataloader = torch.utils.data.DataLoader(
        dataset=AIHubDataset(
            dir=config_data_split.path.aihub_dataroot,
            pairs_path=config_data_split.path.pairs_path(config_data_split),
            transform=transform,
        ),
        batch_size=config_data_split.batch_size,
        num_workers=config_data_split.num_workers,
        shuffle=False,
    )
    return aihub_dataloader

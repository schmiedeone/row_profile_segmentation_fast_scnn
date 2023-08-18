from .cityscapes import CitySegmentation
from .croplanes import CropLaneSegmentation

datasets = {
    'citys': CitySegmentation,
    'crops': CropLaneSegmentation
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)

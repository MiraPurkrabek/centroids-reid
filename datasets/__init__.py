# encoding: utf-8
"""
Partially based on work by:
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com

Adapted and extended by:
@author: mikwieczorek
"""

from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .df1 import DF1
from .street2shop import Street2Shop
from .custom import CustomDataset
from .frame_triplets_dataset import FrameTripletsDataset
from .GPHMER_texture_bf import GPHMERTextureBF
from .GPHMER_texture_smpl import GPHMERTextureSMPL
from .NUMBERS_dataset import NUMBERSDataset
from .floorball_manual_market import FloorballManualMarket
from .patches_market import PatchesMarket
from .patches_market_separate import PatchesMarketSeparate

__factory = {
    "market1501": Market1501,
    "dukemtmcreid": DukeMTMCreID,
    "df1": DF1,
    "street2shop": Street2Shop,
    "custom_market_dataset": CustomDataset,
    "frame_triplets_dataset": FrameTripletsDataset,
    "GPHMER_texture_BF": GPHMERTextureBF,
    "GPHMER_texture_SMPL": GPHMERTextureSMPL,
    "NUMBERS_dataset": NUMBERSDataset,
    "floorball_manual_market": FloorballManualMarket,
    "patches_market": PatchesMarket,
    "patches_market_separate": PatchesMarketSeparate,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)

from __future__ import absolute_import
import warnings

from .dukemtmc import DukeMTMC
from .market1501 import Market1501
from .msmt17v2 import MSMT17_V2
from .msmt17v1 import MSMT17_V1
from .cuhk03 import CUHK03
from .cuhknp import CUHK_NP

__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMC,
    'msmt17v2': MSMT17_V2,
    'msmt17v1': MSMT17_V1,
    'cuhk03': CUHK03,
    'cuhknp': CUHK_NP
}


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. 
    root : str
        The path to the dataset directory.
    split_id : int, optional
        The index of data split. Default: 0
    num_val : int or float, optional
        When int, it means the number of validation identities. When float,
        it means the proportion of validation to all the trainval. Default: 100
    download : bool, optional
        If True, will download the dataset. Default: False
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)


def get_dataset(name, root, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, root, *args, **kwargs)

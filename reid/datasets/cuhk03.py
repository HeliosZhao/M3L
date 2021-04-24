from __future__ import print_function
import os.path as osp
import json
import numpy as np

def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def _pluck(identities, indices, relabel=False):
    ret = []
    for index, pid in enumerate(indices):
        pid_images = identities[pid]
        for camid, cam_images in enumerate(pid_images):
            for fname in cam_images:
                name = osp.splitext(fname)[0]
                x, y, _ = map(int, name.split('_'))
                assert pid == x and camid == y
                if relabel:
                    ret.append((fname, index, camid, 0))
                else:
                    ret.append((fname, pid, camid, 0))
    return ret

def _pluck_gallery(identities, indices, relabel=False):
    ret = []
    for index, pid in enumerate(indices):
        pid_images = identities[pid]
        for camid, cam_images in enumerate(pid_images):
            if len(cam_images[:-1])==0:
                for fname in cam_images:
                    name = osp.splitext(fname)[0]
                    x, y, _ = map(int, name.split('_'))
                    assert pid == x and camid == y
                    if relabel:
                        ret.append((fname, index, camid))
                    else:
                        ret.append((fname, pid, camid))
            else:
                for fname in cam_images[:-1]:
                    name = osp.splitext(fname)[0]
                    x, y, _ = map(int, name.split('_'))
                    assert pid == x and camid == y
                    if relabel:
                        ret.append((fname, index, camid, 0))
                    else:
                        ret.append((fname, pid, camid, 0))
    return ret

def _pluck_query(identities, indices, relabel=False):
    ret = []
    for index, pid in enumerate(indices):
        pid_images = identities[pid]
        for camid, cam_images in enumerate(pid_images):
            for fname in cam_images[-1:]:
                name = osp.splitext(fname)[0]
                x, y, _ = map(int, name.split('_'))
                assert pid == x and camid == y
                if relabel:
                    ret.append((fname, index, camid, 0))
                else:
                    ret.append((fname, pid, camid, 0))
    return ret


class CUHK03(object):
    def __init__(self, root, split_id=0, verbose=True):
        super(CUHK03, self).__init__()
        self.root = osp.join(root,'cuhk03_release')
        self.split_id = split_id
        self.verbose = verbose
        self.meta = None
        self.split = None
        self.train, self.val, self.trainval = [], [], []
        self.query, self.gallery = [], []
        self.num_train_ids, self.num_val_ids, self.num_trainval_ids = 0, 0, 0
        self._check_integrity()
        self.load()
    @property
    def images_dir(self):
        return osp.join(self.root, 'images')

    def load(self):
        splits = read_json(osp.join(self.root, 'splits.json'))
        if self.split_id >= len(splits):
            raise ValueError("split_id exceeds total splits {}"
                             .format(len(splits)))
        self.split = splits[self.split_id]

        # Randomly split train / val
        trainval_pids = np.asarray(self.split['trainval'])
        np.random.shuffle(trainval_pids)
        num = len(trainval_pids)

        train_pids = sorted(trainval_pids)

        self.meta = read_json(osp.join(self.root, 'meta.json'))
        identities = self.meta['identities']
        self.train = _pluck(identities, train_pids, relabel=True)
        self.query = _pluck_query(identities, self.split['query'])
        #self.gallery = _pluck(identities, self.split['gallery'])
        self.gallery = _pluck_gallery(identities, self.split['gallery'])
        self.num_train_pids = len(train_pids)

        if self.verbose:
            print(self.__class__.__name__, "dataset loaded")
            print("  subset   | # ids | # images")
            print("  ---------------------------")
            print("  train    | {:5d} | {:8d}"
                  .format(self.num_train_pids, len(self.train)))
            print("  trainval | {:5d} | {:8d}"
                  .format(self.num_trainval_ids, len(self.trainval)))
            print("  query    | {:5d} | {:8d}"
                  .format(len(self.split['query']), len(self.query)))
            print("  gallery  | {:5d} | {:8d}"
                  .format(len(self.split['gallery']), len(self.gallery)))

    def _check_integrity(self):
        return osp.isdir(osp.join(self.root, 'images')) and \
               osp.isfile(osp.join(self.root, 'meta.json')) and \
               osp.isfile(osp.join(self.root, 'splits.json'))

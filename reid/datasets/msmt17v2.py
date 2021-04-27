from __future__ import print_function, absolute_import
import os.path as osp


class MSMT17_V2(object):

    def __init__(self, root, combine_all=False):

        self.images_dir = osp.join(root,'MSMT17_V2')
        self.combine_all = combine_all
        self.train_path = 'mask_train_v2'
        self.test_path = 'mask_test_v2'
        self.train_list_file = 'list_train.txt'
        self.val_list_file = 'list_val.txt'
        self.gallery_list_file = 'list_gallery.txt'
        self.query_list_file = 'list_query.txt'
        self.gallery_path = self.test_path
        self.query_path = self.test_path
        self.train, self.val, self.query, self.gallery = [], [], [], []
        self.num_train_pids, self.num_val_ids, self.num_query_ids, self.num_gallery_ids = 0, 0, 0, 0
        self.has_time_info = False
        self.load()

    def preprocess(self, list_file, subpath):
        with open(osp.join(self.images_dir, list_file), 'r') as txt:
            lines = txt.readlines()

        data = []
        all_pids = {}

        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid) # no need to relabel
            if pid not in all_pids:
                all_pids[pid] = pid
            camid = int(img_path.split('_')[2]) - 1  # index starts from 0
            data.append((osp.join(subpath,img_path), pid, camid, 3))
        return data, int(len(all_pids))

    def load(self):
        self.train, self.num_train_pids = self.preprocess(self.train_list_file,self.train_path)
        self.val, self.num_val_ids = self.preprocess(self.val_list_file, self.train_path)
        self.gallery, self.num_gallery_ids = self.preprocess(self.gallery_list_file,self.test_path)
        self.query, self.num_query_ids = self.preprocess(self.query_list_file,self.test_path)

        self.train += self.val
        if self.combine_all:
            for item in self.train:
                item[0] = osp.join(self.train_path, item[0])
            for item in self.gallery:
                item[0] = osp.join(self.gallery_path, item[0])
                item[1] += self.num_train_pids
            for item in self.query:
                item[0] = osp.join(self.query_path, item[0])
                item[1] += self.num_train_pids
            self.train += self.gallery
            self.train += self.query
            self.num_train_pids += self.num_gallery_ids
            self.train_path = ''

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  train    | {:5d} | {:8d}"
              .format(self.num_train_pids, len(self.train)))
        print("  query    | {:5d} | {:8d}"
              .format(self.num_query_ids, len(self.query)))
        print("  gallery  | {:5d} | {:8d}"
              .format(self.num_gallery_ids, len(self.gallery)))

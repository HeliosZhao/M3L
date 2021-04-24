from __future__ import print_function, absolute_import, division
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .utils.meters import AverageMeter
from .models import *
from .evaluation_metrics import accuracy
from .models.MetaModules import MixUpBatchNorm1d as MixUp1D

class Trainer(object):
    def __init__(self, args, model, memory, criterion):
        super(Trainer, self).__init__()
        self.model = model
        self.memory = memory
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = criterion
        self.args = args

    def train(self, epoch, data_loaders, optimizer, print_freq=10, train_iters=400):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_meta_train = AverageMeter()
        losses_meta_test = AverageMeter()
        metaLR = optimizer.param_groups[0]['lr'] 

        source_count = len(data_loaders)


        end = time.time()
        for i in range(train_iters):
            metaTestID = np.random.choice(source_count)
            network_bns = [x for x in list(self.model.modules()) if isinstance(x, MixUp1D)]

            for bn in network_bns:
                bn.meta_mean1 = torch.zeros(bn.meta_mean1.size()).float().cuda()
                bn.meta_var1 = torch.zeros(bn.meta_var1.size()).float().cuda()
                bn.meta_mean2 = torch.zeros(bn.meta_mean2.size()).float().cuda()
                bn.meta_var2 = torch.zeros(bn.meta_var2.size()).float().cuda()

            # with torch.autograd.set_detect_anomaly(True):
            if True:
                data_loader_index = [i for i in range(source_count)] ## 0 2
                del data_loader_index[metaTestID]
                batch_data = [data_loaders[i].next() for i in range(source_count)]
                metaTestinputs = batch_data[metaTestID]
                data_time.update(time.time() - end)
                # process inputs
                testInputs, testPids, _, _, _ = self._parse_data(metaTestinputs)
                loss_meta_train = 0.
                save_index = 0
                for t in data_loader_index: # 0 1
                    data_time.update(time.time() - end)
                    traininputs = batch_data[t]
                    save_index += 1
                    inputs, targets, _, _, _ = self._parse_data(traininputs)

                    f_out, tri_features = self.model(inputs, MTE='', save_index=save_index)
                    loss_mtr_tri = self.criterion(tri_features, targets)
                    loss_s = self.memory[t](f_out, targets).mean()

                    loss_meta_train = loss_meta_train + loss_s + loss_mtr_tri

                loss_meta_train = loss_meta_train / (source_count - 1)

                self.model.zero_grad()
                grad_info = torch.autograd.grad(loss_meta_train, self.model.module.params(), create_graph=True)
                self.newMeta = create(self.args.arch, norm=True, BNNeck=self.args.BNNeck)
                # creatmodel = time.time()
                self.newMeta.copyModel(self.model.module)
                # copymodel = time.time()
                self.newMeta.update_params(
                    lr_inner=metaLR, source_params=grad_info, solver='adam'
                )

                del grad_info

                self.newMeta = nn.DataParallel(self.newMeta).to(self.device)

                f_test, mte_tri = self.newMeta(testInputs, MTE=self.args.BNtype)

                loss_meta_test = 0.
                if isinstance(f_test, list):
                    for feature in f_test:
                        loss_meta_test += self.memory[metaTestID](feature, testPids).mean()
                    loss_meta_test /= len(f_test)

                else:
                    loss_meta_test = self.memory[metaTestID](f_test, testPids).mean()

                loss_mte_tri = self.criterion(mte_tri, testPids)
                loss_meta_test = loss_meta_test + loss_mte_tri

                loss_final = loss_meta_train + loss_meta_test
                losses_meta_train.update(loss_meta_train.item())
                losses_meta_test.update(loss_meta_test.item())

                optimizer.zero_grad()
                loss_final.backward()
                optimizer.step()

                with torch.no_grad():
                    for m_ind in range(source_count):
                        imgs, pids, _, _, _ = self._parse_data(batch_data[m_ind])
                        f_new, _ = self.model(imgs)
                        self.memory[m_ind].module.MomentumUpdate(f_new, pids)


                losses.update(loss_final.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()


            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Total loss {:.3f} ({:.3f})\t'
                      'Loss {:.3f}({:.3f})\t'
                      'LossMeta {:.3f}({:.3f})'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              losses.val, losses.avg,
                              losses_meta_train.val, losses_meta_train.avg,
                              losses_meta_test.val, losses_meta_test.avg))

    def _parse_data(self, inputs):
        imgs, names, pids, cams, dataset_id, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda(), cams.cuda(), dataset_id.cuda()



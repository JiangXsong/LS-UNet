# Created on 2018/12
# Author: Kaituo XU

import os
import time

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Solver(object):
    
    def __init__(self, data, model, optimizer, args):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.model = model
        self.optimizer = optimizer
        self.mse_loss = nn.MSELoss()
        self.sl1_loss = nn.L1Loss()

        # Training config
        self.use_cuda = args.use_cuda
        self.epochs = args.epochs
        self.half_lr = args.half_lr
        self.early_stop = args.early_stop
        self.max_norm = args.max_norm
        # save and load model
        self.save_folder = args.save_folder
        self.checkpoint = args.checkpoint
        self.continue_from = args.continue_from
        self.model_path = args.model_path
        # logging
        self.print_freq = args.print_freq
        # visualizing loss using visdom
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)
        self._reset()

    def _reset(self):
        # Reset
        if self.continue_from:
            print('Loading checkpoint model %s' % self.continue_from)
            package = torch.load(self.continue_from)
            self.model.module.load_state_dict(package['state_dict'])
            self.optimizer.load_state_dict(package['optim_dict'])
            self.start_epoch = int(package.get('epoch', 1))
            self.tr_loss[:self.start_epoch] = package['tr_loss'][:self.start_epoch]
            self.cv_loss[:self.start_epoch] = package['cv_loss'][:self.start_epoch]
        else:
            self.start_epoch = 0
        # Create save folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.prev_val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.halving = False
        self.val_no_impv = 0

    def train(self):
        # Train model multi-epoches
        tr_loss_value, val_loss_value = [], []
        loss_weigth = [0.5, 0.5] #[mse_weigth, sl1_weigth]
        
        for epoch in range(self.start_epoch, self.epochs):
            # Train one epoch
            print("Training...")
            self.model.train()  # Turn on BatchNorm & Dropout
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch, loss_weigth)
            tr_loss_value.append(tr_avg_loss)
            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Train Loss {2:.5f}'.format(
                      epoch + 1, time.time() - start, tr_avg_loss))
            print('-' * 85)

            # Save model each epoch
            if self.checkpoint:
                file_path = os.path.join(
                    self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))
                torch.save(self.model.state_dict(), file_path)
                print('Saving checkpoint model to %s' % file_path)
            
            # Cross validation
            print('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            val_loss = self._run_one_epoch(epoch, loss_weigth, cross_valid=True)
            val_loss_value.append(val_loss)
            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Valid Loss {2:.5f}'.format(
                      epoch + 1, time.time() - start, val_loss))
            print('-' * 85)

            # Adjust learning rate (halving)
            if self.half_lr:
                if val_loss >= self.prev_val_loss:
                    self.val_no_impv += 1
                    if self.val_no_impv >= 3:
                        self.halving = True
                    if self.val_no_impv >= 10 and self.early_stop:
                        print("No imporvement for 10 epochs, early stopping.")
                        break
                else:
                    self.val_no_impv = 0
            if self.halving:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = \
                    optim_state['param_groups'][0]['lr'] / 2.0
                self.optimizer.load_state_dict(optim_state)
                print('Learning rate adjusted to: {lr:.6f}'.format(
                    lr=optim_state['param_groups'][0]['lr']))
                self.halving = False
            self.prev_val_loss = val_loss

            # Save the best model
            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = val_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                file_path = os.path.join(self.save_folder, self.model_path)
                torch.save(self.model.state_dict(), file_path)
                print("Find better validated model, saving to %s" % file_path)
        
        plt.plot(tr_loss_value, 'r', label='tr')
        plt.plot(val_loss_value, 'b', label='val')
        plt.legend() # 圖例
        plt.title("unet pytorch Loss") # title
        plt.ylabel("loss") # y label
        plt.xlabel("epoch") # x label

        plt.show()
        plt.savefig(self.save_folder+"/Loss.png")
            
    def _run_one_epoch(self, epoch, loss_weigth:list, cross_valid=False):
        start = time.time()
        total_loss = 0

        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        for i, (xt,yftm,clean) in enumerate(data_loader):
            #padded_mixture, mixture_lengths, padded_source = data
            if self.use_cuda:
                xt = xt.cuda()
                yftm = yftm.cuda()
                clean = clean.cuda()
            
            #print("xt ", xt.shape)
            #print("yt ", yt.shape)
            
            ya = clean.squeeze(0)
            yb = yftm.squeeze(0)
            
            #print("xa ", xa.shape)
            #print("ya ", ya.shape)
            
            output, denoise = self.model(xt)
            loss = loss_weigth[0] * self.mse_loss(denoise, ya) + loss_weigth[1] * self.sl1_loss(output, yb)
            
            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.max_norm)
                self.optimizer.step()

            total_loss += loss.item()

            if i % self.print_freq == 0:
                
                print('Epoch {0} | Iter {1} | Average Loss {2:.5f} | '
                      'Current Loss {3:.7f} | {4:.1f} ms/batch'.format(
                          epoch + 1, i + 1, total_loss / (i + 1),
                          loss.item(), 1000 * (time.time() - start) / (i + 1)),
                      flush=True)
        
        return total_loss / (i + 1)

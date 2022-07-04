"""
Trainer class of PartI and PartII.
"""


import os
import sys
sys.path.append('..')
import torch
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils.utils as utils
import utils.dataset as dataset
import utils.network as network
import train.loss_val as loss_val


#based on configuration
class Trainer_partI:
    def __init__(self,cfg):
        self.cfg=cfg
        self.model_dir=f'{self.cfg.model_fn}/{self.cfg.train_network_type}'
        if not os.path.exists(self.model_dir): os.makedirs(self.model_dir)
        
        #config write
        self.pth_fn=os.path.join(self.model_dir,'model.pth')
        self.best_pth_fn=os.path.join(self.model_dir,'model_best.pth')

        self._init_dataset()
        self._init_network()
        self._init_logger()

    def _init_dataset(self):
        self.train_set=dataset.name2traindataset[self.cfg.trainset_type](self.cfg,is_training=True)
        self.val_set=dataset.name2traindataset[self.cfg.trainset_type](self.cfg,is_training=False)
        self.train_set=DataLoader(self.train_set,1,shuffle=True,num_workers=self.cfg.worker_num)
        self.val_set=DataLoader(self.val_set,self.cfg.batch_size,shuffle=False,num_workers=self.cfg.worker_num,drop_last=True)
        print(f'train set len {len(self.train_set)}')
        print(f'val set len {len(self.val_set)}')

    def _init_network(self):
        utils.config_writer(self.cfg,f'{self.model_dir}/train_info.txt')
        self.network=network.name2network[self.cfg.train_network_type](self.cfg).cuda()
        self.optimizer = Adam(self.network.parameters(), lr=self.cfg.lr_init)
        self.loss=loss_val.name2loss[self.cfg.loss_type](self.cfg)
        self.val_evaluator=loss_val.name2val[self.cfg.val_type](self.cfg)
        self.lr_setter=utils.ExpDecayLR(self.cfg,len(self.train_set)*self.cfg.lr_decay_step)
    
    def _load_model(self):
        best_para,start_step=0,0
        if os.path.exists(self.pth_fn):
            checkpoint=torch.load(self.pth_fn)
            best_para = checkpoint['best_para']
            start_step = checkpoint['step']
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f'==> resuming from step {start_step} best para {best_para}')

        return best_para, start_step

    def _save_model(self, step, best_para, save_fn=None):
        save_fn=self.pth_fn if save_fn is None else save_fn
        torch.save({
            'step':step,
            'best_para':best_para,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        },save_fn)

    def _init_logger(self):
        self.logger = utils.Logger(self.model_dir)

    def _log_data(self,results,step,prefix='train',verbose=False):
        log_results={}
        for k, v in results.items():
            if isinstance(v,float) or np.isscalar(v):
                log_results[k] = v
            elif type(v)==np.ndarray:
                log_results[k]=np.mean(v)
            else:
                log_results[k]=np.mean(v.detach().cpu().numpy())
        self.logger.log(log_results,prefix,step,verbose)

    def run(self):
        best_para,start_step=self._load_model()
        pbar=tqdm(total=self.cfg.epochs*len(self.train_set),bar_format='{r_bar}')
        pbar.update(start_step)
        step=start_step
        wholeloss=0
        start_epoch=start_step//len(self.train_set)
        start_step=start_step-start_epoch*len(self.train_set)
        whole_step=len(self.train_set)*self.cfg.epochs
        for epoch in range(start_epoch,self.cfg.epochs):
            for i,train_data in enumerate(self.train_set):
                step+=1
                if not self.cfg.multi_gpus:
                    train_data = utils.to_cuda(train_data)

                self.network.train()
                utils.reset_learning_rate(self.optimizer,self.lr_setter(step))
                self.optimizer.zero_grad()
                self.network.zero_grad()

                log_info={}
                outputs=self.network(train_data)
                loss=self.loss(outputs)
                loss.backward()
                self.optimizer.step()
                wholeloss+=loss.detach()

                if (step+1) % self.cfg.train_log_step == 0:
                    loss_info={'loss':wholeloss/self.cfg.train_log_step}
                    self._log_data(loss_info,step+1,'train')
                    wholeloss=0

                if (step+1)%self.cfg.val_interval==0:
                    val_results=self.val_evaluator(self.network, self.val_set)
                    val_para=val_results['whole_recall']
                    print(f'New model recall: {val_para:.5f}')
                    if val_para>=best_para:
                        print(f'New best model recall: {val_para:.5f} previous {best_para:.5f}')
                        best_para=val_para
                        self._save_model(step+1,best_para,self.best_pth_fn)
                    self._log_data(val_results,step+1,'val')

                if (step+1)%self.cfg.save_interval==0:
                    self._save_model(step+1,best_para)

                pbar.set_postfix(loss=float(loss.detach().cpu().numpy()),lr=self.optimizer.state_dict()['param_groups'][0]['lr'])
                pbar.update(1)
                if step>=whole_step:
                    break
            if step>=whole_step:
                    break
        pbar.close()
   
class Trainer_partII:
    def __init__(self,cfg):
        self.cfg=cfg
        self.model_dir=f'{self.cfg.model_fn}/{self.cfg.train_network_type}'
        if not os.path.exists(self.model_dir): os.makedirs(self.model_dir)
        
        self.pth_fn=os.path.join(self.model_dir,'model.pth')
        self.best_pth_fn=os.path.join(self.model_dir,'model_best.pth')

        self._init_dataset()
        self._init_network()
        self._init_logger()

    def _init_dataset(self):
        utils.config_writer(self.cfg,f'{self.model_dir}/train_info.txt')
        self.train_set=dataset.name2traindataset[self.cfg.trainset_type](self.cfg,is_training=True)
        self.val_set=dataset.name2traindataset[self.cfg.trainset_type](self.cfg,is_training=False)
        self.train_set=DataLoader(self.train_set,1,shuffle=True,num_workers=self.cfg.worker_num)
        self.val_set=DataLoader(self.val_set,self.cfg.batch_size,shuffle=False,num_workers=self.cfg.worker_num,drop_last=True)
        print(f'train set len {len(self.train_set)}')
        print(f'val set len {len(self.val_set)}')

    def _init_network(self):
        self.network=network.name2network[self.cfg.train_network_type](self.cfg).cuda()
        pretrained_partI_model=torch.load(self.cfg.PartI_pretrained_model_fn)['network_state_dict']
        pretrained_partI_model_for_partII={}
        for key,val in pretrained_partI_model.items():
            pretrained_partI_model_for_partII[f'PartI_net.{key}']=val
        self.network.load_state_dict(pretrained_partI_model_for_partII,strict=False)
        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.network.parameters()), lr=self.cfg.lr_init)
        self.loss=loss_val.name2loss[self.cfg.loss_type](self.cfg)
        self.val_evaluator=loss_val.name2val[self.cfg.val_type](self.cfg)
        self.lr_setter=utils.ExpDecayLR(self.cfg,len(self.train_set)*self.cfg.lr_decay_step)
    
    def _load_model(self):
        best_para,start_step=100,0
        print(self.pth_fn)
        if os.path.exists(self.pth_fn): 
            checkpoint=torch.load(self.pth_fn)
            best_para = checkpoint['best_para']
            start_step = checkpoint['step']
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f'==> resuming from step {start_step} best para {best_para}')

        return best_para, start_step

    def _save_model(self, step, best_para, save_fn=None):
        save_fn=self.pth_fn if save_fn is None else save_fn
        torch.save({
            'step':step,
            'best_para':best_para,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        },save_fn)

    def _init_logger(self):
        self.logger = utils.Logger(self.model_dir)

    def _log_data(self,results,step,prefix='train',verbose=False):
        log_results={}
        for k, v in results.items():
            if isinstance(v,float) or np.isscalar(v):
                log_results[k] = v
            elif type(v)==np.ndarray:
                log_results[k]=np.mean(v)
            else:
                log_results[k]=np.mean(v.detach().cpu().numpy())
        self.logger.log(log_results,prefix,step,verbose)

    def run(self):
        best_para,start_step=self._load_model()
        pbar=tqdm(total=self.cfg.epochs*len(self.train_set),bar_format='{r_bar}')
        pbar.update(start_step)
        step=start_step
        wholeloss=0
        start_epoch=start_step//len(self.train_set)
        start_step=start_step-start_epoch*len(self.train_set)
        whole_step=len(self.train_set)*self.cfg.epochs
        for epoch in range(start_epoch,self.cfg.epochs):
            for i,train_data in enumerate(self.train_set):
                step+=1
                if not self.cfg.multi_gpus:
                    train_data = utils.to_cuda(train_data)

                self.network.train()
                utils.reset_learning_rate(self.optimizer,self.lr_setter(step))
                self.optimizer.zero_grad()
                self.network.zero_grad()

                log_info={}
                outputs=self.network(train_data)
                pre=outputs['quaternion_pre']
                gt=torch.squeeze(train_data['deltaR'])
                loss=self.loss(pre,gt)
                loss.backward()
                self.optimizer.step()
                wholeloss+=loss.detach()
                
                if (step+1) % self.cfg.train_log_step == 0:
                    loss_info={'loss':wholeloss/self.cfg.train_log_step}
                    self._log_data(loss_info,step+1,'train')
                    wholeloss=0

                if (step+1)%self.cfg.val_interval==0:
                    val_results=self.val_evaluator(self.network, self.val_set)
                    val_R=val_results['R_error']
                    val_R_static=val_results['R_error_statics']
                    print(f'R difference now: {val_R:.5f}')
                    print(f'R difference static: {val_R_static[0]}, {val_R_static[1]}, {val_R_static[2]}, {val_R_static[3]}, {val_R_static[4]}, {val_R_static[5]}')
                    if val_R<=best_para:
                        print(f'R difference now: {val_R:.5f} previous {best_para:.5f}')
                        best_para=val_R
                        self._save_model(step+1,best_para,self.best_pth_fn)
                    self._log_data(val_results,step+1,'val')

                if (step+1)%self.cfg.save_interval==0:
                    self._save_model(step+1,best_para)

                pbar.set_postfix(loss=float(loss.detach().cpu().numpy()),lr=self.optimizer.state_dict()['param_groups'][0]['lr'])
                pbar.update(1)
                if step>=whole_step:
                    break
            if step>=whole_step:
                    break
        pbar.close()

name2trainer={
    'PartI':Trainer_partI,
    'PartII':Trainer_partII
}
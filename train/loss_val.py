#loss and validation
import torch
import abc
import utils.utils as utils
import time
from tqdm import tqdm
import numpy as np

#loss PartI
class Loss(abc.ABC):
    def __init__(self,keys: list or tuple):
        self.keys=list(keys)

    @abc.abstractmethod
    def __call__(self, data_pr, data_gt, **kwargs):
        pass

class Batch_hard_Rindex_loss(Loss):
    def __init__(self,cfg):
        super().__init__(['triplet_ranking_Rindex_loss'])
        self.R_perm=torch.from_numpy(np.load(f'{cfg.SO3_related_files}/60_60.npy').astype(np.int).reshape([-1])).cuda()
        self.class_loss=torch.nn.CrossEntropyLoss()

    def eqvloss(self,eqvfeat0,eqvfeat1):
        B,F,G=eqvfeat0.shape
        eqvfeat0=eqvfeat0[:,:,self.R_perm].reshape([B,F,G,G])
        score=torch.einsum('bfgk,bfk->bg',eqvfeat0,eqvfeat1)
        return score


    def __call__(self, data_pr):
        Index= data_pr['DR_true_index'].type(torch.int64)
        
        B=Index.shape[0]
        feats0=data_pr['feats0_inv'] # bn,f
        feats1=data_pr['feats1_inv'] # bn,f

        B,L = feats1.shape
        q_vec=feats0.contiguous().view(B, 1, L)
        ans_vecs=feats1.contiguous().view(1, B, L)
        dist = ((q_vec - ans_vecs) ** 2).sum(-1)
        dist = torch.nn.functional.log_softmax(dist, 1)
        loss_true = torch.diag(dist)
        loss_false=torch.min(dist+torch.eye(B).cuda(),dim=1)[0]
        loss=torch.mean(torch.clamp_min(loss_true-loss_false+0.3,0))

        score=self.eqvloss(data_pr['feats0_eqv_af_conv'],data_pr['feats1_eqv_af_conv'])
        sign=torch.nn.functional.one_hot(Index,60)
        eqv_loss=self.class_loss(score,Index)
        return 5*loss+eqv_loss

#loss PartII
class L1_loss(Loss):
    def __init__(self,cfg):
        super().__init__(['L1_Loss'])
        self.loss=torch.nn.SmoothL1Loss(reduction='sum')
    
    def __call__(self,patch_op,patch_gt):
        return self.loss(patch_op,patch_gt)


class L2_loss(Loss):
    def __init__(self,cfg):
        super().__init__(['L2_Loss'])
        self.loss=torch.nn.MSELoss(reduction='sum')
    
    def __call__(self,patch_op,patch_gt):
        return self.loss(patch_op,patch_gt)

#validation part

class Validation_PartI:
    def __init__(self,cfg):
        self.cfg=cfg
        self.loss=name2loss[self.cfg.loss_type](cfg)
    
    def recall(self,data):
        feats0=data["feats0_inv"]
        feats1=data['feats1_inv']
        bn,fn=feats0.shape
        scores=torch.norm(feats0[None,:,:]-feats1[:,None,:],dim=-1)
        idxs_pr=torch.argmin(scores,1)
        idxs_gt=torch.arange(bn).to(feats0.device).long()
        correct_rate=torch.mean((idxs_pr==idxs_gt).float())
        return correct_rate
    
    def recall_index(self,data):
        feats0=data["feats0_inv"]
        feats1=data['feats1_inv']
        bn,fn=feats0.shape
        scores=torch.norm(feats0[None,:,:]-feats1[:,None,:],dim=-1)
        idxs_pr=torch.argmin(scores,1)
        idxs_gt=torch.arange(bn).to(feats0.device).long()
        return torch.where(idxs_gt==idxs_pr)[0]

    def __call__(self, model, eval_dataset):
        model.eval()
        eval_results={}
        begin=time.time()
        alloutput0=[]
        alloutput1=[]
        allloss=[]
        all_batch_recall=[]
        all_DR_ability=[]
        all_DR_ok=[]
        
        for data in tqdm(eval_dataset):
            data = utils.to_cuda(data)
            with torch.no_grad():
                outputs=model(data)
                #for whole cal
                alloutput0.append(outputs['feats0_inv'].cpu())
                alloutput1.append(outputs['feats1_inv'].cpu())
                all_DR_ability.append(outputs['DR_pre_ability'].cpu())
                all_DR_ok.append((outputs['DR_true_index']==outputs['DR_pre_index']).cpu().numpy())
                #val loss
                loss=self.loss(outputs)
                allloss.append(loss)
                #batch val
                bc_recall=self.recall(outputs)
                all_batch_recall.append(bc_recall)

        #val loss
        val_loss=torch.mean(torch.tensor(allloss))
        #batch_recall
        batch_recall=torch.mean(torch.tensor(all_batch_recall))
        #whole dataset recall
        alloutput0=torch.cat(alloutput0,dim=0)
        alloutput1=torch.cat(alloutput1,dim=0)
        
        alloutputs={'feats0_inv':alloutput0,'feats1_inv':alloutput1}
        whole_recall=self.recall(alloutputs)
        ok_index=self.recall_index(alloutputs).cpu().numpy().astype(np.int)
        all_DR_ok=np.concatenate(all_DR_ok)
        double_ok_rate=np.mean(all_DR_ok[ok_index])

        return {"val_loss":val_loss, "whole_recall":whole_recall, 'batch_recall':batch_recall,'PartI_DR_ability':double_ok_rate}

class Validation_PartII:
    def __init__(self,cfg):
        self.cfg=cfg
        self.loss=name2loss[self.cfg.loss_type](self.cfg)
        
    def diff_cal(self,R_pre,R_gt):
        result=0
        eps=1e-7
        result=[]
        R_pre=R_pre/torch.clamp_min(torch.norm(R_pre,dim=1,keepdim=True),min=1e-4)#q
        for i in range(R_pre.shape[0]):
            loss_q = torch.clamp_min((1.0 - torch.sum(R_pre[i]*R_gt[i])**2),min=eps)
            err_q = torch.acos(1 - 2 * loss_q)
            result.append(err_q/np.pi*180)
        return result
    
    def static(self,errors):
        result=torch.zeros(6)
        for e in errors:
            e_index=int(e)
            if e_index<6:
                result[e_index]+=1
        result/=errors.shape[0]
        return result

    def __call__(self, model, eval_dataset):
        model.eval()
        eval_results={}
        begin=time.time()
        alloutput0=[]
        part1_ability=[]
        all_loss=[]
        all_R_error=[]
        
        for data in tqdm(eval_dataset):
            quaternion_gt=torch.squeeze(data['deltaR'])
            data = utils.to_cuda(data)
            with torch.no_grad():
                outputs=model(data)
                part1_ability.append(outputs['part1_ability'])
                quaternion=outputs['quaternion_pre'].cpu()
                #val loss
                loss=self.loss(quaternion,quaternion_gt)
                all_loss.append(loss)
                #batch val
                ok_idxs=torch.where(outputs['pre_idxs'].cpu()==outputs['true_idxs'].cpu())
                R_error=self.diff_cal(quaternion,quaternion_gt)
                for i in range(len(R_error)):
                    all_R_error.append(R_error[i])
                
        #val loss
        all_loss=torch.Tensor(all_loss)
        all_R_error=torch.Tensor(all_R_error)
        part1_ability=torch.Tensor(part1_ability)
        R_error_statics=self.static(all_R_error)
        
        return {'val_loss':torch.mean(all_loss), 'R_error':torch.mean(all_R_error),'part1_ability':torch.mean(part1_ability),'R_error_statics':R_error_statics}


name2loss={
    'Batch_hard_Rindex_loss':Batch_hard_Rindex_loss,
    'L2_loss_partII':L2_loss
}

name2val={
    "Val_partI":Validation_PartI,
    'Val_partII':Validation_PartII
}
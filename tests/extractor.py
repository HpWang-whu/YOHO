import os,sys
sys.path.append('..')
import time
import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
from utils.r_eval import compute_R_diff,matrix_from_quaternion
from torch.utils.data import DataLoader
from utils.utils import transform_points, read_pickle,make_non_exists_dir,to_cuda
from utils.network import name2network
from tests.estimator import name2estimator


class extractor_PartI():
    def __init__(self,cfg):
        self.cfg=cfg
        self.network=name2network[f'{self.cfg.test_network_type}'](self.cfg).cuda()
        self.model_fn=f'{self.cfg.model_fn}/{self.cfg.train_network_type}/model.pth'
        self.best_model_fn=f'{self.cfg.model_fn}/{self.cfg.train_network_type}/model_best.pth'

    #Model
    def _load_model(self):
        best_para,start_step=0,0
        if os.path.exists(self.best_model_fn):
            checkpoint=torch.load(self.best_model_fn)
            best_para = checkpoint['best_para']
            self.network.load_state_dict(checkpoint['network_state_dict'])
            print(f'Resuming best para {best_para}')
        else:
            raise ValueError("No model exists")

    #Extract
    def Extract(self,dataset):
        #data input 5000*32*60
        #output: 5000*32*69->save
        self._load_model()
        self.network.eval()
        FCGF_input_dir=f'{self.cfg.output_cache_fn}/Testset/{dataset.name}/FCGF_Input_Group_feature'
        YOMO_output_dir=f'{self.cfg.output_cache_fn}/Testset/{dataset.name}/YOMO_Output_Group_feature'
        make_non_exists_dir(YOMO_output_dir)
        print(f'Extracting the PartI descriptors on {dataset.name}')
        for pc_id in tqdm(dataset.pc_ids):
            if os.path.exists(f'{YOMO_output_dir}/{pc_id}.npy'):continue
            Input_feature=np.load(f'{FCGF_input_dir}/{pc_id}.npy') #5000*32*60
            output_feature=[]
            bi=0
            while(bi*self.cfg.test_batch_size<Input_feature.shape[0]):
                start=bi*self.cfg.test_batch_size
                end=(bi+1)*self.cfg.test_batch_size
                batch=torch.from_numpy(Input_feature[start:end,:,:].astype(np.float32)).cuda()
                with torch.no_grad():
                    batch_output=self.network(batch)
                output_feature.append(batch_output['eqv'].cpu().numpy())
                bi+=1
            output_feature=np.concatenate(output_feature,axis=0)
            np.save(f'{YOMO_output_dir}/{pc_id}.npy',output_feature) #5000*32*60



class extractor_dr_index():
    def __init__(self,cfg):
        self.cfg=cfg
        self.Nei_in_SO3=torch.from_numpy(np.load(f'{self.cfg.SO3_related_files}/60_60.npy').reshape([-1]).astype(np.int)).cuda()

    def Des2R_torch(self,des1_eqv,des2_eqv):#beforerot afterrot
        des1_eqv=des1_eqv[:,self.Nei_in_SO3].reshape([-1,60,60])
        cor=torch.einsum('fag,fg->a',des1_eqv,des2_eqv)
        return torch.argmax(cor)

    def Batch_Des2R_torch(self,des1_eqv,des2_eqv):#beforerot afterrot
        B,F,G=des1_eqv.shape
        des1_eqv=des1_eqv[:,:,self.Nei_in_SO3].reshape([B,F,60,60])
        cor=torch.einsum('bfag,bfg->ba',des1_eqv,des2_eqv)
        return torch.argmax(cor,dim=1)
  
    def PartI_Rindex(self,dataset):
        match_dir=f'{self.cfg.output_cache_fn}/Testset/{dataset.name}/Match'
        Save_dir=f'{match_dir}/DR_index'
        make_non_exists_dir(Save_dir)
        if dataset.name[0:4]=='3dLo':
            datasetname=f'3d{dataset.name[4:]}'
        else:
            datasetname=dataset.name
        Feature_dir=f'{self.cfg.output_cache_fn}/Testset/{datasetname}/YOMO_Output_Group_feature'
        
        print(f'extract the drindex of the matches on {dataset.name}')
        for pair in tqdm(dataset.pair_ids):
            id0,id1=pair
            if os.path.exists(f'{Save_dir}/{id0}-{id1}.npy'):continue
            match_pps=np.load(f'{match_dir}/{id0}-{id1}.npy')
            feats0=np.load(f'{Feature_dir}/{id0}.npy') #5000,32,60
            feats1=np.load(f'{Feature_dir}/{id1}.npy') #5000,32,60
            feats0=torch.from_numpy(feats0[match_pps[:,0]].astype(np.float32)).cuda()
            feats1=torch.from_numpy(feats1[match_pps[:,1]].astype(np.float32)).cuda()
            pre_idxs=self.Batch_Des2R_torch(feats1,feats0).cpu().numpy()
            np.save(f'{Save_dir}/{id0}-{id1}.npy',pre_idxs)

           

class extractor_PartII():
    def __init__(self,cfg):
        self.cfg=cfg
        self.network=name2network[f'{self.cfg.test_network_type}'](self.cfg).cuda()
        self.model_fn=f'{self.cfg.model_fn}/{self.cfg.train_network_type}/model.pth'
        self.best_model_fn=f'{self.cfg.model_fn}/{self.cfg.train_network_type}/model_best.pth'
        self.Rgroup=np.load(f'{self.cfg.SO3_related_files}/Rotation.npy').astype(np.float32)

    #Model_import
    def _load_model(self):
        best_para,start_step=0,0
        if os.path.exists(self.best_model_fn):
            print(self.best_model_fn)
            checkpoint=torch.load(self.best_model_fn)
            best_para = checkpoint['best_para']
            self.network.load_state_dict(checkpoint['network_state_dict'],strict=False)
            print(f'Resuming best para {best_para}')
        else:
            raise ValueError("No model exists")
    

    def batch_create(self,feats0_fcgf,feats1_fcgf,feats0_yomo,feats1_yomo,index_pre,start,end):
        #attention: here feats0->feats1_in_batch for it is afterrot
        feats0_fcgf=torch.from_numpy(feats0_fcgf[start:end,:,:].astype(np.float32))
        feats1_fcgf=torch.from_numpy(feats1_fcgf[start:end,:,:].astype(np.float32))
        feats0_yomo=torch.from_numpy(feats0_yomo[start:end,:,:].astype(np.float32))
        feats1_yomo=torch.from_numpy(feats1_yomo[start:end,:,:].astype(np.float32))
        index_pre=torch.from_numpy(index_pre[start:end].astype(np.int))
        return {
                'before_eqv0':feats1_fcgf,#exchanged
                'before_eqv1':feats0_fcgf,
                'after_eqv0':feats1_yomo,
                'after_eqv1':feats0_yomo,
                'pre_idx':index_pre
        }



    def PartII_R_pre(self,dataset):
        self._load_model()
        self.network.eval()
        #dataset: (5000*32*60->pp*32*60)*4 + pre_index_trans-> pp*128*60
        match_dir=f'{self.cfg.output_cache_fn}/Testset/{dataset.name}/Match'
        DRindex_dir=f'{match_dir}/DR_index'
        Save_dir=f'{match_dir}/Trans_pre'
        make_non_exists_dir(Save_dir)
        
        if dataset.name[0:4]=='3dLo':
            datasetname=f'3d{dataset.name[4:]}'
        else:
            datasetname=dataset.name
        Keys_dir=f'{self.cfg.origin_data_dir}/{datasetname}/Keypoints_PC'
        FCGF_dir=f'{self.cfg.output_cache_fn}/Testset/{datasetname}/FCGF_Input_Group_feature'
        YOMO_dir=f'{self.cfg.output_cache_fn}/Testset/{datasetname}/YOMO_Output_Group_feature'

        #feat1:beforrot feat0:afterrot
        print(f'extracting the PartII feature on {dataset.name}')       
        alltime=0 
        for pair in tqdm(dataset.pair_ids):
            id0,id1=pair
            if os.path.exists(f'{Save_dir}/{id0}-{id1}.npy'):continue
            pps=np.load(f'{match_dir}/{id0}-{id1}.npy')
            feats0_fcgf=np.load(f'{FCGF_dir}/{id0}.npy')[pps[:,0],:,:] #pps*32*60
            feats1_fcgf=np.load(f'{FCGF_dir}/{id1}.npy')[pps[:,1],:,:] #pps*32*60
            feats0_yomo=np.load(f'{YOMO_dir}/{id0}.npy')[pps[:,0],:,:] #pps*32*60
            feats1_yomo=np.load(f'{YOMO_dir}/{id1}.npy')[pps[:,1],:,:] #pps*32*60
            Index_pre=np.load(f'{DRindex_dir}/{id0}-{id1}.npy')        #pps

            Keys0=dataset.get_kps(id0)[pps[:,0],:]  #pps*3
            Keys1=dataset.get_kps(id1)[pps[:,1],:]  #pps*3

            bi=0
            Rs=[]
            while(bi*self.cfg.test_batch_size<feats0_fcgf.shape[0]):
                start=bi*self.cfg.test_batch_size
                end=(bi+1)*self.cfg.test_batch_size
                batch=self.batch_create(feats0_fcgf,feats1_fcgf,feats0_yomo,feats1_yomo,Index_pre,start,end)
                batch=to_cuda(batch)
                with torch.no_grad():
                    batch_output=self.network(batch)
                bi+=1
                deltaR=batch_output['quaternion_pre'].cpu().numpy()
                anchorR=batch_output['pre_idxs'].cpu().numpy()
                for i in range(deltaR.shape[0]):
                    R_residual=matrix_from_quaternion(deltaR[i])
                    R_anchor=self.Rgroup[int(anchorR[i])]
                    Rs.append((R_residual@R_anchor)[None,:,:])
            Rs=np.concatenate(Rs,axis=0) #pps*3*3
            Trans=[]
            for R_id in range(Rs.shape[0]):
                R=Rs[R_id]
                key0=Keys0[R_id] #after rot key0=t+key1@R.T
                key1=Keys1[R_id] #before rot
                t=key0-key1@R.T
                trans_one=np.concatenate([R,t[:,None]],axis=1)
                Trans.append(trans_one[None,:,:])
            Trans=np.concatenate(Trans,axis=0)
            np.save(f'{Save_dir}/{id0}-{id1}.npy',Trans)

                
name2extractor={
    'PartI':extractor_PartI,
    'PartII':extractor_PartII
}
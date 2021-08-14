import os,sys
import time
sys.path.append('..')
import numpy as np
import open3d as o3d
from tqdm import tqdm
import utils.RR_cal as RR_cal
from utils.dataset import EvalDataset,get_dataset
from utils.utils import transform_points, read_pickle
from utils.r_eval import compute_R_diff
from tests.extractor import name2extractor,extractor_dr_index
from tests.matcher import name2matcher
from tests.estimator import name2estimator

def make_non_exists_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class Evaluator_PartI:
    def __init__(self,cfg,max_iter,TR_max_iter):
        self.max_iter=max_iter
        self.TR_max_iter=TR_max_iter
        self.cfg=cfg
        self.extractor=name2extractor[self.cfg.extractor](self.cfg)
        self.matcher=name2matcher[self.cfg.matcher](self.cfg)
        self.drindex_extractor=extractor_dr_index(self.cfg)
        est=self.cfg.estimator
        if self.max_iter>500:
            est='yohoc_mul'
        self.estimator=name2estimator[est](self.cfg)

    def run_onescene(self,dataset):
        #extractor:
        if not dataset.name[0:4]=='3dLo':
            self.extractor.Extract(dataset)
        self.matcher.match(dataset)
        self.drindex_extractor.PartI_Rindex(dataset)
        self.estimator.ransac(dataset,self.max_iter,self.TR_max_iter)

    def Feature_match_Recall(self,dataset,ratio=0.05):
        if dataset.name[0:4]=='3dLo':
            datasetname=f'3d{dataset.name[4:]}'
        else:
            datasetname=dataset.name
        Keys_dir=f'{self.cfg.origin_data_dir}/{datasetname}/Keypoints_PC'
        pair_fmrs=[]
        for pair in tqdm(dataset.pair_ids):
            id0,id1=pair
            #match
            matches=np.load(f'{self.cfg.output_cache_fn}/Testset/{dataset.name}/Match/{id0}-{id1}.npy')
            keys0=np.load(f'{Keys_dir}/cloud_bin_{id0}Keypoints.npy')[matches[:,0],:]
            keys1=np.load(f'{Keys_dir}/cloud_bin_{id1}Keypoints.npy')[matches[:,1],:]
            #gt
            gt=dataset.get_transform(id0,id1)
            #ratio
            keys1=transform_points(keys1,gt)
            dist=np.sqrt(np.sum(np.square(keys0-keys1),axis=-1))
            pair_fmr=np.mean(dist<self.cfg.ok_match_dist_threshold) #ok ratio in one pair
            pair_fmrs.append(pair_fmr)                              
        pair_fmrs=np.array(pair_fmrs)                               #ok ratios in one scene
        FMR=np.mean(pair_fmrs>ratio)                                #FMR in one scene
        return FMR, pair_fmrs



    def eval(self):
        correct_ratios_wholeset={}
        datasets=get_dataset(self.cfg,False)
        FMRS=[]
        all_pair_fmrs=[]
        for scene,dataset in datasets.items():
            if scene=='wholesetname':continue
            self.run_onescene(dataset)
            print(f'eval the FMR result on {dataset.name}')
            FMR,pair_fmrs=self.Feature_match_Recall(dataset,ratio=self.cfg.fmr_ratio)
            FMRS.append(FMR)
            all_pair_fmrs.append(pair_fmrs)
        FMRS=np.array(FMRS)
        all_pair_fmrs=np.concatenate(all_pair_fmrs,axis=0)

        #RR
        datasetname=datasets['wholesetname']
        Mean_Registration_Recall,c_flags,c_errors=RR_cal.benchmark(self.cfg,datasets,self.max_iter,yoho_sign='YOHO_C')

        #print and save:
        msg=f'{datasetname}-{self.cfg.descriptor}-{self.cfg.extractor}-{self.cfg.matcher}-{self.cfg.estimator}-{self.max_iter}iterations\n'
        msg+=f'correct ratio avg {np.mean(all_pair_fmrs):.5f}\n' \
             f'correct ratio>0.05 avg {np.mean(FMRS):.5f}  std {np.std(FMRS):.5f}\n' \
             f'Mean_Registration_Recall {Mean_Registration_Recall}\n'

        with open('data/results.log','a') as f:
            f.write(msg+'\n')
        print(msg)
               
class Evaluator_PartII:
    def __init__(self,cfg,max_iter,TR_max_iter):
        self.max_iter=max_iter
        self.TR_max_iter=TR_max_iter
        self.cfg=cfg
        self.extractor=name2extractor[self.cfg.extractor](self.cfg)
        self.matcher=name2matcher[self.cfg.matcher](self.cfg)
        self.estimator=name2estimator[self.cfg.estimator](self.cfg)
        self.drindex_extractor=extractor_dr_index(self.cfg)

    def run_onescene(self,dataset):
        #extractor:
        self.matcher.match(dataset)
        self.drindex_extractor.PartI_Rindex(dataset)
        self.extractor.PartII_R_pre(dataset)
        self.estimator.ransac(dataset,self.max_iter,self.TR_max_iter)


    def Feature_match_Recall(self,dataset,ratio=0.05):
        if dataset.name[0:4]=='3dLo':
            datasetname=f'3d{dataset.name[4:]}'
        else:
            datasetname=dataset.name
        Keys_dir=f'{self.cfg.origin_data_dir}/{datasetname}/Keypoints_PC'
        pair_fmrs=[]
        for pair in tqdm(dataset.pair_ids):
            id0,id1=pair
            #match
            matches=np.load(f'{self.cfg.output_cache_fn}/Testset/{dataset.name}/Match/{id0}-{id1}.npy')
            keys0=np.load(f'{Keys_dir}/cloud_bin_{id0}Keypoints.npy')[matches[:,0],:]
            keys1=np.load(f'{Keys_dir}/cloud_bin_{id1}Keypoints.npy')[matches[:,1],:]
            #gt
            gt=dataset.get_transform(id0,id1)
            #ratio
            keys1=transform_points(keys1,gt)
            dist=np.sqrt(np.sum(np.square(keys0-keys1),axis=-1))
            pair_fmr=np.mean(dist<self.cfg.ok_match_dist_threshold) #ok ratio in one pair
            pair_fmrs.append(pair_fmr)                              
        pair_fmrs=np.array(pair_fmrs)                               #ok ratios in one scene
        FMR=np.mean(pair_fmrs>ratio)                                #FMR in one scene
        return FMR, pair_fmrs



    def eval(self):
        correct_ratios_wholeset={}
        Rtpres=[]
        Rtgts=[]
        datasets=get_dataset(self.cfg,False)
        
        FMRS=[]
        all_pair_fmrs=[]
        for scene,dataset in datasets.items():
            if scene=='wholesetname':continue
            self.run_onescene(dataset)
            print(f'eval the FMR result on {dataset.name}')
            FMR,pair_fmrs=self.Feature_match_Recall(dataset,ratio=self.cfg.fmr_ratio)
            FMRS.append(FMR)
            all_pair_fmrs.append(pair_fmrs)
        FMRS=np.array(FMRS)
        all_pair_fmrs=np.concatenate(all_pair_fmrs,axis=0)


        #RR
        datasetname=datasets['wholesetname']
        Mean_Registration_Recall,c_flags,c_errors=RR_cal.benchmark(self.cfg,datasets,self.max_iter,yoho_sign='YOHO_O')
        #print and save:
        msg=f'{datasetname}-{self.cfg.descriptor}-{self.cfg.extractor}-{self.cfg.matcher}-{self.cfg.estimator}-{self.max_iter}iterations\n'
        msg+=f'correct ratio avg {np.mean(all_pair_fmrs):.5f}\n' \
             f'correct ratio>0.05 avg {np.mean(FMRS):.5f}  std {np.std(FMRS):.5f}\n' \
             f'Mean_Registration_Recall {Mean_Registration_Recall}\n'

        with open('data/results.log','a') as f:
            f.write(msg+'\n')
        print(msg)


name2evaluator={
    'PartI':Evaluator_PartI,
    'PartII':Evaluator_PartII
}
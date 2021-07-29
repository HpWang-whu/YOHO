import os
import numpy as np
import argparse
import open3d as o3d
import torch
import random
from tqdm import tqdm
from utils.r_eval import compute_R_diff,quaternion_from_matrix
from utils.dataset import get_dataset_name
from utils.utils import make_non_exists_dir,random_rotation_matrix,read_pickle,save_pickle
from utils.misc import extract_features
from fcgf_model import load_model


class trainset_create():
    def __init__(self):
        self.dataset_name='3dmatch_train'
        self.origin_data_dir='./data/origin_data'
        self.datasets=get_dataset_name(self.dataset_name,self.origin_data_dir)
        self.output_dir='./data/YOHO_FCGF'
        self.Rgroup=np.load('./group_related/Rotation.npy')

    def PCA_keys_sample(self):
        for name,dataset in tqdm(self.datasets.items()):
            if name=='wholesetname':continue

            Save_keys_dir=f'{self.output_dir}/Filtered_Keys/{dataset.name}'
            Save_pair_dir=f'{self.output_dir}/Pairs_0.03/{dataset.name}'
            make_non_exists_dir(Save_keys_dir)
            make_non_exists_dir(Save_pair_dir)

            for pc_id in tqdm(dataset.pc_ids): #index in pc
                if os.path.exists(f'{Save_keys_dir}/{pc_id}_index.npy'):continue
                Keys_index=np.loadtxt(dataset.get_key_dir(pc_id)).astype(np.int)
                Keys=dataset.get_kps(pc_id)
                Pcas=np.load(f'{dataset.root}/pca_0.3/{pc_id}.npy')
                Ok_index=np.arange(Pcas.shape[0])[Pcas[:,0]>0.03].astype(np.int)
                Keys=Keys[Ok_index]
                Keys_index=Keys_index[Ok_index]
                #Save the filtered index
                np.save(f'{Save_keys_dir}/{pc_id}_coor.npy',Keys)
                np.save(f'{Save_keys_dir}/{pc_id}_index.npy',Keys_index) #in pc
            
            #pair with the filtered keypoints: index in keys
            for pair in tqdm(dataset.pair_ids):
                pc0,pc1=pair
                if os.path.exists(f'{Save_pair_dir}/{pc0}-{pc1}.npy'):continue
                keys0=torch.from_numpy(np.load(f'{Save_keys_dir}/{pc0}_coor.npy').astype(np.float32)).cuda()
                keys1=torch.from_numpy(np.load(f'{Save_keys_dir}/{pc1}_coor.npy').astype(np.float32)).cuda()
                diff=torch.norm(keys0[:,None,:]-keys1[None,:,:],dim=-1).cpu().numpy()
                pair=np.where(diff<0.02)
                pair=np.concatenate([pair[0][:,None],pair[1][:,None]],axis=1)# pairnum*2
                np.save(f'{Save_pair_dir}/{pc0}-{pc1}.npy',pair)

    
    def FCGF_Group_Feature_Extractor(self,args,Point,Keys_index): #index in pc
        #output:kn*32*60
        output=[]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(args.model)
        config = checkpoint['config']

        num_feats = 1
        Model = load_model(config.model)
        model = Model(
            num_feats,
            config.model_n_out,
            bn_momentum=0.05,
            normalize_feature=config.normalize_feature,
            conv1_kernel_size=config.conv1_kernel_size,
            D=3)
            
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        model = model.to(device)
        
        for i in range(self.Rgroup.shape[0]):
            one_R_output=[]
            R_i=self.Rgroup[i]
            Point_i=Point@R_i.T
            Keys_i=Point_i[Keys_index]
            with torch.no_grad():
                xyz_down, feature = extract_features(
                                    model,
                                    xyz=Point_i,
                                    voxel_size=config.voxel_size,
                                    device=device,
                                    skip_check=True)
            feature=feature.cpu().numpy()
            xyz_down_pcd = o3d.geometry.PointCloud()
            xyz_down_pcd.points = o3d.utility.Vector3dVector(xyz_down)
            pcd_tree = o3d.geometry.KDTreeFlann(xyz_down_pcd)
            for k in range(Keys_i.shape[0]):
                [_, idx, _] = pcd_tree.search_knn_vector_3d(Keys_i[k], 1)
                one_R_output.append(feature[idx[0]][None,:])
            one_R_output=np.concatenate(one_R_output,axis=0)#kn*32
            output.append(one_R_output[:,:,None])
        return np.concatenate(output,axis=-1) #kn*32*60


    def PC_random_rot_feat(self,args):
        for key,dataset in tqdm(self.datasets.items()):
            if key=="wholesetname":continue
            for pc_id in tqdm(dataset.pc_ids):
                Feats_save_dir=f'{self.output_dir}/Rotated_Features/{dataset.name}'
                make_non_exists_dir(Feats_save_dir)
                if os.path.exists(f'{Feats_save_dir}/{pc_id}_feats.npz'):continue
                Random_Rs=[]
                Feats=[]

                PC=dataset.get_pc(pc_id)
                Key_idx=np.load(f'{self.output_dir}/Filtered_Keys/{dataset.name}/{pc_id}_index.npy')
                
                for R_i in range(5):
                    R_one=random_rotation_matrix()
                    Random_Rs.append(R_one[None,:,:])
                Random_Rs=np.concatenate(Random_Rs,axis=0)# 5*3*3

                for R_i in range(5):
                    PC_one=PC@Random_Rs[R_i].T
                    feat_one=self.FCGF_Group_Feature_Extractor(args,PC_one,Key_idx) #kn*32*60
                    Feats.append(feat_one[None,:,:,:])
                Feats=np.concatenate(Feats,axis=0)#5*kn*32*60

                np.save(f'{Feats_save_dir}/{pc_id}_Rs.npy',Random_Rs)
                np.savez(f'{Feats_save_dir}/{pc_id}_feats.npz',Rs=Random_Rs,feats=Feats)
    

    def Train_val_list_creation(self):
        val_pc_pair=[]
        train_pc_pair=[]
        val_pp_list=[]
        train_pc_pair_only=[]
        for key,dataset in tqdm(self.datasets.items()):
            if key=="wholesetname":continue
            for pair in dataset.pair_ids:
                pc0,pc1=pair
                Key_pps=np.load(f'{self.output_dir}/Pairs_0.03/{dataset.name}/{pc0}-{pc1}.npy') #index in keys
                if Key_pps.shape[0]<64:
                    val_pc_pair.append(tuple([dataset.name,pc0,pc1]))
                    for pp_index in range(Key_pps.shape[0]):
                        #need the R_indexs
                        for Ri_id in range(5):
                            for Rj_id in range(5):
                                val_pp_list.append(tuple([dataset.name,pc0,pc1,Ri_id,Rj_id,Key_pps[pp_index,0],Key_pps[pp_index,1]])) #datasetname,pc0,pc1,key_m0,key_m1
                else: #train list contains the pairs with more than 64 pps
                    train_pc_pair_only.append(tuple([dataset.name,pc0,pc1]))
                    BaseIndex=np.arange(5).astype(np.int)
                    Index_i=np.random.choice(BaseIndex, size=3, replace=False)
                    Index_j=np.random.choice(BaseIndex, size=3, replace=False)
                    for Ri_id in range(Index_i.shape[0]):
                        for Rj_id in range(Index_j.shape[0]):
                            train_pc_pair.append(tuple([dataset.name,pc0,pc1,Index_i[Ri_id],Index_j[Rj_id]]))
        random.shuffle(val_pp_list)
        val_pp_list=val_pp_list[0:20000]

        Save_list_dir=f'{self.output_dir}/Train_val_list'
        make_non_exists_dir(Save_list_dir)
        random.shuffle(train_pc_pair)
        save_pickle(train_pc_pair,f'{Save_list_dir}/train_pc.pkl')
        save_pickle(train_pc_pair_only,f'{Save_list_dir}/train_pc_only.pkl')
        save_pickle(val_pc_pair,f'{Save_list_dir}/val_pc.pkl')
        save_pickle(val_pp_list,f'{Save_list_dir}/val_pc_pt.pkl') #have been shuffled



    def R2DR_id(self,R):
        min_diff=180
        best_id=0
        for R_id in range(self.Rgroup.shape[0]):
            R_diff=compute_R_diff(self.Rgroup[R_id],R)
            if R_diff<min_diff:
                min_diff=R_diff
                best_id=R_id
        return best_id


    def DeltaR(self,R,index):
        R_anchor=self.Rgroup[index]#3*3
        #R=Rres@Ranc->Rres=R@Ranc.T
        deltaR=R@R_anchor.T
        return quaternion_from_matrix(deltaR)


    def trainset(self):
        Save_list_dir=f'{self.output_dir}/Train_val_list/trainset'
        make_non_exists_dir(Save_list_dir)
        former_list=read_pickle(f'{self.output_dir}/Train_val_list/train_pc_only.pkl')
        batch_i=-1
        trainlist_pair=[]
        for i in tqdm(range(len(former_list))):
            #information
            #if os.path.exists(f'{Save_list_dir}/{i*16)}.pth'):continue
            datasetname,pc0,pc1=former_list[i]
            #feature readin
            Feats0=np.load(f'{self.output_dir}/Rotated_Features/{datasetname}/{pc0}_feats.npz')
            Feats1=np.load(f'{self.output_dir}/Rotated_Features/{datasetname}/{pc1}_feats.npz')
            Feats0_f=Feats0['feats']
            Feats1_f=Feats1['feats']
            Feats0_R=Feats0['Rs']
            Feats1_R=Feats1['Rs']
            AllRs=[]
            AllR_indexs=[]
            AlldeltaRs=[]
            for Ri_id in range(Feats0_R.shape[0]):
                for Rj_id in range(Feats1_R.shape[0]):
                    R_i=Feats0_R[Ri_id]
                    R_j=Feats1_R[Rj_id]
                    R=R_j@R_i.T
                    true_idx=self.R2DR_id(R)
                    delR=self.DeltaR(R,true_idx)
                    AllRs.append(R[None,:,:])
                    AllR_indexs.append(true_idx)
                    AlldeltaRs.append(delR[None,:])
            AllRs=np.concatenate(AllRs,axis=0).reshape([5,5,3,3])
            AllR_indexs=np.array(AllR_indexs).reshape([5,5])
            AlldeltaRs=np.concatenate(AlldeltaRs,axis=0).reshape([5,5,4])

            #pps
            Key_pps=np.load(f'{self.output_dir}/Pairs_0.03/{datasetname}/{pc0}-{pc1}.npy') #index in keys
            pps_all=np.arange(Key_pps.shape[0]) #index
            for i in range(10):
                #pair pps (choose 32):
                np.random.shuffle(pps_all)
                pps=Key_pps[pps_all[0:32]]# bn*2
                
                BaseIndex=np.arange(5).astype(np.int)
                Index_i=np.random.choice(BaseIndex, size=32, replace=True)
                Index_j=np.random.choice(BaseIndex, size=32, replace=True)
                
                Rs=[]
                R_indexs=[]
                deltaR=[]
                feats_one_batch_i=[]
                feats_one_batch_j=[]
                for b in range(32):
                    Rs.append(AllRs[Index_i[b],Index_j[b]][None,:,:])
                    R_indexs.append(AllR_indexs[Index_i[b],Index_j[b]])
                    deltaR.append(AlldeltaRs[Index_i[b],Index_j[b]][None,:])
                    #feat
                    feats_one_batch_i.append(Feats0_f[Index_i[b],pps[b,0]][None,:,:])
                    feats_one_batch_j.append(Feats1_f[Index_j[b],pps[b,1]][None,:,:])
                Rs=np.concatenate(Rs,axis=0)
                R_indexs=np.array(R_indexs)
                deltaR=np.concatenate(deltaR,axis=0)
                feats_one_batch_i=np.concatenate(feats_one_batch_i,axis=0)
                feats_one_batch_j=np.concatenate(feats_one_batch_j,axis=0)

                item={
                    'feats0':torch.from_numpy(feats_one_batch_i.astype(np.float32)), #before enhanced rot
                    'feats1':torch.from_numpy(feats_one_batch_j.astype(np.float32)), #after enhanced rot
                    'R':torch.from_numpy(Rs.astype(np.float32)),
                    'true_idx':torch.from_numpy(R_indexs.astype(np.int)),
                    'deltaR':torch.from_numpy(deltaR.astype(np.float32))
                }
                batch_i+=1
                torch.save(item,f'{Save_list_dir}/{batch_i}.pth',_use_new_zipfile_serialization=False)
                trainlist_pair.append(batch_i)
        save_pickle(trainlist_pair,f'{self.output_dir}/Train_val_list/train.pkl')


    def valset(self):
        Save_list_dir=f'{self.output_dir}/Train_val_list/valset'
        make_non_exists_dir(Save_list_dir)
        Vallist=read_pickle(f'{self.output_dir}/Train_val_list/val_pc_pt.pkl')
        for i in tqdm(range(len(Vallist))):
            datasetname,pc0,pc1,Ri_id,Rj_id,pt0,pt1=Vallist[i]
            Feats0=np.load(f'{self.output_dir}/Rotated_Features/{datasetname}/{pc0}_feats.npz')
            Feats1=np.load(f'{self.output_dir}/Rotated_Features/{datasetname}/{pc1}_feats.npz')
            R_i=Feats0['Rs'][Ri_id]
            R_j=Feats1['Rs'][Rj_id]
            R=R_j@R_i.T
            true_idx=self.R2DR_id(R)
            feat0=Feats0['feats'][Ri_id,pt0]
            feat1=Feats1['feats'][Rj_id,pt1]
            item={
                'feats0':torch.from_numpy(feat0.astype(np.float32)), #before enhanced rot 32*60
                'feats1':torch.from_numpy(feat1.astype(np.float32)), #after enhanced rot 32*60
                'R':torch.from_numpy(R.astype(np.float32)),
                'true_idx':torch.from_numpy(np.array([true_idx]))
            }
            torch.save(item,f'{Save_list_dir}/{i}.pth',_use_new_zipfile_serialization=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--model',
        default='./model/Backbone/best_val_checkpoint.pth',
        type=str,
        help='path to latest checkpoint (default: None)')
    parser.add_argument(
        '--voxel_size',
        default=0.025,
        type=float,
        help='voxel size to preprocess point cloud')
    args = parser.parse_args()

    trainset_creater=trainset_create()
    trainset_creater.PCA_keys_sample()
    trainset_creater.PC_random_rot_feat(args)
    trainset_creater.Train_val_list_creation()
    trainset_creater.trainset()
    trainset_creater.valset()
    
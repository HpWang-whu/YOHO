import os
import time
import numpy as np
import argparse
import open3d as o3d
import torch
from tqdm import tqdm
from utils.dataset import get_dataset_name
from utils.utils import make_non_exists_dir
from fcgf_model import load_model
from knn_search import knn_module
import MinkowskiEngine as ME
from utils.pointcloud import make_open3d_point_cloud



class FCGFDataset():
    def __init__(self,datasets,config):
        self.points={}
        self.pointlist=[]
        self.voxel_size = config.voxel_size
        self.datasets=datasets
        self.Rgroup=np.load('./group_related/Rotation.npy')
        for scene,dataset in self.datasets.items():
            if scene=='wholesetname':continue
            for pc_id in dataset.pc_ids:
                for g_id in range(60):
                    self.pointlist.append((scene,pc_id,g_id))
                self.points[f'{scene}_{pc_id}']=self.datasets[scene].get_pc(pc_id)


    def __getitem__(self, idx):
        scene,pc_id,g_id=self.pointlist[idx]
        xyz0 = self.points[f'{scene}_{pc_id}']
        xyz0=xyz0@self.Rgroup[g_id].T
        # Voxelization
        _, sel0 = ME.utils.sparse_quantize(xyz0 / self.voxel_size, return_index=True)
        # Make point clouds using voxelized points
        pcd0 = make_open3d_point_cloud(xyz0)
        # Select features and points using the returned voxelized indices
        pcd0.points = o3d.utility.Vector3dVector(np.array(pcd0.points)[sel0])
        # Get coords
        xyz0 = np.array(pcd0.points)
        feats=np.ones((xyz0.shape[0], 1))
        coords0 = np.floor(xyz0 / self.voxel_size)
        
        return (xyz0, coords0, feats ,self.pointlist[idx])
    
    def __len__(self):
        return len(self.pointlist)




class testset_create():
    def __init__(self,config):
        self.config=config
        self.dataset_name=self.config.dataset
        self.output_dir='./data/YOHO_FCGF'
        self.origin_dir='./data/origin_data'
        self.datasets=get_dataset_name(self.dataset_name,self.origin_dir)
        self.Rgroup=np.load('./group_related/Rotation.npy')
        self.knn=knn_module.KNN(1)


    def collate_fn(self,list_data):
        xyz0, coords0, feats0, scenepc = list(
            zip(*list_data))
        xyz_batch0 = []
        dsxyz_batch0=[]
        batch_id = 0
        def to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x
            elif isinstance(x, np.ndarray):
                return torch.from_numpy(x)
            else:
                raise ValueError(f'Can not convert to torch tensor, {x}')
        
        
        for batch_id, _ in enumerate(coords0):
            xyz_batch0.append(to_tensor(xyz0[batch_id]))
            _, inds = ME.utils.sparse_quantize(coords0[batch_id], return_index=True)
            dsxyz_batch0.append(to_tensor(xyz0[batch_id][inds]))

        coords_batch0, feats_batch0 = ME.utils.sparse_collate(coords0, feats0)

        # Concatenate all lists
        xyz_batch0 = torch.cat(xyz_batch0, 0).float()
        dsxyz_batch0=torch.cat(dsxyz_batch0, 0).float()
        cuts_node=0
        cuts=[0]
        for batch_id, _ in enumerate(coords0):
            cuts_node+=coords0[batch_id].shape[0]
            cuts.append(cuts_node)

        return {
            'pcd0': xyz_batch0,
            'dspcd0':dsxyz_batch0,
            'scenepc':scenepc,
            'cuts':cuts,
            'sinput0_C': coords_batch0,
            'sinput0_F': feats_batch0.float(),
        }

    def Feature_extracting(self, data_loader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(self.config.model)
        config = checkpoint['config']
        features={}
        features_gid={}
        for scene,dataset in self.datasets.items():
            if scene=='wholesetname':continue
            for pc_id in dataset.pc_ids:
                features[f'{scene}_{pc_id}']=[]
                features_gid[f'{scene}_{pc_id}']=[]

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
        model = model.to(device)
        model.eval()
         
        with torch.no_grad():
            for i, input_dict in enumerate(tqdm(data_loader)):
                starttime=time.time()
                sinput0 = ME.SparseTensor(
                        input_dict['sinput0_F'].to(device),
                        coordinates=input_dict['sinput0_C'].to(device))
                torch.cuda.synchronize()
                F0 = model(sinput0).F
            
                cuts=input_dict['cuts']
                scene_pc=input_dict['scenepc']
                for inb in range(len(scene_pc)):
                    scene,pc_id,g_id=scene_pc[inb]
                    make_non_exists_dir(f'{self.output_dir}/Testset/{self.dataset_name}/{scene}/FCGF_Input_Group_feature')
                    feature=F0[cuts[inb]:cuts[inb+1]].cpu().numpy()
                    pts=input_dict['dspcd0'][cuts[inb]:cuts[inb+1]].numpy()#*config.voxel_size

                    Keys=self.datasets[scene].get_kps(pc_id)
                    Keys=Keys@self.Rgroup[g_id].T
                    Keys_i=torch.from_numpy(np.transpose(Keys)[None,:,:]).cuda() #1,3,k
                    xyz_down=torch.from_numpy(np.transpose(pts)[None,:,:]).cuda() #1,3,n
                    d,nnindex=self.knn(xyz_down,Keys_i)
                    nnindex=nnindex[0,0].cpu().numpy()
                    one_R_output=feature[nnindex,:]#5000*32
                    features[f'{scene}_{pc_id}'].append(one_R_output[:,:,None])
                    features_gid[f'{scene}_{pc_id}'].append(g_id)
                    if len(features_gid[f'{scene}_{pc_id}'])==60:
                        sort_args=np.array(features_gid[f'{scene}_{pc_id}'])
                        sort_args=np.argsort(sort_args)
                        output=np.concatenate(features[f'{scene}_{pc_id}'],axis=-1)[:,:,sort_args]
                        np.save(f'{self.output_dir}/Testset/{self.dataset_name}/{scene}/FCGF_Input_Group_feature/{pc_id}.npy',output)
                        features[f'{scene}_{pc_id}']=[]


    def batch_feature_extraction(self):
        dset=FCGFDataset(self.datasets,self.config)
        loader = torch.utils.data.DataLoader(
            dset,
            batch_size=4, #6 is timely better(but out of memory easily)
            shuffle=False,
            num_workers=10,
            collate_fn=self.collate_fn,
            pin_memory=False,
            drop_last=False)
        self.Feature_extracting(loader)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--model',
        default='./model/Backbone/best_val_checkpoint.pth',
        type=str,
        help='path to backbone latest checkpoint (default: None)')
    parser.add_argument(
        '--voxel_size',
        default=0.025,
        type=float,
        help='voxel size to preprocess point cloud')
    parser.add_argument(
        '--dataset',
        default='demo',
        type=str,
        help='datasetname')
    args = parser.parse_args()

    testset_creater=testset_create(args)
    testset_creater.batch_feature_extraction()
    
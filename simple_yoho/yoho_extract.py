import torch
import argparse
import numpy as np
from copy import deepcopy
from utils.network import PartI_test
from utils.utils import transform_points
from utils.knn_search import modified_knn_matcher
from simple_yoho.fcgf_feat import fcgf_extractor

# fake config
parser = argparse.ArgumentParser()
parser.add_argument('--SO3_related_files',default='./group_related',type=str)
args = parser.parse_args()
    
class yoho_extractor():
    def __init__(self,
                 fcgf_ckpt = 'model/Backbone/best_val_checkpoint.pth',
                 yoho_ckpt = 'model/PartI_train/model_best.pth'):
        # basic
        self.grs = np.load('group_related/Rotation.npy')
        self.fcgf = fcgf_extractor(pth=fcgf_ckpt)
        self.yoho_ckpt = yoho_ckpt
        self.network = PartI_test(args)
        self._load_model()
        self.nn_searcher = modified_knn_matcher()
        self.bs = 500

    def _load_model(self):
        pth = torch.load(self.yoho_ckpt)['network_state_dict']
        self.network.load_state_dict(pth,strict=True)
        self.network.cuda().eval()
        
    def _feature_transfer_xyz(self, query, source, source_f):
        # nn
        query = torch.from_numpy(query.astype(np.float32)).cuda()
        source = torch.from_numpy(source.astype(np.float32)).cuda()
        idx,dist = self.nn_searcher.find_nn_gpu(query, source, nn_max_n=1000, return_distance=True, dist_type='SquareL2')
        qf = source_f[idx]
        return qf

    def extract_features(self,pc,voxel_size,nkpts=5000):
        kpts_index = np.random.permutation(len(pc))[0:nkpts]
        kpts = pc[kpts_index]
        kpts_f = []
        # pc rotation + feature extraction
        for i in range(self.grs.shape[0]):
            kptsi = transform_points(deepcopy(kpts),self.grs[i])
            pci = transform_points(deepcopy(pc),self.grs[i])
            # fcgf extraction
            pci_ds,pci_f = self.fcgf.run(pci,voxel_size)
            kptsi_f = self._feature_transfer_xyz(kptsi,pci_ds,pci_f)
            kpts_f.append(kptsi_f[:,:,None])
        kpts_f = torch.concat(kpts_f,dim=-1).cuda()
        yoho_inv,yoho_eqv = [],[]
        nbatch,nlast = len(kpts_f)//self.bs, len(kpts_f)%self.bs
        with torch.no_grad():
            if nlast<2:
                nbatch -= 1
            for i in range(nbatch):
                batch = kpts_f[self.bs*i:self.bs*(i+1)]
                output = self.network(batch)
                yoho_inv.append(output['inv'])
                yoho_eqv.append(output['eqv'])
            batch = kpts_f[self.bs*nbatch:]
            output = self.network(batch)
            yoho_inv.append(output['inv'])       
            yoho_eqv.append(output['eqv'])       
            # output: 5000*32; 5000*32*60
            yoho_inv = torch.cat(yoho_inv,dim=0).cpu()
            yoho_eqv = torch.cat(yoho_eqv,dim=0).cpu()
        return kpts,yoho_inv,yoho_eqv

    def run(self, pc, voxel_size = 0.025, nkpts=5000):
        # get features. inds is the indexes in the input pc (indexes of down-sampled keypoints)
        kpts, feat_inv, feat_eqv = self.extract_features(pc, voxel_size, nkpts=nkpts)
        # feat l2 normalization
        return kpts, feat_inv, feat_eqv

if __name__ == '__main__':
    import open3d as o3d
    extractor = yoho_extractor()
    gt = np.loadtxt('data/origin_data/demo/kitchen/PointCloud/gt.txt')
    spc = o3d.io.read_point_cloud(f'data/origin_data/demo/kitchen/PointCloud/cloud_bin_0.ply')
    tpc = o3d.io.read_point_cloud(f'data/origin_data/demo/kitchen/PointCloud/cloud_bin_1.ply')
    spc = np.array(spc.points)
    tpc = np.array(tpc.points)
    skpts, sfeat_inv, sfeat_eqv = extractor.run(spc,voxel_size=0.025)
    tkpts, tfeat_inv, tfeat_eqv = extractor.run(tpc,voxel_size=0.025)
    # match
    matcher = modified_knn_matcher()
    sid,tid = matcher.find_corr(sfeat_inv.cuda(),tfeat_inv.cuda(),mutual=True)
    mskpts,mtkpts = skpts[sid],tkpts[tid]
    mtkpts = transform_points(mtkpts,gt)
    disp = np.sqrt(np.sum(np.square(mskpts-mtkpts),axis=-1))
    disp = disp<0.1
    print('Inlier ratio: ',np.mean(disp))
    print('Inlier number: ',np.sum(disp))
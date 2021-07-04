import os
import copy
import numpy as np
import open3d as o3d
from utils.utils_o3d import draw_registration_result
import utils.dataset as Dataset
from utils.utils import transform_points,evaluate_the_match
import parses.Demo_c_parses as parses_C
import parses.Demo_o_parses as parses_O
#extractor
from tests.extractor import name2extractor,extractor_dr_index
#matcher
from tests.matcher import name2matcher
#RANSACer
from tests.estimator import name2estimator


config_c, unparsed_arguments = parses_C.get_config()
config_o, unparsed_arguments = parses_O.get_config()

datasets=Dataset.get_dataset(config_c,training=False)


for name,dataset in datasets.items():
    if name=='wholesetname':
        continue

    #PartI descriptor
    des_extractor=name2extractor[config_c.extractor](config_c)
    des_extractor.Extract(dataset)

    #Matcher
    matcher=name2matcher[config_c.matcher](config_c)
    matcher.match(dataset)

    #FMR
    keys0=dataset.get_kps('0')
    keys1=dataset.get_kps('1')
    match_dir=f'{config_c.output_cache_fn}/Testset/{dataset.name}/Match'
    matches=np.load(f'{match_dir}/0-1.npy')
    print(matches.shape)
    print('Feature matching rate:',evaluate_the_match(keys0,keys1,matches,dataset.get_transform('0','1'),config_c.ok_match_dist_threshold))

    #Dr index extraction
    drindex_extractor=extractor_dr_index(config_c)
    drindex_extractor.PartI_Rindex(dataset)

    #PartII rotation extraction
    extractor_II=name2extractor[config_o.extractor](config_o)
    extractor_II.PartII_R_pre(dataset)

    #YOMO-C
    estimator_c=name2estimator[config_c.estimator](config_c)
    estimator_c.ransac(dataset,max_iter=1000,TR_max_iter=1000)

    #YOMO-O
    estimator_o=name2estimator[config_o.estimator](config_o)
    estimator_o.ransac(dataset,max_iter=1000,TR_max_iter=1000)

    yomoc_result=np.load(f'{match_dir}/YOMO_C/1000iters/0-1.npz')
    yomoo_result=np.load(f'{match_dir}/YOMO_O/1000iters/0-1.npz')
    yomoc_trans=np.concatenate([yomoc_result['trans'],np.array([[0,0,0,1]])],axis=0)
    yomoo_trans=np.concatenate([yomoo_result['trans'],np.array([[0,0,0,1]])],axis=0)
    
    yomoc_iters=np.load(f'{match_dir}/YOMO_C_TR/1000iters/0-1.npz')['recalltime']
    yomoo_iters=np.load(f'{match_dir}/YOMO_O_TR/1000iters/0-1.npz')['recalltime']

    #visual
    target=dataset.get_pc_o3d('0')
    source=dataset.get_pc_o3d('1')
    draw_registration_result(source,target,np.eye(4))

    #yomoc
    print(f'YOMO-C get the true (Rerror<{config_c.TR_rerror}, terror<{config_c.TR_terror}) result with ',yomoc_iters,' iterations.')
    draw_registration_result(source,target,yomoc_trans)

    #yomoo
    print(f'YOMO-O get the true (Rerror<{config_o.TR_rerror}, terror<{config_o.TR_terror}) result with ',yomoo_iters,' iterations.')
    draw_registration_result(source,target,yomoo_trans)

# Import python dependencies
import argparse


base_dir='./data'
arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

Dirs=add_argument_group('Dirs')
Dataset_Args=add_argument_group('Dataset')
Network_Args=add_argument_group("Network_Args")
Test_Args=add_argument_group("Test_Args")

############################################# Base ###################################################
#Dirs
Dirs.add_argument('--base_dir',type=str,default=base_dir,
                        help="base dir containing the whole project")
Dirs.add_argument("--origin_data_dir",type=str,default=f"{base_dir}/origin_data",
                        help="the dir containing whole datas")
Dirs.add_argument("--output_cache_fn",type=str,default=f"{base_dir}/YOMO_FCGF",
                        help="eval cache dir")
Dirs.add_argument("--model_fn",type=str,default=f"./model",
                        help='well trained model path')
Dirs.add_argument('--SO3_related_files',type=str,default=f"./group_related",
                        help='SO3 related files path')

############################################# Test_YOMO-C ###################################################
Pipeline=add_argument_group('Pipeline')
Pipeline.add_argument('--descriptor',type=str,default='YOMO',
                        help="name of extractor")
Pipeline.add_argument('--extractor',type=str,default='PartI',
                        help="name of extractor")
Pipeline.add_argument('--matcher',type=str,default='Match',
                        help="name of matcher")
Pipeline.add_argument('--estimator',type=str,default='yomoc',
                        help="name of estimator")
Pipeline.add_argument('--evaluator',type=str,default='PartI',
                        help="name of evaluator")
Network_Args.add_argument("--train_network_type",type=str,default="PartI_train",
                        help="network type for test")


Dataset_Args.add_argument("--testset_name",type=str,default='demo',
                        help="eval dataset name")
Network_Args.add_argument("--test_network_type",type=str,default="PartI_test",
                        help="network type for test")
Test_Args.add_argument("--test_batch_size",type=int,default=900,
                        help="TEST batch size")
Test_Args.add_argument("--worker_num",type=int,default=16,
                        help="the threads used for dataloader")

Test_Args.add_argument("--ransac_c_inlinerdist",type=float,default=0.07,
                        help="inliner distance threshold distance for YOMO-c")
Test_Args.add_argument("--TR_rerror",type=float,default=3,
                        help="The R error to determain the true transformation")
Test_Args.add_argument("--TR_terror",type=float,default=0.1,
                        help="The t error to determain the true transformation")       
Test_Args.add_argument("--ok_match_dist_threshold",type=float,default=0.1,
                        help="The t error to determain the true transformation")
Test_Args.add_argument("--RR_dist_threshold",type=float,default=0.2,
                        help="The t error to determain the true transformation") 

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

def print_usage():
    parser.print_usage()



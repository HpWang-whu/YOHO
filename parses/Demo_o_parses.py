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
Train_Args=add_argument_group("Training_Args")
Val_Args=add_argument_group("Validation_Args")
Network_Args=add_argument_group("Network_Args")
Test_Args=add_argument_group("Test_Args")

############################################# Base ###################################################
#Dirs
Dirs.add_argument('--base_dir',type=str,default=base_dir,
                        help="base dir containing the whole project")
Dirs.add_argument("--origin_data_dir",type=str,default=f"{base_dir}",
                        help="the dir containing whole datas")
Dirs.add_argument("--output_cache_fn",type=str,default=f"{base_dir}/YOHO_FCGF",
                        help="eval cache dir")
Dirs.add_argument("--model_fn",type=str,default=f"./model",
                        help='well trained model path')
Dirs.add_argument('--SO3_related_files',type=str,default=f"./group_related",
                        help='SO3 related files path')

Network_Args.add_argument("--train_network_type",type=str,default="PartII_train",
                        help="network type for train")

############################################# Test_YOHO-O ###################################################
Pipeline=add_argument_group('Pipeline')
Pipeline.add_argument('--descriptor',type=str,default='YOHO',
                        help="name of extractor")
Pipeline.add_argument('--extractor',type=str,default='PartII',
                        help="name of extractor")
Pipeline.add_argument('--matcher',type=str,default='Match',
                        help="name of matcher")
Pipeline.add_argument('--estimator',type=str,default='yohoo',
                        help="name of estimator")
Pipeline.add_argument('--evaluator',type=str,default='PartII',
                        help="name of evaluator")

Dataset_Args.add_argument("--testset_name",type=str,default='demo',
                        help="eval dataset name")
Network_Args.add_argument("--test_network_type",type=str,default="PartII_test",
                        help="network type for test")
Test_Args.add_argument("--test_batch_size",type=int,default=1000,
                        help="TEST batch size")
Test_Args.add_argument("--worker_num",type=int,default=16,
                        help="the threads used for dataloader")

Test_Args.add_argument("--ransac_o_inlinerdist",type=float,default=0.07,
                        help="inliner distance threshold distance for YOHO-o")
Test_Args.add_argument("--RR_dist_threshold",type=float,default=0.2,
                        help="The t error to determain the true transformation") 

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

def print_usage():
    parser.print_usage()



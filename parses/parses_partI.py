# Import python dependencies
import argparse


base_dir='.'
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
Dirs.add_argument("--origin_data_dir",type=str,default=f"{base_dir}/data/origin_data",
                        help="the dir containing whole datas")
Dirs.add_argument("--output_cache_fn",type=str,default=f"{base_dir}/data/YOHO_FCGF",
                        help="eval cache dir")
Dirs.add_argument("--model_fn",type=str,default=f"./model",
                        help='well trained model path')
Dirs.add_argument('--SO3_related_files',type=str,default=f"./group_related",
                        help='SO3 related files path')

############################################# Train ###################################################
#Train Args
Dataset_Args.add_argument("--trainset_name",type=str,default="3dmatch_train",
                        help="train dataset name")
Dataset_Args.add_argument("--trainset_type",type=str,default="Enhanced_train_dataset_PartI",
                        help="train dataset type")
Dataset_Args.add_argument("--train_pcpair_list_fn",type=str,default=f'{base_dir}/data/YOHO_FCGF/Train_val_list/train.pkl',
                        help="Training tuples (station,pc0,pc1,pt0,pt1)")
Network_Args.add_argument("--train_network_type",type=str,default="PartI_train",
                        help="network type for train")
Train_Args.add_argument('--trainer',type=str,default='PartI',
                        help="name of trainer")
Train_Args.add_argument("--val_type",type=str,default="Val_partI",
                        help="val_type")
Train_Args.add_argument("--loss_type",type=str,default="Batch_hard_Rindex_loss",
                        help="loss type")                        
Train_Args.add_argument("--epochs",type=int,default=10,
                        help="num of epoches")
Train_Args.add_argument("--worker_num",type=int,default=16,
                        help="the threads used for dataloader")
Train_Args.add_argument("--batch_size",type=int,default=32,
                        help="Training batch size")
Train_Args.add_argument("--multi_gpus",type=bool,default=False,
                        help="whether use the mutli gpus")
Train_Args.add_argument("--lr_init",type=float,default=0.0001,
                        help="The initial learning rate")
Train_Args.add_argument("--lr_decay_rate",type=float,default=0.5,
                        help="the decay rate of the learning rate per epoch")
Train_Args.add_argument("--lr_decay_step",type=float,default=1.8,
                        help="the decay step of the learning rate (how many epoches)")
Train_Args.add_argument("--train_log_step",type=int,default=500,
                        help="logger internal")

#Validation Args
Dataset_Args.add_argument("--val_pppair_list_fn",type=str,default=f'{base_dir}/data/YOHO_FCGF/Train_val_list/val_pc_pt.pkl',
                        help="validation tuples (station,pc0,pc1,R_i,R_j,pt0,pt1)")
Val_Args.add_argument("--val_interval",type=int,default=500,
                        help="the interval to validation")
Val_Args.add_argument("--save_interval",type=int,default=500,
                        help="the interval to save the model")

############################################# Test ###################################################
Pipeline=add_argument_group('Pipeline')
Pipeline.add_argument('--descriptor',type=str,default='YOHO',
                        help="name of descriptor")
Pipeline.add_argument('--extractor',type=str,default='PartI',
                        help="name of extractor")
Pipeline.add_argument('--matcher',type=str,default='Match',
                        help="name of matcher")
Pipeline.add_argument('--estimator',type=str,default='yohoc',
                        help="name of estimator")
Pipeline.add_argument('--evaluator',type=str,default='PartI',
                        help="name of evaluator")


Dataset_Args.add_argument("--testset_name",type=str,default='3dmatch',
                        help="eval dataset name")
Network_Args.add_argument("--test_network_type",type=str,default="PartI_test",
                        help="network type for test")
Test_Args.add_argument("--test_batch_size",type=int,default=900,
                        help="TEST batch size")
Test_Args.add_argument("--ransac_c_inlinerdist",type=float,default=0.07,
                        help="inliner distance threshold distance for YOHO-c")
Test_Args.add_argument("--fmr_ratio",type=float,default=0.05,
                        help="tau 1 for FMR") 
Test_Args.add_argument("--ok_match_dist_threshold",type=float,default=0.1,
                        help="tau 2 for FMR")
Test_Args.add_argument("--RR_dist_threshold",type=float,default=0.2,
                        help="tau 3 for RR") 



def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

def print_usage():
    parser.print_usage()



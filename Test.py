import argparse
import parses.parses_partI as parses_partI
import parses.parses_partII as parses_partII
from tests.evaluator import name2evaluator

parser = argparse.ArgumentParser()
parser.add_argument(
    '--Part',
    default='PartI',
    type=str,
    help='PartI or PartII')
parser.add_argument(
    '--max_iter',
    default=1000,
    type=int,
    help='ransac iterations')
parser.add_argument(
    '--dataset',
    default='3dmatch',
    type=str,
    help='dataset name')
parser.add_argument(
    '--ransac_d',
    default=-1,
    type=float,
    help='inliner threshold of ransac')
parser.add_argument(
    '--tau_1',
    default=0.05,
    type=float,
    help='tau 1 for FMR')
parser.add_argument(
    '--tau_2',
    default=0.1,
    type=float,
    help='tau 2 for FMR')
parser.add_argument(
    '--tau_3',
    default=0.2,
    type=float,
    help='tau 3 for RR')

args = parser.parse_args()

sign=args.Part
if sign=='PartI':
    config,nouse=parses_partI.get_config()
    config.fmr_ratio=args.tau_1
    config.ok_match_dist_threshold=args.tau_2
    config.RR_dist_threshold=args.tau_3
    if args.ransac_d>0:
        config.ransac_c_inlinerdist=args.ransac_d
    config.testset_name=args.dataset
    eval_net=name2evaluator[config.evaluator](config,max_iter=args.max_iter)
    eval_net.eval()
elif sign=='PartII':
    config,nouse=parses_partII.get_config()
    config.fmr_ratio=args.tau_1
    config.ok_match_dist_threshold=args.tau_2
    config.RR_dist_threshold=args.tau_3
    if args.ransac_d>0:
        config.ransac_o_inlinerdist=args.ransac_d
    config.testset_name=args.dataset
    eval_net=name2evaluator[config.evaluator](config,max_iter=args.max_iter)
    eval_net.eval()
else:
    print('wrong sign')
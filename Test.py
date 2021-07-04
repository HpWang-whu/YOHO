import sys
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
args = parser.parse_args()

sign=args.Part
if sign=='PartI':
    config,nouse=parses_partI.get_config()
    config.testset_name=args.dataset
    eval_net=name2evaluator[config.evaluator](config,max_iter=args.max_iter,TR_max_iter=1000)
    eval_net.eval()
elif sign=='PartII':
    config,nouse=parses_partII.get_config()
    config.testset_name=args.dataset
    eval_net=name2evaluator[config.evaluator](config,max_iter=args.max_iter,TR_max_iter=1000)
    eval_net.eval()
else:
    print('wrong sign')
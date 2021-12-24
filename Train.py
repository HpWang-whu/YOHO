import argparse
import parses.parses_partI as parses_partI
import parses.parses_partII as parses_partII
import train.trainer as trainer

parser = argparse.ArgumentParser()
parser.add_argument(
    '--Part',
    default='PartI',
    type=str,
    help='PartI or PartII')
args = parser.parse_args()

sign=args.Part
if sign=='PartI':
    config,nouse=parses_partI.get_config()
    train_net=trainer.Trainer_partI(config)
    train_net.run()
elif sign=='PartII':
    config,nouse=parses_partII.get_config()
    train_net=trainer.Trainer_partII(config)
    train_net.run()
else:
    print('wrong sign')
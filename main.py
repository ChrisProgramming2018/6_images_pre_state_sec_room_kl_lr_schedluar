import sys
import json
import gym
import argparse
from train_iql import train
from utils import write_into_file, mkdir, write_parameter


def main(args):
    """ """
    with open (args.param, "r") as f:
        param = json.load(f)

    print("use the env {} ".format(param["env_name"]))
    param["mode"] = args.mode
    param["run"] = args.run
    param["buffer_path"] = args.buffer_path
    print("Start Programm in {}  mode".format(args.mode))
    env = gym.make(param["env_name"])
    if args.mode == "hypersearch":
        param['file_path'] = str(args.locexp)
        param["lr_pre"] = args.lr_pre
        param["lr"] = args.lr
    param["fc1_units"] = args.fc1_units
    param["clip"] = args.clip
    param["eval"] = args.eval
    param["batch_size"] = args.batch_size
    param["locexp"] = str(args.locexp) +  "/run-{}".format(param["run"])
    mkdir("", str(param['locexp']))
    text = str(param)
    print(param)
    write_parameter(str(param['locexp']) + '/parameter', text)
    train(env, args, param)







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', default="param.json", type=str)
    parser.add_argument('--locexp', default="results", type=str)
    parser.add_argument('--buffer_path', default="expert_policy", type=str)
    parser.add_argument('--lr_pre', default=5e-4, type=float)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--fc1_units', default=512, type=int)
    parser.add_argument('--clip', default=-5, type=int)
    parser.add_argument('--mode', default="hypersearch", type=str) 
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--run', default=1, type=int)
    parser.add_argument('--size', default=84, type=int)
    parser.add_argument('--history_length', default=3, type=int)
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--limit_data', default=False, type=bool)
    parser.add_argument('--eval', default=500, type=int)
    arg = parser.parse_args()
    main(arg)

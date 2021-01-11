import os
import sys
import argparse



def main(args):
    """ try different values for epsilon and the decay
    and the two parameter from per alpha and beta
    Args:
        param1 (args) :
    """
    pathname = "param.json"
    counter = 1
    for fc1 in [64, 128]:
        for fc2 in [64, 128]:
            for fc3 in [64, 128]:
                for lr in [1e-4, 5e-4, 1e-3]:
                    for b in [32, 64, 128, 256]:
                        print("Round {}".format(counter))
                        os.system(f'python3 ./main.py \
                                --param {pathname} \
                                --fc1_units {fc1} \
                                --fc2_units {fc2} \
                                --fc3_units {fc3} \
                                --batch_size {b} \
                                --run {counter} \
                                --lr {lr}')
                        counter += 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameter search')
    main(parser.parse_args())

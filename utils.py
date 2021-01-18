# Copyright 2020
# Author: Christian Leininger <info2016frei@gmail.com>


import os
import numpy as np




def write_into_file(pathname, text):
    """
    """
    with open(pathname+".txt", "a") as myfile:
        myfile.write(text)
        myfile.write('\n')

def write_parameter(pathname, text):
    """
    """
    with open(pathname+".txt", "a") as myfile:
        for word in text.split(","):
            myfile.write(word)
            myfile.write('\n')


def mkdir(base, name):
    """
    Creates a direction if its not exist
    Args:
       param1(string): base first part of pathname
       param2(string): name second part of pathname
    Return: pathname
    """
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# Copyright 2021
# Author: Christian Leininger <info2016frei@gmail.com>
import os
import numpy as np


def time_format(sec):
    """
    Computes from the given int in sec
    the time format h, min, sec
    Args:
        param1(int): sec
    """
    hours = sec // 3600
    rem = sec - hours * 3600
    mins = rem // 60
    secs = rem - mins * 60
    return hours, mins, round(secs, 2)


def write_into_file(pathname, text):
    """

    Args:
       param1(string) pathname
       param2(string) text

    """
    with open(pathname+".txt", "a") as myfile:
        myfile.write(text)
        myfile.write('\n')


def write_parameter(pathname, text):
    """ Creates file with param
        from given text and writes in dir of
        pathname

    Args:
       param1(string) pathname
       param2(string) text
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

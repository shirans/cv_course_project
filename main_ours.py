import os
import torch
import torch.nn as nn
from load_data import load_data


def main():
    dirname = os.path.dirname(__file__)
    _path_to_data_dir = os.path.join(dirname, 'data')
    load_data.load(_path_to_data_dir)


    x = torch.rand(5, 3)
    print(x)


main()

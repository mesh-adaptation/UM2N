# Author: Chunyang Wang
# GitHub Username: acse-cw1722
import os

__all__ = ['mkdir_if_not_exist']


def mkdir_if_not_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        # delete all files under the directory
        filelist = [f for f in os.listdir(dir)]
        for f in filelist:
            os.remove(os.path.join(dir, f))

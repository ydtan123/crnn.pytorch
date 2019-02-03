#!/usr/bin/python3
import argparse
import cv2
import os
import lmdb # install lmdb by "pip install lmdb"
import math
import numpy as np
import pathlib

files = {}
for f in pathlib.Path("./trainset").glob("**/*.jpg"):
    files[f.stem] = 1

with open("trainset/labels.txt") as f:
    for l in f:
        s = l.split(' ')[0].split('_')[1]
        if (s not in files):
            print("Cannot find {} in labels".format(s))
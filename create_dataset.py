#!/usr/bin/python3
import argparse
import cv2
import os
import lmdb # install lmdb by "pip install lmdb"
import math
import numpy as np
import pathlib


def groupByY(letters):
    groups = []
    for l in letters:
        found = False
        for g in groups:
            if ((math.fabs(l[4] - g[-1][4]) < (l[4] - l[2]) / 2)
                and (math.fabs(l[3] - g[-1][1]) < (l[3] - l[1]) * 4)):
                g.append(l)
                found = True
        if (not found):
            groups.append([l])
    return groups

def getLocation(txtfile, img_file, data_out):
    #0 0.056711 0.629032 0.030246 0.204301
    symbols = []
    img = cv2.imread(str(img_file))
    imgh, imgw, _ = img.shape
    img_label_list = []
    with open(str(txtfile)) as f:
        for line in f:
            data = [d for d in line.split(' ')]
            if (len(data) < 5):
                print("Incorrect data {0} in {1}".format(line, txtfile))
                continue
            dval = int(data[0])
            if (dval < 0 or dval > 9):
                if (dval == 10):
                    print("Skip 10 in {0}".format(txtfile))
                else:
                    print("Invalid number {0} in {1}".format(data[0], txtfile))
                continue
            cx = float(data[1]) * imgw
            cy = float(data[2]) * imgh
            w = float(data[3]) * imgw / 2
            h = float(data[4]) * imgh / 2
            symbols.append((data[0], int(cx - w), int(cy - h), int(cx + w), int(cy + h)))
    if (len(symbols) == 0):
        return None
    letter_groups = groupByY(sorted(symbols, key=lambda x: x[1]))
    gidx = 0
    for g in letter_groups:
        for word_len in [1, 2, 3, 4]:
            for idx in range(len(g) - word_len):
                last_idx = idx + word_len
                sf = g[idx]
                sl = g[last_idx - 1]
                ty = min(sf[2], sl[2])
                tx = sf[1]
                by = max(sf[4], sl[4])
                bx = sl[3]
                img_name = 'img_{}_g{}_i{}_l{}.jpg'.format(img_file.stem, gidx, idx, word_len)
                cv2.imwrite(os.path.join(data_out, img_name), img[ty:by+1, tx:bx+1])
                img_label_list.append((img_name, [s[0] for s in g[idx : last_idx]]))
        gidx += 1
    return img_label_list

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.iteritems():
            txn.put(k, v)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in xrange(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'r') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    ## Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, help="Path to the input images", default='')
    parser.add_argument("-d", "--debug", action='store_true', help="Debug mode", default=0)
    parser.add_argument("-n", "--number", type=int, help="the number of training images", default=0)

    parser.add_argument("-a", "--data-root", type=str, help='dataset root')

    args = vars(parser.parse_args())
    if not os.path.exists(args['data_root']):
        os.makedirs(args['data_root'])

    file_dict = {}
    count = 0
    for f in pathlib.Path(args["image"]).glob("**/*.jpg"):
        if (args['debug']):
            print("processing {}".format(f))
        if (f in file_dict):
            print("{0} has more than one copy".format(f))
            continue
        file_dict[str(f)] = 1
        txtfile = f.with_suffix(".txt")
        if (not os.path.isfile(str(txtfile))):
            print("GT file for {0} does not exist".format(f))
            continue

        img_labels = getLocation(txtfile, f, args["data_root"])
        with open(os.path.join(args['data_root'], "labels.txt"), "a+") as f:
            for l in img_labels:
                f.write("{} {}\n".format(l[0], ''.join(l[1])))
        count += 1
        if (count > args['number']):
            break
        



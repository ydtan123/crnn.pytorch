#!/usr/bin/python3
import argparse
import math
import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np

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
            w = float(data[3]) * imgw
            h = float(data[4]) * imgh
            symbols.append((data[0], cx, cy, w/2, h/2))
    if (len(symbols) == 0):
        return None
    symbols_sorted = sorted(symbols, key=lambda x: x[1])
    words = []
    for words_len in [1, 2, 3, 4]:
        for idx in range(0, len(symbols_sorted), words_len):
            sf = symbols_sorted[idx]
            sl = symbols_sorted[idx + words_len - 1]
            ty = math.min(sf[2] - sf[4], sl[2] - sl[4])
            tx = sf[1] - sf[3]
            by = math.max(sl[2] + sl[4], sl[2] + sl[4])
            bx = sl[1] + sl[3]
            img_name = 'img_{}_{}_{}.jpg'.format(img_file, idx, words_len)
            imwrite(img_name, img[ty:by+1, tx:bx+1])
            img_label_list.append((img_name, [s[0] for s in symbols_sorted[idx : idx + word_len]]))
    return

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
    parser.add_argument("-a", "--data-root", type=str, help='dataset root')

    args = vars(parser.parse_args())


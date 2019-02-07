#!/usr/bin/python
import argparse
import dataset
import os
import pathlib
from PIL import Image
import random
import torch
from torch.autograd import Variable
import utils

import models.crnn as crnn


def predict_one(model, imagefile, transformer, converter):
    image = Image.open(imagefile).convert('L')
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    return raw_pred, sim_pred

if __name__ == '__main__':
    ## Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, help="Path to the input images", default='')
    parser.add_argument("-l", "--image-list", type=str, help="Path to the directory of images", default='')
    parser.add_argument("-b", "--labels", type=str, help="Specify the label file", default='')
    parser.add_argument("-d", "--debug", action='store_true', help="Debug mode", default=0)
    parser.add_argument("-m", "--model", type=str, help="the model file")
    parser.add_argument("-n", "--number", type=int, help="the number of images to predict", default=100)

    args = parser.parse_args()

    alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

    if (args.labels == ''):
        print("A label file is missing")
        sys.exit(0)
 
    model = crnn.CRNN(32, 1, len(alphabet) + 1, 256)    
    if torch.cuda.is_available():
        model = model.cuda()
    print('loading pretrained model from %s' % args.model)

    state_dict = torch.load(args.model)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    model.eval()

    converter = utils.strLabelConverter(alphabet)
    transformer = dataset.resizeNormalize((100, 32))

    if (args.image != ''):
        raw, sim = predict_one(model, args.image, transformer, converter)
        print('%20s: %-20s => %-20s' % (args.image, raw, sim))

    if (args.image_list != ''):
        gt = {}
        with open(os.path.join(args.image_list, args.labels)) as f:
            for l in f:
                k, v = l.strip().split(' ')
                gt[k] = v
        wrong = {}
        wrong_count = 0
        images = []
        for f in pathlib.Path(args.image_list).glob("**/*.jpg"):
            images.append(f)
        random.shuffle(images)
        predicted = 0
        for f in images:
            r, p = predict_one(model, str(f), transformer, converter)
            if (p != gt[f.name]):
                wrong[f.name] = (gt[f.name], p)
                print("{}, gt: {}, pred:{}".format(f.name, gt[f.name], p))
                wrong_count += 1
            predicted += 1
            if (predicted > args.number):
                break
        print("Total={}, Wrong={}, Error rate={}".format(len(gt), wrong_count, float(wrong_count) / predicted))

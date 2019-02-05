#!/usr/bin/python
import argparse
import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import models.crnn as crnn


if __name__ == '__main__':
    ## Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, help="Path to the input images", default='')
    parser.add_argument("-d", "--debug", action='store_true', help="Debug mode", default=0)
    parser.add_argument("-m", "--model", type=str, help="the model file")

    args = parser.parse_args()

    alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'


    model = crnn.CRNN(32, 1, len(alphabet) + 1, 256)    
    if torch.cuda.is_available():
        model = model.cuda()
    print('loading pretrained model from %s' % args.model)
    model.load_state_dict(torch.load(args.model))

    converter = utils.strLabelConverter(alphabet)

    transformer = dataset.resizeNormalize((100, 32))
    image = Image.open(args.image).convert('L')
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('%-20s => %-20s' % (raw_pred, sim_pred))

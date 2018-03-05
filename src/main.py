import os
import time
import torch
import random
import numpy as np

from datetime import datetime

from utils import load_data, build_vocab
from config import args
from model import Model

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if __name__ == '__main__':

    build_vocab()
    train_data = load_data('./data/train-data-processed.json')
    train_data += load_data('./data/trial-data-processed.json')
    dev_data = load_data('./data/dev-data-processed.json')
    if args.test_mode:
        # use validation data as training data
        train_data += dev_data
        dev_data = []
    model = Model(args)

    best_dev_acc = 0.0
    os.makedirs('./checkpoint', exist_ok=True)
    checkpoint_path = './checkpoint/%d-%s.mdl' % (args.seed, datetime.now().isoformat())
    print('Trained model will be saved to %s' % checkpoint_path)

    for i in range(args.epoch):
        print('Epoch %d...' % i)
        if i == 0:
            dev_acc = model.evaluate(dev_data)
            print('Dev accuracy: %f' % dev_acc)
        start_time = time.time()
        np.random.shuffle(train_data)
        cur_train_data = train_data

        model.train(cur_train_data)
        train_acc = model.evaluate(train_data[:2000], debug=False, eval_train=True)
        print('Train accuracy: %f' % train_acc)
        dev_acc = model.evaluate(dev_data, debug=True)
        print('Dev accuracy: %f' % dev_acc)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            os.system('mv ./data/output.log ./data/best-dev.log')
            model.save(checkpoint_path)
        elif args.test_mode:
            model.save(checkpoint_path)
        print('Epoch %d use %d seconds.' % (i, time.time() - start_time))

    print('Best dev accuracy: %f' % best_dev_acc)

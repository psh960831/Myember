"""
This python module refer to Ember Porject(https://github.com/endgameinc/ember.git)
"""
import os
import sys
import tqdm
import ember
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import OrderedDict
import utility
import pickle


class Predictor:
    def __init__(self, modelpath, testdir, features, output):
        # load model with pickle to predict
        with open(modelpath, 'rb') as f:
            self.model = lgb.Booster(model_file=modelpath)
        self.testdir = testdir
        self.output = output
        self.features = features

    def run(self):
        y_pred = []
        name = []
        err = 0
        end = len(next(os.walk(self.testdir))[2])

        for sample in tqdm.tqdm(utility.directory_generator(self.testdir), total=end):
            fullpath = os.path.join(self.testdir, sample)

            if os.path.isfile(fullpath):
                binary = open(fullpath, "rb").read()
                name.append(sample)

                try:
                    y_pred.append(ember.predict_sample(self.model, binary, self.features))           
                except KeyboardInterrupt:
                    sys.exit()
                except Exception as e:
                    print('error')            
                    y_pred.append(0)
                    err += 1
        #print(np.array(y_pred))
        y_pred = np.where(np.array(y_pred) > 0.5, 1, 0)
        series = OrderedDict([('hash', name),('y_pred', y_pred)])
        r = pd.DataFrame.from_dict(series)
        r.to_csv(self.output, index=False, header=None)

        #logger.info('{} error is occured'.format(err))

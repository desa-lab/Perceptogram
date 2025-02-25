import numpy as np
import pickle

with open('cache/regression_weights/BIGMEG1/thingsmeg_regress_autokl1b_weights_sub-BIGMEG1.pkl',"rb") as f:
    datadict = pickle.load(f)
    reg_w = datadict['weight']
    reg_b = datadict['bias']
import sys
import os
import re
import numpy as np
import SVM

import random
import time

"""
SVM
"""
start = time.time()


def run(test_in, model_file, sys_output):
    start = time.time()
    with open(test_in, 'r', encoding='utf8') as f:
        test_lines = f.readlines()
    with open(model_file, 'r', encoding='utf8') as f:
        model_lines = f.readlines()

    # test_lines = test_lines[0:2]
    clf = SVM.SVMClassifier()
    clf.load_model(model_lines)
    clf.test_raw = test_lines
    clf.process_test()

    y_pred_ts = clf.predict(clf.X_test, save='test')
    end = time.time()
    total_time = end - start

    clf.save_sys_output(sys_output)
    clf.classification_report()
    end = time.time()
    total_time = end - start


if __name__ == "__main__":
    TEST = False
    if TEST:
        TEST_IN = '/Users/Karl/Documents/_UW_Compling/LING572/hw8/hw8/examples/test'
        MODEL_FILE = 'model.5'
        SYS_OUTPUT = 'sys.5'
    else:
        TEST_IN = sys.argv[1]
        MODEL_FILE = sys.argv[2]
        SYS_OUTPUT = sys.argv[3]
    run(TEST_IN, MODEL_FILE,  SYS_OUTPUT)

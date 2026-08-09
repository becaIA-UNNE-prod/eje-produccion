#!/bin/bash

cd /mnt/yacy_1/prod/ferreyra/dataset/train_cordoba_mnc
for t in 20HMJ 20HMK 20JML 20HPG 20JNL 20HNK; do
  python3 -c "
import numpy as np
try:
    with np.load('dataset_mnc_$t.npz') as d:
        d['X']; d['Y']
    print('OK  $t')
except Exception as e:
    print('BAD $t ->', e)
"
done

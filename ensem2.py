import os
import gzip
import pickle
import pandas as pd
import numpy as np
from a00_common_functions import *


pyto_preds, class_names = pickle.load(open('preds.pkl', 'rb'))

idx_back, idx_forw = get_classes_to_index_dicts()
idx_map = np.zeros(len(class_names), dtype=np.int32)
for i, class_name in enumerate(class_names):
    if class_name not in idx_back.keys():
        print('not found', i, class_name)
        idx_map[i] = -1
        continue
    idx_map[i] = idx_back[class_name]
tf_preds = pickle.load(gzip.open('/dataset/openimages/test_preds.pkl', 'rb'))

assert pyto_preds.shape[0] == tf_preds.shape[0]


subm = pd.read_csv('/dataset/openimages/tf/subm/subm-anil-5.csv')
class_description_file = os.path.join('openimages', 'meta', 'class-descriptions.csv')
class_descriptions = pd.read_csv(class_description_file, index_col=None)
name_map = dict(class_descriptions.values)

assert pyto_preds.shape[0] == subm.shape[0]

w = 0.95
for i in range(pyto_preds.shape[1]):
    j = idx_map[i]
    if j == -1:
        continue
    tf_preds[:, j] = tf_preds[:, j] * w + pyto_preds[:, i] * (1 - w)

eps = 1e-7
add_count = 0
del_count = 0
thresh = 0.6

for i, row in subm.iterrows():
    image_id = row['image_id']
    row_labels = [] if type(row['labels']) == float else row['labels'][:-1].split(' ')
    pred_row = pyto_preds[i]
    tf_pred_row = tf_preds[i]

    new_row_labels = []
    sel = np.where(tf_pred_row > thresh)[0]
    for sel_idx in sel:
        sel_name = idx_forw[sel_idx]
        new_row_labels.append(sel_name)
        if sel_name not in row_labels:
            actual_name = name_map[sel_name] if sel_name in name_map else 'unknown'
            print('adding %s to %s' % (actual_name, image_id))
            add_count += 1

    for sel_name in row_labels:
        if sel_name not in new_row_labels:
            actual_name = name_map[sel_name] if sel_name in name_map else 'unknown'
            print('deleting %s from %s' % (actual_name, image_id))
            del_count += 1

    subm.iloc[i]['labels'] = ' '.join(new_row_labels)

subm.to_csv('subm-anil-6.csv', index=False)
print('added %d items. deleted %d items' % (add_count, del_count))







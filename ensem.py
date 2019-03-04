import os
import pickle
import pandas as pd
import numpy as np


preds, class_names = pickle.load(open('preds.pkl', 'rb'))
subm = pd.read_csv(os.path.join('openimages', 'meta', 'best-submission.csv'))
class_description_file = os.path.join('openimages', 'meta', 'class-descriptions.csv')
class_descriptions = pd.read_csv(class_description_file, index_col=None)
name_map = dict(class_descriptions.values)

assert preds.shape[0] == subm.shape[0]
eps = 1e-7
add_count = 0
del_count = 0
for i, row in subm.iterrows():
    image_id = row['image_id']
    row_labels = row['labels'].split(' ')
    pred = preds[i]

    if False:
        sel = np.where(pred > (1 - eps))[0]
        for sel_idx in sel:
            sel_name = class_names[sel_idx]
            if sel_name not in row_labels:
                actual_name = name_map[sel_name] if sel_name in name_map else 'unknown'
                print('adding %s to %s' % (actual_name, image_id))
                row_labels.append(sel_name)
                add_count += 1

    sel = np.where(pred < eps)[0]
    for sel_idx in sel:
        sel_name = class_names[sel_idx]
        if sel_name in row_labels:
            actual_name = name_map[sel_name] if sel_name in name_map else 'unknown'
            print('deleting %s from %s' % (actual_name, image_id))
            row_labels.remove(sel_name)
            del_count += 1

    subm.iloc[i]['labels'] = ' '.join(row_labels)

subm.to_csv('subm-anil.csv', index=False)
print('added %d items. deleted %d items' % (add_count, del_count))







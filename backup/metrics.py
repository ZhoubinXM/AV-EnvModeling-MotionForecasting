import pandas as pd
import os

root = "/data/guifeng/crash_dataset/result"

# collision
file = os.path.join(root, "collision.csv")
df = pd.read_csv(file, low_memory=False)
dynamic_num = sum(df['crashclassification'] == 'dynamicObj')
true_num0 = df.apply(lambda x: True
                    if x['risk'] and x['crashclassification'] == 'dynamicObj' and x['time_before_crash'] > 0.1 else False,
                    axis=1).sum()
true_num1 = df.apply(lambda x: True
                    if x['risk'] and x['crashclassification'] == 'dynamicObj' and x['time_before_crash'] > 1 else False,
                    axis=1).sum()
true_num2 = df.apply(lambda x: True
                    if x['risk'] and x['crashclassification'] == 'dynamicObj' and x['time_before_crash'] > 2 else False,
                    axis=1).sum()
tp_num = df.apply(lambda x: True if x['risk'] and x['crashclassification'] == 'dynamicObj' and x['predict_crash_time'] >= 1 and
                  x['time_before_crash'] >= 1 else False,
                  axis=1).sum()
print(dynamic_num, tp_num / dynamic_num, true_num2 / dynamic_num, true_num1 / dynamic_num, true_num0 / dynamic_num)

# risk
file = os.path.join(root, "risk.csv")
df = pd.read_csv(file, low_memory=False)
dynamic_num = sum(df['crashclassification'] == 'dynamicObj')
true_num0 = df.apply(lambda x: True
                    if x['risk'] and x['crashclassification'] == 'dynamicObj' and x['time_before_crash'] > 0.1 else False,
                    axis=1).sum()
true_num1 = df.apply(lambda x: True
                    if x['risk'] and x['crashclassification'] == 'dynamicObj' and x['time_before_crash'] > 1 else False,
                    axis=1).sum()
true_num2 = df.apply(lambda x: True
                    if x['risk'] and x['crashclassification'] == 'dynamicObj' and x['time_before_crash'] > 2 else False,
                    axis=1).sum()
tp_num = df.apply(lambda x: True if x['risk'] and x['crashclassification'] == 'dynamicObj' and x['predict_crash_time'] >= 1 and
                  x['time_before_crash'] >= 1 else False,
                  axis=1).sum()
print(dynamic_num, tp_num / dynamic_num, true_num2 / dynamic_num, true_num1 / dynamic_num, true_num0 / dynamic_num)

# collision
file = os.path.join(root, "normal.csv")
df = pd.read_csv(file, low_memory=False)
total_num = df.shape[0]
tn_num = df.apply(lambda x: True if x['risk'] == False else False, axis=1).sum()
print(total_num, tn_num / total_num)

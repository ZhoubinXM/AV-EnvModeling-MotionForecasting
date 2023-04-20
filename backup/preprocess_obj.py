import pandas as pd
import numpy as np 
import math 
from operator import itemgetter, mul
from itertools import groupby
from multiprocessing import Pool
import time

# pd.options.display.max_columns = None
# pd.options.display.max_rows = None

def depart_uuid(start, end, df): 
    '''depart by uuid
    Args:
        start: start index
        end: end index
        df:  dataframe of csv
    Returns:
        the last element of change index
    '''
    lst_uuid = ''
    change_idx = []
    uuid = df['c_dlb_uuid'] 
    for i in range(start, end+1):
        cur_uuid = uuid.iloc[i]
        if lst_uuid and cur_uuid!=lst_uuid:
            change_idx.append(i)
        lst_uuid = cur_uuid
    if not change_idx:
        return -1
    else:
        return change_idx[-1]

def age_changed(start, end, idx, df):  #age突变 1-2-3-4  1-2-3-4-70 
    '''depart by age suddenly changed
    Args:
        start: start index
        end: end index
        idx: obj index
        df:  dataframe of csv
    Returns:
        the last element of change index
    '''
    age_df = df.iloc[start : end+1]['l_age_'+str(idx+1)]
    diff_lst = np.where((np.diff(age_df.values)!=1) & (np.diff(age_df.values)!=0))[0].tolist()
    diff_lst = [start+i for i in diff_lst]
    if not diff_lst:
        return -1
    return diff_lst[-1]+1

def obj_id_not_null(start, end, idx, df):
    '''depart by null obj
    Args:
        start: start index
        end: end index
        idx: obj index
        df:  dataframe of csv
    Returns:
        the next of index of last element of null obj
    '''
    null_lst = [] 
    for i in range(start, end+1):
        if pd.isnull(df.iloc[i]['c_obf_id_'+str(idx+1)]):
            null_lst.append(i)
    if not null_lst:
        return -1
    return null_lst[-1]+1

def set_vector(vector, start_idx, end_idx, idx, length, df, frame): 
    '''set vector value
    Args:
        start_idx: start index
        end_idx: end index
        idx: obj index
        length: the third dimention of vector, for now is 50(-1). 
        df:  dataframe of csv
        frame: frame extracting proportion
    Returns:
        vector
    '''
    start = time.time()
    for ele in range(start_idx, end_idx+1):
        if not ele % frame:
            slice_idx = int(ele / length / frame)
            current_idx = int((ele - length*slice_idx*frame) / frame)
            if (current_idx + 1) % length == 0:
                continue
            vector[idx, slice_idx, current_idx,:] = [df.iloc[ele]['l_pos_x_'+str(idx+1)], df.iloc[ele]['l_pos_y_'+str(idx+1)], df.iloc[ele+1]['l_pos_x_'+str(idx+1)], df.iloc[ele+1]['l_pos_y_'+str(idx+1)], df.iloc[ele]['l_vx_'+str(idx+1)], df.iloc[ele]['l_vy_'+str(idx+1)], df.iloc[ele+1]['l_vx_'+str(idx+1)], df.iloc[ele+1]['l_vy_'+str(idx+1)], df.iloc[ele]['l_heading_'+str(idx+1)], df.iloc[ele+1]['l_heading_'+str(idx+1)], df.iloc[ele]['c_main_class_'+str(idx+1)], df.iloc[ele]['c_sub_class_'+str(idx+1)]]
    end = time.time()
    return vector

def main(idx): 
    '''Args:
        idx: obj index
    Returns:
    '''
    for i in range(l): 
        time_start = time.time()
        if pd.isnull(df.iloc[length * (frame * i + frame) -1]['c_obf_id_'+str(idx+1)]):
            continue
        start = length * frame * i 
        end = length * (frame* i + frame) -1 
        last_changed_age = age_changed(start, end,idx,df)
        last_changed_uuid = depart_uuid(start, end,df)
        last_id_not_null = obj_id_not_null(start, end,idx,df)
        idx_changed = max([last_changed_age, last_changed_uuid, last_id_not_null])
        global vector 
        if idx_changed == -1:
            vector = set_vector(vector, start, end, idx, length, df, frame)
        else: 
            vector = set_vector(vector, idx_changed, end, idx, length, df, frame)
        time_end = time.time()
        print("i: ", i, " and time: ", time_end-time_start)

if __name__ == "__main__":
    df = pd.read_csv('/data/nio_dataset/database/db_abnormal_takeover_np_nop_20220913_prod_462.csv')
    length = 50
    frame = 3 #按照1:1 去抽    frame=2 按照2:1去抽
    l = int(len(df)/frame/length) #一共l段50
    obj_num = 128
    vector = np.zeros((obj_num, l, length-1, 12), np.float32)
    start_ts = time.time()
    p = Pool(40)
    for i in range(obj_num):
        p.apply_async(func=main, args=(i,))
    p.close()
    p.join()
    end_ts = time.time()
    run_ts = end_ts - start_ts
    print("Average time={}".format(run_ts/obj_num))

    np.save(file="db_abnormal_takeover_np_nop_20220913_obj.npy", arr=vector)

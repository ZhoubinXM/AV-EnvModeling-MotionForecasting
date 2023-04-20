import pandas as pd
import numpy as np 
from re import L

# pd.options.display.max_columns = None
# pd.options.display.max_rows = None

def get_lane_pts(df, param, i, length):
    '''depart by uuid
    Args:
        df:  dataframe of csv
        param: lane role(left/right/left_right...) 
        i: index of df
        length: the third dimention of vector/ length of lane
    Returns:
        points of the lane 
    '''
    distance_diff = 1.5
    points_lst = []
    if param != -1:
        start = df.iloc[i]['l_LD_Start_' + str(param)]
        c0 = df.iloc[i]['l_line_C0_' + str(param)]
        c1 = df.iloc[i]['l_line_C1_' + str(param)]
        c2 = df.iloc[i]['l_line_C2_' + str(param)]
        c3 = df.iloc[i]['l_line_C3_' + str(param)]
        for i in range(length):
            x = start + i * distance_diff # 1.5米一个点 
            points_lst.append([x, c0 + c1*x + c2*x*x + c3*x*x*x])
    return points_lst

def set_vector(vector, role, role_lst, df_lanes, i, frame, length):
    '''
    Args:
        vector: vector to be set
        role: lane role definition
        role_lst: lanes in the i frame
        df_lanes: columns related to lane in df
        i: i before frame extracting 
        frame: frame extracting proportion
        length: the third dimention of vector/ length of lane
    Returns:
        vector 
    '''
    left_right = 4.0
    left = 1.0
    right_left = 5.0
    right = 2.0
    left_left = 3.0
    leftleft_right = 8.0
    right_right = 6.0
    rightright_left = 9.0

    con1 = (role == left_right) and (left in role_lst)
    con2 = (role == right_left) and (right in role_lst)
    con3 = (role == leftleft_right) and (left_left in role_lst)
    con4 = (role == rightright_left) and (right_right in role_lst)
    role_idx = -1
    if (con1) or (con2) or (con3) or (con4):
        return vector 
    if role in role_lst:
        role_idx = role_lst.index(role)
        print("role: ", role_idx)
        if role_idx != -1:
            lane_points_lst = get_lane_pts(df_lanes, role_idx, i*frame, length)
            for count in range(len(lane_points_lst) - 1):
                vector[i, 0, count, :] = [lane_points_lst[count][0], lane_points_lst[count][1], lane_points_lst[count+1][0], lane_points_lst[count+1][1], df_lanes.iloc[i*frame]['c_LD_Type_'+str(role_idx)], df_lanes.iloc[i*frame]['c_LD_Track_ID_'+str(role_idx)]] 
    return vector

if __name__ == "__main__":
    # 这里可能会有稀疏矩阵问题 
    df = pd.read_csv('/data/nio_dataset/database/db_abnormal_takeover_np_nop_20220913_prod_462.csv')
    length = 50 # 50 slice of one lane 
    frame = 3
    df_lanes = df.iloc[:, 2764:2876]
    df_ld_role = df[['c_LD_Role_'+str(i) for i in range(10)]]
    vector = np.zeros((int(len(df_lanes)/frame), 10, length - 1, 6), np.float32) # 10表示10条lane, 6: s_x, s_y, e_x, e_y, c_LD_Type_, c_LD_Track_ID_,
    for i in range(int(len(df_lanes)/frame)):
        for role in range(1,11,1):
            vector = set_vector(vector, role, df_ld_role.iloc[i*frame].tolist(), df_lanes, i, frame, length)

    np.save(file="db_abnormal_takeover_np_nop_20220913_lane.npy", arr=vector)


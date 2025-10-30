import random
random.seed(2023)
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import os
import pandas as pd
import numpy as np
np.random.seed(2023)
import time
import pickle
import multiprocessing

def dist_dtw(all_data, keys, compare_num, process_num, multiple, step):
    traj_DTW = {}
    index_set = list(range(0, len(keys)))
    unit = int((1/4)*len(keys)/process_num)
    #print('unit:', unit)
    for i in range(multiple*unit, (multiple+1)*unit):
        avg_distance = 0
        seg_index_set = list(range(0, len(all_data[keys[i]]), step))
        x = np.array(list(map(lambda h: all_data[keys[i]][h], seg_index_set)))
        rand_select_index = random.sample(index_set, compare_num)
        max_value = -10
        min_value = 1000000000000
        for j in rand_select_index:
            seg_index_set = list(range(0, len(all_data[keys[j]]), step))
            y = np.array(list(map(lambda h: all_data[keys[j]][h], seg_index_set)))
            distance, dtw_path = fastdtw(x, y, dist=euclidean)
            if distance > max_value:
                max_key = keys[j]
                max_value = distance
            if distance < min_value:
                min_key = keys[j]
                min_value = distance
            avg_distance += distance
        avg_distance = avg_distance/compare_num
        traj_DTW[keys[i]] = [str(avg_distance), max_key, min_key]
        print(str(i) + ' ' + str(avg_distance) + ' '+ str(max_key)+' '+ str(min_key))
    # with open('fast_DTW/traj_fast_DTW_5w_part1_'+str(multiple)+'.pkl', 'wb') as f:
    #     pickle.dump(traj_DTW, f)

if __name__ == '__main__':
    random.seed(2023)
    temp_path = 'data/lon_lat_data/data_5w_dilute_step3.pkl'
    with open(temp_path, 'rb') as file:
        file_temp=pickle.load(file, encoding='latin1')
    keys = []
    for key in file_temp.keys():
        keys.append(key)
    keys = keys[0:10]
    print('len keys:', len(keys))
    process_num = 1
    compare_num = 10
    step = 1
    temp = time.time()
    record = []
    for multiple in range(process_num):
        p1 = multiprocessing.Process(target=dist_dtw, args=(file_temp, keys, compare_num, process_num, multiple, step))
        record.append(p1)
        p1.start()
    for process in record:
        process.join()
    cal_time = time.time() - temp
    print('cal_time:', cal_time)
    print('success')
    
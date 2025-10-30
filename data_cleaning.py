import pandas as pd
from math import radians, cos, sin, asin, sqrt
from queue import Queue
from tqdm import tqdm
import logging
from datetime import datetime
import os
import numpy as np

"""
数据清洗类 Class DataCleaning;
         使用方法 :
         clean = DataCleaning(data_source)
         data_after_clean = clean.cleaning(args)
         clean.log()
源数据data_source为api返回的单个农机轨迹数据的Json、list或dataframe，至少但不限于包含字段’2602’,‘2603’,‘2204’,‘3014’
args为功能选择参数，默认全为True
               bool reference_point_enable, 起点筛选
               bool pre_screen_enable, 筛除异常坐标
               bool offset_enable, 筛除偏移点
               bool repeat_point_enable, 筛除时间/坐标重复点
               bool zero_drift_enable, 筛除零漂
返回值为dataframe
各功能可单独调用，如需调参请单独调用将超参数传入函数变量。
"""


class DataCleaning:
    _LON_COLUMN = 'longitude'
    _LAT_COLUMN = 'latitude'
    _SPEED_COLUMN = 'speed'
    _TIME_COLUMN = 'datetime'
    _MAX_SPEED = 60
    log = {"source_points": 0, "points_after_cleaning": 0, "start_point": 0, "pre_screen_out_points": 0,
           "repeat_points": 0, "offset_points": 0, "wrong_speed": 0, "zero_drift": 0, }

    def __init__(self, data_input):
        """
        :param data_input:
               为AIP返回数据
               格式为  json:{"data":[{},{},{}]...}
                  或  list:[{},{},{}...]
                  或  dataframe:columns=["longitude","latitude","speed","datetime",...]
        """
        if self._type_transform(data_input):
            self.log['source_points'] = len(self.data_per_day)

    def _type_transform(self, data_input):
        """
        识别并转换数据格式
        """
        if isinstance(data_input, list):
            self.data_per_day = pd.DataFrame(data_input)
        elif isinstance(data_input, dict):
            self.data_per_day = pd.DataFrame(data_input.json()['data'])
        elif isinstance(data_input, pd.core.frame.DataFrame):
            data = data_input.reset_index()
            del data['index']
            self.data_per_day = data
        else:
            print("data error! fail to init DataCleaning")
            self.data_per_day = None
            return False

        try:
            self.data_per_day.rename(columns={self._TIME_COLUMN: 'time'}, inplace=True)
        except BaseException:
            print('wrong columns names!')
        return True

    def _haversine(self, start, end):
        """
        计算两点路程(m)
        star:起点index下标
        end:终点index下标
        """
        lon1 = self.data_per_day['longitude'].iloc[start]
        lon2 = self.data_per_day['longitude'].iloc[end]
        lat1 = self.data_per_day['latitude'].iloc[start]
        lat2 = self.data_per_day['latitude'].iloc[end]
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371
        return c * r * 1000

    def _get_row_time(self):
        _time_list = self.data_per_day[self._TIME_COLUMN]
        for i in range(len(_time_list)):
            _time_list[i] = _time_list[i][-6:]
        return _time_list

    def _get_interval(self, end, start):
        """
        计算时间差（s）
        star:起点index下标
        end:终点index下标
        """
        end = datetime.strptime(
            str(self.data_per_day.loc[end, 'time']), "%Y%m%d%H%M%S")
        start = datetime.strptime(
            str(self.data_per_day.loc[start, 'time']), "%Y%m%d%H%M%S")
        interval = (end-start).seconds
        return interval

    def pre_screen_out(self, lon_max=135, lon_min=73, lat_max=53, lat_min=3):
        """初步筛除坐标异常点(出界或0坐标)"""
        self.data_per_day = self.data_per_day[
            (self.data_per_day['longitude'] >= lon_min) & (self.data_per_day['longitude'] <= lon_max)]
        self.data_per_day = self.data_per_day[
            (self.data_per_day['latitude'] >= lat_min) & (self.data_per_day['latitude'] <= lat_max)]
        logging.info('pre_screen_out: 50% complete')
        self.data_per_day.reset_index(inplace=True, drop=True)
        d = len(self.data_per_day)
        self.log["pre_screen_out_points"] = self.log["source_points"] - \
            len(self.data_per_day)
        self.log["points_after_cleaning"] = len(self.data_per_day)
        logging.info('pre_screen_out: 100% complete')

    def find_start_point(self, sliding_steps=5):
        """
        找到一个相对准确的点作为数据清洗的参照起点,
        每次入队sliding_steps个最邻近待选点（非重复，非零速点），
        若多边形符合速度限制则将第一个点设为起点，否则窗口滑动到不符合的下一个点，直到找到起点或遍历结束
        成功则返回起点下标，失败则返回原始点并在log中置‘start_point’为-1
        """
        index = self.data_per_day.index
        counter = 0
        #  定义一个定长队列，逐个检查数据，将合理数据下标入队直至队满
        windows = Queue(maxsize=sliding_steps)
        windows.put(index[0])
        index_candidate = [0, 0, 0]  # 候选起点、上一个出队点、当前出队点
        while True:
            while not windows.full():
                counter += 1
                if counter < len(index):
                    pointer_index = index[counter]
                    interval = self._get_interval(
                        pointer_index, index[counter - 1])
                    distance = self._haversine(
                        pointer_index, index[counter - 1])
                    if (interval != 0) and (distance != 0) and (self.data_per_day.loc[pointer_index, 'speed'] != 0):
                        windows.put(pointer_index)
                else:
                    self.log["start_point"] = -1
                    return index[0]
            # 记录待选下标组candidate，将元素逐个出队并计算speed，若全部通过输出起点下标
            if not windows.empty():
                index_candidate = [windows.get(), 0, 0]
                index_candidate[2] = index_candidate[0]
            while not windows.empty():
                success = True
                index_candidate[1] = index_candidate[2]
                index_candidate[2] = windows.get()
                interval = self._get_interval(
                    index_candidate[2], index_candidate[1])
                distance = self._haversine(
                    index_candidate[1], index_candidate[2])
                # 若在index=candidate[2]处失败，保留剩余点重新装填windows队列
                if (distance * 1.0 / interval) >= self._MAX_SPEED * 0.2777778:
                    success = False
                    break
            # 待选点两两符合配速后，查看第一个和最后一个点间是否满足
            if success and (self._haversine(index_candidate[0], index_candidate[2]) * 1.0 /
                            self._get_interval(index_candidate[0], index_candidate[2])) <= self._MAX_SPEED * 0.2777778:
                # 查找成功则输出起点并删除之前的点，失败则重新装填windows
                self.log["start_point"] = index_candidate[0]
                if index_candidate[0] != index[0]:
                    self.data_per_day.drop(index=range(
                        0, list(index).index(index_candidate[0])), inplace=True)
                    # self.data_per_day=self.data_per_day.loc[index_candidate[0]:]
                return index_candidate[0]

    def drop_outspeed(self, repeat_point_enable=True, speed_max=60):
        """
                筛除连续点中时间或坐标重复点
                筛除超速偏移点
                以平均速度替换异常瞬时速度
                速度阈值为 km/h
        """
        repeat_point_indexes = []
        offset_point_indexes = []
        speed_max_s = speed_max * 0.2777778
        index = self.data_per_day.index
        index_ordered_last = 0  # 已清洗序列的最后一个下标
        speed_list = self.data_per_day['speed']
        for i in range(1, len(index)):
            try:
                interval = self._get_interval(index[i], index_ordered_last)
                distance = self._haversine(index[i], index_ordered_last)
                # (1)时间或坐标重复点
                if repeat_point_enable and ((interval == 0.0) or (distance == 0.0)):
                    repeat_point_indexes.append(index[i])
                    continue
                if distance * 1.0 / interval > speed_max_s:  # (2)超速偏移点
                    offset_point_indexes.append(index[i])
                    continue
                index_ordered_last = index[i]  # 都不匹配则该点视为正常点
                if not speed_list[i] >= 0:  # (3)瞬时速度异常点 None或小于0
                    self.data_per_day.loc[index_ordered_last,
                                          'speed'] = distance * 1.0 / interval
                    self.log['wrong_speed'] += 1
            except BaseException:
                print("error row")
                print(self.data_per_day.iloc[index[i]])
                continue
        self.data_per_day.drop(index=repeat_point_indexes, inplace=True)
        self.data_per_day.drop(index=offset_point_indexes, inplace=True)
        self.log["offset_points"] = len(offset_point_indexes)
        self.log["repeat_points"] = len(repeat_point_indexes)
        self.log["points_after_cleaning"] = len(self.data_per_day)

    def drop_zero_drift(self):
        """筛除零点漂移,将连续的0速点视为停留点"""
        self.data_per_day.reset_index(inplace=True, drop=True)
        drop_list = []
        zero_speed_index = self.data_per_day[self.data_per_day['speed'] == 0].index
        for i in range(1, len(zero_speed_index)):
            if zero_speed_index[i - 1] == zero_speed_index[i] - 1:
                drop_list.append(zero_speed_index[i])
        logging.info('drop_outspeed: 50% complete')
        self.data_per_day.drop(index=drop_list, inplace=True)
        self.data_per_day.reset_index(inplace=True, drop=True)
        self.log["zero_drift"] = len(drop_list)
        self.log["points_after_cleaning"] = len(self.data_per_day)
        logging.info('drop_outspeed: 100% complete')


    def cleaning(self, reference_point_enable=True, pre_screen_enable=True,
                 offset_enable=True, zero_drift_enable=True, repeat_point_enable=True):
        """总调用函数，选择启用哪些功能"""
        good_quality = True
        if pre_screen_enable and good_quality:
            logging.info('Running pre_screen_out function...')
            self.pre_screen_out()
            good_quality = True if len(self.data_per_day) > 1 else False
        if reference_point_enable and good_quality:
            logging.info('Running find_start_point function...')
            self.find_start_point()
            self.data_per_day.reset_index(inplace=True, drop=True)
            good_quality = True if len(self.data_per_day) > 1 else False
        if offset_enable and good_quality:
            logging.info('Running drop_outspeed function...')
            self.drop_outspeed(repeat_point_enable=repeat_point_enable)
            good_quality = True if len(self.data_per_day) > 1 else False
        if zero_drift_enable and good_quality:
            logging.info('Running drop_zero_drift function...')
            self.drop_zero_drift()
            good_quality = True if len(self.data_per_day) > 1 else False
        self.data_per_day.reset_index(inplace=True, drop=True)
        logging.info("Done cleaning!")
        return self.data_per_day
        #return good_quality

    def print_log(self):
        print(self.log)

def get_cleaned_data(data_list):
    return DataCleaning(data_list).cleaning()


def data_clean(data):
    data = get_cleaned_data(data)
    return data

def data_cleaning_paddy():
    print('数据清洗')
    # data_path = 'data/trajectory_data/raw_wheat/'
    # clean_data_save_path = 'data/trajectory_data/raw_wheat_cleaned/'
    data_path = 'data/trajectory_data/raw_paddy/'
    clean_data_save_path = 'data/trajectory_data/raw_paddy_cleaned/'
    all_file = []
    for filepath, dirnames, filenames in os.walk(data_path):
        for filename in filenames:
            all_file.append(filename)
    for i in range(len(all_file)):
        print(i)
        print(all_file[i])
        traj_name = all_file[i].replace('.xlsx', '')
        data = pd.read_excel(data_path + all_file[i])
        if len(data) > 200:
            #print('start:', len(data))
            data.rename(columns={'时间': 'datetime', '经度': 'longitude', '纬度': 'latitude','速度':'speed','方向':'direction',
                                 '高度':'altitude'}, inplace=True)
            time_list = data['datetime'].tolist()
            for n in range(len(time_list)):
                time_list[n] = time_list[n].replace('/','').replace(':','').replace(' ','').replace('-','')
            data['datetime'] = time_list
            cleaned_data = data_clean(data)
            lon = cleaned_data['longitude'].tolist()
            lat = cleaned_data['latitude'].tolist()
            times = cleaned_data['time'].tolist()
            speeds = cleaned_data['speed'].tolist()
            directions = cleaned_data['direction'].tolist()
            altitudes = cleaned_data['altitude'].tolist()
            tags = cleaned_data['标记'].tolist()
            point_index = []
            for h in range(len(tags)):
                point_index.append(traj_name+'_'+str(h))
            # same_data = []
            # for m in range(len(cleaned_data)):
            #     same_data.append([point_index[m], times[m], lon[m], lat[m], speeds[m], directions[m], altitudes[m], tags[m]])
            # ddaad = pd.DataFrame(same_data)
            # ddaad.columns = ['index', 'time', 'longitude', 'latitude', 'speed', 'direction', 'altitude', 'tag']
            # print(clean_data_save_path + 'cleaned_' +all_file[i])
            # ddaad.to_excel(clean_data_save_path + 'cleaned_' +all_file[i], index=False)
    return ddaad

def data_cleaning_wheat():
    print('数据清洗')
    data_path = 'data/trajectory_data/raw_wheat/'
    clean_data_save_path = 'data/trajectory_data/raw_wheat_cleaned/'
    all_file = []
    for filepath, dirnames, filenames in os.walk(data_path):
        for filename in filenames:
            all_file.append(filename)
    for i in range(len(all_file)):
        print(i)
        traj_name = all_file[i].replace('.xlsx', '')
        data = pd.read_excel(data_path + all_file[i])
        if len(data) > 200:
            #print('start:', len(data))
            data.rename(columns={'时间': 'datetime', '经度': 'longitude', '纬度': 'latitude','速度':'speed','方向':'direction',
                                 '高度':'altitude'}, inplace=True)
            time_list = data['datetime'].tolist()
            for n in range(len(time_list)):
                time_list[n] = time_list[n].replace('/','').replace(':','').replace(' ','').replace('-','')
            data['datetime'] = time_list
            cleaned_data = data_clean(data)
            lon = cleaned_data['longitude'].tolist()
            lat = cleaned_data['latitude'].tolist()
            times = cleaned_data['time'].tolist()
            speeds = cleaned_data['speed'].tolist()
            directions = cleaned_data['direction'].tolist()
            altitudes = cleaned_data['altitude'].tolist()
            tags = cleaned_data['标签'].tolist()
            point_index = []
            for h in range(len(tags)):
                point_index.append(traj_name+'_'+str(h))
            same_data = []
            for m in range(len(cleaned_data)):
                same_data.append([point_index[m], times[m], lon[m], lat[m], speeds[m], directions[m], altitudes[m], tags[m]])
            ddaad = pd.DataFrame(same_data)
            ddaad.columns = ['index', 'time', 'longitude', 'latitude', 'speed', 'direction', 'altitude', 'tag']
            print(clean_data_save_path + 'cleaned_' +all_file[i])
            ddaad.to_excel(clean_data_save_path + 'cleaned_' +all_file[i], index=False)
    return ddaad

if __name__ == '__main__':
    #数据清洗函数
    ddaad = data_cleaning_paddy()
    print('paddy clean ending')
    ddaad = data_cleaning_wheat()
    print('wheat clean ending')
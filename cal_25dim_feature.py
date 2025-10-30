import os
import datetime
#from datetime import datetime
import pandas as pd
import numpy as np
cit1=5
cit2=50

def get_5_feature(path, final_path):
    files = os.listdir(path)
    for i in range(len(files)):
        print(i)
        size = int(os.path.getsize(path + "/" + files[i]))
        if size == 0:
            continue
        data = pd.read_excel(path + "/" + files[i])
        try:
            lon_list = data['经度']
            lat_list = data['纬度']
            speed_list = data['速度']
            angular_list = data['方向']
            time_list = data['时间']
            '''
            try:
                tag_list = data['标签']
            except:
                tag_list = data['标记']
            '''
        except:
            lon_list=data['longitude']
            lat_list=data['latitude']
            speed_list = data['speed']
            angular_list = data['direction']
            time_list = data['time']
        time_format = '%Y%m%d%H%M%S'
        for j in range(len(time_list)):
            time_list[j] = datetime.datetime.strptime( str(time_list[j]), time_format)
        speed_diff = [0 for x in range(len(speed_list))]
        time_diff = [0 for x in range(len(speed_list))]
        acclec = [0 for x in range(len(speed_list))]
        angular_speed_diff = [0 for x in range(len(speed_list))]
        angular_acclec = [0 for x in range(len(speed_list))]
        angular_diff = [0 for x in range(len(speed_list))]
        angular_speed = [0 for x in range(len(speed_list))]
        for j in range(len(list(speed_list))):
            if j == 0:
                speed_diff[j] = 0
                time_diff[j] = 0
                acclec[j] = 0
            else:
                speed_diff[j] = speed_list[j] - speed_list[j - 1]
                try:
                    d1 = datetime.datetime.strptime(str(time_list[j - 1]), '%Y/%m/%d %H:%M:%S')
                    d2 = datetime.datetime.strptime(str(time_list[j]), '%Y/%m/%d %H:%M:%S')
                except:
                    d1 = datetime.datetime.strptime(str(time_list[j - 1]), '%Y-%m-%d %H:%M:%S')
                    d2 = datetime.datetime.strptime(str(time_list[j]), '%Y-%m-%d %H:%M:%S')
                time_diff[j] = (d2 - d1).seconds
                acclec[j] = float(speed_diff[j]) / time_diff[j]
        for temp in range(len(list(speed_list))):
            if temp == 0:
                angular_speed[temp] = 0
            else:
                if speed_list[temp] == 0:
                    angular_speed[temp] = 0
                else:
                    angular_speed[temp] = angular_list[temp] / time_diff[temp]
        for k in range(len(list(speed_list))):
            if k == 0:
                angular_speed_diff[k] = 0
                time_diff[k] = 0
                angular_acclec[k] = 0
            else:
                angular_speed_diff[k] = angular_speed[k] - angular_speed[k - 1]
                try:
                    d1 = datetime.datetime.strptime(str(time_list[k - 1]), '%Y/%m/%d %H:%M:%S')
                    d2 = datetime.datetime.strptime(str(time_list[k]), '%Y/%m/%d %H:%M:%S')
                except:
                    d1 = datetime.datetime.strptime(str(time_list[k - 1]), '%Y-%m-%d %H:%M:%S')
                    d2 = datetime.datetime.strptime(str(time_list[k]), '%Y-%m-%d %H:%M:%S')
                time_diff[k] = (d2 - d1).seconds
                angular_acclec[k] = angular_speed_diff[k] / time_diff[k]
        for l in range(len(list(speed_list))):
            if l == 0:
                angular_diff[l] = 0
            else:
                angular_diff[l] = angular_list[l] - angular_list[l - 1]
        columns = ['lon','lat','speed', 'acceleration', 'angular_speed', 'angular_acceleration', 'angle_diff']
        df = pd.DataFrame(columns=columns)
        df["lon"] = lon_list
        df["lat"] = lat_list
        df["speed"] = speed_list
        df['acceleration'] = acclec
        df['angular_speed'] = angular_speed
        df['angular_acceleration'] = angular_acclec
        df['angle_diff'] = angular_diff
        #df['tag'] = tag_list
        x = pd.DataFrame(df)
        index1 = files[i].index('.xlsx')
        name = files[i][0:index1]+'_feature5.xlsx'
        x.to_excel(final_path+"/" + name, index=False)

def get_median(data):
   data = sorted(data)
   size = len(data)
   if size % 2 == 0: # 判断列表长度为偶数
    #median = (data[size//2]+data[size//2-1])/2
    median =  data[size//2]
    data[0] = median
   if size % 2 == 1: # 判断列表长度为奇数
    median = data[(size-1)//2]
    data[0] = median
   return data[0]

def med(num,lst):
    med=[0 for x in range(len(lst))]
    med[0]=lst[0]
    for i in range(1,len(lst)):
        if(i<=num):
            med[i]=get_median(lst[:i+1])
        else:
            med[i] = get_median(lst[i-num:i+1])
    return med
def SD(num,lst):
    SD=[0 for x in range(len(lst))]
    SD[0]=np.std([lst[0]])
    for i in range(1,len(lst)):
        if(i<=num):
            SD[i]=np.std(lst[:i+1])
        else:
            SD[i] = np.std(lst[i-num:i+1])
    return SD
if __name__=="__main__":

    path = "data/trajectory_data/raw_paddy_cleaned"
    if not os.path.exists(path):
        os.mkdir(path)
    middle_path = path
    final_path=path+"_feature_5"
    if not os.path.exists(final_path):
        os.mkdir(final_path)
    end_path=path+"_feature_25"
    if not os.path.exists(end_path):
        os.mkdir(end_path)

    get_5_feature(middle_path, final_path)
    
    
    files=os.listdir(final_path)
    error_num = 0
    for i in range(len(files)):
        print(i)
        print(files[i])
        #angular_speed	angular_acceleration	angle_diff
        data=pd.read_excel(final_path+"/"+files[i])
        size = int(os.path.getsize(final_path + "/" + files[i]))
        if size == 0:
            print(size)
            continue
        ang_speed_list = data['angular_speed']
        ang_acc_list = data['angular_acceleration']
        ang_diff_list = data['angle_diff']
        #print(np.sum(ang_speed_list), np.sum(ang_acc_list), np.sum(ang_diff_list))
        # if np.sum(ang_speed_list)==0 or np.sum(ang_acc_list)==0 or np.sum(ang_diff_list)==0:
        #     print('ang_speed_list:', np.sum(ang_speed_list))
        #     print('ang_acc_list:', np.sum(ang_acc_list))
        #     print('ang_diff_list:', np.sum(ang_diff_list))
        #     error_num += 1
        #     continue

        lon_list = data['lon']
        lat_list = data['lat']

        speed_list=data['speed']
        speed_med_5=med(cit1,speed_list)
        speed_med_20=med(cit2,speed_list)
        speed_SD_5=SD(cit1,speed_list)
        speed_SD_20 = SD(cit2, speed_list)

        acc_list = data['acceleration']
        acc_med_5 = med(cit1, acc_list)
        acc_med_20 = med(cit2, acc_list)
        acc_SD_5 = SD(cit1, acc_list)
        acc_SD_20 = SD(cit2, acc_list)

        
        ang_speed_med_5 = med(cit1, ang_speed_list)
        ang_speed_med_20 = med(cit2, ang_speed_list)
        ang_speed_SD_5 = SD(cit1, ang_speed_list)
        ang_speed_SD_20 = SD(cit2, ang_speed_list)

        
        ang_acc_med_5 = med(cit1, ang_acc_list)
        ang_acc_med_20 = med(cit2, ang_acc_list)
        ang_acc_SD_5 = SD(cit1, ang_acc_list)
        ang_acc_SD_20 = SD(cit2, ang_acc_list)

        ang_diff_med_5 = med(cit1, ang_diff_list)
        ang_diff_med_20 = med(cit2, ang_diff_list)
        ang_diff_SD_5 = SD(cit1, ang_diff_list)
        ang_diff_SD_20 = SD(cit2, ang_diff_list)
        columns = ['speed', 'speed_med_5', 'speed_med_20','speed_SD_5','speed_SD_20',
                   'acceleration','acceleration_med_5', 'acceleration_med_20','acceleration_SD_5','acceleration_SD_20',
                   'angular_speed','angular_speed_med_5', 'angular_speed_med_20','angular_speed_SD_5','angular_speed_SD_20',
                   'angular_acceleration','angular_acceleration_med_5','angular_acceleration_med_20','angular_acceleration_SD_5','angular_acceleration_SD_20',
                   'angle_diff','angle_diff_med_5','angle_diff_med_20','angle_diff_SD_5','angle_diff_SD_20',
                   'lon','lat']
        df = pd.DataFrame(columns=columns)
        df["speed"] = speed_list
        df["speed_med_5"] = speed_med_5
        df["speed_med_20"] = speed_med_20
        df["speed_SD_5"] = speed_SD_5
        df["speed_SD_20"] = speed_SD_20

        df['acceleration'] = acc_list
        df['acceleration_med_5'] = acc_med_5
        df['acceleration_med_20'] = acc_med_20
        df['acceleration_SD_5'] = acc_SD_5
        df['acceleration_SD_20'] = acc_SD_20



        df['angular_speed'] = ang_speed_list
        df['angular_speed_med_5'] = ang_speed_med_5
        df['angular_speed_med_20'] = ang_speed_med_20
        df['angular_speed_SD_5'] = ang_speed_SD_5
        df['angular_speed_SD_20'] = ang_speed_SD_20



        df['angular_acceleration'] = ang_acc_list
        df['angular_acceleration_med_5'] = ang_acc_med_5
        df['angular_acceleration_med_20'] = ang_acc_med_20
        df['angular_acceleration_SD_5'] = ang_acc_SD_5
        df['angular_acceleration_SD_20'] = ang_acc_SD_20



        df['angle_diff'] = ang_diff_list
        df['angle_diff_med_5'] = ang_diff_med_5
        df['angle_diff_med_20'] = ang_diff_med_20
        df['angle_diff_SD_5'] = ang_diff_SD_5
        df['angle_diff_SD_20'] = ang_diff_SD_20

        df['lon']=lon_list
        df['lat']=lat_list
        x = pd.DataFrame(df)
        files[i] = files[i].replace('feature5', 'feature25')
        #print('files[i]:', files[i])
        x.to_excel(end_path+"/"+ files[i], index=False)
    print('error_num:', error_num)
    
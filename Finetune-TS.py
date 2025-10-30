import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pickle
import os
import tqdm
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from self_model.model.bert import BERT
from self_model.model.language_model import BERTLM_detction
import os
import random
import torch.nn.functional as F
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import warnings
import time
import statistics
from argparse import ArgumentParser
import json
import argparse
parser = ArgumentParser()
parser.add_argument('--config', default='config_file/point_detection.json', type=str)
arg_ = parser.parse_args()
with open(arg_.config) as config_file:
    args = json.load(config_file)
args = argparse.Namespace(**args)
warnings.filterwarnings('ignore')

EPOCH = args.epochs
lr = args.lr
betas = (0.9, 0.999)
weight_decay = 1e-4
cuda = True
batch = args.batch_size
splot_len = args.splot_len
hidden_size = args.hidden_size
LSTM_hidden_size = args.LSTM_hidden_size
n_layers_size = args.n_layers_size
attn_heads_size = args.attn_heads_size
split_path = args.split_path
LSTM_layers = args.LSTM_layers
pretrain_model_path = args.pretrain_model_path
device = torch.device("cuda:0" if cuda else "cpu")

def seed_torch(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

class Traj_BERT_Dataset(Dataset):
    def __init__(self, files_path, spots_len, split_path, kfold=0, embed_size=27, type = 'train'):
        print('初始化............')
        
        filename = 'traj_feature_wheat_二分类_2间隔_e300.pkl'
        print('use filename:', filename)
        with open(filename, "rb") as file:
            self.bert_feature = pickle.load(file, encoding="latin1")
        self.processed_data = None
        self.files_path = files_path
        self.spots_len = spots_len
        self.embed_size = embed_size
        self.cls = np.array([0.001 for _ in range(embed_size)])
        self.sep = np.array([0.002 for _ in range(embed_size)])
        self.pad = np.array([0 for _ in range(embed_size)])
        self.train_data_25 = []
        self.train_data_bert = []
        self.train_tag = []
        self.cnndata = []
        self.data = []
        self.count = 0
        self.segment_keys = None

        with open(args.segment_vocab_path, 'rb') as file:
            self.segment_vocabs = pickle.load(file, encoding='latin1')
        #print('self.segment_vocabs:',self.segment_vocabs.keys())
        with open(args.segment_len_path, 'rb') as file:
            self.segment_lens = pickle.load(file, encoding='latin1')

        with open(split_path, 'rb') as file:
            data = pickle.load(file, encoding='latin1')
            train = data['train']
            valid = data['valid']
            test = data['test']
        self.train_vid = []
        self.valid_vid = []
        self.test_vid = []
        segment_files = os.listdir(args.segment_path)
        train_tag = []
        for i in range(len(train[kfold])):
            train_tag.append(train[kfold][i].replace('.xlsx', ''))
        test_tag = []
        for i in range(len(test[kfold])):
            test_tag.append(test[kfold][i].replace('.xlsx', ''))
        valid_tag = []
        for i in range(len(valid[kfold])):
            valid_tag.append(valid[kfold][i].replace('.xlsx', ''))
        for i in range(len(segment_files)):
            loc1 = segment_files[i].find('_feature25')
            single = segment_files[i][0:loc1]
            if single in train_tag:
                self.train_vid.append(segment_files[i])
            if single in valid_tag:
                self.valid_vid.append(segment_files[i])
            if single in test_tag:
                self.test_vid.append(segment_files[i])
        print('self.train_vid:',len(self.train_vid))
        print('self.valid_vid:',len(self.valid_vid))
        print('self.test_vid:',len(self.test_vid))
        i = 0
        pt_exists = False
        if pt_exists:
            print('loading data from pt')
            datas = torch.load('trajectory_25_sample_2000.pt')
            self.result = datas['information']
        else:
            if type == "train":
                temp_files = self.train_vid
                self.segment_keys = temp_files
            elif type == "valid":
                temp_files = self.valid_vid
                self.segment_keys = temp_files
            else:
                temp_files = self.test_vid
                self.segment_keys = temp_files
            #temp_files = temp_files[0:2]
            for file_name in temp_files:
                #file_25_name = file_name.replace('.xlsx', '_feature5.xlsx')
                # file_25_name = file_name.replace('.xlsx', '_feature25.xlsx')
                # file_name = file_name.replace('.xlsx', '_feature25.xlsx')
                file_25_name = file_name
                file_name = file_name
                norm_input_file_path = args.norm_input_file_path + file_25_name
                input_file_path = args.input_file_path + file_name
                norm_df = pd.read_excel(norm_input_file_path)
                df = pd.read_excel(input_file_path)
                tags = norm_df['tag'].tolist()
                lon = norm_df['lon'].tolist()
                lat = norm_df['lat'].tolist()
                df = df.drop(columns=['tag',])
                norm_df = norm_df.drop(columns=['tag',]) #, 'lon', 'lat'
                '''
                std = StandardScaler()
                df = std.fit_transform(df)
                '''
                segment_num = int(len(norm_df)/self.spots_len)
                norm_df = np.array(norm_df)
                df = np.array(df)
                
                for num_index in range(segment_num):
                    #print('max:', (num_index+1)*self.spots_len)
                    norm_segment = norm_df[num_index*self.spots_len:(num_index+1)*self.spots_len]
                    segment = df[num_index*self.spots_len:(num_index+1)*self.spots_len]
                    segment_tag = tags[num_index*self.spots_len:(num_index+1)*self.spots_len]
                    self.train_data_25.append(segment)
                    self.train_tag.append(segment_tag)
                    '''
                    bert_segment = []
                    for x in range(num_index*self.spots_len, (num_index+1)*self.spots_len):
                        bert_segment.append(self.bert_feature[file_name+'_'+str(x)])
                    '''
                    #训练水稻数据没有预训练的bertfeature,拿这个替换
                    bert_segment = []
                    for x in range(num_index*self.spots_len, (num_index+1)*self.spots_len):
                        bert_segment.append(segment[0])
                    self.train_data_bert.append(bert_segment)

                    lon_dist = []
                    lat_dist = []
                    new_lon = lon[num_index*self.spots_len:(num_index+1)*self.spots_len]
                    new_lat = lat[num_index*self.spots_len:(num_index+1)*self.spots_len]
                    for i in range(len(new_lon)-1):
                        lon_dist.append(abs(new_lon[i+1]-new_lon[i]))
                        lat_dist.append(abs(new_lat[i+1]-new_lat[i]))
                    avg_lon_dist = statistics.median(lon_dist)
                    avg_lat_dist = statistics.median(lat_dist)
                    multiple = 2.2
                    avg_lon_dist = multiple * avg_lon_dist
                    avg_lat_dist = multiple * avg_lat_dist
                    CNN_feature = []
                    padding = []
                    for i in range(len(norm_segment[0])):
                        padding.append(0)
                    cnn_len = args.cnn_len
                    for i in range(len(norm_segment)):
                        num = 0
                        feature = []
                        for j in range(len(new_lon)):
                            if new_lon[j] <= (new_lon[i] + avg_lon_dist) and new_lon[j] >= (new_lon[i] - avg_lon_dist) \
                               and new_lat[j] <= (new_lat[i] + avg_lat_dist) and new_lat[j] >= (new_lat[i] - avg_lat_dist):
                                num += 1
                                feature.append(norm_segment[j])
                        #print('len feature:', len(feature))
                        if num <= cnn_len:
                            for i in range(cnn_len-num):
                                feature.append(padding)
                        else:
                            feature = feature[0:cnn_len]
                        '''
                        sta = MinMaxScaler(feature_range=[0, 224])
                        feature = np.array(feature)
                        feature = sta.fit_transform(feature)
                        #feature = np.array(abs(np.array(feature)) * 224).astype(int)
                        feature = feature.astype(int)
                        '''
                        CNN_feature.append(feature)
                    self.cnndata.append(CNN_feature)
                i += 1
            #total_data = [self.train_data_25, self.train_tag, self.train_data_bert]
            # save_path = 'extract_feature/trajectory_25_bert'+type+'.npy'
            # np.save(save_path, np.array(total_data), allow_pickle=True)
            # print('************************************')
    def __len__(self):
        return max(len(self.train_data_25) - 1, 0)
    
    def __getitem__(self, item):
        segment = self.train_data_25[item]
        bert_segment = self.train_data_bert[item]
        segment_point_label = self.train_tag[item]
        segment_label = ([1 for _ in range(self.spots_len)])
        label_bool = [1 if sum(_) != 0 else 0 for _ in segment]
        cnn_data = self.cnndata[item]
        seg_key = self.segment_keys[item]
        vocabs_vector = self.segment_vocabs[seg_key]
        vocabs_len = self.segment_lens[seg_key]
        output = {"segment": segment, #经过各种mask，数据填充的向量
                  "vocabs_vector": vocabs_vector,
                  "vocabs_len": vocabs_len,
                  "bert_segment": bert_segment,
                  "cnn_data":cnn_data,
                  "segment_point_label": segment_point_label,
                  "segment_label": segment_label,
                  'label_bool':label_bool
                  }
        data_items = {}
        for key, value in output.items():
            if key != "vocabs_vector" and key != "vocabs_len":
                if key == 'segment_point_label' or key == 'segment_label':
                    data_items[key] = torch.LongTensor(value)
                else:
                    data_items[key] = torch.FloatTensor(value)
            else:
                data_items[key] = value
        return data_items

class detection_model(nn.Module):
    def __init__(self, with_cuda: bool = True, feature_size = 128, class_num = 2):
        super().__init__()
        cuda_condition = torch.cuda.is_available() and with_cuda
        #self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        self.bert = bert
        self.BERTLM = BERTLM_detction(bert, None)#.to(self.device)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(512, 2)
        self.LSTM = nn.LSTM(512+128, LSTM_hidden_size, num_layers = LSTM_layers, bidirectional = True, batch_first=True) # + 25 #, bidirectional = True, batch_first=True
    def forward(self, x, vocabs_vector, vocabs_len, bert_segment, cnn_data, segment_label, label_bool):
        # self.BERTLM.eval()
        # self.bert.eval()
        #print('x:', x.shape)
        output, sim_features, trajs_hidden = self.BERTLM(x, vocabs_vector, vocabs_len, cnn_data, segment_label, label_bool)
        # output_1 = output[:, 1:513, :] #1025
        # output_2 = output[:, 514:1026, :] #1025
        #print('cnn_data:', cnn_data.shape)
        #x = self.norm(x)
        #feature = torch.cat([x, output], dim = 2)
        #feature = output
        feature = torch.cat([trajs_hidden, output], dim = 2)
        output, _ = self.LSTM(feature)
        #output = feature
        #output = self.linear(output)
        hidden = F.relu(output)
        segment_hidden = self.dropout(hidden)
        #print('segment_hidden:', segment_hidden.shape)
        log_prob = F.log_softmax(self.linear(segment_hidden), 2)  # seq_len, batch, n_classes
        #log_prob = self.linear(hidden) # seq_len, batch, n_classes
        return log_prob, sim_features

# data_path = "paddy_wheat_data/wheat_150_cleaned_feature_5/"
# split_path = "paddy_wheat_data/split/wheat_1_data_split.pkl"
data_path = "xxxxx"

bert = BERT(spots_len=splot_len, size = 25, hidden=hidden_size, n_layers=n_layers_size, attn_heads=attn_heads_size)
bert = bert.cuda()
loss_function = nn.CrossEntropyLoss() #weight = class_weights

def info_nce_loss(features, batch_size):
    labels = torch.cat([torch.tensor([i]*2) for i in range(batch_size)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(features.device)
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

    logits = logits / 0.07
    return logits, labels

def train_model(model, optim, epoch):
    model.train()
    max_f1score = 0
    data_iter = train_data_loader
    loss_list=[]
    preds = []
    labels = []
    for data__ in data_iter:
        optim.zero_grad()
        data = {key: value.to(device) for key, value in data__.items()}
        segment = data["segment"]
        bert_segment = data['bert_segment']
        cnn_data = data['cnn_data']
        segment_label = data["segment_label"]
        label_bool = data['label_bool']
        segment_point_label = data['segment_point_label']
        batch_size = data['vocabs_vector'].shape[0]
        data['vocabs_vector'] = data['vocabs_vector'].view(-1, data['vocabs_vector'].shape[-1])
        data['vocabs_len'] = data['vocabs_len'].view(-1)
        output, sim_features = model(segment, data['vocabs_vector'], data['vocabs_len'], bert_segment, cnn_data, segment_label, label_bool)
        segment_point_label = segment_point_label.view(-1)
        output = output.contiguous().view(-1, output.size()[2])
        logits, temp_labels = info_nce_loss(sim_features, batch_size)
        sim_loss = loss_function(logits, temp_labels)
        CR_loss = loss_function(output, segment_point_label)
        loss = 0.5 * sim_loss + 0.5 * CR_loss
        loss.backward()
        optim.step()
        loss_list.append(loss.item())
        pred_ = torch.argmax(output, 1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(segment_point_label.data.cpu().numpy())
    
    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    f1score = round(f1_score(labels, preds, average='macro') * 100, 2)
    #print('epoch:', epoch, 'loss:', np.sum(loss_list)/len(loss_list), 'avg_accuracy:', avg_accuracy, 'f1score:', f1score)
    return round(np.sum(loss_list)/len(loss_list), 6), avg_accuracy, f1score, labels, preds

def eval_model(model, optim, epoch, data_loader):
    model.eval()
    # data_iter = tqdm.tqdm(enumerate(data_loader),
    #                         desc="EP_%s:%d" % (str_code, epoch),
    #                         total=len(train_data_loader),
    #                         bar_format="{l_bar}{r_bar}")
    data_iter = data_loader
    loss_list=[]
    preds = []
    labels = []
   
    for data__ in data_iter:
        
        optim.zero_grad()
        data = {key: value.to(device) for key, value in data__.items()}
        segment = data["segment"]
        segment_label = data["segment_label"]
        label_bool = data['label_bool']
        segment_point_label = data['segment_point_label']
        bert_segment = data['bert_segment']
        cnn_data = data['cnn_data']
        batch_size = data['vocabs_vector'].shape[0]
        data['vocabs_vector'] = data['vocabs_vector'].view(-1, data['vocabs_vector'].shape[-1])
        data['vocabs_len'] = data['vocabs_len'].view(-1)
        # print('segment_point_label:', segment_point_label.shape)
        # print('segment:', segment.shape)
        output, sim_features = model(segment, data['vocabs_vector'], data['vocabs_len'], bert_segment, cnn_data, segment_label, label_bool)
        segment_point_label = segment_point_label.view(-1)
        output = output.contiguous().view(-1, output.size()[2])
        logits, temp_labels = info_nce_loss(sim_features, batch_size)
        sim_loss = loss_function(logits, temp_labels)
        CR_loss = loss_function(output, segment_point_label)
        loss = 0.5 * sim_loss + 0.5 * CR_loss
        loss_list.append(loss.item())
        pred_ = torch.argmax(output, 1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(segment_point_label.data.cpu().numpy())
    
    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    f1score = round(f1_score(labels, preds, average='macro') * 100, 2)
    #print('epoch:', epoch, 'loss:', np.sum(loss_list)/len(loss_list), 'avg_accuracy:', avg_accuracy, 'f1score:', f1score)

    return round(np.sum(loss_list)/len(loss_list), 6), avg_accuracy, f1score, labels, preds

if __name__ == '__main__':
    
    pid = os.getpid()
    print("当前进程的PID为:", pid)
    seed = args.seed
    print('seed:', seed)
    seed_torch(seed)
    avg_best_f1score = 0
    avg_max_f1score = 0
    fold = int(args.fold_1)
    print('fold:', fold, 'lr:', lr, 'batch:', batch, 'splot_len:', splot_len)
    total_pred = []
    total_label = []
    model = detection_model(with_cuda = True, feature_size = 128, class_num = 2)
    model.cuda()
    #model = nn.DataParallel(model, device_ids=[0, 1])
    #model.BERTLM.load_state_dict(torch.load(pretrain_model_path))
    #print(next(model.parameters()).device)
    temp = time.time()
    best_label = None
    best_preds = None
    best_test_label = None
    best_test_preds = None
    valid_flag = 0
    max_test = 0
    #print('weight load success')
    optim = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_dataset = Traj_BERT_Dataset(files_path=data_path, spots_len=splot_len, split_path=split_path, kfold=fold, embed_size=25, type = 'train')
    train_data_loader = DataLoader(train_dataset, batch_size=batch, num_workers= args.num_workers)
    valid_dataset = Traj_BERT_Dataset(files_path=data_path, spots_len=splot_len, split_path=split_path, kfold=fold, embed_size=25, type = 'valid')
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch, num_workers= args.num_workers)
    test_dataset = Traj_BERT_Dataset(files_path=data_path, spots_len=splot_len, split_path=split_path, kfold=fold, embed_size=25, type = 'test')
    test_data_loader = DataLoader(test_dataset, batch_size=batch, num_workers= args.num_workers)
    print('len train_data_loader:', len(train_data_loader))
    print('len valid_data_loader:', len(valid_data_loader))
    print('len test_data_loader:', len(test_data_loader))
    for e in range(EPOCH):
        if (e + 1) % 10 == 0:
            print('fold:', fold, ' e:', e)
        train_loss, train_acc, train_f1score, _, _ = train_model(model,optim, e)
        valid_loss, valid_acc, valid_f1score, _, _  = eval_model(model, optim, e, valid_data_loader)
        test_loss, test_acc, test_f1score, labels, preds  = eval_model(model, optim, e, test_data_loader)
        #print(classification_report(labels, preds, digits=4))
        if valid_f1score > valid_flag:
            best_test_f1score = test_f1score
            best_epoch = e
            valid_flag = valid_f1score
            best_label = labels
            best_preds = preds
        if test_f1score > max_test:
            max_test = test_f1score
            best_test_label = labels
            best_test_preds = preds
    avg_best_f1score += best_test_f1score
    avg_max_f1score += max_test
    print('best_epoch:', best_epoch)
    print('best_test_f1score:', best_test_f1score)
    print('max_test:', max_test)
    print(classification_report(best_test_label, best_test_preds, digits=4))
    fold_time = time.time() - temp
    print('fold_time:', fold_time)
    result = classification_report(best_test_label, best_test_preds, digits=10, output_dict=True)
    result_pkl = {}
    result_pkl['test_0_precision'] = float(result['0']['precision'])
    result_pkl['test_0_recall'] = float(result['0']['recall'])
    result_pkl['test_0_f1score'] = float(result['0']['f1-score'])

    result_pkl['test_1_precision'] = float(result['1']['precision'])
    result_pkl['test_1_recall'] = float(result['1']['recall'])
    result_pkl['test_1_f1score'] = float(result['1']['f1-score'])

    result_pkl['test_accuracy'] = float(result['accuracy'])

    result_pkl['test_macro_precision'] = float(result['macro avg']['precision'])
    result_pkl['test_macro_recall'] = float(result['macro avg']['recall'])
    result_pkl['test_macro_f1score'] = float(result['macro avg']['f1-score'])

    result_pkl['test_weight_precision'] = float(result['weighted avg']['precision'])
    result_pkl['test_weight_recall'] = float(result['weighted avg']['recall'])
    result_pkl['test_weight_f1score'] = float(result['weighted avg']['f1-score'])
    with open('shell_result/shell_'+str(fold)+'_result.pkl', 'wb') as f:
        pickle.dump(result_pkl, f)
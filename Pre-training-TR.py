import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pickle
import os
import tqdm
from self_model.model.language_model import BERTLM_detction
from self_model.model.language_model import BERTLM
from self_model.model.bert import BERT
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import f1_score, accuracy_score, classification_report
import os
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch.nn.functional as F
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import warnings
warnings.filterwarnings('ignore')

EPOCH = 150
fold = 1
lr = 1e-3
betas = (0.9, 0.999)
weight_decay = 1e-4
cuda = True
batch = 32
splot_len = 512
hidden_size = 256
LSTM_hidden_size = 256
n_layers_size = 2
attn_heads_size = 4
pretrain_model_path = "model_1_bert/traj_BERT_model_12.pkl"
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
        filename = 'traj_classification_dbscan_2.pkl'
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

        self.data = []
        self.count = 0

        with open(split_path, 'rb') as file:
            data = pickle.load(file, encoding='latin1')
            train = data['train']
            valid = data['valid']
            test = data['test']
        self.train_vid = train[kfold]
        self.valid_vid = valid[kfold]
        self.test_vid = test[kfold]
        i = 0
        pt_exists = False
        if pt_exists:
            print('loading data from pt')
            datas = torch.load('trajectory_25_sample_2000.pt')
            self.result = datas['information']
        else:
            if type == "train":
                temp_files = self.train_vid
            elif type == "valid":
                temp_files = self.valid_vid
            else:
                temp_files = self.test_vid
            for file_name in temp_files:
                print(i)
                # if i == 100:
                #     break
                file_name = file_name.replace('.xlsx', '_feature25.xlsx')
                #input_file_path = 'paddy_wheat_data/wheat_150_cleaned_feature_25_normalized/' + file_name
                input_file_path = 'paddy_wheat_data/wheat_150_cleaned_feature_25/' + file_name
                df = pd.read_excel(input_file_path)
                tags = df['tag'].tolist()
                df = df.drop(columns=['lon', 'lat', 'tag'])
                #data = data.drop(columns=['lon', 'lat', 'tag'])
                std = StandardScaler()
                df = std.fit_transform(df)
                #df = df.drop(columns=['tag'])
                '''
                index_list = []
                for h in range(df_len):
                    index_list.append(h/df_len)
                df['index_list'] = index_list
                '''
                '''
                df = np.array(df).tolist()
                print('len df:',len(df))
                segment_num = int(len(df)/self.spots_len)
                for j in range(len(df) - segment_num*self.spots_len):
                    df.append(self.pad)
                '''
                segment_num = int(len(df)/self.spots_len)
                df = np.array(df)
                for num_index in range(segment_num):
                    #print('max:', (num_index+1)*self.spots_len)
                    segment = df[num_index*self.spots_len:(num_index+1)*self.spots_len]
                    segment_tag = tags[num_index*self.spots_len:(num_index+1)*self.spots_len]
                    self.train_data_25.append(segment)
                    self.train_tag.append(segment_tag)
                    bert_segment = []
                    #temp_name = file_name.replace('feature5', 'feature25')
                    for x in range(num_index*self.spots_len, (num_index+1)*self.spots_len):
                        bert_segment.append(self.bert_feature[file_name+'_'+str(x)])
                    self.train_data_bert.append(bert_segment)
                    
                i += 1
            '''
            save_data = {"information": self.result }
            torch.save(save_data, 'trajectory_25_sample_2000.pt')
            print('save success')
            '''
            print('************************************')
    def __len__(self):
        return max(len(self.train_data_25) - 1, 0)
    
    def __getitem__(self, item):
        
        segment = self.train_data_25[item]
        bert_segment = self.train_data_bert[item]
        segment_point_label = self.train_tag[item]
        segment_label = ([1 for _ in range(self.spots_len + 2)])
        label_bool = [1 if sum(_) != 0 else 0 for _ in segment]
        output = {"segment": segment, #经过各种mask，数据填充的向量
                  "bert_segment": bert_segment,
                  "segment_point_label": segment_point_label,
                  "segment_label": segment_label,
                  'label_bool':label_bool
                  }
        data_items = {}
        for key, value in output.items():
            if key == 'segment_point_label' or key == 'segment_label':
                data_items[key] = torch.LongTensor(value)
            else:
                data_items[key] = torch.FloatTensor(value)
        return data_items

class detection_model(nn.Module):
    def __init__(self, with_cuda: bool = True, feature_size = 128, class_num = 2):
        super().__init__()
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        # self.bert = bert
        # self.BERTLM = BERTLM_detction(bert, None).to(self.device)
        self.norm = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(512, 2)
        self.linear_2 = nn.Linear(64, 25)
        #self.linear_1 = nn.Linear(25, 25)
        self.LSTM = nn.LSTM(50, LSTM_hidden_size, num_layers = 2, bidirectional = True, batch_first=True) # + 25 #, bidirectional = True, batch_first=True
        #self.LSTM = nn.GRU(25 + 64, LSTM_hidden_size, 2, bidirectional = True, batch_first=True) # + 25
    def forward(self, x, bert_segment, segment_label, label_bool):
        # self.BERTLM.eval()
        # self.bert.eval()
        # output = self.BERTLM(x, segment_label, label_bool)
        # output_1 = output[:, 1:513, :] #1025
        # output_2 = output[:, 514:1026, :] #1025
        # print('x:', x.shape)
        # print('bert_segment:', bert_segment.shape)
        #
        #print('sum:', torch.sum(bert_segment))
        # x = self.linear_1(x)
        # print('bert_segment:', bert_segment)
        #x = self.norm(x)
        #print('sum x:', torch.sum(x))
        bert_segment = self.linear_2(bert_segment)
        feature = torch.cat([x, bert_segment], dim = 2)
        output, _ = self.LSTM(feature)
        #output = self.linear(output)
        hidden = F.relu(output)
        segment_hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.linear(segment_hidden), 2)  # seq_len, batch, n_classes
        return log_prob

data_path = "paddy_wheat_data/wheat_150_cleaned_feature_5/"
split_path = "paddy_wheat_data/split/wheat_1_data_split.pkl"
train_dataset = Traj_BERT_Dataset(files_path=data_path, spots_len=splot_len, split_path=split_path, kfold=fold, embed_size=25, type = 'train')
train_data_loader = DataLoader(train_dataset, batch_size=batch, num_workers=6)
valid_dataset = Traj_BERT_Dataset(files_path=data_path, spots_len=splot_len, split_path=split_path, kfold=fold, embed_size=25, type = 'valid')
valid_data_loader = DataLoader(valid_dataset, batch_size=batch, num_workers=6)
test_dataset = Traj_BERT_Dataset(files_path=data_path, spots_len=splot_len, split_path=split_path, kfold=fold, embed_size=25, type = 'test')
test_data_loader = DataLoader(test_dataset, batch_size=batch, num_workers=6)
print('len train_data_loader:', len(train_data_loader))
print('len valid_data_loader:', len(valid_data_loader))
print('len test_data_loader:', len(test_data_loader))

train = True
str_code = "train" if train else "test"
#bert = BERT(spots_len=splot_len, size = 25, hidden=hidden_size, n_layers=n_layers_size, attn_heads=attn_heads_size)
model = detection_model(with_cuda = True, feature_size = 128, class_num = 2)
model.cuda()
#model = nn.DataParallel(model, device_ids=[0, 1])
#print(model)
#model.load_state_dict(torch.load(pretrain_model_path))
optim = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
class_weights = torch.FloatTensor([7.0, 1.0]).cuda()
loss_function = nn.CrossEntropyLoss() #weight = class_weights
def train_model(epoch):
    model.train()
    max_f1score = 0
    data_iter = tqdm.tqdm(enumerate(train_data_loader),
                            desc="EP_%s:%d" % (str_code, epoch),
                            total=len(train_data_loader),
                            bar_format="{l_bar}{r_bar}")
    loss_list=[]
    preds = []
    labels = []
    for i, data__ in data_iter:
        optim.zero_grad()
        data = {key: value.to(device) for key, value in data__.items()}
        segment = data["segment"]
        bert_segment = data['bert_segment']
        segment_label = data["segment_label"]
        label_bool = data['label_bool']
        segment_point_label = data['segment_point_label']
        # print('segment_point_label:', segment_point_label.shape)
        # print('segment:', segment.shape)
        output = model(segment, bert_segment, segment_label, label_bool)
        segment_point_label = segment_point_label.view(-1)
        output = output.contiguous().view(-1, output.size()[2])
        loss = loss_function(output, segment_point_label)
        
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

def eval_model(epoch, data_loader):
    model.eval()
    data_iter = tqdm.tqdm(enumerate(data_loader),
                            desc="EP_%s:%d" % (str_code, epoch),
                            total=len(train_data_loader),
                            bar_format="{l_bar}{r_bar}")
    loss_list=[]
    preds = []
    labels = []
    for i, data__ in data_iter:
        optim.zero_grad()
        data = {key: value.to(device) for key, value in data__.items()}
        segment = data["segment"]
        segment_label = data["segment_label"]
        label_bool = data['label_bool']
        segment_point_label = data['segment_point_label']
        bert_segment = data['bert_segment']
        # print('segment_point_label:', segment_point_label.shape)
        # print('segment:', segment.shape)
        output = model(segment, bert_segment, segment_label, label_bool)
        segment_point_label = segment_point_label.view(-1)
        output = output.contiguous().view(-1, output.size()[2])
        loss = loss_function(output, segment_point_label)
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
    seed_torch(3407)
    valid_flag = 0
    max_test = 0
    best_label = None
    best_preds = None
    for e in range(EPOCH):
        train_loss, train_acc, train_f1score, _, _ = train_model(e)
        valid_loss, valid_acc, valid_f1score, _, _  = eval_model(e, valid_data_loader)
        test_loss, test_acc, test_f1score, labels, preds  = eval_model(e, test_data_loader)
        print('train_loss:', train_loss, 'train_acc:', train_acc, 'train_f1score:', train_f1score, 
              'valid_loss:', valid_loss, 'valid_acc:', valid_acc, 'valid_f1score:', valid_f1score,
              'test_loss:', test_loss, 'test_acc:', test_acc, 'test_f1score:', test_f1score)
        if valid_f1score > valid_flag:
            best_test_f1score = test_f1score
            best_epoch = e
            valid_flag = valid_f1score
            best_label = labels
            best_preds = preds
        if test_f1score > max_test:
            max_test = test_f1score
    print('best_epoch:', best_epoch)
    print('best_test_f1score:', best_test_f1score)
    print('max_test:', max_test)
    print(classification_report(best_label, best_preds, digits=4))
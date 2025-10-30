import torch.nn as nn
import torch.nn.functional as F
import torch
# from .Resnet_model import resnet18
# from .Resnet_model import resnet50
# from .Resnet_model import resnet152
# from .Resnet_model import resnet101
from .bert import BERT
import time

class LSTMEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, bidirectional, n_layers, batch_first=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first,
                            bidirectional=bidirectional, num_layers=n_layers)

    def forward(self, trajs_hidden, trajs_len):
        # trajs_hidden: batch_size * 2 * n_views, seq_len, hidden_size
        # trajs_len: batch_size * 2 * n_views
        #packed_trajs_hidden = pack_padded_sequence(trajs_hidden, trajs_len.detach().cpu(), batch_first=True, enforce_sorted=False)
        # hn: num_layers * n_direction, batch_size * 2 * n_views, hidden_size
        #_, (hn, _) = self.lstm(packed_trajs_hidden)
        #outputs, _ = self.lstm(packed_trajs_hidden)
        outputs, _ = self.lstm(trajs_hidden)
        #hn = hn.transpose(0, 1).reshape(trajs_hidden.shape[0], -1)
        # outputs: batch_size * 2 * n_views, seq_len, hidden_size * n_direction
        # outputs, _ = self.lstm(trajs_hidden)
        # hn: batch_size * 2 * n_views, hidden_size * n_direction
        hn = outputs[torch.arange(trajs_hidden.shape[0]), trajs_len-1]
        #unpacked_output, hn = pad_packed_sequence(packed_output, batch_first=True)
        #return hn.transpose(0, 1).reshape(trajs_hidden.shape[0], -1)
        return hn


class LSTMSimCLR(nn.Module):

    def __init__(self, vocab_size, hidden_size, bidirectional, n_layers):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.bidirectional = True if bidirectional else False
        self.n_direction = 2 if self.bidirectional else 1
        self.n_layers = n_layers

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.encoder = LSTMEncoder(self.hidden_size, self.hidden_size, self.bidirectional, self.n_layers)
        self.predictor = nn.Sequential(
            nn.Linear(self.n_direction*self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

    def load_pretrained_embedding(self, embedding_matrix, freeze):
        # other number of freeze means do not load the pretraining embeddings
        if freeze not in {0, 1}:
            print("No pretraining embeddings")
            return
        freeze = True if freeze else False
        self.embedding = self.embedding.from_pretrained(embedding_matrix, freeze=freeze)

    def forward(self, trajs, trajs_len):
        # trajs: batch_size * 2 , seq_len
        # trajs_len: batch_size * 2
        # trajs_hidden: batch_size * 2, seq_len, hidden_size
        trajs_hidden = self.embedding(trajs)
        # features: batch_size * 2, hidden_size * n_direction
        features = self.encoder(trajs_hidden, trajs_len)
        features = self.predictor(features)
        #print('features:', features.shape)
        return features, trajs_hidden

    def encode_by_encoder(self, trajs, trajs_len):
        trajs_hidden = self.embedding(trajs)
        features = self.encoder(trajs_hidden, trajs_len)
        return features

    def encode_by_predictor(self, trajs, trajs_len):
        return self.forward(trajs, trajs_len)

    def encode_by_middle_layer(self, trajs, trajs_len):
        trajs_hidden = self.embedding(trajs)
        features = self.encoder(trajs_hidden, trajs_len)
        for module in self.predictor:
            features = module(features)
            break
        return features

class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """
        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.hidden)
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)
        self.linear_1 = nn.Linear(599, 512) #二分类和多分类都是这个
        self.linear_2 = nn.Linear(512, 2) #二分类和多分类都是这个
        self.linear_3 = nn.Linear(54, 60)
        self.conv2d_1 = nn.Conv2d(in_channels = 1, out_channels=1, kernel_size= (8,1), stride=1)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        self.conv2d_2 = nn.Conv2d(in_channels = 1, out_channels=1, kernel_size= (4,1), stride=1)
        self.maxpool_2 = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        self.LSTMSimCLR = LSTMSimCLR(1000004, 128, 0, 1)
        self.norm = nn.BatchNorm1d(512)
        self.relu = nn.ReLU(False)

    def forward(self, x, vocabs_vector, vocabs_len, CNN_feature, segment_label, label_bool, ignore_info, bert_label):
        #print("CNN_feature:", CNN_feature)
        #x = self.norm(x)
        #print('x:', x)
        flag = CNN_feature
        dim_0 = CNN_feature.shape[0]
        dim_1 = CNN_feature.shape[1]
        print('CNN_feature:', CNN_feature.shape)
        CNN_feature = CNN_feature.reshape(dim_0*dim_1, CNN_feature.shape[2], 27)
        # print('inin CNN_feature:', CNN_feature)
        # print('sum:', torch.sum(CNN_feature))
        CNN_feature = torch.unsqueeze(CNN_feature, dim = 1)
        result = torch.nonzero(torch.isnan(CNN_feature)==True)
        print('result:', result.shape)
        print('result222:', result)
        CNN_feature = self.conv2d_1(CNN_feature)
        weight = self.conv2d_1.weight
        weight_nan = torch.nonzero(torch.isnan(weight)==True)
        print('weight:', weight)
        print('weight_nan:', weight_nan.shape)
        if weight_nan.shape[0] > 0:
            for i in range(len(CNN_feature[0])):
                print(flag[0][i])
            time.sleep(360000000)
        # print('sum222:', torch.sum(CNN_feature))
        # print('2222:', CNN_feature)
        CNN_feature = self.relu(CNN_feature)
        
        maxpool_feature = self.maxpool(CNN_feature)
        #print('maxpool_feature:', maxpool_feature)
        CNN_feature_2 = self.conv2d_2(maxpool_feature)
        CNN_feature_2 = self.relu(CNN_feature_2)
        maxpool_feature_2 = self.maxpool_2(CNN_feature_2)
        CNN_feature = torch.squeeze(maxpool_feature_2, dim = 1)
        CNN_feature = torch.mean(CNN_feature, dim = 2)
        print('CNN_feature:', CNN_feature.shape)
        CNN_feature = self.linear_3(CNN_feature)
        CNN_feature = CNN_feature.reshape(dim_0, dim_1, 60)
        temp_x = x
        #print('CNN_feature:', CNN_feature)
        x = torch.cat([x, CNN_feature], dim = 2)
        #print('x:', x)
        x = self.bert(x, segment_label, label_bool)
        #print('x111:', x)
        x = torch.cat([x, CNN_feature, temp_x], dim = 2)
        #print('x222:', x)
        sim_features, trajs_hidden = self.LSTMSimCLR(vocabs_vector, vocabs_len)
        next_sentence = self.next_sentence(x)
        #print('x:', x)
        x = self.linear_1(x)
        save_x = self.linear_2(x)
        #print('save_x:', save_x)
        save_x = F.softmax(save_x, -1)
        return next_sentence, save_x, sim_features, trajs_hidden

class BERTLM_detction(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """
        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.hidden)
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)
        self.linear_1 = nn.Linear(599, 512)
        
        #self.conv2d = nn.Conv2d(in_channels = 1, out_channels=32, kernel_size= (4,1), stride=1)
        self.conv2d_1 = nn.Conv2d(in_channels = 1, out_channels=1, kernel_size= (4,1), stride=1)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        self.conv2d_2 = nn.Conv2d(in_channels = 1, out_channels=1, kernel_size= (2,1), stride=1)
        self.maxpool_2 = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        self.LSTMSimCLR = LSTMSimCLR(663272, 128, 0, 1)
        #self.img_linear = nn.Linear(25, 3)
        #self.resnet_18 = resnet50(pretrained=False)

    def forward(self, x, vocabs_vector, vocabs_len, CNN_feature, segment_label, label_bool):
        '''
        save_x = x
        x = self.bert(x, segment_label, label_bool)
        speed_x = save_x[:, :, 0:1]
        acceleration_x = save_x[:, :, 5:6]
        angular_speed_x = save_x[:, :, 10:11]
        angular_acceleration_x = save_x[:, :, 15:16]
        angle_diff_x = save_x[:, :, 20:21]
        cat_x = torch.cat([speed_x, acceleration_x, angular_speed_x, angular_acceleration_x, angle_diff_x], dim = 2)
        cat_x = self.bert(cat_x, segment_label, label_bool)
        #feature = (x + cat_x)/2
        feature = torch.cat([x, cat_x], dim = 2)
        '''
        #print('CNN_feature:', CNN_feature.shape)
        dim_0 = CNN_feature.shape[0]
        dim_1 = CNN_feature.shape[1]
        #print('CNN_feature111:', CNN_feature.shape)
        #CNN_feature = CNN_feature.reshape(dim_0*dim_1, CNN_feature.shape[2], 25) # 前面删除 lon lat
        CNN_feature = CNN_feature.reshape(dim_0*dim_1, CNN_feature.shape[2], 27)
        #print('CNN_feature:', CNN_feature)
        '''
        CNN_feature = torch.unsqueeze(CNN_feature, dim = 1)
        CNN_feature = CNN_feature.repeat(1, 3, 1, 1)
        #print('CNN_feature:', CNN_feature.shape)
        CNN_feature_out = self.resnet_18(CNN_feature)
        CNN_feature = CNN_feature_out.reshape(dim_0, dim_1, 32)
        '''
        CNN_feature = torch.unsqueeze(CNN_feature, dim = 1)
        CNN_feature = self.conv2d_1(CNN_feature)
        #print('CNN_feature111:', CNN_feature.shape)
        maxpool_feature = self.maxpool(CNN_feature)
        #print('maxpool_feature:', maxpool_feature.shape)
        CNN_feature_2 = self.conv2d_2(maxpool_feature)
        #print('CNN_feature222:', CNN_feature_2.shape)
        maxpool_feature_2 = self.maxpool_2(CNN_feature_2)
        # print('maxpool_feature_2:', maxpool_feature_2.shape)
        # CNN_feature = torch.mean(maxpool_feature_2, dim = 3)
        # CNN_feature = torch.mean(CNN_feature, dim = 2)
        sim_features, trajs_hidden = self.LSTMSimCLR(vocabs_vector, vocabs_len)
        CNN_feature = torch.squeeze(maxpool_feature_2, dim = 1)
        CNN_feature = torch.mean(CNN_feature, dim = 2)
        CNN_feature = CNN_feature.reshape(dim_0, dim_1, 60)
        temp_x = x
        x = torch.cat([x, CNN_feature], dim = 2)
        x = self.bert(x, segment_label, label_bool)
        x = torch.cat([x, CNN_feature, temp_x], dim = 2)
        '''
        save_x = self.linear_1(x)
        speed_x = save_x[:, :, 0:1]
        acceleration_x = save_x[:, :, 5:6]
        angular_speed_x = save_x[:, :, 10:11]
        angular_acceleration_x = save_x[:, :, 15:16]
        angle_diff_x = save_x[:, :, 20:21]
        cat_x = torch.cat([speed_x, acceleration_x, angular_speed_x, angular_acceleration_x, angle_diff_x], dim = 2)
        '''
        #print('x:', x.shape)
        feature = self.linear_1(x)
        index_list = []
        for h in range(0, trajs_hidden.shape[0], 2):
            index_list.append(h)
        index_list = torch.LongTensor(index_list).cuda()
        trajs_hidden = torch.index_select(trajs_hidden, dim=0, index=index_list)
        return feature, sim_features, trajs_hidden

class NextSentencePrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(599, 2)
        self.softmax = nn.LogSoftmax(dim=1)
        #self.layernorm = nn.LayerNorm(hidden+7, eps=1e-6)

    def forward(self, x):
        # print('x222:', x.shape)
        # print('x[:, 0]:', x[:, 0].shape)
        # print('x[:, 0]:', x[:, 0].shape)
        #print('x:', x.shape)
        #return self.softmax(self.linear(x[:, 0]))
        #print('x[:, 0]:', x[:, 0].shape)
        #temp = self.layernorm(x[:, 0])
        #print('sum:', torch.sum(x[:, 0]))
        return self.linear(x[:, 0])


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 5)
        # self.softmax = nn.LogSoftmax(dim=-1)
    def forward(self, x):
        # print('+++++++++++++9+++++++++++++++++++++')
        # return self.softmax(self.linear(x))
        return self.linear(x)

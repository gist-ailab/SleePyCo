import math
import torch
import torch.nn as nn


feature_len_dict = {
    'SleePyCo': [[5, 24, 120], [10, 48, 240], [15, 72, 360], [20, 96, 480], [24, 120, 600], [29, 144, 720], [34, 168, 840], [39, 192, 960], [44, 216, 1080], [48, 240, 1200]],
    'XSleepNet': [[6, 12, 24], [12, 24, 47], [18, 36, 71], [24, 47, 94], [30, 59, 118], [36, 71, 141], [42, 83, 165], [47, 94, 188], [53, 106, 211], [59, 118, 236]],
    'UTime': [[7, 15, 62], [15, 31, 125], [23, 45, 187], [31, 62, 250], [39, 78, 312], [46, 93, 375], [54, 109, 437], [62, 125, 500], [70, 140, 562], [78, 156, 625]],
}


class PlainRNN(nn.Module):
    def __init__(self, config):
        super(PlainRNN, self).__init__()
        self.cfg = config['classifier']
        self.num_classes = config['num_classes']
        self.input_dim = config['comp_chn']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_rnn_layers']
        self.bidirectional = config['bidirectional']
        
        # architecture
        self.rnn = nn.RNN(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional
        )
        self.fc = nn.Linear(self.hidden_dim * 2 if self.bidirectional else self.hidden_dim, self.num_classes)

    def init_hidden(self, x):
        h0 = torch.zeros((self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_dim)).cuda()
        
        return h0

    def forward(self, x):
        hidden = self.init_hidden(x)
        rnn_output, hidden = self.rnn(x, hidden)

        if self.bidirectional:
            output_f = rnn_output[:, -1, :self.hidden_dim]
            output_b = rnn_output[:, 0, self.hidden_dim:]
            output = torch.cat((output_f, output_b), dim=1)
        else:
            output = rnn_output[:, -1, :]
        
        output = self.fc(output)

        return output


class PlainGRU(PlainRNN):
    def __init__(self, config):
        super(PlainGRU, self).__init__(config)
        self.rnn = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional
        )


class PlainLSTM(PlainRNN):
    def __init__(self, config):
        super(PlainLSTM, self).__init__(config)
        self.rnn = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional
        )
    
    def init_hidden(self, x):
        h0 = torch.zeros((self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_dim)).cuda()
        c0 = torch.zeros((self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_dim)).cuda()
        
        return h0, c0


class AttRNN(PlainRNN):
    def __init__(self, config):
        super(AttRNN, self).__init__(config)
        # architecture
        self.fc = nn.Linear(self.hidden_dim, self.num_classes)
        self.w_ha = nn.Linear(self.hidden_dim * 2 if self.bidirectional else self.hidden_dim, self.hidden_dim, bias=True)
        self.w_att = nn.Linear(self.hidden_dim, 1, bias=False)

    def forward(self, x):
        hidden = self.init_hidden(x)
        rnn_output, hidden = self.rnn(x, hidden)
        a_states = self.w_ha(rnn_output)
        alpha = torch.softmax(self.w_att(a_states), dim=1).view(x.size(0), 1, x.size(1))
        weighted_sum = torch.bmm(alpha, a_states)

        output = weighted_sum.view(x.size(0), -1)
        output = self.fc(output)

        return output


class AttGRU(AttRNN):
    def __init__(self, config):
        super(AttGRU, self).__init__(config)
        self.rnn = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional
        )


class AttLSTM(AttRNN):
    def __init__(self, config):
        super(AttLSTM, self).__init__(config)
        self.rnn = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional
        )
    
    def init_hidden(self, x):
        h0 = torch.zeros((self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_dim)).cuda()
        c0 = torch.zeros((self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_dim)).cuda()
        
        return h0, c0


class PositionalEncoding(nn.Module):
    
    def __init__(self, config, in_features, out_features, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.cfg = config['classifier']['pos_enc']
        self.num_scales = config['feature_pyramid']['num_scales']
        
        if self.cfg['dropout']:
            self.dropout = nn.Dropout(p=dropout)
        
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.act_fn = nn.PReLU()
        
        if self.num_scales > 1:        
            self.max_len = feature_len_dict[config['backbone']['name']][config['dataset']['seq_len'] - 1][config['feature_pyramid']['num_scales'] - 1]
        else:
            self.max_len = 5000
        
        print('[INFO] Maximum length of pos_enc: {}'.format(self.max_len))

        pe = torch.zeros(self.max_len, out_features)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, out_features, 2).float() * (-math.log(10000.0) / out_features))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = self.act_fn(self.fc(x))

        if self.num_scales > 1:
            hop = self.max_len // x.size(0)
            pe = self.pe[hop//2::hop, :]
        else:
            pe = self.pe

        if pe.shape[0] != x.size(0):
            pe = pe[:x.size(0), :]

        x = x + pe

        if self.cfg['dropout']:
            x = self.dropout(x)

        return x


class Transformer(nn.Module):

    def __init__(self, config, nheads, num_encoder_layers, pool='mean'):

        super(Transformer, self).__init__()
        
        self.cfg = config['classifier']
        self.model_dim = self.cfg['model_dim']
        self.feedforward_dim = self.cfg['feedforward_dim']
        
        self.in_features = config['feature_pyramid']['dim']
        self.out_features = self.cfg['model_dim']
        
        self.pos_encoding = PositionalEncoding(config, self.in_features, self.out_features)
        
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=nheads,
            dim_feedforward=self.feedforward_dim,
            dropout=0.1 if self.cfg['dropout'] else 0.0
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_encoder_layers)

        self.pool = pool

        if self.cfg['dropout']:
            self.dropout = nn.Dropout(p=0.5)
        
        if pool == 'attn':
            self.w_ha = nn.Linear(self.model_dim, self.model_dim, bias=True)
            self.w_at = nn.Linear(self.model_dim, 1, bias=False)
        
        self.fc = nn.Linear(self.model_dim, self.cfg['num_classes'])

    def forward(self, x):
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x.transpose(0, 1)

        if self.pool == 'mean':
            x = x.mean(dim=1)
        elif self.pool == 'last':
            x = x[:, -1]
        elif self.pool == 'attn':
            a_states = torch.tanh(self.w_ha(x))
            alpha = torch.softmax(self.w_at(a_states), dim=1).view(x.size(0), 1, x.size(1))
            x = torch.bmm(alpha, a_states).view(x.size(0), -1)
        elif self.pool == None:
            x = x
        else:
            raise NotImplementedError

        if self.cfg['dropout']:
            x = self.dropout(x)
        
        out = self.fc(x)
        
        return out


def get_classifier(config):
    classifier_name = config['classifier']['name']
    
    if classifier_name == 'PlainRNN':
        classifier = PlainRNN(config)
        
    elif classifier_name == 'AttentionRNN':
        classifier = AttRNN(config)
    
    if classifier_name == 'PlainLSTM':
        classifier = PlainLSTM(config)

    elif classifier_name == 'AttentionLSTM':
        classifier = AttLSTM(config)
        
    elif classifier_name == 'PlainGRU':
        classifier = PlainGRU(config)

    elif classifier_name == 'AttentionGRU':
        classifier = AttGRU(config)

    elif classifier_name == 'Transformer':
        classifier = Transformer(config, nheads=8, num_encoder_layers=6, pool=config['classifier']['pool'])
    
    return classifier

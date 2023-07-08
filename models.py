from setup import *

import torch
from torch import nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


# Simple Neural Net
class MultilabelClassifier_NeuralNet(nn.Module):

    def __init__(self, n_classes=NUM_LABELS, model_name=MODEL_NAME):

        super(MultilabelClassifier_NeuralNet, self).__init__()
        self.n_classes = n_classes

        # Architecture
        self.bert = AutoModel.from_pretrained(model_name, return_dict=True) # Backbone
        self.hidden = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.dropout = nn.Dropout(0.3)

        # Initialization
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, input_ids, attention_masks):#, token_type_ids):

        # Bert (fine-tuning)
        out = self.bert(input_ids=input_ids, attention_mask=attention_masks)
        pooled_out = torch.mean(out.last_hidden_state, 1)

        # Neural network
        x = self.dropout(pooled_out)
        x = self.hidden(x)
        x = F.relu(x)
        x = self.dropout(x)
        logits = self.classifier(x)

        return logits




# TextCNN
class MultilabelClassifier_TextCNN(nn.Module):

    def __init__(self, n_classes=NUM_LABELS, model_name=MODEL_NAME):

        super(MultilabelClassifier_TextCNN, self).__init__()
        self.n_classes = n_classes

        # Architecture
        self.bert = AutoModel.from_pretrained(model_name, return_dict=True) # Backbone
        self.cnn = nn.Conv1d(self.bert.config.hidden_size, 256, kernel_size=3, padding=1)
        self.classifier = nn.Linear(256, self.n_classes)
        self.dropout = nn.Dropout(0.3)

        # Initialization
        nn.init.xavier_uniform_(self.cnn.weight)
        nn.init.constant_(self.cnn.bias, 0)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, input_ids, attention_masks):#, token_type_ids):

        # Bert (fine-tuning)
        out = self.bert(input_ids=input_ids, attention_mask=attention_masks)
        word_embeddings = out.last_hidden_state.permute(0, 2, 1) # Reshape for CNN input

        # CNN
        cnn_out = self.cnn(word_embeddings)
        cnn_out = F.relu(cnn_out)
        pooled_out = F.max_pool1d(cnn_out, kernel_size=cnn_out.size(2)).squeeze(2)

        # Dropout
        x = self.dropout(pooled_out)

        # Classifier
        logits = self.classifier(x)

        return logits




# Bi-LSTM
class MultilabelClassifier_BiLSTM(nn.Module):

    def __init__(self, n_classes=NUM_LABELS, model_name=MODEL_NAME):

        super(MultilabelClassifier_BiLSTM, self).__init__()
        self.n_classes = n_classes

        # Architecture
        self.bert = AutoModel.from_pretrained(model_name, return_dict=True) # Backbone
        self.b_lstm = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(2 * self.b_lstm.hidden_size, self.n_classes)
        self.dropout = nn.Dropout(0.3)

        # Initialization
        nn.init.xavier_uniform_(self.b_lstm.weight_ih_l0)
        nn.init.xavier_uniform_(self.b_lstm.weight_hh_l0)
        nn.init.constant_(self.b_lstm.bias_ih_l0, 0)
        nn.init.constant_(self.b_lstm.bias_hh_l0, 0)

    def forward(self, input_ids, attention_masks):#, token_type_ids):

        # Bert (fine-tuning)
        out = self.bert(input_ids=input_ids, attention_mask=attention_masks)
        pooled_out = torch.mean(out.last_hidden_state, 1)

        # Bi-LSTM
        lstm_out, _ = self.b_lstm(pooled_out.unsqueeze(0))
        lstm_out = lstm_out.squeeze(0)

        # Dropout
        x = self.dropout(lstm_out)

        # Classifier
        logits = self.classifier(x)

        return logits




# Bi-GRU
class MultilabelClassifier_BiGRU(nn.Module):

    def __init__(self, n_classes=NUM_LABELS, model_name=MODEL_NAME):

        super(MultilabelClassifier_BiGRU, self).__init__()
        self.n_classes = n_classes

        # Architecture
        self.bert = AutoModel.from_pretrained(model_name, return_dict=True) # Backbone
        self.b_gru = nn.GRU(self.bert.config.hidden_size, self.bert.config.hidden_size, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(2 * self.b_gru.hidden_size, self.n_classes)
        self.dropout = nn.Dropout(0.3)

        # Initialization
        nn.init.xavier_uniform_(self.b_gru.weight_ih_l0)
        nn.init.xavier_uniform_(self.b_gru.weight_hh_l0)
        nn.init.constant_(self.b_gru.bias_ih_l0, 0)
        nn.init.constant_(self.b_gru.bias_hh_l0, 0)

    def forward(self, input_ids, attention_masks):#, token_type_ids):

        # Bert (fine-tuning)
        out = self.bert(input_ids=input_ids, attention_mask=attention_masks)
        pooled_out = torch.mean(out.last_hidden_state, 1)

        # Bi-GRU
        gru_out, _ = self.b_gru(pooled_out.unsqueeze(0))
        gru_out = gru_out.squeeze(0)

        # Dropout
        x = self.dropout(gru_out)

        # Classifier
        logits = self.classifier(x)

        return logits
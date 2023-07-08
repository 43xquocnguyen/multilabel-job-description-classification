import pickle
from models import *

def return_label(y):
    y = y.flatten()
    re = []
    for i in range(0, len(y)):
        if y[i] == 1:
            re.append(ALL_LABELS[i])

    return re

def load_model_ML(model_name):
    if model_name == 'Linear Regression':
        path = 'models/LR.pkl'
    elif model_name == 'Stochastic Gradient Descent':
        path = 'models/SGD.pkl'
    elif model_name == 'Support Vector Machine':
        path = 'models/SVC.pkl'
    else:
        raise ValueError('model_name for ML models is wrong!')
    # Loading models
    with open(path, 'rb') as file:
        model = pickle.load(file)
    
    return model

# Load model
def load_model_DL(c, pretrained, fe):
    if c == 'MLP':
        model = MultilabelClassifier_NeuralNet(model_name=pretrained)
        c = 'NeuralNet'
    elif c == 'TextCNN':
        model = MultilabelClassifier_TextCNN(model_name=pretrained)
    elif c == 'Bi-LSTM':
        model = MultilabelClassifier_BiLSTM(model_name=pretrained)
        c = 'BiLSTM'
    else:
        model = MultilabelClassifier_BiGRU(model_name=pretrained)
        c = 'BiGRU'

    # load weights
    model.load_state_dict(torch.load('models/' + fe + '_' + c + '_9.pth', map_location=torch.device('cpu')))

    return model

# Get the name of BERT (for get tokenizer)
def get_backbone_name(fe):
    if fe == 'phoBERT':
        pretrained_model = 'vinai/phobert-base'
    elif fe == 'XLMBert':
        pretrained_model = 'bert-base-multilingual-cased'
    else:
        pretrained_model = 'distilbert-base-cased'

    return pretrained_model


def tensor_to_numpy(tensor):
    return tensor.cpu().numpy()

def get_prediction(model, ids, attn):

    ids = ids.unsqueeze(0)
    attn = attn.unsqueeze(0)

    with torch.no_grad():
        logits = model(ids, attn)

    probs = torch.sigmoid(logits)
    preds = torch.round(probs)

    return logits, probs, preds
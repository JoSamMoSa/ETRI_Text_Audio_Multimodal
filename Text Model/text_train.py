'''
text data 학습을 위한 코드
- input : df_all_txt.csv (preprocessing.py 실행 후 생성됨.)
- output : text_train_model.pt (text data 학습 모델)

- git clone https://github.com/SKTBrain/KoBERT.git
- 경우에 따라 https://huggingface.co/skt/kobert-base-v1 에서 다운로드 후 사용해야 함.
- 필요 라이브러리 설치 : pip install -r requirements.txt (올바른 디렉토리에 파일 위치)
'''

from kobert_hf.kobert_tokenizer import KoBERTTokenizer

from transformers import BertModel
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from torch import nn
from torch.utils.data import Dataset

import torch
import numpy as np
import pandas as pd
import gluonnlp as nlp

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')

# GPU 사용 시
device = torch.device("cuda:0")

# 데이터셋 전처리, df_all_txt.csv 파일 올바른 디렉토리에 위치
df_all_txt = pd.read_csv("df_all_txt.csv")
df_all_txt = df_all_txt[:8231] # train data

df_all_txt.loc[(df_all_txt['Label'] == "neutral"), 'Label'] = 0
df_all_txt.loc[(df_all_txt['Label'] == "angry"), 'Label'] = 1
df_all_txt.loc[(df_all_txt['Label'] == "happy"), 'Label'] = 2
df_all_txt.loc[(df_all_txt['Label'] == "surprise"), 'Label'] = 3
df_all_txt.loc[(df_all_txt['Label'] == "sad"), 'Label'] = 4
df_all_txt.loc[(df_all_txt['Label'] == "fear"), 'Label'] = 5
df_all_txt.loc[(df_all_txt['Label'] == "disgust"), 'Label'] = 6

train_data_list = []
for ques, label in zip(df_all_txt['text'], df_all_txt['Label'])  :
    train_data = []   
    train_data.append(ques)
    train_data.append(str(label))
    train_data_list.append(train_data)

text_all_emotion = []
for i in range(len(train_data_list)):
    text_all_emotion.append(train_data_list[i][1])

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer,vocab, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len,vocab=vocab, pad=pad, pair=pair)
        
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))
         
    def __len__(self):
        return (len(self.labels))

# 파라미터 세팅
max_len = 128
batch_size = 16
warmup_ratio = 0.1
num_epochs = 16
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

tok = tokenizer.tokenize
data_train = BERTDataset(train_data_list, 0, 1, tok, vocab, max_len, True, False)
train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=7,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device),return_dict=False)
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

# BERT 모델 불러오기
model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
 
# optimizer와 schedule 설정
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss() # 다중 분류를 위한 loss func

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

# 정확도 측정을 위한 함수 정의
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

train_history = []
val_history = []
loss_history = []

for e in range(num_epochs):
    train_acc = 0.0
    val_acc = 0.0

    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(train_dataloader):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
         
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
            train_history.append(train_acc / (batch_id+1))
            loss_history.append(loss.data.cpu().numpy())
    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))

# 학습 모델 저장
torch.save(model, "text_train.pt")
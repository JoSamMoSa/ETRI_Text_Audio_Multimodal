from kobert_hf.kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel
from transformers import AdamW
import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import gluonnlp as nlp
from sklearn.metrics import f1_score, accuracy_score

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')

# GPU 사용 시
device = torch.device("cuda:0")

# 데이터셋 전처리 - test 데이터
df_all_txt = pd.read_csv("df_all_txt.csv")
df_all_txt = df_all_txt[8231:]

df_all_txt.loc[(df_all_txt['Label'] == "neutral"), 'Label'] = 0
df_all_txt.loc[(df_all_txt['Label'] == "angry"), 'Label'] = 1
df_all_txt.loc[(df_all_txt['Label'] == "happy"), 'Label'] = 2
df_all_txt.loc[(df_all_txt['Label'] == "surprise"), 'Label'] = 3
df_all_txt.loc[(df_all_txt['Label'] == "sad"), 'Label'] = 4
df_all_txt.loc[(df_all_txt['Label'] == "fear"), 'Label'] = 5
df_all_txt.loc[(df_all_txt['Label'] == "disgust"), 'Label'] = 6

test_data_list = []
for ques, label in zip(df_all_txt['text'], df_all_txt['Label'])  :
    test_data = []   
    test_data.append(ques)
    test_data.append(str(label))
    test_data_list.append(test_data)


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
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

tok = tokenizer.tokenize
data_test = BERTDataset(test_data_list, 0, 1, tok, vocab,  max_len, True, False)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)

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

# train 된 BERT 모델 불러오기
model = torch.load("text_train.pt")
 
# optimizer와 schedule 설정
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss() # 다중 분류를 위한 loss func

# 정확도 측정을 위한 함수 정의
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

truth_label = []
pred_label = []

output_list = []

test_acc = 0.0
model.eval()
for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
    
    token_ids = token_ids.long().to(device)
    segment_ids = segment_ids.long().to(device)
    valid_length = valid_length
    label = label.long().to(device)
    out = model(token_ids, valid_length, segment_ids)
    
    test_acc += calc_accuracy(out, label)

    predlabel = torch.argmax(out, dim=1)
    label = label.tolist()
    predlabel = predlabel.tolist()
    truth_label.extend(label)
    pred_label.extend(predlabel)

    out = out.detach().cpu().numpy()
    output_list.append(out)

output_np = np.array(output_list)

# np.save("./text_output_tmp.npy", output_np)
np.save("./text_pred.npy", pred_label)
np.save("./text_truth.npy", truth_label)

text_output_tmp = []
for i in range(len(output_np)):
    text_output_tmp.extend(output_np[i])
text_output_tmp = np.array(text_output_tmp)
np.save("./text_output.npy", text_output_tmp) # audio 와 tensor 통일한 output

f1score = f1_score(truth_label, pred_label, average="weighted")
print("fi score :", f1score)

acc = accuracy_score(truth_label, pred_label)
print("accuracy :", acc)
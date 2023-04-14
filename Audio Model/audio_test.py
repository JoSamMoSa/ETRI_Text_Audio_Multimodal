'''
audio data 학습 결과 test를 위한 코드
- input : df_all_txt.csv, /merged_wav_folder/ (preprocessing.py 실행 후 생성됨.), audio_train_model.pt (audio_train.py 실행 후 생성됨.)
- output : audio_pred.npy (모델 예측 라벨), audio_truth.npy (실제 라벨), audio_output.npy (모델 예측 값)
'''
from scipy import signal
from scipy.io import wavfile

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

import numpy as np
import pandas as pd

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from sklearn.metrics import f1_score, accuracy_score

df_all_txt = pd.read_csv("df_all_txt.csv") # df_all_txt.csv 파일 경로 입력

all_wav_tmp = df_all_txt["Seg"].tolist()
all_txt = df_all_txt["text"].tolist()

df_all_txt.loc[(df_all_txt['Label'] == "neutral"), 'Label'] = 0
df_all_txt.loc[(df_all_txt['Label'] == "angry"), 'Label'] = 1
df_all_txt.loc[(df_all_txt['Label'] == "happy"), 'Label'] = 2
df_all_txt.loc[(df_all_txt['Label'] == "surprise"), 'Label'] = 3
df_all_txt.loc[(df_all_txt['Label'] == "sad"), 'Label'] = 4
df_all_txt.loc[(df_all_txt['Label'] == "fear"), 'Label'] = 5
df_all_txt.loc[(df_all_txt['Label'] == "disgust"), 'Label'] = 6

all_emotion = df_all_txt["Label"].tolist()

list_files = []

for i in all_wav_tmp:
    i = i + ".wav"
    list_files.append(i)

# audio data -> spectrogram
def audio2spectrogram(filepath):
    samplerate, test_sound  = wavfile.read(filepath,mmap=True)
    _, spectrogram = log_specgram(test_sound, samplerate)
    return spectrogram

def log_specgram(audio, sample_rate, window_size=40, step_size=20, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, _, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, np.log(spec.T.astype(np.float32) + eps)

# AlexNet에 맞게 Spectrogram을 3차원으로 변경
N_CHANNELS = 3
def get_3d_spec(Sxx_in, moments=None):
    if moments is not None:
        (base_mean, base_std, delta_mean, delta_std, delta2_mean, delta2_std) = moments
    else:
        base_mean, delta_mean, delta2_mean = (0, 0, 0)
        base_std, delta_std, delta2_std = (1, 1, 1)
    h, w = Sxx_in.shape
    right1 = np.concatenate([Sxx_in[:, 0].reshape((h, -1)), Sxx_in], axis=1)[:, :-1]
    delta = (Sxx_in - right1)[:, 1:]
    delta_pad = delta[:, 0].reshape((h, -1))
    delta = np.concatenate([delta_pad, delta], axis=1)
    right2 = np.concatenate([delta[:, 0].reshape((h, -1)), delta], axis=1)[:, :-1]
    delta2 = (delta - right2)[:, 1:]
    delta2_pad = delta2[:, 0].reshape((h, -1))
    delta2 = np.concatenate([delta2_pad, delta2], axis=1)
    base = (Sxx_in - base_mean) / base_std
    delta = (delta - delta_mean) / delta_std
    delta2 = (delta2 - delta2_mean) / delta2_std
    stacked = [arr.reshape((h, w, 1)) for arr in (base, delta, delta2)]
    return np.concatenate(stacked, axis=2)

no_rows=len(list_files)
index=0
sprectrogram_shape=[]
docs = []
bookmark=0
extraLabel=0

# audio data의 파일 이름, 레이블, spectrogram 저장
for i in range(len(list_files)):
    filename = all_wav_tmp[i]
    label = all_emotion[i]

    spector = audio2spectrogram("./merged_wav_folder/"+list_files[i])

    pad2d = lambda a, i: np.vstack((a, np.zeros((i-a.shape[0], a.shape[1]), dtype=np.float32))) if a.shape[0] < i else a[:, :]
    spector = pad2d(spector, 100)
    spector = get_3d_spec(spector)
    npimg = np.transpose(spector,(2,0,1))
    input_tensor = torch.tensor(npimg)

    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
      
    docs.append({
        'fileName': filename,
        'sprectrome': input_batch,
        'label': label
        })
    index += 1
    #print(index)

__all__ = ['AlexNet', 'alexnet']

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

# 모델 구성
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.num_classes=num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((12, 12))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return x
    
def alexnet(pretrained=False, progress=True, **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'], progress=progress)
        model.load_state_dict(state_dict)
    return model

# 가변 길이의 입력을 위한 모델로 변경
class ModifiedAlexNet(nn.Module):
    def __init__(self, num_classes=7):
        super(ModifiedAlexNet, self).__init__()
        self.num_classes=num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=2)
        x = torch.sum(x, dim=2)
        x = self.classifier(x)
        return x
   
def modifiedAlexNet(pretrained=False, progress=True, **kwargs):
    model_modified = ModifiedAlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'], progress=progress)
        model_modified.load_state_dict(state_dict)
    return model_modified

original_model=alexnet(pretrained=True)
original_dict = original_model.state_dict()
modifiedAlexNet=modifiedAlexNet(pretrained=False)
modified_model_dict = modifiedAlexNet.state_dict()
pretrained_modified_model_dict = {k: v for k, v in original_dict.items() if k in modified_model_dict}
modifiedAlexNet.to('cuda')

# train / test 8:2 split
train_list = docs[:8231]
test_list = docs[8231:]

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(modifiedAlexNet.parameters(),
                  lr =  2e-4, 
                  eps = 1e-8
                )

NUM_EPOCHS = 16
total_steps = len(train_list) * NUM_EPOCHS

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

total_steps = 1

# train 된 결과 불러오기
model = torch.load("audio_train_model.pt")
model.eval()
model.to('cuda')

y_actu=[]
y_pred=[]
output_list = []

# test
for every_test_list in test_list:
    label1 = every_test_list['label']
    label1 = torch.tensor([label1])
    sprectrome=every_test_list['sprectrome']

    with torch.no_grad():
      if sprectrome.shape[2]:
        sprectrome = sprectrome.to('cuda')

        output = model(sprectrome)
        
        _, preds = torch.max(output, 1)
        y_actu.append(label1.numpy()[0])
        y_pred.append(preds.cpu().numpy()[0])

        out = output.detach().cpu().numpy()
        output_list.append(out)

# 결과 저장
output_np = np.array(output_list)
#np.save("./audio_output_tmp.npy", output_np)

audio_output_tmp = output_np.reshape(2025,7)
np.save("./audio_output.npy", audio_output_tmp)

np.save("./audio_truth", y_actu) 
np.save("./audio_pred", y_pred) 

# 결과 출력
f1score = f1_score(y_actu, y_pred, average="weighted")
print("f1 socre :", f1score)

acc = accuracy_score(y_actu, y_pred)
print("acc :", acc)
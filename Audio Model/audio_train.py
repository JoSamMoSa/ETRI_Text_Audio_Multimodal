from scipy import signal
from scipy.io import wavfile

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

import numpy as np
import pandas as pd

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

df_all_txt = pd.read_csv("df_all_txt.csv")

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

def audio2spectrogram(filepath):
    samplerate, test_sound  = wavfile.read(filepath, mmap=True)
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

no_rows = len(list_files)
index = 0
sprectrogram_shape = []
docs = []
bookmark = 0
extraLabel = 0

for i in range(len(list_files)):
    filename = all_wav_tmp[i]
    label = all_emotion[i]

    spector = audio2spectrogram("./merged_wav_folder/" + list_files[i])

    pad2d = lambda a, i: np.vstack((a, np.zeros((i-a.shape[0], a.shape[1]), dtype=np.float32))) if a.shape[0] < i else a[:, :]
    spector = pad2d(spector, 100)
    spector = get_3d_spec(spector)
    npimg = np.transpose(spector, (2,0,1))
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

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
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
        print('features',x.shape)
        return x
    
def alexnet(pretrained=False, progress=True, **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'], progress=progress)
        model.load_state_dict(state_dict)
    return model

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

train_list = docs[:8231]
test_list = docs[8231:]

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(modifiedAlexNet.parameters(),
                  lr =  2e-4, 
                  eps = 1e-8
                )

NUM_EPOCHS=16

total_steps = len(train_list) * NUM_EPOCHS

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

total_steps = 1

for epoch in range(NUM_EPOCHS):
    modifiedAlexNet.train()
    for every_trainlist in train_list:
        label1 = every_trainlist['label']
        label1 = torch.tensor([label1])
        sprectrome = every_trainlist['sprectrome']

        if sprectrome.shape[2]:
            optimizer.zero_grad()
            sprectrome = sprectrome.to('cuda')
            label1 = label1.to('cuda')
            modifiedAlexNet.zero_grad()

            output = modifiedAlexNet(sprectrome)
            loss = criterion(output, label1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(modifiedAlexNet.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        if total_steps % 1000 == 0:
            with torch.no_grad():
                print('Epoch: {} \tStep: {} \tLoss: {:.4f}'.format(epoch + 1, total_steps, loss.item()))                  
        total_steps += 1

torch.save(modifiedAlexNet, './audio_train_model.pt')
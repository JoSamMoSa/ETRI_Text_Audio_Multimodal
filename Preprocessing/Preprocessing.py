# 라이브러리 import
import os
import glob
import pandas as pd
import shutil
import warnings
import librosa
import numpy as np
import pickle
import sklearn
from sklearn.preprocessing import scale
from tqdm import tqdm 

#패키지가 없을 시 설치 진행
#pip install librosa 
#pip install tqdm

# Label preprocessing
# Annotation 폴더 내에서 csv 추출
annotation = 'C:/Users/jisuj/Desktop/KSC2023/annotation/*.csv'
folders = glob.glob(annotation)
df_all_label = pd.DataFrame()
#folders 예시 : C:/Users/jisuj/Desktop/KSC2023/annotation\\Session01_F_res.csv

#발화자의 label만 추출
for files in folders:
    sex = files.split('_')[1]
    Label = pd.read_csv(files, usecols=[9, 10])
    Label = Label[1:]
    Label =  Label[Label['Segment ID'].str.contains('_'+sex, na=False, case=False)]
    df_all_label = pd.concat([df_all_label, Label])

df_all_label.rename(columns={'Segment ID':'Seg', 'Total Evaluation':'Label'}, inplace=True)
#print('df_all_label', df_all_label)
df_all_label.sort_values(by='Seg',inplace=True)

# Label 파일 저장
df_all_label.to_csv("C:/Users/jisuj/Desktop/KSC2023/df_label.csv", mode='w')
print('Label 데이터 추출 끝')

# Label csv 불러오기
label= pd.read_csv('C:/Users/jisuj/Desktop/KSC2023/df_label.csv')

# 감정레이블이 여러 개일 경우 제일 처음 감정만 추출
def one_emo(x):
  if ";" in x:
    idx_number = x.find(";")
    return x[:idx_number]
  else:
    return x


label['Label'] = label['Label'].apply(lambda x:one_emo(x))
print('label', label)
label.sort_values(by='Seg',inplace=True)
label = label.drop(columns=['Unnamed: 0'])

# 세그먼트당 1개의 감정만을 담은 csv
label.to_csv("C:/Users/jisuj/Desktop/KSC2023/df_all_label.csv", mode='w')
print('Label 데이터 정제 끝')



# TEXT preprocessing
# 폴더 내에 있는 .txt 파일만 읽고 하나의 .txt파일에 입력하기
targetPattern = 'C:/Users/jisuj/Desktop/KSC2023/wav/**/**/*.txt'
allTextFile = glob.glob(targetPattern)

# merged_seg_text.txt파일에 raw text 입력
mergedText = open('C:/Users/jisuj/Desktop/KSC2023/merged_seg_text.txt', 'w', encoding="UTF-8")
## 'C:/Users/jisuj/Desktop/KSC2023/wav\\Session02\\Sess02_script06\\Sess02_script06_F015.txt'


for i in range(len(allTextFile)):
    myText = allTextFile[i]
    first = myText.rfind('Sess') #60
    last = myText.find('.txt') #79
    sessID = myText[first:last] #Sess02_script06_F015
    myText = open(myText, 'r', encoding="UTF-8")
    text = sessID + '|' + myText.readline()
    mergedText.write(text)
mergedText.close()
print('==============Text 데이터 추출완료==============')


#text 데이터와 label 데이터와 병합 후 각 row에 중복되는 값 제거
df_all_text = pd.read_csv('C:/Users/jisuj/Desktop/KSC2023/merged_seg_text.txt', names=['Seg', 'text'], sep='|')


label = pd.read_csv('C:/Users/jisuj/Desktop/KSC2023/df_all_label.csv', names=['Seg', 'Label'])
label = label[1:]

# text와 label 데이터를 Segment ID를 기준으로 병합하기
df_all_txt_label = pd.merge(label, df_all_text, on='Seg')

# 정규 표현식을 사용하여 한글을 제외한 단어 제거 및 공백 제거하기
df_all_txt_label['text'] = df_all_txt_label['text'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
df_all_txt_label['text'].replace('', np.nan, inplace=True)

#텍스트가 없는 것 지움 ! -> 10256,3
df_all_txt_label = df_all_txt_label.dropna(how='any')



# df_all_txt csv 파일로 저장
df_all_txt_label.to_csv('C:/Users/jisuj/Desktop/KSC2023/df_all_txt.csv', mode='w', encoding="utf-8-sig")
print('텍스트 파일 저장 끝')

# Wav preprocessing
# Wav 파일내 .wav파일들을 하나의 폴더에 copy하기
targetPattern = 'C:/Users/jisuj/Desktop/KSC2023/wav/**/**/*.wav'
allWavFile = glob.glob(targetPattern)
#wav_file 예시 C:/Users/jisuj/Desktop/KSC2023/wav\\Session02\\Sess02_script06\\Sess02_script06_F013.wav

#merged_wav_folder로 파일 옮기기
for wav_file in allWavFile:
    shutil.copy(wav_file, 'C:/Users/jisuj/Desktop/KSC2023/merged_wav_folder')

# 텍스트와 라벨이 있는 데이터만 wav 파일 가져오기
need_wav_files = []
df_all_txt_label = df_all_txt_label.reset_index()
df_all_txt_label = df_all_txt_label.drop(columns=['index'])
for i in range(len(df_all_txt_label)):
    a = df_all_txt_label['Seg'][i]
    need_wav_files.append('C:/Users/jisuj/Desktop/KSC2023/merged_wav_folder\\'+a+'.wav')
        
#forders C:/Users/jisuj/Desktop/KSC2023/merged_wav_folder\\Sess02_script06_F015.wav

# mfcc 추출시 발생되는 warning 제거하기
warnings.filterwarnings(action='ignore')

append_list=[]
extend_list=[]
for files in tqdm(need_wav_files):
    audio, sr = librosa.load(files, sr=16000)
    #mfcc추출 파라미터 설정
    mfcc = librosa.feature.mfcc(audio, sr=16000, n_mfcc=100, n_fft=400, hop_length=160) 
    #전처리 scaling
    mfcc = sklearn.preprocessing.scale(mfcc, axis=1)
    pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i-a.shape[1]))))
    padded_mfcc = pad2d(mfcc, 465)
    
    append_list.append(padded_mfcc)
    extend_list.extend(append_list)
    append_list.clear()

extend_list_array=np.array(extend_list)

# mfcc가 담긴 extend_list_array를 .npy로 저장
#텍스트 순서에 맞도록 함
np.save(r'C:/Users/jisuj/Desktop/KSC2023/all_mfcc.npy', extend_list_array)
print('mfcc 데이터 저장')

##wav 코드를 text, label값이 들어있는 table의 seg로 불러왔기 때문에 wav 파일의 label은 text의 label과 같음


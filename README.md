# Text-Audio Multimodal Emotion Recognization
[제2회 ETRI 휴먼이해 인공지능 논문경진대회](https://aifactory.space/competition/detail/2234)

### Feature Fusion 방식과 Decision Fusion 방식을 결합한 멀티모달 감정 인식에 관한 연구
최근 사용자의 발화로부터 감정을 구별하는 모델에 대한 연구가 활발하게 이어짐에 따라 언어적 표현 과 더불어 반언어적 표현을 반영하기 위한 멀티모달 데이터의 활용이 주목을 받고 있다. 따라서 본 논문에서는 텍스트 데이터와 음성 데이터의 상호작용 정보를 반영하고자 Feature Fusion 방식을 사용하는 모델과 Decision Fusion 방식으로 단일 모델들을 결합하는 방법론을 제안한다. 연구 결과, 텍스트 데이터, 음성 데이터, 텍스트-음성 멀티모달 데이터를 각각 입력으로 받아 감정 인식을 진행하는 세 단일 모델보다 Stacking 기법의 앙상블을 적용한 복합 모델이 약 0.148 향상된 Accuracy를, 약 0.158 향상된 F1 Score를 달성하였다.
<img width="450" alt="Model Architecture" src="https://user-images.githubusercontent.com/38968449/231982829-352a052b-3b2c-486e-9dd7-abd60179abb3.png">

## Environment

  Python version: 3.8
  
  Required packages: 각 Model 폴더 내의 requirements.txt 참조

```
  # install packages
  pip install -r requirements.txt
```

## Dataset
[한국어 멀티모달 감정 데이터셋 2019](https://nanum.etri.re.kr/share/kjnoh/KEMDy19?lang=ko_KR) 사용
|Directory|Format|Description|
|-----|-----|------------|
|./annotation|.csv|세션/참여자 발화 세그먼트에 대한 관찰자의 감정 레이블 평가 파일|
|./wav|.wav / .txt|세션/감정 상황극 내 발화 세그먼트 웨이브 파일(.wav) <br> 세션/감정 상황극 내 발화 세그먼트 텍스트 파일(.txt)|

<br/>

### Preprocessing
- 발화자와 청자의 감정 Label 중 발화자의 Label만 추출
- 감정 Label이 여러 개일 경우, 1개만 채택
- 한글을 제외한 단어 및 공백 제거
- 인덱스가 일치하는 텍스트 파일(.tsv), 오디오 파일(.npy) 생성

<br/>

## Usage
### Text Model (KoBERT)

```
  git clone https://github.com/SKTBrain/KoBERT.git
  # 경우에 따라 https://huggingface.co/skt/kobert-base-v1 에서 다운로드 후 사용해야 함.
```
```
  python text_train.py
  python text_test.py
```

### Audio Model (ModifiedAlexNet)
```
  python audio_train.py
  python audio_test.py
```

### Text-Audio Model (CM-BERT)
```
  python run_classifier.py
```

## Experiments (Ensemble)
Text, Audio, Text-Audio 세 모델로부터 추출한 _truth / _pred / _output npy 파일들을 사용해 앙상블 학습 진행

| |Accuracy|F1 Score|Precision|Recall|
|-----|-----|-----|-----|-----|
|<strong>Stacking</strong>|<strong>0.7269</strong>|<strong>0.7138</strong>|<strong>0.7226</strong>|<strong>0.7269</strong>|
|Soft voting|0.6914|0.6651|0.6814|0.6914|
|Weighted soft voting|0.7012|0.6833|0.7016|0.7012|
|Hard voting|0.6844|0.6633|0.6861|0.6844|


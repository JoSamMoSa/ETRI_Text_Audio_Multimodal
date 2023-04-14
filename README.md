# Text-Audio Multimodal Emotion Recognization
[제2회 ETRI 휴먼이해 인공지능 논문경진대회](https://aifactory.space/competition/detail/2234)

## Feature Fusion 방식과 Decision Fusion 방식을 결합한 멀티모달 감정 인식에 관한 연구
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
| |Accuracy|F1 Score|Precision|Recall|
|-----|-----|-----|-----|-----|
|Stacking|<strong>0.7269</strong>|0.7138|0.7226|0.7269|
|Soft voting|0.6914|0.6651|0.6814|0.6914|
|Weighted soft voting|0.7012|0.6833|0.7016|0.7012|
|Hard voting|0.6844|0.6633|0.6861|0.6844|


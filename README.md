# 2024년 2학기 중간 경진대회
> - 딥러닝/클라우드 2분반
> - 시계열 데이터 예측
>

## 사용방법
1. 필요한 패키지 설치
    - `requirements.txt`
2. 데이터 셋 준비
    - `./data`
3. 파일 설명
    - `eval.py` : 각 모델별 성능 평가
    - `eda.py` : 데이터 특성 EDA
    - `feature_selection.py` :  최적의 feature를 선택하는 코드
    - `config.py` : Hyper parameters 등 학습 및 추론 조건
    - `inference.py` : 추론결과 `./output`에 csv 파일 저장
    - `ensemble.py` : `./output`에 있는 모든 csv 파일 Hard voting

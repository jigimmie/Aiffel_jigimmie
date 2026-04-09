
Autoint_mlp_train.ipynb
Autointmlp.py
Show_st_plus.py

Base line:
# 에포크, 학습률, 드롭아웃, 배치사이즈, 임베딩 크기 등 정의
epochs=5
learning_rate= 0.0001
dropout= 0.4
batch_size = 2048
embed_dim= 16

<img width="173" height="37" alt="image" src="https://github.com/user-attachments/assets/645f1c85-9c7f-4439-90e5-6868ecfee7d3" />

 

<streamlit 실행 결과>
<img width="289" height="232" alt="image" src="https://github.com/user-attachments/assets/d848086b-d0a5-437a-9091-a0de5d82d7b9" />

<img width="244" height="254" alt="image" src="https://github.com/user-attachments/assets/99ca43a3-4e7b-45dc-9edf-609d3bdee6f4" />


 
 

<추가 실험>
-epoch 5까지 val_loss가 계속해서 줄어듦.
Epoch를 키우면 더 줄어들 수 있겠다는 판단? -> 12으로 증가!
혹시나 epoch을 늘리면 과적합될수도 – dropout을 0.5로 조금 증가

<img width="339" height="274" alt="image" src="https://github.com/user-attachments/assets/ef21f34c-fedf-4214-bfff-d3455ec3027b" />

 
계속 줄어들다가 마지막쯤에는 0.538~0.539쯤에서 맴도는 것을 보아 이 정도의 epoch이 최적일 듯하다. 혹은 조금 더 한다면 미세하게는 더 줄일 수는 있을수도..


 <img width="165" height="32" alt="image" src="https://github.com/user-attachments/assets/1361ad22-61ea-45c3-96a6-590ed6b21860" />

Test set으로 평가한 결과, 아주 미묘하게 성능이 향상한 것을 확인할 수 있었다. 

이 이외에도 embedding 크기를 늘리거나 DNN 구조를 확장하는 등의 튜닝을 시도해볼 수 있을듯하다. 
다만 시간관계 상 그 내용들은 추가로 진행해보지 못하여서 아쉬운 마음이다.

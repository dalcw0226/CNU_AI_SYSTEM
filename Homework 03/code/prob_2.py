# 2. 인공 신경망 (ANN)

# 라이브러리 임포트
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1.py에서 sigmoid_list 함수 가져오기
from sigmoid import sigmoid_list  # 앞에서 작성한 sigmoid 와 sigmoid_list 함수를 모듈화 함.

# 2-1) 데이터 읽어오기
# 데이터의 위치는 보고서에 디렉토리 구조를 따로 첨부하겠음
df = pd.read_csv("../source/ann_data.csv")
print(df) # 해당 출력은 vsc에서의 출력이므로 주피터 노트북에서 실행한 과제 파일과 다를 수 있음


# 2-2) 초기의 인공신경망이 주어진 학습데이터를 잘 분류 하는가?
w = [0.5, 0.5]; b = 0

def visualize(df,w,b):
    for idx in range(len(df)):

        x1_1 = 0
        y1_1 = (-w[0]*x1_1-b)/w[1]

        x1_2 = 0.8
        y1_2 = (-w[0]*x1_2-b)/w[1]

        # plot samples
        if df.iloc[idx,2] == 1:
            plt.plot(df.iloc[idx,0],df.iloc[idx,1],'bo')
        else:
            plt.plot(df.iloc[idx,0],df.iloc[idx,1],'ro')

    # plot classifier
    # 과제에서 indent 문제 있음..
    plt.plot([x1_1,x1_2],[y1_1,y1_2])
    plt.show()

visualize(df,w,b)

'''
초기의 인공신경망은 제대로된 분류를 하지못한다. 일단 위 코드에서의 이유로는 제대로 분류하지 못하는 가중치들로 초기화를 하였기 때문이다.
보통 딥러닝 학습에서는 특정 규칙에 따라서 초기화를 하는데, 보통은 데이터의 특성을 반영하지 못하는 초깃값이기 때문에 데이터를 잘 분류시키지 못한다.

따라서 학습을 통해서 가중치와 절편을 수정해 주어야한다.
'''

# 2-3) 입력으로 w,b,x1,x2를 받아 z를 계산하는 zeta 함수를 작성하여라
# 사실 마음같아서는 class로 wrapping 하고 싶은데..
x1 = np.array(df.iloc[:, 0:1]) 
x2 = np.array(df.iloc[:, 1:2])
tn = np.array(df.iloc[:, 2:3]) 

def zeta(w, b, x1, x2):
    data_size = len(x1) # 데이터의 사이즈를 x1의 개수를 가지고 계산한다.

    z = np.zeros((data_size, 1))

    for idx in range(data_size):  # 입력의 개수를 유동적으로 처리할 수 있도록 코드를 작성한다.
        z[idx] = w[0] * x1[idx] + w[1] * x2[idx] + b  # 퍼셉트론 연산
    
    return z

y = sigmoid_list(zeta(w, b, x1, x2))
print(y)

# 2-4) 경사하강법을 이용하여 학습 파라미터 w와 b를 학습하여라
# 모든 하이퍼파라미터 한 번 에 설정
w = [0.5, 0.5] ; b = 0 ; mu = 0.2 ; epoch = 5000

x1 = np.array(df.iloc[:, 0:1])
x2 = np.array(df.iloc[:, 1:2])
tn = np.array(df.iloc[:, 2:3])

# 전체 샘플의 개수
size = len(x1)

for i in range(epoch):
    y = sigmoid_list(zeta(w, b, x1, x2))

    # 업데이트 관련 식을 유도없이 그대로 사용하겠음
    delta_x1 = np.sum(-((tn - y) * y * (np.ones(size) - y) * x1)) # x1의 업데이트 값
    delta_x2 = np.sum(-((tn - y) * y * (np.ones(size) - y) * x2)) # x2의 업데이트 값
    delta_b  = np.sum(-((tn - y) * y * (np.ones(size) - y))) # b의 업데이트 값

    # 업데이트
    # 업데이트는 lr * delta로 한다.
    w[0] -= mu * delta_x1
    w[1] -= mu * delta_x2
    b    -= mu * delta_b

# 2-5) 최종 학습이 완료된 w, b를 이용하여 학습된 분류기 및 학습 데이터들을 시각화 하라
# 확인
print(f"파라미터 w1 : {w[0]}, w2 : {w[1]}, b : {b}")
visualize(df,w,b) 
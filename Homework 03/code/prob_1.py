# 1. sigmoid 함수
# 라이브러리 임포트
import math 
import numpy as np
import matplotlib.pyplot as plt

# 1-1) sigmoid 함수를 작성하라

def sigmoid(x):
    y = 1 / (1 + math.exp(-x))
    
    return y

# 실행 예
print(sigmoid(0.5))


# 1-2) 리스트를 입력으로 받아서 sigmoid 연산을 계산해주는 sigmoid_list 함수를 작성하여라
def sigmoid_list(x_list):

    y = np.zeros((len(x_list), 1))


    for idx, x in enumerate(x_list):
        # 위에서 정의한 시그모이드 함수를 다시 재활용하겠음.
        
        y[idx][0] = sigmoid(x)

    return y

# 실행 예
print(sigmoid_list([0,1,2])) # PRECISION에 대한 체크는 하지 않겠음.


# 1-3) ploting
x_list = np.arange(-10, 10.01, 0.01)  # 구간 설정
plt.plot(x_list, sigmoid_list(x_list))  # plot 그리기
plt.show()
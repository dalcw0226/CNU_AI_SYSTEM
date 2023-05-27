import numpy as np
import math

def sigmoid(x):
    y = 1 / (1 + math.exp(-x))
    
    return y

def sigmoid_list(x_list):

    y = np.zeros((len(x_list), 1))

    # 이런식으로 베이스라인코드를 준 것에 대한 의문..
    # numpy 객체는 브로드캐스팅 연산을 지원하는데..
    for idx, x in enumerate(x_list):
        # 위에서 정의한 시그모이드 함수를 다시 재활용하겠음.
        # 함수의 목적은 재사용이므로! (사실 진짜 재사용을 원한다면 class로 작성해야 맞긴하지만..)
        
        y[idx][0] = sigmoid(x)

    return y

# 3. 유전 알고리즘

# library import
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import random
import math

from plot_3d import plot_3d  # plot을 그리는 함수를 plot_3d.py 파일로 모듈화 하였다. 따라서 본 코드에는 해당 함수가 존재하지 않는다.

# randomseed fix
random.seed(10)
np.random.seed(10)

data = pd.read_csv("../source/p1_training_data.csv")
np_data = np.array(data)

# 1-1) 1세대 유전자를 초기화하여라
INIT_GENE = []  # 초기 유전자를 담고 있는 리스트
POPULATION = 100  # 인구 수를 지정하는 하이퍼파라미터

for i in range(POPULATION):

    # 1 세대를 위한 유전자 초기화
    w1 = random.uniform(-1, 1)
    w2 = random.uniform(-1, 1)
    w3 = random.uniform(-1, 1)
    b  = random.randint(-100, 100)

    # INIT_GENE에 차곡차곡 집어 넣는다.
    INIT_GENE.append([w1, w2, w3, b])

# 1-2) 각 유전자에 대한 fitness 계산법을 설계하고 가장 fitness가 높은 유전자들을 선별하라

# 데이터 셋 정의하기
x1 = np_data[0:, 0]
x2 = np_data[0:, 1]
x3 = np_data[0:, 2]
target = np_data[0:, 3]

# fitness는 Mean Squared Error 로 설정하고, 해당 수치가 낮은 값이 성능이 좋은 모델이다.
def MSE(target, y): # x, y는 numpy.array() 객체 형태로 들어온다.
    return np.mean((target - y) ** 2)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 모델의 feed forward 를 계산하는 함수
def forward(weight, x1, x2, x3):  # weight : 유전자 (리스트로 들어와야한다.), , x1 : 입력 1, x2 : 입력 2, x3 : 입력 3
    return sigmoid(weight[0] * x1 + weight[1] * x2 + weight[2] * x3 + weight[3])

# 가장 좋은 유전자를 선택하는 함수
# weight : 모든 유전자, x1 : 입력 1, x2 : 입력 2, x3 : 입력 3, k : 상위 몇 개의 가중치를 선별할 것인지, fitness_ok : fitness를 포함할 것인지 묻기 (default = True)
def select(weights, x1, x2, x3, target, k, fitness_ok=True):  
    result_pool = []

    # weights에서 한 개씩 뽑아와서 MSE를 계산한 후 가장 낮은 값 k개를 추출하는 과정
    for weight in weights:
        feed_forward_result = forward(weight, x1, x2, x3)
        fitness = MSE(target, feed_forward_result)

        result_pool.append((fitness, weight))

    # 정렬하기 -> key : fitness
    result_pool = sorted(result_pool, key=lambda x : x[0])

    if fitness_ok:
        return result_pool[:k]
    
    else: 
        return [x[1]  for x in result_pool][:k]

# 외부에 정의된 함수를 불러와 사용
# np_data, GENE을 매개변수로 받는다.
print("앞으로 4번의 창이 나옵니다. 해당 plot은 1세대 rank1 ~ 4를 나타냅니다.")
print(select(INIT_GENE, x1, x2, x3, target, 4))
for i in range(4):
    plot_3d(np_data, select(INIT_GENE, x1, x2, x3, target, 4, False)[i]) 

# 1-3) 선택된 유전자들에 대해 Crossover와 mutation을 수행하여 2세대 유전자들을 생성하시오.

# 2개의 선택된 개체를 교차 시킨다.
# 이 때 결정해 주어야 하는 값으로는 몇 개의 유전자를 교차 할 것인지,
# 몇 변 인덱스의 유전자를 교차할 것인지를 결정한다.
def crossover(weight1, weight2):
    weight1_ = weight1.copy() ; weight2_ = weight2.copy()  # 얕은 복사를 하면 중복되는 값이 생기는 문제가 발생. 따라서 깊은 복사로 전환
    # print(weight1_, weight2_)
    num = random.randint(0, 4) # 몇 개의 유전자를 교차할 것인지. (weight1을 기준으로 결정한다.)
    idx_ls = []

    for _ in range(num): # 만약 같은게 2번 나온다면 그것도 자연법칙이라고 간주한다.
        idx = random.randint(0, 3)

        if not (idx in idx_ls):
            idx_ls.append(idx)
    
    # 파이썬에서만 성립하는 문법 : temp 없이 swap 하기
    for idx in idx_ls:
        weight1_[idx], weight2_[idx] = weight2_[idx], weight1_[idx]

    return weight1_, weight2_


# 돌연변이 (변이)
# 확률은 0.0001로 결정한다. 아마 전체 세대에서 2개만 일어나는것이 기대값임.
def mutatue(weight):
    # 변이의 확률은 0.0001로 결정한다
    num = 0
    random_num = random.randint(0, 10000)

    # 0.0001 확률이 성립하는 경우에만 아래 코드를 실행
    if num == random_num:
        # 한 개의 유전자에 대해서만 변이를 일으킴
        mutate_idx = random.randint(0, 3)
        # 다시 새로운 값으로 갈아 끼움
        weight[mutate_idx] = random.uniform(-1, 1)
    
    # 보통은 입력 받은 값을 그대로 리턴할 것이다.
    return weight

# 자식을 생성하는 함수
def hybridization(weight):
    hybrid_weight = []
    for i in range(100):
        for j in range(i+1, 100):
            gene1 = weight[i]
            gene2 = weight[j]
            # 교차 실행 -> 교차 후 자식은 2개이다.
            gene1, gene2 = crossover(gene1, gene2)

            # 변이 실행
            gene1 = mutatue(gene1)
            gene2 = mutatue(gene2)

            hybrid_weight.append(gene1)
            hybrid_weight.append(gene2)

    # 중복되는 값을 제거
    # 중복되는 값이 왜 발생하나? => crossover을 하는데, 유전자가 한 개도 교차되지 않는 경우도 있기 때문이다.
    # 설계한 정책에 따라 움직인다.
    hybrid_weight_ = []
    for w in hybrid_weight:
        if w in hybrid_weight:
            hybrid_weight_.append(w)

    return hybrid_weight_

# 2세대
all_hybreid_2 = hybridization(INIT_GENE)
generation_2 = select(all_hybreid_2, x1, x2, x3, target, 100, False)  # 가장 좋은순서대로 100개의 유전자 선택

# 1-4) (2, 3) 과정을 반복하여 2세대, 3세대에서 가장 fitness가 높은 상위 4개의 유전자에 대해 fitness score를 적고, 분류 평면을 도식화 하라.
# 2세대 생성
# fitness 지표 출력
print("앞으로 4번의 창이 나옵니다. 해당 plot은 2세대 rank1 ~ 4를 나타냅니다.")
print(select(all_hybreid_2, x1, x2, x3, target, 4))

# 그래프 출력
for i in range(4):
    plot_3d(np_data, select(generation_2, x1, x2, x3, target, 4, False)[i])


# 3세대 생성
all_hybreid_3 = hybridization(generation_2) # 교배
generation_3 = select(all_hybreid_3, x1, x2, x3, target, 100, False)  # 가장 좋은순서대로 100개의 유전자 선택

print("앞으로 4번의 창이 나옵니다. 해당 plot은 3세대 rank1 ~ 4를 나타냅니다.")
# fitness 지표 출력
print(select(all_hybreid_3, x1, x2, x3, target, 4))

# 그래프 출력
for i in range(4):
    plot_3d(np_data, select(generation_3, x1, x2, x3, target, 4, False)[i])

# 1-5) 최종적으로 유전 알고리즘을 통해 얻어진 유전자 중 가장 Fitness가 높은 유전자에 대해 분류 평면을 도식화 하라.
# 또한 어떤 조건으로 유전 알고리즘을 종료하였는지 작성하라.
print("유전 알고리즘은 3세대에서 종료, 다음은 최 상위 모델 파라미터에 대한 분류 결과입니다.")
print(select(generation_3, x1, x2, x3, target, 1))
plot_3d(np_data, select(generation_3, x1, x2, x3, target, 1, False)[0])
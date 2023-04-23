# 4 Decision Tree
# 4 - 1 : train_data.csv 파일의 학습데이터 100개를 이용하여 decision tree를 학습하시오

# library import
import math
import numpy as np
import pandas as pd

# 데이터 불러오기
data = pd.read_csv("../data/train_data.csv")
X = np.array(data.iloc[:, 0:4])
y_sub = np.array(data.iloc[:, -1])

# 고양으를 0으로 강아지를 1로 재정의 한다.
# 데이터 전처리
y = []
for i in y_sub:
    if i == "Dog": y.append(1)  # 개를 1로 정의하고
    else: y.append(0)  # 고양이를 0으로 정의한다.
y = np.array(y)


# decision tree node  결정하기
class Node:
    def __init__(self, idx):
        self.left = None  # True 인 경우의 객체 저장
        self.right = None  # False 인 경우의 객체 저장
        self.condition = None  # 어떤 조건을 기지고 다음 노드를 구별할 것인지 판별

        # 데이터는 전체를 들고다니면서 관리하지 않고 인덱스로 접근해서 관리한다.
        self.current_idx = idx  # 현재 노드가 가지고 있는 인덱스

        # 노드의 상태를 저장 (고양이? 개?)
        self.status = None

# 분할하는 함수 정하기
def split_node(X, condition, elements): # X는 전체 학습 데이터, condition은 분할 조건을 의미, elements는 분할 대상에 놓여있는 원소들의 인덱스를 의미
    positive_idx = []
    negative_idx = []

    for idx, state in zip(elements, X[elements, condition]):
        if state == 1: positive_idx.append(idx)
        else: negative_idx.append(idx)
    
    return positive_idx, negative_idx

# entropy를 계산하는 함수 정하기
def entropy(y, idx): # y는 정답 데이터 셋을 의미하고, idx는 분할 결과로 나온 positive 또는 negative의 인덱스 리스트이다.
    length = len(idx)
    zero = sum(y[idx] == 0)  # 고양이를 0으로 정의한다.
    one = sum(y[idx] == 1)  # 강아지를 1로 정의한다.

    try : # log의 연산상의 문제때문에 예외처리 도입
        entropy_result = -(zero/length * (math.log2(zero/length)) + one/length * (math.log2(one/length)))
        return entropy_result
    except:
        return -999  # 엄청 큰 값을 주어 순수 노드가 될 시 무조건 해당 조건으로 분할되도록 설계

# entropy 차이 (gain)를 계산하는 함수
def gain(X, y, condition, elements):  # X: 전체 데이터 셋, y: 정답 데이터 셋, condition: 어떤 상태의 특성으로 분할할지, element: 분할 대상 원소의 인덱스
    pos_idx, neg_idx = split_node(X, condition, elements)

    total_entropy = entropy(y, elements)

    pos_entropy = entropy(y, pos_idx)
    neg_entropy = entropy(y, neg_idx)

    return total_entropy - (len(pos_idx)/len(elements) * pos_entropy + len(neg_idx)/len(elements) * neg_entropy)


# 트리를 정의할 엔진 구현
def engine(node, X, y):

    # 현재 노드에서 가르키고 있는 상태 기입하기
    if sum(y[node.current_idx] == 0) > sum(y[node.current_idx] == 1):  # 고양이가 더 많은 경우 고양이로 분류
        node.status = 0
    elif sum(y[node.current_idx] == 0) < sum(y[node.current_idx] == 1):  # 강아지가 더 많은 경우 강아지로 분류
        node.status = 1
    else :  # 그 외의 경우 if 같은 숫자의 개수
        node.status = -1

    # 분할 종료 조건 -> 재귀함수로 구현하므로 재귀 종료조건이 필요하다.
    if all(y[node.current_idx] == 1) or all(y[node.current_idx] == 0):
        return node
    
    # 최적의 특징 찾기 <- GAIN을 이용해서 구한다
    max_feature = -1
    max_gain = -1
    for condition in range(4):
        GAIN = gain(X, y, condition, node.current_idx)
        if GAIN > max_gain:
            max_feature = condition
            max_gain = GAIN
    # 가장 최적의 컨디션을 노드에 기입
    node.condition = max_feature

    # 찾은 조건을 가지고 분할하기
    pos_idx, neg_idx = split_node(X, max_feature, node.current_idx)

    # 트리를 확장하기
    # C언어에서의 포인터 느낌으로 확장하기
    node.left = Node(pos_idx)
    node.right = Node(neg_idx)

    # 재귀적으로 확장하기
    engine(node.left, X, y)
    engine(node.right, X, y)

# 학습된 트리를 가지고 결과를 예측하는 함수
def predict(root, X):  # tree : 학습된 트리 객체, X : 입력 데이터 - 2차원
    predict = []

    for data in X:
        node = root

        while node.left:  # 왼쪽과 오른쪽이 둘 중 한개만 None인 경우는 없기 때문에 한개만 비교한다. => 자체 shortcut
            if data[node.condition]: node = node.left
            else : node = node.right

        predict.append(node.status)

    return predict

root = Node(list(range(100)))  # 확장성 측면에서 아쉬운 하드코딩, but pass

# 결정 트리를 생성한다.
engine(root, X, y)
# print(predict(root, X))  # 테스트를 위한 훈련 세트로 확인



# 트리의 모양 결정을 위한 코드 <- 실제 동작에서는 무의미한 코드이므로 주석처리.

'''
print(root.status) # -1
print(root.left.status) # 1
print(root.left.left.status) # 0
print(root.left.left.left.status) # 1
print(root.left.left.right.status) # 0
print(root.left.right.status) # 1
print()
print(root.right.status) # 0
print(root.right.left.status) # 0
print(root.right.right.status) # -1
print(root.right.right.left.status) # 1
print(root.right.right.right) # 0
'''

# 4 - 2 test_data.csv 파일의 테스트 데이터 10개에 대한 테스트 결과를 출력하시오

# 데이터 읽어오고 분할하기
test_data = pd.read_csv("../data/test_data.csv")
test_data = np.array(test_data)

X_test = list(test_data[:, 0:4])
y_hat = predict(root, X_test)

# 숫자로 표기된 것을 문자로 바꿔준다.
y_hat_cat = []
for y in y_hat:
    cat = "Cat" if y == 0 else "Dog"
    y_hat_cat.append(cat)


for idx, (source, pred) in enumerate(zip(test_data, y_hat_cat)):
    print(f"Test #{idx} {source} -> {pred}")
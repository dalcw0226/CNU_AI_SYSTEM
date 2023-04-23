# library import
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Linear Classification
# 3 - 1 : Linear_classification.txt 파일을 파일 입출력 객ㄱ체를 활용하여 읽고 각각 list에 데이터를 저장하시오.
data = []

f = open("..\data\Linear_classification.txt", "r")
while True:
    line = f.readline().split()
    if not line:
        break

    data.append(line)

f.close()
# 데이터를 넘파이 어레이로 바꾸어준다
data_array = np.array(data)

# 3 - 2 : 최소제곱법를 이용하여 문제 3-1에서 읽은 데이터에 대한 linear classification을 수행하라
# 학습을 위해서 데이터를 전처리한다.
X = np.array(data_array[:, 1:3], dtype=float)
y = np.array(data_array[:, 3:], dtype=float)

# 데이터 전처리 확인
    # 전처리로는 학습 데이터와 정답 데이터의 분리 및 실수화를 진행하였음
# print(X.shape, y.shape)

# 데이터에 대한 학습 모델은 class 형태로 작성하며,
# 클래스 명은 Linear_Classification으로 정의

class Linear_Classification:
    def __init__ (self):
        # 객체내 멤버 변수를 3개 선언한 함
        # 코드의 안전성을 위해서 일단은 모두 NULL(None)으로 채워둠
        self.W = None  # [[w1, w2, b]] 의 순서로 저장됨.
        self.coef_ = None
        self.intercept_ = None

    # 학습을 위한 멤버 함수 fit 을 정의함
    # 값을 입력을 받아서 최적의 가중치를 찾아주는 함수 임.
    # sklearn에서의 fit함수와의 동작은 동일함.
    def fit(self, X, y):
        # 내부적으로 bias의 값을 처리하기 위해서 입력 데이터의 맨 마지막 원소를 1로 재워준다
        X_ = np.c_[X, np.ones((len(X), 1))]
        self.W = np.linalg.inv(X_.T.dot(X_)).dot(X_.T).dot(y)

        self.coef_ = self.W[0:2]
        self.intercept_ = self.W[2]
        return self.W
    
    def predict(self, X): # X는 반드시 2차원으로 들어온다. (다중 입력을 대비하기 위해서 확장성을 고려함
        predict = []
        # fit함수에서 구한 선형 방정식을 구함.
        for i in X:
            predict.append(self.W[0] * i[0] + self.W[1] * i[1] + self.W[2] * 1)

        return np.array(predict)
    
# 결과 출력
clf = Linear_Classification()
clf.fit(X, y)
print(f"가중치 정보 출력 : w = ({clf.coef_[0].item()}, {clf.coef_[1].item()}), b = {clf.intercept_.item()}")

# 3 - 3 : 구해진 선형 모델 및 학습 데이터를 matplotlib을 이용하여 시각화하시오.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_min = min(X[:, 0]) ; x_max = max(X[:, 0])
y_min = min(X[:, 1]) ; y_max = max(X[:, 1])

# scatter 값 그리기
ax.scatter(X[:, 0], X[:, 1], y.reshape(-1))


# surface 그리기
x_ = np.linspace(x_min, x_max, 100)
y_ = np.linspace(y_min, y_max, 100)
x_, y_ = np.meshgrid(x_, y_)

z_ = clf.coef_[0].item()*x_ + clf.coef_[1].item()*y_ + clf.intercept_.item()
ax.plot_surface(x_, y_, z_, alpha=0.5)

plt.show()

# 3 - 4 : 학습된 선형 분류 모델에 의해 용액S 47, 용액T 29인 실험은 성공인가? 실패인가?
result = clf.predict([[47, 29]]).item()
status = "success" if result > 0 else "fail"
print(f"용액S 47ml, 용액T 29ml인 실험은 {result:.3f} 이므로 {status}이다.")
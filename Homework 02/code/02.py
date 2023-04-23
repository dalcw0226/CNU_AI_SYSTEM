# 라이브러리 임포트
import numpy as np
import matplotlib.pyplot as plt 


# 2 - 1 : Linear_regresion.txt 파일을 파일 입출력 객체를 활용하여 읽고 각각 list에 데이터에 저장하시오
data = []

f = open("..\data\Linear_regression.txt", "r")
while True:
    line = f.readline().split()
    if not line:
        break

    # data 리스트에 값을 추가한다.
    data.append(line)
# 파일 객체 닫기
f.close()

# data 리스트를 넘파이 객체로 바꾼다
data_array = np.array(data)

# 2 - 2 : 최소 제곱법을 이용하여 문제 2 - 1에서 읽은 데이터에 대한 linear regression을 수행하라
# numpy 패키지 활용 가능 (sklearn과 같은 머신러닝 패키지 활용 불가능)
    # 쓰지 못하면 만들면 되는거지!

# 데이터 셋을 preprocessing을 한다.
# 데이터 셋은 2차원 행렬 형태로 주어져야한다.
    # 왜냐? 선형대수적 연산을 수행하기 위해서
X = np.array(np.c_[data_array[:, 1], np.ones(len(data_array))], dtype=float) # bias를 의미하는 1의 값을 추가
y = np.array(data_array[:, 2].reshape(1, -1), dtype=float)

# 문제에 대한 모듈은 클래스로 구성
# 사이킷런과는 다르게 그냥 데이터를 통째로 받고, 클래스 내부에서 처리한다.
    # 물론 이렇게 처리하면, 확장성 측면에서 이슈가 생기기는 하지만.. 과제니까..

# 나중에 다시 돌리는 과정이 있을 수 있지만. 정규 방정식의 원형을 그대로 살리기 위해서
# 손실 함수는 MSE를 사용한다.
class Linear_Regression:
    def __init__(self):
        self.W = None # 가중치에 대한 정보
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        y_reshape = y.reshape(1, -1)
        self.W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y_reshape.T)

        self.coef_ = self.W[0].item()
        self.intercept_ = self.W[1].item()
        return self.W
    
    def predict(self, X): # 일관성있게 2차원으로 만든다.
        predict = []
        for i in X:
            predict.append(self.W[0] * i[0] + self.W[1] * 1)

        return np.array(predict)

# 결과 출력
lr = Linear_Regression()
lr.fit(X, y)
print(f"가중치 정보 출력 : w = {lr.coef_}, b = {lr.intercept_}")

# 2 - 3 : 구해진 선형 모델 및 학습 데이터를 matplotlib를 이용하여 시각화하시오
min_X = X.min()
max_X = X.max()

plt.scatter(X[:, 0], y)
# 그냥 플롯 그리던대로 그리면 된다.
plt.plot([min_X, max_X], [lr.coef_ * min_X + lr.intercept_, lr.coef_ * max_X + lr.intercept_])
plt.show()

# 2 - 4 : 학습된 선형 모델에 의해 7.3 시간 작업하였을 때 완성되는 인형 수는 몇 개라 예측할 수 있는가?
# 2차원으로 주어야한다.
print(f"학습 모델에 의해 예측한 7.3시간 작업한 결과 완성된 인형은 {lr.predict([[7.3]]).item():.2f}개 이다.") # 소수점 이하 둘째자리 절삭
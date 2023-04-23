# Numerical differential

# 1 - 1
def numerical_diff(x):
    epsilon = 1e-10
    second_point = x + epsilon  # 원래의 x와 아주 가까운 가장 작은 값을 규정

    p1 = x*x+2*x+1
    p2 = (lambda x : x*x+2*x+1)(second_point) # 두 번째 점에 대한 y 좌표 정의
    
    return (p2 - p1) / epsilon # 기울기 계산

print("x^2+2x+1 함수의 x=2지점에서의 미분값은", numerical_diff(2), "입니다.")


# 1 - 2
# numerical_differential 값과 symbolic differential 값 간의 오차

# Numerical differential
def numerical_diff(x):
    epsilon = 1e-10
    second_point = x + epsilon  # 원래의 x와 아주 가까운 가장 작은 값을 규정

    p1 = x*x+2*x+1
    p2 = (lambda x : x*x+2*x+1)(second_point) # 두 번째 점에 대한 y 좌표 정의
    
    return (p2 - p1) / epsilon # 기울기 계산

result_num = numerical_diff(2)

# symbolic differential을 이용하여 미분 결과를 도출
import sympy as sym 

x = sym.Symbol('x')
y = x**2 + 2*x + 1

y_ = sym.diff(y)
result_sym = float(y_.subs(x, 2))

# 두 연산 사이의 오차
print(f"오차(result_num - result_sym)는 {result_num - result_sym:.10f} 이다.")

# 오차는 0.0000004964 정도 된다.
# 오차를 줄이기 위해서는 epsilon으로 설정한 두 점 사이의 거리가 좁으면 좋을수록 좋다.

# 사실 이 근사치로 얻은 값은 인공지능분야에서 유용한 값이라고 생각한다.
# 인공지능 분야에서 미분을 사용하는 이유로는 목적함수의 값을 최소화하기 위해서 사용하는데, 이때 사용하는 옵티마이저 기법에 의하면,
# 물론 기울기의 값도 중요하지만, 더 중요하게 영향을 받는것은 경사 하강의 방향, 즉 부호이다.

# 이렇게 근사치로 이용을 한다면, 분명 오차는 존재한다. 그러나, 미분한 결과의 값이 엄밀한 값을 원하는 경우를 제외하고는 해당 근사치를 사용해도 문제가 되지 않는다.

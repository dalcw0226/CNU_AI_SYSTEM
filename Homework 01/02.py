# 2. 인공지능학부 디저터 나눔 사건
from math import dist
from collections import deque

# 좌표 변환 해석 코드
def transfer(x, y, cx, cy, r):
    return (x + cx - r, y + cy - r)

# 이동할 방향 정의
dx = [-1,1,0,0]
dy = [0,0,-1,1]

# BSF Engine
# 너비우선 탐색을 하면서 중심으로부터 1인 사람을 셀 것이다.
def bfs(graph, radius):
    count = 0
    # 초기값으로 중앙 좌표를 집어 넣는다.
    queue = deque([(radius, radius)])

    while queue:
        # 큐에 들어간 값을 한개씩 빼면서 탐색을 진행한다.
        x, y = queue.popleft()
        # 이미 지나간 자리에 대해서는 0으로 전환한 후 다시 돌아오지 않도록 한다. => 일종의 마킹
        graph[x][y] = 0
        
        # 상하좌우 탐색
        for i in range(4):
            new_x = x + dx[i] ; new_y = y + dy[i]
            
            # 범위를 벗어나는 경우에는 무시한다.
            if new_x < 0 or new_x >= 2*radius+1 or new_y < 0 or new_y >= 2*radius+1:
                continue
            
            # 갈 수 있는 노드에 대해서 탐색을 진행한 후 새로운 좌표를 append 한 후 
            # 디저트를 전달 하고 나서 전달 횟수를 업데이트한다.
            if graph[new_x][new_y]:
                queue.append((new_x, new_y))
                graph[new_x][new_y] = 0
                count += 1

    return count

# 원의 중심좌표와 가로, 세로 길이 입력
cx, cy = map(int, input("중심 좌표 입력: ").split())
radius = int(input("원의 반지름 입력: "))
max_x = cx + radius + 1 ; max_y = cy + radius + 1 # 최대 x 와 최대 y 설정

# 초기 설정 <- 인공지능학부 = 1, 그 외의 사람 = 0
# 초기 그래프 생성 <- 중심 (cx, cy) 를 (r, r)로 이동시킨것을 가정하고 그래프를 구상한다. 
# (그러면 그래프의 최대 크기는 2*r by 2*r이 된다.)
# 이제부터의 중심은 (radius, radius) 이다.
# 초기값는 모두 인공지능학부 사람인 것을 가정한다.
graph = [[1] * (2*radius+1) for _ in range(2*radius+1)]

# 그래프 세부 마킹 <- 인공지능학부 = 1 표시하는 작업
for i in range(2 * radius + 1):
    for j in range(2 * radius + 1):
        # 원 밖에 있는 사람들은 0으로 바꿔준다.
        if dist((i, j), (radius, radius)) >= radius: graph[i][j] = 0
        # 규칙에 따라 인공지능학부가 아닌 사람은 0으로 바꿔준다.
        x = list(map(int, list(str(abs(transfer(i, j, cx, cy, radius)[0])))))
        y = list(map(int, list(str(abs(transfer(i, j, cx, cy, radius)[1])))))
        result = sum(x + y)

        # 드모르간 법칙 이용하여 논리식을 바꿔준다.
        if (result % 2 != 0) and (result > 16):
            graph[i][j] = 0 

answer = bfs(graph, radius)
print(f"디저트를 받은 인공지능학부 학생은 총 {answer}명 입니다.")
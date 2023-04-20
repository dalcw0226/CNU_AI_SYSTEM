# 1번 문제 신입생에게 전대마을 소개하기

# dfs engine
def dfs(village, x, y, n_row, n_col):
    # 초기 블럭 칸 수 ; 1로 표기된 한 개의 구역을 의미 함.
    cnt = 1
    # 예외처리
    # 만약 블럭을 넘어서서 탐색을 한다면 return 0를 준다. (넘어선 공간에 대해서는 0개의 블럭이 있다는 의미)
    if x < 0 or x >= n_row or y < 0 or y >= n_col:
        return 0
    
    # 탐색 시작
    # 만약 (x, y)에 대해서 1인 블럭이 있다면, 연결된 블럭을 탐색
    if village[x][y]:
        # 일단 탐색을 marking 한다.
        # 1 -> 0
        village[x][y] = 0
        
        # 탐색의 시작이 되는 블럭을 중심으로 8개의 블럭을 재귀적으로 탐색한다.
        # cnt를 accumulate로 이용하여 1로 marking 된 블럭을 탐색한다.
        cnt += dfs(village, x-1, y-1, n_row, n_col)
        cnt += dfs(village, x  , y-1, n_row, n_col)
        cnt += dfs(village, x+1, y-1, n_row, n_col)
        cnt += dfs(village, x-1, y,   n_row, n_col)
        cnt += dfs(village, x+1, y,   n_row, n_col)
        cnt += dfs(village, x-1, y+1, n_row, n_col)
        cnt += dfs(village, x  , y+1, n_row, n_col)
        cnt += dfs(village, x+1, y+1, n_row, n_col)
        # 모든 탐색이 종료되면 누적된 cnt를 리턴한다.
        return cnt
    # 만약 (x, y)가 1이 아니라면 0을 리턴한다.
    return 0

# 실행 코드
def run(dfs, village, n_row, n_col):
    storage = 0 # 창고의 개수
    house = 0 # 집의 개수

    # 행렬에서 한 개씩 순차적으로 탐색한다.
    for i in range(n_row):
        for j in range(n_col):
            # 결과를 받은 후 리턴값의 크기를 통해 집과 창고를 구분하여 누적한다.
            result = dfs(village, i, j, n_row, n_col)
            # 1이라면 창고
            if result == 1:
                storage += 1
            # 아니라면 집
            elif result > 1:
                house += 1
    # 튜플로 리턴
    return (storage, house)
            
# 전대 마을 입력
n_row, n_col = map(int, input("마을의 행과 열 개수를 공백 기준으로 분리하여 입력: ").split())
village = []
for i in range(n_row):
    col = list(map(int, input(f"{i} 행 입력: ").split()))
    village.append(col)

storage, house = run(dfs, village, n_row, n_col)

print(f"전대마을에는 {house}개의 집, {storage}개의 창고가 있습니다.")
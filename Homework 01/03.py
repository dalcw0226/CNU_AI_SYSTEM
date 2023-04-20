#  3. 감시 시스템을 피해 탈출하기
# 일단, 본 문제는 위치 변환에 따른 비용이 모두 1이므로 BFS를 이용하면 최단 경로를 구할 수 있다.

from collections import deque

# 다음 나올 수 있는 우주선의 위치를 추정하는 함수
# x : 현재 위치 Form : [(), ()]
dx = [-1, 1,  0, 0]
dy = [ 0, 0, -1, 1]

# 그냥 평범한 이동 (상하좌우)
def udlr(board, locate):
    # 가능한 위치를 리스트에 담는다.
    avail_ls = []

    # 사용하기 편하도록 다시 변수를 선언
    x1 = locate[0][0]
    y1 = locate[0][1]
    x2 = locate[1][0]
    y2 = locate[1][1]

    # 다음 가증한 위치에 대해서 살펴본다.
    for i, j in zip(dx, dy):
        # 다음 이동 할 수 있는 위치에 대해서 훑어본다.
        if board[x1+i][y1+j] == 0 and board[x2+i][y2+j] == 0:
            avail_ls.append([(x1+i, y1+j), (x2+i, y2+j)])
    return avail_ls

# 안 평범한 이동 == 회전 고려
def lotate(board, locate):
    # 가능한 위치를 리스트에 담는다.
    avail_ls = []

    # 사용하기 편하도록 다시 변수를 선언
    x1 = locate[0][0]
    y1 = locate[0][1]
    x2 = locate[1][0]
    y2 = locate[1][1]

    # 논리식으로 모두 처리한다. 
    # 하드코딩, 어떻게 처리할까 생각을 해봤는데 색다른 아이디어가 떠오르지 않아 모두 하드코딩한다.
    # 누워 있는 경우
    if(x1 == x2):
        dx = [-1, 1] # 이걸 반복하여 누워있는 케이스에 대해서 처리한다.
        # 누워있는 경우에 대해서 회전을 할 때는 x의 좌표가 변하지 않으므로 delta x에 대해서는 0으로 지정한다.
        for i, j  in zip(dx, [0, 0]):
            if(board[x1+i][y1] == 0 and board[x2+i][y2] == 0):
                avail_ls.append([(x1+i, y1), (x1, y1)]) 
                # 음.. 순서는 상관없다. 어차피 목적지에 도달했는지의 상태만 확인할것이기 때문에
                avail_ls.append([(x2+i, y2), (x2, y2)])
    # 로케트가 서 있는 경우
    elif(y1 == y2):
        dy = [-1, 1] # 이걸 반복해서 서 있는 케이스에 대해서 처리한다.
        # 음.. 유사한 논리로 처리
        for i, j in zip([0, 0], dy):
            if(board[x1][y1+j] == 0 and board[x2][y2+j] == 0):
                avail_ls.append([(x1, y1+j), (x1, y1)])
                avail_ls.append([(x2, y2+j), (x2, y2)])

    return avail_ls

# board 에 padding을 씌운다.  - 차용한 아이디어
# (1로, 왜냐! 조건 처리하기 귀찮으니까, padding을 씌워서 그냥 우주선을 밖으로 못나가게 한다.)
def padding_by_1(board, n, m):
    board_padding = []
    board_padding.append([1] * (m+2))
    for i in range(n):
        board_padding.append([1] + board[i] + [1])
    board_padding.append([1] * (m+2))
    return board_padding


# BSF 엔진 (BSF 사용 이유 : BFS는 각 변화의 비용이 같은 경우에는 최단 경로를 탐색한다.)
def BFS(board, n, m):
    new_board = padding_by_1(board, n, m)

    # queue 정의
    queue = deque()
    visit = [] # 방문한 노드 관리
    # 초기값 queue에 append 후 visit에 추가
    queue.append(([(1,1), (1,2)], 0))
    visit.append([(1,1), (1,2)]) 

    # queue 가 비면 종료
    while queue:
        locate, count = queue.popleft()
        
        # short cut : 모든 노드를 탐색하기 전에 원하는 결과가 나오면 종료한다.
        if (n, m) in locate:
            return count
    
        # queue 에 가능한 상황 추가
        for i in udlr(new_board, locate) + lotate(new_board, locate):
            if i not in visit: # 방문 하지 않은 노드에 대해서만 탐색
                queue.append((i, count+1)) # 이동 횟수 업데이트
                visit.append(i) # 방문 노드 처리

# 감시 공간 크기 입력
n, m = map(int, input("감시 공간 크기 입력: ").split())

board = []
for s in range(n):
    tmp = list(map(int, input(f'{s}행 입력: ').split()))
    board.append(tmp)

answer = BFS(board, n, m)
print(f"감시 공간을 탈출할 수 있는 최단 시간은 {answer}초 입니다.")
import math

class Game():
    def __init__(self, size, board, player, computer):
        self.size = size         # 구름게임 맵 크기 (정방행렬)
        self.board = board       # 구름게임 맵 
        self.player = player     # 플레이어 위치 
        self.computer = computer # 컴퓨터 위치 

    # 구름게임 출력 함수 
    def board_visualization(self):
        for idx in range(self.size):
            for jdx in range(self.size):
                if self.board[idx][jdx] == 0:   # 낭떠러지 
                    print("□ ", end='')
                elif self.board[idx][jdx] == 1: # 구름 
                    print('▨ ', end='')
                elif self.board[idx][jdx] == 2: # 플레이어
                    print('◆ ', end='')
                elif self.board[idx][jdx] == 3: # 컴퓨터 
                    print('● ', end='')
                else:
                    pass
            print()


    # 플레이어 이동 위치 함수 
    def position(self):
        player = list(map(int, input(f'플레이어가 이동할 위치 입력: ').split()))
        # 맵을 벗어나지 않도록 설정 
        if player[0] >= self.size or player[1] >= self.size:
            self.position()
        
        # 이동할 위치가 구름이고 상하좌우 방향이라면 플레이어 위치로 변환 / 기존 위치는 낭떠러지로 변환 
        if self.board[player[0]][player[1]] == 1 and (abs(self.player[0]-player[0]) + abs(self.player[1]-player[1])) == 1:
            self.board[self.player[0]][self.player[1]] = 0
            self.player = player
            self.board[player[0]][player[1]] = 2
        else:
            self.position()


    # 게임이 끝났는지 확인하는 함수 
    def win_condition(self, board):
        # 리턴값 0: 컴퓨터 승리 / 1: 플레이어 승리 / 2: 승패 결정 X 
        dx = [-1, 1, 0, 0]
        dy = [0, 0, -1 ,1]

        computer = None # 컴퓨터의 좌표 저장
        player = None # 플레이어의 좌표 저장
        computer_ls = [] # 컴퓨터의 주변 좌표 저장
        player_ls = [] # 플레이어의 주변 좌표 저장
        computer_state = None # 처음에는 리스트로 사용, 이후에는 컴퓨터의 상태 0 : 진 것, 1 이긴 것
        player_state = None # 처음에는 리스트로 사용, 이후에는 플레이어의 상태 0 : 진 것, 1 이긴 것

        # 반복문을 돌력서 컴퓨터와 플레이어의 위치를 찾는다.
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == 3:
                    computer = (i, j)
                if board[i][j] == 2:
                    player = (i, j)
        
        
        # 컴퓨터의 상태와 플레이어의 상태를 결정한다. 주변의 값이 모두 비어있으면 0으로 유지된다.
        for i, j in zip(dx, dy):
            if computer[0]+i >= self.size or computer[0]+i < 0 or computer[1]+j >= self.size or computer[1]+j < 0: continue
            computer_ls.append(board[computer[0]+i][computer[1]+j])

        # 위와 동일한 과정을 플레이어에 대해서 반복한다.
        for i, j in zip(dx, dy):
            if player[0]+i >= self.size or player[0]+i < 0 or player[1]+j >= self.size or player[1]+j < 0: continue
            player_ls.append(board[player[0]+i][player[1]+j])

        # 0 또는 1로 모두 인코딩을 한 후 any 함수를 통해 한 개라도 1이 있으면 리턴한다.
        computer_state = any(list(map(lambda x: 0 if x == 2 or x == 0 else 1, computer_ls)))
        player_state = any(list(map(lambda x: 0 if x == 3 or x == 0 else 1, player_ls)))

        # 조건에 맞도록 리턴한다.
        if computer_state == 1 and player_state == 1: return 2
        if computer_state == 1 and player_state == 0: return 0
        if computer_state == 0 and player_state == 1: return 1
        
        return 2
    

    # 게임 결과에 따른 minimax 알고리즘 점수를 리턴하는 함수 
    def evaluate(self, board):
        result = self.win_condition(board) # 게임 결과를 받아옴

        if result == 1: return 100 # 사용자가 이긴 경우에는 점수를 100을 준다.
        if result == 0: return -100 # 컴퓨터가 이긴 경우에는 점수를 -100을 준다.
        return 0 # 승패를 결정할 수 없는 경우 0을 준다.

    # 타겟이 이동할 수 있는 위치를 찾는 함수 
    def ismove(self, board, target): # target = 2: 플레이어 | target = 3: 컴퓨터
        # 변화량
        dx = [-1, 1, 0, 0]
        dy = [0, 0, -1 ,1]

        moves = [] # 초기화 
        locate = None

        # 조건에 맞는 타깃의 좌표를 찾는다.
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == target:
                    locate = (i, j)
        
        # 변화를 시켜가면서 주변이 1로 된 좌표를 찾는다. 그리고 좌표의 형태로 append를 하여 리턴
        for i, j in zip(dx, dy):
            # 범위를 벗어나는지 검증
            if locate[0]+i >= self.size or locate[0]+i < 0 or locate[1]+j >= self.size or locate[1]+j < 0: continue
            if board[locate[0]+i][locate[1]+j] == 1:
                moves.append((locate[0]+i, locate[1]+j))

        return moves # 해당 값은 튜플 형태로 [(), (), ...] 로 구성되어 있음


    # MiniMax 알고리즘 수행 함수 
    def minimax(self, board, locate, turn): # turn 2 : 플레이어 / turn 3 : 컴퓨터 
        # 리턴값 : best_score, best_move (플레이어가 갈 수 있는 최적의 위치와 이에 따른 점수 리턴)
        
        # 현재 플레이어와 컴퓨터의 위치
        player = None # (i, j) form
        computer = None # (i, j) form

        # 반복문을 돌력서 컴퓨터와 플레이어의 위치를 찾는다.
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == 3:
                    computer = (i, j)
                if board[i][j] == 2:
                    player = (i, j)

        # 게임이 끝나는 경우 <- 확실히 승패가 결정된 상태로 끝나는 경우
        if self.evaluate(board)  == 100:
            return [self.evaluate(board), player]
        
        if self.evaluate(board) == -100:
            return [self.evaluate(board), computer]

        # 승패가 결정되지않고 무승부로 게임이 끝나는 경우
        if len(self.ismove(board, 2)) == 0 and len(self.ismove(board, 3)) == 0:
            if turn == 2: return [0, player]
            else: return [0, computer]
        
        # 플레이어인 경우 turn = True
        if turn == 2:
            minimax_val_loc = [-999, locate]  # player은 휴리스틱 값을 최대로 하여야하기 때문에 초기 설정을 최소로 지정한다.

            # 루프를 돌아 가능한 경우를 탐색할 수 있도록 한다.
            for (i, j) in self.ismove(board, 2):
                board[i][j] = 2 # 인간이 위치를 이동한다.
                board[player[0]][player[1]] = 0 # 보드판에서 원래 플레이어의 위치를 바군다.
                tmp_tuple = (i, j)
                value, next_locate = self.minimax(board, tmp_tuple, 3) # 재귀를 호출하여 턴을 컴퓨터로 넘긴다.
                
                # 보드판을 원래대로 돌려놓는다.
                board[player[0]][player[1]] = 2
                board[i][j] = 1
                
                # 값을 업데이트 해야하는 경우가 발생하면 한다.
                if value > minimax_val_loc[0]:
                    minimax_val_loc = [value, tmp_tuple]
        
            return minimax_val_loc # next_locate
        
        if turn == 3:
            minimax_val_loc = [999, locate]  # computer 휴리스틱 값을 최소로 하여야하기 때문에 초기 설정을 최대로 지정한다.

            # 루프를 돌아 가능한 경우를 탐색할 수 있도록 한다.
            for (i, j) in self.ismove(board, 3):
                board[i][j] = 3 # 컴퓨터가 위치를 이동한다.
                board[computer[0]][computer[1]] = 0 # 보드판에서 원래 위치를 바군다.
                tmp_tuple = (i, j)
                value, next_locate = self.minimax(board, tmp_tuple ,2) # 재귀를 호출하여 턴을 인간으로 넘긴다.
                
                # 보드판을 원래대로 돌려놓는다.
                board[computer[0]][computer[1]] = 3
                board[i][j] = 1
                
                # 값을 업데이트 해야하는 경우가 발생하면 한다.
                if value < minimax_val_loc[0]:
                    minimax_val_loc = [value, tmp_tuple]

            return minimax_val_loc



    # 컴퓨터가 이동할 최적의 위치를 받는 함수 
    def best_pos(self):
    	# 컴퓨터가 이동할 최적의 위치 
        position = self.minimax(self.board, (0,0), 3)[1]

        # 위치를 받아와서 구름게임 맵과 컴퓨터 위치 변경 
        self.board[self.computer[0]][self.computer[1]] = 0
        self.computer = [position[0], position[1]]
        self.board[position[0]][position[1]] = 3


# 구름게임 맵 입력 
board = []
size = int(input('구름게임 맵 크기(정방행렬): '))
print('낭떠러지 : 0     구름 : 1')
for s in range(size):
    tmp = list(map(int, input(f'{s}행 입력: ').split()))
    board.append(tmp)

while True: # 플레이어 시작 위치 입력 
    player = list(map(int, input(f'플레이어 시작 위치 입력: ').split()))
    if player[0] >= 0 and player[0] < size and player[1] >= 0 and player[1] < size:
        if board[player[0]][player[1]] == 1:
            break
board[player[0]][player[1]] = 2

while True: # 컴퓨터 시작 위치 입력 
    computer = list(map(int, input(f'컴퓨터 시작 위치 입력: ').split()))
    if computer[0] >= 0 and computer[0] < size and computer[1] >= 0 and computer[1] < size:
        if board[computer[0]][computer[1]] == 1:
            break
board[computer[0]][computer[1]] = 3

# 구름게임 세팅 
game = Game(size, board, player, computer)

print("============= Initial State =============")
game.board_visualization()

# 구름게임 시작 
while True:
    print("=============== Computer ================")
    result = game.win_condition(game.board) # 컴퓨터가 진행하기 전 게임 상황 확인 
    if result == 0: # 컴퓨터가 이긴 경우 게임 종료 
        game.board_visualization()
        print('컴퓨터가 승리하였습니다.')
        break
    elif result == 1: # 플레이어가 이긴 경우 게임 종료 
        game.board_visualization()
        print('플레이어가 승리하였습니다.')
        break
    game.best_pos()
    game.board_visualization()

    print("=============== Your turn ===============")
    result = game.win_condition(game.board) # 플레이어가 진행하기 전 게임 상황 확인 
    if result == 0:
        game.board_visualization()
        print('컴퓨터가 승리하였습니다.')
        break
    elif result == 1:
        game.board_visualization()
        print('플레이어가 승리하였습니다.')
        break
    game.position()
    game.board_visualization()
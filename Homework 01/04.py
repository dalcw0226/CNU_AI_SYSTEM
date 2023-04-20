# 04.py 틱택토

import random

def board_visualization(board):
    ### 틱택토 게임 출력 함수  
    for i_idx in range(3):
        for j_idx in range(3):
            
            if board[i_idx][j_idx] == 0: # not occupied
                print("▨", end='')
            elif board[i_idx][j_idx] == 1: # user
                print("○", end='')
            elif board[i_idx][j_idx] == 2: # computer
                print("×", end='')
            else:
                pass
        print()        
    
def board_turn(board, row, col, val):
    
    if 0 <= row < 3 and 0 <= col < 3:
        if board[row][col] == 0:
            board[row][col] = val
            return 1
        else:
            print("그곳에 돌을 둘 수 없습니다. 다시 입력하세요.")
            return 0
    else:
        print("그곳에 돌을 둘 수 없습니다. 다시 입력하세요.")
        return 0
        
    
def win_condition(v1,v2,v3): # 사용자 승리 return 100, 컴퓨터 승리 return -100, 무승부 return 0
    if v1 == v2 == v3:  
        if v1 == 1: return 100
        elif v2 == 2: return -100
        else: return 0
    else:
        return 0
        
    
def game_result(board): # 사용자 승리 return 100, 컴퓨터 승리 return -100, 무승부 return 0
    result = 0
    # check rows
    for i_idx in range(3):
        result = win_condition(board[i_idx][0],board[i_idx][1],board[i_idx][2])      
        if result != 0: return result
      
    # check cols
    for i_idx in range(3):
        result = win_condition(board[0][i_idx],board[1][i_idx],board[2][i_idx])
        if result != 0: return result
        
    #check diagonals
    result = win_condition(board[0][0],board[1][1],board[2][2])      
    if result != 0: return result
    result = win_condition(board[0][2],board[1][1],board[2][0])      
    if result != 0: return result
        
    return result

def position():
    row = int(input("돌을 놓을 행을 입력하세요 (0~2):"))
    col = int(input("돌을 놓을 열을 입력하세요 (0~2):"))
    
    return row, col


def minimax(board, turn):

    # 승패가 결정된 경우
    if game_result(board) in [-100, 100]:
        return game_result(board)
    
    # 승패가 결정되지 않고 무승부로 게임이 끝난 경우
    # 모든 판이 채워진 경우를 의미하는 조건식
    if not False in list(map(lambda x : not (0 in x), board)):
        return 0
    
    # 플레이어의 경우 turn = True
    # 플레이어는 휴리스틱 값이 최대가 되는 경우를 지향해야한다.
    if turn:
        # player는 휴리스틱 값을 최대로 만들어야하기 때문에 초기 설정을 최소로 지정한다.
        minimax_val = -999  

        # 루프를 두어 모든 판을 탐색할 수 있도록 코드 작성
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:  # 판에 빈 공간 탐색
                    # board를 손댔으면 이후에는 다시 돌려놔야한다.
                    board[i][j] = 1  # 인간이 수를 둔 것으로 가정한 후 다음 탐색 진행
                    # 재귀적 탐색 이용, 
                    # 여기서는 turn 매개변수를 False로 두어 turn이 computer로 넘어가도록 함
                    value = minimax(board, False) 

                    # 다시 돌려놔야한다.
                    board[i][j] = 0
                    # 플레이어가 둔 수는 이길 확률이 가장 높도록 함
                    minimax_val = max(minimax_val, value) 
        return minimax_val # 리프에 도달했을때를 대비하여 리턴
    
    else:
        # computer은 휴리스틱 값을 최소로 만들어야하기 때문에 초기 설정을 최대로 둔다.
        minimax_val = 999 
        
        # 루프를 두어 모든 판을 탐색할 수 있도록 코드 작성
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:  # 판에 빈 공간 탐색
                    board[i][j] = 2  # computer가 수를 둔 것으로 가정한 후 다음 탐색 진행
                    # 재귀적 탐색 이용, 
                    # 여기서는 turn 매개변수를 True로 두어 turn이 player로 넘어가도록 함
                    value = minimax(board, True)

                    # 이것도 마찬가지로 손댔으면 다시 돌려놔야 한다.
                    board[i][j] = 0
                    # 컴퓨터가 둔 수는 플레이어가 이길 확률이 가장 낮도록 함.
                    minimax_val = min(minimax_val, value) 
        return minimax_val
    
def computer(board):

    # 가장 최적의 값을 찾기 위한 도움 리스트 2개를 둔다.
    score_ls = []  # 해당 값에 수를 두었을 때 얻을 수 있는 휴리스틱 값
    point_ls = []  # 해당 점의 좌표

    # 플레이어가 둔 보드판의 상태를 기반으로 탐색을 진행
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:  # 빈 공간 발견
                # 일단 빈 공간에 대해서는 point_ls 에 좌표 추가
                point_ls.append((i, j))
                # 빈 공간에 일단 수를 둔 것으로 가정하고 탐색을 진행
                board[i][j] = 2
                # score_ls에 minimax를 통해 얻은 최적의 휴리스틱 값을 추가한다. 
                # computer가 수를 두고 player가 수를 둘 차례이므로 turn매개변수는 True로 둔다.
                score_ls.append(minimax(board, True))
                # 원상 복귀
                board[i][j] = 0
    
    # 이중에서 computer가 이기도록 하기위해서는 휴리스틱 값이 가장 낮은 값을 찾을 수 있도록 해야한다.
    # 가장 낮은 값과 좌표를 찾아서 선언한 리스트에 추가하는 알고리즘
    min_score_point_idx = None
    min_score = 999
    for idx, score in enumerate(score_ls):
        if min_score > score : 
            min_score = score
            min_score_point_idx = idx
    
    # 만약 판이 모두 채워져서 두 리스트의 길이가 0인 경우에는 무승부를 판단할 수 있는 조건인 (-1,-1)을 리턴한다.
    if len(point_ls) == 0:
        result = (-1,-1)
    else:  # 그 외에는 휴리스틱값을 최소로 만드는 좌표를 리턴한다.
        result = point_ls[min_score_point_idx]
    
    return result


### 게임 시작
print("게임 시작")
board = [[0,0,0],[0,0,0],[0,0,0]]
board_visualization(board)

for i_idx in range(5): # 플레이어는 최대 5번 돌을 둔다
    ###### User ######
    print("============= Your turn =============")
    row, col = position()
    while board_turn(board, row, col,1) == 0:
        row, col = position()
        
    board_visualization(board)
    if game_result(board) == 100:
        print("사용자가 승리하였습니다.")
        break
    elif game_result(board) == -100:
        print("컴퓨터가 승리하였습니다.")
        break      
    ########################
    
    
    
    ###### Computer ######
    print("============= Computer =============")
    row, col = computer(board) # computer 함수를 다시 코딩 하시오
    board_turn(board, row, col, 2)
    board_visualization(board)   
    if game_result(board) == 100:
        print("사용자가 승리하였습니다.")
        break
    elif game_result(board) == -100:
        print("컴퓨터가 승리하였습니다.")
        break
    ########################

    
if game_result(board) == 0:
    print("비겼습니다.")
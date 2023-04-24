#######################################################
# Gomoku Platform (single game)
# Version 0.5
# 
# Swansea University
# Mar 2023
#

import sys, time, signal
import numpy as np
import os

import concurrent.futures

from time import time
from random import randint

from misc import winningTest, legalMove

BOARD_SIZE = 11   # size of the board is 11-by-11
X_IN_A_LINE = 5   # play the standard game with 5 stones in a line
TIME_OUT = 20      # player must return a move within 5 seconds

# an empty class to host the time-out exception
class TimeOutException(Exception):
    pass

# handler for time out
def handler(signum, frame):
    print("Player timeout - signal -")
    raise TimeOutException()


# turn taking function
def turn(board, player, turn_id):

    # make a copy of the board, which is passed to the agent
    tempBoard = np.array(board)

    # TIME_OUT seconds Timer
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            playerThread = executor.submit(player.move, tempBoard)
            moveLoc = playerThread.result(TIME_OUT+1)
    except TimeOutException:        
        # Caught the signal timeout exception
        print("before pass")
        pass
    except concurrent.futures.TimeoutError:
        print("Player" + str(turn_id) + " time out.")
        return turn_id*1, board        

    # test if the move is legal - on the original board
    if legalMove(board, moveLoc):
        board[moveLoc] = player.ID
    else:
        print("Player " + str(player.ID) + " illegal move at " + str(moveLoc))
        return turn_id*-1, board

    # test if any player wins the game
    if winningTest(player.ID, board, X_IN_A_LINE):
        return turn_id, board

    # move to the next turn
    return 0, board

def main():
    if len(sys.argv) < 3:
        print("Error. To use: python gomoku.py PLAYER1 PLAYER2");
        print("Example: python gomoku.py GomokuAgentRand GomokuAgentRand");
        return -1;

    # two directory names
    p1Dir, p2Dir = sys.argv[1], sys.argv[2]
    
    # creating the two players
    P1 = getattr(__import__(p1Dir, fromlist=["player"]), "player")
    P2 = getattr(__import__(p2Dir, fromlist=["player"]), "player")

    player1 = P1.Player(1, BOARD_SIZE, X_IN_A_LINE)
    player2 = P2.Player(-1, BOARD_SIZE, X_IN_A_LINE)

    # initialize the board
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)

    # connect the alarm signal with the handler
    # signal.signal(signal.SIGALRM, handler)

    # play the game
    winner = 0
    while True:
        end = False
        for player, turn_id in [(player1, 1), (player2, -1)]:
            id, board = turn(board, player, turn_id)
            print(board)
            if not 0 in board:
                print("Draw.")
                end = True
                break
            if id != 0:
                print("Winner: " + str(id))
                end = True
                break
        if end:
            break        

if __name__ == '__main__':
    sys.exit(main());

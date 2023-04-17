from gomokuAgent import GomokuAgent
import numpy as np
from misc import legalMove

class Player(GomokuAgent):
   class GomokuAgent:
    def __init__(self, ID, BOARD_SIZE, X_IN_A_LINE):
        self.ID = ID
        self.BOARD_SIZE = BOARD_SIZE
        self.X_IN_A_LINE = X_IN_A_LINE
        self.playerSymbol = ID
        self.opponentSymbol = -ID
    
    def move(self, board):
        while True:
            moveLoc = self.minimax(board, True)
            if legalMove(board, moveLoc):
                return moveLoc
    
    def minimax(self, board, maximizingPlayer):
        if self.gameOver(board):
            return self.evaluate(board), None
        
        if maximizingPlayer:
            value = float('-inf')
            bestMove = None
            for moveLoc in self.generateMoves(board):
                newBoard = board.copy()
                newBoard[moveLoc] = self.playerSymbol
                newValue, _ = self.minimax(newBoard, False)
                if newValue > value:
                    value = newValue
                    bestMove = moveLoc
            return value, bestMove
        else:
            value = float('inf')
            bestMove = None
            for moveLoc in self.generateMoves(board):
                newBoard = board.copy()
                newBoard[moveLoc] = self.opponentSymbol
                newValue, _ = self.minimax(newBoard, True)
                if newValue < value:
                    value = newValue
                    bestMove = moveLoc
            return value, bestMove
    
    def gameOver(self, board):
        return self.getWinner(board) != 0 or np.count_nonzero(board == 0) == 0
    
    def getWinner(self, board):
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                if board[r][c] == 0:
                    continue
                player = board[r][c]
                if c + self.X_IN_A_LINE <= self.BOARD_SIZE and np.all(board[r,c:c+self.X_IN_A_LINE] == player):
                    return player
                if r + self.X_IN_A_LINE <= self.BOARD_SIZE and np.all(board[r:r+self.X_IN_A_LINE,c] == player):
                    return player
                if r + self.X_IN_A_LINE <= self.BOARD_SIZE and c + self.X_IN_A_LINE <= self.BOARD_SIZE and np.all(board[r:r+self.X_IN_A_LINE,c:c+self.X_IN_A_LINE].diagonal() == player):
                    return player
                if r + self.X_IN_A_LINE <= self.BOARD_SIZE and c - self.X_IN_A_LINE >= -1 and np.all(np.fliplr(board[r:r+self.X_IN_A_LINE,c-self.X_IN_A_LINE+1:c+1]).diagonal() == player):
                    return player
        return 0
    
    def evaluate(self, board):
        winner = self.getWinner(board)
        if winner == self.playerSymbol:
            return float('inf')
        elif winner == self.opponentSymbol:
            return float('-inf')
        else:
            score = self.evaluateLine(board, self.playerSymbol) - self.evaluateLine(board, self.opponentSymbol)
            return score
    
    def evaluateLine(self, board, player):
        score = 0
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                if board[r][c] != player:
                    continue
                if c + self.X_IN_A_LINE <= self.BOARD_SIZE:
                    line = board[r,c:c+self.X_IN_A_LINE]
                    if np.count_nonzero(line == player) == self.X_IN_A_LINE:
                        score += 1
                    elif np.count_nonzero(line == player) == self.X_IN_A_LINE - 2 and np.count_nonzero(line == 0) == 2:
                        score += self.TWO_IN_A_LINE_SCORE
                                # Check for two in a line with one empty space
                    elif np.count_nonzero(line == player) == self.X_IN_A_LINE - 2 and np.count_nonzero(line == 0) == 1:
                        score += self.ONE_IN_A_LINE_SCORE

                return score

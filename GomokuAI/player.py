import numpy as np
from gomokuAgent import GomokuAgent
from misc import legalMove, winningTest

class Player(GomokuAgent):
    def __init__(self, ID, BOARD_SIZE, X_IN_A_LINE):
        super().__init__(ID, BOARD_SIZE, X_IN_A_LINE)
        self.MAX_DEPTH = 0

    def move(self, board):
        best_move = None
        best_score = -np.inf
        for move in self.generate_moves(board):
            new_board = np.copy(board)
            new_board[move] = self.ID
            score = self.minimax(new_board, self.MAX_DEPTH, -np.inf, np.inf, False)
            if score > best_score:
                best_move = move
                best_score = score
        return best_move

    def generate_moves(self, board):
        moves = []
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                if legalMove(board, (r, c)):
                    moves.append((r, c))
                if legalMove(board, (r - 1, c - 1)):
                    moves.append((r - 1, c - 1))
                if legalMove(board, (r - 1, c + 1)):
                    moves.append((r - 1, c + 1))
                if legalMove(board, (r + 1, c - 1)):
                    moves.append((r + 1, c - 1))
                if legalMove(board, (r + 1, c + 1)):
                    moves.append((r + 1, c + 1))
        return moves

    def minimax(self, board, depth, alpha, beta, maximizing_player):
        if winningTest(self.ID, board, self.X_IN_A_LINE):
            return 1000000 - depth
        elif winningTest(-self.ID, board, self.X_IN_A_LINE):
            return -1000000 + depth
        elif depth == 0:
            return self.heuristic_score(board)

        if maximizing_player:
            max_score = -np.inf
            for move in self.generate_moves(board):
                new_board = np.copy(board)
                new_board[move] = self.ID
                score = self.minimax(new_board, depth - 1, alpha, beta, False)
                max_score = max(max_score, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break  # beta cutoff
            return max_score
        else:
            min_score = np.inf
            for move in self.generate_moves(board):
                new_board = np.copy(board)
                new_board[move] = -self.ID
                score = self.minimax(new_board, depth - 1, alpha, beta, True)
                min_score = min(min_score, score)
                beta = min(beta, score)
                if beta <= alpha:
                    break  # alpha cutoff
            return min_score


    def heuristic_score(self, board):
        score = 0
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                if board[r, c] == self.ID:
                    score += self.get_score_for_position(board, r, c)
                    score += self.get_score_for_potential_moves(board, r, c)
                    if (r,c) == (3,3) or (r,c) == (4,4) or (r,c) == (3,4) or (r,c) == (4,3):
                        score += 50  # Bonus for having a piece in the center of the board
                    if self.ID == 1:
                        if board[r, c] == 1:
                            score += 5  # Bonus for having more pieces on the board
                        else:
                            score -= 5
                    else:
                        if board[r, c] == -1:
                            score += 5
                        else:
                            score -= 5
                elif board[r, c] == -self.ID:
                    score -= self.get_score_for_position(board, r, c)
                    score -= self.get_score_for_potential_moves(board, r, c)
                    if (r,c) == (3,3) or (r,c) == (4,4) or (r,c) == (3,4) or (r,c) == (4,3):
                        score -= 50  # Penalty for opponent having a piece in the center of the board
                    if self.ID == 1:
                        if board[r, c] == 1:
                            score -= 5  # Penalty for opponent having more pieces on the board
                        else:
                            score += 5
                    else:
                        if board[r, c] == -1:
                            score -= 5
                        else:
                            score += 5
        return score

    def get_score_for_position(self, board, row, col):
        score = 0
        # check if opponent is close to getting 5 in a row
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                # check horizontally
                opponent_row = row - dr
                opponent_col = col - dc
                if opponent_row >= 0 and opponent_row < self.BOARD_SIZE and opponent_col >= 0 and opponent_col < self.BOARD_SIZE and board[opponent_row][opponent_col] == -self.ID:
                    opponent_piece_count = 0
                    for i in range(self.X_IN_A_LINE):
                        r = opponent_row + i * dr
                        c = opponent_col + i * dc

                        if r < 0 or r >= self.BOARD_SIZE or c < 0 or c >= self.BOARD_SIZE:
                            break
                        if board[r][c] == -self.ID:
                            opponent_piece_count += 1
                        elif board[r][c] == 0:
                            break
                        else:
                            opponent_piece_count = 0
                            break
                    if opponent_piece_count == self.X_IN_A_LINE - 1:
                        score -= 1000000
                # check vertically
                opponent_row = row - dr * (self.X_IN_A_LINE - 1)
                opponent_col = col - dc * (self.X_IN_A_LINE - 1)
                if opponent_row >= 0 and opponent_row < self.BOARD_SIZE and opponent_col >= 0 and opponent_col < self.BOARD_SIZE and board[opponent_row][opponent_col] == -self.ID:
                    opponent_piece_count = 0
                    for i in range(self.X_IN_A_LINE):
                        r = opponent_row + i * dr
                        c = opponent_col + i * dc

                        if r < 0 or r >= self.BOARD_SIZE or c < 0 or c >= self.BOARD_SIZE:
                            break
                        if board[r][c] == -self.ID:
                            opponent_piece_count += 1
                        elif board[r][c] == 0:
                            break
                        else:
                            opponent_piece_count = 0
                            break
                    if opponent_piece_count == self.X_IN_A_LINE - 1:
                        score -= 1000000
                # check diagonally
                opponent_row = row - dr * (self.X_IN_A_LINE - 1)
                opponent_col = col - dc * (self.X_IN_A_LINE - 1)
                if opponent_row >= 0 and opponent_row < self.BOARD_SIZE and opponent_col >= 0 and opponent_col < self.BOARD_SIZE and board[opponent_row][opponent_col] == -self.ID:
                    opponent_piece_count = 0
                    for i in range(self.X_IN_A_LINE):
                        r = opponent_row + i * dr
                        c = opponent_col + i * dc

                        if r < 0 or r >= self.BOARD_SIZE or c < 0 or c >= self.BOARD_SIZE:
                            break
                        if board[r][c] == -self.ID:
                            opponent_piece_count += 1
                        elif board[r][c] == 0:
                            break
                        else:
                            opponent_piece_count = 0
                            break
                    if opponent_piece_count == self.X_IN_A_LINE - 1:
                        score -= 1000000

        return score
            
    def get_score_for_direction(self, board, row, col, dr, dc):
        score = 0
        player_piece_count = 0
        open_end = False

        for i in range(self.X_IN_A_LINE):
            r = row + i * dr
            c = col + i * dc

            if r < 0 or r >= self.BOARD_SIZE or c < 0 or c >= self.BOARD_SIZE:
                break
            if board[r][c] == self.ID:
                player_piece_count += 1
            elif board[r][c] == 0:
                if i == 0 or i == self.X_IN_A_LINE - 1:
                    open_end = True
                else:
                    open_end = False
                break
            else:
                open_end = False
                break
            
        # Count the number of opponent's stones in the given direction
        countOpponent = 0
        for i in range(self.X_IN_A_LINE):
            r = row + i * dr
            c = col + i * dc

            if r < 0 or r >= self.BOARD_SIZE or c < 0 or c >= self.BOARD_SIZE:
                break
            if board[r][c] == self.ID:
                countOpponent += 1
            elif board[r][c] == 0:
                if i == 0 or i == self.X_IN_A_LINE - 1:
                    open_end = True
                else:
                    open_end = False
                break
            else:
                open_end = False
                break

        if player_piece_count > 0 and player_piece_count < self.X_IN_A_LINE and open_end:
            score += player_piece_count
        elif player_piece_count == self.X_IN_A_LINE:
            score += self.X_IN_A_LINE * 2

        return score
    
    
    def get_score_for_potential_moves(self, board, r, c):
        score = 0
        
        # Check horizontally to the right
        if c < self.BOARD_SIZE - 1 and board[r, c+1] == 0:
            board[r, c+1] = self.ID
            score += self.get_score_for_position(board, r, c+1)
            board[r, c+1] = 0
            
        # Check horizontally to the left
        if c > 0 and board[r, c-1] == 0:
            board[r, c-1] = self.ID
            score += self.get_score_for_position(board, r, c-1)
            board[r, c-1] = 0
            
        # Check vertically downwards
        if r < self.BOARD_SIZE - 1 and board[r+1, c] == 0:
            board[r+1, c] = self.ID
            score += self.get_score_for_position(board, r+1, c)
            board[r+1, c] = 0
            
        # Check diagonally downwards to the right
        if r < self.BOARD_SIZE - 1 and c < self.BOARD_SIZE - 1 and board[r+1, c+1] == 0:
            board[r+1, c+1] = self.ID
            score += self.get_score_for_position(board, r+1, c+1)
            board[r+1, c+1] = 0
            
        # Check diagonally downwards to the left
        if r < self.BOARD_SIZE - 1 and c > 0 and board[r+1, c-1] == 0:
            board[r+1, c-1] = self.ID
            score += self.get_score_for_position(board, r+1, c-1)
            board[r+1, c-1] = 0
        
        return score

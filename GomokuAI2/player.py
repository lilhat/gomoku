import numpy as np
from gomokuAgent import GomokuAgent
from misc import legalMove, winningTest
from multiprocessing import Pool

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
            score = self.minimax_parallel(new_board, self.MAX_DEPTH, -np.inf, np.inf, False)
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

    def minimax_parallel(state, depth, alpha, beta, is_maximizing_player):
        if depth == 0 or state.is_game_over():
            return state.evaluate(), None
        sub_trees = state.get_sub_trees()
        pool = Pool(processes=len(sub_trees))
        results = []
        for sub_tree in sub_trees:
            result = pool.apply_async(
                minimax_parallel, (sub_tree, depth - 1, alpha, beta, not is_maximizing_player)
            )
            results.append(result)
        sub_tree_results = [result.get() for result in results]
        if is_maximizing_player:
            best_value = float("-inf")
            best_move = None
            for value, move in sub_tree_results:
                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, best_value)
                if alpha >= beta:
                    break
        else:
            best_value = float("inf")
            best_move = None
            for value, move in sub_tree_results:
                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, best_value)
                if alpha >= beta:
                    break
        return best_value, best_move


    def heuristic_score(self, board):
        score = 0
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                if board[r, c] == self.ID:
                    score += self.get_score_for_position(board, r, c)
                elif board[r, c] == -self.ID:
                    score -= self.get_score_for_position(board, r, c)
        return score

    def get_score_for_position(self, board, row, col):
        score = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                score += self.get_score_for_direction(board, row, col, dr, dc)
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
    
# Import required libraries
import numpy as np
from gomokuAgent import GomokuAgent
from misc import legalMove, winningTest

# Player class definition, inherits from GomokuAgent
class Player(GomokuAgent):
    # Class constructor
    def __init__(self, ID, BOARD_SIZE, X_IN_A_LINE):
        # Calls the constructor from GomokuAgent
        super().__init__(ID, BOARD_SIZE, X_IN_A_LINE)
        # Sets the max depth of the minimax algorithm to 0
        self.MAX_DEPTH = 0

    # Overwriting the move function from GomokuAgent
    def move(self, board):
        # Initialize variables
        best_move = None
        best_score = -np.inf
        # Loop through all possible moves
        for move in self.generate_moves(board):
            # Create copy of the board, make current move on copy
            new_board = np.copy(board)
            new_board[move] = self.ID
            # Calculate the score for current move using minimax algorithm
            score = self.minimax(new_board, self.MAX_DEPTH, -np.inf, np.inf, False)
            # If score is greater than the previous best score then update the best move and best score
            if score > best_score:
                best_move = move
                best_score = score
        # Return best move
        print(best_score)
        print(best_move)
        return best_move

    '''
    Generates a list of legal moves for a given board
    Parameters:
        - board: the current state of the game
    returns:
        - moves: a list of legal moves for the given game board
    '''
    def generate_moves(self, board):
        moves = []
        # Iterate through every position on the board
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                # If position is a legal move, add it to list of moves
                if legalMove(board, (r, c)):
                    moves.append((r, c))
                # Check for diagonal moves
                if legalMove(board, (r - 1, c - 1)):
                    # Check if diagonal move is valid then add valid diagonal moves to the list of moves
                    # Checks if there is enough space on either side to form 5 in a row
                    # Also checks if the next space on either end is occupied by the opponent for possibility to block
                    if (board[r - 1, c - 1] == -self.ID and 
                        np.sum(board[r - self.X_IN_A_LINE + 1:r, c - self.X_IN_A_LINE + 1:c]) == -(self.X_IN_A_LINE - 1) and 
                        legalMove(board, (r - self.X_IN_A_LINE, c - self.X_IN_A_LINE))):
                        moves.append((r - self.X_IN_A_LINE, c - self.X_IN_A_LINE))
                    if (c + self.X_IN_A_LINE <= self.BOARD_SIZE - 1 and
                        board[r - 1, c + 1] == -self.ID and 
                        np.sum(board[r - self.X_IN_A_LINE + 1:r, c + 1:c + self.X_IN_A_LINE]) == -(self.X_IN_A_LINE - 1) and 
                        legalMove(board, (r - self.X_IN_A_LINE, c + self.X_IN_A_LINE))):
                        moves.append((r - self.X_IN_A_LINE, c + self.X_IN_A_LINE))
                    if (r + self.X_IN_A_LINE <= self.BOARD_SIZE - 1 and
                        board[r + 1, c - 1] == -self.ID and 
                        np.sum(board[r + 1:r + self.X_IN_A_LINE, c - self.X_IN_A_LINE + 1:c]) == -(self.X_IN_A_LINE - 1) and 
                        legalMove(board, (r + self.X_IN_A_LINE, c - self.X_IN_A_LINE))):
                        moves.append((r + self.X_IN_A_LINE, c - self.X_IN_A_LINE))
                    if (r + self.X_IN_A_LINE <= self.BOARD_SIZE - 1 and c + self.X_IN_A_LINE <= self.BOARD_SIZE - 1 and
                        board[r + 1, c + 1] == -self.ID and 
                        np.sum(board[r + 1:r + self.X_IN_A_LINE, c + 1:c + self.X_IN_A_LINE]) == -(self.X_IN_A_LINE - 1) and 
                        legalMove(board, (r + self.X_IN_A_LINE, c + self.X_IN_A_LINE))):
                        moves.append((r + self.X_IN_A_LINE, c + self.X_IN_A_LINE))
                # Check for vertical moves
                if legalMove(board, (r - 1, c)):
                    # Check if vertical move is valid then add valid vertical moves to the list of moves
                    if (board[r - 1, c] == -self.ID and 
                        np.sum(board[r - self.X_IN_A_LINE + 1:r, c]) == -(self.X_IN_A_LINE - 1) and 
                        legalMove(board, (r - self.X_IN_A_LINE, c))):
                        moves.append((r - self.X_IN_A_LINE, c))
                    if (r + self.X_IN_A_LINE <= self.BOARD_SIZE - 1 and
                        board[r + 1, c] == -self.ID and 
                        np.sum(board[r + 1:r + self.X_IN_A_LINE, c]) == -(self.X_IN_A_LINE - 1) and 
                        legalMove(board, (r + self.X_IN_A_LINE, c))):
                                moves.append((r + self.X_IN_A_LINE, c))
                # Check for horizontal moves
                if legalMove(board, (r, c - 1)):
                    # Check if the horizontal move is valid then add valid horizontal moves to the list of moves
                    if (board[r, c - 1] == -self.ID and 
                        np.sum(board[r, c - self.X_IN_A_LINE + 1:c]) == -(self.X_IN_A_LINE - 1) and 
                        legalMove(board, (r, c - self.X_IN_A_LINE))):
                        moves.append((r, c - self.X_IN_A_LINE))
                    if (c + self.X_IN_A_LINE <= self.BOARD_SIZE - 1 and
                        board[r, c + 1] == -self.ID and 
                        np.sum(board[r, c + 1:c + self.X_IN_A_LINE]) == -(self.X_IN_A_LINE - 1) and 
                        legalMove(board, (r, c + self.X_IN_A_LINE))):
                        moves.append((r, c + self.X_IN_A_LINE))

        return moves


    '''
    This function is an implementation of the minimax algorith with alpha-beta pruning
    Parameters:
        - board: current state of the game
        - depth: maximum depth for the algorithm to go through
        - alpha: the best value the maximising player can guarantee
        - beta: the best value the minimising player can guarantee
        - maximizing_player: the current player that is maximising
    Returns:
        - score: the best score found by the algorithm
    '''
    def minimax(self, board, depth, alpha, beta, maximizing_player):
        # Check if current player has won the game
        if winningTest(self.ID, board, self.X_IN_A_LINE):
            return 1000000 - depth
        # Check if other player has won the game
        elif winningTest(-self.ID, board, self.X_IN_A_LINE):
            return -1000000 + depth
        # Check if maximum depth has been reached
        elif depth == 0:
            return self.heuristic_score(board)
        # Find best move if the current player is maximising
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
        # Find worst move for the other player if the current player is minimising
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
        
    '''
    This function calculates the heuristic score of the current board for the current player.
    Parameters:
        - board: The current state of the game board
    Returns:
        - score: The heuristic score of the board for the current player
    '''
    
    def heuristic_score(self, board):
        score = 0
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                if board[r, c] == self.ID:
                    # Add the score for the current position and possible moves from this position
                    score += self.get_score_for_position(board, r, c)
                    score += self.get_score_for_potential_moves(board, r, c)
                elif board[r, c] == -self.ID:
                    # Subtract the score for the current position and possible moves from this position
                    score -= self.get_score_for_position(board, r, c)
                    score -= self.get_score_for_potential_moves(board, r, c)                      
        return score

    """
        Returns the score for a given position on the board.
        Parameters:
            - board: The current state of the game board
            - row: The row of the position to check
            - col: The column of the position to check
        Returns:
            - Score: The score for the given position on the board.
        """
    def get_score_for_position(self, board, row, col):
        score = 0
        # check if close to getting 5 in a row
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                # check horizontally
                check_row = row - dr
                check_col = col - dc
                if check_row >= 0 and check_row < self.BOARD_SIZE and check_col >= 0 and check_col < self.BOARD_SIZE and board[check_row][check_col] == self.ID:
                    piece_count = 0
                    # Calculate the row/column index of the next piece
                    for i in range(self.X_IN_A_LINE):
                        r = check_row + i * dr
                        c = check_col + i * dc
                    
                        if r < 0 or r >= self.BOARD_SIZE or c < 0 or c >= self.BOARD_SIZE:
                            break
                        if board[r][c] == self.ID:
                            piece_count += 1
                        elif board[r][c] == 0:
                            break
                        else:
                            piece_count = 0
                            break

                    if piece_count == self.X_IN_A_LINE - 2:
                        score += 1000
                    if piece_count >= self.X_IN_A_LINE - 1:
                        score += 10000
                        
                # check vertically
                check_row = row - dr * (self.X_IN_A_LINE - 1)
                check_col = col - dc * (self.X_IN_A_LINE - 1)
                if check_row >= 0 and check_row < self.BOARD_SIZE and check_col >= 0 and check_col < self.BOARD_SIZE and board[check_row][check_col] == self.ID:
                    piece_count = 0
                    for i in range(self.X_IN_A_LINE):
                        r = check_row + i * dr
                        c = check_col + i * dc

                        if r < 0 or r >= self.BOARD_SIZE or c < 0 or c >= self.BOARD_SIZE:
                            break
                        if board[r][c] == self.ID:
                            piece_count += 1
                        elif board[r][c] == 0:
                            break
                        else:
                            piece_count = 0
                            break

                    if piece_count == self.X_IN_A_LINE - 2:
                        score += 1000
                    if piece_count >= self.X_IN_A_LINE - 1:
                        score += 10000
                    
        directions = [(0, 1), (1, 0), (1, 1), (-1, 1)] # diagonal directions
        for dr, dc in directions:
            piece_count = 0
            for i in range(-self.X_IN_A_LINE + 1, self.X_IN_A_LINE):
                r = row + i * dr
                c = col + i * dc
                if r < 0 or r >= self.BOARD_SIZE or c < 0 or c >= self.BOARD_SIZE:
                    continue
                if board[r, c] == self.ID:
                    piece_count += 1
                elif board[r, c] == 0:
                    continue
                else:
                    piece_count = 0
                    break
        
                if piece_count == self.X_IN_A_LINE - 2:
                    score += 1000
                if piece_count >= self.X_IN_A_LINE - 1:
                    score += 10000


        return score
    

    '''
    This function is to get the score for a given direction with direction vector on the board
    Parameters:
        - board: the game board
        - row: row of the starting position
        - col: column of the starting position
        - dr: direction vector for the row
        - dc: direction vector for the column
    Returns:
        - score: the score for the given direction
    '''       
    def get_score_for_direction(self, board, row, col, dr, dc):
        score = 0
        player_piece_count = 0
        open_end = False
        #Check over the positions in the given direction
        for i in range(self.X_IN_A_LINE):
            r = row + i * dr
            c = col + i * dc
            #Break the loop if the position is out of the board
            if r < 0 or r >= self.BOARD_SIZE or c < 0 or c >= self.BOARD_SIZE:
                break
            # Increment the player piece count if the position contains the player's piece
            if board[r][c] == self.ID:
                player_piece_count += 1
            # If the position is empty check if it is a open end
            elif board[r][c] == 0:
                if i == 0 or i == self.X_IN_A_LINE - 1:
                    open_end = True
                else:
                    open_end = False
                break
            #Break loop if position has the opponent's piece
            else:
                open_end = False
                break
            
        # Count the number of opponent's pieces in the given direction
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
        # Update the score based on the player piece count and open end status
        if player_piece_count > 0 and player_piece_count < self.X_IN_A_LINE and open_end:
            score += player_piece_count
        elif player_piece_count == self.X_IN_A_LINE:
            score += self.X_IN_A_LINE * 2

        return score
    
    """
    This function calculate the score for potential moves at a given position.
    Parameters:
        - board: The current state of the board.
        - r: The row number of the potential move.
        - c: The column number of the potential move.
    Returns:
        - score: The score of the potential move.
    """
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

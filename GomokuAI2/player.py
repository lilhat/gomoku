# Import required libraries
import numpy as np
from gomokuAgent import GomokuAgent
from misc import legalMove, winningTest
import math


class Node:
    def __init__(self, board, player):
        self.board = board
        self.player = player
        self.children = []
        self.wins = 0
        self.visits = 0
        self.parent = None
        self.BOARD_SIZE = len(board)
        self.X_IN_A_LINE = X_IN_A_LINE

    def add_child(self, node):
        self.children.append(node)
        node.parent = self

    def get_ucb(self, exploration_constant):
        if self.visits == 0:
            return math.inf
        exploitation = self.wins / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def select_child(self, exploration_constant):
        return max(self.children, key=lambda child: child.get_ucb(exploration_constant))

    def expand(self):
        for move in self.generate_moves(self.board):
            new_board = np.copy(self.board)
            new_board[move] = self.get_opponent(self.player)
            child_node = Node(new_board, self.get_opponent(self.player))
            self.add_child(child_node)

    def rollout(self):
        board = np.copy(self.board)
        player = self.player
        while self.check_win_loss(board, player) is None:
            move = self.random_move(board)
            board[move] = player
            player = self.get_opponent(player)
        return self.check_win_loss(board, self.player)

    def backpropagate(self, result):
        self.visits += 1
        self.wins += result
        if self.parent is not None:
            self.parent.backpropagate(result)

    def get_opponent(self, player):
        return 3 - player

    def check_win_loss(self, board, player):
        if self.check_winner(board, player):
            return 1
        elif self.check_winner(board, self.get_opponent(player)):
            return -1
        elif len(self.generate_moves(board)) == 0:
            return 0
        else:
            return None

    def check_winner(self, board, player):
        for i in range(self.BOARD_SIZE):
            for j in range(self.BOARD_SIZE):
                if board[i][j] == player:
                    if self.check_row(board, i, j, player):
                        return True
                    if self.check_col(board, i, j, player):
                        return True
                    if self.check_diag(board, i, j, player):
                        return True
                    if self.check_antidiag(board, i, j, player):
                        return True
        return False

    def check_row(self, board, row, col, player):
        if col + self.X_IN_A_LINE > self.BOARD_SIZE:
            return False
        for j in range(col, col + self.X_IN_A_LINE):
            if board[row][j] != player:
                return False
        return True

    def check_col(self, board, row, col, player):
        if row + self.X_IN_A_LINE > self.BOARD_SIZE:
            return False
        for i in range(row, row + self.X_IN_A_LINE):
            if board[i][col] != player:
                return False
        return True

    def check_diag(self, board, row, col, player):
        if row + self.X_IN_A_LINE > self.BOARD_SIZE or col + self.X_IN_A_LINE > self.BOARD_SIZE:
            return False
        for i, j in zip(range(row, row + self.X_IN_A_LINE), range(col, col + self.X_IN_A_LINE)):
            if board[i][j] != player:
                return False
        return True

class Player(GomokuAgent):
    # Class constructor
    def __init__(self, ID, BOARD_SIZE, X_IN_A_LINE, MCTS_SIMS):
        # Calls the constructor from GomokuAgent
        super().__init__(ID, BOARD_SIZE, X_IN_A_LINE)
        # Set the number of MCTS simulations to run
        self.MCTS_SIMS = MCTS_SIMS

    # Overwriting the move function from GomokuAgent
    def move(self, board):
        # Initialize the root node of the MCTS tree
        root = Node(board)

        # Run MCTS for a set number of simulations
        for _ in range(self.MCTS_SIMS):
            # Selection
            node = root
            while not node.is_leaf():
                node = node.select_child()

            # Expansion
            if not node.is_terminal():
                node.expand()

            # Simulation
            winner = node.rollout()

            # Backpropagation
            node.backpropagate(winner)

        # Choose the best move based on the MCTS results
        best_move = root.select_child(0).move

        return best_move
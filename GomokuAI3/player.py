import math
import copy
import numpy as np
from gomokuAgent import GomokuAgent
import time
from misc import legalMove, winningTest
from random import randint, choice

class Node:
    def __init__(self, board, parent, current_player, move_loc):
        self.board = board
        self.parent = parent
        self.current_player = current_player
        self.move_loc = move_loc
        self.children = []
        self.visits = 0
        self.wins = 0
        self.expanded = False

    def expand(self):
        legal_moves = [(i, j) for i in range(len(self.board)) for j in range(len(self.board)) if self.board[i][j] == 0]

        for move in legal_moves:
            new_board = copy.deepcopy(self.board)
            new_board[move[0]][move[1]] = self.current_player
            new_node = Node(new_board, self, -self.current_player, move)
            self.children.append(new_node)

        self.expanded = True

class Player(GomokuAgent):
    def __init__(self, ID, BOARD_SIZE, X_IN_A_LINE):
        self.ID = ID
        self.board_size = BOARD_SIZE
        self.x_in_a_line = X_IN_A_LINE
        self.TIME_OUT = 5
        self.transposition_table = {}

    def move(self, board):
        start_time = time.time()
        end_time = start_time + self.TIME_OUT

        root = Node(board, None, self.ID, None)

        while time.time() < end_time:
            node = root
            # selection
            while node.children:
                node = self.select_child(node)

            # expansion
            if not node.expanded:
                node.expand()

            # simulation
            winner = self.simulate(node)

            # backpropagation
            while node:
                node.visits += 1
                if winner == self.ID:
                    node.wins += 1
                node = node.parent

        best_child = self.select_best_child(root)
        return best_child.move_loc

    def select_child(self, node):
        total_visits = sum(child.visits for child in node.children)
        log_total = math.log(total_visits or 1)

        best_score = float("-inf")
        best_child = None

        for child in node.children:
            exploit = child.wins / (child.visits + 0.1)
            explore = math.sqrt(log_total / (child.visits + 0.1))
            score = exploit + explore

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def select_best_child(self, node):
        most_visits = float("-inf")
        best_child = None

        for child in node.children:
            if child.visits > most_visits:
                best_child = child
                most_visits = child.visits
            elif child.visits == most_visits:
                if child.wins / child.visits > best_child.wins / best_child.visits:
                    best_child = child

        return best_child

    def simulate(self, node, alpha=float("-inf"), beta=float("inf")):
        board = copy.deepcopy(node.board)
        current_player = node.current_player
        board_size = self.board_size
        x_in_a_line = self.x_in_a_line

        while True:
            legal_moves = [(i, j) for i in range(board_size) for j in range(board_size) if board[i][j] == 0]
            if not legal_moves:
                return 0

            # Evaluate each move using a simple heuristic
            scores = []
            for move in legal_moves:
                score = 0
                for direction in [(1,0), (0,1), (1,1), (1,-1)]:
                    count = 0
                    for i in range(-x_in_a_line+1, x_in_a_line):
                        row = move[0] + i * direction[0]
                        col = move[1] + i * direction[1]
                        if (row < 0 or row >= board_size or col < 0 or col >= board_size or
                                board[row][col] != current_player):
                            break
                        count += 1
                    if count >= x_in_a_line:
                        score += count * count  # Add a bonus for longer lines
                scores.append(score)

            # Choose the move with the highest score
            best_moves = [i for i in range(len(legal_moves)) if scores[i] == max(scores)]
            move_index = choice(best_moves)
            move = legal_moves[move_index]

            board[move] = current_player

            if winningTest(current_player, board, x_in_a_line):
                return current_player

            current_player = -current_player

            if current_player == self.ID:
                alpha = max(alpha, max(scores))
                if alpha >= beta:
                    return self.ID
            else:
                beta = min(beta, min(scores))
                if beta <= alpha:
                    return -self.ID
        
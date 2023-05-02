# group: 851108, 1913899

# Initially created a solution using the Minimax algorithm with Alpha-Beta pruning alongside some in depth heuristics.
# The algorithm had a depth of 1 which worked with a time out of 10 seconds, but would time out before finding a solution for each move at 5 seconds.

# The solution was recreated using Monte Carlo Tree Search (MCTS), which has a simpler 'anytime' behaviour. This means the iterations can continue to run
# until the time limit is reached, which then returns the best move found so far. This required significantly less processing time, and allowed for simpler
# heuristics to be integrated, while still producing promising results.


# Import required libraries
import math
import copy
from gomokuAgent import GomokuAgent
import time
from misc import winningTest
from random import choice

class Node:

    '''
    Initializing the node with the necessary attributes for the Node class
    '''
    def __init__(self, board, parent, current_player, move_loc):
        self.board = board # The current game state of the board
        self.parent = parent # Parent node
        self.current_player = current_player # The current player
        self.move_loc = move_loc # The move location
        # Initializing the node with an empty list of children
        self.children = []
        # Initializing the number of times the nodes has been visited and the number of wins 
        self.visits = 0
        self.wins = 0
        # Initialize a flag indicating whether the node has been expanded yet or not
        self.expanded = False

    '''
    Create child nodes for the current node 
    '''
    def expand(self):
        # Create a list of all legal moves that can be made by the current player
        legal_moves = [(i, j) for i in range(len(self.board)) for j in range(len(self.board)) if self.board[i][j] == 0]

        # Create a new board for each legal move by making that move and add it as a child node to the current node
        for move in legal_moves:
            new_board = copy.deepcopy(self.board)
            new_board[move[0]][move[1]] = self.current_player
            new_node = Node(new_board, self, -self.current_player, move)
            self.children.append(new_node)

        # Mark the current node as expanded so that its does not get expanded again in the future
        self.expanded = True

class Player(GomokuAgent):
    '''
    Initializing the node with the necessary attributes for the Player class
    '''
    def __init__(self, ID, BOARD_SIZE, X_IN_A_LINE):
        self.ID = ID # Player ID
        self.board_size = BOARD_SIZE # Size of the game board
        self.x_in_a_line = X_IN_A_LINE # Number of stones in a row required to win the game
        self.TIME_OUT = 5 # The amount of time the player has to make a move

    '''
    The purpose of this method is to use monte carlo tree search to find
    the best move for a player.
    Parameters:
        - board: The current state of the board
    Returns:
        - best_child.mov_loc: The child node with the highest win rate
    '''
    def move(self, board):
        # Get the current time and the time at which the search should end
        start_time = time.time()
        end_time = start_time + self.TIME_OUT

        root = Node(board, None, self.ID, None)

        # Loop until time runs out
        while time.time() < end_time:
            # Starting at the root
            node = root
            # selection
            while node.children:
                # Select the child with the highest UCB1 score
                node = self.select_child(node)

            # expansion
            # If the selected node is not expanded yet, generate all possible child nodes
            if not node.expanded:
                node.expand()

            # simulation
            # Simulate a game from the selected child node until the end of the game
            winner = self.simulate(node)

            # backpropagation
            # Update the statitics of all nodes visited during the search based on the result of the simulated game.
            while node:
                node.visits += 1
                if winner == self.ID:
                    node.wins += 1
                node = node.parent

        # Select a child node with the highest win rate after the search is complete.
        best_child = self.select_best_child(root)
        return best_child.move_loc

    '''
    Selects the child node of the current node with the highest UCT score
    Parameters:
        - node: The current node being checked
    Returns:
        - best_child: The child node with the highest UCT score
    '''
    def select_child(self, node):
        # Calculate total visits for all children of current node
        total_visits = sum(child.visits for child in node.children)
        # Calculate logarithm of total visits, 1 if there are none
        log_total = math.log(total_visits or 1)

        best_score = float("-inf")
        best_child = None

        # Iterate over each child of current node
        for child in node.children:
            # Calculate ratio of wins to visits, add 0.01 to avoid ZeroDivisionError
            exploit = child.wins / (child.visits + 0.01)
            # Calculate term based on total visists and visits to current child, add 0.01 to avoid ZeroDivisionError
            explore = math.sqrt(log_total / (child.visits + 0.01))
            # Combine terms to give score for current child
            score = exploit + explore

            # If the score is higher than the current best score, replace with current child
            if score > best_score:
                best_score = score
                best_child = child

        # Return highest scoring child
        return best_child

    '''
    Selects the child node of the current node with the most visits, or if many children have 
    the most visits, the one with the highest ratio wins.
    Parameters:
        - node: The current node being checked
    Returns:
        - best_child: The child node with the highest number of visits, or if many children have 
        the most visits, the one with the highest ratio wins.
    '''
    def select_best_child(self, node):
        most_visits = float("-inf")
        best_child = None
        
        # Iterate over each child of current node
        for child in node.children:
            # If current child has more visits than the best child update the best child and most visits variables
            if child.visits > most_visits:
                best_child = child
                most_visits = child.visits
            # If current child has same number of visits as the best child, compare win ratio to make decision
            elif child.visits == most_visits:
                if child.wins / child.visits > best_child.wins / best_child.visits:
                    best_child = child
                    
        # Return highest scoring child
        return best_child

    '''
    Simulate a game from the given node by selecting moves using a heuristic.
    Parameters:
        - node: A node object representing the current state of the game
    Returns:
        - An integer (1 or -1) which indicates the winning player. 0 if draw.
    '''
    def simulate(self, node):
        # Create a copy of the board and set the current player
        board = copy.deepcopy(node.board)
        current_player = node.current_player
        # Get the board size and number of pieces in a row needed to win
        board_size = self.board_size
        x_in_a_line = self.x_in_a_line

        while True:
            # Get a list of all legal moves
            legal_moves = [(i, j) for i in range(board_size) for j in range(board_size) if board[i][j] == 0]
            # Check for a draw
            if not legal_moves:
                return 0

            # Check for potential opponent wins and block them
            opponent = -current_player
            # Iterate over each legal move
            for move in legal_moves:
                # Check in each possible direction
                for direction in [(1,0), (0,1), (1,1), (1,-1)]:
                    count = 0
                    # Iterate over each space around the move
                    for i in range(-x_in_a_line+1, x_in_a_line):
                        # Calculate the row and column coordinates
                        row = move[0] + i * direction[0]
                        col = move[1] + i * direction[1]
                        # Check if the move is out of bounds or not an opponent piece
                        if (row < 0 or row >= board_size or col < 0 or col >= board_size or
                                board[row][col] != opponent):
                            break # Stop counting if conditions are met
                        count += 1
                    if count >= x_in_a_line - 1:
                        # Found a potential opponent win, block it
                        board[move] = current_player
                        # Check if the current player wins
                        if winningTest(current_player, board, x_in_a_line):
                            return current_player
                        # Switch to other player
                        current_player = -current_player

            # Evaluate each move using a simple heuristic
            scores = []
            # Iterate over each legal move
            for move in legal_moves:
                score = 0
                # Check in each possible direction
                for direction in [(1,0), (0,1), (1,1), (1,-1)]:
                    count = 0
                    # Iterate over each space around the move
                    for i in range(-x_in_a_line+1, x_in_a_line):
                        # Calculate the row and column coordinates
                        row = move[0] + i * direction[0]
                        col = move[1] + i * direction[1]
                        # Check if the move is out of bound or not a current player piece
                        if (row < 0 or row >= board_size or col < 0 or col >= board_size or
                                board[row][col] != current_player):
                            break # Stop counting if conditions are met
                        count += 1 # Increase count if conditions are not met
                    if count >= x_in_a_line - 1: # Check if count is close or equal to 5 in a row
                        score += count * count  # Add a bonus for longer lines
                scores.append(score)

            # Choose the move with the highest score
            best_moves = [i for i in range(len(legal_moves)) if scores[i] == max(scores)]
            move_index = choice(best_moves)
            move = legal_moves[move_index]
        
            # Make the move on the board 
            board[move] = current_player
            
            # Check if the current player wins
            if winningTest(current_player, board, x_in_a_line):
                return current_player

            # Switch to the other player
            current_player = -current_player

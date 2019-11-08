import numpy as np
from TicTacToe import boardCheck
import matplotlib.pyplot as plt



def run_from_ipython():
    """
    Check if the script is run from command line or from a Jupyter Notebook.

    :return: bool that is True if run from Jupyter Notebook
    """
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

if run_from_ipython():
    # Selectively import the right version of tqdm
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

class State:
    def __init__(self, p1, p2, **settings):
        self.WIN = settings['WIN']
        self.MAX_SCORE = settings['MAX_SCORE']
        self.BOARD_ROWS = settings['BOARD_ROWS']
        self.BOARD_COLS = settings['BOARD_COLS']
        
        self.board = np.zeros((self.BOARD_ROWS, self.BOARD_COLS), dtype=np.int8)
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None
        # init p1 plays first
        self.playerSymbol = 1
        self.available_positions = {j+i*self.BOARD_COLS: (i, j) for i in range(self.BOARD_ROWS) for j in range(self.BOARD_COLS)}
        self.results = {
            'played': 0,
            'p1': [],
            'p2': [],
            'tie': []
        }
        
        
        self.max_score = 1
    
    def displayMetrics(self):
        # Sanity checks
        assert len(self.results["tie"])+len(self.results["p1"]) + len(self.results["p2"]) == self.results["played"]
        print("[INFO] {} games played".format(self.results["played"]))
        print("P1 won: {} | P2 won: {} | Ties: {}"
              .format(len(self.results["p1"]), len(self.results["p2"]), len(self.results["tie"])))
        
        expand_result = np.zeros((self.results['played'],))
        for game in self.results['p1']:
            expand_result[game[0]] = 1
        for game in self.results['p2']:
            expand_result[game[0]] = -1
        
        """
        # Linear scatter plot
        plt.figure(figsize=(15, 5))
        plt.scatter(range(self.results["played"]), expand_result, alpha=0.1)
        plt.title('Overall training result')
        plt.xlabel('Game number')
        plt.ylabel('Won by')
        
        # Do the same with a semilogx instead
        plt.figure(figsize=(15, 5))
        plt.semilogx(range(self.results["played"]), expand_result, linestyle="", marker=".", alpha=0.1)
        plt.title('Overall training result')
        plt.xlabel('Game number')
        plt.ylabel('Won by')
        """
        
        # Cum sum plot might be better?
        cum_p1 = np.zeros((self.results['played'],))
        cum_p2 = np.zeros((self.results['played'],))
        cum_tie = np.zeros((self.results['played'],))
        if expand_result[0] == 1: cum_p1[0] = 1
        if expand_result[0] == -1: cum_p2[0] = 1
        if expand_result[0] == 0: cum_tie[0] = 1
        for i in range(1, self.results['played']):
            cum_p1[i] = cum_p1[i-1] + 1 if expand_result[i] == 1 else cum_p1[i-1]
            cum_p2[i] = cum_p2[i-1] + 1 if expand_result[i] == -1 else cum_p2[i-1]
            cum_tie[i] = cum_tie[i-1] + 1 if expand_result[i] == 0 else cum_tie[i-1]
        plt.figure(figsize=(15, 5))
        plt.plot(cum_p1, label='Player 1')
        plt.plot(cum_p2, label='Player 2')
        plt.plot(cum_tie, label='Tie')
        plt.legend()
        plt.title("Cumulative win sum")
        plt.xlabel('Game number')
        plt.ylabel('Cumulative score')
        
        # Learning & exploration rate over time
        plt.figure(figsize=(15, 5))
        lr = [self.p1.learningRate(k) for k in range(self.results['played'])]
        explo = [self.p1.explorationRate(k) for k in range(self.results['played'])]
        plt.plot(lr, label='Learning rate')
        plt.plot(explo, label='Exploration rate')
        plt.title('Evolution of the learning & exploration rates')
        plt.xlabel('Game number')
        plt.ylabel('Rate')
        plt.legend()
    
    # get unique hash of current board state
    def getHash(self, method='tostring'):
        if method == 'str':
            self.boardHash = str(self.board.reshape(self.BOARD_COLS*self.BOARD_ROWS))
        elif method == 'tostring':
            self.boardHash = self.board.tostring()
        return self.boardHash
    
    def old_winner(self):
        # row
        for i in range(self.BOARD_ROWS):
            if sum(self.board[i, :]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[i, :]) == -30:
                self.isEnd = True
                return -1
        # col
        for i in range(self.BOARD_COLS):
            if sum(self.board[:, i]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[:, i]) == -30:
                self.isEnd = True
                return -1
        # diagonal
        diag_sum1 = sum([self.board[i, i] for i in range(self.BOARD_COLS)])
        diag_sum2 = sum([self.board[i, self.BOARD_COLS-i-1] for i in range(self.BOARD_COLS)])
        #diag_sum = max(diag_sum1, diag_sum2)
        if diag_sum1 == 3 or diag_sum2 == 3:
            self.isEnd = True
            return 1
        if diag_sum1 == -30 or diag_sum2 == -30:
            self.isEnd = True
            return -1
        
        # tie
        # no available positions
        if len(self.available_positions) == 0:
            self.isEnd = True
            return 0
        # not end
        self.isEnd = False
        return None

    def winner(self, player):
        """
        Checks if there is a winner already or not by hecking rows, cols and diag on the board.
        The whole board is rechecked, which wastes some time.
        return: 1 of player 1 won, -1 if player 2, 0 if tie
        """
        #print("Checking the board - still {} positions to play".format(len(self.available_positions)))
        # row
        for i in range(self.BOARD_ROWS):
            if sum(self.board[i, :]) in player.win and player.id == 1:
                self.isEnd = True
                return 1
            if sum(self.board[i, :]) in player.win and player.id == 2:
                self.isEnd = True
                return -1
        # col
        for i in range(self.BOARD_COLS):
            if sum(self.board[:, i]) in player.win and player.id == 1:
                self.isEnd = True
                return 1
            if sum(self.board[:, i]) in player.win and player.id == 2:
                self.isEnd = True
                return -1
        
        """Should use hills/dales"""
        # diagonal
        diag_sum1 = sum([self.board[i, i] for i in range(self.BOARD_COLS)])
        diag_sum2 = sum([self.board[i, self.BOARD_COLS-i-1] for i in range(self.BOARD_COLS)])
        #diag_sum = max(diag_sum1, diag_sum2)
        if (diag_sum1 in player.win or diag_sum2 in player.win) and player.id == 1:
            self.isEnd = True
            return 1
        if (diag_sum1 in player.win or diag_sum2 in player.win) and player.id == 2:
            self.isEnd = True
            return -1
        
        # tie
        # no available positions
        #if len(self.availablePositions()) == 0:
        if len(self.available_positions) == 0:
            self.isEnd = True
            return 0
        # not end
        self.isEnd = False
        return None
    
    def fastWinner(self, action, player):
        boardCheck.updateScores(self.board.copy(), *action, player, self.WIN)
        if player.score >= self.MAX_SCORE:
            if player.id == 1:
                win = 1
            elif player.id == 2:
                win = -1
            else:
                raise ValueError
            #win = 1 if player.id == 1 else -1  # Really not great but works
        elif len(self.available_positions) == 0 and player.score==0:  # Tied
            win = 0
        elif player.score == 0:  # Keep playing
            win = None
        else:
            raise ValueError('Problem! Fix me!')
        return win

    
    def updateState(self, position, player):
        #self.board[position] = self.playerSymbol
        
        # A player always lay a pc1. These can be turned into pc2 if counted in a win
        self.board[position] = player.pc1
        # switch to another player
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1
    
    # only when game ends
    def giveReward(self, result):
        #result = self.winner()
        # backpropagate reward
        if result == 1:  # p1 won
            self.p1.feedReward(1, self.game_number)
            self.p2.feedReward(0, self.game_number)
        elif result == -1:  # p2 won
            self.p1.feedReward(0, self.game_number)
            self.p2.feedReward(1, self.game_number)
        else:  # There was a tie or the game is still running
            self.p1.feedReward(0.1, self.game_number)
            self.p2.feedReward(0.5, self.game_number)
        
    
    # board reset
    def reset(self):
        self.board = np.zeros((self.BOARD_ROWS, self.BOARD_COLS), dtype=np.int8)
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1
        self.available_positions = {j+i*self.BOARD_COLS: (i, j) for i in range(self.BOARD_ROWS) for j in range(self.BOARD_COLS)}
        self.p1.score = 0
        self.p2.score = 0
    
    def play(self, rounds=100, save_policies=None, show_board=False):
        for game in tqdm(range(rounds)):
            self.game_number = game
            if save_policies and game%save_policies == 0 and game != 0:
                print("Saving policy at round {}".format(game))
                self.p1.savePolicy(game)
                self.p2.savePolicy(game)
            #assert not self.isEnd
            turns = 0  # Tracks how many turns were played before a winner was found
            
            while not self.isEnd:
                # Player 1
                #positions = self.availablePositions()
                #p1_action = self.p1.chooseAction(self.available_positions, self.board, self.playerSymbol, self.game_number)
                p1_action = self.p1.chooseAction(self.available_positions, self.board, self.game_number)
                # take action and upate board state
                self.updateState(p1_action, self.p1)
                board_hash = self.getHash()
                self.p1.addState(board_hash)
                
                # check board status if it is end

                #win = self.winner(self.p1)
                #win2 = self.old_winner()
                win = self.fastWinner(p1_action, self.p1)
                """
                try:
                    assert win == win2
                except:
                    print("Debug: win != win2")
                    print(win, win2)
                    print(self.board)
                    print(p1_action)
                    print(p1.win)
                    raise
                """

                if win is not None:
                    # self.showBoard()
                    # ended with p1 either win or draw
                    if win == 1:
                        self.results['p1'].append((game, turns))
                        #print("x won game ", self.game_number)
                    elif win == 0:
                        self.results['tie'].append((game, turns))
                    else:
                        raise ValueError('[ERROR] Only p1 can have won at this stage or tied')
                    
                    
                    if show_board:
                        self.showBoard()
                    self.giveReward(win)
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                else:
                    # Player 2
                    #positions = self.availablePositions()
                    #p2_action = self.p2.chooseAction(self.available_positions, self.board, self.playerSymbol, self.game_number)
                    p2_action = self.p2.chooseAction(self.available_positions, self.board, self.game_number)
                    self.updateState(p2_action, self.p2)
                    board_hash = self.getHash()
                    self.p2.addState(board_hash)
                    
                    #win = self.winner(self.p2)
                    #win2 = self.old_winner()
                    win = self.fastWinner(p2_action, self.p2)
                    """
                    try:
                        assert win == win2
                    except:
                        print("Debug: win != win2")
                        print(win, win2)
                        print(self.board)
                        print(p2_action)
                        print(p2.win)
                        raise
                    """
                    
                    if win is not None:
                        # self.showBoard()
                        # ended with p2 either win or draw
                        if win == -1:
                            self.results['p2'].append((game, turns))
                            #print("o won game ", self.game_number)
                        elif win == 0:
                            self.results['tie'].append((game, turns))
                        else:
                            raise ValueError('[ERROR] Only p2 can have won at this stage or tied')
                        
                        
                        if show_board:
                            self.showBoard()
                        self.giveReward(win)
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break
                turns += 1  # Another turn was played
            
            
            self.results['played'] += 1
        
        # Save last policy
        if save_policies:
            print("Saving policy at round {}".format(game))
            self.p1.savePolicy(game)
            self.p2.savePolicy(game)
            
            
    
    # play with human
    def playComputerVsHuman(self):
        while not self.isEnd:
            # Player 1
            #positions = self.availablePositions()
            p1_action = self.p1.chooseAction(self.available_positions, self.board, self.playerSymbol)
            # take action and upate board state
            #self.updateState(p1_action)
            self.updateState(p1_action, self.p1)
            self.showBoard()
            # check board status if it is end
            # win = self.winner()
            win = self.fastWinner(p1_action, self.p1)
            if win is not None:
                if win == 1:
                    print(self.p1.name, "wins!")
                else:
                    print("tie!")
                self.reset()
                break

            else:
                # Player 2
                #positions = self.availablePositions()
                p2_action = self.p2.chooseAction(self.available_positions)

                self.updateState(p2_action, self.p2)
                self.showBoard()
                #win = self.winner()
                win = self.fastWinner(p2_action, self.p2)
                if win is not None:
                    if win == -1:
                        print(self.p2.name, "wins!")
                    else:
                        print("tie!")
                    self.reset()
                    break
    

    def showBoard(self):
        # p1: x  p2: o
        for i in range(0, self.BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(0, self.BOARD_COLS):
                if self.board[i, j] in (self.p1.pc1, self.p1.pc2):
                    token = 'x'
                if self.board[i, j] in (self.p2.pc1, self.p2.pc2):
                    token = 'o'
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('-------------')    

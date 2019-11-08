import numpy as np
import pickle


class Player:
    def __init__(self, name, player_id, pc, max_l, exp_rate=0.3):
        self.name = name
        self.states = []  # record all positions taken
        self.lr_scheduler = {
            'start': 0.2,
            'decreasing': (500, 1500),
            'end': 0.2
        }
        #self.exp_rate = exp_rate  # Exploration rate
        self.explo_scheduler = {
            'start': 0.3,
            'decreasing': (1000, 45000),
            'end': 0
        }
        self.decay_gamma = 0.9  # Discount rate
        self.states_value = {}  # state -> value
        self.score = 0
        self.id = player_id
        self.pc1, self.pc2 = pc
        self.win = {max_l*self.pc1, (max_l-1)*self.pc1 + self.pc2}
        
        # Sanity checks - make sure the schedulers have con
    def test_learningRate(self):
        # TO DO: Test 4 points: start, first inflexion, second inflexion, end
        assert learningRate(self, self.lr_scheduler['decreasing'][0]) == self.lr_scheduler['start']
        assert learningRate(self, self.lr_scheduler['decreasing'][0]) == self.lr_scheduler['start']
        assert learningRate(self, self.lr_scheduler['decreasing'][1]) == self.lr_scheduler['end']
        assert learningRate(self, self.lr_scheduler['decreasing'][0]) == self.lr_scheduler['end']
    
    def test_explorationRate(self):
        assert explorationRate(self, self.explo_scheduler['decreasing'][0]) == self.explo_scheduler['start']
        assert explorationRate(self, self.explo_scheduler['decreasing'][1]) == self.explo_scheduler['end']
        
        
    def learningRate(self, iteration):
        if iteration < self.lr_scheduler['decreasing'][0]:
            return self.lr_scheduler['start']
        elif self.lr_scheduler['decreasing'][0] <= iteration < self.lr_scheduler['decreasing'][1]:
            offset = self.lr_scheduler['decreasing'][0]
            span = self.lr_scheduler['decreasing'][1] - self.lr_scheduler['decreasing'][0]
            return self.lr_scheduler['start']-(self.lr_scheduler['start']-self.lr_scheduler['end'])*(iteration-offset)/span
        else:
            return self.lr_scheduler['end']
    
    def explorationRate(self, iteration):
        if iteration < self.explo_scheduler['decreasing'][0]:
            return self.explo_scheduler['start']
        elif self.explo_scheduler['decreasing'][0] <= iteration < self.explo_scheduler['decreasing'][1]:
            offset = self.explo_scheduler['decreasing'][0]
            span = self.explo_scheduler['decreasing'][1] - self.explo_scheduler['decreasing'][0]
            return self.explo_scheduler['start']-(self.explo_scheduler['start']-self.explo_scheduler['end'])*(iteration-offset)/span
        else:
            return self.explo_scheduler['end']
    
    
    def getHash(self, board, method='tostring'):
        if method == 'str':
            boardHash = str(board.reshape(BOARD_COLS*BOARD_ROWS))
        elif method == 'tostring':
            boardHash = board.tostring()
        return boardHash
    
    def chooseAction(self, positions, current_board, iteration):
        #if np.random.uniform(0, 1) <= self.exp_rate:
        if np.random.uniform(0, 1) <= self.explorationRate(iteration):
            # take random action
            idx = np.random.choice(len(positions))
            action = positions[idx]
            self.deletePosition(positions, idx)
        else:  # Calculate the score for all the possible upcoming states (maximum n*m for the first move)
            value_max = -999
            #i = -1
            #print(positions)
            for idx, pos in positions.items():  # Evaluate all next moves
                """
                # Option 1: copy the whole array
                next_board = current_board.copy()
                next_board[pos] = symbol  # Valid because this positions has all the valid moves
                next_boardHash = self.getHash(next_board)  # Do we need that function call?
                
                """
                # Option 2: incremental move & revert
                temp_ = current_board[pos]
                #current_board[pos] = symbol
                current_board[pos] = self.pc1
                next_boardHash = self.getHash(current_board)
                current_board[pos] = temp_
                

                #value = self.states_value.get(next_boardHash, 0)
                #print(value, value_max)
                value = 0 if self.states_value.get(next_boardHash) is None else self.states_value.get(next_boardHash)
                # print("value", value)
                
                # Selects the action with the highest value
                if value >= value_max:
                    value_max = value
                    action = pos
                    i = idx
        # print("{} takes action {}".format(self.name, action))
            self.deletePosition(positions, i)  # Played position i
        return action
    
    def deletePosition(self, positions, idx):
        # Remove that position from the dict by copying the last entry to idx
        # As random deletes are made, the index stop meaning anything and
        # are just a convenient way to randomly sample the distribution.
        len_ = len(positions)  # O(1)
        positions[idx] = positions[len_-1]  # O(1)
        del positions[len_-1]  # O(1)
            
    # append a hash state
    def addState(self, state):
        self.states.append(state)
    
    # at the end of game, backpropagate and update states value
    def feedReward(self, reward, iteration):
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
            #self.states_value[st] += self.lr*(self.decay_gamma*reward - self.states_value[st])
            self.states_value[st] += self.learningRate(iteration)*(self.decay_gamma*reward - self.states_value[st])
            reward = self.states_value[st]
            
    def reset(self):
        self.states = []
        
    def savePolicy(self, game):
        title = 'policies/' + str(self.name) + '_' + str(game)
        with open(title , 'wb') as fw:
            pickle.dump(self.states_value, fw)

    def loadPolicy(self, file):
        with open(file,'rb') as fr:
            self.states_value = pickle.load(fr)


class HumanPlayer:
    def __init__(self, name, player_id, pc, max_l):
        self.name = name
        self.score = 0
        self.id = player_id
        self.pc1, self.pc2 = pc
        self.win = {max_l*self.pc1, (max_l-1)*self.pc1 + self.pc2}
    
    def chooseAction(self, positions):
        while True:
            row = int(input("Input your action row:"))
            col = int(input("Input your action col:"))
            action = (row, col)
            if action in positions.values():
                # Find the corresponding key - gettho but ok as we play vs human
                idx = next(k for k, v in positions.items() if v == action)
                self.deletePosition(positions, idx)
                return action
    
    def deletePosition(self, positions, idx):
        # Remove that position from the dict by copying the last entry to idx
        # As random deletes are made, the index stop meaning anything and
        # are just a convenient way to randomly sample the distribution.
        len_ = len(positions)  # O(1)
        positions[idx] = positions[len_-1]  # O(1)
        del positions[len_-1]  # O(1)
        
    # append a hash state
    def addState(self, state):
        pass
    
    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        pass
            
    def reset(self):
        pass

from TicTacToe import player, state
import numpy as np
import glob

def loadAllPolicies(folder='policies/*'):
    list_policies = glob.glob(folder)
    p1_policies = [p for p in list_policies if p.split('/')[-1].split('_')[0] == 'p1']
    p2_policies = [p for p in list_policies if p.split('/')[-1].split('_')[0] == 'p2']
    return sorted(p1_policies), sorted(p2_policies)

def compositeScore(results):
    p1win = len(results["p1"])
    p2win = len(results["p2"])
    tie = len(results["tie"])
    return p1win - p2win

def benchmarkPolicies(p1_policies, p2_policies, **settings):
    nb_games = 100
    p1win = np.zeros((len(p1_policies), len(p2_policies)), dtype=np.int8)
    p2win = np.zeros((len(p1_policies), len(p2_policies)), dtype=np.int8)
    tie = np.zeros((len(p1_policies), len(p2_policies)), dtype=np.int8)
    for idxp1, policy_p1 in enumerate(p1_policies):
        for idxp2, policy_p2 in enumerate(p2_policies):
            # 1. Create players and load respective policies
            p1 = player.Player("p1", 1, settings['pc1'], settings['WIN'], exp_rate=0)
            p1.loadPolicy(policy_p1)
            p2 = player.Player("p2", 2, settings['pc2'], settings['WIN'], exp_rate=0)
            p2.loadPolicy(policy_p2)

            # 2. Play 100 games
            st = state.State(p1, p2, **settings)
            st.play(rounds=nb_games, save_policies=False)

            # 3. Store the results
            assert st.results["played"] == nb_games
            p1win[idxp1, idxp2] = len(st.results['p1'])
            p2win[idxp1, idxp2] = len(st.results['p2'])
            tie[idxp1, idxp2] = len(st.results['tie'])
    return p1win, p2win, tie




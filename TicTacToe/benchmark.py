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
    
    # I. Populate the result matrix with 100 games played at each level
    for idxp1, policy_p1 in enumerate(p1_policies):
        for idxp2, policy_p2 in enumerate(p2_policies):
            # I.1. Create players and load respective policies
            p1 = player.Player("p1", 1, settings['pc1'], settings['WIN'], explo_schedule=(0, (0, 0), 0))
            p1.loadPolicy(policy_p1)
            p2 = player.Player("p2", 2, settings['pc2'], settings['WIN'], explo_schedule=(0, (0, 0), 0))
            p2.loadPolicy(policy_p2)

            # I.2. Play 100 games
            st = state.State(p1, p2, **settings)
            st.play(rounds=nb_games, save_policies=False)

            # I.3. Store the results
            assert st.results["played"] == nb_games
            p1win[idxp1, idxp2] = len(st.results['p1'])
            p2win[idxp1, idxp2] = len(st.results['p2'])
            tie[idxp1, idxp2] = len(st.results['tie'])
    
    # II. Calculate how "ideal" that matrix is
    return tournament_score(p1win), tournament_score(p2win)
    # return p1win, p2win, tie


def tournament_score(a, debug=False):
    """
    Measures how close you are from the ideal tournament result between the agent at different iterations.
    A game between p1 with i iterations and p2 with j iterations should have a higher p1 win count than p1 at the i-1th iteration and/or p2 at the j-1th iteration.
    The ideal result matrix has 3 points in each entry, 1 on the top and left edges and 0 in [0, 0]
    """
    n, m = a.shape
    res = np.zeros((n, m), dtype=np.uint8)
    for i in range(n):
        for j in range(m):
            if i > 0 and j > 0:  
                if a[i, j] >= a[i-1, j]: res[i, j] += 1
                if a[i, j] >= a[i-1, j-1]: res[i, j] += 1
                if a[i, j] >= a[i, j-1]: res[i, j] += 1
            elif i > 0 and j == 0:
                if a[i, j] >= a[i-1, j]: res[i, j] += 1
            elif i == 0 and j > 0:
                if a[i, j] >= a[i, j-1]: res[i, j] += 1
            else:  # i == 0 and j == 0:
                pass
    
    if debug: return res, sum(sum(res))/(3*(n-1)*(m-1)+(n-1)+(m-1))
    return sum(sum(res))/(3*(n-1)*(m-1)+(n-1)+(m-1))

def test_score():
    n = m = 10
    test = np.zeros((n, m), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            test[i, j] = i+j
    res, v = tournament_score(test, debug=True)
    expected_res = np.zeros((n, m), dtype=np.int8) + 3
    expected_res[:, 0] = 1
    expected_res[0, :] = 1
    expected_res[0, 0] = 0
    assert np.array_equal(res, expected_res)  # Check matrix result
    assert v == 1
    return True
test_score()


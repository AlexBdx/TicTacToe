import numpy as np

def updateScores(grid, i, j, p, max_l=5):
    # Get subgrid centered around where was just played
    sg, r, c, window = extractSubgrid(grid, i, j, max_l)
    # Check all 
    p.score += check_row(sg, r, p, max_l)
    p.score += check_col(sg, c, p, max_l)
    p.score += check_hills(sg, r, c, p, max_l)
    p.score += check_dales(sg, r, c, p, max_l)
    writeSubgrid(sg, grid, window)  # Write back the updated grid to the main one
    #return grid

def extractSubgrid(g, i, j, max_l):
    rmax, cmax = g.shape
    rstart, rend = max(0, i-(max_l-1)), min(rmax, i+(max_l))
    cstart, cend = max(0, j-(max_l-1)), min(cmax, j+(max_l))
    rl, cl = i-rstart, j-cstart  # Verify the local index mapping
    window = (rstart, rend, cstart, cend)
    assert rend-rstart >= max_l  # Check the size of the new
    assert cend-cstart >= max_l
    return g[rstart:rend, cstart:cend], rl, cl, window


def writeSubgrid(sg, g, w):
    # w is (rstart, rend, cstart, cend)
    g[w[0]:w[1], w[2]:w[3]] = sg


# Row check
def check_row(g, r, p, max_l):
    """[MAIN ALGO] If something has to be moded here, probably in the others too.
    Returns nothing, modifies p's score in place if needed
    grid g is also modified in place
    """
    points = 0
    _, cmax = g.shape
    s = sum(g[r, :max_l-1]) + g[r, 0]  # Double first elem and rm last
    for k in range(cmax-max_l+1):  # Iterate on cols
        s += g[r, max_l-1+k]  # Add head
        s -= g[r, k]  # Remove tail
        if s in p.win:
            #print('Row win on:', k, c, g)
            g[r, k:max_l+k] = p.pc2  # Assigns values
            points += 1
            if k == 0 and cmax == 2*max_l-1:  # CORNER CASE: what if you joined a whole line?
                s = sum(g[r, max_l-1:])
                if s in p.win:  # Only works if all other symbols are not assigned
                    g[r, max_l-1:] = p.pc2  # Assigns values
                    #p['score'] += 1
                    points += 1
                    break
            else:  # That is the only point you could score in that line
                break
    return points


# Col check
def check_col(g, c, p, max_l):
    """Returns nothing, modifies p's score in place if needed
    grid g is also modified in place
    """
    points = 0
    rmax, _ = g.shape
    s = sum(g[:max_l-1, c]) + g[0, c]  # Double first elem and rm last
    for k in range(rmax-max_l+1):  # Iterate on rows
        s += g[max_l-1+k, c]  # Add head
        s -= g[k, c]  # Remove tail
        if s in p.win:
            #print('Col win on:', k, c, g)
            g[k:max_l+k, c] = p.pc2  # Assigns values
            points += 1
            if k == 0 and rmax == 2*max_l-1:  # CORNER CASE: what if you joined a whole line?
                s = sum(g[max_l-1:, c])
                if s in p.win:  # Only works if all other symbols are not assigned
                    g[max_l-1:, c] = p.pc2
                    points += 1
                    break
            else:  # That is the only point you could score in that line
                break
    return points  # 0, 1 or 2


def edgeHills(g, r, c):
    rmax, cmax = g.shape
    rmax -= 1
    cmax -= 1
    
    # Coordinates of two extreme points
    kmax = min(r, cmax-c)  # max number of steps in that direction
    upper_right = (r-kmax, c+kmax)
    kmax = min(rmax-r, c)
    bottom_left = (r+kmax, c-kmax)
    return upper_right, bottom_left

def test_edgeHills():
    s = 9
    test = np.zeros((s, s))
    row = {0, s-1}
    col = {0, s-1}
    for i in range(s):
        for j in range(s):
            ur, bl = edgeHills(test, i, j)
            assert ur[0] in row or ur[1] in col
            assert bl[0] in row or bl[1] in col
            assert ur[0]+ur[1] == i+j == bl[0]+bl[1]
    return True
test_edgeHills()

def edgeDales(g, r, c):
    rmax, cmax = g.shape
    rmax -= 1
    cmax -= 1
    
    # Coordinates of two extreme points
    kmax = min(rmax-r, cmax-c)
    bottom_right = (r+kmax, c+kmax)
    kmax = min(r, c)
    upper_left = (r-kmax, c-kmax)
    return bottom_right, upper_left

def test_edgeDales():
    s = 9
    test = np.zeros((s, s))
    row = {0, s-1}
    col = {0, s-1}
    for i in range(s):
        for j in range(s):
            ur, bl = edgeDales(test, i, j)
            assert ur[0] in row or ur[1] in col
            assert bl[0] in row or bl[1] in col
            assert ur[0]-ur[1] == i-j == bl[0]-bl[1]
    return True
test_edgeDales()

def check_hills(g, r, c, p, max_l):
    points = 0
    ur, bl = edgeHills(g, r, c)
    # Isolate what we will be working on
    hill_length = (bl[0] - ur[0]) + 1  # |slope| = 1 so we can use any distance
    coord = [(bl[0]-k, bl[1]+k) for k in range(hill_length)]  # All coords to select
    if len(coord) < max_l: return points  # No way we can score
    hill = [g[k] for k in coord]  # Back to simple 1D format
    s = sum(hill[:max_l-1]) + hill[0]  # Double first elem and rm last
    
    for k in range(hill_length - (max_l-1)):  # Iterate on distance
        s += hill[max_l-1+k]  # Add head
        s -= hill[k]  # Remove tail
        if s in p.win:
            # Reassign values to g
            cwin = coord[k:max_l+k]
            #print('Hill win on:', cwin)
            for c_ in cwin:
                g[c_] = p.pc2
            # g[k:max_l+k, c] = p.pc2  # Assigns values
            #p['score'] += 1
            points += 1
            if k == 0 and hill_length == 2*max_l-1:  # CORNER CASE: what if you joined a whole line?
                s = sum(hill[max_l-1:])
                if s in p.win:  # Only works if all other symbols are not assigned
                    cwin = coord[max_l-1:]
                    for c_ in cwin:
                        g[c_] = p.pc2
                    #p['score'] += 1
                    points += 1
                    break
            else:  # That is the only point you could score in that line
                break
    
    return points  # 0, 1 or 2

def check_dales(g, r, c, p, max_l):
    points = 0
    br, ul = edgeDales(g, r, c)
    # Isolate what we will be working on
    dale_length = (br[1] - ul[1]) + 1  # |slope| = 1 so we can use any distance
    coord = [(ul[0]+k, ul[1]+k) for k in range(dale_length)]  # All coords to select
    if len(coord) < max_l: return points  # No way we can score - abort
    dale = [g[k] for k in coord]  # Back to simple 1D format
    s = sum(dale[:max_l-1]) + dale[0]  # Double first elem and rm last
    
    for k in range(dale_length - (max_l-1)):  # Iterate on distance
        s += dale[max_l-1+k]  # Add head
        s -= dale[k]  # Remove tail
        if s in p.win:
            # Reassign values to g
            cwin = [coord[max_l-1+k-j] for j in range(max_l)]
            #print('Dale win on:', cwin)
            for c_ in cwin:
                g[c_] = p.pc2
            # g[k:max_l+k, c] = p.pc2  # Assigns values
            #p['score'] += 1
            points += 1
            if k == 0 and dale_length == 2*max_l-1:  # CORNER CASE: what if you joined a whole line?
                s = sum(dale[max_l-1:])
                
                if s in p.win:  # Only works if all other symbols are not assigned
                    cwin = coord[max_l-1:]
                    for c_ in cwin:
                        g[c_] = p.pc2
                    #p['score'] += 1
                    points += 1
                    break
            else:  # That is the only point you could score in that line
                break
    return points

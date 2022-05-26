import math
C = 1.0
inBoard = lambda x,y: not (x>7 or x<0 or y>7 or y<0)#判断是否在棋盘里

normalizeFunction = lambda x:1 if x>0 else(0 if x==0 else -1)

score = lambda xChild,nParent,nChild:xChild+C*math.sqrt(math.log(nParent)/nChild)

winner = lambda state:normalizeFunction(state.sum())#根据state判断谁是赢家，-1表示黑赢，0表示平局，1表示白赢

def check(state,turn,x,y):#在state棋盘下 当前turn(-1黑1白)方下子 判断(x,y)是否可以下 返回下在该处造成的翻转列表
    lis = [(1,1),(1,0),(1,-1),(0,1),(0,-1),(-1,1),(-1,0),(-1,-1)]
    stacklis = []
    for i,j in lis:
        if (not inBoard(x+i,y+j)) or state[x+i][y+j]==0:
            continue
        m = 1
        flag = False
        while inBoard(x+m*i,y+m*j) and state[x+m*i][y+m*j]!=0:
            templis = []
            if state[x+m*i][y+m*j] == -1*turn:
                templis.append((x+m*i,y+m*j))
            else:
                flag = True
            m += 1
        if flag == True:
            stacklis.extend(templis)
    return stacklis

def updateEdge(state,edge,x,y):#在state棋盘下 在(x,y)下子，原地更新边缘
    edge.remove((x,y))
    lis = [(1,1),(1,0),(1,-1),(0,1),(0,-1),(-1,1),(-1,0),(-1,-1)]
    for i,j in lis:
        if inBoard(x+i,y+j) and state[x+i][y+j] == 0:
            edge.add((x+i,y+j))

def findCanGoes(state,edge,turn):#在state棋盘下,根据edge和turn获得可以落子的坐标 返回{落子坐标：在落子坐标处落子造成的翻转列表}
    canGoes = dict()
    for x,y in edge:
        a = check(state,turn,x,y)
        if a != []:
            canGoes[(x,y)] = a
    return canGoes

def updateState(state,canGoes,x,y,turn):#在state棋盘下，根据turn选择的x，y和计算出的canGoes原地更新state,返回turn单步相对于上个state多的子数
    state[x][y] = turn
    for m,n in canGoes[(x,y)]:
        state[m][n] = -1*state[m][n]
    return len(canGoes[(x,y)])

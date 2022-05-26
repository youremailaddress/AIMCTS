import numpy as np
from multiprocessing import Pool
import itertools
import math,random
from copy import deepcopy
from functools import partial

class Board:
    def __init__(self):
        self._board = np.zeros((8,8),dtype=np.int8)
        self._board[3][4],self._board[4][3] = -1,-1 # 黑棋为 -1
        self._board[3][3],self._board[4][4] = 1,1  # 白棋为 1
    
    def count(self,color):
        '''
        @description: 统计 color 一方棋子的数量。 color[-1,0,1]
        @return 返回 color 棋子在棋盘上的总数
        '''        
        return np.sum(self._board == color)
    
    def get_winner(self):
        '''
         @description: 判断黑棋和白棋的输赢，通过棋子的个数进行判断
         @return {*} 哪种棋赢了，赢子个数
        '''        
        diff = np.sum(self._board)
        if diff > 0:
            return 1,diff
        elif diff == 0:
            return 0,0
        else:
            return -1,-diff
    
    def _move(self,action,color):
        '''
         @description: 落子并获取反转棋子的坐标 action: 落子的坐标 color:当前落子人的颜色
         @return 返回反转棋子的坐标列表，落子失败则返回False
        '''        
        filp = self._can_filped(action, color)
        if filp == []:
            return False
        for item in filp:
            self._board[item[0]][item[1]] = color
        self._board[action[0]][action[1]] = color

    def is_on_board(self,x,y):
        '''
         @description: 是否出界
         @return {*}
        '''        
        return x >= 0 and x <= 7 and y >= 0 and y <= 7

    def _can_filped(self,action,color):
        x,y = action
        if not self.is_on_board(x, y) or self._board[x][y] != 0:
            return False
        self._board[x][y] = color
        op_color = -1*color
        flipped_pos = []
        d = [[0, 1], [1, 1], [1, 0], [1, -1],
             [0, -1], [-1, -1], [-1, 0], [-1, 1]]
        for xdirection, ydirection in d:
            m,n = x,y
            m += xdirection
            n += ydirection
            if self.is_on_board(m, n) and self._board[m][n] == op_color:
                m += xdirection
                n += ydirection
                if not self.is_on_board(m, n):
                    continue
                while self._board[m][n] == op_color:
                    m += xdirection
                    n += ydirection
                    if not self.is_on_board(m, n):
                        break
                if not self.is_on_board(m, n):
                    break
                if self._board[m][n] == color:
                    while True:
                        m -= xdirection
                        n -= ydirection
                        if m == x and n == y:
                            break
                        flipped_pos.append([m,n])
        self._board[x][y] = 0
        if len(flipped_pos) == 0:
            return False
        return flipped_pos

    def backpropagation(self,action,filpped_pos,color):
        '''
         @description: 回溯
         @return {*}
        '''        
        self._board[action[0]][action[1]] = 0
        op_color = -1*color
        for p in filpped_pos:
            self._board[p[0]][p[1]] = op_color
    
    def _can_filp(self,color,action):
        if not self._can_filped(action, color):
            return False
        else:
            return action

    def get_legal_actions(self,color):
        d = [[0, 1], [1, 1], [1, 0], [1, -1],
             [0, -1], [-1, -1], [-1, 0], [-1, 1]]
        op_color = -1*color
        op_color_near_points = []
        board = self._board
        for i in range(8):
            for j in range(8):
                if board[i][j] == op_color:
                    for dx, dy in d:
                        x, y = i + dx, j + dy
                        if 0 <= x <= 7 and 0 <= y <= 7 and board[x][y] == 0 and (
                                x, y) not in op_color_near_points:
                            op_color_near_points.append((x, y))
        retlis =  [p for p in op_color_near_points if self._can_filped(p, color)]
        return retlis

class Node:
    def __init__(self,parent,board,action,color):
        self.board = board
        self.preaction = action #父节点到该节点的action
        self.color = color
        self.N = 0 #访问到的总次数
        self.parent = parent
        self.children = []
        self.unvisitedQueue = board.get_legal_actions(color)
        if self.unvisitedQueue ==  [] and self.Isend(board) == False:
            self.unvisitedQueue = [None]
        self.reward = [0,0] # index 0,1 -> -1,1 -> 黑,白
        self.maxVal = [0,0]

    def maxValue(self,color):
        self.maxVal[(color+1)//2] = self.reward[(color+1)//2] /self.N + math.sqrt(2*math.log(self.parent.N)/self.N)

    def Isend(self,board):
        return len(board.get_legal_actions(1))==0 and len(board.get_legal_actions(-1))==0

class MTCSTree():
    def __init__(self):
        pass

    def choice(self,node):
        returnNode = node
        while not returnNode.Isend(returnNode.board):
            if len(returnNode.unvisitedQueue)>0:
                return self.expand(returnNode)
            else:
                returnNode = self.maxValue(returnNode,returnNode.color)
        return returnNode

    def maxValue(self,node,color):
        for childnode in node.children:
            childnode.maxValue(color)
        return sorted(node.children,key=lambda x:x.maxVal[(color+1)//2],reverse=True)[0]

    def expand(self,node):
        nextBoard = deepcopy(node.board)
        action = random.choice(node.unvisitedQueue)
        node.unvisitedQueue.remove(action)
        if action != None:
            nextBoard._move(action,node.color)
        op_color = -1*node.color
        nextNode = Node(node, nextBoard, action, op_color)
        node.children.append(nextNode)
        return nextNode

    def simulate(self,board,color):
        nextBoard = deepcopy(board)
        op_color = color
        
        while not (nextBoard.get_legal_actions(-1) == [] and nextBoard.get_legal_actions(1) == []):
            actions = nextBoard.get_legal_actions(op_color)
            if actions == []:
                action = None
            else:
                action = random.choice(actions)
            if action == None:
                op_color = -1*op_color
                continue
            else:
                nextBoard._move(action,op_color)
                op_color = -1*op_color
        winner,diff = nextBoard.get_winner()
        return winner,diff/64

    def backup(self, node, reward):
        prevNode = node
        while prevNode is not None:
            prevNode.N += 1
            if reward[0] == -1:
                prevNode.reward[0] += reward[1]
                prevNode.reward[1] -= reward[1]
            if reward[0] == 1:
                prevNode.reward[0] -= reward[1]
                prevNode.reward[0] += reward[1]
            prevNode = prevNode.parent

    def decision(self,board,color,times):
        actions = board.get_legal_actions(color)
        if len(actions) == 1:
            return actions[0]
        nextboard = deepcopy(board)
        root = Node(None,nextboard,None,color)
        while times:
            expand = self.choice(root)
            reward = self.simulate(expand.board, expand.color)
            self.backup(expand, reward)
            times -= 1
        return self.maxValue(root, color).preaction

    
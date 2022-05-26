import math
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from itertools import chain
from datetime import datetime
import pytz,sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log_filepath = sys.argv[1]

def log(level,message):
    try:
        f = open(log_filepath,"a+")
    except:
        print("Fail to open/write .log file!")
        exit(0)
    if level == 1:
        f.write("["+str(datetime.now(pytz.timezone('Asia/Shanghai')))+"] [Info]:"+message+'\n')
        f.close()
    if level == 2:
        f.write("["+str(datetime.now(pytz.timezone('Asia/Shanghai')))+"] [Warning]:"+message+'\n')
        f.close()
    if level == 3:
        f.write("["+str(datetime.now(pytz.timezone('Asia/Shanghai')))+"] [Error]:"+message+'\n')
        f.close()
        exit(-1)

class TrainSet(Dataset):
    def __init__(self, X, Y):
        # 定义好 image 的路径
        self.X, self.Y = X, Y

    def __getitem__(self, index):
        return self.X[index], self.Y[index].flatten()

    def __len__(self):
        return len(self.X)

class Node():
    def __init__(self,parent,state,action,selfnum = 0):
        self.S = state
        self.action = action #父节点到该节点的action
        self.P = 0 #选择的概率
        self.Q = 0 #访问到的平均value
        self.W = 0 #访问到的总value
        self.N = 0 #访问到的总次数
        self.UCT = 65537
        self.parent = parent
        self.child_action_list = []
        self.selfnum = selfnum

    def Update(self,value):#回溯更新沿途节点
        self.N += 1
        self.W += value
        self.Q = self.W/self.N
        if self.parent != None:
            self.UCT = self.Q + self.P + math.sqrt(2*math.log(self.parent.N)/self.N)

    def isAllowedAction(self,xx,yy):#判断当前棋盘下xx,yy是不是可行点?
        d = [[0, 1], [1, 1], [1, 0], [1, -1],
             [0, -1], [-1, -1], [-1, 0], [-1, 1]]
        for dx, dy in d:
            x, y = xx, yy
            x += dx
            y += dy
            if x+dx <= 7 and x+dx >= 0 and y+dy <= 7 and y+dy >= 0 and self.S[x][y] == -1:
                x += dx
                y += dy
                while x+dx <= 7 and x+dx >= 0 and y+dy <= 7 and y+dy >= 0 and self.S[x][y] == -1:
                    x += dx
                    y += dy
                if x <= 7 and x >= 0 and y <= 7 and y >= 0 and self.S[x][y] == 1:
                    return True
        return False

    def GetAllAllowedAction(self):#获取所有可以走的点
        validMoves = []
        for x in range(8):
            for y in range(8):
                if self.S[x][y] == 0 and self.isAllowedAction(x, y):
                    validMoves.append([x, y])
        return validMoves

    def GetFlipedState(self,temp,action):
        mm,nn = action
        temp[mm][nn] = 1
        d = [[0, 1], [1, 1], [1, 0], [1, -1],
             [0, -1], [-1, -1], [-1, 0], [-1, 1]]
        for x,y in d:
            flag = 0
            for count in range(1,8):
                if mm+count*x>7 or mm+count*x <0 or nn+count*y>7 or nn+count*y<0:
                    continue
                if temp[mm+count*x][nn+count*y] == 1:
                    flag = count
                    break
            while flag > 0:
                temp[mm+flag*x][nn+flag*y] = 1
                flag -= 1
        return temp

    def GetNextState(self,action):#从action获取状态
        if action == None:
            return -copy.deepcopy(self.S)
        temp = copy.deepcopy(self.S)
        temp = self.GetFlipedState(temp,action)
        return -temp

    def Expand(self):#扩展结点
        allActions = self.GetAllAllowedAction()
        count = 0
        if allActions == []:
            childnode = Node(self,self.GetNextState(None),None,selfnum=count)
            self.child_action_list.append((childnode,None))
            return
        for action in allActions:
            childnode = Node(self,self.GetNextState(action),action,selfnum=count)
            count += 1
            self.child_action_list.append((childnode,action))

    def Differaction(self,child):#从父节点怎么到子节点的
        return child.action
        # templist = self.S + child.S
        # if np.transpose(np.nonzero(templist)) == []:
        #     return None
        # return np.transpose(np.nonzero(templist))[0]

    def ModifyP(self,pmatrix):#给子节点分配p
        actionlist = []
        plist = []
        if self.parent == None:
            self.P = 1
        for child,action in self.child_action_list:
            actionlist.append(action)
        for action in actionlist:
            if action == None:
                self.child_action_list[0][0].P = 1
                return
            plist.append(pmatrix[action[0]][action[1]])
        # plist = self.NormalizeP(plist)
        for i in range(len(self.child_action_list)):
            self.child_action_list[i][0].P = plist[i]
        return
    
    # def NormalizeP(self,plist):#按照需求修改标准化p
    #     pass

    def Isleaf(self):#判断是不是叶子
        if self.child_action_list == []:
            return True
        else:
            return False

    def Chooseleaf(self):#选择叶子
        childlist = [i[0] for i in self.child_action_list]
        uctlist = [child.UCT for child in childlist]
        return childlist[uctlist.index(max(uctlist))]

    def Isend(self):
        if self.parent == None:
            return False
        if self.action == None and self.parent.action == None:
            return True
        else:
            return False

class MTCSTree():
    def __init__(self,root,model,player,times=800):
        self.root = root
        self.model = model
        self.times = times
        self.player = player
        self.child = self.NextChild()

    def winorlose(self,p):
        value = sum(map(sum,p.S))
        if value > 0:
            return 1
        elif value == 0:
            return 0
        else:
            return -1

    def SingleApproach(self):
        p = self.root
        pathlist = []
        while not p.Isleaf():
            pathlist.append(p)
            p = p.Chooseleaf()
        if not p.Isend():
            p.Expand()
            pmatrix,value = self.model.NN(p.S)
            p.ModifyP(pmatrix)
            p.Update(value)
        else:
            value = self.winorlose(p)
            p.Update(value)
        while len(pathlist):
            pathlist.pop().Update(value)

    def NextAction(self):
        return self.root.Differaction(self.child)

    def NextState(self):
        return self.child.state

    def NextChild(self,typ = 2):
        for i in range(self.times):
            self.SingleApproach()
        nlist = [child.N for child,action in self.root.child_action_list]
        childlist = [child for child,action in self.root.child_action_list]
        if typ == 1:
            maxofN = max(nlist)
            return childlist[nlist.index(maxofN)]
        else:
            normalizeN = [i/sum(nlist) for i in nlist]
            if childlist == []:
                return None
            return np.random.choice(childlist,1,p=normalizeN)[0]
    
    def DelNodes(self,parents,exception):
        childlist = [child for child,action in parents.child_action_list]
        del parents
        if childlist == []:
            return
        if exception != None:
            childlist.remove(exception)
        for child in childlist:
            self.DelNodes(child,None)

    def UpdateTree(self):#上一次走棋的结尾，修改root，删减树
        p = self.root
        self.root = self.child
        self.DelNodes(p,self.root)
        # self.child = self.NextChild()

    def UpdateActionTree(self,action):#下一次走棋的开始->计算root位置
        self.UpdateTree()
        for child,act in self.root.child_action_list:
            # print(child.action,action)
            if act == action:
                p = self.root
                self.root = child
                self.DelNodes(p,self.root)
                self.child = self.NextChild()

class PlayGames():
    def __init__(self,model_white,model_black,times=100):
        self.model_white = model_white
        self.model_black = model_black
        self.preisbetter = 0
        startstate = np.array([[0]*8]*8)
        startstate[3][3] = startstate[4][4] = 1
        startstate[3][4] = startstate[4][3] = -1
        root = Node(None,startstate,None)
        self.allstatus = []
        for i in range(times):
            self.player_white = MTCSTree(root,self.model_white,0)
            self.player_black = None
            self.selfplay()
            log(1,"battling;times:"+str(i))

    def selfplay(self):
        case = 0
        whitestatestatus = []
        blackstatestatus = []
        whitestate = self.player_white.root.S
        child = self.player_white.child#白走的下一步
        action = child.action#白走的位置
        if action!=None:
            whitestatestatus.append([whitestate,action,0])
        state = self.player_white.root.GetNextState(action)#对黑来说state
        root_1 = Node(None,state,None)
        self.player_black = MTCSTree(root_1,self.model_black,1)#生成黑树
        while True:
            blackstate = self.player_black.root.S
            child = self.player_black.child#黑下一步
            action = child.action#黑位置
            if action != None:
                blackstatestatus.append([blackstate,action,0])
            # print(blackstatestatus[-1])
            self.player_white.UpdateActionTree(action)#白更新root
            if self.player_white.root.Isend():
                case = 1
                break
            whitestate = self.player_white.root.S
            child = self.player_white.child#白下一步
            action = child.action#白位置
            self.player_black.UpdateActionTree(action)#黑更新root
            if action != None:
                whitestatestatus.append([whitestate,action,0])
            # print(whitestatestatus[-1])
            if self.player_black.root.Isend():
                case = -1
                break
        final = self.player_white.root.S
        score = sum(map(sum,final))
        if score > 0:
            for i in range(len(whitestatestatus)):
                whitestatestatus[i][2] = case
            for i in range(len(blackstatestatus)):
                blackstatestatus[i][2] = -case
            self.preisbetter += case
        elif score == 0:
            for i in range(len(whitestatestatus)):
                whitestatestatus[i][2] = 0
            for i in range(len(blackstatestatus)):
                blackstatestatus[i][2] = 0
        else:
            for i in range(len(whitestatestatus)):
                whitestatestatus[i][2] = -case
            for i in range(len(blackstatestatus)):
                blackstatestatus[i][2] = case
            self.preisbetter -= case
        self.allstatus.extend(whitestatestatus)
        self.allstatus.extend(blackstatestatus)

    def gameend(self):
        if self.player_white.root.Isend() and self.player_black.root.Isend():
            return True
        else:
            return False

    def outboundData(self,state,action,winp):
        state1 = state[::-1]
        action1 = [7-action[0],action[1]]
        self.allstatus.append((state1,action1,winp))
        state2 = list(map(list,zip(*state)))
        action2 = [action[1],action[0]]
        self.allstatus.append((state2,action2,winp))
        state3 = list(map(list,zip(*state)))[::-1]
        action3 = [7-action[1],action[0]]
        self.allstatus.append((state3,action3,winp))
        state4 = list(map(list,zip(*state[::-1])))
        action4 = [action[1],7- action[0]]
        self.allstatus.append((state4,action4,winp))
        state5 = list(map(list,zip(*state[::-1])))[::-1]
        action5 =  [7-action[1],7-action[0]]
        self.allstatus.append((state5,action5,winp))
        state6 = list(map(list,zip(*state2[::-1])))
        action6 = [action[0],7 - action[1]]
        self.allstatus.append((state6,action6,winp))
        state7 = state6[::-1]
        action7 = [7-action[0],7-action[1]]
        self.allstatus.append((state7,action7,winp))

    def processData(self):
        newlist = []
        for state,action,winp in copy.deepcopy(self.allstatus):
            self.outboundData(state,action,winp)#扩展
        statelis = [state for state,action,winp in self.allstatus]#获得statelist
        statelis = np.array(list(set([tuple(tuple(i) for i in x) for x in statelis])))
        statelist = [statelis[i] for i in range(len(statelis))]
        actionlist = np.array([[[0]*8]*8]*len(statelist))#对应去重后的动作矩阵列表
        winplist = [0]*len(statelist)#对应是否获胜列表
        Nlist = [0]*len(statelist)#记录不同state总数
        for state,action,winp in self.allstatus:
            for k,v in enumerate(statelist):
                if (v==state).all():
                    index = k
                    actionlist[index][action[0]][action[1]] += 1 #获得对应state对应位置的矩阵值+1
                    winplist[index] += winp #获得对应state的获胜率
                    Nlist[index] += 1 #对应state总数+1
        for i in range(len(statelist)):
            newlist.append(list(chain(*actionlist[i])))
            newlist[i].append(winplist[i])
        newlist = [np.array(np.array(newlist[i])/Nlist[i]) for i in range(len(statelist))] #除以总数获得率
        
        return torch.from_numpy(np.array(statelist)),torch.from_numpy(np.array(newlist))
        
class Model(nn.Module):
    def __init__(self) -> None:
        super(Model,self).__init__()
        # 卷积层1
        self.conv1 = nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        # 卷积层2
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
        self.fc1 = nn.Linear(64*2*2,200)
        self.fc2 = nn.Linear(200,100)
        self.fc3 = nn.Linear(100,65)

    # 前向传播的过程
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 2* 2)           #将数据平整为一维的 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = F.log_softmax(x,dim=1) NLLLoss()才需要，交叉熵不需要
        return x

class CNN():
    # 训练过程
    train_accs = []             # 训练准确率
    train_loss = []             # 训练损失率
    test_accs = []              # 测试准确率

    def __init__(self,lr=0.001, epochs=15,PATH="./alphazero_net.pth"):
        self.model = Model()
        self.criterion = nn.MultiLabelSoftMarginLoss()      # 交叉熵
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9) # 随机梯度下降优化器
        #也可以选择Adam优化方法
        # self.optimizer = torch.optim.Adam(net.parameters(),lr=1e-2)
        # self.dp = nn.Dropout(p=0.5)
        self.epochs = epochs
        self.PATH=PATH

    # 训练拟合
    def fit(self,train_loader,path1=None,path2=None):
        '''
        path1:原来model地址
        path2:要保存的model地址
        '''
        net = self.model
        if path1 != None:
            net.load_state_dict(torch.load(path1))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = net.to(device)
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i,data in enumerate(train_loader,0):#0是下标起始位置默认为0    
                # inputs,labels = data
                print(i,data)
                inputs,labels = data[0].to(device), data[1].to(device)
                #初始为0，清除上个batch的梯度信息
                self.optimizer.zero_grad()
                
                #前向+后向+优化     
                outputs = net(inputs)
                loss = self.criterion(outputs,labels)
                loss.backward()
                self.optimizer.step()
                
                # loss 的输出，每个一百个batch输出，平均的loss
                running_loss += loss.item()
                if i%100 == 99:
                    log(1,'[%d,%5d] loss :%.3f' %
                        (epoch+1,i+1,running_loss/100))
                    running_loss = 0.0
                self.train_loss.append(loss.item())
        torch.save(net.state_dict(), path2)
    
    def NN(self,state):
        self.model.load_state_dict(torch.load(self.PATH))
        self.model.eval()
        tenso = torch.from_numpy(state)
        tenso = Variable(torch.unsqueeze(tenso, dim=0).float(), requires_grad=False)
        tenso = Variable(torch.unsqueeze(tenso, dim=0).float(), requires_grad=False)
        self.model.to(device)
        tenso = tenso.to(device)
        predicted = self.model.forward(tenso)
        predicted = predicted.view(1,65).detach().cpu().numpy()[0]
        plist = predicted[:-1]
        mini = min(plist)
        plist = [i+mini for i in plist]
        value = predicted[-1]
        xishu = sum(plist)
        plist = plist/xishu
        plist.resize((8,8))
        value = value/xishu
        return plist,value

    def save(self,path):
        net = self.model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = net.to(device)
        torch.save(net.state_dict(), path)

    def load(self):
        net = self.model
        net.load_state_dict(torch.load(self.PATH))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = net.to(device)


if __name__ == "__main__":
    cnn_alphazero = CNN(PATH="./alphazero_net.pth")
    cnn_temp = CNN(PATH="./temp.pth")
    cnn_alphazero.save("./alphazero_net.pth")
    while True:
        a = PlayGames(cnn_alphazero,cnn_alphazero,times=20)#自己跟自己对弈
        log(1,"processing data.")
        x,y = a.processData()#获得每步数据
        x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False)
        x = np.transpose(x,(1,0,2,3))
        train_data = TrainSet(x,y)#生成训练数据集
        train_loader = torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True,num_workers=2)
        log(1,"train model.")
        cnn_alphazero.fit(train_loader,path1="./alphazero_net.pth",path2="./temp.pth")#利用训练集训练数据,暂存新模型
        cnn_temp.load()
        b = PlayGames(cnn_alphazero,cnn_temp,times=10)
        log(1,"complete a circle!")
        if b.preisbetter < -2:
            cnn_temp.save("./alphazero_net.pth")
            log(1,"New best model!")
    # except Exception as e:
    #     print(str(e))
        # log(3,str(e))
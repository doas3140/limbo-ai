from collections import deque
import os

class LivePlot():
    def __init__(self,MAXLEN,N):
        self.dir = os.path.join(os.getcwd(),'liveplot')
        with open(os.path.join(self.dir,'data.txt'),'w'):
            pass # delete content
        self.N = N
        self.counter = 0 # for N
        self.x = deque(maxlen=int(MAXLEN/N))
        self.y = deque(maxlen=int(MAXLEN/N))
        self.x.append(0)
        self.y.append(0)

    def emit(self,y):
        self.counter += 1
        if self.counter > self.N-1:
            self.x.append(self.x[-1] + self.N)
            self.y.append(y)
            with open(os.path.join(self.dir,'data.txt'),'w') as f:
                for x,y in zip(self.x,self.y):
                    f.write(str(x)+','+str(y)+'\n')
            self.counter = 0
import time
import numpy as np
import termplotlib as tpl
import os

def showProgress(agent, x, y, y2, meanOfN):
    os.system('clear')
    print('+-------------------------------------+')
    agent.printName()
    print('+-------------------------------------+')
    agent.printParameters()
    print('+-------------------------------------+')
    print('+ Episode ' + str(len(x)) + '              score: '+ str(y[len(y)-1]))
    print('+ Mean of last ' + str(meanOfN) + ' = ' + str(meanOfLast(x,y, meanOfN)) + '   Highest Score: ' + str(np.max(y)))
    print('+-------------------------------------+')
    fig = tpl.figure()
    fig.plot(x, y, width=100, height=30)
    fig.show()
    fig = tpl.figure()
    fig.plot(x, y2, width=100, height=30)
    fig.show()

def meanOfLast(x, y, meanOfN):
     divider = meanOfN if meanOfN < len(x) else len(x)
     return np.sum(y[-divider:]) / divider
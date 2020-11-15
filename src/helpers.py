import time
import numpy as np
import termplotlib as tpl
import os

def showProgress(agent, x, y, meanOfN):
    os.system('clear')
    print('+-------------------------------------+')
    agent.printName()
    print('+-------------------------------------+')
    agent.printParameters()
    print('+-------------------------------------+')
    print('+ Episode ' + str(len(x)) + '              score: '+ str(y[len(y)-1]))
    divider = meanOfN if meanOfN < len(x) else len(x)
    print('+ Mean of last ' + str(meanOfN) + ' = ' + str(np.sum(y[-divider:]) / divider) + '   Highest Score: ' + str(np.max(y)))
    print('+-------------------------------------+')
    fig = tpl.figure()
    fig.plot(x, y, width=100, height=30)
    fig.show()
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import sys
from pylab import *
import os

def main():

    if len(sys.argv) > 1:
        interval = sys.argv[1]
    else:
        interval = 100

    if len(sys.argv) > 5:
        x = int(sys.argv[2])
        y = int(sys.argv[3])
        sizex = int(sys.argv[4])
        sizey = int(sys.argv[5])
    else:
        x = 1025
        y = 0
        sizex = 610
        sizey = 610

    style.use('fivethirtyeight')

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)

    def animate(i):
        graph_data = open(os.path.join(os.getcwd(),'data.txt'),'r').read()
        lines = graph_data.split('\n')
        xs = []
        ys = []
        for line in lines:
            if len(line) > 1:
                x, y = line.split(',')
                xs.append(float(x))
                ys.append(float(y))
        ax1.clear()
        ax1.set_ylim([-500, 500])
        ax1.plot(xs, ys)

    thismanager = get_current_fig_manager()
    # thismanager.window.SetPosition((x,y))
    thismanager.window.setGeometry(x,y,sizex,sizey)

    ani = animation.FuncAnimation(fig, animate, interval=interval)

    plt.show()

if __name__ == '__main__':
    main()
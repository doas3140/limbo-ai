import win32gui
import numpy as np

ORIGINAL = np.array([1920,1080])
WIN_GUI = np.array([1535,863])

N = ORIGINAL/WIN_GUI

while True:
    flags, hcursor, (x,y) = win32gui.GetCursorInfo()
    x,y = int(x*N[0]),int(y*N[1])
    string = 'x - {:>5} , y - {:>5}'.format(x,y)
    print(string,end='\r')
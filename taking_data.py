import pyautogui
import numpy as np
import matplotlib.pyplot as plt

data_x = []
data_y = []

drawing = True

plt.axis('equal')

while drawing:

    a = str(input())

    pos = pyautogui.position()
    x,y = pos.x, -pos.y
    if a == 'a':
        drawing = False
    elif a == 'b':
        data_x.pop()
        data_y.pop()
        pass
    elif a == 's':
        plt.plot(data_x,data_y,'.')
        plt.show()
    else:
        print(x,y)
        data_x.append(x), data_y.append(y)

data = np.array([data_x, data_y])
np.savetxt('pearl2.txt', data)
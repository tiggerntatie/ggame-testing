from ggame.point import Point
from ggame.mathapp import MathApp

x = y = 0.0
d = 0.01

def getxy():
    global x
    global y
    global d
    x += d
    if x > 1 or x < 0:
        d = d*-1
    return (x, y)

p = Point(getxy)

a = MathApp()
a.run()
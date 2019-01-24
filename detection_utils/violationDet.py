import numpy as np

"""
This Class checks whether the vehicle has crossed the middle road line 
"""
def calculate_line_equation(x1, x2, y1, y2):
    m = (y1 - y2) / (x1 - x2)
    b = (x1 * y1 - x2 * y1) / (x1 - x2) #y - intercept
    #y = m*x + b
    line = (m, b)
    return line

def lineFromPoints(x1 , x2 , y1, y2):
    m = (y1 - y2) * 1.0 / (x1 - x2)
    c =  y1 - x1 * m
    line = (m, c)
    return line

def intercept(m1, b1 ,m2, b2):
    A = np.matrix([[1, -m1], [1, -m2]])
    B = np.matrix([[b1],[b2]])
    A_inv = np.linalg.inv(A)
    x = A_inv * B
    return x

def is_violation(b1x1, b1x2, b1y1, b1y2, b2x1, b2y1, b2x2, b2y2, label):
    roadLineEq = lineFromPoints(b2x1, b2x2, b2y1, b2y2)
    boxLineEq = lineFromPoints(b1x1, b1x2, b1y2, b1y2)
    if not(roadLineEq[0] == 0 and boxLineEq[0] == 0):
        newIntercept = intercept(roadLineEq[0], roadLineEq[1], boxLineEq[0], boxLineEq[1])
        boxLineXDifference = abs(b1x2 - b1x1)
        if 'Front' in label:
            vialationX = b1x1 + (boxLineXDifference * 5 / 16)
            #vialationY = (boxLineEq[0] * vialationX) + boxLineEq[1]
            if (vialationX < newIntercept[1]):
                return True
            else:
                return False
        elif 'Back' in label:
            vialationX = b1x2 - (boxLineXDifference * 5 / 16)
            if (vialationX > newIntercept[1]):
                return True
            else:
                return False
    else:
        return False
#print(is_violation(100, 300, 300, 305, 150, 50, 240, 300, True))









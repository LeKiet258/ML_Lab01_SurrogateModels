# Define some metric functions here

def MSE(y, y2):
    m = len(y)
    return sum([sum([(y2[i] - y[i]) ** 2]) for i in range(m)])[0] / m

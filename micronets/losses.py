def mean_squared_error(y, ypred):
    cost = sum((ygrt - ypred)**2 for ypred, ygrt in zip(ypred, y))
    return cost

def binary_cross_entropy(y, ypred):
    epsilon = 1e-15  # Small value to avoid log(0)
    loss = [-(yi * (ypredi + epsilon).log() + (1 - yi) * ((1 - ypredi).log())) for yi, ypredi in zip(y, ypred)]
    cost = sum(loss)/len(loss)
    return cost
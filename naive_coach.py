def naive_coach(theta):
    if theta > 0.1:
        action = 1
    elif theta < 0.1:
        action = 2
    else:
        action = 0
    return action
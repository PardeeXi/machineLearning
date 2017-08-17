# assignment 6 Xi Pardee
import numpy as np
import matplotlib.pyplot as plot
import threading
import time

learnRate = 0.9
discount = 0.9
M = 200               # steps
N = 4000            # episodes
actions = np.array(["n", "s", "e", "w", "c"])
qmatrix = np.zeros((243, 5))
score = 0

size = 10
robot = [0, 0]
canBoard = np.zeros((10, 10))

# set up the game
def setUpGame():
    global robot, canBoard, score
    robot = np.random.randint(size, size=2)
    indexC = np.random.choice([True, False], (10, 10), p=[0.5, 0.5])
    canBoard = np.zeros((10, 10))
    canBoard[indexC] = 1.0
    score = 0

# choose an action
def chooseAction(epsilon):
    global robot, canBoard, qmatrix
    [i, j] = robot
    stateNumber = calculateState(i, j)
    maxV = np.amax(qmatrix[stateNumber])
    indexs = np.where(qmatrix[stateNumber]==maxV)[0]
    indexH = np.random.choice(indexs)
    #indexH = np.argmax(qmatrix[stateNumber])
    probility = np.full((5), epsilon/5)
    probility[indexH] += 1 - epsilon
    return np.random.choice(actions, p=probility), stateNumber

# calculate the state
def calculateState(i, j):
    stateNumber = 0
    stateNumber += calculateStateNumber(i, j, 4)    # Here
    stateNumber += calculateStateNumber(i-1, j, 3)  # North
    stateNumber += calculateStateNumber(i+1, j, 2)  # South
    stateNumber += calculateStateNumber(i, j+1, 1)  # East
    stateNumber += calculateStateNumber(i, j-1, 0)  # West
    return stateNumber

# calculate the state number
def calculateStateNumber(x, y, k):
    global canBoard
    if x < 0 or y < 0 or x > 9 or y > 9:
        return np.power(3, k) * 2
    elif canBoard[x, y] == 1:
        return np.power(3, k) * 1
    else:
        return 0

def updateQ(action, oldStatenumber, reward):
    global qmatrix, actions, discount
    newStateNumber = calculateState(robot[0], robot[1])
    maxQ = np.amax(qmatrix[newStateNumber])
    index = np.where( actions == action )[0][0]
    update = learnRate * (float(reward) + discount * maxQ - qmatrix[oldStatenumber][index])
    qmatrix[oldStatenumber][index] += update

def doAction(action):
    # print (action)
    global robot, canBoard, score
    # print (robot)
    [i, j] = robot
    oldLocation = [i, j]
    r = 0
    if action == "n":
        newLocation = [i-1, j]
    elif action == "s":
        newLocation = [i+1, j]
    elif action == "e":
        newLocation = [i, j+1]
    elif action == "w":
        newLocation = [i, j-1]
    else:
        newLocation = [i, j]

    # print(newLocation)
    robot = newLocation
    if (newLocation[0] < 0 or newLocation[0] >= size or newLocation[1] < 0 or newLocation[1] >=size):
        r = -5
        robot = oldLocation
    elif (action == "c" and canBoard[newLocation[0], newLocation[1]] == 1):
        r = 10
        canBoard[newLocation[0], newLocation[1]] = 0
    elif (action == "c" and canBoard[newLocation[0], newLocation[1]] == 0):
        r = -1
    else:
        r = 0

    #r -= 0.5
    score += r
    return r

def game():
    global robot
    epsolon = 1.0
    epoch = np.zeros(50)
    scores = np.zeros(50)
    testScores = np.zeros(5000)
    times = 0
    # training
    for i in range(N):
        setUpGame()
        if ((i+1)%50 == 0):
            epsolon -= 0.01
        if (epsolon <= 0.1):
            epsolon = 0.1

        for j in range(M):
            action, oldStatenumber = chooseAction(epsolon)
            reward = doAction(action)
            updateQ(action, oldStatenumber, reward)

        print (i, score)
        if ((i+1)%80 == 0):
            epoch[times] = (times+1) * 80
            scores[times] = score
            times += 1


    print (epoch)
    print (scores)
    plot.plot(epoch, scores, color='darkorange', lw=2)
    plot.title('Training Reward')
    plot.xlabel('Epoch')
    plot.ylabel('Reward')
    plot.show()

    # testing
    for k in range(N):
        setUpGame()
        for j in range(M):
            action, oldStatenumber = chooseAction(0.0)
            reward = doAction(action)

        print (k, score)
        testScores[k] = score

    testAverage = np.mean(testScores)
    testSD = np.std(testScores)
    print ('Test-Average', testAverage)
    print ('Test-Standard-Deviation', testSD)

if __name__ == "__main__":
    game()


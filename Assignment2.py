import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(precision=3, suppress=True) #lower precision of printed Q_values to 3 decimal places for easy reading.

# init parameters
alpha = 0.5  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.1  # exploration probability

# init the grid
grid = np.zeros((5, 5))
startPos = (0, 0)
goalPos = (4, 4)
holes = [(1, 0), (1, 3), (3, 1), (4, 2)]

# Initialize the Q-values (used for storing Qvals for each state and action)
Q = np.zeros((5, 5, 4)) # 5 and 5 refer to rows/cols, 4 refers to actions. (e.g up/down/left/right)

# state rewards
rewardGrid = np.zeros((5, 5))
rewardGrid.fill(-1) #set non terminals to -1 reward
    #Setting goal reward = 10
rewardGrid[goalPos] = 10
    #Setting hole rewards = -5
rewardGrid[holes[0]] = -5
rewardGrid[holes[1]] = -5
rewardGrid[holes[2]] = -5
rewardGrid[holes[3]] = -5

#hold reward values for later plotting
episodeRewards = np.array([])

#Print of initial grid of reward values.
print("\n")
print("Initial grid: ")
print(rewardGrid)
print("\n")

# action definations
actions = ["up", "down", "left", "right"] #use in random state chosing
actionMap = {"up": 0, "down": 1, "left": 2, "right": 3} #for mapping actions to q value in Q[].


# Q learning
for episode in range(10000): #10,000 episodes

    rewardTotal = 0 #For tracking reward over an episode
    steps = 0 #Used for counting steps taken in an episode. (Didn't use this anywhere, but cool information to print.)

    state = startPos #set initial state = start
    action = actions[np.random.choice(len(actions))] #select a random action as per epsilon greedy policy

    #Continue so long as not in a hole or goal
    while state != goalPos and state not in holes:
        # Take the chosen action and observe the next state and reward
        # access random action and get the state on the grid it is moving to.
        # max()/min() prevents the agent moving outside the grid (e.g moving up from row 0, would result in 0 instead of -1)
        if action == "up":
            nextState = (max(state[0] - 1, 0), state[1]) #state[1] to keep row index
            steps += 1
        elif action == "down":
            nextState = (min(state[0] + 1, 4), state[1])
            steps += 1
        elif action == "left":
            nextState = (state[0], max(state[1] - 1, 0)) #state[0] to keep column index
            steps += 1
        else:
            nextState = (state[0], min(state[1] + 1, 4))
            steps += 1

        reward = rewardGrid[nextState[0], nextState[1]] #get reward on current state for use in Q learning

        # Choosing next action based on eps-greedy policy
        if np.random.uniform(0, 1) < epsilon:
            nextAction = np.random.choice(actions)
        else:
            # if random value > epsilon, get the action with the highest Q value for next state
            nextAction = actions[np.argmax(Q[nextState[0], nextState[1], :])]

        oldQ = Q[state[0], state[1], actionMap[action]]
        maxQ = Q[nextState[0], nextState[1], actionMap[nextAction]]
        # Update the Q-value for agents current state and action
        Q[state[0], state[1], actionMap[action]] += alpha * (reward + gamma * maxQ - oldQ)

        # update state and action to previously calculated versions.
        state = nextState
        action = nextAction

        #Track total reward for the episode
        rewardTotal += reward


    #Add total episode reward to list for plotting 
    episodeRewards = np.append(episodeRewards,rewardTotal)


    #--------

    #Uncomment to enable linear epsilon decay.
    #epsilon = epsilon - 0.00001 # this value will insure it reachs 0 by 10,000th episode

    #--------


#Plot too dense to read with 10,000 episodes. using rolling average over 100 episodes.
rollingAverageReward = pd.Series(episodeRewards).rolling(window=100).mean()

# for visual of final grid with calculated Q-values.
finalGrid = np.copy(rewardGrid)
for i in range(5):
    for j in range(5):
        if (i, j) != goalPos and (i, j) not in holes:
            # update new grid to that of the best Q-values in Q[state,state]
            finalGrid[i, j] = np.max(Q[i, j])


# display grid with final Q-values
print("Final Q-Values: ")
#print(Q)
print(finalGrid)
print("\n")

#Plot reward over episodes (rolling average per 100 episodes)
plt.figure(figsize=(10, 6))
plt.plot(rollingAverageReward)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()

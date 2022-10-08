import math
import random
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Generate Data
bandit1 = np.random.poisson(lam=2, size=10000)
bandit2 = np.random.poisson(lam=3, size=10000)
bandit3 = np.random.poisson(lam=3.5, size=10000)
bandit4 = np.random.poisson(lam=10, size=10000)

df = pd.DataFrame({"b1": bandit1, "b2": bandit2, "b3": bandit3, "b4": bandit4})
sns.boxplot(x="variable", y="value", data=pd.melt(df[['b1','b2','b3','b4']]))
plt.title("Distribution of Rewards by Bandit \ Lamda")
plt.show()

def UCB():
    N = len(df.index)       # the time (or round) 
    d = 4                   # number of possible bandits
    Qt_a = 0
    Nt_a = np.zeros(d)      #number of times action a has been selected prior to T
                            #If Nt(a) = 0, then a is considered to be a maximizing action.
    c = 1                   #a number greater than 0 that controls the degree of exploration
    sum_rewards = np.zeros(d) #cumulative sum of reward for a particular message
    #helper variables to perform analysis
    hist_t = [] #holds the natural log of each round
    hist_achieved_rewards = [] #holds the history of the UCB CHOSEN cumulative rewards
    hist_best_possible_rewards = [] #holds the history of OPTIMAL cumulative rewards
    hist_random_choice_rewards = [] #holds the history of RANDONMLY selected actions rewards

    #loop through no of rounds #t = time
    for t in range(0,N):
        UCB_Values = np.zeros(d) #array holding the ucb values. we pick the max  
        action_selected = 0
        for a in range(0, d):
            if (Nt_a[a] > 0):
                ln_t = math.log(t) #natural log of t
                hist_t.append(ln_t) #to plot natural log of t
                #calculate the UCB
                Qt_a = sum_rewards[a]/Nt_a[a]
                ucb_value = Qt_a + c*(ln_t/Nt_a[a]) 
                UCB_Values[a] = ucb_value
            #if this equals zero, choose as the maximum. Cant divide by negative     
            elif (Nt_a[a] == 0):
                UCB_Values[a] = 1e500 #make large value
            
        #select the max UCB value
        action_selected = np.argmax(UCB_Values)
        #update Values as of round t
        Nt_a[action_selected] += 1
        reward = df.values[t, action_selected]
        sum_rewards[action_selected] += reward

        #these are to allow us to perform analysis of our algorithmm
        
        r_ = df.values[t,[0,1,2,3]]     #get all rewards for time t to a vector
        r_best = r_[np.argmax(r_)]      #select the best action
        
        pick_random = random.randrange(d) #choose an action randomly
        r_random = r_[pick_random] #np.random.choice(r_) #select reward for random action
        if len(hist_achieved_rewards)>0:
            hist_achieved_rewards.append(hist_achieved_rewards[-1]+reward)
            hist_best_possible_rewards.append(hist_best_possible_rewards[-1]+r_best)
            hist_random_choice_rewards.append(hist_random_choice_rewards[-1]+r_random)
        else:
            hist_achieved_rewards.append(reward)
            hist_best_possible_rewards.append(r_best)
            hist_random_choice_rewards.append(r_random)

    print("Reward if we choose randonmly {0}".format(hist_random_choice_rewards[-1]))
    print("Reward of our UCB method {0}".format(hist_achieved_rewards[-1]))
    plt.bar(['b1','b2','b3','b4'],Nt_a)
    plt.title("Number of times each bandit was Selected (UCB)")
    plt.show()

def lamdaUCB():
    N = len(df.index)
    d = 4 
    Nt_a = np.zeros(d)      

    sum_rewards = np.zeros(d) 
   
    hist_achieved_rewards = [] 
    hist_best_possible_rewards = []

    for t in range(0,N):
        UCB_Values = np.zeros(d)
        action_selected = 0
        for a in range(0, d):
            if (Nt_a[a] > 0):
                #MLE for lamdas
                ucb_value = sum_rewards[a]/Nt_a[a]
                UCB_Values[a] = ucb_value
            #if this equals zero, choose as the maximum. Cant divide by negative     
            elif (Nt_a[a] == 0):
                UCB_Values[a] = 1e500 #make large value
        print(UCB_Values)
        #select the max UCB value
        action_selected = np.argmax(UCB_Values)
        print("action selected: " + str(action_selected))
        #update Values as of round t
        Nt_a[action_selected] += 1
        reward = df.values[t, action_selected]
        sum_rewards[action_selected] += reward
        
        r_ = df.values[t,[0,1,2,3]]     #get all rewards for time t to a vector
        r_best = r_[np.argmax(r_)]      #select the best action

        if len(hist_achieved_rewards)>0:
            hist_achieved_rewards.append(hist_achieved_rewards[-1]+reward)
            hist_best_possible_rewards.append(hist_best_possible_rewards[-1]+r_best)
        else:
            hist_achieved_rewards.append(reward)
            hist_best_possible_rewards.append(r_best)
    print("Reward of our lamda-UCB method {0}".format(hist_achieved_rewards[-1]))
    plt.bar(['b1','b2','b3','b4'],Nt_a)
    plt.title("Number of times each bandit was Selected (lamda-UCB)")
    plt.show()
UCB()
lamdaUCB()

"""
Implementation of the Axelrod model of cultural convergence and polarisation
AXELROD R (1997) The dissemination of culture - A model with local convergence and global polarization. 
    Journal of Conflict Resolution 41(2), pp. 203-226.]
"""

# Import libraries
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from matplotlib import cm

# Parameters
PROB_SHIFT = 0.1 # additional winning probability for the agent with more money
MONEY_GAMBLE_SHARE = 0.004 # play in a game for this share of total money in the economy (relates to total money for normalization)
MONEY_GAMBLE_FIXED = 0.01 # fixed money gamble parameter
TAX_RATE = 0 # flat tax on wealth, redistributed lump-sum every iteration. Set to 0 to get model without state

# Variables for different versions of the model
FIXED_GAMBLE = 1 # 1: agents play for a fixed amount of money, 0: play for share of cake via MONEY_GAMBLE_SHARE
ALLOW_DEBT = 0 # 1: allow debt, 0: agents stop playing when they run out of money
START_WEALTH_DISTRIBUTION = 4 # 1: everyone gets 2/7, 2: random draw from uniform(0,4/7), 3: random draw from beta(2,5): replicates realistic wealth distribution between 0 and 1 with expected value 2/7; 4: everyone gets 1
GREEDINESS_DISTRIBUTION = 4 # 1: everyone gets 0.5, 2: random draw from uniform(0,1), 3: random draw from beta(5,5): some agents are very greedy and some not at all; 4: half is 0.6, other half is 0.4
VISUAL = 0 # 1: shows plots while calculating, 0: does not show plots
ITER_VISUAL = 100 # show visualization of model every ITER_VISUAL number of iterations
VISUAL_MIN = 0.8 # min money value for visualization of agent grid
VISUAL_MAX = 1.2 # max money value for visualization of agent grid

# Variables related to computation
SIZE = 7 # grid of agents with SIZE * SIZE
RUNS = 100 # number of iterations
N_SIM = 1 # number of model simulations

# Define the model and the agents as classes
class Agent():
    "Agents to populate the model"
    
    def __init__(self):
        if GREEDINESS_DISTRIBUTION == 1:
            self.greediness = 0.5
        elif GREEDINESS_DISTRIBUTION == 2:
            self.greediness = rd.uniform(0, 1)
        elif GREEDINESS_DISTRIBUTION == 3:
            self.greediness = rd.betavariate(5, 5)
        elif GREEDINESS_DISTRIBUTION == 4:
            self.greediness = 0.6 if rd.random() < 0.5 else 0.4
            
        if START_WEALTH_DISTRIBUTION == 1:
            self.money = 2/7
        elif START_WEALTH_DISTRIBUTION == 2:
            self.money = rd.uniform(0, 4/7)
        elif START_WEALTH_DISTRIBUTION == 3:
            self.money = rd.betavariate(2, 5)
        elif START_WEALTH_DISTRIBUTION == 4:
            self.money = 1

    def game(self, MONEY_GAMBLE, target, model):
        interaction_probability = self.greediness

        if ALLOW_DEBT == 0:
            if self.money <= 0 or model.agents[target].money <= 0:
                interaction_probability = 0

        if self.money > model.agents[target].money:
            win_probability = 0.5 + PROB_SHIFT
        elif self.money == model.agents[target].money:
            win_probability = 0.5
        elif self.money < model.agents[target].money:
            win_probability = 0.5 - PROB_SHIFT

        if rd.uniform(0, 1) < interaction_probability:
            if rd.uniform(0, 1) < win_probability:    
                model.agents[target].money = model.agents[target].money - MONEY_GAMBLE
                self.money = self.money + MONEY_GAMBLE
            else: 
                model.agents[target].money = model.agents[target].money + MONEY_GAMBLE
                self.money = self.money - MONEY_GAMBLE

class Axelrod():
    "This is the model"
    
    def __init__(self):
        self.agents = [Agent() for i in range(SIZE**2)]
        self.n_agents = SIZE**2

        self.results = {}
        self.results['money'] = {i: [] for i in range(SIZE**2)}
        self.results['greediness'] = {i: [] for i in range(SIZE**2)}
    
    def tick(self):
        "Runs a single cycle of the simulation: each agent plays once"
        for i in range(SIZE**2):    
            try:
                active = list(enumerate(self.agents))[i]
                passive = rd.choice(list(enumerate(self.agents)))
                while active[0] == passive[0]: # [0] gives index, [1] gives agent object
                    passive = rd.choice(list(enumerate(self.agents))) 
                
                if FIXED_GAMBLE == 1:
                    active[1].game(MONEY_GAMBLE_FIXED, passive[0], self)
                elif FIXED_GAMBLE == 0:
                    active[1].game(MONEY_GAMBLE_SHARE * sum([self.agents[i].money for i in range(SIZE**2)]), passive[0], self)
                
                for i in range(SIZE**2): # taxation
                   total_wealth = sum([self.agents[i].money for i in range(SIZE**2)])
                   self.agents[i].money = (1-TAX_RATE) * self.agents[i].money + (TAX_RATE * total_wealth / (SIZE**2))
            except IndexError:
               self.tick()
    
    def run_sim(self):
        for i in range (SIZE**2):
            self.results['greediness'][i].append(self.agents[i].greediness)
       
        for r in range(RUNS):
            for i in range (SIZE**2):
                self.results['money'][i].append(self.agents[i].money)
            self.tick()

            if r == RUNS:
                for i in range (SIZE**2):
                    self.results['money'][i] = {}
                    self.results['money'][i].append(self.agents[i].money)

            if VISUAL == 1:
                if r % ITER_VISUAL == 0:
                    print('run {}'.format(r))
                    self.visualize_output()

    def visualize_output(self):
        color = []
        for i in range(len(self.agents)):
            color.append(self.agents[i].money)
        color = np.reshape(color, (SIZE,SIZE))
        plt.imshow(color, interpolation='nearest', cmap=cm.Blues, vmin=VISUAL_MIN, vmax=VISUAL_MAX)
        plt.show()

if __name__ == "__main__":
    results_sim = {i: [] for i in range(N_SIM)}
    for s in range(N_SIM):
        model = Axelrod()
        model.run_sim()
        results_sim[s].append(model.results)
    #print(model.results['money'][0])
    #print(model.results['greediness'][1][0] > 0) # need second 0 index to access greediness in list
    #print(results_sim[0][0]['money'][0]) # need second 0 index to access model in list

def plot_money_evolve(results_sim, n_run):
    count_greedy = 0
    count_not_greedy = 0
    for i in range(SIZE**2):
        if results_sim[n_run][0]['greediness'][i][0] > 0.5:
            if count_greedy == 0:
                plt.plot(results_sim[n_run][0]['money'][i], 'r', label = 'Greedy')
                count_greedy += 1
            else:
                plt.plot(results_sim[n_run][0]['money'][i], 'r')
        else:
            if count_not_greedy == 0:
                plt.plot(results_sim[n_run][0]['money'][i], 'b', label = 'Not greedy')
                count_not_greedy += 1
            else: 
                plt.plot(results_sim[n_run][0]['money'][i], 'b')
    plt.title('Evolution of money over time: ' + str(100*TAX_RATE) + '% taxes/period')
    plt.xlabel('Time')
    plt.ylabel('Money')
    plt.legend(framealpha=1, frameon=False, loc='lower center', ncol=2)
    plt.show()
plot_money_evolve(results_sim, 0) # plot money evolution for the first model run

def show_histogram(results_sim, state, n_run):
    distribution = []
    for i in range (SIZE**2):
        distribution.append(results_sim[n_run][0]['money'][i][state])
    plt.hist(distribution)
    plt.title('Money distribution after ' + str(RUNS) + ' periods')
    plt.xlabel('Money')
    plt.ylabel('No. agents')
    plt.show()
show_histogram(results_sim, RUNS-1, 0) # plot final money distribution for the last model run

def gini(x):
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x)**2 * np.mean(x))

def plot_gini_evolve(results_sim, n_run):
    money_distribution = np.zeros((SIZE**2, RUNS))
    for i in range(SIZE**2):
        money_distribution[i] = results_sim[n_run][0]['money'][i]
    n_features, n_timepoints = money_distribution.shape
    gini_index = np.zeros(n_timepoints)
    for t in range(n_timepoints):
        gini_index[t] = gini(money_distribution[:,t])
    plt.plot(gini_index)
    plt.title('Gini coefficient of money over time: ' + str(100*TAX_RATE) + '% taxes/period')
    plt.xlabel('Time')
    plt.ylabel('Gini coefficient')    
    plt.show()
plot_gini_evolve(results_sim, 0) # plot gini evolution for the first model run
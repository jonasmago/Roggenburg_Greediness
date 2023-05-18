"""
Implementation of the Axelrod model of cultural convergence and polarisation
AXELROD R (1997) The dissemination of culture - A model with local convergence and global polarization. 
    Journal of Conflict Resolution 41(2), pp. 203-226.]
"""

# Import libraries
import numpy as np
import random as rd
import matplotlib.pyplot as plt

# Parameters
PROB_SHIFT = 0.1 # additional winning probability for the agent with more money
MONEY_GAMBLE_SHARE = 0.004 # play in a game for this share of total money in the economy (relates to total money for normalization)
MONEY_GAMBLE_FIXED = 0.01 # fixed money gamble parameter, alternative to MONEY_GAMBLE_SHARE
TAX_RATE = 0 # flat tax on wealth (share of wealth between 0 and 1), redistributed lump-sum every iteration

# Variables for different versions of the model
FIXED_GAMBLE = 1 # 1: agents play for a fixed amount of money via MONEY_GAMBLE_FIXED, 0: play for share of total money via MONEY_GAMBLE_SHARE
ALLOW_DEBT = 0 # 1: allow debt (agents can have negative money and continue playing when they run out of money), 0: agents stop playing when they run out of money
START_WEALTH_DISTRIBUTION = 4 # 1: everyone gets 2/7, 2: random draw from uniform(0,4/7), 3: random draw from beta(2,5): replicates realistic wealth distribution between 0 and 1 (expected value is 2/7); 4: everyone gets 1. Expected value of 2/7 in scenarios 1-3 normalizes the expected wealth agents start with to align with the expected value from scenario 3
GREEDINESS_DISTRIBUTION = 4 # 1: everyone gets 0.5, 2: random draw from uniform(0,1), 3: random draw from beta(5,5): some agents are very greedy and some not at all (expected value is 0.5); 4: half is 0.6, other half is 0.4

# Variables related to computation
N_AGENTS = 50 # number of agents
N_ITER = 100 # number of iterations
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
        self.agents = [Agent() for i in range(N_AGENTS)]
        self.results = {}
        self.results['money'] = {i: [] for i in range(N_AGENTS)}
        self.results['greediness'] = {i: [] for i in range(N_AGENTS)}
    
    def tick(self):
        "Runs a single cycle of the simulation: each agent plays once"
        for i in range(N_AGENTS):    
            try:
                active = list(enumerate(self.agents))[i]
                passive = rd.choice(list(enumerate(self.agents)))
                while active[0] == passive[0]: # [0] gives index, [1] gives agent object
                    passive = rd.choice(list(enumerate(self.agents))) 
                
                # money gamble game
                if FIXED_GAMBLE == 1:
                    active[1].game(MONEY_GAMBLE_FIXED, passive[0], self)
                elif FIXED_GAMBLE == 0:
                    active[1].game(MONEY_GAMBLE_SHARE * sum([self.agents[i].money for i in range(N_AGENTS)]), passive[0], self)
                
                # taxation
                for i in range(N_AGENTS): 
                   total_wealth = sum([self.agents[i].money for i in range(N_AGENTS)])
                   self.agents[i].money = (1-TAX_RATE) * self.agents[i].money + (TAX_RATE * total_wealth / (N_AGENTS))
            except IndexError:
               self.tick()
    
    def run_sim(self):
        for i in range (N_AGENTS):
            self.results['greediness'][i].append(self.agents[i].greediness)
       
        for r in range(N_ITER):
            for i in range (N_AGENTS):
                self.results['money'][i].append(self.agents[i].money)
            self.tick()

            if r == N_ITER:
                for i in range (N_AGENTS):
                    self.results['money'][i] = {}
                    self.results['money'][i].append(self.agents[i].money)

if __name__ == "__main__":
    results_sim = {i: [] for i in range(N_SIM)}
    for s in range(N_SIM):
        model = Axelrod()
        model.run_sim()
        results_sim[s].append(model.results)

def plot_money_evolve(results_sim, n_run):
    count_greedy = 0
    count_not_greedy = 0
    for i in range(N_AGENTS):
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
    for i in range (N_AGENTS):
        distribution.append(results_sim[n_run][0]['money'][i][state])
    plt.hist(distribution)
    plt.title('Money distribution after ' + str(N_ITER) + ' periods')
    plt.xlabel('Money')
    plt.ylabel('No. agents')
    plt.show()
show_histogram(results_sim, N_ITER-1, 0) # plot final money distribution for the first model run

def gini(x):
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x)**2 * np.mean(x))

def plot_gini_evolve(results_sim, n_run):
    money_distribution = np.zeros((N_AGENTS, N_ITER))
    for i in range(N_AGENTS):
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
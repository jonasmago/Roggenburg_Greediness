"""
Implementation of the Axelrod model of cultural convergence and polarisation

AXELROD R (1997) The dissemination of culture - A model with local convergence and global polarization. 
    Journal of Conflict Resolution 41(2), pp. 203-226.]
"""

# Convention: greediness is the first component, money the second

# Import libraries
import numpy as np
from matplotlib import pyplot
from numpy.random import random
from matplotlib import cm
import random as rd

# Define constants
SIZE = 7 # grid of agents with SIZE * SIZE
RUNS = 100000 # number of iterations
PROB_SHIFT = 0.1 # additional winning probability for the agent with more money
MONEY_GAMBLE_SHARE = 0.004 # play in a game for this share of total money in the economy
START_WEALTH_DISTRIBUTION = 4 # 1: everyone gets 2/7, 2: random draw from uniform(0,4/7), 3: random draw from beta(2,5): replicates realistic wealth distribution between 0 and 1 with expected value 2/7; 4: everyone gets 1
GREEDINESS_DISTRIBUTION = 4 # 1: everyone gets 0.5, 2: random draw from uniform(0,1), 3: random draw from beta(5,5): some agents are very greedy and some not at all; 4: half is 0.6, other half is 0.4
TAX_RATE = 0.00003 # flat tax on wealth, redistributed lump-sum every iteration. Set to 0 to get model without state
NO_DEBT = 1 # 1: switch off that agents can have money smaller or equal to zero, 0: allow debt
VISUAL = 0 # 1: shows plots while calculating, 0: does not show plots
# Seed pseudo-random number generator

#rd.seed(1)

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
        self.performance=[]

    def game(self, MONEY_GAMBLE, target, model):
        interaction_probability = self.greediness

        if NO_DEBT == 1:
            if self.money <= 0 or model.agents[target].money <= 0:
                interaction_probability = 0

        if self.money > model.agents[target].money:
            win_probability = 0.5 + PROB_SHIFT
        else:
            win_probability = 0.5 - PROB_SHIFT

        if rd.uniform(0, 1) < interaction_probability:
            if rd.uniform(0, 1) < win_probability:    
                model.agents[target].money = model.agents[target].money - MONEY_GAMBLE
                self.money = self.money + MONEY_GAMBLE
                model.agent[target].performance.append(0)
                self.performance.append(1)
                
            else: 
                model.agents[target].money = model.agents[target].money + MONEY_GAMBLE
                self.money = self.money - MONEY_GAMBLE
                model.agent[target].performance.append(1)
                self.performance.append(0)

class Axelrod():
    "This is the model"
    
    def __init__(self):
        self.agents = [Agent() for i in range(SIZE**2)]
        self.n_agents = SIZE**2
        
        # init result vector 
        self.results = {} # empty dictionary
        self.results = {i: [] for i in range(SIZE**2)}
        # add results for each participant to their corresponding key
        for i in range(self.n_agents):
            self.results[i] = []

    
    def tick(self):
        "Runs a single cycle of the simulation"
        try:
            active = rd.choice(list(enumerate(self.agents)))
            passive = rd.choice(list(enumerate(self.agents)))
            while active[0] == passive[0]:
                passive = rd.choice(list(enumerate(self.agents)))     
            active[1].game(MONEY_GAMBLE_SHARE * sum([self.agents[i].money for i in range(SIZE**2)]), passive[0], self)
            total_wealth = sum([self.agents[i].money for i in range(0, SIZE**2)])
            for i in range(0, SIZE**2):
                self.agents[i].money = (1-TAX_RATE) * self.agents[i].money + (TAX_RATE * total_wealth / (SIZE**2))
        except IndexError:
            self.tick()
 
    def show_state(self):
        for n in range(0, SIZE):
            #row = [ [self.agents[i].greediness, self.agents[i].money] for i in range(n * SIZE, n * (SIZE) + SIZE)]
            row = [ [self.agents[i].money] for i in range(n * SIZE, n * (SIZE) + SIZE)]
            for agent in row:
                print("[", end="")
                print(*agent, sep=",", end="")
                print("]", end="")
            print()
            print("-" * (SIZE*2*2+SIZE))
    
    def run_sim(self):
        print("Initial state of the model:")
        print()
        self.show_state()
        print()
        print("Running simulation...")
        for r in range(1,RUNS):
            for i in range (SIZE**2):
                self.results[i].append(self.agents[i].money)
            if VISUAL == 1:
                if r % (RUNS/5) == 0:
                    print('run {}'.format(r))
                    self.visualize_output()
            self.tick()
        print("Final state of the model:")
        self.show_state()

    def visualize_output(self):
        color = []
        for i in range(len(self.agents)):
            color.append(self.agents[i].money)
        color = np.reshape(color, (SIZE,SIZE))
        pyplot.imshow(color, interpolation='nearest', cmap=cm.Blues, vmin=0, vmax=1)
        pyplot.show()

if __name__ == "__main__":
    model = Axelrod()
    model.run_sim()


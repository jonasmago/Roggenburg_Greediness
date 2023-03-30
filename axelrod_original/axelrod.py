"""
Implementation of the Axelrod model of cultural convergence and polarisation

AXELROD R (1997) The dissemination of culture - A model with local convergence and global polarization. 
    Journal of Conflict Resolution 41(2), pp. 203-226.

"""

# Convention: greediness is the first component, money the second

# Import libraries
import random as rd

# Define constants
SIZE = 5 # grid of agents with SIZE * SIZE
RUNS = 1000 # number of iterations
PROB_SHIFT = 0.1 # additional winning probability for the agent with more money
MONEY_GAMBLE_SHARE = 0.004 # play in a game for this share of total money in the economy
MONEY_GAMBLE = 0 # initialize absolute value of gambled money
START_WEALTH_DISTRIBUTION = 1 # 1: everyone gets 2/7, 2: random draw from uniform(0,4/7), 3: random draw from beta(2,5): replicates realistic wealth distribution between 0 and 1 with expected value 2/7

# Seed pseudo-random number generator
#rd.seed(1)

# Define the model and the agents as classes
class Agent():
    "Agents to populate the model"
    
    def __init__(self):
        self.greediness = rd.uniform(0, 1)
        if START_WEALTH_DISTRIBUTION == 1:
            self.money = 2/7
        elif START_WEALTH_DISTRIBUTION == 2:
            self.money = rd.uniform(0, 4/7)
        elif START_WEALTH_DISTRIBUTION == 3:
            self.money = rd.betavariate(2, 5)
    def game(self, target, model):
        interaction_probability = self.greediness

        if self.money > model.agents[target].money:
            win_probability = 0.5 + PROB_SHIFT
        else:
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
    
    def tick(self):
        "Runs a single cycle of the simulation"
        try:
            active = rd.choice(list(enumerate(self.agents)))
            passive = rd.choice(list(enumerate(self.agents)))
            while active[0] == passive[0]:
                passive = rd.choice(list(enumerate(self.agents)))
            active[1].game(passive[0], self)
        except IndexError:
            self.tick()
 
    def show_state(self):
        for n in range(0, SIZE):
            row = [ [self.agents[i].greediness, self.agents[i].money] for i in range(n * SIZE, n * (SIZE) + SIZE)]
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
        MONEY_GAMBLE = MONEY_GAMBLE_SHARE * sum([self.agents[i].money for i in range(0, SIZE**2)])
        print()
        print("Running simulation...")
        for r in range(1,RUNS):
            self.tick()
        print("Final state of the model:")
        self.show_state()
        print("Money gamble:")
        print(MONEY_GAMBLE)
        
model = Axelrod()

model.run_sim()

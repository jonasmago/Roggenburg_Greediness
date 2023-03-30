from axelrod import Agent, Axelrod
%load_ext autoreload
%autoreload 2
%matplotlib qt

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr



model = Axelrod()
model.run_sim()




def plot_money_evolve(model):
    plt.figure(figsize=(3, 2), dpi=200)
    my_labels = {"1" : "rich & greedy", "2" : "rich & not greedy", "3" : "poor & greedy", "4" : "poor & not greedy"}

    for i in range (model.n_agents):
        if (model.agents[i].greediness>0.5) and (model.agents[i].money>0.5):
            plt.plot (model.results[i][::500], 'blue', label=my_labels["1"])
            my_labels["1"] = "_nolegend_"
        elif (model.agents[i].greediness>0.5) and (model.agents[i].money<0.5):
            plt.plot (model.results[i][::500], 'orange', label=my_labels["2"])
            my_labels["2"] = "_nolegend_"
        elif (model.agents[i].greediness<0.5) and (model.agents[i].money>0.5):
            plt.plot (model.results[i][::500], 'g', label=my_labels["3"])
            my_labels["3"] = "_nolegend_"
        elif (model.agents[i].greediness<0.5) and (model.agents[i].money<0.5):
            plt.plot (model.results[i][::500], 'y', label=my_labels["4"])
            my_labels["4"] = "_nolegend_"

    plt.legend()
    plt.title('money over time')
    plt.xlabel('time')
    plt.ylabel('money')
    plt.tight_layout()
    plt.show()
plot_money_evolve(model)





def money_violin (model):
    distribution_start=[]
    distribution_end=[]
    for i in range (model.n_agents):
        distribution_start.append(model.results[i][0])
        distribution_end.append(model.results[i][-1])
    data = {'start': distribution_start, 'end': distribution_end}
    df = pd.DataFrame(data)
    fig, axes = plt.subplots(figsize=(5,5))
    sns.set(style="whitegrid")
    sns.violinplot(data=df, ax = axes, orient ='h')
money_violin(model)



# compute the greediness and delta for our simulation
delta = []
greediness = []
for i in range(model.n_agents):
    delta.append(model.results[i][0]-model.results[i][-1])
    greediness.append(model.agents[i].greediness)




# calculate the correlation and p-value
corr, p_value = pearsonr(greediness, np.abs(delta))

# print the results
print("Correlation: ", corr)
print("P-value: ", p_value)




plt.figure(figsize=(3, 2), dpi=200)
# create a dataframe with the two vectors
data = {'greediness': greediness, 'delta': np.abs(delta)}
df = pd.DataFrame(data)

# plot the scatter plot with regression line and 95% confidence interval
sns.regplot(x='greediness', y='delta', data=df, ci=95)

# add a title and axis labels
plt.title(r'correlation between $likelihood to play$ and $|\Delta|$')
plt.xlabel('likelihood to play')
plt.ylabel('$|\Delta|$')

plt.show()




### implementation ideas

- delta money (veränderung) abhängig von greed 
- Varianz (SD von den 25 Geld werten zu zeit t) / Zeit 


- Einkommensverteilung 
- Graph mit Einkommensverteilung 
- correlation between 
    - reich & greedy
    - arm.. 


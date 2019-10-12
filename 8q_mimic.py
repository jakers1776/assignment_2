#8-Queens MIMIC
import mlrose
import mlrose
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

def queens_max(state): # Initialize counter
    fitness_cnt = 0
          # For all pairs of queens
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):
        # Check for horizontal, diagonal-up and diagonal-down attacks
          if (state[j] != state[i]) \
          and (state[j] != state[i] + (j - i)) and (state[j] != state[i] - (j - i)):
                           # If no attacks, then increment counter
            fitness_cnt += 1 
    return fitness_cnt
# Initialize custom fitness function object
fitness_cust = mlrose.CustomFitness(queens_max)

# Initialize fitness function object using pre-defined class
fitness = mlrose.Queens()
# Define optimization problem object
problem = mlrose.DiscreteOpt(length = 8, fitness_fn = fitness_cust, maximize=True, max_val=8)
# Define decay schedule
schedule = mlrose.ExpDecay()
# Start Timer
from timeit import default_timer as timer
start = timer()

# Run Randomized Hill Climbing        
best_state, best_fitness,fit_curve = mlrose.mimic(problem,max_attempts = 1000, 
                                                      max_iters = 1000,
                                                      curve=True,random_state = 1)
# Stop Timer
elapsed_time = timer() - start # in seconds
print('Time elapsed time in seconds is: ',elapsed_time)
print('The best state found is: ', best_state)
print('The fitness at the best state is: ', best_fitness)
# Plot Curve
import matplotlib.pyplot as plt
plt.plot(fit_curve)
plt.ylabel('Fitness')
plt.xlabel('Iterations')
plt.xticks(np.arange(0,1001,100))
plt.show()
#plt.savefig('8q_mimic.png')

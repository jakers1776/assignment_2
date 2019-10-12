#8-Queens SA
import mlrose
import mlrose
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score


# Initialize fitness function object using pre-defined class
fitness = mlrose.Queens()
# Define optimization problem object
problem = mlrose.DiscreteOpt(length = 8, fitness_fn = fitness, maximize=True, max_val=8)
# Define decay schedule
schedule = mlrose.ExpDecay()
# Start Timer
from timeit import default_timer as timer
start = timer()

# Run Randomized Hill Climbing        
init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
best_state, best_fitness,fit_curve = mlrose.simulated_annealing(problem,max_attempts = 1000, 
                                                      max_iters = 1000, init_state = init_state,
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
#plt.savefig('8q_sa.png')

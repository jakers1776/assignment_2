  #Four Peaks Mimic
import mlrose
import mlrose
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score


# Initialize fitness function object using pre-defined class
fitness = mlrose.FourPeaks(t_pct=0.15)
# Define optimization problem object
problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
# Define decay schedule
schedule = mlrose.ExpDecay()
# Start Timer
from timeit import default_timer as timer
start = timer()
# Run Mimic
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
#plt.savefig('4p_mimic.png')

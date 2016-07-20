import numpy as np
import matplotlib.pyplot as plt

# number if samples
greyhounds = 500
labs = 500

grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)

plt.hist([grey_height, lab_height], stacked=True, 
         color=['r', 'b'], label=['greyhounds','labs'])
plt.title('Dog\'s height distribution')
plt.ylabel("# of dogs")
plt.xlabel("height")
plt.legend()
plt.show()

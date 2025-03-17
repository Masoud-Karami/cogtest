import numpy as np
import matplotlib.pyplot as plt

# Parameters
mean = 50  
std_dev = 8  
size = 1000  

# Generate samples
samples = np.random.normal(loc=mean, scale=std_dev, size=size)
samples = np.clip(np.round(samples), 1, 100)  # Round & clip between 1-100

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(samples, bins=20, alpha=0.7, edgecolor='black', density=True)
plt.title("Gaussian Distribution of Points (Mean=50, Std Dev=8)")
plt.xlabel("Points")
plt.ylabel("Density")
plt.grid(True)

# Save plot instead of showing
plt.savefig("gaussian_distribution.png")

print("Plot saved as 'gaussian_distribution.png'. Open it to view.")


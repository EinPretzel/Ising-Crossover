import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

""" Module Function"""
def initialize_spin_config(size_x, size_y, size_z):
    return 2 * np.random.randint(2, size=(size_z, size_x, size_y)) - 1

def metropolis_step(lattice, temperature):
    for _ in range(lattice.size):
        x, y, z = np.random.randint(0, lattice.shape[1]), np.random.randint(0, lattice.shape[2]),
        np.random.randint(0, lattice.shape[0])
        spin = lattice[z, x, y]
        if int(lattice.shape[0]) > 1: # 3D Config
            neighbor_sum = (
                lattice[(z + 1) % lattice.shape[0], x, y] +
                lattice[(z - 1) % lattice.shape[0], x, y] +
                lattice[z, (x + 1) % lattice.shape[1], y] +
                lattice[z, (x - 1) % lattice.shape[1], y] +
                lattice[z, x, (y + 1) % lattice.shape[2]] +
                lattice[z, x, (y - 1) % lattice.shape[2]]
            )
        elif int(lattice.shape[0]) == 1: # 2D Config
            neighbor_sum = (
                    lattice[z, (x + 1) % lattice.shape[1], y] +
                    lattice[z, (x - 1) % lattice.shape[1], y] +
                    lattice[z, x, (y + 1) % lattice.shape[2]] +
                    lattice[z, x, (y - 1) % lattice.shape[2]]
                )
            
        delta_energy = 2 * spin * neighbor_sum
        if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / temperature):
            lattice[z, x, y] *= -1

def calculate_magnetization(lattice):
    return np.sum(lattice)

def calculate_susceptibility(magnetizations, temperature):
    magnetization_variance = np.var(magnetizations)
    return magnetization_variance / temperature

def find_critical_temperature(temperature_range, magnetizations):
    susceptibilities = []

    for temperature, magnetization in tqdm(zip(temperature_range, magnetizations), 
                                           total=len(temperature_range), 
                                           desc="Calculating Susceptibility"):
        susceptibility = calculate_susceptibility(magnetization, temperature)
        susceptibilities.append(susceptibility)

    max_index = np.argmax(susceptibilities)
    critical_temperature = temperature_range[max_index]
    return critical_temperature

def simulate(X, Y, Z, temperature, steps, thermalization_steps):
    lattice = initialize_spin_config(X, Y, Z)
    magnetizations = []

    for _ in tqdm(range(thermalization_steps), desc= "Thermalizing Spin Configuration"):
        metropolis_step(lattice, temperature)

    for _ in tqdm(range(steps), desc="Applying Algorithm"):
        metropolis_step(lattice, temperature)
        magnetizations.append(calculate_magnetization(lattice))

    return np.array(magnetizations) / lattice.size

# Simulation parameters
size_x = 10
size_y = 10
size_z = 12
temperature_range = np.linspace(1, 6, 200)
steps = 10000
thermalization_steps = 1000

# Perform simulations for different temperatures
magnetizations_at_critical_temp = []

for temperature in tqdm(temperature_range, desc="Running Simulation"):
    magnetizations = simulate(size_x, size_y, size_z, temperature, steps, thermalization_steps)
    magnetizations_at_critical_temp.append(magnetizations)

# Find the critical temperature
critical_temperature = find_critical_temperature(temperature_range, 
                                                 magnetizations_at_critical_temp)
print(f"Estimated critical temperature: {critical_temperature:.3f}")

# Plot magnetization as a function of temperature
plt.scatter(temperature_range, [np.abs(m.mean()) for m in magnetizations_at_critical_temp], 
            s=10, marker='o', color='RoyalBlue')
plt.xlabel(r'Temperature, T in $\frac{J}{k_B}$')
plt.ylabel('Magnetization, <|M|>')
plt.axvline(x = critical_temperature, color = 'red', 
            label=f'Estimated Tc {critical_temperature:.3f}')
plt.title(f'Magnetization vs Temperature {size_x}x{size_y}x{size_z}')
plt.legend()
plt.savefig(f"Magnetization vs Temperature {size_x}x{size_y}x{size_z}")
plt.show()

# Plot susceptibility as a function of temperature
susceptibilities = [calculate_susceptibility(m, temp) for m, temp in 
                    zip(magnetizations_at_critical_temp, temperature_range)]
plt.scatter(temperature_range, susceptibilities, s=10, marker='o', color='LimeGreen')
plt.xlabel(r'Temperature, T in $\frac{J}{k_B}$')
plt.ylabel('Susceptibility, \u03C7')
plt.axvline(x = critical_temperature, color = 'red', 
            label=f'Estimated Tc {critical_temperature:.3f}')
plt.legend()
plt.title(f'Susceptibility vs Temperature {size_x}x{size_y}x{size_z}')
plt.savefig(f"Susceptibility vs Temperature {size_x}x{size_y}x{size_z}")
plt.show()

# Create data frame
Results_df = pd.DataFrame({"Temperature": temperature_range, 
                           "Magnetization": [np.abs(m.mean()) 
                                             for m in magnetizations_at_critical_temp], 
                           "Susceptibility": susceptibilities})
Results_df.to_csv(f"{size_x}x{size_y}x{size_z}_Results.csv", index = False)
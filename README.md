The following code is written for my thesis during my Bachelor's degree in Physics investigating the dimensional crossover of the critical temperature of the Ising model.

For testing purposes in 3D lattice sizes please only use 5x5x5, higher sizes take days to complete. 2D sizes work fine as is fast though. Code optimization is to be done eventually.

ising_simulation is used to generate the data which is automatically saved into a pandas dataframe and converted to a csv for individual lattice.
dim_curve is used for data visualization for critical temperature vs lattice shape across various lattices.

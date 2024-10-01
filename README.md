The following code is written for my thesis during my Bachelors degree in Physics investigating the dimensional crossover of the critical temperature of the Ising model.

It is slow and errors are known to occur at lattice sizes greater than 100x100x100.

For testing purposes in 3D lattice sizes please only use 5x5x5, higher sizes take days to complete. 2D sizes works fine as is fast though.

ising_simulation is used to generate the data which is automatically saved into a pandas dataframe and converted to a csv for individual lattice.
dim_curve is used for data visualization for critical temperature vs lattice shape across various lattices.

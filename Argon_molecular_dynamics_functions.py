"""
This code contains the core functions to simulate the classical dynamics of Argon atoms in a box of 1/2/3 dimensions, 
interacting with each other via the Lennard-Jones potential.

Goals of the following code:
-> Convert to dimensionless units to skip cumbersome calculations.
-> Implement an initialization of the atomic positions on an FCC lattice.
-> Store the initial positions and velocities of each particle.
-> Calculate the Lennard-Jones potential for all particle pairs.
-> Calculate the forces on each particle using the Lennard-Jones potential.
-> Impose periodic boundary conditions to the system.
-> Implement a hard-sphere potential for short-distanced interactions (optional).
-> Implement the Euler/Verlet methods to determine the time evolution of the system.
-> Store the positions and velocities of each particle at every time instant.
-> Calculate the kinetic, potential, and total energies of the system.
-> Compute errors for the statistical analysis of observables.
"""

# Import necessary libraries
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm
from scipy.optimize import curve_fit

class molecular_motion:

    
    def __init__(self, m, N, size, dim, r, temp, Dt, tsteps):
        """
        Enter the parameters of the problem.
        
        Parameters
        ----------
        m : float
        The mass of the particles.
        N : int
        The total number of atoms in the system.
        size : float
        The length of the sides of the box. (Assumed to be cubic.)
        dim : int
        The dimensionality of the system
        r : float
        The radius of the atoms.
        temp : float
        The temperature of the system.
        Dt : float
        The difference between two consecutive time instances.
        tsteps : int
        The number of time steps passed after the initialization.
        """

        if not isinstance(N, int):
            raise TypeError("The number of atoms must be an integer!")

        if not isinstance(dim, int):
            raise TypeError("The dimensionality of the system must be an integer!")

        if not isinstance(tsteps, int):
            raise TypeError("The number of time steps must be an integer!")

        self.k_B = 1.380649e-23
        self.sigma = 3.405e-10
        self.m = m
        self.N = N
        self.size = size
        self.dim = dim
        self.r = r
        self.temp = temp
        self.Dt = Dt
        self.tsteps = tsteps
        self.epsilon = self.k_B * 119.8 
        

    def fcc_atoms(self, n):
        """
        Calculates the total number of atoms in a FCC lattice given the number of
        atoms on each edge.

        Parameters
        ----------
        n: int
            The number of atoms on the edges of the FCC lattice.

        Returns
        -------
        int
            The total numbers of the atoms in the lattice for the chosen number of 
            atoms per edge.
        """

        if not isinstance(n, int):
            raise TypeError("The number of atoms must be an integer!")
        
        return  n**3 + 3 * n * (n - 1)**2

    
    def periodic_fcc_atoms(self, N):
        """
        Calculates the number of atoms in a FCC lattice which is closest to the
        given number of atoms in the system and the number of atoms to span the 
        corresponding periodic FCC lattice when applying periodic boundary conditions.

        Parameters
        ----------
        N: int
            The number of atoms of the system.

        Returns
        -------
        tuple
            The closest number of atoms that can create an FCC lattice, the number of unit cell edges
            along each direction, and the corresponding number of atoms that can span a periodic FCC 
            when applying periodic boundary conditions.
        """

        if not isinstance(N, int):
            raise TypeError("The number of atoms must be an integer!")

        
        # Number of atoms per edge for a simple cubic cell:
        n_s_cubic = int(N**(1/3))
        # Corresponding number of atoms for FCC lattices with number of atoms per side
        #ranging from 1 to n_s_cubic.
        choices = np.array([4*i**3 for i in range(1, n_s_cubic + 1)])
        # Choose minimum difference between FCC atoms and the original number of atoms:
        N_sides = int(np.argmin(abs(choices - N)) + 1)
        N_periodic_fcc =  4 * N_sides**3
        N_full_fcc = self.fcc_atoms(N_sides + 1)

        return N_sides, N_periodic_fcc, N_full_fcc

        
    
    def init_positions(self, init_type):
        """
        Initializes the particle positions on an FCC lattice inside a cubic simulation box.
        Ensures that the number of particles is compatible with the FCC structure.
        
        Parameters
        ----------
        init_type: str
            Specifies the preferred type of initialization of the positions. Can be either 'random',
            where all atoms are placed randomly in space, or 'fcc', where the atoms are placed
            in a FCC lattice.
            
        Returns
        -------
        np.ndarray
            (N, dim) array of FCC lattice positions in dimensionless units.
        """

        if not isinstance(init_type, str):
            raise TypeError("The chosen initialization must be a string input!")
            
        if init_type not in ['random', 'fcc']:
            raise TypeError("Please enter one of the two options: 'random' or 'fcc'")

        if init_type == 'random':

            min_dist = 2 * (self.r / self.sigma)
        
            # Preallocate a NumPy array to store the positions:
            init_pos = np.empty((self.N, self.dim))
        
            # Counter for the number of atoms placed:
            placed_atoms = 0
    
            init_pos[placed_atoms] = np.random.rand(self.dim) * (self.size / self.sigma)
        
            while placed_atoms < self.N:
                
                # Generate a new position within [0, size] in dimensionless units
                new_pos = np.random.rand(self.dim) * (self.size / self.sigma)
                    
                # Calculate distances between new_pos and all existing positions
                distances = np.linalg.norm(new_pos - init_pos[:placed_atoms], axis=1)
                    
                if np.all(distances > min_dist):
                        
                    init_pos[placed_atoms] = new_pos
                    placed_atoms += 1

        elif init_type == 'fcc':

            N_sides, N_periodic_fcc, N_full_fcc = self.periodic_fcc_atoms(self.N)
            self.N = N_periodic_fcc
            
            # Lattice constant:
            a = self.size / N_sides
                    
            # FCC basis:
            basis = np.array([
                            [0.0, 0.0, 0.0],
                            [0.5, 0.5, 0.0],
                            [0.5, 0.0, 0.5],
                            [0.0, 0.5, 0.5]
                            ])
            
            # Translate the basis to all possible positions inside the box:
            x, y, z = np.meshgrid(np.arange(N_sides), np.arange(N_sides), np.arange(N_sides), indexing='ij')
            unit_cell_indices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)
            init_pos = (basis[:, np.newaxis, :] + unit_cell_indices[np.newaxis, :, :]).reshape(-1, 3)

            # Convert to dimensionless units:
            init_pos *= a / self.sigma  # Scale positions by the lattice constant

            print(f'The closest number of atoms that can span a virtual periodic FCC lattice is {self.N}. The number of atoms of the corresponding full lattice is {N_full_fcc}.')

        return init_pos

    
    def init_velocities(self):
        """
        Generates initial particle velocities according to the Boltzmann distribution.
        
        Parameters
        ----------
        None
    
        Returns
        -------
        np.ndarray
            The Boltzmann velocity distribution for the initial velocities of the atoms.
        """  

        # Standard deviations of velocities in Boltzmann ensemble:
        std_dev = np.sqrt(self.k_B * (self.temp / self.m)) 
        # Generate random velocity components from a Gaussian distribution:
        init_vel = np.random.normal(0, std_dev, (self.N, self.dim)) 
        # Ensure that the net momentum of the system is zero:
        init_vel -= np.mean(init_vel, axis=0)
        # Convert to dimensionless units:
        init_vel *= np.sqrt(self.m / self.epsilon)  
    
        return init_vel 
        

    def atomic_distances(self, pos):
        """ 
        Calculates relative positions and distances between particles.
        
        Parameters
        ----------
        pos : np.ndarray
            The positions of the atoms in Cartesian space at a specific time instant.
    
        Returns
        -------
        tuple
            The arrays containing the relative positions/distances for all atom pairs at
            a specific time instance, having applied the minimal image convention, so that
            only the shortest distance between two particles and the image of one of the 
            particles is taken into account.
        """
    
        if not isinstance(pos, np.ndarray):
            raise TypeError("The positions must be stored in a numpy array!")
        
        # Use broadcasting to compute relative positions
        rel_pos = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]

        # Apply minimal image convention in dimensionless units
        rel_pos = (rel_pos + self.size / (2*self.sigma)) % (self.size / self.sigma) - self.size / (2*self.sigma)
        
        # Compute distances:
        rel_dist = np.linalg.norm(rel_pos, axis=2)

        rel_dist = np.where(rel_dist < 2*self.r/self.sigma, 2*self.r/self.sigma, rel_dist)
        np.fill_diagonal(rel_dist, 0.0)
    
        return rel_pos, rel_dist


    def energies(self, vel, rel_dist):
        """
        Calculates the energy of the system of particles at a given time instance.
        
        Parameters
        ----------
        vel : np.ndarray
            The velocities of the atoms in Cartesian space at all time instants.
        rel_dist : np.ndarray
            The relative distances between for all atom pairs in the lattice.
    
        Returns
        -------
        tuple
            The kinetic, potential, and total energies of the system at a given time instant.
        """

        if not isinstance(vel, np.ndarray):
            raise TypeError("The velocities must be stored in a numpy array!")
    
        if not isinstance(rel_dist, np.ndarray):
            raise TypeError("The relative distances must be stored in a numpy array!")

        # Reset the diagonal elements to not encounter division by zero afterwards:
        dist = rel_dist.copy()
        np.fill_diagonal(dist, 1.0)
        # Compute LJ potential energy for each pair:
        U = 4 * ((1 / dist)**12 - (1 / dist)**6)
        # Set diagonal to zero to avoid self-interaction
        np.fill_diagonal(U, 0)
        # Total potential energy of the system:
        U_total = 0.5 * np.sum(U)

        # Total kinetic energy of the system:
        kin_total = 0.5 * np.sum(vel**2)

        E_total = kin_total + U_total
    
        return kin_total, U_total, E_total


    def lj_force(self, rel_pos, rel_dist):
        """ 
        Calculates the Lennard-Jones force acting on each atom due to the interactions
        with the other atoms in the lattice.
    
        Parameters
        ----------
        rel_pos : np.ndarray
            The relative position vectors for all atom pairs in the lattice.
        rel_dist : np.ndarray
            The relative distances between for all atom pairs in the lattice.
    
        Returns
        -------
        np.ndarray
            The force vectors for each interacting atom-pair.
        """
    
        if not isinstance(rel_pos, np.ndarray):
            raise TypeError("The relative positions must be stored in a numpy array!")
    
        if not isinstance(rel_dist, np.ndarray):
            raise TypeError("The relative distances must be stored in a numpy array!")

        # Reset the diagonal elements to not encounter division by zero afterwards:
        dist = rel_dist.copy()
        np.fill_diagonal(dist, 1.0)
        # Compute the derivative of the LJ potential
        dU_dr = 4 * (6 / dist**7 - 12 / dist**13)
        # Set diagonal to zero to avoid self-interaction
        np.fill_diagonal(dU_dr, 0)
        # Compute force vectors
        forces = - dU_dr[:, :, np.newaxis] * (rel_pos / dist[:, :, np.newaxis])
        # Sum over all pairs to get the net force on each atom
        forces = np.sum(forces, axis=1)
                    
        return forces


    def collisions(self, pos, vel, t, collision_state):
        """
        Implements a hard-sphere potential for short-distanced interactions.
        Handles collisions between atoms by updating their velocities.
        Ensures that collisions are processed only once per collision event.
    
        Parameters
        ----------
        pos : np.ndarray
            Positions of the atoms at time step t.
        vel : np.ndarray
            Velocities of the atoms at time step t.
        t : int
            Current time step.
        collision_state : np.ndarray
            Matrix tracking whether two atoms are currently in a collision state.
    
        Returns
        -------
        tuple
            Updated velocities and collision state matrix.
        """

        if not isinstance(pos, np.ndarray):
            raise TypeError("The positions must be stored in a numpy array!")
    
        if not isinstance(vel, np.ndarray):
            raise TypeError("The velocities must be stored in a numpy array!")

        if not isinstance(t, int):
            raise TypeError("The number of time steps must be an integer!")
            
        if not isinstance(t, int):
            raise TypeError("The collision states must be stored in a numpy array!")

        tree = KDTree(pos[t, :, :])
        collision_pairs = tree.query_pairs(r=2 * self.r / self.sigma, output_type='ndarray')
    
        # Reset collision state for pairs that are no longer colliding
        previously_colliding_pairs = np.transpose(np.where(collision_state))
        
        for i, j in previously_colliding_pairs:
            
            if (i, j) not in collision_pairs and (j, i) not in collision_pairs:
                
                collision_state[i, j] = False
                collision_state[j, i] = False

        # Process collisions for the detected pairs
        for i, j in collision_pairs:
            
            if i != j and not collision_state[i, j]:
                
                # Compute relative position and velocity
                r_diff = pos[t, i, :] - pos[t, j, :]
                vel_diff = vel[t, i, :] - vel[t, j, :]
    
                # Elastic collision response
                vel[t, i, :] -= r_diff.dot(vel_diff) / r_diff.dot(r_diff) * r_diff
                vel[t, j, :] += r_diff.dot(vel_diff) / r_diff.dot(r_diff) * r_diff
    
                # Mark this pair as currently colliding
                collision_state[i, j] = True
                collision_state[j, i] = True
    
        return pos[t, :, :], vel[t, :, :], collision_state


    def periodicBC(self, pos):
        """
        Imposes periodic boundary conditions to the system of particles in the box, i.e.,
        particles should not reflect on the boundary. On the contrary, a particle exiting a
        boundary with certain momentum must enter the opposite boundary with the same momentum.
    
        Parameters
        ----------
        pos : np.ndarray
            The positions of the atoms in Cartesian space at all time instants.
    
        Returns
        -------
        np.ndarray
            The updated position array with imposed periodic 
            boundary conditions at a specific time instance.
        """    
        
        if not isinstance(pos, np.ndarray):
            raise TypeError("The positions must be stored in a numpy array!")
        
        # Use modulo operator to wrap around the box:
        pos = pos % (self.size / self.sigma)

        return pos

    def normal_autocorr(self, mu, sigma, tau, N):
        """
        Generates an autocorrelated sequence of Gaussian random numbers.

        Parameters:
        ----------
        mu : float
            Mean of each Gaussian random number.
        sigma : float
            Standard deviation of each Gaussian random number.
        tau : float
            Autocorrelation time.
        N : int
            Number of desired random numbers.

        Returns:
        --------
        np.ndarray
            Array of autocorrelated random numbers.
        """
        f = np.exp(-1.0 / tau)
        sequence = np.zeros(shape=(N,))
        sequence[0] = np.random.normal(0, 1)
        for i in range(1, N):
            sequence[i] = f * sequence[i - 1] + np.sqrt(1 - f ** 2) * np.random.normal(0, 1)
        return mu + sigma * sequence

    def autocorr_func(self, data, max_lag):
        """
        Computes the normalized autocorrelation function using the correct formula for finite time series.

        Parameters:
        ----------
        data : np.ndarray
            The input time series data.
        max_lag : int
            Maximum lag for autocorrelation calculation.

        Returns:
        --------
        np.ndarray
            Normalized autocorrelation function.
        """
        N = len(data)
        mean_A = np.mean(data)
        var_A = np.var(data)

        autocorr = np.zeros(max_lag)

        for t in range(1, max_lag + 1):  # 1 ≤ n ≤ N − t
            num = (N - t) * np.sum(data[:-t] * data[t:]) - np.sum(data[:-t]) * np.sum(data[t:])
            den1 = np.sqrt((N - t) * np.sum(data[:-t] ** 2) - np.sum(data[:-t]) ** 2)
            den2 = np.sqrt((N - t) * np.sum(data[t:] ** 2) - np.sum(data[t:]) ** 2)

            if den1 == 0 or den2 == 0:  # Avoid division by zero
                autocorr[t - 1] = 0
            else:
                autocorr[t - 1] = num / (den1 * den2)

        return autocorr

    def estimate_error(self, data, max_lag):
        """
        Estimates the correlation time τ and computes the raw error estimate.

        Parameters:
        ----------
        data : np.ndarray
            The input time series data.
        max_lag : int, optional
            Maximum lag for autocorrelation calculation.

        Returns:
        --------
        tuple:
            (Estimated correlation time τ, Raw error estimate)
        """
        # Compute autocorrelation function
        acf_values = self.autocorr_func(data, max_lag)

        # Check if ACF is constant or NaN
        if np.isnan(acf_values).any() or np.all(acf_values == acf_values[0]):
            raise ValueError("ACF is constant or contains NaN. Cannot perform exponential fit.")

        def exp_decay(t, a, b, c):
            return a * np.exp(-b * t) + c

        # Fit exponential decay to estimate correlation time
        lags = np.arange(max_lag)

        # Initial parameter guesses: a ~ max(acf), b ~ 1/50, c ~ min(acf)
        p0 = [acf_values[0], 1/50, acf_values[-1]]

        try:
            popt, _ = curve_fit(exp_decay, lags, acf_values, p0=p0)
            a_est, b_est, c_est = popt
            tau_estimated = 1 / b_est  # Compute correlation time τ
            if tau_estimated <= 0 or np.isnan(tau_estimated):
                raise RuntimeError("Invalid correlation time")
        except RuntimeError:
            tau_estimated = np.nan  # If fitting fails, return NaN

        # Compute error
        if not np.isnan(tau_estimated):
            tau_rounded = int(np.round(tau_estimated))
            thinned_data = data[::tau_rounded]  # Thin dataset using τ
            var_A = np.var(thinned_data)

            # Compute raw error (before conversion)
            error = np.sqrt((2 * tau_estimated / len(thinned_data)) * var_A)
        else:
            error = np.nan  # If τ estimation fails, return NaN

        return tau_estimated, error, thinned_data

    def data_blocking(self, data, tau, min_block_size=1):
        """
        Implements the data blocking method to estimate statistical errors.

        Parameters
        ----------
        data : np.ndarray
            The input data series for which the error is to be estimated.
        tau : float
            The estimated correlation time to determine max block size.
        min_block_size : int, optional
            The minimum block size to start with (default is 1).

        Returns
        -------
        np.ndarray
            Block sizes used.
        np.ndarray
            Estimated statistical errors for each block size.
        np.ndarray
            Block-averaged data (statistically independent).
        """
        max_block_size = int(np.round(tau*5))   # Ensure it's an integer
        N = len(data)
    
        block_sizes = []
        errors = []

        block_size = min_block_size
        while block_size <= max_block_size and block_size <= N // 2:
            num_blocks = N // block_size  # Only full blocks considered
        
            if num_blocks < 2:
                break  # Avoid division by zero when num_blocks - 1 = 0

            block_sizes.append(block_size)

            # Compute the mean of each block
            blocked_data = data[: num_blocks * block_size].reshape(num_blocks, block_size).mean(axis=1)

            # Compute ⟨a⟩ and ⟨a²⟩
            mean_blocked = np.mean(blocked_data)
            mean_sq_blocked = np.mean(blocked_data**2)

            # Compute error
            error = np.sqrt((mean_sq_blocked - mean_blocked**2) / (num_blocks - 1))
            errors.append(error)

            block_size += 1

        return np.array(block_sizes), np.array(errors), blocked_data

    def block_bootstrap(self, data, num_resamples = 1000):
        """
        Implements the block bootstrap method for resampling independent data.

        Parameters:
        ----------
        data : np.ndarray
            The input block-averaged dataset (should be independent after data blocking).
        num_resamples : int
            Number of bootstrap resamples to generate.

        Returns:
        -------
        np.ndarray
            Bootstrap resampled datasets (num_resamples x len(data)).
        """
        N = len(data)  # Number of independent blocks
        bootstrap_samples = np.zeros((num_resamples, N))  # Store resampled datasets
        bootstrap_means = np.zeros(num_resamples)  # Store means of each bootstrap dataset

        # Perform bootstrap resampling
        for i in range(num_resamples):
            resampled_data = np.random.choice(data, size=N, replace=True)  # Sample with replacement
            bootstrap_samples[i] = resampled_data

        return bootstrap_samples
    
    def time_evol(self, init_pos, init_vel, method, units):
        """
        Computes the time evolution of the system within a time interval, 
        i.e. the particles positions/velocities/relative positions/distances between
        the particles, as well as the corresponding kinetic/potential, and total energies
        of the system, at each time instance. Implements the Euler or velocity-Verlet 
        methods to solve the system of equations that govern the particles' motion.
    
        Parameters
        ----------
        init_pos : np.ndarray
            The initial positions of the atoms in Cartesian space.
        init_vel : np.ndarray
            The initial velocities of the atoms in Cartesian space.
        method: str
            The chosen approximation method to compute the time evolutions of the system.
            Enter 'euler' to implement the Euler method and 'verlet' to implement the
            velocity Verlet method.
        units: str
            If 'on', converts back to S.I. units. If 'off', keeps the dimensionless units.
    
        Returns
        -------
        tuple
            The position, velocity, relative position, and relative distance arrays 
            at all time instances within the time interval of the system's evolution.
        """

        if not isinstance(method, str):
            raise TypeError("The chosen method must be a string input!")
            
        if method not in ['euler', 'verlet']:
            raise TypeError("Please enter one of the two methods: 'euler' or 'verlet'")

        if not isinstance(units, str):
            raise TypeError("The option must be a string input!")
            
        if units not in ['on', 'off']:
            raise TypeError("Please enter one of the two inputs: 'on' or 'off'")
        
        # Specify the dimensions of the arrays that will contain the positions/velocities,
        #relative positions/distances, and kinetic/potential/total energies.
        pos = np.zeros((self.tsteps, self.N, self.dim))  # positions
        vel = np.zeros((self.tsteps, self.N, self.dim))  # velocities
        rel_pos = np.zeros((self.tsteps, self.N, self.N, self.dim))  # relative positions
        rel_dist = np.zeros((self.tsteps, self.N, self.N))  # relative distances
        energy_values = np.zeros((self.tsteps, 3)) 
    
        # Initialize the positions/velocities.
        pos[0] = init_pos
        vel[0] = init_vel
        
        if method == 'euler':
        
            # Implement the Euler method to compute positions/velocities, relative
            #positions/distances, and energies at each time instant:
            for t in tqdm(range(1, self.tsteps), desc = "Time Evolution (Euler)"):
                
                # Compute relative positions/distances at a specific time instance:
                rel_pos_t, rel_dist_t = self.atomic_distances(pos[t - 1])
                # Store the relative positions/distances, and energies:
                rel_pos[t - 1], rel_dist[t - 1] = rel_pos_t, rel_dist_t
                energy_values[t - 1] = self.energies(vel[t-1], rel_dist[t-1]) 
        
                # Implement the Euler method and store the positions/velocities
                # at all time instances:
                pos[t] = pos[t - 1] + vel[t - 1] * self.Dt
                vel[t] = vel[t - 1] + self.lj_force(rel_pos_t, rel_dist_t) * self.Dt
        
                # Apply periodic B.C. if particles cross a boundary:
                pos[t] = self.periodicBC(pos[t])

        elif method == 'verlet':
            
            # Implement the velocity-Verlet method to compute positions/velocities, relative
            #positions/distances, and energies at each time instant:

            # Compute relative positions/distances at the previous time step:
            rel_pos_t1, rel_dist_t1 = self.atomic_distances(pos[0])
            # Compute the Lennard-Jones forces:
            forces_t1 = self.lj_force(rel_pos_t1, rel_dist_t1)
            
            for t in tqdm(range(1, self.tsteps), desc = "Time Evolution (Verlet)"):
                
                # Store the relative positions/distances, and energies:
                rel_pos[t-1], rel_dist[t-1] = rel_pos_t1, rel_dist_t1
                energy_values[t-1] = self.energies(vel[t-1], rel_dist[t-1]) 
                
                # Implement the Verlet method and store the positions/velocities
                # at all time instances:
                pos[t] = pos[t - 1] + vel[t - 1] * self.Dt + forces_t1 * (self.Dt**2 / 2)

                # Compute relative positions/distances at the current time step:
                rel_pos_t2, rel_dist_t2 = self.atomic_distances(pos[t])
                forces_t2 = self.lj_force(rel_pos_t2, rel_dist_t2)
                
                vel[t] = vel[t-1] + 0.5 * (forces_t1 + forces_t2) * self.Dt

                # Update relative positions/distances for next iteration:
                rel_pos_t1, rel_dist_t1 = rel_pos_t2, rel_dist_t2 
                forces_t1 = forces_t2  # Update forces for next iteration
        
                # Apply periodic B.C. if particles cross a boundary:
                pos[t] = self.periodicBC(pos[t])
    
        # Compute final relative positions, distances, and energies:x
        rel_pos[-1], rel_dist[-1] = self.atomic_distances(pos[-1])
        energy_values[-1] = self.energies(vel[-1], rel_dist[-1]) 

        if units == 'on':
            
            # Convert back to non-dimensionless units:
            pos *= self.sigma
            vel *= np.sqrt(self.epsilon / self.m)
            rel_pos *= self.sigma
            rel_dist *= self.sigma
            energy_values *= self.epsilon
    
        return pos, vel, rel_pos, rel_dist, energy_values
import numpy as np
import matplotlib.pyplot as plt
import random as rand
import math
from tqdm import tqdm  # Import tqdm for the progress bar

class IsingModel():
    '''Maintain a 2D Ising model with n*n spins and coupling constant beta'''
    def __init__(self, n, beta):
        # n: the size of the lattice
        # beta: the coupling constant
        self.n = n
        self.beta = beta
        # -1 or 1 lattice with spin config
        self._lattice_spin_config = np.random.choice([-1,1], size=(n,n)) # uniformly
        # print(self._lattice_spin_config)
        self._energy = self.get_energy()
        self._samples = []
    
    def get_energy(self):
        '''Return the energy of the current lattice'''
        energy = 0
        # we should pay attention to the boundary condition
        # get the edge list
        edge_set = set()
        for i in range(self.n):
            for j in range(self.n):
                if i != self.n-1:
                    edge_set.add(((i,j), (i+1,j)))
                if j != self.n-1:
                    edge_set.add(((i,j), (i,j+1)))
        # print(edge_set)
        # print(len(edge_set)) # 2*n*(n-1)
        # calculate the energy
        for edge in edge_set:
            energy += -self._lattice_spin_config[edge[0]]*self._lattice_spin_config[edge[1]]

        return energy
    
    def get_magnetization(self):
        '''Return the magnetization of the current lattice, but not used in the simulation'''
        return np.sum(self._lattice_spin_config)
    
    def _get_neighbor(self, i, j):
        '''Return the neighbor of vertex (i,j)'''
        neighbor_list = []
        if i != 0:
            neighbor_list.append((i-1,j))
        if i != self.n-1:
            neighbor_list.append((i+1,j))
        if j != 0:
            neighbor_list.append((i,j-1))
        if j != self.n-1:
            neighbor_list.append((i,j+1))
        return neighbor_list
    
    def _get_neighbor_spins(self, i, j):
        '''Return the spin of the neighbors of vertex (i,j)'''
        neighbor_list = self._get_neighbor(i,j)
        neighbor_spin_list = []
        for neighbor in neighbor_list:
            neighbor_spin_list.append(self._lattice_spin_config[neighbor])
        return neighbor_spin_list
    
    def _get_neighbor_spin_sum(self, i, j):
        '''Return the spin sum of the neighbors of vertex (i,j), for cond prob calculation'''
        neighbor_spin_list = self._get_neighbor_spins(i,j)
        return np.sum(neighbor_spin_list)
        
    
    def _get_positive_conditional_probability(self, i, j):
        '''Return the conditional probability of vertex (i,j) for spin value +1 given its neighbor in the current lattice'''
        neighbor_spin_sum = self._get_neighbor_spin_sum(i,j)
        return 1/(1+math.exp(-2*self.beta*neighbor_spin_sum))
    
    def _get_next_in_sweeping(self, i, j):
        '''Return the next vertex in the sweeping order'''
        if j == self.n-1 and i == self.n-1:
            return (0, 0)
        elif j == self.n-1:
            return (i+1, 0)
        else:
            return (i, j+1)
    
    def gibbs_sampling(self, warmup_sweepings, save_sweeping_interval, steps):
        '''Return a list of samples of the lattice spin configuration using Gibbs sampling'''
        # steps: steps = max_sweepings * n**2, but we explicitly specify it.
        self._samples = []
        warmup_sweepings = min(warmup_sweepings, (steps -1)/ self.n**2)
        warmup_steps = warmup_sweepings*self.n**2
        
        save_steps = save_sweeping_interval*self.n**2
        i, j = 0, 0
        # for sweep, 
        for step in range(steps):
            # for each vertex, sample its spin value
            # sample the spin value of vertex (i,j)
            p = self._get_positive_conditional_probability(i,j)
            if rand.random() < p:
                self._lattice_spin_config[(i,j)] = 1
            else:
                self._lattice_spin_config[(i,j)] = -1
            # move to the next vertex
            i, j = self._get_next_in_sweeping(i,j)
            # if we have finished a sweeping, we should save the sample
            if step >= warmup_steps and step % save_steps == 0:
                # print('step', step, 'sample', self._lattice_spin_config)
                self._samples.append(self._lattice_spin_config.copy())

    def gibbs_sampling_max_sweepings(self, warmup_sweepings, save_sweeping_interval, max_sweepings):
        self.gibbs_sampling(warmup_sweepings, save_sweeping_interval=save_sweeping_interval, steps=max_sweepings*self.n**2)

    def simulate(self, warmup_sweepings, save_sweeping_interval, max_sweepings, temperature_range):
        '''Simulate the Ising model using Gibbs sampling over a range of temperatures'''
        temperatures = np.linspace(temperature_range[0], temperature_range[1], max_sweepings)
        energies = []
        magnetizations = []

        for temp in tqdm(temperatures, desc='Temperature Progress'):
            self.beta = temp
            self._lattice_spin_config = np.random.choice([-1, 1], size=(self.n, self.n))  # Reset initial configuration
            self.gibbs_sampling_max_sweepings(warmup_sweepings, save_sweeping_interval, max_sweepings)
            energies.append(self.get_energy())
            magnetizations.append(self.get_magnetization())

        self.plot_energy_and_magnetization(temperatures, energies, magnetizations)

    def plot_energy_and_magnetization(self, temperatures, energies, magnetizations):
        '''Plot energy and magnetization as functions of temperature'''
        plt.figure(figsize=(10, 6))

        plt.subplot(2, 1, 1)
        plt.plot(temperatures, energies, label='Energy')
        plt.title('Energy vs Temperature')
        plt.xlabel('Temperature')
        plt.ylabel('Energy')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(temperatures, magnetizations, label='Magnetization')
        plt.title('Magnetization vs Temperature')
        plt.xlabel('Temperature')
        plt.ylabel('Magnetization')
        plt.legend()

        plt.tight_layout()
        plt.show()

# Example Usage
ising_model = IsingModel(n=64, beta=0.8)
ising_model.simulate(warmup_sweepings=5, save_sweeping_interval=1, max_sweepings=100, temperature_range=(0.2, 2.5))

import os
import numpy as np
import matplotlib.pyplot as plt
import random as rand
import math
from matplotlib.animation import FuncAnimation

class IsingModel():
    def __init__(self, n, beta):
        self.n = n
        self.beta = beta
        self._lattice_spin_config = np.random.choice([-1, 1], size=(n, n))
        self._energy = self.get_energy()
        self._samples = []

    def get_energy(self):
        energy = 0
        edge_set = set()
        for i in range(self.n):
            for j in range(self.n):
                if i != self.n - 1:
                    edge_set.add(((i, j), (i + 1, j)))
                if j != self.n - 1:
                    edge_set.add(((i, j), (i, j + 1)))

        for edge in edge_set:
            energy += -self._lattice_spin_config[edge[0]] * self._lattice_spin_config[edge[1]]

        return energy

    def _get_neighbor(self, i, j):
        neighbor_list = []
        if i != 0:
            neighbor_list.append((i - 1, j))
        if i != self.n - 1:
            neighbor_list.append((i + 1, j))
        if j != 0:
            neighbor_list.append((i, j - 1))
        if j != self.n - 1:
            neighbor_list.append((i, j + 1))
        return neighbor_list

    def _get_neighbor_spins(self, i, j):
        neighbor_list = self._get_neighbor(i, j)
        neighbor_spin_list = []
        for neighbor in neighbor_list:
            neighbor_spin_list.append(self._lattice_spin_config[neighbor])
        return neighbor_spin_list

    def _get_neighbor_spin_sum(self, i, j):
        neighbor_spin_list = self._get_neighbor_spins(i, j)
        return np.sum(neighbor_spin_list)

    def _get_positive_conditional_probability(self, i, j):
        neighbor_spin_sum = self._get_neighbor_spin_sum(i, j)
        return 1 / (1 + math.exp(-2 * self.beta * neighbor_spin_sum))

    def _get_next_in_sweeping(self, i, j):
        if j == self.n - 1 and i == self.n - 1:
            return (0, 0)
        elif j == self.n - 1:
            return (i + 1, 0)
        else:
            return (i, j + 1)

    def gibbs_sampling(self, warmup_sweepings, save_sweeping_interval, steps):
        self._samples = []
        warmup_sweepings = min(warmup_sweepings, (steps - 1) / self.n ** 2)
        warmup_steps = warmup_sweepings * self.n ** 2
        save_steps = save_sweeping_interval * self.n ** 2
        i, j = 0, 0

        for step in range(steps):
            p = self._get_positive_conditional_probability(i, j)
            if rand.random() < p:
                self._lattice_spin_config[(i, j)] = 1
            else:
                self._lattice_spin_config[(i, j)] = -1

            i, j = self._get_next_in_sweeping(i, j)

            if step >= warmup_steps and step % save_steps == 0:
                self._samples.append(self._lattice_spin_config.copy())

    def gibbs_sampling_max_sweepings(self, warmup_sweepings, save_sweeping_interval, max_sweepings):
        self.gibbs_sampling(warmup_sweepings, save_sweeping_interval=save_sweeping_interval, steps=max_sweepings * self.n ** 2)

    def simulate(self, warmup_sweepings, save_sweeping_interval, max_sweepings):
        self._lattice_spin_config = np.random.choice([-1, 1], size=(self.n, self.n))
        self.gibbs_sampling_max_sweepings(warmup_sweepings, save_sweeping_interval, max_sweepings)

    def plot_samples(self, mode='print_and_last', lasts=1):
        print('The number of samples is', len(self._samples))
        if mode == 'print_and_all':
            for sample in self._samples:
                print(sample)
            return
        elif mode == 'print_and_last':
            for sample in self._samples[-lasts:]:
                print(sample)
            return
        elif mode == 'plot_and_last':
            for sample in self._samples[-lasts:]:
                plt.imshow(sample)
                plt.show()
            return

    def animate_samples(self, interval, save_path=None):
        fig, ax = plt.subplots()

        def update(frame):
            ax.clear()
            im = ax.imshow(self._samples[frame], cmap='viridis')  # 添加 cmap='viridis'
            ax.set_title(f'Step {frame}')

            n_beta_text = f'n = {self.n}, β = {self.beta:.3f}'
                    
            ax.text(0.02, 0.98, n_beta_text, transform=fig.transFigure,
                    fontsize=10, horizontalalignment='left', verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.2))

                    
        animation = FuncAnimation(fig, update, frames=len(self._samples), interval=interval, repeat=False)

        if save_path:
            absolute_path = os.path.abspath(save_path)
            animation.save(absolute_path, writer='pillow', fps=1000 // interval)
            print(f"Animation saved at: {absolute_path}")
        else:
            plt.show()

# Example Usage with Animation for different β values
betas = [-5, 0.2, 0.441, 0.6]

for beta_value in betas:
    ising_model = IsingModel(n=300, beta=beta_value)
    ising_model.simulate(warmup_sweepings=5, save_sweeping_interval=1, max_sweepings=100)

    # Set the interval for displaying each frame in milliseconds
    animation_interval = 500

    # Generate and display the animation with n and β values
    save_path = f'animation_graph/ising_model_animation_n300_beta{beta_value:.3f}.gif'
    ising_model.animate_samples(interval=animation_interval, save_path=save_path)
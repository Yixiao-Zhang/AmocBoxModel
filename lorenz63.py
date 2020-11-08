#!/usr/bin/env python3

from typing import Mapping
from itertools import count

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import bmode

class Lorenz63(bmode.BalanceModel):
    class Lorenz63State(bmode.State):
        _variables_ = ['x', 'y', 'z']

    def __init__(self, sigma, rho, beta, *args, **kwargs):
        super().__init__(*args, **kwargs)

        @self.add_flux(sink='x')
        def _tend_x(state):
            return sigma*(state['y'] - state['x'])

        @self.add_flux(sink='y')
        def _tend_y(state):
            return state['x']*(rho - state['z']) - state['y']

        @self.add_flux(sink='z')
        def _tend_z(state):
            return state['x']*state['y'] - beta*state['z']

    def run(self, ic:Mapping[str, float], nstep):
        ic = self.Lorenz63State.from_dict(ic)
        states = self.Lorenz63State(
            shape=(len(self.Lorenz63State._variables_), nstep),
            dtype=float
        )

        states[:, 0] = ic[:]
        for i, state in zip(range(1, nstep), self.iter_states(ic)):
            states[:, i] = state[:]
        return states

parameters = dict(sigma=10, rho=28, beta=8/3)

def plot_traj():
    ic = dict(x=5.2, y=5.2, z=28)
    dt, nstep = 0.01, int(1.0e4)
    schemes = ('rk4', 'euler', 'ab2', 'leapfrog')

    fig = plt.figure(figsize=(8, 8))
    for k, scheme in enumerate(schemes):
        model = Lorenz63(scheme=scheme, dt=dt, **parameters)
        states = model.run(ic=ic, nstep=nstep)
        ax = fig.add_subplot(2, 2, k+1, projection='3d')

        ax.set_title(scheme.capitalize())
        ax.plot(*(states[v] for v in 'xyz'),
            linewidth=0.3, marker='o', markersize=0.2)
    # ax = fig.gca()
    # ax.plot('x', data=states)
    # plt.draw()
    plt.tight_layout()
    plt.savefig('schemes.eps')

    # clearing
    fig.clear()
    plt.close(fig)

def plot_sens():
    fig, ax_matrix = plt.subplots(3, 3, figsize=(8, 6), sharex=True)

    dt, nstep = 0.01, int(1.5e3)
    time = np.arange(nstep)*dt
    model = Lorenz63(dt=dt, **parameters)

    ic0 = dict(x=50, y=50, z=50)

    ref = model.run(ic=ic0, nstep=nstep)

    for j, error in enumerate((0.01, 0.1, 1.0)):
        ax_array = ax_matrix[:, j]
        ic = {key:value+error for key, value in ic0.items()}
        states = model.run(ic=ic, nstep=nstep)

        tot_error = np.sum(np.abs(ref - states), axis=0)
        time_error = np.where(tot_error > 6)[0][0]*dt

        print(time_error)

        for i, v in enumerate('xyz'):
            ax = ax_array[i]
            ax.plot(time, ref[v],
                linewidth=0.3, marker='o', markersize=0.2, label='Reference')
            ax.plot(time, states[v],
                linewidth=0.3, marker='o', markersize=0.2, label='Perturbed')
            ax.axvline(x=time_error, linestyle='--', color='k')
            if j == 0:
                ax.set_ylabel(v)
            if (i, j) == (0, 0):
                ax.legend()


        ax_array[0].set_title(f'E$_i$ = {error:0.02f}')
        ax_array[-1].set_xlabel('Time')


    plt.tight_layout()
    plt.savefig('butterfly.eps')

    # clearing
    fig.clear()
    plt.close(fig)

def main():
    plot_traj()
    plot_sens()

if __name__ == '__main__':
    main()

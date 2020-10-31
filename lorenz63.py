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

def main():
    ic = dict(x=5.2, y=5.2, z=28)
    parameters = dict(sigma=10, rho=28, beta=8/3)
    dt, nstep = 0.01, int(1.0e4)
    schemes = ('rk4', 'euler', 'ab2', 'leapfrog')

    fig = plt.figure()
    for k, scheme in enumerate(schemes):
        model = Lorenz63(scheme=scheme, dt=dt, **parameters)
        states = model.run(ic=ic, nstep=nstep)
        ax = fig.add_subplot(2, 2, k+1, projection='3d')

        ax.set_title(scheme)
        ax.plot(*(states[v] for v in 'xyz'),
            linewidth=0.5, marker='o', markersize=0.5)
    # ax = fig.gca()
    # ax.plot('x', data=states)
    # plt.draw()
    plt.show()

if __name__ == '__main__':
    main()
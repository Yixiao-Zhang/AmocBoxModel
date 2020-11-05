# -*- coding: utf-8 -*-
'''A general and abstract framework for box models. The core idea is
that the system is governed by the laws of conservation.
'''

from typing import AnyStr, Callable
from collections import defaultdict

import numpy as np

class State(np.ndarray):
    _variables_ = []

    @classmethod
    def from_dict(cls, dictlike):
        return np.asarray([dictlike[varbl]
            for varbl in cls._variables_]).view(cls)

    def __getitem__(self, key):
        if isinstance(key, str):
            key = self._variables_.index(key)
        return np.ndarray.__getitem__(self, key)

    def __str__(self):
        return ''.join((type(self).__name__, '(',
            ', '.join('{0:s}:{1:f}'.format(varbl, self[varbl])
                for varbl in self._variables_),
            ')'))

class BalanceBase:
    def __init__(self):
        self.fluxes = list()
        self.inertias = defaultdict(lambda : 1.0)

    def set_inertia(self, **kwargs):
        for varbl, inertia in kwargs.items():
            self.inertias[varbl] = inertia

    def add_flux(self, sink:AnyStr=None, source:AnyStr=None):
        def wrapper(func):
            self.fluxes.append((func, source, sink))
            return func
        return wrapper

    def tend(self, state:State):
        convs = defaultdict(float)
        for flux, source, sink in self.fluxes:
            conv = flux(state)
            # print('{0:e}'.format(conv), flux.__name__,
            #     source, '{0:e}'.format(-conv/self.inertias[source]),
            #     sink,'{0:e}'.format(conv/self.inertias[sink]))
            if source is not None:
                convs[source] -= conv
            if sink is not None:
                convs[sink] += conv
        result = defaultdict(float)
        for varbl, conv in convs.items():
            result[varbl] = convs[varbl]/self.inertias[varbl]
        # print(result)
        return type(state).from_dict(result)

class OdeSolver:
    def __init__(self, tend:Callable, dt:float, scheme:AnyStr):
        self.tend = tend
        self.dt = dt
        self.scheme = getattr(self, scheme)

    @staticmethod
    def euler(state, dt, tend):
        while True:
            state += dt*tend(state)
            yield state

    @staticmethod
    def rk4(state, dt, tend):
        while True:
            k1 = dt*tend(state)
            k2 = dt*tend(state+k1/2)
            k3 = dt*tend(state+k2/2)
            k4 = dt*tend(state+k3)
            state = state + (k1+2*(k2+k3)+k4)/6
            yield state

    @staticmethod
    def leapfrog(state, dt, tend):
        prev, state = (state, state + dt*tend(state)) # euler
        while True:
            yield state
            prev, state = (state, prev + (2*dt)*tend(state))

    @staticmethod
    def ab2(state, dt, tend):
        tprev = dt*tend(state)
        prev, state = (state, state + tprev) # euler
        tnow = dt*tend(state)
        while True:
            prev, state = (state, state + 1.5*tnow -0.5*tprev)
            yield state
            tprev, tnow = (tnow, dt*tend(state))

    def iter_states(self, state:State):
        return self.scheme(state, self.dt, self.tend)

class BalanceModel(BalanceBase, OdeSolver):
    def __init__(self, dt:float, scheme:AnyStr='rk4'):
        BalanceBase.__init__(self)
        OdeSolver.__init__(self, tend=self.tend, dt=dt, scheme=scheme)

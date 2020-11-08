#!/usr/bin/env python3

import math
from typing import Mapping
from itertools import count

import numpy as np
import matplotlib.pyplot as plt

import bmode

# earth radius, in meter
rds = 6.371e6

# ocean fraction
fraco = np.array([0.16267, 0.22222, 0.21069])

# box boundaries, in rad
latia = np.array([-90.0, -30.0, 45.0, 90.0])*(np.pi/180.0)
latio = np.array([-60.0, -30.0, 45.0, 80.0])*(np.pi/180.0)

# sine of latitdues
sinlatia = np.sin(latia)
sinlata = (sinlatia[1:] + sinlatia[:-1])/2

# box mass centers, in rad
lata = np.arcsin(sinlata)

# area of each atmospheric box, in square meter
areaa = (2 * np.pi * rds**2)*np.diff(sinlatia)

# meridional distance between boxes, in meter
ydis = rds*np.diff(lata)

# perimeter of boundaries, in meter
perim = (2 * np.pi* rds)*np.cos(latia[1:-1])

# atmospheric heat capacities, in J/K
ca = areaa * (5300 * 1004.0)

# area of each oceanic box
areao = np.diff(np.sin(latio))*(80.0*np.pi/180.0*rds**2)
areao = np.array([*areao, areao[1]])

# height of each oceanic box
z1, z2 = 600, 4000
heighto = np.array([z2, z1, z2, z2-z1])

# volume of each oceanic box
vo = areao*heighto

# heat capacity of unit mass sea water, in J/(kg K)
cswt = 1025 * 4200

# oceanic heat capacities, in J/K
co = vo * cswt

def lht(tc, delta_y):
    'Latent heat transport'
    sat = 6.112*math.exp(17.67*tc/(tc + 243.5))
    dqsdt = (243.5*17.67*0.622*1.0e-3)*sat/(tc + 243.5)**2
    return (1.5*5.1e17*0.8/delta_y)*dqsdt

class Amoc(bmode.BalanceModel):
    class AmocState(bmode.State):
        _variables_ = ['tos', 'tom', 'ton', 'tod',
                        'sos', 'som', 'son', 'sod',
                        'tas', 'tam', 'tan'
                    ]

        def __getitem__(self, name):
            if name in ('lht', 'amoc'):
                return getattr(self, name)
            else:
                return super().__getitem__(name)

        @property
        def lht(self):
            if '_lht' not in self.__dict__:
                tc0 = (self['tam']*(latia[1] - lata[0])
                    + self['tas']*(lata[1]-latia[1]))/(lata[1] - lata[0])
                lht0 = lht(tc0, ydis[0])
                tc1 = (self['tam']*(latia[2] - lata[2])
                    + self['tan']*(lata[1]-latia[2]))/(lata[1] - lata[2])
                lht1 = lht(tc1, ydis[1])
                self._lht = np.array([lht0, lht1])
                self.setflags(write=False)
            return self._lht

        @property
        def amoc(self):
            if '_amoc' not in self.__dict__:
                phi = 1.5264e10*(8.0e-4*(self['son'] - self['sos'])
                    - 1.5e-4*(self['ton'] - self['tos']))
                phi = phi*(phi > 0)
                self._amoc = phi
                self.setflags(write=False)
            return self._amoc

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        inertias = [*co, *vo, *ca]
        self.set_inertia(
            **dict(zip(self.AmocState._variables_, inertias))
        )

        do_rad = True # radiative transfer
        do_srf_hlf = True # heat exchange between the ocean and the atmosphere
        do_atm_ht = True # atmospheric heat transport
        do_salt_raise = True # evaporation increases the salinity of the ocean
        do_salt_amoc = True # salt transport by AMOC
        do_heat_amoc = True # heat transport by AMOC

        if do_rad:
            @self.add_flux(sink='tas')
            def _rad_s(state):
                return areaa[0]*(320.0*(1 - 0.4) - 213.35 - 2.22*state['tas'])

            @self.add_flux(sink='tam')
            def _rad_m(state):
                return areaa[1]*(390.0*(1 - 0.25) - 213.35 - 2.22*state['tam'])

            @self.add_flux(sink='tan')
            def _rad_n(state):
                return areaa[2]*(270.0*(1 - 0.42) - 213.35 - 2.22*state['tan'])

        if do_srf_hlf:
            @self.add_flux(source='tas', sink='tos')
            def _srf_hflx_s(state):
                return areao[0]*(10 - 50*(state['tos'] - state['tas']))

            @self.add_flux(source='tam', sink='tom')
            def _srf_hflx_m(state):
                return areao[1]*(70 - 50*(state['tom'] - state['tam']))

            @self.add_flux(source='tan', sink='ton')
            def _srf_hflx_n(state):
                return areao[2]*(20 - 50*(state['ton'] - state['tan']))

        if do_atm_ht:
            @self.add_flux(source='tam', sink='tas')
            def _aht_m2s(state):
                return (perim[0])*(
                        (2.5e13/ydis[0])*(state['tam'] - state['tas'])
                        + state.lht[0]
                )

            @self.add_flux(source='tam', sink='tan')
            def _aht_m2n(state):
                return (perim[1])*(
                    (2.5e13/ydis[1])*(state['tam'] - state['tan'])
                    + state.lht[1]
            )

        if do_salt_raise:
            @self.add_flux(source='sos', sink='som')
            def _ers_s2m(state):
                return 34.9*perim[0]*(80.0/(360.0*2.5e9))*state.lht[0]

            @self.add_flux(source='son', sink='som')
            def _ers_n2m(state):
                return 34.9*perim[1]*(2.5*80.0/(360.0*2.5e9))*state.lht[1]

        if do_heat_amoc:
            @self.add_flux(sink='tos', source='tod')
            def _heat_amoc_d2s(state):
                return cswt*state.amoc*state['tod']

            @self.add_flux(sink='tom', source='tos')
            def _heat_amoc_s2m(state):
                return cswt*state.amoc*state['tos']

            @self.add_flux(sink='ton', source='tom')
            def _heat_amoc_m2n(state):
                return cswt*state.amoc*state['tom']

            @self.add_flux(sink='tod', source='ton')
            def _heat_amoc_n2d(state):
                return cswt*state.amoc*state['ton']

        if do_salt_amoc:
            @self.add_flux(sink='sos', source='sod')
            def _salt_amoc_d2s(state):
                return state.amoc*state['sod']

            @self.add_flux(sink='som', source='sos')
            def _salt_amoc_s2m(state):
                return state.amoc*state['sos']

            @self.add_flux(sink='son', source='som')
            def _salt_amoc_m2n(state):
                return state.amoc*state['som']

            @self.add_flux(sink='sod', source='son')
            def _salt_amoc_n2d(state):
                return state.amoc*state['son']

    def run(self, ic:Mapping[str, float], nstep, nhist=100, nprint=1000):

        time = np.arange(0, nstep, nhist)*self.dt

        ic = self.AmocState.from_dict(ic)
        states = self.AmocState(
            shape=(len(self.AmocState._variables_), nstep//nhist),
            dtype=float
        )

        states[:, 0] = ic[:]
        for i, state in zip(range(1, nstep), self.iter_states(ic)):
            qut, rmd = divmod(i, nhist)
            if rmd == 0:
                states[:, qut] = state
            if i%nprint == 0:
                print('*'*16)
                print('NSTEP', i)
                print(state)
        return time, states

def main():
    ic_values = [4.777404031, 24.42876625, 2.66810894, 2.67598915,
        34.40753555, 35.62585068, 34.92513657, 34.91130066, 4.67439556,
        23.30437851, 0.94061828
    ]
    ic = dict(zip(Amoc.AmocState._variables_, ic_values))

    NSEC_YR = 86400*365
    dt_yr = 0.01

    dt, nstep = dt_yr*NSEC_YR, int(5.0e5)
    scheme = 'rk4'

    # run until the equilibrium has been reached
    model = Amoc(scheme=scheme, dt=dt)
    time0, states0 = model.run(ic=ic, nstep=nstep)

    print('=*16')
    print('salinity drop experiment')
    print('=*16')
    # salinity drop experiment
    ic1 = np.array(states0[:, -1])
    ic1[Amoc.AmocState._variables_.index('son')] -= 0.7
    ic1 = dict(zip(Amoc.AmocState._variables_, ic1))
    time1, states1 = model.run(ic=ic1, nstep=nstep)

    time1 += time0[-1]

    for time in (time0, time1):
        time /= NSEC_YR # in year

    # for legend
    def txt2tex(txt):
        return f'${txt[0].capitalize()}_{txt[2].capitalize()}^{txt[1]}$'

    # colors for different regions
    colors = [plt.get_cmap("tab10")(i) for i in range(4)]

    # the first 3000 years
    def xy0(v):
        nstep = 3000
        return time0[:nstep], states0[v][:nstep]

    # the whole 10000 years
    def xy1(v):
        return map(np.concatenate,
            ((time0, time1), (states0[v], states1[v])))

    for j, xy in enumerate((xy0, xy1)):

        fig, (ax0, ax1, ax2) = plt.subplots(ncols=1, nrows=3, figsize=(8, 8))

        for i, v in enumerate(('tos', 'tom', 'ton', 'tod')):
            ax0.plot(*xy(v), label=txt2tex(v),
                linestyle='--', color=colors[i])
        for i, v in enumerate(('tas', 'tam', 'tan')):
            ax0.plot(*xy(v), label=txt2tex(v),
                color=colors[i])
        for i, v in enumerate(('sos', 'som', 'son', 'sod')):
            ax1.plot(*xy(v), label=txt2tex(v),
                color=colors[i])

        time, amoc = xy('amoc')
        ax2.plot(time, amoc*1.0e-6, color=colors[0])

        ax0.set_ylabel(r'Temperature (C$^\circ$)')
        ax1.set_ylabel(r'Salinity (psu)')
        ax2.set_ylabel(r'$\Phi$ (10$^6$ Sv)')
        ax2.set_xlabel(r'Time (yr)')

        for ax in (ax0, ax1):
            ax.legend(ncol=2)
            ax.grid()
        ax2.grid()

        plt.tight_layout()
        plt.savefig(f'amoc{j}.eps')
        fig.clear()
        plt.close(fig)
    # plt.show()

if __name__ == '__main__':
    main()

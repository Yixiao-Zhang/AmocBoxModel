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
lati = np.array([-60.0, -30.0, 45.0, 80.0])*(np.pi/180.0)

# sine of latitdues
sinlati = np.sin(lati)
sinlat = (sinlati[1:] + sinlati[:-1])/2

# box mass centers, in rad
lat = np.arcsin(sinlat)

# area of each atmospheric box, in square meter
areaa = (2 * np.pi * rds**2)*np.diff(sinlati)

# meridional distance between boxes, in meter
ydis = rds*np.diff(lat)

# perimeter of boundaries, in meter
perim = (2 * np.pi* rds)*np.cos(lati[1:-1])

# atmospheric heat capacities, in J/K
ca = areaa * (5300 * 1004.0)

# area of each oceanic box
areao = areaa*fraco
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

        @property
        def lht(self):
            if '_lht' not in self.__dict__:
                tc0 = (self['tam']*(lati[1] - lat[0])
                    + self['tas']*(lat[1]-lati[1]))/(lat[1] - lat[0])
                lht0 = lht(tc0, ydis[0])
                tc1 = (self['tam']*(lati[2] - lat[2])
                    + self['tan']*(lat[1]-lati[2]))/(lat[1] - lat[2])
                lht1 = lht(tc1, ydis[1])
                self._lht = np.array([lht0, lht1])
                self.setflags(write=False)
            return self._lht

        @property
        def amoc(self):
            if '_amoc' not in self.__dict__:
                phi = 1.5264e10*(8.0e-4*(self['son'] - self['sos'])
                    - 1.5e-4*(self['ton'] - self['tos']))
                phi = max(0.0, phi)
                self._amoc = phi
                self.setflags(write=False)
            return self._amoc

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        inertias = [*co, *vo, *ca]
        self.set_inertia(
            **dict(zip(self.AmocState._variables_, inertias))
        )

        print(self.inertias)

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
                return (perim[0])*((2.5e13/ydis[0])*(state['tam'] - state['tas']) + state.lht[0])

            @self.add_flux(source='tam', sink='tan')
            def _aht_m2n(state):
                return (perim[1])*((2.5e13/ydis[1])*(state['tam'] - state['tan']) + state.lht[1])

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

    def run(self, ic:Mapping[str, float], nstep):
        ic = self.AmocState.from_dict(ic)
        states = self.AmocState(
            shape=(len(self.AmocState._variables_), nstep),
            dtype=float
        )

        states[:, 0] = ic[:]
        for i, state in zip(range(1, nstep), self.iter_states(ic)):
            states[:, i] = state[:]
            if i%100 == 0:
                print('*'*16)
                print('NSTEP', i)
        return states

def main():
    ic_values = [4.777404031, 24.42876625, 2.66810894, 2.67598915, 34.40753555,
        35.62585068, 34.92513657, 34.91130066, 4.67439556, 23.30437851, 0.94061828
    ]
    ic = dict(zip(Amoc.AmocState._variables_, ic_values))

    NSEC_YR = 86400*365
    dt_yr = 0.01

    dt, nstep = 0.01*NSEC_YR, int(3.0e5)
    scheme = 'rk4'

    model = Amoc(scheme=scheme, dt=dt)
    states = model.run(ic=ic, nstep=nstep)
    fig, (ax0, ax1) = plt.subplots(ncols=1, nrows=2)

    time = np.arange(nstep)*dt_yr

    def txt2tex(txt):
        return f'${txt[0].capitalize()}_{txt[2]}^{txt[1]}$'

    colors = [plt.get_cmap("tab10")(i) for i in range(4)]
    pstep = 100
    pslice = slice(None, None, pstep)
    for i, v in enumerate(('tos', 'tom', 'ton', 'tod')):
        ax0.plot(time[pslice], states[v][pslice], label=txt2tex(v),
            linestyle='--', color=colors[i])
    for i, v in enumerate(('tas', 'tam', 'tan')):
        ax0.plot(time[pslice], states[v][pslice], label=txt2tex(v),
            color=colors[i])
    for i, v in enumerate(('sos', 'som', 'son', 'sod')):
        ax1.plot(time[pslice], states[v][pslice], label=txt2tex(v),
            color=colors[i])

    ax0.set_ylabel(r'Temperature (C$^\circ$)')
    ax1.set_ylabel(r'Salinity (psu)')
    ax1.set_xlabel(r'Time (yr)')

    for ax in (ax0, ax1):
        ax.legend(ncol=2)
        ax.grid()
    # ax = fig.gca()
    # ax.plot('x', data=states)
    # plt.draw()
    plt.show()

if __name__ == '__main__':
    main()
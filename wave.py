import numbers
from collections.abc import Iterable

import numpy as np 
from matplotlib import pyplot as plt
from matplotlib import animation

import seaborn as sns
sns.set_style("whitegrid")

class Wave:
    def __init__(self, fre, Amp, Phi, modu_wave, info):
        self.fre = fre
        self.Amp = Amp
        self.Phi = Phi
        self.info = info 
        self.modu_wave = modu_wave
    
    def __str__(self):
        return '{cate}Wave(Frequency is {fre},Amplitude is {amp},Initial phase is {phi}),modulate wave is {modu_wave}'.format(
             cate=self.info,fre=self.fre, amp=self.Amp, phi=self.Phi, modu_wave=self.modu_wave
        )

class SinWave(Wave):
    def __init__(self, fre, Amp, Phi, modu_wave=None):
        super().__init__(fre, Amp, Phi, modu_wave, info='Sin')
        self.modu_wave = modu_wave

    def __call__(self, t):
        if self.modu_wave:
            modu_value = self.modu_wave(t)
            # print(modu_value)
            return self.Amp * np.sin(2*np.pi*self.fre*t + self.Phi + modu_value)
        else:
            return self.Amp * np.sin(2*np.pi*self.fre*t + self.Phi)

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return SinWave(fre=self.fre, Amp=self.Amp*scalar, Phi=self.Phi, modu_wave=self.modu_wave)
        elif isinstance(scalar, SinWave):
            diff_fre =  (self.fre - scalar.fre)
            sum_fre =  (self.fre + scalar.fre)
            diff_Phi = self.Phi - scalar.Phi
            sum_Phi = self.Phi + scalar.Phi
            new_amp = 0.5 * self.Amp * scalar.Amp 
            return CosWave(fre=sum_fre, Amp=(-new_amp), Phi=sum_Phi, modu_wave=self.modu_wave), CosWave(fre=diff_fre, Amp=(new_amp), Phi=diff_Phi, modu_wave=self.modu_wave)
        elif isinstance(scalar, CosWave):
            diff_fre = (self.fre - scalar.fre)
            sum_fre =  (self.fre + scalar.fre)
            diff_Phi = self.Phi - scalar.Phi
            sum_Phi = self.Phi + scalar.Phi
            new_amp = 0.5 * self.Amp * scalar.Amp 
            return SinWave(fre=sum_fre, Amp=(new_amp), Phi=sum_Phi, modu_wave=self.modu_wave), SinWave(fre=diff_fre, Amp=(new_amp), Phi=diff_Phi, modu_wave=self.modu_wave)
        else:
            return NotImplemented

    def __rmul__(self, scalar):
        return self * scalar
   
class CosWave(Wave):
    def __init__(self, fre, Amp, Phi, modu_wave=None):
        super().__init__(fre, Amp, Phi, modu_wave, info='Cos')
        self.modu_wave = modu_wave

    def __call__(self, t):
        if self.modu_wave:
            return self.Amp * np.cos(2*np.pi*self.fre*t + self.Phi + self.modu_wave(t))
        else:
            return self.Amp * np.cos(2*np.pi*self.fre*t + self.Phi)

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return CosWave(fre=self.fre, Amp=self.Amp*scalar, Phi=self.Phi, modu_wave=self.modu_wave)
        elif isinstance(scalar, SinWave):
            diff_fre = (self.fre - scalar.fre)
            sum_fre =  (self.fre + scalar.fre)
            diff_Phi = self.Phi - scalar.Phi
            sum_Phi = self.Phi + scalar.Phi
            new_amp = 0.5 * self.Amp * scalar.Amp 
            return SinWave(fre=sum_fre, Amp=(new_amp), Phi=sum_Phi, modu_wave=self.modu_wave), SinWave(fre=diff_fre, Amp=(-new_amp), Phi=diff_Phi, modu_wave=self.modu_wave)
        elif isinstance(scalar, CosWave):
            diff_fre =  (self.fre - scalar.fre)
            sum_fre =  (self.fre + scalar.fre)
            diff_Phi = self.Phi - scalar.Phi
            sum_Phi = self.Phi + scalar.Phi
            new_amp = 0.5 * self.Amp * scalar.Amp 
            return CosWave(fre=sum_fre, Amp=(new_amp), Phi=sum_Phi, modu_wave=self.modu_wave), CosWave(fre=diff_fre, Amp=(new_amp), Phi=diff_Phi, modu_wave=self.modu_wave)
        else:
            return NotImplemented

    def __rmul__(self, scalar):
        return self * scalar

class Signal:
    def __init__(self, sig=None):
        self.signal = [] if sig == None else sig

    def max_fre(self):
        max_fre = 0
        for sig in self.signal:
            if isinstance(sig, Wave):
                if sig.fre > max_fre:
                    max_fre = sig.fre
        return max_fre

    def max_Amp(self):
        max_Amp = 0
        for sig in self.signal:
            if isinstance(sig, Wave):
                if sig.Amp > max_Amp:
                    max_Amp = sig.Amp
        return max_Amp

    def __iter__(self):
        return iter(self.signal)

    def __add__(self, sig):
        if isinstance(sig, Iterable):
            # self.signal.append([i for i in sig])
            for si in sig:
                self.signal.append(si)
        else:
            self.signal.append(sig)
        return Signal(self.signal)
    
    def __radd__(self, sig):
        return self + sig
    
    def __mul__(self, val):
        return Signal( i * val for i in self.signal)
    
    def __rmul__(self, val):
        return self * val
    
    def __call__(self, t):
        sum = 0
        for func in self.signal:
            if isinstance(func, numbers.Real):
                sum += func
            elif isinstance(func, Wave):
                # print(func)
                sum += func(t)
            else:
                raise Exception
        return sum
    
    def __str__(self):
        return 'Signal {}'.format([str(i) for i in self.signal])
    
class UpdateDist:

    def __init__(self, fig, x, *func_y):
        self.fig = fig
        self.func_msg, self.func_car, self.func_modu = func_y
        self.xlim = 4
        self.ylim = 4

        self.ax1 = self.fig.add_subplot(3,1,1) # message
        self.ax1.set_xlim(0, self.xlim)
        self.ax1.set_ylim(-self.ylim, self.ylim)
        self.ax1.set_title('message')
        self.ax1.set_xticks([])

        self.ax2 = self.fig.add_subplot(3,1,2) # carrier
        self.ax2.set_xlim(0, self.xlim)
        self.ax2.set_ylim(-self.ylim, self.ylim)
        self.ax2.set_title('carrier')
        self.ax2.set_xticks([])

        self.ax3 = self.fig.add_subplot(3,1,3) # AM 
        self.ax3.set_xlim(0, self.xlim)
        self.ax3.set_ylim(-self.ylim, self.ylim)
        self.ax3.set_title('modulate')
        self.ax3.set_xticks([])

        self.t = np.arange(0, 2*np.pi, 0.01) # tim  scale

        self.time = x
        self.line_msg, = self.ax1.plot(self.time, self.func_msg(self.time), 'k-')
        self.line_car, = self.ax2.plot(self.time, self.func_car(self.time), 'k-')
        self.line_modu, = self.ax3.plot(self.time, self.func_modu(self.time), 'k-')

    
    def __call__(self, i):
        self.line_msg.set_data(self.time, self.func_msg(self.time + i/10))
        self.line_car.set_data(self.time, self.func_car(self.time + i/10))
        self.line_modu.set_data(self.time, self.func_modu(self.time + i/10))
        return self.line_msg, self.line_car, self.line_modu

def AM(information_signal, carrier_signal, A):
    am_sig = Signal() + 1 
    for mes_sig in information_signal:
        am_sig += mes_sig * (1/A)
    true_sig = Signal()
    # print(am_sig)
    # print(carrier_signal)
    for sig in am_sig:
        true_sig += sig * carrier_signal
    # print(true_sig)
    return true_sig

def PM(information_signal, carrier_signal):
    pm = Signal()
    if isinstance(carrier_signal, SinWave):
        pm += SinWave(fre=carrier_signal.fre, Amp=carrier_signal.Amp, Phi=carrier_signal.Phi, modu_wave=information_signal)
    elif isinstance(carrier_signal, CosWave):
        pm += CosWave(fre=carrier_signal.fre, Amp=carrier_signal.Amp, Phi=carrier_signal.Phi, modu_wave=information_signal)
    else:
        raise Exception('carrier_signal not define')
    # print(pm)
    return pm

def FM(information_signal, carrier_signal, h=1):
    fm_sig = Signal()
    modu_sig = Signal()
    # print('h is ->>>'+ str(h))
    for mes_sig in information_signal:
        modu_sig += mes_sig * h
    if isinstance(carrier_signal, SinWave):
        fm_sig += SinWave(fre=carrier_signal.fre, Amp=carrier_signal.Amp, Phi=carrier_signal.Phi, modu_wave=modu_sig)
    elif isinstance(carrier_signal, CosWave):
        fm_sig += CosWave(fre=carrier_signal.fre, Amp=carrier_signal.Amp, Phi=carrier_signal.Phi, modu_wave=modu_sig)
    else:
        raise Exception('carrier_signal not define')
    # print('------------------------fm')
    # print(fm_sig)
    return fm_sig

if __name__ == "__main__": 
    fig = plt.figure()

    A = 2 # carrier wave amplitude 
    m = 1  #  m is the amplitude sensitivity
    mes_fre = 1 # message wave frequence 
    car_fre = 10  # carrier  wave frequence 
    signal = Signal()
    time = np.arange(0, 2*np.pi, 0.01)

    test1_wave = CosWave(fre=mes_fre*2, Amp=0.5, Phi=0)
    test2_wave = CosWave(fre=mes_fre*3, Amp=0.5, Phi=0)
    # print(test1_wave)
    # print(test2_wave)

    mes_wave = SinWave(fre=mes_fre, Amp=A*m, Phi=0)
    information_signal = signal  + mes_wave #+ test1_wave * test2_wave
    # print(information_signal)

    carrier_signal = CosWave(fre=car_fre, Amp=A, Phi=0)
    # print(carrier_signal)

    # AM modulate
    # am_sig = AM(information_signal, carrier_signal, A=A)
    # ud = UpdateDist(fig, time, information_signal, carrier_signal, am_sig)
    # PM modulate
    # pm_sig = PM(information_signal, carrier_signal)
    # # print(pm_sig)
    # ud = UpdateDist(fig, time, information_signal, carrier_signal, pm_sig)

    # FM modulate
    k_f = 1 # delta is the frequence sensitivity
    delta = k_f * information_signal.max_Amp()
    fm = information_signal.max_fre()
    h = delta/fm
    fm_sig = FM(information_signal, carrier_signal, h=h)
    ud = UpdateDist(fig, time, information_signal, carrier_signal, fm_sig)
    ani = animation.FuncAnimation(fig=fig,
                                func=ud,
                                frames=20,
                                interval=100,
                                blit=True)
    plt.show()
    ani.save('FM.gif', writer='pillow')
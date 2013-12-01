import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.animation as animation

N=512
Nt=400
tmax=400
Lx=1.5
Nx=N
dt=tmax/Nt
tol=0.1**12
A=0.04
B=0.1
Du=0.001
Dv=0.0001

x=(2*np.pi/Nx)*np.array([n for n in range(-Nx/2, Nx/2 )]).transpose()*Lx
I=complex(0,1)
kx=np.array([I*n for n in range(0,Nx/2) + [0] + range(-Nx/2+1, 0)]).transpose()/Lx

t=0

u=0.2+np.exp(-2*x**2)
v=0.1+np.exp(-4*(x-0.01)**2)

gamma=[[complex(.053475778387618596606, .006169356340079532510)],[complex(.041276342845804256647, .069948574390707814951)],[complex(.086533558604675710289, .023112501636914874384)],[complex(.079648855663021043369, .049780495455654338124)],[complex(.069981052846323122899, .052623937841590541286)],[complex(.087295480759955219242, .010035268644688733950)],[complex(.042812886419632082126, .076059456458843523862)],[complex(.077952088945939937643, .007280873939894204350)],[complex(.042812886419632082126, .076059456458843523862)],[complex(.087295480759955219242, .010035268644688733950)],[complex(.069981052846323122899, .052623937841590541286)],[complex(.079648855663021043369, .049780495455654338124)],[complex(.086533558604675710289, .023112501636914874384)],[complex(.041276342845804256647, .069948574390707814951)],[complex(.053475778387618596606, .006169356340079532510)]]

Ahat=np.multiply(A,sp.fft([1]*(len(x))))
datau = np.zeros((Nt, N))
datav = np.zeros((Nt, N))
datau[0,:] = u
datav[0,:] = v
tdata = [t]

for n in range(1, Nt):
	print n
	chg=1
	for m in range(1, 15):
		chg=1
		uold=u
		vold=v
		while chg>tol:
			utemp=u
			vtemp=v
			umean=0.5*(u+uold)
			vmean=0.5*(v+vold)
			u=uold+np.dot(0.5, gamma[m])*dt*(np.multiply(-umean, vmean**2))
			v=vold+np.multiply(np.dot(0.5,gamma[m])*dt*umean, vmean**2)
			chg=max(abs(u-utemp))+max(abs(v-vtemp))
		uhat=sp.fft(u)
		vhat=sp.fft(v)

		uhat=np.multiply(np.exp(gamma[m]*dt*(-A+np.multiply(Du*kx,kx))), (uhat-np.divide(Ahat, (A+np.multiply(Du*kx, kx)))))+np.divide(Ahat, (A+np.multiply(Du*kx,kx)))
		vhat=np.multiply(np.exp(gamma[m]*dt*(-B+np.multiply(Dv*kx,kx))), vhat)
		u = sp.ifft(uhat)
		v = sp.ifft(vhat)
		chg=1
		uold=u
		vold=v
		while chg>tol:
			utemp=u
			vtemp=v
			umean=0.5*(u+uold)
			vmean=0.5*(v+vold)
			u=uold+np.dot(0.5, gamma[m])*dt*(np.multiply(-umean, vmean**2))
			v=vold+np.multiply(np.dot(0.5,gamma[m])*dt*umean, vmean**2)
			chg=max(abs(u-utemp))+max(abs(v-vtemp))
	t=n*dt
	# draw on graph
	
	datau[n,:]=np.real(u)
	datav[n,:]=np.real(v)
	tdata.append(t)

xx,tt = (np.mat(A) for A in (np.meshgrid(x, tdata)))

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(xx, tt, datav,rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

plt.style.use('dark_background')

# FROM SMITH'S WAVES AND OSCILLATION
#--------------------------------------
#y(t_j)= \sum_{n=0}^{N-1} C_n e^{i\omega_n t_j}
#C_n = \sum_{n=0}^{N-1} y(t_j) e^{-i \omega_n t_j}
#\omega_1 = \frac{2 \pi}{N \Delta}
#\omega_n = n \omega_1
#T = N \Delta = total measurment time
#--------------------------------------

#C_n = RELATIVE STRENGTH OF DIFFERENT MODES
def compute_Cn(data, n_modes): #data has to be a 1-d array of complex values
    #MATH
    N = len(data)
    t_j = np.arange(0,N)
    omega_1 = 2*np.pi/N
    omega_n = np.arange(0,n_modes)*omega_1

    C_n = np.zeros(n_modes, dtype=complex)

    for n in range(1,n_modes):
        tmp_C_n = np.sum(data * np.exp(-1j * omega_n[n] * t_j))
        C_n[n] = tmp_C_n

    return C_n

#EXTRACTING DATA AND DEFINING VARIABLES
#--------------------------------------
data = np.loadtxt('mona_lisa.txt')

complex_data = data[0] + 1j * data[1]

n_modes = len(complex_data)
C_n = compute_Cn(complex_data,n_modes)
N = len(complex_data)
t = np.linspace(0,N,N+1)
omega_1 = 2*np.pi/N
omega_n = np.arange(0,n_modes)*omega_1

#SIGNAL PROCESSING
#--------------------------------------
modulus_C_n = np.real(C_n)**2+np.imag(C_n)**2
normalized_C_n_modulus = modulus_C_n/max(modulus_C_n)

#ONLY THE n_active_modes BIGGEST MODES ARE ACTIVE
n_active_modes = 150
argsorted_C_n_moduli = np.flip(np.argsort(normalized_C_n_modulus))
active_modes = argsorted_C_n_moduli[:n_active_modes]

#REMOVE SMALL MODES
# active_modes = (normalized_C_n_modulus>=0.00007).nonzero()[0]

#APPROXIMATION OF NEW FUNCTION y(t_n) (MATH)
#--------------------------------------
y_t = 0
for n in active_modes:
    y_t += C_n[n] * np.exp(1j*omega_n[n]*t)
y_t/=N

#--------------------------------------

#SHOWING PICTURE
plt.axis('equal')
plt.plot(np.real(y_t), np.imag(y_t))

#--------------------------------------
#SHOWING ANIMATION
fig, ax = plt.subplots(1)

line, = ax.plot(np.real(y_t), np.imag(y_t))

arrows = np.zeros(n_modes, dtype='object')

for i in active_modes:
    arrows[i], = ax.plot(0,0)

def animate(k):

    line.set_xdata(np.real(y_t)[:k])
    line.set_ydata(np.imag(y_t)[:k])
    previous_pos = 0 + 0j
    for n in active_modes:
        pos = C_n[n] * np.exp(1j*omega_n[n]*t)/N
        arrows[n].set_data([np.real(previous_pos), np.real(previous_pos)+np.real(pos[k-1])], [np.imag(previous_pos), np.imag(previous_pos)+np.imag(pos[k-1])])
        previous_pos += np.copy(pos[k-1])


ax.axis('equal')
nframes = N+1
ani = animation.FuncAnimation(fig, animate, interval=1, frames = nframes)
#ani.save('.mp4')
plt.show()

#--------------------------------------
#SHOWING MODES IN k SPACE (kind of)
plt.style.use('default')
plt.title('Relative mode strength')
plt.xlabel('n')
plt.ylabel(r'$\dfrac{|C_n|}{|C_{max}|}$')
plt.vlines( x = active_modes, ymin = np.zeros(len(active_modes)), ymax = normalized_C_n_modulus[active_modes])
plt.hlines( y = 0, xmin=0, xmax = max(active_modes))
plt.show()

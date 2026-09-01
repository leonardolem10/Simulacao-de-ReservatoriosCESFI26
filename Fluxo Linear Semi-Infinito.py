
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

p0 = 300        
k = 400         
mi = 2          
phi = 0.25      
ct = 100e-6     
L = 200         
qw = 50          
A = 1          

eta = (k/(phi*mi*ct))*0.0003484

print(f"eta = {eta:.3e} m²/h")

def solucao_analitica(x,t):

    if t == 0:
        return np.ones_like(x)*p0


    termo1 = np.sqrt((4*eta*t)/(np.pi*L**2)) * np.exp(-x**2/(4*eta*t))


    termo2 = (x/L) * erfc(x/np.sqrt(4*eta*t))
    pressao = p0 - ((qw*mi*L)/(k*A)*(termo1-termo2))
    return pressao



# plot

x = np.linspace(0,2000,500)

tempos = [0.5,5,24,48]


plt.figure(figsize=(9,5))


for t in tempos:

    p = solucao_analitica(x,t)

    plt.plot( x, p,label=f"t = {t} h")


plt.xlabel("Distância x (m)")
plt.ylabel("Pressão (kgf/cm²)")


plt.title("Solução Analítica 1D - Fluxo Linear Semi-Infinito (Caso 2)")
plt.grid(True)
plt.legend()
plt.show()
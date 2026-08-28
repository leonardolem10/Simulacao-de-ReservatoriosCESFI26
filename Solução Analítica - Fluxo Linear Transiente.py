import numpy as np
import matplotlib.pyplot as plt

#Parâmetros

p0 = 300 
pe=p0      
pw = 150             
k = 20         
mi = 0.8       
phi = 0.18
ct = 150e-6    
L = 250
          
eta=(k/(phi*mi*ct))*0.0003484

# Solução analítica
def analitica(x,t):

    if t == 0:
        return np.ones(len(x))*p0

    soma = np.zeros(len(x))

    for n in range(1,1000):
        soma += (1/n)*np.exp(-((n*np.pi/L)**2)*eta*t)*np.sin(n*np.pi*x/L)

    return (p0-pw)*(x/L+(2/np.pi)*soma)+pw

# Domínio

x = np.linspace(0,L,200)


tempos = [0.5,5,24,48]


plt.figure(figsize=(8,5))


for t in tempos:
    plt.plot(x,analitica(x,t),label="t = "+str(t)+" h")


plt.xlabel(" x (m)")
plt.ylabel("Pressão")
plt.title("Solução Analítica - Fluxo Linear Transiente")
plt.grid()
plt.legend()
plt.show()
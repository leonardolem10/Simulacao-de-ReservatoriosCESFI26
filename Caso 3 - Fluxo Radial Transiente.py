import numpy as np
import matplotlib.pyplot as plt
from scipy.special import exp1

# ----------------------------------------------------
# 1. Parâmetros Físicos (Sistema Internacional - SI)
# ----------------------------------------------------
phi = 0.2          # Porosidade
k = 1e-13          # Permeabilidade (m²) (~100 mD)
h = 10             # Espessura (m)
mu = 1e-3          # Viscosidade (Pa.s) (1 cP)
ct = 1e-9          # Compressibilidade total (1/Pa)
q = 0.001          # Vazão (m³/s)
pi = 2e7           # Pressão inicial (Pa) (20 MPa)
rw = 0.1           # Raio do poço (m)

# ----------------------------------------------------
# 2. Domínio Espacial e Temporal
# ----------------------------------------------------
x = np.linspace(rw, 200 * rw, 200)   # Distância a partir do poço (0.1 m até 20 m)
tempos = [60, 300, 1800, 3600]       # Tempos em segundos

# ----------------------------------------------------
# 3. Cálculo da Pressão
# ----------------------------------------------------
plt.figure(figsize=(7, 4.5), dpi=150)

for t in tempos:
    u = (phi * mu * ct * (x**2)) / (4 * k * t)
    p = pi - (q * mu / (4 * np.pi * k * h)) * exp1(u)

    plt.plot(x, p, label=f't = {t} s')

# ----------------------------------------------------
# 4. Gráfico
# ----------------------------------------------------
plt.title('Solução 1D - Fluxo Radial Transiente', fontsize=11, fontweight='bold')
plt.xlabel('Coordenada x (m)', fontsize=10)
plt.ylabel('Pressão (Pa)', fontsize=10)
plt.tick_params(labelsize=9)
plt.grid(True)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig('grafico_caso3.png', dpi=300)
plt.show()
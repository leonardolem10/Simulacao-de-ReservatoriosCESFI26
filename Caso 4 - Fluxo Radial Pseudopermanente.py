import numpy as np
import matplotlib.pyplot as plt

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
re = 300           # Raio externo selado (m)

# ----------------------------------------------------
# 2. Domínio Espacial e Temporal
# ----------------------------------------------------
x = np.linspace(rw, re, 200)  # Coordenada espacial de rw até re

# Tempos em segundos
tempos = [2 * 3600, 5 * 3600, 10 * 3600, 20 * 3600]

# ----------------------------------------------------
# 3. Cálculo da Pressão
# ----------------------------------------------------
plt.figure(figsize=(9, 5), dpi=300)

for t in tempos:
    termo = (2 * k * t) / (phi * mu * ct * re**2) + np.log(re / x) + 0.5 * (x / re)**2 - 0.75
    p = pi - (q * mu / (2 * np.pi * k * h)) * termo

    plt.plot(x, p, label=f't = {t / 3600:.1f} h')


# ----------------------------------------------------
# 4. Gráfico
# ----------------------------------------------------

plt.title('Solução 1D - Fluxo Radial Pseudopermanente', fontsize=11, fontweight='bold')
plt.xlabel('Coordenada x (m)', fontsize=10)
plt.ylabel('Pressão (Pa)', fontsize=10)
plt.tick_params(labelsize=9)
plt.grid(True)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig('grafico_caso4.png', dpi=300)
plt.show()
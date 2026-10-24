import numpy as np

def gerar_campo_lognormal(nx, ny, media_mD=100.0, sigma=0.5, seed=42):
    """Gera campo de permeabilidade heterogéneo log-normal conforme Aula 10."""
    np.random.seed(seed)
    mu_ln = np.log(media_mD)
    k_mD = np.random.lognormal(mean=mu_ln, sigma=sigma, size=(nx, ny))
    # Limita valores entre patamares físicos realistas (ex: 10 mD a 1500 mD)
    return np.clip(k_mD, 5.0, 2000.0)

import numpy as np


def gerar_permeabilidade_p1(nx, ny):
  perm_mD = np.full((nx, ny), 500.0, dtype=float)
  perm_mD[8:23, 8:15] = 50.0
  perm_mD[8:23, 16:23] = 150.0
  return perm_mD


def gerar_permeabilidade_p2(nx, ny):
  perm_mD = np.full((nx, ny), 100.0, dtype=float)
  col_falha = int(ny * 0.35)
  linha_ini_falha = int(nx * 0.15)
  linha_fim_falha = int(nx * 0.85)
  perm_mD[linha_ini_falha:linha_fim_falha, col_falha] = 1.0e-5
  return (perm_mD, col_falha, linha_ini_falha, linha_fim_falha)


def gerar_permeabilidade_p3(nx, ny):
  perm_mD = np.full((nx, ny), 100.0, dtype=float)
  perm_mD[8:23, 8:15] = 50.0
  perm_mD[8:23, 16:23] = 150.0
  return perm_mD

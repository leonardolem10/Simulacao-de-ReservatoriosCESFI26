import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# CONFIGURAÇÃO DE SALVAMENTO AUTOMÁTICO DE GRÁFICOS

PASTA_GRAFICOS = os.path.join(
    os.path.expanduser("~"),
    "Downloads",
    "graficos_simulacao_t4"
)
os.makedirs(PASTA_GRAFICOS, exist_ok=True)

contador_graficos = 0

def salvar_e_fechar():
    global contador_graficos
    contador_graficos += 1
    caminho = os.path.join(PASTA_GRAFICOS, f"grafico_{contador_graficos:02d}.png")
    try:
        plt.savefig(caminho, dpi=150, bbox_inches="tight", pil_kwargs={"optimize": True})
        print(f"Gráfico salvo (otimizado para LaTeX): {caminho}")
    except Exception as erro:
        print(f"Erro ao salvar gráfico: {erro}")
        raise
    finally:
        plt.close()

from gerador_permeabilidade import (
    gerar_permeabilidade_p1,
    gerar_permeabilidade_p2,
    gerar_permeabilidade_p3
)

# 1. PARÂMETROS GERAIS DO RESERVATÓRIO

Lx, Ly, h = 100.0, 100.0, 12.0
nx, ny = 31, 31
N = nx * ny

dx, dy = Lx / ny, Ly / nx
Vb = dx * dy * h
x_c = (np.arange(ny) + 0.5) * dx
y_c = (np.arange(nx) + 0.5) * dy

p0 = 160.0 * 98066.5       # 160 kgf/cm² -> 156.9064 bar -> Pa
pwf = 100.0 * 98066.5      # 100 kgf/cm² -> 98.0665 bar -> Pa
mu = 2.5e-3                # 2.5 cP -> Pa.s
phi = 0.20                 # Porosidade
c = 15e-5 / 98066.5        # Compressibilidade total -> Pa^-1
rw, skin = 0.10, 0.0       # Raio do poço e dano nulo

# 2. OPERADORES NUMÉRICOS DO MÉTODO DOS VOLUMES FINITOS (MVF - AULA 13)

def velocidades_darcy_mvf(p, perm, mu, dx, dy):
    """Calcula a velocidade de Darcy nas faces dos volumes finitos e interpola no centro."""
    n0, n1 = p.shape
    u_face = np.zeros((n0, n1 + 1))
    kx = 2.0 * perm[:, :-1] * perm[:, 1:] / np.maximum(perm[:, :-1] + perm[:, 1:], 1e-300)
    u_face[:, 1:-1] = -(kx / mu) * (p[:, 1:] - p[:, :-1]) / dx

    v_face = np.zeros((n0 + 1, n1))
    ky = 2.0 * perm[:-1, :] * perm[1:, :] / np.maximum(perm[:-1, :] + perm[1:, :], 1e-300)
    v_face[1:-1, :] = -(ky / mu) * (p[1:, :] - p[:-1, :]) / dy

    Vx = 0.5 * (u_face[:, :-1] + u_face[:, 1:])
    Vy = 0.5 * (v_face[:-1, :] + v_face[1:, :])
    return Vx, Vy

def montar_sistema_mvf(perm, dt, condicoes_fronteira, pocos_produtores):
    """
    Monta a matriz do sistema linear pelo MVF totalmente implícito (Aula 13).
    A equação integral é dividida por V_b, gerando:
      - Termo temporal: C_dt = (phi * c_t) / dt
      - Condutâncias de face: Cx = k_face / (mu * dx^2), Cy = k_face / (mu * dy^2)
      - Contorno de Pressão prescrita (Dirichlet): t_ext = 2*k/(mu*delta^2) na diagonal e t_ext*p_bc no vetor B.
      - Contorno de Fluxo prescrito (Neumann): q_bc (m³/s, positivo entrando) adicionado como +q_bc / V_b no vetor B.
      - Termo de poço na diagonal: WI / Vb
    """
    C_dt = (phi * c) / dt

    # Peaceman para malha retangular (Aula 11): r0 = 0.140365*sqrt(dx^2+dy^2) -> 0.198*dx
    r0 = 0.140365 * np.sqrt(dx**2 + dy**2)

    Cy = np.zeros((nx - 1, ny))
    for i in range(nx - 1):
        for j in range(ny):
            k_harm = 2.0 * perm[i, j] * perm[i + 1, j] / (perm[i, j] + perm[i + 1, j])
            Cy[i, j] = k_harm / (mu * dy**2)

    Cx = np.zeros((nx, ny - 1))
    for i in range(nx):
        for j in range(ny - 1):
            k_harm = 2.0 * perm[i, j] * perm[i, j + 1] / (perm[i, j] + perm[i, j + 1])
            Cx[i, j] = k_harm / (mu * dx**2)

    A = sp.lil_matrix((N, N))
    B_bc = np.zeros(N)

    wi_pocos = {}
    for (i_w, j_w), pwf_val in pocos_produtores.items():
        wi_pocos[(i_w, j_w)] = (2.0 * np.pi * perm[i_w, j_w] * h) / (mu * (np.log(r0 / rw) + skin))

    for i in range(nx):
        for j in range(ny):
            row = i * ny + j
            diag = C_dt

            # Face Oeste (j = 0)
            if j > 0:
                t = Cx[i, j - 1]
                A[row, row - 1] = -t
                diag += t
            else:
                bc = condicoes_fronteira['oeste']
                if bc['tipo'] == 'pressao_prescrita':
                    t_ext = (2.0 * perm[i, j]) / (mu * dx**2)
                    diag += t_ext
                    B_bc[row] += t_ext * bc['valor']
                elif bc['tipo'] == 'fluxo_prescrito':
                    # bc['valor'] em m³/s por célula de contorno (positivo entrando no reservatório)
                    B_bc[row] += bc['valor'] / Vb

            # Face Leste (j = ny - 1)
            if j < ny - 1:
                t = Cx[i, j]
                A[row, row + 1] = -t
                diag += t
            else:
                bc = condicoes_fronteira['leste']
                if bc['tipo'] == 'pressao_prescrita':
                    t_ext = (2.0 * perm[i, j]) / (mu * dx**2)
                    diag += t_ext
                    B_bc[row] += t_ext * bc['valor']
                elif bc['tipo'] == 'fluxo_prescrito':
                    B_bc[row] += bc['valor'] / Vb

            # Face Sul (i = 0)
            if i > 0:
                t = Cy[i - 1, j]
                A[row, row - ny] = -t
                diag += t
            else:
                bc = condicoes_fronteira['sul']
                if bc['tipo'] == 'pressao_prescrita':
                    t_ext = (2.0 * perm[i, j]) / (mu * dy**2)
                    diag += t_ext
                    B_bc[row] += t_ext * bc['valor']
                elif bc['tipo'] == 'fluxo_prescrito':
                    B_bc[row] += bc['valor'] / Vb

            # Face Norte (i = nx - 1)
            if i < nx - 1:
                t = Cy[i, j]
                A[row, row + ny] = -t
                diag += t
            else:
                bc = condicoes_fronteira['norte']
                if bc['tipo'] == 'pressao_prescrita':
                    t_ext = (2.0 * perm[i, j]) / (mu * dy**2)
                    diag += t_ext
                    B_bc[row] += t_ext * bc['valor']
                elif bc['tipo'] == 'fluxo_prescrito':
                    B_bc[row] += bc['valor'] / Vb

            # Acoplamento de Poço
            if (i, j) in wi_pocos:
                diag += wi_pocos[(i, j)] / Vb

            A[row, row] = diag

    return A.tocsr(), B_bc, C_dt, wi_pocos

def validar_simulacao(nome, P_final, pwf_ref, vazoes, Np_final, Ninj_acum=0.0):
    """
    Verificação física e balanço global de volume.
    Restaura tolerância estrita de 1e-4 Pa e imprime todas as variáveis das Tabelas 3 e 4 em bar.
    """
    erros = []
    if not np.all(np.isfinite(P_final)):
        erros.append("Pressões NaN ou infinitas.")
    if np.any(P_final <= 0.0):
        erros.append("Pressões negativas detectadas.")
    # Tolerância rigorosa do T3: no máximo 1e-4 Pa abaixo de pwf
    if P_final.min() < pwf_ref - 1e-4:
        erros.append(f"Pressão mínima ({P_final.min()/1e5:.2f} bar) menor que pwf imposto.")

    delta_armaz = np.sum(Vb * phi * c * (P_final - p0))
    Np_balanco = Ninj_acum - delta_armaz
    erro_abs = abs(Np_final - Np_balanco)
    erro_rel = erro_abs / max(abs(Np_final), abs(Np_balanco), 1e-12)

    p_min_bar = P_final.min() / 1e5
    p_max_bar = P_final.max() / 1e5
    p_med_bar = P_final.mean() / 1e5

    q1_cm3s = vazoes[0] * 1e6 if len(vazoes) > 0 else 0.0
    qf_cm3s = vazoes[-1] * 1e6 if len(vazoes) > 0 else 0.0

    print(f"\n--- RELATÓRIO DE VALIDAÇÃO: {nome} ---")
    print(f"Pmin: {p_min_bar:.2f} bar | Pmax: {p_max_bar:.2f} bar | Pmedia: {p_med_bar:.2f} bar  [TABELA 3]")
    print(f"Vazão inicial q1: {q1_cm3s:.0f} cm³/s | Vazão final qf: {qf_cm3s:.0f} cm³/s       [TABELA 3]")
    print(f"Injeção total: {Ninj_acum:.4f} m³ | Delta Armazenamento: {delta_armaz:.4f} m³       [TABELA 4]")
    print(f"Produção acumulada Np: {Np_final:.6f} m³ | Pelo balanço: {Np_balanco:.6f} m³")
    print(f"Erro absoluto: {erro_abs:.6e} m³ | Erro relativo: {erro_rel:.3e}")

    if erro_rel > 1e-5 or erros:
        print("STATUS: VERIFICAR")
        for e in erros:
            print(f"  -> {e}")
        return False
    print("STATUS: PASSOU (Fisicamente coerente e conservação estrita).")
    return True

def simular_cenario(nome, perm, tempo_total, nt, pocos_produtores, pocos_injetores, passos_salvos):
    """Marcha temporal implícita pelo MVF."""
    dt = tempo_total / nt
    bc_selada = {f: {'tipo': 'fluxo_nulo', 'valor': 0.0} for f in ['oeste', 'leste', 'sul', 'norte']}
    A, B_bc, C_dt, wi_pocos = montar_sistema_mvf(perm, dt, bc_selada, pocos_produtores)

    P_vetor = np.full(N, p0)
    tempo_list, q_prod_list = [0.0], [0.0]
    campos_transientes = {}

    t0 = time.time()
    for step in range(1, nt + 1):
        B = C_dt * P_vetor + B_bc

        # Termo fonte de produtores (WI * pwf / Vb)
        for (i_w, j_w), pwf_val in pocos_produtores.items():
            B[i_w * ny + j_w] += wi_pocos[(i_w, j_w)] * pwf_val / Vb

        # Termo fonte de injetores (q_inj / Vb)
        for (i_inj, j_inj), q_inj in pocos_injetores.items():
            B[i_inj * ny + j_inj] += q_inj / Vb

        P_vetor = spla.spsolve(A, B)
        P_matriz = P_vetor.reshape((nx, ny))

        q_step = sum(wi_pocos[(i_w, j_w)] * (P_matriz[i_w, j_w] - pwf_val)
                     for (i_w, j_w), pwf_val in pocos_produtores.items())

        t_h = (step * dt) / 3600.0
        tempo_list.append(t_h)
        q_prod_list.append(q_step)

        if step in passos_salvos:
            campos_transientes[t_h] = P_matriz.copy()

    dt_exec = time.time() - t0
    Np = np.cumsum([0.0] + [q * dt for q in q_prod_list[1:]])
    print(f"[{nome}] Simulação MVF concluída em {dt_exec:.2f} s | Np: {Np[-1]:.4f} m³")

    return {
        "campos": campos_transientes,
        "P_final": P_matriz,
        "tempo": tempo_list,
        "q_prod": q_prod_list,
        "Np": Np,
        "wi": wi_pocos,
        "dt": dt
    }

# 3. FUNÇÕES GRÁFICAS COMPACTAS (GRADE 2x2, MESMO PADRÃO VISUAL DO T3)

def plotar_evolucao_pressao(campos, titulo, pwf_min, p0_max, poco_pos, falha_coords=None, inj_coords=None):
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 9.8), sharex=True, sharey=True)
    axs = axs.ravel()

    for ax, (t_h, p_field) in zip(axs, campos.items()):
        im = ax.imshow(
            p_field / 1.0e5, extent=[0, Lx, 0, Ly], origin="lower",
            cmap="PuRd", vmin=pwf_min / 1.0e5, vmax=p0_max / 1.0e5
        )

        if falha_coords:
            ax.plot([falha_coords[0], falha_coords[0]], [falha_coords[1], falha_coords[2]], "k--", linewidth=2.2, label="Falha", zorder=3)

        if inj_coords:
            for ii, jj in inj_coords:
                ax.plot(x_c[jj], y_c[ii], "^", color="#49006a", markersize=7, zorder=5)

        for iw, jw in poco_pos:
            ax.plot(x_c[jw], y_c[iw], "kD", markersize=7, zorder=6)

        ax.set_title(f"t = {t_h:.2f} h", fontsize=11, pad=10)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)

    fig.subplots_adjust(left=0.08, right=0.84, bottom=0.08, top=0.88, wspace=0.05, hspace=0.25)
    cbar_ax = fig.add_axes([0.87, 0.12, 0.025, 0.75])
    fig.colorbar(im, cax=cbar_ax, label="Pressão [bar]")
    fig.suptitle(titulo, fontsize=13, y=0.96)
    salvar_e_fechar()

def plotar_velocidade_darcy(campos, perm, titulo, pwf_min, p0_max, poco_pos, falha_coords=None, inj_coords=None):
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 9.8), sharex=True, sharey=True)
    axs = axs.ravel()
    step_q = 2
    Xq, Yq = np.meshgrid(x_c, y_c)

    for ax, (t_h, p_field) in zip(axs, campos.items()):
        Vx, Vy = velocidades_darcy_mvf(p_field, perm, mu, dx, dy)
        V_mag = np.sqrt(Vx**2 + Vy**2)

        mask = V_mag > 1e-15
        Vx_dir = np.zeros_like(Vx)
        Vy_dir = np.zeros_like(Vy)
        Vx_dir[mask] = Vx[mask] / V_mag[mask]
        Vy_dir[mask] = Vy[mask] / V_mag[mask]

        im = ax.imshow(
            p_field / 1.0e5, extent=[0, Lx, 0, Ly], origin="lower",
            cmap="PuRd", vmin=pwf_min / 1.0e5, vmax=p0_max / 1.0e5, aspect="equal"
        )

        ax.quiver(
            Xq[::step_q, ::step_q], Yq[::step_q, ::step_q],
            Vx_dir[::step_q, ::step_q], Vy_dir[::step_q, ::step_q],
            color="black",
            pivot="mid",
            scale=28,
            width=0.003,
            headwidth=3.5,
            headlength=4.5,
            alpha=0.85,
            zorder=4
        )

        if falha_coords:
            ax.plot([falha_coords[0], falha_coords[0]], [falha_coords[1], falha_coords[2]], "k--", linewidth=2.0, zorder=3)

        if inj_coords:
            for ii, jj in inj_coords:
                ax.plot(x_c[jj], y_c[ii], "^", color="#49006a", markersize=7, zorder=5)

        for iw, jw in poco_pos:
            ax.plot(x_c[jw], y_c[iw], "kD", markersize=7, zorder=6)

        ax.set_title(f"t = {t_h:.2f} h", fontsize=11, pad=10)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)

    fig.subplots_adjust(left=0.08, right=0.84, bottom=0.08, top=0.88, wspace=0.05, hspace=0.25)
    cbar_ax = fig.add_axes([0.87, 0.12, 0.025, 0.75])
    fig.colorbar(im, cax=cbar_ax, label="Pressão [bar]")
    fig.suptitle(titulo, fontsize=13, y=0.96)
    salvar_e_fechar()

def plotar_desempenho(tempo_list, q_prod_list, Np_list, titulo, q_inj_total=None):
    fig, ax_q = plt.subplots(figsize=(6.5, 4))
    if q_inj_total:
        ax_q.plot(tempo_list[1:], np.full(len(tempo_list) - 1, q_inj_total * 1e6), "m--", linewidth=2, label="Vazão Total Injetada")
    ax_q.plot(tempo_list[1:], np.array(q_prod_list[1:]) * 1.0e6, 'k-', linewidth=2, label="Vazão Produzida")
    ax_q.set_xlabel("Tempo [h]")
    ax_q.set_ylabel("Vazão [cm³/s]")
    ax_q.grid(True, linestyle="--", alpha=0.5)

    ax_q_twin = ax_q.twinx()
    ax_q_twin.plot(tempo_list, Np_list, 'b:', linewidth=2, label="Produção Acumulada")
    ax_q_twin.set_ylabel("Np [m³]", color='blue')

    h1, l1 = ax_q.get_legend_handles_labels()
    h2, l2 = ax_q_twin.get_legend_handles_labels()
    ax_q.legend(h1 + h2, l1 + l2, loc="upper right" if not q_inj_total else "lower right")
    plt.title(titulo)
    plt.tight_layout()
    salvar_e_fechar()

# 4. EXECUÇÃO — PROBLEMA 1: PRODUTOR CENTRAL

res_p1 = gerar_permeabilidade_p1(nx, ny)
perm_p1_mD = np.asarray(res_p1[0] if isinstance(res_p1, tuple) else res_p1, dtype=float)
perm_p1 = perm_p1_mD * 9.869233e-16

plt.figure(figsize=(5.5, 4.5))
im_k1 = plt.imshow(perm_p1_mD, extent=[0, Lx, 0, Ly], origin="lower", cmap="PuRd")
plt.colorbar(im_k1, label="Permeabilidade [mD]")
plt.title("Problema 1: Campo de Permeabilidade")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.tight_layout()
salvar_e_fechar()

i_w1, j_w1 = nx // 2, ny // 2
poco_p1 = {(i_w1, j_w1): pwf}
passos_p1 = [int(100 * 0.05), int(100 * 0.25), int(100 * 0.50), 100]

sim_p1 = simular_cenario("Problema 1", perm_p1, 2.0 * 3600.0, 100, poco_p1, {}, passos_p1)

plotar_evolucao_pressao(sim_p1["campos"], "Problema 1: Evolução Transiente da Pressão 2D", pwf, p0, [(i_w1, j_w1)])
plotar_velocidade_darcy(sim_p1["campos"], perm_p1, "Problema 1: Campo Transiente de Velocidades de Darcy", pwf, p0, [(i_w1, j_w1)])
plotar_desempenho(sim_p1["tempo"], sim_p1["q_prod"], sim_p1["Np"], "Problema 1: Histórico de Produção")

validar_simulacao("Problema 1 — Produtor Central", sim_p1["P_final"], pwf, sim_p1["q_prod"][1:], sim_p1["Np"][-1])

# 5. EXECUÇÃO — PROBLEMA 2: FALHA SELANTE E PRODUTOR DESCENTRALIZADO

res_p2 = gerar_permeabilidade_p2(nx, ny)
if len(res_p2) == 5:
    perm_p2_mD, _, col_falha, l_ini, l_fim = res_p2
elif len(res_p2) == 4:
    perm_p2_mD, col_falha, l_ini, l_fim = res_p2
else:
    raise ValueError(f"Formato inesperado no gerador: {len(res_p2)} elementos.")

perm_p2_mD = np.asarray(perm_p2_mD, dtype=float)
perm_p2 = perm_p2_mD * 9.869233e-16

c_f = int(np.asarray(col_falha).ravel()[0])
l_i = int(np.asarray(l_ini).ravel()[0])
l_f = int(np.asarray(l_fim).ravel()[0])

x_falha = float((c_f + 0.5) * dx)
y_ini_falha = float(l_i * dy)
y_fim_falha = float((l_f + 1) * dy)
falha_coords = (x_falha, y_ini_falha, y_fim_falha)

i_w2, j_w2 = int(nx * 0.50), int(ny * 0.75)
poco_p2 = {(i_w2, j_w2): pwf}

plt.figure(figsize=(6, 5))
im_k2 = plt.imshow(np.log10(perm_p2_mD), extent=[0, Lx, 0, Ly], origin="lower", cmap="PuRd", vmin=-5.0, vmax=2.0)
cbar2 = plt.colorbar(im_k2)
cbar2.set_label(r"$\log_{10}(k\ [\mathrm{mD}])$")
plt.plot([x_falha, x_falha], [y_ini_falha, y_fim_falha], "k--", linewidth=2, label="Traço da Falha", zorder=3)
plt.plot(x_c[j_w2], y_c[i_w2], marker="D", color="black", markersize=8, label=f"Produtor ({pwf/1e5:.0f} bar)", zorder=5)
plt.title("Problema 2: Campo de Permeabilidade com Falha")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.legend(loc="upper right", framealpha=0.9)
plt.tight_layout()
salvar_e_fechar()

sim_p2 = simular_cenario("Problema 2", perm_p2, 2.0 * 3600.0, 100, poco_p2, {}, passos_p1)

plotar_evolucao_pressao(sim_p2["campos"], "Problema 2: Evolução Transiente da Pressão 2D com Barreira", pwf, p0, [(i_w2, j_w2)], falha_coords)
plotar_velocidade_darcy(sim_p2["campos"], perm_p2, "Problema 2: Campo Transiente de Velocidades de Darcy", pwf, p0, [(i_w2, j_w2)], falha_coords)
plotar_desempenho(sim_p2["tempo"], sim_p2["q_prod"], sim_p2["Np"], "Problema 2: Histórico de Produção")

validar_simulacao("Problema 2 — Produtor com Falha", sim_p2["P_final"], pwf, sim_p2["q_prod"][1:], sim_p2["Np"][-1])

# 6. EXECUÇÃO — PROBLEMA 3: FIVE-SPOT COMPLETO

res_p3 = gerar_permeabilidade_p3(nx, ny)
perm_p3_mD = np.asarray(res_p3[0] if isinstance(res_p3, tuple) else res_p3, dtype=float)
perm_p3 = perm_p3_mD * 9.869233e-16

plt.figure(figsize=(5.5, 4.5))
im_k3 = plt.imshow(perm_p3_mD, extent=[0, Lx, 0, Ly], origin="lower", cmap="PuRd")
plt.colorbar(im_k3, label="Permeabilidade [mD]")
plt.title("Problema 3: Campo de Permeabilidade")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.tight_layout()
salvar_e_fechar()

i_prod3, j_prod3 = nx // 2, ny // 2
poco_p3_prod = {(i_prod3, j_prod3): pwf}
offset = 3
inj_coords = [
    (offset, offset),
    (offset, ny - 1 - offset),
    (nx - 1 - offset, offset),
    (nx - 1 - offset, ny - 1 - offset)
]
qw_cada = 1.5e-3  # m³/s cada injetor
pocos_p3_inj = {pos: qw_cada for pos in inj_coords}
passos_p3 = [int(100 * 0.05), int(100 * 0.25), int(100 * 0.60), 100]

sim_p3 = simular_cenario("Problema 3", perm_p3, 4.0 * 3600.0, 100, poco_p3_prod, pocos_p3_inj, passos_p3)

p_max_p3 = max(campo.max() for campo in sim_p3["campos"].values())
plotar_evolucao_pressao(sim_p3["campos"], "Problema 3: Evolução Transiente da Pressão 2D (Five-Spot)", pwf, p_max_p3, [(i_prod3, j_prod3)], None, inj_coords)
plotar_velocidade_darcy(sim_p3["campos"], perm_p3, "Problema 3: Campo Transiente de Velocidades de Darcy (Five-Spot)", pwf, p_max_p3, [(i_prod3, j_prod3)], None, inj_coords)
plotar_desempenho(sim_p3["tempo"], sim_p3["q_prod"], sim_p3["Np"], "Problema 3: Resposta Temporal dos Poços", q_inj_total=4.0 * qw_cada)

Ninj_total3 = 4.0 * qw_cada * 4.0 * 3600.0
validar_simulacao("Problema 3 — Five-Spot", sim_p3["P_final"], pwf, sim_p3["q_prod"][1:], sim_p3["Np"][-1], Ninj_acum=Ninj_total3)

# 7. VALIDAÇÕES ANALÍTICAS 1D (AULA 12/13)

def validar_analitica_1d_heterogeneo_mvf():
    p_w_1d, p_e_1d = 100.0 * 1e5, 160.0 * 1e5
    bc_1d = {
        'oeste': {'tipo': 'pressao_prescrita', 'valor': p_w_1d},
        'leste': {'tipo': 'pressao_prescrita', 'valor': p_e_1d},
        'sul':   {'tipo': 'fluxo_nulo', 'valor': 0.0},
        'norte': {'tipo': 'fluxo_nulo', 'valor': 0.0}
    }

    k_1d_mD = perm_p1_mD[nx // 2, :]
    perm_hetero_1d = np.tile(k_1d_mD, (nx, 1)) * 9.869233e-16

    A_1d, B_bc_1d, C_dt_1d, _ = montar_sistema_mvf(perm_hetero_1d, 1000.0, bc_1d, {})
    A_est = A_1d - C_dt_1d * sp.eye(N, format="csr")
    P_num_1d = spla.spsolve(A_est, B_bc_1d).reshape((nx, ny))
    perfil_numerico = P_num_1d[nx // 2, :]

    resistencias = dx / perm_hetero_1d[nx // 2, :]
    R_total = np.sum(resistencias)
    p_analitica_hetero = np.zeros(ny)
    for j in range(ny):
        R_acumulada = np.sum(resistencias[:j]) + 0.5 * resistencias[j]
        p_analitica_hetero[j] = p_w_1d + (p_e_1d - p_w_1d) * (R_acumulada / R_total)

    erro_max = np.max(np.abs(perfil_numerico - p_analitica_hetero)) / 1e5
    print(f"\nErro máximo analítico MVF (Heterogêneo 1D Permanente): {erro_max:.4e} bar")

    plt.figure(figsize=(6.5, 4))
    plt.plot(x_c, p_analitica_hetero / 1e5, 'r-', linewidth=2, label='Analítica (Série Harmônica)')
    plt.plot(x_c, perfil_numerico / 1e5, 'ko', markersize=4, label='MVF 2D (Perfil Central)')
    plt.title("Validação 1D MVF em Meio Heterogêneo (Regime Permanente)")
    plt.xlabel("X [m]")
    plt.ylabel("Pressão [bar]")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    salvar_e_fechar()

def validar_analitica_1d_transiente_homogeneo_mvf():
    """
    Pressão prescrita em W e E, fluxo nulo em N e S, k homogêneo (Aulas 9 e 12: caso 1D q_t = q_b = 0).
    Analítica: p = p_ss + sum b_n sin(n pi x/L) exp(-eta (n pi/L)^2 t).
    """
    p_i = 130.0 * 1e5
    p_w_1d, p_e_1d = 100.0 * 1e5, 160.0 * 1e5
    k_hom = 150.0 * 9.869233e-16
    perm_hom = np.full((nx, ny), k_hom)
    bc_1d = {
        'oeste': {'tipo': 'pressao_prescrita', 'valor': p_w_1d},
        'leste': {'tipo': 'pressao_prescrita', 'valor': p_e_1d},
        'sul':   {'tipo': 'fluxo_nulo', 'valor': 0.0},
        'norte': {'tipo': 'fluxo_nulo', 'valor': 0.0}
    }
    dt_1d, t_alvo = 5.0, 3600.0
    n_passos = int(round(t_alvo / dt_1d))
    A_1d, B_bc_1d, C_dt_1d, _ = montar_sistema_mvf(perm_hom, dt_1d, bc_1d, {})
    lu = spla.splu(A_1d.tocsc())
    P = np.full(N, p_i)
    for _ in range(n_passos):
        P = lu.solve(C_dt_1d * P + B_bc_1d)
    perfil_num = P.reshape((nx, ny))[nx // 2, :]

    eta = k_hom / (phi * mu * c)
    p_an = p_w_1d + (p_e_1d - p_w_1d) * x_c / Lx
    for n in range(1, 201):
        bn = 2.0 / (n * np.pi) * ((p_i - p_w_1d) * (1 - (-1)**n) + (p_e_1d - p_w_1d) * (-1)**n)
        p_an += bn * np.sin(n * np.pi * x_c / Lx) * np.exp(-eta * (n * np.pi / Lx)**2 * t_alvo)

    erro_max = np.max(np.abs(perfil_num - p_an)) / 1e5
    print(f"Erro máximo analítico MVF (Homogêneo 1D Transiente, t = {t_alvo/3600:.1f} h): {erro_max:.4e} bar")

    plt.figure(figsize=(6.5, 4))
    plt.plot(x_c, p_an / 1e5, 'r-', linewidth=2, label='Analítica (Série de Fourier)')
    plt.plot(x_c, perfil_num / 1e5, 'ko', markersize=4, label='MVF 2D (Perfil Central)')
    plt.title("Validação 1D MVF Transiente em Meio Homogêneo")
    plt.xlabel("X [m]")
    plt.ylabel("Pressão [bar]")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    salvar_e_fechar()

validar_analitica_1d_heterogeneo_mvf()
validar_analitica_1d_transiente_homogeneo_mvf()

# 8. TESTE DE CONVERGÊNCIA TEMPORAL (REFINAMENTO DE dt)

def testar_convergencia_temporal():
    nt_refinado = 200
    sim_ref = simular_cenario("Convergência MVF P1 (nt=200)", perm_p1, 2.0 * 3600.0, nt_refinado, poco_p1, {}, [nt_refinado])
    diff_rel_Np = abs(sim_ref["Np"][-1] - sim_p1["Np"][-1]) / sim_p1["Np"][-1]
    diff_max_P = np.max(np.abs(sim_ref["P_final"] - sim_p1["P_final"])) / 1e5

    print("\n--- TESTE DE REFINAMENTO TEMPORAL (MVF P1: dt=72s vs dt=36s) ---")
    print(f"Np (nt=100): {sim_p1['Np'][-1]:.4f} m³ | Np (nt=200): {sim_ref['Np'][-1]:.4f} m³")
    print(f"Diferença relativa em Np: {diff_rel_Np:.3e}")
    print(f"Diferença máxima local de pressão: {diff_max_P:.4e} bar")

testar_convergencia_temporal()

# 9. TESTE DA CONDIÇÃO DE CONTORNO DE FLUXO PRESCRITO

def validar_teste_fluxo_prescrito():

    dt_teste = 36.0
    nt_teste = 50
    tempo_teste = nt_teste * dt_teste
    q_in_face = 1.0e-4  # m³/s entrando em cada célula da fronteira Oeste

    bc_teste = {
        'oeste': {'tipo': 'fluxo_prescrito', 'valor': q_in_face},
        'leste': {'tipo': 'fluxo_nulo', 'valor': 0.0},
        'sul':   {'tipo': 'fluxo_nulo', 'valor': 0.0},
        'norte': {'tipo': 'fluxo_nulo', 'valor': 0.0}
    }

    A, B_bc, C_dt, _ = montar_sistema_mvf(perm_p1, dt_teste, bc_teste, {})
    P_vetor = np.full(N, p0)
    for _ in range(nt_teste):
        P_vetor = spla.spsolve(A, C_dt * P_vetor + B_bc)

    P_final = P_vetor.reshape((nx, ny))
    volume_injetado = q_in_face * nx * tempo_teste
    delta_armaz = np.sum(Vb * phi * c * (P_final - p0))
    erro_balanco = abs(volume_injetado - delta_armaz)
    erro_relativo = erro_balanco / volume_injetado

    print("\n--- TESTE DE CONDIÇÃO DE CONTORNO: FLUXO PRESCRITO NA FRONTEIRA ---")
    print(f"Volume injetado via fronteira Oeste: {volume_injetado:.6f} m³")
    print(f"Variação de armazenamento calculada: {delta_armaz:.6f} m³")
    print(f"Erro relativo de balanço de volume:  {erro_relativo:.3e}")
    if erro_relativo < 1e-10:
        print("STATUS: PASSOU (Opção de contorno 'fluxo_prescrito' funcional e conservativa).")
    else:
        print("STATUS: FALHOU (balanço de volume não fechou).")

validar_teste_fluxo_prescrito()

print("\nPasta onde os gráficos foram salvos:")
print(os.path.abspath(PASTA_GRAFICOS))

if sys.platform.startswith("win"):
    os.startfile(os.path.abspath(PASTA_GRAFICOS))

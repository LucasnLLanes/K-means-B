# ============================================================
# Clustering de Estudiantes Universitarios con K-Means
# Autor: Lucas Nahuel Llanes
# Materia: Aprendizaje Automático
# ISFT N° 197 — 2026
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Paleta púrpura / coral / turquesa / amarillo
PURPURA  = "#5e35b1"
CORAL    = "#ff6f61"
TURQUESA = "#26c6da"
AMARILLO = "#f4b400"
TEXTO    = "#2d2d3a"

# ------------------------------------------------------------
# 1. Dataset
# ------------------------------------------------------------
registros = [
    {'edad': 42, 'horas_estudio': 2, 'promedio_academico': 8.87},
    {'edad': 26, 'horas_estudio': 8, 'promedio_academico': 9.11},
    {'edad': 56, 'horas_estudio': 5, 'promedio_academico': 7.29},
    {'edad': 29, 'horas_estudio': 14, 'promedio_academico': 7.27},
    {'edad': 43, 'horas_estudio': 8, 'promedio_academico': 8.5},
    {'edad': 34, 'horas_estudio': 10, 'promedio_academico': 4.45},
    {'edad': 51, 'horas_estudio': 3, 'promedio_academico': 6.12},
    {'edad': 22, 'horas_estudio': 12, 'promedio_academico': 4.33},
    {'edad': 47, 'horas_estudio': 6, 'promedio_academico': 7.0},
    {'edad': 38, 'horas_estudio': 9, 'promedio_academico': 4.75},
    {'edad': 60, 'horas_estudio': 2, 'promedio_academico': 9.0},
    {'edad': 31, 'horas_estudio': 11, 'promedio_academico': 2.0},
    {'edad': 45, 'horas_estudio': 4, 'promedio_academico': 8.0},
    {'edad': 27, 'horas_estudio': 13, 'promedio_academico': 3.2},
    {'edad': 50, 'horas_estudio': 5, 'promedio_academico': 7.0},
    {'edad': 36, 'horas_estudio': 7, 'promedio_academico': 5.0},
    {'edad': 40, 'horas_estudio': 6, 'promedio_academico': 6.0},
    {'edad': 24, 'horas_estudio': 15, 'promedio_academico': 3.0},
    {'edad': 55, 'horas_estudio': 3, 'promedio_academico': 9.0},
    {'edad': 33, 'horas_estudio': 10, 'promedio_academico': 3.0},
    {'edad': 48, 'horas_estudio': 4, 'promedio_academico': 8.0},
    {'edad': 28, 'horas_estudio': 12, 'promedio_academico': 2.0},
    {'edad': 52, 'horas_estudio': 5, 'promedio_academico': 7.0},
    {'edad': 30, 'horas_estudio': 11, 'promedio_academico': 2.0},
    {'edad': 46, 'horas_estudio': 6, 'promedio_academico': 6.0},
    {'edad': 35, 'horas_estudio': 9, 'promedio_academico': 4.0},
    {'edad': 41, 'horas_estudio': 7, 'promedio_academico': 5.0},
    {'edad': 23, 'horas_estudio': 14, 'promedio_academico': 1.0},
    {'edad': 53, 'horas_estudio': 3, 'promedio_academico': 8.0},
    {'edad': 32, 'horas_estudio': 10, 'promedio_academico': 3.0},
    {'edad': 49, 'horas_estudio': 4, 'promedio_academico': 8.0},
    {'edad': 25, 'horas_estudio': 13, 'promedio_academico': 2.0},
    {'edad': 54, 'horas_estudio': 5, 'promedio_academico': 7.0},
    {'edad': 37, 'horas_estudio': 8, 'promedio_academico': 4.0},
    {'edad': 39, 'horas_estudio': 7, 'promedio_academico': 5.0},
    {'edad': 21, 'horas_estudio': 15, 'promedio_academico': 9.0},
    {'edad': 58, 'horas_estudio': 2, 'promedio_academico': 9.0},
    {'edad': 44, 'horas_estudio': 6, 'promedio_academico': 6.0},
    {'edad': 29, 'horas_estudio': 12, 'promedio_academico': 2.0},
    {'edad': 50, 'horas_estudio': 5, 'promedio_academico': 7.0},
    {'edad': 34, 'horas_estudio': 9, 'promedio_academico': 3.0},
    {'edad': 42, 'horas_estudio': 7, 'promedio_academico': 6.0},
    {'edad': 26, 'horas_estudio': 14, 'promedio_academico': 3.0},
    {'edad': 57, 'horas_estudio': 3, 'promedio_academico': 9.0},
    {'edad': 30, 'horas_estudio': 11, 'promedio_academico': 2.0},
    {'edad': 47, 'horas_estudio': 4, 'promedio_academico': 7.0},
    {'edad': 31, 'horas_estudio': 10, 'promedio_academico': 3.0},
    {'edad': 45, 'horas_estudio': 6, 'promedio_academico': 6.0},
    {'edad': 27, 'horas_estudio': 13, 'promedio_academico': 1.0},
    {'edad': 55, 'horas_estudio': 5, 'promedio_academico': 8.0},
    {'edad': 33, 'horas_estudio': 8, 'promedio_academico': 6.5},
]

df = pd.DataFrame(registros)
print("Vista previa:")
print(df.head(), "\n")

# ------------------------------------------------------------
# 2. Estandarización
# ------------------------------------------------------------
X = df[["edad", "horas_estudio", "promedio_academico"]].values
escalador = StandardScaler()
X_esc = escalador.fit_transform(X)

# ------------------------------------------------------------
# 3. Método del codo
# ------------------------------------------------------------
inercias = []
valores_k = range(1, 11)

for k in valores_k:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_esc)
    inercias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(valores_k, inercias, marker="D", color=PURPURA,
         linewidth=2.5, markersize=8,
         markerfacecolor=CORAL, markeredgecolor=TEXTO)
plt.title("Método del Codo — Elección de K", color=PURPURA, fontsize=13)
plt.xlabel("Cantidad de Clusters (K)")
plt.ylabel("Inercia")
plt.xticks(valores_k)
plt.grid(True, alpha=0.25, linestyle="--")
plt.savefig("codo_estudiantes.png", dpi=150, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# 4. Modelo final K=4
# ------------------------------------------------------------
modelo = KMeans(n_clusters=4, random_state=42)
df["cluster"] = modelo.fit_predict(X_esc)

print("\nInercia final:", modelo.inertia_)
print("\nEstudiantes por cluster:")
print(df.groupby("cluster").size())

# ------------------------------------------------------------
# 5. Visualización
# ------------------------------------------------------------
paleta = {0: PURPURA, 1: CORAL, 2: TURQUESA, 3: AMARILLO}

plt.figure(figsize=(10, 6))
for c in range(4):
    grupo = df[df["cluster"] == c]
    plt.scatter(grupo["edad"], grupo["promedio_academico"],
                label=f"Grupo {c}", color=paleta[c],
                s=90, edgecolor=TEXTO, linewidth=0.6, alpha=0.9)

plt.xlabel("Edad")
plt.ylabel("Promedio académico")
plt.title("Segmentación de Estudiantes — K-Means", color=PURPURA, fontsize=13)
plt.legend(frameon=True, edgecolor=PURPURA)
plt.grid(True, alpha=0.25, linestyle="--")
plt.tight_layout()
plt.savefig("grafico_estudiantes.png", dpi=150, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# 6. Resumen
# ------------------------------------------------------------
print("\n=== RESUMEN DE CLUSTERS ===")
for c in range(4):
    grupo = df[df["cluster"] == c]
    print(f"\nGrupo {c} ({paleta[c]}): {len(grupo)} estudiantes")
    print(f"  Edad promedio: {grupo['edad'].mean():.1f}")
    print(f"  Horas estudio promedio: {grupo['horas_estudio'].mean():.1f}")
    print(f"  Promedio académico: {grupo['promedio_academico'].mean():.2f}")

print("\nListo.")
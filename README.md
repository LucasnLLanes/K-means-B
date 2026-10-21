# Clustering de Estudiantes Universitarios con K-Means

**Autor:** Lucas Nahuel Llanes  
**Materia:** Aprendizaje Automático  
**Carrera:** Tecnicatura en Ciencia de Datos e Inteligencia Artificial  
**Institución:** Instituto Superior de Formación Técnica N° 197  
**Año:** 2026

---

## 🎯 Propósito

Este repositorio reúne un notebook y un script de Python que aplican **K-Means** para segmentar 50 estudiantes universitarios según tres variables: **edad**, **horas de estudio semanales** y **promedio académico**. El objetivo es descubrir agrupamientos naturales que permitan diseñar intervenciones educativas personalizadas.

### Pipeline de trabajo

1. Carga del dataset sintético embebido.
2. Estandarización con `StandardScaler`.
3. Selección de K mediante el **método del codo**.
4. Entrenamiento final con `K = 4` (según consigna).
5. Visualización con paleta **púrpura / coral / turquesa**.
6. Interpretación pedagógica de los clusters.

---

## ⚙️ Dependencias

- Python 3.10+
- `pandas`
- `scikit-learn`
- `matplotlib`
- `numpy`

```bash
pip install pandas scikit-learn matplotlib numpy# K-means-B

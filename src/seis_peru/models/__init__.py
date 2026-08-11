"""Modelos de pronóstico y evaluación (Fase 4+).

Baselines sismológicos que definen el 'listón' a superar por el ML (H5):
climatología, Poisson y Gutenberg-Richter. Todos son funciones deterministas
del PASADO (features ≤ t), así que la evaluación en un periodo de test es
walk-forward por construcción.
"""

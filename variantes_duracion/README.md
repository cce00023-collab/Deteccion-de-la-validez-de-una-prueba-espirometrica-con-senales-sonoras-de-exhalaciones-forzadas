# Variantes de duración

Estos tres WAV son alternativas de los archivos publicados en `audios_metrica/`. No se incluyen en las 125 señales utilizadas para calcular las métricas de la memoria y no sustituyen los originales.

| Variante | Original | Cambio | Conclusión antes y después |
| --- | --- | --- | --- |
| [`sano18_1.20s.wav`](sano18_1.20s.wav) | [`sano18_S.wav`](../audios_metrica/sanos/sano18_S.wav), 0,82 s | Se prolongó el mismo corte hasta 1,20 s tomando las muestras siguientes de `SANOSMEMORIA/20260912141723_S.wav`. El original de 0,82 s coincide muestra a muestra con ese registro desde 2,99 s. No se añadió silencio ni se sintetizó audio. | `valida` en ambos; 0 eventos de tos y 1 PEF. |
| [`tos19_9.00s.wav`](tos19_9.00s.wav) | [`tos19_S.wav`](../audios_metrica/tos/tos19_S.wav), 11,726 s | Se conservaron los primeros 9,00 s y se retiró el tramo final del archivo. | `tos` en ambos; 2 eventos de tos y 1 PEF. |
| [`esfmult25_8.00s.wav`](esfmult25_8.00s.wav) | [`esfmult25_S.wav`](../audios_metrica/multiple/esfmult25_S.wav), 8,788 s | Se conservaron los primeros 8,00 s. | `esfuerzo_insuficiente` en ambos; 0 eventos de tos, 1 PEF y el mismo tiempo de PEF retrasado (6,99875 s). |

La comparación se hizo con `software/escritorio/script.py`, configurado para estetoscopio, con umbral de PEF retrasado de 0,150 s y umbral de terminación precoz de 0,60. Los WAV siguen siendo mono, PCM de 16 bits y 16 000 Hz. La grabación fuente utilizada para ampliar `sano18` no forma parte de este repositorio.

`esfmult25_S.wav` figura en la carpeta de esfuerzo múltiple, pero el programa clasifica tanto el original como el recorte como esfuerzo insuficiente. Además, su nivel digital es muy bajo (muestras entre -6 y 17 en PCM de 16 bits). El recorte conserva la salida del programa, pero no verifica la etiqueta de esfuerzo múltiple. Que se mantenga una salida tampoco demuestra que las variantes tengan la misma calidad clínica que las grabaciones originales.

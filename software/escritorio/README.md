# Código Python del TFG

| Archivo | Función |
| --- | --- |
| [Descargar script.py](https://github.com/cce00023-collab/Deteccion-de-la-validez-de-una-prueba-espirometrica-con-senales-sonoras-de-exhalaciones-forzadas/raw/refs/heads/main/software/escritorio/script.py) | Programa principal. Analiza uno o dos WAV y permite guardar los resultados. |
| [Descargar script_auxiliar.py](https://github.com/cce00023-collab/Deteccion-de-la-validez-de-una-prueba-espirometrica-con-senales-sonoras-de-exhalaciones-forzadas/raw/refs/heads/main/software/escritorio/script_auxiliar.py) | Analiza un WAV y permite seleccionar la fuente de captura. |
| [Descargar record_v3.py](https://github.com/cce00023-collab/Deteccion-de-la-validez-de-una-prueba-espirometrica-con-senales-sonoras-de-exhalaciones-forzadas/raw/refs/heads/main/software/escritorio/record_v3.py) | Graba desde el estetoscopio y el micrófono del ordenador. |
| [Descargar requirements.txt](https://github.com/cce00023-collab/Deteccion-de-la-validez-de-una-prueba-espirometrica-con-senales-sonoras-de-exhalaciones-forzadas/raw/refs/heads/main/software/escritorio/requirements.txt) | Dependencias de Python con sus versiones. |

## Ejecución en Windows

Instala Python 3.10 con Tkinter y guarda los archivos en la misma carpeta. Desde esa carpeta, instala las dependencias y abre el programa principal.

```powershell
py -3.10 -m pip install -r requirements.txt
py -3.10 script.py
```

Para abrir el programa auxiliar o el de grabación, ejecuta `py -3.10 script_auxiliar.py` o `py -3.10 record_v3.py`, respectivamente. Puedes probar el análisis con los [WAV del repositorio](../../audios_metrica/). Las salidas del programa auxiliar corresponden al análisis del archivo seleccionado. Las métricas históricas se recogen en la [memoria del TFG](../../memoria/memoria.pdf).

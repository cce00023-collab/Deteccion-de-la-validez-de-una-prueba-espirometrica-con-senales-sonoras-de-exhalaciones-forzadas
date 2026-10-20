# Detección de la validez de una prueba espirométrica mediante el análisis de señales sonoras de exhalaciones forzadas

**Trabajo Fin de Grado** · Grado en Ingeniería de Tecnologías de Telecomunicación  
**Universidad de Jaén** · Escuela Politécnica Superior de Linares  
**Autor:** Carlos Chica Extremera  
**Tutores:** Francisco Jesús Cañadas Quesada y Alejandro Antonio Salvador Navarro  
**Año:** 2026

El TFG estudia si el sonido de una exhalación forzada permite identificar maniobras espirométricas que requieren revisión. Este repositorio público reúne la memoria, las grabaciones utilizadas y los programas de análisis y adquisición. La base de datos contiene **125 archivos WAV**, organizados en cinco clases de 25. Cada archivo corresponde a una maniobra o segmento de maniobra; **125 archivos no equivalen a 125 participantes**.

**Acceso directo:** [Memoria (PDF)](memoria/memoria.pdf) · [Descargar memoria](https://github.com/cce00023-collab/Deteccion-de-la-validez-de-una-prueba-espirometrica-con-senales-sonoras-de-exhalaciones-forzadas/raw/refs/heads/main/memoria/memoria.pdf) · [Base de datos](audios_metrica/) · [Inventario de señales](inventario.csv) · [Aplicación Android (APK)](software/android/aplicacion_android.apk) · [Programas de escritorio](software/escritorio/)

<p align="center"><img src="figuras/proceso_grabacion.jpg" alt="Adquisición de una maniobra con espirómetro y estetoscopio digital" width="430"></p>
<p align="center"><em>Adquisición de la señal durante una maniobra de exhalación forzada. Fotografía incluida en la memoria.</em></p>

## Contenido y etiquetas

| Carpeta | Archivos | Significado de la etiqueta |
| --- | ---: | --- |
| `audios_metrica/sanos/` | 25 | Maniobras tomadas como referencia de ejecución válida desde el punto de vista acústico. `sano` no es un diagnóstico ni certifica la salud del participante. |
| `audios_metrica/tos/` | 25 | Maniobras con al menos un evento de tos. |
| `audios_metrica/multiple/` | 25 | Maniobras con más de un esfuerzo espiratorio distinguible. |
| `audios_metrica/esfuerzo_insuficiente/` | 25 | Maniobras con inicio o esfuerzo insuficiente, asociado en el análisis a un pico acústico retrasado. |
| `audios_metrica/precoz/` | 25 | Maniobras con terminación anticipada de la exhalación. |

Los nombres siguen el patrón `<clase><numero>_S.wav`, con números del 1 al 25 dentro de cada carpeta. Los prefijos son `sano`, `tos`, `esfmult`, `esfinsuf` y `precoz`. Por ejemplo, `esfmult7_S.wav` es el archivo 7 de la clase de esfuerzo múltiple. El sufijo `_S` indica que la fuente conservada es el **estetoscopio**; `_PC` se usó para el micrófono del ordenador y `_M` para el móvil durante otras pruebas, pero esas fuentes no forman parte de este conjunto. El número es un identificador de archivo dentro de la clase, no un identificador de persona ni una secuencia cronológica de adquisición.

## Formato y duración

Todos los archivos son WAV mono. **123** son PCM de 16 bits con frecuencia de muestreo de **16 000 Hz**. Las dos excepciones son `audios_metrica/esfuerzo_insuficiente/esfinsuf3_S.wav` y `audios_metrica/esfuerzo_insuficiente/esfinsuf14_S.wav`: son WAV de coma flotante de 32 bits a **4 000 Hz**. Estos valores describen los archivos distribuidos, no la frecuencia de trabajo del algoritmo, que remuestrea las entradas a 4 000 Hz durante el análisis.

El tamaño conjunto de los WAV es de aproximadamente **14,22 MB**. `inventario.csv` ofrece, para cada archivo, su clase, frecuencia de muestreo, canales, resolución, codificación, número de muestras, duración, tamaño y SHA-256. Las duraciones se calculan a partir del número de tramas y la frecuencia indicada en la cabecera WAV.

## Adquisición y preparación

Las grabaciones se obtuvieron entre abril y septiembre de 2026. Se registraron exhalaciones forzadas y se provocaron intencionadamente las incidencias de tos, esfuerzo múltiple, esfuerzo insuficiente y terminación precoz. La persona que realizó la prueba estaba sentada, con el tronco erguido; el estetoscopio digital **Thinklabs One** se colocó en el cuello y se procuró mantenerlo estable para reducir artefactos de movimiento. Durante las maniobras se utilizó un espirómetro **CONTEC SP-10**. No se confirmó clínicamente el estado de salud del participante.

El programa de adquisición permitía grabar simultáneamente el estetoscopio y el micrófono del ordenador. También se ensayó un móvil en pruebas preliminares. Para este conjunto se conservaron únicamente señales de estetoscopio. Durante la preparación se revisaron la fuente, la clase y el tramo útil de cada registro. Los WAV publicados aquí se copiaron byte por byte del paquete `documentacion.zip` facilitado con el proyecto: el empaquetado de este repositorio no aplica remuestreo, normalización ni reducción de ruido adicional.

## Representación de las señales

El espectrograma muestra cómo se distribuye la energía de una exhalación forzada en el tiempo y la frecuencia. Es una de las representaciones empleadas en la memoria para explicar el análisis acústico; no corresponde a un resultado diagnóstico por sí solo.

<p align="center"><img src="figuras/espectrograma_exhalacion.png" alt="Espectrograma STFT de una exhalación forzada, con tiempo en el eje horizontal y frecuencia en el vertical" width="760"></p>
<p align="center"><em>Espectrograma STFT de una exhalación forzada. Figura incluida en la memoria.</em></p>

## Organización

```text
audios_metrica/
  sanos/                  25 WAV
  tos/                    25 WAV
  multiple/               25 WAV
  esfuerzo_insuficiente/  25 WAV
  precoz/                 25 WAV
memoria/
  memoria.pdf             memoria del TFG
software/
  escritorio/             análisis y grabación en Python
  android/                aplicación para analizar un WAV local
inventario.csv            metadatos técnicos y SHA-256 de cada WAV
variantes_duracion/       tres versiones alternativas, fuera de la base de datos de 125 WAV
figuras/                  imágenes de la memoria utilizadas en este README
```

La selección de archivos se reconstruyó posteriormente a partir del material del proyecto y no constituye una validación independiente del clasificador. Las etiquetas describen las clases de trabajo del TFG; no son diagnósticos clínicos ni anotaciones independientes de especialistas. No se incluyen curvas espirométricas sincronizadas para confirmar por sí solas la aceptabilidad clínica de cada maniobra.

## Variantes de duración

En [`variantes_duracion/`](variantes_duracion/) hay tres versiones alternativas: una ampliación de `sano18_S.wav` de 0,82 a 1,20 s, un recorte de `tos19_S.wav` de 11,726 a 9,00 s y otro de `esfmult25_S.wav` de 8,788 a 8,00 s. El programa conserva en cada caso su conclusión original. En `esfmult25_S.wav`, esa conclusión es «esfuerzo insuficiente», aunque el archivo pertenece a la carpeta de esfuerzo múltiple. Los archivos de `audios_metrica/` no se han modificado; el [detalle de las pruebas](variantes_duracion/README.md) explica el origen y la comprobación de cada variante.

## Memoria y programas

- [`memoria/memoria.pdf`](memoria/memoria.pdf) es la memoria del TFG. Si la vista previa de GitHub no carga, puede [descargarse directamente](https://github.com/cce00023-collab/Deteccion-de-la-validez-de-una-prueba-espirometrica-con-senales-sonoras-de-exhalaciones-forzadas/raw/refs/heads/main/memoria/memoria.pdf).
- [`software/escritorio/script.py`](software/escritorio/script.py) es el programa de análisis; permite cargar uno o dos WAV y guardar los resultados en una carpeta elegida por el usuario.
- [`software/escritorio/script_auxiliar.py`](software/escritorio/script_auxiliar.py) analiza un WAV y permite elegir la fuente de captura. Es una variante del programa principal: sus salidas no deben confundirse con las métricas históricas de la memoria.
- [`software/escritorio/record_v3.py`](software/escritorio/record_v3.py) graba las entradas del estetoscopio y del micrófono del ordenador.
- [`software/escritorio/requirements.txt`](software/escritorio/requirements.txt) fija las versiones de las bibliotecas comprobadas en el equipo de desarrollo; este archivo se añadió para facilitar la instalación y no formaba parte del ZIP original.
- [`software/android/aplicacion_android.apk`](software/android/aplicacion_android.apk) es la aplicación Android incluida en el paquete. Abre un WAV local para analizarlo; no graba audio. Requiere Android 7.0 o superior y compatibilidad con aplicaciones de 32 bits.

Para ejecutar los programas de escritorio en Windows se necesita Python 3.10 con Tkinter. Desde `software/escritorio/`:

```powershell
py -3.10 -m pip install -r requirements.txt
py -3.10 script.py
```

También pueden ejecutarse `script_auxiliar.py` y `record_v3.py` con el mismo intérprete. El grabador necesita acceso a las entradas de audio del ordenador. Los tres archivos Python proceden del paquete de documentación y se han copiado sin modificar su código.

## Consulta y uso

El repositorio es público. Cualquier persona puede consultar los archivos o descargar el conjunto completo mediante **Code > Download ZIP**, sin solicitar acceso ni contactar con la persona responsable. No se ha asignado una licencia de reutilización al material.

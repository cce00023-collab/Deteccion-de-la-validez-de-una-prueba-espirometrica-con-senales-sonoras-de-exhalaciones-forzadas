"""Análisis acústico de exhalaciones forzadas.

Notación: y representa x[n], n es el índice de muestra y t_n=n/fs, en s.
fs se expresa en Hz; C es el número de canales. Los registros propios son
mono (C=1); para C>1 se prepara x_mono[n] promediando los canales del WAV.
Las amplitudes y potencias son relativas, sin calibración en Pa o W.
Las matrices espectrales usan filas de frecuencia/banda y columnas de tiempo.
raw identifica la rama original y den/denoised la rama tras reducción de ruido.
PEF es el máximo acústico definido por el detector, no un flujo medido en L/s.
La clasificación es la salida de un prototipo; no sustituye la espirometría.

El módulo puede importarse sin abrir la interfaz; main() selecciona los WAV.
"""

import os
import math
import tempfile
import platform
import subprocess
import builtins
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import scipy.ndimage as ndimage
from scipy.fft import next_fast_len
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from tkinter import Tk, filedialog


# Los diagnósticos internos se silencian; builtins.print muestra la salida final.
print = lambda *args, **kwargs: None


# lmediant se expresa en tramas; gain_floor limita la atenuación del módulo.
CONFIG_FUENTES = {
    "MOVIL": {
        "lmediant": 20,
        "denoising_strength": 0.10,
        "gain_floor": 0.85,
        "tos_onset_db": 5.0,
        "tos_offset_db": 2.5,
        "tos_pad_pre": 0.06,
        "tos_pad_post": 0.10
    },
    "PC": {
        "lmediant": 40,
        "denoising_strength": 0.70,
        "gain_floor": 0.85,
        "tos_onset_db": 5.0,
        "tos_offset_db": 2.5,
        "tos_pad_pre": 0.06,
        "tos_pad_post": 0.10
    },
    "ESTETOSCOPIO": {
        "lmediant": 40,
        "denoising_strength": 0.70,
        "gain_floor": 0.90,
        "tos_onset_db": 4.5,
        "tos_offset_db": 2.0,
        "tos_pad_pre": 0.06,
        "tos_pad_post": 0.10
    }
}


# fs=4000 Hz fija Nyquist en 2000 Hz.
target_fs = 4000


# Hamming de 256 muestras (64 ms a 4 kHz), con solapamiento del 50 %.
N_denoising = 256
S_denoising = 0.5
window_denoising = signal.windows.hamming(N_denoising)


# Hann para representación; B=40 bandas Mel.
N_stft_graficas = 256
S_stft_graficas = 0.5
n_mels=40


tiempo_suavizado_env = 0.25


preservar_tos = True
tos_duracion_min = 0.05
tos_duracion_max = 0.30
tos_rms_ventana = 0.020
tos_rms_salto = 0.010
tos_gain_floor = 0.90


def seleccionar_audio(nombre_dispositivo):
    """Selecciona un archivo WAV; la cancelación genera ValueError."""
    print(f"Selecciona el audio correspondiente a: {nombre_dispositivo}")
    ruta = filedialog.askopenfilename(
        title=f"Selecciona el audio de {nombre_dispositivo}",
        filetypes=[("Archivos WAV", "*.wav"), ("Todos los archivos", "*.*")]
    )
    if ruta == "":
        raise ValueError(f"No se ha seleccionado archivo para {nombre_dispositivo}")
    return ruta


def seleccionar_audio_opcional(nombre_dispositivo):
    """Selecciona un archivo WAV o devuelve None si se cancela."""
    print(f"Selecciona el audio correspondiente a: {nombre_dispositivo}")
    ruta = filedialog.askopenfilename(
        title=f"Selecciona el audio de {nombre_dispositivo}",
        filetypes=[("Archivos WAV", "*.wav"), ("Todos los archivos", "*.*")]
    )
    if ruta == "":
        print(f"No se ha seleccionado archivo para {nombre_dispositivo}.")
        return None
    return ruta

def convertir_a_float(y):
    """Convierte las muestras PCM a float64, sin calibración acústica.

    PCM16: x[n]=q[n]/32768; PCM32: q[n]/2147483648.
    PCM8 sin signo: x[n]=(q[n]-128)/128. Para otros tipos,
    divide por max(abs(x)) únicamente si ese máximo supera 1.
    """
    if y.dtype == np.int16: return y.astype(np.float64) / 32768.0
    elif y.dtype == np.int32: return y.astype(np.float64) / 2147483648.0
    elif y.dtype == np.uint8: return (y.astype(np.float64) - 128.0) / 128.0
    else:
        y = y.astype(np.float64)
        max_y = np.max(np.abs(y))
        if max_y > 1: y = y / max_y
        return y

def normalizar_audio(y):
    """Normaliza x[n] por a=max(abs(x[n])) si a>0.

    Conserva registros vacíos o nulos. La amplitud resultante es relativa;
    no representa presión sonora en Pa ni flujo espiratorio en L/s.
    """
    y = np.asarray(y, dtype=np.float64)
    if len(y) == 0: return y
    max_y = np.max(np.abs(y))
    if max_y > 0: y = y / max_y
    return y

def remuestrear_audio(y, fs_original, fs_objetivo):
    """Remuestrea mediante filtrado polifásico antialias.

    Con d=mcd(fs_original, fs_objetivo), los factores enteros son
    up=fs_objetivo/d y down=fs_original/d. Devuelve señal y nueva fs.
    """
    if fs_original == fs_objetivo: return y, fs_original
    mcd = math.gcd(fs_original, fs_objetivo)
    up = fs_objetivo // mcd
    down = fs_original // mcd
    y_resampled = signal.resample_poly(y, up, down)
    return y_resampled, fs_objetivo


def suavizar_senal_cruda_25ms(y, fs):
    """Obtiene la RMS centrada de 25 ms para representación temporal.

    r[n]=sqrt(mean(x[n+k]**2)) en una ventana de round(0.025*fs)
    muestras, con extensión de los extremos. Normaliza la señal y la RMS
    por sus respectivos máximos. Esta curva no es la envolvente e(t)
    ni modifica las muestras utilizadas por los detectores.
    """

    y = np.asarray(y, dtype=np.float64)

    if len(y) == 0 or fs <= 0:
        return y.copy()


    max_y = np.max(np.abs(y))
    if max_y > 0:
        y = y / max_y

    ventana = max(1, int(round(0.025 * fs)))


    pad_izq = ventana // 2
    pad_der = ventana - 1 - pad_izq

    y_pad = np.pad(y, (pad_izq, pad_der), mode="edge")
    kernel = np.ones(ventana, dtype=np.float64) / ventana

    energia = np.convolve(y_pad ** 2, kernel, mode="valid")
    rms = np.sqrt(np.maximum(energia, 0.0))


    max_rms = np.max(rms)
    if max_rms > 0:
        rms = rms / max_rms

    return rms


def quitar_repunte_final_logmel_por_actividad(
    logmel,
    t_logmel,
    y_ref,
    fs_ref,
    umbral_rms=0.025,
    ventana_rms_s=0.025,
    zona_final_s=0.50
):
    """Aplica un suelo de -80 dB a columnas finales con RMS baja.

    Usa y_ref para localizar tramas con RMS relativa < umbral_rms en
    zona_final_s. Devuelve una copia de Log-Mel; no modifica el WAV.
    preparar_graficas_fuente usa esta matriz para indicadores posteriores;
    por tanto, el enmascaramiento puede influir en la decisión espectral.
    """

    logmel = np.asarray(logmel, dtype=np.float64).copy()
    t_logmel = np.asarray(t_logmel, dtype=np.float64)
    y_ref = np.asarray(y_ref, dtype=np.float64)

    if logmel.size == 0 or len(t_logmel) == 0 or len(y_ref) == 0 or fs_ref <= 0:
        return logmel


    if logmel.ndim == 1:
        logmel = logmel.reshape(1, -1)

    n_frames = min(logmel.shape[1], len(t_logmel))
    if n_frames == 0:
        return logmel

    logmel[:, :n_frames] = logmel[:, :n_frames]


    y_norm = normalizar_audio(y_ref)
    ventana = max(1, int(round(ventana_rms_s * fs_ref)))

    energia = np.convolve(
        y_norm ** 2,
        np.ones(ventana, dtype=np.float64) / ventana,
        mode="same"
    )
    rms = np.sqrt(np.maximum(energia, 0.0))

    max_rms = np.max(rms)
    if max_rms > 0:
        rms = rms / max_rms

    duracion = len(y_ref) / fs_ref


    idx = np.round(t_logmel[:n_frames] * fs_ref).astype(int)
    idx = np.clip(idx, 0, len(rms) - 1)

    rms_en_frames = rms[idx]


    zona_final = t_logmel[:n_frames] >= max(0.0, duracion - zona_final_s)
    baja_actividad = rms_en_frames < umbral_rms

    mascara_borrar = zona_final & baja_actividad

    if np.any(mascara_borrar):
        logmel[:, :n_frames][:, mascara_borrar] = -80.0

    return logmel


def detectar_mascara_tos_en_muestras(fs, y, config):
    """Detecta candidatos de tos y máximos acústicos sobre x[n].

    Construye u[n] a partir de RMS de 25 ms, paso bajo de 6 Hz y
    suavizado gaussiano de 20 ms; normaliza u[n] por su máximo.
    Los límites de cada evento son los cruces al 10 % de su altura.
    La tos requiere duración de 0.05-0.30 s y condiciones de forma;
    Welch sobre el audio, no sobre u[n], aporta confirmación espectral.
    Devuelve una máscara booleana por muestra y conserva los eventos
    en los atributos ultimos_eventos y ultimos_maximos de la función.
    PEF designa aquí un máximo acústico, no una medida clínica de flujo.
    """

    y = normalizar_audio(y)
    n = len(y)

    mascara_tos = np.zeros(n, dtype=bool)

    if n < 10 or np.max(np.abs(y)) <= 0:
        detectar_mascara_tos_en_muestras.ultimos_eventos = []
        detectar_mascara_tos_en_muestras.ultimos_maximos = []
        return mascara_tos


    # Welch recibe el audio; la curva RMS suavizada solo caracteriza la forma temporal.
    y_audio_para_espectro = y.copy()


    def confirmar_tos_por_banda_500_1500(audio_real_evento, fs_audio, idx_inicio, idx_fin):
        """Confirma si el máximo de Welch entre 100 y 1800 Hz cae en 500-1500 Hz.

        La razón sum(P[500:1500])/sum(P[f>=100]) es un descriptor;
        no se compara con un umbral independiente para confirmar tos.
        """
        idx_inicio = max(0, int(idx_inicio))
        idx_fin = min(len(audio_real_evento), int(idx_fin))

        segmento = audio_real_evento[idx_inicio:idx_fin]

        if len(segmento) < max(16, int(0.020 * fs_audio)):
            return {
                "confirmacion_espectral": False,
                "frecuencia_dominante": 0.0,
                "ratio_500_1500": 0.0
            }

        segmento = segmento - np.mean(segmento)

        if np.max(np.abs(segmento)) > 0:
            segmento = segmento / np.max(np.abs(segmento))

        nperseg = min(len(segmento), max(64, int(0.064 * fs_audio)))
        nperseg = max(16, nperseg)

        try:
            f, pxx = signal.welch(
                segmento,
                fs=fs_audio,
                window="hann",
                nperseg=nperseg,
                noverlap=nperseg // 2,
                scaling="spectrum"
            )
        except Exception:
            return {
                "confirmacion_espectral": False,
                "frecuencia_dominante": 0.0,
                "ratio_500_1500": 0.0
            }

        if len(pxx) == 0 or np.max(pxx) <= 0:
            return {
                "confirmacion_espectral": False,
                "frecuencia_dominante": 0.0,
                "ratio_500_1500": 0.0
            }


        banda_util = (f >= 100) & (f <= min(1800, fs_audio / 2))

        if np.any(banda_util):
            idx_util = np.where(banda_util)[0]
            idx_dom = idx_util[int(np.argmax(pxx[idx_util]))]
        else:
            idx_dom = int(np.argmax(pxx))

        frecuencia_dominante = float(f[idx_dom])

        banda_tos = (f >= 500) & (f <= 1500)
        energia_500_1500 = float(np.sum(pxx[banda_tos]))
        energia_total_util = float(np.sum(pxx[f >= 100]))

        if energia_total_util > 0:
            ratio_500_1500 = energia_500_1500 / energia_total_util
        else:
            ratio_500_1500 = 0.0

        confirmacion_espectral = 500.0 <= frecuencia_dominante <= 1500.0

        return {
            "confirmacion_espectral": bool(confirmacion_espectral),
            "frecuencia_dominante": frecuencia_dominante,
            "ratio_500_1500": float(ratio_500_1500)
        }


    y_abs = np.abs(y)

    ventana_rms = max(1, int(0.025 * fs))
    energia = np.convolve(y_abs ** 2, np.ones(ventana_rms) / ventana_rms, mode="same")
    curva = np.sqrt(np.maximum(energia, 0))

    nyq = fs / 2.0
    fc = 6.0
    fc = min(fc, nyq * 0.90)

    if fc > 0 and len(curva) > 20:
        try:
            b, a = signal.butter(4, fc / nyq, btype="low")
            padlen = min(3 * max(len(a), len(b)), len(curva) - 1)

            if padlen > 0:
                curva = signal.filtfilt(b, a, curva, padlen=padlen)
            else:
                curva = signal.lfilter(b, a, curva)

        except Exception:
            pass

    curva = np.maximum(curva, 0)

    sigma = max(1, int(0.020 * fs))
    curva = ndimage.gaussian_filter1d(curva, sigma=sigma, mode="reflect")

    if np.max(curva) > 0:
        curva = curva / np.max(curva)


    def medir_evento_al_10(idx_pico):
        """Mide los cruces al 10 % del máximo y T=(n_d-n_i)/fs."""
        valor_pico = curva[idx_pico]

        if valor_pico <= 0:
            return None

        nivel_10 = 0.10 * valor_pico

        idx_izq = idx_pico
        while idx_izq > 0 and curva[idx_izq] > nivel_10:
            idx_izq -= 1

        idx_der = idx_pico
        while idx_der < n - 1 and curva[idx_der] > nivel_10:
            idx_der += 1

        duracion_s = (idx_der - idx_izq) / fs

        return idx_izq, idx_der, duracion_s, float(valor_pico)


    duracion_min_pef_al_10_s = 0.30
    muestras_min_pef_al_10 = int(round(duracion_min_pef_al_10_s * fs))


    def detectar_pef_por_curvas_reales(curva, fs):
        """Agrupa máximos de u[n] mediante duración, separación y valle.

        Un segundo esfuerzo exige separación >=0.30 s, duración al 10 %
        >=0.30 s, caída y recuperación >=0.30 y valle relativo <=0.65.
        La lógica de solapamiento se aplica después de estas condiciones.
        """

        if len(curva) == 0 or np.max(curva) <= 0:
            return []

        curva = np.asarray(curva, dtype=np.float64)

        umbral_actividad = 0.12
        prominencia_minima = 0.08


        tiempo_min_segundo_pef_s = 0.30


        caida_minima_valle_segundo_pef = 0.30


        valle_relativo_max_segundo_pef = 0.65


        subida_minima_desde_valle_segundo_pef = 0.30


        tolerancia_solape_misma_curva_s = 0.0
        valle_relativo_max_pef_solapado_independiente = 0.45
        caida_min_pef_solapado_independiente = 0.50
        subida_min_pef_solapado_independiente = 0.45
        tiempo_max_pef_solapado_independiente_s = 1.05
        solape_min_pef_solapado_independiente_s = -1.60
        pico_candidato_min_pef_solapado_independiente = 0.70
        pico_candidato_max_pef_solapado_independiente = 0.90


        def detectar_primer_pef_antes_de_bajada(curva_local, fs_local):
            """Localiza el máximo del primer intervalo activo.

            El intervalo empieza con u>=0.12 y termina al mantenerse
            u<=0.40 durante 0.12 s; devuelve su máximo absoluto.
            """

            curva_local = np.asarray(curva_local, dtype=np.float64)

            if len(curva_local) == 0 or np.max(curva_local) <= 0:
                return None

            idx_activos = np.where(curva_local >= umbral_actividad)[0]

            if len(idx_activos) == 0:
                return None

            idx_inicio = int(idx_activos[0])

            umbral_basal_bloque = 0.40
            min_basal_sostenido_s = 0.120
            muestras_basal = max(1, int(round(min_basal_sostenido_s * fs_local)))

            idx_fin_bloque = len(curva_local)
            debajo_basal = curva_local <= umbral_basal_bloque

            for i in range(idx_inicio + 1, max(idx_inicio + 1, len(curva_local) - muestras_basal + 1)):
                if np.all(debajo_basal[i:i + muestras_basal]):
                    idx_fin_bloque = i
                    break

            if idx_fin_bloque <= idx_inicio:
                idx_fin_bloque = len(curva_local)

            tramo_primer_bloque = curva_local[idx_inicio:idx_fin_bloque]

            if len(tramo_primer_bloque) == 0:
                return None

            idx_pef_real = int(idx_inicio + np.argmax(tramo_primer_bloque))

            if curva_local[idx_pef_real] < umbral_actividad:
                return None

            return idx_pef_real

        idx_primer_pef = detectar_primer_pef_antes_de_bajada(curva, fs)

        if idx_primer_pef is None:
            return []

        if curva[idx_primer_pef] < umbral_actividad:
            return []

        pefs = [int(idx_primer_pef)]


        distancia_min = max(1, int(0.080 * fs))

        try:
            maximos, props = signal.find_peaks(
                curva,
                height=umbral_actividad,
                distance=distancia_min,
                prominence=prominencia_minima
            )
        except Exception:
            maximos = []

        maximos = list(maximos)


        if idx_primer_pef not in maximos:
            maximos.append(idx_primer_pef)

        maximos = sorted([int(m) for m in maximos])


        candidatos_post_primer_pef = [idx for idx in maximos if idx > idx_primer_pef]

        for idx in candidatos_post_primer_pef:
            idx = int(idx)
            ultimo = int(pefs[-1])

            tiempo_entre_pefs = (idx - ultimo) / fs


            if tiempo_entre_pefs < tiempo_min_segundo_pef_s:
                continue

            ini = min(ultimo, idx)
            fin = max(ultimo, idx)

            if fin <= ini:
                continue


            tramo_entre_picos = curva[ini:fin + 1]
            idx_valle_local = int(np.argmin(tramo_entre_picos))
            idx_valle = ini + idx_valle_local
            valle = float(curva[idx_valle])

            pico_anterior = float(curva[ultimo])
            pico_candidato = float(curva[idx])
            pico_menor = float(min(pico_anterior, pico_candidato))


            caida_desde_pef_anterior = pico_anterior - valle


            if pico_menor > 0:
                valle_relativo = valle / pico_menor
            else:
                valle_relativo = 1.0


            subida_desde_valle = pico_candidato - valle

            med_candidato = medir_evento_al_10(idx)

            if med_candidato is not None:
                idx_izq_cand, idx_der_cand, duracion_candidata_s, _ = med_candidato
                duracion_candidata_muestras = idx_der_cand - idx_izq_cand
            else:
                duracion_candidata_s = 0.0
                duracion_candidata_muestras = 0

            med_ultimo = medir_evento_al_10(ultimo)

            misma_curva_continua = False
            separacion_basal_entre_eventos_s = 0.0

            if med_ultimo is not None and med_candidato is not None:
                _, idx_der_ult, _, _ = med_ultimo
                idx_izq_cand = int(idx_izq_cand)
                idx_der_ult = int(idx_der_ult)
                separacion_basal_entre_eventos_s = (idx_izq_cand - idx_der_ult) / fs
                misma_curva_continua = (
                    separacion_basal_entre_eventos_s
                    <= tolerancia_solape_misma_curva_s
                )

            valle_profundo_con_reascenso = (
                caida_desde_pef_anterior >= caida_min_pef_solapado_independiente
                and valle_relativo <= valle_relativo_max_pef_solapado_independiente
                and subida_desde_valle >= subida_min_pef_solapado_independiente
                and tiempo_entre_pefs <= tiempo_max_pef_solapado_independiente_s
                and separacion_basal_entre_eventos_s <= solape_min_pef_solapado_independiente_s
                and pico_candidato >= pico_candidato_min_pef_solapado_independiente
                and pico_candidato <= pico_candidato_max_pef_solapado_independiente
            )

            if misma_curva_continua and valle_profundo_con_reascenso:
                misma_curva_continua = False

            if misma_curva_continua:
                if pico_candidato > pico_anterior:
                    pefs[-1] = idx
                continue

            segundo_pef_valido = (
                tiempo_entre_pefs >= tiempo_min_segundo_pef_s
                and caida_desde_pef_anterior >= caida_minima_valle_segundo_pef
                and valle_relativo <= valle_relativo_max_segundo_pef
                and subida_desde_valle >= subida_minima_desde_valle_segundo_pef
                and duracion_candidata_muestras >= muestras_min_pef_al_10
            )

            if segundo_pef_valido:
                pefs.append(idx)

        return pefs

    idx_pefs = detectar_pef_por_curvas_reales(curva, fs)

    eventos_pef = []

    for idx_pico in idx_pefs:
        med = medir_evento_al_10(idx_pico)

        if med is None:
            continue

        idx_izq, idx_der, duracion_s, valor_pico = med
        duracion_muestras_pef = idx_der - idx_izq


        if duracion_muestras_pef >= muestras_min_pef_al_10:
            eventos_pef.append({
                "idx_inicio": int(idx_izq),
                "idx_pico": int(idx_pico),
                "idx_fin": int(idx_der),
                "inicio": idx_izq / fs,
                "pico": idx_pico / fs,
                "fin": idx_der / fs,
                "duracion": duracion_s,
                "duracion10_muestras": int(duracion_muestras_pef),
                "valor_pico": float(valor_pico),
                "tipo": "PEF",
                "frecuencia_dominante": 0.0,
                "confirmacion_espectral": False,
                "ratio_500_1500": 0.0
            })


    primera_derivada = np.gradient(curva)
    segunda_derivada = np.gradient(primera_derivada)

    candidatos = []

    for i in range(1, n - 1):
        cambio_signo = primera_derivada[i - 1] > 0 and primera_derivada[i + 1] <= 0
        concavidad_negativa = segunda_derivada[i] < 0

        if cambio_signo and concavidad_negativa:
            candidatos.append(i)

    distancia_min_picos_tos = max(1, int(0.080 * fs))
    maximos_agrupados = []

    if len(candidatos) > 0:
        grupo = [candidatos[0]]

        for idx in candidatos[1:]:
            if idx - grupo[-1] <= distancia_min_picos_tos:
                grupo.append(idx)
            else:
                mejor = grupo[int(np.argmax(curva[grupo]))]
                maximos_agrupados.append(mejor)
                grupo = [idx]

        mejor = grupo[int(np.argmax(curva[grupo]))]
        maximos_agrupados.append(mejor)

    maximos_forma_valida = []
    maximos_descartados_forma = []

    ventana_check = max(1, int(0.160 * fs))
    ventana_pre = max(1, int(0.080 * fs))

    caida_minima_relativa = 0.10
    subida_minima_relativa = 0.06
    tolerancia_pico_posterior = 1.03

    for idx_pico in maximos_agrupados:
        valor_pico = curva[idx_pico]

        ini_pre = max(0, idx_pico - ventana_pre)
        fin_post = min(n - 1, idx_pico + ventana_check)

        min_antes = np.min(curva[ini_pre:idx_pico + 1])
        min_despues = np.min(curva[idx_pico:fin_post + 1])
        max_despues = np.max(curva[idx_pico:fin_post + 1])

        subida_previa = valor_pico - min_antes
        caida_posterior = valor_pico - min_despues

        hay_otro_mas_alto_despues = max_despues > (valor_pico * tolerancia_pico_posterior)

        if subida_previa < subida_minima_relativa:
            maximos_descartados_forma.append((idx_pico, "sin_subida_previa"))
            continue

        if caida_posterior < caida_minima_relativa:
            maximos_descartados_forma.append((idx_pico, "caida_insuficiente"))
            continue

        if hay_otro_mas_alto_despues:
            maximos_descartados_forma.append((idx_pico, "rampa_creciente"))
            continue

        maximos_forma_valida.append(idx_pico)

    muestras_min_tos = int(0.05 * fs)
    muestras_max_tos = int(0.30 * fs)

    eventos_tos = []
    eventos_tos_segura = []
    eventos_tos_duracion = []
    eventos_artefacto = []

    for idx_pico in maximos_forma_valida:
        med = medir_evento_al_10(idx_pico)

        if med is None:
            continue

        idx_izq, idx_der, duracion_s, valor_pico = med
        duracion_muestras = idx_der - idx_izq


        cerca_de_pef = False
        for ev_pef in eventos_pef:
            if abs(ev_pef["idx_pico"] - idx_pico) <= int(0.080 * fs):
                cerca_de_pef = True
                break

        if cerca_de_pef:
            continue

        if duracion_muestras < muestras_min_tos:
            eventos_artefacto.append({
                "idx_inicio": int(idx_izq),
                "idx_pico": int(idx_pico),
                "idx_fin": int(idx_der),
                "inicio": idx_izq / fs,
                "pico": idx_pico / fs,
                "fin": idx_der / fs,
                "duracion": duracion_s,
                "valor_pico": float(valor_pico),
                "tipo": "artefacto",
                "frecuencia_dominante": 0.0,
                "confirmacion_espectral": False,
                "ratio_500_1500": 0.0
            })
            continue

        if duracion_muestras <= muestras_max_tos:
            info_espectral = confirmar_tos_por_banda_500_1500(
                y_audio_para_espectro,
                fs,
                idx_izq,
                idx_der
            )

            if info_espectral["confirmacion_espectral"]:
                tipo = "tos_segura"
            else:
                tipo = "tos_por_duracion"

            evento_tos = {
                "idx_inicio": int(idx_izq),
                "idx_pico": int(idx_pico),
                "idx_fin": int(idx_der),
                "inicio": idx_izq / fs,
                "pico": idx_pico / fs,
                "fin": idx_der / fs,
                "duracion": duracion_s,
                "valor_pico": float(valor_pico),
                "tipo": tipo,
                "frecuencia_dominante": info_espectral["frecuencia_dominante"],
                "confirmacion_espectral": info_espectral["confirmacion_espectral"],
                "ratio_500_1500": info_espectral["ratio_500_1500"]
            }

            eventos_tos.append(evento_tos)

            if tipo == "tos_segura":
                eventos_tos_segura.append(evento_tos)
            else:
                eventos_tos_duracion.append(evento_tos)

        else:


            eventos_artefacto.append({
                "idx_inicio": int(idx_izq),
                "idx_pico": int(idx_pico),
                "idx_fin": int(idx_der),
                "inicio": idx_izq / fs,
                "pico": idx_pico / fs,
                "fin": idx_der / fs,
                "duracion": duracion_s,
                "valor_pico": float(valor_pico),
                "tipo": "artefacto",
                "frecuencia_dominante": 0.0,
                "confirmacion_espectral": False,
                "ratio_500_1500": 0.0
            })


    padding_pre = int(config.get("tos_pad_pre", 0.06) * fs)
    padding_post = int(config.get("tos_pad_post", 0.10) * fs)

    for ev in eventos_tos:
        ini_p = max(0, ev["idx_inicio"] - padding_pre)
        fin_p = min(n, ev["idx_fin"] + padding_post)
        mascara_tos[ini_p:fin_p] = True


    eventos = eventos_tos + eventos_artefacto + eventos_pef
    eventos = sorted(eventos, key=lambda e: e["pico"])

    print("\n" + "="*110)
    print("        DETECCION DE TOS + PEF POR CURVAS REALES")
    print("="*110)
    print("  Curva:                         RMS 25 ms + LPF Butterworth 4º fc=6 Hz + gauss 20 ms")
    print("  PEF:                           1 curva real = 1 máximo principal")
    print("  Criterio PEF:                  1º PEF por curva real; 2º PEF exige tiempo + valle basal bajo + duración inicio10-fin10 >= 300 ms")
    print("  Tos:                           máximo local + duración 0.05-0.30 s")
    print("  Tos segura:                    duración compatible + frecuencia dominante 500-1500 Hz")
    print("  Eventos largos no principales: descartados como hombros/ondulaciones")
    print(f"  Máximos locales candidatos tos: {len(maximos_agrupados)}")
    print(f"  Máximos descartados forma:      {len(maximos_descartados_forma)}")
    print(f"  PEF por curvas reales:          {len(eventos_pef)}")
    print(f"  Tos totales detectadas:         {len(eventos_tos)}")
    print(f"  Tos seguras 500-1500 Hz:        {len(eventos_tos_segura)}")
    print(f"  Tos solo por duración:          {len(eventos_tos_duracion)}")
    print(f"  Artefactos/hombros descartados: {len(eventos_artefacto)}")

    if len(eventos) > 0:
        print("\n  EVENTOS:")
        for k, ev in enumerate(eventos, 1):
            print(
                f"    Evento {k}: "
                f"tipo={ev['tipo']} | "
                f"inicio10={ev['inicio']:.3f}s | "
                f"pico={ev['pico']:.3f}s | "
                f"fin10={ev['fin']:.3f}s | "
                f"duracion10={ev['duracion']*1000:.1f} ms | "
                f"f_dom={ev.get('frecuencia_dominante', 0.0):.1f} Hz | "
                f"ratio500_1500={ev.get('ratio_500_1500', 0.0):.3f} | "
                f"valor={ev['valor_pico']:.3f}"
            )

    if len(maximos_descartados_forma) > 0:
        print("\n  MAXIMOS DESCARTADOS POR FORMA:")
        for k, (idx, motivo) in enumerate(maximos_descartados_forma, 1):
            print(
                f"    Descartado {k}: "
                f"t={idx/fs:.3f}s | "
                f"motivo={motivo} | "
                f"valor={curva[idx]:.3f}"
            )

    print("="*110 + "\n")

    detectar_mascara_tos_en_muestras.ultimos_eventos = eventos
    detectar_mascara_tos_en_muestras.ultimos_maximos = [
        ev["idx_pico"] for ev in eventos if ev["tipo"] != "artefacto"
    ]

    return mascara_tos


def convertir_mascara_muestras_a_frames(mascara_muestras, times, fs, longitud_senal):
    """Asigna a cada centro de trama el estado de la máscara de muestras."""
    mascara_frames = np.zeros(len(times), dtype=bool)
    for i, tiempo_frame in enumerate(times):
        muestra = int(tiempo_frame * fs)
        muestra = min(max(muestra, 0), longitud_senal - 1)
        mascara_frames[i] = mascara_muestras[muestra]
    return mascara_frames


def aplicar_denoising(fs, y, N, S, window, config):
    """Atenúa el módulo espectral y reconstruye manteniendo la fase.

    A=abs(X), D=min(mediana_temporal(A), A), R=max(A-D, 0).
    La máscara es M=D**2/(D**2+R**2+eps). Con intensidad alpha,
    A_hat=max((1-alpha+alpha*M)*A, gain_floor*A).
    En las tramas de tos se añade el suelo tos_gain_floor*A.
    La ISTFT reconstruye A_hat*exp(1j*angle(X)); se ajusta la longitud
    al audio de entrada. No restaura información perdida por clipping.
    """
    lmediant = config["lmediant"]
    denoising_strength = config["denoising_strength"]
    denoising_gain_floor_global = config["gain_floor"]

    if y.ndim == 2: y = np.mean(y, axis=1)
    y = y.astype(float)

    if np.max(np.abs(y)) == 0:
        raise ValueError("La señal no tiene contenido util.")
    y = y / np.max(np.abs(y))

    if preservar_tos:
        mascara_tos_muestras = detectar_mascara_tos_en_muestras(fs, y, config)
    else:
        mascara_tos_muestras = np.zeros(len(y), dtype=bool)

    noverlap = int(N * S)
    nfft = next_fast_len(2 * N)

    frequencies, times, Spec_complexX = signal.spectrogram(
        y, fs, window=window, nperseg=N, noverlap=noverlap, nfft=nfft, mode='complex'
    )

    Spec_absX = np.abs(Spec_complexX)
    Spec_angleX = np.angle(Spec_complexX)
    INPUT = Spec_absX

    mascara_tos_frames = convertir_mascara_muestras_a_frames(
        mascara_tos_muestras, times, fs, len(y)
    )


    # La mediana es temporal: size=(1,lmediant) no mezcla frecuencias.
    TMP_DEN = ndimage.median_filter(INPUT, size=(1, lmediant), mode='reflect')
    TMP_DEN = np.minimum(TMP_DEN, INPUT)

    TMP_NONDEN = np.maximum(0, Spec_absX - TMP_DEN)
    p = 2
    MDEN = (TMP_DEN ** p) / (TMP_DEN ** p + TMP_NONDEN ** p + np.finfo(float).eps)

    Xdenois_fuerte = MDEN * Spec_absX
    Xdenois = denoising_strength * Xdenois_fuerte + (1.0 - denoising_strength) * Spec_absX

    Xdenois = np.maximum(Xdenois, denoising_gain_floor_global * Spec_absX)

    if preservar_tos and np.any(mascara_tos_frames):
        Xdenois[:, mascara_tos_frames] = np.maximum(
            Xdenois[:, mascara_tos_frames],
            tos_gain_floor * Spec_absX[:, mascara_tos_frames]
        )

    _, y_den = signal.istft(
        Xdenois * np.exp(1j * Spec_angleX),
        fs, window=window, nperseg=N, noverlap=noverlap, nfft=nfft
    )

    if len(y_den) > len(y): y_den = y_den[:len(y)]
    elif len(y_den) < len(y): y_den = np.pad(y_den, (0, len(y) - len(y_den)))

    max_original = np.max(np.abs(y))
    if max_original > 0:
       y_den = np.clip(y_den, -max_original, max_original)
    return y_den


def controlar_tos_saturada_antes_normalizar(fs, y, config):
    """Reduce una tos detectada si domina la referencia de normalización.

    La referencia es el percentil 99 de abs(x) fuera de la máscara;
    si no existe esa región se usa el percentil 95 global. Si el pico
    de tos supera 1.2*referencia, aplica ganancia proporcional y una
    transición de 25 ms. No mide clipping ni recupera picos recortados.
    """

    y = np.asarray(y, dtype=np.float64)

    if len(y) == 0 or np.max(np.abs(y)) <= 0:
        return y


    y_pre = y / np.max(np.abs(y))
    mascara_tos = detectar_mascara_tos_en_muestras(fs, y_pre, config)

    if not np.any(mascara_tos):
        return y

    mascara_no_tos = ~mascara_tos

    if np.any(mascara_no_tos):
        ref_soplido = np.percentile(np.abs(y[mascara_no_tos]), 99.0)
    else:
        ref_soplido = np.percentile(np.abs(y), 95.0)

    if ref_soplido <= 0:
        return y

    pico_tos = np.max(np.abs(y[mascara_tos]))


    limite_tos = 1.20 * ref_soplido

    if pico_tos <= limite_tos:
        return y

    gain_tos = limite_tos / (pico_tos + 1e-12)

    transicion = max(1, int(0.025 * fs))
    alpha = np.convolve(
        mascara_tos.astype(float),
        np.ones(transicion) / transicion,
        mode="same"
    )
    alpha = np.clip(alpha, 0.0, 1.0)

    y_tos_bajada = y * gain_tos
    y_bajada = alpha * y_tos_bajada + (1.0 - alpha) * y

    print("\n" + "="*80)
    print("        CONTROL DE TOS SATURADA ANTES DE NORMALIZAR")
    print("="*80)
    print("  La tos no será la referencia principal de normalización.")
    print(f"  Muestras de tos:             {np.sum(mascara_tos)}")
    print(f"  Ref. soplido percentil 99:   {ref_soplido:.6f}")
    print(f"  Pico tos original:           {pico_tos:.6f}")
    print(f"  Límite tos aplicado:         {limite_tos:.6f}")
    print(f"  Gain aplicado a tos:         {gain_tos:.3f}")
    print("="*80 + "\n")

    return y_bajada


def recortar_espacio_blanco_inicio(y, fs,
                                   umbral_relativo=0.015,
                                   ventana_rms_s=0.020,
                                   muestras_consecutivas_s=0.050,
                                   margen_pre_s=0.000):
    """Recorta el inicio anterior a una actividad RMS persistente.

    rho[n]=RMS_20ms[n]/(max(RMS_20ms)+eps). Busca el primer tramo
    consecutivo de 50 ms con rho>=0.015 y resta el margen previo.
    Devuelve (señal_recortada, muestras_retiradas). El bucle conserva
    sus límites originales: en registros cortos puede evaluar una
    porción menor que la persistencia solicitada.
    """

    y = np.asarray(y, dtype=np.float64)

    if len(y) == 0 or np.max(np.abs(y)) <= 0:
        return y, 0

    y_abs = np.abs(y)

    ventana = max(1, int(ventana_rms_s * fs))
    energia = np.convolve(y_abs ** 2, np.ones(ventana) / ventana, mode="same")
    rms = np.sqrt(np.maximum(energia, 0))

    if np.max(rms) <= 0:
        return y, 0

    rms_norm = rms / (np.max(rms) + 1e-12)

    n_consec = max(1, int(muestras_consecutivas_s * fs))
    idx_inicio = None

    for i in range(0, max(1, len(rms_norm) - n_consec)):
        if np.all(rms_norm[i:i + n_consec] >= umbral_relativo):
            idx_inicio = i
            break

    if idx_inicio is None:
        return y, 0

    margen_pre = max(0, int(margen_pre_s * fs))
    idx_corte = max(0, idx_inicio - margen_pre)

    if idx_corte <= 0:
        return y, 0

    y_recortada = y[idx_corte:]

    print("\n" + "="*80)
    print("        RECORTE DE ESPACIO BLANCO INICIAL")
    print("="*80)
    print(f"  Inicio útil detectado:       {idx_inicio / fs:.4f} s")
    print(f"  Margen previo conservado:    {margen_pre_s:.4f} s (sin espacio blanco)")
    print(f"  Recorte aplicado:            {idx_corte / fs:.4f} s")
    print("  Nuevo inicio de gráfica:     t = 0 s")
    print("="*80 + "\n")

    return y_recortada, idx_corte


def cargar_y_procesar(ruta_audio, nombre_dispositivo, output_folder):
    """Lee el WAV, prepara x[n], aplica denoising y exporta a 4 kHz.

    Las grabaciones propias son mono (C=1). Para C>1, el análisis usa
    x_mono[n]=sum_c(x_c[n])/C; la vista original usa el primer canal.
    Normaliza, remuestrea, recorta el inicio y, en estetoscopio, limita
    la dominancia de una tos detectada antes de normalizar de nuevo.
    La máscara y los PEF temporales proceden de denoising_entrada,
    anterior al filtrado espectral. El WAV procesado se guarda como
    float32. Cada archivo conserva su análisis independiente.
    """
    fs_raw, y_raw = wav.read(ruta_audio)

    print(f"\n{nombre_dispositivo}")
    print(f"Archivo original = {os.path.basename(ruta_audio)}")

    y_raw_float = convertir_a_float(y_raw)
    # Para WAV multicanal, vista original=canal 1; análisis=promedio de canales.
    y_raw_graficas = normalizar_audio(y_raw_float[:, 0] if y_raw_float.ndim == 2 else y_raw_float)

    y_proc = np.mean(y_raw_float, axis=1) if y_raw_float.ndim == 2 else y_raw_float.copy()
    if np.max(np.abs(y_proc)) > 0: y_proc = y_proc / np.max(np.abs(y_proc))

    y_proc, fs_proc = remuestrear_audio(y_proc, fs_raw, target_fs)


    y_proc, muestras_recorte_inicio = recortar_espacio_blanco_inicio(
        y_proc,
        fs_proc,
        umbral_relativo=0.015,
        ventana_rms_s=0.020,
        muestras_consecutivas_s=0.050,
        margen_pre_s=0.000
    )


    if np.max(np.abs(y_proc)) > 0:
        y_proc = y_proc / np.max(np.abs(y_proc))


    nombre_config = "ESTETOSCOPIO" if str(nombre_dispositivo).startswith("ESTETOSCOPIO") else nombre_dispositivo
    config_disp = CONFIG_FUENTES.get(nombre_config, CONFIG_FUENTES["MOVIL"])


    if nombre_config == "ESTETOSCOPIO":
        y_proc = controlar_tos_saturada_antes_normalizar(fs_proc, y_proc, config_disp)

    if np.max(np.abs(y_proc)) > 0: y_proc = y_proc / np.max(np.abs(y_proc))

    y_denoising_entrada = y_proc.copy()


    if muestras_recorte_inicio > 0:
        muestras_recorte_raw = int(round(muestras_recorte_inicio * fs_raw / fs_proc))
        if 0 < muestras_recorte_raw < len(y_raw_graficas):
            y_raw_graficas = y_raw_graficas[muestras_recorte_raw:]

    y_den = aplicar_denoising(
        fs_proc, y_denoising_entrada, N_denoising, S_denoising, window_denoising, config_disp
    )

    if preservar_tos:
        mascara_tos = detectar_mascara_tos_en_muestras(fs_proc, y_denoising_entrada, config_disp)
        eventos_picos = getattr(detectar_mascara_tos_en_muestras, "ultimos_eventos", [])
    else:
        mascara_tos = np.zeros(len(y_denoising_entrada), dtype=bool)
        eventos_picos = []

    nombre_salida = os.path.splitext(os.path.basename(ruta_audio))[0] + "_denoised_4000Hz.wav"
    ruta_salida = os.path.join(output_folder, nombre_salida)
    wav.write(ruta_salida, fs_proc, y_den.astype(np.float32))

    return {
        "nombre_dispositivo": nombre_dispositivo, "nombre_archivo": os.path.basename(ruta_audio),
        "ruta_audio": ruta_audio, "ruta_salida": ruta_salida,
        "fs_raw": fs_raw, "fs": fs_proc,
        "original_raw_escucha": y_raw_float, "original_raw_graficas": y_raw_graficas,
        "denoising_entrada": y_denoising_entrada,
        "denoised": y_den,
        "muestras_recorte_inicio": muestras_recorte_inicio,
        "mascara_tos": mascara_tos,
        "eventos_picos": eventos_picos
    }


def limitar_tos_sin_aplanar_curva(envolvente_norm, mascara_tos, fs,
                                  limite_visual=1.0,
                                  transicion_s=0.025):
    """Ajusta la escala visual de la envolvente dentro de la máscara de tos.

    g=limite_visual/max(env[tos]); una transición temporal mezcla
    g*env y env. El clip final limita la representación a [0, limite].
    No modifica los eventos detectados ni el audio exportado.
    """

    envolvente_norm = np.asarray(envolvente_norm, dtype=np.float64).copy()

    if mascara_tos is None:
        return np.clip(envolvente_norm, 0.0, limite_visual)

    mascara_tos = np.asarray(mascara_tos, dtype=bool)

    if len(mascara_tos) != len(envolvente_norm) or not np.any(mascara_tos):
        return np.clip(envolvente_norm, 0.0, limite_visual)

    max_tos = float(np.max(envolvente_norm[mascara_tos]))

    if max_tos <= limite_visual:
        return np.clip(envolvente_norm, 0.0, limite_visual)


    gain_tos = limite_visual / (max_tos + 1e-12)

    tos_escalada = envolvente_norm * gain_tos

    transicion = max(1, int(transicion_s * fs))
    alpha = np.convolve(
        mascara_tos.astype(float),
        np.ones(transicion) / transicion,
        mode="same"
    )
    alpha = np.clip(alpha, 0.0, 1.0)

    envolvente_out = alpha * tos_escalada + (1.0 - alpha) * envolvente_norm


    envolvente_out = np.clip(envolvente_out, 0.0, limite_visual)

    return envolvente_out


def calcular_envolvente_temporal(y, fs, mascara_tos=None, normalizar_entrada=True, normalizar_salida=True):
    """Obtiene e(t_n) mediante rectificación y medias móviles centradas.

    Usa 250 ms en general y 50 ms en la máscara de tos; mezcla ambas
    curvas mediante la máscara suavizada. Normaliza por el máximo
    fuera de tos cuando esa región es útil y limita su escala visual.
    No utiliza la transformada de Hilbert. La media de abs(x) no es
    energía física ni RMS; los indicadores de decisión se calculan
    sobre las curvas específicas de cada detector.
    """
    if normalizar_entrada: y = normalizar_audio(y)
    else: y = np.asarray(y, dtype=np.float64)

    envolvente = np.abs(y)

    vent_resto = max(int(tiempo_suavizado_env * fs), 1)
    env_resto = np.convolve(envolvente, np.ones(vent_resto) / vent_resto, mode="same")

    if mascara_tos is not None and np.any(mascara_tos):
        vent_tos = max(int(0.05 * fs), 1)
        env_tos = np.convolve(envolvente, np.ones(vent_tos) / vent_tos, mode="same")
        mascara_suave = np.convolve(mascara_tos.astype(float), np.ones(vent_tos) / vent_tos, mode="same")
        envolvente_suave = (env_tos * mascara_suave) + (env_resto * (1.0 - mascara_suave))
    else:
        envolvente_suave = env_resto

    if normalizar_salida and np.max(envolvente_suave) > 0:


        if mascara_tos is not None and len(mascara_tos) == len(envolvente_suave) and np.any(mascara_tos):
            mascara_no_tos = ~np.asarray(mascara_tos, dtype=bool)

            if np.any(mascara_no_tos):
                ref_curva = np.max(envolvente_suave[mascara_no_tos])
            else:
                ref_curva = np.max(envolvente_suave)
        else:
            ref_curva = np.max(envolvente_suave)

        if ref_curva <= 0:
            ref_curva = np.max(envolvente_suave)

        envolvente_suave = envolvente_suave / (ref_curva + 1e-12)


        envolvente_suave = limitar_tos_sin_aplanar_curva(
            envolvente_suave,
            mascara_tos,
            fs,
            limite_visual=1.0,
            transicion_s=0.025
        )

    return envolvente_suave


def calcular_envolvente_logmel_denoised(logmel_den, t_logmel, fs_audio, mascara_tos=None, normalizar_salida=True):
    """Promedia por bandas la potencia relativa reconstruida desde Log-Mel.

    E[m,b]=10**(L_rel[m,b]/10); la curva inicial es mean_b(E[m,b]).
    Aplica medias móviles de 250/50 ms según la máscara de tos y
    normalización relativa. Devuelve tiempos y e_LM(t); no representa
    energía absoluta y su amplitud no es equivalente a la de e(t).
    """

    logmel_den = np.asarray(logmel_den, dtype=np.float64)
    t_logmel = np.asarray(t_logmel, dtype=np.float64)

    if logmel_den.size == 0 or len(t_logmel) == 0:
        return t_logmel, np.zeros(len(t_logmel), dtype=np.float64)


    if logmel_den.ndim == 1:
        logmel_den = logmel_den.reshape(1, -1)

    n_frames = min(logmel_den.shape[1], len(t_logmel))
    logmel_den = logmel_den[:, :n_frames]
    t_logmel = t_logmel[:n_frames]

    if n_frames == 0:
        return t_logmel, np.zeros(0, dtype=np.float64)


    energia_lineal = 10.0 ** (logmel_den / 10.0)


    envolvente = np.mean(energia_lineal, axis=0)

    if len(t_logmel) > 1:
        dt = float(np.median(np.diff(t_logmel)))
        if dt <= 0:
            dt = 1.0 / max(float(fs_audio), 1.0)
    else:
        dt = 1.0 / max(float(fs_audio), 1.0)

    fs_env = 1.0 / dt


    vent_resto = max(int(tiempo_suavizado_env * fs_env), 1)
    env_resto = np.convolve(envolvente, np.ones(vent_resto) / vent_resto, mode="same")

    mascara_tos_logmel = None

    if mascara_tos is not None and len(mascara_tos) > 0:
        mascara_tos = np.asarray(mascara_tos, dtype=bool)


        idx_tos = np.round(t_logmel * fs_audio).astype(int)
        idx_tos = np.clip(idx_tos, 0, len(mascara_tos) - 1)
        mascara_tos_logmel = mascara_tos[idx_tos]

    if mascara_tos_logmel is not None and np.any(mascara_tos_logmel):

        vent_tos = max(int(0.05 * fs_env), 1)
        env_tos = np.convolve(envolvente, np.ones(vent_tos) / vent_tos, mode="same")
        mascara_suave = np.convolve(
            mascara_tos_logmel.astype(float),
            np.ones(vent_tos) / vent_tos,
            mode="same"
        )
        mascara_suave = np.clip(mascara_suave, 0.0, 1.0)
        envolvente_suave = (env_tos * mascara_suave) + (env_resto * (1.0 - mascara_suave))
    else:
        envolvente_suave = env_resto

    if normalizar_salida and np.max(envolvente_suave) > 0:

        if (
            mascara_tos_logmel is not None
            and len(mascara_tos_logmel) == len(envolvente_suave)
            and np.any(mascara_tos_logmel)
        ):
            mascara_no_tos = ~np.asarray(mascara_tos_logmel, dtype=bool)

            if np.any(mascara_no_tos):
                ref_curva = np.max(envolvente_suave[mascara_no_tos])
            else:
                ref_curva = np.max(envolvente_suave)
        else:
            ref_curva = np.max(envolvente_suave)

        if ref_curva <= 0:
            ref_curva = np.max(envolvente_suave)

        envolvente_suave = envolvente_suave / (ref_curva + 1e-12)


        if mascara_tos_logmel is not None:
            envolvente_suave = limitar_tos_sin_aplanar_curva(
                envolvente_suave,
                mascara_tos_logmel,
                fs_env,
                limite_visual=1.0,
                transicion_s=0.025
            )
        else:
            envolvente_suave = np.clip(envolvente_suave, 0.0, 1.0)

    return t_logmel, envolvente_suave


def calcular_stft_power(y, fs, normalizar_entrada=True, ref_power_max=None):
    """Calcula el espectrograma Hann y P=abs(X)**2.

    L=10*log10(P/(P_ref+eps)+eps), con máximo propio o referencia
    compartida. Devuelve frecuencias, tiempos, potencia y dB relativos.
    """
    if normalizar_entrada:
        y = normalizar_audio(y)
    else:
        y = np.asarray(y, dtype=np.float64)

    nperseg = min(N_stft_graficas, len(y))
    nperseg = max(8, nperseg)
    noverlap = int(nperseg * S_stft_graficas)
    noverlap = min(noverlap, nperseg - 1)

    f, t, Sxx = signal.spectrogram(
        y, fs=fs, window="hann", nperseg=nperseg,
        noverlap=noverlap, scaling="spectrum", mode="magnitude"
    )

    power = Sxx ** 2

    if ref_power_max is None:
        ref_power_max = np.max(power)

    power_db = 10 * np.log10((power / (ref_power_max + 1e-12)) + 1e-12)

    return f, t, power, power_db


def calcular_score_terminacion_precoz(
    y,
    fs,
    mascara_tos: np.ndarray | None = None,
    umbral_bajo: float = 0.15,
    tiempo_optimo_caida: float = 0.5,
    tiempo_max_caida: float = 2,
    umbral_score: float = 0.85
) -> dict:
    """Calcula indicadores auxiliares desde el máximo hasta el nivel bajo.

    Puntúa el tiempo de caída de la envolvente y calcula la varianza
    entre ocho bandas de potencia, excluyendo tramas de tos. La salida
    score_terminacion no activa directamente la clasificación final;
    esta se resuelve en calcular_conclusion_final_prueba.
    """
    y = normalizar_audio(y)

    if len(y) < 32 or np.max(np.abs(y)) <= 0:
        print("\n" + "="*50)
        print("     ANÁLISIS TERMINACIÓN PRECOZ (ERROR)     ")
        print("="*50)
        print("  Señal vacía o demasiado corta.")
        print("="*50 + "\n")
        return {
            "score": 0.0, "tiempo": 0.0, "caida_total": 0.0, "homogeneidad": 0.0, "correlacion": 0.0,
            "terminacion_precoz": False, "mensaje": "No se trata de una posible terminación precoz",
            "varianza_ventana": 0.0
        }


    envolvente = calcular_envolvente_temporal(y, fs, mascara_tos=None, normalizar_entrada=False, normalizar_salida=True)
    n = len(envolvente)
    tiempos = np.arange(n) / fs


    valido = np.ones(n, dtype=bool)
    if mascara_tos is not None and np.any(mascara_tos):
        mascara_tos = np.asarray(mascara_tos, dtype=bool)
        if len(mascara_tos) == n:
            valido = ~mascara_tos
        else:
            x_old = np.linspace(0, 1, len(mascara_tos))
            x_new = np.linspace(0, 1, n)
            valido = ~(np.interp(x_new, x_old, mascara_tos.astype(float)) > 0.5)

    if not np.any(valido):
        print("\n" + "="*50)
        print("     ANÁLISIS TERMINACIÓN PRECOZ (ERROR)     ")
        print("="*50)
        print("  La señal solo contiene tos o no hay zonas válidas.")
        print("="*50 + "\n")
        return {
            "score": 0.0, "tiempo": 0.0, "caida_total": 0.0, "homogeneidad": 0.0, "correlacion": 0.0,
            "terminacion_precoz": False, "mensaje": "No se trata de una posible terminación precoz",
            "varianza_ventana": 0.0
        }


    idx_validos = np.where(valido)[0]
    idx_pef = idx_validos[np.argmax(envolvente[idx_validos])]
    valor_pef = envolvente[idx_pef]

    if valor_pef <= 0:
        return {
            "score": 0.0, "tiempo": float(tiempos[idx_pef]), "caida_total": 0.0, "homogeneidad": 0.0, "correlacion": 0.0,
            "terminacion_precoz": False, "mensaje": "No se trata de una posible terminación precoz",
            "varianza_ventana": 0.0
        }


    segmento_post_pef = envolvente[idx_pef:]
    indices_caida = np.where(segmento_post_pef <= umbral_bajo)[0]
    if len(indices_caida) > 0:
        idx_bajo = idx_pef + indices_caida[0]
    else:
        idx_bajo = n - 1

    valor_bajo = envolvente[idx_bajo]
    caida_total = float(np.clip(valor_pef - valor_bajo, 0.0, 1.0))


    muestras_caida = int(idx_bajo - idx_pef + 1)
    tiempo_caida_total = muestras_caida / float(fs)


    if tiempo_caida_total <= tiempo_optimo_caida:
        score_final = 1.0
    elif tiempo_caida_total >= tiempo_max_caida:
        score_final = 0.0
    else:
        rango = tiempo_max_caida - tiempo_optimo_caida
        score_final = 1.0 - ((tiempo_caida_total - tiempo_optimo_caida) / rango)

    score_final = float(np.clip(score_final, 0.0, 1.0))

    if score_final >= umbral_score:
        mensaje_texto = "se trata de una posible terminación precoz"
        terminacion = True
    else:
        mensaje_texto = "no se trata de una posible terminación precoz"
        terminacion = False


    tiempo_pef = tiempos[idx_pef]
    tiempo_bajo = tiempos[idx_bajo]


    f, t_stft, power, _ = calcular_stft_power(y, fs, normalizar_entrada=False)


    dentro_ventana = (t_stft >= tiempo_pef) & (t_stft <= tiempo_bajo)


    idx_valido_stft = np.clip((t_stft * fs).astype(int), 0, n - 1)
    frames_sin_tos = valido[idx_valido_stft]


    frames_finales = dentro_ventana & frames_sin_tos

    if np.any(frames_finales):
        power_ventana = power[:, frames_finales]


        fmax = min(2000, fs / 2)
        n_bandas = 8
        bordes = np.linspace(0, fmax, n_bandas + 1)

        energia_bandas = []
        for i in range(n_bandas):
            idx_f = (f >= bordes[i]) & (f < bordes[i + 1])
            if np.any(idx_f):
                energia_bandas.append(np.sum(power_ventana[idx_f, :], axis=0))
            else:
                energia_bandas.append(np.zeros(power_ventana.shape[1]))

        energia_bandas = np.array(energia_bandas)


        varianzas_por_frame = np.var(energia_bandas, axis=0)


        varianza_bandas_final = float(np.mean(varianzas_por_frame))
    else:
        varianza_bandas_final = 0.0


    print("\n" + "="*60)
    print("                ANÁLISIS DE TRANSMISIÓN / PRECOCIDAD        ")
    print("="*60)
    print(f"  Veredicto:             {mensaje_texto.upper()}")
    print(f"  Score de Precocidad:   {score_final:.3f} / 1.000 (Umbral: {umbral_score:.2f})")
    print(f"  Muestras de caída:     {muestras_caida} muestras (PEF -> {int(umbral_bajo*100)}%)")
    print(f"  Tiempo de caída:       {tiempo_caida_total:.3f} segundos")
    print(f"  Rango Evaluado STFT:   {tiempo_pef:.3f} s  -->  {tiempo_bajo:.3f} s")
    print(f"  Varianza Espectral:    {varianza_bandas_final:.6f} (Buen estado: casi 0)")
    print("="*60 + "\n")

    return {
        "score": score_final,
        "tiempo": float(tiempos[idx_bajo]),
        "caida_total": caida_total,
        "homogeneidad": score_final,
        "correlacion": tiempo_caida_total,
        "terminacion_precoz": terminacion,
        "muestras_caida": muestras_caida,
        "tiempo_caida": tiempo_caida_total,
        "mensaje": mensaje_texto,
        "varianza_ventana": varianza_bandas_final
    }
def calcular_envolvente_stft_en_tiempo(y, fs, normalizar_entrada=True, normalizar_salida=True):
    """Obtiene sqrt(sum(P, axis=0)) en los tiempos de trama de la STFT, con normalización opcional."""
    f, t, power, power_db = calcular_stft_power(y, fs, normalizar_entrada=normalizar_entrada)
    envolvente_stft = np.sqrt(np.sum(power, axis=0))
    if normalizar_salida and np.max(envolvente_stft) > 0: envolvente_stft = envolvente_stft / np.max(envolvente_stft)
    return t, envolvente_stft


# Mel(f)=2595*log10(1+f/700), con f en Hz; la función siguiente es su inversa.
def hz_a_mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
def mel_a_hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

def crear_banco_mel(fs, nfft, n_mels):
    """Construye B filtros triangulares entre 0 y fs/2.

    Sus vértices son equidistantes en Mel, no en Hz; cada fila del
    banco pesa las componentes de la FFT de una banda.
    """
    puntos_hz = mel_a_hz(np.linspace(hz_a_mel(0), hz_a_mel(fs / 2), n_mels + 2))
    freqs_fft = np.fft.rfftfreq(nfft, 1 / fs)
    banco = np.zeros((n_mels, len(freqs_fft)))
    for m in range(1, n_mels + 1):
        f_izq, f_cen, f_der = puntos_hz[m - 1], puntos_hz[m], puntos_hz[m + 1]
        for k, f in enumerate(freqs_fft):
            if f_izq < f <= f_cen: banco[m - 1, k] = (f - f_izq) / (f_cen - f_izq + 1e-12)
            elif f_cen < f < f_der: banco[m - 1, k] = (f_der - f) / (f_der - f_cen + 1e-12)
    return banco

def frecuencias_centro_bandas_mel(fs, n_mels):
    """Devuelve en Hz los B centros de los filtros Mel, excluidos sus extremos."""
    puntos_hz = mel_a_hz(np.linspace(hz_a_mel(0), hz_a_mel(fs / 2), n_mels + 2))
    return puntos_hz[1:-1]


def aplicar_eje_frecuencias_mel(ax, freqs_centro, n_mels, num_ticks=6):
    """Rotula los índices de banda con sus frecuencias centrales en Hz."""
    freqs_centro = np.asarray(freqs_centro, dtype=np.float64)

    if len(freqs_centro) == 0:
        return

    num_ticks = min(num_ticks, len(freqs_centro))
    idx_ticks = np.linspace(0, len(freqs_centro) - 1, num_ticks).astype(int)


    posiciones = idx_ticks + 0.5
    etiquetas = [f"{int(round(freqs_centro[i]))}" for i in idx_ticks]

    ax.set_yticks(posiciones)
    ax.set_yticklabels(etiquetas, fontsize=7)
    ax.set_ylabel("Hz", fontsize=8)

def calcular_mel_logmel(y, fs, ref_mel_max=None, normalizar_entrada=True):
    """Calcula M[m,b] y L_rel[m,b] sobre ventanas Hann del audio.

    M=H@abs(rFFT(x*w))**2. Normaliza con una referencia propia o
    compartida: M_norm=clip(M/(M_ref+eps),0,1).
    L_rel=clip(10*log10(M_norm+eps),-80,0).
    La última ventana se desplaza para terminar en la última muestra;
    no se añade una trama temporal nula. La FFT puede completar ceros
    hasta nfft, que es distinto de añadir audio al final del registro.
    Devuelve centros y bordes temporales para dibujar con pcolormesh.
    """

    if normalizar_entrada:
        y = normalizar_audio(y)
    else:
        y = np.asarray(y, dtype=np.float64)

    y = np.asarray(y, dtype=np.float64)

    if len(y) == 0:
        t = np.array([], dtype=np.float64)
        t_edges = np.array([0.0], dtype=np.float64)
        mel_norm = np.zeros((n_mels, 0), dtype=np.float64)
        logmel = np.zeros((n_mels, 0), dtype=np.float64)
        if ref_mel_max is None:
            ref_mel_max = 0.0
        return t, mel_norm, logmel, ref_mel_max, t_edges

    nperseg = max(8, min(N_stft_graficas, len(y)))
    noverlap = min(int(nperseg * S_stft_graficas), nperseg - 1)
    hop = max(1, nperseg - noverlap)
    nfft = next_fast_len(nperseg)

    banco_mel = crear_banco_mel(fs, nfft, n_mels)

    frames_power = []
    tiempos_centro = []
    inicios_frames = []
    finales_frames = []

    inicio = 0

    while inicio < len(y):
        fin = inicio + nperseg


        if fin > len(y):
            fin_real = len(y)
            inicio_real = max(0, fin_real - min(nperseg, len(y)))
        else:
            inicio_real = inicio
            fin_real = fin

        segmento = y[inicio_real:fin_real]


        if len(segmento) < 2:
            break


        while len(inicios_frames) > 0 and inicio_real <= inicios_frames[-1]:
            frames_power.pop()
            tiempos_centro.pop()
            inicios_frames.pop()
            finales_frames.pop()


        ventana = signal.windows.hann(len(segmento), sym=False)
        segmento_win = segmento * ventana


        espectro = np.fft.rfft(segmento_win, n=nfft)
        power = np.abs(espectro) ** 2

        frames_power.append(power)
        tiempos_centro.append((inicio_real + len(segmento) / 2.0) / fs)
        inicios_frames.append(inicio_real)
        finales_frames.append(fin_real)

        if fin_real >= len(y):
            break

        inicio += hop

    if len(frames_power) == 0:
        t = np.array([], dtype=np.float64)
        t_edges = np.array([0.0], dtype=np.float64)
        mel_norm = np.zeros((n_mels, 0), dtype=np.float64)
        logmel = np.zeros((n_mels, 0), dtype=np.float64)
        if ref_mel_max is None:
            ref_mel_max = 0.0
        return t, mel_norm, logmel, ref_mel_max, t_edges

    power_frames = np.array(frames_power).T
    t = np.array(tiempos_centro, dtype=np.float64)


    t_edges = np.zeros(len(t) + 1, dtype=np.float64)

    if len(t) == 1:
        t_edges[0] = 0.0
        t_edges[1] = len(y) / fs
    else:
        t_edges[0] = 0.0
        t_edges[1:-1] = (t[:-1] + t[1:]) / 2.0
        t_edges[-1] = len(y) / fs


    t_edges = np.maximum.accumulate(t_edges)
    if len(t_edges) > 1 and t_edges[-1] <= t_edges[-2]:
        t_edges[-1] = t_edges[-2] + (1.0 / fs)

    mel = np.dot(banco_mel, power_frames)

    if ref_mel_max is None:
        ref_mel_max = np.max(mel)

    if ref_mel_max > 0:
        mel_norm = mel / (ref_mel_max + 1e-12)
    else:
        mel_norm = mel


    mel_norm = np.clip(mel_norm, 0, 1)

    logmel = np.clip(10 * np.log10(mel_norm + 1e-12), -80, 0)

    return t, mel_norm, logmel, ref_mel_max, t_edges


def calcular_pendiente_logmel_antes_energia_nula(
    y_den,
    fs,
    duracion_ventana=0.40,
    umbral_energia_nula=0.08,
    min_frames_nulos=2
):
    """Caracteriza un corte candidato en las primeras 18 bandas Mel.

    El nombre y_den es histórico: preparar_graficas_fuente llama a
    esta función con el audio original. Usa Hann de 32 muestras,
    nfft=128 y salto aproximado de 1 ms. Compara tres tramas antes
    y después: D_dB=10*log10((E_pre+eps)/(E_post+eps)) y
    D_lin=E_pre-E_post. Devuelve también colapso de bandas, dispersión
    temporal y pendiente global firmada en dB/s; no toma su módulo.
    """


    y_den = normalizar_audio(y_den)

    if len(y_den) < 32 or np.max(np.abs(y_den)) <= 0:
        print("\n" + "="*95)
        print("        ANALISIS DE CORTE BRUSCO EN BANDAS BAJAS LOG-MEL ORIGINAL")
        print("="*95)
        print("  Señal vacía o demasiado corta.")
        print("="*95 + "\n")

        return {
            "pendientes_bandas": None,
            "pendiente_media": 0.0,
            "dispersion_pendientes": 0.0,
            "correlacion_banda_pendiente": 0.0,
            "spread_tiempos_caida": 0.0,
            "tiempo_colapso_bandas": 0.0,
            "pendiente_global_bajas": 0.0,
            "tiempo_inicio_ventana": 0.0,
            "tiempo_fin_ventana": 0.0,
            "tiempo_energia_nula": 0.0,
            "tiempos_apagado_bandas": None,
            "bandas_activas_por_frame": None,
            "mensaje": "Señal vacía o demasiado corta"
        }


    nperseg_hr = min(32, len(y_den))
    nperseg_hr = max(16, nperseg_hr)

    hop_hr = max(1, int(0.001 * fs))
    hop_hr = min(hop_hr, nperseg_hr - 1)

    noverlap_hr = nperseg_hr - hop_hr
    nfft_hr = next_fast_len(max(128, nperseg_hr))

    f_hr, t_mel, Zxx_hr = signal.stft(
        y_den,
        fs=fs,
        window="hann",
        nperseg=nperseg_hr,
        noverlap=noverlap_hr,
        nfft=nfft_hr,
        boundary=None,
        padded=False
    )

    if Zxx_hr.size == 0 or len(t_mel) < 5:
        print("\n" + "="*95)
        print("        ANALISIS DE CORTE BRUSCO EN BANDAS BAJAS LOG-MEL ORIGINAL")
        print("="*95)
        print("  No hay frames suficientes para análisis de alta resolución.")
        print("="*95 + "\n")

        return {
            "pendientes_bandas": None,
            "pendiente_media": 0.0,
            "dispersion_pendientes": 0.0,
            "correlacion_banda_pendiente": 0.0,
            "spread_tiempos_caida": 0.0,
            "tiempo_colapso_bandas": 0.0,
            "pendiente_global_bajas": 0.0,
            "tiempo_inicio_ventana": 0.0,
            "tiempo_fin_ventana": 0.0,
            "tiempo_energia_nula": 0.0,
            "tiempos_apagado_bandas": None,
            "bandas_activas_por_frame": None,
            "mensaje": "No hay frames suficientes"
        }

    mel_hr = np.dot(
        crear_banco_mel(fs, nfft_hr, n_mels),
        np.abs(Zxx_hr) ** 2
    )

    if np.max(mel_hr) > 0:
        mel_hr_norm_global = mel_hr / np.max(mel_hr)
    else:
        mel_hr_norm_global = mel_hr

    logmel_hr = np.clip(10 * np.log10(mel_hr_norm_global + 1e-12), -80, 0)


    num_bandas_totales = mel_hr.shape[0]
    num_bandas_bajas = max(4, int(num_bandas_totales * 0.45))

    mel_bajas = mel_hr[:num_bandas_bajas, :]
    logmel_bajas = logmel_hr[:num_bandas_bajas, :]


    energia_bajas = np.sum(mel_bajas, axis=0)

    if np.max(energia_bajas) > 0:
        energia_bajas_norm = energia_bajas / np.max(energia_bajas)
    else:
        energia_bajas_norm = np.zeros_like(energia_bajas)

    if len(energia_bajas_norm) < 5:
        return {
            "pendientes_bandas": None,
            "pendiente_media": 0.0,
            "dispersion_pendientes": 0.0,
            "correlacion_banda_pendiente": 0.0,
            "spread_tiempos_caida": 0.0,
            "tiempo_colapso_bandas": 0.0,
            "pendiente_global_bajas": 0.0,
            "tiempo_inicio_ventana": 0.0,
            "tiempo_fin_ventana": 0.0,
            "tiempo_energia_nula": 0.0,
            "tiempos_apagado_bandas": None,
            "bandas_activas_por_frame": None,
            "mensaje": "Muy pocos frames Mel"
        }


    idx_pef_bajas = int(np.argmax(energia_bajas_norm))


    pre_frames = 3
    post_frames = 3

    eps = 1e-12
    caidas_db = []
    caidas_lineales = []
    pre_vals = []
    post_vals = []
    indices_candidatos = []

    for i in range(idx_pef_bajas + pre_frames, len(energia_bajas_norm) - post_frames):

        pre = float(np.mean(energia_bajas_norm[i - pre_frames:i]))
        post = float(np.mean(energia_bajas_norm[i:i + post_frames]))

        caida_db = 10 * np.log10((pre + eps) / (post + eps))
        caida_lineal = pre - post

        caidas_db.append(caida_db)
        caidas_lineales.append(caida_lineal)
        pre_vals.append(pre)
        post_vals.append(post)
        indices_candidatos.append(i)

    if len(indices_candidatos) == 0:
        idx_corte_global = len(t_mel) - 1
        caida_maxima_db = 0.0
        caida_maxima_lineal = 0.0
        energia_pre_corte = 0.0
        energia_post_corte = 0.0
    else:
        caidas_db = np.array(caidas_db)
        caidas_lineales = np.array(caidas_lineales)
        pre_vals = np.array(pre_vals)
        post_vals = np.array(post_vals)
        indices_candidatos = np.array(indices_candidatos)


        validos = pre_vals >= 0.05

        if np.any(validos):
            idx_mejor_local = int(np.argmax(caidas_db[validos]))
            indices_validos = np.where(validos)[0]
            pos = indices_validos[idx_mejor_local]
        else:
            pos = int(np.argmax(caidas_db))

        idx_corte_global = int(indices_candidatos[pos])
        caida_maxima_db = float(caidas_db[pos])
        caida_maxima_lineal = float(caidas_lineales[pos])
        energia_pre_corte = float(pre_vals[pos])
        energia_post_corte = float(post_vals[pos])

    tiempo_corte = float(t_mel[idx_corte_global])


    mascara_baja = energia_bajas_norm <= umbral_energia_nula

    idx_apagado_global = None

    for i in range(idx_pef_bajas + 1, len(mascara_baja) - min_frames_nulos + 1):
        if np.all(mascara_baja[i:i + min_frames_nulos]):
            idx_apagado_global = i
            break

    if idx_apagado_global is None:
        idx_apagado_global = min(len(t_mel) - 1, idx_corte_global + post_frames)
        aviso = "No se encontró apagado bajo claro; se usa el entorno del corte máximo"
    else:
        aviso = "Apagado de bandas bajas detectado correctamente"

    tiempo_apagado_global = float(t_mel[idx_apagado_global])


    tiempo_fin = min(float(t_mel[-1]), tiempo_corte + 0.030)
    tiempo_inicio = max(0, tiempo_fin - duracion_ventana)

    frames_analisis = (t_mel >= tiempo_inicio) & (t_mel <= tiempo_fin)

    if np.sum(frames_analisis) < 5:
        print("\n" + "="*95)
        print("        ANALISIS DE CORTE BRUSCO EN BANDAS BAJAS LOG-MEL ORIGINAL")
        print("="*95)
        print("  Hay muy pocos frames en la ventana de análisis.")
        print("="*95 + "\n")

        return {
            "pendientes_bandas": None,
            "pendiente_media": 0.0,
            "dispersion_pendientes": 0.0,
            "correlacion_banda_pendiente": 0.0,
            "spread_tiempos_caida": 0.0,
            "tiempo_colapso_bandas": 0.0,
            "pendiente_global_bajas": 0.0,
            "tiempo_inicio_ventana": tiempo_inicio,
            "tiempo_fin_ventana": tiempo_fin,
            "tiempo_energia_nula": tiempo_apagado_global,
            "tiempos_apagado_bandas": None,
            "bandas_activas_por_frame": None,
            "mensaje": "Muy pocos frames en la ventana de análisis"
        }

    t_final = t_mel[frames_analisis]
    t_rel = t_final - t_final[0]

    mel_final = mel_bajas[:, frames_analisis]
    logmel_final = logmel_bajas[:, frames_analisis]


    idx_corte_ventana = int(np.argmin(np.abs(t_final - tiempo_corte)))


    mel_final_norm = np.zeros_like(mel_final)

    for banda in range(mel_final.shape[0]):
        max_banda = np.max(mel_final[banda, :])
        if max_banda > 0:
            mel_final_norm[banda, :] = mel_final[banda, :] / max_banda
        else:
            mel_final_norm[banda, :] = 0.0


    umbral_apagado_banda = 0.10

    ini_pre = max(0, idx_corte_ventana - pre_frames)
    fin_pre = max(ini_pre + 1, idx_corte_ventana)

    ini_post = min(mel_final_norm.shape[1] - 1, idx_corte_ventana)
    fin_post = min(mel_final_norm.shape[1], idx_corte_ventana + post_frames)

    energia_banda_pre = np.mean(mel_final_norm[:, ini_pre:fin_pre], axis=1)
    energia_banda_post = np.mean(mel_final_norm[:, ini_post:fin_post], axis=1)

    activas_antes_corte = float(np.mean(energia_banda_pre > umbral_apagado_banda))
    activas_despues_corte = float(np.mean(energia_banda_post > umbral_apagado_banda))
    colapso_activos_frame = float(activas_antes_corte - activas_despues_corte)

    bandas_activas_por_frame = np.sum(mel_final_norm > umbral_apagado_banda, axis=0)
    bandas_activas_norm = bandas_activas_por_frame / float(num_bandas_bajas)


    tiempos_apagado_bandas = []

    for banda in range(mel_final_norm.shape[0]):

        curva = mel_final_norm[banda, :]
        mascara_apagada = curva <= umbral_apagado_banda

        idx_apagado_banda = None

        for i in range(0, len(mascara_apagada) - min_frames_nulos + 1):
            if np.all(mascara_apagada[i:i + min_frames_nulos]):
                idx_apagado_banda = i
                break

        if idx_apagado_banda is None:
            tiempos_apagado_bandas.append(t_rel[-1])
        else:
            tiempos_apagado_bandas.append(t_rel[idx_apagado_banda])

    tiempos_apagado_bandas = np.array(tiempos_apagado_bandas)

    spread_tiempos_caida = float(
        np.max(tiempos_apagado_bandas) - np.min(tiempos_apagado_bandas)
    )


    indices_80 = np.where(bandas_activas_norm <= 0.80)[0]
    indices_20 = np.where(bandas_activas_norm <= 0.20)[0]

    if len(indices_80) > 0 and len(indices_20) > 0:
        tiempo_80 = t_rel[indices_80[0]]
        tiempo_20 = t_rel[indices_20[0]]
        tiempo_colapso_bandas = float(max(0.0, tiempo_20 - tiempo_80))
    else:
        tiempo_colapso_bandas = float(t_rel[-1] - t_rel[0])


    energia_bajas_final = np.sum(mel_final, axis=0)

    if np.max(energia_bajas_final) > 0:
        energia_bajas_final_norm = energia_bajas_final / np.max(energia_bajas_final)
    else:
        energia_bajas_final_norm = np.zeros_like(energia_bajas_final)

    energia_bajas_db = 10 * np.log10(energia_bajas_final_norm + 1e-12)
    energia_bajas_db = np.clip(energia_bajas_db, -80, 0)

    if len(t_rel) >= 2 and np.all(np.isfinite(energia_bajas_db)):
        pendiente_global_bajas, ordenada = np.polyfit(t_rel, energia_bajas_db, 1)
    else:
        pendiente_global_bajas = 0.0

    pendiente_global_bajas = float(pendiente_global_bajas)


    pendientes = []

    for banda in range(logmel_final.shape[0]):

        y_banda = logmel_final[banda, :]

        if np.all(np.isfinite(y_banda)) and len(y_banda) >= 2:
            pendiente, ordenada = np.polyfit(t_rel, y_banda, 1)
        else:
            pendiente = 0.0

        pendientes.append(pendiente)

    pendientes = np.array(pendientes)

    pendiente_media = float(np.mean(pendientes))
    dispersion_pendientes = float(np.std(pendientes))

    indices_bandas = np.arange(len(pendientes))

    if np.std(pendientes) > 0:
        correlacion_banda_pendiente = float(
            np.corrcoef(indices_bandas, pendientes)[0, 1]
        )
    else:
        correlacion_banda_pendiente = 0.0


    if (
        caida_maxima_db >= 10.0
        and caida_maxima_lineal >= 0.20
        and energia_pre_corte >= 0.12
        and energia_post_corte <= 0.35
    ):
        mensaje = "Corte brusco detectado: caída brusca de energía en bandas bajas"

    elif (
        caida_maxima_db >= 6.0
        and caida_maxima_lineal >= 0.12
        and energia_pre_corte >= 0.08
    ):
        mensaje = "Posible corte_brusco: caída rápida en bandas bajas, revisar visualmente"

    elif (
        spread_tiempos_caida > 0.20
        and tiempo_colapso_bandas > 0.15
    ):
        mensaje = "Desaparición progresiva por bandas bajas: efecto escalera"

    elif pendiente_global_bajas < -120:
        mensaje = "Caída rápida global de energía baja, pero sin corte_brusco claro"

    else:
        mensaje = "Final sin corte_brusco claro; evolución baja más gradual o irregular"


    print("\n" + "="*95)
    print("        ANALISIS DE CORTE BRUSCO EN BANDAS BAJAS LOG-MEL ORIGINAL")
    print("="*95)
    print(f"  Estado detección:                    {aviso}")
    print(f"  Resolución temporal usada:            {hop_hr / fs:.4f} s por frame")
    print(f"  Ventana STFT interna:                 {nperseg_hr / fs:.4f} s")
    print(f"  Tiempo apagado global bajas:          {tiempo_apagado_global:.4f} s")
    print(f"  Tiempo corte máximo:                  {tiempo_corte:.4f} s")
    print(f"  Ventana analizada:                    {tiempo_inicio:.4f} s  -->  {tiempo_fin:.4f} s")
    print(f"  Frames analizados:                    {np.sum(frames_analisis)}")
    print(f"  Bandas bajas analizadas:              {num_bandas_bajas}/{num_bandas_totales}")
    print(f"  Energía pre-corte:                    {energia_pre_corte:.4f}")
    print(f"  Energía post-corte:                   {energia_post_corte:.4f}")
    print(f"  Caída máxima en dB:                   {caida_maxima_db:.3f} dB")
    print(f"  Caída máxima lineal:                  {caida_maxima_lineal:.4f}")
    print(f"  Bandas activas antes del corte:       {activas_antes_corte:.4f}")
    print(f"  Bandas activas después del corte:     {activas_despues_corte:.4f}")
    print(f"  Colapso bandas activas:               {colapso_activos_frame:.4f}")
    print(f"  Spread tiempos apagado bandas:        {spread_tiempos_caida:.4f} s")
    print(f"  Tiempo colapso bandas activas 80-20:  {tiempo_colapso_bandas:.4f} s")
    print(f"  Pendiente global bajas:               {pendiente_global_bajas:.3f} dB/s")
    print(f"  Pendiente media por banda:            {pendiente_media:.3f} dB/s")
    print(f"  Dispersión pendientes bandas:         {dispersion_pendientes:.3f}")
    print(f"  Correlación banda-pendiente:          {correlacion_banda_pendiente:.3f}")
    print(f"  Interpretación:                       {mensaje}")
    print("="*95 + "\n")

    return {
        "pendientes_bandas": pendientes,
        "pendiente_media": pendiente_media,
        "dispersion_pendientes": dispersion_pendientes,
        "correlacion_banda_pendiente": correlacion_banda_pendiente,
        "spread_tiempos_caida": spread_tiempos_caida,
        "tiempo_colapso_bandas": tiempo_colapso_bandas,
        "pendiente_global_bajas": pendiente_global_bajas,
        "tiempo_inicio_ventana": tiempo_inicio,
        "tiempo_fin_ventana": tiempo_fin,
        "tiempo_energia_nula": tiempo_apagado_global,
        "tiempos_apagado_bandas": tiempos_apagado_bandas,
        "bandas_activas_por_frame": bandas_activas_por_frame,
        "caida_maxima_frame": caida_maxima_lineal,
        "caida_maxima_db": caida_maxima_db,
        "tiempo_corte": tiempo_corte,
        "colapso_activos_frame": colapso_activos_frame,
        "activas_antes_corte": activas_antes_corte,
        "activas_despues_corte": activas_despues_corte,
        "energia_pre_corte": energia_pre_corte,
        "energia_post_corte": energia_post_corte,
        "mensaje": mensaje
    }


def forzar_inicio_curva_desde_suelo_visual(t, curva,
                                           umbral_inicio=0.08,
                                           muestras_consecutivas=8,
                                           pre_suelo_s=0.08,
                                           rampa_s=0.18):
    """Aplica una rampa inicial a una copia de la curva para representación."""

    curva = np.asarray(curva, dtype=np.float64).copy()
    t = np.asarray(t, dtype=np.float64)

    if len(curva) == 0 or len(t) != len(curva):
        return curva

    if np.max(curva) <= 0:
        return curva

    curva_ref = curva / (np.max(curva) + 1e-12)

    if len(t) > 1:
        dt = float(np.median(np.diff(t)))
        if dt <= 0:
            dt = 1.0
    else:
        dt = 1.0

    n_consec = max(1, int(muestras_consecutivas))
    pre_muestras = max(0, int(pre_suelo_s / dt))
    rampa_muestras = max(1, int(rampa_s / dt))

    idx_inicio = None

    for i in range(0, max(1, len(curva_ref) - n_consec)):
        if np.all(curva_ref[i:i + n_consec] >= umbral_inicio):
            idx_inicio = i
            break

    if idx_inicio is None:
        return curva

    idx_arranque = max(0, idx_inicio - pre_muestras)

    curva[:idx_arranque] = 0.0

    idx_fin_rampa = min(len(curva), idx_arranque + rampa_muestras)

    if idx_fin_rampa > idx_arranque:
        rampa = np.linspace(0.0, 1.0, idx_fin_rampa - idx_arranque)
        curva[idx_arranque:idx_fin_rampa] = curva[idx_arranque:idx_fin_rampa] * rampa

    return curva


def guardar_temp_wav(y, fs, prefijo, normalizar=True):
    """Guarda un WAV temporal para escucha; la normalización es opcional."""
    y = np.asarray(y, dtype=np.float64)
    if normalizar: y = normalizar_audio(y)
    ruta_temp = os.path.join(tempfile.gettempdir(), prefijo + ".wav")
    wav.write(ruta_temp, fs, np.int16(np.clip(y, -1, 1) * 32767))
    return ruta_temp

def reproducir_audio(ruta_wav):
    """Abre el WAV con el reproductor disponible en el sistema operativo."""
    sistema = platform.system()
    try:
        if sistema == "Windows": os.startfile(ruta_wav)
        elif sistema == "Darwin": subprocess.Popen(["open", ruta_wav])
        else: subprocess.Popen(["xdg-open", ruta_wav])
    except Exception as e: print("No se pudo reproducir:", e)

def sincronizar_zoom(fig, grupos_ejes):
    """Comparte los límites horizontales entre los ejes temporales."""
    bloqueado = {"estado": False}
    def crear_callback(ejes_grupo):
        """Crea el manejador que propaga el intervalo temporal seleccionado."""
        def callback(ax):
            """Actualiza los demás ejes del grupo al cambiar los límites del eje origen."""
            if bloqueado["estado"]: return
            bloqueado["estado"] = True
            xlim = ax.get_xlim()
            for otro_ax in ejes_grupo:
                if otro_ax is not ax: otro_ax.set_xlim(xlim)
            fig.canvas.draw_idle()
            bloqueado["estado"] = False
        return callback
    for ejes_grupo in grupos_ejes:
        callback = crear_callback(ejes_grupo)
        for ax in ejes_grupo: ax.callbacks.connect("xlim_changed", callback)


def calcular_promedio_abs_ultimo_cuarto(y, fs, duracion_s=0.25):
    """Compara mean(abs(x)) en dos ventanas finales de duracion_s.

    Para W=0.25 s: anterior=[T-2W,T-W) y final=[T-W,T).
    Calcula además RMS y razones. mean(abs(x)) es amplitud media,
    aunque algunas claves históricas del resultado se llamen energía.
    """

    y = np.asarray(y, dtype=np.float64)

    if len(y) == 0 or fs <= 0:
        return {
            "media_abs_ultimo": 0.0,
            "media_abs_anterior": 0.0,
            "media_abs_total": 0.0,
            "ratio_ultimo_vs_anterior": 0.0,
            "ratio_ultimo_vs_total": 0.0,
            "diferencia_abs_anterior_menos_ultimo": 0.0,
            "rms_ultimo": 0.0,
            "rms_anterior": 0.0,
            "muestras_ultimo": 0,
            "muestras_anterior": 0,
            "duracion_real_ultimo_s": 0.0,
            "duracion_real_anterior_s": 0.0
        }

    n_ventana = int(round(duracion_s * fs))
    n_ventana = max(1, min(n_ventana, len(y)))

    tramo_ultimo = y[-n_ventana:]


    fin_anterior = max(0, len(y) - n_ventana)
    inicio_anterior = max(0, fin_anterior - n_ventana)
    tramo_anterior = y[inicio_anterior:fin_anterior]

    if len(tramo_anterior) == 0:
        tramo_anterior = tramo_ultimo.copy()

    media_abs_ultimo = float(np.mean(np.abs(tramo_ultimo)))
    media_abs_anterior = float(np.mean(np.abs(tramo_anterior)))
    media_abs_total = float(np.mean(np.abs(y)))

    if media_abs_anterior > 0:
        ratio_ultimo_vs_anterior = float(media_abs_ultimo / (media_abs_anterior + 1e-12))
    else:
        ratio_ultimo_vs_anterior = 0.0

    if media_abs_total > 0:
        ratio_ultimo_vs_total = float(media_abs_ultimo / (media_abs_total + 1e-12))
    else:
        ratio_ultimo_vs_total = 0.0

    diferencia = float(media_abs_anterior - media_abs_ultimo)

    rms_ultimo = float(np.sqrt(np.mean(tramo_ultimo ** 2)))
    rms_anterior = float(np.sqrt(np.mean(tramo_anterior ** 2)))

    return {
        "media_abs_ultimo": media_abs_ultimo,
        "media_abs_anterior": media_abs_anterior,
        "media_abs_total": media_abs_total,
        "ratio_ultimo_vs_anterior": ratio_ultimo_vs_anterior,
        "ratio_ultimo_vs_total": ratio_ultimo_vs_total,
        "diferencia_abs_anterior_menos_ultimo": diferencia,
        "rms_ultimo": rms_ultimo,
        "rms_anterior": rms_anterior,
        "muestras_ultimo": int(len(tramo_ultimo)),
        "muestras_anterior": int(len(tramo_anterior)),
        "duracion_real_ultimo_s": float(len(tramo_ultimo) / fs),
        "duracion_real_anterior_s": float(len(tramo_anterior) / fs)
    }


def obtener_tiempo_primer_pef(eventos_picos):
    """Devuelve el tiempo del primer evento PEF acústico, o None si no existe."""
    pefs = [
        ev for ev in eventos_picos
        if isinstance(ev, dict) and ev.get("tipo", "") == "PEF"
    ]

    if len(pefs) == 0:
        return None

    pefs = sorted(pefs, key=lambda ev: float(ev.get("pico", 0.0)))
    return float(pefs[0].get("pico", 0.0))


def calcular_promedio_abs_desde_primer_pef(y, fs, eventos_picos):
    """Resume las muestras desde el primer PEF acústico hasta el final.

    Calcula mean(abs(x)) y sqrt(mean(x**2)), junto con el número de
    muestras y duración analizada. No estima energía acústica en julios.
    """

    y = np.asarray(y, dtype=np.float64)

    if len(y) == 0 or fs <= 0:
        return {
            "media_abs_desde_pef": 0.0,
            "rms_desde_pef": 0.0,
            "idx_primer_pef": None,
            "tiempo_primer_pef_s": None,
            "muestras_analizadas": 0,
            "duracion_desde_pef_s": 0.0,
            "pef_detectado": False
        }

    tiempo_primer_pef = obtener_tiempo_primer_pef(eventos_picos)

    if tiempo_primer_pef is None:
        return {
            "media_abs_desde_pef": 0.0,
            "rms_desde_pef": 0.0,
            "idx_primer_pef": None,
            "tiempo_primer_pef_s": None,
            "muestras_analizadas": 0,
            "duracion_desde_pef_s": 0.0,
            "pef_detectado": False
        }

    idx_primer_pef = int(round(tiempo_primer_pef * fs))
    idx_primer_pef = max(0, min(idx_primer_pef, len(y) - 1))

    tramo = y[idx_primer_pef:]

    if len(tramo) == 0:
        return {
            "media_abs_desde_pef": 0.0,
            "rms_desde_pef": 0.0,
            "idx_primer_pef": int(idx_primer_pef),
            "tiempo_primer_pef_s": float(tiempo_primer_pef),
            "muestras_analizadas": 0,
            "duracion_desde_pef_s": 0.0,
            "pef_detectado": True
        }

    media_abs_desde_pef = float(np.mean(np.abs(tramo)))
    rms_desde_pef = float(np.sqrt(np.mean(tramo ** 2)))

    return {
        "media_abs_desde_pef": media_abs_desde_pef,
        "rms_desde_pef": rms_desde_pef,
        "idx_primer_pef": int(idx_primer_pef),
        "tiempo_primer_pef_s": float(tiempo_primer_pef),
        "muestras_analizadas": int(len(tramo)),
        "duracion_desde_pef_s": float(len(tramo) / fs),
        "pef_detectado": True
    }


def calcular_energia_final_por_ventanas_10ms(y, fs, duracion_total_s=0.25, ventana_s=0.010):
    """Calcula amplitud media absoluta y RMS en ventanas finales de 10 ms."""

    y = np.asarray(y, dtype=np.float64)

    if len(y) == 0 or fs <= 0:
        return []

    n_total = int(round(duracion_total_s * fs))
    n_ventana = int(round(ventana_s * fs))

    n_total = max(1, min(n_total, len(y)))
    n_ventana = max(1, n_ventana)

    idx_inicio_total = len(y) - n_total
    idx_fin_total = len(y)

    resultados = []
    idx = idx_inicio_total

    while idx < idx_fin_total:
        idx_fin = min(idx + n_ventana, idx_fin_total)
        tramo = y[idx:idx_fin]

        if len(tramo) == 0:
            break

        energia_abs_media = float(np.mean(np.abs(tramo)))
        rms = float(np.sqrt(np.mean(tramo ** 2)))

        inicio_rel_ms = -1000.0 * (idx_fin_total - idx) / fs
        fin_rel_ms = -1000.0 * (idx_fin_total - idx_fin) / fs

        resultados.append({
            "inicio_rel_ms": float(inicio_rel_ms),
            "fin_rel_ms": float(fin_rel_ms),
            "energia_abs_media": energia_abs_media,
            "rms": rms,
            "muestras": int(len(tramo))
        })

        idx = idx_fin

    return resultados


def imprimir_comparacion_final_250ms_10ms(datos, usar="denoised"):
    """Presenta, si se invoca, la comparación de ventanas finales de 10 ms."""

    nombres = list(datos.keys())

    if len(nombres) == 0:
        return

    tablas = {}

    for nombre in nombres:
        d = datos[nombre]

        if usar == "original":
            y = d.get("original_raw_graficas", np.array([]))
            fs = d.get("fs_raw", 1)
        else:
            y = d.get("denoised", np.array([]))
            fs = d.get("fs", 1)

        tablas[nombre] = calcular_energia_final_por_ventanas_10ms(
            y,
            fs,
            duracion_total_s=0.25,
            ventana_s=0.010
        )

    builtins.print("\n" + "="*90)
    builtins.print("COMPARACIÓN FINAL 250 ms EN VENTANAS DE 10 ms")
    builtins.print(f"Señal usada: {usar.upper()}")
    builtins.print("="*90)

    if len(nombres) == 1:
        nombre = nombres[0]
        tabla = tablas[nombre]

        builtins.print(f"\n--- {nombre} ---")
        builtins.print("ventana_relativa_final_ms | energia_abs_media | RMS | muestras")

        for fila in tabla:
            builtins.print(
                f"{fila['inicio_rel_ms']:7.1f} a {fila['fin_rel_ms']:6.1f} ms | "
                f"{fila['energia_abs_media']:.6f} | "
                f"{fila['rms']:.6f} | "
                f"{fila['muestras']}"
            )

    else:
        nombre_1 = nombres[0]
        nombre_2 = nombres[1]

        tabla_1 = tablas[nombre_1]
        tabla_2 = tablas[nombre_2]

        n = min(len(tabla_1), len(tabla_2))

        builtins.print(f"\nComparando: {nombre_1}  vs  {nombre_2}")
        builtins.print(
            "ventana_relativa_final_ms | "
            f"E_{nombre_1} | E_{nombre_2} | diferencia_1-2 | ratio_2/1"
        )

        for i in range(n):
            f1 = tabla_1[i]
            f2 = tabla_2[i]

            e1 = float(f1["energia_abs_media"])
            e2 = float(f2["energia_abs_media"])
            diferencia = e1 - e2

            if e1 > 0:
                ratio_2_1 = e2 / (e1 + 1e-12)
            else:
                ratio_2_1 = 0.0

            builtins.print(
                f"{f1['inicio_rel_ms']:7.1f} a {f1['fin_rel_ms']:6.1f} ms | "
                f"{e1:.6f} | "
                f"{e2:.6f} | "
                f"{diferencia:.6f} | "
                f"{ratio_2_1:.3f}"
            )

    builtins.print("="*90 + "\n")


def obtener_tiempo_pef_desde_eventos(eventos_picos):
    """Devuelve el primer PEF temporal, sin recalcular su detección.

    Si no existe PEF, usa el evento de mayor valor_pico que no sea
    artefacto; ese sustituto puede ser una tos. Sin candidatos devuelve None.
    """

    if eventos_picos is None or len(eventos_picos) == 0:
        return None

    pefs = [ev for ev in eventos_picos if ev.get("tipo") == "PEF"]

    if len(pefs) > 0:
        pefs = sorted(pefs, key=lambda ev: ev.get("pico", 0.0))
        return float(pefs[0].get("pico", 0.0))

    candidatos = [
        ev for ev in eventos_picos
        if ev.get("tipo") not in ["artefacto"]
    ]

    if len(candidatos) == 0:
        return None

    mejor = max(candidatos, key=lambda ev: ev.get("valor_pico", 0.0))

    return float(mejor.get("pico", 0.0))


def calcular_ratio_ventanas_post_pef(
    y,
    fs,
    eventos_picos,
    offset_inicio_s,
    duracion_ventana_s,
    nombre_ratio
):
    """Compara amplitudes medias absolutas en dos ventanas post-PEF.

    Para cada ventana V, a_V=mean(abs(x[V])); calcula a_V2/(a_V1+eps),
    (a_V1-a_V2)/(a_V1+eps) y 20*log10(a_V2/(a_V1+eps)+eps).
    Los campos denominados energia contienen amplitud media absoluta.
    Si a_V1 es nula, devuelve cero para esas tres magnitudes.
    """

    y = np.asarray(y, dtype=np.float64)

    t_pef_s = obtener_tiempo_pef_desde_eventos(eventos_picos)

    if len(y) == 0 or fs <= 0 or t_pef_s is None:
        return {
            "nombre_ratio": nombre_ratio,
            "pef_detectado": False,
            "t_pef_s": None,
            "energia_v1": 0.0,
            "energia_v2": 0.0,
            "diferencia_v1_menos_v2": 0.0,
            "ratio_v1_v2": 0.0,
            "aumento_relativo_v1_sobre_v2": 0.0,
            "caida_relativa": 0.0,
            "delta_db_v1_v2": 0.0
        }

    idx_pef = int(round(float(t_pef_s) * fs))
    idx_pef = max(0, min(idx_pef, len(y) - 1))

    n_offset = int(round(float(offset_inicio_s) * fs))
    n_ventana = max(1, int(round(float(duracion_ventana_s) * fs)))

    idx_1_ini = max(0, min(idx_pef + n_offset, len(y)))
    idx_1_fin = max(0, min(idx_1_ini + n_ventana, len(y)))
    idx_2_ini = idx_1_fin
    idx_2_fin = max(0, min(idx_2_ini + n_ventana, len(y)))

    tramo_1 = y[idx_1_ini:idx_1_fin]
    tramo_2 = y[idx_2_ini:idx_2_fin]

    energia_1 = float(np.mean(np.abs(tramo_1))) if len(tramo_1) > 0 else 0.0
    energia_2 = float(np.mean(np.abs(tramo_2))) if len(tramo_2) > 0 else 0.0

    diferencia = float(energia_1 - energia_2)

    if energia_1 > 0:
        ratio = float(energia_2 / (energia_1 + 1e-12))
        caida_relativa = float((energia_1 - energia_2) / (energia_1 + 1e-12))
        delta_db = float(20.0 * np.log10(ratio + 1e-12))
    else:
        ratio = 0.0
        caida_relativa = 0.0
        delta_db = 0.0

    return {
        "nombre_ratio": nombre_ratio,
        "pef_detectado": True,
        "t_pef_s": float(t_pef_s),

        "t_v1_ini_s": float(idx_1_ini / fs),
        "t_v1_fin_s": float(idx_1_fin / fs),
        "t_v2_ini_s": float(idx_2_ini / fs),
        "t_v2_fin_s": float(idx_2_fin / fs),

        "energia_v1": energia_1,
        "energia_v2": energia_2,
        "diferencia_v1_menos_v2": diferencia,
        "ratio_v2_v1": ratio,
        "caida_relativa": caida_relativa,
        "delta_db": delta_db
    }


def calcular_ratios_post_pef_optimizados(y, fs, eventos_picos):
    """Evalúa las ventanas auxiliares post-PEF predefinidas en esta función."""

    return {
        "R0_600ms_desde_PEF": calcular_ratio_ventanas_post_pef(
            y,
            fs,
            eventos_picos,
            offset_inicio_s=0.000,
            duracion_ventana_s=0.600,
            nombre_ratio="R0: PEF a +600 ms vs +600 a +1200 ms"
        ),
        "R1_rapido": calcular_ratio_ventanas_post_pef(
            y,
            fs,
            eventos_picos,
            offset_inicio_s=0.100,
            duracion_ventana_s=0.250,
            nombre_ratio="R1 rápido: PEF+100 a +350 ms vs +350 a +600 ms"
        ),
        "R2_principal": calcular_ratio_ventanas_post_pef(
            y,
            fs,
            eventos_picos,
            offset_inicio_s=0.150,
            duracion_ventana_s=0.600,
            nombre_ratio="R2 principal: PEF+150 a +750 ms vs +750 a +1350 ms"
        )
    }


def imprimir_ratios_post_pef_optimizados(etiqueta, ratios):
    """Presenta, si se invoca, las razones auxiliares post-PEF."""

    builtins.print(f"\nRATIOS POST-PEF {etiqueta}:")

    for clave, r in ratios.items():
        builtins.print(f"\n  {r.get('nombre_ratio', clave)}")

        if not r.get("pef_detectado", False):
            builtins.print("    No se ha detectado PEF válido.")
            continue

        builtins.print(f"    PEF usado:              {r['t_pef_s']:.3f} s")
        builtins.print(f"    Ventana 1:              {r['t_v1_ini_s']:.3f} - {r['t_v1_fin_s']:.3f} s")
        builtins.print(f"    Ventana 2:              {r['t_v2_ini_s']:.3f} - {r['t_v2_fin_s']:.3f} s")
        builtins.print(f"    Energía media abs V1:   {r['energia_v1']:.6f}")
        builtins.print(f"    Energía media abs V2:   {r['energia_v2']:.6f}")
        builtins.print(f"    Diferencia V1 - V2:     {r['diferencia_v1_menos_v2']:.6f}")
        builtins.print(f"    Ratio V1 / V2:          {r['ratio_v2_v1']:.3f}")
        builtins.print(f"    Caída relativa:         {r['caida_relativa']:.3f}")
        builtins.print(f"    Delta dB V1/V2:         {r['delta_db']:.2f} dB")


def calcular_ratio_final_050s_antes_logmel_negro(
    logmel_den,
    t_logmel_den,
    eventos_picos=None,
    duracion_ventana_s=0.50,
    umbral_negro_db=-79.9,
    min_frames_negros=1,
    t_logmel_den_edges=None
):
    """Compara dos ventanas de 0.50 s anteriores a la referencia terminal.

    Usa el borde izquierdo de la primera columna bajo umbral tras
    actividad; si faltan bordes usa su centro. Las ventanas son
    [t_ref-1,t_ref-0.5) y [t_ref-0.5,t_ref). El color negro depende
    del suelo visual, por lo que no equivale necesariamente a silencio.
    """

    logmel_den = np.asarray(logmel_den, dtype=np.float64)
    t_logmel_den = np.asarray(t_logmel_den, dtype=np.float64)

    if t_logmel_den_edges is not None:
        t_logmel_den_edges = np.asarray(t_logmel_den_edges, dtype=np.float64)

    if logmel_den.size == 0 or len(t_logmel_den) == 0:
        return {
            "detectado": False,
            "motivo": "logmel_vacio",
            "t_negro_s": None,
            "energia_v1": 0.0,
            "energia_v2": 0.0,
            "ratio_v1_v2": 0.0,
            "aumento_relativo_v1_sobre_v2": 0.0,
            "caida_relativa": 0.0,
            "delta_db_v1_v2": 0.0
        }

    if logmel_den.ndim == 1:
        logmel_den = logmel_den.reshape(1, -1)

    n_frames = min(logmel_den.shape[1], len(t_logmel_den))

    if n_frames <= 0:
        return {
            "detectado": False,
            "motivo": "sin_frames",
            "t_negro_s": None,
            "energia_v1": 0.0,
            "energia_v2": 0.0,
            "ratio_v1_v2": 0.0,
            "aumento_relativo_v1_sobre_v2": 0.0,
            "caida_relativa": 0.0,
            "delta_db_v1_v2": 0.0
        }

    logmel_use = logmel_den[:, :n_frames]
    t_use = t_logmel_den[:n_frames]


    energia_por_banda = 10.0 ** (logmel_use / 10.0)
    energia_frame = np.mean(energia_por_banda, axis=0)


    max_db_frame = np.max(logmel_use, axis=0)
    negro = max_db_frame <= float(umbral_negro_db)


    t_pef_s = obtener_tiempo_pef_desde_eventos(eventos_picos)

    if t_pef_s is None:
        idx_pef_frame = int(np.argmax(energia_frame))
        t_pef_s = float(t_use[idx_pef_frame])
    else:
        idx_pef_frame = int(np.argmin(np.abs(t_use - float(t_pef_s))))

    min_frames_negros = max(1, int(min_frames_negros))

    idx_negro = None


    for i in range(idx_pef_frame + 1, n_frames - min_frames_negros + 1):
        if np.all(negro[i:i + min_frames_negros]):
            idx_negro = int(i)
            break

    if idx_negro is None:
        return {
            "detectado": False,
            "motivo": "no_llega_a_negro_real",
            "t_pef_s": float(t_pef_s),
            "t_negro_s": None,
            "energia_v1": 0.0,
            "energia_v2": 0.0,
            "ratio_v1_v2": 0.0,
            "aumento_relativo_v1_sobre_v2": 0.0,
            "caida_relativa": 0.0,
            "delta_db_v1_v2": 0.0
        }


    if (
        t_logmel_den_edges is not None
        and len(t_logmel_den_edges) >= idx_negro + 1
    ):

        t_negro_ref_s = float(t_logmel_den_edges[idx_negro])
        t_negro_centro_s = float(t_use[idx_negro])
        referencia_usada = "borde_izquierdo_primer_cuadro_negro"
    else:


        t_negro_ref_s = float(t_use[idx_negro])
        t_negro_centro_s = float(t_use[idx_negro])
        referencia_usada = "centro_frame_negro_fallback"


    t_v2_fin_s = t_negro_ref_s
    t_v2_ini_s = max(0.0, t_v2_fin_s - float(duracion_ventana_s))

    t_v1_fin_s = t_v2_ini_s
    t_v1_ini_s = max(0.0, t_v1_fin_s - float(duracion_ventana_s))

    mascara_v1 = (t_use >= t_v1_ini_s) & (t_use < t_v1_fin_s)
    mascara_v2 = (t_use >= t_v2_ini_s) & (t_use < t_v2_fin_s)

    if not np.any(mascara_v1) or not np.any(mascara_v2):
        return {
            "detectado": False,
            "motivo": "ventanas_500ms_insuficientes",
            "t_pef_s": float(t_pef_s),
            "t_negro_s": float(t_negro_ref_s),
            "t_negro_centro_s": float(t_negro_centro_s),
            "referencia_usada": referencia_usada,
            "t_v1_ini_s": float(t_v1_ini_s),
            "t_v1_fin_s": float(t_v1_fin_s),
            "t_v2_ini_s": float(t_v2_ini_s),
            "t_v2_fin_s": float(t_v2_fin_s),
            "energia_v1": 0.0,
            "energia_v2": 0.0,
            "ratio_v1_v2": 0.0,
            "aumento_relativo_v1_sobre_v2": 0.0,
            "caida_relativa": 0.0,
            "delta_db_v1_v2": 0.0
        }

    energia_v1 = float(np.mean(energia_frame[mascara_v1]))
    energia_v2 = float(np.mean(energia_frame[mascara_v2]))

    diferencia = float(energia_v1 - energia_v2)

    if energia_v2 > 0:
        ratio = float(energia_v1 / (energia_v2 + 1e-12))
        aumento_relativo_v1_sobre_v2 = float((energia_v1 - energia_v2) / (energia_v2 + 1e-12))
        delta_db = float(10.0 * np.log10(ratio + 1e-12))
    else:
        ratio = 0.0
        aumento_relativo_v1_sobre_v2 = 0.0
        delta_db = 0.0


    if energia_v1 > 0:
        caida_relativa = float((energia_v1 - energia_v2) / (energia_v1 + 1e-12))
    else:
        caida_relativa = 0.0

    return {
        "detectado": True,
        "motivo": "primer_punto_del_cuadro_negro_logmel_denoised",
        "referencia_usada": referencia_usada,
        "idx_negro": int(idx_negro),
        "t_pef_s": float(t_pef_s),


        "t_negro_s": float(t_negro_ref_s),
        "t_negro_ref_s": float(t_negro_ref_s),
        "t_negro_centro_s": float(t_negro_centro_s),

        "duracion_ventana_s": float(duracion_ventana_s),
        "umbral_negro_db": float(umbral_negro_db),
        "min_frames_negros": int(min_frames_negros),

        "t_v1_ini_s": float(t_v1_ini_s),
        "t_v1_fin_s": float(t_v1_fin_s),
        "t_v2_ini_s": float(t_v2_ini_s),
        "t_v2_fin_s": float(t_v2_fin_s),

        "energia_v1": energia_v1,
        "energia_v2": energia_v2,
        "diferencia_v1_menos_v2": diferencia,
        "ratio_v1_v2": ratio,
        "aumento_relativo_v1_sobre_v2": aumento_relativo_v1_sobre_v2,
        "caida_relativa": caida_relativa,
        "delta_db_v1_v2": delta_db,

        "frames_v1": int(np.sum(mascara_v1)),
        "frames_v2": int(np.sum(mascara_v2))
    }


def imprimir_ratio_final_050s_logmel_negro(info):
    """Prepara el resumen de las ventanas terminales de 0.50 s."""

    builtins.print("\nRATIO LOG-MEL DENOISED: REFERENCIA = PRIMER PUNTO DEL CUADRO NEGRO:")

    if not info.get("detectado", False):
        builtins.print(f"  No detectado: {info.get('motivo', 'desconocido')}")
        return

    builtins.print(f"  PEF usado:                              {info['t_pef_s']:.3f} s")
    builtins.print(f"  Referencia usada:                       {info.get('referencia_usada', '')}")
    builtins.print(f"  Primer punto cuadro negro:              {info['t_negro_ref_s']:.3f} s")
    builtins.print(f"  Centro del frame negro:                 {info['t_negro_centro_s']:.3f} s")
    builtins.print(f"  Umbral negro real usado:                {info['umbral_negro_db']:.1f} dB")

    builtins.print(
        f"  V1 anterior 500 ms:                     "
        f"{info['t_v1_ini_s']:.3f} - {info['t_v1_fin_s']:.3f} s"
    )
    builtins.print(
        f"  V2 final 500 ms antes del cuadro negro: "
        f"{info['t_v2_ini_s']:.3f} - {info['t_v2_fin_s']:.3f} s"
    )

    builtins.print(f"  Energía V1 anterior:                    {info['energia_v1']:.8f}")
    builtins.print(f"  Energía V2 final:                       {info['energia_v2']:.8f}")
    builtins.print(f"  Diferencia V1 - V2:                     {info['diferencia_v1_menos_v2']:.8f}")
    builtins.print(f"  Ratio V1 / V2:                          {info['ratio_v1_v2']:.3f}")
    builtins.print(f"  Aumento relativo V1 sobre V2:           {info['aumento_relativo_v1_sobre_v2']:.3f}")
    builtins.print(f"  Caída relativa respecto a V1:           {info['caida_relativa']:.3f}")
    builtins.print(f"  Delta dB V1/V2:                         {info['delta_db_v1_v2']:.2f} dB")


def detectar_pefs_envolvente_logmel_denoised(t_env, env_logmel_den):
    """Obtiene PEF acústicos auxiliares de e_LM(t).

    Usa condiciones de separación, duración y valle análogas a la
    curva temporal. Sus máximos pueden diferir por la representación;
    no se suman directamente al recuento temporal de esfuerzos.
    """

    t_env = np.asarray(t_env, dtype=np.float64)
    curva = np.asarray(env_logmel_den, dtype=np.float64)

    if len(t_env) == 0 or len(curva) == 0 or np.max(curva) <= 0:
        return []

    n = min(len(t_env), len(curva))
    t_env = t_env[:n]
    curva = curva[:n]


    if np.max(curva) > 0:
        curva = curva / np.max(curva)

    if len(t_env) > 1:
        dt = float(np.median(np.diff(t_env)))
        if dt <= 0:
            fs_env = 1.0
        else:
            fs_env = 1.0 / dt
    else:
        fs_env = 1.0

    umbral_actividad = 0.12
    prominencia_minima = 0.08

    tiempo_min_segundo_pef_s = 0.30
    caida_minima_valle_segundo_pef = 0.30
    valle_relativo_max_segundo_pef = 0.65
    subida_minima_desde_valle_segundo_pef = 0.30
    tolerancia_solape_misma_curva_s = 0.0
    valle_relativo_max_pef_solapado_independiente = 0.45
    caida_min_pef_solapado_independiente = 0.50
    subida_min_pef_solapado_independiente = 0.45
    tiempo_max_pef_solapado_independiente_s = 1.05
    solape_min_pef_solapado_independiente_s = -1.60
    pico_candidato_min_pef_solapado_independiente = 0.70
    pico_candidato_max_pef_solapado_independiente = 0.90

    duracion_min_pef_al_10_s = 0.30
    muestras_min_pef_al_10 = int(round(duracion_min_pef_al_10_s * fs_env))

    def medir_evento_al_10_env(idx_pico):
        """Obtiene los límites al 10 % de altura de un máximo de e_LM(t)."""
        idx_pico = int(idx_pico)

        if idx_pico < 0 or idx_pico >= len(curva):
            return None

        valor_pico = float(curva[idx_pico])

        if valor_pico <= 0:
            return None

        nivel_10 = 0.10 * valor_pico

        idx_izq = idx_pico
        while idx_izq > 0 and curva[idx_izq] > nivel_10:
            idx_izq -= 1

        idx_der = idx_pico
        while idx_der < len(curva) - 1 and curva[idx_der] > nivel_10:
            idx_der += 1

        duracion_s = float(t_env[idx_der] - t_env[idx_izq])

        return idx_izq, idx_der, duracion_s, valor_pico

    def detectar_primer_pef_antes_de_bajada(curva_local, fs_local):
        """Localiza el máximo del primer intervalo activo.

        El intervalo empieza con u>=0.12 y termina al mantenerse
        u<=0.40 durante 0.12 s; devuelve su máximo absoluto.
        """

        curva_local = np.asarray(curva_local, dtype=np.float64)

        idx_activos = np.where(curva_local >= umbral_actividad)[0]

        if len(idx_activos) == 0:
            return None

        idx_inicio = int(idx_activos[0])

        umbral_basal_bloque = 0.40
        min_basal_sostenido_s = 0.120
        muestras_basal = max(1, int(round(min_basal_sostenido_s * fs_local)))

        idx_fin_bloque = len(curva_local)
        debajo_basal = curva_local <= umbral_basal_bloque

        for i in range(idx_inicio + 1, max(idx_inicio + 1, len(curva_local) - muestras_basal + 1)):
            if np.all(debajo_basal[i:i + muestras_basal]):
                idx_fin_bloque = i
                break

        if idx_fin_bloque <= idx_inicio:
            idx_fin_bloque = len(curva_local)

        tramo_primer_bloque = curva_local[idx_inicio:idx_fin_bloque]

        if len(tramo_primer_bloque) == 0:
            return None

        idx_pef_real = int(idx_inicio + np.argmax(tramo_primer_bloque))

        if curva_local[idx_pef_real] < umbral_actividad:
            return None

        return idx_pef_real

    idx_primer_pef = detectar_primer_pef_antes_de_bajada(curva, fs_env)

    if idx_primer_pef is None:
        return []

    if curva[idx_primer_pef] < umbral_actividad:
        return []

    pefs = [int(idx_primer_pef)]

    distancia_min = max(1, int(0.080 * fs_env))

    try:
        maximos, props = signal.find_peaks(
            curva,
            height=umbral_actividad,
            distance=distancia_min,
            prominence=prominencia_minima
        )
    except Exception:
        maximos = []

    maximos = list(maximos)

    if idx_primer_pef not in maximos:
        maximos.append(idx_primer_pef)

    maximos = sorted([int(m) for m in maximos])

    candidatos_post_primer_pef = [idx for idx in maximos if idx > idx_primer_pef]

    for idx in candidatos_post_primer_pef:
        idx = int(idx)
        ultimo = int(pefs[-1])

        tiempo_entre_pefs = float(t_env[idx] - t_env[ultimo])

        if tiempo_entre_pefs < tiempo_min_segundo_pef_s:
            continue

        ini = min(ultimo, idx)
        fin = max(ultimo, idx)

        if fin <= ini:
            continue

        tramo_entre_picos = curva[ini:fin + 1]
        idx_valle_local = int(np.argmin(tramo_entre_picos))
        idx_valle = ini + idx_valle_local
        valle = float(curva[idx_valle])

        pico_anterior = float(curva[ultimo])
        pico_candidato = float(curva[idx])
        pico_menor = float(min(pico_anterior, pico_candidato))

        caida_desde_pef_anterior = pico_anterior - valle

        if pico_menor > 0:
            valle_relativo = valle / pico_menor
        else:
            valle_relativo = 1.0

        subida_desde_valle = pico_candidato - valle

        med_candidato = medir_evento_al_10_env(idx)

        if med_candidato is not None:
            idx_izq_cand, idx_der_cand, duracion_candidata_s, _ = med_candidato
            duracion_candidata_muestras = idx_der_cand - idx_izq_cand
        else:
            duracion_candidata_s = 0.0
            duracion_candidata_muestras = 0

        med_ultimo = medir_evento_al_10_env(ultimo)

        misma_curva_continua = False
        separacion_basal_entre_eventos_s = 0.0

        if med_ultimo is not None and med_candidato is not None:
            _, idx_der_ult, _, _ = med_ultimo
            idx_izq_cand = int(idx_izq_cand)
            idx_der_ult = int(idx_der_ult)
            separacion_basal_entre_eventos_s = float(t_env[idx_izq_cand] - t_env[idx_der_ult])
            misma_curva_continua = (
                separacion_basal_entre_eventos_s
                <= tolerancia_solape_misma_curva_s
            )

        valle_profundo_con_reascenso = (
            caida_desde_pef_anterior >= caida_min_pef_solapado_independiente
            and valle_relativo <= valle_relativo_max_pef_solapado_independiente
            and subida_desde_valle >= subida_min_pef_solapado_independiente
            and tiempo_entre_pefs <= tiempo_max_pef_solapado_independiente_s
            and separacion_basal_entre_eventos_s <= solape_min_pef_solapado_independiente_s
            and pico_candidato >= pico_candidato_min_pef_solapado_independiente
            and pico_candidato <= pico_candidato_max_pef_solapado_independiente
        )

        if misma_curva_continua and valle_profundo_con_reascenso:
            misma_curva_continua = False

        if misma_curva_continua:
            if pico_candidato > pico_anterior:
                pefs[-1] = idx
            continue

        segundo_pef_valido = (
            tiempo_entre_pefs >= tiempo_min_segundo_pef_s
            and caida_desde_pef_anterior >= caida_minima_valle_segundo_pef
            and valle_relativo <= valle_relativo_max_segundo_pef
            and subida_desde_valle >= subida_minima_desde_valle_segundo_pef
            and duracion_candidata_muestras >= muestras_min_pef_al_10
        )

        if segundo_pef_valido:
            pefs.append(idx)

    eventos = []

    for idx_pico in pefs:
        med = medir_evento_al_10_env(idx_pico)

        if med is None:
            continue

        idx_izq, idx_der, duracion_s, valor_pico = med
        duracion_muestras_pef = idx_der - idx_izq

        if duracion_muestras_pef >= muestras_min_pef_al_10:
            eventos.append({
                "idx_inicio": int(idx_izq),
                "idx_pico": int(idx_pico),
                "idx_fin": int(idx_der),
                "inicio": float(t_env[idx_izq]),
                "pico": float(t_env[idx_pico]),
                "fin": float(t_env[idx_der]),
                "duracion": float(duracion_s),
                "duracion10_muestras": int(duracion_muestras_pef),
                "valor_pico": float(valor_pico),
                "tipo": "PEF",
                "fuente": "envolvente_logmel_denoised",
                "frecuencia_dominante": 0.0,
                "confirmacion_espectral": False,
                "ratio_500_1500": 0.0
            })

    return eventos


def calcular_variacion_bandas_070s_antes_final_logmel_denoised(
    logmel_den,
    t_logmel_den,
    freqs_mel_den=None,
    eventos_picos=None,
    t_logmel_den_edges=None,
    duracion_ventana_s=0.70,
    umbral_negro_db=-79.9,
    umbral_banda_activa_db=-70.0,
    peso_altas=2.5
):
    """Compara actividad por banda en los 0.70 s previos al final estimado.

    Contrasta los cuartos inicial y final de esa ventana y pondera
    las diferencias de dB hacia las bandas altas. Devuelve indicadores
    auxiliares; no constituye por sí sola una regla de clase.
    """

    logmel_den = np.asarray(logmel_den, dtype=np.float64)
    t_logmel_den = np.asarray(t_logmel_den, dtype=np.float64)

    if t_logmel_den_edges is not None:
        t_logmel_den_edges = np.asarray(t_logmel_den_edges, dtype=np.float64)

    if logmel_den.size == 0 or len(t_logmel_den) == 0:
        return {"detectado": False, "motivo": "logmel_vacio"}

    if logmel_den.ndim == 1:
        logmel_den = logmel_den.reshape(1, -1)

    n_frames = min(logmel_den.shape[1], len(t_logmel_den))
    if n_frames <= 1:
        return {"detectado": False, "motivo": "frames_insuficientes"}

    logmel_use = logmel_den[:, :n_frames]
    t_use = t_logmel_den[:n_frames]
    n_bandas = logmel_use.shape[0]

    if freqs_mel_den is None:
        freqs_use = np.arange(n_bandas, dtype=np.float64)
    else:
        freqs_use = np.asarray(freqs_mel_den, dtype=np.float64)
        if len(freqs_use) != n_bandas:
            freqs_use = np.linspace(0.0, 1.0, n_bandas)

    max_db_frame = np.max(logmel_use, axis=0)
    negro = max_db_frame <= float(umbral_negro_db)
    activo = max_db_frame > float(umbral_banda_activa_db)

    energia_frame = np.mean(10.0 ** (logmel_use / 10.0), axis=0)

    t_pef_s = obtener_tiempo_pef_desde_eventos(eventos_picos)
    if t_pef_s is None:
        idx_pef_frame = int(np.argmax(energia_frame))
        t_pef_s = float(t_use[idx_pef_frame])
    else:
        idx_pef_frame = int(np.argmin(np.abs(t_use - float(t_pef_s))))

    idx_final = None
    tipo_final = None


    for i in range(idx_pef_frame + 1, n_frames):
        if negro[i]:
            idx_final = int(i)
            tipo_final = "primer_cuadro_negro_real"
            break

    if idx_final is not None:
        if t_logmel_den_edges is not None and len(t_logmel_den_edges) >= idx_final + 1:
            t_final_s = float(t_logmel_den_edges[idx_final])
            referencia_final = "borde_izquierdo_primer_cuadro_negro"
        else:
            t_final_s = float(t_use[idx_final])
            referencia_final = "centro_primer_frame_negro_fallback"
    else:

        activos_post = np.where(activo[idx_pef_frame:])[0]
        if len(activos_post) == 0:
            return {
                "detectado": False,
                "motivo": "sin_actividad_post_pef",
                "t_pef_s": float(t_pef_s)
            }

        idx_final = int(idx_pef_frame + activos_post[-1])
        tipo_final = "ultima_columna_activa"

        if t_logmel_den_edges is not None and len(t_logmel_den_edges) >= idx_final + 2:
            t_final_s = float(t_logmel_den_edges[idx_final + 1])
            referencia_final = "borde_derecho_ultima_columna_activa"
        else:
            t_final_s = float(t_use[idx_final])
            referencia_final = "centro_ultima_columna_activa_fallback"

    t_inicio_s = max(0.0, t_final_s - float(duracion_ventana_s))
    mascara_ventana = (t_use >= t_inicio_s) & (t_use < t_final_s)

    if not np.any(mascara_ventana):
        return {
            "detectado": False,
            "motivo": "ventana_070s_vacia",
            "t_pef_s": float(t_pef_s),
            "t_final_s": float(t_final_s),
            "referencia_final": referencia_final
        }

    idxs = np.where(mascara_ventana)[0]
    if len(idxs) < 2:
        return {
            "detectado": False,
            "motivo": "ventana_070s_insuficiente",
            "t_pef_s": float(t_pef_s),
            "t_final_s": float(t_final_s),
            "referencia_final": referencia_final
        }

    logmel_win = logmel_use[:, idxs]
    n_win = logmel_win.shape[1]
    n_bloque = max(1, int(round(0.25 * n_win)))

    bloque_inicio = logmel_win[:, :n_bloque]
    bloque_final = logmel_win[:, -n_bloque:]

    media_inicio_db = np.mean(bloque_inicio, axis=1)
    media_final_db = np.mean(bloque_final, axis=1)


    variacion_db_por_banda = media_inicio_db - media_final_db

    activo_inicio = media_inicio_db > float(umbral_banda_activa_db)
    activo_final = media_final_db > float(umbral_banda_activa_db)

    bandas_activas_inicio = int(np.sum(activo_inicio))
    bandas_activas_final = int(np.sum(activo_final))
    diferencia_bandas_activas = int(bandas_activas_inicio - bandas_activas_final)

    if bandas_activas_inicio > 0:
        ratio_bandas_activas_final_inicio = float(
            bandas_activas_final / (bandas_activas_inicio + 1e-12)
        )
    else:
        ratio_bandas_activas_final_inicio = 0.0

    orden = np.argsort(freqs_use)
    tercio = max(1, n_bandas // 3)

    idx_bajas = orden[:tercio]
    idx_medias = orden[tercio:2 * tercio]
    idx_altas = orden[2 * tercio:]

    variacion_bajas_db = float(np.mean(variacion_db_por_banda[idx_bajas])) if len(idx_bajas) else 0.0
    variacion_medias_db = float(np.mean(variacion_db_por_banda[idx_medias])) if len(idx_medias) else 0.0
    variacion_altas_db = float(np.mean(variacion_db_por_banda[idx_altas])) if len(idx_altas) else 0.0

    if np.max(freqs_use) > np.min(freqs_use):
        freq_norm = (freqs_use - np.min(freqs_use)) / (np.max(freqs_use) - np.min(freqs_use))
    else:
        freq_norm = np.linspace(0.0, 1.0, n_bandas)

    pesos = 1.0 + (float(peso_altas) - 1.0) * freq_norm
    variacion_ponderada_altas_db = float(
        np.sum(variacion_db_por_banda * pesos) / (np.sum(pesos) + 1e-12)
    )

    bandas_altas_activas_inicio = int(np.sum(activo_inicio[idx_altas])) if len(idx_altas) else 0
    bandas_altas_activas_final = int(np.sum(activo_final[idx_altas])) if len(idx_altas) else 0

    if bandas_altas_activas_inicio > 0:
        ratio_altas_activas_final_inicio = float(
            bandas_altas_activas_final / (bandas_altas_activas_inicio + 1e-12)
        )
    else:
        ratio_altas_activas_final_inicio = 0.0

    if variacion_altas_db > 0:
        indice_corte_frecuencias_altas = float(
            ratio_altas_activas_final_inicio / (variacion_altas_db + 1e-6)
        )
    else:
        indice_corte_frecuencias_altas = float(ratio_altas_activas_final_inicio * 10.0)

    return {
        "detectado": True,
        "motivo": "variacion_bandas_070s_antes_final_logmel_denoised",
        "t_pef_s": float(t_pef_s),
        "t_final_s": float(t_final_s),
        "idx_final": int(idx_final),
        "tipo_final": tipo_final,
        "referencia_final": referencia_final,
        "t_inicio_ventana_s": float(t_inicio_s),
        "duracion_ventana_s": float(duracion_ventana_s),
        "umbral_negro_db": float(umbral_negro_db),
        "umbral_banda_activa_db": float(umbral_banda_activa_db),

        "bandas_activas_inicio": bandas_activas_inicio,
        "bandas_activas_final": bandas_activas_final,
        "diferencia_bandas_activas": diferencia_bandas_activas,
        "ratio_bandas_activas_final_inicio": ratio_bandas_activas_final_inicio,

        "bandas_altas_activas_inicio": bandas_altas_activas_inicio,
        "bandas_altas_activas_final": bandas_altas_activas_final,
        "ratio_altas_activas_final_inicio": ratio_altas_activas_final_inicio,

        "variacion_media_db": float(np.mean(variacion_db_por_banda)),
        "variacion_bajas_db": variacion_bajas_db,
        "variacion_medias_db": variacion_medias_db,
        "variacion_altas_db": variacion_altas_db,
        "variacion_ponderada_altas_db": variacion_ponderada_altas_db,
        "indice_corte_frecuencias_altas": indice_corte_frecuencias_altas,

        "variacion_db_por_banda": variacion_db_por_banda,
        "freqs_mel_den": freqs_use
    }


def imprimir_variacion_bandas_070s_final_logmel_denoised(info):
    """Prepara el resumen de variación de bandas en la ventana terminal."""

    builtins.print("\nVARIACIÓN DE BANDAS 0.7 s ANTES DEL FINAL LOG-MEL DENOISED:")

    if not info.get("detectado", False):
        builtins.print(f"  No detectado: {info.get('motivo', 'desconocido')}")
        return

    builtins.print(f"  PEF usado:                              {info['t_pef_s']:.3f} s")
    builtins.print(f"  Final Log-Mel denoised:                 {info['t_final_s']:.3f} s")
    builtins.print(f"  Referencia final:                       {info['referencia_final']}")
    builtins.print(f"  Tipo final:                             {info['tipo_final']}")
    builtins.print(f"  Ventana analizada:                      {info['t_inicio_ventana_s']:.3f} - {info['t_final_s']:.3f} s")
    builtins.print(f"  Umbral banda activa:                    {info['umbral_banda_activa_db']:.1f} dB")

    builtins.print(f"  Bandas activas inicio ventana:          {info['bandas_activas_inicio']}")
    builtins.print(f"  Bandas activas final ventana:           {info['bandas_activas_final']}")
    builtins.print(f"  Diferencia bandas activas:              {info['diferencia_bandas_activas']}")
    builtins.print(f"  Ratio bandas activas final/inicio:      {info['ratio_bandas_activas_final_inicio']:.3f}")

    builtins.print(f"  Bandas ALTAS activas inicio:            {info['bandas_altas_activas_inicio']}")
    builtins.print(f"  Bandas ALTAS activas final:             {info['bandas_altas_activas_final']}")
    builtins.print(f"  Ratio ALTAS activas final/inicio:       {info['ratio_altas_activas_final_inicio']:.3f}")

    builtins.print(f"  Variación media todas bandas:           {info['variacion_media_db']:.3f} dB")
    builtins.print(f"  Variación bandas bajas:                 {info['variacion_bajas_db']:.3f} dB")
    builtins.print(f"  Variación bandas medias:                {info['variacion_medias_db']:.3f} dB")
    builtins.print(f"  Variación bandas altas:                 {info['variacion_altas_db']:.3f} dB")
    builtins.print(f"  Variación ponderada peso altas:         {info['variacion_ponderada_altas_db']:.3f} dB")
    builtins.print(f"  Índice corte frecuencias altas:         {info['indice_corte_frecuencias_altas']:.3f}")


def calcular_ratios_ventanas_100ms_a_1s_logmel_denoised(
    logmel_den,
    t_logmel_den,
    freqs_mel_den=None,
    eventos_picos=None,
    t_logmel_den_edges=None,
    umbral_negro_db=-79.9,
    umbral_banda_activa_db=-70.0
):
    """Compara ventanas contiguas de W=0.10,0.20,...,1.00 s.

    V1=[t_ref-2W,t_ref-W), V2=[t_ref-W,t_ref). Promedia 10**(L/10)
    en cada ventana y forma razones globales y de bandas altas.
    Las ventanas sin tramas suficientes se identifican como no válidas.
    """

    logmel_den = np.asarray(logmel_den, dtype=np.float64)
    t_logmel_den = np.asarray(t_logmel_den, dtype=np.float64)

    if t_logmel_den_edges is not None:
        t_logmel_den_edges = np.asarray(t_logmel_den_edges, dtype=np.float64)

    if logmel_den.size == 0 or len(t_logmel_den) == 0:
        return {"detectado": False, "motivo": "logmel_vacio", "ratios": []}

    if logmel_den.ndim == 1:
        logmel_den = logmel_den.reshape(1, -1)

    n_frames = min(logmel_den.shape[1], len(t_logmel_den))

    if n_frames <= 1:
        return {"detectado": False, "motivo": "frames_insuficientes", "ratios": []}

    L = logmel_den[:, :n_frames]
    t = t_logmel_den[:n_frames]
    n_bandas = L.shape[0]

    if freqs_mel_den is None:
        freqs = np.arange(n_bandas, dtype=np.float64)
    else:
        freqs = np.asarray(freqs_mel_den, dtype=np.float64)
        if len(freqs) != n_bandas:
            freqs = np.linspace(0.0, 1.0, n_bandas)

    E = 10.0 ** (L / 10.0)
    E_frame = np.mean(E, axis=0)

    t_pef = obtener_tiempo_pef_desde_eventos(eventos_picos)

    if t_pef is None:
        idx_pef = int(np.argmax(E_frame))
        t_pef = float(t[idx_pef])
    else:
        idx_pef = int(np.argmin(np.abs(t - float(t_pef))))

    max_db = np.max(L, axis=0)
    negro = max_db <= float(umbral_negro_db)
    activo = max_db > float(umbral_banda_activa_db)

    idx_final = None
    tipo_final = None

    for i in range(idx_pef + 1, n_frames):
        if negro[i]:
            idx_final = int(i)
            tipo_final = "primer_cuadro_negro_real"
            break

    if idx_final is not None:
        if t_logmel_den_edges is not None and len(t_logmel_den_edges) >= idx_final + 1:
            t_final = float(t_logmel_den_edges[idx_final])
            referencia_final = "borde_izquierdo_primer_cuadro_negro"
        else:
            t_final = float(t[idx_final])
            referencia_final = "centro_frame_negro_fallback"
    else:
        activos_post = np.where(activo[idx_pef:])[0]

        if len(activos_post) == 0:
            return {
                "detectado": False,
                "motivo": "sin_actividad_post_pef",
                "t_pef_s": float(t_pef),
                "ratios": []
            }

        idx_final = int(idx_pef + activos_post[-1])
        tipo_final = "ultima_columna_activa"

        if t_logmel_den_edges is not None and len(t_logmel_den_edges) >= idx_final + 2:
            t_final = float(t_logmel_den_edges[idx_final + 1])
            referencia_final = "borde_derecho_ultima_columna_activa"
        else:
            t_final = float(t[idx_final])
            referencia_final = "centro_ultima_columna_activa_fallback"

    orden = np.argsort(freqs)
    tercio = max(1, n_bandas // 3)
    idx_altas = orden[2 * tercio:]

    if len(idx_altas) == 0:
        idx_altas = orden[-tercio:]

    resultados = []

    for dur_ms in range(100, 1001, 100):
        W = dur_ms / 1000.0

        t_v1_ini = t_final - 2.0 * W
        t_v1_fin = t_final - W
        t_v2_ini = t_final - W
        t_v2_fin = t_final

        if t_v1_ini < 0:
            resultados.append({
                "duracion_ms": int(dur_ms),
                "valido": False,
                "motivo": "no_hay_tiempo_suficiente",
                "t_v1_ini_s": float(t_v1_ini),
                "t_v1_fin_s": float(t_v1_fin),
                "t_v2_ini_s": float(t_v2_ini),
                "t_v2_fin_s": float(t_v2_fin)
            })
            continue

        m1 = (t >= t_v1_ini) & (t < t_v1_fin)
        m2 = (t >= t_v2_ini) & (t < t_v2_fin)

        if not np.any(m1) or not np.any(m2):
            resultados.append({
                "duracion_ms": int(dur_ms),
                "valido": False,
                "motivo": "ventanas_sin_frames",
                "t_v1_ini_s": float(t_v1_ini),
                "t_v1_fin_s": float(t_v1_fin),
                "t_v2_ini_s": float(t_v2_ini),
                "t_v2_fin_s": float(t_v2_fin)
            })
            continue

        e1 = float(np.mean(E[:, m1]))
        e2 = float(np.mean(E[:, m2]))

        e1_altas = float(np.mean(E[idx_altas, :][:, m1]))
        e2_altas = float(np.mean(E[idx_altas, :][:, m2]))

        ratio_v1_v2 = float(e1 / (e2 + 1e-12)) if e2 > 0 else 0.0
        ratio_v2_v1 = float(e2 / (e1 + 1e-12)) if e1 > 0 else 0.0
        caida_relativa = float((e1 - e2) / (e1 + 1e-12)) if e1 > 0 else 0.0

        ratio_altas_v1_v2 = float(e1_altas / (e2_altas + 1e-12)) if e2_altas > 0 else 0.0
        ratio_altas_v2_v1 = float(e2_altas / (e1_altas + 1e-12)) if e1_altas > 0 else 0.0
        caida_relativa_altas = float((e1_altas - e2_altas) / (e1_altas + 1e-12)) if e1_altas > 0 else 0.0

        media_db_v1 = np.mean(L[:, m1], axis=1)
        media_db_v2 = np.mean(L[:, m2], axis=1)

        bandas_v1 = int(np.sum(media_db_v1 > float(umbral_banda_activa_db)))
        bandas_v2 = int(np.sum(media_db_v2 > float(umbral_banda_activa_db)))
        bandas_altas_v1 = int(np.sum(media_db_v1[idx_altas] > float(umbral_banda_activa_db)))
        bandas_altas_v2 = int(np.sum(media_db_v2[idx_altas] > float(umbral_banda_activa_db)))

        resultados.append({
            "duracion_ms": int(dur_ms),
            "valido": True,
            "t_v1_ini_s": float(t_v1_ini),
            "t_v1_fin_s": float(t_v1_fin),
            "t_v2_ini_s": float(t_v2_ini),
            "t_v2_fin_s": float(t_v2_fin),

            "energia_v1": e1,
            "energia_v2": e2,
            "ratio_v1_v2": ratio_v1_v2,
            "ratio_v2_v1": ratio_v2_v1,
            "caida_relativa": caida_relativa,
            "delta_db_v1_v2": float(10.0 * np.log10(ratio_v1_v2 + 1e-12)),
            "delta_db_v2_v1": float(10.0 * np.log10(ratio_v2_v1 + 1e-12)),

            "energia_altas_v1": e1_altas,
            "energia_altas_v2": e2_altas,
            "ratio_altas_v1_v2": ratio_altas_v1_v2,
            "ratio_altas_v2_v1": ratio_altas_v2_v1,
            "caida_relativa_altas": caida_relativa_altas,
            "delta_db_altas_v1_v2": float(10.0 * np.log10(ratio_altas_v1_v2 + 1e-12)),
            "delta_db_altas_v2_v1": float(10.0 * np.log10(ratio_altas_v2_v1 + 1e-12)),

            "bandas_activas_v1": bandas_v1,
            "bandas_activas_v2": bandas_v2,
            "bandas_altas_activas_v1": bandas_altas_v1,
            "bandas_altas_activas_v2": bandas_altas_v2
        })

    validos = [r for r in resultados if r.get("valido", False)]

    if len(validos) > 0:
        mejor_altas = max(validos, key=lambda r: r.get("ratio_altas_v2_v1", 0.0))
        mejor_global = max(validos, key=lambda r: r.get("ratio_v2_v1", 0.0))
    else:
        mejor_altas = None
        mejor_global = None

    return {
        "detectado": True,
        "motivo": "ratios_ventanas_100ms_a_1s_logmel_denoised",
        "t_pef_s": float(t_pef),
        "t_final_s": float(t_final),
        "tipo_final": tipo_final,
        "referencia_final": referencia_final,
        "umbral_banda_activa_db": float(umbral_banda_activa_db),
        "ratios": resultados,
        "mejor_global": mejor_global,
        "mejor_altas": mejor_altas
    }


def imprimir_ratios_ventanas_100ms_a_1s_logmel_denoised(info):
    """Prepara la salida de razones para las diez duraciones de ventana."""

    builtins.print("\nRATIOS COMPARATIVOS LOG-MEL DENOISED: VENTANAS 100 ms -> 1 s:")

    if not info.get("detectado", False):
        builtins.print(f"  No detectado: {info.get('motivo', 'desconocido')}")
        return

    builtins.print(f"  Final Log-Mel denoised:             {info['t_final_s']:.3f} s")
    builtins.print(f"  Referencia final:                   {info['referencia_final']}")
    builtins.print(f"  Tipo final:                         {info['tipo_final']}")
    builtins.print("  V1 = ventana anterior | V2 = ventana final antes del final Log-Mel")

    for r in info.get("ratios", []):
        dur_ms = r.get("duracion_ms", 0)

        if not r.get("valido", False):
            builtins.print(f"\n  W = {dur_ms:4d} ms -> NO válido ({r.get('motivo', '')})")
            continue

        builtins.print(f"\n  W = {dur_ms:4d} ms")
        builtins.print(f"    V1: {r['t_v1_ini_s']:.3f}-{r['t_v1_fin_s']:.3f} s | V2: {r['t_v2_ini_s']:.3f}-{r['t_v2_fin_s']:.3f} s")
        builtins.print(f"    Ratio V1/V2 global:      {r['ratio_v1_v2']:.3f}")
        builtins.print(f"    Ratio V2/V1 global:      {r['ratio_v2_v1']:.3f}")
        builtins.print(f"    Caída relativa global:   {r['caida_relativa']:.3f}")
        builtins.print(f"    Delta dB V1/V2 global:   {r['delta_db_v1_v2']:.2f} dB")
        builtins.print(f"    Ratio V1/V2 ALTAS:       {r['ratio_altas_v1_v2']:.3f}")
        builtins.print(f"    Ratio V2/V1 ALTAS:       {r['ratio_altas_v2_v1']:.3f}")
        builtins.print(f"    Caída relativa ALTAS:    {r['caida_relativa_altas']:.3f}")
        builtins.print(f"    Bandas activas V1->V2:   {r['bandas_activas_v1']} -> {r['bandas_activas_v2']} | ALTAS {r['bandas_altas_activas_v1']} -> {r['bandas_altas_activas_v2']}")

    mejor_altas = info.get("mejor_altas")
    if mejor_altas is not None:
        builtins.print("\n  Ventana más crítica en frecuencias ALTAS:")
        builtins.print(f"    W = {mejor_altas['duracion_ms']} ms | Ratio V2/V1 ALTAS = {mejor_altas['ratio_altas_v2_v1']:.3f}")


def calcular_ratio_100ms_antes_ruido_basal_logmel_denoised(
    logmel_den,
    t_logmel_den,
    freqs_mel_den=None,
    eventos_picos=None,
    t_logmel_den_edges=None,
    duracion_ventana_s=0.100,
    duracion_basal_s=0.100,
    suelo_db=-80.0,
    techo_db=0.0,
    tolerancia_basal_visual=0.035
):
    """Compara dos ventanas de 100 ms anteriores al final estimado.

    Usa q=clip((L-L_min)/(L_max-L_min),0,1), una escala visual
    adimensional, no potencia lineal. Estima el nivel basal con mediana
    y MAD de la cola y busca actividad persistente desde el final.
    Devuelve razones, ocupación y dinámica frecuencial auxiliares.
    Las claves heredadas que incluyen energía no implican unidades físicas.
    """

    logmel_den = np.asarray(logmel_den, dtype=np.float64)
    t_logmel_den = np.asarray(t_logmel_den, dtype=np.float64)

    if t_logmel_den_edges is not None:
        t_logmel_den_edges = np.asarray(t_logmel_den_edges, dtype=np.float64)

    if logmel_den.size == 0 or len(t_logmel_den) == 0:
        return {
            "detectado": False,
            "motivo": "logmel_vacio"
        }

    if logmel_den.ndim == 1:
        logmel_den = logmel_den.reshape(1, -1)

    n_frames = min(logmel_den.shape[1], len(t_logmel_den))

    if n_frames <= 2:
        return {
            "detectado": False,
            "motivo": "frames_insuficientes"
        }

    L = logmel_den[:, :n_frames]
    t = t_logmel_den[:n_frames]
    n_bandas = L.shape[0]

    if freqs_mel_den is None:
        freqs = np.arange(n_bandas, dtype=np.float64)
    else:
        freqs = np.asarray(freqs_mel_den, dtype=np.float64)
        if len(freqs) != n_bandas:
            freqs = np.linspace(0.0, 1.0, n_bandas)


    visual = np.clip((L - float(suelo_db)) / (float(techo_db) - float(suelo_db)), 0.0, 1.0)

    energia_visual_frame = np.mean(visual, axis=0)


    t_pef_s = obtener_tiempo_pef_desde_eventos(eventos_picos)

    if t_pef_s is None:
        idx_pef = int(np.argmax(energia_visual_frame))
        t_pef_s = float(t[idx_pef])
    else:
        idx_pef = int(np.argmin(np.abs(t - float(t_pef_s))))


    if len(t) > 1:
        dt = float(np.median(np.diff(t)))
        if dt <= 0:
            dt = 0.01
    else:
        dt = 0.01

    n_tail = max(3, int(round(0.300 / dt)))
    n_tail = min(n_tail, n_frames)

    basal_visual = float(np.median(energia_visual_frame[-n_tail:]))
    mad = float(np.median(np.abs(energia_visual_frame[-n_tail:] - basal_visual)))
    umbral_basal = basal_visual + max(float(tolerancia_basal_visual), 3.0 * mad)

    n_basal = max(1, int(round(float(duracion_basal_s) / dt)))


    umbral_energia = basal_visual + max(float(tolerancia_basal_visual), 3.0 * mad)


    umbral_energia = min(float(umbral_energia), 0.08)

    hay_energia = energia_visual_frame > umbral_energia


    n_energia_consecutiva = max(1, int(round(0.030 / dt)))

    idx_ultimo_con_energia = None

    i = n_frames - 1

    while i >= idx_pef:
        if hay_energia[i]:
            ini_check = max(idx_pef, i - n_energia_consecutiva + 1)
            bloque = hay_energia[ini_check:i + 1]

            if np.sum(bloque) >= max(1, int(0.60 * len(bloque))):
                idx_ultimo_con_energia = int(i)
                break

        i -= 1

    if idx_ultimo_con_energia is not None:

        idx_basal = min(idx_ultimo_con_energia + 1, n_frames - 1)
        criterio_basal = "recorrido_desde_atras_hasta_ultima_energia"
    else:


        idx_ultimo_con_energia = int(idx_pef)
        idx_basal = n_frames - 1
        criterio_basal = "fallback_recorrido_atras_sin_energia"

    idx_ultimo_activo = int(idx_ultimo_con_energia)


    if t_logmel_den_edges is not None and len(t_logmel_den_edges) >= idx_basal + 1:
        t_basal_s = float(t_logmel_den_edges[idx_basal])
        referencia_basal = "borde_izquierdo_primer_negro_tras_ultima_energia"
    else:
        t_basal_s = float(t[idx_basal])
        referencia_basal = "centro_frame_primer_negro_tras_ultima_energia_fallback"


    W = float(duracion_ventana_s)

    t_v2_ini = t_basal_s - W
    t_v2_fin = t_basal_s

    t_v1_ini = t_basal_s - 2.0 * W
    t_v1_fin = t_basal_s - W

    if t_v1_ini < 0:
        return {
            "detectado": False,
            "motivo": "no_hay_200ms_antes_del_inicio_basal_final",
            "t_basal_s": float(t_basal_s),
            "referencia_basal": referencia_basal,
            "criterio_basal": criterio_basal,
            "idx_basal": int(idx_basal),
            "idx_ultimo_activo": int(idx_ultimo_activo)
        }

    m1 = (t >= t_v1_ini) & (t < t_v1_fin)
    m2 = (t >= t_v2_ini) & (t < t_v2_fin)

    if not np.any(m1) or not np.any(m2):
        return {
            "detectado": False,
            "motivo": "ventanas_100ms_sin_frames",
            "t_basal_s": float(t_basal_s),
            "referencia_basal": referencia_basal,
            "t_v1_ini_s": float(t_v1_ini),
            "t_v1_fin_s": float(t_v1_fin),
            "t_v2_ini_s": float(t_v2_ini),
            "t_v2_fin_s": float(t_v2_fin)
        }


    v1 = float(np.mean(visual[:, m1]))
    v2 = float(np.mean(visual[:, m2]))

    ratio_v2_v1 = float(v2 / (v1 + 1e-12)) if v1 > 0 else 0.0
    ratio_v1_v2 = float(v1 / (v2 + 1e-12)) if v2 > 0 else 0.0
    diferencia_v1_menos_v2 = float(v1 - v2)

    if v1 > 0:
        variacion_relativa = float((v1 - v2) / (v1 + 1e-12))
    else:
        variacion_relativa = 0.0


    orden = np.argsort(freqs)
    tercio = max(1, n_bandas // 3)
    idx_altas = orden[2 * tercio:]

    if len(idx_altas) == 0:
        idx_altas = orden[-tercio:]

    v1_altas = float(np.mean(visual[idx_altas, :][:, m1]))
    v2_altas = float(np.mean(visual[idx_altas, :][:, m2]))

    ratio_altas_v2_v1 = float(v2_altas / (v1_altas + 1e-12)) if v1_altas > 0 else 0.0
    ratio_altas_v1_v2 = float(v1_altas / (v2_altas + 1e-12)) if v2_altas > 0 else 0.0
    diferencia_altas_v1_menos_v2 = float(v1_altas - v2_altas)

    if v1_altas > 0:
        variacion_relativa_altas = float((v1_altas - v2_altas) / (v1_altas + 1e-12))
    else:
        variacion_relativa_altas = 0.0


    visual_sobre_basal = np.maximum(visual - basal_visual, 0.0)


    umbral_activo_visual = max(0.030, umbral_energia - basal_visual)

    activo_visual = visual_sobre_basal > umbral_activo_visual


    if np.max(freqs) > np.min(freqs):
        freq_norm = (freqs - np.min(freqs)) / (np.max(freqs) - np.min(freqs))
    else:
        freq_norm = np.linspace(0.0, 1.0, n_bandas)

    pesos_freq = 1.0 + 2.0 * freq_norm
    pesos_freq_col = pesos_freq[:, None]

    energia_activa = visual_sobre_basal * activo_visual

    energia_ponderada_v1 = float(
        np.sum(energia_activa[:, m1] * pesos_freq_col)
        / (np.sum(pesos_freq_col * activo_visual[:, m1]) + 1e-12)
    ) if np.any(activo_visual[:, m1]) else 0.0

    energia_ponderada_v2 = float(
        np.sum(energia_activa[:, m2] * pesos_freq_col)
        / (np.sum(pesos_freq_col * activo_visual[:, m2]) + 1e-12)
    ) if np.any(activo_visual[:, m2]) else 0.0


    ratio_mejorado_v2_v1 = float(
        energia_ponderada_v2 / (energia_ponderada_v1 + 1e-12)
    ) if energia_ponderada_v1 > 0 else 0.0

    ratio_mejorado_v1_v2 = float(
        energia_ponderada_v1 / (energia_ponderada_v2 + 1e-12)
    ) if energia_ponderada_v2 > 0 else 0.0

    caida_mejorada = float(
        (energia_ponderada_v1 - energia_ponderada_v2)
        / (energia_ponderada_v1 + 1e-12)
    ) if energia_ponderada_v1 > 0 else 0.0


    ocupacion_v1 = float(np.mean(activo_visual[:, m1]))
    ocupacion_v2 = float(np.mean(activo_visual[:, m2]))

    ratio_ocupacion_v2_v1 = float(
        ocupacion_v2 / (ocupacion_v1 + 1e-12)
    ) if ocupacion_v1 > 0 else 0.0


    ocupacion_altas_v1 = float(np.mean(activo_visual[idx_altas, :][:, m1]))
    ocupacion_altas_v2 = float(np.mean(activo_visual[idx_altas, :][:, m2]))

    ratio_ocupacion_altas_v2_v1 = float(
        ocupacion_altas_v2 / (ocupacion_altas_v1 + 1e-12)
    ) if ocupacion_altas_v1 > 0 else 0.0


    energia_activa_altas_v1 = float(
        np.mean(energia_activa[idx_altas, :][:, m1])
    )
    energia_activa_altas_v2 = float(
        np.mean(energia_activa[idx_altas, :][:, m2])
    )

    ratio_energia_activa_altas_v2_v1 = float(
        energia_activa_altas_v2 / (energia_activa_altas_v1 + 1e-12)
    ) if energia_activa_altas_v1 > 0 else 0.0


    score_escalon_visual = float(
        0.50 * max(0.0, 1.0 - ratio_mejorado_v2_v1)
        + 0.30 * max(0.0, 1.0 - ratio_ocupacion_v2_v1)
        + 0.20 * max(0.0, 1.0 - ratio_ocupacion_altas_v2_v1)
    )


    def calcular_dinamica_frecuencial(mascara_tiempo):
        """Calcula centroide, máximo dominante, borde alto y sus variaciones entre tramas."""
        idx_frames = np.where(mascara_tiempo)[0]

        if len(idx_frames) < 2:
            return {
                "valido": False,
                "centroide_medio_hz": 0.0,
                "velocidad_centroide_hz_s": 0.0,
                "dominante_media_hz": 0.0,
                "velocidad_dominante_hz_s": 0.0,
                "borde_alto_medio_hz": 0.0,
                "velocidad_borde_alto_hz_s": 0.0,
                "flujo_espectral_norm": 0.0,
                "salto_max_centroide_hz": 0.0,
                "salto_max_dominante_hz": 0.0,
                "salto_max_borde_alto_hz": 0.0
            }


        A = energia_activa[:, idx_frames].astype(np.float64)


        energia_total_frame = np.sum(A, axis=0)

        if np.max(energia_total_frame) <= 0:
            return {
                "valido": False,
                "centroide_medio_hz": 0.0,
                "velocidad_centroide_hz_s": 0.0,
                "dominante_media_hz": 0.0,
                "velocidad_dominante_hz_s": 0.0,
                "borde_alto_medio_hz": 0.0,
                "velocidad_borde_alto_hz_s": 0.0,
                "flujo_espectral_norm": 0.0,
                "salto_max_centroide_hz": 0.0,
                "salto_max_dominante_hz": 0.0,
                "salto_max_borde_alto_hz": 0.0
            }


        centroide = np.zeros(len(idx_frames), dtype=np.float64)

        for k in range(len(idx_frames)):
            if energia_total_frame[k] > 0:
                centroide[k] = float(
                    np.sum(freqs * A[:, k]) / (energia_total_frame[k] + 1e-12)
                )
            else:
                centroide[k] = centroide[k - 1] if k > 0 else 0.0


        dominante = np.zeros(len(idx_frames), dtype=np.float64)

        for k in range(len(idx_frames)):
            if energia_total_frame[k] > 0:
                dominante[k] = float(freqs[int(np.argmax(A[:, k]))])
            else:
                dominante[k] = dominante[k - 1] if k > 0 else 0.0


        borde_alto = np.zeros(len(idx_frames), dtype=np.float64)

        for k in range(len(idx_frames)):
            bandas_activas_k = np.where(A[:, k] > 0)[0]

            if len(bandas_activas_k) > 0:
                borde_alto[k] = float(np.max(freqs[bandas_activas_k]))
            else:
                borde_alto[k] = borde_alto[k - 1] if k > 0 else 0.0

        t_local = t[idx_frames]

        if len(t_local) > 1:
            dt_local = np.diff(t_local)
            dt_local = np.where(dt_local <= 0, dt, dt_local)
        else:
            dt_local = np.array([dt], dtype=np.float64)

        def velocidad_media_abs(x):
            """Calcula mean(abs(diff(f))/diff(t)), en Hz/s cuando f está en Hz."""
            x = np.asarray(x, dtype=np.float64)

            if len(x) < 2:
                return 0.0

            return float(np.mean(np.abs(np.diff(x)) / (dt_local + 1e-12)))

        def salto_maximo(x):
            """Devuelve max(abs(diff(f))), o cero si no hay dos valores."""
            x = np.asarray(x, dtype=np.float64)

            if len(x) < 2:
                return 0.0

            return float(np.max(np.abs(np.diff(x))))


        flujo = []

        for k in range(1, A.shape[1]):
            p0 = A[:, k - 1]
            p1 = A[:, k]

            p0 = p0 / (np.sum(p0) + 1e-12)
            p1 = p1 / (np.sum(p1) + 1e-12)

            flujo.append(float(np.sum(np.abs(p1 - p0))))

        flujo_espectral_norm = float(np.mean(flujo)) if len(flujo) > 0 else 0.0

        return {
            "valido": True,
            "centroide_medio_hz": float(np.mean(centroide)),
            "velocidad_centroide_hz_s": velocidad_media_abs(centroide),
            "dominante_media_hz": float(np.mean(dominante)),
            "velocidad_dominante_hz_s": velocidad_media_abs(dominante),
            "borde_alto_medio_hz": float(np.mean(borde_alto)),
            "velocidad_borde_alto_hz_s": velocidad_media_abs(borde_alto),
            "flujo_espectral_norm": flujo_espectral_norm,
            "salto_max_centroide_hz": salto_maximo(centroide),
            "salto_max_dominante_hz": salto_maximo(dominante),
            "salto_max_borde_alto_hz": salto_maximo(borde_alto)
        }

    dinamica_v1 = calcular_dinamica_frecuencial(m1)
    dinamica_v2 = calcular_dinamica_frecuencial(m2)

    def ratio_seguro(a, b):
        """Devuelve a/(b+eps) para b>0 y cero en otro caso."""
        return float(a / (b + 1e-12)) if b > 0 else 0.0

    ratio_velocidad_centroide_v2_v1 = ratio_seguro(
        dinamica_v2["velocidad_centroide_hz_s"],
        dinamica_v1["velocidad_centroide_hz_s"]
    )

    ratio_velocidad_dominante_v2_v1 = ratio_seguro(
        dinamica_v2["velocidad_dominante_hz_s"],
        dinamica_v1["velocidad_dominante_hz_s"]
    )

    ratio_velocidad_borde_alto_v2_v1 = ratio_seguro(
        dinamica_v2["velocidad_borde_alto_hz_s"],
        dinamica_v1["velocidad_borde_alto_hz_s"]
    )

    ratio_flujo_espectral_v2_v1 = ratio_seguro(
        dinamica_v2["flujo_espectral_norm"],
        dinamica_v1["flujo_espectral_norm"]
    )


    score_transicion_frecuencial = float(
        0.35 * min(ratio_velocidad_centroide_v2_v1, 5.0)
        + 0.25 * min(ratio_velocidad_borde_alto_v2_v1, 5.0)
        + 0.25 * min(ratio_flujo_espectral_v2_v1, 5.0)
        + 0.15 * min(ratio_velocidad_dominante_v2_v1, 5.0)
    )


    def calcular_extremos_frecuenciales(mascara_tiempo):
        """Localiza extremos entre bandas activas al menos un 15 % de la ventana.

        Si ninguna alcanza esa persistencia, utiliza cualquier banda
        activa. Las frecuencias corresponden a centros de bandas Mel.
        """

        idx_frames = np.where(mascara_tiempo)[0]

        if len(idx_frames) == 0:
            return {
                "valido": False,
                "f_min_hz": 0.0,
                "f_max_hz": 0.0,
                "rango_hz": 0.0,
                "f_min_frame_mediana_hz": 0.0,
                "f_max_frame_mediana_hz": 0.0,
                "n_bandas_activas": 0,
                "n_frames": 0
            }

        A = activo_visual[:, idx_frames]

        if not np.any(A):
            return {
                "valido": False,
                "f_min_hz": 0.0,
                "f_max_hz": 0.0,
                "rango_hz": 0.0,
                "f_min_frame_mediana_hz": 0.0,
                "f_max_frame_mediana_hz": 0.0,
                "n_bandas_activas": 0,
                "n_frames": int(len(idx_frames))
            }


        persistencia = np.mean(A, axis=1)


        umbral_persistencia = 0.15
        idx_bandas_robustas = np.where(persistencia >= umbral_persistencia)[0]

        if len(idx_bandas_robustas) == 0:

            idx_bandas_robustas = np.where(np.any(A, axis=1))[0]

        if len(idx_bandas_robustas) == 0:
            return {
                "valido": False,
                "f_min_hz": 0.0,
                "f_max_hz": 0.0,
                "rango_hz": 0.0,
                "f_min_frame_mediana_hz": 0.0,
                "f_max_frame_mediana_hz": 0.0,
                "n_bandas_activas": 0,
                "n_frames": int(len(idx_frames))
            }

        f_min = float(np.min(freqs[idx_bandas_robustas]))
        f_max = float(np.max(freqs[idx_bandas_robustas]))
        rango = float(f_max - f_min)


        fmins_frame = []
        fmaxs_frame = []

        for k in range(A.shape[1]):
            idx_act_k = np.where(A[:, k])[0]

            if len(idx_act_k) == 0:
                continue

            fmins_frame.append(float(np.min(freqs[idx_act_k])))
            fmaxs_frame.append(float(np.max(freqs[idx_act_k])))

        if len(fmins_frame) > 0:
            f_min_frame_mediana = float(np.median(fmins_frame))
            f_max_frame_mediana = float(np.median(fmaxs_frame))
        else:
            f_min_frame_mediana = f_min
            f_max_frame_mediana = f_max

        return {
            "valido": True,
            "f_min_hz": f_min,
            "f_max_hz": f_max,
            "rango_hz": rango,
            "f_min_frame_mediana_hz": f_min_frame_mediana,
            "f_max_frame_mediana_hz": f_max_frame_mediana,
            "n_bandas_activas": int(len(idx_bandas_robustas)),
            "n_frames": int(len(idx_frames))
        }

    extremos_v1 = calcular_extremos_frecuenciales(m1)
    extremos_v2 = calcular_extremos_frecuenciales(m2)

    f_min_v1 = float(extremos_v1.get("f_min_hz", 0.0))
    f_max_v1 = float(extremos_v1.get("f_max_hz", 0.0))
    rango_v1 = float(extremos_v1.get("rango_hz", 0.0))

    f_min_v2 = float(extremos_v2.get("f_min_hz", 0.0))
    f_max_v2 = float(extremos_v2.get("f_max_hz", 0.0))
    rango_v2 = float(extremos_v2.get("rango_hz", 0.0))


    ratio_extremo_precoz_fmax_v2_fmin_v1 = (
        float(f_max_v2 / (f_min_v1 + 1e-12)) if f_min_v1 > 0 else 0.0
    )


    ratio_fmax_v2_fmax_v1 = (
        float(f_max_v2 / (f_max_v1 + 1e-12)) if f_max_v1 > 0 else 0.0
    )

    ratio_fmin_v2_fmin_v1 = (
        float(f_min_v2 / (f_min_v1 + 1e-12)) if f_min_v1 > 0 else 0.0
    )

    ratio_rango_v2_v1 = (
        float(rango_v2 / (rango_v1 + 1e-12)) if rango_v1 > 0 else 0.0
    )


    f_global_min = float(np.min(freqs))
    f_global_max = float(np.max(freqs))
    rango_global = max(f_global_max - f_global_min, 1e-12)

    score_extremos_frecuenciales = float(
        np.clip((f_max_v2 - f_min_v1) / rango_global, 0.0, 1.0)
    )


    score_direccion_fmax = float(
        (f_max_v2 - f_max_v1) / rango_global
    )


    db_v1 = float(np.mean(L[:, m1]))
    db_v2 = float(np.mean(L[:, m2]))
    db_altas_v1 = float(np.mean(L[idx_altas, :][:, m1]))
    db_altas_v2 = float(np.mean(L[idx_altas, :][:, m2]))

    return {
        "detectado": True,
        "motivo": "ratio_100ms_antes_ruido_basal_logmel_denoised",
        "t_pef_s": float(t_pef_s),
        "t_basal_s": float(t_basal_s),
        "referencia_basal": referencia_basal,
        "criterio_basal": criterio_basal,
        "idx_basal": int(idx_basal),
        "idx_ultimo_activo": int(idx_ultimo_activo),
        "basal_visual": float(basal_visual),
        "umbral_basal": float(umbral_basal),
        "umbral_energia_recorrido_atras": float(umbral_energia),

        "t_v1_ini_s": float(t_v1_ini),
        "t_v1_fin_s": float(t_v1_fin),
        "t_v2_ini_s": float(t_v2_ini),
        "t_v2_fin_s": float(t_v2_fin),

        "media_visual_v1": v1,
        "media_visual_v2": v2,
        "ratio_v2_v1": ratio_v2_v1,
        "ratio_v1_v2": ratio_v1_v2,
        "diferencia_v1_menos_v2": diferencia_v1_menos_v2,
        "variacion_relativa": variacion_relativa,

        "media_visual_altas_v1": v1_altas,
        "media_visual_altas_v2": v2_altas,
        "ratio_altas_v2_v1": ratio_altas_v2_v1,
        "ratio_altas_v1_v2": ratio_altas_v1_v2,
        "diferencia_altas_v1_menos_v2": diferencia_altas_v1_menos_v2,
        "variacion_relativa_altas": variacion_relativa_altas,

        "umbral_activo_visual": float(umbral_activo_visual),
        "energia_ponderada_activa_v1": energia_ponderada_v1,
        "energia_ponderada_activa_v2": energia_ponderada_v2,
        "ratio_mejorado_v2_v1": ratio_mejorado_v2_v1,
        "ratio_mejorado_v1_v2": ratio_mejorado_v1_v2,
        "caida_mejorada": caida_mejorada,
        "ocupacion_v1": ocupacion_v1,
        "ocupacion_v2": ocupacion_v2,
        "ratio_ocupacion_v2_v1": ratio_ocupacion_v2_v1,
        "ocupacion_altas_v1": ocupacion_altas_v1,
        "ocupacion_altas_v2": ocupacion_altas_v2,
        "ratio_ocupacion_altas_v2_v1": ratio_ocupacion_altas_v2_v1,
        "energia_activa_altas_v1": energia_activa_altas_v1,
        "energia_activa_altas_v2": energia_activa_altas_v2,
        "ratio_energia_activa_altas_v2_v1": ratio_energia_activa_altas_v2_v1,
        "score_escalon_visual": score_escalon_visual,

        "extremos_frecuenciales_v1": extremos_v1,
        "extremos_frecuenciales_v2": extremos_v2,
        "ratio_extremo_precoz_fmax_v2_fmin_v1": ratio_extremo_precoz_fmax_v2_fmin_v1,
        "ratio_fmax_v2_fmax_v1": ratio_fmax_v2_fmax_v1,
        "ratio_fmin_v2_fmin_v1": ratio_fmin_v2_fmin_v1,
        "ratio_rango_v2_v1": ratio_rango_v2_v1,
        "score_extremos_frecuenciales": score_extremos_frecuenciales,
        "score_direccion_fmax": score_direccion_fmax,

        "dinamica_frecuencial_v1": dinamica_v1,
        "dinamica_frecuencial_v2": dinamica_v2,
        "ratio_velocidad_centroide_v2_v1": ratio_velocidad_centroide_v2_v1,
        "ratio_velocidad_dominante_v2_v1": ratio_velocidad_dominante_v2_v1,
        "ratio_velocidad_borde_alto_v2_v1": ratio_velocidad_borde_alto_v2_v1,
        "ratio_flujo_espectral_v2_v1": ratio_flujo_espectral_v2_v1,
        "score_transicion_frecuencial": score_transicion_frecuencial,

        "media_db_v1": db_v1,
        "media_db_v2": db_v2,
        "media_db_altas_v1": db_altas_v1,
        "media_db_altas_v2": db_altas_v2,
        "frames_v1": int(np.sum(m1)),
        "frames_v2": int(np.sum(m2))
    }


def imprimir_SOLO_ratio_100ms_ruido_basal_logmel_denoised(info):
    """Prepara el resumen auxiliar de las dos ventanas previas al nivel basal."""

    builtins.print("\n" + "="*90)
    builtins.print("RATIO SINCERO LOG-MEL DENOISED: 100 ms ANTES DEL BASAL FINAL RECORRIDO DESDE ATRÁS")
    builtins.print("="*90)

    if not info.get("detectado", False):
        builtins.print(f"No detectado: {info.get('motivo', 'desconocido')}")
        builtins.print("="*90 + "\n")
        return

    builtins.print(f"PEF usado:                         {info['t_pef_s']:.3f} s")
    builtins.print(f"Momento ruido basal:               {info['t_basal_s']:.3f} s")
    builtins.print(f"Referencia basal:                  {info['referencia_basal']}")
    builtins.print(f"Criterio basal:                    {info['criterio_basal']}")
    builtins.print(f"Umbral energía recorrido atrás:    {info.get('umbral_energia_recorrido_atras', 0.0):.6f}")

    builtins.print(
        f"V1 anterior 100 ms:                "
        f"{info['t_v1_ini_s']:.3f} - {info['t_v1_fin_s']:.3f} s"
    )
    builtins.print(
        f"V2 final 100 ms antes basal:       "
        f"{info['t_v2_ini_s']:.3f} - {info['t_v2_fin_s']:.3f} s"
    )

    builtins.print("\nRATIO PRINCIPAL POR EXTREMOS FRECUENCIALES")
    e1 = info.get("extremos_frecuenciales_v1", {})
    e2 = info.get("extremos_frecuenciales_v2", {})

    builtins.print("V1 = 100 ms anteriores | V2 = 100 ms justo antes del basal")
    builtins.print("\nV1 anterior:")
    builtins.print(f"  Frecuencia mínima activa V1:       {e1.get('f_min_hz', 0.0):.2f} Hz")
    builtins.print(f"  Frecuencia máxima activa V1:       {e1.get('f_max_hz', 0.0):.2f} Hz")
    builtins.print(f"  Rango frecuencial V1:              {e1.get('rango_hz', 0.0):.2f} Hz")
    builtins.print(f"  Bandas activas robustas V1:        {e1.get('n_bandas_activas', 0)}")

    builtins.print("\nV2 final:")
    builtins.print(f"  Frecuencia mínima activa V2:       {e2.get('f_min_hz', 0.0):.2f} Hz")
    builtins.print(f"  Frecuencia máxima activa V2:       {e2.get('f_max_hz', 0.0):.2f} Hz")
    builtins.print(f"  Rango frecuencial V2:              {e2.get('rango_hz', 0.0):.2f} Hz")
    builtins.print(f"  Bandas activas robustas V2:        {e2.get('n_bandas_activas', 0)}")

    builtins.print("\nRatios de extremos:")
    builtins.print(f"  Ratio principal fmax(V2)/fmin(V1): {info['ratio_extremo_precoz_fmax_v2_fmin_v1']:.6f}")
    builtins.print(f"  Ratio fmax(V2)/fmax(V1):           {info['ratio_fmax_v2_fmax_v1']:.6f}")
    builtins.print(f"  Ratio fmin(V2)/fmin(V1):           {info['ratio_fmin_v2_fmin_v1']:.6f}")
    builtins.print(f"  Ratio rango(V2)/rango(V1):         {info['ratio_rango_v2_v1']:.6f}")
    builtins.print(f"  Score extremos frecuenciales:      {info['score_extremos_frecuenciales']:.6f}")
    builtins.print(f"  Score dirección fmax V2-V1:        {info['score_direccion_fmax']:.6f}")

    builtins.print("\nCONTROL SECUNDARIO")
    builtins.print(f"  Ratio MEJORADO energía V2/V1:      {info['ratio_mejorado_v2_v1']:.6f}")
    builtins.print(f"  Score transición frecuencial:      {info['score_transicion_frecuencial']:.6f}")

    builtins.print("\nCONTROL SIMPLE EN ESCALA VISUAL LOG-MEL [-80, 0] -> [0, 1]")
    builtins.print(f"Media visual V1:                   {info['media_visual_v1']:.6f}")
    builtins.print(f"Media visual V2:                   {info['media_visual_v2']:.6f}")
    builtins.print(f"Ratio simple V2/V1:                {info['ratio_v2_v1']:.6f}")
    builtins.print(f"Ratio simple V1/V2:                {info['ratio_v1_v2']:.6f}")
    builtins.print(f"Diferencia simple V1 - V2:         {info['diferencia_v1_menos_v2']:.6f}")
    builtins.print(f"Variación simple relativa:         {info['variacion_relativa']:.6f}")

    builtins.print("\nCONTROL EN dB DIRECTO")
    builtins.print(f"Media dB V1:                       {info['media_db_v1']:.3f} dB")
    builtins.print(f"Media dB V2:                       {info['media_db_v2']:.3f} dB")
    builtins.print(f"Media dB altas V1:                 {info['media_db_altas_v1']:.3f} dB")
    builtins.print(f"Media dB altas V2:                 {info['media_db_altas_v2']:.3f} dB")
    builtins.print("="*90 + "\n")


def calcular_ratio_variacion_frecuencial_200ms_logmel_denoised(
    logmel_den,
    t_logmel_den,
    freqs_mel_den=None,
    eventos_picos=None,
    t_logmel_den_edges=None,
    duracion_ventana_s=0.200,
    suelo_db=-80.0,
    techo_db=0.0,
    tolerancia_basal_visual=0.035,
    t_referencia_fija_s=None,
    etiqueta_logmel="DENOISED"
):
    """Describe actividad frecuencial en los 200 ms previos al final.

    Estima actividad mediante mediana y MAD de la cola en escala q.
    Calcula extremos de bandas y razones f_max/f_min. La puntuación
    S_T=clip(6*sum(f_b*q[m,b])/(N_tramas*sum(f_b)+eps),0,1)
    es auxiliar: no se compara con 0.60 en la decisión final actual.
    Original y procesada pueden compartir t_ref para comparar el mismo
    intervalo. Si faltan frecuencias válidas, los índices de banda
    usados como respaldo no deben interpretarse como hercios medidos.
    """

    logmel_den = np.asarray(logmel_den, dtype=np.float64)
    t_logmel_den = np.asarray(t_logmel_den, dtype=np.float64)

    if t_logmel_den_edges is not None:
        t_logmel_den_edges = np.asarray(t_logmel_den_edges, dtype=np.float64)

    if logmel_den.size == 0 or len(t_logmel_den) == 0:
        return {
            "detectado": False,
            "motivo": "logmel_vacio"
        }

    if logmel_den.ndim == 1:
        logmel_den = logmel_den.reshape(1, -1)

    n_frames = min(logmel_den.shape[1], len(t_logmel_den))

    if n_frames <= 2:
        return {
            "detectado": False,
            "motivo": "frames_insuficientes"
        }

    L = logmel_den[:, :n_frames]
    t = t_logmel_den[:n_frames]
    n_bandas = L.shape[0]

    if freqs_mel_den is None:
        freqs = np.arange(n_bandas, dtype=np.float64)
    else:
        freqs = np.asarray(freqs_mel_den, dtype=np.float64)
        if len(freqs) != n_bandas:
            freqs = np.linspace(1.0, float(n_bandas), n_bandas)


    visual = np.clip(
        (L - float(suelo_db)) / (float(techo_db) - float(suelo_db)),
        0.0,
        1.0
    )

    energia_visual_frame = np.mean(visual, axis=0)


    t_pef_s = obtener_tiempo_pef_desde_eventos(eventos_picos)

    if t_pef_s is None:
        idx_pef = int(np.argmax(energia_visual_frame))
        t_pef_s = float(t[idx_pef])
    else:
        idx_pef = int(np.argmin(np.abs(t - float(t_pef_s))))


    if len(t) > 1:
        dt = float(np.median(np.diff(t)))
        if dt <= 0:
            dt = 0.01
    else:
        dt = 0.01


    n_tail = max(3, int(round(0.300 / dt)))
    n_tail = min(n_tail, n_frames)

    basal_visual = float(np.median(energia_visual_frame[-n_tail:]))
    mad = float(np.median(np.abs(energia_visual_frame[-n_tail:] - basal_visual)))


    umbral_frame_energia = basal_visual + max(float(tolerancia_basal_visual), 3.0 * mad)
    umbral_frame_energia = min(float(umbral_frame_energia), 0.08)


    umbral_pixel_activo = max(0.030, umbral_frame_energia - basal_visual)

    visual_sobre_basal = np.maximum(visual - basal_visual, 0.0)
    activo_pixel = visual_sobre_basal > umbral_pixel_activo


    if t_referencia_fija_s is not None:
        t_ref_s = float(t_referencia_fija_s)
        idx_referencia = int(np.argmin(np.abs(t - t_ref_s)))
        idx_ultimo_con_energia = -1
        referencia = "misma_referencia_temporal_del_logmel_denoised"
    else:


        hay_energia_frame = np.any(activo_pixel, axis=0)

        n_energia_consecutiva = max(1, int(round(0.030 / dt)))
        idx_ultimo_con_energia = None

        i = n_frames - 1

        while i >= idx_pef:
            if hay_energia_frame[i]:
                ini_check = max(idx_pef, i - n_energia_consecutiva + 1)
                bloque = hay_energia_frame[ini_check:i + 1]

                if np.sum(bloque) >= max(1, int(0.60 * len(bloque))):
                    idx_ultimo_con_energia = int(i)
                    break

            i -= 1

        if idx_ultimo_con_energia is None:
            return {
                "detectado": False,
                "motivo": "no_se_encuentra_ultima_energia_desde_atras",
                "t_pef_s": float(t_pef_s),
                "etiqueta_logmel": etiqueta_logmel
            }

        idx_referencia = min(idx_ultimo_con_energia + 1, n_frames - 1)

        if t_logmel_den_edges is not None and len(t_logmel_den_edges) >= idx_referencia + 1:
            t_ref_s = float(t_logmel_den_edges[idx_referencia])
            referencia = "borde_izquierdo_primer_negro_tras_ultima_energia"
        else:
            t_ref_s = float(t[idx_referencia])
            referencia = "centro_frame_referencia_fallback"


    t_win_ini = t_ref_s - float(duracion_ventana_s)
    t_win_fin = t_ref_s

    if t_win_ini < 0:
        return {
            "detectado": False,
            "motivo": "no_hay_200ms_antes_referencia",
            "etiqueta_logmel": etiqueta_logmel,
            "t_referencia_s": float(t_ref_s),
            "referencia": referencia
        }

    mascara_win = (t >= t_win_ini) & (t < t_win_fin)

    if not np.any(mascara_win):
        return {
            "detectado": False,
            "motivo": "ventana_200ms_sin_frames",
            "etiqueta_logmel": etiqueta_logmel,
            "t_referencia_s": float(t_ref_s),
            "referencia": referencia,
            "t_win_ini_s": float(t_win_ini),
            "t_win_fin_s": float(t_win_fin)
        }

    activo_win = activo_pixel[:, mascara_win]

    if not np.any(activo_win):
        return {
            "detectado": False,
            "motivo": "ventana_200ms_sin_frecuencias_activas",
            "etiqueta_logmel": etiqueta_logmel,
            "t_referencia_s": float(t_ref_s),
            "referencia": referencia,
            "t_win_ini_s": float(t_win_ini),
            "t_win_fin_s": float(t_win_fin)
        }


    idx_bandas_activas = np.where(np.any(activo_win, axis=1))[0]

    f_min_200 = float(np.min(freqs[idx_bandas_activas]))
    f_max_200 = float(np.max(freqs[idx_bandas_activas]))
    rango_200 = float(f_max_200 - f_min_200)

    ratio_fmax_fmin_200 = float(f_max_200 / (f_min_200 + 1e-12)) if f_min_200 > 0 else 0.0
    ratio_variacion_relativa_200 = float(rango_200 / (f_min_200 + 1e-12)) if f_min_200 > 0 else 0.0
    velocidad_variacion_hz_s = float(rango_200 / (float(duracion_ventana_s) + 1e-12))


    t_mitad = t_win_ini + float(duracion_ventana_s) / 2.0

    m_primera = (t >= t_win_ini) & (t < t_mitad)
    m_segunda = (t >= t_mitad) & (t < t_win_fin)

    activo_primera = activo_pixel[:, m_primera]
    activo_segunda = activo_pixel[:, m_segunda]

    def extremos_de_mascara(activo_sub):
        """Obtiene frecuencias mínima y máxima entre las bandas activas de la máscara."""
        if activo_sub.size == 0 or not np.any(activo_sub):
            return {
                "valido": False,
                "f_min": 0.0,
                "f_max": 0.0,
                "rango": 0.0,
                "n_bandas": 0
            }

        idx_bandas = np.where(np.any(activo_sub, axis=1))[0]

        fmin = float(np.min(freqs[idx_bandas]))
        fmax = float(np.max(freqs[idx_bandas]))

        return {
            "valido": True,
            "f_min": fmin,
            "f_max": fmax,
            "rango": float(fmax - fmin),
            "n_bandas": int(len(idx_bandas))
        }

    e_primera = extremos_de_mascara(activo_primera)
    e_segunda = extremos_de_mascara(activo_segunda)

    if e_primera["valido"] and e_segunda["valido"]:
        ratio_dinamico_fmax_segunda_fmin_primera = float(
            e_segunda["f_max"] / (e_primera["f_min"] + 1e-12)
        ) if e_primera["f_min"] > 0 else 0.0

        diferencia_fmax_segunda_menos_fmin_primera = float(
            e_segunda["f_max"] - e_primera["f_min"]
        )

        velocidad_dinamica_hz_s = float(
            diferencia_fmax_segunda_menos_fmin_primera
            / (float(duracion_ventana_s) + 1e-12)
        )
    else:
        ratio_dinamico_fmax_segunda_fmin_primera = 0.0
        diferencia_fmax_segunda_menos_fmin_primera = 0.0
        velocidad_dinamica_hz_s = 0.0


    idx_frames_win = np.where(mascara_win)[0]
    fmin_frame = []
    fmax_frame = []

    for idx_frame in idx_frames_win:
        idx_act = np.where(activo_pixel[:, idx_frame])[0]

        if len(idx_act) == 0:
            continue

        fmin_frame.append(float(np.min(freqs[idx_act])))
        fmax_frame.append(float(np.max(freqs[idx_act])))

    if len(fmax_frame) >= 2:
        salto_max_fmax_hz = float(np.max(np.abs(np.diff(fmax_frame))))
        salto_max_fmin_hz = float(np.max(np.abs(np.diff(fmin_frame))))
        variacion_total_fmax_hz = float(np.sum(np.abs(np.diff(fmax_frame))))
        variacion_total_fmin_hz = float(np.sum(np.abs(np.diff(fmin_frame))))
    else:
        salto_max_fmax_hz = 0.0
        salto_max_fmin_hz = 0.0
        variacion_total_fmax_hz = 0.0
        variacion_total_fmin_hz = 0.0


    f_global_min = float(np.min(freqs))
    f_global_max = float(np.max(freqs))
    rango_global = max(f_global_max - f_global_min, 1e-12)


    freqs_safe = np.asarray(freqs, dtype=np.float64)
    L_win = L[:, mascara_win]

    db_relativo_win = np.clip(
        (L_win - float(suelo_db)) / (float(techo_db) - float(suelo_db)),
        0.0,
        1.0
    )

    baremo_freq_por_db = freqs_safe[:, None] * db_relativo_win

    baremo_total_freq_db_200 = float(np.sum(baremo_freq_por_db))
    baremo_medio_freq_db_200 = float(np.mean(baremo_freq_por_db))


    max_baremo_teorico = float(np.sum(freqs_safe) * max(1, np.sum(mascara_win)))

    baremo_normalizado_freq_db_200 = float(
        baremo_total_freq_db_200 / (max_baremo_teorico + 1e-12)
    ) if max_baremo_teorico > 0 else 0.0


    ratio_solo_baremo_freq_db_200 = baremo_normalizado_freq_db_200


    factor_peso_ratio_baremo = 6.0

    ratio_baremo_freq_db_200_mas_peso = float(
        np.clip(ratio_solo_baremo_freq_db_200 * factor_peso_ratio_baremo, 0.0, 1.0)
    )


    mascara_altas_1000 = freqs_safe >= 1000.0

    if np.any(mascara_altas_1000):
        baremo_altas_1000_total = float(
            np.sum(baremo_freq_por_db[mascara_altas_1000, :])
        )

        max_baremo_altas_1000_teorico = float(
            np.sum(freqs_safe[mascara_altas_1000]) * max(1, np.sum(mascara_win))
        )

        baremo_altas_1000_normalizado = float(
            baremo_altas_1000_total / (max_baremo_altas_1000_teorico + 1e-12)
        ) if max_baremo_altas_1000_teorico > 0 else 0.0
    else:
        baremo_altas_1000_total = 0.0
        baremo_altas_1000_normalizado = 0.0

    proporcion_baremo_altas_1000 = float(
        baremo_altas_1000_total / (baremo_total_freq_db_200 + 1e-12)
    ) if baremo_total_freq_db_200 > 0 else 0.0


    factor_baremo_freq_db = float(
        1.0 + 3.0 * baremo_normalizado_freq_db_200 + 3.0 * proporcion_baremo_altas_1000
    )

    ratio_mixto_variacion_baremo_freq_db = float(
        ratio_fmax_fmin_200 * factor_baremo_freq_db
    )


    score_variacion_frecuencial_200ms = float(
        np.clip(rango_200 / rango_global, 0.0, 1.0)
    )

    score_mixto_variacion_baremo_freq_db = float(
        np.clip(
            0.50 * score_variacion_frecuencial_200ms
            + 0.30 * baremo_normalizado_freq_db_200
            + 0.20 * proporcion_baremo_altas_1000,
            0.0,
            1.0
        )
    )


    energia_total_sin_ponderar_200 = baremo_total_freq_db_200
    energia_total_ponderada_200 = baremo_total_freq_db_200
    energia_altas_1000_200 = baremo_altas_1000_total
    energia_altas_1000_ponderada_200 = baremo_altas_1000_total
    proporcion_energia_altas_1000 = proporcion_baremo_altas_1000
    proporcion_energia_altas_1000_ponderada = proporcion_baremo_altas_1000
    factor_energia_1000hz = factor_baremo_freq_db
    ratio_mixto_variacion_energia_1000hz = ratio_mixto_variacion_baremo_freq_db
    score_mixto_variacion_energia_1000hz = score_mixto_variacion_baremo_freq_db

    return {
        "detectado": True,
        "motivo": "ratio_variacion_frecuencial_200ms_logmel_denoised",
        "etiqueta_logmel": etiqueta_logmel,
        "t_pef_s": float(t_pef_s),
        "t_referencia_s": float(t_ref_s),
        "referencia": referencia,
        "idx_ultimo_con_energia": int(idx_ultimo_con_energia),
        "idx_referencia": int(idx_referencia),
        "t_win_ini_s": float(t_win_ini),
        "t_win_fin_s": float(t_win_fin),
        "t_mitad_s": float(t_mitad),
        "duracion_ventana_s": float(duracion_ventana_s),
        "umbral_pixel_activo": float(umbral_pixel_activo),

        "f_min_200_hz": f_min_200,
        "f_max_200_hz": f_max_200,
        "rango_200_hz": rango_200,
        "ratio_fmax_fmin_200": ratio_fmax_fmin_200,
        "ratio_variacion_relativa_200": ratio_variacion_relativa_200,
        "velocidad_variacion_hz_s": velocidad_variacion_hz_s,

        "baremo_total_freq_db_200": baremo_total_freq_db_200,
        "baremo_medio_freq_db_200": baremo_medio_freq_db_200,
        "baremo_normalizado_freq_db_200": baremo_normalizado_freq_db_200,
        "ratio_solo_baremo_freq_db_200": ratio_solo_baremo_freq_db_200,
        "factor_peso_ratio_baremo": factor_peso_ratio_baremo,
        "ratio_baremo_freq_db_200_mas_peso": ratio_baremo_freq_db_200_mas_peso,
        "baremo_altas_1000_total": baremo_altas_1000_total,
        "baremo_altas_1000_normalizado": baremo_altas_1000_normalizado,
        "proporcion_baremo_altas_1000": proporcion_baremo_altas_1000,
        "factor_baremo_freq_db": factor_baremo_freq_db,
        "ratio_mixto_variacion_baremo_freq_db": ratio_mixto_variacion_baremo_freq_db,
        "score_mixto_variacion_baremo_freq_db": score_mixto_variacion_baremo_freq_db,

        "energia_total_sin_ponderar_200": energia_total_sin_ponderar_200,
        "energia_total_ponderada_200": energia_total_ponderada_200,
        "energia_altas_1000_200": energia_altas_1000_200,
        "energia_altas_1000_ponderada_200": energia_altas_1000_ponderada_200,
        "proporcion_energia_altas_1000": proporcion_energia_altas_1000,
        "proporcion_energia_altas_1000_ponderada": proporcion_energia_altas_1000_ponderada,
        "factor_energia_1000hz": factor_energia_1000hz,
        "ratio_mixto_variacion_energia_1000hz": ratio_mixto_variacion_energia_1000hz,
        "score_variacion_frecuencial_200ms": score_variacion_frecuencial_200ms,
        "score_mixto_variacion_energia_1000hz": score_mixto_variacion_energia_1000hz,

        "f_min_primera_100ms_hz": float(e_primera["f_min"]),
        "f_max_primera_100ms_hz": float(e_primera["f_max"]),
        "f_min_segunda_100ms_hz": float(e_segunda["f_min"]),
        "f_max_segunda_100ms_hz": float(e_segunda["f_max"]),
        "ratio_dinamico_fmax_segunda_fmin_primera": ratio_dinamico_fmax_segunda_fmin_primera,
        "diferencia_fmax_segunda_menos_fmin_primera_hz": diferencia_fmax_segunda_menos_fmin_primera,
        "velocidad_dinamica_hz_s": velocidad_dinamica_hz_s,

        "salto_max_fmax_hz": salto_max_fmax_hz,
        "salto_max_fmin_hz": salto_max_fmin_hz,
        "variacion_total_fmax_hz": variacion_total_fmax_hz,
        "variacion_total_fmin_hz": variacion_total_fmin_hz,
        "n_bandas_activas_200ms": int(len(idx_bandas_activas))
    }


def calcular_caida_post_pef_logmel(
    mel,
    logmel,
    t_mel,
    intervalo_s=0.80,
    etiqueta_logmel="DENOISED"
):
    """Compara potencia Mel relativa tras su máximo espectral.

    E[m]=sum_b(M[m,b])/max_m(sum_b(M[m,b])) y media móvil de 5 tramas.
    t_PEFe es el máximo de esa curva, no necesariamente el PEF temporal.
    E_i promedia [t_PEFe+0.05,t_PEFe+0.25); E_f promedia los 200 ms
    finales de intervalo_s=0.80. delta=max(0,1-E_f/(E_i+eps)).
    La actividad por bandas usa L>-65 dB. Si no cabe el intervalo
    completo se devuelve detectado=False, sin activar esta vía terminal.
    """

    mel = np.asarray(mel, dtype=np.float64)
    logmel = np.asarray(logmel, dtype=np.float64)
    t = np.asarray(t_mel, dtype=np.float64)

    if (
        mel.ndim != 2
        or logmel.ndim != 2
        or t.ndim != 1
        or mel.shape[1] != len(t)
        or logmel.shape[1] != len(t)
        or len(t) < 8
    ):
        return {
            "detectado": False,
            "motivo": "sin_matriz_logmel",
            "etiqueta_logmel": etiqueta_logmel
        }

    energia = np.sum(np.maximum(mel, 0.0), axis=0)

    if np.max(energia) <= 0:
        return {
            "detectado": False,
            "motivo": "energia_nula",
            "etiqueta_logmel": etiqueta_logmel
        }

    energia = energia / (np.max(energia) + 1e-12)

    if len(energia) >= 5:
        kernel = np.ones(5, dtype=np.float64) / 5.0
        energia_suave = np.convolve(energia, kernel, mode="same")
    else:
        energia_suave = energia

    idx_pef = int(np.argmax(energia_suave))
    t_pef = float(t[idx_pef])

    inicio_ini = t_pef + 0.05
    inicio_fin = t_pef + 0.25
    final_ini = t_pef + float(intervalo_s) - 0.20
    final_fin = t_pef + float(intervalo_s)

    if final_fin > float(t[-1]) + 1e-9:
        return {
            "detectado": False,
            "motivo": "no_cabe_intervalo_post_pef",
            "etiqueta_logmel": etiqueta_logmel,
            "t_pef_s": t_pef,
            "t_fin_audio_s": float(t[-1]),
            "intervalo_s": float(intervalo_s)
        }

    m_inicio = (t >= inicio_ini) & (t < inicio_fin)
    m_final = (t >= final_ini) & (t < final_fin)

    if not np.any(m_inicio) or not np.any(m_final):
        return {
            "detectado": False,
            "motivo": "ventanas_sin_frames",
            "etiqueta_logmel": etiqueta_logmel,
            "t_pef_s": t_pef,
            "intervalo_s": float(intervalo_s)
        }

    e_inicio = float(np.mean(energia_suave[m_inicio]))
    e_final = float(np.mean(energia_suave[m_final]))
    e_min_final = float(np.min(energia_suave[m_final]))

    caida_lineal = float(e_inicio - e_final)
    ratio_final_inicio = float(e_final / (e_inicio + 1e-12)) if e_inicio > 0 else 1.0
    caida_relativa = float(max(0.0, 1.0 - ratio_final_inicio))
    caida_db = float(10.0 * np.log10((e_inicio + 1e-12) / (e_final + 1e-12)))

    activo = logmel > -65.0
    activo_inicio = float(np.mean(activo[:, m_inicio]))
    activo_final = float(np.mean(activo[:, m_final]))
    caida_activos = float(activo_inicio - activo_final)

    m_intervalo = (t >= inicio_ini) & (t < final_fin)

    if np.sum(m_intervalo) >= 2:
        pendiente_post_pef = float(
            np.polyfit(t[m_intervalo] - t_pef, energia_suave[m_intervalo], 1)[0]
        )
    else:
        pendiente_post_pef = 0.0

    return {
        "detectado": True,
        "motivo": "caida_post_pef_logmel",
        "etiqueta_logmel": etiqueta_logmel,
        "t_pef_s": t_pef,
        "intervalo_s": float(intervalo_s),
        "e_inicio": e_inicio,
        "e_final": e_final,
        "e_min_final": e_min_final,
        "caida_lineal": caida_lineal,
        "caida_relativa": caida_relativa,
        "ratio_final_inicio": ratio_final_inicio,
        "caida_db": caida_db,
        "activo_inicio": activo_inicio,
        "activo_final": activo_final,
        "caida_activos": caida_activos,
        "pendiente_post_pef": pendiente_post_pef,
        "inicio_ini_s": float(inicio_ini),
        "inicio_fin_s": float(inicio_fin),
        "final_ini_s": float(final_ini),
        "final_fin_s": float(final_fin)
    }


def imprimir_SOLO_ratio_baremo_freq_db(info):
    """Muestra mediante builtins.print la puntuación frecuencial auxiliar S_T."""

    etiqueta = info.get("etiqueta_logmel", "")

    if not info.get("detectado", False):
        builtins.print(f"{etiqueta}: No detectado ({info.get('motivo', 'desconocido')})")
        return

    builtins.print(
        f"{etiqueta}: Ratio baremo freq*dBrel 200 ms MÁS PESO = "
        f"{info['ratio_baremo_freq_db_200_mas_peso']:.6f}"
    )


def evaluar_terminacion_precoz_brusca_logmel(g):
    """Combina condiciones booleanas de corte y caída post-PEF.

    D_T=Q_ext or Q_brus or Q_esp or Q_post. Cada Q requiere todas
    las comparaciones de su bloque. Q_post exige delta>=0.98 y
    E_f<=0.005; sus variantes de bandas son subconjuntos de esa regla.
    La pendiente global conserva su signo: p_L>=150/300 no es abs(p_L).
    score resume intensidades y no sustituye la decisión booleana D_T.
    """

    info = g.get("pendiente_logmel_final", {}) or {}

    def num(campo, defecto=0.0):
        """Lee un indicador finito o devuelve el valor de respaldo."""
        try:
            valor = float(info.get(campo, defecto))
            if np.isfinite(valor):
                return valor
        except Exception:
            pass
        return float(defecto)

    caida_db = num("caida_maxima_db")
    caida_lineal = num("caida_maxima_frame")
    energia_pre = num("energia_pre_corte")
    energia_post = num("energia_post_corte", 1.0)
    colapso = num("colapso_activos_frame")
    spread = num("spread_tiempos_caida")
    tiempo_colapso = num("tiempo_colapso_bandas")
    # Se conserva la pendiente firmada; una caída negativa no cumple >=150 o >=300.
    pendiente_global = num("pendiente_global_bajas")

    post = g.get("caida_post_pef_logmel_den", {}) or {}

    def num_post(campo, defecto=0.0):
        """Lee un indicador post-PEF finito o devuelve el valor de respaldo."""
        try:
            valor = float(post.get(campo, defecto))
            if np.isfinite(valor):
                return valor
        except Exception:
            pass
        return float(defecto)

    post_detectado = bool(post.get("detectado", False))
    post_e_inicio = num_post("e_inicio")
    post_e_final = num_post("e_final", 1.0)
    post_caida_relativa = num_post("caida_relativa")
    post_caida_db = num_post("caida_db")
    post_activo_final = num_post("activo_final", 1.0)
    post_caida_activos = num_post("caida_activos")
    post_t_pef = num_post("t_pef_s")
    post_intervalo = num_post("intervalo_s", 0.80)

    corte_brusco_extremo = (
        caida_db >= 15.0
        and caida_lineal >= 0.50
        and energia_pre >= 0.50
        and energia_post <= 0.04
        and colapso >= 0.85
        and tiempo_colapso <= 0.02
        and spread <= 0.03
    )

    corte_brusco_brusco = (
        caida_db >= 10.0
        and caida_lineal >= 0.25
        and energia_pre >= 0.25
        and energia_post <= 0.04
        and colapso >= 0.70
        and tiempo_colapso <= 0.02
        and spread <= 0.03
        and pendiente_global >= 150.0
    )

    corte_brusco_espectral_fuerte = (
        caida_db >= 10.0
        and caida_lineal >= 0.18
        and energia_pre >= 0.20
        and energia_post <= 0.025
        and colapso >= 0.70
        and tiempo_colapso <= 0.02
        and spread <= 0.03
        and pendiente_global >= 300.0
    )

    post_pef_apagado_extremo = (
        post_detectado
        and post_caida_relativa >= 0.98
        and post_e_final <= 0.005
        and post_activo_final <= 0.20
    )

    post_pef_apagado_con_colapso = (
        post_detectado
        and post_caida_relativa >= 0.98
        and post_e_final <= 0.005
        and post_activo_final <= 0.25
        and post_caida_activos >= 0.70
    )

    post_pef_caida_energia_fuerte = (
        post_detectado
        and post_caida_relativa >= 0.98
        and post_e_final <= 0.005
    )

    apagado_final_precoz = bool(
        corte_brusco_extremo
        or corte_brusco_brusco
        or corte_brusco_espectral_fuerte
    )

    post_pef_precoz = bool(
        post_pef_apagado_extremo
        or post_pef_apagado_con_colapso
        or post_pef_caida_energia_fuerte
    )

    if post_pef_apagado_extremo:
        regla = "post_pef_apagado_extremo"
    elif post_pef_apagado_con_colapso:
        regla = "post_pef_apagado_con_colapso"
    elif post_pef_caida_energia_fuerte:
        regla = "post_pef_caida_energia_fuerte"
    elif corte_brusco_extremo:
        regla = "apagado_final_extremo"
    elif corte_brusco_brusco:
        regla = "apagado_final_brusco"
    elif corte_brusco_espectral_fuerte:
        regla = "apagado_final_espectral_fuerte"
    else:
        regla = "sin_corte_brusco_precoz"

    apagado_relativo = 0.0
    if energia_pre > 1e-9:
        apagado_relativo = max(0.0, 1.0 - (energia_post / energia_pre))

    score = max(
        min(caida_db / 20.0, 1.0),
        min(caida_lineal / 0.70, 1.0),
        min(colapso, 1.0),
        min(apagado_relativo, 1.0),
        min(post_caida_relativa, 1.0),
    )

    return {
        "terminacion_precoz": bool(post_pef_precoz or apagado_final_precoz),
        "corte_brusco_extremo": bool(corte_brusco_extremo),
        "corte_brusco_brusco": bool(corte_brusco_brusco),
        "corte_brusco_espectral_fuerte": bool(corte_brusco_espectral_fuerte),
        "apagado_final_precoz": bool(apagado_final_precoz),
        "post_pef_precoz": bool(post_pef_precoz),
        "post_pef_apagado_extremo": bool(post_pef_apagado_extremo),
        "post_pef_apagado_con_colapso": bool(post_pef_apagado_con_colapso),
        "post_pef_caida_energia_fuerte": bool(post_pef_caida_energia_fuerte),
        "regla": regla,
        "score": float(score),
        "caida_db": float(caida_db),
        "caida_lineal": float(caida_lineal),
        "energia_pre": float(energia_pre),
        "energia_post": float(energia_post),
        "colapso": float(colapso),
        "spread": float(spread),
        "tiempo_colapso": float(tiempo_colapso),
        "pendiente_global": float(pendiente_global),
        "post_e_inicio": float(post_e_inicio),
        "post_e_final": float(post_e_final),
        "post_caida_relativa": float(post_caida_relativa),
        "post_caida_db": float(post_caida_db),
        "post_activo_final": float(post_activo_final),
        "post_caida_activos": float(post_caida_activos),
        "post_t_pef": float(post_t_pef),
        "post_intervalo": float(post_intervalo),
    }


def evaluar_esfuerzo_multiple_apagon_logmel(g):
    """Busca descenso y recuperación en bandas Mel de al menos 350 Hz.

    Promedia q=(L+80)/80 limitado a [0,1] y suaviza con gaussiana
    de 60 ms. Exige valle<=0.20 durante al menos 50 ms desde t>=0.35 s;
    busca pre/post>=0.32 en 0.45 s a cada lado y cambios>=0.10.
    Esta vía puede activar esfuerzo múltiple aunque haya un solo PEF
    temporal; n_pef en la conclusión es entonces una salida inferida.
    """

    logmel = np.asarray(g.get("logmel_den", np.zeros((0, 0))), dtype=np.float64)
    t = np.asarray(g.get("t_mel_den", np.array([])), dtype=np.float64)
    freqs = np.asarray(g.get("freqs_mel_den", np.array([])), dtype=np.float64)

    if logmel.ndim != 2 or logmel.size == 0 or len(t) < 3:
        return {"detectado": False, "motivo": "sin_logmel"}

    if len(freqs) != logmel.shape[0]:
        return {"detectado": False, "motivo": "frecuencias_no_validas"}

    n = min(logmel.shape[1], len(t))
    logmel = logmel[:, :n]
    t = t[:n]

    bandas_medias_altas = freqs >= 350.0

    if not np.any(bandas_medias_altas):
        return {"detectado": False, "motivo": "sin_bandas_medias_altas"}

    energia_altas = np.mean(
        np.clip((logmel[bandas_medias_altas, :] + 80.0) / 80.0, 0.0, 1.0),
        axis=0
    )

    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.032
    sigma = max(0.5, 0.060 / max(dt, 1e-6))
    energia_altas = ndimage.gaussian_filter1d(energia_altas, sigma=sigma, mode="nearest")

    umbral_apagon = 0.20
    umbral_recuperacion = 0.32
    duracion_min_apagon_s = 0.05
    ventana_busqueda_s = 0.45
    t_min_apagon_s = 0.35
    cambio_minimo = 0.10

    debajo = energia_altas <= umbral_apagon
    mejor = None
    i = 0

    while i < len(energia_altas):
        if not debajo[i]:
            i += 1
            continue

        j = i
        while j < len(energia_altas) and debajo[j]:
            j += 1

        duracion_apagon = float(t[j - 1] - t[i]) if j > i else 0.0

        if float(t[i]) >= t_min_apagon_s and duracion_apagon >= duracion_min_apagon_s:
            pre0 = max(0, int(np.searchsorted(t, t[i] - ventana_busqueda_s)))
            pre1 = i
            post0 = j
            post1 = min(len(energia_altas), int(np.searchsorted(t, t[j - 1] + ventana_busqueda_s)))

            if pre1 > pre0 and post1 > post0:
                energia_pre = float(np.max(energia_altas[pre0:pre1]))
                energia_valle = float(np.min(energia_altas[i:j]))
                energia_post = float(np.max(energia_altas[post0:post1]))

                apagon_con_recuperacion = (
                    energia_pre >= umbral_recuperacion
                    and energia_post >= umbral_recuperacion
                    and (energia_pre - energia_valle) >= cambio_minimo
                    and (energia_post - energia_valle) >= cambio_minimo
                )

                if apagon_con_recuperacion:
                    score = min(energia_pre, energia_post) - energia_valle
                    candidato = {
                        "score": float(score),
                        "t_inicio": float(t[i]),
                        "t_fin": float(t[j - 1]),
                        "duracion": float(duracion_apagon),
                        "energia_pre": float(energia_pre),
                        "energia_valle": float(energia_valle),
                        "energia_post": float(energia_post),
                    }

                    if mejor is None or candidato["score"] > mejor["score"]:
                        mejor = candidato

        i = j + 1

    if mejor is None:
        return {"detectado": False, "motivo": "sin_apagon_con_recuperacion"}

    mejor["detectado"] = True
    mejor["motivo"] = "apagon_logmel_medias_altas_con_recuperacion"
    return mejor


def calcular_conclusion_final_prueba(d, g, umbral_pef_retrasado_s=0.150, umbral_terminacion_precoz=0.60):
    """Asigna una clase según la prioridad técnica del prototipo.

    Orden: tos; esfuerzo múltiple temporal o espectral; PEF retrasado
    (>0.150 s desde el cruce inicial al 10 %); terminación precoz;
    ausencia de criterios. Retorna al cumplirse la primera condición.
    El argumento umbral_terminacion_precoz se conserva por compatibilidad,
    pero no se usa como umbral activo. La salida válida significa que
    estas reglas no detectan un evento, no aceptabilidad clínica probada.
    """

    eventos = d.get("eventos_picos", [])
    mascara_tos = d.get("mascara_tos", None)


    hay_tos_mascara = False

    try:
        hay_tos_mascara = mascara_tos is not None and bool(np.any(mascara_tos))
    except Exception:
        hay_tos_mascara = False

    eventos_tos = [
        ev for ev in eventos
        if ev.get("tipo", "") in ["tos_segura", "tos_por_duracion"]
    ]

    if hay_tos_mascara or len(eventos_tos) > 0:
        return {
            "valida": False,
            "motivo": "EVENTO INVÁLIDO POR TOS",
            "codigo": "tos",
            "detalle": f"Se detecta máscara/evento de tos. Nº eventos tos = {len(eventos_tos)}",
            "n_tos": int(len(eventos_tos)),
            "n_pef": int(len([ev for ev in eventos if ev.get('tipo', '') == 'PEF'])),
            "pef_retrasado": False,
            "tiempo_pef_desde_inicio_s": None,
            "score_terminacion_precoz": None,
            "fuente_score_terminacion": None
        }


    eventos_pef_temporal = [
        ev for ev in eventos
        if ev.get("tipo", "") == "PEF"
    ]

    eventos_pef_logmel = g.get("eventos_pef_logmel_den", [])
    eventos_pef_logmel = [
        ev for ev in eventos_pef_logmel
        if ev.get("tipo", "") == "PEF"
    ]

    n_pef_temporal = len(eventos_pef_temporal)
    n_pef_logmel = len(eventos_pef_logmel)


    n_pef_multiple = n_pef_temporal
    n_pef = n_pef_temporal if n_pef_temporal > 0 else n_pef_logmel

    # También puede activar la clase el descenso espectral con recuperación.
    multiple_apagon_logmel = evaluar_esfuerzo_multiple_apagon_logmel(g)

    if n_pef_multiple >= 2 or multiple_apagon_logmel.get("detectado", False):
        if n_pef_multiple >= 2:
            detalle_multiple = f"Se detectan {n_pef_multiple} PEF."
            n_pef_final = int(n_pef_multiple)
        else:
            detalle_multiple = (
                "Se detecta apagón espectral Log-Mel en bandas medias-altas "
                "con recuperación posterior: "
                f"{multiple_apagon_logmel['t_inicio']:.3f}-{multiple_apagon_logmel['t_fin']:.3f} s, "
                f"energía pre={multiple_apagon_logmel['energia_pre']:.3f}, "
                f"valle={multiple_apagon_logmel['energia_valle']:.3f}, "
                f"post={multiple_apagon_logmel['energia_post']:.3f}."
            )
            # Recuento inferido por la vía espectral; no son dos PEF temporales medidos.
            n_pef_final = int(max(n_pef_multiple, 2))

        return {
            "valida": False,
            "motivo": "EVENTO INVÁLIDO POR ESFUERZO MÚLTIPLE",
            "codigo": "esfuerzo_multiple",
            "detalle": detalle_multiple,
            "n_tos": 0,
            "n_pef": int(n_pef_final),
            "pef_retrasado": False,
            "tiempo_pef_desde_inicio_s": None,
            "score_terminacion_precoz": None,
            "fuente_score_terminacion": None
        }


    precoz_brusca = evaluar_terminacion_precoz_brusca_logmel(g)

    if precoz_brusca.get("regla", "").startswith("post_pef"):
        detalle_precoz = (
            f"Caída post-PEF extrema ({precoz_brusca['regla']}): "
            f"PEF espectral={precoz_brusca['post_t_pef']:.3f} s, "
            f"intervalo={precoz_brusca['post_intervalo']:.2f} s, "
            f"energía inicial={precoz_brusca['post_e_inicio']:.3f}, "
            f"energía final={precoz_brusca['post_e_final']:.4f}, "
            f"caída={precoz_brusca['post_caida_relativa']:.3f}, "
            f"bandas activas finales={precoz_brusca['post_activo_final']:.3f}."
        )
    else:
        detalle_precoz = (
            f"Corte brusco final ({precoz_brusca['regla']}): "
            f"caída={precoz_brusca['caida_db']:.2f} dB, "
            f"caída lineal={precoz_brusca['caida_lineal']:.3f}, "
            f"energía pre={precoz_brusca['energia_pre']:.3f}, "
            f"energía post={precoz_brusca['energia_post']:.3f}, "
            f"colapso bandas={precoz_brusca['colapso']:.3f}."
        )


    pef_evento = None

    if len(eventos_pef_temporal) >= 1:
        pef_evento = eventos_pef_temporal[0]
    elif len(eventos_pef_logmel) >= 1:
        pef_evento = eventos_pef_logmel[0]

    tiempo_pef_desde_inicio_s = None
    pef_retrasado = False

    if pef_evento is not None:
        try:
            t_inicio = float(pef_evento.get("inicio", 0.0))
            t_pico = float(pef_evento.get("pico", 0.0))
            tiempo_pef_desde_inicio_s = max(0.0, t_pico - t_inicio)
            pef_retrasado = tiempo_pef_desde_inicio_s > float(umbral_pef_retrasado_s)
        except Exception:
            tiempo_pef_desde_inicio_s = None
            pef_retrasado = False

    # El retardo usa el evento detectado, no la posición corregida del marcador visual.
    if pef_retrasado:
        return {
            "valida": False,
            "motivo": "EVENTO INVÁLIDO POR ESFUERZO INSUFICIENTE",
            "codigo": "esfuerzo_insuficiente",
            "detalle": (
                f"PEF retrasado: tarda {tiempo_pef_desde_inicio_s:.3f} s "
                f"desde el inicio del soplido. Umbral = {umbral_pef_retrasado_s:.3f} s."
            ),
            "n_tos": 0,
            "n_pef": int(n_pef),
            "pef_retrasado": True,
            "tiempo_pef_desde_inicio_s": float(tiempo_pef_desde_inicio_s),
            "score_terminacion_precoz": None,
            "fuente_score_terminacion": None
        }


    info_raw = g.get("ratio_variacion_frecuencial_200ms_logmel_raw", {})
    info_den = g.get("ratio_variacion_frecuencial_200ms_logmel_den", {})

    score_terminacion = None
    fuente_score = None

    if info_raw.get("detectado", False) and "ratio_baremo_freq_db_200_mas_peso" in info_raw:
        score_terminacion = float(info_raw.get("ratio_baremo_freq_db_200_mas_peso", 0.0))
        fuente_score = "ORIGINAL"
    elif info_den.get("detectado", False) and "ratio_baremo_freq_db_200_mas_peso" in info_den:
        score_terminacion = float(info_den.get("ratio_baremo_freq_db_200_mas_peso", 0.0))
        fuente_score = "DENOISED"
    else:
        score_terminacion = 0.0
        fuente_score = "NO_DETECTADO"

    # La decisión usa condiciones booleanas, no score_terminacion >= 0.60.
    if precoz_brusca.get("terminacion_precoz", False):
        detalle_precoz_final = detalle_precoz + (
            f" Score frecuencial auxiliar={score_terminacion:.3f} ({fuente_score})."
        )

        return {
            "valida": False,
            "motivo": "EVENTO INVÁLIDO POR TERMINACIÓN PRECOZ",
            "codigo": "terminacion_precoz",
            "detalle": detalle_precoz_final,
            "n_tos": 0,
            "n_pef": int(n_pef),
            "pef_retrasado": False,
            "tiempo_pef_desde_inicio_s": tiempo_pef_desde_inicio_s,
            "score_terminacion_precoz": float(precoz_brusca["score"]),
            "fuente_score_terminacion": "LOGMEL_CORTE_BRUSCO"
        }


    return {
        "valida": True,
        "motivo": "PRUEBA TOTALMENTE VÁLIDA",
        "codigo": "valida",
        "detalle": "No se detecta tos, esfuerzo múltiple, PEF retrasado ni terminación precoz.",
        "n_tos": 0,
        "n_pef": int(n_pef),
        "pef_retrasado": False,
        "tiempo_pef_desde_inicio_s": tiempo_pef_desde_inicio_s,
        "score_terminacion_precoz": float(max(score_terminacion, precoz_brusca.get("score", 0.0))),
        "fuente_score_terminacion": f"{fuente_score}+LOGMEL_CORTE_BRUSCO"
    }


def imprimir_conclusion_final_prueba(nombre, conclusion):
    """Muestra la clase y su motivo prioritario mediante builtins.print."""

    builtins.print("\nCONCLUSIÓN FINAL")
    builtins.print("-" * 70)
    builtins.print(f"{nombre}: {conclusion['motivo']}")
    builtins.print(f"Detalle: {conclusion['detalle']}")
    builtins.print("-" * 70)

def preparar_graficas_fuente(d):
    """Prepara representaciones e indicadores para cada archivo por separado.

    Conserva las ramas original y procesada y referencias compartidas.
    El análisis de corte brusco consulta la original; la caída post-PEF
    efectiva consulta la procesada. Las rampas iniciales y el suelo terminal
    se aplican antes de algunos indicadores espectrales: pueden influir
    en ellos aunque no cambien el WAV ni los eventos temporales de d.
    """
    fs_raw, fs_den, y_raw, y_den = d["fs_raw"], d["fs"], d["original_raw_graficas"], d["denoised"]

    t_raw = np.arange(len(y_raw)) / fs_raw
    t_den = np.arange(len(y_den)) / fs_den


    y_raw = forzar_arranque_en_suelo_sin_blanco(t_raw, y_raw, duracion_rampa_s=0.035)
    y_den = forzar_arranque_en_suelo_sin_blanco(t_den, y_den, duracion_rampa_s=0.035)


    y_raw_suavizada_25ms = suavizar_senal_cruda_25ms(y_raw, fs_raw)

    env_raw = calcular_envolvente_temporal(y_raw, fs_raw, mascara_tos=None, normalizar_entrada=False)
    env_den = calcular_envolvente_temporal(y_den, fs_den, mascara_tos=d.get("mascara_tos"), normalizar_entrada=False)


    env_raw = forzar_envolvente_inicio_suelo_sin_blanco(t_raw, env_raw, duracion_rampa_s=0.040)
    env_den = forzar_envolvente_inicio_suelo_sin_blanco(t_den, env_den, duracion_rampa_s=0.040)

    t_env_stft_raw, env_stft_raw = calcular_envolvente_stft_en_tiempo(y_raw, fs_raw, normalizar_entrada=False)
    t_env_stft_den, env_stft_den = calcular_envolvente_stft_en_tiempo(y_den, fs_den, normalizar_entrada=False)

    env_stft_raw = forzar_envolvente_inicio_suelo_sin_blanco(
        t_env_stft_raw,
        env_stft_raw,
        duracion_rampa_s=0.040
    )

    env_stft_den = forzar_envolvente_inicio_suelo_sin_blanco(
        t_env_stft_den,
        env_stft_den,
        duracion_rampa_s=0.040
    )

    f_raw, t_stft_raw, power_raw, _ = calcular_stft_power(
    y_raw, fs_raw, normalizar_entrada=False
    )

    ref_power_max = np.max(power_raw)

    f_raw, t_stft_raw, _, stft_raw_db = calcular_stft_power(
    y_raw, fs_raw, normalizar_entrada=False, ref_power_max=ref_power_max
    )

    f_den, t_stft_den, _, stft_den_db = calcular_stft_power(
    y_den, fs_den, normalizar_entrada=False, ref_power_max=ref_power_max
    )


    t_mel_raw, mel_raw, logmel_raw, ref_mel_max, t_mel_raw_edges = calcular_mel_logmel(
        y_raw,
        fs_raw,
        ref_mel_max=None,
        normalizar_entrada=True
    )

    t_mel_den, mel_den, logmel_den, _, t_mel_den_edges = calcular_mel_logmel(
        y_den,
        fs_den,
        ref_mel_max=ref_mel_max,
        normalizar_entrada=True
    )


    logmel_den = quitar_repunte_final_logmel_por_actividad(
        logmel_den,
        t_mel_den,
        y_den,
        fs_den,
        umbral_rms=0.025,
        ventana_rms_s=0.025,
        zona_final_s=0.50
    )


    freqs_mel_raw = frecuencias_centro_bandas_mel(fs_raw, n_mels)
    freqs_mel_den = frecuencias_centro_bandas_mel(fs_den, n_mels)


    def _t0(t):
        """Traslada el primer tiempo al origen de la representación."""
        t = np.asarray(t, dtype=np.float64)
        if len(t) == 0:
            return t
        return t - float(t[0])

    t_env_stft_raw = _t0(t_env_stft_raw)
    t_env_stft_den = _t0(t_env_stft_den)
    t_stft_raw = _t0(t_stft_raw)
    t_stft_den = _t0(t_stft_den)
    t_mel_raw = _t0(t_mel_raw)
    t_mel_den = _t0(t_mel_den)


    t_env_logmel_den, env_logmel_den = calcular_envolvente_logmel_denoised(
        logmel_den,
        t_mel_den,
        fs_den,
        mascara_tos=d.get("mascara_tos"),
        normalizar_salida=True
    )

    env_logmel_den = forzar_envolvente_inicio_suelo_sin_blanco(
        t_env_logmel_den,
        env_logmel_den,
        duracion_rampa_s=0.040
    )

    eventos_pef_logmel_den = detectar_pefs_envolvente_logmel_denoised(
        t_env_logmel_den,
        env_logmel_den
    )

    pendiente_logmel_final = calcular_pendiente_logmel_antes_energia_nula(
    y_raw,
    fs_raw,
    duracion_ventana=0.40,
    umbral_energia_nula=0.08,
    min_frames_nulos=2
    )


    score_terminacion = calcular_score_terminacion_precoz(
        d["denoising_entrada"], d["fs"], mascara_tos=d.get("mascara_tos")
    )


    energia_abs_final_raw = calcular_promedio_abs_ultimo_cuarto(
        y_raw,
        fs_raw,
        duracion_s=0.25
    )

    energia_abs_final_den = calcular_promedio_abs_ultimo_cuarto(
        y_den,
        fs_den,
        duracion_s=0.25
    )


    energia_abs_desde_pef_raw = calcular_promedio_abs_desde_primer_pef(
        y_raw,
        fs_raw,
        d.get("eventos_picos", [])
    )

    energia_abs_desde_pef_den = calcular_promedio_abs_desde_primer_pef(
        y_den,
        fs_den,
        d.get("eventos_picos", [])
    )

    ratios_post_pef_raw = calcular_ratios_post_pef_optimizados(
        y_raw,
        fs_raw,
        d.get("eventos_picos", [])
    )

    ratios_post_pef_den = calcular_ratios_post_pef_optimizados(
        y_den,
        fs_den,
        d.get("eventos_picos", [])
    )

    ratio_final_050s_logmel_negro_den = calcular_ratio_final_050s_antes_logmel_negro(
        logmel_den,
        t_mel_den,
        eventos_picos=eventos_pef_logmel_den,
        duracion_ventana_s=0.50,
        umbral_negro_db=-79.9,
        min_frames_negros=1,
        t_logmel_den_edges=t_mel_den_edges
    )

    variacion_bandas_070s_final_logmel_den = calcular_variacion_bandas_070s_antes_final_logmel_denoised(
        logmel_den,
        t_mel_den,
        freqs_mel_den=freqs_mel_den,
        eventos_picos=eventos_pef_logmel_den,
        t_logmel_den_edges=t_mel_den_edges,
        duracion_ventana_s=0.70,
        umbral_negro_db=-79.9,
        umbral_banda_activa_db=-70.0,
        peso_altas=2.5
    )

    ratios_ventanas_100ms_a_1s_logmel_den = calcular_ratios_ventanas_100ms_a_1s_logmel_denoised(
        logmel_den,
        t_mel_den,
        freqs_mel_den=freqs_mel_den,
        eventos_picos=eventos_pef_logmel_den,
        t_logmel_den_edges=t_mel_den_edges,
        umbral_negro_db=-79.9,
        umbral_banda_activa_db=-70.0
    )

    ratio_100ms_ruido_basal_logmel_den = calcular_ratio_100ms_antes_ruido_basal_logmel_denoised(
        logmel_den,
        t_mel_den,
        freqs_mel_den=freqs_mel_den,
        eventos_picos=eventos_pef_logmel_den,
        t_logmel_den_edges=t_mel_den_edges,
        duracion_ventana_s=0.100,
        duracion_basal_s=0.100,
        suelo_db=-80.0,
        techo_db=0.0,
        tolerancia_basal_visual=0.035
    )

    ratio_variacion_frecuencial_200ms_logmel_den = calcular_ratio_variacion_frecuencial_200ms_logmel_denoised(
        logmel_den,
        t_mel_den,
        freqs_mel_den=freqs_mel_den,
        eventos_picos=eventos_pef_logmel_den,
        t_logmel_den_edges=t_mel_den_edges,
        duracion_ventana_s=0.200,
        suelo_db=-80.0,
        techo_db=0.0,
        tolerancia_basal_visual=0.035,
        etiqueta_logmel="DENOISED"
    )

    referencia_denoised_200ms = (
        ratio_variacion_frecuencial_200ms_logmel_den.get("t_referencia_s")
        if ratio_variacion_frecuencial_200ms_logmel_den.get("detectado", False)
        else None
    )

    ratio_variacion_frecuencial_200ms_logmel_raw = calcular_ratio_variacion_frecuencial_200ms_logmel_denoised(
        logmel_raw,
        t_mel_raw,
        freqs_mel_den=freqs_mel_raw,
        eventos_picos=eventos_pef_logmel_den,
        t_logmel_den_edges=t_mel_raw_edges,
        duracion_ventana_s=0.200,
        suelo_db=-80.0,
        techo_db=0.0,
        tolerancia_basal_visual=0.035,
        t_referencia_fija_s=referencia_denoised_200ms,
        etiqueta_logmel="ORIGINAL"
    )

    caida_post_pef_logmel_den = calcular_caida_post_pef_logmel(
        mel_den,
        logmel_den,
        t_mel_den,
        intervalo_s=0.80,
        etiqueta_logmel="DENOISED"
    )

    caida_post_pef_logmel_raw = calcular_caida_post_pef_logmel(
        mel_raw,
        logmel_raw,
        t_mel_raw,
        intervalo_s=0.80,
        etiqueta_logmel="ORIGINAL"
    )

    return {
        "t_raw": t_raw, "t_den": t_den, "y_raw": y_raw, "y_den": y_den,
        "y_raw_suavizada_25ms": y_raw_suavizada_25ms,
        "env_raw": env_raw, "env_den": env_den,
        "t_env_stft_raw": t_env_stft_raw, "env_stft_raw": env_stft_raw,
        "t_env_stft_den": t_env_stft_den, "env_stft_den": env_stft_den,
        "t_stft_raw": t_stft_raw, "t_stft_den": t_stft_den,
        "stft_raw_db": stft_raw_db, "stft_den_db": stft_den_db,
        "t_mel_raw": t_mel_raw, "t_mel_den": t_mel_den,
        "t_mel_raw_edges": t_mel_raw_edges, "t_mel_den_edges": t_mel_den_edges,
        "t_env_logmel_den": t_env_logmel_den, "env_logmel_den": env_logmel_den,
        "eventos_pef_logmel_den": eventos_pef_logmel_den,
        "freqs_mel_raw": freqs_mel_raw, "freqs_mel_den": freqs_mel_den,
        "mel_raw": mel_raw, "mel_den": mel_den,
        "logmel_raw": logmel_raw, "logmel_den": logmel_den,
        "score_terminacion": score_terminacion,
        "pendiente_logmel_final": pendiente_logmel_final,
        "energia_abs_final_raw": energia_abs_final_raw,
        "energia_abs_final_den": energia_abs_final_den,
        "energia_abs_desde_pef_raw": energia_abs_desde_pef_raw,
        "energia_abs_desde_pef_den": energia_abs_desde_pef_den,
        "ratios_post_pef_raw": ratios_post_pef_raw,
        "ratios_post_pef_den": ratios_post_pef_den,
        "ratio_final_050s_logmel_negro_den": ratio_final_050s_logmel_negro_den,
        "variacion_bandas_070s_final_logmel_den": variacion_bandas_070s_final_logmel_den,
        "ratios_ventanas_100ms_a_1s_logmel_den": ratios_ventanas_100ms_a_1s_logmel_den,
        "ratio_100ms_ruido_basal_logmel_den": ratio_100ms_ruido_basal_logmel_den,
        "ratio_variacion_frecuencial_200ms_logmel_den": ratio_variacion_frecuencial_200ms_logmel_den,
        "ratio_variacion_frecuencial_200ms_logmel_raw": ratio_variacion_frecuencial_200ms_logmel_raw,
        "caida_post_pef_logmel_den": caida_post_pef_logmel_den,
        "caida_post_pef_logmel_raw": caida_post_pef_logmel_raw,
    }

def anadir_botones_audio(ax, d, nombre):
    """Añade controles de escucha de los WAV original y procesado."""
    botones = []

    ax_b_ori = ax.inset_axes([0.015, 0.055, 0.17, 0.23])
    b_ori = Button(ax_b_ori, "Play orig")
    ruta_ori = guardar_temp_wav(d["original_raw_escucha"], d["fs_raw"], f"{nombre}_orig")
    b_ori.on_clicked(lambda e, r=ruta_ori: reproducir_audio(r))
    botones.append(b_ori)

    ax_b_den = ax.inset_axes([0.200, 0.055, 0.17, 0.23])
    b_den = Button(ax_b_den, "Play den")
    ruta_den = guardar_temp_wav(d["denoised"], d["fs"], f"{nombre}_den")
    b_den.on_clicked(lambda e, r=ruta_den: reproducir_audio(r))
    botones.append(b_den)

    for b in botones: b.label.set_fontsize(7)
    return botones


def forzar_arranque_en_suelo_sin_blanco(t, y, duracion_rampa_s=0.035):
    """Aplica una rampa visual inicial sin desplazar el eje temporal.

    Actúa sobre una copia, sin cambiar el WAV exportado. Su resultado
    alimenta representaciones e indicadores en preparar_graficas_fuente;
    por ello no debe considerarse completamente ajena a la clasificación.
    """

    y = np.asarray(y, dtype=np.float64).copy()
    t = np.asarray(t, dtype=np.float64)

    if len(y) == 0 or len(t) != len(y):
        return y

    if len(t) > 1:
        dt = float(np.median(np.diff(t)))
        if dt <= 0:
            dt = 1.0
    else:
        dt = 1.0

    n_rampa = max(1, int(duracion_rampa_s / dt))
    n_rampa = min(n_rampa, len(y))

    if n_rampa <= 1:
        y[0] = 0.0
        return y


    y0 = y[0]
    y[:n_rampa] = y[:n_rampa] - y0


    rampa = np.linspace(0.0, 1.0, n_rampa)
    y[:n_rampa] = y[:n_rampa] * rampa

    return y


def forzar_envolvente_inicio_suelo_sin_blanco(t, env, duracion_rampa_s=0.040):
    """Aplica una rampa inicial; e_LM modificada se usa también para PEF auxiliares."""

    env = np.asarray(env, dtype=np.float64).copy()
    t = np.asarray(t, dtype=np.float64)

    if len(env) == 0 or len(t) != len(env):
        return env

    if len(t) > 1:
        dt = float(np.median(np.diff(t)))
        if dt <= 0:
            dt = 1.0
    else:
        dt = 1.0

    n_rampa = max(1, int(duracion_rampa_s / dt))
    n_rampa = min(n_rampa, len(env))

    if n_rampa <= 1:
        env[0] = 0.0
        return env

    rampa = np.linspace(0.0, 1.0, n_rampa)
    env[:n_rampa] = env[:n_rampa] * rampa
    env[0] = 0.0

    return env


def _extent(t, y0, y1):
    """Establece los límites de imagen y el origen visual t=0."""
    if len(t) == 0:
        return [0, 1, y0, y1]

    t = np.asarray(t, dtype=np.float64)

    if len(t) == 1:
        return [0, max(float(t[0]), 1e-6), y0, y1]

    duracion = float(t[-1] - t[0])

    if duracion <= 0:
        duracion = float(t[-1])

    duracion = max(duracion, 1e-6)

    return [0.0, duracion, y0, y1]


def ajustar_pico_a_maximo_visible(t_pico, t_ref, y_ref, ventana_s=0.12):
    """Sitúa el marcador en el máximo visual cercano al evento detectado.

    Busca en una ventana de +/-ventana_s sobre la curva representada.
    No modifica el tiempo del evento empleado para medir el retardo
    del PEF acústico ni para determinar la clase.
    """

    if len(t_ref) == 0 or len(y_ref) == 0:
        return None, None

    t_ref = np.asarray(t_ref)
    y_ref = np.asarray(y_ref)

    mascara = (t_ref >= t_pico - ventana_s) & (t_ref <= t_pico + ventana_s)

    if not np.any(mascara):
        idx = int(np.argmin(np.abs(t_ref - t_pico)))
        return float(t_ref[idx]), float(y_ref[idx])

    indices = np.where(mascara)[0]
    idx_local = indices[int(np.argmax(y_ref[indices]))]

    return float(t_ref[idx_local]), float(y_ref[idx_local])


def dibujar_logmel_suave(ax, t_edges, logmel, freqs_centro):
    """Interpola los colores entre bandas y tramas solo para representar Log-Mel.

    La matriz de analisis no se modifica. La imagen se interpola sobre los
    tiempos reales de las celdas, incluida la ultima trama no uniforme.
    Su tamano esta limitado para evitar una malla de triangulos en memoria.
    """
    logmel = np.asarray(logmel, dtype=np.float64)
    t_edges = np.asarray(t_edges, dtype=np.float64)
    if logmel.ndim != 2 or len(t_edges) != logmel.shape[1] + 1:
        raise ValueError("Dimensiones incompatibles para representar Log-Mel.")
    if logmel.size == 0:
        ax.set_facecolor(plt.get_cmap("magma")(0.0))
        ax.grid(False, which="both")
        return None
    if np.any(np.diff(t_edges) <= 0):
        raise ValueError("Los bordes temporales de Log-Mel deben ser crecientes.")

    n_bandas = logmel.shape[0]
    centros_t = (t_edges[:-1] + t_edges[1:]) / 2.0
    ancho_px = max(1, int(np.ceil(ax.get_window_extent().width)))
    n_columnas = min(8192, max(ancho_px, 2 * logmel.shape[1]))
    tiempos_imagen = t_edges[0] + (
        np.arange(n_columnas, dtype=np.float64) + 0.5
    ) * (t_edges[-1] - t_edges[0]) / n_columnas
    colores = np.empty((n_bandas, n_columnas), dtype=np.float32)
    for banda in range(n_bandas):
        colores[banda] = np.interp(tiempos_imagen, centros_t, logmel[banda])

    imagen = ax.imshow(
        colores,
        origin="lower",
        aspect="auto",
        extent=(t_edges[0], t_edges[-1], 0.0, float(n_bandas)),
        interpolation="bilinear",
        interpolation_stage="data",
        cmap="magma",
        vmin=-80,
        vmax=0,
        rasterized=True,
    )
    aplicar_eje_frecuencias_mel(ax, freqs_centro, n_bandas)
    ax.grid(False, which="both")
    return imagen


def dibujar_ventana_unica(datos):
    """Representa uno o dos archivos en columnas con seis filas.

    Filas: formas de onda; RMS original de 25 ms; envolventes;
    Log-Mel original; Log-Mel procesado; envolvente e_LM(t).
    Los ejes temporales comparten límites; la escala de color es relativa.
    STFT y Mel se calculan como pasos intermedios, sin filas propias.
    """

    columnas = list(datos.keys())
    ncols = len(columnas)

    filas = [
        "Señal",
        "RMS cruda 25 ms",
        "Env.",
        "Log-Mel orig.",
        "Log-Mel den.",
        "Env. Log-Mel den."
    ]

    nfilas = 6

    fig_ancho = 16 if ncols == 1 else 22
    fig, axs = plt.subplots(nfilas, ncols, figsize=(fig_ancho, 18), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.915, top=0.90, bottom=0.075, hspace=0.62, wspace=0.18)

    axs = np.asarray(axs)
    if ncols == 1:
        axs = axs.reshape(nfilas, 1)

    ejes_temporales, botones = [], []
    ultimo_im_db = None

    def obtener_tmax_comun(datos_local):
        """Obtiene el límite temporal compartido de las columnas representadas."""
        claves_tiempo = [
            "t_raw", "t_den",
            "t_mel_raw_edges", "t_mel_den_edges",
            "t_env_logmel_den"
        ]

        tmax = 0.0

        for nombre_local in columnas:
            g_local = datos_local[nombre_local]["graficas"]

            for clave in claves_tiempo:
                if clave in g_local and len(g_local[clave]) > 0:
                    arr = np.asarray(g_local[clave], dtype=np.float64)
                    tmax = max(tmax, float(np.max(arr)))

        return max(tmax, 1e-6)

    tmax_comun = obtener_tmax_comun(datos)

    for col, nombre in enumerate(columnas):
        d = datos[nombre]
        g, mascara_tos = d["graficas"], d["mascara_tos"]
        eventos_picos = d.get("eventos_picos", [])

        axs[0, col].set_title(f"{nombre}\n{d['nombre_archivo']}", fontsize=11, pad=12)

        for f in range(nfilas):
            ax = axs[f, col]

            if f == 0:
                ax.plot(g["t_raw"], g["y_raw"], lw=0.45, alpha=0.8, label="Ori")
                ax.plot(g["t_den"], g["y_den"], lw=0.45, alpha=0.9, label="Den")

                if np.any(mascara_tos):
                    ax.fill_between(g["t_den"], -1, 1, where=mascara_tos, alpha=0.11)


                eventos_logmel_plot = g.get("eventos_pef_logmel_den", eventos_picos)

                if len(eventos_logmel_plot) > 0:
                    for ev in eventos_logmel_plot:
                        tipo_ev = ev.get("tipo", "")

                        if tipo_ev == "artefacto":
                            continue

                        t_pico = float(ev.get("pico", 0.0))

                        if tipo_ev == "PEF":
                            ax.scatter(
                                t_pico,
                                0.95,
                                color="red",
                                marker="o",
                                s=30,
                                zorder=8
                            )

                        elif tipo_ev in ["tos_segura", "tos_por_duracion"]:
                            ax.scatter(
                                t_pico,
                                0.85,
                                color="green",
                                marker="x",
                                s=45,
                                linewidths=1.8,
                                zorder=9
                            )

                ax.set_ylim(-1.05, 1.05)
                ax.legend(fontsize=7, loc="upper right")
                botones.extend(anadir_botones_audio(ax, d, nombre))

            elif f == 1:

                ax.plot(
                    g["t_raw"],
                    g["y_raw_suavizada_25ms"],
                    lw=0.75,
                    label="RMS cruda 25 ms"
                )

                ax.set_ylim(0, 1.05)
                ax.legend(fontsize=7, loc="upper right")

            elif f == 2:
                ax.plot(g["t_raw"], g["env_raw"], lw=0.8, label="Ori")
                ax.plot(g["t_den"], g["env_den"], lw=0.8, label="Den")

                if np.any(mascara_tos):
                    ax.fill_between(
                        g["t_den"],
                        0,
                        np.max(g["env_den"]) if len(g["env_den"]) > 0 else 1,
                        where=mascara_tos,
                        alpha=0.09
                    )


                if len(eventos_picos) > 0:
                    for ev in eventos_picos:
                        tipo_ev = ev.get("tipo", "")

                        if tipo_ev == "artefacto":
                            continue

                        t_pico = float(ev.get("pico", 0.0))
                        t_plot, y_plot = ajustar_pico_a_maximo_visible(
                            t_pico,
                            g["t_den"],
                            g["env_den"],
                            ventana_s=0.12
                        )

                        if t_plot is not None:
                            if tipo_ev == "PEF":
                                ax.scatter(
                                    t_plot,
                                    y_plot,
                                    color="red",
                                    marker="o",
                                    s=42,
                                    zorder=8
                                )

                            elif tipo_ev in ["tos_segura", "tos_por_duracion"]:
                                ax.scatter(
                                    t_plot,
                                    y_plot,
                                    color="green",
                                    marker="x",
                                    s=55,
                                    linewidths=2.0,
                                    zorder=9
                                )

                ax.set_ylim(0, 1.05)
                ax.legend(fontsize=7, loc="upper right")

            elif f == 3:

                ultimo_im_db = dibujar_logmel_suave(
                    ax,
                    g["t_mel_raw_edges"],
                    g["logmel_raw"],
                    g["freqs_mel_raw"],
                )

            elif f == 4:

                ultimo_im_db = dibujar_logmel_suave(
                    ax,
                    g["t_mel_den_edges"],
                    g["logmel_den"],
                    g["freqs_mel_den"],
                )

            elif f == 5:


                ax.plot(
                    g["t_env_logmel_den"],
                    g["env_logmel_den"],
                    lw=0.9,
                    label="Env Log-Mel den"
                )


                if np.any(mascara_tos) and len(g["t_env_logmel_den"]) > 0:
                    idx_tos_logmel = np.round(g["t_env_logmel_den"] * d["fs"]).astype(int)
                    idx_tos_logmel = np.clip(idx_tos_logmel, 0, len(mascara_tos) - 1)
                    mascara_tos_logmel_plot = mascara_tos[idx_tos_logmel]

                    ax.fill_between(
                        g["t_env_logmel_den"],
                        0,
                        np.max(g["env_logmel_den"]) if len(g["env_logmel_den"]) > 0 else 1,
                        where=mascara_tos_logmel_plot,
                        alpha=0.09
                    )


                if len(eventos_picos) > 0:
                    for ev in eventos_picos:
                        tipo_ev = ev.get("tipo", "")

                        if tipo_ev == "artefacto":
                            continue

                        t_pico = float(ev.get("pico", 0.0))
                        t_plot, y_plot = ajustar_pico_a_maximo_visible(
                            t_pico,
                            g["t_env_logmel_den"],
                            g["env_logmel_den"],
                            ventana_s=0.12
                        )

                        if t_plot is not None:
                            if tipo_ev == "PEF":
                                ax.scatter(
                                    t_plot,
                                    y_plot,
                                    color="red",
                                    marker="o",
                                    s=42,
                                    zorder=8
                                )

                            elif tipo_ev in ["tos_segura", "tos_por_duracion"]:
                                ax.scatter(
                                    t_plot,
                                    y_plot,
                                    color="green",
                                    marker="x",
                                    s=55,
                                    linewidths=2.0,
                                    zorder=9
                                )

                ax.set_ylim(0, 1.05)
                ax.legend(fontsize=7, loc="upper right")


            ax.set_xlim(0.0, tmax_comun)
            ax.margins(x=0)

            if f in (3, 4):
                ax.grid(False, which="both")
            else:
                ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=7)

            if f < nfilas - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Tiempo (s)", fontsize=8)

            ejes_temporales.append(ax)


    for i, etiqueta in enumerate(filas):
        fig.text(
            0.030,
            axs[i, 0].get_position().y0 + axs[i, 0].get_position().height / 2,
            etiqueta,
            ha="center",
            va="center",
            fontsize=9,
            rotation=90
        )

    sincronizar_zoom(fig, [ejes_temporales])

    if ultimo_im_db is not None:
        cax = fig.add_axes([0.930, 0.30, 0.012, 0.38])
        fig.colorbar(ultimo_im_db, cax=cax).set_label("dB relativos", fontsize=8)

    fig.suptitle("Comparación de audio espiratorio", fontsize=14, y=0.970)
    fig._botones_audio = botones

    try:
        plt.get_current_fig_manager().window.state("zoomed")
    except Exception:
        pass

    return fig


def main():
    """Selecciona hasta dos WAV, procesa cada uno y muestra conclusión y gráficas."""
    root = Tk()
    root.withdraw()


    ruta_1 = seleccionar_audio_opcional("AUDIO 1 / ESTETOSCOPIO 1")
    ruta_2 = seleccionar_audio_opcional("AUDIO 2 / ESTETOSCOPIO 2 (opcional)")

    archivos = {}

    if ruta_1 is not None:
        archivos["ESTETOSCOPIO 1"] = ruta_1

    if ruta_2 is not None:
        archivos["ESTETOSCOPIO 2"] = ruta_2

    if len(archivos) == 0:
        root.destroy()
        raise ValueError("No se ha seleccionado ningún audio.")

    output_folder = filedialog.askdirectory(title="Carpeta de guardado")
    if not output_folder:
        root.destroy()
        raise ValueError("Carpeta no seleccionada.")

    root.destroy()
    os.makedirs(output_folder, exist_ok=True)

    datos = {}

    for nombre, ruta in archivos.items():
        datos[nombre] = cargar_y_procesar(ruta, nombre, output_folder)
        datos[nombre]["graficas"] = preparar_graficas_fuente(datos[nombre])


    for nombre in datos:
        g = datos[nombre]["graficas"]

        builtins.print(f"\n--- {nombre} ---")

        imprimir_SOLO_ratio_baremo_freq_db(
            g.get("ratio_variacion_frecuencial_200ms_logmel_den", {})
        )

        imprimir_SOLO_ratio_baremo_freq_db(
            g.get("ratio_variacion_frecuencial_200ms_logmel_raw", {})
        )

        conclusion = calcular_conclusion_final_prueba(
            datos[nombre],
            g,
            umbral_pef_retrasado_s=0.150,
            umbral_terminacion_precoz=0.60
        )

        imprimir_conclusion_final_prueba(nombre, conclusion)

    dibujar_ventana_unica(datos)
    plt.show()


if __name__ == "__main__":
    main()

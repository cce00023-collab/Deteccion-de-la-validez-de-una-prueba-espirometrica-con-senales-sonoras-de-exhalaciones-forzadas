# -*- coding: utf-8 -*-
# =====================================================
# RESUMEN MATEMATICO DEL PROGRAMA
# =====================================================
# Senal discreta: x[n], n = 0,...,N-1; tiempo: t_n = n/fs.
# Normalizacion: x_norm[n] = x[n] / max_n |x[n]|, si max_n |x[n]| > 0.
# Ventana de analisis: w[n], longitud Nw; salto H = Nw*(1-S).
# RMS por ventana: RMS[k] = sqrt((1/Nw) * sum_{n=0}^{Nw-1} x^2[kH+n]).
# STFT: X[m,k] = sum_n x[n] w[n-kH] exp(-j*2*pi*m*n/NFFT).
# Potencia espectral: P[m,k] = |X[m,k]|^2.
# Escala Mel: Mel(f) = 2595*log10(1 + f/700).
# Inversa Mel: f = 700*(10^(Mel/2595) - 1).
# Banco Mel: E[b,k] = sum_m M[b,m] P[m,k].
# Log-Mel: L[b,k] = log10(E[b,k] + eps).
# Envolvente temporal: e[k] = media_b(L[b,k]), suavizada y normalizada a [0,1].
# Reduccion de ruido: se estima ruido por mediana temporal y se aplica ganancia limitada.
# Ganancia espectral: G[b,k] = max(gain_floor, 1 - alpha*R[b,k]/(P[b,k]+eps)).
# Senal denoised: Y[m,k] = G[m,k] X[m,k]; y_den[n] = iSTFT(Y[m,k]).
# PEF acustico: t_PEF = argmax_t e(t) dentro del evento espiratorio aceptado.
# Esfuerzo insuficiente: PEF retrasado si t_PEF - t_inicio > 0.150 s.
# Tos: evento breve de alta energia, contrastado con RMS y banda 500-1500 Hz.
# Esfuerzo multiple: reaparicion significativa de energia tras una bajada o apagon.
# Terminacion precoz: caida brusca de energia tras el PEF y/o perdida final de energia.
# =====================================================

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

# =====================================================
# SALIDA POR PANTALLA CONTROLADA
# =====================================================
# Se silencian todos los print internos antiguos.
# Para mostrar el resumen final se usa builtins.print().
print = lambda *args, **kwargs: None

# =====================================================
# CONFIGURACION ESPECIFICA POR DISPOSITIVO
# =====================================================
# Aquí se controlan la fuerza del filtro y la sensibilidad de la tos
# para que no elimine la subida en PC/Estetoscopio y no introduzca picos artificiales en Móvil.
# IMPORTANTE, al tomarse la decisión de solo operar con el estetosocopio, se deja la configuración


CONFIG_FUENTES = {
    "MOVIL": {
        "lmediant": 20,              # Ventana LARGA para NO eliminar parte del soplido
        "denoising_strength": 0.10,  # Denoising poco agresivo
        "gain_floor": 0.85,          # Protege la amplitud original
        "tos_onset_db": 5.0,
        "tos_offset_db": 2.5,
        "tos_pad_pre": 0.06,
        "tos_pad_post": 0.10      # Engloba la tos continua
    },
    "PC": {
        "lmediant": 40,              # Ventana LARGA para NO eliminar parte del soplido
        "denoising_strength": 0.70,  # Denoising poco agresivo
        "gain_floor": 0.85,          # Protege la amplitud original
        "tos_onset_db": 5.0,
        "tos_offset_db": 2.5,
        "tos_pad_pre": 0.06,
        "tos_pad_post": 0.10   
    },
    "ESTETOSCOPIO": {
        "lmediant": 40,              # Ventana moderada: estable sin ser excesiva
        "denoising_strength": 0.70,  # Denoising mínimo
        "gain_floor": 0.90,          # Mantiene intacta la curva
        "tos_onset_db": 4.5,
        "tos_offset_db": 2.0,
        "tos_pad_pre": 0.06,
        "tos_pad_post": 0.10
    }
}

# =====================================================
# PARAMETROS GLOBALES
# =====================================================

target_fs = 4000

# Parametros base del denoising
N_denoising = 256
S_denoising = 0.5
window_denoising = signal.windows.hamming(N_denoising)

# Parametros para graficas
N_stft_graficas = 256
S_stft_graficas = 0.5
n_mels=40


# Suavizado temporal envolvente
tiempo_suavizado_env = 0.25  # Segundos para el resto (tos será de 0.05)

# Tos
preservar_tos = True
tos_duracion_min = 0.05
tos_duracion_max = 0.30
tos_rms_ventana = 0.020
tos_rms_salto = 0.010
tos_gain_floor = 0.90


# =====================================================
# UTILIDADES BASICAS
# =====================================================

def seleccionar_audio(nombre_dispositivo):
    print(f"Selecciona el audio correspondiente a: {nombre_dispositivo}")
    ruta = filedialog.askopenfilename(
        title=f"Selecciona el audio de {nombre_dispositivo}",
        filetypes=[("Archivos WAV", "*.wav"), ("Todos los archivos", "*.*")]
    )
    if ruta == "":
        raise ValueError(f"No se ha seleccionado archivo para {nombre_dispositivo}")
    return ruta


def seleccionar_audio_opcional(nombre_dispositivo):
    """
    Igual que seleccionar_audio(), pero si se cancela devuelve None.
    Sirve para comparar dos audios o mostrar solo uno.
    """
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
    if y.dtype == np.int16: return y.astype(np.float64) / 32768.0
    elif y.dtype == np.int32: return y.astype(np.float64) / 2147483648.0
    elif y.dtype == np.uint8: return (y.astype(np.float64) - 128.0) / 128.0
    else:
        y = y.astype(np.float64)
        max_y = np.max(np.abs(y))
        if max_y > 1: y = y / max_y
        return y

def normalizar_audio(y):
    y = np.asarray(y, dtype=np.float64)
    if len(y) == 0: return y
    max_y = np.max(np.abs(y))
    if max_y > 0: y = y / max_y
    return y

def remuestrear_audio(y, fs_original, fs_objetivo):
    if fs_original == fs_objetivo: return y, fs_original
    mcd = math.gcd(fs_original, fs_objetivo)
    up = fs_objetivo // mcd
    down = fs_original // mcd
    y_resampled = signal.resample_poly(y, up, down)
    return y_resampled, fs_objetivo


def suavizar_senal_cruda_25ms(y, fs):
    """
    Calcula la forma de la señal cruda mediante RMS de 25 ms.

    Esto NO aplica:
    - denoising
    - STFT
    - Mel / Log-Mel
    - filtro frecuencial
    - mediana espectral
    - eliminación de ruido

    Solo hace:
    señal cruda -> energía local RMS en ventanas de 25 ms

    Es lo más adecuado para ver la forma del soplido, porque la señal acústica
    cruda oscila positivo/negativo y una media móvil directa sobre y puede
    cancelarse y tender a cero.
    """

    y = np.asarray(y, dtype=np.float64)

    if len(y) == 0 or fs <= 0:
        return y.copy()

    # Normalización solo para visualización.
    max_y = np.max(np.abs(y))
    if max_y > 0:
        y = y / max_y

    ventana = max(1, int(round(0.025 * fs)))  # 25 ms

    # RMS centrado sin eliminar los bordes.
    pad_izq = ventana // 2
    pad_der = ventana - 1 - pad_izq

    y_pad = np.pad(y, (pad_izq, pad_der), mode="edge")
    kernel = np.ones(ventana, dtype=np.float64) / ventana

    energia = np.convolve(y_pad ** 2, kernel, mode="valid")
    rms = np.sqrt(np.maximum(energia, 0.0))

    # Normalizar para que la forma se vea entre 0 y 1.
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
    """
    Quita repuntes falsos al final del Log-Mel usando una máscara de actividad.

    Motivo:
    En los últimos frames puede aparecer una columna de energía alta falsa
    por ruido residual, borde de ventana o normalización visual del Log-Mel.
    Si la señal temporal denoised ya está prácticamente a cero, esa columna
    no representa soplido real.

    Qué hace:
    - Calcula RMS temporal de la señal de referencia y_ref.
    - Normaliza esa RMS entre 0 y 1.
    - Solo mira la zona final de la señal.
    - Si RMS < umbral_rms, fuerza esa columna Log-Mel a -80 dB.

    Importante:
    - No cambia el audio.
    - No cambia la señal denoised exportada.
    - No cambia la detección de PEF/tos.
    - Solo limpia la representación visual del espectrograma.
    """

    logmel = np.asarray(logmel, dtype=np.float64).copy()
    t_logmel = np.asarray(t_logmel, dtype=np.float64)
    y_ref = np.asarray(y_ref, dtype=np.float64)

    if logmel.size == 0 or len(t_logmel) == 0 or len(y_ref) == 0 or fs_ref <= 0:
        return logmel

    # Asegurar forma bandas x tiempo.
    if logmel.ndim == 1:
        logmel = logmel.reshape(1, -1)

    n_frames = min(logmel.shape[1], len(t_logmel))
    if n_frames == 0:
        return logmel

    logmel[:, :n_frames] = logmel[:, :n_frames]

    # RMS de 25 ms sobre la señal temporal denoised.
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

    # Mapear cada columna Log-Mel a una muestra temporal.
    idx = np.round(t_logmel[:n_frames] * fs_ref).astype(int)
    idx = np.clip(idx, 0, len(rms) - 1)

    rms_en_frames = rms[idx]

    # Solo se aplica al tramo final para no borrar partes válidas del soplido.
    zona_final = t_logmel[:n_frames] >= max(0.0, duracion - zona_final_s)
    baja_actividad = rms_en_frames < umbral_rms

    mascara_borrar = zona_final & baja_actividad

    if np.any(mascara_borrar):
        logmel[:, :n_frames][:, mascara_borrar] = -80.0

    return logmel


# =====================================================
# DETECCION DE TOS EXACTA (CON CONFIG ESPECIFICA)
# =====================================================

def detectar_mascara_tos_en_muestras(fs, y, config):
    """
    Detección de tos + PEF por curvas reales.

    CAMBIO IMPORTANTE:
    - La tos se sigue detectando por máximos locales + duración 0.05-0.30 s.
    - El PEF ya NO se obtiene de todos los máximos locales.
    - El PEF se detecta como: 1 curva real = 1 máximo principal.

    Así se evita que hombros u ondulaciones internas del soplido sean marcados
    como PEF adicionales.
    """

    y = normalizar_audio(y)
    n = len(y)

    mascara_tos = np.zeros(n, dtype=bool)

    if n < 10 or np.max(np.abs(y)) <= 0:
        detectar_mascara_tos_en_muestras.ultimos_eventos = []
        detectar_mascara_tos_en_muestras.ultimos_maximos = []
        return mascara_tos

    # Copia del AUDIO REAL para la confirmación espectral de tos.
    # No es la curva suavizada.
    y_audio_para_espectro = y.copy()

    # -------------------------------------------------
    # Confirmación espectral 500-1500 Hz para tos
    # -------------------------------------------------

    def confirmar_tos_por_banda_500_1500(audio_real_evento, fs_audio, idx_inicio, idx_fin):
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

        # Para que el grave del soplido no domine la decisión.
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

    # -------------------------------------------------
    # Curva suavizada para detectar forma temporal
    # -------------------------------------------------
    # Esta curva sirve para:
    # - detectar máximos
    # - medir inicio10/fin10
    # - separar curvas reales de PEF
    #
    # NO sirve para analizar frecuencias 500-1500 Hz.

    y_abs = np.abs(y)

    ventana_rms = max(1, int(0.025 * fs))  # 25 ms
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

    sigma = max(1, int(0.020 * fs))  # 20 ms
    curva = ndimage.gaussian_filter1d(curva, sigma=sigma, mode="reflect")

    if np.max(curva) > 0:
        curva = curva / np.max(curva)

    # -------------------------------------------------
    # Función auxiliar: medir inicio10 y fin10
    # -------------------------------------------------

    def medir_evento_al_10(idx_pico):
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

    # -------------------------------------------------
    # CRITERIO CRÍTICO DE DURACIÓN PEF AL 10%
    # -------------------------------------------------
    # Para marcar PEF, el tramo entre inicio10 y fin10 debe durar
    # como mínimo 300 ms. Se comprueba contando muestras.
    duracion_min_pef_al_10_s = 0.30
    muestras_min_pef_al_10 = int(round(duracion_min_pef_al_10_s * fs))

    # -------------------------------------------------
    # PEF POR CURVAS REALES
    # -------------------------------------------------
    # Idea:
    # 1. Se buscan máximos candidatos.
    # 2. Entre dos máximos se mira si hay un valle real.
    # 3. Si NO hay valle real, pertenecen a la misma curva y se queda solo el mayor.
    # 4. Si SÍ hay valle real, se considera una curva nueva.
    #
    # Resultado:
    # 1 curva real = 1 PEF.

    def detectar_pef_por_curvas_reales(curva, fs):
        """
        Detecta PEF reales con las condiciones pedidas:

        REGLA 1:
        El primer PEF ya NO es el máximo global de toda la curva.
        Ahora es el punto más alto antes de que la curva empiece a bajar
        de forma sostenida.

        REGLA 2:
        Nunca se puede marcar un PEF antes de ese primer PEF.

        REGLA 3:
        Solo se buscan posibles segundos PEF después del primer PEF.

        REGLA 4:
        Para aceptar un segundo PEF real debe cumplirse:
        - mínimo 300 ms de separación respecto al PEF anterior;
        - bajada significativa desde el PEF anterior hasta el valle;
        - valle suficientemente claro para separar curvas;
        - subida significativa desde el valle hasta el segundo PEF;
        - duración del evento al 10% >= 300 ms.
        """

        if len(curva) == 0 or np.max(curva) <= 0:
            return []

        curva = np.asarray(curva, dtype=np.float64)

        umbral_actividad = 0.12
        prominencia_minima = 0.08

        # -------------------------------------------------
        # CONDICIONES REALES PARA SEGUNDO PEF
        # -------------------------------------------------
        tiempo_min_segundo_pef_s = 0.30

        # Bajada desde el PEF anterior hasta el valle.
        caida_minima_valle_segundo_pef = 0.30

        # El valle debe ser claro, pero no se exige caer hasta el 10%.
        # Evita doble cresta de la misma subida sin cargarse esfuerzos reales.
        valle_relativo_max_segundo_pef = 0.65

        # Subida desde el valle hasta el segundo PEF.
        subida_minima_desde_valle_segundo_pef = 0.30

        # Si dos máximos comparten la misma ventana energética al 10%,
        # pertenecen al mismo bloque de soplido: no son dos PEF.
        # Esto corrige dobles marcadores dentro de la misma subida sin exigir
        # un reposo extra cuando sí existe una bajada real entre esfuerzos.
        tolerancia_solape_misma_curva_s = 0.0
        valle_relativo_max_pef_solapado_independiente = 0.45
        caida_min_pef_solapado_independiente = 0.50
        subida_min_pef_solapado_independiente = 0.45
        tiempo_max_pef_solapado_independiente_s = 1.05
        solape_min_pef_solapado_independiente_s = -1.60
        pico_candidato_min_pef_solapado_independiente = 0.70
        pico_candidato_max_pef_solapado_independiente = 0.90

        # -------------------------------------------------
        # PRIMER PEF = MAXIMO DEL PRIMER BLOQUE ACTIVO
        # -------------------------------------------------
        # Desde que empieza la actividad, el primer PEF debe ser el punto
        # mas alto de ese primer bloque de soplido. Solo se cierra el bloque
        # cuando la curva vuelve a un nivel basal bajo de forma sostenida.
        def detectar_primer_pef_antes_de_bajada(curva_local, fs_local):
            """
            Primer PEF:
            maximo del primer bloque activo desde el inicio real del soplido.
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

        # -------------------------------------------------
        # CANDIDATOS A SEGUNDO PEF
        # -------------------------------------------------
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

        # Asegurar que el primer PEF está considerado como referencia.
        if idx_primer_pef not in maximos:
            maximos.append(idx_primer_pef)

        maximos = sorted([int(m) for m in maximos])

        # Nunca se puede marcar un PEF antes del primer PEF detectado.
        candidatos_post_primer_pef = [idx for idx in maximos if idx > idx_primer_pef]

        for idx in candidatos_post_primer_pef:
            idx = int(idx)
            ultimo = int(pefs[-1])

            tiempo_entre_pefs = (idx - ultimo) / fs

            # Condición 1: mínimo 300 ms entre picos.
            if tiempo_entre_pefs < tiempo_min_segundo_pef_s:
                continue

            ini = min(ultimo, idx)
            fin = max(ultimo, idx)

            if fin <= ini:
                continue

            # Valle entre el PEF anterior y el posible segundo PEF.
            tramo_entre_picos = curva[ini:fin + 1]
            idx_valle_local = int(np.argmin(tramo_entre_picos))
            idx_valle = ini + idx_valle_local
            valle = float(curva[idx_valle])

            pico_anterior = float(curva[ultimo])
            pico_candidato = float(curva[idx])
            pico_menor = float(min(pico_anterior, pico_candidato))

            # Condición 2: bajada significativa desde el PEF anterior hasta el valle.
            caida_desde_pef_anterior = pico_anterior - valle

            # Condición 3: valle claro respecto al pico menor.
            if pico_menor > 0:
                valle_relativo = valle / pico_menor
            else:
                valle_relativo = 1.0

            # Condición 4: subida significativa desde el valle hasta el segundo PEF.
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

        # Para que un máximo principal de curva sea PEF, debe durar >= 300 ms
        # entre inicio10 y fin10. Se comprueba contando muestras.
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

    # -------------------------------------------------
    # TOS POR MÁXIMOS LOCALES + DURACIÓN 0.05-0.30 s
    # -------------------------------------------------
    # Esta parte se mantiene independiente de los PEF:
    # una tos pegada al soplido puede detectarse aunque no sea una curva PEF.

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

        # Si coincide prácticamente con un PEF, no lo metemos como tos.
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
            # Evento largo que NO es máximo principal de curva PEF.
            # Se descarta como hombro/ondulación larga para que no salga como PEF falso.
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

    # -------------------------------------------------
    # Crear máscara solo con eventos de tos
    # -------------------------------------------------

    padding_pre = int(config.get("tos_pad_pre", 0.06) * fs)
    padding_post = int(config.get("tos_pad_post", 0.10) * fs)

    for ev in eventos_tos:
        ini_p = max(0, ev["idx_inicio"] - padding_pre)
        fin_p = min(n, ev["idx_fin"] + padding_post)
        mascara_tos[ini_p:fin_p] = True

    # Eventos que se mostrarán en la gráfica:
    # - PEF principales por curvas reales
    # - Tos
    # - Artefactos no se pintan si tu gráfica los omite
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
    mascara_frames = np.zeros(len(times), dtype=bool)
    for i, tiempo_frame in enumerate(times):
        muestra = int(tiempo_frame * fs)
        muestra = min(max(muestra, 0), longitud_senal - 1)
        mascara_frames[i] = mascara_muestras[muestra]
    return mascara_frames


# =====================================================
# DENOISING OPTIMIZADO (CON CONFIG ESPECIFICA)
# =====================================================

def aplicar_denoising(fs, y, N, S, window, config):
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

    # Optimizacion masiva: usamos ndimage.median_filter en el eje temporal
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



# =====================================================
# CONTROL DE TOS SATURADA PARA QUE NO DOMINE LA NORMALIZACION
# =====================================================

def controlar_tos_saturada_antes_normalizar(fs, y, config):
    """
    Baja solo la tos si está saturando antes de normalizar.

    La tos no se elimina: solo se reduce si domina demasiado el máximo.
    Así no se convierte en la referencia de normalización y no hunde el soplido.
    """

    y = np.asarray(y, dtype=np.float64)

    if len(y) == 0 or np.max(np.abs(y)) <= 0:
        return y

    # Normalización provisional solo para detectar la tos.
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

    # La tos puede quedar algo por encima del soplido, pero no dominar toda la escala.
    limite_tos = 1.20 * ref_soplido

    if pico_tos <= limite_tos:
        return y

    gain_tos = limite_tos / (pico_tos + 1e-12)

    transicion = max(1, int(0.025 * fs))  # 25 ms
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
    """
    Recorta el espacio blanco/silencioso del inicio de la señal.

    La idea es que el audio empiece ya en t=0 justo antes del soplido real,
    pero conservando un pequeño margen previo para que la curva arranque
    desde el suelo y suba.

    Devuelve:
    - y_recortada
    - muestras_recortadas

    No modifica la forma de la tos ni la detección: simplemente desplaza
    toda la señal para que el inicio útil esté en t=0.
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


# =====================================================
# CARGAR AUDIO CRUDO Y PROCESAR
# =====================================================

def cargar_y_procesar(ruta_audio, nombre_dispositivo, output_folder):
    fs_raw, y_raw = wav.read(ruta_audio)

    print(f"\n{nombre_dispositivo}")
    print(f"Archivo original = {os.path.basename(ruta_audio)}")

    y_raw_float = convertir_a_float(y_raw)
    y_raw_graficas = normalizar_audio(y_raw_float[:, 0] if y_raw_float.ndim == 2 else y_raw_float)

    y_proc = np.mean(y_raw_float, axis=1) if y_raw_float.ndim == 2 else y_raw_float.copy()
    if np.max(np.abs(y_proc)) > 0: y_proc = y_proc / np.max(np.abs(y_proc))

    y_proc, fs_proc = remuestrear_audio(y_proc, fs_raw, target_fs)

    # -------------------------------------------------
    # RECORTE DE ESPACIO BLANCO INICIAL
    # -------------------------------------------------
    # Si al principio hay silencio/espacio blanco, se recorta para que
    # el soplido empiece en t=0 con la curva desde el suelo.
    y_proc, muestras_recorte_inicio = recortar_espacio_blanco_inicio(
        y_proc,
        fs_proc,
        umbral_relativo=0.015,
        ventana_rms_s=0.020,
        muestras_consecutivas_s=0.050,
        margen_pre_s=0.000
    )

    # Tras recortar, normalizamos de nuevo si hay contenido.
    if np.max(np.abs(y_proc)) > 0:
        y_proc = y_proc / np.max(np.abs(y_proc))

    # OBTENER CONFIGURACION
    # Si se comparan dos audios como ESTETOSCOPIO 1 / ESTETOSCOPIO 2,
    # ambos usan la configuración del estetoscopio original.
    nombre_config = "ESTETOSCOPIO" if str(nombre_dispositivo).startswith("ESTETOSCOPIO") else nombre_dispositivo
    config_disp = CONFIG_FUENTES.get(nombre_config, CONFIG_FUENTES["MOVIL"])

    # Si la tos está saturando en estetoscopio, se baja antes de normalizar por máximo.
    if nombre_config == "ESTETOSCOPIO":
        y_proc = controlar_tos_saturada_antes_normalizar(fs_proc, y_proc, config_disp)

    if np.max(np.abs(y_proc)) > 0: y_proc = y_proc / np.max(np.abs(y_proc))

    y_denoising_entrada = y_proc.copy()

    # Recorte equivalente para la señal original usada SOLO en gráficas.
    # Así Ori y Den empiezan alineadas en t=0.
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


# =====================================================
# ENVOLVENTES DINAMICAS ADAPTATIVAS (0.05s Tos / 0.25s Resto)
# =====================================================


def limitar_tos_sin_aplanar_curva(envolvente_norm, mascara_tos, fs,
                                  limite_visual=1.0,
                                  transicion_s=0.025):
    """
    Limita la tos para que entre en [0, 1] SIN aplanarla.

    Problema:
    Si se usa np.clip directamente, la parte alta de la tos se convierte
    en una meseta horizontal.

    Solución:
    - Detecta cuánto supera la tos el límite.
    - Aplica una ganancia SOLO a la zona de tos.
    - Usa una transición suave para no crear cortes.
    - Mantiene la forma relativa de la tos.

    No cambia la máscara ni la detección.
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

    # Ganancia única para toda la tos: conserva la forma.
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

    # Última seguridad: si algún borde supera por transición, se recorta muy poco.
    # Como la zona de tos ya fue escalada, no debería crear meseta en el pico real.
    envolvente_out = np.clip(envolvente_out, 0.0, limite_visual)

    return envolvente_out


def calcular_envolvente_temporal(y, fs, mascara_tos=None, normalizar_entrada=True, normalizar_salida=True):
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
        # -------------------------------------------------
        # NORMALIZACIÓN DE LA CURVA GENERAL SIN TENER EN CUENTA LA TOS
        # -------------------------------------------------
        # Si hay máscara de tos, la referencia de normalización se calcula
        # solo con la parte NO tos. Así la curva general del soplido gana
        # amplitud y la tos no hunde la escala.
        #
        # IMPORTANTE:
        # - La tos NO se elimina.
        # - La tos NO se baja.
        # - La máscara sigue igual.
        # - Solo cambia la referencia usada para escalar la curva.
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

        # -------------------------------------------------
        # LIMITAR TOS SIN APLANARLA
        # -------------------------------------------------
        # La referencia NO usa la tos, por eso la curva general mantiene amplitud.
        # Si la tos supera 1, se reduce SOLO la zona de tos con una ganancia suave,
        # conservando su forma. No se usa clip duro sobre la tos.
        envolvente_suave = limitar_tos_sin_aplanar_curva(
            envolvente_suave,
            mascara_tos,
            fs,
            limite_visual=1.0,
            transicion_s=0.025
        )

    return envolvente_suave


def calcular_envolvente_logmel_denoised(logmel_den, t_logmel, fs_audio, mascara_tos=None, normalizar_salida=True):
    """
    Calcula una envolvente 1D a partir de la Log-Mel denoised.

    Idea:
    - La Log-Mel está en dB relativos [-80, 0].
    - Para obtener energía real positiva, se pasa de dB a escala lineal:
          energia_lineal = 10 ** (logmel_den / 10)
    - Después se promedia la energía de todas las bandas Mel en cada instante.
    - Sobre esa curva se aplican los mismos suavizados que en la envolvente temporal:
        * ventana larga general: tiempo_suavizado_env
        * ventana corta en tos: 0.05 s
        * normalización sin usar la tos como referencia
        * limitación suave de tos sin aplanarla

    Devuelve:
    - t_logmel
    - envolvente_logmel_den
    """

    logmel_den = np.asarray(logmel_den, dtype=np.float64)
    t_logmel = np.asarray(t_logmel, dtype=np.float64)

    if logmel_den.size == 0 or len(t_logmel) == 0:
        return t_logmel, np.zeros(len(t_logmel), dtype=np.float64)

    # Asegurar forma bandas x tiempo.
    if logmel_den.ndim == 1:
        logmel_den = logmel_den.reshape(1, -1)

    n_frames = min(logmel_den.shape[1], len(t_logmel))
    logmel_den = logmel_den[:, :n_frames]
    t_logmel = t_logmel[:n_frames]

    if n_frames == 0:
        return t_logmel, np.zeros(0, dtype=np.float64)

    # Convertir dB relativos a energía lineal positiva.
    energia_lineal = 10.0 ** (logmel_den / 10.0)

    # Promedio entre bandas Mel: envolvente de energía Log-Mel denoised.
    envolvente = np.mean(energia_lineal, axis=0)

    if len(t_logmel) > 1:
        dt = float(np.median(np.diff(t_logmel)))
        if dt <= 0:
            dt = 1.0 / max(float(fs_audio), 1.0)
    else:
        dt = 1.0 / max(float(fs_audio), 1.0)

    fs_env = 1.0 / dt

    # Mismo suavizado general que la envolvente temporal.
    vent_resto = max(int(tiempo_suavizado_env * fs_env), 1)
    env_resto = np.convolve(envolvente, np.ones(vent_resto) / vent_resto, mode="same")

    mascara_tos_logmel = None

    if mascara_tos is not None and len(mascara_tos) > 0:
        mascara_tos = np.asarray(mascara_tos, dtype=bool)

        # Pasar la máscara de tos del eje de muestras de audio al eje temporal Log-Mel.
        idx_tos = np.round(t_logmel * fs_audio).astype(int)
        idx_tos = np.clip(idx_tos, 0, len(mascara_tos) - 1)
        mascara_tos_logmel = mascara_tos[idx_tos]

    if mascara_tos_logmel is not None and np.any(mascara_tos_logmel):
        # Mismo suavizado corto usado para tos en la envolvente temporal.
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
        # Igual que en la envolvente temporal: la tos no define la escala general.
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

        # Misma protección para que la tos no quede aplanada.
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


# =====================================================
# DETECCION VECTORIZADA (ULTRA-RÁPIDA)
# =====================================================

def calcular_score_terminacion_precoz(
    y,
    fs,
    mascara_tos: np.ndarray | None = None,
    umbral_bajo: float = 0.15,        # Umbral del 15% (Final de nuestra ventana crítica)
    tiempo_optimo_caida: float = 0.5, 
    tiempo_max_caida: float = 2,    
    umbral_score: float = 0.85        
) -> dict:
    """
    Score de terminación precoz sobre envolvente temporal.
    Calcula de forma independiente la varianza espectral ESTRICTA entre bandas 
    en la ventana desde el PEF hasta el 15%, ignorando frames de tos.
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

    # 1. Envolvente temporal (normalizada entre 0 y 1.0)
    envolvente = calcular_envolvente_temporal(y, fs, mascara_tos=None, normalizar_entrada=False, normalizar_salida=True)
    n = len(envolvente)
    tiempos = np.arange(n) / fs

    # 2. Gestionar máscara de tos para exclusión del PEF
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

    # 3. PEF ACÚSTICO: Máximo libre de tos
    idx_validos = np.where(valido)[0]
    idx_pef = idx_validos[np.argmax(envolvente[idx_validos])]
    valor_pef = envolvente[idx_pef]

    if valor_pef <= 0:
        return {
            "score": 0.0, "tiempo": float(tiempos[idx_pef]), "caida_total": 0.0, "homogeneidad": 0.0, "correlacion": 0.0, 
            "terminacion_precoz": False, "mensaje": "No se trata de una posible terminación precoz",
            "varianza_ventana": 0.0
        }

    # 4. PUNTO DE CAÍDA (Para el Score original al 15%)
    segmento_post_pef = envolvente[idx_pef:]
    indices_caida = np.where(segmento_post_pef <= umbral_bajo)[0]
    if len(indices_caida) > 0:
        idx_bajo = idx_pef + indices_caida[0]
    else:
        idx_bajo = n - 1

    valor_bajo = envolvente[idx_bajo]
    caida_total = float(np.clip(valor_pef - valor_bajo, 0.0, 1.0))

    # 5. TIEMPO Y MUESTRAS TOTALES DEL SCORE (Línea continua con tos incluida)
    muestras_caida = int(idx_bajo - idx_pef + 1)
    tiempo_caida_total = muestras_caida / float(fs)

    # 6. CÁLCULO DEL SCORE ORIGINAL
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

    # =========================================================================
    # 7. NUEVO: CÁLCULO ESPECTRAL ULTRA-ESTRICTO (VENTANA PEF -> 15% SIN TOS)
    # =========================================================================
    tiempo_pef = tiempos[idx_pef]
    tiempo_bajo = tiempos[idx_bajo]

    # Calcular espectrograma
    f, t_stft, power, _ = calcular_stft_power(y, fs, normalizar_entrada=False)
    
    # Filtro 1: Encontrar qué frames están dentro del rango temporal (PEF hasta 15%)
    dentro_ventana = (t_stft >= tiempo_pef) & (t_stft <= tiempo_bajo)
    
    # Filtro 2: Mapear y excluir frames que coincidan con la máscara de tos
    idx_valido_stft = np.clip((t_stft * fs).astype(int), 0, n - 1)
    frames_sin_tos = valido[idx_valido_stft]
    
    # Intersección final: Dentro de la ventana crítica Y libres de tos
    frames_finales = dentro_ventana & frames_sin_tos

    if np.any(frames_finales):
        power_ventana = power[:, frames_finales]
        
        # Definimos las 8 bandas de frecuencia hasta 2000 Hz
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
        
        # Varianza entre las 8 bandas para cada frame de forma individual
        varianzas_por_frame = np.var(energia_bandas, axis=0)
        
        # Promedio final de las varianzas en este tramo limpio
        varianza_bandas_final = float(np.mean(varianzas_por_frame))
    else:
        varianza_bandas_final = 0.0

    # =========================================================================
    # 8. IMPRESIÓN UNIFICADA EN TERMINAL
    # =========================================================================
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
        "varianza_ventana": varianza_bandas_final  # Clave actualizada en el diccionario
    }
def calcular_envolvente_stft_en_tiempo(y, fs, normalizar_entrada=True, normalizar_salida=True):
    f, t, power, power_db = calcular_stft_power(y, fs, normalizar_entrada=normalizar_entrada)
    envolvente_stft = np.sqrt(np.sum(power, axis=0))
    if normalizar_salida and np.max(envolvente_stft) > 0: envolvente_stft = envolvente_stft / np.max(envolvente_stft)
    return t, envolvente_stft


# =====================================================
# MEL Y LOG-MEL SIN LIBROSA
# =====================================================

def hz_a_mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
def mel_a_hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

def crear_banco_mel(fs, nfft, n_mels):
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
    """
    Devuelve la frecuencia central aproximada de cada banda Mel.
    La banda i de la matriz Mel/Log-Mel corresponde aproximadamente
    a freqs_centro[i] Hz.
    """
    puntos_hz = mel_a_hz(np.linspace(hz_a_mel(0), hz_a_mel(fs / 2), n_mels + 2))
    return puntos_hz[1:-1]


def aplicar_eje_frecuencias_mel(ax, freqs_centro, n_mels, num_ticks=6):
    """
    Cambia el eje vertical de índice de banda Mel a frecuencia aproximada en Hz.
    La imagen sigue teniendo coordenadas 0..n_mels, pero las etiquetas muestran Hz.
    """
    freqs_centro = np.asarray(freqs_centro, dtype=np.float64)

    if len(freqs_centro) == 0:
        return

    num_ticks = min(num_ticks, len(freqs_centro))
    idx_ticks = np.linspace(0, len(freqs_centro) - 1, num_ticks).astype(int)

    # Como pcolormesh usa bordes 0..n_mels, el centro visual de la banda i es i + 0.5
    posiciones = idx_ticks + 0.5
    etiquetas = [f"{int(round(freqs_centro[i]))}" for i in idx_ticks]

    ax.set_yticks(posiciones)
    ax.set_yticklabels(etiquetas, fontsize=7)
    ax.set_ylabel("Hz", fontsize=8)

def calcular_mel_logmel(y, fs, ref_mel_max=None, normalizar_entrada=True):
    """
    Calcula Mel y Log-Mel para las gráficas.

    CAMBIO:
    No se usa scipy.stft con padding temporal para Mel/Log-Mel.
    Se crean las ventanas manualmente.

    Si una ventana supera el final de la señal:
    - se acorta hasta el máximo de muestras reales disponibles;
    - NO se rellena con ceros;
    - NO se crea un frame nulo artificial.

    Además devuelve t_edges, que son los bordes temporales reales para dibujar
    la imagen con pcolormesh sin recuadros blancos ni estiramientos falsos.
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

        # -------------------------------------------------
        # ÚLTIMA VENTANA SIN SOBREPASAR LA SEÑAL
        # -------------------------------------------------
        # Si la ventana normal sobrepasa el final, NO se cogen ceros a la derecha.
        # En vez de eso, la ventana se desplaza hacia la izquierda y termina
        # exactamente en la última muestra real de la señal.
        #
        # Caso importante:
        # si al final solo quedaría 1 muestra, no se crea una ventana de 1 muestra.
        # Se toma la máxima ventana posible hacia la izquierda.
        if fin > len(y):
            fin_real = len(y)
            inicio_real = max(0, fin_real - min(nperseg, len(y)))
        else:
            inicio_real = inicio
            fin_real = fin

        segmento = y[inicio_real:fin_real]

        # Si aun así quedara una única muestra porque la señal completa es mínima,
        # no se crea una ventana falsa.
        if len(segmento) < 2:
            break

        # Si al desplazar la última ventana hacia la izquierda se solapa con la
        # última ventana ya guardada de forma no creciente, se elimina la anterior.
        # Así no quedan ventanas duplicadas ni huecos raros al final.
        while len(inicios_frames) > 0 and inicio_real <= inicios_frames[-1]:
            frames_power.pop()
            tiempos_centro.pop()
            inicios_frames.pop()
            finales_frames.pop()

        # Ventana con la longitud REAL disponible.
        # No hay padding temporal con ceros.
        ventana = signal.windows.hann(len(segmento), sym=False)
        segmento_win = segmento * ventana

        # FFT con nfft fijo para mantener el mismo banco Mel.
        # La señal NO se rellena temporalmente; nfft solo da resolución frecuencial.
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

    # Bordes reales para graficar:
    # - empieza en 0;
    # - los bordes intermedios son puntos medios entre centros reales;
    # - termina exactamente en la duración real de la señal.
    # Si la última ventana tuvo que desplazarse hacia la izquierda, estos centros
    # ya reflejan ese cálculo real, sin ventanas de 1 muestra ni padding a cero.
    t_edges = np.zeros(len(t) + 1, dtype=np.float64)

    if len(t) == 1:
        t_edges[0] = 0.0
        t_edges[1] = len(y) / fs
    else:
        t_edges[0] = 0.0
        t_edges[1:-1] = (t[:-1] + t[1:]) / 2.0
        t_edges[-1] = len(y) / fs

    # Seguridad por si algún redondeo deja bordes no crecientes.
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

    # No se renormaliza el denoised. Solo se limita el máximo para visualizar.
    mel_norm = np.clip(mel_norm, 0, 1)

    logmel = np.clip(10 * np.log10(mel_norm + 1e-12), -80, 0)

    return t, mel_norm, logmel, ref_mel_max, t_edges

# =====================================================
# ANALISIS DE CORTE BRUSCO EN BANDAS BAJAS LOG-MEL ORIGINAL
# =====================================================

def calcular_pendiente_logmel_antes_energia_nula(
    y_den,
    fs,
    duracion_ventana=0.40,
    umbral_energia_nula=0.08,
    min_frames_nulos=2
):
    """
    Analiza el final del soplido buscando un corte brusco tipo "corte_brusco"
    en las bandas bajas de la señal que se le pase. En este código se llama con la señal ORIGINAL.

    CAMBIO IMPORTANTE:
    Esta versión NO depende de que el número de bandas activas caiga exactamente
    a cero entre dos frames. Eso era demasiado estricto.

    Ahora detecta el corte_brusco por caída brusca de energía baja entre dos
    pequeñas zonas temporales:
    - media de unos frames antes del corte
    - media de unos frames después del corte

    Esto es más robusto cuando la STFT deja una pequeña energía residual por
    fuga espectral o por el suavizado de la ventana.
    """

    # -------------------------------------------------
    # Preparar señal
    # -------------------------------------------------

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

    # -------------------------------------------------
    # Mel/log-Mel de MUY alta resolución temporal
    # -------------------------------------------------
    # Para fs = 4000:
    # nperseg = 32  -> 8 ms de ventana
    # hop = 4       -> 1 ms entre frames
    #
    # Esto reduce mucho el problema de no ver cortes de pocos milisegundos.

    nperseg_hr = min(32, len(y_den))
    nperseg_hr = max(16, nperseg_hr)

    hop_hr = max(1, int(0.001 * fs))      # aprox. 1 ms
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

    # -------------------------------------------------
    # Seleccionar bandas bajas
    # -------------------------------------------------
    # Se analiza el 45 % inferior de bandas Mel para incluir la zona baja
    # que queda visible al final del soplido.

    num_bandas_totales = mel_hr.shape[0]
    num_bandas_bajas = max(4, int(num_bandas_totales * 0.45))

    mel_bajas = mel_hr[:num_bandas_bajas, :]
    logmel_bajas = logmel_hr[:num_bandas_bajas, :]

    # -------------------------------------------------
    # Energía total de bandas bajas por frame
    # -------------------------------------------------

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

    # -------------------------------------------------
    # Buscar máximo energético en bajas frecuencias
    # -------------------------------------------------

    idx_pef_bajas = int(np.argmax(energia_bajas_norm))

    # -------------------------------------------------
    # Detectar el corte por caída PRE vs POST
    # -------------------------------------------------
    # En vez de mirar solo E[i]-E[i+1], se compara una pequeña media antes
    # y después del posible corte. Esto evita fallos por fuga espectral.
    #
    # pre_frames/post_frames:
    # con hop de 1 ms, 3 frames son unos 3 ms.

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

        # Priorizamos una caída grande en dB, pero evitando falsos cortes
        # cuando la energía antes del corte ya era prácticamente cero.
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

    # -------------------------------------------------
    # Buscar apagado global bajo para mantener compatibilidad
    # -------------------------------------------------

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

    # -------------------------------------------------
    # Ventana de análisis alrededor del corte detectado
    # -------------------------------------------------
    # Aunque duracion_ventana sea 0.40, la ventana termina un poco después
    # del corte para que entren los frames posteriores al corte_brusco.

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

    # Índice del corte dentro de la ventana final
    idx_corte_ventana = int(np.argmin(np.abs(t_final - tiempo_corte)))

    # -------------------------------------------------
    # Normalización por banda dentro de la ventana
    # -------------------------------------------------

    mel_final_norm = np.zeros_like(mel_final)

    for banda in range(mel_final.shape[0]):
        max_banda = np.max(mel_final[banda, :])
        if max_banda > 0:
            mel_final_norm[banda, :] = mel_final[banda, :] / max_banda
        else:
            mel_final_norm[banda, :] = 0.0

    # -------------------------------------------------
    # Bandas activas antes y después del corte
    # -------------------------------------------------
    # Se usa media de varios frames antes/después, no un único frame.
    # Así detecta mejor el corte_brusco aunque haya un pequeño residuo.

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

    # -------------------------------------------------
    # Tiempo de apagado de cada banda baja
    # -------------------------------------------------

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

    # -------------------------------------------------
    # Tiempo de colapso 80 % -> 20 % de bandas activas
    # -------------------------------------------------

    indices_80 = np.where(bandas_activas_norm <= 0.80)[0]
    indices_20 = np.where(bandas_activas_norm <= 0.20)[0]

    if len(indices_80) > 0 and len(indices_20) > 0:
        tiempo_80 = t_rel[indices_80[0]]
        tiempo_20 = t_rel[indices_20[0]]
        tiempo_colapso_bandas = float(max(0.0, tiempo_20 - tiempo_80))
    else:
        tiempo_colapso_bandas = float(t_rel[-1] - t_rel[0])

    # -------------------------------------------------
    # Pendiente global de energía baja en dB
    # -------------------------------------------------

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

    # -------------------------------------------------
    # Pendiente por banda baja en dB/s
    # -------------------------------------------------

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

    # -------------------------------------------------
    # Interpretación final
    # -------------------------------------------------
    # Esta versión detecta corte_brusco si:
    # - antes del corte todavía había energía baja suficiente
    # - después cae bastante
    # - la caída en dB es grande
    #
    # No obliga a que todas las bandas activas pasen a cero, porque en STFT
    # puede quedar residuo por la propia ventana.

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

    # -------------------------------------------------
    # Imprimir resultados
    # -------------------------------------------------

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
    """
    SOLO PARA VISUALIZACIÓN.

    Hace que la envolvente empiece en el suelo y vaya creciendo,
    evitando que aparezca ya alta tras un tramo blanco inicial.

    No cambia:
    - detección de tos
    - PEF
    - máscara
    - denoising
    - audio exportado
    """

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


# =====================================================
# UI Y VISUALIZACION
# =====================================================

def guardar_temp_wav(y, fs, prefijo, normalizar=True):
    y = np.asarray(y, dtype=np.float64)
    if normalizar: y = normalizar_audio(y)
    ruta_temp = os.path.join(tempfile.gettempdir(), prefijo + ".wav")
    wav.write(ruta_temp, fs, np.int16(np.clip(y, -1, 1) * 32767))
    return ruta_temp

def reproducir_audio(ruta_wav):
    sistema = platform.system()
    try:
        if sistema == "Windows": os.startfile(ruta_wav)
        elif sistema == "Darwin": subprocess.Popen(["open", ruta_wav])
        else: subprocess.Popen(["xdg-open", ruta_wav])
    except Exception as e: print("No se pudo reproducir:", e)

def sincronizar_zoom(fig, grupos_ejes):
    bloqueado = {"estado": False}
    def crear_callback(ejes_grupo):
        def callback(ax):
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
    """
    Calcula la energía media en valor absoluto del último cuarto de segundo
    y la compara con el cuarto de segundo inmediatamente anterior.

    Esto es más útil para terminación precoz que mirar solo el último tramo:
    - en una terminación progresiva, la diferencia entre ambos tramos suele ser menor;
    - en una terminación precoz/corte brusco, el tramo final suele caer mucho
      respecto al tramo anterior.

    Tramos:
    - anterior: [-0.50 s, -0.50 s]
    - último:   [-0.50 s,  0.00 s]
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

    # Tramo inmediatamente anterior. Si la señal es corta, se usa lo que haya.
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
    """
    Devuelve el tiempo del primer PEF detectado.
    Si no hay PEF, devuelve None.
    """
    pefs = [
        ev for ev in eventos_picos
        if isinstance(ev, dict) and ev.get("tipo", "") == "PEF"
    ]

    if len(pefs) == 0:
        return None

    pefs = sorted(pefs, key=lambda ev: float(ev.get("pico", 0.0)))
    return float(pefs[0].get("pico", 0.0))


def calcular_promedio_abs_desde_primer_pef(y, fs, eventos_picos):
    """
    Calcula el promedio de energía en valor absoluto desde el PRIMER PEF
    hasta el final de la señal.

    Es importante que sea promedio y no suma:
    así no afecta que un audio dure más que otro.

    Devuelve:
    - media_abs_desde_pef: promedio de abs(y) desde primer PEF hasta final
    - rms_desde_pef: RMS desde primer PEF hasta final
    - idx_primer_pef: índice de muestra del primer PEF
    - tiempo_primer_pef_s: tiempo del primer PEF
    - muestras_analizadas: muestras desde PEF hasta final
    - duracion_desde_pef_s: duración del tramo analizado
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
    """
    Divide los últimos 250 ms de una señal en ventanas de 10 ms y calcula
    la energía media en valor absoluto de cada ventana.

    Ejemplo:
    -250 a -240 ms, -240 a -230 ms, ..., -10 a 0 ms.
    """

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
    """
    Imprime la comparación de energía de los últimos 250 ms en ventanas de 10 ms.

    Si hay dos audios, compara ventana a ventana.
    Si hay un audio, imprime su evolución ventana a ventana.
    """

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



# =====================================================
# RATIOS POST-PEF OPTIMIZADOS
# =====================================================

def obtener_tiempo_pef_desde_eventos(eventos_picos):
    """
    Obtiene el tiempo del PEF ya detectado por el propio script.
    No cambia la detección de PEF.
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
    """
    Calcula energía entre dos ventanas consecutivas después del PEF.
    Energía = media del valor absoluto de la señal.
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
    """
    Calcula SOLO los ratios post-PEF añadidos.
    """

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
    """
    Imprime SOLO los ratios post-PEF.
    """

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


# =====================================================
# RATIO FINAL 0.50 s ANTES DEL LOG-MEL DENOISED NEGRO
# =====================================================

def calcular_ratio_final_050s_antes_logmel_negro(
    logmel_den,
    t_logmel_den,
    eventos_picos=None,
    duracion_ventana_s=0.50,
    umbral_negro_db=-79.9,
    min_frames_negros=1,
    t_logmel_den_edges=None
):
    """
    Calcula el ratio energético usando como referencia EXACTA el primer
    punto del primer cuadro negro del Log-Mel denoised.

    Corrección importante:
    - Antes se podía usar el centro del frame negro.
    - Ahora se usa el borde izquierdo del primer rectángulo negro de la gráfica.

    En la gráfica pcolormesh, una columna Log-Mel se dibuja entre:
        t_logmel_den_edges[i]  y  t_logmel_den_edges[i+1]

    Si la columna i es la primera columna completamente negra, entonces la
    referencia correcta es:
        t_negro_ref_s = t_logmel_den_edges[i]

    Ventanas:
        V1 = desde t_negro_ref_s - 1.00 s hasta t_negro_ref_s - 0.50 s
        V2 = desde t_negro_ref_s - 0.50 s hasta t_negro_ref_s

    Ratio:
        V1 / V2
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

    # Energía lineal positiva por frame:
    # -80 dB -> 1e-8 aprox.; 0 dB -> 1.
    energia_por_banda = 10.0 ** (logmel_use / 10.0)
    energia_frame = np.mean(energia_por_banda, axis=0)

    # -------------------------------------------------
    # DETECCIÓN DEL CUADRO NEGRO REAL
    # -------------------------------------------------
    # Una columna es negra solo si TODAS sus bandas están prácticamente
    # en el suelo de la escala (-80 dB). Usamos max <= -79.9 dB.
    max_db_frame = np.max(logmel_use, axis=0)
    negro = max_db_frame <= float(umbral_negro_db)

    # Tiempo PEF: si hay evento PEF se usa ese; si no, se usa el máximo
    # de energía del propio Log-Mel.
    t_pef_s = obtener_tiempo_pef_desde_eventos(eventos_picos)

    if t_pef_s is None:
        idx_pef_frame = int(np.argmax(energia_frame))
        t_pef_s = float(t_use[idx_pef_frame])
    else:
        idx_pef_frame = int(np.argmin(np.abs(t_use - float(t_pef_s))))

    min_frames_negros = max(1, int(min_frames_negros))

    idx_negro = None

    # Primer cuadro negro después del PEF.
    # min_frames_negros=1 para coger exactamente el primer cuadro negro,
    # no esperar a varios cuadros y desplazar la referencia.
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

    # -------------------------------------------------
    # REFERENCIA EXACTA = PRIMER PUNTO DEL CUADRO NEGRO
    # -------------------------------------------------
    if (
        t_logmel_den_edges is not None
        and len(t_logmel_den_edges) >= idx_negro + 1
    ):
        # Borde izquierdo de la columna negra.
        t_negro_ref_s = float(t_logmel_den_edges[idx_negro])
        t_negro_centro_s = float(t_use[idx_negro])
        referencia_usada = "borde_izquierdo_primer_cuadro_negro"
    else:
        # Fallback: si por cualquier razón no hay edges, se usa el centro.
        # Pero en tu código normal sí existen t_mel_den_edges.
        t_negro_ref_s = float(t_use[idx_negro])
        t_negro_centro_s = float(t_use[idx_negro])
        referencia_usada = "centro_frame_negro_fallback"

    # -------------------------------------------------
    # VENTANAS CORRECTAS RESPECTO AL PRIMER PUNTO NEGRO
    # -------------------------------------------------
    # V2: los 500 ms justo antes del primer punto negro.
    # V1: los 500 ms anteriores a V2.
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

    # Se mantiene también la caída relativa antigua por si quieres verla:
    # caída respecto a V1 = (V1 - V2) / V1
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

        # t_negro_s queda como la referencia principal para compatibilidad.
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
    """
    Imprime el ratio entre los 500 ms finales antes del primer punto del
    cuadro negro y los 500 ms anteriores a esa ventana.
    """

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


# =====================================================
# PEF EN ENVOLVENTE LOG-MEL DENOISED
# =====================================================

def detectar_pefs_envolvente_logmel_denoised(t_env, env_logmel_den):
    """
    Aplica a la envolvente del Log-Mel denoised la misma lógica de PEF:

    - Primer PEF:
      punto más alto antes de que la curva empiece a bajar de forma sostenida.

    - Nunca se marca un PEF antes de ese primer PEF.

    - Segundo PEF:
      se mantiene el criterio real:
        * mínimo 300 ms entre picos;
        * bajada significativa al valle;
        * valle claro, no solo una ondulación de la misma subida;
        * subida significativa desde el valle;
        * duración al 10% >= 300 ms.

    Devuelve eventos con el mismo formato básico que eventos_picos.
    """

    t_env = np.asarray(t_env, dtype=np.float64)
    curva = np.asarray(env_logmel_den, dtype=np.float64)

    if len(t_env) == 0 or len(curva) == 0 or np.max(curva) <= 0:
        return []

    n = min(len(t_env), len(curva))
    t_env = t_env[:n]
    curva = curva[:n]

    # Normalización por seguridad.
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
        """
        Primer PEF de la envolvente Log-Mel denoised:
        maximo del primer bloque activo desde el inicio real del soplido.
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


# =====================================================
# VARIACIÓN DE BANDAS EN 0.7 s ANTES DEL FINAL LOG-MEL DENOISED
# =====================================================

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
    """
    Detecta el punto final de la prueba en Log-Mel denoised y calcula
    variación de bandas en los 0.7 s anteriores, dando más peso a altas
    frecuencias.
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

    # Final principal: primer cuadro negro real después del PEF.
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
        # Fallback: última columna activa si no llega a negro real.
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

    # Positivo = pérdida de energía al final.
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
    """
    Imprime la variación de bandas en los 0.7 s anteriores al final
    del Log-Mel denoised.
    """

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


# =====================================================
# RATIOS DE VENTANAS 100 ms -> 1 s ANTES DEL FINAL LOG-MEL DENOISED
# =====================================================

def calcular_ratios_ventanas_100ms_a_1s_logmel_denoised(
    logmel_den,
    t_logmel_den,
    freqs_mel_den=None,
    eventos_picos=None,
    t_logmel_den_edges=None,
    umbral_negro_db=-79.9,
    umbral_banda_activa_db=-70.0
):
    """
    Para cada ventana W = 100, 200, ..., 1000 ms:

        V1 = ventana anterior: [t_final - 2W, t_final - W]
        V2 = ventana final:    [t_final - W,  t_final]

    Calcula ratios globales y ratios solo en bandas altas.
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
    """
    Imprime ratios comparativos de ventanas de 100 ms a 1 s.
    """

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


# =====================================================
# RATIO SINCERO 100 ms ANTES DEL RUIDO BASAL LOG-MEL DENOISED
# =====================================================

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
    """
    Detecta el momento en el que el Log-Mel denoised entra en ruido basal y
    compara SOLO dos ventanas de 100 ms:

        V1 = [t_basal - 0.200 s, t_basal - 0.100 s]
        V2 = [t_basal - 0.100 s, t_basal]

    Ratio sincero:
        Se calcula sobre la escala visual del Log-Mel:
            visual = clip((logmel_db - suelo_db) / (techo_db - suelo_db), 0, 1)

    Esto evita que pequeñas diferencias en escala lineal den ratios exagerados.
    Si la imagen se ve prácticamente igual, el ratio debe salir cerca de 1.
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

    # Escala visual sincera: lo que se ve en el mapa de color.
    visual = np.clip((L - float(suelo_db)) / (float(techo_db) - float(suelo_db)), 0.0, 1.0)

    energia_visual_frame = np.mean(visual, axis=0)

    # PEF de referencia.
    t_pef_s = obtener_tiempo_pef_desde_eventos(eventos_picos)

    if t_pef_s is None:
        idx_pef = int(np.argmax(energia_visual_frame))
        t_pef_s = float(t[idx_pef])
    else:
        idx_pef = int(np.argmin(np.abs(t - float(t_pef_s))))

    # -------------------------------------------------
    # DETECCIÓN DEL RUIDO BASAL
    # -------------------------------------------------
    # Basal estimado con la cola final del Log-Mel.
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

    # -------------------------------------------------
    # CORRECCIÓN DEFINITIVA: RECORRIDO DESDE ATRÁS
    # -------------------------------------------------
    # Se recorre el Log-Mel denoised desde el final hacia la izquierda.
    #
    # Idea:
    # 1) Al final debe haber zona negra / ruido basal.
    # 2) Recorremos hacia atrás mientras NO haya energía.
    # 3) En cuanto vuelve a aparecer energía, la referencia es el primer
    #    frame negro inmediatamente a la derecha de esa energía.
    #
    # Es decir:
    #     energía energía energía | negro negro negro negro
    #                              ^
    #                         referencia basal
    #
    # Desde esa referencia se toman:
    #     V2 = 100 ms justo a la izquierda
    #     V1 = los 100 ms anteriores a V2
    #
    # Esto evita por completo coger un negro intermedio o un falso basal temprano.

    # Umbral de "hay energía" en escala visual. Se calcula respecto a la cola final,
    # pero nunca se deja por debajo de tolerancia_basal_visual.
    umbral_energia = basal_visual + max(float(tolerancia_basal_visual), 3.0 * mad)

    # Si el umbral queda demasiado alto por algún artefacto en la cola, se limita.
    # En escala visual [0,1], 0.08 ya representa energía visible baja pero real.
    umbral_energia = min(float(umbral_energia), 0.08)

    hay_energia = energia_visual_frame > umbral_energia

    # Para evitar que una sola línea vertical/artefacto se considere energía,
    # se exige un pequeño bloque consecutivo de energía.
    n_energia_consecutiva = max(1, int(round(0.030 / dt)))  # 30 ms

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
        # El primer punto basal final es justo después del último punto con energía.
        idx_basal = min(idx_ultimo_con_energia + 1, n_frames - 1)
        criterio_basal = "recorrido_desde_atras_hasta_ultima_energia"
    else:
        # Si no se encuentra energía al recorrer desde atrás, se usa el último frame
        # como fallback para no bloquear el cálculo.
        idx_ultimo_con_energia = int(idx_pef)
        idx_basal = n_frames - 1
        criterio_basal = "fallback_recorrido_atras_sin_energia"

    idx_ultimo_activo = int(idx_ultimo_con_energia)

    # Referencia temporal exacta:
    # borde izquierdo del primer frame negro tras la última energía.
    if t_logmel_den_edges is not None and len(t_logmel_den_edges) >= idx_basal + 1:
        t_basal_s = float(t_logmel_den_edges[idx_basal])
        referencia_basal = "borde_izquierdo_primer_negro_tras_ultima_energia"
    else:
        t_basal_s = float(t[idx_basal])
        referencia_basal = "centro_frame_primer_negro_tras_ultima_energia_fallback"

    # -------------------------------------------------
    # VENTANAS EXACTAS DE 100 ms
    # -------------------------------------------------
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

    # -------------------------------------------------
    # RATIO SIMPLE EN ESCALA VISUAL
    # -------------------------------------------------
    # Se conserva como control, pero NO es el más sensible al escalón.
    v1 = float(np.mean(visual[:, m1]))
    v2 = float(np.mean(visual[:, m2]))

    ratio_v2_v1 = float(v2 / (v1 + 1e-12)) if v1 > 0 else 0.0
    ratio_v1_v2 = float(v1 / (v2 + 1e-12)) if v2 > 0 else 0.0
    diferencia_v1_menos_v2 = float(v1 - v2)

    if v1 > 0:
        variacion_relativa = float((v1 - v2) / (v1 + 1e-12))
    else:
        variacion_relativa = 0.0

    # Altas frecuencias = tercio superior.
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

    # -------------------------------------------------
    # RATIO MEJORADO PARA ESCALÓN VISUAL
    # -------------------------------------------------
    # Se resta el suelo/basal visual y solo se mide energía realmente visible.
    # Esto hace que, si V2 tiene mucho negro por el escalón, el ratio baje.
    visual_sobre_basal = np.maximum(visual - basal_visual, 0.0)

    # Umbral de actividad visual:
    # valores por debajo se consideran ruido/fondo, no energía útil.
    umbral_activo_visual = max(0.030, umbral_energia - basal_visual)

    activo_visual = visual_sobre_basal > umbral_activo_visual

    # Pesos por frecuencia: las bandas altas pesan más para que el escalón
    # de desaparición en altas frecuencias cuente más.
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

    # OJO: este es el ratio que interesa para ver el escalón:
    # si V2 pierde energía/bandas respecto a V1, baja claramente.
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

    # Ocupación espectral: mide cuánta superficie activa queda.
    ocupacion_v1 = float(np.mean(activo_visual[:, m1]))
    ocupacion_v2 = float(np.mean(activo_visual[:, m2]))

    ratio_ocupacion_v2_v1 = float(
        ocupacion_v2 / (ocupacion_v1 + 1e-12)
    ) if ocupacion_v1 > 0 else 0.0

    # Ocupación solo en altas.
    ocupacion_altas_v1 = float(np.mean(activo_visual[idx_altas, :][:, m1]))
    ocupacion_altas_v2 = float(np.mean(activo_visual[idx_altas, :][:, m2]))

    ratio_ocupacion_altas_v2_v1 = float(
        ocupacion_altas_v2 / (ocupacion_altas_v1 + 1e-12)
    ) if ocupacion_altas_v1 > 0 else 0.0

    # Energía activa solo en altas, sin diluir con bajas.
    energia_activa_altas_v1 = float(
        np.mean(energia_activa[idx_altas, :][:, m1])
    )
    energia_activa_altas_v2 = float(
        np.mean(energia_activa[idx_altas, :][:, m2])
    )

    ratio_energia_activa_altas_v2_v1 = float(
        energia_activa_altas_v2 / (energia_activa_altas_v1 + 1e-12)
    ) if energia_activa_altas_v1 > 0 else 0.0

    # Score de escalón:
    # combina caída de energía ponderada y pérdida de ocupación.
    # Cerca de 0 -> sin escalón claro.
    # Alto -> escalón claro.
    score_escalon_visual = float(
        0.50 * max(0.0, 1.0 - ratio_mejorado_v2_v1)
        + 0.30 * max(0.0, 1.0 - ratio_ocupacion_v2_v1)
        + 0.20 * max(0.0, 1.0 - ratio_ocupacion_altas_v2_v1)
    )

    # -------------------------------------------------
    # MÉTRICA NUEVA: TRANSICIÓN FRECUENCIAL RÁPIDA
    # -------------------------------------------------
    # Aquí NO se mide cuánta energía hay acumulada en bandas,
    # sino cómo cambia la localización frecuencial de la energía.
    #
    # Métricas por ventana:
    # - Centroide espectral medio.
    # - Velocidad media del centroide: Hz/s.
    # - Frecuencia dominante media.
    # - Velocidad de la frecuencia dominante: Hz/s.
    # - Borde frecuencial alto activo: frecuencia máxima que sigue activa.
    # - Velocidad de caída/subida de ese borde.
    # - Flujo espectral normalizado: cuánto cambia el patrón de frecuencias
    #   entre frames consecutivos.
    #
    # Esto detecta el "escalón" como un cambio rápido de distribución
    # frecuencial, no como simple energía acumulada.

    def calcular_dinamica_frecuencial(mascara_tiempo):
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

        # Energía activa por frecuencia y frame.
        A = energia_activa[:, idx_frames].astype(np.float64)

        # Si no queda energía activa, no hay movimiento frecuencial útil.
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

        # Centroide espectral por frame.
        centroide = np.zeros(len(idx_frames), dtype=np.float64)

        for k in range(len(idx_frames)):
            if energia_total_frame[k] > 0:
                centroide[k] = float(
                    np.sum(freqs * A[:, k]) / (energia_total_frame[k] + 1e-12)
                )
            else:
                centroide[k] = centroide[k - 1] if k > 0 else 0.0

        # Frecuencia dominante por frame.
        dominante = np.zeros(len(idx_frames), dtype=np.float64)

        for k in range(len(idx_frames)):
            if energia_total_frame[k] > 0:
                dominante[k] = float(freqs[int(np.argmax(A[:, k]))])
            else:
                dominante[k] = dominante[k - 1] if k > 0 else 0.0

        # Borde alto activo: frecuencia máxima con energía activa.
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
            x = np.asarray(x, dtype=np.float64)

            if len(x) < 2:
                return 0.0

            return float(np.mean(np.abs(np.diff(x)) / (dt_local + 1e-12)))

        def salto_maximo(x):
            x = np.asarray(x, dtype=np.float64)

            if len(x) < 2:
                return 0.0

            return float(np.max(np.abs(np.diff(x))))

        # Flujo espectral normalizado:
        # mide si el patrón por frecuencia cambia rápido entre frames.
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

    # Score de transición frecuencial:
    # sube si en V2 hay cambios frecuenciales más bruscos que en V1.
    score_transicion_frecuencial = float(
        0.35 * min(ratio_velocidad_centroide_v2_v1, 5.0)
        + 0.25 * min(ratio_velocidad_borde_alto_v2_v1, 5.0)
        + 0.25 * min(ratio_flujo_espectral_v2_v1, 5.0)
        + 0.15 * min(ratio_velocidad_dominante_v2_v1, 5.0)
    )


    # -------------------------------------------------
    # MÉTRICA PRINCIPAL: EXTREMOS FRECUENCIALES
    # -------------------------------------------------
    # No se usa la cantidad de energía acumulada como métrica principal.
    # Solo se usa la máscara activa para saber QUÉ frecuencias aparecen.
    #
    # V1 = 100 ms anteriores.
    # V2 = 100 ms justo antes del basal.
    #
    # Para cada ventana se calcula:
    # - frecuencia mínima activa;
    # - frecuencia máxima activa;
    # - rango frecuencial activo.
    #
    # Criterio:
    # terminación precoz sospechosa si V2 todavía alcanza frecuencias altas
    # mientras V1 tenía frecuencias más bajas.

    def calcular_extremos_frecuenciales(mascara_tiempo):
        """
        Calcula extremos frecuenciales activos de forma robusta.

        No promedia energía.
        Solo mira qué frecuencias están activas.

        Para evitar que una línea aislada de ruido cuente como frecuencia alta,
        una banda debe aparecer activa en al menos un porcentaje mínimo de la
        ventana. Si ninguna cumple, se usa cualquier banda activa como fallback.
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

        # Persistencia por banda: porcentaje de frames donde esa frecuencia aparece.
        persistencia = np.mean(A, axis=1)

        # Frecuencia activa robusta: debe aparecer al menos en el 15% de la ventana.
        umbral_persistencia = 0.15
        idx_bandas_robustas = np.where(persistencia >= umbral_persistencia)[0]

        if len(idx_bandas_robustas) == 0:
            # Fallback: cualquier banda activa.
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

        # Extremos frame a frame, para comprobar si el salto es consistente.
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

    # Ratio principal que pediste:
    # frecuencia alta alcanzada en la ventana final / frecuencia pequeña de la ventana anterior.
    ratio_extremo_precoz_fmax_v2_fmin_v1 = (
        float(f_max_v2 / (f_min_v1 + 1e-12)) if f_min_v1 > 0 else 0.0
    )

    # Ratios de control.
    ratio_fmax_v2_fmax_v1 = (
        float(f_max_v2 / (f_max_v1 + 1e-12)) if f_max_v1 > 0 else 0.0
    )

    ratio_fmin_v2_fmin_v1 = (
        float(f_min_v2 / (f_min_v1 + 1e-12)) if f_min_v1 > 0 else 0.0
    )

    ratio_rango_v2_v1 = (
        float(rango_v2 / (rango_v1 + 1e-12)) if rango_v1 > 0 else 0.0
    )

    # Score proporcional de precocidad por extremos.
    # Normalizado al rango total de frecuencias Mel.
    f_global_min = float(np.min(freqs))
    f_global_max = float(np.max(freqs))
    rango_global = max(f_global_max - f_global_min, 1e-12)

    score_extremos_frecuenciales = float(
        np.clip((f_max_v2 - f_min_v1) / rango_global, 0.0, 1.0)
    )

    # Score direccional:
    # positivo si la ventana final alcanza más frecuencia máxima que la anterior.
    score_direccion_fmax = float(
        (f_max_v2 - f_max_v1) / rango_global
    )

    # También guardo la media dB directa, sin pasar a lineal.
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
    """
    Única salida por pantalla: ratio sincero de 100 ms antes del ruido basal.
    """

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


# =====================================================
# RATIO DE VARIACIÓN FRECUENCIAL EN VENTANA DE 200 ms
# =====================================================

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
    """
    Calcula un ratio de VARIACIÓN FRECUENCIAL en el espectrograma Log-Mel denoised.

    No mide concentración de energía acumulada.

    Referencia:
    - Se recorre el Log-Mel desde atrás hacia la izquierda.
    - Se localiza la última energía visible.
    - La referencia es el primer punto negro justo después de esa última energía.

    Ventana:
    - Se toman los 200 ms inmediatamente anteriores a esa referencia.

    Métrica principal:
    - frecuencia mínima activa alcanzada en la ventana de 200 ms
    - frecuencia máxima activa alcanzada en la ventana de 200 ms
    - ratio = f_max_200ms / f_min_200ms

    Métrica dinámica:
    - primera mitad de la ventana: 100 ms antiguos
    - segunda mitad de la ventana: 100 ms finales
    - ratio dinámico = f_max_segunda_mitad / f_min_primera_mitad

    La idea es medir si el espectrograma pasa rápidamente de frecuencias bajas
    a frecuencias altas, no cuánta energía total hay.
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

    # Escala visual del espectrograma, solo para decidir si una frecuencia está activa.
    visual = np.clip(
        (L - float(suelo_db)) / (float(techo_db) - float(suelo_db)),
        0.0,
        1.0
    )

    energia_visual_frame = np.mean(visual, axis=0)

    # PEF de referencia para no buscar antes de la maniobra útil.
    t_pef_s = obtener_tiempo_pef_desde_eventos(eventos_picos)

    if t_pef_s is None:
        idx_pef = int(np.argmax(energia_visual_frame))
        t_pef_s = float(t[idx_pef])
    else:
        idx_pef = int(np.argmin(np.abs(t - float(t_pef_s))))

    # Paso temporal aproximado.
    if len(t) > 1:
        dt = float(np.median(np.diff(t)))
        if dt <= 0:
            dt = 0.01
    else:
        dt = 0.01

    # Estimar basal visual con la cola final.
    n_tail = max(3, int(round(0.300 / dt)))
    n_tail = min(n_tail, n_frames)

    basal_visual = float(np.median(energia_visual_frame[-n_tail:]))
    mad = float(np.median(np.abs(energia_visual_frame[-n_tail:] - basal_visual)))

    # Umbral de energía visible para saber si una frecuencia aparece.
    umbral_frame_energia = basal_visual + max(float(tolerancia_basal_visual), 3.0 * mad)
    umbral_frame_energia = min(float(umbral_frame_energia), 0.08)

    # Umbral por píxel/frecuencia.
    # No se usa para medir energía, solo para decidir frecuencia activa/no activa.
    umbral_pixel_activo = max(0.030, umbral_frame_energia - basal_visual)

    visual_sobre_basal = np.maximum(visual - basal_visual, 0.0)
    activo_pixel = visual_sobre_basal > umbral_pixel_activo

    # -------------------------------------------------
    # REFERENCIA TEMPORAL
    # -------------------------------------------------
    # DENOISED:
    #   se calcula recorriendo desde atrás hasta encontrar la última energía.
    #
    # ORIGINAL:
    #   puede usar EXACTAMENTE la misma referencia temporal del denoised.
    #
    # Así el ratio del Log-Mel original se calcula desde el mismo punto temporal.
    if t_referencia_fija_s is not None:
        t_ref_s = float(t_referencia_fija_s)
        idx_referencia = int(np.argmin(np.abs(t - t_ref_s)))
        idx_ultimo_con_energia = -1
        referencia = "misma_referencia_temporal_del_logmel_denoised"
    else:
        # -------------------------------------------------
        # RECORRIDO DESDE ATRÁS
        # -------------------------------------------------
        # Se localiza la última energía visible al recorrer desde el final.
        hay_energia_frame = np.any(activo_pixel, axis=0)

        n_energia_consecutiva = max(1, int(round(0.030 / dt)))  # 30 ms
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

    # -------------------------------------------------
    # VENTANA ÚNICA DE 200 ms
    # -------------------------------------------------
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

    # Frecuencias activas alcanzadas en TODA la ventana.
    idx_bandas_activas = np.where(np.any(activo_win, axis=1))[0]

    f_min_200 = float(np.min(freqs[idx_bandas_activas]))
    f_max_200 = float(np.max(freqs[idx_bandas_activas]))
    rango_200 = float(f_max_200 - f_min_200)

    ratio_fmax_fmin_200 = float(f_max_200 / (f_min_200 + 1e-12)) if f_min_200 > 0 else 0.0
    ratio_variacion_relativa_200 = float(rango_200 / (f_min_200 + 1e-12)) if f_min_200 > 0 else 0.0
    velocidad_variacion_hz_s = float(rango_200 / (float(duracion_ventana_s) + 1e-12))

    # -------------------------------------------------
    # DINÁMICA DENTRO DE LA VENTANA 200 ms
    # -------------------------------------------------
    # Primera mitad = 100 ms antiguos.
    # Segunda mitad = 100 ms finales.
    t_mitad = t_win_ini + float(duracion_ventana_s) / 2.0

    m_primera = (t >= t_win_ini) & (t < t_mitad)
    m_segunda = (t >= t_mitad) & (t < t_win_fin)

    activo_primera = activo_pixel[:, m_primera]
    activo_segunda = activo_pixel[:, m_segunda]

    def extremos_de_mascara(activo_sub):
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

    # Extremos frame a frame para medir salto brusco de frecuencia.
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

    # -------------------------------------------------
    # RATIO MIXTO: VARIACIÓN FRECUENCIAL + ENERGÍA
    # -------------------------------------------------
    # Ahora se mezcla:
    #   1) cuánto se abre la ventana en frecuencia;
    #   2) cuánta energía real hay en esas frecuencias;
    #   3) dando mucho más peso a partir de 1000 Hz.
    #
    # La variación frecuencial sigue siendo la base.
    # La energía solo refuerza el ratio si esas frecuencias realmente tienen presencia.

    f_global_min = float(np.min(freqs))
    f_global_max = float(np.max(freqs))
    rango_global = max(f_global_max - f_global_min, 1e-12)

    # -------------------------------------------------
    # BAREMO FRECUENCIA x dB RELATIVO
    # -------------------------------------------------
    # Esto es lo que has pedido:
    #
    #   baremo = frecuencia * peso_db_relativo
    #
    # donde:
    #   0 dB   -> peso 1.0, máxima ponderación
    #   -80 dB -> peso 0.0, mínima ponderación
    #
    # Por tanto una frecuencia alta solo aporta mucho si además tiene
    # un nivel dB relativo alto. No basta con que aparezca una frecuencia aislada.

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

    # Normalización 0-1:
    # máximo teórico = todas las bandas a 0 dB durante todos los frames.
    max_baremo_teorico = float(np.sum(freqs_safe) * max(1, np.sum(mascara_win)))

    baremo_normalizado_freq_db_200 = float(
        baremo_total_freq_db_200 / (max_baremo_teorico + 1e-12)
    ) if max_baremo_teorico > 0 else 0.0

    # Ratio PURO del baremo:
    # No se mezcla con fmax/fmin ni con ningún ratio frecuencial.
    # Es solo:
    #     sum(frecuencia * peso_db_relativo) / máximo_teórico
    ratio_solo_baremo_freq_db_200 = baremo_normalizado_freq_db_200

    # -------------------------------------------------
    # RATIO BAREMO CON MÁS PESO
    # -------------------------------------------------
    # El baremo puro suele salir bajo porque está normalizado contra un máximo
    # teórico muy exigente: todas las frecuencias a 0 dB durante toda la ventana.
    #
    # Para que sea más útil en comparación, se aplica un factor de escala.
    # Ejemplo:
    #   0.12 -> 0.72
    #   0.06 -> 0.36
    #
    # Así mantiene la diferencia relativa entre precoz y sano, pero no queda
    # todo comprimido cerca de cero.
    factor_peso_ratio_baremo = 6.0

    ratio_baremo_freq_db_200_mas_peso = float(
        np.clip(ratio_solo_baremo_freq_db_200 * factor_peso_ratio_baremo, 0.0, 1.0)
    )

    # Parte del baremo que viene de frecuencias >= 1000 Hz.
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

    # Ratio mixto principal:
    # mezcla variación frecuencial con este baremo frecuencia*dB relativo.
    # Si hay rango frecuencial pero está a muy bajo nivel, no sube tanto.
    factor_baremo_freq_db = float(
        1.0 + 3.0 * baremo_normalizado_freq_db_200 + 3.0 * proporcion_baremo_altas_1000
    )

    ratio_mixto_variacion_baremo_freq_db = float(
        ratio_fmax_fmin_200 * factor_baremo_freq_db
    )

    # Score 0-1 para comparar entre casos.
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

    # Mantengo nombres antiguos como compatibilidad, pero ahora salen de este baremo.
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
    """
    Mide si la energia del espectrograma se desploma justo despues del PEF.

    PEF = frame de maxima concentracion energetica del espectrograma.
    Se compara:
    - energia media 50-250 ms tras el PEF
    - energia media en los ultimos 200 ms de una ventana fija de 800 ms tras el PEF

    La decision usa caida y apagado espectral, no la duracion total del audio.
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
    """
    Única salida por pantalla:
    SOLO el ratio puro del baremo frecuencia*dB relativo.

    Fórmula:
        ratio = sum(frecuencia * peso_db_relativo) / máximo_teórico

    donde:
        0 dB   -> peso 1.0
        -80 dB -> peso 0.0
    """

    etiqueta = info.get("etiqueta_logmel", "")

    if not info.get("detectado", False):
        builtins.print(f"{etiqueta}: No detectado ({info.get('motivo', 'desconocido')})")
        return

    builtins.print(
        f"{etiqueta}: Ratio baremo freq*dBrel 200 ms MÁS PESO = "
        f"{info['ratio_baremo_freq_db_200_mas_peso']:.6f}"
    )


# =====================================================
# CONCLUSIÓN FINAL DE VALIDEZ CON PRIORIDAD CLÍNICA
# =====================================================

def evaluar_terminacion_precoz_brusca_logmel(g):
    """
    Decide terminacion precoz por brusquedad real del apagado:
    caida fuerte de energia en la senal y perdida brusca de bandas bajas
    del espectrograma Log-Mel.

    No usa duracion total de la prueba como criterio.
    """

    info = g.get("pendiente_logmel_final", {}) or {}

    def num(campo, defecto=0.0):
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
    pendiente_global = num("pendiente_global_bajas")

    post = g.get("caida_post_pef_logmel_den", {}) or {}

    def num_post(campo, defecto=0.0):
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
    """
    Apoyo espectral para esfuerzo multiple:
    busca un apagon breve de energia en bandas medias-altas del Log-Mel denoised
    seguido de recuperacion. No sustituye a la tos ni a los PEF temporales.
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
    """
    Conclusión final con orden de prioridad:

    1) Tos detectada              -> inválida por tos
    2) Dos PEF o más              -> inválida por esfuerzo múltiple
    3) PEF retrasado              -> inválida por esfuerzo insuficiente
    4) Terminación precoz         -> inválida por terminación precoz
    6) Nada de lo anterior        -> prueba totalmente válida

    La terminación precoz se decide por brusquedad de caída: descenso fuerte
    de energía y pérdida súbita de bandas bajas Log-Mel, no por duración.
    """

    eventos = d.get("eventos_picos", [])
    mascara_tos = d.get("mascara_tos", None)

    # -------------------------------------------------
    # 1) TOS
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 2) ESFUERZO MÚLTIPLE: DOS PEF O MÁS
    # -------------------------------------------------
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
    # La decisión de esfuerzo múltiple se toma con la curva temporal.
    # El Log-Mel denoised puede separar ondulaciones internas del mismo soplido
    # y generar dobles PEF visuales si se usa solo. Por eso se conserva como
    # apoyo gráfico, pero no dispara por sí solo la invalidez por múltiple.
    n_pef_multiple = n_pef_temporal
    n_pef = n_pef_temporal if n_pef_temporal > 0 else n_pef_logmel

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

    # -------------------------------------------------
    # TERMINACION PRECOZ: se calcula aquí, pero se decide al final.
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 3) PEF RETRASADO: ESFUERZO INSUFICIENTE
    # -------------------------------------------------
    # Se mide desde el inicio10 del evento hasta el pico PEF.
    # Esto es mejor que usar el tiempo absoluto del audio, porque puede haber
    # pequeñas zonas previas de silencio o recorte.
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

    # -------------------------------------------------
    # 4) TERMINACIÓN PRECOZ: BRUSQUEDAD DE CAÍDA LOG-MEL
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 5) VÁLIDA
    # -------------------------------------------------
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
    """
    Imprime solo la conclusión final, con el motivo prioritario.
    """

    builtins.print("\nCONCLUSIÓN FINAL")
    builtins.print("-" * 70)
    builtins.print(f"{nombre}: {conclusion['motivo']}")
    builtins.print(f"Detalle: {conclusion['detalle']}")
    builtins.print("-" * 70)

def preparar_graficas_fuente(d):
    fs_raw, fs_den, y_raw, y_den = d["fs_raw"], d["fs"], d["original_raw_graficas"], d["denoised"]
    
    t_raw = np.arange(len(y_raw)) / fs_raw
    t_den = np.arange(len(y_den)) / fs_den

    # -------------------------------------------------
    # ARRANQUE EN t=0 DESDE AMPLITUD PRÁCTICAMENTE NULA
    # -------------------------------------------------
    # Solo afecta a la visualización temporal. No cambia la detección ni el audio exportado.
    y_raw = forzar_arranque_en_suelo_sin_blanco(t_raw, y_raw, duracion_rampa_s=0.035)
    y_den = forzar_arranque_en_suelo_sin_blanco(t_den, y_den, duracion_rampa_s=0.035)

    # Señal cruda representada como RMS de 25 ms.
    # No se aplica denoising, STFT, Mel, Log-Mel ni filtro frecuencial.
    y_raw_suavizada_25ms = suavizar_senal_cruda_25ms(y_raw, fs_raw)

    env_raw = calcular_envolvente_temporal(y_raw, fs_raw, mascara_tos=None, normalizar_entrada=False)
    env_den = calcular_envolvente_temporal(y_den, fs_den, mascara_tos=d.get("mascara_tos"), normalizar_entrada=False)

    # También forzamos que la envolvente empiece desde el suelo en t=0.
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

    # -------------------------------------------------
    # MEL / LOG-MEL CON REFERENCIA COMÚN
    # -------------------------------------------------
    # La referencia se toma del original/general.
    # El denoised se normaliza contra esa referencia, no contra sí mismo.
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

    # -------------------------------------------------
    # QUITAR REPUNTE FALSO FINAL DEL LOG-MEL DENOISED
    # -------------------------------------------------
    # Si al final la señal temporal denoised tiene RMS prácticamente nula,
    # cualquier columna de Log-Mel que aparezca ahí se considera ruido residual
    # o artefacto de borde/normalización, no soplido real.
    logmel_den = quitar_repunte_final_logmel_por_actividad(
        logmel_den,
        t_mel_den,
        y_den,
        fs_den,
        umbral_rms=0.025,
        ventana_rms_s=0.025,
        zona_final_s=0.50
    )

    # Frecuencia central aproximada de cada banda Mel para etiquetar el eje vertical.
    freqs_mel_raw = frecuencias_centro_bandas_mel(fs_raw, n_mels)
    freqs_mel_den = frecuencias_centro_bandas_mel(fs_den, n_mels)

    # -------------------------------------------------
    # SINCRONIZACIÓN TEMPORAL VISUAL
    # -------------------------------------------------
    # scipy devuelve tiempos de STFT/Mel centrados en ventana, por eso a veces
    # empiezan en 0.02-0.03 s y aparece espacio blanco. Para visualizar, todos
    # los ejes empiezan en t=0.
    def _t0(t):
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
    # Los bordes de Mel/Log-Mel ya se calculan desde 0 hasta la duración real.
    # No se desplazan ni se estiran.

    # -------------------------------------------------
    # ENVOLVENTE DE LA LOG-MEL DENOISED
    # -------------------------------------------------
    # Se calcula sobre la Log-Mel denoised y se suaviza con la misma lógica
    # que la envolvente temporal.
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
    
    # Pasamos la máscara de tos para ignorarla en el cálculo del score si está disponible
    # y usamos la señal remuestreada (denoising_entrada) para que su longitud coincida
    # con la longitud de la máscara. Esto evita que la máscara no se aplique por
    # diferencias de longitud entre la señal original y la señal procesada.
    score_terminacion = calcular_score_terminacion_precoz(
        d["denoising_entrada"], d["fs"], mascara_tos=d.get("mascara_tos")
    )

    # -------------------------------------------------
    # ENERGÍA ABSOLUTA EN EL ÚLTIMO CUARTO DE SEGUNDO
    # -------------------------------------------------
    # Se calcula sobre la señal original y sobre la denoised para comparar
    # cuánta energía queda al final de la espiración.
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

    # -------------------------------------------------
    # ENERGÍA ABSOLUTA MEDIA DESDE EL PRIMER PEF HASTA EL FINAL
    # -------------------------------------------------
    # Es promedio, no suma, para que no dependa de la duración de cada audio.
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
    """
    SOLO PARA VISUALIZACIÓN.

    Mantiene t=0, pero evita que la señal/curva aparezca ya con altura.
    No añade espacio blanco: simplemente hace que los primeros milisegundos
    suban desde amplitud prácticamente nula.

    No modifica detección, máscara, PEF, tos ni audio exportado.
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

    # Quitamos el offset inicial local para que empiece pegado al suelo/cero.
    y0 = y[0]
    y[:n_rampa] = y[:n_rampa] - y0

    # Rampa desde 0 hasta la señal real. No añade silencio, solo evita inicio alto.
    rampa = np.linspace(0.0, 1.0, n_rampa)
    y[:n_rampa] = y[:n_rampa] * rampa

    return y


def forzar_envolvente_inicio_suelo_sin_blanco(t, env, duracion_rampa_s=0.040):
    """
    SOLO PARA VISUALIZACIÓN DE ENVOLVENTES.

    Hace que la envolvente empiece en 0 y suba progresivamente durante
    unos pocos milisegundos, sin desplazar el eje temporal ni añadir blancos.
    """

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
    """
    Extensión temporal para imágenes.

    Forzada a empezar SIEMPRE en 0 para que no aparezca una banda blanca
    inicial en STFT/Mel/Log-Mel cuando scipy devuelve tiempos centrados
    en la ventana.
    """
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
    """
    Ajusta el punto rojo al máximo visible de la curva que se está dibujando.

    Motivo:
    La detección se hace sobre una curva filtrada agresiva, pero en la gráfica
    se estaba pintando el índice directamente sobre env_den. Por eso algún punto
    podía caer en una zona baja de la curva visible.

    Esta función busca, alrededor del pico detectado, el máximo real de la curva
    visible y devuelve ese tiempo y valor.
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


def dibujar_ventana_unica(datos):
    """
    Visualización comparativa.

    Permite 1 o 2 audios:
    - si se cargan dos, se muestran en dos columnas;
    - si solo se carga uno, se muestra una columna.

    Se conservan 6 gráficas:
    1) Señal original/denoised
    2) RMS 25 ms de señal cruda
    3) Envolvente temporal original/denoised
    4) Log-Mel original
    5) Log-Mel denoised
    6) Envolvente de Log-Mel denoised

    Se eliminan de la visualización:
    - Env. STFT
    - STFT original
    - STFT denoised
    - Mel original
    - Mel denoised
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

                # Marcadores:
                # - PEF: punto rojo.
                # - Tos: cruz verde.
                # - Artefactos: no se pintan.
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
                # Forma de la señal cruda mediante RMS de 25 ms.
                ax.plot(
                    g["t_raw"],
                    g["y_raw_suavizada_25ms"],
                    lw=0.75,
                    label="RMS cruda 25 ms"
                )
                # Al ser RMS/valor energético, no tiene parte negativa.
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

                # Marcadores sobre la envolvente:
                # - PEF: punto rojo.
                # - Tos: cruz verde.
                # - Artefactos: no se pintan.
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
                # Log-Mel original
                ultimo_im_db = ax.pcolormesh(
                    g["t_mel_raw_edges"],
                    np.arange(n_mels + 1),
                    g["logmel_raw"],
                    shading="auto",
                    cmap="magma",
                    vmin=-80,
                    vmax=0
                )

                aplicar_eje_frecuencias_mel(ax, g["freqs_mel_raw"], n_mels)

            elif f == 4:
                # Log-Mel denoised
                ultimo_im_db = ax.pcolormesh(
                    g["t_mel_den_edges"],
                    np.arange(n_mels + 1),
                    g["logmel_den"],
                    shading="auto",
                    cmap="magma",
                    vmin=-80,
                    vmax=0
                )

                aplicar_eje_frecuencias_mel(ax, g["freqs_mel_den"], n_mels)

            elif f == 5:
                # -------------------------------------------------
                # ENVOLVENTE DE LA LOG-MEL DENOISED
                # -------------------------------------------------
                # Nuevo cuadro independiente debajo de Log-Mel denoised.
                # Lleva la misma detección visual que la envolvente temporal:
                # PEF rojo y tos verde.
                ax.plot(
                    g["t_env_logmel_den"],
                    g["env_logmel_den"],
                    lw=0.9,
                    label="Env Log-Mel den"
                )

                # Sombreado de tos trasladado al eje temporal de la envolvente Log-Mel.
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

                # Misma detección que en la envolvente temporal:
                # - PEF: punto rojo
                # - Tos: cruz verde
                # - Artefactos: no se pintan
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

            # Ejes sincronizados para poder comparar.
            ax.set_xlim(0.0, tmax_comun)
            ax.margins(x=0)

            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=7)

            if f < nfilas - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Tiempo (s)", fontsize=8)

            ejes_temporales.append(ax)

    # Etiquetas de filas en el lateral izquierdo.
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


# =====================================================
# MAIN
# =====================================================

def main():
    root = Tk()
    root.withdraw()

    # Puedes cargar uno o dos audios.
    # Si cancelas uno de los dos, se muestra solo el que hayas seleccionado.
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

    # =====================================================
    # ÚNICA SALIDA POR PANTALLA
    # =====================================================
    # Solo se muestra el ratio sincero de 100 ms justo antes del ruido basal
    # y los 100 ms anteriores a esa ventana.
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

# =====================================================
# PROGRAMA AUXILIAR INDEPENDIENTE
# =====================================================
# Este bloque permite analizar un unico WAV eligiendo la fuente de grabacion.
# No importa ningun script local: todas las funciones necesarias estan en este
# mismo archivo. Las unicas dependencias externas son las librerias usadas por
# el programa final: numpy, scipy, matplotlib y tkinter.

import re
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk


FUENTES_AUXILIARES = (
    ("Estetoscopio", "ESTETOSCOPIO"),
    ("Microfono del PC", "PC"),
    ("Movil", "MOVIL"),
)


class SelectorFuenteAuxiliar(simpledialog.Dialog):
    """Selecciona el perfil de procesamiento aplicado al WAV."""

    def body(self, master):
        self.title("Fuente de grabacion")
        self.resizable(False, False)
        ttk.Label(master, text="Fuente de grabacion").grid(
            row=0, column=0, sticky="w", padx=12, pady=(12, 6)
        )
        self.selector = ttk.Combobox(
            master,
            values=[etiqueta for etiqueta, _ in FUENTES_AUXILIARES],
            state="readonly",
            width=28,
        )
        self.selector.current(0)
        self.selector.grid(row=1, column=0, padx=12, pady=(0, 12))
        return self.selector

    def buttonbox(self):
        botones = ttk.Frame(self)
        ttk.Button(botones, text="Seleccionar audio", command=self.ok).pack(
            side="left", padx=(0, 8)
        )
        ttk.Button(botones, text="Cancelar", command=self.cancel).pack(side="left")
        botones.pack(padx=16, pady=(0, 12))
        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def validate(self):
        return 0 <= self.selector.current() < len(FUENTES_AUXILIARES)

    def apply(self):
        self.result = FUENTES_AUXILIARES[self.selector.current()][1]


def seleccionar_fuente_auxiliar(parent):
    """Devuelve MOVIL, PC o ESTETOSCOPIO; None si se cancela."""
    return SelectorFuenteAuxiliar(parent).result


def aplicar_denoising_alineado_auxiliar(fs, y, N, S, window, config):
    """Reconstruye la senal denoised con STFT/ISTFT temporalmente alineadas.

    La mascara espectral se mantiene igual que en el programa final:

        G[f,k] = max(gain_floor, (1-alpha+alpha*M[f,k]))
        Y[f,k] = G[f,k] * |X[f,k]| * exp(j*angle(X[f,k]))

    donde M[f,k] compara energia persistente y residual. Esta funcion se usa
    para que la grafica auxiliar muestre correctamente el Log-Mel denoised.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 2:
        y = np.mean(y, axis=1)
    if y.ndim != 1 or not y.size or not np.all(np.isfinite(y)):
        raise ValueError("La senal debe contener muestras finitas.")
    if fs <= 0 or N < 1 or not 0 <= S < 1:
        raise ValueError("Configuracion temporal no valida.")
    peak = np.max(np.abs(y))
    if peak == 0:
        raise ValueError("La senal no tiene contenido util.")
    y = y / peak

    nperseg = min(int(N), len(y))
    window = np.asarray(window, dtype=np.float64)
    if len(window) != N:
        raise ValueError("La ventana debe tener N muestras.")
    if len(y) == 1:
        return y.copy()
    if nperseg < N:
        window = signal.windows.hamming(nperseg)

    noverlap = min(int(nperseg * S), nperseg - 1)
    nfft = next_fast_len(2 * nperseg)
    _, times, spectrum = signal.stft(
        y, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap,
        nfft=nfft, detrend=False, boundary="zeros", padded=True,
        scaling="spectrum",
    )
    magnitude = np.abs(spectrum)

    median_size = int(config["lmediant"])
    if median_size < 1:
        raise ValueError("La mediana requiere al menos una trama.")
    if magnitude.shape[1] < median_size:
        extended = np.pad(
            magnitude,
            ((0, 0), (median_size // 2, (median_size - 1) // 2)),
            mode="symmetric",
        )
        windows = np.lib.stride_tricks.sliding_window_view(
            extended, median_size, axis=1
        )
        median = np.partition(windows, median_size // 2, axis=-1)[
            ..., median_size // 2
        ]
    else:
        median = ndimage.median_filter(
            magnitude, size=(1, median_size), mode="reflect"
        )

    persistent = np.minimum(median, magnitude)
    residual = np.maximum(magnitude - persistent, 0)
    mask = persistent ** 2 / (persistent ** 2 + residual ** 2 + np.finfo(float).eps)
    alpha = config["denoising_strength"]
    filtered = np.maximum(
        (1 - alpha + alpha * mask) * magnitude,
        config["gain_floor"] * magnitude,
    )

    if preservar_tos:
        sample_mask = detectar_mascara_tos_en_muestras(fs, y, config)
        frame_mask = convertir_mascara_muestras_a_frames(sample_mask, times, fs, len(y))
        filtered[:, frame_mask] = np.maximum(
            filtered[:, frame_mask], tos_gain_floor * magnitude[:, frame_mask]
        )

    _, result = signal.istft(
        filtered * np.exp(1j * np.angle(spectrum)),
        fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
        boundary=True, scaling="spectrum",
    )
    if len(result) < len(y):
        raise RuntimeError("La reconstruccion no cubre todo el registro.")
    if not np.all(np.isfinite(result)):
        raise RuntimeError("La reconstruccion contiene muestras no finitas.")
    return np.clip(result[:len(y)], -1.0, 1.0)


def preparar_comparacion_logmel_auxiliar(datos):
    """Prepara Log-Mel de entrada y Log-Mel denoised con referencia comun.

    La referencia comun evita comparar dos imagenes normalizadas por separado:

        L_raw[b,k] = log10(E_raw[b,k] + eps)
        L_den[b,k] = log10(E_den[b,k] + eps)

    Ambas se representan respecto al maximo de energia Mel de la entrada.
    """
    fs = datos["fs"]
    raw = calcular_mel_logmel(
        datos["denoising_entrada"], fs, normalizar_entrada=False
    )
    den = calcular_mel_logmel(
        datos["denoised"], fs, ref_mel_max=raw[3], normalizar_entrada=False
    )
    graficas = dict(datos["graficas"])
    for suffix, values in (("raw", raw), ("den", den)):
        graficas["t_mel_" + suffix] = values[0]
        graficas["mel_" + suffix] = values[1]
        graficas["logmel_" + suffix] = values[2]
        graficas["t_mel_" + suffix + "_edges"] = values[4]
        graficas["freqs_mel_" + suffix] = frecuencias_centro_bandas_mel(fs, n_mels)
    vista = dict(datos)
    vista["graficas"] = graficas
    return vista


def calcular_espectrograma_original_auxiliar(datos):
    """Calcula una STFT de inspeccion sin remuestrear el WAV original."""
    fs = float(datos["fs_raw"])
    y = np.asarray(datos["original_raw_graficas"], dtype=np.float64)
    if fs <= 0 or not np.isfinite(fs) or y.ndim != 1 or not y.size:
        raise ValueError("La vista original requiere una senal mono y una fs valida.")
    if not np.isfinite(y).all():
        raise ValueError("La senal original contiene muestras no finitas.")

    nperseg = min(len(y), max(8, int(round(0.064 * fs))))
    nfft = next_fast_len(max(8, nperseg))
    frequencies, times, spectrum = signal.stft(
        y, fs=fs, window=signal.windows.hann(nperseg, sym=False),
        nperseg=nperseg, noverlap=nperseg // 2, nfft=nfft,
        detrend=False, boundary="zeros", padded=True, scaling="spectrum",
    )
    duration = len(y) / fs
    keep = times < duration
    times, spectrum = times[keep], spectrum[:, keep]
    power = np.abs(spectrum) ** 2
    reference = float(power.max())
    db = np.clip(
        10 * np.log10(np.maximum(power / max(reference, np.finfo(float).tiny), 1e-8)),
        -80, 0,
    )
    return {
        "fs": fs,
        "nperseg": nperseg,
        "nfft": nfft,
        "time_edges": np.r_[0, (times[:-1] + times[1:]) / 2, duration],
        "frequency_edges": np.r_[0, (frequencies[:-1] + frequencies[1:]) / 2, fs / 2],
        "frequencies": frequencies,
        "db": db,
        "reference": reference,
    }


def analizar_audio_auxiliar(ruta_audio, fuente, carpeta_salida):
    """Analiza un unico WAV con el procesamiento original del programa final."""
    if fuente not in {clave for _, clave in FUENTES_AUXILIARES}:
        raise ValueError("Fuente no reconocida: " + str(fuente))
    ruta_audio = Path(ruta_audio)
    if not ruta_audio.is_file() or ruta_audio.suffix.lower() != ".wav":
        raise ValueError("Selecciona un archivo WAV existente.")
    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    datos = cargar_y_procesar(str(ruta_audio), fuente, str(carpeta_salida))
    graficas = preparar_graficas_fuente(datos)
    datos["graficas"] = graficas
    conclusion = calcular_conclusion_final_prueba(
        datos,
        graficas,
        umbral_pef_retrasado_s=0.150,
        umbral_terminacion_precoz=0.60,
    )
    return datos, conclusion


def organizar_espectrogramas_auxiliar(figura, datos):
    """Reordena la ventana para visualizar entrada y Log-Mel denoised."""
    if len(figura.axes) < 6:
        return
    ejes = figura.axes[:6]
    barra_comparacion = figura.axes[6] if len(figura.axes) > 6 else None
    incluir_original = datos["nombre_dispositivo"] in {"PC", "MOVIL"}
    alturas = [1, 1, 1, 1.8, 1.5, 1.5, 1.1] if incluir_original else [1, 1, 1, 1.5, 1.5, 1.1]
    filas = [0, 1, 2, 4, 5, 6] if incluir_original else list(range(6))
    figura.set_size_inches(16, 20 if incluir_original else 18, forward=False)
    rejilla = figura.add_gridspec(
        len(alturas), 1, height_ratios=alturas,
        left=0.095, right=0.89, top=0.90, bottom=0.06, hspace=0.70,
    )
    for eje, fila in zip(ejes, filas):
        eje.set_subplotspec(rejilla[fila])

    fs = datos["fs"]
    ejes[3].set_title(f"Log-Mel entrada del denoising | fs = {fs:g} Hz", loc="left", fontsize=8, pad=5)
    ejes[4].set_title(f"Log-Mel denoised | fs = {fs:g} Hz", loc="left", fontsize=8, pad=5)
    if barra_comparacion is not None:
        base, techo = ejes[4].get_position().y0, ejes[3].get_position().y1
        barra_comparacion.set_position([0.915, base, 0.012, techo - base])
        barra_comparacion.set_ylabel("Nivel relativo a la entrada (dB)", fontsize=8)

    if not incluir_original:
        return

    original = calcular_espectrograma_original_auxiliar(datos)
    eje = figura.add_subplot(rejilla[3], sharex=ejes[3])
    imagen = eje.pcolormesh(
        original["time_edges"], original["frequency_edges"], original["db"],
        shading="flat", cmap="magma", vmin=-80, vmax=0, rasterized=True,
    )
    eje.set_ylabel("STFT original\nFrecuencia (Hz)", fontsize=8)
    eje.set_title(
        f"Original | fs = {original['fs']:g} Hz | banda 0-{original['fs']/2:g} Hz",
        loc="left", fontsize=8, pad=5,
    )
    eje.set_ylim(0, original["fs"] / 2)
    eje.set_yticks([i * original["fs"] / 8 for i in range(5)])
    eje.set_xlim(ejes[3].get_xlim())
    eje.callbacks.connect("xlim_changed", lambda origen: ejes[3].set_xlim(origen.get_xlim()))
    eje.tick_params(labelsize=7, labelbottom=False)
    posicion = eje.get_position()
    barra_original = figura.add_axes([0.915, posicion.y0, 0.012, posicion.height])
    figura.colorbar(imagen, cax=barra_original).set_label("Original (dB relativos)", fontsize=8)
    barra_original.tick_params(labelsize=7)
    figura._eje_espectrograma_original = eje
    figura._vista_espectrograma_original = original


def mostrar_resultados_auxiliar(datos, conclusion):
    """Imprime la conclusion y muestra la figura original con Log-Mel denoised."""
    fuente = datos["nombre_dispositivo"]
    graficas = datos["graficas"]
    builtins.print("\n--- " + fuente + " ---")
    for clave in (
        "ratio_variacion_frecuencial_200ms_logmel_den",
        "ratio_variacion_frecuencial_200ms_logmel_raw",
    ):
        imprimir_SOLO_ratio_baremo_freq_db(graficas.get(clave, {}))

    salida = dict(conclusion)
    if isinstance(salida.get("motivo"), str):
        salida["motivo"] = salida["motivo"].lower()
    if isinstance(salida.get("detalle"), str):
        salida["detalle"] = re.sub(
            r"\b(tos|esfuerzo multiple|esfuerzo insuficiente|terminacion precoz|valida)\b",
            lambda m: m.group().lower(), salida["detalle"], flags=re.IGNORECASE,
        )
    imprimir_conclusion_final_prueba(fuente, salida)

    figura = dibujar_ventana_unica({fuente: datos})
    return figura


def main():
    """Selecciona fuente, WAV y carpeta de salida para analizar una sola senal."""
    root = Tk()
    root.withdraw()
    try:
        fuente = seleccionar_fuente_auxiliar(root)
        if fuente is None:
            return
        ruta_audio = filedialog.askopenfilename(
            parent=root,
            title="Selecciona un audio de " + fuente,
            filetypes=[("Archivos WAV", "*.wav")],
            multiple=False,
        )
        if not ruta_audio:
            return
        carpeta_salida = filedialog.askdirectory(
            parent=root, title="Carpeta de guardado"
        )
        if not carpeta_salida:
            return
        datos, conclusion = analizar_audio_auxiliar(ruta_audio, fuente, carpeta_salida)
    except Exception as error:
        messagebox.showerror("No se pudo analizar el audio", str(error), parent=root)
        return
    finally:
        root.destroy()

    mostrar_resultados_auxiliar(datos, conclusion)
    plt.show()


if __name__ == "__main__":
    main()

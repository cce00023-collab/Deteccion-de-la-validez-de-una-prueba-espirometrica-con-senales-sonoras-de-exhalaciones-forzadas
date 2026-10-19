import tkinter as tk
from tkinter import ttk, filedialog
import sounddevice as sd
import soundfile as sf
import threading
import queue
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
from scipy.signal import spectrogram
import os


class AudioRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplicación de audio doble")
        self.root.geometry("1050x820")
        self.root.configure(bg="#f0f2f5")

        self.stethoscope_index = tk.StringVar()
        self.pc_mic_index = tk.StringVar()
        self.output_folder = tk.StringVar(value=os.getcwd())

        self.is_recording = False
        self.recording_threads = []

        self.queue_S = queue.Queue()
        self.queue_PC = queue.Queue()

        self.recorded_file_S = None
        self.recorded_file_PC = None
        self.error_message = None

        self.sampling_rate = 16000

        self.data_S = np.empty((0,))
        self.data_PC = np.empty((0,))

        self.audio_level_S = tk.DoubleVar(value=0)
        self.audio_level_PC = tk.DoubleVar(value=0)

        self.start_time = None

        self.create_widgets()

    def create_widgets(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "TButton",
            font=("Segoe UI", 10, "bold"),
            padding=8,
            background="#1877f2",
            foreground="white"
        )

        style.map(
            "TButton",
            background=[("active", "#166fe5"), ("disabled", "#bcc0c4")]
        )

        style.configure("TLabel", font=("Segoe UI", 10), background="#ffffff")
        style.configure("TCombobox", font=("Segoe UI", 10))
        style.configure("TEntry", font=("Segoe UI", 10))

        main_frame = tk.Frame(self.root, bg="#f0f2f5")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=18, pady=18)

        card = tk.Frame(
            main_frame,
            bg="#ffffff",
            highlightbackground="#dddfe2",
            highlightthickness=1
        )
        card.pack(fill=tk.X, pady=(0, 12), ipady=10)

        device_frame = tk.Frame(card, bg="#ffffff")
        device_frame.pack(pady=10, padx=16, fill=tk.X)

        ttk.Label(device_frame, text="Entrada 1 - Estetoscopio (_S):").grid(
            row=0, column=0, sticky="w", padx=5, pady=5
        )

        self.stethoscope_menu = ttk.Combobox(
            device_frame,
            textvariable=self.stethoscope_index,
            width=75,
            state="readonly"
        )
        self.stethoscope_menu.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(device_frame, text="Entrada 2 - Micrófono PC (_PC):").grid(
            row=1, column=0, sticky="w", padx=5, pady=5
        )

        self.pc_mic_menu = ttk.Combobox(
            device_frame,
            textvariable=self.pc_mic_index,
            width=75,
            state="readonly"
        )
        self.pc_mic_menu.grid(row=1, column=1, padx=5, pady=5)

        self.refresh_button = ttk.Button(
            device_frame,
            text="Actualizar",
            command=self.refresh_devices
        )
        self.refresh_button.grid(row=0, column=2, rowspan=2, padx=8)

        folder_frame = tk.Frame(card, bg="#ffffff")
        folder_frame.pack(pady=6, padx=16, fill=tk.X)

        ttk.Label(folder_frame, text="Carpeta de guardado:").grid(
            row=0, column=0, sticky="w", padx=5
        )

        self.folder_entry = ttk.Entry(
            folder_frame,
            textvariable=self.output_folder,
            width=75
        )
        self.folder_entry.grid(row=0, column=1, padx=5)

        self.folder_button = ttk.Button(
            folder_frame,
            text="Examinar",
            command=self.select_output_folder
        )
        self.folder_button.grid(row=0, column=2, padx=5)

        control_frame = tk.Frame(card, bg="#ffffff")
        control_frame.pack(pady=12)

        self.start_button = ttk.Button(
            control_frame,
            text="Iniciar grabación",
            command=self.start_recording
        )
        self.start_button.grid(row=0, column=0, padx=8)

        self.stop_button = ttk.Button(
            control_frame,
            text="Detener",
            command=self.stop_recording,
            state="disabled"
        )
        self.stop_button.grid(row=0, column=1, padx=8)

        self.save_spec_button = ttk.Button(
            control_frame,
            text="Guardar espectrogramas",
            command=self.save_spectrogram_images
        )
        self.save_spec_button.grid(row=0, column=2, padx=8)

        info_frame = tk.Frame(card, bg="#ffffff")
        info_frame.pack(pady=(0, 8))

        self.timer_label = ttk.Label(info_frame, text="Tiempo: 00:00")
        self.timer_label.grid(row=0, column=0, padx=15)

        ttk.Label(info_frame, text="Nivel _S:").grid(row=0, column=1, padx=5)

        self.level_bar_S = ttk.Progressbar(
            info_frame,
            variable=self.audio_level_S,
            maximum=1.0,
            length=180
        )
        self.level_bar_S.grid(row=0, column=2, padx=5)

        ttk.Label(info_frame, text="Nivel _PC:").grid(row=0, column=3, padx=5)

        self.level_bar_PC = ttk.Progressbar(
            info_frame,
            variable=self.audio_level_PC,
            maximum=1.0,
            length=180
        )
        self.level_bar_PC.grid(row=0, column=4, padx=5)

        spectrogram_card = tk.Frame(
            main_frame,
            bg="#ffffff",
            highlightbackground="#dddfe2",
            highlightthickness=1
        )
        spectrogram_card.pack(fill=tk.BOTH, expand=True)

        self.fig, (self.ax_S, self.ax_PC) = plt.subplots(2, 1, figsize=(11, 7))

        self.ax_S.set_title("Espectrograma estetoscopio (_S)")
        self.ax_S.set_xlabel("Tiempo [s]")
        self.ax_S.set_ylabel("Frecuencia [Hz]")
        self.ax_S.set_ylim(0, 8000)

        self.ax_PC.set_title("Espectrograma micrófono PC (_PC)")
        self.ax_PC.set_xlabel("Tiempo [s]")
        self.ax_PC.set_ylabel("Frecuencia [Hz]")
        self.ax_PC.set_ylim(0, 8000)

        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=spectrogram_card)
        self.canvas.get_tk_widget().pack(pady=10, fill=tk.BOTH, expand=True)

        self.status_label = tk.Label(
            main_frame,
            text="Lista para grabar",
            anchor="center",
            font=("Segoe UI", 10),
            bg="#f0f2f5",
            fg="#65676b"
        )
        self.status_label.pack(pady=8)

        self.refresh_devices()

    def refresh_devices(self):
        devices = self.get_input_device_names()

        self.stethoscope_menu["values"] = devices
        self.pc_mic_menu["values"] = devices

        if devices:
            self.stethoscope_menu.current(0)
            self.pc_mic_menu.current(0)

            self.stethoscope_index.set(devices[0].split(":")[0])
            self.pc_mic_index.set(devices[0].split(":")[0])

            self.status_label.config(text="Dispositivos de entrada actualizados.")
        else:
            self.stethoscope_index.set("")
            self.pc_mic_index.set("")
            self.status_label.config(text="No se han encontrado dispositivos de entrada.")

    def get_input_device_names(self):
        devices = sd.query_devices()
        device_names = []

        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                name = d["name"]
                channels = d["max_input_channels"]
                default_sr = int(d["default_samplerate"])

                device_info = (
                    f"{i}: {name} "
                    f"- Canales de entrada: {channels} "
                    f"- Frecuencia por defecto: {default_sr} Hz"
                )

                device_names.append(device_info)
                print(device_info)

        return device_names

    def select_output_folder(self):
        folder = filedialog.askdirectory()

        if folder:
            self.output_folder.set(folder)

    def generate_base_filename(self):
        return datetime.now().strftime("%Y%m%d%H%M%S")

    def callback_S(self, indata, frames, time, status):
        if status:
            print("Estado _S:", status)

        self.queue_S.put(indata.copy())

    def callback_PC(self, indata, frames, time, status):
        if status:
            print("Estado _PC:", status)
 
        self.queue_PC.put(indata.copy())

    def record_audio(self, device_index, suffix, audio_queue, callback):
        base_filename = self.base_filename

        file_path = os.path.join(
            self.output_folder.get(),
            f"{base_filename}_{suffix}.wav"
        )

        if suffix == "S":
            self.recorded_file_S = file_path
        elif suffix == "PC":
            self.recorded_file_PC = file_path

        try:
            device_info = sd.query_devices(device_index, "input")

            if int(device_info["max_input_channels"]) < 1:
                raise ValueError(f"El dispositivo _{suffix} no tiene canales de entrada válidos.")

            channels_to_use = 1

            sd.check_input_settings(
                device=device_index,
                channels=channels_to_use,
                samplerate=self.sampling_rate
            )

            print(f"Grabando _{suffix} con:")
            print("Dispositivo:", device_info["name"])
            print("Frecuencia de muestreo:", self.sampling_rate)
            print("Canales:", channels_to_use)

            with sf.SoundFile(
                file_path,
                mode="w",
                samplerate=self.sampling_rate,
                channels=1
            ) as file:

                with sd.InputStream(
                    device=device_index,
                    channels=channels_to_use,
                    samplerate=self.sampling_rate,
                    dtype="float32",
                    callback=callback
                ):

                    while self.is_recording:
                        try:
                            data = audio_queue.get(timeout=0.2)
                        except queue.Empty:
                            continue

                        data = data.flatten()
                        file.write(data)

                        rms = np.sqrt(np.mean(data ** 2))
                        level = min(rms * 20, 1.0)

                        if suffix == "S":
                            self.audio_level_S.set(level)
                            self.data_S = np.concatenate((self.data_S, data))
                        elif suffix == "PC":
                            self.audio_level_PC.set(level)
                            self.data_PC = np.concatenate((self.data_PC, data))

        except Exception as e:
            self.error_message = str(e)
            print(f"Error de grabación _{suffix}: {e}")

            self.root.after(
                0,
                lambda: self.status_label.config(
                    text=f"Error de grabación _{suffix}: {self.error_message}"
                )
            )

            self.is_recording = False

    def draw_spectrogram(self, ax, data, title):
        if len(data) <= 1024:
            return

        f, t, Sxx = spectrogram(
            data,
            fs=self.sampling_rate,
            nperseg=1024,
            noverlap=512,
            scaling="spectrum"
        )

        Sxx = 10 * np.log10(Sxx + 1e-10)

        freq_mask = f <= 8000
        f = f[freq_mask]
        Sxx = Sxx[freq_mask, :]

        ax.clear()

        ax.imshow(
            Sxx,
            aspect="auto",
            origin="lower",
            extent=[t.min(), t.max(), f.min(), f.max()]
        )

        ax.set_title(title)
        ax.set_xlabel("Tiempo [s]")
        ax.set_ylabel("Frecuencia [Hz]")
        ax.set_ylim(0, 8000)

    def update_spectrograms(self):
        self.draw_spectrogram(
            self.ax_S,
            self.data_S,
            "Espectrograma estetoscopio (_S)"
        )

        self.draw_spectrogram(
            self.ax_PC,
            self.data_PC,
            "Espectrograma micrófono PC (_PC)"
        )

        self.fig.tight_layout()
        self.canvas.draw_idle()

        if self.is_recording:
            self.root.after(1000, self.update_spectrograms)

    def update_timer(self):
        if self.is_recording and self.start_time is not None:
            elapsed = datetime.now() - self.start_time
            total_seconds = int(elapsed.total_seconds())
            minutes = total_seconds // 60
            seconds = total_seconds % 60

            self.timer_label.config(text=f"Tiempo: {minutes:02d}:{seconds:02d}")
            self.root.after(1000, self.update_timer)

    def start_recording(self):
        if self.is_recording:
            return

        if not self.stethoscope_index.get() or not self.pc_mic_index.get():
            self.status_label.config(text="Seleccione dos dispositivos de entrada válidos.")
            return

        output_folder = self.output_folder.get()

        if not os.path.isdir(output_folder):
            self.status_label.config(text="La carpeta de guardado no es válida.")
            return

        try:
            device_S = int(self.stethoscope_index.get().split(":")[0])
            device_PC = int(self.pc_mic_index.get().split(":")[0])
        except ValueError:
            self.status_label.config(text="Algún dispositivo de audio no es válido.")
            return

        self.data_S = np.empty((0,))
        self.data_PC = np.empty((0,))

        self.queue_S = queue.Queue()
        self.queue_PC = queue.Queue()

        self.recorded_file_S = None
        self.recorded_file_PC = None
        self.error_message = None

        self.audio_level_S.set(0)
        self.audio_level_PC.set(0)

        self.sampling_rate = 16000
        self.base_filename = self.generate_base_filename()

        self.is_recording = True
        self.start_time = datetime.now()

        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.refresh_button.config(state="disabled")
        self.folder_button.config(state="disabled")

        thread_S = threading.Thread(
            target=self.record_audio,
            args=(device_S, "S", self.queue_S, self.callback_S),
            daemon=True
        )

        thread_PC = threading.Thread(
            target=self.record_audio,
            args=(device_PC, "PC", self.queue_PC, self.callback_PC),
            daemon=True
        )

        self.recording_threads = [thread_S, thread_PC]

        thread_S.start()
        thread_PC.start()

        self.status_label.config(text="Grabando dos entradas de audio...")
        self.update_timer()
        self.update_spectrograms()

    def stop_recording(self):
        if not self.is_recording:
            return

        self.is_recording = False

        for thread in self.recording_threads:
            thread.join(timeout=3)

        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.refresh_button.config(state="normal")
        self.folder_button.config(state="normal")

        self.audio_level_S.set(0)
        self.audio_level_PC.set(0)

        if self.error_message:
            self.status_label.config(
                text=f"Grabación detenida con error:\n{self.error_message}"
            )
        else:
            self.status_label.config(
                text=(
                    "Grabación finalizada.\n"
                    f"Archivo estetoscopio: {self.recorded_file_S}\n"
                    f"Archivo micrófono PC: {self.recorded_file_PC}"
                )
            )

            print("Grabación _S guardada:", self.recorded_file_S)
            print("Grabación _PC guardada:", self.recorded_file_PC)

    def save_spectrogram_images(self):
        if len(self.data_S) <= 1024 and len(self.data_PC) <= 1024:
            self.status_label.config(
                text="No hay suficientes datos para guardar los espectrogramas."
            )
            return

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        combined_path = os.path.join(
            self.output_folder.get(),
            f"{timestamp}_spectrograms_S_PC.png"
        )

        self.fig.savefig(combined_path, dpi=300, bbox_inches="tight")

        if len(self.data_S) > 1024:
            fig_S, ax_S = plt.subplots(figsize=(10, 4))
            self.draw_spectrogram(ax_S, self.data_S, "Espectrograma estetoscopio (_S)")

            path_S = os.path.join(
                self.output_folder.get(),
                f"{timestamp}_spectrogram_S.png"
            )

            fig_S.savefig(path_S, dpi=300, bbox_inches="tight")
            plt.close(fig_S)

        if len(self.data_PC) > 1024:
            fig_PC, ax_PC = plt.subplots(figsize=(10, 4))
            self.draw_spectrogram(ax_PC, self.data_PC, "Espectrograma micrófono PC (_PC)")

            path_PC = os.path.join(
                self.output_folder.get(),
                f"{timestamp}_spectrogram_PC.png"
            )

            fig_PC.savefig(path_PC, dpi=300, bbox_inches="tight")
            plt.close(fig_PC)

        self.status_label.config(
            text=f"Espectrogramas guardados:\n{combined_path}"
        )

        print("Espectrogramas guardados:", combined_path)


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioRecorderApp(root)
    root.mainloop()
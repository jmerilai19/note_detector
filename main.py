import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import librosa
import soundfile as sf

CHUNK_SIZE = 2048  
SR = 44100 
DURATION = 10  

def read_audio_chunks(filename, chunk_size, duration):
    with sf.SoundFile(filename) as audio_file:
        sr = audio_file.samplerate
        total_samples = int(duration * sr)
        chunks = []

        for _ in range(0, total_samples, chunk_size):
            chunk = audio_file.read(chunk_size, dtype='float32')
            if len(chunk) == 0:
                break
            chunks.append(chunk)
        return chunks, sr

filename = 'c.wav'
audio_chunks, sample_rate = read_audio_chunks(filename, CHUNK_SIZE, DURATION)

fig, (ax_waveform, ax_spectrum) = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle("Real-Time Audio Waveform and Frequency Spectrum")

waveform_line, = ax_waveform.plot([], [], lw=2)
ax_waveform.set_xlim(0, CHUNK_SIZE / sample_rate)
ax_waveform.set_ylim(-1, 1)
ax_waveform.set_xlabel('Time (s)')
ax_waveform.set_ylabel('Amplitude')
ax_waveform.grid()

spectrum_line, = ax_spectrum.plot([], [], lw=2)
ax_spectrum.set_xlim(0, 5000)
ax_spectrum.set_ylim(0, 50)
ax_spectrum.set_xlabel('Frequency (Hz)')
ax_spectrum.set_ylabel('Amplitude')
ax_spectrum.grid()

c_freq = 261.63
ax_spectrum.axvline(c_freq, color='r', linestyle='--', label='C note frequency (261.63 Hz)')
ax_spectrum.legend()

def update_plot(frame):
    if frame >= len(audio_chunks):
        return waveform_line, spectrum_line

    chunk = audio_chunks[frame]

    time_axis = np.linspace(0, len(chunk) / sample_rate, len(chunk))
    waveform_line.set_data(time_axis, chunk)

    yf = np.fft.fft(chunk)
    xf = np.fft.fftfreq(len(chunk), 1 / sample_rate)

    idx = np.arange(len(chunk) // 2)
    xf = xf[idx]
    yf = np.abs(yf[idx])

    spectrum_line.set_data(xf, yf)
    
    return waveform_line, spectrum_line

ani = animation.FuncAnimation(fig, update_plot, frames=len(audio_chunks), interval=1000 * CHUNK_SIZE / sample_rate, blit=True)

plt.tight_layout()
plt.show()

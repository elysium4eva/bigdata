"""05_datatype_audio.py -- audio data: synthetic WAV clips plus waveform and spectrum plots."""  # module comment
import wave                                                           # standard library: write WAV files
import numpy as np                                                    # numeric computation
import matplotlib.pyplot as plt                                       # plotting
from _common import DAT, save                                         # shared helpers

SR = 8000                                                             # sample rate 8 kHz, telephone speech level
T = 1.0                                                               # each clip lasts 1 second
t = np.linspace(0, T, int(SR * T), endpoint=False)                    # build the time axis, 8000 samples per second
clips = {}                                                            # store the three clips in a dictionary

clips["tone_440"] = 0.6 * np.sin(2 * np.pi * 440 * t)                 # clip 1: pure 440 Hz tone, one single frequency
clips["chirp_200_2000"] = 0.6 * np.sin(2 * np.pi * (200 + 900 * t) * t)  # clip 2: chirp, frequency rising over time
env = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))                           # clip 3: 3 Hz amplitude envelope, speech rhythm
clips["am_noise"] = env * np.random.default_rng(7).normal(0, 0.3, t.size)  # clip 3: modulated noise, speech-like

for name, x in clips.items():                                         # loop over the three clips
    pcm = np.int16(np.clip(x, -1, 1) * 32767)                         # convert floats to 16 bit PCM, the WAV standard
    path = DAT / ("audio_%s.wav" % name)                              # build the output path
    with wave.open(str(path), "wb") as w:                             # open the WAV file in binary write mode
        w.setnchannels(1)                                             # mono channel
        w.setsampwidth(2)                                             # 2 bytes per sample, that is 16 bit
        w.setframerate(SR)                                            # write the sample rate
        w.writeframes(pcm.tobytes())                                  # write the raw byte stream
    print("[data]", path)                                             # print the output path

# ---------------------------------------------------------------- visualization
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))                   # one row, three columns: waveform / spectrum / comparison
axes[0].plot(t, clips["tone_440"], color="#8A0C3C", lw=0.8)           # left: pure tone waveform in the time domain
axes[0].set_title("波形（时域）：440 Hz 纯音")                         # left panel title
axes[0].set_xlabel("时间 / 秒")                                       # left panel x label
axes[0].set_ylabel("振幅")                                            # left panel y label

axes[1].specgram(clips["chirp_200_2000"], NFFT=256, Fs=SR,           # middle: spectrogram from a short-time Fourier transform
                 cmap="magma", noverlap=128)                          # set the overlap and the colour map
axes[1].set_title("频谱图（时频）：200→2000 Hz 扫频")                  # middle panel title
axes[1].set_xlabel("时间 / 秒")                                       # middle panel x label
axes[1].set_ylabel("频率 / Hz")                                       # middle panel y label
axes[1].grid(False)                                                   # a spectrogram needs no grid

axes[2].plot(t, clips["am_noise"], color="#66717D", lw=0.6)           # right: amplitude modulated noise waveform
axes[2].plot(t, env, color="#C9A227", lw=1.8, label="包络")           # overlay the envelope to show the rhythm
axes[2].set_title("调幅噪声（模拟语音节奏）")                           # right panel title
axes[2].set_xlabel("时间 / 秒")                                       # right panel x label
axes[2].legend(fontsize=8)                                            # show the legend
plt.tight_layout()                                                    # adjust the layout
save(fig, "fig_audio.png")                                            # save the figure

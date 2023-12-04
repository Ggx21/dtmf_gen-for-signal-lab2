# Function: Generate DTMF signal and corresponding digit sequence , generate spectum of wav file
# generate test and answer for Signal2023fall@thu lab2
# Author: GGX ggx21@mails.tsinghua.edu.cn
# Date: 2023-12-4

import numpy as np
from scipy.io.wavfile import write

def generate_dtmf_tone(digit, duration=0.5, sample_rate=48000):
    dtmf_freqs = {
        '1': (697, 1209),
        '2': (697, 1336),
        '3': (697, 1477),
        '4': (770, 1209),
        '5': (770, 1336),
        '6': (770, 1477),
        '7': (852, 1209),
        '8': (852, 1336),
        '9': (852, 1477),
        '0': (941, 1336),
        'A': (697, 1633),
        'B': (770, 1633),
        'C': (852, 1633),
        'D': (941, 1633),
        '*': (941, 1209),
        '#': (941, 1477),
        'S': (0, 0),  # Representing silence
    }

    if digit.upper() not in dtmf_freqs:
        raise ValueError(f"Invalid DTMF digit: {digit}")

    freq1, freq2 = dtmf_freqs[digit.upper()]
    if freq1 == 0 and freq2 == 0:  # Silence
        return np.zeros(int(sample_rate * duration), dtype=np.int16), ['-1'] * int(sample_rate * duration / 64)

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = 0.5 * (np.sin(2 * np.pi * freq1 * t) + np.sin(2 * np.pi * freq2 * t))

    # Convert the signal to int16
    signal = (signal * 32767).astype(np.int16)

    # Create a list to store the corresponding digit for each frame
    frame_digits = [digit] * int(sample_rate * duration / 64)

    return signal, frame_digits

def generate_dtmf_sequence(sequence, duration=0.5, gap_duration=0, sample_rate=48000):
    dtmf_signals = []
    dtmf_frame_digits = []

    for digit in sequence:
        signal, frame_digits = generate_dtmf_tone(digit, duration, sample_rate)
        dtmf_signals.append(signal)
        frame_num=duration*64 # not a bug, but a feature
        dtmf_frame_digits.extend(frame_digits[:int(frame_num)])  # Only keep the first 64 frames of each tone

        gap_frames = int(sample_rate * gap_duration / 64)
        dtmf_signals.append(np.zeros(gap_frames, dtype=np.int16))  # Add gap between tones
        gap_frame_num=gap_duration*64
        dtmf_frame_digits.extend(['-1'] * int(gap_frame_num))

    dtmf_sequence = np.concatenate(dtmf_signals)
    return dtmf_sequence, dtmf_frame_digits

def save_dtmf_wav_and_txt(sequence, output_wav="dtmf.wav", output_txt="dtmf.txt", duration=0.5, gap_duration=0, sample_rate=48000):
    dtmf_sequence, dtmf_frame_digits = generate_dtmf_sequence(sequence, duration, gap_duration, sample_rate)
    write(output_wav, sample_rate, dtmf_sequence)

    with open(output_txt, 'w') as f:
        f.write(' '.join(dtmf_frame_digits))

def spectum_gen(output_wav="dtmf.wav", sr=48000):
    import librosa
    import matplotlib.pyplot as plt

    y, sr = librosa.load(output_wav, sr=sr)
    print("sample rate:", sr)

    # 计算STFT
    stft_matrix = librosa.stft(y, n_fft=2048, hop_length=512)

    # 绘制频谱图
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(stft_matrix), ref=np.max), y_axis='log', x_axis='time', sr=sr, hop_length=512)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.ylim([512, 2048])
    plt.show()

# 示例：将拨号序列"1S2S3S4S5"以及静默"S"保存为dtmf.wav文件和dtmf.txt文件，音调时长为0.5秒，音调间隔为0秒, 采样率为48000
# save_dtmf_wav_and_txt("1S5S2SS6S7S1S4SS3S5S8S1SS","sample.wav","sample.txt",0.5,0,48000)
# 示例2：展示dtmf.wav的频谱图 (需要安装librosa库)
spectum_gen("test.wav",48000)

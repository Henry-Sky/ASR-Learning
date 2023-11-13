import wave
import numpy as np
from scipy import fft

# 读取音频文件
def read_wave(file_path):
    """
    读取wav文件并返回音频数据和时间表。

    参数:
    - file_path: str
      wav文件的路径。

    返回:
    - wave_data: numpy数组
      音频数据，每行为一个通道。
    - time_table: numpy数组
      与音频数据对应的时间点。
    """
    with wave.open(file_path, "rb") as file:
        # 获取音频文件的参数信息
        params = file.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        # 读取所有帧的音频数据并将其转换为numpy数组
        wave_data = np.frombuffer(file.readframes(nframes), dtype=np.int16)
    wave_data = wave_data.reshape(-1,nchannels).T
    time_table = np.arange(0, nframes) * (1.0/framerate)
    return wave_data, time_table

# 快速傅里叶变换
def wave_fft(wave_data, time_table, time_range=(-1,-1), inverse=False):
    """
    在指定的时间范围内对多通道音频数据执行FFT或傅里叶逆变换。

    参数:
    - wave_data: numpy数组, 形状为 (nchannels, nsamples)
      多通道音频数据.
    - time_table: numpy数组
      与音频数据对应的时间点.
    - time_range: 元组, 可选
      指定FFT的时间范围。默认为 (-1, -1), 表示完整的时间范围。
    - inverse: 布尔值, 可选
      如果为True,则执行傅里叶逆变换.默认为False。

    返回:
    - fft_data: numpy数组
      傅里叶变换或傅里叶逆变换的结果。
    - fft_time: numpy数组
      与FFT结果对应的时间点。
    """
    assert(len(time_range) == 2)
    time_start, time_end = time_range if time_range != (-1,-1) else (0, time_table[-1])
    selected_indices = np.where((time_table >= time_start) & (time_table <= time_end))[0]
    fft_time = time_table[selected_indices]
    if not inverse:
        fft_data = fft.fft(wave_data[:, selected_indices], axis=1)
    else:
        fft_data = fft.ifft(wave_data[:, selected_indices], axis=1)
    return fft_data, fft_time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import argparse
from nptdms import TdmsWriter,ChannelObject
from nptdms import TdmsFile, TdmsWriter, ChannelObject
from matplotlib.ticker import FuncFormatter
import threading
import queue
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

USE_CUPY = False
def add_cuda_to_path():
    cuda_path = os.environ.get('CUDA_PATH', 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6')
    cuda_bin = os.path.join(cuda_path, 'bin')
    if cuda_bin not in os.environ['PATH']:
        os.environ['PATH'] = cuda_bin + os.pathsep + os.environ['PATH']

try:
    # add_cuda_to_path()
    import cupy as cp
    if cp.cuda.is_available():
        USE_CUPY = True
        cp.cuda.Device(0).use()
        print("Using CuPy for FFT calculations.")
    else:
        USE_CUPY = False
        print("CUDA not available. Using NumPy for FFT calculations.")
except ImportError:
    USE_CUPY = False
    print("CuPy not installed. Using NumPy for FFT calculations.")

def format_frequency(freq):
    """
    根据频率值自动转换为合适的单位显示：Hz、KHz、MHz 或 GHz。
    """
    if freq < 1e3:
        return f"{freq:.2f} Hz"
    elif freq < 1e6:
        return f"{freq/1e3:.2f} KHz"
    elif freq < 1e9:
        return f"{freq/1e6:.2f} MHz"
    else:
        return f"{freq/1e9:.2f} GHz"


def scan_directory(directory_path):
    # 初始化两个列表，分别存储 csv 文件名和 tdms 文件名
    csv_files = []
    tdms_files = []
    
    # 遍历指定目录下的所有文件
    for file in os.listdir(directory_path):
        full_path = os.path.join(directory_path, file)
        # 确保当前项是文件而非目录
        if os.path.isfile(full_path):
            # 如果文件扩展名为 .csv，则加入 csv_files 列表
            if file.lower().endswith('.csv'):
                csv_files.append(file)
            # 如果文件扩展名为 .tdms，则加入 tdms_files 列表
            elif file.lower().endswith('.tdms'):
                tdms_files.append(file)
    return csv_files, tdms_files

def autofft(file_path,filter_threshold=0.0):
    # 如果文件名包含 5151 250e6
    sample_rate = 250e6
    if "5171" in file_path:
        if filter_threshold <0.01:
            filter_threshold = 25.0
        sample_rate = 250e6  # 采样率为 250 MHz (250 * 10^6)
    elif "5785" in file_path:
        if filter_threshold <0.01:
            filter_threshold = 50000.0
        sample_rate = 3.2e9  # 或者采样率为 3.2 GHz (3.2 * 10^9)
    else:
        raise ValueError("不支持的采样率")

    # 修改读取CSV文件的部分
    try:
        start = time.time()
        with open(file_path, 'r') as file:
            first_line = file.readline().strip()
            if first_line:
                data = first_line.split(',')
                # 确保数据不为空且是有效的数值
                data = [float(x) for x in data if x.strip()]
                if not data:
                    raise ValueError("CSV文件中没有有效的数据")
                data = data * 5  # 如果需要重复数据
                data = np.array(data, dtype=np.float32)
            else:
                raise ValueError("CSV文件为空")
        end = time.time()
        print(f'读取CSV文件耗时: {end-start:.3f}秒')
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        raise
    
    N = len(data)  # 数据长度
    T = 1.0 / sample_rate  # 采样时间间隔
    if USE_CUPY:
        data_gpu = cp.asarray(data)

        fft_result_gpu = cp.fft.fft(data_gpu)
        fft_freq_gpu = cp.fft.fftfreq(N, d=1 / sample_rate)

        fft_magnitude_gpu = cp.abs(fft_result_gpu) / (N / 2)
        fft_magnitude_gpu[0] = fft_magnitude_gpu[0] / 2  # 直流分量不需要乘以2
        # 将 FFT 幅值转换为有效值
        fft_magnitude_rms_gpu = fft_magnitude_gpu / cp.sqrt(2)

        # 只取正频率部分
        pos_mask_gpu = fft_freq_gpu >= 0
        fft_freq_gpu = fft_freq_gpu[pos_mask_gpu]
        fft_magnitude_rms_gpu = fft_magnitude_rms_gpu[pos_mask_gpu]

        # 转换为 NumPy 数组
        fft_freq = cp.asnumpy(fft_freq_gpu)
        fft_magnitude_rms = cp.asnumpy(fft_magnitude_rms_gpu)

        fft_amplitude = 20 * np.log10(fft_magnitude_rms) + 120

        peak_index = np.argmax(fft_amplitude)
        peak_freq = fft_freq[peak_index]
        peak_amplitude = fft_amplitude[peak_index]
        print("峰值频率: {:.2f} Hz, 峰值幅值: {:.2f} dB".format(peak_freq, peak_amplitude))
    else:
        fft_result = np.fft.fft(data)  # 计算 FFT
        fft_freq = np.fft.fftfreq(N, d=T)  # 计算频率

        fft_magnitude = np.abs(fft_result) / (N / 2)
        fft_magnitude[0] = fft_magnitude[0] / 2  # 处理直流分量

        fft_magnitude_rms = fft_magnitude / np.sqrt(2)  # 转换为有效值

        # 只保留正频率部分
        pos_mask = fft_freq >= 0
        fft_freq = fft_freq[pos_mask]
        fft_magnitude_rms = fft_magnitude_rms[pos_mask]

        # 将幅值转换为 dB
        fft_amplitude = 20 * np.log10(fft_magnitude_rms) + 120

        ## 计算频谱图中最大幅值（峰值）的频率和幅值
        peak_index = np.argmax(fft_amplitude)
        peak_freq = fft_freq[peak_index]
        peak_amplitude = fft_amplitude[peak_index]
        print("峰值频率: {}, 峰值幅值: {:.2f} dB".format(format_frequency(peak_freq), peak_amplitude))

    if filter_threshold > 0:
        mask = fft_freq >= filter_threshold
        fft_freq = fft_freq[mask]
        fft_amplitude = fft_amplitude[mask]

    if fft_amplitude.size > 0:
        peak_index = np.argmax(fft_amplitude)
        peak_freq = fft_freq[peak_index]
        peak_amplitude = fft_amplitude[peak_index]
        print("过滤后峰值频率: {}, 过滤后峰值幅值: {:.2f} dB".format(format_frequency(peak_freq), peak_amplitude))
    else:
        print("过滤阈值过高，没有数据满足条件")


   
    
    # FN = len(fft_freq)
    # with open(file_path.replace('.csv', '_fft.csv'), 'w') as f:
    #     for i, (freq, amplitude) in enumerate(zip(fft_freq, fft_amplitude)):
    #         # Write comma-separated values, but avoid trailing comma at the end
    #         if i == len(fft_freq) - 1:  # Last item, no comma
    #             f.write(f'({freq},{amplitude})')
    #         else:  # Other items, add a comma
    #             f.write(f'({freq},{amplitude}),')

    # df = pd.DataFrame({'Frequency': fft_freq, 'Amplitude(dB)': fft_amplitude})
    # df.to_csv(file_path.replace('.csv', '_fft.csv'), index=False, header=False)

    # np.savez(file_path.replace('.csv', '_fft.npz'), frequency=fft_freq, amplitude=fft_amplitude)
    
    start = time.time()
    tdms_file_path = file_path.replace('.csv', '_fft.tdms')
    with TdmsWriter(tdms_file_path) as writer:
        # Adding properties as an empty dictionary
        writer.write_segment([
            ChannelObject('Group', 'Frequency', fft_freq, properties={}),
            ChannelObject('Group', 'Amplitude_dB', fft_amplitude, properties={})
        ])
    end = time.time()
    print(f'save tdms {end -start}')

    print(f'{tdms_file_path}')
 


def plot_fft_tdms(file_path):
    """
    从 TDMS 文件读取 FFT 数据并绘制频率图。
    """
    # 读取 TDMS 文件
    tdms_file = TdmsFile.read(file_path)
    group = tdms_file['Group']
    
    # 获取频率和幅值数据
    fft_freq = group['Frequency'][:]
    fft_amplitude = group['Amplitude_dB'][:]
    peak_index = np.argmax(fft_amplitude)
    peak_freq = fft_freq[peak_index]
    peak_amplitude = fft_amplitude[peak_index]
    print("峰值频率: {} , 峰值幅值: {:.2f} dB".format(format_frequency(peak_freq), peak_amplitude))

    # 绘制频谱图
    plt.figure(figsize=(10, 6))
    plt.plot(fft_freq, fft_amplitude)
    # 标记峰值点
    plt.scatter(peak_freq, peak_amplitude, color='red', zorder=5)
    plt.annotate("   Peak\n   {}\n   {:.2f} dB".format(format_frequency(peak_freq), peak_amplitude),
                xy=(peak_freq, peak_amplitude),
                xytext=(peak_freq, peak_amplitude-30),
                # arrowprops=dict(facecolor='black', shrink=0.05)
                )
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.grid(True)
    ax = plt.gca()
    # 设置 symlog：低于1Hz部分采用线性显示，高于1Hz部分采用对数显示，使低频间隔较宽，高频间隔较窄
    ax.set_xscale("symlog", linthresh=1)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: format_frequency(x)))
    plt.show(block=False)
    # plt.pause(1)
    # plt.close()

# 添加一个全局队列用于线程间通信
plot_queue = queue.Queue()

class FileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith('_fft.tdms'):
            plot_queue.put(event.src_path)
        elif event.src_path.endswith('.csv'):
            print(f'发现新的CSV文件: {event.src_path}')
            try:
                # 等待文件写入完成
                time.sleep(0.1)  # 短暂延迟以确保文件写入完成
                if not os.path.exists(event.src_path):
                    print(f"文件不存在: {event.src_path}")
                    return
                file_size = os.path.getsize(event.src_path)
                if file_size == 0:
                    print(f"文件为空: {event.src_path}")
                    return
                
                autofft(event.src_path)
                tdms_path = event.src_path.replace('.csv', '_fft.tdms')
                plot_queue.put(tdms_path)
            except Exception as e:
                print(f"处理新CSV文件时出错: {event.src_path}")
                print(f"错误详情: {str(e)}")
                print(f"错误类型: {type(e).__name__}")

def process_plot_queue():
    """在主线程中处理绘图队列"""
    try:
        # 非阻塞方式获取新的TDMS文件路径
        tdms_path = plot_queue.get_nowait()
        
        # 清除当前图形
        plt.clf()
        
        # 绘制新的频谱图
        tdms_file = TdmsFile.read(tdms_path)
        group = tdms_file['Group']
        
        fft_freq = group['Frequency'][:]
        fft_amplitude = group['Amplitude_dB'][:]
        peak_index = np.argmax(fft_amplitude)
        peak_freq = fft_freq[peak_index]
        peak_amplitude = fft_amplitude[peak_index]
        
        plt.plot(fft_freq, fft_amplitude)
        plt.scatter(peak_freq, peak_amplitude, color='red', zorder=5)
        plt.annotate(
            f"   Peak\n   {format_frequency(peak_freq)}\n   {peak_amplitude:.2f} dB",
            xy=(peak_freq, peak_amplitude),
            xytext=(peak_freq, peak_amplitude-30)
        )
        
        plt.title("Frequency Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude (dB)")
        plt.grid(True)
        
        ax = plt.gca()
        ax.set_xscale("symlog", linthresh=1)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: format_frequency(x)))
        
        plt.draw()
        
    except queue.Empty:
        pass
    except Exception as e:
        print(f"绘图错误: {e}")

def main():
    parser = argparse.ArgumentParser(description="解析 LabView 存储的 TDMS 文件")
    path = ''
    try:
        parser.add_argument('file', help="TDMS 或 CSV 文件路径")
        args = parser.parse_args()
        path = args.file
    except:
        path = 'C:\\Data'

    # 确保目录存在
    if not os.path.exists(path):
        try:
            if path.endswith('.csv') or path.endswith('.tdms'):
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            else:
                os.makedirs(path, exist_ok=True)
        except Exception as e:
            print(f"创建目录失败: {e}")
            return

    # 获取要监视的目录
    watch_path = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    
    # 初始化matplotlib
    plt.ion()  # 启用交互模式
    plt.figure(figsize=(10, 6))

    # 设置文件监视器
    event_handler = FileHandler()
    observer = Observer()
    observer.schedule(event_handler, path=watch_path, recursive=False)
    observer.start()

    try:
        # 处理已有文件
        if path.endswith('.csv'):
            filepath = os.path.abspath(path)
            autofft(filepath)
            tdms_path = filepath.replace('.csv', '_fft.tdms')
            plot_queue.put(tdms_path)
        elif path.endswith('_fft.tdms'):
            plot_queue.put(path)
        elif os.path.isdir(path):
            csvfiles, tdmsfiles = scan_directory(path)
            print(f' {path} 当前目录下有 {len(csvfiles)} 个CSV文件，{len(tdmsfiles)} 个TDMS文件')
            for csv_file in csvfiles:
                tdms_filename = csv_file.replace('.csv','_fft.tdms') 
                if tdms_filename not in tdmsfiles:
                    print(f'需要转换{csv_file} -> {tdms_filename}')
                    filepath = f'{os.path.abspath(path)}\\{csv_file}'
                    autofft(filepath) 
                    tdms_filename_path = f'{os.path.abspath(path)}\\{tdms_filename}'
                    plot_queue.put(tdms_filename_path)
                else:
                    print(f'已转换{csv_file} -> {os.path.abspath(path)}\\{tdms_filename}')

        # 主循环
        while True:
            process_plot_queue()  # 在主线程中处理绘图
            plt.pause(0.1)  # 更新图形并处理事件

    except KeyboardInterrupt:
        observer.stop()
        observer.join()
    except Exception as e:
        print(f"发生错误: {e}")
        observer.stop()
        observer.join()
    finally:
        plt.close('all')

if __name__ == "__main__":
    main()
    
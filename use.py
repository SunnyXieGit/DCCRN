import torch
import librosa as lib
import noisereduce as nr
import tkinter as tk
from tkinter import ttk, filedialog
import threading
from tkinterdnd2 import TkinterDnD, DND_FILES
import os
import ffmpeg
import subprocess
#from pydub import AudioSegment
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pydub import AudioSegment



class AudioProcessorApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("网课音频降噪处理器")
        self.geometry("800x500+500+250")#("宽度x高度+X坐标+Y坐标")
        # 初始化全局样式
        self.init_style()
        # 模型配置参数
        self.sr = 16000
        self.model_path = './logs/parameter_epoch1_2025-03-01 18-04-11.pth'
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda:0")
        self.create_widgets()
        self.load_model()

    def init_style(self):
        """自定义全局组件样式"""
        self.style = ttk.Style()
        # 设置全局字体（微软雅黑，16号）
        self.style.configure(".", font=("Microsoft YaHei", 16))
        # 按钮样式
        self.style.configure(
            "Big.TButton",
            font=("Microsoft YaHei", 18, "bold"),
            padding=15,
            width=20
        )
        # 输入框样式
        self.style.configure(
            "Big.TEntry",
            padding=10,
            font=("Microsoft YaHei", 14)
        )

    def create_widgets(self):
        """创建界面组件"""
        # 主容器
        main_frame = ttk.Frame(self)
        main_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        # 文件输入区域
        input_frame = ttk.LabelFrame(main_frame, text=" 输入设置 ", padding=(20, 15))
        input_frame.pack(fill=tk.X, pady=10)

        ttk.Label(input_frame, text="需要降噪的文件路径：").grid(row=0, column=0, sticky=tk.W, padx=5, pady=10)

        self.file_entry = ttk.Entry(
            input_frame,
            width=60,
            style="Big.TEntry"
        )
        self.file_entry.grid(row=1, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=5)

        # 拖放支持
        self.file_entry.drop_target_register(DND_FILES)
        self.file_entry.dnd_bind('<<Drop>>', self.on_drop)

        # 操作按钮区域
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=20)

        ttk.Button(
            btn_frame,
            text="📁 浏览本地文件",
            command=self.browse_file,
            style="Big.TButton"
        ).pack(side=tk.LEFT, expand=True, padx=10)

        self.process_btn = ttk.Button(
            btn_frame,
            text="⚡ 开始降噪处理",
            command=self.start_processing,
            style="Big.TButton"
        )
        self.process_btn.pack(side=tk.LEFT, expand=True, padx=10)

        # 状态显示区域
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=15)

        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(
            status_frame,
            textvariable=self.status_var,
            font=("Microsoft YaHei", 14, "bold"),
            foreground="#444"
        ).pack(expand=True)

    def load_model(self):
        try:
            self.dccrn_model = torch.load(self.model_path).to(self.device)
            self.dccrn_model.eval()
            if torch.cuda.is_available():
                gpu_info = f"检测到显卡：{torch.cuda.get_device_name(0)}"
            else:
                gpu_info = "警告：未检测到可用显卡！"
            self.status_var.set("✅ 模型载入成功！"+gpu_info)
        except Exception as e:
            self.status_var.set(f"❌ 模型载入错误: {str(e)}")

    def on_drop(self, event):
        files = event.data.split()
        if files:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, files[0].strip("{}"))

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("音频文件", "*.wav *.mp3")]
        )
        if file_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)

    def start_processing(self):
        input_path = self.file_entry.get()
        if not input_path:
            self.status_var.set("⚠ 请先选择音频文件！")
            return

        self.process_btn.config(state="disabled")
        self.status_var.set("⏳ 正在处理中，请稍候...")

        # 在后台线程运行处理
        threading.Thread(target=self.process_audio, args=(input_path,)).start()

    def process_audio(self, input_path):
        try:
            # 加载音频
            noisy_audio, _ = lib.load(input_path, sr=self.sr)
            nr_noise = nr.reduce_noise(noisy_audio, sr=self.sr)
            sf.write('./output/nrdenoised_audio.wav', nr_noise, self.sr)

            # FFmpeg 音量调整
            subprocess.run([
                'ffmpeg', '-i', './output/nrdenoised_audio.wav',
                '-filter:a', 'volume=3.0', '-y', './output/fdenoised.wav'
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # 保存结果
            temp_path = './output/fdenoised.wav'

            """打开输出文件夹"""
            output_path = os.path.abspath('./output')
            # 如果文件夹不存在则创建
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            os.startfile(output_path)
            # 二次降噪和音量调整
            self.post_process(temp_path)
            self.status_var.set("处理成功!")

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
        finally:
            self.process_btn.config(state="normal")

    def load_wav(self, signal, frame_dur=37.5):
        print("load_signal", signal.shape)
        win = int(frame_dur / 1000 * self.sr)
        lten = torch.tensor(np.array(np.split(signal, int(len(signal) / win), axis=0)))
        print("lten", lten.shape)
        return lten

    def save_reconstructed_audio(self, output_frames, save_path):
        full_signal = np.concatenate(output_frames, axis=0)
        full_signal = np.clip(full_signal, -1.0, 1.0)
        sf.write(save_path, full_signal.astype(np.float32), self.sr)

    def post_process(self, input_file):
        # 二次降噪处理
        faudio, _ = lib.load(input_file, sr=self.sr)
        # 预处理
        noisy_audio_tensor = self.load_wav(faudio).float().to(self.device)

        # 模型推理
        with torch.no_grad():
            clean_audio_tensor = self.dccrn_model(noisy_audio_tensor).to(self.device)
            print("clean_audio_tensor", clean_audio_tensor.shape)
        # 后处理
        clean_audio = clean_audio_tensor.squeeze(1).cpu().detach().numpy()
        save_path = "./output/final_denoised.wav"
        self.save_reconstructed_audio(clean_audio, save_path)


if __name__ == "__main__":
    app = AudioProcessorApp()
    app.mainloop()








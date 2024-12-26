#!/usr/bin/env python3
"""
Tsunami Wave Simulation Application
Main entry point for the GUI application that simulates and visualizes tsunami wave motion.
"""
import sys
from typing import Optional, Dict, List
import json
import csv
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np

# Cấu hình matplotlib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from models.wave_physics import TsunamiWaveModel, WaveParameters

class TsunamiSimulator(tk.Tk):
    """Ứng dụng mô phỏng sóng thần."""
    
    def __init__(self):
        super().__init__()
        
        self.title("Mô phỏng Sóng Thần")
        self.geometry("1200x800")
        
        # Khởi tạo mô hình
        self.wave_model = TsunamiWaveModel()
        self.time = 0.0
        
        # Tạo giao diện
        self._create_widgets()
        self._create_layout()
        
        # Cập nhật đồ thị ban đầu
        self._update_plot()
    
    def _create_widgets(self):
        """Tạo các widget cho giao diện."""
        # Frame điều khiển
        self.control_frame = ttk.LabelFrame(self, text="Điều khiển")
        
        # Thông số sóng
        wave_params_frame = ttk.LabelFrame(self.control_frame, text="Thông số sóng")
        
        # Biên độ
        self.amp_label = ttk.Label(wave_params_frame, text="Biên độ (m):")
        self.amp_var = tk.DoubleVar(value=1.0)
        self.amp_spin = ttk.Spinbox(
            wave_params_frame,
            from_=0.1,
            to=50.0,
            increment=0.1,
            textvariable=self.amp_var,
            command=self._update_plot
        )
        self.amp_scale = ttk.Scale(
            wave_params_frame,
            from_=0.1,
            to=50.0,
            orient='horizontal',
            variable=self.amp_var,
            command=lambda _: self._update_plot()
        )
        
        # Bước sóng
        self.wave_label = ttk.Label(wave_params_frame, text="Bước sóng (m):")
        self.wave_var = tk.DoubleVar(value=100.0)
        self.wave_spin = ttk.Spinbox(
            wave_params_frame,
            from_=10.0,
            to=1000.0,
            increment=10.0,
            textvariable=self.wave_var,
            command=self._update_plot
        )
        self.wave_scale = ttk.Scale(
            wave_params_frame,
            from_=10.0,
            to=1000.0,
            orient='horizontal',
            variable=self.wave_var,
            command=lambda _: self._update_plot()
        )
        
        # Độ sâu
        self.depth_label = ttk.Label(wave_params_frame, text="Độ sâu (m):")
        self.depth_var = tk.DoubleVar(value=1000.0)
        self.depth_spin = ttk.Spinbox(
            wave_params_frame,
            from_=100.0,
            to=5000.0,
            increment=100.0,
            textvariable=self.depth_var,
            command=self._update_plot
        )
        self.depth_scale = ttk.Scale(
            wave_params_frame,
            from_=100.0,
            to=5000.0,
            orient='horizontal',
            variable=self.depth_var,
            command=lambda _: self._update_plot()
        )
        
        # Hiệu ứng vật lý
        self.effects_frame = ttk.LabelFrame(self.control_frame, text="Hiệu ứng vật lý")
        self.effects_vars = {}
        self.effects_checks = {}
        self.effects_descriptions = {
            "nonlinear": "Phi tuyến - Mô phỏng các hiệu ứng phi tuyến của sóng",
            "dispersion": "Tán sắc - Mô phỏng sự phân tán sóng theo tần số",
            "bottom_friction": "Ma sát đáy - Tính đến ảnh hưởng của ma sát đáy biển",
            "coriolis": "Coriolis - Hiệu ứng do sự quay của Trái Đất",
            "wind": "Gió - Ảnh hưởng của gió lên bề mặt sóng"
        }
        
        for key, desc in self.effects_descriptions.items():
            frame = ttk.Frame(self.effects_frame)
            self.effects_vars[key] = tk.BooleanVar()
            check = ttk.Checkbutton(
                frame,
                text=desc.split(' - ')[0],
                variable=self.effects_vars[key],
                command=self._update_plot
            )
            info_label = ttk.Label(frame, text=desc.split(' - ')[1], wraplength=200)
            
            check.pack(side=tk.LEFT)
            info_label.pack(side=tk.LEFT, padx=5)
            frame.pack(fill=tk.X, padx=5, pady=2)
            self.effects_checks[key] = check
        
        # Điều khiển mô phỏng
        sim_frame = ttk.LabelFrame(self.control_frame, text="Điều khiển mô phỏng")
        
        # Nút Start/Stop
        self.running = False
        self.start_button = ttk.Button(
            sim_frame,
            text="Bắt đầu",
            command=self._toggle_simulation
        )
        
        # Nút Reset
        self.reset_button = ttk.Button(
            sim_frame,
            text="Đặt lại",
            command=self._reset_simulation
        )
        
        # Thanh thời gian
        self.time_var = tk.DoubleVar(value=0.0)
        self.time_scale = ttk.Scale(
            sim_frame,
            from_=0.0,
            to=10.0,
            orient='horizontal',
            variable=self.time_var,
            command=lambda _: self._update_plot()
        )
        
        # Notebook cho các tab
        self.notebook = ttk.Notebook(self)
        
        # Tab sóng 2D
        self.tab_2d = ttk.Frame(self.notebook)
        self.fig_2d = Figure(figsize=(8, 6))
        self.ax_2d = self.fig_2d.add_subplot(111)
        self.canvas_2d = FigureCanvasTkAgg(self.fig_2d, master=self.tab_2d)
        
        # Tab sóng 3D
        self.tab_3d = ttk.Frame(self.notebook)
        
        # Frame chứa đồ thị 3D
        plot_frame = ttk.Frame(self.tab_3d)
        self.fig_3d = Figure(figsize=(8, 6))
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=plot_frame)
        
        # Frame điều khiển 3D
        control_3d_frame = ttk.LabelFrame(self.tab_3d, text="Điều khiển 3D")
        
        # Điều khiển góc nhìn
        view_frame = ttk.LabelFrame(control_3d_frame, text="Góc nhìn")
        
        # Góc elevation
        elev_label = ttk.Label(view_frame, text="Góc đứng:")
        self.elev_var = tk.DoubleVar(value=30)
        self.elev_scale = ttk.Scale(
            view_frame,
            from_=0,
            to=90,
            orient='horizontal',
            variable=self.elev_var,
            command=self._update_3d_view
        )
        
        # Góc azimuth
        azim_label = ttk.Label(view_frame, text="Góc quay:")
        self.azim_var = tk.DoubleVar(value=-60)
        self.azim_scale = ttk.Scale(
            view_frame,
            from_=0,
            to=360,
            orient='horizontal',
            variable=self.azim_var,
            command=self._update_3d_view
        )
        
        # Điều khiển zoom
        zoom_frame = ttk.LabelFrame(control_3d_frame, text="Zoom")
        zoom_label = ttk.Label(zoom_frame, text="Khoảng cách:")
        self.zoom_var = tk.DoubleVar(value=1000)
        self.zoom_scale = ttk.Scale(
            zoom_frame,
            from_=500,
            to=2000,
            orient='horizontal',
            variable=self.zoom_var,
            command=self._update_3d_view
        )
        
        # Điều khiển màu sắc
        color_frame = ttk.LabelFrame(control_3d_frame, text="Màu sắc")
        
        # Chọn colormap
        cmap_label = ttk.Label(color_frame, text="Bảng màu:")
        self.cmap_var = tk.StringVar(value='viridis')
        self.cmap_combo = ttk.Combobox(
            color_frame,
            values=['viridis', 'plasma', 'inferno', 'magma', 'ocean', 'coolwarm'],
            textvariable=self.cmap_var,
            state='readonly'
        )
        self.cmap_combo.bind('<<ComboboxSelected>>', lambda e: self._update_plot())
        
        # Độ trong suốt
        alpha_label = ttk.Label(color_frame, text="Độ trong suốt:")
        self.alpha_var = tk.DoubleVar(value=1.0)
        self.alpha_scale = ttk.Scale(
            color_frame,
            from_=0.1,
            to=1.0,
            orient='horizontal',
            variable=self.alpha_var,
            command=lambda _: self._update_plot()
        )
        
        # Hiệu ứng bề mặt
        surface_frame = ttk.LabelFrame(control_3d_frame, text="Hiệu ứng bề mặt")
        
        # Kiểu bề mặt
        self.wireframe_var = tk.BooleanVar(value=False)
        self.wireframe_check = ttk.Checkbutton(
            surface_frame,
            text="Hiển thị lưới",
            variable=self.wireframe_var,
            command=self._update_plot
        )
        
        self.contour_var = tk.BooleanVar(value=False)
        self.contour_check = ttk.Checkbutton(
            surface_frame,
            text="Hiển thị đường đồng mức",
            variable=self.contour_var,
            command=self._update_plot
        )
        
        # Tab phân tích
        self.tab_analysis = ttk.Frame(self.notebook)
        
        # Frame cho bảng thông số
        params_frame = ttk.LabelFrame(self.tab_analysis, text="Thông số phân tích")
        self.analysis_text = tk.Text(params_frame, height=10, width=50)
        
        # Frame cho đồ thị phổ
        spectrum_frame = ttk.LabelFrame(self.tab_analysis, text="Phổ năng lượng")
        self.fig_spectrum = Figure(figsize=(8, 4))
        self.ax_spectrum = self.fig_spectrum.add_subplot(111)
        self.canvas_spectrum = FigureCanvasTkAgg(self.fig_spectrum, master=spectrum_frame)
        
        # Thêm các tab
        self.notebook.add(self.tab_2d, text="Sóng 2D")
        self.notebook.add(self.tab_3d, text="Sóng 3D")
        self.notebook.add(self.tab_analysis, text="Phân tích")
        
        # Khởi tạo timer cho animation
        self.after_id = None
    
    def _create_layout(self):
        """Sắp xếp layout cho giao diện."""
        # Layout điều khiển
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Layout thông số sóng
        wave_params_frame = self.control_frame.winfo_children()[0]
        wave_params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.amp_label.pack(pady=2)
        self.amp_spin.pack(pady=2)
        self.amp_scale.pack(fill=tk.X, padx=5, pady=2)
        
        self.wave_label.pack(pady=2)
        self.wave_spin.pack(pady=2)
        self.wave_scale.pack(fill=tk.X, padx=5, pady=2)
        
        self.depth_label.pack(pady=2)
        self.depth_spin.pack(pady=2)
        self.depth_scale.pack(fill=tk.X, padx=5, pady=2)
        
        # Layout hiệu ứng vật lý
        self.effects_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Layout điều khiển mô phỏng
        sim_frame = self.control_frame.winfo_children()[2]
        sim_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.reset_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.time_scale.pack(fill=tk.X, padx=5, pady=5)
        
        # Layout notebook và các tab
        self.notebook.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        # Layout tab 2D
        self.canvas_2d.get_tk_widget().pack(expand=True, fill=tk.BOTH)
        
        # Layout tab 3D
        plot_frame = self.tab_3d.winfo_children()[0]
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        control_3d_frame = self.tab_3d.winfo_children()[1]
        control_3d_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # Layout góc nhìn
        view_frame = control_3d_frame.winfo_children()[0]
        view_frame.pack(fill=tk.X, padx=5, pady=5)
        elev_label = ttk.Label(view_frame, text="Góc đứng:")
        elev_label.pack()
        self.elev_scale.pack(fill=tk.X, padx=5, pady=2)
        azim_label = ttk.Label(view_frame, text="Góc quay:")
        azim_label.pack()
        self.azim_scale.pack(fill=tk.X, padx=5, pady=2)
        
        # Layout zoom
        zoom_frame = control_3d_frame.winfo_children()[1]
        zoom_frame.pack(fill=tk.X, padx=5, pady=5)
        zoom_label = ttk.Label(zoom_frame, text="Khoảng cách:")
        zoom_label.pack()
        self.zoom_scale.pack(fill=tk.X, padx=5, pady=2)
        
        # Layout màu sắc
        color_frame = control_3d_frame.winfo_children()[2]
        color_frame.pack(fill=tk.X, padx=5, pady=5)
        cmap_label = ttk.Label(color_frame, text="Bảng màu:")
        cmap_label.pack()
        self.cmap_combo.pack(fill=tk.X, padx=5, pady=2)
        alpha_label = ttk.Label(color_frame, text="Độ trong suốt:")
        alpha_label.pack()
        self.alpha_scale.pack(fill=tk.X, padx=5, pady=2)
        
        # Layout hiệu ứng bề mặt
        surface_frame = control_3d_frame.winfo_children()[3]
        surface_frame.pack(fill=tk.X, padx=5, pady=5)
        self.wireframe_check.pack(anchor=tk.W, padx=5, pady=2)
        self.contour_check.pack(anchor=tk.W, padx=5, pady=2)
        
        # Layout tab phân tích
        params_frame = self.tab_analysis.winfo_children()[0]
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        self.analysis_text.pack(fill=tk.X, padx=5, pady=5)
        
        spectrum_frame = self.tab_analysis.winfo_children()[1]
        spectrum_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.canvas_spectrum.get_tk_widget().pack(expand=True, fill=tk.BOTH)
    
    def _toggle_simulation(self):
        """Bắt đầu/dừng mô phỏng."""
        self.running = not self.running
        if self.running:
            self.start_button.configure(text="Dừng")
            self._animate()
        else:
            self.start_button.configure(text="Bắt đầu")
            if self.after_id:
                self.after_cancel(self.after_id)
    
    def _reset_simulation(self):
        """Đặt lại mô phỏng về trạng thái ban đầu."""
        self.running = False
        self.start_button.configure(text="Bắt đầu")
        if self.after_id:
            self.after_cancel(self.after_id)
        self.time_var.set(0.0)
        self._update_plot()
    
    def _animate(self):
        """Cập nhật animation."""
        if self.running:
            current_time = self.time_var.get()
            new_time = current_time + 0.1
            if new_time > 10.0:
                new_time = 0.0
            self.time_var.set(new_time)
            self._update_plot()
            self.after_id = self.after(50, self._animate)
    
    def _update_3d_view(self, _=None):
        """Cập nhật góc nhìn của đồ thị 3D."""
        self.ax_3d.view_init(
            elev=self.elev_var.get(),
            azim=self.azim_var.get()
        )
        self.ax_3d.dist = self.zoom_var.get()
        self.canvas_3d.draw()
    
    def _update_plot(self):
        """Cập nhật tất cả các đồ thị."""
        # Lấy tham số hiện tại
        params = WaveParameters(
            amplitude=self.amp_var.get(),
            wavelength=self.wave_var.get(),
            depth=self.depth_var.get()
        )
        
        # Lấy trạng thái các hiệu ứng
        effects = {name: var.get() for name, var in self.effects_vars.items()}
        
        # Tính toán profile sóng
        eta, analysis = self.wave_model.calculate_wave_profile(params, self.time_var.get(), effects)
        
        # Cập nhật đồ thị 2D
        self.ax_2d.clear()
        self.ax_2d.plot(self.wave_model.x, eta)
        self.ax_2d.set_xlabel('Khoảng cách (m)')
        self.ax_2d.set_ylabel('Độ cao sóng (m)')
        self.ax_2d.set_title('Profile sóng thần 2D')
        self.ax_2d.grid(True)
        self.canvas_2d.draw()
        
        # Cập nhật đồ thị 3D
        self.ax_3d.clear()
        x = self.wave_model.x
        y = np.linspace(0, 100, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.tile(eta, (100, 1)) * np.cos(0.1 * Y)
        
        # Vẽ bề mặt với các tùy chọn mới
        if self.wireframe_var.get():
            self.ax_3d.plot_wireframe(X, Y, Z, alpha=self.alpha_var.get())
        else:
            surf = self.ax_3d.plot_surface(
                X, Y, Z,
                cmap=self.cmap_var.get(),
                alpha=self.alpha_var.get()
            )
            if self.contour_var.get():
                offset_z = np.min(Z) - 5
                self.ax_3d.contour(
                    X, Y, Z,
                    zdir='z',
                    offset=offset_z,
                    cmap=self.cmap_var.get()
                )
        
        self.ax_3d.set_xlabel('Khoảng cách (m)')
        self.ax_3d.set_ylabel('Chiều rộng (m)')
        self.ax_3d.set_zlabel('Độ cao sóng (m)')
        self.ax_3d.set_title('Mô phỏng sóng thần 3D')
        
        # Cập nhật góc nhìn
        self._update_3d_view()
        
        # Cập nhật phân tích
        self.analysis_text.delete(1.0, tk.END)
        for key, value in analysis.items():
            if key != 'energy_spectrum':
                self.analysis_text.insert(tk.END, f"{key}: {value}\n")
        
        # Cập nhật đồ thị phổ năng lượng
        self.ax_spectrum.clear()
        freq, spectrum = analysis['energy_spectrum']
        self.ax_spectrum.plot(freq[1:len(freq)//2], spectrum[1:len(spectrum)//2])
        self.ax_spectrum.set_xlabel('Tần số (Hz)')
        self.ax_spectrum.set_ylabel('Năng lượng phổ')
        self.ax_spectrum.set_title('Phổ năng lượng sóng')
        self.ax_spectrum.grid(True)
        self.canvas_spectrum.draw()

def main():
    app = TsunamiSimulator()
    app.mainloop()

if __name__ == "__main__":
    main()

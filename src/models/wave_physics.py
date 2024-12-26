"""
Module chứa các tính toán vật lý cho mô phỏng sóng thần.
"""
from typing import Tuple, Dict, List
from dataclasses import dataclass
import numpy as np
from scipy import signal

@dataclass
class WaveParameters:
    """Lớp chứa các thông số sóng."""
    amplitude: float
    wavelength: float
    depth: float
    bottom_friction: float = 0.001
    coriolis_param: float = 0.0001
    wind_speed: float = 0.0
    wind_direction: float = 0.0

class TsunamiWaveModel:
    """Mô hình tính toán sóng thần với các hiệu ứng vật lý nâng cao."""
    
    def __init__(self, domain_size: float = 1000.0, nx: int = 1000):
        self.domain_size = domain_size
        self.nx = nx
        self.dx = domain_size / nx
        self.x = np.linspace(0, domain_size, nx)
        self.g = 9.81  # Gia tốc trọng trường (m/s^2)
        self.rho = 1025.0  # Mật độ nước biển (kg/m^3)
        self.earth_radius = 6371000  # Bán kính Trái Đất (m)
        
    def calculate_wave_profile(self, 
                             params: WaveParameters,
                             t: float,
                             include_effects: Dict[str, bool]) -> Tuple[np.ndarray, Dict]:
        """
        Tính toán profile sóng với nhiều hiệu ứng vật lý.
        
        Args:
            params: Thông số sóng
            t: Thời gian (s)
            include_effects: Dict chỉ định các hiệu ứng cần tính
            
        Returns:
            Tuple chứa profile sóng và các thông số tính toán
        """
        k = 2 * np.pi / params.wavelength
        omega = np.sqrt(self.g * k * np.tanh(k * params.depth))
        
        # Sóng cơ bản
        eta = params.amplitude * np.cos(k * self.x - omega * t)
        
        # Thêm các hiệu ứng
        if include_effects.get('nonlinear', False):
            eta = self._add_nonlinear_effects(eta, k, omega, t, params)
            
        if include_effects.get('dispersion', False):
            eta = self._add_dispersion_effects(eta, k, params.depth)
            
        if include_effects.get('bottom_friction', False):
            eta = self._add_bottom_friction(eta, params.bottom_friction, t)
            
        if include_effects.get('coriolis', False):
            eta = self._add_coriolis_effect(eta, params.coriolis_param, t)
            
        if include_effects.get('wind', False):
            eta = self._add_wind_effect(eta, params.wind_speed, params.wind_direction)
        
        # Tính các thông số phân tích
        analysis = self._analyze_wave(eta, k, omega, params)
        
        return eta, analysis
    
    def _add_nonlinear_effects(self, eta: np.ndarray, k: float, omega: float, 
                              t: float, params: WaveParameters) -> np.ndarray:
        """Thêm các hiệu ứng phi tuyến bậc cao."""
        # Stokes bậc 2
        eta2 = 0.5 * k * params.amplitude**2 * np.cos(2 * (k * self.x - omega * t))
        
        # Stokes bậc 3
        eta3 = (3/8) * k**2 * params.amplitude**3 * np.cos(3 * (k * self.x - omega * t))
        
        return eta + eta2 + eta3
    
    def _add_dispersion_effects(self, eta: np.ndarray, k: float, depth: float) -> np.ndarray:
        """Thêm hiệu ứng tán sắc."""
        beta = k * depth
        dispersion_factor = np.sqrt(np.tanh(beta) / beta)
        return eta * dispersion_factor
    
    def _add_bottom_friction(self, eta: np.ndarray, friction: float, t: float) -> np.ndarray:
        """Thêm hiệu ứng ma sát đáy."""
        decay = np.exp(-friction * t)
        return eta * decay
    
    def _add_coriolis_effect(self, eta: np.ndarray, f: float, t: float) -> np.ndarray:
        """Thêm hiệu ứng Coriolis."""
        phase_shift = f * t
        return eta * np.cos(phase_shift)
    
    def _add_wind_effect(self, eta: np.ndarray, wind_speed: float, 
                        wind_direction: float) -> np.ndarray:
        """Thêm hiệu ứng gió."""
        wind_factor = 0.0015 * wind_speed  # Hệ số kinh nghiệm
        wind_component = wind_factor * np.cos(wind_direction) * np.sin(2*np.pi*self.x/self.domain_size)
        return eta + wind_component
    
    def _analyze_wave(self, eta: np.ndarray, k: float, omega: float, 
                     params: WaveParameters) -> Dict:
        """Phân tích các đặc tính sóng."""
        # Tính năng lượng sóng
        E = 0.5 * self.rho * self.g * np.mean(eta**2)
        
        # Phổ năng lượng
        freq = np.fft.fftfreq(len(eta), d=self.dx)
        spectrum = np.abs(np.fft.fft(eta))**2
        
        # Tính momentum flux
        momentum_flux = self.rho * self.g * np.mean(eta**2) / 2
        
        # Tính vận tốc pha và nhóm
        c_phase = omega / k
        c_group = 0.5 * omega / k * (1 + 2*k*params.depth/np.sinh(2*k*params.depth))
        
        return {
            'energy': E,
            'energy_spectrum': (freq, spectrum),
            'momentum_flux': momentum_flux,
            'phase_velocity': c_phase,
            'group_velocity': c_group,
            'max_amplitude': np.max(np.abs(eta)),
            'rms_amplitude': np.sqrt(np.mean(eta**2)),
            'mean_wavelength': 2*np.pi/k,
            'period': 2*np.pi/omega
        }
        
    def calculate_tsunami_risk(self, eta: np.ndarray, params: WaveParameters) -> Dict:
        """Đánh giá rủi ro sóng thần."""
        max_height = np.max(np.abs(eta))
        energy = 0.5 * self.rho * self.g * np.mean(eta**2)
        
        # Phân loại mức độ rủi ro
        if max_height < 0.5:
            risk_level = "Thấp"
        elif max_height < 2.0:
            risk_level = "Trung bình"
        elif max_height < 5.0:
            risk_level = "Cao"
        else:
            risk_level = "Rất cao"
            
        return {
            'risk_level': risk_level,
            'max_height': max_height,
            'energy': energy,
            'potential_damage': self._estimate_damage(max_height, energy)
        }
    
    def _estimate_damage(self, height: float, energy: float) -> str:
        """Ước tính thiệt hại dựa trên chiều cao và năng lượng sóng."""
        if height < 1.0:
            return "Thiệt hại nhẹ đến các công trình ven biển"
        elif height < 3.0:
            return "Thiệt hại đáng kể đến các công trình ven biển, ngập lụt vùng thấp"
        elif height < 6.0:
            return "Thiệt hại nghiêm trọng đến công trình, ngập lụt sâu trong đất liền"
        else:
            return "Thiệt hại toàn diện đến cơ sở hạ tầng ven biển"

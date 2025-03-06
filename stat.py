import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import pandas as pd

class ESP32DataMonitor:
    def __init__(self, port='COM6', baud_rate=115200, history_size=100, 
                 z_score_threshold=3, rate_threshold=5, 
                 absolute_humidity_threshold=10, absolute_temp_threshold=5,
                 consecutive_anomalies_threshold=3):
        """
        Initialize the ESP32 data monitor with enhanced anomaly detection capabilities.
        
        Args:
            port: Serial port for ESP32 connection
            baud_rate: Baud rate for serial communication
            history_size: Number of data points to keep in memory
            z_score_threshold: Z-score threshold for statistical anomalies
            rate_threshold: Maximum allowed rate of change between consecutive readings
            absolute_humidity_threshold: Absolute change threshold for humidity (%)
            absolute_temp_threshold: Absolute change threshold for temperature (°C)
            consecutive_anomalies_threshold: Number of consecutive anomalies to trigger an alert
        """
        self.port = port
        self.baud_rate = baud_rate
        self.serial_conn = None
        self.history_size = history_size
        self.z_score_threshold = z_score_threshold
        self.rate_threshold = rate_threshold
        self.absolute_humidity_threshold = absolute_humidity_threshold
        self.absolute_temp_threshold = absolute_temp_threshold
        self.consecutive_anomalies_threshold = consecutive_anomalies_threshold
        
        self.timestamps = deque(maxlen=history_size)
        self.humidity_data = deque(maxlen=history_size)
        self.temperature_data = deque(maxlen=history_size)
        
        self.humidity_ewma = deque(maxlen=history_size)
        self.temperature_ewma = deque(maxlen=history_size)
        
        self.humidity_anomalies = deque(maxlen=history_size)
        self.temperature_anomalies = deque(maxlen=history_size)
        self.consecutive_h_anomalies = 0
        self.consecutive_t_anomalies = 0
        self.last_h_anomaly_time = 0
        self.last_t_anomaly_time = 0
        
        self.humidity_stats = {
            'mean': 0, 'std': 0, 'median': 0, 
            'ewma': 0, 'ewmstd': 0, 'alpha': 0.1, 'lambda': 0.94,
            'last_value': 0, 'baseline': None
        }
        self.temperature_stats = {
            'mean': 0, 'std': 0, 'median': 0, 
            'ewma': 0, 'ewmstd': 0, 'alpha': 0.1, 'lambda': 0.94,
            'last_value': 0, 'baseline': None
        }
        
        self.humidity_anomaly_score = 0
        self.temperature_anomaly_score = 0
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 10))
        self.setup_plot()
        
    def setup_plot(self):
        """Set up the real-time plot with moving averages and thresholds."""
        # Raw data and anomalies
        self.humidity_line, = self.ax1.plot([], [], 'b-', label='Humidity (%)')
        self.humidity_ewma_line, = self.ax1.plot([], [], 'g-', label='Humidity EWMA', linewidth=1.5, alpha=0.8)
        self.humidity_anomaly_points = self.ax1.scatter([], [], color='red', s=50, label='Anomalies')
        
        self.temperature_line, = self.ax2.plot([], [], 'r-', label='Temperature (°C)')
        self.temperature_ewma_line, = self.ax2.plot([], [], 'g-', label='Temperature EWMA', linewidth=1.5, alpha=0.8)
        self.temperature_anomaly_points = self.ax2.scatter([], [], color='red', s=50, label='Anomalies')
        
        self.h_upper_threshold, = self.ax1.plot([], [], 'k--', alpha=0.3)
        self.h_lower_threshold, = self.ax1.plot([], [], 'k--', alpha=0.3)
        self.t_upper_threshold, = self.ax2.plot([], [], 'k--', alpha=0.3)
        self.t_lower_threshold, = self.ax2.plot([], [], 'k--', alpha=0.3)
        
        self.ax1.set_title('ESP32 DHT11 Sensor Data with Anomaly Detection')
        self.ax1.set_ylabel('Humidity (%)')
        self.ax1.set_ylim(0, 100)
        self.ax1.legend(loc='upper right')
        self.ax1.grid(True)
        
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Temperature (°C)')
        self.ax2.set_ylim(0, 50)
        self.ax2.legend(loc='upper right')
        self.ax2.grid(True)
        
    def connect_to_esp32(self):
        """Connect to the ESP32 device."""
        try:
            self.serial_conn = serial.Serial(self.port, self.baud_rate, timeout=1)
            self.serial_conn.reset_input_buffer()
            print(f"Connected to ESP32 on {self.port}")
            time.sleep(2)  
            return True
        except Exception as e:
            print(f"Error connecting to ESP32: {e}")
            return False
    
    def read_data(self):
        """Read and parse data from ESP32."""
        if not self.serial_conn:
            return None, None
        
        try:
            if self.serial_conn.in_waiting:
                line = self.serial_conn.readline().decode('utf-8', errors='replace').strip()
                print(f"DEBUG: Received raw data: {line}") 
                if "Humidity" in line and "Temperature" in line:
                    parts = line.split('\t')
                    humidity = float(parts[0].split(':')[1].strip().split('%')[0])
                    temperature = float(parts[1].split(':')[1].strip().split('*C')[0])
                    return humidity, temperature
                else:
                    if any(keyword in line for keyword in ["Wi-Fi", "MQTT", "connecting"]):
                        print("DEBUG: Ignoring debug message from ESP32.")
        except Exception as e:
            print(f"Error reading data: {e}")
        
        return None, None
    
    def update_statistics(self, humidity, temperature):
        """Update statistical models with new data using enhanced methods."""
        current_time = time.time()
        self.timestamps.append(current_time)
        self.humidity_data.append(humidity)
        self.temperature_data.append(temperature)
        
        if self.humidity_stats['last_value'] != 0:
            h_rate_of_change = abs(humidity - self.humidity_stats['last_value'])
            t_rate_of_change = abs(temperature - self.temperature_stats['last_value'])
        else:
            h_rate_of_change = 0
            t_rate_of_change = 0
            
        self.humidity_stats['last_value'] = humidity
        self.temperature_stats['last_value'] = temperature
        
        if len(self.humidity_data) >= 10 and self.humidity_stats['baseline'] is None:
            self.humidity_stats['baseline'] = np.mean(self.humidity_data)
            self.temperature_stats['baseline'] = np.mean(self.temperature_data)
            
            self.humidity_stats['ewma'] = self.humidity_stats['baseline']
            self.temperature_stats['ewma'] = self.temperature_stats['baseline']
            
            self.humidity_stats['ewmstd'] = np.std(self.humidity_data) or 1.0  # Ensure non-zero
            self.temperature_stats['ewmstd'] = np.std(self.temperature_data) or 1.0  # Ensure non-zero
        
        if len(self.humidity_data) > 1:
            self.humidity_stats['mean'] = np.mean(self.humidity_data)
            self.humidity_stats['std'] = np.std(self.humidity_data) or 0.1  # Avoid division by zero
            self.humidity_stats['median'] = np.median(self.humidity_data)
            
            self.temperature_stats['mean'] = np.mean(self.temperature_data)
            self.temperature_stats['std'] = np.std(self.temperature_data) or 0.1  # Avoid division by zero
            self.temperature_stats['median'] = np.median(self.temperature_data)
            
            alpha = self.humidity_stats['alpha']
            self.humidity_stats['ewma'] = (alpha * humidity) + ((1 - alpha) * self.humidity_stats['ewma'])
            self.temperature_stats['ewma'] = (alpha * temperature) + ((1 - alpha) * self.temperature_stats['ewma'])
            
            lambda_param = self.humidity_stats['lambda']
            h_diff = humidity - self.humidity_stats['ewma']
            t_diff = temperature - self.temperature_stats['ewma']
            
            self.humidity_stats['ewmstd'] = np.sqrt(lambda_param * (self.humidity_stats['ewmstd']**2) + 
                                                   (1-lambda_param) * (h_diff**2))
            self.temperature_stats['ewmstd'] = np.sqrt(lambda_param * (self.temperature_stats['ewmstd']**2) + 
                                                     (1-lambda_param) * (t_diff**2))
            
            self.humidity_ewma.append(self.humidity_stats['ewma'])
            self.temperature_ewma.append(self.temperature_stats['ewma'])
            
            h_z_score = abs((humidity - self.humidity_stats['mean']) / max(0.1, self.humidity_stats['std']))
            t_z_score = abs((temperature - self.temperature_stats['mean']) / max(0.1, self.temperature_stats['std']))
            
            h_ewma_deviation = abs(humidity - self.humidity_stats['ewma']) / max(0.1, self.humidity_stats['ewmstd'])
            t_ewma_deviation = abs(temperature - self.temperature_stats['ewma']) / max(0.1, self.temperature_stats['ewmstd'])
            
            h_abs_deviation = abs(humidity - self.humidity_stats['baseline']) if self.humidity_stats['baseline'] else 0
            t_abs_deviation = abs(temperature - self.temperature_stats['baseline']) if self.temperature_stats['baseline'] else 0
            
            self.humidity_anomaly_score = max(
                h_z_score / self.z_score_threshold,
                h_ewma_deviation / 2.0, 
                h_rate_of_change / self.absolute_humidity_threshold,
                h_abs_deviation / (self.absolute_humidity_threshold * 2)  
            )
            
            self.temperature_anomaly_score = max(
                t_z_score / self.z_score_threshold,
                t_ewma_deviation / 2.0,  
                t_rate_of_change / self.absolute_temp_threshold,
                t_abs_deviation / (self.absolute_temp_threshold * 2)  
            )
    
    def detect_anomalies(self):
        """Detect anomalies using multiple methods and track consecutive anomalies."""
        if len(self.humidity_data) < 10:
            self.humidity_anomalies.append(False)
            self.temperature_anomalies.append(False)
            return
        
        latest_humidity = self.humidity_data[-1]
        latest_temperature = self.temperature_data[-1]
        
        h_anomaly = self.humidity_anomaly_score > 1.0
        t_anomaly = self.temperature_anomaly_score > 1.0
        
        if h_anomaly:
            self.consecutive_h_anomalies += 1
        else:
            self.consecutive_h_anomalies = 0
            
        if t_anomaly:
            self.consecutive_t_anomalies += 1
        else:
            self.consecutive_t_anomalies = 0
        
        self.humidity_anomalies.append(h_anomaly)
        self.temperature_anomalies.append(t_anomaly)
        
        current_time = time.time()
        min_report_interval = 5  
        
        should_report_h = h_anomaly and (
            self.humidity_anomaly_score > 2.0 or 
            self.consecutive_h_anomalies >= self.consecutive_anomalies_threshold or  
            (current_time - self.last_h_anomaly_time) > min_report_interval 
        )
        
        should_report_t = t_anomaly and (
            self.temperature_anomaly_score > 2.0 or 
            self.consecutive_t_anomalies >= self.consecutive_anomalies_threshold or  
            (current_time - self.last_t_anomaly_time) > min_report_interval  
        )
        
        if should_report_h or should_report_t:
            print(f"\nANOMALY DETECTED at {time.strftime('%H:%M:%S')}:")
            
            if should_report_h:
                severity = "HIGH" if self.humidity_anomaly_score > 2.5 else "MEDIUM" if self.humidity_anomaly_score > 1.5 else "LOW"
                pattern_msg = f", PATTERN DETECTED ({self.consecutive_h_anomalies} consecutive)" if self.consecutive_h_anomalies >= self.consecutive_anomalies_threshold else ""
                print(f"  Humidity: {latest_humidity:.1f}% (Anomaly score: {self.humidity_anomaly_score:.2f}, Severity: {severity}{pattern_msg})")
                print(f"  Expected range: {self.humidity_stats['ewma'] - 2*self.humidity_stats['ewmstd']:.1f}% - {self.humidity_stats['ewma'] + 2*self.humidity_stats['ewmstd']:.1f}%")
                self.last_h_anomaly_time = current_time
                
            if should_report_t:
                severity = "HIGH" if self.temperature_anomaly_score > 2.5 else "MEDIUM" if self.temperature_anomaly_score > 1.5 else "LOW"
                pattern_msg = f", PATTERN DETECTED ({self.consecutive_t_anomalies} consecutive)" if self.consecutive_t_anomalies >= self.consecutive_anomalies_threshold else ""
                print(f"  Temperature: {latest_temperature:.1f}°C (Anomaly score: {self.temperature_anomaly_score:.2f}, Severity: {severity}{pattern_msg})")
                print(f"  Expected range: {self.temperature_stats['ewma'] - 2*self.temperature_stats['ewmstd']:.1f}°C - {self.temperature_stats['ewma'] + 2*self.temperature_stats['ewmstd']:.1f}°C")
                self.last_t_anomaly_time = current_time
    
    def update_plot(self, frame):
        """Update the plot with new data including moving averages and thresholds."""
        timestamps_list = list(self.timestamps)
        if not timestamps_list:
            return self.humidity_line, self.temperature_line, self.humidity_anomaly_points, self.temperature_anomaly_points
        
        x_data = np.array(timestamps_list) - timestamps_list[0]
        
        self.humidity_line.set_data(x_data, self.humidity_data)
        self.temperature_line.set_data(x_data, self.temperature_data)
        
        if len(self.humidity_ewma) > 0:
            self.humidity_ewma_line.set_data(x_data[-len(self.humidity_ewma):], self.humidity_ewma)
            self.temperature_ewma_line.set_data(x_data[-len(self.temperature_ewma):], self.temperature_ewma)
            
            h_upper = [min(100, self.humidity_stats['ewma'] + 2*self.humidity_stats['ewmstd'])] * len(x_data)
            h_lower = [max(0, self.humidity_stats['ewma'] - 2*self.humidity_stats['ewmstd'])] * len(x_data)
            self.h_upper_threshold.set_data(x_data, h_upper)
            self.h_lower_threshold.set_data(x_data, h_lower)
            
            t_upper = [min(50, self.temperature_stats['ewma'] + 2*self.temperature_stats['ewmstd'])] * len(x_data)
            t_lower = [max(0, self.temperature_stats['ewma'] - 2*self.temperature_stats['ewmstd'])] * len(x_data)
            self.t_upper_threshold.set_data(x_data, t_upper)
            self.t_lower_threshold.set_data(x_data, t_lower)
        
        h_anomaly_indices = [i for i, anomaly in enumerate(self.humidity_anomalies) if anomaly]
        t_anomaly_indices = [i for i, anomaly in enumerate(self.temperature_anomalies) if anomaly]
        
        self.humidity_anomaly_points.set_offsets(np.column_stack([
            np.array(x_data)[h_anomaly_indices],
            np.array(self.humidity_data)[h_anomaly_indices]
        ]) if h_anomaly_indices else np.empty((0, 2)))
        
        self.temperature_anomaly_points.set_offsets(np.column_stack([
            np.array(x_data)[t_anomaly_indices],
            np.array(self.temperature_data)[t_anomaly_indices]
        ]) if t_anomaly_indices else np.empty((0, 2)))
        
        if len(x_data) > 0:
            self.ax1.set_xlim(max(0, x_data[-1] - 60), x_data[-1] + 5)
            self.ax2.set_xlim(max(0, x_data[-1] - 60), x_data[-1] + 5)
        
        return self.humidity_line, self.temperature_line, self.humidity_anomaly_points, self.temperature_anomaly_points
    
    def run(self):
        """Main loop to run the data monitor."""
        if not self.connect_to_esp32():
            print("Failed to connect to ESP32.")
            return
        
        ani = FuncAnimation(self.fig, self.update_plot, interval=1000, cache_frame_data=False)
        
        print("Starting ESP32 Data Monitor with Enhanced Anomaly Detection")
        print("═" * 70)
        print("Detection parameters:")
        print(f"• Z-score threshold: {self.z_score_threshold}")
        print(f"• Rate threshold: {self.rate_threshold} units/reading")
        print(f"• Absolute thresholds: {self.absolute_humidity_threshold}% (humidity), {self.absolute_temp_threshold}°C (temperature)")
        print(f"• Consecutive anomalies threshold: {self.consecutive_anomalies_threshold}")
        print("═" * 70)
        
        while True:
            humidity, temperature = self.read_data()
            if humidity is not None and temperature is not None:
                self.update_statistics(humidity, temperature)
                self.detect_anomalies()
            
            plt.pause(0.01)

if __name__ == "__main__":
    monitor = ESP32DataMonitor()
    monitor.run()

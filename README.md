# mmWave Vital Signs & Blood Pressure Estimation

A high-performance Python application for real-time vital signs monitoring and blood pressure estimation using **TI mmWave Radar (xWR68xx)**. This project extracts 13+ distinct physiological features from heart and respiration waveforms to estimate Systolic and Diastolic Blood Pressure.

## 🚀 Features

- **Real-time Monitoring**: Tracks Heart Rate (HR) and Respiration Rate (RR).
- **BP Estimation**: Standalone estimation of SBP and DBP using morphologic and spectral pulse wave features.
- **Precision Data Collection**: Automated 30-second data collection sessions.
- **ML-Ready Datasets**: Generates unique timestamped CSV files with 21+ columns of raw and processed features.
- **Signal Validation**: Includes a "Reliability Engine" that detects signal quality, motion artifacts, and "fake" signatures.
- **Automated Reporting**: Generates a visual validation report (`.png`) for every run showing vitals stability, BP trends, and quality scores.

## 🛠️ Hardware Requirements

- **Radar**: TI IWR6843AOPEVM / IWR6843ISK or similar xWR68xx series.
- **Mounting**: Front-facing (Chest) or Hand-top placement.
- **Connectivity**: Two USB-UART ports (Application/User UART and Auxiliary Data Port).

## 📋 Software Dependencies

Ensure you have Python 3.8+ installed.

```bash
pip install pyserial numpy scipy matplotlib
```

## 📂 Project Structure

- `vitalsigns.py`: Main execution script for data collection and estimation.
- `xwr68xx_profile_VitalSigns_20fps_Front.cfg`: Radar configuration profile.
- `BP_Datasets/`: Directory where CSV datasets and validation reports are saved.

## 🚦 Getting Started

1. **Connect the Radar**: Plug in your TI mmWave device.
2. **Check Ports**: Open Device Manager and identify the COM ports for "XDS110 Class Application/User UART" and "XDS110 Class Auxiliary Data Port".
3. **Configure Script**: Update `USER_PORT` and `DATA_PORT` in `vitalsigns.py` if they differ from the defaults (COM4/COM5).
4. **Run the Application**:
   ```bash
   python vitalsigns.py
   ```

## 📊 Extracted Features (CSV Schema)

Each row in the dataset provides:
- **Metadata**: Timestamp, User, SessionTime.
- **Core Vitals**: HR, RR, Range (Distance).
- **Time-Domain**: RMS, Energy, Peak-to-Peak, IBI Mean, IBI STD, Pulse Width, Rise Time, Skewness, Kurtosis.
- **Frequency-Domain**: Dominant Freq, Spectral Centroid, Spectral Entropy, Band Power (0.8-2.5Hz).
- **Outputs**: SBP_Estimate, DBP_Estimate.
- **Validation**: ReliabilityScore (0-100%).

## 🛡️ Validation System

The script evaluates data quality using three markers:
- **HR Variability**: Detects if the sensor is stuck on a static object.
- **Range Locking**: Filters out secondary reflections and minor body jitters.
- **Physiological Checking**: Ensures values are within humanly possible ranges.

---

**Disclaimer**: This project is for research and development purposes. It is not a medical-grade diagnostic tool.
"# Blood_pressure_estimation" 

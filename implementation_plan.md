# Aligning BP Estimation Project with Signal Processing Syllabus

This plan covers the comparison between your current code and your syllabus, establishes the clear white-box architecture you requested, and proposes the necessary code changes.

## 1. Topic Comparison: Current Code vs. Syllabus

Your goal is to use *only* the topics taught in your syllabus (the image provided) or replace non-compliant methods with syllabus-approved techniques.

| Current Feature / Method | Status | Corresponding Syllabus Topic | Proposed White-Box Replacement |
| :--- | :--- | :--- | :--- |
| Smoothing (`np.convolve`) | ✅ Compliant | **UNIT 1 & 4**: Convolution, Moving Average filter | Keep as is. |
| Filtering (`scipy.signal.butter`) | ✅ Compliant | **UNIT 4**: Band-pass filtering | Keep as is. |
| Frequency Analysis (`np.fft`) | ✅ Compliant | **UNIT 2**: DFT, Power Spectral Density | Keep as is. |
| Time Domain Metrics (RMS, Energy) | ✅ Compliant | **UNIT 1**: Energy and Power Signals | Keep as is. |
| **Skewness & Kurtosis** | ❌ Non-Compliant | *Not in syllabus (Pure statistics)* | **Replace** with **STFT Variance / Energy** (Unit 4). |
| **Spectral Entropy** | ❌ Non-Compliant | *Not in syllabus (Information Theory)* | **Replace** with **Wavelet Decomposition (DWT)** detail energy (Unit 3). |
| **Peak Finding (`find_peaks`)** | ❌ Non-Compliant | *Not in syllabus (Algorithmic Heuristic)* | **Replace** with **Autocorrelation** (Unit 1) to mathematically track signal repetition and find the Inter-Beat Interval (IBI). |
| **Pulse Width & Rise Time** | ❌ Non-Compliant | *Not in syllabus* | **Replace** with **Wavelet Edge Detection** (Unit 3) to quantify the sharpness of the heartbeat. |

## 2. The "White-Box" Pipeline Architecture

Here is the proper, transparent pipeline of how your signal is transformed into Blood Pressure. Because it is a "White Box" model, we rely strictly on deterministic math and signal processing, avoiding hidden neural network layers.

### Phase A: Raw Signal Collection (Sensor Hardware)
1. **Transmission:** The Texas Instruments Radar emits Frequency Modulated Continuous Wave (FMCW) millimeter-wave signals towards the user's chest.
2. **Reflection & Mixing:** The waves bounce off the chest wall. The radar mixes the returned wave with the transmitted wave to form an Intermediate Frequency (IF).
3. **On-Board Processing:** The radar’s DSP algorithms perform a **Range-FFT** to isolate the chest's distance. It then measures the **sub-millimeter phase shifts** of these waves over time. These minuscule shifts represent the physical expansion of the chest from breathing and the beating heart.

### Phase B: Data Transfer
* The radar’s internal processor packs this unwrapped phase data into binary Type-Length-Value (TLV) packets.
* It streams these packets over a high-speed serial UART connection (921600 Baud) to the computer in real-time.

### Phase C: Signal Processing & Feature Extraction (Computer Software)
Once the computer receives the raw phase data, we apply our syllabus-approved mathematical operations:
1. **Denoising (UNIT 4):** A Butterworth **Band-pass Filter** zeroes out non-heart frequencies, and a **Moving Average (Convolution)** smooths out jagged hardware noise.
2. **Computing Features (The White-Box Math):**
   * **Autocorrelation (UNIT 1):** We slide the signal over itself and multiply ($R[k] = \sum x[n]x[n-k]$). The time delay where the signal perfectly aligns with its own next repetition gives us the exact **Inter-Beat Interval (IBI)** and Heart Rate.
   * **Signal Energy (UNIT 1):** We calculate the sum of squared amplitudes to determine the mechanical force of the heart.
   * **Discrete Fourier Transform (UNIT 2):** We convert the time-domain signal to the frequency domain to calculate the **Dominant Frequency** and **Power Spectral Density**.
   * **Discrete Wavelet Transform (UNIT 3):** We decompose the signal into approximation and detail coefficients. The energy of the high-frequency *detail* coefficients represents the sharp, sudden edge of the systolic blood pump.

### Phase D: Blood Pressure Estimation (Linear White Box)
We map the physical features directly to BP using a multivariable linear equation. It ensures interpretable cause-and-effect:
* `Systolic BP  = Base + (w1 * Autocorrelation_HR) + (w2 * Signal_Energy) + (w3 * Wavelet_Detail_Energy)`
* `Diastolic BP = Base + (k1 * Autocorrelation_HR) + (k2 * Signal_Energy) + (k3 * Wavelet_Detail_Energy)`

---

## User Review Required
Please review the analysis above. 
If you are satisfied with this architecture and the proposed feature replacements (using Autocorrelation instead of peak finding, and wavelets instead of stats), **reply with an approval**.
I will then rewrite the `extract_bp_features` function in your `vitalsigns.py` script to be 100% strictly aligned with these syllabus methods and remove the non-compliant code.

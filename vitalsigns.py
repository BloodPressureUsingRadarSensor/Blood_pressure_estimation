import serial, struct, time, sys, os, csv, datetime, statistics, json
import numpy as np
from scipy.signal import butter, filtfilt, detrend, find_peaks, peak_widths
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

# ================= CONFIG =================
USER_EMAIL = sys.argv[1] if len(sys.argv) > 1 else "data_collection_user"
CONFIG_TYPE = int(sys.argv[2]) if len(sys.argv) > 2 else 0

USER_PORT = "COM4"
DATA_PORT = "COM5"
USER_BAUD = 115200
DATA_BAUD = 921600

CFG_FILE = r"E:\Document\spml\xwr68xx_profile_VitalSigns_20fps_Front.cfg"

FPS = 20
RUN_TIME = 30  # Fixed 30 seconds collection
BUF_LEN = 400  # Buffer for 20 seconds of data for feature extraction
MAGIC = b'\x02\x01\x04\x03\x06\x05\x08\x07'

# ================= CSV SETUP =================
CSV_DIR = r"E:\Document\spml\BP_Datasets"
os.makedirs(CSV_DIR, exist_ok=True)

# Generate unique filename with timestamp
timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
MASTER_CSV = os.path.join(CSV_DIR, f"bp_dataset_{timestamp_str}.csv")

csv_file = open(MASTER_CSV, "w", newline="") # Open in write mode for new file
csv_writer = csv.writer(csv_file)

# Always write header for new unique file
csv_writer.writerow([
    "Timestamp","User","SessionTime",
    "HR","RR","Range_m",
    "RMS","Energy","PeakToPeak",
    "IBI_Mean","IBI_STD",
    "Skewness","Kurtosis",
    "DominantFreq","SpectralCentroid","SpectralEntropy",
    "PulseWidth","RiseTime","BandPower",
    "SBP_Estimate","DBP_Estimate",
    "ReliabilityScore"
])
csv_file.flush()

last_status_len = 0

def write_status(message):
    global last_status_len
    padded = message
    if last_status_len > len(message):
        padded += " " * (last_status_len - len(message))
    sys.stdout.write("\r" + padded)
    sys.stdout.flush()
    last_status_len = len(message)

# ================= DATA VALIDATION HELPERS =================
def calculate_reliability(hr, rr, range_m, hr_history, range_history):
    score = 100
    
    # 1. Check for "Frozen" HR (TI chip sometimes gets stuck)
    if len(hr_history) > 20: 
        # Real HR always has some jitter (HRV)
        if statistics.stdev(hr_history[-20:]) < 0.01:
            score -= 40 # High penalty for artificial-looking data
            
    # 2. Check for Range stability
    if len(range_history) > 10:
        if abs(range_m - statistics.median(range_history[-10:])) > 0.2:
            score -= 30 # Tracking loss or sudden large movement
            
    # 3. Physiological Plausibility
    if not (45 <= hr <= 160): score -= 20
    if not (8 <= rr <= 35): score -= 10
    
    return max(0, score)

# ================= SIGNAL PROCESSING =================
def bandpass(sig, low, high, fs, order=4):
    if len(sig) < fs * 2:
        return sig
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype="band")
    return filtfilt(b, a, sig)

def smooth(x, w=5):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w)/w, mode="same")

def normalize(x):
    s = np.std(x)
    return x if s == 0 else x/s

def extract_bp_features(sig, fs):
    if len(sig) < fs * 5:
        return None
    sig = np.array(sig)
    
    # Time Domain
    rms = np.sqrt(np.mean(sig**2))
    energy = np.sum(sig**2)
    p2p = np.max(sig) - np.min(sig)
    try:
        sk = skew(sig)
        ku = kurtosis(sig)
    except:
        sk, ku = 0, 0
        
    peaks, _ = find_peaks(sig, distance=fs*0.4)
    ibi_mean = ibi_std = pulse_width = rise_time = 0
    
    if len(peaks) > 1:
        ibi = np.diff(peaks) / fs
        ibi_mean = np.mean(ibi)
        ibi_std = np.std(ibi)
        
        # Pulse Width & Rise Time
        try:
            w_data = peak_widths(sig, peaks, rel_height=0.5)[0]
            pulse_width = np.mean(w_data) / fs if len(w_data) > 0 else 0
        except:
            pulse_width = 0
        
        r_times = []
        for p in peaks:
            start = max(0, p - int(0.3 * fs))
            pre = sig[start:p]
            if len(pre) > 0:
                l_min = np.argmin(pre) + start
                amp = sig[p] - sig[l_min]
                if amp > 0:
                    i10 = np.where(sig[l_min:p] >= sig[l_min] + 0.1*amp)[0]
                    i90 = np.where(sig[l_min:p] >= sig[l_min] + 0.9*amp)[0]
                    if len(i10) > 0 and len(i90) > 0:
                        r_times.append((i90[0] - i10[0]) / fs)
        rise_time = np.mean(r_times) if len(r_times) > 0 else 0

    # Frequency Domain
    fft = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(len(sig), 1/fs)
    power = np.abs(fft)**2
    
    dom_freq = freqs[np.argmax(power)]
    spec_centroid = np.sum(freqs * power) / (np.sum(power) + 1e-8)
    prob = power / (np.sum(power) + 1e-8)
    spec_entropy = -np.sum(prob * np.log(prob + 1e-8))
    
    # Power in 0.8-2.5 Hz band
    mask = (freqs >= 0.8) & (freqs <= 2.5)
    band_power = np.sum(power[mask])

    return [
        rms, energy, p2p, ibi_mean, ibi_std, sk, ku, 
        dom_freq, spec_centroid, spec_entropy, pulse_width, rise_time, band_power
    ]

def estimate_blood_pressure(hr, rr, features, config_type=0):
    if hr < 30: return 0, 0
    if config_type == 0: # Chest
        sbp = 102 + 0.38 * hr + 0.22 * rr + 4.5 * features[2]
        dbp = 64 + 0.22 * hr + 0.12 * rr + 1.8 * features[2]
    else: # Hand
        sbp = 106 + 0.42 * hr + 3.8 * features[2]
        dbp = 68 + 0.26 * hr + 1.4 * features[2]
    return np.clip(sbp, 85, 170), np.clip(dbp, 55, 105)

def extract_range_m(tlv, last_range=None):
    # Scan for potential range values in the Vital Signs TLV payload
    candidates = []
    for offset in range(64, min(128, len(tlv)), 4):
        try:
            val = struct.unpack_from("<f", tlv, offset)[0]
            if 0.2 < val < 3.0:
                candidates.append(val)
        except:
            pass
    
    if not candidates:
        return None
        
    if last_range is None:
        return candidates[0] # Initial guess
        
    # Find the candidate closest to the last known stable range
    # This prevents "flickering" to background objects or different body parts
    best_val = min(candidates, key=lambda x: abs(x - last_range))
    
    # Only update if the change is significant (>5cm) to filter small jitters
    # but small enough to be real motion (<30cm)
    diff = abs(best_val - last_range)
    if diff < 0.05: # Ignore jitters smaller than 5cm
        return last_range
    
    return best_val

# ================= SERIAL INIT =================
print("Initializing Radar...")
user = serial.Serial(USER_PORT, USER_BAUD, timeout=2)
data = serial.Serial(DATA_PORT, DATA_BAUD, timeout=0)

time.sleep(1)
user.write(b"sensorStop\n")
time.sleep(0.5)

with open(CFG_FILE) as f:
    for l in f:
        if l.strip() and not l.startswith("%"):
            user.write((l.strip()+"\n").encode())
            time.sleep(0.01)

user.write(b"sensorStart\n")
print(f"Radar started. Collecting BP data for 30 seconds for User: {USER_EMAIL}...")

# ================= BUFFERS =================
rx = bytearray()
heart_phase = []
range_history = [] # Median filter buffer
all_range_history = [] # For plotting
hr_history = []
rr_history = []
sbp_history = []
dbp_history = []
reliability_history = []
time_history = []
start_time = time.time()
sample_count = 0

# ================= MAIN LOOP =================
try:
    while time.time() - start_time < RUN_TIME:
        rx.extend(data.read(4096))
        
        while True:
            i = rx.find(MAGIC)
            if i < 0 or len(rx) < i + 40:
                break

            plen = struct.unpack("<I", rx[i+12:i+16])[0]
            if len(rx) < i + plen:
                break

            pkt = rx[i:i+plen]
            rx = rx[i+plen:]
            pay = pkt[40:]

            off = 0
            while off + 8 <= len(pay):
                t, l = struct.unpack_from("<II", pay, off)
                off += 8

                if t == 6: # Vital Signs TLV
                    # Reference: 68xx Vital Signs lab offsets
                    # bp (breath filter) at offset 28
                    # hp (heart filter) at offset 32
                    # hr (heart rate) at offset 36
                    # rr (breath rate) at offset 52
                    try:
                        bp = struct.unpack_from("<f", pay, off+28)[0]
                        hp = struct.unpack_from("<f", pay, off+32)[0]
                        hr = struct.unpack_from("<f", pay, off+36)[0]
                        rr = struct.unpack_from("<f", pay, off+52)[0]
                        
                        # Improved Range Tracking
                        last_r = range_history[-1] if range_history else None
                        range_m = extract_range_m(pay[off:off+l], last_r)
                        
                        range_sm = 0
                        if range_m:
                            range_history.append(range_m)
                            range_history[:] = range_history[-15:] # Larger median buffer for stability
                            range_sm = statistics.median(range_history)

                        heart_phase.append(hp)
                        heart_phase[:] = heart_phase[-BUF_LEN:]
                        sample_count += 1

                        # During the warmup period, show collection progress until
                        # enough heart samples are available for feature extraction.
                        if len(heart_phase) < 100 and sample_count % 20 == 0:
                            write_status(
                                f"Warming up: {len(heart_phase):3d}/100 heart samples"
                                f" | Range: {range_sm:.2f}m"
                            )

                        if len(heart_phase) >= 100:
                            heart_wave = bandpass(smooth(detrend(np.unwrap(heart_phase))), 0.8, 2.5, FPS)
                            feats = extract_bp_features(heart_wave, FPS)
                            
                            if feats:
                                sbp, dbp = estimate_blood_pressure(hr, rr, feats, CONFIG_TYPE)
                                
                                # Validation logic
                                rel_score = calculate_reliability(hr, rr, range_sm, hr_history, range_history)
                                
                                hr_history.append(hr)
                                rr_history.append(rr)
                                sbp_history.append(sbp)
                                dbp_history.append(dbp)
                                reliability_history.append(rel_score)
                                all_range_history.append(range_sm)
                                time_history.append(time.time() - start_time)

                                # Log to CSV
                                csv_writer.writerow([
                                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                                    USER_EMAIL,
                                    f"{time.time() - start_time:.2f}",
                                    f"{hr:.2f}", f"{rr:.2f}", f"{range_sm:.3f}",
                                    *feats,
                                    f"{sbp:.1f}", f"{dbp:.1f}",
                                    f"{rel_score}"
                                ])
                                csv_file.flush()
                                
                                # Console Update with Quality Feedback (Fixed flickering)
                                status = "TRUSTED" if rel_score > 70 else "UNSTABLE" if rel_score > 40 else "FAKE/NOISY"
                                write_status(
                                    f"Time: {time.time()-start_time:4.1f}s"
                                    f" | HR: {hr:5.1f}"
                                    f" | RR: {rr:4.1f}"
                                    f" | BP: {sbp:.0f}/{dbp:.0f}"
                                    f" | Quality: {rel_score}% ({status})"
                                )
                    except Exception as e:
                        print(f"\nUnpacking error: {e}")

                off += l
                
except KeyboardInterrupt:
    print("\nInterrupted by user.")

# ================= CLEANUP =================
user.write(b"sensorStop\n")
user.close()
data.close()
csv_file.close()

print(f"\nData collection complete. Dataset saved at: {MASTER_CSV}")

# ================= GENERATE VALIDATION REPORT =================
if time_history:
    print("\nGenerating Validation Report Chart...")
    plt.figure(figsize=(12, 10))
    
    # 1. Heart Rate & Respiration
    plt.subplot(4, 1, 1)
    plt.plot(time_history, hr_history, 'r-', label='Heart Rate (BPM)')
    plt.plot(time_history, [r * 4 for r in rr_history], 'c--', alpha=0.5, label='RR (x4 for scale)')
    plt.ylabel('BPM')
    plt.title('Vitals & Reliability Report')
    plt.legend()
    plt.grid(True)

    # 2. Blood Pressure
    plt.subplot(4, 1, 2)
    plt.plot(time_history, sbp_history, 'm-', label='Systolic BP (SBP)')
    plt.plot(time_history, dbp_history, 'm--', label='Diastolic BP (DBP)')
    plt.ylabel('mmHg')
    plt.legend()
    plt.grid(True)

    # 3. Reliability Score
    plt.subplot(4, 1, 3)
    # Fill quality regions
    plt.axhspan(70, 100, color='green', alpha=0.1)
    plt.axhspan(40, 70, color='yellow', alpha=0.1)
    plt.axhspan(0, 40, color='red', alpha=0.1)
    plt.plot(time_history, reliability_history, 'g-', lw=2, label='Reliability Score (%)')
    plt.ylim(0, 110)
    plt.ylabel('Quality %')
    plt.legend()
    plt.grid(True)

    # 4. Range Stability
    plt.subplot(4, 1, 4)
    plt.plot(time_history, all_range_history, 'b-', label='Range Stability (m)')
    plt.ylabel('Distance (m)')
    plt.xlabel('Session Time (s)')
    plt.legend()
    plt.grid(True)

    report_img = MASTER_CSV.replace(".csv", "_report.png")
    plt.tight_layout()
    plt.savefig(report_img)
    print(f"Report image saved: {report_img}")

    avg_qual = sum(reliability_history) / len(reliability_history)
    print("\n" + "="*40)
    print(f"FINAL DATA QUALITY REPORT")
    print(f"Overall Reliability: {avg_qual:.1f}%")
    if avg_qual > 80:
        print("RESULT: HIGH QUALITY DATA (Authenticated)")
    elif avg_qual > 50:
        print("RESULT: MEDIUM QUALITY (Minor Noise/Artifacts)")
    else:
        print("RESULT: REJECTED (Abnormal/Fake signature detected)")
    print("="*40)
    
    print("\nDisplaying Blood Pressure Chart...")
    plt.show() # Keep chart window open

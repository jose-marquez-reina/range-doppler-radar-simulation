import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

# ============================================================
# Basic radar and simulation parameters
# ============================================================

c = 3.0e8              # speed of light in m per s
fc = 10.0e9            # carrier frequency in Hz
bandwidth = 20.0e6     # chirp bandwidth in Hz
pulse_width = 20e-6    # pulse width in s
pri = 200e-6           # pulse repetition interval in s
fs = 80.0e6            # sampling frequency in Hz
num_pulses = 64        # number of pulses in the coherent burst

# Target parameters
target_range_0 = 1000
target_velocity = 0  # 120 m/s (~270 mph), nice Doppler
rcs = 1.0              # radar cross section scale
radar_constant = 1e10 

noise_power = 1e-6
# noise power for additive white Gaussian noise

# ============================================================
# Helper functions
# ============================================================

def generate_chirp(pulse_width, bandwidth, fs):
    """Baseband linear FM chirp."""
    t = np.arange(0, pulse_width, 1.0 / fs)
    k = bandwidth / pulse_width  # chirp rate
    # complex baseband chirp
    s = np.exp(1j * np.pi * k * t**2)
    return t, s

def simulate_received_pulses(tx_pulse, t_pulse, num_pulses, fs,
                             target_range_0, target_velocity,
                             pri, rcs, noise_power):
    """
    Simulate a single moving target.
    Returns array with shape num_pulses x num_samples.
    """
    num_samples = len(tx_pulse)
    rx = np.zeros((num_pulses, num_samples), dtype=complex)

    for m in range(num_pulses):
        # time of this pulse
        tm = m * pri

        # target range at this pulse
        R = target_range_0 + target_velocity * tm

        # propagation delay round trip
        tau = 2.0 * R / c

        # integer sample delay
        sample_delay = int(np.round(tau * fs))

        # if delay is beyond our window, target is not seen
        if sample_delay >= num_samples:
            # only noise
            rx[m, :] = np.sqrt(noise_power / 2) * (
                np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
            )
            continue

        # amplitude loss with one over r to the fourth
        path_loss = radar_constant * rcs / (R**4)

        # Doppler frequency from radial velocity
        fd = 2.0 * target_velocity * fc / c  # Hz

        # create shifted version of the transmit pulse
        echo = np.zeros(num_samples, dtype=complex)
        valid_len = num_samples - sample_delay
        echo[sample_delay:] = tx_pulse[:valid_len]

        # apply Doppler and path loss
        t = np.arange(num_samples) / fs
        doppler_phase = np.exp(1j * 2.0 * np.pi * fd * (t + tm))
        echo = echo * doppler_phase * path_loss

        # add white Gaussian noise
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
        )

        rx[m, :] = echo + noise

    return rx

def matched_filter(rx, tx):
    """
    Range compression using matched filter.
    Uses frequency domain convolution for speed.
    """
    # time reversed conjugate
    h = np.conj(tx[::-1])

    num_pulses, num_samples = rx.shape
    out = np.zeros_like(rx, dtype=complex)

    for m in range(num_pulses):
        conv = fftconvolve(rx[m, :], h, mode="same")
        out[m, :] = conv

    return out

def doppler_processing(range_compressed):
    """
    Doppler FFT across pulses for each range bin.
    range_compressed shape: num_pulses x num_samples
    Returns range_doppler_map shape: num_pulses x num_samples
    """
    # apply window across pulses if desired
    num_pulses, num_samples = range_compressed.shape
    window = np.hanning(num_pulses).reshape(-1, 1)

    data_win = range_compressed * window

    # Doppler FFT along pulse dimension
    rd = np.fft.fftshift(np.fft.fft(data_win, axis=0), axes=0)
    return rd

def simple_cfar_1d(power_db, guard_cells=2, training_cells=8, threshold_scale=8.0):
    return np.zeros_like(power_db, dtype=bool)

    """
    Simple one dimensional CA CFAR across range for each Doppler bin.
    power_db is 2D array in dB  shape: num_doppler x num_range.
    Returns boolean mask of detections.
    """
    num_doppler, num_range = power_db.shape
    detections = np.zeros_like(power_db, dtype=bool)

    for d in range(num_doppler):
        for r in range(num_range):
            start = max(0, r - guard_cells - training_cells)
            end = min(num_range, r + guard_cells + training_cells + 1)

            # exclude guard cells and cell under test
            guard_start = max(0, r - guard_cells)
            guard_end = min(num_range, r + guard_cells + 1)

            ref_indices = list(range(start, guard_start)) + list(range(guard_end, end))
            if len(ref_indices) < 1:
                continue

            noise_level = np.mean(power_db[d, ref_indices])
            threshold = noise_level + threshold_scale
            if power_db[d, r] > threshold:
                detections[d, r] = True

    return detections

# ============================================================
# Run the simulation
# ============================================================

# 1. transmit waveform
t_pulse, tx_pulse = generate_chirp(pulse_width, bandwidth, fs)

# 2. simulate received data for many pulses
rx = simulate_received_pulses(
    tx_pulse,
    t_pulse,
    num_pulses,
    fs,
    target_range_0,
    target_velocity,
    pri,
    rcs,
    noise_power,
)

# 3. range compression by matched filter
rc = matched_filter(rx, tx_pulse)

# 4. build range Doppler map
rd = doppler_processing(rc)

# 5. magnitude in dB
rd_power = np.abs(rd) ** 2
rd_db = 10.0 * np.log10(rd_power + 1e-12)

# 6. simple CFAR across range for each Doppler bin
detections = simple_cfar_1d(rd_db)

# ============================================================
# Axes for plots
# ============================================================

num_samples = len(t_pulse)
rng_res = c / (2.0 * bandwidth)
ranges = np.arange(num_samples) * (c / (2.0 * fs))  # crude range axis
doppler_bins = np.arange(num_pulses) - num_pulses // 2
prf = 1.0 / pri
doppler_freqs = doppler_bins * prf / num_pulses
velocities = doppler_freqs * c / (2.0 * fc)

# ============================================================
# Plot time waveform and matched filter output
# ============================================================

plt.figure()
plt.plot(t_pulse * 1e6, np.real(tx_pulse))
plt.xlabel("Time microseconds")
plt.ylabel("Real part of transmit pulse")
plt.title("Transmit chirp pulse real part")
plt.grid(True)

plt.figure()
plt.plot(ranges, np.abs(rc[num_pulses // 2, :]))
plt.xlabel("Range meters")
plt.ylabel("Magnitude")
plt.title("Range compressed output for one pulse")
plt.grid(True)

# ============================================================
# Plot range Doppler map
# ============================================================

plt.figure()
extent = [ranges[0], ranges[-1], velocities[0], velocities[-1]]
plt.imshow(
    rd_db,
    aspect="auto",
    origin="lower",
    extent=extent,
    cmap="viridis",
)
plt.xlabel("Range meters")
plt.ylabel("Velocity m per s")
plt.title("Range Doppler map dB")
plt.colorbar(label="Power dB")

# overlay detections
det_y, det_x = np.where(detections)
plt.scatter(ranges[det_x], velocities[det_y], s=10, edgecolors="white", facecolors="none")

plt.tight_layout()
# Save high-res range–Doppler figure for portfolio / resume
plt.figure(3)  # assuming RD map is figure 3
plt.savefig("range_doppler_map.png", dpi=300)
print("Saved range_doppler_map.png")

plt.show()

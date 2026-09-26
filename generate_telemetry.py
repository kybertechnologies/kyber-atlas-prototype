"""Simulate 30 days of 15-minute grid telemetry for the IEEE 14-bus network.

Run: python generate_telemetry.py   ->   writes atlas_training_data.csv
"""

import numpy as np
import pandas as pd

OUTPUT_FILE = "atlas_training_data.csv"
START = "2026-09-01 00:00"
DAYS = 30
INTERVAL = "15min"
SEED = 42

V_OVERNIGHT, V_PEAK = 0.98, 1.02
MIN_LOAD_FRACTION = 0.6
VOLTAGE_NOISE_STD = 0.005
POWER_NOISE_STD = 0.5
NOMINAL_FREQUENCY_HZ = 60.0
SYSTEM_FREQUENCY_NOISE_STD = 0.01
SENSOR_FREQUENCY_NOISE_STD = 0.002
FREQUENCY_LOAD_DROOP_HZ = 0.02

# IEEE 14-bus (MATPOWER case14) loads, used as each bus's peak demand: bus -> (P MW, Q MVAr).
# Buses 1, 7 and 8 carry no load, so their power readings are sensor noise around 0.
BUS_LOADS = {
    1: (0.0, 0.0), 2: (21.7, 12.7), 3: (94.2, 19.0), 4: (47.8, -3.9), 5: (7.6, 1.6),
    6: (11.2, 7.5), 7: (0.0, 0.0), 8: (0.0, 0.0), 9: (29.5, 16.6), 10: (9.0, 5.8),
    11: (3.5, 1.8), 12: (6.1, 1.6), 13: (13.5, 5.8), 14: (14.9, 5.0),
}
BUS_TYPES = {1: "slack", 2: "generator", 3: "generator", 6: "generator", 8: "generator", 7: "junction"}


def daily_shape(hour: np.ndarray) -> np.ndarray:
    """0 at the 02:00-05:00 minimum, 1 across 09:00-17:00, smooth cosine ramps in between."""
    hour = np.asarray(hour, dtype=float) % 24
    shape = np.zeros_like(hour)

    rising = (hour >= 5) & (hour < 9)
    shape[rising] = 0.5 - 0.5 * np.cos(np.pi * (hour[rising] - 5) / 4)

    shape[(hour >= 9) & (hour <= 17)] = 1.0

    # Evening decline runs 17:00 -> 02:00 the next day (9 hours).
    falling = (hour > 17) | (hour < 2)
    hours_since_17 = (hour[falling] - 17) % 24
    shape[falling] = 0.5 + 0.5 * np.cos(np.pi * hours_since_17 / 9)
    return shape


def simulate(rng: np.random.Generator) -> pd.DataFrame:
    timestamps = pd.date_range(START, periods=DAYS * 24 * 4, freq=INTERVAL)
    hours = timestamps.hour + timestamps.minute / 60
    shape = daily_shape(hours)
    n_times = len(timestamps)

    # Frequency is system-wide, dipping slightly as load rises; each bus adds its own sensor noise.
    system_frequency = (
        NOMINAL_FREQUENCY_HZ
        - FREQUENCY_LOAD_DROOP_HZ * (shape - 0.5)
        + rng.normal(0, SYSTEM_FREQUENCY_NOISE_STD, n_times)
    )
    load_factor = MIN_LOAD_FRACTION + (1 - MIN_LOAD_FRACTION) * shape
    voltage_curve = V_OVERNIGHT + (V_PEAK - V_OVERNIGHT) * shape

    frames = []
    for bus, (peak_p, peak_q) in BUS_LOADS.items():
        frames.append(pd.DataFrame({
            "timestamp": timestamps,
            "bus_number": bus,
            "bus": f"Bus {bus}",
            "bus_type": BUS_TYPES.get(bus, "load"),
            "voltage_pu": voltage_curve + rng.normal(0, VOLTAGE_NOISE_STD, n_times),
            "active_power_mw": peak_p * load_factor + rng.normal(0, POWER_NOISE_STD, n_times),
            "reactive_power_mvar": peak_q * load_factor + rng.normal(0, POWER_NOISE_STD, n_times),
            "frequency_hz": system_frequency + rng.normal(0, SENSOR_FREQUENCY_NOISE_STD, n_times),
        }))

    data = pd.concat(frames).sort_values(["timestamp", "bus_number"]).drop(columns="bus_number")
    return data.reset_index(drop=True).round({
        "voltage_pu": 4, "active_power_mw": 3, "reactive_power_mvar": 3, "frequency_hz": 4,
    })


def print_summary(data: pd.DataFrame) -> None:
    measurements = ["voltage_pu", "active_power_mw", "reactive_power_mvar", "frequency_hz"]
    print(f"Saved {OUTPUT_FILE}")
    print(f"Total rows:  {len(data):,} ({data.timestamp.nunique():,} timestamps x {data.bus.nunique()} buses)")
    print(f"Date range:  {data.timestamp.min()} to {data.timestamp.max()} ({DAYS} days, every 15 minutes)")

    print("\nDescriptive statistics")
    print(data[measurements].describe().round(4).to_string())

    hourly = data.assign(hour=data.timestamp.dt.hour).groupby("hour")
    by_hour = pd.DataFrame({
        "Mean voltage (pu)": hourly.voltage_pu.mean(),
        "Total load (MW)": hourly.active_power_mw.mean() * data.bus.nunique(),
    })
    print("\nDaily pattern check (averaged over all days)")
    print(by_hour.loc[[0, 3, 6, 9, 12, 15, 18, 21]].round(3).to_string())


def main() -> None:
    data = simulate(np.random.default_rng(SEED))
    data.to_csv(OUTPUT_FILE, index=False)
    print_summary(data)


if __name__ == "__main__":
    main()
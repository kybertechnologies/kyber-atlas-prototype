"""Per-bus Isolation Forest anomaly detection for the IEEE 14-bus telemetry dataset.

Run: python train_anomaly_models.py
Needs atlas_training_data.csv from generate_telemetry.py. Saves models to atlas_models/.
"""

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

DATA_FILE = "atlas_training_data.csv"
MODEL_DIR = Path("atlas_models")
SEED = 42

MEASUREMENTS = ["voltage_pu", "active_power_mw", "reactive_power_mvar", "frequency_hz"]
FEATURES = [f"{m}_z" for m in MEASUREMENTS] + ["max_abs_z", "rms_z"]
BUSES = [f"Bus {i}" for i in range(1, 15)]

READINGS_PER_DAY = 96
FAULT_START = 48  # 12:00, when normal voltage sits at its 1.02 pu plateau
FAULT_DURATION = 8  # 2 hours of 15-minute readings
NOMINAL_FREQUENCY_HZ = 60.0

# The decision threshold sits at the 99.5th percentile of training scores. An alarm needs PERSISTENCE
# consecutive anomalous readings, which filters out one-off noise at the cost of one extra reading.
CONTAMINATION = 0.005
PERSISTENCE = 2

# Isolation Forest anomaly score is -score_samples(): ~0.45 for typical readings, ~0.62 at the
# alarm threshold. Scores level off around 0.71-0.77 once a reading is ~6 noise standard deviations
# from normal, so the bands separate borderline, moderate and severe anomalies but cannot rank two
# faults that are both far outside normal. Severity applies to the peak score during a fault.
SEVERITY_BANDS = [(0.75, "CRITICAL"), (0.70, "HIGH"), (0.66, "MEDIUM"), (0.0, "LOW")]


@dataclass
class Injection:
    bus: str
    column: str
    value: float
    delay: int = 0  # readings after the scenario's fault start


@dataclass
class Scenario:
    name: str
    description: str
    injections: list[Injection] = field(default_factory=list)

    @property
    def expected_buses(self) -> set[str]:
        return {i.bus for i in self.injections}


SCENARIOS = [
    Scenario("1. Voltage sag", "Bus 5 voltage drops to 0.72 pu (downstream short circuit)",
             [Injection("Bus 5", "voltage_pu", 0.72)]),
    Scenario("2. Voltage spike", "Bus 9 voltage rises to 1.18 pu (load rejection)",
             [Injection("Bus 9", "voltage_pu", 1.18)]),
    Scenario("3. Disconnection", "Bus 3 active and reactive power drop to ~0 (complete disconnection)",
             [Injection("Bus 3", "active_power_mw", 0.2), Injection("Bus 3", "reactive_power_mvar", 0.0)]),
    Scenario("4. Cascading fault",
             "Bus 7 faults to 0.75 pu, then connected buses 4 and 9 sag 15 and 30 minutes later",
             [Injection("Bus 7", "voltage_pu", 0.75),
              Injection("Bus 4", "voltage_pu", 0.90, delay=1),
              Injection("Bus 9", "voltage_pu", 0.88, delay=2)]),
    Scenario("5. Frequency deviation", "System frequency rises to 60.8 Hz (+0.8 Hz) on all buses",
             [Injection(bus, "frequency_hz", NOMINAL_FREQUENCY_HZ + 0.8) for bus in BUSES]),
]


def time_slot(timestamps: pd.Series) -> pd.Series:
    return timestamps.dt.hour * 4 + timestamps.dt.minute // 15


def normal_profile(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-bus expected value for each 15-minute slot of the day, and each bus's noise level."""
    slot = time_slot(data.timestamp).rename("slot")
    mean = data.groupby(["bus", slot])[MEASUREMENTS].mean()
    noise_std = (data.set_index(["bus", slot])[MEASUREMENTS] - mean).groupby("bus").std()
    return mean, noise_std


def make_features(bus_data: pd.DataFrame, profile: pd.DataFrame, noise_std: pd.Series) -> pd.DataFrame:
    """Model input: each reading's deviation from that bus's normal value at that time of day,
    in units of that bus's sensor noise, plus the largest and RMS deviation across measurements.

    Removing the daily curve turns a fault into a clean outlier. Isolation Forest scores any
    reading beyond the training range about the same as the most extreme training reading along
    the same splits, so a fault in a single measurement would otherwise score no higher than normal
    readings that are mildly unusual in several measurements. The max/RMS features make a
    single-measurement fault extreme in three dimensions at once, which pushes its score clear of
    everything seen in training.
    """
    expected = profile.loc[time_slot(bus_data.timestamp)].to_numpy()
    z = (bus_data[MEASUREMENTS].to_numpy() - expected) / noise_std[MEASUREMENTS].to_numpy()
    features = pd.DataFrame(z, columns=FEATURES[:4])
    features["max_abs_z"] = np.abs(z).max(axis=1)
    features["rms_z"] = np.sqrt((z**2).mean(axis=1))
    return features


def features_for(bus_data: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    return make_features(bus_data, bundle["profile"], bundle["noise_std"])


def train_models(data: pd.DataFrame, mean: pd.DataFrame, noise_std: pd.DataFrame) -> dict[str, dict]:
    MODEL_DIR.mkdir(exist_ok=True)
    models = {}
    for bus in BUSES:
        profile, bus_noise = mean.loc[bus], noise_std.loc[bus]
        features = make_features(data[data.bus == bus], profile, bus_noise)
        model = IsolationForest(n_estimators=150, max_samples=512,
                                contamination=CONTAMINATION, random_state=SEED)
        model.fit(features)
        models[bus] = {"model": model, "profile": profile, "noise_std": bus_noise,
                       "features": FEATURES, "persistence": PERSISTENCE, "bus": bus}
        joblib.dump(models[bus], MODEL_DIR / f"bus_{bus.split()[1].zfill(2)}.joblib", compress=3)
    return models


def make_normal_day(mean: pd.DataFrame, noise_std: pd.DataFrame, day: pd.Timestamp,
                    rng: np.random.Generator) -> pd.DataFrame:
    timestamps = pd.date_range(day, periods=READINGS_PER_DAY, freq="15min")
    frames = []
    for bus in BUSES:
        values = mean.loc[bus].to_numpy() + rng.normal(0, noise_std.loc[bus].to_numpy(), (READINGS_PER_DAY, 4))
        frame = pd.DataFrame(values, columns=MEASUREMENTS)
        frames.append(frame.assign(timestamp=timestamps, bus=bus, reading=np.arange(READINGS_PER_DAY)))
    return pd.concat(frames, ignore_index=True)


def inject(day: pd.DataFrame, scenario: Scenario, rng: np.random.Generator) -> pd.DataFrame:
    day = day.copy()
    day["faulted"] = False
    fault_end = FAULT_START + FAULT_DURATION
    for inj in scenario.injections:
        mask = (day.bus == inj.bus) & day.reading.between(FAULT_START + inj.delay, fault_end - 1)
        noise_std = 0.002 if inj.column == "frequency_hz" else (0.005 if inj.column == "voltage_pu" else 0.1)
        day.loc[mask, inj.column] = inj.value + rng.normal(0, noise_std, mask.sum())
        day.loc[mask, "faulted"] = True
    return day


def severity(score: float) -> str:
    return next(label for bound, label in SEVERITY_BANDS if score >= bound)


def persistent(flags: np.ndarray, count: int) -> np.ndarray:
    """True where this reading and the previous count-1 readings were all flagged."""
    run = np.zeros(len(flags), dtype=int)
    for i, flagged in enumerate(flags):
        run[i] = run[i - 1] + 1 if flagged and i > 0 else int(flagged)
    return run >= count


def evaluate(scenario: Scenario, day: pd.DataFrame, models: dict[str, dict]) -> dict:
    window = day.reading.between(FAULT_START, FAULT_START + FAULT_DURATION - 1)
    rows, outside_alarms, outside_points = [], 0, 0
    onset = {inj.bus: FAULT_START + inj.delay for inj in scenario.injections}

    for bus in BUSES:
        bus_day = day[day.bus == bus]
        features = features_for(bus_day, models[bus])
        flags = models[bus]["model"].predict(features) == -1
        scores = -models[bus]["model"].score_samples(features)
        alarms = persistent(flags, PERSISTENCE)

        in_window = window[bus_day.index].to_numpy()
        outside_alarms += int(alarms[~in_window].sum())
        outside_points += int((~in_window).sum())

        start = onset.get(bus, FAULT_START)
        flagged_readings = bus_day.reading.to_numpy()[alarms & in_window]
        flagged_readings = flagged_readings[flagged_readings >= start]
        detected = len(flagged_readings) > 0
        peak = float(scores[in_window].max())
        rows.append({
            "Bus": bus,
            "Expected fault": bus in scenario.expected_buses,
            "Flagged": detected,
            "Points to detection": int(flagged_readings[0] - start + 1) if detected else None,
            "Peak anomaly score": round(peak, 3),
            "Severity": severity(peak) if detected else "-",
            "Max deviation (sigma)": round(float(features.max_abs_z[in_window].max()), 1),
        })

    return {"table": pd.DataFrame(rows), "outside_alarms": outside_alarms, "outside_points": outside_points}


def print_report(scenario: Scenario, result: dict) -> None:
    table = result["table"]
    title = f"Scenario {scenario.name}"
    print(f"\n{title}\n{'=' * len(title)}\n{scenario.description}")

    shown = table[table.Flagged | table["Expected fault"]].copy()
    shown["Result"] = np.select(
        [shown.Flagged & shown["Expected fault"], shown["Expected fault"]],
        ["DETECTED", "MISSED"], default="FALSE ALARM",
    )
    shown["Points to detection"] = shown["Points to detection"].map(lambda v: "-" if pd.isna(v) else int(v))
    columns = ["Bus", "Result", "Points to detection", "Peak anomaly score", "Severity", "Max deviation (sigma)"]
    print(shown[columns].to_string(index=False))

    tp = int((table.Flagged & table["Expected fault"]).sum())
    print(f"Detected {tp}/{int(table['Expected fault'].sum())} faulted buses; "
          f"{result['outside_alarms']} false alarms on {result['outside_points']} normal readings outside the fault window")


def main() -> None:
    data = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    print(f"Loaded {len(data):,} rows from {DATA_FILE} "
          f"({data.timestamp.min():%Y-%m-%d} to {data.timestamp.max():%Y-%m-%d})")

    mean, noise_std = normal_profile(data)
    models = train_models(data, mean, noise_std)
    print(f"Trained {len(models)} Isolation Forest models and saved them to {MODEL_DIR}/")

    rng = np.random.default_rng(SEED)
    test_start = data.timestamp.max().normalize() + pd.Timedelta(days=1)

    correct = total = outside_alarms = outside_points = 0
    fully_detected = 0
    for i, scenario in enumerate(SCENARIOS):
        day = inject(make_normal_day(mean, noise_std, test_start + pd.Timedelta(days=i), rng), scenario, rng)
        result = evaluate(scenario, day, models)
        print_report(scenario, result)

        table = result["table"]
        correct += int((table.Flagged == table["Expected fault"]).sum())
        total += len(table)
        outside_alarms += result["outside_alarms"]
        outside_points += result["outside_points"]
        fully_detected += int(table[table["Expected fault"]].Flagged.all())

    print("\nOverall results\n===============")
    print(f"Detection accuracy: {100 * correct / total:.1f}% "
          f"({correct}/{total} bus-scenario decisions correct: faulted buses flagged, healthy buses not flagged)")
    print(f"Scenarios with every faulted bus detected: {fully_detected}/{len(SCENARIOS)}")
    print(f"False alarm rate on normal readings: {100 * outside_alarms / outside_points:.2f}% "
          f"({outside_alarms}/{outside_points})")
    print(f"Models saved in {MODEL_DIR.resolve()}")


if __name__ == "__main__":
    main()
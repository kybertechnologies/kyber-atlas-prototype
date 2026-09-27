"""Atlas by Kyber Technologies: Grid Topology AI operations dashboard for the IEEE 14-bus network.

Run: streamlit run atlas_app.py
Needs topology_map.py and generate_telemetry.py in the same folder.
"""

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from generate_telemetry import daily_shape
from topology_map import BRANCHES, build_graph, build_network, voltage_status

BUS_NUMBERS = range(1, 15)


def healthy_bus(bus: int) -> dict:
    return {"bus": bus, "faulted": False, "voltage_pu": None, "fault": None, "severity": None}


# Session state survives every rerun (each button press reruns this script from the top),
# but is cleared when the browser tab is refreshed or closed.
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "health_score" not in st.session_state:
    st.session_state.health_score = 100
if "bus_status" not in st.session_state:
    st.session_state.bus_status = {bus: healthy_bus(bus) for bus in BUS_NUMBERS}
if "recommended_response" not in st.session_state:
    st.session_state.recommended_response = None
if "faulted_branches" not in st.session_state:
    st.session_state.faulted_branches = []
if "tripped_branches" not in st.session_state:
    st.session_state.tripped_branches = []

NAVY = "#0A0E1A"
PANEL = "#111A2E"
BORDER = "#1E2A44"
BLUE = "#00D4FF"
GREEN = "#22C55E"
YELLOW = "#FACC15"
RED = "#EF4444"
MUTED = "#8B97B5"
STATUS_COLORS = {"Normal": GREEN, "Warning": YELLOW, "Critical": RED}

# Generator voltage setpoints for a healthy operating point: every bus sits inside 0.95-1.05 pu.
# (The textbook case14 setpoints of up to 1.09 pu would put nine buses in the warning band.)
HEALTHY_SETPOINTS = {1: 1.04, 2: 1.03, 3: 1.01, 6: 1.03, 8: 1.04}
GENERATOR_BUSES = set(HEALTHY_SETPOINTS)
HEALTH_PENALTY = {"LOW": 2, "MEDIUM": 6, "HIGH": 12, "CRITICAL": 20}

LINES = [(f, t) for f, t, *_, ratio, _ in BRANCHES if ratio == 0]
# Fault voltages are only forced on buses without generators: holding a generator bus (especially
# the slack bus, which sets the system voltage reference) low would drag the whole grid down.
# Generator buses near a fault still sag through the power flow.
FAULTABLE_LINES = [(f, t) for f, t in LINES if not {f, t} <= set(HEALTHY_SETPOINTS)]
TRANSFORMERS = [(f, t) for f, t, *_, ratio, _ in BRANCHES if ratio != 0]
LINE_REACTANCE_OHM = {(f, t): x * 135.0**2 / 100.0 for f, t, _, x, _, ratio, _ in BRANCHES if ratio == 0}
OHMS_PER_KM = 0.4

CREWS = {
    "line": [("Crew Alpha-3", "Line Maintenance", "J. Martinez"), ("Crew Foxtrot-1", "Line Maintenance", "A. Chen")],
    "substation": [("Crew Bravo-7", "Substation Services", "R. Okafor"), ("Crew Golf-4", "Substation Services", "S. Patel")],
    "protection": [("Crew Delta-2", "Protection & Control", "M. Novak"), ("Crew Hotel-6", "Protection & Control", "L. Duarte")],
    "emergency": [("Crew Echo-5", "Emergency Response", "K. Williams"), ("Crew India-9", "Emergency Response", "T. Nguyen")],
}


@dataclass
class Fault:
    kind: str
    severity: str
    title: str
    description: str
    buses: list[int]
    forced_voltages: dict[int, float]
    location: str
    steps: list[str]
    crew: dict
    faulted_branches: list[str] = field(default_factory=list)
    tripped_branches: list[str] = field(default_factory=list)
    time: datetime = field(default_factory=datetime.now)


# ---------------------------------------------------------------- grid model

def base_graph() -> nx.Graph:
    return solve_grid((), ())


@st.cache_data(show_spinner=False)
def solve_grid(forced: tuple[tuple[int, float], ...], tripped: tuple[str, ...]) -> nx.Graph:
    """AC power flow with fault voltages held at their faulted buses and failed branches removed."""
    n = build_network()
    for bus, v in HEALTHY_SETPOINTS.items():
        n.buses.loc[f"Bus {bus}", "v_mag_pu_set"] = v
    for bus, v in forced:
        if not (n.generators.bus == f"Bus {bus}").any():
            n.add("Generator", f"Fault {bus}", bus=f"Bus {bus}", control="PV", p_set=0.0)
        n.buses.loc[f"Bus {bus}", "v_mag_pu_set"] = v
    for name in tripped:
        n.remove("Transformer" if name.startswith("Trafo") else "Line", name)

    result = n.pf()
    if not bool(np.all(result.converged)):
        graph = solve_grid((), ()) if forced or tripped else None
        if graph is None:
            raise RuntimeError("Base power flow did not converge")
        graph = graph.copy()
        for bus, v in forced:
            graph.nodes[f"Bus {bus}"]["v_mag"] = v
        return graph

    graph = build_graph(n)
    for name in tripped:
        f, t = (int(b) for b in name.split()[1].split("-"))
        kind = "Transformer" if name.startswith("Trafo") else "Line"
        graph.add_edge(f"Bus {f}", f"Bus {t}", name=name, kind=kind, flow_mva=0.0,
                       rating_mva=0.0, loading_pct=0.0)
    return graph


def nearest_generator(graph: nx.Graph, bus: int) -> int:
    lengths = nx.single_source_shortest_path_length(graph, f"Bus {bus}")
    return min(GENERATOR_BUSES - {bus}, key=lambda g: lengths.get(f"Bus {g}", 99))


def neighbours(graph: nx.Graph, bus: int) -> list[int]:
    return sorted(int(b.split()[1]) for b in graph.neighbors(f"Bus {bus}"))


def dispatch(team: str, rng: random.Random) -> dict:
    name, specialty, lead = rng.choice(CREWS[team])
    distance = round(rng.uniform(4, 28), 1)
    eta_min = int(round(10 + distance / 55 * 60))
    return {"name": name, "specialty": specialty, "lead": lead, "distance_km": distance,
            "eta_min": eta_min, "eta_clock": (datetime.now() + timedelta(minutes=eta_min)).strftime("%H:%M")}


# ---------------------------------------------------------------- fault scenarios

def line_fault(rng: random.Random) -> Fault:
    a, b = rng.choice(FAULTABLE_LINES)
    candidates = [bus for bus in (a, b) if bus not in GENERATOR_BUSES]
    near = rng.choice(candidates)
    far = b if near == a else a
    length_km = LINE_REACTANCE_OHM[(a, b)] / OHMS_PER_KM
    pct = rng.randint(10, 45)
    forced = {near: round(rng.uniform(0.76, 0.82), 3)}
    if far not in GENERATOR_BUSES:
        forced[far] = round(rng.uniform(0.84, 0.88), 3)
    graph = base_graph()
    alternates = [f"Bus {near}-{n}" for n in neighbours(graph, near) if n != far][:2]
    return Fault(
        kind="Line Fault", severity="HIGH",
        title=f"Line Fault — Line {a}-{b}",
        description=(f"Phase-to-ground fault on Line {a}-{b}. Voltage collapsed to "
                     + " and ".join(f"{v:.2f} pu at Bus {bus}" for bus, v in forced.items()) + "."),
        buses=list(forced), forced_voltages=forced,
        location=(f"Line {a}-{b}, approx. {length_km * pct / 100:.1f} km from the Bus {near} substation "
                  f"({pct}% of the {length_km:.0f} km line), based on impedance-to-fault ratio."),
        steps=[
            f"Open breakers CB-{a}{b}-A at Bus {a} and CB-{a}{b}-B at Bus {b} to isolate Line {a}-{b}.",
            f"Confirm zero voltage on Line {a}-{b} and apply safety grounds at both line terminals.",
            f"Reroute power via {' and '.join(alternates) if alternates else 'parallel paths'} and hold "
            f"their loading below 90% until the line is repaired.",
        ],
        crew=dispatch("line", rng), faulted_branches=[f"Line {a}-{b}"],
    )


def transformer_failure(rng: random.Random) -> Fault:
    hv, lv = rng.choice(TRANSFORMERS)
    v_lv = round(rng.uniform(0.78, 0.84), 3)
    graph = base_graph()
    tie = next((n for n in neighbours(graph, lv) if n != hv), None)
    return Fault(
        kind="Transformer Failure", severity="CRITICAL",
        title=f"Transformer Failure — T{hv}-{lv}",
        description=(f"Transformer T{hv}-{lv} tripped on internal fault (differential protection). "
                     f"LV side Bus {lv} dropped to {v_lv:.2f} pu."),
        buses=[lv], forced_voltages={lv: v_lv},
        location=f"Transformer T{hv}-{lv} at the Bus {hv} substation (HV Bus {hv} → LV Bus {lv}).",
        steps=[
            f"Confirm HV breaker at Bus {hv} and LV breaker at Bus {lv} are open; lock out T{hv}-{lv}.",
            f"Transfer Bus {lv} load to the Bus {tie} tie via its interconnecting branch and close the tie switch."
            if tie else f"Transfer Bus {lv} load to an alternate supply and close the tie switch.",
            f"Run dissolved-gas and Buchholz relay checks on T{hv}-{lv} before any attempt to re-energise.",
        ],
        crew=dispatch("substation", rng), tripped_branches=[f"Trafo {hv}-{lv}"],
    )


def voltage_sag(rng: random.Random) -> Fault:
    bus = rng.choice([4, 5, 9, 10, 11, 12, 13, 14])
    v = round(rng.uniform(0.84, 0.88), 3)
    graph = base_graph()
    gen = nearest_generator(graph, bus)
    return Fault(
        kind="Voltage Sag", severity="MEDIUM",
        title=f"Voltage Sag — Bus {bus}",
        description=f"Sustained voltage sag at Bus {bus}: {v:.2f} pu (normal band 0.95-1.05 pu).",
        buses=[bus], forced_voltages={bus: v},
        location=f"Distribution feeder downstream of Bus {bus}, estimated {rng.uniform(1.5, 9):.1f} km from the substation.",
        steps=[
            f"Raise the AVR setpoint at the Bus {gen} generator and switch in the Bus {bus} capacitor bank.",
            f"Open the sectionaliser on the affected feeder downstream of Bus {bus} to isolate the sag source.",
            f"If voltage stays below 0.95 pu, shed non-critical load at Bus {bus} in 5 MW steps.",
        ],
        crew=dispatch("protection", rng),
    )


def cascading_fault(rng: random.Random) -> Fault:
    origin = rng.choice([4, 5, 7, 9, 13])
    graph = base_graph()
    secondary = [b for b in neighbours(graph, origin) if b not in GENERATOR_BUSES]
    forced = {origin: round(rng.uniform(0.72, 0.78), 3)}
    forced |= {b: round(rng.uniform(0.84, 0.89), 3) for b in secondary}
    gen = nearest_generator(graph, origin)
    return Fault(
        kind="Cascading Fault", severity="CRITICAL",
        title=f"Cascading Fault — Bus {origin} → {', '.join(f'Bus {b}' for b in secondary)}",
        description=(f"Primary fault at Bus {origin} ({forced[origin]:.2f} pu) propagated to "
                     f"{len(secondary)} connected buses: "
                     + ", ".join(f"Bus {b} {forced[b]:.2f} pu" for b in secondary) + "."),
        buses=[origin, *secondary], forced_voltages=forced,
        location=f"Origin at the Bus {origin} substation; secondary sags at connected Buses {', '.join(map(str, secondary))}.",
        steps=[
            f"Isolate Bus {origin} immediately: open every breaker connecting it to Buses {', '.join(map(str, secondary))}.",
            f"Arm under-voltage load shedding at Buses {', '.join(map(str, secondary))} to stop further propagation.",
            f"Re-dispatch the Bus {gen} generator for voltage support, then restore Bus {origin} in stages once all buses exceed 0.95 pu.",
        ],
        crew=dispatch("emergency", rng),
    )


FAULT_BUTTONS = {
    "Trigger Line Fault": line_fault,
    "Trigger Transformer Failure": transformer_failure,
    "Trigger Voltage Sag": voltage_sag,
    "Trigger Cascading Fault": cascading_fault,
}


def trigger(builder) -> None:
    fault = builder(random.Random())
    state = st.session_state

    state.alerts.append({
        "time": fault.time, "severity": fault.severity, "buses": fault.buses,
        "title": fault.title, "description": fault.description,
    })
    state.health_score = max(0, state.health_score - HEALTH_PENALTY[fault.severity])
    for bus, voltage in fault.forced_voltages.items():
        state.bus_status[bus] = {"bus": bus, "faulted": True, "voltage_pu": voltage,
                                 "fault": fault.kind, "severity": fault.severity}
    state.faulted_branches.extend(b for b in fault.faulted_branches if b not in state.faulted_branches)
    state.tripped_branches.extend(b for b in fault.tripped_branches if b not in state.tripped_branches)
    state.recommended_response = {
        "kind": fault.kind, "severity": fault.severity, "location": fault.location,
        "steps": fault.steps, "crew": fault.crew,
    }


def reset_grid() -> None:
    state = st.session_state
    state.alerts = []
    state.health_score = 100
    state.bus_status = {bus: healthy_bus(bus) for bus in BUS_NUMBERS}
    state.recommended_response = None
    state.faulted_branches = []
    state.tripped_branches = []


# ---------------------------------------------------------------- rendering

CSS = f"""
<style>
  .stApp {{ background: {NAVY}; }}
  header[data-testid="stHeader"] {{ background: transparent; }}
  .block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; max-width: 100%; }}
  section[data-testid="stSidebar"] {{ background: #0D1426; border-right: 1px solid {BORDER}; }}
  .atlas-header {{ display: flex; align-items: center; justify-content: space-between; gap: 24px;
      flex-wrap: wrap; background: linear-gradient(90deg, #0F1830, #0B1122); border: 1px solid {BORDER};
      border-radius: 14px; padding: 14px 22px; margin-bottom: 18px; }}
  .brand {{ font-size: 1.05rem; font-weight: 600; letter-spacing: .04em; color: {BLUE}; white-space: nowrap; }}
  .brand b {{ color: #fff; font-size: 1.6rem; letter-spacing: .18em; margin-left: 10px; font-weight: 800; }}
  .status {{ display: flex; align-items: center; gap: 10px; color: {GREEN}; font-weight: 700;
      letter-spacing: .14em; white-space: nowrap; }}
  .pulse {{ width: 14px; height: 14px; border-radius: 50%; background: {GREEN};
      box-shadow: 0 0 0 0 rgba(34,197,94,.7); animation: pulse 1.8s infinite; }}
  @keyframes pulse {{ 0% {{ box-shadow: 0 0 0 0 rgba(34,197,94,.7); }}
      70% {{ box-shadow: 0 0 0 12px rgba(34,197,94,0); }} 100% {{ box-shadow: 0 0 0 0 rgba(34,197,94,0); }} }}
  .metrics {{ display: flex; gap: 12px; flex-wrap: wrap; }}
  .metric {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px; padding: 8px 16px; min-width: 150px; }}
  .metric .label {{ color: {MUTED}; font-size: .7rem; text-transform: uppercase; letter-spacing: .1em; }}
  .metric .value {{ color: #fff; font-size: 1.6rem; font-weight: 700; line-height: 1.2; }}
  .panel-title {{ color: {BLUE}; font-weight: 700; letter-spacing: .16em; font-size: .85rem; margin: 0 0 8px 2px; }}
  .alert-feed {{ height: 540px; overflow-y: auto; background: {PANEL}; border: 1px solid {BORDER};
      border-radius: 12px; padding: 10px; }}
  .alert {{ background: #0C1427; border: 1px solid {BORDER}; border-left: 4px solid var(--sev); border-radius: 8px;
      padding: 10px 12px; margin-bottom: 10px; }}
  .alert .row {{ display: flex; justify-content: space-between; align-items: center; gap: 8px; }}
  .alert .time {{ color: {MUTED}; font-family: ui-monospace, monospace; font-size: .8rem; }}
  .alert .bus {{ color: {BLUE}; font-weight: 700; font-size: .8rem; }}
  .alert .title {{ color: #fff; font-weight: 600; margin: 4px 0 2px; }}
  .alert .desc {{ color: #C9D2E8; font-size: .85rem; }}
  .badge {{ font-size: .68rem; font-weight: 800; letter-spacing: .08em; padding: 3px 8px; border-radius: 999px;
      color: #0A0E1A; background: var(--sev); }}
  .empty {{ color: {MUTED}; text-align: center; padding: 60px 20px; }}
  .empty b {{ color: {GREEN}; display: block; font-size: 1.05rem; margin-bottom: 6px; }}
  .sidebar-title {{ color: {BLUE}; font-weight: 800; letter-spacing: .16em; font-size: .95rem; margin: 4px 0 10px; }}
  section[data-testid="stSidebar"] .stButton button {{ border: 1px solid {BLUE}; color: #fff; background: #0F1A33;
      font-weight: 600; }}
  section[data-testid="stSidebar"] .stButton button:hover {{ background: {BLUE}; color: {NAVY}; }}
  .response {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px; padding: 14px; margin-top: 6px; }}
  .response h4 {{ color: {MUTED}; font-size: .7rem; letter-spacing: .12em; text-transform: uppercase; margin: 10px 0 4px; }}
  .response p, .response li {{ color: #fff; font-size: .85rem; }}
  .response ol {{ padding-left: 18px; margin: 0; }}
  .crew {{ background: #0C1427; border: 1px solid {BLUE}; border-radius: 10px; padding: 10px 12px; margin-top: 6px; }}
  .crew .name {{ color: {BLUE}; font-weight: 700; }}
  .crew .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 6px; }}
  .crew .k {{ color: {MUTED}; font-size: .68rem; text-transform: uppercase; letter-spacing: .08em; }}
  .crew .v {{ color: #fff; font-weight: 700; }}
</style>
"""


def severity_color(severity: str) -> str:
    return {"LOW": GREEN, "MEDIUM": YELLOW}.get(severity, RED)


def render_header() -> None:
    health = st.session_state.health_score
    alerts = st.session_state.alerts
    health_color = GREEN if health >= 80 else YELLOW if health >= 50 else RED
    alert_color = RED if alerts else "#fff"
    st.markdown(f"""
    <div class="atlas-header">
      <div class="brand">Kyber Technologies<b>ATLAS</b></div>
      <div class="status"><span class="pulse"></span>GRID ONLINE</div>
      <div class="metrics">
        <div class="metric"><div class="label">Total Nodes Monitored</div><div class="value">14</div></div>
        <div class="metric"><div class="label">Active Alerts</div>
          <div class="value" style="color:{alert_color}">{len(alerts)}</div></div>
        <div class="metric"><div class="label">Grid Health Score</div>
          <div class="value" style="color:{health_color}">{health}%</div></div>
      </div>
    </div>""", unsafe_allow_html=True)


def topology_figure(graph: nx.Graph) -> go.Figure:
    bus_status = st.session_state.bus_status
    faulted_branches = set(st.session_state.faulted_branches)
    tripped = set(st.session_state.tripped_branches)
    fault_buses = [f"Bus {bus}" for bus, s in bus_status.items() if s["faulted"]]
    fig = go.Figure()

    styles = {
        "normal": dict(color="#4A5B82", width=2, dash="solid"),
        "transformer": dict(color="#4A5B82", width=2, dash="dash"),
        "faulted": dict(color=RED, width=4, dash="solid"),
        "tripped": dict(color=RED, width=3, dash="dot"),
    }
    groups = {k: ([], []) for k in styles}
    mid_x, mid_y, hover = [], [], []
    for u, v, d in graph.edges(data=True):
        status = ("tripped" if d["name"] in tripped else "faulted" if d["name"] in faulted_branches
                  else "transformer" if d["kind"] == "Transformer" else "normal")
        (x0, y0), (x1, y1) = graph.nodes[u]["pos"], graph.nodes[v]["pos"]
        groups[status][0].extend([x0, x1, None])
        groups[status][1].extend([y0, y1, None])
        mid_x.append((x0 + x1) / 2)
        mid_y.append((y0 + y1) / 2)
        state = {"tripped": "TRIPPED — out of service", "faulted": "FAULTED"}.get(status, "In service")
        hover.append(f"<b>{d['name']}</b><br>Connects: {u} ↔ {v}<br>Status: {state}<br>"
                     f"Loading: {d['loading_pct']:.1f}% of capacity")
    for status, (xs, ys) in groups.items():
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=styles[status], hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=mid_x, y=mid_y, mode="markers", text=hover, hoverinfo="text", showlegend=False,
                             marker=dict(size=16, color="rgba(0,0,0,0)")))

    if fault_buses:
        fig.add_trace(go.Scatter(
            x=[graph.nodes[b]["pos"][0] for b in fault_buses], y=[graph.nodes[b]["pos"][1] for b in fault_buses],
            mode="markers", hoverinfo="skip", showlegend=False,
            marker=dict(size=52, color="rgba(239,68,68,0.18)", line=dict(color=RED, width=2)),
        ))

    nodes = list(graph.nodes(data=True))
    loads = np.array([d["load_mw"] for _, d in nodes])
    faults = [bus_status[int(n.split()[1])] for n, _ in nodes]
    statuses = ["Critical" if f["faulted"] else voltage_status(d["v_mag"])[0] for f, (_, d) in zip(faults, nodes)]
    fig.add_trace(go.Scatter(
        x=[d["pos"][0] for _, d in nodes], y=[d["pos"][1] for _, d in nodes],
        mode="markers+text", text=[n.replace("Bus ", "") for n, _ in nodes], textposition="middle center",
        textfont=dict(color=NAVY, size=11, family="Arial Black"),
        hovertext=[f"<b>{n}</b><br>Voltage: {d['v_mag']:.4f} pu ({s})<br>Load: {d['load_mw']:.1f} MW"
                   + (f"<br>FAULTED: {f['fault']} ({f['severity']})" if f["faulted"] else "")
                   for (n, d), s, f in zip(nodes, statuses, faults)],
        hoverinfo="text", showlegend=False,
        marker=dict(size=18 + 30 * loads / loads.max(), color=[STATUS_COLORS[s] for s in statuses],
                    line=dict(color="#fff", width=1.5), opacity=1),
    ))
    for label, color in [("Normal 0.95–1.05 pu", GREEN), ("Warning 0.90–0.95 / 1.05–1.10 pu", YELLOW),
                         ("Critical <0.90 / >1.10 pu", RED)]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=label, marker=dict(size=11, color=color)))

    fig.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, height=540, margin=dict(l=10, r=10, t=10, b=10),
        font=dict(color="#fff"), hoverlabel=dict(bgcolor=NAVY, font_color="#fff", bordercolor=BLUE),
        legend=dict(orientation="h", x=0, y=0, bgcolor="rgba(10,14,26,0.7)", font=dict(size=11)),
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
    )
    return fig


def render_alerts() -> None:
    alerts = st.session_state.alerts
    st.markdown('<div class="panel-title">LIVE ALERTS</div>', unsafe_allow_html=True)
    if not alerts:
        body = ('<div class="empty"><b>All systems nominal</b>No active alerts. All 14 buses are within '
                'normal voltage limits.<br>Use the Fault Simulator in the sidebar to inject a fault.</div>')
    else:
        body = "".join(
            f'<div class="alert" style="--sev:{severity_color(a["severity"])}">'
            f'<div class="row"><span class="time">{a["time"]:%Y-%m-%d %H:%M:%S}</span>'
            f'<span class="badge">{a["severity"]}</span></div>'
            f'<div class="row"><span class="bus">{"BUS " + ", ".join(map(str, a["buses"]))}</span></div>'
            f'<div class="title">{a["title"]}</div><div class="desc">{a["description"]}</div></div>'
            for a in reversed(alerts)
        )
    st.markdown(f'<div class="alert-feed">{body}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def trend_baseline(end: datetime) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray]]:
    times = pd.date_range(end=end, periods=96, freq="15min")
    shape = daily_shape(times.hour + times.minute / 60)
    now_shape = shape[-1]
    rng = np.random.default_rng(7)
    graph = solve_grid((), ())
    series = {bus: graph.nodes[bus]["v_mag"] + 0.02 * (shape - now_shape) + rng.normal(0, 0.002, len(times))
              for bus in graph.nodes}
    return times, series


def trend_figure(graph: nx.Graph) -> go.Figure:
    end = pd.Timestamp.now().floor("15min").to_pydatetime()
    times, series = trend_baseline(end)
    palette = px.colors.qualitative.Light24
    fig = go.Figure()
    for i, (bus, values) in enumerate(series.items()):
        values = values.copy()
        values[-1] = graph.nodes[bus]["v_mag"]
        fig.add_trace(go.Scatter(x=times, y=values, mode="lines", name=bus,
                                 line=dict(color=palette[i % len(palette)], width=2),
                                 hovertemplate=f"{bus}<br>%{{x|%H:%M}}<br>%{{y:.4f}} pu<extra></extra>"))
    for y, color in [(1.05, YELLOW), (0.95, YELLOW), (0.90, RED)]:
        fig.add_hline(y=y, line=dict(color=color, width=1, dash="dot"), opacity=0.6)
    fig.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, height=440, margin=dict(l=10, r=10, t=40, b=10),
        font=dict(color="#fff"), hovermode="x unified",
        legend=dict(orientation="v", x=1.01, y=1, font=dict(size=11)),
        xaxis=dict(title="Time (last 24 hours)", gridcolor=BORDER, showline=False),
        yaxis=dict(title="Voltage magnitude (pu)", gridcolor=BORDER, zeroline=False),
    )
    return fig


def render_sidebar() -> None:
    response = st.session_state.recommended_response
    with st.sidebar:
        st.markdown('<div class="sidebar-title">FAULT SIMULATOR</div>', unsafe_allow_html=True)
        for label, builder in FAULT_BUTTONS.items():
            st.button(label, on_click=trigger, args=(builder,), width="stretch")
        st.button("Reset Grid", on_click=reset_grid, type="tertiary", width="stretch",
                  disabled=not st.session_state.alerts)

        if response is not None:
            c = response["crew"]
            steps = "".join(f"<li>{s}</li>" for s in response["steps"])
            st.markdown(f"""
            <div class="sidebar-title" style="margin-top:18px">RECOMMENDED RESPONSE</div>
            <div class="response">
              <span class="badge" style="--sev:{severity_color(response['severity'])}">{response['severity']}</span>
              <span style="color:#fff;font-weight:700;margin-left:6px">{response['kind']}</span>
              <h4>Estimated fault location</h4><p>{response['location']}</p>
              <h4>Isolation &amp; switching steps</h4><ol>{steps}</ol>
              <h4>Crew dispatch</h4>
              <div class="crew">
                <div class="name">{c['name']}</div>
                <div style="color:{MUTED};font-size:.8rem">{c['specialty']} · Lead: {c['lead']}</div>
                <div class="grid">
                  <div><div class="k">Distance</div><div class="v">{c['distance_km']} km</div></div>
                  <div><div class="k">ETA</div><div class="v">{c['eta_min']} min ({c['eta_clock']})</div></div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Atlas | Kyber Technologies", page_icon="⚡", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    forced = tuple(sorted((bus, s["voltage_pu"]) for bus, s in st.session_state.bus_status.items() if s["faulted"]))
    tripped = tuple(sorted(st.session_state.tripped_branches))
    with st.spinner("Solving AC power flow..."):
        graph = solve_grid(forced, tripped)

    render_sidebar()
    render_header()

    left, right = st.columns([3, 2], gap="medium")
    with left:
        st.markdown('<div class="panel-title">GRID TOPOLOGY — IEEE 14-BUS</div>', unsafe_allow_html=True)
        st.plotly_chart(topology_figure(graph), width="stretch", config={"displaylogo": False})
    with right:
        render_alerts()

    st.markdown('<div class="panel-title" style="margin-top:12px">VOLTAGE MAGNITUDE TRENDS — LAST 24 HOURS</div>',
                unsafe_allow_html=True)
    st.plotly_chart(trend_figure(graph), width="stretch", config={"displaylogo": False})


main()
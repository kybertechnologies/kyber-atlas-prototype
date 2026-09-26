"""IEEE 14-bus AC power flow rendered as an interactive Plotly topology map.

Run: python topology_map.py   ->   writes topology_map.html (normal operation)
                                     and topology_map_fault.html (fault at bus 9)
"""

import logging
import warnings

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import pypsa

logging.getLogger("pypsa").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning, module="pypsa")

BASE_MVA = 100.0
V_NOM = 135.0
OUTPUT_FILE = "topology_map.html"
FAULT_OUTPUT_FILE = "topology_map_fault.html"
FAULT_VOLTAGE_PU = 0.75

# MATPOWER case14 branches: (from, to, r pu, x pu, b pu, tap ratio, rating MVA).
# case14 ships without thermal ratings, so the ratings here are assumed values:
# 200 MVA for the HV lines between buses 1-5, 65 MVA for transformers and the
# bus 7 branches, 32 MVA for the lower-voltage lines.
BRANCHES = [
    (1, 2, 0.01938, 0.05917, 0.0528, 0, 200),
    (1, 5, 0.05403, 0.22304, 0.0492, 0, 200),
    (2, 3, 0.04699, 0.19797, 0.0438, 0, 200),
    (2, 4, 0.05811, 0.17632, 0.034, 0, 200),
    (2, 5, 0.05695, 0.17388, 0.0346, 0, 200),
    (3, 4, 0.06701, 0.17103, 0.0128, 0, 200),
    (4, 5, 0.01335, 0.04211, 0.0, 0, 200),
    (4, 7, 0.0, 0.20912, 0.0, 0.978, 65),
    (4, 9, 0.0, 0.55618, 0.0, 0.969, 65),
    (5, 6, 0.0, 0.25202, 0.0, 0.932, 65),
    (6, 11, 0.09498, 0.1989, 0.0, 0, 32),
    (6, 12, 0.12291, 0.25581, 0.0, 0, 32),
    (6, 13, 0.06615, 0.13027, 0.0, 0, 32),
    (7, 8, 0.0, 0.17615, 0.0, 0, 65),
    (7, 9, 0.0, 0.11001, 0.0, 0, 65),
    (9, 10, 0.03181, 0.0845, 0.0, 0, 32),
    (9, 14, 0.12711, 0.27038, 0.0, 0, 32),
    (10, 11, 0.08205, 0.19207, 0.0, 0, 32),
    (12, 13, 0.22092, 0.19988, 0.0, 0, 32),
    (13, 14, 0.17093, 0.34802, 0.0, 0, 32),
]

# (bus, control, P MW, voltage setpoint pu). Buses 3, 6 and 8 are synchronous condensers.
GENERATORS = [
    (1, "Slack", 232.4, 1.06),
    (2, "PV", 40.0, 1.045),
    (3, "PV", 0.0, 1.01),
    (6, "PV", 0.0, 1.07),
    (8, "PV", 0.0, 1.09),
]

# (bus, P MW, Q MVAr)
LOADS = [
    (2, 21.7, 12.7), (3, 94.2, 19.0), (4, 47.8, -3.9), (5, 7.6, 1.6),
    (6, 11.2, 7.5), (9, 29.5, 16.6), (10, 9.0, 5.8), (11, 3.5, 1.8),
    (12, 6.1, 1.6), (13, 13.5, 5.8), (14, 14.9, 5.0),
]

# Positions loosely following the standard IEEE 14-bus one-line diagram.
LAYOUT = {
    1: (0.0, 3.0), 2: (1.0, 0.0), 3: (4.2, 0.0), 4: (4.0, 2.0), 5: (1.4, 2.4),
    6: (1.4, 4.4), 7: (5.4, 3.0), 8: (6.8, 3.0), 9: (4.6, 4.2), 10: (3.5, 4.9),
    11: (2.4, 5.2), 12: (0.3, 6.2), 13: (1.9, 6.6), 14: (4.4, 6.4),
}


def build_network() -> pypsa.Network:
    z_base = V_NOM**2 / BASE_MVA
    n = pypsa.Network(name="IEEE 14-bus")
    for i in range(1, 15):
        n.add("Bus", f"Bus {i}", v_nom=V_NOM)

    for f, t, r, x, b, ratio, rating in BRANCHES:
        if ratio == 0:
            n.add("Line", f"Line {f}-{t}", bus0=f"Bus {f}", bus1=f"Bus {t}",
                  r=r * z_base, x=x * z_base, b=b / z_base, s_nom=rating)
        else:
            # Transformer impedances are per unit on their own rating.
            scale = rating / BASE_MVA
            n.add("Transformer", f"Trafo {f}-{t}", bus0=f"Bus {f}", bus1=f"Bus {t}",
                  r=r * scale, x=x * scale, b=b / scale, s_nom=rating, tap_ratio=ratio)

    for bus, control, p, v in GENERATORS:
        n.add("Generator", f"Gen {bus}", bus=f"Bus {bus}", control=control, p_set=p)
        n.buses.loc[f"Bus {bus}", "v_mag_pu_set"] = v

    for bus, p, q in LOADS:
        n.add("Load", f"Load {bus}", bus=f"Bus {bus}", p_set=p, q_set=q)

    n.add("ShuntImpedance", "Shunt 9", bus="Bus 9", b=19.0 / V_NOM**2)
    return n


def voltage_status(v: float) -> tuple[str, str]:
    if 0.95 <= v <= 1.05:
        return "Normal", "#2ecc71"
    if 0.90 <= v <= 1.10:
        return "Warning", "#f1c40f"
    return "Critical", "#e74c3c"


def build_graph(n: pypsa.Network) -> nx.Graph:
    snapshot = n.snapshots[0]
    v_mag = n.buses_t.v_mag_pu.loc[snapshot]
    load_mw = n.loads.groupby("bus").p_set.sum().reindex(n.buses.index, fill_value=0.0)

    g = nx.Graph()
    for bus in n.buses.index:
        g.add_node(bus, v_mag=float(v_mag[bus]), load_mw=float(load_mw[bus]),
                   pos=LAYOUT[int(bus.split()[1])])

    for component, flows in [("Line", n.lines_t), ("Transformer", n.transformers_t)]:
        static = n.lines if component == "Line" else n.transformers
        s0 = np.hypot(flows.p0.loc[snapshot], flows.q0.loc[snapshot])
        s1 = np.hypot(flows.p1.loc[snapshot], flows.q1.loc[snapshot])
        for name, row in static.iterrows():
            flow = max(s0[name], s1[name])
            g.add_edge(row.bus0, row.bus1, name=name, kind=component, flow_mva=float(flow),
                       rating_mva=float(row.s_nom), loading_pct=100 * flow / row.s_nom)
    return g


def build_figure(g: nx.Graph, faulted_bus: str | None = None) -> go.Figure:
    fig = go.Figure()

    for kind, dash in [("Line", "solid"), ("Transformer", "dash")]:
        xs, ys = [], []
        for u, v, data in g.edges(data=True):
            if data["kind"] == kind:
                (x0, y0), (x1, y1) = g.nodes[u]["pos"], g.nodes[v]["pos"]
                xs += [x0, x1, None]
                ys += [y0, y1, None]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", hoverinfo="skip", name=f"{kind}s",
            line=dict(color="#8a9bb0", width=2, dash=dash),
        ))

    # Plotly can't hover on line segments, so each edge gets an invisible marker at its midpoint.
    mid_x, mid_y, edge_text = [], [], []
    for u, v, data in g.edges(data=True):
        (x0, y0), (x1, y1) = g.nodes[u]["pos"], g.nodes[v]["pos"]
        mid_x.append((x0 + x1) / 2)
        mid_y.append((y0 + y1) / 2)
        edge_text.append(
            f"<b>{data['name']}</b> ({data['kind'].lower()})<br>"
            f"Connects: {u} ↔ {v}<br>"
            f"Loading: {data['loading_pct']:.1f}% of capacity<br>"
            f"Flow: {data['flow_mva']:.1f} / {data['rating_mva']:.0f} MVA"
        )
    fig.add_trace(go.Scatter(
        x=mid_x, y=mid_y, mode="markers", text=edge_text, hoverinfo="text",
        marker=dict(size=18, color="rgba(0,0,0,0)"), showlegend=False,
    ))

    nodes = list(g.nodes(data=True))
    loads = np.array([d["load_mw"] for _, d in nodes])
    statuses = [voltage_status(d["v_mag"]) for _, d in nodes]
    # Size is proportional to load; the minimum keeps zero-load buses visible and hoverable.
    sizes = 14 + 46 * loads / loads.max()
    fig.add_trace(go.Scatter(
        x=[d["pos"][0] for _, d in nodes],
        y=[d["pos"][1] for _, d in nodes],
        mode="markers+text",
        text=[name for name, _ in nodes],
        textposition="top center",
        textfont=dict(color="#e0e0e0", size=11),
        hovertext=[
            f"<b>{name}</b><br>Voltage: {d['v_mag']:.4f} pu ({status})<br>Load: {d['load_mw']:.1f} MW"
            for (name, d), (status, _) in zip(nodes, statuses)
        ],
        hoverinfo="text",
        marker=dict(size=sizes, color=[c for _, c in statuses], opacity=1,
                    line=dict(color="#e0e0e0", width=1.5)),
        showlegend=False,
    ))

    for label, color in [
        ("Normal (0.95–1.05 pu)", "#2ecc71"),
        ("Warning (0.90–0.95 or 1.05–1.10 pu)", "#f1c40f"),
        ("Critical (outside 0.90–1.10 pu)", "#e74c3c"),
    ]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=label,
                                 marker=dict(size=12, color=color)))

    if faulted_bus is not None:
        fx, fy = g.nodes[faulted_bus]["pos"]
        fault_v = g.nodes[faulted_bus]["v_mag"]
        fig.add_trace(go.Scatter(
            x=[fx, fx], y=[fy, fy], mode="markers", hoverinfo="skip", showlegend=False,
            # Open symbols take their stroke colour from marker.color.
            marker=dict(size=[95, 70], color=["rgba(255,40,40,0.45)", "#ff2828"],
                        symbol="circle-open", line=dict(width=[10, 5])),
        ))
        fig.add_trace(go.Scatter(
            x=[fx], y=[fy], mode="markers", name="Faulted bus",
            hovertext=[f"<b>FAULT — {faulted_bus}</b><br>Voltage: {fault_v:.4f} pu"], hoverinfo="text",
            marker=dict(size=26, color="#ff2828", symbol="x", line=dict(color="#fff", width=2)),
        ))
        bus_number = faulted_bus.split()[1]
        fig.add_annotation(
            x=fx, y=fy, ax=160, ay=-70,
            text=f"<b>FAULT DETECTED — BUS {bus_number} — VOLTAGE CRITICAL</b>",
            font=dict(color="#fff", size=15), bgcolor="#b00020", bordercolor="#ff2828",
            borderwidth=2, borderpad=8, showarrow=True, arrowhead=2, arrowsize=1.5,
            arrowwidth=3, arrowcolor="#ff2828",
        )

    counts = {s: sum(1 for status, _ in statuses if status == s) for s in ("Normal", "Warning", "Critical")}
    heading = "IEEE 14-Bus Grid Topology — AC Power Flow"
    if faulted_bus is not None:
        heading += f" — <span style='color:#ff4d4d'>FAULT SIMULATION: {faulted_bus}</span>"
    fig.update_layout(
        title=dict(
            text=(
                heading
                + f"<br><sup>{counts['Normal']} normal · {counts['Warning']} warning · "
                f"{counts['Critical']} critical · node size = load (MW)</sup>"
            ),
            x=0.02,
        ),
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#2b2b2b",
        font=dict(color="#e0e0e0", family="Arial"),
        hoverlabel=dict(bgcolor="#111", font_color="#fff", bordercolor="#555"),
        legend=dict(bgcolor="rgba(30,30,30,0.8)", bordercolor="#555", borderwidth=1,
                    orientation="h", yanchor="bottom", y=-0.12, x=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x"),
        margin=dict(l=20, r=20, t=80, b=60),
    )
    return fig


def solve_and_save(network: pypsa.Network, output_file: str, faulted_bus: str | None = None) -> nx.Graph:
    result = network.pf()
    if not bool(np.all(result.converged)):
        raise RuntimeError("AC power flow did not converge")

    graph = build_graph(network)
    print(f"Power flow converged. Graph: {graph.number_of_nodes()} buses (nodes), "
          f"{graph.number_of_edges()} branches (edges), connected: {nx.is_connected(graph)}")

    build_figure(graph, faulted_bus).write_html(
        output_file,
        default_height="100vh",
        post_script="document.body.style.margin='0'; document.body.style.background='#1e1e1e';",
    )
    return graph


def simulate_fault(bus_number: int, output_file: str = FAULT_OUTPUT_FILE) -> nx.Graph:
    """Hold a bus at FAULT_VOLTAGE_PU, re-run the AC power flow and save a faulted topology map.

    The fault is modelled as a zero-MW voltage-controlling source at the bus, so the power flow
    solves the rest of the network around the depressed voltage.
    """
    if not 1 <= bus_number <= len(LAYOUT):
        raise ValueError(f"bus_number must be between 1 and {len(LAYOUT)}, got {bus_number}")
    bus = f"Bus {bus_number}"

    network = build_network()
    if not (network.generators.bus == bus).any():
        network.add("Generator", f"Fault {bus_number}", bus=bus, control="PV", p_set=0.0)
    network.buses.loc[bus, "v_mag_pu_set"] = FAULT_VOLTAGE_PU

    print(f"\nSimulating fault at {bus}: voltage forced to {FAULT_VOLTAGE_PU} pu")
    graph = solve_and_save(network, output_file, faulted_bus=bus)

    print(f"{'Bus':<8}{'Voltage (pu)':>13}  Status")
    for name, data in graph.nodes(data=True):
        v = data["v_mag"]
        if voltage_status(v)[0] != "Normal":
            marker = "  <-- FAULT" if name == bus else ""
            print(f"{name:<8}{v:>13.4f}  {voltage_status(v)[0]}{marker}")
    overloaded = [d for *_, d in graph.edges(data=True) if d["loading_pct"] > 100]
    for d in overloaded:
        print(f"Overloaded: {d['name']} at {d['loading_pct']:.1f}% of capacity")
    print(f"FAULT DETECTED — BUS {bus_number} — VOLTAGE CRITICAL")
    print(f"Fault topology map saved to {output_file}")
    return graph


def main() -> None:
    solve_and_save(build_network(), OUTPUT_FILE)
    print(f"Topology map saved to {OUTPUT_FILE}")

    simulate_fault(9)


if __name__ == "__main__":
    main()
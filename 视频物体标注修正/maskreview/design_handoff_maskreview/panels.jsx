/* ============================================================
   Icons + chrome: TopBar, SetupPanel, KpiStrip, QueueList
   ============================================================ */
const MI = {
  mask: "M4 7l8-4 8 4-8 4-8-4zM4 12l8 4 8-4M4 17l8 4 8-4",
  video: "M4 6a2 2 0 012-2h8a2 2 0 012 2v12a2 2 0 01-2 2H6a2 2 0 01-2-2zM16 10l5-3v10l-5-3",
  cpu: "M6 6h12v12H6zM9 9h6v6H9M3 9h3M3 14h3M18 9h3M18 14h3M9 3v3M14 3v3M9 18v3M14 18v3",
  play: "M7 5l12 7-12 7V5z",
  pause: "M8 5v14M16 5v14",
  prevf: "M18 6v12L9 12l9-6zM6 5v14",
  nextf: "M6 6v12l9-6-9-6zM18 5v14",
  check: "M5 13l4 4L19 7",
  plus: "M12 5v14M5 12h14",
  minus: "M5 12h14",
  box: "M4 7l8-4 8 4v10l-8 4-8-4zM4 7l8 4 8-4M12 11v10",
  tightbox: "M4 8V5a1 1 0 011-1h3M16 4h3a1 1 0 011 1v3M20 16v3a1 1 0 01-1 1h-3M8 20H5a1 1 0 01-1-1v-3",
  download: "M12 4v11m0 0l-4-4m4 4l4-4M5 20h14",
  alert: "M12 9v4m0 4h.01M10.3 4.3L2.6 18a1.5 1.5 0 001.3 2.2h16.2a1.5 1.5 0 001.3-2.2L13.7 4.3a1.5 1.5 0 00-3.4 0z",
  sparkle: "M12 3l1.7 5L19 9.7l-5.3 1.7L12 17l-1.7-5.6L5 9.7 10.3 8 12 3z",
  sliders: "M4 6h10M18 6h2M4 12h2M10 12h10M4 18h7M15 18h5M14 4v4M6 10v4M11 16v4",
  compare: "M9 4v16M5 8l-2 4 2 4M19 4v16M15 8l2 4-2 4",
  upload: "M12 16V5m0 0l-4 4m4-4l4 4M5 19h14",
  image: "M4 5h16v14H4zM4 15l4-4 4 4 3-3 5 5M9 9a1.5 1.5 0 11-3 0 1.5 1.5 0 013 0z",
  target: "M12 12m-9 0a9 9 0 1018 0a9 9 0 10-18 0M12 12m-4 0a4 4 0 108 0a4 4 0 10-8 0M12 12m-0.5 0a.5.5 0 101 0a.5.5 0 10-1 0",
  eye: "M2 12s4-7 10-7 10 7 10 7-4 7-10 7-10-7-10-7z|M12 12m-3 0a3 3 0 106 0a3 3 0 10-6 0",
  braces: "M8 4c-2 0-2 2-2 4s0 3-2 3c2 0 2 1 2 3s0 4 2 4M16 4c2 0 2 2 2 4s0 3 2 3c-2 0-2 1-2 3s0 4-2 4",
  layers: "M12 3l9 5-9 5-9-5 9-5zM3 13l9 5 9-5",
  refresh: "M21 12a9 9 0 11-3-6.7M21 4v5h-5",
  x: "M6 6l12 12M18 6L6 18",
  arrowr: "M5 12h14M13 6l6 6-6 6",
  jump: "M13 5l7 7-7 7M20 12H4",
};
function Ic({ n, style }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
      strokeLinecap="round" strokeLinejoin="round" style={style}>
      {(MI[n] || "").split("|").map((d, i) => <path key={i} d={d} />)}
    </svg>
  );
}
window.Ic = Ic;

const riskColor = (r) => (r >= 0.8 ? "#ff4d5e" : r >= 0.65 ? "#ffb02e" : "#ffd24d");
const confColor = (c) => (c >= 0.55 ? { bg: "rgba(255,176,46,0.16)", fg: "#ffb02e" } : c >= 0.3 ? { bg: "rgba(255,122,69,0.16)", fg: "#ff7a45" } : { bg: "rgba(255,77,94,0.16)", fg: "#ff4d5e" });

function TopBar({ tab, setTab, queued, running, onExport }) {
  const TABS = [
    { id: "review", label: "Review", icon: "mask", badge: queued },
    { id: "reprop", label: "Re-propagate", icon: "compare" },
    { id: "calib", label: "Calibration", icon: "sliders" },
  ];
  return (
    <div className="topbar">
      <div className="brand">
        <div className="brand-mark"><Ic n="mask" /></div>
        <div className="brand-name">Mask<b>Review</b></div>
      </div>
      <div className="sep" />
      <div className="session-chip">
        <Ic n="video" style={{ width: 14, height: 14, color: "var(--tx-3)" }} />
        <span className="fn">{window.VIDEO.name}</span>
        <span className="meta">{window.VIDEO.w}×{window.VIDEO.h} · {window.VIDEO.fps}fps · {(window.VIDEO.frames / window.VIDEO.fps)}s</span>
      </div>
      <div className="spacer" />
      <div className="tabs">
        {TABS.map((t) => (
          <button key={t.id} className={"tab" + (tab === t.id ? " on" : "")} onClick={() => setTab(t.id)}>
            <Ic n={t.icon} style={{ width: 14, height: 14 }} />{t.label}
            {t.badge ? <span className="badge">{t.badge}</span> : null}
          </button>
        ))}
      </div>
      <div className="sep" />
      <div className="device">
        <span className={"dot " + (running ? "busy" : "live")} />
        <Ic n="cpu" style={{ width: 13, height: 13, color: "var(--tx-3)" }} />
        <span className="mono">{window.VIDEO.device.split(" · ")[1]}</span>
        <span style={{ color: "var(--tx-3)" }}>{running ? "running" : "idle"}</span>
      </div>
      <button className="btn" onClick={onExport}><Ic n="download" />Export</button>
    </div>
  );
}
window.TopBar = TopBar;

function KpiStrip({ kpi }) {
  return (
    <div className="kpi-strip">
      <div className="kpi">
        <div className="kpi-label">Total frames</div>
        <div className="kpi-val">{kpi.total_frames.toLocaleString()}</div>
        <div className="kpi-sub">propagated in {kpi.pass_seconds}s</div>
      </div>
      <div className="kpi accent">
        <div className="kpi-label">Queued for review</div>
        <div className="kpi-val">{kpi.queued}</div>
        <div className="kpi-sub">{((kpi.queued / kpi.total_frames) * 100).toFixed(1)}% of frames</div>
      </div>
      <div className="kpi">
        <div className="kpi-label">Est. interactions</div>
        <div className="kpi-val">{kpi.est_interactions}</div>
        <div className="kpi-sub">clicks to fix queue</div>
      </div>
      <div className="kpi">
        <div className="kpi-label">Interactions / min</div>
        <div className="kpi-val">{kpi.interactions_per_min.toFixed(1)}</div>
        <div className="kpi-sub">of source video</div>
      </div>
      <div className="kpi good">
        <div className="kpi-label">Saved vs fixed-interval</div>
        <div className="kpi-val">{kpi.saved_interactions}<small> / {kpi.saved_pct}%</small></div>
        <div className="kpi-sub">vs every {kpi.fixed_interval} frames ({kpi.fixed_checks})</div>
      </div>
    </div>
  );
}
window.KpiStrip = KpiStrip;

function SetupPanel({ running, onRun, elapsed }) {
  return (
    <div className="setup">
      <div className="panel-h" style={{ paddingLeft: 0, paddingRight: 0 }}>
        <div className="t"><Ic n="upload" />Setup &amp; run</div>
        <div className="c">pass 1</div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
        <div className="media-mini">
          <div className="tag">input</div>
          <div className="ph"><Ic n="video" style={{ width: 18, height: 18 }} /></div>
        </div>
        <div className="media-mini">
          <div className="tag" style={{ color: "var(--mask)" }}>overlay</div>
          <div className="ph"><Ic n="layers" style={{ width: 18, height: 18 }} /></div>
        </div>
      </div>
      <div className="field">
        <div className="field-l"><span>Initial object box · x1,y1,x2,y2</span>
          <span className="ok"><Ic n="check" />set</span></div>
        <input className="input" defaultValue="89,216,346,633" />
      </div>
      <button className="btn btn-primary btn-block" onClick={onRun} disabled={running}>
        {running ? <><span className="dot busy" />Running review pass…</> : <><Ic n="sparkle" />Re-run review pass</>}
      </button>
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 8, fontSize: 10.5, color: "var(--tx-3)", fontFamily: "var(--mono)" }}>
        <span>{window.VIDEO.device}</span>
        <span>{running ? "—" : "last: " + window.KPI.pass_seconds + "s"}</span>
      </div>
    </div>
  );
}
window.SetupPanel = SetupPanel;

function QueueList({ queue, corrected, selId, onSelect, filter, setFilter }) {
  const items = queue.filter((q) => {
    if (filter === "pending") return !corrected[q.id];
    if (filter === "done") return corrected[q.id];
    return true;
  });
  const counts = {
    all: queue.length,
    pending: queue.filter((q) => !corrected[q.id]).length,
    done: queue.filter((q) => corrected[q.id]).length,
  };
  return (
    <div className="queue-wrap">
      <div className="panel-h">
        <div className="t"><Ic n="braces" />Review queue</div>
        <div className="c">{counts.pending} pending</div>
      </div>
      <div className="queue-filter">
        {[["all", "All"], ["pending", "Pending"], ["done", "Corrected"]].map(([id, lbl]) => (
          <button key={id} className={"qf" + (filter === id ? " on" : "")} onClick={() => setFilter(id)}>
            {lbl} {counts[id]}
          </button>
        ))}
      </div>
      <div className="queue-list">
        {items.map((q) => {
          const done = corrected[q.id];
          const rec = window.RECS[q.rec];
          const cc = confColor(q.conf);
          return (
            <div key={q.id} className={"qitem" + (q.id === selId ? " sel" : "") + (done ? " done" : "")} onClick={() => onSelect(q)}>
              <div className="qitem-top">
                <span className="qrisk" style={{ background: done ? "#3ddc97" : riskColor(q.risk) }} />
                <span className="qframe">f{q.frame}<span> · risk {q.risk.toFixed(2)}</span></span>
                <span className="qitem-spacer" />
                {done
                  ? <span className="qdone-badge"><Ic n="check" />fixed</span>
                  : <span className="qconf" style={{ background: cc.bg, color: cc.fg }}>{q.conf.toFixed(2)}</span>}
              </div>
              <div className="reason-row">
                {q.reasons.map((r) => {
                  const rr = window.REASONS[r];
                  return <span key={r} className="rtag" style={{ background: rr.color + "22", color: rr.color }}>{rr.short}</span>;
                })}
              </div>
              <div className="rec-row">
                <span style={{ color: "var(--tx-3)" }}>fix:</span>
                <span className="rec-chip" style={{ color: rec.color }}>
                  <Ic n={rec.icon === "box" ? "tightbox" : rec.icon} />{rec.label}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
window.QueueList = QueueList;

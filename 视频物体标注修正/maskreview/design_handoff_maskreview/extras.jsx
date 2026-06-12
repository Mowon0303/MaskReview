/* ============================================================
   Re-propagate (before/after) + Calibration (threshold sweep)
   ============================================================ */
const { useState: useStateE } = React;

function MiniMetric({ k, v, color }) {
  return <div className="m" style={{ display: "flex", flexDirection: "column", gap: 2 }}>
    <span style={{ fontSize: 9, color: "var(--tx-3)", fontFamily: "var(--mono)" }}>{k}</span>
    <span style={{ fontSize: 12, fontWeight: 600, fontFamily: "var(--mono)", color: color || "var(--tx-1)" }}>{v}</span>
  </div>;
}

function RepropView({ correctedCount }) {
  const [state, setState] = useStateE("idle"); // idle | running | done
  const R = window.REPROP;
  const run = () => { setState("running"); setTimeout(() => setState("done"), 2200); };

  return (
    <div className="fullview">
      <div className="fv-head">
        <div className="fv-title">Re-propagation</div>
        <div className="fv-sub">Apply saved corrections as new SAM2 prompts and re-run propagation from each corrected frame. Compare the review queue and mask quality before and after.</div>
      </div>
      <div className="fv-grid">
        <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 22 }}>
          <button className="btn btn-primary" style={{ height: 38, padding: "0 20px" }} onClick={run} disabled={state === "running"}>
            {state === "running" ? <><span className="dot busy" />Re-propagating…</> : <><Ic n="refresh" />Re-propagate {correctedCount} corrections</>}
          </button>
          <div style={{ fontSize: 11.5, color: "var(--tx-3)", fontFamily: "var(--mono)" }}>
            {state === "done" ? "completed · 38.2s · writes comparison.json" : correctedCount + " corrections staged"}
          </div>
        </div>

        {state !== "done" ? (
          <div className="card" style={{ display: "grid", placeItems: "center", height: 280, color: "var(--tx-3)", textAlign: "center" }}>
            <div>
              <Ic n="compare" style={{ width: 30, height: 30, color: "var(--tx-4)", marginBottom: 12 }} />
              <div style={{ fontSize: 12.5 }}>{state === "running" ? "Propagating corrected prompts across the clip…" : "Run re-propagation to see the before / after comparison."}</div>
            </div>
          </div>
        ) : (
          <React.Fragment>
            <div className="compare-row">
              <div className="compare-card">
                <div className="ch"><span className="lbl">Before</span><span className="badge2" style={{ background: "rgba(255,77,94,0.16)", color: "var(--neg)" }}>{R.before_queue} queued</span></div>
                <div className="compare-media"><FrameView frame={R.sample_frame} view="overlay" corrected={false} /><div className="frame-badge"><span className="fb">f{R.sample_frame}</span></div>
                  <div className="iou-readout"><MiniMetric k="mean IoU" v={R.before_mean_iou.toFixed(2)} color="var(--neg)" /></div>
                </div>
              </div>
              <div className="compare-arrow"><Ic n="arrowr" /></div>
              <div className="compare-card">
                <div className="ch"><span className="lbl">After</span><span className="badge2" style={{ background: "rgba(61,220,151,0.16)", color: "var(--pos)" }}>{R.after_queue} queued</span></div>
                <div className="compare-media"><FrameView frame={R.sample_frame} view="overlay" corrected={true} /><div className="frame-badge"><span className="fb ok">f{R.sample_frame} · corrected</span></div>
                  <div className="iou-readout"><MiniMetric k="mean IoU" v={R.after_mean_iou.toFixed(2)} color="var(--pos)" /></div>
                </div>
              </div>
            </div>

            <div className="delta-grid">
              <div className="delta-card good">
                <div className="dl">Queue frames</div>
                <div className="dv"><span className="from">{R.before_queue}</span><span className="arrow"><Ic n="arrowr" style={{ width: 14, height: 14 }} /></span><span className="to">{R.after_queue}</span></div>
                <div className="chg">−{R.before_queue - R.after_queue} frames</div>
              </div>
              <div className="delta-card good">
                <div className="dl">Queue reduction</div>
                <div className="dv"><span className="to">{R.queue_reduction}%</span></div>
                <div className="chg">fewer frames to review</div>
              </div>
              <div className="delta-card">
                <div className="dl">Actual interactions</div>
                <div className="dv"><span className="to" style={{ color: "var(--tx-1)" }}>{R.actual_interactions}</span></div>
                <div className="chg" style={{ color: "var(--tx-3)" }}>vs 17 estimated</div>
              </div>
              <div className="delta-card good">
                <div className="dl">Mean mask IoU</div>
                <div className="dv"><span className="from">{R.before_mean_iou.toFixed(2)}</span><span className="arrow"><Ic n="arrowr" style={{ width: 14, height: 14 }} /></span><span className="to">{R.after_mean_iou.toFixed(2)}</span></div>
                <div className="chg">+{(R.after_mean_iou - R.before_mean_iou).toFixed(2)} IoU</div>
              </div>
            </div>
          </React.Fragment>
        )}
      </div>
    </div>
  );
}
window.RepropView = RepropView;

function Bar({ v, color }) {
  return <div className="bar-cell"><span style={{ minWidth: 34 }}>{v.toFixed(2)}</span><div className="bar-track"><div className="bar-fill" style={{ width: v * 100 + "%", background: color }} /></div></div>;
}

function CalibView() {
  const C = window.CALIB;
  return (
    <div className="fullview">
      <div className="fv-head">
        <div className="fv-title">Evaluation &amp; threshold calibration</div>
        <div className="fv-sub">Tune queue sensitivity against an annotated eval set. Each preset trades queue precision (fewer false alarms) for recall (fewer missed bad frames). Goal: minimise interactions while catching every real failure.</div>
      </div>
      <div className="fv-grid">
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 22 }}>
          <div className="card" style={{ padding: 14, display: "flex", alignItems: "center", gap: 12 }}>
            <span className="ai" style={{ width: 34, height: 34, borderRadius: 8, background: "var(--bg-3)", display: "grid", placeItems: "center" }}><Ic n="braces" style={{ width: 16, height: 16, color: "var(--tx-2)" }} /></span>
            <div><div style={{ fontFamily: "var(--mono)", fontSize: 12.5 }}>{C.manifest}</div><div style={{ fontSize: 10.5, color: "var(--tx-3)" }}>eval manifest · loaded</div></div>
          </div>
          <div className="card" style={{ padding: 14, display: "flex", alignItems: "center", gap: 12 }}>
            <span className="ai" style={{ width: 34, height: 34, borderRadius: 8, background: "var(--bg-3)", display: "grid", placeItems: "center" }}><Ic n="image" style={{ width: 16, height: 16, color: "var(--tx-2)" }} /></span>
            <div><div style={{ fontFamily: "var(--mono)", fontSize: 12.5 }}>{C.cases_annotated} / {C.cases_total} cases annotated</div><div style={{ fontSize: 10.5, color: "var(--tx-3)" }}>expected_review_frames labelled</div></div>
          </div>
        </div>

        <div className="subhead"><Ic n="sliders" />Threshold presets</div>
        <table className="table" style={{ marginBottom: 26 }}>
          <thead><tr>
            <th>Preset</th><th>Queue precision</th><th>Queue recall</th><th>F1</th><th>Missed</th><th>False positives</th><th>Saved vs fixed</th>
          </tr></thead>
          <tbody>
            {C.presets.map((p) => (
              <tr key={p.id} className={p.active ? "active-preset" : ""}>
                <td><span className="preset-name">{p.active && <span className="dot live" />}{p.label}</span></td>
                <td><Bar v={p.precision} color="#5b9dff" /></td>
                <td><Bar v={p.recall} color="#1fdac6" /></td>
                <td><Bar v={p.f1} color="#3ddc97" /></td>
                <td style={{ color: p.missed > 5 ? "var(--neg)" : "var(--tx-1)" }}>{p.missed}</td>
                <td style={{ color: p.fp > 5 ? "var(--warn)" : "var(--tx-1)" }}>{p.fp}</td>
                <td style={{ color: "var(--pos)" }}>{p.saved}%</td>
              </tr>
            ))}
          </tbody>
        </table>

        <div className="subhead"><Ic n="image" />Eval video cases</div>
        <table className="table">
          <thead><tr><th>Case</th><th>Frames</th><th>Expected review frames</th><th>Status</th></tr></thead>
          <tbody>
            {C.cases.map((c) => (
              <tr key={c.name}>
                <td style={{ fontFamily: "var(--sans)", fontWeight: 500 }}><span style={{ fontFamily: "var(--mono)", fontSize: 12 }}>{c.name}</span></td>
                <td>{c.frames.toLocaleString()}</td>
                <td>{c.status === "annotated" ? c.expected : <span style={{ color: "var(--tx-3)" }}>—</span>}</td>
                <td><span className={"pill-status " + c.status}>{c.status}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
window.CalibView = CalibView;

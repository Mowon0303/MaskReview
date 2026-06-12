/* ============================================================
   Center viewer (frame + mask + correction canvas + transport)
   and right panel (selected item JSON + correction + artifacts)
   ============================================================ */
const { useRef: useRefR, useState: useStateR, useEffect: useEffectR } = React;

function CenterViewer({
  frame, setFrame, view, setView, playing, setPlaying,
  sel, corrected, corrType, corrPoint, corrBox, setCorrPoint, setCorrBox, jumpQueue,
}) {
  const stageRef = useRefR(null);
  const layerRef = useRefR(null);
  const [dim, setDim] = useStateR({ W: 880, H: 495 });
  const [draft, setDraft] = useStateR(null);
  const drawing = useRefR(null);

  useEffectR(() => {
    const measure = () => {
      const el = stageRef.current; if (!el) return;
      const aw = el.clientWidth - 36, ah = el.clientHeight - 36;
      let W = aw, H = (aw * 9) / 16;
      if (H > ah) { H = ah; W = (ah * 16) / 9; }
      setDim({ W: Math.round(W), H: Math.round(H) });
    };
    measure();
    const ro = new ResizeObserver(measure);
    if (stageRef.current) ro.observe(stageRef.current);
    return () => ro.disconnect();
  }, []);

  const norm = (e) => {
    const r = layerRef.current.getBoundingClientRect();
    return [Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)), Math.max(0, Math.min(1, (e.clientY - r.top) / r.height))];
  };
  const onLayerDown = (e) => {
    if (!sel) return;
    if (corrType === "tight_box") {
      const [x, y] = norm(e); drawing.current = { x, y };
      setDraft({ x, y, w: 0, h: 0 });
      const mv = (ev) => { const [nx, ny] = norm(ev); const d = drawing.current; setDraft({ x: Math.min(d.x, nx), y: Math.min(d.y, ny), w: Math.abs(nx - d.x), h: Math.abs(ny - d.y) }); };
      const up = (ev) => {
        window.removeEventListener("pointermove", mv); window.removeEventListener("pointerup", up);
        const [nx, ny] = norm(ev); const d = drawing.current;
        const b = [Math.min(d.x, nx), Math.min(d.y, ny), Math.abs(nx - d.x), Math.abs(ny - d.y)];
        setDraft(null);
        if (b[2] > 0.01 && b[3] > 0.01) setCorrBox(b);
      };
      window.addEventListener("pointermove", mv); window.addEventListener("pointerup", up);
    } else {
      const [x, y] = norm(e); setCorrPoint({ x, y });
    }
  };

  const qHere = window.queueAtFrame(Math.round(frame));
  const isDone = qHere && corrected[qHere.id];
  const metrics = qHere ? qHere.metrics : { iou: 0.93, dArea: 0.0, dCtr: 0.0 };
  const showCorrected = qHere && corrected[qHere.id];

  return (
    <div className="col center">
      <div className="viewer-bar">
        <div className="viewseg">
          {[["overlay", "Overlay", "layers"], ["mask", "Mask only", "mask"], ["source", "Source", "image"]].map(([id, lbl, ic]) => (
            <button key={id} className={view === id ? "on" : ""} onClick={() => setView(id)}><Ic n={ic} />{lbl}</button>
          ))}
        </div>
        <div className="tool-pills">
          <span style={{ fontSize: 10.5, color: "var(--tx-3)", alignSelf: "center", padding: "0 6px" }}>tool</span>
          {[["positive_point", "Positive", "var(--pos)", "plus"], ["negative_point", "Negative", "var(--neg)", "minus"], ["tight_box", "Box", "var(--orange)", "tightbox"]].map(([id, lbl, col, ic]) => (
            <button key={id} className={"tpill" + (corrType === id ? " on" : "")} onClick={() => window.__setCorrType(id)} title={lbl}>
              <span className="ic" style={{ background: col }}><Ic n={ic} style={{ width: 10, height: 10, color: "#08110d" }} /></span>{lbl}
            </button>
          ))}
        </div>
        <div className="spacer" />
        <div style={{ fontSize: 11, color: "var(--tx-3)", fontFamily: "var(--mono)" }}>
          {sel ? "editing q" + sel.id.slice(1) : "select a queue item to correct"}
        </div>
      </div>

      <div className="stage" ref={stageRef}>
        <div className="frame-wrap" style={{ width: dim.W, height: dim.H }}>
          <FrameView W={dim.W} H={dim.H} frame={Math.round(frame)} view={view} corrected={showCorrected} />

          <div className="frame-badge">
            <span className="fb">f{Math.round(frame)} · {window.fmtTC(frame)}</span>
            {qHere && !isDone && <span className="fb flag"><Ic n="alert" style={{ width: 11, height: 11, display: "inline", verticalAlign: "-1px" }} /> flagged</span>}
            {isDone && <span className="fb ok">corrected</span>}
          </div>

          {/* correction layer */}
          <div ref={layerRef} className={"corr-layer" + (sel ? " placing" : "")} onPointerDown={onLayerDown}>
            {sel && corrPoint && corrType !== "tight_box" && (
              <div className="cpoint" style={{ left: corrPoint.x * dim.W, top: corrPoint.y * dim.H, background: corrType === "positive_point" ? "var(--pos)" : "var(--neg)" }}>
                <Ic n={corrType === "positive_point" ? "plus" : "minus"} />
              </div>
            )}
            {sel && corrBox && corrType === "tight_box" && (
              <div className="cbox" style={{ left: corrBox[0] * dim.W, top: corrBox[1] * dim.H, width: corrBox[2] * dim.W, height: corrBox[3] * dim.H }} />
            )}
            {draft && <div className="cbox-draft" style={{ left: draft.x * dim.W, top: draft.y * dim.H, width: draft.w * dim.W, height: draft.h * dim.H }} />}
          </div>

          {view !== "source" && (
            <div className="iou-readout">
              <div className="m"><span className="k">mask IoU</span><span className="v" style={{ color: showCorrected ? "var(--pos)" : metrics.iou < 0.5 ? "var(--neg)" : "var(--tx-1)" }}>{showCorrected ? "0.94" : metrics.iou.toFixed(2)}</span></div>
              <div className="m"><span className="k">Δarea</span><span className="v">{(showCorrected ? 0 : metrics.dArea > 0 ? metrics.dArea : metrics.dArea).toFixed(2)}</span></div>
              <div className="m"><span className="k">Δcenter</span><span className="v">{(showCorrected ? 0 : metrics.dCtr).toFixed(2)}</span></div>
            </div>
          )}
        </div>
      </div>

      <div className="transport-bar">
        <div className="tp-row">
          <div className="tp-btns">
            <button className="tpb" onClick={() => setFrame((f) => Math.max(0, Math.round(f) - 1))}><Ic n="prevf" /></button>
            <button className="tpb play" onClick={() => setPlaying((p) => !p)}><Ic n={playing ? "pause" : "play"} /></button>
            <button className="tpb" onClick={() => setFrame((f) => Math.min(window.VIDEO.frames, Math.round(f) + 1))}><Ic n="nextf" /></button>
          </div>
          <div className="tc-display">{window.fmtTC(frame).split(":").map((p, i) => <React.Fragment key={i}>{i ? <span>:</span> : null}{p}</React.Fragment>)}</div>
          <div className="frame-num">frame {Math.round(frame)} / {window.VIDEO.frames}</div>
          <button className="jump-q" onClick={jumpQueue}><Ic n="jump" style={{ width: 13, height: 13 }} />jump to next flagged</button>
        </div>
        <Timeline frame={frame} setFrame={setFrame} corrected={corrected} sel={sel} />
      </div>
    </div>
  );
}
window.CenterViewer = CenterViewer;

function Timeline({ frame, setFrame, corrected, sel }) {
  const ref = useRefR(null);
  const N = window.VIDEO.frames;
  const scrub = (e) => {
    const r = ref.current.getBoundingClientRect();
    setFrame(Math.max(0, Math.min(N, Math.round(((e.clientX - r.left) / r.width) * N))));
  };
  const down = (e) => {
    scrub(e);
    const mv = (ev) => scrub(ev);
    const up = () => { window.removeEventListener("pointermove", mv); window.removeEventListener("pointerup", up); };
    window.addEventListener("pointermove", mv); window.addEventListener("pointerup", up);
  };
  return (
    <React.Fragment>
      <div className="timeline" ref={ref} onPointerDown={down}>
        {window.QUEUE.map((q) => {
          const done = corrected[q.id];
          const col = done ? "#3ddc97" : riskColor(q.risk);
          return (
            <div key={q.id} className="tl-marker" style={{ left: (q.frame / N) * 100 + "%" }}>
              <div className="tick" style={{ background: col, opacity: sel && sel.id === q.id ? 1 : 0.5 }} />
              <div className="cap" style={{ background: col }} />
            </div>
          );
        })}
        <div className="tl-playhead" style={{ left: (frame / N) * 100 + "%" }} />
      </div>
      <div className="tl-axis">
        {[0, 0.25, 0.5, 0.75, 1].map((p) => <span key={p}>{window.fmtTC(p * N)}</span>)}
      </div>
    </React.Fragment>
  );
}

/* ---------------- RIGHT PANEL ---------------- */
function RightPanel({ sel, corrected, corrType, corrPoint, corrBox, note, setNote, onSave, justSaved }) {
  return (
    <div className="col right">
      <div className="panel-h">
        <div className="t"><Ic n="braces" />Selected queue item</div>
        {sel && <div className="c">q{sel.id.slice(1)}</div>}
      </div>
      {sel ? (
        <div className="right-scroll">
          <div className="section">
            <SelectedJson sel={sel} done={corrected[sel.id]} />
          </div>
          <div className="section">
            <div className="panel-h"><div className="t"><Ic n="tightbox" />Correction</div></div>
            <div className="corr-controls">
              <div className="corr-types">
                {[["positive_point", "Positive", "pos", "plus", "var(--pos)"], ["negative_point", "Negative", "neg", "minus", "var(--neg)"], ["tight_box", "Tight box", "box", "tightbox", "var(--orange)"]].map(([id, lbl, cls, ic, col]) => (
                  <button key={id} className={"ctype " + cls + (corrType === id ? " on" : "") + (sel.rec === id ? " rec-flag" : "")} onClick={() => window.__setCorrType(id)}>
                    <span className="badge-ic" style={{ background: col }}><Ic n={ic} /></span>{lbl}
                  </button>
                ))}
              </div>

              {corrType === "tight_box" ? (
                <div className="field">
                  <div className="field-l"><span>Tight box · x1,y1,x2,y2</span></div>
                  <input className="input" value={corrBox ? corrBox.map((v, i) => Math.round((i < 2 ? v : (i === 2 ? corrBox[0] + v : corrBox[1] + v)) * (i % 2 ? window.VIDEO.h : window.VIDEO.w))).join(",") : ""} placeholder="drag on the frame" readOnly />
                </div>
              ) : (
                <div className="field">
                  <div className="field-l"><span>Point · x,y (px)</span></div>
                  <input className="input" value={corrPoint ? Math.round(corrPoint.x * window.VIDEO.w) + "," + Math.round(corrPoint.y * window.VIDEO.h) : ""} placeholder="click on the frame" readOnly />
                </div>
              )}

              <div className="corr-hint">
                {corrType === "tight_box"
                  ? <>Drag a <b>tight box</b> around the true object on the frame.</>
                  : corrType === "positive_point"
                    ? <>Click a point <b>inside</b> the object to pull the mask in.</>
                    : <>Click a point on <b>background</b> to push the mask out.</>}
                {sel.rec === corrType && <span style={{ color: "var(--orange)" }}> · model-recommended</span>}
              </div>

              <div className="field">
                <div className="field-l"><span>Note</span></div>
                <input className="input" value={note} onChange={(e) => setNote(e.target.value)} placeholder="optional" style={{ fontFamily: "var(--sans)" }} />
              </div>

              <button className="btn btn-primary btn-block" onClick={onSave}
                disabled={corrType === "tight_box" ? !corrBox : !corrPoint}>
                <Ic n="check" />Save correction
              </button>
              {justSaved && <div className="save-status"><Ic n="check" />Saved to corrections.json · frame re-queued for re-propagation</div>}
            </div>
          </div>
          <div className="section" style={{ borderBottom: "none" }}>
            <div className="panel-h"><div className="t"><Ic n="download" />Artifacts</div></div>
            {window.ARTIFACTS.map((a) => (
              <div key={a.name} className="artifact">
                <span className="ai"><Ic n={a.kind === "video" ? "video" : "braces"} /></span>
                <span className="an"><span className="nm">{a.name}</span><span className="sz">{a.size}</span></span>
                <button className="dl"><Ic n="download" /></button>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="empty-pane">
          <div><Ic n="braces" /><div>Pick a frame from the review queue<br />to inspect metrics and correct its mask.</div></div>
        </div>
      )}
    </div>
  );
}
window.RightPanel = RightPanel;

function SelectedJson({ sel, done }) {
  const L = ({ k, v, cls }) => (
    <div><span className="jk">"{k}"</span><span className="jb">: </span><span className={cls}>{v}</span><span className="jb">,</span></div>
  );
  return (
    <div className="json-view">
      <div><span className="jb">{"{"}</span></div>
      <div style={{ paddingLeft: 14 }}>
        <L k="frame" v={sel.frame} cls="jn" />
        <L k="timecode" v={'"' + window.fmtTC(sel.frame) + '"'} cls="js" />
        <div><span className="jk">"reasons"</span><span className="jb">: [</span>
          {sel.reasons.map((r, i) => <span key={r}><span className="js">"{r}"</span>{i < sel.reasons.length - 1 ? <span className="jb">, </span> : null}</span>)}
          <span className="jb">],</span></div>
        <div><span className="jk">"metrics"</span><span className="jb">: {"{"}</span></div>
        <div style={{ paddingLeft: 14 }}>
          <L k="iou" v={sel.metrics.iou.toFixed(2)} cls="jn" />
          <L k="d_area" v={sel.metrics.dArea.toFixed(2)} cls="jn" />
          <L k="d_center" v={sel.metrics.dCtr.toFixed(2)} cls="jn" />
          <div><span className="jk">"edge_contact"</span><span className="jb">: </span><span className="jn">{sel.metrics.edge.toFixed(2)}</span></div>
        </div>
        <div><span className="jb">{"},"}</span></div>
        <L k="confidence" v={sel.conf.toFixed(2)} cls="jn" />
        <L k="risk" v={sel.risk.toFixed(2)} cls="jn" />
        <L k="recommended" v={'"' + sel.rec + '"'} cls="js" />
        <div><span className="jk">"status"</span><span className="jb">: </span><span className="js">"{done ? "corrected" : "pending"}"</span></div>
      </div>
      <div><span className="jb">{"}"}</span></div>
    </div>
  );
}

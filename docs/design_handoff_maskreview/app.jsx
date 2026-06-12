/* ============================================================
   MaskReview — app state + flows
   ============================================================ */
const { useState: useS, useEffect: useE, useRef: useRf } = React;

function MaskApp() {
  const [tab, setTab] = useS("review");
  const [frame, setFrame] = useS(147);
  const [playing, setPlaying] = useS(false);
  const [view, setView] = useS("overlay");
  const [sel, setSel] = useS(window.QUEUE[1]);
  const [queueFilter, setQueueFilter] = useS("all");
  const [corrected, setCorrected] = useS({});
  const [corrType, setCorrType] = useS(window.QUEUE[1].rec);
  const [corrPoint, setCorrPoint] = useS(null);
  const [corrBox, setCorrBox] = useS(null);
  const [note, setNote] = useS("");
  const [justSaved, setJustSaved] = useS(false);
  const [running, setRunning] = useS(false);

  window.__setCorrType = setCorrType;

  const selectItem = (q) => {
    setSel(q); setFrame(q.frame); setCorrType(q.rec);
    setCorrPoint(null); setCorrBox(null); setNote(""); setJustSaved(false);
    setView("overlay");
  };

  const onSave = () => {
    if (!sel) return;
    setCorrected((c) => ({ ...c, [sel.id]: true }));
    setJustSaved(true);
    setTimeout(() => setJustSaved(false), 2600);
  };

  const jumpQueue = () => {
    const f = Math.round(frame);
    const next = window.QUEUE.find((q) => q.frame > f) || window.QUEUE[0];
    selectItem(next);
  };

  const onRun = () => { setRunning(true); setTimeout(() => setRunning(false), 2200); };

  // playback
  useE(() => {
    if (!playing) return;
    let raf, last = performance.now();
    const tick = (t) => {
      const dt = (t - last) / 1000; last = t;
      setFrame((f) => { let nf = f + dt * window.VIDEO.fps; if (nf >= window.VIDEO.frames) nf = 0; return nf; });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  // keyboard
  useE(() => {
    const onKey = (e) => {
      if (e.target.tagName === "INPUT") return;
      const k = e.key.toLowerCase();
      if (k === " ") { e.preventDefault(); setPlaying((p) => !p); }
      else if (k === "arrowright") setFrame((f) => Math.min(window.VIDEO.frames, Math.round(f) + 1));
      else if (k === "arrowleft") setFrame((f) => Math.max(0, Math.round(f) - 1));
      else if (k === "1") setCorrType("positive_point");
      else if (k === "2") setCorrType("negative_point");
      else if (k === "3") setCorrType("tight_box");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const correctedCount = Object.keys(corrected).filter((k) => corrected[k]).length;

  return (
    <div className="app">
      <TopBar tab={tab} setTab={setTab} queued={window.KPI.queued - correctedCount} running={running} onExport={() => {}} />

      {tab === "review" && (
        <React.Fragment>
          <KpiStrip kpi={window.KPI} />
          <div className="work">
            <div className="col left">
              <SetupPanel running={running} onRun={onRun} />
              <QueueList queue={window.QUEUE} corrected={corrected} selId={sel && sel.id}
                onSelect={selectItem} filter={queueFilter} setFilter={setQueueFilter} />
            </div>
            <CenterViewer
              frame={frame} setFrame={setFrame} view={view} setView={setView}
              playing={playing} setPlaying={setPlaying} sel={sel} corrected={corrected}
              corrType={corrType} corrPoint={corrPoint} corrBox={corrBox}
              setCorrPoint={setCorrPoint} setCorrBox={setCorrBox} jumpQueue={jumpQueue}
            />
            <RightPanel
              sel={sel} corrected={corrected} corrType={corrType}
              corrPoint={corrPoint} corrBox={corrBox} note={note} setNote={setNote}
              onSave={onSave} justSaved={justSaved}
            />
          </div>
        </React.Fragment>
      )}

      {tab === "reprop" && <RepropView correctedCount={correctedCount} />}
      {tab === "calib" && <CalibView />}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<MaskApp />);

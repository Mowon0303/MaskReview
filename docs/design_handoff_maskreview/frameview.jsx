/* ============================================================
   FrameView — draws the clip frame the SAM2 mask is reviewed on.
   Muted plaza backdrop + a walking subject + the propagated mask,
   which is rendered WRONG at flagged frames (drift / bleed / lost
   / shrink / edge) unless the frame has been corrected.
   view: "overlay" | "mask" | "source"
   ============================================================ */
const VBW = 1000, VBH = 562;

function personLocal(fill, mask) {
  // local coords 0..100 wide, 0..200 tall
  if (mask) {
    return (
      <g fill={fill}>
        <circle cx="50" cy="26" r="17" />
        <path d="M28 46 Q50 36 72 46 L70 120 L30 120 Z" />
        <path d="M30 116 L44 116 L42 196 L30 196 Z" />
        <path d="M56 116 L70 116 L70 196 L58 196 Z" />
        <path d="M28 50 L18 104 L26 106 L36 56 Z" />
        <path d="M72 50 L82 104 L74 106 L64 56 Z" />
      </g>
    );
  }
  return (
    <g>
      <path d="M30 116 L44 116 L42 196 L30 196 Z" fill="#1b2433" />
      <path d="M56 116 L70 116 L70 196 L58 196 Z" fill="#222c3d" />
      <path d="M28 50 L18 104 L26 106 L36 56 Z" fill="#2c3346" />
      <path d="M72 50 L82 104 L74 106 L64 56 Z" fill="#343c52" />
      <path d="M28 46 Q50 36 72 46 L70 120 L30 120 Z" fill="#384768" />
      <circle cx="50" cy="26" r="17" fill="#caa078" />
      <path d="M34 20 Q50 6 66 20 Q58 14 50 15 Q42 14 34 20 Z" fill="#241d28" />
    </g>
  );
}

function Backdrop({ mode }) {
  if (mode === "mask") return (
    <svg className="frame-svg" viewBox={`0 0 ${VBW} ${VBH}`} preserveAspectRatio="none">
      <rect width={VBW} height={VBH} fill="#06080b" />
    </svg>
  );
  return (
    <svg className="frame-svg" viewBox={`0 0 ${VBW} ${VBH}`} preserveAspectRatio="none">
      <defs>
        <linearGradient id="wall" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#262b33" /><stop offset="1" stopColor="#1c2128" />
        </linearGradient>
        <linearGradient id="ground" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#33373d" /><stop offset="1" stopColor="#181b20" />
        </linearGradient>
      </defs>
      <rect width={VBW} height={VBH} fill="url(#wall)" />
      <rect y="300" width={VBW} height={VBH - 300} fill="url(#ground)" />
      {/* wall seams */}
      <g stroke="#2e333b" strokeWidth="1.5" opacity="0.6">
        <line x1="0" y1="300" x2={VBW} y2="300" />
        <line x1="180" y1="120" x2="180" y2="300" /><line x1="520" y1="120" x2="520" y2="300" />
        <line x1="820" y1="120" x2="820" y2="300" />
        <line x1="60" y1="200" x2="940" y2="200" opacity="0.5" />
      </g>
      {/* ground markings (perspective) */}
      <g stroke="#454a52" strokeWidth="2" opacity="0.4">
        <line x1="300" y1="300" x2="120" y2="562" /><line x1="700" y1="300" x2="880" y2="562" />
        <line x1="0" y1="430" x2={VBW} y2="430" opacity="0.5" />
      </g>
      {/* a planter + pole for context */}
      <rect x="60" y="250" width="56" height="58" rx="4" fill="#23272e" />
      <rect x="64" y="236" width="48" height="18" rx="3" fill="#2c5a3c" opacity="0.7" />
      <rect x="900" y="150" width="6" height="160" fill="#2a2f37" />
      <rect x="884" y="146" width="38" height="10" rx="3" fill="#3a4150" />
    </svg>
  );
}

function placeAt(box) {
  const [x, y, w, h] = box;
  return `translate(${x * VBW} ${y * VBH}) scale(${(w * VBW) / 100} ${(h * VBH) / 200})`;
}

function maskBoxFromErr(box, err) {
  let [x, y, w, h] = box;
  const s = err.scale || 1;
  const mw = w * s, mh = h * s;
  let mx = x - (mw - w) / 2 + (err.dx || 0);
  let my = y - (mh - h) / 2 + (err.dy || 0);
  if (err.edge) mx = Math.max(box[0] + 0.02, mx);
  return [mx, my, mw, mh];
}

function FrameView({ W, H, frame, view, corrected }) {
  const box = window.subjectBox(frame);
  const err = corrected ? {} : window.maskErrAt(frame);
  const hasMask = !(err.empty && !corrected);
  const mbox = maskBoxFromErr(box, err);
  const clipFrac = err.clip; // show only bottom fraction
  const uid = "mk" + frame;

  return (
    <React.Fragment>
      <Backdrop mode={view} />
      {/* source subject (hidden in mask-only view) */}
      {view !== "mask" && (
        <svg className="frame-svg" viewBox={`0 0 ${VBW} ${VBH}`} preserveAspectRatio="none">
          <g transform={placeAt(box)}>{personLocal(null, false)}</g>
          <ellipse cx={(box[0] + box[2] / 2) * VBW} cy={(box[1] + box[3]) * VBH} rx={box[2] * VBW * 0.5} ry="7" fill="#000" opacity="0.3" />
        </svg>
      )}
      {/* SAM2 mask */}
      {hasMask && (view === "overlay" || view === "mask") && (
        <svg className="overlay-layer" viewBox={`0 0 ${VBW} ${VBH}`} preserveAspectRatio="none">
          <defs>
            {clipFrac && (
              <clipPath id={uid}>
                <rect x={mbox[0] * VBW - 20} y={(mbox[1] + mbox[3] * (1 - clipFrac)) * VBH}
                  width={mbox[2] * VBW + 40} height={mbox[3] * clipFrac * VBH + 20} />
              </clipPath>
            )}
          </defs>
          <g clipPath={clipFrac ? `url(#${uid})` : undefined}>
            <g transform={placeAt(mbox)} opacity={view === "mask" ? 1 : 0.5}>
              {personLocal(view === "mask" ? "#1fdac6" : "#1fdac6", true)}
            </g>
            {/* bright mask edge */}
            <g transform={placeAt(mbox)} fill="none" stroke="#3df0dc" strokeWidth={view === "mask" ? 1.6 : 2.2} opacity="0.95">
              <circle cx="50" cy="26" r="17" />
              <path d="M28 46 Q50 36 72 46 L70 120 L30 120 Z" />
              <path d="M30 116 L44 116 L42 196 L30 196 Z" />
              <path d="M56 116 L70 116 L70 196 L58 196 Z" />
            </g>
          </g>
        </svg>
      )}
      {/* corrected confirmation tint */}
      {corrected && view === "overlay" && (
        <svg className="overlay-layer" viewBox={`0 0 ${VBW} ${VBH}`} preserveAspectRatio="none">
          <g transform={placeAt(box)} fill="none" stroke="#3ddc97" strokeWidth="1.6" opacity="0.8" strokeDasharray="5 4">
            <path d="M28 46 Q50 36 72 46 L70 196 L30 196 Z" />
          </g>
        </svg>
      )}
    </React.Fragment>
  );
}
window.FrameView = FrameView;

"use strict";

// ---------------------------------------------------------------------------
// Static metadata
// ---------------------------------------------------------------------------

const STYLE_NAMES = {
  BN: "Bossa Nova",
  Funk: "Funk",
  Jazz: "Jazz",
  Rock: "Rock",
  SS: "Singer-Songwriter",
};

const MODELS = [
  { id: "main_pipeline", label: "2 stage" },
  { id: "velocity_joint_peak", label: "1 stage" },
  { id: "ddsp_guitar", label: "ddsp-guitar" },
  { id: "ground_truth", label: "ground-truth" },
];

// tab10 colors matplotlib names in scripts/pipeline/plot_ctrl_preview.py's COLORS map to,
// so channel colors here match that script's PNGs exactly.
const TAB_COLORS = {
  "tab:blue": "#2f6fed",
  "tab:orange": "#e0820a",
  "tab:green": "#1f9d55",
  "tab:purple": "#8b5cf6",
  "tab:red": "#e0393e",
};

const INPUT_CHANNELS = [
  "midi_pitch",
  "pseudo_velocity",
  "segment_phase",
  "segment_onset",
  "voiced",
];
const TARGET_CHANNELS = ["pitch", "envelope"];

// distinct per-string colors for the MIDI piano-roll (low E -> high e), independent of
// the control-curve channel palette above.
const STRING_COLORS = [
  "#e0393e", // string 1 - low E
  "#e0820a", // string 2 - A
  "#c9a227", // string 3 - D
  "#1f9d55", // string 4 - G
  "#2f6fed", // string 5 - B
  "#8b5cf6", // string 6 - high e
];
const STRING_NAMES = ["Low E", "A", "D", "G", "B", "High E"];

// Output-comparison plot: one fixed color per system, independent of the string/channel
// palettes above, and the draw order (ground truth drawn first/underneath, so the three
// synthesized systems' lines aren't hidden behind it).
const COMPARE_ORDER = ["ground_truth", "main_pipeline", "velocity_joint_peak", "ddsp_guitar"];
const ARM_COMPARE_COLORS = {
  ground_truth: "#6b7280",
  main_pipeline: "#2f6fed",
  velocity_joint_peak: "#e0820a",
  ddsp_guitar: "#8b5cf6",
};

// Per-legend-item click toggles a system's line on/off in both comparison canvases, so
// two curves can be isolated for a direct A/B read. All visible by default. Single
// module-level state since only one "Single String Study" instance exists.
const compareVisible = {};
for (const arm of COMPARE_ORDER) compareVisible[arm] = true;

// ---------------------------------------------------------------------------
// Exclusive audio playback: never two clips at once, across every section.
// ---------------------------------------------------------------------------

let currentAudioEl = null;
function registerExclusive(el) {
  el.addEventListener("play", () => {
    if (currentAudioEl && currentAudioEl !== el) currentAudioEl.pause();
    currentAudioEl = el;
  });
}

// ---------------------------------------------------------------------------
// Shared playhead: whichever registered <audio> is currently playing drives every
// registered canvas (MIDI rolls + every curve/comparison panel, across every section)
// via one rAF loop.
// ---------------------------------------------------------------------------

let lastPlayheadT = null; // last active playback time, kept while paused so a legend
                           // toggle mid-pause redraws at the same position instead of blank
function tickPlayhead() {
  if (currentAudioEl && !currentAudioEl.paused) {
    lastPlayheadT = currentAudioEl.currentTime;
    redrawAllRegisteredCanvases(lastPlayheadT);
  }
  requestAnimationFrame(tickPlayhead);
}
requestAnimationFrame(tickPlayhead);

// ---------------------------------------------------------------------------
// Scroll-position preservation: selecting a model/string/sample can hide/show or resize
// sections above whatever the user is currently looking at, and the browser has no way
// to know what content the user cares about staying put -- it just leaves window.scrollY
// numerically unchanged, which visually "jumps" to different content once the layout
// above it grows/shrinks. This finds whichever stable landmark element's top edge is
// closest to the viewport's top edge before the mutation, then nudges scroll after the
// mutation so that same element lands at the same screen position again -- ephemeral
// rebuilt content (individual curve rows/canvases) is deliberately excluded from the
// candidate set since those get torn down and rebuilt, and <summary> elements are
// excluded too since several <details> sections can themselves be hidden by the very
// mutation being measured -- a hidden element's getBoundingClientRect() reports all
// zeros, which would compute a bogus delta.
// ---------------------------------------------------------------------------

const SCROLL_ANCHOR_SELECTOR = "h1, .panel, .audio-block, h3, .curve-section-header";

function _captureScrollAnchor() {
  let anchor = null;
  let anchorTop = null;
  for (const candidate of document.querySelectorAll(SCROLL_ANCHOR_SELECTOR)) {
    const top = candidate.getBoundingClientRect().top;
    if (anchorTop === null || Math.abs(top) < Math.abs(anchorTop)) {
      anchor = candidate;
      anchorTop = top;
    }
  }
  return { anchor, before: anchor ? anchor.getBoundingClientRect().top : null };
}

function _restoreScrollAnchor({ anchor, before }) {
  if (anchor && before != null && document.contains(anchor)) {
    const delta = anchor.getBoundingClientRect().top - before;
    if (delta !== 0) window.scrollBy(0, delta);
  }
}

function preserveScrollPosition(mutate) {
  const state = _captureScrollAnchor();
  mutate();
  _restoreScrollAnchor(state);
}

// Same idea, but spans an async sequence (e.g. loadSample's awaited fetch + nested
// loadCurves calls) as ONE before/after measurement instead of several small ones --
// sequential separate captures can each pick a slightly different anchor as intermediate
// mutations shift what's closest to the viewport top, compounding into a residual drift
// none of them individually catches.
async function preserveScrollPositionAsync(mutate) {
  const state = _captureScrollAnchor();
  await mutate();
  _restoreScrollAnchor(state);
}

// ---------------------------------------------------------------------------
// Shared state / data loading
// ---------------------------------------------------------------------------

let manifest = null;
let currentModel = MODELS[0].id; // shared across every section that has a model selector

const el = (id) => document.getElementById(id);

async function fetchJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`failed to fetch ${path}: ${res.status}`);
  return res.json();
}

async function loadManifest() {
  manifest = await fetchJSON("data/manifest.json");
}

// "Single String Study" defaults to whichever string actually carries the performance
// for this clip (most notes played), rather than always string 1 -- most GuitarSet
// takes concentrate the melody/comping on one or two strings, and the rest are mostly
// silent open/drone notes not worth landing on by default.
function mostActiveString(availableStrings, notes) {
  let best = availableStrings[0];
  let bestCount = -1;
  for (const s of availableStrings) {
    const count = (notes[String(s)] || []).length;
    if (count > bestCount) {
      best = s;
      bestCount = count;
    }
  }
  return best;
}

// ---------------------------------------------------------------------------
// MIDI piano-roll visualizer -- generic: any section's own canvas/notes/duration.
// ---------------------------------------------------------------------------

function buildStringLegend(container, strings) {
  container.innerHTML = "";
  for (const s of strings) {
    const span = document.createElement("span");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = STRING_COLORS[s - 1];
    span.appendChild(swatch);
    span.appendChild(document.createTextNode(`String ${s} (${STRING_NAMES[s - 1]})`));
    container.appendChild(span);
  }
}

// Fixed pitch range covering every note across all 5 demo recordings (MIDI 40-73),
// snapped out to whole-octave C boundaries so the C-note gridlines/labels land cleanly
// and the range never rescales per sample/string.
const MIDI_PITCH_MIN = 36; // C2
const MIDI_PITCH_MAX = 84; // C6

function niceTimeStep(duration) {
  // one tick per second up to 12s, else coarsen so labels don't collide
  if (duration <= 12) return 1;
  if (duration <= 30) return 2;
  return 5;
}

function drawMidiRoll(canvas, notes, duration, playheadT) {
  const cssWidth = Math.max(canvas.clientWidth, 300);
  const cssHeight = 300;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(cssWidth * dpr);
  canvas.height = Math.round(cssHeight * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const width = cssWidth;
  const height = cssHeight;
  ctx.clearRect(0, 0, width, height);

  if (!notes) return;
  const minPitch = MIDI_PITCH_MIN;
  const maxPitch = MIDI_PITCH_MAX;

  const padL = 38;
  const padR = 8;
  const padT = 8;
  const padB = 20;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;
  const xOf = (t) => padL + (t / duration) * plotW;
  const yOf = (p) => padT + (1 - (p - minPitch) / (maxPitch - minPitch)) * plotH;

  // horizontal guide line + "Cn" label at every octave (C2, C3, ... C6)
  ctx.font = "12px -apple-system, BlinkMacSystemFont, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let p = minPitch; p <= maxPitch; p += 12) {
    const y = Math.round(yOf(p)) + 0.5;
    ctx.strokeStyle = p === minPitch || p === maxPitch
      ? "rgba(0,0,0,0.22)"
      : "rgba(0,0,0,0.09)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(width - padR, y);
    ctx.stroke();
    ctx.fillStyle = "rgba(0,0,0,0.6)";
    ctx.fillText(`C${p / 12 - 1}`, padL - 6, y);
  }

  // vertical guide line + time label every niceTimeStep(duration) seconds
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const step = niceTimeStep(duration);
  for (let t = 0; t <= duration + 1e-6; t += step) {
    const x = Math.round(xOf(t)) + 0.5;
    ctx.strokeStyle = "rgba(0,0,0,0.07)";
    ctx.beginPath();
    ctx.moveTo(x, padT);
    ctx.lineTo(x, height - padB);
    ctx.stroke();
    ctx.fillStyle = "rgba(0,0,0,0.6)";
    ctx.fillText(`${Math.round(t)}s`, x, height - padB + 4);
  }

  // fixed, sane note thickness (independent of pitch-range span, which was the source of
  // the "incredibly thick" notes -- it used to scale with plotH / (maxPitch-minPitch))
  const noteH = 9;
  for (const sKey of Object.keys(notes)) {
    const sIdx = parseInt(sKey, 10) - 1;
    ctx.fillStyle = STRING_COLORS[sIdx];
    for (const n of notes[sKey]) {
      const x0 = xOf(n.start_s);
      const x1 = xOf(n.start_s + Math.max(n.dur_s, 0.02));
      const y = yOf(n.pitch) - noteH / 2;
      ctx.globalAlpha = 0.92;
      const r = 2;
      const w = Math.max(x1 - x0, 2);
      roundRect(ctx, x0, y, w, noteH, r);
      ctx.fill();
    }
  }
  ctx.globalAlpha = 1;

  drawPlayhead(ctx, xOf, height, playheadT, duration, padT, height - padB);
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawPlayhead(ctx, xOf, height, t, duration, yTop, yBottom) {
  if (t == null) return;
  const clamped = Math.max(0, Math.min(duration, t));
  const x = Math.round(xOf(clamped)) + 0.5;
  const y0 = yTop == null ? 0 : yTop;
  const y1 = yBottom == null ? height : yBottom;
  ctx.strokeStyle = "#1b1f27";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(x, y0);
  ctx.lineTo(x, y1);
  ctx.stroke();
}

// ---------------------------------------------------------------------------
// Playhead-canvas registry: every canvas that needs to redraw on each rAF tick
// registers a draw(playheadT) closure here, keyed by its own canvas element (dedup) and
// by a "container" used to bulk-evict stale entries when a container's canvases are
// thrown away and rebuilt from scratch (buildCurveGroup, buildSynthInputCurves, ...).
// MIDI rolls register here too, one per section, alongside every curve/comparison panel.
// ---------------------------------------------------------------------------

const playheadCanvases = [];

function registerPlayheadCanvas(canvas, container, drawFn) {
  for (let i = playheadCanvases.length - 1; i >= 0; i--) {
    if (playheadCanvases[i].canvas === canvas) playheadCanvases.splice(i, 1);
  }
  playheadCanvases.push({ canvas, container, draw: drawFn });
}

function clearContainerCanvases(container) {
  for (let i = playheadCanvases.length - 1; i >= 0; i--) {
    if (playheadCanvases[i].container === container) playheadCanvases.splice(i, 1);
  }
}

function redrawAllRegisteredCanvases(playheadT) {
  for (const entry of playheadCanvases) entry.draw(playheadT);
}

// ---------------------------------------------------------------------------
// Control-curve mini panels (Single String Study only, but kept as generic, duration-
// parameterized functions rather than closures, so they stay easy to reason about).
// ---------------------------------------------------------------------------

function _addCurveRow(container, labelText) {
  const row = document.createElement("div");
  row.className = "curve-row";
  const label = document.createElement("div");
  label.className = "curve-label";
  label.textContent = labelText;
  const canvas = document.createElement("canvas");
  row.appendChild(label);
  row.appendChild(canvas);
  container.appendChild(row);
  return canvas;
}

function buildCurveGroup(container, channels, dataByChannel, duration) {
  container.innerHTML = "";
  clearContainerCanvases(container);
  for (const name of channels) {
    const chan = dataByChannel[name];
    if (!chan) continue;
    const canvas = _addCurveRow(container, chan.label);
    const draw = (t) => drawCurvePanel(canvas, chan, name, duration, t);
    registerPlayheadCanvas(canvas, container, draw);
    draw(null);
  }
}

function _drawSeries(ctx, t, v, xOf, yOf) {
  ctx.beginPath();
  let drawing = false;
  for (let i = 0; i < t.length; i++) {
    if (v[i] == null) {
      drawing = false;
      continue;
    }
    const x = xOf(t[i]);
    const y = yOf(v[i]);
    if (!drawing) {
      ctx.moveTo(x, y);
      drawing = true;
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.stroke();
}

function _formatAxisValue(v) {
  const abs = Math.abs(v);
  if (abs >= 100) return Math.round(v).toString();
  if (abs >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

// Shared left margin reserved for y-axis tick labels, so every row in a curve-group
// (line curves, the voiced block, the onset/voiced strip) lines up on the same x origin
// even though only the numeric line curves actually draw tick labels into it.
const Y_AXIS_PAD_L = 44;

function _drawYAxisTicks(ctx, minV, maxV, yOf, padL, padR, width) {
  ctx.font = "10px -apple-system, BlinkMacSystemFont, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const val of [maxV, (minV + maxV) / 2, minV]) {
    const y = Math.round(yOf(val)) + 0.5;
    ctx.strokeStyle = "rgba(0,0,0,0.08)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(width - padR, y);
    ctx.stroke();
    ctx.fillStyle = "rgba(0,0,0,0.6)";
    ctx.fillText(_formatAxisValue(val), padL - 6, y);
  }
}

function drawCurvePanel(canvas, chan, name, duration, playheadT, overlayChan) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(rect.width, 200);
  const height = Math.max(rect.height, 40);
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const t = chan.t;
  const v = chan.v;
  const dur = duration || (t.length ? t[t.length - 1] : 1);
  const padT = 6;
  const padB = 6;
  const padL = Y_AXIS_PAD_L;
  const padR = 6;
  const plotH = height - padT - padB;
  const plotW = width - padL - padR;
  const xOf = (tt) => padL + (tt / dur) * plotW;

  if (name === "voiced") {
    // binary block, matches plot_ctrl_preview.py's _plot_block: a filled gray span
    // wherever the mask is truthy, nothing drawn otherwise.
    ctx.fillStyle = "rgba(0,0,0,0.12)";
    let runStart = null;
    for (let i = 0; i < v.length; i++) {
      const on = v[i] != null && v[i] > 0.5;
      if (on && runStart == null) runStart = t[i];
      if ((!on || i === v.length - 1) && runStart != null) {
        const end = t[i];
        ctx.fillRect(xOf(runStart), padT, Math.max(xOf(end) - xOf(runStart), 1), plotH);
        runStart = null;
      }
    }
  } else {
    let minV = Infinity;
    let maxV = -Infinity;
    const scan = (arr) => {
      for (const val of arr) {
        if (val == null) continue;
        minV = Math.min(minV, val);
        maxV = Math.max(maxV, val);
      }
    };
    scan(v);
    if (overlayChan) scan(overlayChan.v);
    if (!isFinite(minV)) {
      drawPlayhead(ctx, xOf, height, playheadT, dur);
      return;
    }
    if (minV === maxV) {
      minV -= 1;
      maxV += 1;
    }
    const margin = (maxV - minV) * 0.1;
    minV -= margin;
    maxV += margin;
    const yOf = (val) => padT + (1 - (val - minV) / (maxV - minV)) * plotH;

    _drawYAxisTicks(ctx, minV, maxV, yOf, padL, padR, width);

    if (overlayChan) {
      ctx.save();
      ctx.globalAlpha = 0.4;
      ctx.strokeStyle = "#5d6673";
      ctx.lineWidth = 1.3;
      ctx.setLineDash([3, 3]);
      _drawSeries(ctx, overlayChan.t, overlayChan.v, xOf, yOf);
      ctx.restore();
    }

    ctx.strokeStyle = TAB_COLORS[chan.color] || "#2f6fed";
    ctx.lineWidth = 1.6;
    _drawSeries(ctx, t, v, xOf, yOf);
  }

  drawPlayhead(ctx, xOf, height, playheadT, dur);
}

function drawOnsetVoicedPanel(canvas, voicedChan, onsetTimes, duration, playheadT) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(rect.width, 200);
  const height = Math.max(rect.height, 40);
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const dur = duration || 1;
  const padL = Y_AXIS_PAD_L;
  const padR = 6;
  const plotW = width - padL - padR;
  const xOf = (tt) => padL + (tt / dur) * plotW;

  // gray voiced blocks
  const t = voicedChan.t;
  const v = voicedChan.v;
  ctx.fillStyle = "rgba(0,0,0,0.12)";
  let runStart = null;
  for (let i = 0; i < v.length; i++) {
    const on = v[i] != null && v[i] > 0.5;
    if (on && runStart == null) runStart = t[i];
    if ((!on || i === v.length - 1) && runStart != null) {
      ctx.fillRect(xOf(runStart), 4, Math.max(xOf(t[i]) - xOf(runStart), 1), height - 8);
      runStart = null;
    }
  }

  // black onset ticks, drawn on top of the voiced blocks
  ctx.strokeStyle = "#1b1f27";
  ctx.lineWidth = 2;
  for (const ot of onsetTimes) {
    const x = Math.round(xOf(ot)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(x, 2);
    ctx.lineTo(x, height - 2);
    ctx.stroke();
  }

  drawPlayhead(ctx, xOf, height, playheadT, dur);
}

function buildSynthInputCurves(container, model, curves) {
  container.innerHTML = "";
  clearContainerCanvases(container);
  const synth = curves.synth_input[model];
  const overlay = model === "main_pipeline" ? curves.target_gt : null;
  const duration = curves.duration_s;

  for (const name of TARGET_CHANNELS) {
    const chan = synth[name];
    const canvas = _addCurveRow(container, chan.label);
    const overlayChan = overlay ? overlay[name] : null;
    const draw = (t) => drawCurvePanel(canvas, chan, name, duration, t, overlayChan);
    registerPlayheadCanvas(canvas, container, draw);
    draw(null);
  }

  if (model === "main_pipeline") {
    const canvas = _addCurveRow(container, "onset (tick) / voiced (block) \u2014 fed to synth");
    const draw = (t) => drawOnsetVoicedPanel(canvas, synth.voiced, synth.onset_t, duration, t);
    registerPlayheadCanvas(canvas, container, draw);
    draw(null);
  }
}

// ---------------------------------------------------------------------------
// Output comparison: pitch/envelope re-extracted from each system's own rendered audio.
// ---------------------------------------------------------------------------

function drawComparisonPanel(canvas, seriesMap, duration, playheadT, voicedMask) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(rect.width, 200);
  const height = Math.max(rect.height, 100);
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const dur = duration || 1;
  const padL = Y_AXIS_PAD_L;
  const padR = 6;
  const padT = 8;
  const padB = 8;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;
  const xOf = (tt) => padL + (tt / dur) * plotW;

  // faint gray background wherever the real recording actually has a note sounding
  // (ground-truth voiced mask), drawn first so every line stays fully legible on top.
  if (voicedMask) {
    ctx.fillStyle = "rgba(0,0,0,0.07)";
    let runStart = null;
    const vt = voicedMask.t;
    const vv = voicedMask.v;
    for (let i = 0; i < vv.length; i++) {
      const on = vv[i] != null && vv[i] > 0.5;
      if (on && runStart == null) runStart = vt[i];
      if ((!on || i === vv.length - 1) && runStart != null) {
        ctx.fillRect(xOf(runStart), padT, Math.max(xOf(vt[i]) - xOf(runStart), 1), plotH);
        runStart = null;
      }
    }
  }

  let minV = Infinity;
  let maxV = -Infinity;
  for (const arm of COMPARE_ORDER) {
    if (!compareVisible[arm]) continue;
    const s = seriesMap[arm];
    if (!s) continue;
    for (const val of s.v) {
      if (val == null) continue;
      minV = Math.min(minV, val);
      maxV = Math.max(maxV, val);
    }
  }
  if (!isFinite(minV)) {
    drawPlayhead(ctx, xOf, height, playheadT, dur, padT, height - padB);
    return;
  }
  if (minV === maxV) {
    minV -= 1;
    maxV += 1;
  }
  const margin = (maxV - minV) * 0.1;
  minV -= margin;
  maxV += margin;
  const yOf = (val) => padT + (1 - (val - minV) / (maxV - minV)) * plotH;

  _drawYAxisTicks(ctx, minV, maxV, yOf, padL, padR, width);

  for (const arm of COMPARE_ORDER) {
    if (!compareVisible[arm]) continue;
    const s = seriesMap[arm];
    if (!s) continue;
    ctx.strokeStyle = ARM_COMPARE_COLORS[arm];
    ctx.lineWidth = arm === "ground_truth" ? 1.3 : 1.6;
    ctx.globalAlpha = arm === "ground_truth" ? 0.7 : 0.95;
    _drawSeries(ctx, s.t, s.v, xOf, yOf);
  }
  ctx.globalAlpha = 1;

  drawPlayhead(ctx, xOf, height, playheadT, dur, padT, height - padB);
}

// Default comparison-plot visibility: ground truth plus whichever model tab is active, so
// switching models re-centers the comparison on "reference vs. this system" without the
// other two systems cluttering the initial view. Selecting ground-truth itself has no
// distinct "other system" to pair it with, so that case opens all four instead. Clicking
// a legend item still overrides this per the usual toggle behavior.
function defaultCompareVisible(modelId) {
  const vis = {};
  if (modelId === "ground_truth") {
    for (const arm of COMPARE_ORDER) vis[arm] = true;
    return vis;
  }
  for (const arm of COMPARE_ORDER) vis[arm] = false;
  vis.ground_truth = true;
  vis[modelId] = true;
  return vis;
}

// ---------------------------------------------------------------------------
// "3. Velocity Prediction": all 6 strings' predicted-vs-ground-truth note gain on one
// canvas, real per-frame time axis (not note index).
// ---------------------------------------------------------------------------

const VELOCITY_MODELS = [
  { id: "three_stage", label: "3 stage" },
  { id: "main_pipeline", label: "2 stage" },
  { id: "ground_truth", label: "ground-truth" },
];

function _stepValueAt(t, v, frameRate, time) {
  if (!t.length) return null;
  // Sample slightly after `time` rather than exactly at it: onset times come from MIDI,
  // frame times from a fixed grid, so an onset can land within half a frame of a frame
  // boundary -- without the nudge that can round down into the PREVIOUS note's held
  // frame and draw the wrong stem height.
  const idx = Math.min(t.length - 1, Math.max(0, Math.round((time - t[0]) * frameRate + 0.5)));
  return v[idx];
}

function _drawStepWithStems(ctx, t, v, onsets, duration, xOf, yOf, bottomY) {
  if (!t.length) return;
  const frameRate = t.length > 1 ? (t.length - 1) / (t[t.length - 1] - t[0]) : 1;
  const sorted = onsets && onsets.length ? [...onsets].sort((a, b) => a - b) : [t[0]];
  for (let i = 0; i < sorted.length; i++) {
    const start = Math.max(0, sorted[i]);
    if (start > duration) break;
    const end = i + 1 < sorted.length ? Math.min(sorted[i + 1], duration) : duration;
    const value = _stepValueAt(t, v, frameRate, start);
    if (value == null) continue;
    const y = yOf(value);
    const x0 = xOf(start);
    const x1 = xOf(end);
    // Vertical stem from the axis baseline up to this note's level, marking the onset.
    ctx.beginPath();
    ctx.moveTo(x0, bottomY);
    ctx.lineTo(x0, y);
    ctx.stroke();
    // This note's held value, deliberately NOT connected to the previous note's segment:
    // gain is a per-note step (see predict_gain_frames), not a continuously-varying
    // signal, so a line drawn between two different note levels would misrepresent it.
    ctx.beginPath();
    ctx.moveTo(x0, y);
    ctx.lineTo(x1, y);
    ctx.stroke();
  }
}

function drawVelocityPanel(canvas, stringsData, notesData, duration, playheadT, showPredicted, visibleStrings) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(rect.width, 200);
  const height = Math.max(rect.height, 160);
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const dur = duration || 1;
  const padL = Y_AXIS_PAD_L;
  const padR = 6;
  const padT = 8;
  const padB = 8;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;
  const xOf = (tt) => padL + (tt / dur) * plotW;
  const isVisible = (s) => !visibleStrings || visibleStrings[s] !== false;

  let minV = Infinity;
  let maxV = -Infinity;
  for (const s of Object.keys(stringsData)) {
    if (!isVisible(s)) continue;
    const entry = stringsData[s];
    const series = showPredicted ? [entry.predicted, entry.gt, entry.extracted] : [entry.gt];
    for (const ser of series) {
      for (const val of ser.v) {
        if (val == null) continue;
        minV = Math.min(minV, val);
        maxV = Math.max(maxV, val);
      }
    }
  }
  if (!isFinite(minV)) {
    drawPlayhead(ctx, xOf, height, playheadT, dur, padT, height - padB);
    return;
  }
  if (minV === maxV) {
    minV -= 1;
    maxV += 1;
  }
  const margin = (maxV - minV) * 0.1;
  minV -= margin;
  maxV += margin;
  const yOf = (val) => padT + (1 - (val - minV) / (maxV - minV)) * plotH;

  _drawYAxisTicks(ctx, minV, maxV, yOf, padL, padR, width);

  for (const s of Object.keys(stringsData)) {
    if (!isVisible(s)) continue;
    const sIdx = parseInt(s, 10) - 1;
    const color = STRING_COLORS[sIdx];
    const entry = stringsData[s];
    const onsets = (notesData && notesData[s] ? notesData[s] : []).map((n) => n.start_s);
    if (showPredicted) {
      // Layered back-to-front so the solid predicted line for this string stays fully
      // legible on top: faintest first (ground truth, for reference), then the
      // re-extracted-from-audio curve (did the synth actually reach that loudness?),
      // then the predicted curve itself, solid.
      ctx.save();
      ctx.globalAlpha = 0.4;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.2;
      ctx.setLineDash([3, 3]);
      _drawStepWithStems(ctx, entry.gt.t, entry.gt.v, onsets, dur, xOf, yOf, height - padB);
      ctx.restore();

      ctx.save();
      ctx.globalAlpha = 0.7;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.4;
      ctx.setLineDash([1, 2]);
      _drawStepWithStems(ctx, entry.extracted.t, entry.extracted.v, onsets, dur, xOf, yOf, height - padB);
      ctx.restore();

      ctx.strokeStyle = color;
      ctx.lineWidth = 1.7;
      _drawStepWithStems(ctx, entry.predicted.t, entry.predicted.v, onsets, dur, xOf, yOf, height - padB);
    } else {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.7;
      _drawStepWithStems(ctx, entry.gt.t, entry.gt.v, onsets, dur, xOf, yOf, height - padB);
    }
  }

  drawPlayhead(ctx, xOf, height, playheadT, dur, padT, height - padB);
}

// ---------------------------------------------------------------------------
// Model selector: one shared currentModel, built into every ".model-select-group" row
// found anywhere on the page (Full Mix Study's and Single String Study's each get their
// own row/DOM, but a click on either updates the same state and both rows' active pill).
// ---------------------------------------------------------------------------

const modelChangeListeners = [];
function onModelChange(fn) {
  modelChangeListeners.push(fn);
}

function setModel(modelId) {
  currentModel = modelId;
  for (const btn of document.querySelectorAll(".model-select-group .pill-btn")) {
    btn.classList.toggle("active", btn.dataset.model === modelId);
  }
  for (const fn of modelChangeListeners) fn(modelId);
}

function buildModelSelectorRows() {
  for (const container of document.querySelectorAll(".model-select-group")) {
    container.innerHTML = "";
    for (const m of MODELS) {
      const btn = document.createElement("button");
      btn.className = "pill-btn";
      btn.dataset.model = m.id;
      btn.textContent = m.label;
      btn.classList.toggle("active", m.id === currentModel);
      btn.addEventListener("click", () => setModel(m.id));
      container.appendChild(btn);
    }
  }
}

// ---------------------------------------------------------------------------
// Reusable per-section sample picker: each section owns its own <select>/meta/MIDI
// canvas/legend, decoupled from every other section's currently-selected sample.
// ---------------------------------------------------------------------------

function createSamplePicker(root) {
  const selectEl = root.querySelector(".sample-select");
  const metaEl = root.querySelector(".sample-meta");
  const midiCanvas = root.querySelector(".midi-canvas");
  const legendEl = root.querySelector(".midi-legend");
  const state = { sample: null, notes: null, duration: 0 };

  function populateOptions() {
    selectEl.innerHTML = "";
    for (const id of Object.keys(manifest.recordings)) {
      const rec = manifest.recordings[id];
      const opt = document.createElement("option");
      opt.value = id;
      const styleName = STYLE_NAMES[rec.style] || rec.style;
      const practice = rec.tier === "practice" ? " \u00b7 practice" : "";
      opt.textContent = `${id} \u2014 ${styleName}, ${rec.take}${practice}`;
      selectEl.appendChild(opt);
    }
  }

  function redrawMidi(t) {
    if (state.notes) drawMidiRoll(midiCanvas, state.notes, state.duration, t ?? null);
  }

  async function selectSample(id, onLoaded) {
    state.sample = id;
    selectEl.value = id;
    const rec = manifest.recordings[id];
    state.duration = rec.duration_s;

    await preserveScrollPositionAsync(async () => {
      const styleName = STYLE_NAMES[rec.style] || rec.style;
      metaEl.textContent =
        `${styleName} \u00b7 player ${rec.player} \u00b7 ${rec.take} \u00b7 ${rec.duration_s.toFixed(2)}s` +
        (rec.tier === "practice" ? " \u00b7 practice recording (unscored)" : "");

      state.notes = await fetchJSON(`data/${id}/notes.json`);
      buildStringLegend(legendEl, rec.available_strings);
      registerPlayheadCanvas(midiCanvas, midiCanvas, redrawMidi);
      redrawMidi(null);

      await onLoaded(rec);
    });
  }

  selectEl.addEventListener("change", () => selectSample(selectEl.value, onSelectHandler));
  let onSelectHandler = async () => {};

  return {
    state,
    populateOptions,
    redrawMidi,
    onSelect(fn) { onSelectHandler = fn; },
    selectSample: (id) => selectSample(id, onSelectHandler),
  };
}

// ---------------------------------------------------------------------------
// Entering any top-level section folds the other currently-open ones, so the reader's
// attention stays on whichever one they just opened -- not locked, they can still
// reopen any of them (multiple can be open simultaneously if the reader does that).
// ---------------------------------------------------------------------------

const SECTION_IDS = ["fullMixSection", "singleStringSection", "velocityPredictionSection", "longSampleSection"];
function foldOtherSections(exceptId) {
  for (const id of SECTION_IDS) {
    if (id === exceptId) continue;
    const details = el(id);
    if (details && details.open) details.open = false;
  }
}

// ---------------------------------------------------------------------------
// "1. Full Mix Study": Query hint + per-model mix audio, own sample/MIDI.
// ---------------------------------------------------------------------------

function createMixStudy(rootId) {
  const root = el(rootId);
  const picker = createSamplePicker(root);
  const queryAudio = root.querySelector(".query-audio");
  const modelAudio = root.querySelector(".model-audio");

  function refreshModelAudio() {
    if (!picker.state.sample) return;
    if (currentAudioEl === modelAudio) currentAudioEl.pause();
    modelAudio.src = `data/${picker.state.sample}/audio/${currentModel}.mp3`;
    modelAudio.load();
  }

  picker.onSelect(async () => {
    if (currentAudioEl === queryAudio) currentAudioEl.pause();
    queryAudio.src = `data/${picker.state.sample}/audio/query.mp3`;
    queryAudio.load();
    refreshModelAudio();
  });

  onModelChange(() => refreshModelAudio());

  root.addEventListener("toggle", () => {
    if (!root.open) return;
    preserveScrollPosition(() => {
      foldOtherSections("fullMixSection");
      picker.redrawMidi(lastPlayheadT);
    });
  });

  registerExclusive(queryAudio);
  registerExclusive(modelAudio);

  return { picker, loadSample: (id) => picker.selectSample(id) };
}

// ---------------------------------------------------------------------------
// "2. Single String Study": own sample/MIDI, own model selector row (synced state),
// string selector, string audio, predictor/synth input, and re-extracted comparison.
// ---------------------------------------------------------------------------

function createSingleStringStudy(rootId) {
  const root = el(rootId);
  const picker = createSamplePicker(root);
  const stringSelectorRow = root.querySelector("#stringSelectorRow");
  const stringAudio = root.querySelector("#stringAudio");
  const predSection = root.querySelector("#predictorInputSection");
  const inputGroup = root.querySelector("#inputCurves");
  const synthSection = root.querySelector("#synthInputSection");
  const synthTitle = root.querySelector("#synthInputTitle");
  const synthNote = root.querySelector("#synthInputNote");
  const synthGroup = root.querySelector("#synthInputCurves");
  const comparisonDetails = root.querySelector("#comparisonSection");
  const comparePitchCanvas = root.querySelector("#comparePitchCanvas");
  const compareEnvCanvas = root.querySelector("#compareEnvelopeCanvas");
  const compareLegendEl = root.querySelector("#compareLegend");

  const state = { string: 1, curves: null };

  function buildStringSelectorRow(strings) {
    stringSelectorRow.innerHTML = "";
    for (const s of strings) {
      const btn = document.createElement("button");
      btn.className = "pill-btn";
      btn.dataset.string = s;
      btn.title = `String ${s} (${STRING_NAMES[s - 1]})`;
      btn.textContent = String(s);
      btn.addEventListener("click", () => loadCurves(s));
      stringSelectorRow.appendChild(btn);
    }
  }

  function updateStringAudio() {
    if (!picker.state.sample) return;
    if (currentAudioEl === stringAudio) currentAudioEl.pause();
    stringAudio.src = `data/${picker.state.sample}/audio/strings/${currentModel}/string${state.string}.mp3`;
    stringAudio.load();
  }

  async function loadCurves(stringNum) {
    if (currentAudioEl === stringAudio) currentAudioEl.pause();
    state.string = stringNum;
    for (const btn of stringSelectorRow.querySelectorAll(".pill-btn")) {
      btn.classList.toggle("active", parseInt(btn.dataset.string, 10) === stringNum);
    }
    updateStringAudio();
    state.curves = await fetchJSON(`data/${picker.state.sample}/ctrl/string${stringNum}.json`);
    renderModelDependentCurves();
  }

  function renderModelDependentCurves() {
    if (!state.curves) return;
    const duration = state.curves.duration_s;

    preserveScrollPosition(() => {
      // Predictor input: only the 2-stage system's predictor is conditioned on this.
      if (currentModel === "main_pipeline") {
        predSection.hidden = false;
        buildCurveGroup(inputGroup, INPUT_CHANNELS, state.curves.inputs, duration);
      } else {
        predSection.hidden = true;
        inputGroup.innerHTML = "";
        clearContainerCanvases(inputGroup);
      }

      // Synthesizer input: differs per system, ddsp-guitar shows nothing at all.
      synthGroup.innerHTML = "";
      clearContainerCanvases(synthGroup);

      if (currentModel === "ddsp_guitar") {
        synthSection.hidden = true;
      } else {
        synthSection.hidden = false;
        if (currentModel === "main_pipeline") {
          synthTitle.textContent = "Synthesizer input";
          synthNote.hidden = false;
          synthNote.textContent = "Faint dashed line: ground-truth, not fed to the synth";
          buildSynthInputCurves(synthGroup, "main_pipeline", state.curves);
        } else if (currentModel === "velocity_joint_peak") {
          synthTitle.textContent = "Synthesizer input";
          synthNote.hidden = true;
          buildSynthInputCurves(synthGroup, "velocity_joint_peak", state.curves);
        } else if (currentModel === "ground_truth") {
          synthTitle.textContent = "Ground truth curve";
          synthNote.hidden = true;
          buildCurveGroup(synthGroup, TARGET_CHANNELS, state.curves.target_gt, duration);
        }
      }

      renderComparisonPlot();
    });
  }

  function buildCompareLegend() {
    compareLegendEl.innerHTML = "";
    for (const arm of COMPARE_ORDER) {
      const meta = MODELS.find((m) => m.id === arm);
      const item = document.createElement("button");
      item.type = "button";
      item.className = "legend-item";
      item.dataset.arm = arm;
      const swatch = document.createElement("span");
      swatch.className = "swatch";
      swatch.style.background = ARM_COMPARE_COLORS[arm];
      item.appendChild(swatch);
      item.appendChild(document.createTextNode(meta.label));
      item.title = "Click to show/hide this system's line";
      item.addEventListener("click", () => {
        compareVisible[arm] = !compareVisible[arm];
        item.classList.toggle("off", !compareVisible[arm]);
        redrawComparisonCanvases();
      });
      compareLegendEl.appendChild(item);
    }
  }

  function redrawComparisonCanvases() {
    for (const entry of playheadCanvases) {
      if (entry.canvas === comparePitchCanvas || entry.canvas === compareEnvCanvas) {
        entry.draw(lastPlayheadT);
      }
    }
  }

  function updateLegendButtonStates() {
    for (const btn of compareLegendEl.querySelectorAll(".legend-item")) {
      btn.classList.toggle("off", !compareVisible[btn.dataset.arm]);
    }
  }

  function applyDefaultCompareVisibility(modelId) {
    const defaults = defaultCompareVisible(modelId);
    for (const arm of COMPARE_ORDER) compareVisible[arm] = defaults[arm];
    updateLegendButtonStates();
  }

  function renderComparisonPlot() {
    const comp = state.curves.comparison;
    const voicedMask = state.curves.voiced_gt;
    const duration = state.curves.duration_s;

    const pitchSeries = {};
    const envSeries = {};
    for (const arm of COMPARE_ORDER) {
      pitchSeries[arm] = comp[arm].pitch;
      envSeries[arm] = comp[arm].envelope;
    }

    const drawPitch = (t) => drawComparisonPanel(comparePitchCanvas, pitchSeries, duration, t, voicedMask);
    const drawEnv = (t) => drawComparisonPanel(compareEnvCanvas, envSeries, duration, t, voicedMask);
    registerPlayheadCanvas(comparePitchCanvas, comparePitchCanvas, drawPitch);
    registerPlayheadCanvas(compareEnvCanvas, compareEnvCanvas, drawEnv);
    drawPitch(null);
    drawEnv(null);
  }

  picker.onSelect(async (rec) => {
    buildStringSelectorRow(rec.available_strings);
    const target = mostActiveString(rec.available_strings, picker.state.notes);
    await loadCurves(target);
  });

  onModelChange((modelId) => {
    updateStringAudio();
    applyDefaultCompareVisibility(modelId);
    renderModelDependentCurves();
  });

  // Collapsed <details> render their children at zero size, so canvases built while
  // folded need a fresh redraw once actually visible -- otherwise they stay stuck at the
  // fallback ~200px width from before the section was ever opened.
  for (const details of [predSection, synthSection, comparisonDetails]) {
    details.addEventListener("toggle", () => {
      if (details.open) renderModelDependentCurves();
    });
  }

  // Entering "Single String Study" folds the other sections out of the way so the
  // reader can focus on the string-level comparison -- not locked, they can still
  // reopen them. Also redraws: this section's own MIDI roll and any nested
  // predictor/synth/comparison section left open from a previous visit rendered at
  // fallback size while THIS outer section was collapsed (collapsed <details> content
  // has zero layout size regardless of a descendant's own open state).
  root.addEventListener("toggle", () => {
    if (!root.open) return;
    preserveScrollPosition(() => {
      foldOtherSections("singleStringSection");
      picker.redrawMidi(lastPlayheadT);
      renderModelDependentCurves();
    });
  });

  buildCompareLegend();
  registerExclusive(stringAudio);

  return { picker, loadSample: (id) => picker.selectSample(id) };
}

// ---------------------------------------------------------------------------
// "3. Velocity Prediction": own sample/MIDI, own 3-option model selector (NOT the shared
// currentModel -- ddsp-guitar/1-stage have no place here, and 3-stage doesn't exist in
// the other two sections), and one 6-string predicted-vs-ground-truth velocity plot.
// ---------------------------------------------------------------------------

function createVelocityStudy(rootId) {
  const root = el(rootId);
  const picker = createSamplePicker(root);
  const queryAudio = root.querySelector(".query-audio");
  const modelAudio = root.querySelector("#velocityModelAudio");
  const modelRow = root.querySelector("#velocityModelSelectorRow");
  const stringSelectorRow = root.querySelector("#velocityStringSelectorRow");
  const stringAudio = root.querySelector("#velocityStringAudio");
  const canvas = root.querySelector("#velocityCanvas");
  const legendEl = root.querySelector("#velocityLegend");
  const noteEl = root.querySelector("#velocityNote");

  const state = {
    model: VELOCITY_MODELS[0].id,
    velocity: null,
    string: null,
    stringVisible: { 1: true, 2: true, 3: true, 4: true, 5: true, 6: true },
  };

  function buildModelRow() {
    modelRow.innerHTML = "";
    for (const m of VELOCITY_MODELS) {
      const btn = document.createElement("button");
      btn.className = "pill-btn";
      btn.dataset.model = m.id;
      btn.textContent = m.label;
      btn.classList.toggle("active", m.id === state.model);
      btn.addEventListener("click", () => setLocalModel(m.id));
      modelRow.appendChild(btn);
    }
  }

  function refreshModelAudio() {
    if (!picker.state.sample) return;
    if (currentAudioEl === modelAudio) currentAudioEl.pause();
    modelAudio.src = `data/${picker.state.sample}/audio/${state.model}.mp3`;
    modelAudio.load();
  }

  function refreshStringAudio() {
    if (!picker.state.sample || !state.string) return;
    if (currentAudioEl === stringAudio) currentAudioEl.pause();
    stringAudio.src = `data/${picker.state.sample}/audio/strings/${state.model}/string${state.string}.mp3`;
    stringAudio.load();
  }

  function buildStringSelectorRow(strings) {
    stringSelectorRow.innerHTML = "";
    for (const s of strings) {
      const btn = document.createElement("button");
      btn.className = "pill-btn";
      btn.dataset.string = s;
      btn.title = `String ${s} (${STRING_NAMES[s - 1]})`;
      btn.textContent = String(s);
      btn.classList.toggle("active", s === state.string);
      btn.addEventListener("click", () => selectString(s));
      stringSelectorRow.appendChild(btn);
    }
  }

  // Selecting a string both drives the single-string audio player AND becomes the sole
  // default-visible line on the velocity comparison plot below -- the other 5 strings
  // stay togglable via the legend, just not shown until the reader asks for them.
  function selectString(stringNum) {
    state.string = stringNum;
    for (const btn of stringSelectorRow.querySelectorAll(".pill-btn")) {
      btn.classList.toggle("active", parseInt(btn.dataset.string, 10) === stringNum);
    }
    for (const s of [1, 2, 3, 4, 5, 6]) state.stringVisible[s] = s === stringNum;
    updateVelocityLegendState();
    refreshStringAudio();
    renderVelocityPlot();
  }

  function updateVelocityLegendState() {
    for (const btn of legendEl.querySelectorAll(".legend-item")) {
      btn.classList.toggle("off", !state.stringVisible[btn.dataset.string]);
    }
  }

  function buildVelocityLegend() {
    legendEl.innerHTML = "";
    for (const s of [1, 2, 3, 4, 5, 6]) {
      const key = String(s);
      const item = document.createElement("button");
      item.type = "button";
      item.className = "legend-item";
      item.dataset.string = key;
      const swatch = document.createElement("span");
      swatch.className = "swatch";
      swatch.style.background = STRING_COLORS[s - 1];
      item.appendChild(swatch);
      item.appendChild(document.createTextNode(`String ${s} (${STRING_NAMES[s - 1]})`));
      item.title = "Click to view this string alone";
      item.addEventListener("click", () => selectString(s));
      legendEl.appendChild(item);
    }
    updateVelocityLegendState();
  }

  function renderVelocityPlot() {
    if (!state.velocity) return;
    const showPredicted = state.model === "three_stage";
    noteEl.textContent = showPredicted
      ? "Solid: predicted, from the score-only note-gain predictor. Dotted: re-extracted " +
        "from the synth's own rendered audio (did it actually reach that loudness?). " +
        "Faint dashed: ground truth, for reference."
      : "Ground truth per-note gain -- this system has no distinct predicted-velocity " +
        "curve of its own.";
    const duration = state.velocity.duration_s;
    const draw = (t) =>
      drawVelocityPanel(canvas, state.velocity.strings, picker.state.notes, duration, t, showPredicted, state.stringVisible);
    registerPlayheadCanvas(canvas, canvas, draw);
    draw(null);
  }

  function setLocalModel(modelId) {
    state.model = modelId;
    for (const btn of modelRow.querySelectorAll(".pill-btn")) {
      btn.classList.toggle("active", btn.dataset.model === modelId);
    }
    refreshModelAudio();
    refreshStringAudio();
    renderVelocityPlot();
  }

  picker.onSelect(async (rec) => {
    if (currentAudioEl === queryAudio) currentAudioEl.pause();
    queryAudio.src = `data/${picker.state.sample}/audio/query.mp3`;
    queryAudio.load();
    refreshModelAudio();
    state.velocity = await fetchJSON(`data/${picker.state.sample}/velocity.json`);
    buildStringSelectorRow(rec.available_strings);
    selectString(mostActiveString(rec.available_strings, picker.state.notes));
  });

  root.addEventListener("toggle", () => {
    if (!root.open) return;
    preserveScrollPosition(() => {
      foldOtherSections("velocityPredictionSection");
      picker.redrawMidi(lastPlayheadT);
      renderVelocityPlot();
    });
  });

  buildModelRow();
  buildVelocityLegend();
  registerExclusive(queryAudio);
  registerExclusive(modelAudio);
  registerExclusive(stringAudio);

  return { picker, loadSample: (id) => picker.selectSample(id) };
}

// ---------------------------------------------------------------------------
// "4. Unseen Long Audio Inference": four whole GAPS-dataset renders, deliberately no
// sample picker / MIDI roll / curves -- just fixed audio players, per this section's own
// request. Static content driven off manifest.long_samples (see
// scripts/pipeline/export_demo_assets.py's export_long_samples).
// ---------------------------------------------------------------------------

const LONG_SAMPLE_DETAIL_PARAGRAPHS = [
  "These four pieces are real GuitarPro/MusicXML scores from the GAPS dataset (004_JSswc, " +
    "008_PSswc, 009_xSswc, 010_QSswc) \u2014 material none of the three checkpoints below saw " +
    "during training, with no recording, no annotations, and no ground-truth curves attached. " +
    "Each one is rendered end-to-end by the same frozen note-gain predictor, pitch/envelope " +
    "predictor, and synth decoder used everywhere else on this page.",
  "Real GAPS pieces run 60-300s, far past what any of the three models were trained on " +
    "(GuitarSet recordings top out around 45s, and the middle model only ever saw random " +
    "6-12s crops), so each stage handles long inputs differently. Velocity prediction is " +
    "fed the whole score's note sequence in one shot \u2014 it operates per note, not per " +
    "time-window, so there is nothing to chunk. Performance (pitch/envelope) prediction " +
    "splits the score into chunks cut only at note boundaries, carries about a second of " +
    "already-decided context into the next chunk, and prefixes every chunk after the first " +
    "with 0.5s of silence \u2014 so each chunk looks like the start of a recording to a model " +
    "that was only ever validated starting from one. Synthesis also renders in overlapping " +
    "windows, but every window is sliced from the same global random noise rather than " +
    "restarting from independent noise, and overlapping regions are blended by averaging the " +
    "latents themselves before decoding, so the six-string audio stays continuous across " +
    "window boundaries instead of seaming or drifting.",
];

function createLongSampleSection(rootId) {
  const root = el(rootId);
  const listEl = root.querySelector("#longSampleList");
  const detailEl = root.querySelector("#longSampleDetailText");

  function build() {
    listEl.innerHTML = "";
    for (const sample of manifest.long_samples || []) {
      const block = document.createElement("div");
      block.className = "audio-block";
      const label = document.createElement("div");
      label.className = "audio-block-label";
      const strong = document.createElement("strong");
      strong.textContent = sample.label;
      const hint = document.createElement("span");
      hint.className = "hint-text";
      hint.textContent = `${sample.duration_s.toFixed(1)}s \u00b7 ${sample.source_gp}`;
      label.appendChild(strong);
      label.appendChild(hint);
      const audio = document.createElement("audio");
      audio.controls = true;
      audio.preload = "none";
      audio.src = `data/long/${sample.id}.mp3`;
      registerExclusive(audio);
      block.appendChild(label);
      block.appendChild(audio);
      listEl.appendChild(block);
    }

    detailEl.innerHTML = "";
    for (const text of LONG_SAMPLE_DETAIL_PARAGRAPHS) {
      const p = document.createElement("p");
      p.textContent = text;
      detailEl.appendChild(p);
    }
  }

  root.addEventListener("toggle", () => {
    if (!root.open) return;
    preserveScrollPosition(() => foldOtherSections("longSampleSection"));
  });

  build();
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

async function main() {
  await loadManifest();

  const mixStudy = createMixStudy("fullMixSection");
  const singleStringStudy = createSingleStringStudy("singleStringSection");
  const velocityStudy = createVelocityStudy("velocityPredictionSection");
  createLongSampleSection("longSampleSection");

  buildModelSelectorRows();

  mixStudy.picker.populateOptions();
  singleStringStudy.picker.populateOptions();
  velocityStudy.picker.populateOptions();

  window.addEventListener("resize", () => {
    const t = currentAudioEl && !currentAudioEl.paused ? currentAudioEl.currentTime : lastPlayheadT;
    redrawAllRegisteredCanvases(t);
  });

  const firstSample = Object.keys(manifest.recordings)[0];
  await mixStudy.loadSample(firstSample);
  await singleStringStudy.loadSample(firstSample);
  await velocityStudy.loadSample(firstSample);
}

main();

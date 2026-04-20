"""
IOTA FRAMEWORK — LIVE DASHBOARD  v0.79.4.9
=======================================
Full run-control web UI + live monitoring.

  python export_flask.py      → open http://localhost:5000

What's here:
  - SSE live console + sparklines + progress bars
  - Pause / Resume  (writes .iota_pause flag between turns)
  - Abort           (terminates subprocess)
  - Launch any preset or custom run set from the browser
  - Run-status grid: 45 cells with phase grouping labels
  - Hypothesis scoreboard: 35+1 chips with status/implication
  - Settings tab: model picker + all parameters + advanced options
  - What-Next recommendation with clickable run numbers
  - Dependency map
  - Keyboard shortcuts: Space=pause  R=run  Esc=abort  1-6=tabs  Ctrl+L/K
  - Console search filter + severity filter buttons + copy-to-clipboard
  - Live metric colourisation (similarity threshold highlighting)
  - Elapsed run timer in toolbar (survives page reload — FIX-7)
  - Run-26 blocking banner with one-click launch
  - Auto-reconnect on server restart

v37.7 changes (critical JS SyntaxError fix + scan n_trials fix):
  - BUG-37B [CRITICAL]: Two addL() calls embedded a literal newline character inside
             JS single-quoted string literals. Python processes \n in triple-quoted
             strings as ASCII 0x0A — the resulting HTML had `text:'<NEWLINE>...'`
             which Firefox/Chrome reject with SyntaxError: '' string literal contains
             an unescaped line break. Because JS aborts the entire <script> block on
             any SyntaxError, ALL dashboard functions (doLaunch, sw, clrC, cpyC,
             loadH, selP, togAS, setSev, doPause, doAbort) threw ReferenceError on
             every click. Fixed: \\n in Python source renders as \n (two chars) in
             the served JS, which is the valid JS newline escape. Lines 1184, 1421.
  - BUG-37C [MED]: _scan() API route called _scan_runs(paths) without n_trials.
             Always used n_trials=100 (default) regardless of session setting.
             If session.trials=50, all complete runs showed 'partial' in the grid.
             Fixed: _scan_runs(paths, s.get('trials', 100)).
  - Version string updated to v37.7 in toolbar, docstring, and CHANGELOG.
v37.2 changes (UI label pass + prereq-banner syntax fix):
  - BUG-37A [CRITICAL]: rmPrereqBanner() named helper (fixes prereq-banner Cancel).
  - WCAG-1: outline:none removed from .si:focus and .csf:focus — keyboard focus
             now visible via 2px solid accent ring on all interactive elements
  - WCAG-2: --t3 (#3a5060) on --bg (#0a0c0f) = 1.77:1 contrast ratio, fails WCAG AA.
             All text-bearing label classes updated to var(--t2) (4.75:1).
             --t3 retained only for decorative non-text elements (borders, dots, chrome)
  - WCAG-3: 8px font-size on meaningful labels below readable minimum.
             Section headers, form labels, phase labels, console title → 10px.
             Chip labels, run grid cells, sev-btn, legend → 9px.
  - WCAG-4: Run grid cells used color as sole status indicator.
             title + aria-label now include explicit status text per cell.
             Hypothesis chip symbols (✓/✗/?/·) already present; chip.pen contrast fixed.
  - WCAG-5: Zero ARIA on interactive elements. Added: role="tab"+aria-selected on tabs;
             role="button"+tabindex+keyboard on chips and run-grid cells;
             role="log"+aria-live on console; aria-pressed on severity buttons;
             aria-label on toolbar buttons, all inputs and selects.
  - WCAG-6: Form inputs had decorative .slb div labels only (not <label> elements).
             aria-label added to every input and select element.
  - WCAG-7: Zero media queries. @media (max-width:900px) added: panels stack vertically.
  - BUG-35A [MED]: Runs 43+44 absent from PHASES array, ALL array, run grid — silently
             unlaunchable from web UI. Phase 4 row added; ALL updated.
  - BUG-35B [MED]: H33/H34/H35 absent from H_PHASES — scoreboard showed no Phase 4 chips.
             'Phase 4' entry added with H33/H34/H35.
  - Version string in toolbar updated to v35.2.
v34.2 changes (post-release bug fixes — see CHANGELOG):
  - BUG-34D: removed erroneous _hypLoaded=true from doStats polling
             (suppressed Hyp tab auto-load on first visit as side-effect)
  Previously in v34.1:
  - BUG-34A: poll double-increment fixed (60s delay was skipped: 30s→120s→stop)
  - BUG-34B: fig_confound max(vals) NaN order-dependency fixed (np.nanmax)
  - BUG-34C: misleading _write_status comment corrected
  Previously in v34.0:
  - FIX-2: H17 metric field corrected to delta_r2_internal_per_condition
  - FIX-3: fig_confound rebuilt as dual-panel (primary: ΔR², secondary: state_similarity_index)
  - FIX-5: doStats() scoreboard refresh replaced with 30/60/120s polling loop
  - FIX-6: H25 inconclusive arm now requires diff < -0.05 (magnitude guard added)
  - FIX-7: startElapsed() accepts optional startTs; applyS passes run_start_ts from
           status JSON so elapsed timer survives browser reload during active runs
  - run_start_ts written to .iota_status.json via orchestration_core (_write_status)
  - Version bump; CHANGELOG v33.3 header restored (FIX-4)
"""

import os, sys, json, re, time, threading, subprocess
from pathlib import Path

def _find_root():
    candidate = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        if os.path.exists(os.path.join(candidate, "start_here.py")):
            return candidate
        parent = os.path.dirname(candidate)
        if parent == candidate:
            break
        candidate = parent
    return os.path.dirname(os.path.abspath(__file__))

ROOT         = _find_root()
STATUS_FILE  = os.path.join(ROOT, ".iota_status.json")
PAUSE_FILE   = os.path.join(ROOT, ".iota_pause")
LOG_FILE     = os.path.join(ROOT, ".iota_log.jsonl")
SESSION_FILE = os.path.join(ROOT, "last_session.json")
PID_FILE     = os.path.join(ROOT, ".iota_pid")
sys.path.insert(0, ROOT)

from flask import Flask, Response, request, jsonify
app = Flask(__name__)

# v0.78.1.0 BUG-STALE-GRID fix: force no-store on every response.
# jsonify() returns with no cache headers, so browsers apply HTTP/1.1
# heuristic freshness and return stale JSON on model switches — causing
# /models_collected, /temp_grid, /scan to show the previous model's data
# until Flask reboots. The reboot "fix" only worked because the dropped
# TCP connection + page reload invalidated the heuristic cache; the
# underlying cause was browser-side storage of untagged JSON.
# no-store (not just no-cache) forbids the browser from storing the
# response at all — correct policy for a live dashboard where the
# underlying data changes constantly underneath.
# SSE stream at /stream already sets no-cache; this hook overwrites with
# no-store which is strictly stricter and works fine for EventSource.
@app.after_request
def _no_browser_cache(resp):
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

# v0.76.0.4: set active quant from session on Flask startup
try:
    from cartography import set_active_quant as _saq
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE) as _sf: _sq = json.load(_sf).get('quantization', '4bit')
    else: _sq = '4bit'
    _saq(_sq)
except Exception: pass

_proc  = None
_plock = threading.Lock()
_pruns = None
_queued_runs = None   # run spec staged to fire automatically when current session ends
_session_start_ts = None  # timestamp when current collection session was started
_SERVER_START = time.time()  # when Flask process started — used for idle clock
_active_total = 0.0  # accumulated seconds where a run (data or analysis) was active
_last_active_start = None  # timestamp when current run started


def _log_separator(label=""):
    """Append a separator line with datetime stamp to the log."""
    try:
        import json as _j, time as _t
        _dt = _t.strftime('%b %d, %H:%M')
        entry = _j.dumps({"text": f"  ━━ {label} — {_dt} ━━", "kind": "sep", "ts": _t.time()})
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(entry + "\n")
    except Exception:
        pass

# ── Startup cleanup (v39.0.10) ─────────────────────────────────────────────────
# On server start no subprocess can be running (fresh process). If a prior
# session left .iota_status.json with run_num/trial/etc still set, every
# subsequent SSE heartbeat would read those stale values and the dashboard
# would show a ghost "now" highlight on the last-run cell permanently.
# Fix: strip run-state fields at import time so the file is clean on first load.
_STALE_FIELDS = ("run_num","run_label","trial","turn","max_turn",
                 "sim_index","sim","power","disrupt","run_index","run_start_ts")

def _clear_stale_pid():
    """At server startup no subprocess can be running. If a PID file exists
    from a prior session, check if that process is actually alive. If not,
    delete the file — it is stale and would block all future launches."""
    try:
        if not os.path.exists(PID_FILE):
            return
        with open(PID_FILE) as f:
            pid = int(f.read().strip())
        if not _pid_alive(pid):
            os.remove(PID_FILE)
    except Exception:
        try: os.remove(PID_FILE)
        except Exception: pass

def _clear_stale_status():
    try:
        if not os.path.exists(STATUS_FILE):
            return
        with open(STATUS_FILE, encoding='utf-8') as f:
            d = json.load(f)
        changed = False
        for k in _STALE_FIELDS:
            if k in d:
                del d[k]
                changed = True
        if changed:
            with open(STATUS_FILE, 'w') as f:
                json.dump(d, f)
    except Exception:
        pass

_clear_stale_pid()
_clear_stale_status()
# Clear stale pause file — if Flask restarted (reboot/crash) no subprocess is
# running so a leftover .iota_pause would show dashboard as paused permanently.
try:
    if os.path.exists(PAUSE_FILE): os.remove(PAUSE_FILE)
except Exception:
    pass
# ── End startup cleanup ───────────────────────────────────────────────────────

_DS = {
    "model_path":"failspy/Meta-Llama-3-8B-Instruct-abliterated-v3",
    "model_name":"Llama-3-8B-Abliterated","model_family":"llama",
    "model_size":"8b","model_variant":"abliterated","quantization":"4bit",
    "temperature":0.0,"trials":100,"seed":42,"runs":"1-45",
    # v37.1: updated from "1-39,41,42" — missing runs 0051, 0025, 0026
}

def _rst():
    try:
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE,encoding='utf-8') as f: return json.load(f)
    except: pass
    return {}

def _rsess():
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE) as f: s=json.load(f)
            return {**_DS,**s}
        except: pass
    return dict(_DS)

def _wsess(upd):
    import datetime
    from cartography import set_active_quant
    s=_rsess()
    ok={"trials","temperature","seed","runs","model_path","model_name",
        "model_family","model_size","model_variant","quantization",
        "clear_cache","force_gc","unload_between_runs","hf_token",
        "r19_passes","et_mode_overrides",
        "patch_modes_17","patch_modes_18","patch_modes_19","r3_cells","mc_conds",
        "restore_runs","low_priority","active_temps","cross_model_include"}
    for k,v in upd.items():
        if k in ok: s[k]=v
    set_active_quant(s.get('quantization', '4bit'))
    s['_saved']=datetime.datetime.now().isoformat()
    import tempfile as _tf
    _fd, _tmp = _tf.mkstemp(dir=os.path.dirname(os.path.abspath(SESSION_FILE)), suffix='.tmp')
    try:
        with os.fdopen(_fd, 'w') as _f: json.dump(s, _f, indent=2)
        os.replace(_tmp, SESSION_FILE)
    except Exception:
        try: os.unlink(_tmp)
        except Exception: pass
        raise
    return s

def _pid_alive(pid):
    """Return True if a process with this PID is currently running."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except Exception:
        return False

def _write_pid(pid):
    try:
        with open(PID_FILE, 'w') as f: f.write(str(pid))
    except Exception: pass

def _clear_pid():
    try:
        if os.path.exists(PID_FILE): os.remove(PID_FILE)
    except Exception: pass

def _running():
    global _proc, _active_total, _last_active_start
    with _plock:
        if _proc is not None:
            if _proc.poll() is not None:
                # Process just ended — accumulate active time
                if _last_active_start is not None:
                    _active_total += time.time() - _last_active_start
                    _last_active_start = None
                _proc = None
                _clear_pid()
                _clear_stale_status()   # BUG-STATUS-GHOST fix (v45.0.7): subprocess
                                        # died but status file still has run_num/trial
                                        # set from the last _write_status call. JS uses
                                        # run_num&&!PAU as a secondary "running" signal
                                        # so the dashboard stays showing "running" even
                                        # after web_running=False. Clear at death point.
                return False
            return True
        # _proc is None — Flask may have restarted while a subprocess was live.
        # Check PID file to catch that case and block duplicate launches.
        try:
            if os.path.exists(PID_FILE):
                with open(PID_FILE) as f: pid = int(f.read().strip())
                if _pid_alive(pid):
                    return True
                _clear_pid()
                _clear_stale_status()   # Same fix for PID-file path
        except Exception:
            _clear_pid()
        return False

def _fire_queued_runs():
    """Called from SSE gen() when session ends and _queued_runs is set.
    Clears the queue and launches the staged runs in a daemon thread.
    Guards against firing while another session is already active.
    """
    global _queued_runs
    if _running():
        # Session still active — do not fire. SSE will retry on next tick.
        return
    qr = _queued_runs
    _queued_runs = None
    if not qr:
        return
    def _autofire():
        import time as _t
        _t.sleep(2)  # Brief pause so SSE sends the cleared-queue state first
        if _running():
            # Another session started in the 2s window — abort auto-fire
            print(f"  [queue] auto-fire aborted — session already active", flush=True)
            return
        _wsess({"runs": qr})
        log = os.path.join(ROOT, ".iota_flask.log")
        _log_separator("queue-autofire: " + qr)
        try:
            lf = open(log, 'a')
            try:
                if qr.startswith('all-temps:'):
                    # Analysis runs at all temperatures
                    _atr = qr.split(':', 1)[1]
                    p = subprocess.Popen(
                        [sys.executable, os.path.join(ROOT, "start_here.py"),
                         "--all-temps-runs", _atr],
                        stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
                else:
                    p = subprocess.Popen(
                        [sys.executable, os.path.join(ROOT, "start_here.py"),
                         "--headless", "--auto-run", qr],
                        stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
            finally:
                lf.close()
            global _proc, _pruns, _last_active_start
            with _plock:
                _proc = p
                _pruns = qr
                _last_active_start = time.time()
            _write_pid(p.pid)
        except Exception as e:
            print(f"  [queue] Auto-fire failed: {e}", flush=True)
    threading.Thread(target=_autofire, daemon=True).start()

def _scan():
    try:
        from cartography import get_paths
        s=_rsess()
        paths=get_paths(s.get('model_family','llama'),s.get('model_size','8b'),
                        s.get('model_variant','abliterated'),s.get('temperature',0.0),
                        create_dirs=False)
        import start_here as sh
        return sh._scan_runs(paths, s.get('trials', 100))
    except: return {}

def _descs():
    try:
        import start_here as sh
        return {k:v[1] for k,v in sh.RUN_MAP.items()}
    except: return {}

def _whatnext():
    try:
        import io,contextlib
        from cartography import get_paths
        s=_rsess()
        paths=get_paths(s.get('model_family','llama'),s.get('model_size','8b'),
                        s.get('model_variant','abliterated'),s.get('temperature',0.0),
                        create_dirs=False)
        import start_here as sh
        buf=io.StringIO()
        with contextlib.redirect_stdout(buf): sh._what_next(paths)
        return buf.getvalue()
    except Exception as e: return f"(error: {e})"

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/stream")
def stream():
    def gen():
        last=0
        ticks=0
        # SSE-FIX (v37.1): original gen() was edge-triggered on STATUS_FILE mtime only.
        # After a run completes, the file stops changing — last SSE message had
        # web_running=True, applyS() never ran with web_running=False, and lBtn stayed
        # disabled until page reload. Fix: emit a heartbeat every 10s unconditionally
        # (ticks%20 at 0.1s sleep = 2s) so applyS() always reflects current state.
        # v39.0.10: sleep reduced 0.5s→0.1s so status changes appear within 100ms.
        while True:
            try:
                mt=os.path.getmtime(STATUS_FILE) if os.path.exists(STATUS_FILE) else 0
                if mt!=last or ticks%20==0:
                    last=mt
                    st=_rst()
                    st["paused"]=os.path.exists(PAUSE_FILE)
                    st["web_running"]=_running()
                    st["web_runs"]=_pruns or ""
                    st["queued_runs"]=_queued_runs or ""
                    st["session_start_ts"]=_session_start_ts
                    st["temperature"]=st.get('temperature') or _rsess().get('temperature', 0.0)
                    # Auto-fire queued runs when session ends
                    if not st["web_running"] and _queued_runs:
                        _fire_queued_runs()
                    # Sanitize NaN/Infinity — valid in Python json.dumps
                    # but invalid in browser JSON.parse
                    _raw = json.dumps(st)
                    _raw = _raw.replace('NaN', 'null').replace('Infinity', 'null').replace('-Infinity', 'null')
                    yield f"data: {_raw}\n\n"
            except: pass
            ticks+=1
            time.sleep(0.1)
    return Response(gen(),mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.route("/log")
def log_ep():
    """Serve log lines efficiently from .iota_log.jsonl.

    v0.65.1.2: rewrote to use sparse line-offset index. Previous version did
    f.readlines() on every request — O(file_size) memory and time. On a 1GB log
    file that's ~10M lines loaded into a Python list every 2 seconds (poll rate).

    Now: index is built once on first request (sequential read, stores byte offset
    of every 1000th line — ~10KB for 10M lines). Updated incrementally on subsequent
    requests (seeks to last known position, reads only new bytes). All three access
    patterns (since/tail/before) seek directly to the relevant position.

    Memory: O(total_lines / 1000) for index + O(chunk_size) for returned lines.
    Time: O(chunk_size) per request after initial index build.
    """
    since=request.args.get('since',None)
    tail =request.args.get('tail', None)
    before=request.args.get('before',None)
    if since is not None:
        try: since=int(since)
        except: since=None
    if tail is not None:
        try: tail=int(tail)
        except: tail=None
    if before is not None:
        try: before=int(before)
        except: before=None
    try:
        _log_index_ready.wait(timeout=2)  # wait for bg index build, max 2s
        _update_log_index()
        total = _log_total

        if since is not None:
            if since >= total:
                return jsonify({"lines":[],"total":total})
            lines = _read_log_lines_from(since, total)
            return jsonify({"lines":lines,"total":total})

        if tail is not None:
            start = max(0, total - tail)
            lines = _read_log_lines_from(start, total)
            return jsonify({"lines":lines,"total":total})

        if before is not None:
            CHUNK = 200
            start = max(0, before - CHUNK)
            lines = _read_log_lines_from(start, min(before, total))
            return jsonify({"lines":lines,"start":start,"total":total})

    except Exception:
        pass
    return jsonify({"lines":[],"total":0})


# ── Log index (sparse, incremental) ──────────────────────────────────────────
_LOG_INDEX_STEP = 1000
_log_sparse_offsets = []   # byte offset of every 1000th line
_log_file_size = 0
_log_total = 0
_log_lock = threading.Lock()

def _update_log_index():
    """Build or incrementally update the sparse line-offset index.
    Thread-safe: concurrent requests queue on the lock instead of racing.
    """
    global _log_sparse_offsets, _log_file_size, _log_total
    with _log_lock:
        if not os.path.exists(LOG_FILE):
            _log_sparse_offsets = []
            _log_file_size = 0
            _log_total = 0
            return
        current_size = os.path.getsize(LOG_FILE)
        if current_size == _log_file_size:
            return  # no change
        if current_size < _log_file_size:
            # File shrank (manual truncation or rotation) — rebuild
            _log_sparse_offsets = []
            _log_file_size = 0
            _log_total = 0
        with open(LOG_FILE, 'rb') as f:
            f.seek(_log_file_size)
            line_num = _log_total
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line_num % _LOG_INDEX_STEP == 0:
                    _log_sparse_offsets.append(offset)
                line_num += 1
            _log_file_size = f.tell()
            _log_total = line_num

# Build index in background so Flask startup is not blocked by large log files.
# First /log request waits up to 2s for index to complete; returns partial if not ready.
_log_index_ready = threading.Event()
def _bg_log_index():
    _update_log_index()
    _log_index_ready.set()
threading.Thread(target=_bg_log_index, daemon=True).start()


def _seek_to_line(f, line_num):
    """Seek file to the start of line_num using the sparse index."""
    idx = line_num // _LOG_INDEX_STEP
    if idx >= len(_log_sparse_offsets):
        idx = max(0, len(_log_sparse_offsets) - 1)
    if not _log_sparse_offsets:
        f.seek(0)
        skip = line_num
    else:
        f.seek(_log_sparse_offsets[idx])
        skip = line_num - idx * _LOG_INDEX_STEP
    for _ in range(skip):
        f.readline()


def _read_log_lines_from(start, end):
    """Read parsed JSON log lines from line start to line end (exclusive)."""
    if start >= end or not os.path.exists(LOG_FILE):
        return []
    lines = []
    with open(LOG_FILE, 'rb') as f:
        _seek_to_line(f, start)
        count = 0
        limit = end - start
        for raw in f:
            if count >= limit:
                break
            raw = raw.strip()
            if raw:
                try:
                    lines.append(json.loads(raw.decode('utf-8', errors='replace')))
                except Exception:
                    pass
            count += 1
    return lines

@app.route("/pause",methods=["POST"])
def pause_ep():
    if os.path.exists(PAUSE_FILE): os.remove(PAUSE_FILE); return jsonify({"paused":False})
    Path(PAUSE_FILE).touch(); return jsonify({"paused":True})

@app.route("/queue",methods=["GET"])
def queue_get():
    return jsonify({"queued_runs": _queued_runs or ""})

@app.route("/queue",methods=["POST"])
def queue_post():
    global _queued_runs
    d = request.get_json(force=True, silent=True) or {}
    runs = str(d.get("runs","")).strip()
    clear = d.get("clear", False)
    if not runs or clear:
        _queued_runs = None
        return jsonify({"ok": True, "queued_runs": ""})
    # v0.76.0.7: strip and preserve all-temps: prefix before naive comma-split.
    # Prior version fed "all-temps:25,27,32" straight into split(',') — the first
    # token "all-temps:25" failed both isdigit() and range parsing, was silently
    # dropped, and the prefix itself was lost. Result: queuing analysis runs
    # from Layer 2 corrupted the run list AND lost the all-temps dispatch signal.
    # Fix: detect prefix up front, peel it off, parse numeric body, re-attach.
    _prefix_new = ""
    _body_new = runs
    if runs.startswith('all-temps:'):
        _prefix_new = 'all-temps:'
        _body_new = runs.split(':', 1)[1]
    # Additive: merge new runs with existing queue
    if _queued_runs:
        _prefix_cur = ""
        _body_cur = _queued_runs
        if _queued_runs.startswith('all-temps:'):
            _prefix_cur = 'all-temps:'
            _body_cur = _queued_runs.split(':', 1)[1]
        # Incompatible prefixes: new dispatch intent overwrites old.
        # Mixing all-temps and per-temp in the same queue would lose information
        # either way — take the new caller's intent as authoritative.
        if _prefix_cur != _prefix_new:
            _body_cur = ""
        existing = set()
        for p in _body_cur.split(','):
            p = p.strip()
            if '-' in p:
                lo, hi = p.split('-', 1)
                try:
                    for n in range(int(lo), int(hi)+1): existing.add(n)
                except ValueError: pass
            elif p.isdigit(): existing.add(int(p))
        for p in _body_new.split(','):
            p = p.strip()
            if '-' in p:
                lo, hi = p.split('-', 1)
                try:
                    for n in range(int(lo), int(hi)+1): existing.add(n)
                except ValueError: pass
            elif p.isdigit(): existing.add(int(p))
        _merged_body = ','.join(str(n) for n in sorted(existing))
        _queued_runs = _prefix_new + _merged_body if _merged_body else None
    else:
        _queued_runs = runs
    return jsonify({"ok": True, "queued_runs": _queued_runs or ""})

@app.route("/session",methods=["GET"])
def sess_get(): return jsonify(_rsess())

@app.route("/session",methods=["POST"])
def sess_post():
    d=request.get_json(force=True,silent=True) or {}
    if "trials" in d:
        try: d["trials"]=int(d["trials"])
        except: d.pop("trials")
    if "temperature" in d:
        try: d["temperature"]=round(float(d["temperature"]),2)
        except: d.pop("temperature")
    if "seed" in d:
        try: d["seed"]=int(d["seed"])
        except: d.pop("seed")
    return jsonify({"ok":True,"session":_wsess(d)})

@app.route("/add_model",methods=["POST"])
def add_model_ep():
    d=request.get_json(force=True,silent=True) or {}
    fam=d.get('model_family','')
    sz=d.get('model_size','')
    var=d.get('model_variant','abliterated')
    if not fam or not sz:
        return jsonify({"ok":False,"error":"Family and size required."}),400
    try:
        from cartography import get_paths
        get_paths(fam, sz, var, 0.0, create_dirs=True)
    except Exception:
        pass
    _wsess({"model_family":fam,"model_size":sz,"model_variant":var,
            "model_path":d.get('model_path',''),"model_name":d.get('model_name','')})
    return jsonify({"ok":True})

@app.route("/delete_model",methods=["POST"])
def delete_model_ep():
    if _running():
        return jsonify({"ok":False,"error":"Cannot delete while a run is active."}),409
    d=request.get_json(force=True,silent=True) or {}
    fam=d.get('family','')
    sz=d.get('size','')
    var=d.get('variant','')
    if not fam or not sz or not var:
        return jsonify({"ok":False,"error":"Family, size, and variant required."}),400
    from cartography import DATA, get_family_size_dir
    import shutil
    model_dir=os.path.join(get_family_size_dir(fam,sz),var)
    deleted=0
    if os.path.isdir(model_dir):
        try:
            deleted=sum(1 for _,_,files in os.walk(model_dir) for _ in files)
            shutil.rmtree(model_dir)
        except Exception as e:
            return jsonify({"ok":False,"error":str(e)}),500
    return jsonify({"ok":True,"deleted":deleted})

@app.route("/factory_reset",methods=["POST"])
def factory_reset_ep():
    if _running():
        return jsonify({"ok":False,"error":"Cannot reset while a run is active."}),409
    import shutil
    from cartography import DATA
    deleted=0
    # Delete all data
    if os.path.isdir(DATA):
        deleted+=sum(1 for _,_,files in os.walk(DATA) for _ in files)
        shutil.rmtree(DATA)
    # Delete logs, session, status, pid files
    for f in ['.iota_flask.log','.iota_flask.log.1','.iota_log.jsonl',
              'last_session.json','.iota_status.json','.iota_flask.pid',
              '.iota_pause','.iota_env.json']:
        fp=os.path.join(ROOT,f)
        if os.path.exists(fp):
            try: os.remove(fp); deleted+=1
            except: pass
    return jsonify({"ok":True,"deleted":deleted})

@app.route("/run",methods=["POST"])
def run_ep():
    global _proc,_pruns,_session_start_ts,_last_active_start
    if _running(): return jsonify({"ok":False,"error":"Already running. Abort first."}),409
    _session_start_ts = time.time()
    _last_active_start = time.time()
    d=request.get_json(force=True,silent=True) or {}
    runs=str(d.get("runs","")).strip()
    if not runs: return jsonify({"ok":False,"error":"No runs specified."}),400
    cfg={}
    for k in ("trials","temperature","seed"):
        if k in d: cfg[k]=d[k]
    if cfg: _wsess(cfg)
    _wsess({"runs":runs})
    # Save pass-mode overrides written by dashboard launch buttons.
    # Write mode overrides — grid launches strip these keys so they're None;
    # popup launches set them explicitly. None clears stale overrides.
    _wsess({"r19_passes": d.get("r19_passes"),
            "et_mode_overrides": d.get("et_mode_overrides"),
            "patch_modes_17": d.get("patch_modes_17"),
            "patch_modes_18": d.get("patch_modes_18"),
            "patch_modes_19": d.get("patch_modes_19"),
            "r3_cells": d.get("r3_cells"),
            "mc_conds": d.get("mc_conds")})
    log=os.path.join(ROOT,".iota_flask.log")
    if os.path.exists(log) and os.path.getsize(log) > 10*1024*1024:
        try: os.replace(log, log+'.1')
        except Exception: pass
    # BUG-43G: was truncation. Now appends separator — scroll-up history preserved.
    _log_separator("run: " + str(runs))
    try:
        lf=open(log,'a')
        try:
            # v0.79.4.9: --all-models dispatch iterates every discovered
            # model in DATA/, firing the requested runs against each in
            # sequence. The runs string itself can be auto-temp, all-temps:N,
            # or a plain run list — --all-models is an outer iteration wrapper.
            all_models_mode = bool(d.get('all_models'))
            if all_models_mode:
                # Pass the runs payload through unchanged — it will be
                # re-interpreted per model inside _run_all_models.
                p=subprocess.Popen(
                    [sys.executable,os.path.join(ROOT,"start_here.py"),"--all-models",runs],
                    stdout=lf,stderr=subprocess.STDOUT,start_new_session=True)
            elif runs == 'auto-temp':
                # Auto-temperature mode: advance through all incomplete rounds
                p=subprocess.Popen(
                    [sys.executable,os.path.join(ROOT,"start_here.py"),"--auto-temp"],
                    stdout=lf,stderr=subprocess.STDOUT,start_new_session=True)
            elif runs.startswith('all-temps:'):
                # All-temps mode: run specified runs at every temperature with data
                _atr = runs.split(':', 1)[1]
                p=subprocess.Popen(
                    [sys.executable,os.path.join(ROOT,"start_here.py"),"--all-temps-runs",_atr],
                    stdout=lf,stderr=subprocess.STDOUT,start_new_session=True)
            else:
                p=subprocess.Popen(
                    [sys.executable,os.path.join(ROOT,"start_here.py"),"--headless","--auto-run",runs],
                    stdout=lf,stderr=subprocess.STDOUT,start_new_session=True)
        finally:
            lf.close()
        with _plock: _proc=p; _pruns=runs
        _write_pid(p.pid)
        return jsonify({"ok":True,"pid":p.pid,"runs":runs})
    except Exception as e: return jsonify({"ok":False,"error":str(e)}),500

@app.route("/abort",methods=["POST"])
def abort_ep():
    global _proc,_pruns
    with _plock: p=_proc
    if p is None or p.poll() is not None:
        with _plock: _proc=None; _pruns=None
        return jsonify({"ok":True,"msg":"No active run."})
    try:
        # Kill entire process tree — not just the orchestrator.
        # The orchestrator (--auto-run) spawns child subprocesses (--single-run N).
        # p.terminate() only kills the orchestrator; the grandchild keeps running.
        # If user then clicks Play, two collection processes run simultaneously,
        # writing duplicate rows to the same CSV and overwriting .npy files.
        if os.name == 'nt':
            # Windows: taskkill /F /T kills process tree rooted at PID
            import subprocess as _abrt_sp
            _abrt_sp.call(['taskkill', '/F', '/T', '/PID', str(p.pid)],
                          stdout=_abrt_sp.DEVNULL, stderr=_abrt_sp.DEVNULL)
        else:
            # Unix: kill the process group (start_new_session=True created one)
            import signal as _sig
            try:
                os.killpg(os.getpgid(p.pid), _sig.SIGTERM)
            except ProcessLookupError:
                pass
        try: p.wait(timeout=5)
        except subprocess.TimeoutExpired: p.kill()
    except Exception as e: return jsonify({"ok":False,"error":str(e)}),500
    with _plock: _proc=None; _pruns=None
    _clear_pid()
    if os.path.exists(PAUSE_FILE): os.remove(PAUSE_FILE)
    return jsonify({"ok":True,"msg":"Run aborted."})

@app.route("/clear_run", methods=["POST"])
def clear_run_ep():
    """Delete all data files for a single run: CSV, hidden states (.npy), and
    analysis outputs. Does not touch other runs or calibration.

    No active-run guard — the OS will raise PermissionError if a file is
    locked by the subprocess, which is returned in 'errors'. .npy files are
    written and closed per turn so they are never locked mid-run.

    POST body: {"run": <run_num>}
    Returns: {"ok": bool, "removed": [...], "errors": [...], "n_removed": N}
    """
    try:
        import glob as _glob
        body = request.get_json(silent=True) or {}
        run_num = int(body.get("run", -1))
        if run_num < 0:
            return jsonify({"ok": False, "error": "Missing run number."}), 400
        s = _rsess()
        from cartography import get_paths
        from cartography import RUN_CSV, ANALYSIS_JSON
        paths = get_paths(s.get("model_family", "llama"), s.get("model_size", "8b"),
                          s.get("model_variant", "abliterated"), s.get("temperature", 0.0),
                          create_dirs=False)
        removed = []; errors = []

        # 1. CSV file
        csv_fname = RUN_CSV.get(run_num)
        if csv_fname:
            csv_path = os.path.join(paths.get("csv", ""), csv_fname)
            if os.path.exists(csv_path):
                try:
                    os.remove(csv_path)
                    removed.append(csv_path)
                except Exception as e:
                    errors.append(str(e))

        # 2. Hidden state .npy files (S_t, E_t, C_t, _alllayers, _et_base)
        #    Pattern: R{NN}_* and Q{NN}_* — both prefixes used depending on run num.
        from cartography import run_prefix as _rpfx
        prefix = f"{_rpfx(run_num)}{run_num:04d}_"
        hid_dir = paths.get("hidden", "")
        if os.path.isdir(hid_dir):
            for f in _glob.glob(os.path.join(hid_dir, prefix + "*.npy")):
                try:
                    os.remove(f)
                    removed.append(f)
                except Exception as e:
                    errors.append(str(e))

        # 3. Analysis JSON if this is a no-GPU analysis run
        ana_fname = ANALYSIS_JSON.get(run_num)
        if ana_fname:
            ana_path = os.path.join(paths.get("analysis", ""), ana_fname)
            if os.path.exists(ana_path):
                try:
                    os.remove(ana_path)
                    removed.append(ana_path)
                except Exception as e:
                    errors.append(str(e))

        return jsonify({"ok": not errors, "removed": removed, "errors": errors,
                        "run": run_num, "n_removed": len(removed),
                        "searched_csv": csv_path if csv_fname else None,
                        "searched_npy": os.path.join(hid_dir, prefix + "*.npy")})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/clear",methods=["POST"])
def clear_ep():
    if _running():
        return jsonify({"ok":False,"error":"Cannot clear while a run is active."}),409
    try:
        import shutil,glob as _glob
        body=request.get_json(silent=True) or {}
        del_data=bool(body.get("delete_data",False))
        del_cal =bool(body.get("delete_calibration",False))
        s=_rsess()
        from cartography import get_paths
        paths=get_paths(s.get("model_family","llama"),s.get("model_size","8b"),
                        s.get("model_variant","abliterated"),s.get("temperature",0.0),
                        create_dirs=False)
        removed=[];errors=[]
        # variant root = base/{family}/{size}/{variant}/
        # csv is at variant_root/{condition}/csv  so go up twice
        variant_root=os.path.dirname(os.path.dirname(paths["csv"]))
        if del_data and os.path.isdir(variant_root):
            for item in os.listdir(variant_root):
                full=os.path.join(variant_root,item)
                if os.path.isdir(full):
                    try: shutil.rmtree(full);removed.append(full)
                    except Exception as e: errors.append(str(e))
        if del_cal and os.path.isdir(variant_root):
            for f in _glob.glob(os.path.join(variant_root,"calibration_*.json")):
                try: os.remove(f);removed.append(f)
                except Exception as e: errors.append(str(e))
        return jsonify({"ok":not errors,"removed":removed,"errors":errors})
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)}),500

@app.route("/clear_data",methods=["POST"])
def clear_data_ep():
    """Clear data with granular scope control."""
    if _running():
        return jsonify({"ok":False,"error":"Cannot clear while a run is active."}),409
    try:
        import shutil, glob as _glob
        body = request.get_json(silent=True) or {}
        scope = body.get("scope", "")
        runs_str = body.get("runs", "")
        s = _rsess()
        from cartography import get_paths, RUN_CSV, DATA, condition_name, get_family_size_dir
        family = s.get("model_family", "llama")
        size = s.get("model_size", "8b")
        variant = s.get("model_variant", "abliterated")
        temp = s.get("temperature", 0.0)
        variant_root = os.path.join(get_family_size_dir(family, size), variant)
        deleted = 0

        def _parse_runs(rs):
            nums = set()
            for part in rs.split(','):
                part = part.strip()
                if '-' in part:
                    lo, hi = part.split('-', 1)
                    for n in range(int(lo), int(hi) + 1):
                        nums.add(n)
                elif part.isdigit():
                    nums.add(int(part))
            return nums

        def _clear_run_at_temp(run_num, temp_val):
            nonlocal deleted
            paths = get_paths(family, size, variant, temp_val, create_dirs=False)
            csv_dir = paths.get('csv', '')
            hid_dir = paths.get('hidden', '')
            ana_dir = paths.get('analysis', '')
            csv_name = RUN_CSV.get(run_num)
            # Delete CSV
            if csv_name and os.path.exists(os.path.join(csv_dir, csv_name)):
                os.remove(os.path.join(csv_dir, csv_name))
                deleted += 1
            # Delete hidden states for this run
            prefix = f"R{run_num:04d}_"
            for d in [hid_dir, ana_dir]:
                if os.path.isdir(d):
                    for f in os.listdir(d):
                        if f.startswith(prefix) or f.startswith(f"Q{run_num:04d}_"):
                            fp = os.path.join(d, f)
                            if os.path.isfile(fp):
                                os.remove(fp); deleted += 1

        if scope == 'run_temp':
            for rn in _parse_runs(runs_str):
                _clear_run_at_temp(rn, temp)
        elif scope == 'run_all':
            # Scan disk for all temperature directories
            _all_temps = []
            if os.path.isdir(variant_root):
                for d in sorted(os.listdir(variant_root)):
                    if d == 'deterministic': _all_temps.append(0.0)
                    elif d.startswith('temp_'):
                        try: _all_temps.append(float(d[5:]))
                        except: pass
            for rn in _parse_runs(runs_str):
                for t in _all_temps:
                    _clear_run_at_temp(rn, t)
        elif scope == 'temp':
            cond = condition_name(temp)
            cond_dir = os.path.join(variant_root, cond)
            if os.path.isdir(cond_dir):
                n = sum(1 for _ in _glob.glob(os.path.join(cond_dir, '**', '*'),
                        recursive=True) if os.path.isfile(_))
                shutil.rmtree(cond_dir)
                deleted = n
        elif scope == 'all_temps':
            # Delete all temperature condition directories but keep calibration
            if os.path.isdir(variant_root):
                for item in os.listdir(variant_root):
                    full = os.path.join(variant_root, item)
                    if os.path.isdir(full):
                        n = sum(1 for _ in _glob.glob(os.path.join(full, '**', '*'),
                                recursive=True) if os.path.isfile(_))
                        shutil.rmtree(full)
                        deleted += n
        elif scope == 'all':
            if os.path.isdir(variant_root):
                n = sum(1 for _ in _glob.glob(os.path.join(variant_root, '**', '*'),
                        recursive=True) if os.path.isfile(_))
                shutil.rmtree(variant_root)
                deleted = n
        else:
            return jsonify({"ok": False, "error": f"Unknown scope: {scope}"}), 400

        return jsonify({"ok": True, "deleted": deleted})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/models_data")
def models_data_ep():
    try:
        sys.path.insert(0, ROOT)
        from vault import IOTA_SUBFAMILY_MAP, FAMILIES, _classify_variant
        # Build hierarchy: family → generation → size → {abliterated, base, instruct}
        # Parse from IOTA_SUBFAMILY_MAP which has verified triplets
        _FAM_DISPLAY = {'llama': 'LLaMA', 'gemma': 'Gemma', 'qwen': 'Qwen',
                        'mistral': 'Mistral', 'phi': 'Phi', 'deepseek': 'DeepSeek',
                        'falcon': 'Falcon', 'yi': 'Yi', 'stablelm': 'StableLM',
                        'intern': 'InternLM', 'olmo': 'OLMo', 'nemotron': 'Nemotron'}
        hierarchy = {}
        # Get family from ALL_MODELS
        from vault import ALL_MODELS
        _fam_lookup = {}
        for _n, _p, _f, _s in ALL_MODELS:
            _fam_lookup[_p] = _f
        def _parse_subfamily(sf, abl_path):
            """Parse 'qwen2.5-0.5b' → (qwen, 2.5, 0.5b)"""
            fam = _fam_lookup.get(abl_path, '')
            if not fam or not sf: return None, None, None
            # Strip family prefix to get gen-size
            # "llama-3-8b" strip "llama-" → "3-8b"
            # "qwen2.5-0.5b" strip "qwen" → "2.5-0.5b"
            # "gemma-2-2b" strip "gemma-" → "2-2b"
            remainder = sf
            if sf.startswith(fam + '-'):
                remainder = sf[len(fam)+1:]
            elif sf.startswith(fam):
                remainder = sf[len(fam):]
                if remainder.startswith('-'): remainder = remainder[1:]
            # Last segment ending in 'b' or known size words is the size
            parts = remainder.split('-')
            size = parts[-1]
            gen = '-'.join(parts[:-1]) if len(parts) > 1 else ''
            if not gen: gen = '1'  # single-gen family
            return fam, gen, size

        for abl_path, info in IOTA_SUBFAMILY_MAP.items():
            sf = info.get('subfamily', '')
            if not sf: continue
            fam_key, gen, size_part = _parse_subfamily(sf, abl_path)
            if not fam_key: continue
            fam_display = _FAM_DISPLAY.get(fam_key, fam_key.title())
            gen_display = f"{fam_display} {gen}"
            size_display = size_part.upper()
            # Determine variant directory name from abliterated path
            v = _classify_variant(abl_path)
            if v == 'unknown': v = 'abliterated'
            if fam_key not in hierarchy:
                hierarchy[fam_key] = {}
            if gen not in hierarchy[fam_key]:
                hierarchy[fam_key][gen] = {}
            hierarchy[fam_key][gen][size_part] = {
                "name": f"{gen_display} {size_display}",
                "path": abl_path,
                "variant": v,
                "base": info.get('base', ''),
                "instruct": info.get('instruct', ''),
                "subfamily": sf,
                "note": info.get('note', ''),
            }
        # Build flat families dict for backward compat
        fams = {k: FAMILIES.get(k, k.title()) for k in hierarchy}
        # Build sizes dict keyed by family for backward compat (first gen's first size)
        sizes = {}
        for fk, gens in hierarchy.items():
            sz_data = {}
            for gk, szs in sorted(gens.items()):
                for sk, info in sorted(szs.items()):
                    # Use subfamily as unique key to avoid collisions
                    sz_data[info['subfamily']] = {
                        "name": info['name'], "path": info['path'],
                        "variant": info['variant'], "gen": gk, "size": sk,
                    }
            if sz_data:
                sizes[fk] = sz_data
        fams_filtered = {k: v for k, v in fams.items() if k in sizes}
        return jsonify({"families": fams_filtered, "sizes": sizes,
                        "hierarchy": hierarchy})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/stats",methods=["POST"])
def stats_ep():
    try:
        # Clear stale status; append separator to log (history preserved for scroll-up).
        _clear_stale_status()
        _log_separator("stats")
        log=os.path.join(ROOT,".iota_flask.log")
        if os.path.exists(log) and os.path.getsize(log) > 10*1024*1024:
            try: os.replace(log, log+'.1')
            except Exception: pass
        s=_rsess()
        lf=open(log,'a')
        p=subprocess.Popen(
            [sys.executable, os.path.join(ROOT,"export_stats.py")],
            stdout=lf, stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "IOTA_SESSION": json.dumps(s)},
        )
        lf.close()
        return jsonify({"ok":True,"pid":p.pid,"msg":"Stats running - check console log."})
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)}),500

@app.route("/all_stats",methods=["POST"])
def all_stats_ep():
    """Launch full analysis + stats across all temperature conditions, then Run 0051 pooled."""
    global _proc,_pruns,_last_active_start
    if _running(): return jsonify({"ok":False,"error":"Already running. Abort first."}),409
    _last_active_start = time.time()
    d=request.get_json(force=True,silent=True) or {}
    mode=d.get('mode','skip')  # skip, overwrite, backup
    try:
        _clear_stale_status()
        _log_separator("all-stats: " + mode)
        log=os.path.join(ROOT,".iota_flask.log")
        if os.path.exists(log) and os.path.getsize(log) > 10*1024*1024:
            try: os.replace(log, log+'.1')
            except Exception: pass
        cmd=[sys.executable, os.path.join(ROOT,"start_here.py"), "--all-stats"]
        if mode=='overwrite': cmd.append('--force')
        elif mode=='backup': cmd.extend(['--force','--backup'])
        lf=open(log,'a')
        try:
            p=subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        finally:
            lf.close()
        with _plock: _proc=p; _pruns="all-stats"
        _write_pid(p.pid)
        return jsonify({"ok":True,"pid":p.pid,"mode":mode,"msg":"All Stats running ("+mode+")."})
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)}),500


@app.route("/report",methods=["POST"])
def report_ep():
    """Generate the figures + markdown report from Run 0051/0046/0050 outputs."""
    if _running(): return jsonify({"ok":False,"error":"A run is active. Wait for it to finish."}),409
    try:
        import report as _rpt
        # Force reimport in case report.py was updated
        import importlib
        importlib.reload(_rpt)
        s = _rsess()
        path = _rpt.generate(session=s)
        # Count figures
        fig_dir = os.path.join(os.path.dirname(path), 'figures')
        import glob as _rglob
        n_figs = len(_rglob.glob(os.path.join(fig_dir, '*.png')))
        return jsonify({"ok":True,"path":path,"n_figs":n_figs,"fig_dir":fig_dir})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok":False,"error":str(e),"tb":traceback.format_exc()}),500


@app.route("/report_status")
def report_status_ep():
    """Check if report prerequisites are met (Run 0051 or 0050 JSON exists)."""
    try:
        from cartography import get_pooled_paths
        s = _rsess()
        pp = get_pooled_paths(s.get('model_family','llama'),
                              s.get('model_size','8b'),
                              s.get('model_variant','abliterated'),
                              create_dirs=False)
        ana = pp.get('analysis', '')
        has_40 = os.path.exists(os.path.join(ana, 'Q0051_temperature_curve.json'))
        has_47 = os.path.exists(os.path.join(ana, 'Q0050_cross_temp_synthesis.json'))
        has_report = os.path.exists(os.path.join(ana, 'REPORT.md'))
        return jsonify({"ready": has_40 or has_47, "has_40": has_40, "has_47": has_47,
                        "has_report": has_report})
    except Exception as e:
        return jsonify({"ready": False, "error": str(e)})

@app.route("/all_temps_complete")
def all_temps_complete_ep():
    """Check if all generation runs are complete across all temperature conditions."""
    try:
        from cartography import DATA, get_paths as _gp, get_family_size_dir
        s=_rsess()
        family  = s.get('model_family','llama')
        size    = s.get('model_size','8b')
        variant = s.get('model_variant','abliterated')
        base_dir = os.path.join(get_family_size_dir(family, size), variant)
        if not os.path.isdir(base_dir):
            return jsonify({"complete":False,"reason":"no data directory","conditions":[]})

        import start_here as sh
        # Generation runs — everything except analysis-only runs
        _GEN_RUNS = set(range(1, 41))  # v0.79.4.0: all 40 data-collection runs
        conditions_status = {}
        all_done = True
        n_conditions = 0
        for cond in sorted(os.listdir(base_dir)):
            if cond == 'pooled': continue
            cond_csv = os.path.join(base_dir, cond, 'csv')
            if not os.path.isdir(cond_csv): continue
            n_conditions += 1
            # Get paths for this condition's temperature
            _COND_TEMPS = {
                'deterministic':0.0,'temp_0.2':0.2,'temp_0.4':0.4,
                'temp_0.6':0.6,'temp_0.8':0.8,'temp_1.0':1.0,
            }
            temp = _COND_TEMPS.get(cond, 0.0)
            paths = _gp(family, size, variant, temp, create_dirs=False)
            status = sh._scan_runs(paths, s.get('trials',100))
            done_runs = {r for r in _GEN_RUNS if status.get(r) == 'done'}
            missing = _GEN_RUNS - done_runs
            cond_done = len(missing) == 0
            conditions_status[cond] = {
                "done": cond_done,
                "n_done": len(done_runs),
                "n_total": len(_GEN_RUNS),
                "missing": sorted(missing) if missing else [],
            }
            if not cond_done:
                all_done = False

        if n_conditions < 6:
            all_done = False  # need all 6 temperature rounds

        return jsonify({
            "complete": all_done,
            "n_conditions": n_conditions,
            "conditions": conditions_status,
        })
    except Exception as e:
        return jsonify({"complete":False,"error":str(e)})


@app.route("/temp_status")
def temp_status_ep():
    """Return per-round completion status for all 6 temperatures.
    Used by Auto mode to show the first incomplete round in the grid."""
    try:
        from cartography import get_paths as _gp
        import start_here as sh
        s = _rsess()
        family  = s.get('model_family', 'llama')
        size    = s.get('model_size', '8b')
        variant = s.get('model_variant', 'abliterated')
        n_trials = s.get('trials', 100)
        _GEN_RUNS = set(range(1, 41))  # v0.79.4.0: all 40 data-collection runs
        _TEMPS = [
            (0.0, 'deterministic'), (0.2, 'temp_0.2'), (0.4, 'temp_0.4'),
            (0.6, 'temp_0.6'), (0.8, 'temp_0.8'), (1.0, 'temp_1.0'),
        ]
        rounds = []
        first_incomplete = None
        for temp, cond in _TEMPS:
            paths = _gp(family, size, variant, temp, create_dirs=False)
            status = sh._scan_runs(paths, n_trials)
            done = {r for r in _GEN_RUNS if status.get(r) == 'done'}
            complete = len(done) == len(_GEN_RUNS)
            rounds.append({"temperature": temp, "condition": cond,
                           "n_done": len(done), "n_total": len(_GEN_RUNS),
                           "complete": complete})
            if not complete and first_incomplete is None:
                first_incomplete = temp
        return jsonify({"rounds": rounds, "first_incomplete": first_incomplete})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/models_collected")
def models_collected_ep():
    """Return models with per-temperature status breakdown."""
    from cartography import DATA, get_paths, RUN_CSV, ANALYSIS_JSON
    from cartography import logical_size as _logical_size, quant_from_dir as _quant_from_dir
    from vault import IOTA_SUBFAMILY_MAP, get_preferred_variant
    _FAM_NAMES = {'llama': 'LLaMA', 'gemma': 'Gemma', 'qwen': 'Qwen',
                  'mistral': 'Mistral', 'phi': 'Phi'}
    # Build reverse lookup: (family, size, variant) → display label from subfamily
    _LABELS = {}
    for hf_path, info in IOTA_SUBFAMILY_MAP.items():
        sf = info.get('subfamily', '')
        if sf:
            # "gemma-2-2b" → "Gemma 2 2B", "llama-3-8b" → "LLaMA 3 8B"
            parts = sf.split('-')
            fam_key = parts[0] if parts else ''
            fam_display = _FAM_NAMES.get(fam_key, fam_key.title())
            rest = ' '.join(p.upper() if p[-1] == 'b' and p[:-1].replace('.','').isdigit() else p
                           for p in parts[1:])
            _LABELS[hf_path] = fam_display + ' ' + rest
    _STANDARD = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    s = _rsess()
    _active = s.get('active_temps', _STANDARD)
    if not isinstance(_active, list) or not _active:
        _active = _STANDARD
    _DATA_CSVS = {k: v for k, v in RUN_CSV.items() if v is not None}
    # v0.79.4.9: post-renumber. Pooled = {50, 51}, cross-model = {52, 53, 54},
    # output = {55, 56}. Naming reflects execution-order positions.
    from cartography import DualKeyRunSet as _DKRSet
    _POOLED_RUNS = _DKRSet({50, 51})
    _CROSS_MODEL_RUNS = _DKRSet({52, 53, 54})
    _OUTPUT_RUNS = _DKRSet({55, 56})
    _PER_TEMP_ANA = {k: v for k, v in ANALYSIS_JSON.items()
                     if k not in _POOLED_RUNS
                     and k not in _CROSS_MODEL_RUNS
                     and k not in _OUTPUT_RUNS}
    _POOLED_ANA = {k: v for k, v in ANALYSIS_JSON.items()
                   if k in _POOLED_RUNS}
    groups = {}
    if not os.path.isdir(DATA):
        return jsonify({"models": []})
    try:
        for family in sorted(os.listdir(DATA)):
            fam_dir = os.path.join(DATA, family)
            if not os.path.isdir(fam_dir): continue
            for size in sorted(os.listdir(fam_dir)):
                sz_dir = os.path.join(fam_dir, size)
                if not os.path.isdir(sz_dir): continue
                for variant in sorted(os.listdir(sz_dir)):
                    var_dir = os.path.join(sz_dir, variant)
                    if not os.path.isdir(var_dir): continue
                    n_with_data = 0
                    temp_rows = []
                    # Merge active temps with temps that have data on disk
                    _disk_temps = set()
                    for d in os.listdir(var_dir):
                        if d == 'deterministic':
                            _disk_temps.add(0.0)
                        elif d.startswith('temp_'):
                            try: _disk_temps.add(float(d[5:]))
                            except: pass
                    _show_temps = sorted(set(_active) | _disk_temps)
                    for temp in _show_temps:
                        paths = get_paths(family, size, variant, temp,
                                          create_dirs=False)
                        csv_dir = paths.get('csv', '')
                        ana_dir = paths.get('analysis', '')
                        if not os.path.isdir(csv_dir):
                            temp_rows.append({"temp": temp, "data": 0, "ana": 0})
                            continue
                        n_with_data += 1
                        td = 0
                        for rn, cf in _DATA_CSVS.items():
                            fp = os.path.join(csv_dir, cf)
                            if os.path.exists(fp):
                                td += 1
                        ta = 0
                        for rn, jf in _PER_TEMP_ANA.items():
                            if os.path.exists(os.path.join(ana_dir, jf)):
                                ta += 1
                        temp_rows.append({"temp": temp, "data": td, "ana": ta})
                    # Pooled analysis
                    pooled_count = 0
                    pooled_ana = os.path.join(var_dir, 'pooled', 'analysis')
                    if os.path.isdir(pooled_ana):
                        for rn, jf in _POOLED_ANA.items():
                            if os.path.exists(os.path.join(pooled_ana, jf)):
                                pooled_count += 1
                    # Summary totals = sum across all temps + pooled
                    total_data = sum(t['data'] for t in temp_rows)
                    total_ana = sum(t['ana'] for t in temp_rows) + pooled_count
                    if n_with_data > 0:
                        _lsize = _logical_size(size)
                        _key = family + '/' + size
                        _n_temps = len(_show_temps)
                        # Read quant from directory name, fall back to .quant file, then session
                        _quant = _quant_from_dir(size) or ''
                        if not _quant:
                            try:
                                _qpath = os.path.join(var_dir, '.quant')
                                if os.path.exists(_qpath):
                                    with open(_qpath) as _qf: _quant = _qf.read().strip()
                            except Exception: pass
                        if not _quant:
                            _quant = s.get('quantization', '4bit')
                        if _key not in groups or n_with_data > groups[_key].get('n_with_data', 0):
                            _pn, _pp, _pv = get_preferred_variant(family, _lsize)
                            _lbl = _LABELS.get(_pp, _FAM_NAMES.get(family, family.title()) + ' ' + _lsize.upper())
                            groups[_key] = {
                                "family": family, "size": _lsize, "variant": variant,
                                "label": _lbl, "quant": _quant,
                                "n_with_data": n_with_data,
                                "temps": temp_rows,
                                "data_per_temp": len(_DATA_CSVS),
                                "ana_per_temp": len(_PER_TEMP_ANA),
                                "pooled_runs": pooled_count,
                                "pooled_total": len(_POOLED_ANA),
                                "total_data": total_data,
                                "total_data_max": len(_DATA_CSVS) * _n_temps,
                                "total_ana": total_ana,
                                "total_ana_max": len(_PER_TEMP_ANA) * _n_temps + len(_POOLED_ANA),
                            }
    except Exception:
        pass
    return jsonify({"models": list(groups.values())})


@app.route("/cross_model_status")
def cross_model_status_ep():
    """Return all discovered models with analysis completion status.
    Used by the dashboard to show checkboxes for Run 0056 (paper assembly).
    v0.75.2.2: new endpoint for cross-model model selection."""
    try:
        sys.path.insert(0, ROOT)
        import export_stats as _cms_es
        models = _cms_es._discover_model_data({})
        result = []
        for m in models:
            pooled_ana = os.path.join(m['data_dir'], 'pooled', 'analysis')
            has_stamp = os.path.exists(os.path.join(pooled_ana, 'Q0055_stats_report.json'))
            # Count temps with Q34 (permutation sensitivity = core analysis)
            n_temps = 0
            for cond in ('deterministic', 'temp_0.2', 'temp_0.4', 'temp_0.6', 'temp_0.8', 'temp_1.0'):
                q34 = os.path.join(m['data_dir'], cond, 'analysis', 'Q0043_sobol_partition.json')
                if os.path.exists(q34):
                    n_temps += 1
            result.append({
                'label': m['label'],
                'family': m['family'],
                'size': m['size'],
                'variant': m['variant'],
                'n_temps_analyzed': n_temps,
                'has_r54_stamp': has_stamp,
                'ready': n_temps >= 1,
            })
        return jsonify({"models": result})
    except Exception as e:
        return jsonify({"error": str(e), "models": []}), 500


@app.route("/temp_grid")
def temp_grid_ep():
    """Per-run completion across all 6 temperatures for the All Temps grid.
    Uses scanner.scan_runs — the same battle-tested logic as the per-temp grid.
    Returns {grid: {run_num: {temp: status}}, temps: [...]}."""
    from cartography import get_paths
    from scanner import scan_runs
    s = _rsess()
    family  = s.get('model_family', 'llama')
    size    = s.get('model_size', '8b')
    variant = s.get('model_variant', 'abliterated')
    n_trials = int(s.get('trials', 100))
    _standard = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    temps = s.get('active_temps', _standard)
    if not isinstance(temps, list) or not temps:
        temps = _standard
    grid = {}
    _debug = [f"model={family}/{size}/{variant}"]
    try:
        for temp in temps:
            paths = get_paths(family, size, variant, temp, create_dirs=False)
            status = scan_runs(paths, n_trials)
            ts = f"{temp:.1f}"
            for run_num, st in status.items():
                rn = str(run_num)
                if rn not in grid:
                    grid[rn] = {}
                grid[rn][ts] = st
            # DEBUG — remove after fixing Run 0019 issue
            if 19 in status:  # v0.79.4.0: old R53 random patching → new 19
                _debug.append(f"T={temp} Run53={status[53]}")
            else:
                csv_dir = paths.get('csv', '')
                csv53 = os.path.join(csv_dir, 'R0019_random_patching.csv')
                _debug.append(f"T={temp} Run53=NOT_IN_STATUS csv_exists={os.path.exists(csv53)}")
    except Exception as e:
        _debug.append(f"CRASHED: {e}")
    return jsonify({"grid": grid, "temps": temps, "_debug": _debug})


@app.route("/outcomes")
def outcomes_ep():
    """Serve hypothesis outcomes aggregated across all temperatures.
    Reads hypothesis_outcomes.json from each temp's analysis dir.
    Returns the most recent/complete set found."""
    try:
        from cartography import get_paths, DATA, get_family_size_dir
        s = _rsess()
        family = s.get('model_family', 'llama')
        size = s.get('model_size', '8b')
        variant = s.get('model_variant', 'abliterated')
        # Try all temps, use the most populated outcomes file
        best = {}
        best_count = 0
        for temp in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            paths = get_paths(family, size, variant, temp, create_dirs=False)
            out_json = os.path.join(paths.get('analysis', ''), 'hypothesis_outcomes.json')
            if os.path.exists(out_json):
                try:
                    with open(out_json, encoding='utf-8') as f:
                        raw = f.read()
                    raw = re.sub(r'\bNaN\b', 'null', raw)
                    raw = re.sub(r'\bInfinity\b', 'null', raw)
                    raw = re.sub(r'\b-Infinity\b', 'null', raw)
                    data = json.loads(raw)
                    # Merge: keep non-pending results, prefer supported/disproven over inconclusive
                    for k, v in data.items():
                        if k not in best or (isinstance(v, dict) and v.get('status') != 'pending'
                                             and (k not in best or best[k].get('status') == 'pending')):
                            best[k] = v
                    n = sum(1 for v in data.values() if isinstance(v, dict) and v.get('status') != 'pending')
                    if n > best_count:
                        best_count = n
                except Exception:
                    pass
        # Also check pooled analysis
        pooled_ana = os.path.join(get_family_size_dir(family, size), variant, 'pooled', 'analysis')
        for fname in ['hypothesis_outcomes.json']:
            fp = os.path.join(pooled_ana, fname)
            if os.path.exists(fp):
                try:
                    with open(fp, encoding='utf-8') as f:
                        raw = f.read()
                    raw = re.sub(r'\bNaN\b', 'null', raw)
                    data = json.loads(raw)
                    for k, v in data.items():
                        if k not in best or (isinstance(v, dict) and v.get('status') != 'pending'):
                            best[k] = v
                except Exception:
                    pass
        return jsonify(best)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Live hypothesis computation (v0.72.1.0) ─────────────────────────────────
# Computes hypothesis outcomes on demand from CSVs + analysis JSONs.
# Cached keyed on max CSV mtime — only recomputes when data changes.
# Falls back to static hypothesis_outcomes.json if computation fails.

_hyp_cache = {"mtime": 0, "data": None, "temp": None}
_hyp_lock = threading.Lock()

def _csv_max_mtime(csv_dir):
    """Return max mtime across all CSVs in a directory."""
    if not os.path.isdir(csv_dir):
        return 0
    mx = 0
    for f in os.listdir(csv_dir):
        if f.endswith('.csv'):
            try:
                mt = os.path.getmtime(os.path.join(csv_dir, f))
                if mt > mx: mx = mt
            except: pass
    return mx

@app.route("/hyp_live")
def hyp_live_ep():
    """Compute hypothesis outcomes live from CSV data. Cached until data changes."""
    try:
        from cartography import get_paths, DATA, get_family_size_dir
        s = _rsess()
        family = s.get('model_family', 'llama')
        size = s.get('model_size', '8b')
        variant = s.get('model_variant', 'abliterated')
        temp = s.get('temperature', 0.0)
        paths = get_paths(family, size, variant, temp, create_dirs=False)
        csv_dir = paths.get('csv', '')
        ana_dir = paths.get('analysis', '')

        mtime = _csv_max_mtime(csv_dir)
        with _hyp_lock:
            if _hyp_cache["data"] and _hyp_cache["mtime"] == mtime and _hyp_cache["temp"] == temp:
                return jsonify(_hyp_cache["data"])

        # Lazy import — pandas + scipy + matplotlib only loaded on first call
        import export_stats as _es

        df = _es.load_all(csv_dir)
        granger = _es.load_granger(ana_dir)
        bs = _es.load_baseline_swap(ana_dir)
        decomp = _es.load_decomposition(ana_dir)
        sobol = _es.load_permutation_sensitivity(ana_dir)
        pooled_ana = os.path.join(get_family_size_dir(family, size), variant, 'pooled', 'analysis')
        pooled_sobol = _es.load_pooled_permutation_sensitivity(pooled_ana)

        outcomes = _es._infer_outcomes(df, granger, bs, decomp, sobol,
                                       pooled_sobol=pooled_sobol)

        # Sanitize NaN/Infinity for JSON
        raw = json.dumps(outcomes, default=str)
        raw = re.sub(r'\bNaN\b', 'null', raw)
        raw = re.sub(r'\bInfinity\b', 'null', raw)
        raw = re.sub(r'\b-Infinity\b', 'null', raw)
        clean = json.loads(raw)

        with _hyp_lock:
            _hyp_cache["data"] = clean
            _hyp_cache["mtime"] = mtime
            _hyp_cache["temp"] = temp

        return jsonify(clean)
    except Exception as e:
        # Fall back to static outcomes
        try:
            return outcomes_ep()
        except:
            return jsonify({"error": str(e)}), 500

@app.route("/resume_state")
def resume_state_ep():
    """Return resume state. Shows button ONLY when _resume_runs exists
    (interrupted session). v0.71.0.11: reverted fallback to session runs."""
    s = _rsess()
    resume = s.get('_resume_runs', '')
    return jsonify({"resume_runs": resume, "has_resume": bool(resume)})

@app.route("/resume",methods=["POST"])
def resume_ep():
    """Stage runs for relaunch. Uses _resume_runs ONLY (interrupted sessions).
    v0.71.0.11: reverted fallback to session runs — resume is for interrupted only."""
    try:
        s = _rsess()
        resume = s.get('_resume_runs', '')
        if not resume:
            return jsonify({"ok": False, "error": "No interrupted session to resume"})
        _wsess({"runs": resume})
        sf = os.path.join(ROOT, 'last_session.json')
        if os.path.exists(sf):
            try:
                with open(sf) as f: raw = json.load(f)
                raw.pop('_resume_runs', None)
                import tempfile as _tf
                _fd, _tmp = _tf.mkstemp(dir=os.path.dirname(os.path.abspath(sf)), suffix='.tmp')
                try:
                    with os.fdopen(_fd, 'w') as _f: json.dump(raw, _f, indent=2)
                    os.replace(_tmp, sf)
                except Exception:
                    try: os.unlink(_tmp)
                    except Exception: pass
            except Exception:
                pass
        return jsonify({"ok": True, "runs": resume})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/scan")
def scan_ep():
    # v0.79.4.9: accept optional query params to scan a specific model/temp
    # without mutating the session. Used by navToAllModels in the dashboard
    # to overlay per-model status into the All-Models aggregate grid.
    fam = request.args.get('family')
    sz  = request.args.get('size')
    var = request.args.get('variant')
    tmp = request.args.get('temperature')
    if fam and sz and var:
        try:
            from cartography import get_paths, logical_size as _ls
            try:
                _t = float(tmp) if tmp is not None else 0.0
            except Exception:
                _t = 0.0
            # size may come in either '2b' or '2b_8bit' form — resolve via
            # logical_size so get_paths can append quant suffix itself.
            _sz_in = _ls(sz) if sz else sz
            s = _rsess()
            paths = get_paths(fam, _sz_in, var, _t, create_dirs=False)
            import start_here as sh
            status = sh._scan_runs(paths, s.get('trials', 100))
            # Return raw dict to match the shape of the session-scoped
            # default path — consumers index by run_num directly.
            return jsonify({str(k): v for k, v in status.items()})
        except Exception as e:
            return jsonify({'error': str(e)})
    # Default: session-scoped scan (existing behaviour).
    return jsonify({str(k):v for k,v in _scan().items()})

@app.route("/scan_debug")
def scan_debug_ep():
    """Debug endpoint — shows scanner status, csv existence, and header info."""
    try:
        from cartography import get_paths
        from cartography import RUN_CSV
        import csv as _dcsv, io as _dio
        s = _rsess()
        paths = get_paths(s.get('model_family','llama'), s.get('model_size','8b'),
                          s.get('model_variant','abliterated'), s.get('temperature',0.0),
                          create_dirs=False)
        csv_dir = paths.get('csv','')
        import start_here as sh
        status = sh._scan_runs(paths, s.get('trials',100))
        detail = {}
        for rn, fname in RUN_CSV.items():
            if fname is None:
                continue
            fpath = os.path.join(csv_dir, fname)
            exists = os.path.exists(fpath)
            st = status.get(rn, 'NOT_IN_STATUS')
            hdr_len = None
            has_rc = None
            n_rows = None
            row_lens = None
            if exists:
                try:
                    with open(fpath, 'rb') as fh:
                        raw = fh.read()
                    text = raw.decode('utf-8', errors='replace').replace('\r\n','\n').replace('\r','\n')
                    rows = list(_dcsv.reader(_dio.StringIO(text)))
                    if rows:
                        hdr = rows[0]
                        hdr_len = len(hdr)
                        has_rc = 'disruption_flag' in hdr
                        n_rows = len(rows) - 1
                        lens = sorted(set(len(r) for r in rows[1:] if len(r) > 2))
                        row_lens = lens[:5]
                except Exception as e2:
                    hdr_len = f'err:{e2}'
            detail[str(rn)] = {
                'status': st, 'csv_exists': exists, 'fname': fname,
                'hdr_len': hdr_len, 'has_rc': has_rc,
                'n_rows': n_rows, 'row_lens': row_lens
            }
        return jsonify({'csv_dir': csv_dir, 'runs': detail})
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()})

@app.route("/run_detail")
def run_detail_ep():
    """Per-run trial breakdown for the grid detail popup."""
    try:
        run_num = int(request.args.get('run', 0))
        from cartography import get_paths
        from cartography import RUN_CSV
        s = _rsess()
        paths = get_paths(s.get('model_family','llama'), s.get('model_size','8b'),
                          s.get('model_variant','abliterated'), s.get('temperature',0.0),
                          create_dirs=False)
        import start_here as sh
        n_trials = s.get('trials', 100)
        desc = sh.RUN_MAP.get(run_num, ('', f'Run {run_num}'))[1]
        csv_fname = RUN_CSV.get(run_num)
        if csv_fname is None:
            return jsonify({'run': run_num, 'desc': desc, 'status': 'analysis',
                            'n_trials': n_trials, 'n_complete': 0,
                            'note': 'Analysis run — output is a JSON file, not a trial CSV.'})
        csv_dir  = paths.get('csv', '')
        hid_dir  = paths.get('hidden', '')
        fpath    = os.path.join(csv_dir, csv_fname)

        # ── Run 0001 special case ───────────────────────────────────────────────
        # Three model passes + vector computation. Only abliterated writes CSV.
        # Show per-pass progress instead of a single trial count.
        if run_num == 1:
            import glob as _g
            abl_trials = {}
            if os.path.exists(fpath):
                from orchestration_core import canonical_read as _cr19
                cols19 = _cr19(fpath, ('run_mode', 'trial', 'priming'))
                for i, t in enumerate(cols19.get('trial', [])):
                    if cols19.get('priming') and cols19['priming'][i] == '1':
                        continue
                    try:
                        tn = int(t)
                        abl_trials[tn] = abl_trials.get(tn, 0) + 1
                    except Exception:
                        pass
            n_abl = len(abl_trials)

            # v42.1.11: base/instruct are sibling variant dirs.
            # Derive from hid_dir: .../abliterated/cond/hidden_states
            #                    → .../base/cond/hidden_states
            def _r19_sibling(base_hid, variant):
                try:
                    _cond    = os.path.dirname(base_hid)
                    _size    = os.path.dirname(os.path.dirname(_cond))
                    return os.path.join(_size, variant,
                                        os.path.basename(_cond), 'hidden_states')
                except Exception:
                    return os.path.join(base_hid, variant)  # v42.1.10 fallback

            base_trials = set()
            for _bd in [_r19_sibling(hid_dir, 'base'), os.path.join(hid_dir, 'base')]:
                for f in _g.glob(os.path.join(_bd, 'R19_*_base_trial*_turn01.npy')):
                    try: base_trials.add(int(os.path.basename(f).split('_trial')[1].split('_')[0]))
                    except Exception: pass
            n_base = len(base_trials)

            inst_trials = set()
            for _id in [_r19_sibling(hid_dir, 'instruct'), os.path.join(hid_dir, 'instruct')]:
                for f in _g.glob(os.path.join(_id, 'R19_*_instruct_trial*_turn01.npy')):
                    try: inst_trials.add(int(os.path.basename(f).split('_trial')[1].split('_')[0]))
                    except Exception: pass
            n_inst = len(inst_trials)

            # BUG-PASS4-SENTINEL: accept ct_global_mean.npy as fallback when
            # sentinel absent — covers vector passes run on pre-v42.1.2 code.
            pass4 = (len(_g.glob(os.path.join(hid_dir, 'R19_*_pass4_ok.stamp'))) > 0 or
                     len(_g.glob(os.path.join(hid_dir, 'R19_*_ct_global_mean.npy'))) > 0)

            all_done = (n_abl >= n_trials and n_base >= n_trials
                        and n_inst >= n_trials and pass4)
            nothing  = (n_abl == 0 and n_base == 0 and n_inst == 0)
            st19 = 'done' if all_done else 'missing' if nothing else 'partial'
            # Diagnostic: include exact paths scanned so misroutes show up
            # in the run detail popup without reading source.
            _base_scan_dir = _r19_sibling(hid_dir, 'base')
            _inst_scan_dir = _r19_sibling(hid_dir, 'instruct')
            return jsonify({
                'run': run_num, 'desc': desc, 'status': st19,
                'n_trials': n_trials * 3,
                'n_complete': n_abl + n_base + n_inst,
                'expected_turns': None,
                'n_partial': 0, 'n_missing': 0, 'partial': [], 'missing': [],
                'run1': {
                    'abliterated': n_abl, 'base': n_base,
                    'instruct': n_inst, 'pass4': pass4, 'n_trials': n_trials,
                    'scan_dirs': {
                        'abliterated': hid_dir,
                        'base': _base_scan_dir,
                        'instruct': _inst_scan_dir,
                    }
                }
            })

        # ── Activation patching runs (Run 0017 and Run 0018) ────────────────────
        # Mixed-schema CSVs: none rows have full trajectory metrics;
        # patched rows have patching-specific fields only (patch_mode /
        # patch_layer, sim_to_reference, output_changed, output, prompt).
        # Counting all rows/trial gives 18 (R21: 3 modes × 6 turns) or
        # 65 (R42: 5 modes × 13 turns) — shown as "N turns/trial" in the
        # standard path, which is completely misleading.
        # Fix: read patch_mode/patch_layer column, count only none rows for
        # trial completion, report per-mode row counts as a sanity check.
        _PATCH_LAYERS_DEFAULT = [8, 16, 24, 31]
        _nl = 32
        try:
            _var_dir = os.path.dirname(os.path.dirname(csv_dir))
            _nl_path = os.path.join(_var_dir, '.n_layers')
            if os.path.exists(_nl_path):
                with open(_nl_path) as _f:
                    _nl = int(_f.read().strip())
        except Exception: pass
        _valid_layers = [li for li in _PATCH_LAYERS_DEFAULT if li < _nl]
        _PATCHING_RUNS = {
            # v0.79.4.0 renumber: old 21→17 (activation), 42→18 (layer iso), 53→19 (random).
            17: {'col': 'patch_mode',  'modes': ['none','partial','full','random'],
                 'turns': 13, 'label': 'patch_mode'},
            18: {'col': 'patch_layer', 'modes': ['none'] + [f'L{li}' for li in _valid_layers],
                 'turns': 13, 'label': 'patch_layer'},
            19: {'col': 'patch_layer', 'modes': ['none','full'] + [f'L{li}' for li in _valid_layers],
                 'turns': 13, 'label': 'patch_layer'},
        }
        if run_num in _PATCHING_RUNS and run_num != 1:
            _pi = _PATCHING_RUNS[run_num]
            _pcol = _pi['col']
            _pmodes = _pi['modes']
            _pt = _pi['turns']
            # BUG-PATCH-SCAN fix (v42.6.5): _read_csv_cols returns patch_mode
            # using its header-index position (~30). Patched rows are 11 fields
            # wide — that index is always out of range, so every patched row
            # returns mode=None and is skipped. Use raw CSV read with per-row
            # logic: short rows (rlen < n_hdr) use fixed pos 4; long rows use
            # header index.
            import csv as _csv_p, io as _io_p
            none_trials = set()
            none_row_count = 0
            mode_counts = {m: 0 for m in _pmodes}
            if os.path.exists(fpath):
                try:
                    with open(fpath, 'rb') as _fhp:
                        _rawp = _fhp.read()
                    _cp = _rawp.decode('utf-8', errors='replace').replace('\r\n','\n').replace('\r','\n')
                    _rowsp = list(_csv_p.reader(_io_p.StringIO(_cp)))
                    if len(_rowsp) >= 2:
                        _hdrp  = _rowsp[0]
                        _rmp   = _hdrp.index('run_mode')   if 'run_mode'  in _hdrp else -1
                        _prp   = _hdrp.index('priming')    if 'priming'   in _hdrp else -1
                        _pmhp  = _hdrp.index(_pcol)        if _pcol       in _hdrp else -1
                        _tihp  = _hdrp.index('trial')      if 'trial'     in _hdrp else -1
                        for _rowp in _rowsp[1:]:
                            _rlenp = len(_rowp)
                            if _rmp >= 0 and (_rmp >= _rlenp or not run_mode_matches(_rowp[_rmp], run_num)): continue
                            if _prp >= 0 and _prp < _rlenp and _rowp[_prp] == '1': continue
                            # Read patch mode from header column (works for all row widths)
                            _pmp = None
                            if _pmhp >= 0 and _pmhp < _rlenp:
                                _pmp = _rowp[_pmhp].strip() or None
                            # Skip NA sentinel
                            if _pmp in (None, 'NA', ''):
                                continue
                            _tp = None
                            if _tihp >= 0 and _tihp < _rlenp:
                                try: _tp = int(float(_rowp[_tihp]))
                                except (ValueError, TypeError): pass
                            if _pmp in mode_counts:
                                mode_counts[_pmp] += 1
                            if _pmp == 'none':
                                none_row_count += 1
                                if _tp is not None:
                                    none_trials.add(_tp)
                except Exception:
                    pass
            # Use distinct trial count if available (trial not nan),
            # otherwise fall back to row count // turns (R21/R42 have trial=nan).
            n_none = len(none_trials) if none_trials else none_row_count // 13
            all_p_done = n_none >= n_trials
            any_p_started = n_none > 0
            _p_st = 'done' if all_p_done else 'missing' if not any_p_started else 'partial'
            # per-mode expected row count: n_trials × turns (for complete runs)
            per_mode_expected = n_trials * _pt
            mode_rows = [{'mode': m, 'n': mode_counts[m],
                          'done': mode_counts[m] >= per_mode_expected,
                          'pct': min(100, int(mode_counts[m]*100/per_mode_expected)) if per_mode_expected else 0}
                         for m in _pmodes]
            return jsonify({'run': run_num, 'desc': desc, 'status': _p_st,
                            'n_trials': n_trials, 'n_complete': n_none,
                            'expected_turns': _pt,
                            'n_partial': 0, 'n_missing': 0, 'partial': [], 'missing': [],
                            'is_patching_run': True,
                            'patch_col': _pcol,
                            'patch_modes': mode_rows,
                            'mode_counts': mode_counts,
                            'patch_turns': _pt,
                            'patch_per_mode_expected': per_mode_expected})

        # ── Run 0003: temperature grid 4×5 (BLOCKING) ─────────────────────────
        # fix — condition and temperature read from correct turn-row positions.
        if run_num == 3:
            _TEMPS_3 = [0.2, 0.4, 0.6, 0.8, 1.0]
            _CONDS_3 = ['introspection', 'null', 'arithmetic', 'neutral_prime']
            _CONDS_SET = set(_CONDS_3)
            _TEMPS_SET = {round(t,1) for t in _TEMPS_3}
            cell_max = {}
            if os.path.exists(fpath):
                try:
                    import csv as _csv26, io as _io26
                    with open(fpath,'rb') as _fh: _raw=_fh.read()
                    _c=_raw.decode('utf-8',errors='replace').replace('\r\n','\n').replace('\r','\n')
                    _rows=list(_csv26.reader(_io26.StringIO(_c)))
                    if len(_rows)>=2:
                        _hdr=_rows[0]; _nhdr=len(_hdr)
                        _rm=_hdr.index('run_mode')   if 'run_mode'     in _hdr else -1
                        _pr=_hdr.index('priming')    if 'priming'      in _hdr else -1
                        # v50: read by column name — all CSVs normalized to canonical schema
                        _TI  = _hdr.index('trial')                if 'trial'                in _hdr else -1
                        _CO  = _hdr.index('condition')            if 'condition'             in _hdr else -1
                        _TE  = _hdr.index('temperature_condition') if 'temperature_condition' in _hdr else -1
                        _RM  = _hdr.index('run_mode')             if 'run_mode'              in _hdr else -1
                        _PR  = _hdr.index('priming')              if 'priming'               in _hdr else -1
                        for _r in _rows[1:]:
                            _rlen=len(_r)
                            if _RM>=0 and _RM<_rlen and _r[_RM]!='26': continue
                            if _PR>=0 and _PR<_rlen and _r[_PR]=='1':  continue
                            if any(i<0 or i>=_rlen for i in [_TI,_CO,_TE]): continue
                            try:
                                _cv=_r[_CO]; _tn=int(float(_r[_TI]))
                                # v0.66.3.0: temperature_condition is "{cond}@{temp}" not a bare float.
                                # Parse temperature from the composite string.
                                _te_raw=_r[_TE]
                                if '@' in _te_raw:
                                    _tv=round(float(_te_raw.split('@',1)[1]),1)
                                else:
                                    _tv=round(float(_te_raw),1)
                            except (ValueError,TypeError): continue
                            if _cv not in _CONDS_SET or _tv not in _TEMPS_SET: continue
                            _k=(_cv,_tv)
                            if _k not in cell_max or _tn>cell_max[_k]: cell_max[_k]=_tn
                except Exception: pass
            grid=[]; n_done_cells=0
            for _cond in _CONDS_3:
                row_data=[]
                for _temp in _TEMPS_3:
                    _k=(_cond,round(_temp,1)); _n=cell_max.get(_k,-1)+1
                    _dn=_n>=n_trials
                    if _dn: n_done_cells+=1
                    row_data.append({'cond':_cond,'temp':round(_temp,1),'n':_n,'done':_dn,
                                     'pct':min(100,int(_n*100/max(n_trials,1)))})
                grid.append({'cond':_cond,'cells':row_data})
            n_total_cells=len(_CONDS_3)*len(_TEMPS_3)
            all_done_26=n_done_cells==n_total_cells
            any_started=any(v>=0 for v in cell_max.values())
            _st26='done' if all_done_26 else 'missing' if not any_started else 'partial'
            return jsonify({'run':run_num,'desc':desc,'status':_st26,
                            'n_trials':n_trials*n_total_cells,
                            'n_complete':n_done_cells*n_trials,
                            'run3_grid':grid,'run3_temps':_TEMPS_3,'run3_conds':_CONDS_3,
                            'run3_cells_done':n_done_cells,'run3_cells_total':n_total_cells,
                            'is_blocking':True})

                # ── Multi-condition runs (Run 0002, 0023, 0024, etc.) ──────────────────────
        # These runs loop over multiple conditions. n_trials is per-condition;
        # total expected = n_trials × n_conditions. Report per-condition progress
        # so the popup shows real status instead of "100/100 done" after T=0.0.
        import start_here as _sh2
        _MC = getattr(_sh2, '_MC_RUNS', {})
        if run_num in _MC and run_num != 1:
            cond_col, cond_vals = _MC[run_num]
            n_conds   = len(cond_vals)
            n_total   = n_trials * n_conds
            is_float  = cond_vals and isinstance(cond_vals[0], float)
            _cond_set = {round(float(v),1) if is_float else str(v) for v in cond_vals}
            mc_status = []
            n_done    = 0
            cond_max  = {}
            if os.path.exists(fpath):
                # v50: read by column name — all CSVs normalized to canonical schema
                import csv as _csv_mc2, io as _io_mc2
                with open(fpath,'rb') as _fhmc2: _rawmc2=_fhmc2.read()
                _cmc2=_rawmc2.decode('utf-8',errors='replace').replace('\r\n','\n').replace('\r','\n')
                _rmc2=list(_csv_mc2.reader(_io_mc2.StringIO(_cmc2)))
                if len(_rmc2)>=2:
                    _hmc2=_rmc2[0]; _nhmc2=len(_hmc2)
                    _rmmc2=_hmc2.index('run_mode') if 'run_mode' in _hmc2 else 0
                    _prmc2=_hmc2.index('priming')  if 'priming'  in _hmc2 else 2
                    _thmc2=_hmc2.index('trial')    if 'trial'    in _hmc2 else -1
                    _cchdr2=_hmc2.index(cond_col)  if cond_col   in _hmc2 else -1
                    # Use distinct trial count per condition rather than max+1.
                    # Global trial values (R20: 0-499) would be cut off by
                    # the old 0<=trial<n_trials filter, showing only condition 0.
                    cond_trials = {cv: set() for cv in _cond_set}
                    for _r2 in _rmc2[1:]:
                        _rlen2=len(_r2)
                        if _prmc2<_rlen2 and _r2[_prmc2]=='1': continue
                        if _rmmc2<_rlen2 and _r2[_rmmc2]!=str(run_num): continue
                        if _thmc2<0 or _thmc2>=_rlen2: continue
                        try: _tn2=int(float(_r2[_thmc2]))
                        except (ValueError,TypeError): continue
                        if _cchdr2<0 or _cchdr2>=_rlen2: continue
                        try: _cv2=round(float(_r2[_cchdr2]),1) if is_float else _r2[_cchdr2]
                        except (ValueError,TypeError): continue
                        if _cv2 not in _cond_set: continue
                        cond_trials.setdefault(_cv2, set()).add(_tn2)

            if 'cond_trials' not in dir():
                cond_trials = {}
            for _cv in cond_vals:
                _k = round(float(_cv), 1) if is_float else _cv
                _n = len(cond_trials.get(_k, set()))
                _done = _n >= n_trials
                if _done: n_done += 1
                mc_status.append({'condition': str(_cv), 'n': _n,
                                   'done': _done, 'pct': min(100, int(_n * 100 / n_trials))})
            all_mc_done = n_done == n_conds
            any_mc_started = any(c['n'] > 0 for c in mc_status)
            _mc_st = 'done' if all_mc_done else 'missing' if not any_mc_started else 'partial'
            # v0.79.4.9: MC-run ET short-circuit removed. Source runs
            # (including Run 0002) report plain done/partial/missing based
            # on CSV only. ET state for all sources aggregated into Run 0016.
            return jsonify({'run': run_num, 'desc': desc, 'status': _mc_st,
                            'n_trials': n_total, 'n_complete': n_done * n_trials,
                            'expected_turns': None, 'n_partial': 0, 'n_missing': 0,
                            'partial': [], 'missing': [],
                            'mc_conditions': mc_status,
                            'mc_cond_col': cond_col,
                            'mc_per_cond': n_trials,
                            'is_et_run': run_num in {20}})

        # ── Standard runs ─────────────────────────────────────────────────────
        if not os.path.exists(fpath):
            return jsonify({'run': run_num, 'desc': desc, 'status': 'missing',
                            'n_trials': n_trials, 'n_complete': 0,
                            'n_partial': 0, 'n_missing': n_trials,
                            'partial': [], 'missing': list(range(min(5, n_trials))),
                            'expected_turns': None})
        from orchestration_core import canonical_read as _cr_std
        cols = _cr_std(fpath, ('run_mode', 'trial', 'priming'))
        trial_turns = {}
        # v0.79.4.9: dual-accept canonical 4-digit and legacy integer-string
        from cartography import run_mode_matches as _rmm_chk_ep
        for i, t in enumerate(cols.get('trial', [])):
            if cols.get('priming') and cols['priming'][i] == '1':
                continue
            if cols.get('run_mode') and not _rmm_chk_ep(cols['run_mode'][i], run_num):
                continue
            try:
                tn = int(t)
                trial_turns[tn] = trial_turns.get(tn, 0) + 1
            except Exception:
                pass
        if not trial_turns:
            return jsonify({'run': run_num, 'desc': desc, 'status': 'missing',
                            'n_trials': n_trials, 'n_complete': 0,
                            'n_partial': 0, 'n_missing': n_trials,
                            'partial': [], 'missing': list(range(min(5, n_trials))),
                            'expected_turns': None})
        expected   = max(trial_turns.values())
        partial    = sorted([{'trial': k, 'turns': v, 'expected': expected}
                              for k, v in trial_turns.items() if v < expected],
                            key=lambda x: x['trial'])
        max_seen   = max(trial_turns.keys()) if trial_turns else 0
        all_trials = set(range(max(n_trials, max_seen + 1)))
        missing    = sorted(all_trials - set(trial_turns.keys()))
        n_complete = sum(1 for v in trial_turns.values() if v >= expected)
        csv_done   = n_complete >= n_trials
        # v0.79.4.9: per-source ET short-circuit removed. Source runs
        # (1-9, 15-17, 20) report plain done/partial/missing based on CSV.
        # ET coverage now aggregated into Run 0016 (E_t recovery meta-run).
        status = 'done' if csv_done else 'partial' if trial_turns else 'missing'
        has_extra = bool(trial_turns) and max_seen >= n_trials and (max_seen + 1) > n_complete
        # BUG-RC-POPUP fix (v45.0.1): run_detail_ep never checked for needs_rc status.
        # _scan_runs correctly detected it (grid cell went red) but the popup read
        # status='done' because csv_done=True. Run button rendered dimmed — unclickable.
        # Fix: mirror the same header check from _scan_runs. If disruption_flag is absent from
        # the header on a csv_done run in _RC_RUNS, override status to needs_rc.
        # The popup JS then enables the Run button and shows a re-collect note.
        # v0.79.4.0: old {10,11,22} → new {39,40,28} (perturbation A/B + self-reference)
        _RC_POPUP_RUNS = {39, 40, 28}
        if status == 'done' and run_num in _RC_POPUP_RUNS:
            try:
                import csv as _csv_rc, io as _io_rc
                with open(fpath, 'rb') as _fh_rc:
                    _raw_rc = _fh_rc.read()
                _c_rc = _raw_rc.decode('utf-8', errors='replace').replace('\r\n','\n').replace('\r','\n')
                _hdr_rc = next(iter(_csv_rc.reader(_io_rc.StringIO(_c_rc))), [])
                if 'disruption_flag' not in _hdr_rc:
                    status = 'needs_rc'
            except Exception:
                pass  # can't read header — leave status as-is
        # ── Summary stats (mean ± std for key metrics) ──────────────────────
        _summary = {}
        try:
            if os.path.exists(fpath):
                import csv as _csv_ss, io as _io_ss, math as _math
                with open(fpath, 'rb') as _fss:
                    _raw_ss = _fss.read()
                _c_ss = _raw_ss.decode('utf-8', errors='replace').replace('\r\n','\n').replace('\r','\n')
                _rows_ss = list(_csv_ss.reader(_io_ss.StringIO(_c_ss)))
                if len(_rows_ss) >= 2:
                    _hdr_ss = _rows_ss[0]
                    _metrics = {'sim_index': 'state_similarity_index', 'sim': 'layer_sim_mean', 'disrupt': 'disruption_magnitude'}
                    _mi = {}
                    for mk, col in _metrics.items():
                        if col in _hdr_ss:
                            _mi[mk] = _hdr_ss.index(col)
                    _pri = _hdr_ss.index('priming') if 'priming' in _hdr_ss else -1
                    _vals = {mk: [] for mk in _mi}
                    for _r_ss in _rows_ss[1:]:
                        if _pri >= 0 and _pri < len(_r_ss) and _r_ss[_pri] == '1':
                            continue
                        for mk, idx in _mi.items():
                            if idx < len(_r_ss):
                                try:
                                    v = float(_r_ss[idx])
                                    if _math.isfinite(v):
                                        _vals[mk].append(v)
                                except (ValueError, TypeError):
                                    pass
                    for mk, vs in _vals.items():
                        if vs:
                            _mean = sum(vs) / len(vs)
                            _std = (sum((x - _mean)**2 for x in vs) / len(vs)) ** 0.5
                            _summary[mk] = {'mean': round(_mean, 6), 'std': round(_std, 6), 'n': len(vs)}
        except Exception:
            pass

        return jsonify({'run': run_num, 'desc': desc, 'status': status,
                        'n_trials': n_trials, 'n_complete': n_complete,
                        'expected_turns': expected,
                        'n_partial': len(partial), 'n_missing': len(missing),
                        'partial': partial[:25], 'missing': missing[:25],
                        'has_extra': has_extra,
                        'summary': _summary,
                        # v0.79.4.0: ET-runs post-renumber = {1, 4..15}
                        'is_et_run': run_num in {1,4,5,6,7,8,9,10,11,12,13,14,15}})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/timers", methods=["GET","POST"])
def timers_ep():
    """Compute timer values from CSV timestamps — ground truth, no drift.

    v0.59.1.0: rewritten. Old implementation used client-side JS accumulators
    persisted to .iota_timers.json — drifted across sessions, showed wrong
    values for model time, collection time, and run time.

    New: all values computed server-side from CSV `timestamp` columns.
    Per-trial duration = max(timestamp) - min(timestamp). Summed across trials
    and runs. Cached by CSV file mtimes — cheap to poll every 3 seconds.

    POST: no-op (backward compat — JS no longer saves client-side timers).
    GET:  returns {collection_sec, model_sec, run_label, run_sec, active}.
    """
    if request.method == "POST":
        return jsonify({"ok": True})  # no-op — server computes everything now

    import time as _time
    from cartography import DATA, get_family_size_dir

    s = _rsess()
    family  = s.get('model_family', 'llama')
    size    = s.get('model_size', '8b')
    variant = s.get('model_variant', 'abliterated')
    cur_temp = float(s.get('temperature', 0.0))
    base_dir = os.path.join(get_family_size_dir(family, size), variant)

    collection_sec = 0.0   # current temperature round
    model_sec      = 0.0   # all temperature rounds for this model
    run_label      = ""
    run_sec        = 0.0
    active         = False

    # ── Per-directory collection time from CSV timestamps ────────────
    # Cached by (filepath, mtime) — only re-parses files that changed since last poll.
    # Avoids reading 100K+ CSV rows every 3 seconds during active collection.
    _timer_cache = getattr(timers_ep, '_cache', {})  # persists across calls via function attribute
    timers_ep._cache = _timer_cache

    def _file_collection(fpath):
        """Sum per-trial durations from one CSV. Cached by mtime.
        Active files change mtime every append — min 10s between re-parses."""
        import csv as _csv
        try:
            mt = os.path.getmtime(fpath)
        except OSError:
            return 0.0
        cache_key = fpath
        if cache_key in _timer_cache:
            cached_mt, cached_val, cached_ts = _timer_cache[cache_key]
            if cached_mt == mt:
                return cached_val
            # mtime changed but re-parsed recently — return stale value
            if (_time.time() - cached_ts) < 10.0:
                return cached_val
        now = _time.time()
        total = 0.0
        try:
            with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                reader = _csv.DictReader(f)
                if 'timestamp' not in (reader.fieldnames or []):
                    _timer_cache[cache_key] = (mt, 0.0, now)
                    return 0.0
                trials = {}
                _has_ft = 'file_trial' in (reader.fieldnames or [])
                for row in reader:
                    try:
                        ts = float(row.get('timestamp', 0))
                        if ts <= 0:
                            continue
                        # file_trial is offset-encoded (unique across conditions).
                        # trial reuses numbers across conditions in multi-condition
                        # runs (26, 28, 30, etc) — keying on trial inflates duration
                        # by spanning across collection sessions (8000+ hours).
                        _tid = row.get('file_trial', '') if _has_ft else ''
                        if not _tid or _tid == 'NA':
                            _tid = row.get('trial', '')
                        key = (row.get('run_mode', ''), _tid)
                        if key not in trials:
                            trials[key] = [ts, ts]
                        else:
                            if ts < trials[key][0]: trials[key][0] = ts
                            if ts > trials[key][1]: trials[key][1] = ts
                    except (ValueError, TypeError):
                        continue
                for (mn, mx) in trials.values():
                    total += mx - mn
        except Exception:
            pass
        _timer_cache[cache_key] = (mt, total, now)
        return total

    def _dir_collection(csv_dir):
        """Sum per-trial durations from all CSVs in a directory."""
        if not os.path.isdir(csv_dir):
            return 0.0
        total = 0.0
        for fname in os.listdir(csv_dir):
            if not fname.endswith('.csv'):
                continue
            total += _file_collection(os.path.join(csv_dir, fname))
        return total

    # Current temperature directory
    _COND_MAP = {0.0: 'deterministic', 0.2: 'temp_0.2', 0.4: 'temp_0.4',
                 0.6: 'temp_0.6', 0.8: 'temp_0.8', 1.0: 'temp_1.0'}
    cur_cond = _COND_MAP.get(cur_temp, 'deterministic')
    cur_csv = os.path.join(base_dir, cur_cond, 'csv')
    collection_sec = _dir_collection(cur_csv)

    # All temperature directories for model total
    if os.path.isdir(base_dir):
        for cond in os.listdir(base_dir):
            cond_csv = os.path.join(base_dir, cond, 'csv')
            if os.path.isdir(cond_csv):
                model_sec += _dir_collection(cond_csv)

    # Current active run — from status file
    try:
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, encoding='utf-8') as f:
                st = json.load(f)
            rn = st.get('run_num')
            rts = st.get('run_start_ts')
            if rn and rts and rts > 0:
                active = True
                run_label = f"Run {rn}"
                run_sec = _time.time() - rts
    except Exception:
        pass

    # Idle = server uptime minus time spent running (data + analysis)
    uptime = _time.time() - _SERVER_START
    _cur_active = (_time.time() - _last_active_start) if _last_active_start else 0.0
    total_active = _active_total + _cur_active
    idle_sec = max(0.0, uptime - total_active)
    duty_cycle = total_active / uptime if uptime > 0 else 0.0

    # Session = time since /run was fired
    session_sec = 0.0
    if _session_start_ts and _session_start_ts > 0:
        session_sec = _time.time() - _session_start_ts

    return jsonify({
        "total_sec":      round(model_sec, 1),
        "session_sec":    round(session_sec, 1),
        "run_label":      run_label,
        "run_sec":        round(run_sec, 1),
        "idle_sec":       round(idle_sec, 1),
        "duty_cycle":     round(duty_cycle, 4),
        "active":         active,
        "collection_sec": round(collection_sec, 1),
        "model_sec":      round(model_sec, 1),
    })


@app.route("/hyp_deps")
def hyp_deps_ep():
    """Return hypothesis->runs dependency data for the hypothesis tab.
    Returns {HID: {name, phase, data_runs, upstream_runs}} for all hypotheses."""
    try:
        import dependency_map as dm
        out = {}
        for hid, h in dm.DEPENDENCY_MAP.items():
            out[hid] = {
                "name":          h.get("name", ""),
                "phase":         h.get("phase", 0),
                "data_runs":     h.get("data_runs", []),
                "upstream_runs": h.get("upstream_runs", []),
            }
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/prereq_check")
def prereq_check_ep():
    try:
        import start_here as sh
        from cartography import get_paths
        s = _rsess()
        runs_spec = request.args.get('runs', s.get('runs', '1-45'))
        # v37.1: fallback updated from '1-39,41,42' — missing runs 0051, 0025, 0026
        runs = sh._order_runs(sh._parse_runs(runs_spec))
        paths = get_paths(s.get('model_family','llama'), s.get('model_size','8b'),
                          s.get('model_variant','abliterated'), s.get('temperature',0.0),
                          create_dirs=False)
        status = sh._scan_runs(paths, s.get('trials', 100))
        problems = {}
        for r in runs:
            ps = sh._check_prereqs(r, status)
            if ps:
                problems[str(r)] = [
                    {"prereq": pnum, "status": pstat, "desc": desc}
                    for pnum, pstat, desc in ps
                ]
        return jsonify({"ok": True, "problems": problems})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/descs")
def descs_ep(): return jsonify({str(k):v for k,v in _descs().items()})

@app.route("/whatnext")
def wn_ep(): return jsonify({"text":_whatnext()})

@app.route("/rawlog")
def rawlog_ep():
    """Return last N lines of .iota_flask.log — raw subprocess stdout.
    Surfaces model loading, errors, tracebacks, anything before JSONL starts.
    Seeks from end of file instead of reading entire contents."""
    n = int(request.args.get('n', 120))
    try:
        log = os.path.join(ROOT, '.iota_flask.log')
        if os.path.exists(log):
            size = os.path.getsize(log)
            # Read last 64KB — enough for ~120 lines
            chunk = min(size, 65536)
            with open(log, 'rb') as f:
                f.seek(max(0, size - chunk))
                tail = f.read().decode('utf-8', errors='replace')
            lines = tail.splitlines()[-n:]
            return jsonify({"lines": lines, "size": size})
    except Exception as e:
        return jsonify({"lines": [f"(rawlog error: {e})"], "size": 0})
    return jsonify({"lines": [], "size": 0})

@app.route("/")
def index(): return Response(DASH, mimetype='text/html',
    headers={'Cache-Control': 'no-cache, no-store, must-revalidate',
             'Pragma': 'no-cache', 'Expires': '0'})

# ── Dashboard HTML ─────────────────────────────────────────────────────────────
# UI rebuilt v37.2 (label pass + syntax fix). v37.7 fixes critical JS SyntaxError
# caused by Python processing \n escape in triple-quoted DASH string to a literal
# newline character, which was then embedded inside JS single-quoted string literals
# in addL() calls — Firefox/Chrome rejected the entire <script> block, leaving every
# button undefined.  Fixed by doubling the backslash (\\n) so Python renders \n as
# two characters (valid JS escape), not a raw newline (invalid in JS string literals).

DASH = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IOTA</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#0a0c0f;--bg2:#10141a;--bg3:#161c24;--b1:#1e2830;--b2:#2a3540;
  --t:#c8d4dc;--t2:#6a8090;--t3:#3a5060;
  --ac:#00d4aa;--bl:#0088ff;--wa:#ff8c42;--ye:#f5d000;--er:#ff4455;--go:#44dd88;
  --fn:'IBM Plex Mono',monospace;
  --lc:30%;--lh:30%}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--t);font-family:var(--fn);font-size:12px;
  line-height:1.5;height:100vh;display:flex;flex-direction:column;overflow:hidden}

/* ── Toolbar (fixed) ── */
#tb{display:flex;align-items:stretch;gap:0;padding:0;
  background:var(--bg2);border-bottom:1px solid var(--b1);flex-shrink:0;min-height:64px;z-index:10}
.tb-left{display:flex;flex-direction:column;justify-content:center;gap:3px;
  flex-shrink:0;padding:8px 14px;min-width:90px}
.logo{font-weight:600;font-size:14px;letter-spacing:.20em;color:var(--ac);
  text-transform:uppercase;line-height:1}
.logo span{color:var(--t2);font-weight:300;font-size:9px;display:block;
  letter-spacing:.12em;margin-top:2px}
.tb-status{display:flex;align-items:center;gap:5px;margin-top:3px}
.tb-temp{font-size:9px;font-weight:600;letter-spacing:.05em;color:var(--ac);
  background:rgba(68,221,136,.1);padding:1px 5px;border-radius:3px;margin-left:4px}
.dot{width:7px;height:7px;border-radius:50%;background:var(--t3);flex-shrink:0;transition:background .3s}
.dot.run{background:var(--ac);box-shadow:0 0 6px var(--ac);animation:dp 1.4s infinite}
.dot.pau{background:var(--wa)}
@keyframes dp{0%,100%{opacity:1}50%{opacity:.3}}
#stx{font-size:9px;color:var(--t2);flex-shrink:0;letter-spacing:.06em;text-transform:uppercase}
/* Clocks — order: Total > Model > Session > Run */
.tb-clocks{display:flex;flex-direction:column;justify-content:center;gap:3px;
  flex-shrink:0;padding:8px 16px}
.timer-row{display:flex;gap:7px;align-items:baseline;
  font-variant-numeric:tabular-nums;white-space:nowrap}
.timer-lbl{color:var(--t3);min-width:54px;letter-spacing:.06em;font-size:8px;
  text-transform:uppercase}
.timer-val{color:var(--t2);font-size:11px}
.timer-val.live{color:var(--ac);font-size:11px}
/* Centre: model + run info */
.tb-mid{flex:1;display:flex;flex-direction:column;justify-content:center;
  gap:2px;min-width:0;padding:4px 16px}
.tb-context{display:flex;align-items:baseline;justify-content:center;gap:4px;flex-wrap:wrap}
.tb-pcts{display:flex;gap:20px;justify-content:center;align-items:baseline}
.tb-pct{text-align:center}
.tb-pct-val{font-size:22px;font-weight:700;color:var(--t1);line-height:1;font-variant-numeric:tabular-nums}
.tb-pct-lbl{font-size:7px;color:var(--t3);text-transform:uppercase;letter-spacing:.08em;margin-top:2px}
.tb-run-info{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap;justify-content:center}
.tb-run-info span{font-size:8px;color:var(--t3);white-space:nowrap}
/* Right: tall action buttons */
.tb-right{display:flex;align-items:stretch;gap:0;flex-shrink:0}
.tb-btn-tall{font-family:var(--fn);font-size:9px;font-weight:500;letter-spacing:.10em;
  padding:0 14px;background:transparent;border:none;border-left:1px solid var(--b1);
  color:var(--t2);cursor:pointer;transition:all .15s;text-transform:uppercase;
  white-space:nowrap;flex-shrink:0;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:4px;min-width:58px}
.tb-btn-tall:hover{background:var(--bg3);color:var(--t)}
.tb-btn-tall:focus-visible{outline:2px solid var(--ac);outline-offset:-2px}
.tb-btn-tall:disabled{opacity:.3;cursor:not-allowed;pointer-events:none}
.tb-btn-tall .tb-btn-icon{font-size:16px;line-height:1}
.tb-btn-tall .tb-btn-lbl{font-size:8px;letter-spacing:.10em}
.tb-btn-tall.go{border-top:2px solid var(--go);color:var(--go)}
.pr-all-stats{
  background:linear-gradient(135deg,var(--ac),#6366f1)!important;color:#fff!important;
  font-weight:600!important;letter-spacing:.08em;border:none!important;
}
.pr-all-stats:hover{opacity:.85}
.pr-all-stats:disabled{opacity:.3;cursor:not-allowed;pointer-events:none}
.tb-btn-tall.go:hover{background:rgba(68,221,136,.08)}
.tb-btn-tall.pa{border-top:2px solid var(--t2);color:var(--t2)}
.tb-btn-tall.pa:hover{background:rgba(200,210,220,.08)}
.tb-btn-tall.ab{border-top:2px solid var(--er);color:var(--er)}
.tb-btn-tall.ab:hover{background:rgba(255,68,85,.08)}
.tb-btn-tall.st{border-top:2px solid var(--er);color:var(--er)}
.tb-btn-tall.st:hover{background:rgba(255,68,85,.08)}
.btn{font-family:var(--fn);font-size:10px;font-weight:500;letter-spacing:.08em;
  padding:4px 11px;background:transparent;border:1px solid var(--b2);color:var(--t2);
  cursor:pointer;transition:all .15s;text-transform:uppercase;white-space:nowrap;flex-shrink:0}
.btn:hover{border-color:var(--t);color:var(--t)}
.btn:focus-visible{outline:2px solid var(--ac);outline-offset:2px}
.btn:disabled{opacity:.3;cursor:not-allowed;pointer-events:none}
[title]{cursor:help}
.btn[title],.pr[title],.cb[title],.sev[title]{cursor:pointer}
.btn.go{border-color:var(--go)!important;color:var(--go)!important}
.btn.go:hover{background:rgba(68,221,136,.15)!important;color:var(--go)!important}
.btn.pa{border-color:var(--wa)!important;color:var(--wa)!important}
.btn.st{border-color:var(--er)!important;color:var(--er)!important}
.btn.st:hover{background:var(--er)!important;color:#fff!important}
.btn.ye{border-color:var(--ye)!important;color:var(--ye)!important}
.btn.ye:hover{background:var(--ye)!important;color:var(--bg)!important}
.btn.er{border-color:var(--er)!important;color:var(--er)!important}
.btn.er:hover{background:var(--er)!important;color:#fff!important}
.nuke-btn{font-family:var(--fn);font-size:11px;font-weight:700;letter-spacing:.06em;
  padding:6px 14px;border:2px solid rgba(255,68,85,.5);color:var(--er);cursor:pointer;
  background:rgba(255,68,85,.08);border-radius:4px;white-space:nowrap;
  transition:all .2s;text-transform:uppercase}
.nuke-btn:hover{background:rgba(255,68,85,.25);border-color:var(--er);transform:scale(1.03)}
.nuke-btn.mega{font-size:13px;padding:8px 18px;border-width:3px;animation:nukePulse 1.5s infinite;
  background:rgba(255,68,85,.15);text-shadow:0 0 8px rgba(255,68,85,.4)}
@keyframes nukePulse{0%,100%{border-color:rgba(255,68,85,.5)}50%{border-color:rgba(255,68,85,1)}}
.qm{display:inline-block;width:14px;height:14px;line-height:14px;text-align:center;
  font-size:9px;border-radius:50%;background:rgba(160,170,180,.15);color:var(--t3);
  cursor:help;margin-left:4px;vertical-align:middle;font-style:normal;font-weight:600}
.btn.bl{border-color:var(--bl)!important;color:var(--bl)!important}

/* ── Tiling layout ── */
/* Three panels: left-col (grid top + metrics bottom) | vspl | right-col (console/tabs) */
#workspace{display:flex;flex:1;overflow:hidden;min-height:0}
#left-col{width:var(--lc);flex-shrink:0;display:flex;flex-direction:column;
  overflow:hidden;min-width:180px}
#panel-grid{flex:var(--lh);overflow:hidden;display:flex;flex-direction:column;
  min-height:120px;background:var(--bg2);border-right:1px solid var(--b1)}
#panel-metrics{flex:calc(1 - var(--lh));overflow:hidden;display:flex;flex-direction:column;
  min-height:100px;background:var(--bg2);border-right:1px solid var(--b1);
  border-top:1px solid var(--b1)}
/* Horizontal splitter between grid and metrics (inside left col) */
#hspl{height:4px;background:var(--b1);cursor:row-resize;flex-shrink:0;
  display:flex;align-items:center;justify-content:center;user-select:none;
  transition:background .15s;border-right:1px solid var(--b1)}
#hspl:hover,#hspl.drag{background:var(--ac)}
#hspl::after{content:'';display:block;height:1px;width:32px;background:var(--b2);border-radius:1px}
#hspl:hover::after,#hspl.drag::after{background:var(--ac)}
/* Vertical splitter between left col and right col */
#vspl{width:4px;background:var(--b1);cursor:col-resize;flex-shrink:0;
  display:flex;align-items:center;justify-content:center;user-select:none;
  transition:background .15s}
#vspl:hover,#vspl.drag{background:var(--ac)}
#vspl::after{content:'';display:block;width:1px;height:32px;background:var(--b2);border-radius:1px}
#vspl:hover::after,#vspl.drag::after{background:var(--ac)}
/* Right panel: console/tabs — takes remaining width */
#panel-right{flex:1;display:flex;flex-direction:column;overflow:hidden;min-width:200px}

/* ── Panel headers ── */
.panel-hdr{display:flex;align-items:center;justify-content:space-between;
  padding:5px 10px;background:var(--bg2);border-bottom:1px solid var(--b1);flex-shrink:0}
.panel-title{font-size:9px;letter-spacing:.14em;color:var(--t3);text-transform:uppercase}
.panel-hdr-right{display:flex;align-items:center;gap:5px}

/* ── Grid panel ── */
#grid-scroll{flex:1;overflow-y:auto;overflow-x:hidden;padding:8px 10px}
.run-order-note{font-size:8px;color:var(--t3);letter-spacing:.08em;text-transform:uppercase;
  margin-bottom:6px;padding-bottom:4px;border-bottom:1px solid var(--b1)}
.model-bar{display:flex;align-items:center;gap:8px;margin-bottom:8px}
.model-bar select{font-family:var(--fn);font-size:10px;padding:3px 8px;background:var(--bg3);
  color:var(--t1);border:1px solid var(--b2);border-radius:3px}
.model-bar .hint{font-size:8px;color:var(--t3);font-style:italic}
/* Navigation breadcrumb */
.nav-bc{display:flex;align-items:center;gap:6px;padding:6px 0;margin-bottom:8px;font-size:11px}
.bc-link{color:var(--ac);cursor:pointer;font-weight:500;padding:2px 6px;border-radius:3px}
.bc-link:hover{background:rgba(0,212,170,.1)}
.bc-cur{color:var(--t1);font-weight:600}
.bc-sep{color:var(--t3);font-size:9px}
.bc-select{font-family:var(--fn);font-size:11px;font-weight:500;padding:2px 8px;
  background:var(--bg3);color:var(--ac);border:1px solid var(--ac);border-radius:3px;cursor:pointer}
/* Layer visibility */
.lyr{display:none}.lyr.on{display:block}
/* Loading overlay — fills parent content area */
.lyr-loading{display:flex;align-items:center;justify-content:center;flex-direction:column;
  gap:10px;padding:40px 20px;color:var(--t3);font-size:13px;min-height:120px}
.lyr-loading .ld-spin{width:28px;height:28px;border:3px solid var(--b2);border-top-color:var(--ac);
  border-radius:50%;animation:ldspin .8s linear infinite}
@keyframes ldspin{to{transform:rotate(360deg)}}
.lyr-loading .ld-text{font-size:11px;color:var(--t3);letter-spacing:.04em}
/* Model cards */
.model-cards{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:16px}
.model-card{background:var(--bg3);border:1px solid var(--b2);border-radius:6px;padding:14px 18px;
  min-width:220px;cursor:pointer;transition:all .15s}
.model-card:hover{border-color:var(--ac);transform:translateY(-1px)}
.model-card.all-models{border-color:var(--ac);background:linear-gradient(135deg,var(--bg3) 0%,rgba(80,140,220,0.06) 100%)}
.model-card.all-models .mc-name{color:var(--ac)}
.model-card.all-models.active{border-color:var(--ac);box-shadow:0 0 0 2px rgba(80,140,220,0.2)}
.model-card .mc-name{font-size:14px;font-weight:600;color:var(--t1);margin-bottom:6px}
.model-card .mc-thead{display:flex;gap:4px;font-size:9px;color:var(--t3);margin-bottom:2px;
  text-transform:uppercase;letter-spacing:.05em}
.model-card .mc-tr{display:flex;gap:4px;font-size:11px;color:var(--t2);padding:1px 0}
.model-card .mc-tr.done{color:var(--go)}
.model-card .mc-tr.part{color:var(--wa)}
.model-card .mc-tr.empty{color:var(--t3);opacity:0.5}
.model-card .mc-tl{min-width:48px;font-family:var(--fn)}
.model-card .mc-td{min-width:52px;text-align:right;font-family:var(--fn)}
.model-card .mc-summary{font-size:11px;font-weight:600;color:var(--t1);margin-top:6px;
  padding-top:6px;border-top:1px solid var(--b2)}
.model-card .mc-summary.done{color:var(--go)}
.tc{display:inline-block;font-size:10px;font-family:var(--fn);padding:3px 7px;border-radius:3px;
  cursor:pointer;border:1px solid var(--b2);color:var(--t2);background:var(--bg3);transition:all .15s;user-select:none}
.tc.on{background:var(--ac);color:#fff;border-color:var(--ac)}
.tc:hover{border-color:var(--ac)}
/* Add model section */
.add-model{border:1px dashed var(--b2);border-radius:6px;padding:12px 16px;margin-top:8px}
.add-model .am-title{font-size:10px;color:var(--t3);text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px}
.add-model .am-row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.add-model select,.add-model button{font-family:var(--fn);font-size:10px;padding:4px 10px;
  background:var(--bg3);color:var(--t1);border:1px solid var(--b2);border-radius:3px}
.add-model button{background:rgba(160,170,180,.2);color:#cdd;border-color:rgba(160,170,180,.4);cursor:pointer;font-weight:500}
.add-model button:hover{background:rgba(160,170,180,.3);color:#eee}
/* Back button */
.back-btn{font-family:var(--fn);font-size:9px;color:var(--ac);background:none;border:1px solid var(--ac);
  border-radius:3px;padding:3px 10px;cursor:pointer;margin-bottom:8px}
.back-btn:hover{background:rgba(0,212,170,.1)}
.set-action{font-size:10px!important;padding:6px 16px!important;font-weight:600;
  background:rgba(160,170,180,.2);border:1px solid rgba(160,170,180,.4);color:#cdd;
  letter-spacing:.04em;border-radius:3px}
.set-action:hover{background:rgba(160,170,180,.3);color:#eee}
/* Grid control buttons */
.grid-controls{display:flex;gap:8px;justify-content:center;margin-top:12px;padding-top:10px;border-top:1px solid var(--b1)}
.gc-btn{flex:1;height:36px;display:flex;align-items:center;justify-content:center;gap:6px;
  font-family:var(--fn);font-size:11px;font-weight:600;letter-spacing:.04em;
  border-radius:4px;cursor:pointer;border:1px solid var(--b2);background:var(--bg3);
  color:var(--t2);transition:all .15s;text-transform:uppercase}
.gc-btn:hover{transform:translateY(-1px)}
.gc-btn.go{background:rgba(68,221,136,.15);border-color:rgba(68,221,136,.4);color:var(--go)}
.gc-btn.go:hover{background:rgba(68,221,136,.25)}
.gc-btn.pa{background:rgba(200,210,220,.1);border-color:rgba(200,210,220,.3);color:var(--t1)}
.gc-btn.pa:hover{background:rgba(200,210,220,.2)}
.gc-btn.rs{background:rgba(200,210,220,.1);border-color:rgba(200,210,220,.3);color:var(--t1)}
.gc-btn.rs:hover{background:rgba(200,210,220,.2)}
.gc-btn:disabled{opacity:.3;cursor:not-allowed;transform:none}
.gc-icon{font-size:13px}
.preset-bar{display:flex;flex-wrap:wrap;gap:3px;margin-bottom:8px}
.pr{font-family:var(--fn);font-size:9px;padding:3px 8px;background:var(--bg3);
  border:1px solid var(--b1);color:var(--t2);cursor:pointer;transition:all .12s;
  white-space:nowrap;letter-spacing:.04em}
.pr:hover{border-color:var(--ac);color:var(--ac)}
.pr:focus-visible{outline:2px solid var(--ac);outline-offset:2px}
.pr.hi{border-color:var(--ac);color:var(--ac);background:rgba(0,212,170,.07)}
.phase-grid{display:flex;flex-direction:column;gap:5px;margin-bottom:7px}
.phase-row{display:flex;flex-direction:column;gap:3px}
.phase-grid.per-temp .phase-row.pooled{display:none}
.phase-lbl{font-size:9px;letter-spacing:.1em;color:var(--t3);text-transform:uppercase}
.phase-lbl.analysis-lbl{color:var(--ac);letter-spacing:.14em}
.rg{display:flex;flex-wrap:wrap;gap:2px}
.rc{width:26px;height:26px;display:flex;align-items:center;justify-content:center;
  font-size:9px;font-weight:500;border:1px solid var(--b1);background:var(--bg3);
  color:var(--t2);cursor:pointer;transition:all .1s;user-select:none}
.rc:hover{border-color:var(--t2);color:var(--t)}
.rc:focus-visible{outline:2px solid var(--ac);outline-offset:2px}
.rc.done{background:rgba(68,221,136,.1);border-color:rgba(68,221,136,.3);color:var(--go)}
.rc.part{background:rgba(255,140,66,.1);border-color:rgba(255,140,66,.4);color:var(--wa)}
.rc.neet{background:rgba(245,208,0,.1);border-color:rgba(245,208,0,.4);color:var(--ye)}
.rc.nrec{background:rgba(255,68,85,.1);border-color:rgba(255,68,85,.4);color:var(--er)}
.rc.miss{background:rgba(58,80,96,.12);border-color:rgba(58,80,96,.5);color:var(--t3)}
.rc.etpt{background:rgba(255,140,66,.1);border-color:rgba(255,140,66,.4);color:var(--wa)}
.rc.sel{border-color:var(--ac)!important;box-shadow:0 0 0 1px var(--ac);color:var(--ac)!important}
.rc.now{animation:rp 1s infinite;border-color:var(--ac)!important;color:var(--ac)!important}
.grid-action{width:56px;height:26px;display:flex;align-items:center;justify-content:center;
  font-size:8px;letter-spacing:.06em;border-radius:3px;cursor:pointer;user-select:none;
  text-transform:uppercase;font-weight:600;transition:all .15s}
.grid-action.stats-btn{background:rgba(100,180,255,.1);border:1px solid rgba(100,180,255,.3);color:var(--ac)}
.grid-action.stats-btn:hover{background:rgba(100,180,255,.2)}
.grid-action.report-btn{background:rgba(221,187,0,.1);border:1px solid rgba(221,187,0,.3);color:var(--ye)}
.grid-action.report-btn:hover{background:rgba(221,187,0,.2)}
.grid-action.disabled{opacity:.3;cursor:not-allowed;pointer-events:none}
/* ── Action bar below grid ── */
@keyframes rp{0%,100%{opacity:1}50%{opacity:.35}}
.legend{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px}
.li{display:flex;align-items:center;gap:3px;font-size:9px;color:var(--t2)}
.ld{width:7px;height:7px;border-radius:1px;flex-shrink:0}
.run-row{display:flex;gap:5px;align-items:center;margin-bottom:4px}
.run-row .si{flex:1;min-width:0}
#selInfo{font-size:9px;color:var(--t2);min-height:13px;letter-spacing:.04em}
#selBox{background:var(--bg3);border:1px solid var(--b2);padding:5px 8px;
  font-size:9px;color:var(--t2);min-height:28px;line-height:1.6;display:none;
  margin-top:4px;word-break:break-all}
#selBox .sel-run{display:inline-block;padding:1px 5px;margin:1px 2px;
  background:rgba(0,212,170,.1);border:1px solid rgba(0,212,170,.4);
  color:var(--ac);cursor:pointer;border-radius:2px;font-size:8px}
#selBox .sel-run:hover{background:rgba(0,212,170,.2)}

/* ── Metrics panel ── */
#metrics-scroll{flex:1;overflow-y:auto;padding:8px 10px}
.expand-btn{background:none;border:1px solid var(--b2);border-radius:3px;color:var(--t3);
  cursor:pointer;font-size:13px;padding:1px 5px;line-height:1;transition:all .15s}
.expand-btn:hover{color:var(--t1);border-color:var(--ac)}
.expand-btn.on{color:var(--ac);border-color:var(--ac)}
.metrics{display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-bottom:8px}
.metric{background:var(--bg3);border:1px solid var(--b1);padding:6px 8px}
.mlab{font-size:10px;font-weight:500;color:var(--t);margin-bottom:1px}
.msub{font-size:8px;letter-spacing:.08em;color:var(--t3);text-transform:uppercase;margin-bottom:2px}
.mval{font-size:17px;font-weight:300;color:var(--ac);font-variant-numeric:tabular-nums;transition:color .4s}
.mval.sm{font-size:13px;padding-top:2px}
.mval.hi{color:var(--go)!important}.mval.lo{color:var(--wa)!important}.mval.vlo{color:var(--er)!important}
.bars{display:flex;flex-direction:column;gap:4px;margin-bottom:8px}
.bar-row{display:flex;flex-direction:column;gap:2px}
.bar-hd{display:flex;justify-content:space-between;font-size:9px;color:var(--t2)}
.bar-track{height:3px;background:var(--bg3);border:1px solid var(--b1);overflow:hidden}
.bar-fill{height:100%;background:var(--ac);transition:width .4s}
.bar-fill.wa{background:var(--wa)}.bar-fill.bl{background:var(--bl)}
.sparks{display:flex;flex-direction:column;gap:5px}
.spk-row{display:flex;flex-direction:column;gap:2px}
.spk-hd{display:flex;justify-content:space-between;font-size:9px;color:var(--t2)}
canvas.spk{width:100%;height:60px;display:block}

/* ── Right panel: tab bar + panes ── */
.tab-bar{display:flex;border-bottom:1px solid var(--b1);flex-shrink:0;
  background:var(--bg2)}
.tab{flex:0 0 auto;padding:7px 16px;font-size:10px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--t2);cursor:pointer;border-bottom:2px solid transparent;
  transition:all .15s;background:none;border-top:none;border-left:none;border-right:none;
  font-family:var(--fn)}
.tab:hover{color:var(--t)}.tab.on{color:var(--ac);border-bottom-color:var(--ac)}
.tab:focus-visible{outline:2px solid var(--ac);outline-offset:-2px}
.pane{display:none;flex-direction:column;flex:1;overflow:hidden;min-height:0}
.pane.on{display:flex}

/* Console pane */
#con{flex:1;overflow-y:auto;padding:10px 13px;font-size:11px;line-height:1.65;
  background:var(--bg);min-height:0}
#con::-webkit-scrollbar{width:3px}
#con::-webkit-scrollbar-thumb{background:var(--b2)}
.ll{white-space:pre;color:var(--t2)}
.ll.turn{color:var(--t)}.ll.ok{color:var(--go)}.ll.warn{color:var(--wa)}
.ll.err{color:var(--er)}.ll.prime{color:var(--bl);opacity:.75}.ll.section{color:var(--t3)}
.ll.raw{color:var(--t3);font-size:10px}
.ll.sep{color:var(--t3);opacity:.5;font-size:10px;letter-spacing:.1em}
.ll .ht{color:var(--ac)}.ll .hs{color:var(--bl)}.ll .hr{color:var(--go);font-weight:600}
#con-load-indicator{font-size:9px;color:var(--ac);text-align:center;padding:6px 0;border-bottom:1px solid var(--b1);cursor:pointer;letter-spacing:.06em;
  letter-spacing:.08em;display:none}
#con-load-indicator.on{display:block}
#con-bottom-btn{flex-shrink:0;text-align:center;padding:5px 0;
  font-size:11px;color:var(--t2);cursor:pointer;background:var(--bg2);border-top:1px solid var(--b1);
  letter-spacing:.06em;user-select:none;z-index:2;font-weight:600}
#con-bottom-btn:hover{color:var(--t1);background:var(--bg3)}

/* Hypothesis pane */
.hyp-hdr{display:flex;align-items:center;justify-content:space-between;
  padding:7px 12px;border-bottom:1px solid var(--b1);flex-shrink:0;background:var(--bg2)}
.hyp-sum{font-size:9px;color:var(--t2)}
.hyp-scroll{flex:1;overflow-y:auto;padding:8px 10px;display:flex;flex-direction:column;gap:6px}
/* Hypothesis phase sections */
.hyp-phase{flex-shrink:0}
.hyp-phase-lbl{font-size:9px;letter-spacing:.12em;color:var(--t2);text-transform:uppercase;
  padding:4px 0 4px 0;border-bottom:1px solid var(--b1);margin-bottom:5px;
  display:flex;justify-content:space-between;align-items:center}
/* Hypothesis rows */
.hyp-row{display:flex;align-items:flex-start;gap:6px;padding:4px 6px;
  border:1px solid var(--b1);background:var(--bg3);margin-bottom:3px;transition:border-color .1s}
.hyp-row:hover{border-color:var(--b2)}
.hyp-row.sel{border-color:var(--ac)}
.hyp-chk{flex-shrink:0;margin-top:2px;accent-color:var(--ac);cursor:pointer}
.hyp-id{font-size:9px;font-weight:600;color:var(--t);min-width:70px;flex-shrink:0}
.hyp-name{font-size:9px;color:var(--t2);flex:1;min-width:0}
.hyp-runs{display:flex;flex-wrap:wrap;gap:2px;align-items:center}
.hyp-run-chip{font-size:8px;padding:1px 4px;border:1px solid var(--b1);
  background:var(--bg);cursor:pointer;transition:all .1s;color:var(--t2)}
.hyp-run-chip:hover{border-color:var(--ac);color:var(--ac)}
.hyp-run-chip.done{border-color:rgba(68,221,136,.3);color:var(--go)}
.hyp-run-chip.part{border-color:rgba(255,140,66,.4);color:var(--wa)}
.hyp-run-chip.miss{border-color:rgba(58,80,96,.5);color:var(--t3)}
.hyp-run-chip.sel{border-color:var(--ac);color:var(--ac);background:rgba(0,212,170,.08)}
/* Queue bar at bottom of hyp panel */
#hyp-queue-bar{flex-shrink:0;padding:6px 10px;border-top:1px solid var(--b1);
  background:var(--bg2);display:flex;align-items:center;gap:6px;min-height:36px}
#hyp-queue-info{font-size:9px;color:var(--t2);flex:1}
.chips{display:flex;flex-wrap:wrap;gap:3px}
.chip{font-family:var(--fn);font-size:9px;padding:3px 7px;border:1px solid var(--b2);
  background:var(--bg3);color:var(--t2);cursor:pointer;transition:all .12s;letter-spacing:.04em}
.chip:hover{border-color:var(--t2);color:var(--t)}
.chip:focus-visible{outline:2px solid var(--ac);outline-offset:2px}
.chip.sel{border-color:var(--ac)!important;color:var(--ac)!important;background:rgba(0,212,170,.08)!important}
.chip.sup{border-color:rgba(68,221,136,.5);color:var(--go);background:rgba(68,221,136,.06)}
.chip.dis{border-color:rgba(255,68,85,.5);color:var(--er);background:rgba(255,68,85,.06)}
.chip.inc{border-color:rgba(255,140,66,.5);color:var(--wa);background:rgba(255,140,66,.06)}
.chip.pen{border-color:var(--b2);color:var(--t2);background:var(--bg3)}
.hyp-det{background:var(--bg3);border-top:2px solid var(--b1);padding:10px 12px;
  font-size:10px;line-height:1.65;color:var(--t2);flex-shrink:0;min-height:52px}
.hid{font-weight:600;color:var(--t);margin-right:6px}
.hstat{font-size:9px;letter-spacing:.12em;text-transform:uppercase;
  padding:1px 5px;border:1px solid;margin-right:6px}
.hstat.sup{border-color:var(--go);color:var(--go)}.hstat.dis{border-color:var(--er);color:var(--er)}
.hstat.inc{border-color:var(--wa);color:var(--wa)}.hstat.pen{border-color:var(--t2);color:var(--t2)}
.hhdr{display:flex;align-items:center;gap:4px;margin-bottom:4px;flex-wrap:wrap}
.hname{font-size:9px;color:var(--t2)}
.himpl{font-size:10px;color:var(--t2);line-height:1.6}

/* Settings pane */
#sp2{overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:11px;flex:1;min-height:0}
.fsect{flex-shrink:0}
.ftitle{font-size:10px;letter-spacing:.1em;color:var(--t2);text-transform:uppercase;
  margin-bottom:7px;padding-bottom:4px;border-bottom:1px solid var(--b1);
  display:flex;justify-content:space-between;align-items:center}
.sg{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.sg3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px}
.frow{display:flex;flex-direction:column;gap:3px}
.flb{font-size:10px;letter-spacing:.08em;color:var(--t2);text-transform:uppercase}
.si{font-family:var(--fn);font-size:11px;background:var(--bg3);border:1px solid var(--b2);
  color:var(--t);padding:5px 8px;width:100%;transition:border-color .15s}
.si:focus{outline:2px solid var(--ac);outline-offset:1px;border-color:var(--ac)}
.tgl{font-family:var(--fn);font-size:10px;background:var(--bg3);border:1px solid var(--b2);
  color:var(--t2);padding:5px 8px;cursor:pointer;transition:all .15s;text-align:left;width:100%}
.tgl:hover{border-color:var(--ac)}.tgl.on{border-color:var(--go);color:var(--go)}
.tgl:focus-visible{outline:2px solid var(--ac);outline-offset:2px}
.mdl-cur{font-size:9px;color:var(--t2);word-break:break-all;margin-bottom:6px;
  line-height:1.4;padding:5px 7px;background:var(--bg3);border:1px solid var(--b1);white-space:pre-wrap}
.save-row{display:flex;gap:6px;margin-top:6px;align-items:center}
.fb{font-size:9px;color:var(--go);opacity:0;transition:opacity .4s}.fb.sh{opacity:1}
.kbd{display:grid;grid-template-columns:1fr 1fr;gap:4px}
.kk{display:flex;gap:5px;align-items:center;font-size:10px;color:var(--t2)}
.kk code{background:var(--b1);border:1px solid var(--b2);padding:1px 5px;
  font-family:var(--fn);font-size:9px;color:var(--t)}

/* ── Run detail popup ── */
#rdpop{display:none;position:fixed;z-index:300;background:var(--bg2);
  border:1px solid var(--b2);min-width:260px;max-width:340px;
  box-shadow:0 8px 32px rgba(0,0,0,.6)}
#rdpop.on{display:block}
.rdp-hd{display:flex;align-items:center;justify-content:space-between;
  padding:9px 12px;border-bottom:1px solid var(--b1);background:var(--bg3);
  cursor:move;user-select:none}
.rdp-title{font-size:11px;font-weight:500;color:var(--t)}
.rdp-close{background:none;border:none;color:var(--t2);cursor:pointer;
  font-size:14px;padding:0 2px;line-height:1}
.rdp-close:hover{color:var(--t)}
.rdp-body{padding:10px 12px;font-size:10px;line-height:1.75;color:var(--t2)}
.rdp-row{display:flex;justify-content:space-between;align-items:center;
  border-bottom:1px solid var(--b1);padding:3px 0;gap:6px}
.rdp-row:last-child{border-bottom:none}
.rdp-lbl{color:var(--t3);letter-spacing:.08em;text-transform:uppercase;flex-shrink:0}
.rdp-val{color:var(--t);font-weight:500}
.rdp-val.ok{color:var(--go)}.rdp-val.wa{color:var(--wa)}.rdp-val.ye{color:var(--ye)}
.rdp-val.er{color:var(--er)}.rdp-val.bl{color:var(--bl)}
.rdp-rval{display:flex;align-items:center;gap:5px}
.patch-mode-row{transition:background .1s}
.r26-cell{transition:border-color .15s,box-shadow .15s}
.patch-mode-row:hover{background:rgba(255,255,255,0.04)}
.patch-mode-row.patch-sel{background:rgba(0,180,200,0.12);border-radius:3px}
.rdp-rbtn{font-family:var(--fn);font-size:8px;padding:1px 6px;background:transparent;
  border:1px solid var(--b2);color:var(--t2);cursor:pointer;transition:all .15s;
  text-transform:uppercase;letter-spacing:.06em;white-space:nowrap;flex-shrink:0}
.rdp-rbtn:hover{border-color:var(--ac);color:var(--ac)}
.rdp-rbtn.go{border-color:var(--go)!important;color:var(--go)!important}
.rdp-rbtn.go:hover{background:var(--go)!important;color:var(--bg)!important}
.rdp-rbtn.ac{border-color:var(--ac)!important;color:var(--ac)!important}
.rdp-rbtn.ac:hover{background:var(--ac)!important;color:var(--bg)!important}
.rdp-rbtn.bl{border-color:var(--bl)!important;color:var(--bl)!important}
.rdp-rbtn.bl:hover{background:var(--bl)!important;color:#fff!important}
.rdp-rbtn.ye{border-color:var(--ye)!important;color:var(--ye)!important}
.rdp-rbtn.ye:hover{background:var(--ye)!important;color:var(--bg)!important}
.rdp-rbtn.wa{border-color:var(--wa)!important;color:var(--wa)!important}
.rdp-rbtn.wa:hover{background:var(--wa)!important;color:var(--bg)!important}
.rdp-rbtn.dim{opacity:0.22!important;cursor:default!important;pointer-events:none!important}
.rdp-foot{padding:8px 12px;border-top:1px solid var(--b1);display:flex;gap:6px}

/* ET popup */
#etpop{display:none;position:fixed;z-index:300;background:var(--bg2);
  border:1px solid var(--b2);min-width:380px;max-width:480px;
  box-shadow:0 8px 32px rgba(0,0,0,.6)}
#etpop.on{display:block}
#etTable tr{border-bottom:1px solid var(--b1)}
#etTable tr:last-child{border-bottom:none}
#etTable td{padding:3px 4px;vertical-align:middle}
#etTable td:first-child{width:18px}
#etTable td:nth-child(2){width:22px}
#etTable td:nth-child(3){max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#etTable .et-status{font-size:9px;padding:1px 5px;border-radius:2px;white-space:nowrap}
#etTable .et-status.neet{color:var(--ye);border:1px solid var(--ye)}
#etTable .et-status.done{color:var(--go);border:1px solid var(--go)}
#etTable .et-status.part{color:var(--er);border:1px solid var(--er)}
#etTable .et-status.etpt{color:var(--wa);border:1px solid var(--wa)}
#etTable .et-status.miss{color:var(--t3);border:1px solid var(--b2)}
.et-mode-btn{font-family:var(--fn);font-size:8px;padding:1px 6px;background:transparent;
  border:1px solid var(--b2);color:var(--t2);cursor:pointer;transition:all .12s;
  text-transform:uppercase;letter-spacing:.06em}
.et-mode-btn:hover{border-color:var(--ac);color:var(--ac)}
.et-mode-btn.sel{background:var(--ac);border-color:var(--ac);color:var(--bg);font-weight:600}
.et-pr{border-color:var(--ye)!important;color:var(--ye)!important}
.et-pr:hover{background:rgba(221,187,0,.15)!important}
.nrec-pr{border-color:rgba(255,68,85,.5)!important;color:var(--er)!important}
.nrec-pr:hover,.nrec-pr.hi{background:rgba(255,68,85,.12)!important}

/* Pause overlay — toolbar banner only, does not block UI */
.pov{display:none;position:relative;background:rgba(247,184,0,.10);
  border-bottom:1px solid var(--wa);padding:4px 12px;
  align-items:center;justify-content:center;gap:10px;z-index:1}
.pov.on{display:flex}
.pt{font-size:11px;letter-spacing:.2em;color:var(--wa);text-transform:uppercase}
.ps{font-size:9px;color:var(--t3);letter-spacing:.12em}

/* Toast + disconnect */
#disc{display:none;position:fixed;bottom:0;left:0;right:0;
  background:rgba(255,68,85,.12);border-top:1px solid var(--er);
  color:var(--er);text-align:center;padding:5px;font-size:10px;z-index:100}
#disc.on{display:block}
#tst{position:fixed;bottom:16px;left:50%;transform:translateX(-50%);
  padding:7px 18px;font-size:10px;z-index:200;white-space:nowrap;display:none}
#tst.ok{background:var(--go);color:var(--bg);display:block}
#tst.er{background:var(--er);color:#fff;display:block}
#tst.wa{background:var(--wa);color:var(--bg);display:block}
#tst.in{background:var(--bl);color:#fff;display:block}

/* Model loading indicator */
.tb-mid.loading #ml{color:var(--t2)}
.tb-mid.loading #ml::after{
  content:'';display:block;height:2px;width:100%;margin-top:4px;
  background:linear-gradient(90deg,transparent 0%,var(--ac) 40%,var(--ac) 60%,transparent 100%);
  background-size:200% 100%;animation:ldbar 1.4s ease-in-out infinite}
@keyframes ldbar{0%{background-position:100% 0}100%{background-position:-100% 0}}

::-webkit-scrollbar{width:3px;height:3px}
::-webkit-scrollbar-track{background:var(--bg2)}
::-webkit-scrollbar-thumb{background:var(--b2)}
</style>
</head>
<body>

<!-- Popups -->
<div id="rdpop" role="dialog" aria-label="Run detail">
  <div class="rdp-hd">
    <span class="rdp-title" id="rdpTitle">Run</span>
    <button class="rdp-close" onclick="closeRdp()" title="Close">&times;</button>
  </div>
  <div class="rdp-body" id="rdpBody"></div>
  <div class="rdp-foot">
    <div id="rdpActs" style="display:flex;gap:4px;flex:1;flex-wrap:wrap;align-items:center"></div>
    <button class="btn" style="font-size:9px;padding:3px 9px" onclick="closeRdp()">Close</button>
  </div>
</div>

<div id="etpop" role="dialog" aria-label="ET Recovery Batch">
  <div class="rdp-hd">
    <span class="rdp-title">E&#x209c; Recovery Batch</span>
    <button class="rdp-close" onclick="closeEtPop()" title="Close">&times;</button>
  </div>
  <div class="rdp-body" id="etpBody" style="padding:6px 10px">
    <div style="font-size:9px;color:var(--t2);margin-bottom:8px;line-height:1.6">
      Select runs and mode. <b>E</b>&nbsp;= base E&#x209c; pass only (you already have abliterated data).
      <b>A</b>&nbsp;= abliterated generation only. <b>B</b>&nbsp;= both.
    </div>
    <table id="etTable" style="width:100%;border-collapse:collapse;font-size:10px"></table>
    <div style="display:flex;gap:6px;margin-top:8px;flex-wrap:wrap">
      <button class="btn" style="font-size:9px;padding:2px 7px" onclick="etSelectAll(true)">All</button>
      <button class="btn" style="font-size:9px;padding:2px 7px" onclick="etSelectAll(false)">None</button>
      <button class="btn" style="font-size:9px;padding:2px 7px" onclick="etSetMode('e')">All &rarr; E</button>
      <button class="btn" style="font-size:9px;padding:2px 7px" onclick="etSetMode('a')">All &rarr; A</button>
      <button class="btn" style="font-size:9px;padding:2px 7px" onclick="etSelectNeedsEt()">needs E&#x209c; only</button>
    </div>
  </div>
  <div class="rdp-foot">
    <div style="flex:1"></div>
    <button class="btn" style="font-size:9px;padding:3px 9px" onclick="closeEtPop()">Close</button>
    <button class="btn" style="font-size:9px;padding:3px 10px;border-color:var(--bl);color:var(--bl)"
      onclick="etSelectNeedsEt();etSetMode('e');launchEtBatch()"
      title="Select all needs-E&#x209c; runs at mode E and launch in one press">&#9654; All Pending</button>
    <button class="btn set-action" style="font-size:9px;padding:3px 10px" onclick="launchEtBatch()">&#9654; Launch Selected</button>
  </div>
</div>

<!-- Run 0056: Cross-model paper assembly — model selection (v0.75.2.2) -->
<div id="r55pop" role="dialog" aria-label="Paper Assembly — Model Selection" style="display:none;
  position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);z-index:9999;
  background:var(--bg2);border:2px solid var(--b1);border-radius:8px;padding:0;
  min-width:340px;max-width:500px;box-shadow:0 8px 32px rgba(0,0,0,.5)">
  <div class="rdp-hd">
    <span class="rdp-title">Run 0056 — Paper Assembly</span>
    <button class="rdp-close" onclick="closeR55Pop()" title="Close">&times;</button>
  </div>
  <div class="rdp-body" id="r55Body" style="padding:8px 12px">
    <div style="font-size:9px;color:var(--t2);margin-bottom:8px;line-height:1.6">
      Select models to include in paper figures (FIG01–FIG13) and JSON assembly.
      Only models with completed analysis can be selected.
    </div>
    <div id="r55Models" style="font-size:10px"></div>
  </div>
  <div class="rdp-foot">
    <div style="flex:1"></div>
    <button class="btn" style="font-size:9px;padding:3px 9px" onclick="closeR55Pop()">Cancel</button>
    <button class="btn set-action" style="font-size:9px;padding:3px 10px" id="r55Go" onclick="launchR55()">&#9654; Generate Paper</button>
  </div>
</div>
<div id="r55overlay" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;
  background:rgba(0,0,0,.4);z-index:9998" onclick="closeR55Pop()"></div>

<!-- Toolbar (fixed) -->
<div id="tb" role="toolbar" aria-label="Run controls">
  <div class="tb-left">
    <div class="logo">IOTA<span>v0.79.4.9</span></div>
    <div class="tb-status">
      <div class="dot" id="dot" title="Green = run active · Grey = idle"></div>
      <span id="stx" title="Current run status">idle</span>
    </div>
  </div>
  <!-- Clocks: Total > Session > Run > Idle -->
  <div class="tb-clocks">
    <div class="timer-row" title="Total collection time for this model across all temperature rounds">
      <span class="timer-lbl">Total</span>
      <span class="timer-val" id="tTotal">&#8212;</span>
    </div>
    <div class="timer-row" title="Time since current collection session was launched">
      <span class="timer-lbl">Session</span>
      <span class="timer-val" id="tSess">&#8212;</span>
    </div>
    <div class="timer-row" title="Elapsed time for current active run">
      <span class="timer-lbl" id="tRunLbl">Run</span>
      <span class="timer-val live" id="tRun">&#8212;</span>
    </div>
    <div class="timer-row" title="Total idle time (program open but not collecting)">
      <span class="timer-lbl">Idle</span>
      <span class="timer-val" id="tIdle">&#8212;</span>
    </div>
  </div>
  <div class="tb-mid">
    <div class="tb-context">
      <span id="ml" title="Active model" style="font-weight:600;font-size:11px">&#8212;</span>
      <span id="tbTempBig" title="Current temperature" style="font-size:13px;font-weight:700;color:var(--ac);margin:0 6px">&#8212;</span>
      <span id="tbRn" title="Current run label" style="font-size:10px"></span>
      <span id="tbTr" title="Trial progress" style="font-size:10px"></span>
    </div>
    <div class="tb-pcts">
      <div class="tb-pct" title="All temperatures progress">
        <div class="tb-pct-val" id="barTempsPct">&#8212;</div>
        <div class="tb-pct-lbl">temps</div>
      </div>
      <div class="tb-pct" title="Current run progress (trials)">
        <div class="tb-pct-val" id="barRunPct">&#8212;</div>
        <div class="tb-pct-lbl">run</div>
      </div>
      <div class="tb-pct" title="Current trial progress">
        <div class="tb-pct-val" id="barTrialPct">&#8212;</div>
        <div class="tb-pct-lbl">trial</div>
      </div>
    </div>
    <div class="tb-run-info" style="font-size:8px;color:var(--t3)">
      <span id="tbTc" title="state_similarity_index"></span>
      <span id="tbSm" title="Layer similarity"></span>
      <span id="tbPw" title="GPU power"></span>
    </div>
  </div>
  <div class="tb-right">
    <button class="tb-btn-tall go" id="lBtn2" onclick="doLaunchOrQueue()"
      aria-label="Launch or queue selected runs" title="Launch selected runs (keyboard: R)">
      <span class="tb-btn-icon">&#9654;</span>
      <span class="tb-btn-lbl" id="lBtnLbl2">Run</span>
    </button>
    <div id="queueInd" style="display:none;font-size:8px;color:var(--wa);
      max-width:52px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
      text-align:center;padding:2px 0" title="Queued runs"></div>
    <button id="clearQBtn" style="display:none;font-size:7px;padding:1px 5px;cursor:pointer;
      background:transparent;border:1px solid rgba(255,68,85,.3);color:var(--er);
      font-family:var(--fn);letter-spacing:.04em" onclick="doClearQueue()" title="Clear queue">&#10005; Q</button>
    <button class="tb-btn-tall pa" id="pBtn2" onclick="doPause()"
      aria-label="Pause or resume" title="Pause or resume (keyboard: Space)">
      <span class="tb-btn-icon">&#9646;&#9646;</span>
      <span class="tb-btn-lbl" id="pBtnLbl2">Pause</span>
    </button>
    <button class="tb-btn-tall ab" id="abBtn" onclick="doAbort()"
      aria-label="Abort running process" title="Abort (keyboard: Esc)">
      <span class="tb-btn-icon">&#9632;</span>
      <span class="tb-btn-lbl">Abort</span>
    </button>
    <button class="tb-btn-tall go" id="rsBtn2" onclick="doResume()"
      style="display:none" title="Resume interrupted session">
      <span class="tb-btn-icon">&#8635;</span>
      <span class="tb-btn-lbl">Resume</span>
    </button>
  </div>
</div>

<!-- Workspace: three tiling panels -->
<div id="workspace">

  <!-- Left column: grid (top) + metrics (bottom) -->
  <div id="left-col">

    <!-- Grid panel -->
    <div id="panel-grid">
      <div class="panel-hdr">
        <span class="panel-title">Runs &#x2193; run in order</span>
        <div class="panel-hdr-right">
          <span id="gsn" style="font-size:9px;color:var(--t2)">scanning&#8230;</span>
        </div>
      </div>
      <div id="grid-scroll">
        <div class="run-order-note" id="gridSummary"></div>
        <!-- Navigation breadcrumb -->
        <div class="nav-bc" id="navBc">
          <span class="bc-link" id="bcModels" onclick="navTo(1)">Models</span>
          <span class="bc-sep" id="bcSep1" style="display:none"> &#x203a; </span>
          <select class="bc-select" id="bcTempSel" style="display:none" onchange="onBcTempChange()">
            <option value="all">All Temps</option>
            <option value="0.0">T=0.0</option>
            <option value="0.2">T=0.2</option>
            <option value="0.4">T=0.4</option>
            <option value="0.6">T=0.6</option>
            <option value="0.8">T=0.8</option>
            <option value="1.0">T=1.0</option>
          </select>
        </div>
        <!-- Layer 1: Model Select -->
        <div id="lyr1" class="lyr on">
          <div class="model-cards" id="modelCards">
            <div class="lyr-loading" id="lyr1Loading"><div class="ld-spin"></div><div class="ld-text">Loading models...</div></div>
          </div>
          <div class="add-model">
            <div class="am-title">Model Setup</div>
            <div class="am-row" style="display:flex;gap:6px;flex-wrap:wrap;align-items:center">
              <select id="amFam" onchange="amLoadGens()" style="flex:1;min-width:80px"></select>
              <select id="amGen" onchange="amLoadSzs()" style="flex:1;min-width:90px"></select>
              <select id="amSz" style="flex:1;min-width:70px"></select>
              <select class="si" id="iqt" onchange="autoSaveParams()" style="min-width:55px">
                <option value="4bit">4-bit</option>
                <option value="8bit">8-bit</option>
                <option value="fp16">FP16</option>
              </select>
            </div>
            <div class="am-row" style="margin-top:6px">
              <div class="frow" style="flex:1"><div class="flb" style="min-width:40px">HF Token</div>
                <div style="position:relative;flex:1;display:flex;align-items:center">
                  <input class="si" id="amHf" type="text" placeholder="hf_..." style="flex:1;padding-right:24px"
                    onfocus="this.type='text';this.value=this.dataset.raw||this.value"
                    onblur="hfBlur(this)">
                  <span id="amHfStatus" style="position:absolute;right:6px;font-size:11px"></span>
                </div></div>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;margin-top:8px">
              <button class="btn set-action" onclick="amStart()">Add Model</button>
              <div style="display:flex;gap:6px">
                <button class="btn set-action" onclick="resetDefaults()" title="Restore defaults">Default</button>
                <button class="btn set-action" onclick="autoSaveParams();refreshParams()" title="Save all model settings">Save All</button>
              </div>
            </div>
            <div style="font-size:9px;color:var(--t2);margin-top:6px">Models with abliterated + instruct + base variants only. Configure temperatures in <a href="#" onclick="sw('S');return false" style="color:var(--ac)">Settings</a>.</div>
          </div>
        </div>
        <!-- Layer 2+3: Grid (All Temps or Per-Temp) -->
        <div id="lyr2" class="lyr">
          <div class="lyr-loading" id="lyr2Loading" style="display:none"><div class="ld-spin"></div><div class="ld-text">Loading temperature data...</div></div>
          <div class="preset-bar" role="group" aria-label="Run presets">
            <button class="pr" onclick="selP('1-55',this)" title="All runs across every phase">All</button>
            <button class="pr" onclick="selP('1-24,26,28-31,35-39,41-44,53',this)" title="Data Collection — GPU generation runs, temp-indep + Phase B + Phase D + Phase EFC">Collection</button>
            <button class="pr" onclick="selP('25,27,32-34,45,46,49,56',this)" title="Per-Temperature Analysis — POOL_DIM calibration, E+C+R, permutation, per-condition R, MLP validation">Analysis</button>
            <button class="pr" onclick="selP('40,47',this)" title="Pooled Analysis — cross-temperature within one model">Pooled</button>
            <button class="pr" onclick="selP('50,51,52',this)" title="Cross-Model Analysis — pairwise R comparison, condition concordance, cross-model summary">Cross-Model</button>
            <button class="pr" onclick="selP('54,55',this)" title="Paper Output — stats report, master JSONs, paper assembly">Paper</button>
            <button class="pr et-pr" onclick="openEtPop()" title="E_t Recovery &#x2014; run base-model pass to extract E_t embeddings for Runs 0004&#x2013;0015">E&#x2094; Recovery</button>
          </div>
          <div class="phase-grid" id="phaseGrid" role="group" aria-label="Individual run selection"></div>
          <div class="legend" aria-label="Run status legend">
            <div class="li"><div class="ld" style="background:rgba(68,221,136,.45)"></div>done</div>
            <div class="li"><div class="ld" style="background:rgba(255,140,66,.55)"></div>partial</div>
            <div class="li"><div class="ld" style="background:rgba(245,208,0,.55)"></div>needs E&#x209c;</div>
            <div class="li"><div class="ld" style="background:rgba(255,68,85,.45)"></div>corrupt</div>
            <div class="li"><div class="ld" style="background:rgba(58,80,96,.5)"></div>not started</div>
            <div class="li"><div class="ld" style="background:var(--ac)"></div>selected</div>
          </div>
          <div class="run-row" style="margin-top:6px">
            <input class="si" id="ri" placeholder="e.g. 26  or  1-5,28"
              title="Enter run numbers or ranges" aria-label="Run selection">
            <button class="btn set-action" style="padding:4px 10px" onclick="doLaunchOrQueue()" title="Launch / Queue">&#9654;</button>
            <button class="btn" style="padding:4px 8px" onclick="clearSel()" title="Clear">&#10005;</button>
          </div>
          <div id="selInfo" aria-live="polite" style="display:none"></div>
          <div id="selBox" aria-live="polite" role="group" aria-label="Selected runs"></div>
          <div class="grid-controls">
            <button class="gc-btn go" id="lBtn" onclick="doLaunchOrQueue()" title="Launch selected runs (R)">
              <span class="gc-icon">&#9654;</span> <span id="lBtnLbl">Run</span>
            </button>
            <button class="gc-btn pa" id="pBtn" onclick="doPause()" title="Pause / resume (Space)">
              <span class="gc-icon">&#9646;&#9646;</span> <span id="pBtnLbl">Pause</span>
            </button>
            <button class="gc-btn go" id="rsBtn" onclick="doResume()"
              style="display:none" title="Resume interrupted session">
              <span class="gc-icon">&#8635;</span> Resume
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Horizontal splitter -->
    <div id="hspl" title="Drag to resize grid / metrics" aria-hidden="true"></div>

    <!-- Metrics panel -->
    <div id="panel-metrics">
      <div class="panel-hdr" style="display:flex;justify-content:space-between;align-items:center">
        <span class="panel-title">Live
          <span id="mRn" style="color:var(--t2);font-size:9px;letter-spacing:0;text-transform:none;margin-left:5px">&#8212;</span>
        </span>
        <button class="expand-btn" id="metricsExpand" onclick="togMetricsExpand()" title="Expand metrics over grid">&#x26F6;</button>
      </div>
      <div id="metrics-scroll">
        <div class="sparks">
          <div class="spk-row">
            <div class="spk-hd">
              <span>Coherence (avg)</span><span id="stcv">&#8212;</span>
            </div>
            <canvas class="spk" id="stc" aria-label="Coherence sparkline"></canvas>
          </div>
          <div class="spk-row">
            <div class="spk-hd">
              <span>Similarity (avg)</span><span id="ssmv">&#8212;</span>
            </div>
            <canvas class="spk" id="ssm" aria-label="Similarity sparkline"></canvas>
          </div>
          <div class="spk-row">
            <div class="spk-hd">
              <span>GPU Power (avg W)</span><span id="spwv">&#8212;</span>
            </div>
            <canvas class="spk" id="spw" aria-label="GPU power sparkline"></canvas>
          </div>
          <div class="spk-row">
            <div class="spk-hd">
              <span>Disruption (avg)</span><span id="srcv">&#8212;</span>
            </div>
            <canvas class="spk" id="src" aria-label="disruption sparkline"></canvas>
          </div>
        </div>
      </div>
    </div>

  </div><!-- /left-col -->

  <!-- Vertical splitter -->
  <div id="vspl" title="Drag to resize panels" aria-hidden="true"></div>

  <!-- Right panel: tabbed -->
  <div id="panel-right">
    <div class="tab-bar" role="tablist" aria-label="Dashboard panels">
      <button class="tab on" id="tC" role="tab" aria-selected="true" aria-controls="pC" onclick="sw('C')">Console</button>
      <button class="tab" id="tH" role="tab" aria-selected="false" aria-controls="pH" onclick="sw('H')">Hypotheses</button>
      <button class="tab" id="tN" role="tab" aria-selected="false" aria-controls="pN" onclick="sw('N')">Settings</button>
    </div>

    <!-- Console pane -->
    <div class="pane on" id="pC" role="tabpanel" aria-labelledby="tC">
      <div style="display:flex;justify-content:center;padding:2px 6px 2px 0;flex-shrink:0;
        background:var(--bg2);border-bottom:1px solid var(--b1)">
        <button class="btn" style="font-size:9px;padding:2px 8px" onclick="clrC()"
          title="Clear console (Ctrl+L)" aria-label="Clear console">Clear</button>
      </div>
      <div id="con-load-indicator" onclick="_loadHistory()" title="Load older log lines">&#x25B2; scroll up or click to load history</div>
      <div id="con" role="log" aria-live="polite" aria-label="Console output"></div>
      <div id="con-bottom-btn" onclick="conScrollBottom()" title="Scroll to bottom and lock auto-scroll">&#x25BC; Bottom</div>
      <div id="con-refresh-btn" onclick="conRefresh()" title="Reconnect console" style="position:absolute;bottom:4px;right:70px;background:var(--bg2);border:1px solid var(--b1);color:var(--t2);font-size:9px;padding:2px 8px;border-radius:3px;cursor:pointer;opacity:.5;z-index:2">&#x21BB;</div>
    </div>

    <!-- Hypothesis pane -->
    <div class="pane" id="pH" role="tabpanel" aria-labelledby="tH">
      <div class="hyp-hdr">
        <div class="hyp-sum" aria-live="polite">
          <span style="color:var(--go)"><span id="hypSup">&#8212;</span> supported</span> &#x2022;
          <span style="color:var(--er)"><span id="hypDis">&#8212;</span> disproven</span> &#x2022;
          <span style="color:var(--wa)"><span id="hypInc">&#8212;</span> inconclusive</span> &#x2022;
          <span style="color:var(--t2)"><span id="hypPen">&#8212;</span> pending</span>
        </div>
        <div style="display:flex;gap:4px">
          <button class="btn" style="font-size:9px;padding:2px 8px" onclick="hypSelectAll(true)" title="Select all runs from all hypotheses">All</button>
          <button class="btn" style="font-size:9px;padding:2px 8px" onclick="hypSelectAll(false)" title="Clear selection">None</button>
          <button class="btn" style="font-size:9px;padding:2px 8px" onclick="loadHyp(true)" title="Reload outcomes (live compute)">Refresh</button>
        </div>
      </div>
      <div class="hyp-scroll" id="hypScroll" role="list" aria-label="Hypotheses">
        <div style="font-size:10px;color:var(--t2)">Loading hypothesis data&#8230;</div>
      </div>
      <!-- v0.76.0.7 / v0.77.1.3: hypothesis detail panel. showHypDetail() writes
           here on row click. Prior: function existed but was never bound AND
           the target DOM element didn't exist. Numerical H values computed,
           never shown. -->
      <div id="hypDetail" class="hyp-detail" role="status" aria-live="polite"
           style="padding:6px 8px;margin-top:6px;border:1px solid var(--b2);
                  border-radius:4px;background:var(--bg);min-height:28px;
                  font-size:10px;color:var(--t2)">
        <span style="color:var(--t3)">Click a hypothesis above to view its value, status, and implication.</span>
      </div>
      <!-- Queue bar removed - run from grid instead -->
    </div>

    <!-- Settings pane -->
    <div class="pane" id="pN" role="tabpanel" aria-labelledby="tN">
      <div id="sp2">
        <div class="fsect">
          <div class="ftitle">Collection</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px 12px;margin-bottom:8px">
            <div class="frow"><div class="flb">Trials</div>
              <input class="si" id="itr" type="number" min="1" max="1000"
                placeholder="100" style="width:60px" onchange="autoSaveParams()"></div>
            <div class="frow"><div class="flb">Seed</div>
              <input class="si" id="ise" type="number" min="0"
                placeholder="42" style="width:60px" onchange="autoSaveParams()"></div>
          </div>
          <div class="fsect" style="margin-bottom:8px">
            <div class="ftitle">Temperatures</div>
            <div class="frow" style="margin-bottom:6px">
              <select class="si" id="iTempPreset" onchange="onTempPresetChange()" style="width:auto">
                <option value="standard">Standard (6)</option>
                <option value="log">Logarithmic (11)</option>
                <option value="custom">Custom</option>
              </select>
            </div>
            <div id="tempChips" style="display:flex;flex-wrap:wrap;gap:3px"></div>
          </div>
        </div>
        <div class="fsect">
          <div class="ftitle">Performance</div>
          <div style="display:grid;grid-template-columns:1fr auto;gap:4px 12px;align-items:center;margin-bottom:8px">
            <span class="flb">GPU Mode</span>
            <select class="si" id="iGpu" onchange="autoSaveParams()" style="width:auto">
              <option value="single">Single GPU</option>
              <option value="multi">Multi-GPU (auto)</option>
              <option value="distributed">Distributed</option>
            </select>
            <span class="flb">Clear cache between turns<i class="qm" title="Flush GPU memory after each generation turn. Prevents VRAM buildup. Default ON for 10GB cards.">?</i></span>
            <button class="tgl" id="tCC" onclick="togCC()">&#8212;</button>
            <span class="flb">Force GC between trials<i class="qm" title="Run Python garbage collector and flush CUDA memory between trials. Prevents slow memory leaks over long sessions.">?</i></span>
            <button class="tgl" id="tGC" onclick="togGC()">&#8212;</button>
            <span class="flb">Unload model between runs<i class="qm" title="Fully remove model from GPU between runs. Slower startup per run but guarantees clean VRAM. Default ON for 10GB cards.">?</i></span>
            <button class="tgl" id="tUL" onclick="togUL()">&#8212;</button>
            <span class="flb">Restore run selection on load<i class="qm" title="Restore the last run selection when the dashboard loads. OFF = grid starts empty.">?</i></span>
            <button class="tgl" id="tRS" onclick="togRS()">&#8212;</button>
            <span class="flb">Low priority (reduce lag)<i class="qm" title="Run analysis and data collection at below-normal CPU priority. Reduces system lag while IOTA runs in the background. Default ON.">?</i></span>
            <button class="tgl" id="tLP" onclick="togLP()">&#8212;</button>
          </div>
        </div>
        <div class="fsect">
          <div class="ftitle">Data Management</div>
          <div style="display:flex;gap:8px;align-items:center;margin-bottom:8px;flex-wrap:wrap">
            <select class="si" id="clearScope" style="flex:1" onchange="updateClearBtn()">
              <option value="run_temp">Selected runs &times; current temp</option>
              <option value="run_all">Selected runs &times; all temps</option>
              <option value="temp">All data at current temperature</option>
              <option value="all_temps">All data at all temperatures</option>
              <option value="delete_model">Delete entire model directory</option>
              <option value="factory_reset">&#9760; Factory reset &mdash; delete EVERYTHING</option>
            </select>
            <button class="nuke-btn" id="nukeBtn" onclick="clearDataPrompt()">&#x1f5d1; Clear</button>
          </div>
          <div id="clearDesc" style="font-size:9px;color:var(--t3);margin-top:-4px;margin-bottom:8px"></div>
        </div>
        <div style="display:flex;justify-content:flex-end;gap:8px;padding:8px 0;border-top:1px solid var(--b1)">
          <button class="btn set-action" onclick="resetDefaults()" title="Restore all settings to defaults">Default</button>
          <button class="btn set-action" onclick="autoSaveParams();savePerfSettings()" title="Save all settings">Save All</button>
        </div>
        <div class="fsect">
          <div class="ftitle">Keyboard Shortcuts</div>
          <div class="kbd">
            <div class="kk"><code>Space</code>Pause / Resume</div>
            <div class="kk"><code>R</code>Launch runs</div>
            <div class="kk"><code>Esc</code>Abort</div>
            <div class="kk"><code>1\u20133</code>Switch tabs</div>
            <div class="kk"><code>Ctrl+L</code>Clear console</div>
            <div class="kk"><code>Ctrl+K</code>Focus run input</div>
          </div>
        </div>
      </div>
    </div>
    <!-- Pause banner -->
    <div class="pov" id="pov" role="status" aria-label="Paused">
      <div class="pt">&#9208; Paused</div>
      <div class="ps">resumes after current turn</div>
      <button class="btn pa" style="margin-top:0;padding:2px 10px;font-size:9px" onclick="doPause()" aria-label="Resume">Resume</button>
    </div>

  </div><!-- /panel-right -->

</div><!-- /workspace -->

<div id="disc" role="alert">&#9888; Disconnected &#x2014; reconnecting&#8230;</div>
<div id="tst" aria-live="assertive"></div>

<script>
// ── State ─────────────────────────────────────────────────────────────────────
const SD={sim_index:[],sim:[],pow:[],disrupt:[]},MX=60;
let AS=true,PAU=false,LLN=0,LOG_START=0,_histLoading=false,_clrMode=false;
let selR=new Set(),scanSt={},descs={},curRN=null,_lastRun=null;
let _CC=true,_modelsData={families:{},sizes:{}};
let _hypLoaded=false,_hypDeps={},_hypQueue=new Set();
let scanInterval,logInterval,rawInterval;
let _prevRunning=null,_searchStr='',_sevFilter=null,_activeRunTemp=null;
let _elapsedTick=null,_selHyp=null,_outcomes={};
let _rawlogActive=false;

// ── Timer stack (v0.59.1.0: server-computed from CSV timestamps) ──────────────
// All times except Session come from /timers endpoint — computed from actual
// CSV timestamp columns. No client-side accumulation, no persistence file, no drift.
// Session = time since page open (client-side, resets on close).
// Run = live tick from run_start_ts while active.
let _totalStart=Date.now();
let _timerTick=null;
let _srvTimers={collection_sec:0,model_sec:0,run_label:'',run_sec:0,active:false};
let _srvFetchTs=0;  // when we last got server data
let _runActiveTs=0;  // Date.now() when we learned run was active

(async function(){
  try{await _fetchTimers();}catch(e){}
  _timerTick=setInterval(_tickTimers,1000);
  setInterval(_fetchTimers,3000);
})();

async function _fetchTimers(){
  try{
    const r=await fetch('/timers');
    const d=await r.json();
    const wasActive=_srvTimers.active;
    _srvTimers=d;
    _srvFetchTs=Date.now();
    if(d.active&&!wasActive)_runActiveTs=Date.now();
    if(!d.active)_runActiveTs=0;
  }catch(e){}
}

let es;

// ── Execution order ──────────────────────────────────────────────────────────
// v0.77.1.3: updated to match start_here.EXECUTION_ORDER exactly.
// Adds Run 0046 (after Run 0044) and cross-model runs 0052,0053,0054,0056 that were
// missing. Primarily used for the hypothesis launcher queue ordering.
// v0.79.4.9: renumber — EXEC_ORDER is now simply 1..56 (IDs = positions).
const EXEC_ORDER=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
  16,
  17,18,19,
  20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
  41,42,43,44,45,46,47,48,49,
  50,51,
  52,53,54,
  55,56];

// Phase groupings for the grid — new-ID phase boundaries
const PHASES=[
  {label:'Data Collection',runs:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    16,
    17,18,19,
    20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]},
  {label:'Analysis',runs:[41,42,43,44,45,46,47,48,49]},
  {label:'Pooled',runs:[50,51]},
  {label:'Cross-Model',runs:[52,53,54]},
  {label:'Paper',runs:[55,56]},
];
const ALL=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,
           25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56];

const H_PHASES={
  // v0.77.1.3: UI labels aligned with HYPOTHESES registry in export_stats.py.
  // 0.77.0.0 renumbered H02→H54, H04→H55, H07→H56, H15→H57 but kept old
  // inverted names. 0.77.1.3 fixes the names so UI matches registry.
  // H33 updated to match registry "Has No ... Compute Advantage".
  // H34 updated to "Does Not Track" (was "Tracks" — inverted).
  // H50 / H51 updated to null-direction-correct form.
  'CSV Only (no analysis needed)':[['H01','Geometry Is Condition-Invariant'],['H54','Baseline Entropy Is Not Degenerate'],
    ['H03','Introspection Has No Geometric Effect'],['H55','Signal Entropy Ratio Is Elevated By Introspection'],
    ['H05','Contradiction Has No Geometric Effect'],['H06','Similarity Does Not Decay'],
    ['H56','No Similarity Drop Early To Late'],['H08','Throughline Has No Trajectory Effect'],
    ['H09','Task Demand Geometry Equals Introspection'],['H10','Disruption Fully Resets Trajectory'],
    ['H12','Similarity Explained By Token Statistics'],['H13','Similarity Collapses Under Temperature'],
    ['H14','Cross-Turn Sim Concentrates In Late Layers'],['H57','Turn-1 Sim Shows Early-To-Late Gradient'],
    ['H23','Onset Delay Uncorrelated With Disruption'],['H25','Entropy Shape Uncorrelated With R'],
    ['H26','R Does Not Persist Across Condition Switch'],['H27','Output Similarity Uncorrelated With R'],
    ['H30','Output Turn-1 Self-Referentiality Grows'],['H31','Arithmetic Accuracy Uncorrelated With Trajectory'],
    ['H32','Shock Phrasing Has No Differential Effect'],['H33','Condition A Has No Task-Equivalent Compute Advantage'],
    ['H34','Disruption Magnitude Does Not Track Contradiction Response'],['H35a','Contradiction Recovery (Group)'],
    ['H35b','Contradiction Recovery (Per-Trial)'],['H39','Impossibility Has No Geometric Effect'],
    ['H41','System Prompt Has No Geometric Effect'],['H42','Similarity Does Not Accumulate With Turn Count'],
    ['H52','Disruption Magnitude Is Stationary'],['H53','Cross-Stochasticity Disruption Monotonic']],
  'Needs Analysis (Runs 0047-0043)':[['H11','Prior State Adds No Predictive Power'],
    ['H16','Permuted Component Adds No Variance'],['H17','Consistency Explained By System Prompt'],
    ['H18','History Mode Has No Effect'],['H19','Two-Instance Coupling Has No Effect'],
    ['H20','R Condition Has No Signal/Watt Effect'],['H21','Prior State Adds No Power Over E+C'],
    ['H22','E C R Permutation Fractions Are Equal'],['H24','Cross-Turn Similarity Has No Layer Locality'],
    ['H28','In-Sample Fractions Do Not Generalise'],
    ['H50','MLP Does Not Improve On Ridge By More Than 0.01'],
    ['H51','Interaction Information Below 5% Of Joint MI'],
    ['H58','E + C + R Fractions Sum To 1.0 Within Tolerance']],
  'Needs Patching (Runs 0017/0018/0019)':[['H29','No Single Layer Is Causally Sufficient'],
    ['H38','Patched State Does Not Change Output'],['H40','Random Noise Matches Real Patching']],
  'Cross-Temperature (Runs 0034/0037)':[['H36','Causal Effect Of Patching Is Temperature-Invariant'],
    ['H37','Layer Causal Sufficiency Is Temperature-Invariant']],
  'Not Yet Testable':[['H43','R Fraction Sensitive To Projection Dimension'],
    ['H44','R Fraction Is Condition-Invariant'],['H45','Resistant Prime No Worse Than Cooperative'],
    ['H46','Contradiction Does Not Drop Similarity'],['H47','Compute Cost Not Reduced By Priming'],
    ['H48','R Decay Explained By KV Cache Growth'],['H49','Introspection R Does Not Transfer']],
};

// ── SSE ───────────────────────────────────────────────────────────────────────
let _lastMsg=Date.now();
function sse(){
  es=new EventSource('/stream');
  es.onmessage=e=>{
    _lastMsg=Date.now();
    document.getElementById('disc').classList.remove('on');
    applyS(JSON.parse(e.data));};
  es.onerror=()=>{document.getElementById('disc').classList.add('on');
    es.close();setTimeout(sse,3000);};
}
sse();
setInterval(()=>{
  if(Date.now()-_lastMsg>15000){
    document.getElementById('disc').classList.add('on');
    es.close();setTimeout(sse,500);_lastMsg=Date.now();}
},5000);

// ── Timers (v0.59.1.0: server-computed from CSV timestamps) ───────────────────
function _fmtTime(ms){
  const s=Math.floor(ms/1000),h=Math.floor(s/3600),m=Math.floor((s%3600)/60),sec=s%60;
  if(h)return h+'h '+p2(m)+'m '+p2(sec)+'s';
  return p2(m)+'m '+p2(sec)+'s';
}
function _tickTimers(){
  const now=Date.now();
  const d=_srvTimers;
  // Total: all CSV collection time for this model — static between polls
  const tEl=document.getElementById('tTotal');
  if(tEl)tEl.textContent=_fmtTime((d.total_sec||0)*1000);
  // Session: time since /run was fired — tick live only when active
  const sEl=document.getElementById('tSess');
  if(sEl){
    if(d.session_sec>0&&d.active){
      const liveSess=(d.session_sec||0)+((now-_srvFetchTs)/1000);
      sEl.textContent=_fmtTime(liveSess*1000);
    } else if(d.session_sec>0){
      sEl.textContent=_fmtTime((d.session_sec||0)*1000);
    } else { sEl.textContent='\u2014'; }
  }
  // Run: live tick if active
  const rEl=document.getElementById('tRun');
  if(rEl){
    if(d.active&&_runActiveTs){
      const liveSec=(d.run_sec||0)+((now-_srvFetchTs)/1000);
      rEl.textContent=_fmtTime(liveSec*1000);
    } else { rEl.textContent='\u2014'; }
  }
  // Idle: server snapshot only — no live tick (v0.71.0.11 fix: idle was growing
  // continuously because (now-_srvFetchTs) was added on every tick while total
  // stayed frozen, causing idle > total within seconds)
  const iEl=document.getElementById('tIdle');
  if(iEl){
    const idleSec=(d.idle_sec||0);
    const dc=d.duty_cycle||0;
    const dcPct=Math.round(dc*100);
    iEl.textContent=_fmtTime(idleSec*1000)+' ('+dcPct+'%)';
  }
  // Update run label if present
  const rl=document.getElementById('tRunLbl');
  if(rl)rl.textContent=d.run_label||'Run';
}

// ── Status apply ──────────────────────────────────────────────────────────────
function applyS(s){
  window._iotaStatus=s;
  PAU=s.paused||false;
  // BUG-STATUS-GHOST fix (v45.1.0): web_running is sole source of truth.
  const run=!!(s.web_running);
  const dot=document.getElementById('dot'),st=document.getElementById('stx');
  if(PAU){dot.className='dot pau';st.textContent='paused';}
  else if(run){dot.className='dot run';st.textContent='running';}
  else{dot.className='dot';st.textContent='idle';}
  document.getElementById('pov').classList.toggle('on',PAU);
  // Sync both button sets (grid + toolbar)
  ['pBtnLbl','pBtnLbl2'].forEach(id=>{const e=document.getElementById(id);if(e)e.textContent=PAU?'Resume':'Pause';});
  ['lBtn','lBtn2'].forEach(id=>{const e=document.getElementById(id);if(e)e.disabled=false;});
  const _runLabel=s.web_running?'Queue':'Run';
  ['lBtnLbl','lBtnLbl2'].forEach(id=>{const e=document.getElementById(id);if(e)e.textContent=_runLabel;});
  // Queue indicator
  const _qi=document.getElementById('queueInd');
  const _cq=document.getElementById('clearQBtn');
  if(_qi){
    if(s.queued_runs){_qi.style.display='block';_qi.textContent='\u2192 '+s.queued_runs;if(_cq)_cq.style.display='inline-block';}
    else{_qi.style.display='none';_qi.textContent='';if(_cq)_cq.style.display='none';}
  }
  const _ml=document.getElementById('ml');
  const _isAnalysis = !!(s.analysis_mode);
  if(_ml){
    if(_isAnalysis && s.web_running){
      _ml.textContent='Analysis — '+( s.run_label||('Run '+s.run_num));
    } else if(s.web_running&&s.model_name){
      _ml.textContent=s.model_name;
    } else {
      _ml.textContent='\u2014';
    }
  }
  // Temperature indicator
  const _tb=document.getElementById('tbTempBig');
  if(_tb){
    if(s.web_running&&s.temperature!=null){_tb.textContent='T='+Number(s.temperature).toFixed(1);_activeRunTemp=Number(s.temperature);}
    else _tb.textContent='';
  }
  const runLbl=document.getElementById('tRunLbl');
  if(runLbl&&s.run_label)runLbl.textContent=s.run_label;
  else if(runLbl&&s.run_num)runLbl.textContent='Run '+s.run_num;
  // Analysis mode: show step + pct instead of trial/turn/similarity/power
  if(_isAnalysis && s.web_running){
    S('tbRn', s.run_label||(s.run_num?'Q'+s.run_num:null));
    S('mRn', s.run_label||(s.run_num?'Q'+s.run_num:null));
    const _aStep = s.analysis_step||'';
    const _aPct = s.analysis_pct!=null ? s.analysis_pct+'%' : '';
    const _aDetail = s.analysis_detail||'';
    S('tbTr', _aStep + (_aPct ? ' ('+_aPct+')' : ''));
    S('tbTc', _aDetail||null);
    S('tbSm', null);
    S('tbPw', null);
    // Use run progress bar for analysis percentage
    if(s.analysis_pct!=null){
      S('barRunPct', s.analysis_pct+'%');
      bar('btr','btp',s.analysis_pct,100);
    }
    S('barTrialPct',null);
  } else {
    S('tbRn',s.run_label||(s.run_num?'R'+p2(s.run_num):null));
    S('mRn', s.run_label||(s.run_num?'R'+p2(s.run_num):null));
    S('tbTr',s.trial!=null?(s.total_trials!=null?s.trial+'/'+s.total_trials:s.trial):null);
    S('tbTc',f4(s.sim_index));
    S('tbSm',f4(s.sim));
    S('tbPw',s.power!=null?s.power.toFixed(1)+'W':null);
  }
  // Progress percentages with ETA: Run (trial/total), Trial (turn/max_turn)
  if(s.trial!=null&&s.total_trials){
    const rPct=Math.round(s.trial/s.total_trials*100);
    let rEta='';
    // Only show ETA after 5+ trials so warmup overhead doesn't skew estimate
    if(s.run_start_ts&&s.ts&&s.trial>=5){
      const elapsed=s.ts-s.run_start_ts;
      const avgPerTrial=elapsed/s.trial;
      const remaining=avgPerTrial*(s.total_trials-s.trial);
      if(remaining>0&&isFinite(remaining))rEta=' ~'+_fmtEta(remaining);
    }
    S('barRunPct',rPct+'%'+rEta);
    bar('btr','btp',s.trial,s.total_trials);
  }
  if(s.turn!=null&&s.max_turn){
    const tPct=Math.round(s.turn/s.max_turn*100);
    S('barTrialPct',tPct+'%');
    bar('bur','bup',s.turn,s.max_turn);
  }
  if(s.run_num!=null&&s.total_runs)bar('brr','brp',s.run_index||0,s.total_runs);
  if(s.sim_index!=null)spk('sim_index',s.sim_index,'stc','stcv',_metricsExpanded?undefined:0);
  if(s.sim!=null)spk('sim',s.sim,'ssm','ssmv',_metricsExpanded?undefined:0);
  if(s.power!=null)spk('pow',s.power,'spw','spwv');
  if(s.disruption_magnitude!=null)spk('disrupt',s.disruption_magnitude,'src','srcv');
  const rn=(run&&s.run_num)?String(s.run_num):null;
  if(rn!==curRN){curRN=rn;paintGrid();}
  if(run!==_prevRunning){
    updateLogRate(run);updateScanRate(run);
    if(run){
      // Timer is server-driven — no client-side accumulation needed
    } else {
      stopRawlog();setTimeout(doScan,500);_stopRdpPoll();
      // Refresh hypothesis outcomes after run completes
      setTimeout(()=>{_hypLoaded=false;loadHyp();},2000);
      // Auto-fire is handled server-side; just show toast if there were queued runs
      if(s.queued_runs){toast('Starting queued runs: '+s.queued_runs,'ok');}
    }
    if(run&&_rdpRun!=null)_startRdpPoll();
    _prevRunning=run;
    checkResumeState();
  }
  const loading=run&&!s.run_num;
  const tbMid=document.querySelector('.tb-mid');
  if(loading){
    const _ml2=document.getElementById('ml');
    if(_ml2)_ml2.textContent='\\u23f3 loading\\u2026';
    if(tbMid)tbMid.classList.add('loading');
    startRawlog();
  } else {
    if(tbMid)tbMid.classList.remove('loading');
    if(run) stopRawlog();
    const _ml3=document.getElementById('ml');
    if(!run&&_ml3)_ml3.textContent='\\u2014';
  }
}
function colorMv(id,v,hi,lo,vlo){
  const e=document.getElementById(id);if(!e)return;
  e.classList.remove('hi','lo','vlo');if(v==null||isNaN(Number(v)))return;
  const n=Number(v);
  if(n>=hi)e.classList.add('hi');else if(n<vlo)e.classList.add('vlo');else if(n<lo)e.classList.add('lo');
}

// ── Console — scroll-to-load history ─────────────────────────────────────────
async function pollLog(){
  try{const r=await fetch('/log?since='+LLN);const d=await r.json();
  const ls=d.lines||[];if(ls.length>0){ls.forEach(addL);}
  if(d.total!=null)LLN=d.total;}catch(e){}
}
function updateLogRate(r){clearInterval(logInterval);if(r){logInterval=setInterval(pollLog,2000);}}

// Load older history when user scrolls near the top
async function _loadHistory(){
  if(_histLoading||LOG_START<=0)return;
  _histLoading=true;
  const ind=document.getElementById('con-load-indicator');
  if(ind){ind.classList.add('on');ind.textContent='Loading history\\u2026';}
  try{
    const r=await fetch('/log?before='+LOG_START);
    const d=await r.json();
    const lines=d.lines||[];
    const start=d.start!=null?d.start:Math.max(0,LOG_START-200);
    LOG_START=start;
    if(lines.length>0){
      const con=document.getElementById('con');
      const prevH=con.scrollHeight;
      // Prepend lines in order
      const frag=document.createDocumentFragment();
      lines.forEach(l=>{
        const el=_makeLogEl(l);
        if(el)frag.appendChild(el);
      });
      con.insertBefore(frag,con.firstChild);
      // Restore scroll position so view doesn't jump
      con.scrollTop=con.scrollTop+(con.scrollHeight-prevH);
    }
    if(ind){
      if(LOG_START<=0){ind.classList.remove('on');}
      else{ind.textContent='\\u25b2 scroll up for more';ind.classList.add('on');}
    }
  }catch(e){
    if(ind)ind.classList.remove('on');
  }
  _histLoading=false;
}

// Scroll listener on #con
document.addEventListener('DOMContentLoaded',()=>{
  const con=document.getElementById('con');
  if(con){
    con.addEventListener('scroll',()=>{
      if(con.scrollTop<50&&LOG_START>0&&!_histLoading&&!_clrMode)_loadHistory();
      AS=(con.scrollTop+con.clientHeight>=con.scrollHeight-40);
    });
  }
});

document.addEventListener('visibilitychange',()=>{
  if(!document.hidden){
    // Resume polling — do NOT clear console or reload from log file.
    // The 1GB log causes /log?tail=200 to block Flask for tens of seconds
    // while the sparse index rebuilds, freezing all other requests.
    clearInterval(logInterval);
    logInterval=setInterval(pollLog,_prevRunning?2000:10000);
    // Resume grid scan — browser throttles setInterval when tab is hidden
    doScan();loadTempGrid();updateScanRate(_prevRunning);
  }
});

function _makeLogEl(l){
  if(!l)return null;
  if(_sevFilter&&l.kind!==_sevFilter)return null;
  if(_searchStr&&!(l.text||'').toLowerCase().includes(_searchStr))return null;
  const d=document.createElement('div');
  d.className='ll'+(l.kind?' '+l.kind:'');
  let prefix='';
  if(l.ts){const dt=new Date(l.ts*1000);prefix='<span style="color:var(--t3);font-size:8px;margin-right:6px">'+
    String(dt.getHours()).padStart(2,'0')+':'+String(dt.getMinutes()).padStart(2,'0')+':'+
    String(dt.getSeconds()).padStart(2,'0')+'</span>';}
  d.innerHTML=prefix+_colourLine(escH(l.text||''));
  return d;
}
function addL(l){
  if(!l)return;
  if(_sevFilter&&l.kind!==_sevFilter)return;
  if(_searchStr&&!(l.text||'').toLowerCase().includes(_searchStr))return;
  const con=document.getElementById('con');if(!con)return;
  const d=_makeLogEl(l);if(!d)return;
  con.appendChild(d);
  if(AS){while(con.childNodes.length>1000)con.removeChild(con.firstChild);con.scrollTop=1e9;}
}
function clrC(){
  var c=document.getElementById('con');if(c)c.innerHTML='';
  _clrMode=true;
  LOG_START=Math.max(0,LLN-200);
  var ind=document.getElementById('con-load-indicator');
  if(ind&&LLN>0){ind.textContent='\u25b2 click to load history';ind.classList.add('on');}
  AS=true;
}
function conScrollBottom(){
  const c=document.getElementById('con');if(!c)return;
  _clrMode=false;AS=true;c.scrollTop=1e9;
}
function conRefresh(){
  // Reconnect SSE and re-poll without clearing visible content
  if(typeof es!=='undefined'&&es){try{es.close();}catch(e){}}
  setTimeout(sse,200);
  clearInterval(logInterval);
  logInterval=setInterval(pollLog,2000);
  pollLog();
  AS=true;
}
let _metricsExpanded=false;
let _savedLh=null;
function togMetricsExpand(){
  const btn=document.getElementById('metricsExpand');
  const grid=document.getElementById('panel-grid');
  const hspl=document.getElementById('hspl');
  const metrics=document.getElementById('panel-metrics');
  _metricsExpanded=!_metricsExpanded;
  if(_metricsExpanded){
    if(grid)grid.style.display='none';
    if(hspl)hspl.style.display='none';
    if(metrics)metrics.style.flex='1';
    setTimeout(()=>{
      if(!metrics)return;
      const avail=metrics.offsetHeight-50;
      const nSpk=document.querySelectorAll('canvas.spk').length||4;
      const hdrPer=22;
      const perSpk=Math.max(80,Math.floor((avail-nSpk*hdrPer)/nSpk));
      document.querySelectorAll('canvas.spk').forEach(c=>{c.style.height=perSpk+'px';});
      // Force redraw at new resolution
      setTimeout(()=>{
        dspk('stc',SD.sim_index);dspk('ssm',SD.sim);
        dspk('spw',SD.pow);dspk('src',SD.disrupt);
      },20);
    },50);
    if(btn){btn.classList.add('on');btn.title='Restore grid';}
  }else{
    if(grid)grid.style.display='';
    if(hspl)hspl.style.display='';
    if(metrics)metrics.style.flex='';
    document.querySelectorAll('canvas.spk').forEach(c=>{c.style.height='';});
    setTimeout(()=>{
      dspk('stc',SD.sim_index,0);dspk('ssm',SD.sim,0);
      dspk('spw',SD.pow);dspk('src',SD.disrupt);
    },20);
    if(btn){btn.classList.remove('on');btn.title='Expand metrics over grid';}
  }
}
function _colourLine(s){
  s=s.replace(/\\bR(\\d+)\\b/g,(m,n)=>'<span class="ht">'+m+'</span>');
  s=s.replace(/sim:(0\\.\\d+)/g,'sim:<span class="hs">$1</span>');
  s=s.replace(/disrupt:(\\d+)/g,(m,v)=>v!=='0'?'disrupt:<span class="hr">'+v+'</span>':m);
  return s;
}
function _addLBatch(lines,onDone,CHUNK){
  CHUNK=CHUNK||80;
  if(!lines||!lines.length){if(onDone)onDone();return;}
  let i=0;
  function _step(){
    const end=Math.min(i+CHUNK,lines.length);
    for(;i<end;i++)addL(lines[i]);
    if(i<lines.length)requestAnimationFrame(_step);
    else if(onDone)onDone();
  }
  requestAnimationFrame(_step);
}

// ── Rawlog ────────────────────────────────────────────────────────────────────
let _rawlogSize=0;
async function startRawlog(){
  if(_rawlogActive)return;_rawlogActive=true;
  _rawlogSize=0;
  async function _poll(){
    if(!_rawlogActive)return;
    if(LLN>0){_rawlogActive=false;
      const _rc=document.getElementById('con');
      if(_rc)_rc.querySelectorAll('.ll.raw').forEach(el=>el.remove());
      return;}
    try{const r=await fetch('/rawlog?n=40');const d=await r.json();
    const sz=d.size||0;
    // Only append if file grew
    if(sz>_rawlogSize){
      _rawlogSize=sz;
      const ls=d.lines||[];
      if(ls.length){
        const con=document.getElementById('con');
        // Clear previous rawlog lines and replace with current tail
        if(con){
          con.querySelectorAll('.ll.raw').forEach(el=>el.remove());
          ls.forEach(t=>{if(t.trim()){const el=document.createElement('div');
            el.className='ll raw';el.textContent=t;con.appendChild(el);}});
          if(AS)con.scrollTop=1e9;
        }
      }
    }}catch(e){}
    if(_rawlogActive)rawInterval=setTimeout(_poll,2000);
  }
  _poll();
}
function stopRawlog(){_rawlogActive=false;clearTimeout(rawInterval);}

// ── Run grid ──────────────────────────────────────────────────────────────────
function buildGrid(){
  const g=document.getElementById('phaseGrid');g.innerHTML='';
  PHASES.forEach(ph=>{
    const row=document.createElement('div');row.className='phase-row'+(ph.label.indexOf('Pooled')>=0?' pooled':'');
    const lbl=document.createElement('div');lbl.className='phase-lbl';
    lbl.dataset.phase=ph.label;lbl.textContent=ph.label;
    if(ph.label.indexOf('Analysis')>=0){lbl.style.cssText='color:var(--ac);border-bottom:1px dashed rgba(100,180,255,.4);padding-bottom:2px';}
    if(ph.label.indexOf('Pooled')>=0){lbl.style.cssText='color:var(--ye);border-bottom:1px dashed rgba(221,187,0,.4);padding-bottom:2px';}
    if(ph.label.indexOf('Cross-Model')>=0){row.className+=' pooled';lbl.style.cssText='color:var(--bl);border-bottom:1px dashed rgba(0,136,255,.4);padding-bottom:2px';}
    if(ph.label.indexOf('no GPU')>=0)lbl.classList.add('analysis-lbl');
    row.appendChild(lbl);
    const rg=document.createElement('div');rg.className='rg';
    ph.runs.forEach(n=>{
      const c=document.createElement('div');
      c.id='rc'+n;c.className='rc miss';
      c.textContent=String(n).padStart(2,'0');
      c.setAttribute('role','button');c.setAttribute('tabindex','0');
      c.title='Run '+n+(descs[String(n)]?' \u2014 '+descs[String(n)]:'');
      const _ttip='Run '+n+(descs[String(n)]?' \u2014 '+descs[String(n)]:'')+
        ' ['+((scanSt[String(n)])||'missing')+']';
      c.setAttribute('aria-label',_ttip);c.setAttribute('title',_ttip);
      c.addEventListener('click',e=>{
        const prevSel = new Set(selR);  // snapshot before toggleRun mutates selR
        toggleRun(n,e.ctrlKey||e.metaKey,e.shiftKey,c);
        const pop=document.getElementById('rdpop');
        if(e.ctrlKey||e.metaKey||e.shiftKey){
          // Multi-select modifier — update or open combined popup if 2+ selected
          if(selR.size>1) showMergedRdp(c);
          else if(selR.size===1) showRdp([...selR][0],c);
          else closeRdp();
        } else {
          // Plain click
          if(prevSel.size>1&&prevSel.has(n)){
            // Was clicking into a multi-selection — show combined popup for prev selection
            const arr=[...prevSel].sort((a,b)=>a-b);
            _rdpRun=arr;
            const _title=document.getElementById('rdpTitle');
            if(_title)_title.textContent=arr.length+' runs selected: '+arr.join(', ');
            if(pop){pop.classList.add('on');pop.style.width='340px';}
            _refreshRdpMulti(arr);
          } else if(pop&&pop.classList.contains('on')&&(Array.isArray(_rdpRun)?false:_rdpRun===n)&&prevSel.size===1){
            closeRdp();
          } else if(selR.size===1){
            showRdp(n,c);
          }
        }
      });
      c.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();
        c.click();}});
      c.addEventListener('contextmenu',e=>{e.preventDefault();showRdp(n,c);});
      c.addEventListener('dblclick',e=>{
        e.preventDefault();
        if(_navLayer===2&&_activeRunTemp!=null){
          // v0.77.1.3: no cache — just clear stale scan state, switch view,
          // and let doScan() fetch fresh data.
          fetch('/session',{method:'POST',headers:{'Content-Type':'application/json'},
            body:JSON.stringify({temperature:_activeRunTemp})})
            .then(()=>{
              scanSt={};descs={};
              navTo(3,_activeRunTemp);
              paintGrid();
              doScan();
            });
        }
      });
      rg.appendChild(c);
    });
    row.appendChild(rg);
    g.appendChild(row);
  });
  // Back cell — same style as run cells, appended to last run group
  const lastRg=g.querySelector('.rg:last-child')||g.lastElementChild?.querySelector('.rg');
  if(lastRg){
    const bc=document.createElement('div');
    bc.id='gridBackCell';bc.className='rc miss';
    bc.textContent='\u2190';bc.title='Back';
    bc.style.cssText='cursor:pointer;opacity:.5';
    bc.setAttribute('role','button');bc.setAttribute('tabindex','0');
    bc.addEventListener('click',()=>{
      if(_navLayer===3)navTo(2);
      else navTo(1);
    });
    lastRg.appendChild(bc);
  }
}
function _compressRuns(nums){
  // Convert [1,2,3,5,6,9] → "1-3,5-6,9"
  if(!nums.length)return '';
  const sorted=[...nums].sort((a,b)=>a-b);
  const parts=[];
  let start=sorted[0],end=sorted[0];
  for(let i=1;i<sorted.length;i++){
    if(sorted[i]===end+1){end=sorted[i];}
    else{parts.push(start===end?String(start):start+'-'+end);start=end=sorted[i];}
  }
  parts.push(start===end?String(start):start+'-'+end);
  return parts.join(', ');
}
function updateSelInfo(){
  const info=document.getElementById('selInfo');
  const box=document.getElementById('selBox');
  if(!info&&!box)return;
  if(selR.size===0){
    if(info){info.style.display='none';info.textContent='';}
    if(box)box.innerHTML='';
    return;
  }
  const ordered=[...EXEC_ORDER].filter(n=>selR.has(n));
  const spec=_compressRuns(ordered);
  if(info){info.style.display='';info.textContent=ordered.length+' run'+(ordered.length!==1?'s':'')+' selected';}
  if(box){
    box.innerHTML='';
    const s=document.createElement('span');
    s.className='sel-run';
    s.style.cssText='cursor:default;letter-spacing:.03em;padding:2px 7px';
    s.textContent=spec;
    s.title='Click to clear selection';
    s.addEventListener('click',()=>{
      selR.clear();paintGrid();
      const ri=document.getElementById('ri');if(ri)ri.value='';
      updateSelInfo();
    });
    box.appendChild(s);
  }
}
function toggleRun(n,multi,shift,cell){
  if(shift&&_lastRun!=null){
    // Range select: fill between _lastRun and n using visual grid order (PHASES)
    const VISUAL_ORDER=PHASES.flatMap(p=>p.runs);
    const ia=VISUAL_ORDER.indexOf(_lastRun),ib=VISUAL_ORDER.indexOf(n);
    if(ia>=0&&ib>=0){
      const lo=Math.min(ia,ib),hi=Math.max(ia,ib);
      VISUAL_ORDER.slice(lo,hi+1).forEach(r=>selR.add(r));
    }
  } else if(multi){
    // Ctrl/Cmd: toggle this run without affecting others
    if(selR.has(n))selR.delete(n);else selR.add(n);
  } else {
    // Single click: select only this run
    const had=selR.has(n)&&selR.size===1;
    selR.clear();
    if(!had)selR.add(n);
  }
  _lastRun=n;
  paintGrid();
  const ri=document.getElementById('ri');
  if(ri)ri.value=[...selR].sort((a,b)=>a-b).join(',');
  updateSelInfo();
}
// v0.77.1.3: grid caching removed entirely. Prior version snapshotted _tempGrid
// and scanSt into _modelCache on every model switch, nav change, scan tick, and
// grid load — then restored them for "instant display" when switching back.
// The cache had no invalidation: stale completion states, stale descs, and
// stale scan data lingered across real file-system changes, making the grid
// lie about what was on disk. Removed per user: "caching the grid files fucks
// shit up. we need to get rid of that whole function and just wait for shit
// to load". Every grid view now goes through /temp_grid and /scan fresh.
// _tempGrid retained as a plain in-memory variable for the current view only —
// it is populated by loadTempGrid() on each model/nav change and cleared when
// navigating away. No persistence across model switches.
let _tempGrid={};
let _navLayer=1; // 1=models, 2=alltemps, 3=per-temp
let _gridBusy=false; // v0.75.2.2: true while grid scan in flight — blocks run button
let _navTemp=null; // which temp when in layer 3
let _navModel=null; // {family,size,variant,label}

function navTo(layer, temp){
  _navLayer=layer;
  _navTemp=temp||null;
  // Show/hide layers
  const l1=document.getElementById('lyr1');
  const l2=document.getElementById('lyr2');
  if(l1){l1.classList.toggle('on',layer===1);}
  if(l2){l2.classList.toggle('on',layer>=2);}
  // Loading overlay
  const l2ld=document.getElementById('lyr2Loading');
  if(l2ld&&layer>=2){
    const tgLoaded=Object.keys(_tempGrid).length>0;
    l2ld.style.display=(layer===2&&!tgLoaded)?'flex':'none';
  }
  // Breadcrumb: show model name + temp dropdown when in grid view
  const s1=document.getElementById('bcSep1');
  const tsel=document.getElementById('bcTempSel');
  if(s1)s1.style.display=layer>=2?'':'none';
  if(tsel){
    tsel.style.display=layer>=2?'':'none';
    if(layer===2)tsel.value='all';
    else if(layer===3&&temp!=null)tsel.value=Number(temp).toFixed(1);
  }
  const bm=document.getElementById('bcModels');
  if(bm){
    const _ql={'4bit':'4-bit','8bit':'8-bit','fp16':'FP16'};
    const _qt=_navModel?_ql[_navModel.quant]||_navModel.quant||'':'';
    bm.textContent=_navModel?_navModel.label+(_qt?' ['+_qt+']':''):'Models';
  }
  // Hide pooled phase in per-temp view
  const pg=document.getElementById('phaseGrid');
  if(pg)pg.classList.toggle('per-temp',layer===3);
  // Summary
  const gs=document.getElementById('gridSummary');
  if(gs){
    if(layer===3)gs.textContent='T='+Number(temp).toFixed(1);
    else gs.textContent='';
  }
  // Save temp to session for per-temp view
  if(layer===3&&temp!=null){
    fetch('/session',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({temperature:Number(temp)})}).then(()=>doScan());
  }
  if(layer>=2)paintGrid();
}

function onBcTempChange(){
  const sel=document.getElementById('bcTempSel');if(!sel)return;
  // v0.77.1.3: no cache — clear stale per-temp scan state and let doScan()
  // repopulate. paintGrid() runs with cleared state (shows loading) until
  // scan returns.
  if(sel.value==='all'){
    navTo(2);
  } else {
    const t=parseFloat(sel.value);
    scanSt={};descs={};
    navTo(3,t);
    paintGrid();
    fetch('/session',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({temperature:t})})
      .then(()=>doScan());
  }
}

function navToModel(m){
  // v0.77.1.3: no cache. Every model switch clears all per-model state and
  // waits for /temp_grid + /scan to return before painting the real grid.
  // Prior version showed cached data instantly for "snappy" switching; in
  // practice the cache never invalidated so users saw whatever the grid
  // looked like last time they visited the model, not what's on disk now.
  // v0.79.4.9: clear the all-models sentinel so a real-model nav resets it.
  _allModelsMode=null;
  _navModel=m;
  scanSt={};_tempGrid={};descs={};
  _hypLoaded=false;
  paintGrid();
  navTo(2);
  const l2ld=document.getElementById('lyr2Loading');
  if(l2ld)l2ld.style.display='flex';
  // v0.75.2.2: block run button while grid loads
  _gridBusy=true;
  _setRunBtnEnabled(false);
  // Save model to session, then fetch fresh data.
  // v0.77.1.3: include quantization so _wsess updates _ACTIVE_QUANT. Prior
  // versions sent only family/size/variant — switching from a fp16 card to
  // a 4bit card left session.quantization stale, so get_paths() resolved to
  // the previous model's folder and completion gates (e.g. Run 0046) fired
  // false-positives against the wrong data.
  fetch('/session',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({model_family:m.family,model_size:m.size,model_variant:m.variant,
                         quantization:m.quant||'4bit'})})
    .then(()=>Promise.all([
      loadTempGrid(),
      new Promise(r=>{doScan();setTimeout(r,500);})
    ]))
    .then(()=>{
      if(l2ld)l2ld.style.display='none';
      _gridBusy=false;
      _setRunBtnEnabled(true);
      paintGrid();
    })
    .catch(()=>{_gridBusy=false;_setRunBtnEnabled(true);});
}

// v0.79.4.9: "All Models" dispatch. Sets _allModelsMode sentinel; the grid
// renders with a neutral "cross-model" status (derived by the highest
// completion across discovered models — shows what's pending across the
// fleet). doLaunch routes to the orchestrator's --all-models path when
// _allModelsMode is truthy.
let _allModelsMode=null;
function navToAllModels(models){
  _allModelsMode={models:models};
  _navModel={label:'All Models',family:'*',size:'*',variant:'*',quant:'*',all_models:true};
  scanSt={};_tempGrid={};descs={};
  _hypLoaded=false;
  paintGrid();
  navTo(2);
  const l2ld=document.getElementById('lyr2Loading');
  if(l2ld)l2ld.style.display='flex';
  // Fetch status by overlaying per-model /scan results: a run shows 'done'
  // only when every model has it done; otherwise 'partial' if any model has
  // it started; 'missing' if none. This is an accurate progress surface
  // for the fleet.
  Promise.all(models.map(m=>
    fetch('/scan?family='+encodeURIComponent(m.family)+
          '&size='+encodeURIComponent(m.size)+
          '&variant='+encodeURIComponent(m.variant)+
          '&temperature=0.0')
      .then(r=>r.json()).catch(()=>({}))
  )).then(results=>{
    const agg={};
    const allRuns=new Set();
    // Each result is a raw {run_num: status} dict (no wrapper).
    results.forEach(r=>{Object.keys(r||{}).forEach(k=>{
      if(k!=='error')allRuns.add(k);
    });});
    allRuns.forEach(k=>{
      const statuses=results.map(r=>(r||{})[k]||'missing');
      const allDone=statuses.every(s=>s==='done');
      const anyStart=statuses.some(s=>s==='done'||s==='partial');
      agg[k]=allDone?'done':anyStart?'partial':'missing';
    });
    scanSt=agg;
    if(l2ld)l2ld.style.display='none';
    _gridBusy=false;
    _setRunBtnEnabled(true);
    paintGrid();
  }).catch(()=>{
    if(l2ld)l2ld.style.display='none';
    _gridBusy=false;
    _setRunBtnEnabled(true);
  });
}

function paintGrid(){
  const allMode=(_navLayer===2);
  // v0.79.4.9: in All-Models view, scanSt already holds the cross-model
  // aggregate (per-run status across the fleet). Bypass the _tempGrid
  // path entirely — per-temp data would require per-model, per-temp scans
  // which are prohibitively expensive and not what this view represents.
  const allModelsView=!!(_allModelsMode&&_allModelsMode.models);
  const tgLoaded=Object.keys(_tempGrid).length>0;
  ALL.forEach(n=>{const c=document.getElementById('rc'+n);if(!c)return;
    let st,m;
    if(allMode && !allModelsView){
      if(tgLoaded&&_tempGrid[String(n)]){
        const tg=_tempGrid[String(n)];
        const vals=Object.values(tg);
        const doneCount=vals.filter(v=>v==='done').length;
        const anyData=vals.some(v=>v!=='missing');
        if(doneCount>=6){st='6/6 temps';m='done';}
        else if(anyData){st=doneCount+'/6 temps';m='part';}
        else{st='0/6 temps';m='miss';}
      } else {
        st='';m='miss';
      }
    } else {
      st=scanSt[String(n)]||'missing';
      m={'done':'done','partial':'part','missing':'miss','needs_et':'neet','et_partial':'etpt',
        'needs_rc':'nrec','corrupt':'nrec'}[st]||'miss';
    }
    c.textContent=p2(n);
    c.className='rc '+m+(selR.has(n)?' sel':'')+(curRN===String(n)?' now':'');
    const _d=descs[String(n)]||'';
    c.title='Run '+n+(_d?' - '+_d:'')+' ['+st+']';
    c.setAttribute('aria-label','Run '+n+(_d?' - '+_d:'')+' - '+st);
  });
  if(allMode && allModelsView){
    // v0.79.4.9: All-Models progress banner — sum across fleet.
    const done=Object.values(scanSt).filter(v=>v==='done').length;
    const part=Object.values(scanSt).filter(v=>v==='partial').length;
    const total=ALL.length;
    const nm=(_allModelsMode&&_allModelsMode.models)?_allModelsMode.models.length:0;
    S('gsn','All Models ('+nm+'): '+done+'/'+total+' done everywhere'+(part?', '+part+' partial':''));
    S('barTempsPct',Math.round((done/Math.max(1,total))*100)+'%');
    return;
  }
  if(allMode){
    const tgKeys=Object.keys(_tempGrid);
    if(tgKeys.length>0){
      const allDone=tgKeys.filter(k=>{const v=Object.values(_tempGrid[k]);return v.filter(s=>s==='done').length>=6;}).length;
      S('gsn','All Temps: '+allDone+'/'+ALL.length+' complete at all 6 temperatures');
      // Weighted progress: data collection ~90% of time, analysis ~10%
      let dataCells=0,dataOk=0,anaCells=0,anaOk=0;
      const _ANA=new Set([41,42,43,44,45,46,47,48,49,50,51]);
      tgKeys.forEach(k=>{
        const rn=parseInt(k);const v=Object.values(_tempGrid[k]);
        const ok=v.filter(s=>s!=='missing').length;
        if(_ANA.has(rn)){anaCells+=v.length;anaOk+=ok;}
        else{dataCells+=v.length;dataOk+=ok;}
      });
      const dataPct=dataCells>0?dataOk/dataCells:0;
      const anaPct=anaCells>0?anaOk/anaCells:0;
      const pct=Math.round(dataPct*90+anaPct*10);
      S('barTempsPct',pct+'%');
    } else {
      S('gsn','Loading...');
    }
  } else {
    // v0.79.4.9: needs_et / et_partial no longer emitted by scanner.
    // Run 0016 (E_t recovery meta-run) carries its own done/partial/missing.
    const done=Object.values(scanSt).filter(v=>v==='done').length;
    const r48=scanSt['48']||'missing';
    const r48note=r48==='done'?'':r48==='partial'?' \\u00b7 E\\u209c partial':' \\u00b7 E\\u209c pending';
    S('gsn','T='+(_navTemp!=null?Number(_navTemp).toFixed(1):'?')+': '+done+'/'+ALL.length+' done'+r48note);
  }
}
async function loadTempGrid(){
  try{const r=await fetch('/temp_grid');const d=await r.json();
    if(d.error){console.error('temp_grid error:',d.error);return;}
    _tempGrid=d.grid||{};
    // v0.77.1.3: no cache write — _tempGrid is the single live source
    console.log('temp_grid loaded:',Object.keys(_tempGrid).length,'runs');
    if(d._debug)console.log('Run 0019 debug:',d._debug);
    const l2ld=document.getElementById('lyr2Loading');
    if(l2ld)l2ld.style.display='none';
    paintGrid();
  }catch(e){console.error('loadTempGrid failed:',e);}
}
async function buildModelCards(){
  try{
    const r=await fetch('/models_collected');const d=await r.json();
    const container=document.getElementById('modelCards');if(!container)return;
    container.innerHTML='';
    const models=d.models||[];
    if(models.length===0){
      container.innerHTML='<div style="color:var(--t3);font-size:11px;padding:12px">No models collected yet. Add a model below to start.</div>';
      return;
    }
    // v0.79.4.9: "All Models" synthetic card at the top of the list.
    // When selected, runs fan out across every discovered model (smart
    // loader groups runs by model to minimize subprocess startup cost).
    // Skipped when only one model is present — no point showing it.
    if(models.length>=2){
      const allCard=document.createElement('div');
      allCard.className='model-card all-models';
      const nModels=models.length;
      const nLabels=models.map(m=>m.label).join(', ');
      allCard.innerHTML='<div class="mc-name">All Models</div>'+
        '<div class="mc-thead" style="color:var(--ac);opacity:0.7">DISPATCH ACROSS EVERY MODEL</div>'+
        '<div class="mc-tr" style="color:var(--t2);padding-top:6px">'+
          '<span class="mc-tl" style="min-width:auto">'+nModels+' models</span></div>'+
        '<div class="mc-tr" style="color:var(--t3);font-size:10px;opacity:0.75">'+
          escH(nLabels.length>60?nLabels.slice(0,57)+'...':nLabels)+'</div>'+
        '<div class="mc-summary" style="border-top-color:rgba(80,140,220,0.3)">'+
          'Select runs \u2192 hit play \u2192 fires on all models sequentially'+
        '</div>';
      allCard.onclick=()=>navToAllModels(models);
      container.appendChild(allCard);
    }
    models.forEach(m=>{
      const card=document.createElement('div');card.className='model-card';
      const dpt=m.data_per_temp||0;
      const apt=m.ana_per_temp||0;
      const td=m.total_data||0;
      const tdm=m.total_data_max||0;
      const ta=m.total_ana||0;
      const tam=m.total_ana_max||0;
      const allDone=td>=tdm&&ta>=tam&&tdm>0;
      // Per-temp rows
      let tempHtml='';
      (m.temps||[]).forEach(t=>{
        const tl=t.temp<=0?'T=0.0':'T='+t.temp.toFixed(1);
        const dd=t.data||0;const aa=t.ana||0;
        const dOk=dd>=dpt&&dpt>0;
        const aOk=aa>=apt&&apt>0;
        const rowDone=dOk&&aOk;
        const empty=dd===0&&aa===0;
        const cls=rowDone?'mc-tr done':empty?'mc-tr empty':'mc-tr part';
        tempHtml+='<div class="'+cls+'"><span class="mc-tl">'+tl+'</span>'+
          '<span class="mc-td"'+(dOk?' style="color:var(--go)"':dd>0?' style="color:var(--wa)"':' style="color:var(--t3);opacity:0.5"')+'>'+dd+'/'+dpt+'</span>'+
          '<span class="mc-td"'+(aOk?' style="color:var(--go)"':aa>0?' style="color:var(--wa)"':' style="color:var(--t3);opacity:0.5"')+'>'+aa+'/'+apt+'</span></div>';
      });
      // Pooled row
      const pr=m.pooled_runs||0;const pt=m.pooled_total||3;
      const pOk=pr>=pt&&pt>0;
      const pCls=pOk?'mc-tr done':pr>0?'mc-tr part':'mc-tr empty';
      tempHtml+='<div class="'+pCls+'"><span class="mc-tl">Pooled</span>'+
        '<span class="mc-td"'+(pOk?' style="color:var(--go)"':pr>0?' style="color:var(--wa)"':'')+'>'+pr+'/'+pt+'</span>'+
        '<span class="mc-td"></span></div>';
      const _qlm={'4bit':'4-bit','8bit':'8-bit','fp16':'FP16'};
      const _qd=m.quant?_qlm[m.quant]||m.quant:'';
      card.innerHTML='<div class="mc-name">'+escH(m.label)+
        (_qd?'<span style="color:var(--t3);font-size:9px;margin-left:6px;font-weight:400">['+escH(_qd)+']</span>':'')+
        '</div>'+
        '<div class="mc-thead"><span class="mc-tl"></span><span class="mc-td">Data</span><span class="mc-td">Analysis</span></div>'+
        tempHtml+
        '<div class="mc-summary'+(allDone?' done':'')+'">'+
        '<span'+(td>=tdm&&tdm>0?' style="color:var(--go)"':td>0?' style="color:var(--wa)"':'')+'>'+td+'/'+tdm+' data</span>'+
        ' &middot; '+
        '<span'+(ta>=tam&&tam>0?' style="color:var(--go)"':ta>0?' style="color:var(--wa)"':'')+'>'+ta+'/'+tam+' analysis</span>'+
        '</div>';
      card.onclick=()=>navToModel(m);
      container.appendChild(card);
    });
  }catch(e){console.error('buildModelCards:',e);}
}
let _amHierarchy={};
async function initAddModelForm(){
  try{
    const r=await fetch('/models_data');const d=await r.json();
    _amHierarchy=d.hierarchy||{};
    const fam=document.getElementById('amFam');if(!fam)return;
    fam.innerHTML='';
    const famNames={'llama':'LLaMA','gemma':'Gemma','qwen':'Qwen','mistral':'Mistral',
      'phi':'Phi','deepseek':'DeepSeek','falcon':'Falcon','yi':'Yi'};
    Object.keys(_amHierarchy).sort().forEach(f=>{
      const o=document.createElement('option');o.value=f;
      o.textContent=famNames[f]||f.charAt(0).toUpperCase()+f.slice(1);
      fam.appendChild(o);
    });
    amLoadGens();
  }catch(e){console.error('initAddModelForm:',e);}
}
function amLoadGens(){
  const fam=document.getElementById('amFam');
  const gen=document.getElementById('amGen');
  if(!fam||!gen)return;
  const gens=_amHierarchy[fam.value]||{};
  const famNames={'llama':'LLaMA','gemma':'Gemma','qwen':'Qwen','mistral':'Mistral',
    'phi':'Phi','deepseek':'DeepSeek'};
  const fn=famNames[fam.value]||fam.value;
  gen.innerHTML='';
  Object.keys(gens).sort((a,b)=>{
    const na=parseFloat(a)||0,nb=parseFloat(b)||0;
    return na-nb||a.localeCompare(b);
  }).forEach(g=>{
    const o=document.createElement('option');o.value=g;
    o.textContent=fn+' '+g;
    gen.appendChild(o);
  });
  amLoadSzs();
}
function amLoadSzs(){
  const fam=document.getElementById('amFam');
  const gen=document.getElementById('amGen');
  const sz=document.getElementById('amSz');
  if(!fam||!gen||!sz)return;
  const szs=(_amHierarchy[fam.value]||{})[gen.value]||{};
  sz.innerHTML='';
  Object.entries(szs).sort((a,b)=>{
    const na=parseFloat(a[0])||0,nb=parseFloat(b[0])||0;
    return na-nb;
  }).forEach(([k,v])=>{
    const o=document.createElement('option');o.value=v.subfamily;
    o.textContent=k.toUpperCase();
    o.dataset.path=v.path;o.dataset.variant=v.variant;o.dataset.name=v.name;
    o.dataset.subfamily=v.subfamily;
    sz.appendChild(o);
  });
}
async function amLoadSizes(){amLoadGens();} // backward compat
function hfBlur(el){
  const raw=el.value.trim();
  if(!raw){el.dataset.raw='';document.getElementById('amHfStatus').textContent='';return;}
  el.dataset.raw=raw;
  // Save to session
  fetch('/session',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({hf_token:raw})});
  // Also update settings field if it exists
  const ihf=document.getElementById('ihf');if(ihf)ihf.value=raw;
  // Mask display
  el.type='password';el.value='\u25cf\u25cf\u25cf\u25cf\u25cf\u25cf\u25cf\u25cf';
  document.getElementById('amHfStatus').innerHTML=raw.startsWith('hf_')?
    '<span style="color:var(--go)">\u2713</span>':
    '<span style="color:var(--wa)">?</span>';
}
async function amStart(){
  const fam=document.getElementById('amFam');
  const gen=document.getElementById('amGen');
  const sz=document.getElementById('amSz');
  if(!fam||!sz||!fam.value||!sz.value){toast('Select a model','wa');return;}
  const sel=sz.options[sz.selectedIndex];
  const sf=sel.dataset.subfamily||sz.value;
  // Extract size from subfamily: "qwen2.5-0.5b" → "0.5b"
  const sfParts=sf.split('-');
  const szKey=sfParts[sfParts.length-1]||sz.value;
  const body={model_family:fam.value,model_size:szKey,
    model_variant:sel&&sel.dataset.variant?sel.dataset.variant:'abliterated',
    model_path:sel&&sel.dataset.path?sel.dataset.path:'',
    model_name:sel&&sel.dataset.name?sel.dataset.name:''};
  const hfEl=document.getElementById('amHf');
  const hf=hfEl&&hfEl.dataset.raw?hfEl.dataset.raw:'';
  if(hf)body.hf_token=hf;
  try{
    await fetch('/add_model',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body)});
    await buildModelCards();
    toast('Model added','ok');
  }catch(e){toast('Error: '+e,'er');}
}
// ── Temperature presets ──────────────────────────────────────────────────────
const _TEMP_PRESETS={
  standard:[0.0,0.2,0.4,0.6,0.8,1.0],
  log:[0.0,0.1,0.2,0.3,0.4,0.6,0.8,1.0,1.2,1.4,2.0]
};
const _ALL_TEMPS=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.4,1.6,1.8,2.0];
let _activeTemps=_TEMP_PRESETS.standard.slice();
function buildTempChips(){
  const c=document.getElementById('tempChips');if(!c)return;
  c.innerHTML='';
  _ALL_TEMPS.forEach(t=>{
    const chip=document.createElement('span');
    chip.className='tc'+((_activeTemps.includes(t))?' on':'');
    chip.textContent=t.toFixed(1);
    chip.onclick=()=>toggleTemp(t);
    c.appendChild(chip);
  });
}
function toggleTemp(t){
  const idx=_activeTemps.indexOf(t);
  if(idx>=0)_activeTemps.splice(idx,1);else _activeTemps.push(t);
  _activeTemps.sort((a,b)=>a-b);
  // Switch to Custom if current preset doesn't match
  const sel=document.getElementById('iTempPreset');
  if(sel){
    let matched=false;
    for(const[k,v] of Object.entries(_TEMP_PRESETS)){
      if(JSON.stringify(v)===JSON.stringify(_activeTemps)){sel.value=k;matched=true;break;}
    }
    if(!matched)sel.value='custom';
  }
  buildTempChips();
  saveActiveTemps();
}
function onTempPresetChange(){
  const sel=document.getElementById('iTempPreset');if(!sel)return;
  const preset=_TEMP_PRESETS[sel.value];
  if(preset){_activeTemps=preset.slice();buildTempChips();saveActiveTemps();}
}
function saveActiveTemps(){
  fetch('/session',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({active_temps:_activeTemps})});
}
async function autoSaveParams(){
  try{
    const body={
      trials:parseInt(document.getElementById('itr').value)||100,
      seed:parseInt(document.getElementById('ise').value)||42,
      quantization:document.getElementById('iqt').value||'4bit'
    };
    const _hfEl=document.getElementById('amHf');
    const hf=_hfEl&&_hfEl.dataset.raw?_hfEl.dataset.raw:'';
    if(hf)body.hf_token=hf;
    await fetch('/session',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body)});
    // Quick flash to confirm
    const lyr=document.getElementById('lyr1');
    if(lyr){lyr.style.opacity='0.7';setTimeout(()=>{lyr.style.opacity='1';},150);}
  }catch(e){}
}
function resetDefaults(){
  document.getElementById('itr').value=100;
  document.getElementById('ise').value=42;
  document.getElementById('iqt').value='4bit';
  const gpu=document.getElementById('iGpu');if(gpu)gpu.value='single';
  _applyCC(true);_applyGC(true);_applyUL(true);
  autoSaveParams();savePerfSettings();
}
async function refreshParams(){
  await loadSt();
  const lyr=document.getElementById('lyr1');
  if(lyr){lyr.style.opacity='0.7';setTimeout(()=>{lyr.style.opacity='1';},150);}
}
async function onTempChange(){
  const sel=document.getElementById('bcTempSel');if(!sel)return;
  const v=sel.value;
  if(v==='all'){
    navTo(2);
  }else if(v!=='auto'){
    navTo(3, Number(v));
    await fetch('/session',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({temperature:Number(v)})});
  }
  paintGrid();doScan();
}
function selP(spec,btn){
  if(btn&&btn.classList.contains('hi')){
    document.querySelectorAll('.pr').forEach(b=>b.classList.remove('hi'));
    selR.clear();paintGrid();
    const ri=document.getElementById('ri');if(ri)ri.value='';
    updateSelInfo();return;
  }
  document.querySelectorAll('.pr').forEach(b=>b.classList.remove('hi'));
  if(btn)btn.classList.add('hi');
  selR.clear();
  spec.split(',').forEach(part=>{
    const m=part.trim().match(/^(\\d+)(?:-(\\d+))?$/);
    if(!m)return;
    const a=parseInt(m[1]),b=m[2]?parseInt(m[2]):a;
    for(let i=a;i<=b;i++)if(ALL.includes(i))selR.add(i);
  });
  paintGrid();
  const ri=document.getElementById('ri');
  if(ri)ri.value=spec;
  updateSelInfo();
}
function goRun(n){selP(String(n),null);}

// ── Scan ──────────────────────────────────────────────────────────────────────
function updateScanRate(r){clearInterval(scanInterval);scanInterval=setInterval(()=>{doScan();if(r)loadTempGrid();},r?30000:120000);}
let _atcTick=0;
async function doScan(){
  // v0.79.4.9: while in All-Models view, a single session-scoped /scan
  // returns the currently-set model's status and would overwrite the
  // cross-model aggregate. Refresh the aggregate instead by re-running
  // navToAllModels' per-model overlay logic inline.
  if(_allModelsMode && _allModelsMode.models){
    try{
      const results=await Promise.all(_allModelsMode.models.map(m=>
        fetch('/scan?family='+encodeURIComponent(m.family)+
              '&size='+encodeURIComponent(m.size)+
              '&variant='+encodeURIComponent(m.variant)+
              '&temperature=0.0')
          .then(r=>r.json()).catch(()=>({}))
      ));
      const agg={};
      const allRuns=new Set();
      results.forEach(r=>{Object.keys(r||{}).forEach(k=>{
        if(k!=='error')allRuns.add(k);
      });});
      allRuns.forEach(k=>{
        const statuses=results.map(r=>(r||{})[k]||'missing');
        const allDone=statuses.every(s=>s==='done');
        const anyStart=statuses.some(s=>s==='done'||s==='partial');
        agg[k]=allDone?'done':anyStart?'partial':'missing';
      });
      scanSt=agg;
      paintGrid();
      if(_hypLoaded)_refreshHypRunStatus();
    }catch(e){}
    // No temp-grid refresh here — /temp_grid is session-scoped and would
    // overwrite the aggregate with whichever model happens to be set. The
    // All-Models view stays at the scan-aggregate level (no per-temp drill).
    _atcTick++;
    return;
  }
  try{
    const r=await fetch('/scan');const d=await r.json();
    scanSt=d;
    // v0.77.1.3: no cache write — scanSt is live-only per model view
    paintGrid();
    if(_hypLoaded)_refreshHypRunStatus();
  }catch(e){}
  // Refresh popup if open and run active
  if(_rdpRun!=null&&_prevRunning&&_navLayer!==2)_refreshRdp(_rdpRun);
  // Check all-temps-complete every ~60s (scan fires every 5-10s)
  _atcTick++;
  // Refresh all-temps grid during runs (~30s), less often when idle (~120s)
  if(_prevRunning&&_atcTick%6===0) loadTempGrid();
  else if(!_prevRunning&&_atcTick%24===0) loadTempGrid();
}

// ── Hypothesis tab ────────────────────────────────────────────────────────────
const STAT_CLS={supported:'sup',disproven:'dis',inconclusive:'inc',pending:'pen',ready:'inc'};
const STAT_SYM={supported:'\\u2713',disproven:'\\u2717',inconclusive:'\\u223c',pending:'\\u00b7',ready:'\\u25cb'};

async function loadHyp(live){
  // live=true uses /hyp_live (computes from CSV, slow first call).
  // live=false uses /outcomes (reads static JSON, fast).
  // Tab click and Refresh use live. Init uses fast.
  const endpoint=live?'/hyp_live':'/outcomes';
  const scroll=document.getElementById('hypScroll');
  try{
    const [ro,rd]=await Promise.all([fetch(endpoint),fetch('/hyp_deps')]);
    if(!ro.ok||!rd.ok){
      if(scroll)scroll.innerHTML='<div style="color:var(--er);font-size:10px">Server error loading hypothesis data ('+
        (!ro.ok?endpoint+' '+ro.status:'/hyp_deps '+rd.status)+').<br>Check .iota_flask.log for details.</div>';
      return;
    }
    const [otxt,dtxt]=await Promise.all([ro.text(),rd.text()]);
    let outcomes,deps;
    try{outcomes=JSON.parse(otxt);}catch(je){
      if(scroll)scroll.innerHTML='<div style="color:var(--er);font-size:10px">'+endpoint+' JSON error: '+escH(String(je))+'<br>First 200 chars: '+escH(otxt.slice(0,200))+'</div>';
      return;
    }
    try{deps=JSON.parse(dtxt);}catch(je){
      if(scroll)scroll.innerHTML='<div style="color:var(--er);font-size:10px">/hyp_deps JSON error: '+escH(String(je))+'<br>First 200 chars: '+escH(dtxt.slice(0,200))+'</div>';
      return;
    }
    if(outcomes.error){
      if(scroll)scroll.innerHTML='<div style="color:var(--er);font-size:10px">Outcomes error: '+escH(outcomes.error)+'</div>';
      return;
    }
    if(deps.error){
      if(scroll)scroll.innerHTML='<div style="color:var(--er);font-size:10px">Deps error: '+escH(deps.error)+'</div>';
      return;
    }
    _outcomes=outcomes;_hypDeps=deps;
    _hypLoaded=true;
    try{
      _renderHyp(outcomes,deps);
    }catch(re){
      console.error('_renderHyp error:',re);
      if(scroll)scroll.innerHTML='<div style="color:var(--er);font-size:10px">Render error: '+escH(String(re))+'<br><small>'+escH(re&&re.stack?re.stack.split('\\n').slice(0,3).join(' | '):'')+'</small></div>';
      _hypLoaded=false;
    }
  }catch(e){
    console.error('loadHyp error:',e);
    if(scroll)scroll.innerHTML='<div style="color:var(--er);font-size:10px">Failed to load hypothesis data: '+
      escH(String(e))+'<br><small>'+escH(e&&e.stack?e.stack.split('\\n')[0]:'')+'</small></div>';
    // Don't set _hypLoaded=true — allow retry on next tab visit
  }
}
function _runChipCls(n){
  const st=scanSt[String(n)]||'miss';
  if(st==='done')return 'done';
  if(st==='missing'||st==='miss')return 'miss';
  return 'part';
}
function _renderHyp(outcomes,deps){
  const scroll=document.getElementById('hypScroll');
  if(!scroll)return;
  scroll.innerHTML='';
  Object.entries(H_PHASES).forEach(([phase,hyps])=>{
    const sec=document.createElement('div');sec.className='hyp-phase';
    const lbl=document.createElement('div');lbl.className='hyp-phase-lbl';
    const doneCount=hyps.filter(([hid])=>(outcomes[hid]||{}).status==='supported').length;
    lbl.innerHTML=escH(phase)+'<span style="font-size:8px;color:var(--t3)">'+
      doneCount+'/'+hyps.length+' supported</span>';
    sec.appendChild(lbl);
    hyps.forEach(([hid,hname])=>{
      const o=outcomes[hid]||{status:'pending'};
      const dep=deps[hid]||{data_runs:[],upstream_runs:[]};
      const allRuns=[...new Set([...(dep.data_runs||[]),...(dep.upstream_runs||[])])].sort((a,b)=>a-b);
      // Detect "ready" — all data runs done but stats not yet run
      let effStatus=o.status||'pending';
      if(effStatus==='pending'&&allRuns.length>0){
        const allDone=allRuns.every(n=>scanSt[String(n)]==='done');
        if(allDone)effStatus='ready';
      }
      const row=document.createElement('div');row.className='hyp-row';
      row.dataset.hid=hid;
      const chk=document.createElement('input');chk.type='checkbox';chk.className='hyp-chk';
      chk.title='Add runs to queue';
      chk.addEventListener('change',()=>{
        allRuns.forEach(n=>{if(chk.checked)_hypQueue.add(n);else _hypQueue.delete(n);});
        _updateHypQueue();
        row.classList.toggle('sel',chk.checked);
      });
      const cls=STAT_CLS[effStatus]||'pen';
      const sym=STAT_SYM[effStatus]||'\\u00b7';
      const idSpan=document.createElement('span');idSpan.className='hyp-id';
      idSpan.style.color=effStatus==='supported'?'var(--go)':
        effStatus==='disproven'?'var(--er)':
        effStatus==='inconclusive'||effStatus==='ready'?'var(--wa)':'var(--t2)';
      idSpan.textContent=sym+' '+hid;
      const nameSpan=document.createElement('span');nameSpan.className='hyp-name';
      nameSpan.title=hname;nameSpan.textContent='null: '+hname;
      if(effStatus==='ready'){
        nameSpan.textContent='null: '+hname+' [data ready — run stats]';
      }
      const runsDiv=document.createElement('div');runsDiv.className='hyp-runs';
      if(allRuns.length>0){
        const lbl2=document.createElement('span');lbl2.style.cssText='font-size:8px;color:var(--t3);margin-right:2px';
        lbl2.textContent='runs:';runsDiv.appendChild(lbl2);
        allRuns.forEach(n=>{
          const chip=document.createElement('span');
          chip.className='hyp-run-chip '+_runChipCls(n);
          chip.textContent=String(n).padStart(2,'0');
          chip.title='Run '+n+(scanSt[String(n)]?' - '+scanSt[String(n)]:'');
          chip.addEventListener('click',e=>{e.stopPropagation();selP(String(n),null);});
          runsDiv.appendChild(chip);
        });
      } else {
        const noRuns=document.createElement('span');noRuns.style.cssText='font-size:8px;color:var(--t3)';
        noRuns.textContent='analysis only';runsDiv.appendChild(noRuns);
      }
      row.appendChild(chk);row.appendChild(idSpan);
      const mid=document.createElement('div');mid.style.cssText='flex:1;min-width:0';
      mid.appendChild(nameSpan);mid.appendChild(runsDiv);
      row.appendChild(mid);
      // v0.76.0.7 / v0.77.1.3: wire row click to showHypDetail. Prior: function
      // existed but was never bound to any event — hypothesis value/p-value/
      // implication were computed and written to JSON but never surfaced in
      // the UI. Checkbox and run chips stop propagation so those interactions
      // don't trigger the detail view.
      chk.addEventListener('click',e=>e.stopPropagation());
      row.style.cursor='pointer';
      row.addEventListener('click',()=>{showHypDetail(hid);});
      sec.appendChild(row);
    });
    scroll.appendChild(sec);
  });
  updateHypSummary(outcomes);
}
function _refreshHypRunStatus(){
  // Update run chip colours after scan
  document.querySelectorAll('.hyp-run-chip[title]').forEach(chip=>{
    const n=parseInt(chip.textContent);if(isNaN(n))return;
    chip.className='hyp-run-chip '+_runChipCls(n);
    chip.title='Run '+n+(scanSt[String(n)]?' - '+scanSt[String(n)]:'');
  });
}
function _updateHypQueue(){
  const info=document.getElementById('hyp-queue-info');
  const btn=document.getElementById('hyp-launch-btn');
  if(_hypQueue.size===0){
    if(info)info.textContent='No runs selected';
    if(btn)btn.disabled=true;
    return;
  }
  // Sort by execution order
  const ordered=[...EXEC_ORDER].filter(n=>_hypQueue.has(n));
  const extra=[..._hypQueue].filter(n=>!EXEC_ORDER.includes(n)).sort((a,b)=>a-b);
  const all=[...ordered,...extra];
  if(info)info.textContent=all.length+' runs queued: '+all.join(', ');
  if(btn)btn.disabled=false;
}
function hypLaunchQueue(){
  const ordered=[...EXEC_ORDER].filter(n=>_hypQueue.has(n));
  const extra=[..._hypQueue].filter(n=>!EXEC_ORDER.includes(n)).sort((a,b)=>a-b);
  const all=[...ordered,...extra];
  if(!all.length)return;
  selP(all.join(','),null);
  doLaunch();
}
function hypSelectAll(sel){
  document.querySelectorAll('.hyp-chk').forEach(chk=>{
    if(chk.checked!==sel){chk.checked=sel;chk.dispatchEvent(new Event('change'));}
  });
}
function showHypDetail(hid){
  _selHyp=hid;
  const o=_outcomes[hid];const det=document.getElementById('hypDetail');
  if(!det)return;
  if(!o){
    det.innerHTML='<div style="font-weight:600;color:var(--t1)">'+escH(hid)+'</div>'+
      '<div style="color:var(--t3);font-size:9px;margin-top:2px">No outcome data yet. Run stats (all-stats) after analysis completes.</div>';
    return;
  }
  // v0.76.0.7 / v0.77.1.3: structured multi-row view. Header + metric+value +
  // extra fields (varies by hypothesis) + full implication text. Long nested
  // values render as "{N fields}" summaries to avoid flooding.
  const cls=STAT_CLS[o.status]||'pen';
  const statColor=o.status==='supported'?'var(--go)':
    o.status==='disproven'?'var(--er)':
    o.status==='inconclusive'?'var(--wa)':'var(--t2)';
  const fmtVal=v=>{
    if(v==null)return '\\u2014';
    if(typeof v==='number'){
      if(!isFinite(v))return String(v);
      if(Math.abs(v)>=1000||Math.abs(v)<0.001&&v!==0)return v.toExponential(3);
      return v.toFixed(4);
    }
    if(typeof v==='boolean')return v?'true':'false';
    if(Array.isArray(v))return '['+v.length+' items]';
    if(typeof v==='object')return '{'+Object.keys(v).length+' fields}';
    return String(v);
  };
  const _skipKeys=new Set(['status','metric','value','implication']);
  const extras=Object.entries(o).filter(([k])=>!_skipKeys.has(k));
  const metric=o.metric||'';
  const val=fmtVal(o.value);
  let html='<div style="display:flex;align-items:center;gap:8px;margin-bottom:3px">'+
    '<span style="font-weight:600;color:var(--t1);font-size:11px">'+escH(hid)+'</span>'+
    '<span style="padding:1px 6px;border-radius:3px;font-size:9px;font-weight:600;'+
    'background:'+statColor+';color:var(--bg);text-transform:uppercase">'+escH(o.status||'pending')+'</span>'+
    (metric?'<span style="color:var(--t2);font-size:9px;font-family:monospace">'+escH(metric)+' = '+escH(val)+'</span>':'')+
    '</div>';
  if(extras.length){
    html+='<div style="font-size:9px;color:var(--t3);font-family:monospace;margin-bottom:3px;'+
      'display:flex;flex-wrap:wrap;gap:2px 10px">';
    extras.forEach(([k,v])=>{
      html+='<span><span style="color:var(--t3)">'+escH(k)+':</span> '+
        '<span style="color:var(--t2)">'+escH(fmtVal(v))+'</span></span>';
    });
    html+='</div>';
  }
  html+='<div style="font-size:10px;color:var(--t1);line-height:1.35">'+escH(o.implication||'No implication text.')+'</div>';
  det.innerHTML=html;
}
function updateHypSummary(outcomes){
  const cnts={supported:0,disproven:0,inconclusive:0,pending:0};
  Object.values(outcomes).forEach(o=>{cnts[o.status]=(cnts[o.status]||0)+1;});
  const total=Object.values(cnts).reduce((a,b)=>a+b,0);
  if(!total){['Sup','Dis','Inc','Pen'].forEach(k=>S('hyp'+k,'\\u2014'));return;}
  S('hypSup',cnts.supported||0);S('hypDis',cnts.disproven||0);
  S('hypInc',cnts.inconclusive||0);S('hypPen',cnts.pending||0);
}

// ── Settings ──────────────────────────────────────────────────────────────────
function _applyCC(v){_CC=v;const b=document.getElementById('tCC');
  b.textContent=v?'ON':'OFF';
  b.classList.toggle('on',v);b.setAttribute('aria-pressed',v.toString());}
function togCC(){_applyCC(!_CC);_savePerf();}
let _GC=true,_UL=true;
function _applyGC(v){_GC=v;const b=document.getElementById('tGC');if(!b)return;
  b.textContent=v?'ON':'OFF';
  b.classList.toggle('on',v);b.setAttribute('aria-pressed',v.toString());}
function togGC(){_applyGC(!_GC);_savePerf();}
function _applyUL(v){_UL=v;const b=document.getElementById('tUL');if(!b)return;
  b.textContent=v?'ON':'OFF';
  b.classList.toggle('on',v);b.setAttribute('aria-pressed',v.toString());}
function togUL(){_applyUL(!_UL);_savePerf();}
let _RS=false;
function _applyRS(v){_RS=v;const b=document.getElementById('tRS');if(!b)return;
  b.textContent=v?'ON':'OFF';
  b.classList.toggle('on',v);b.setAttribute('aria-pressed',v.toString());}
function togRS(){_applyRS(!_RS);_savePerf();}
let _LP=true;
function _applyLP(v){_LP=v;const b=document.getElementById('tLP');if(!b)return;
  b.textContent=v?'ON':'OFF';
  b.classList.toggle('on',v);b.setAttribute('aria-pressed',v.toString());}
function togLP(){_applyLP(!_LP);_savePerf();}
function _savePerf(){
  fetch('/session',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({clear_cache:_CC,force_gc:_GC,unload_between_runs:_UL,restore_runs:_RS,low_priority:_LP})});
}
async function savePerfSettings(){
  try{await fetch('/session',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({clear_cache:_CC,force_gc:_GC,unload_between_runs:_UL,restore_runs:_RS,low_priority:_LP})});
    toast('Performance settings saved','ok');
  }catch(e){toast('Save failed','er');}
}
async function loadSt(){
  try{const r=await fetch('/session');const s=await r.json();
  document.getElementById('itr').value=s.trials??100;
  const _tsel=document.getElementById('bcTempSel');
  // Don't override temp dropdown — it defaults to "All temps" for the grid view.
  // The session temperature is for collection, not for the grid display mode.
  const _tv=Number(s.temperature??0).toFixed(1);
  document.getElementById('ise').value=s.seed??42;
  const _tI=document.getElementById('tbTemp');
  if(_tI)_tI.textContent='T='+_tv;
  const ri=document.getElementById('ri');if(ri){
    if(s.restore_runs&&s.runs){ri.value=s.runs;selP(s.runs,null);}
    else{ri.value='';}
  }
  const qt=document.getElementById('iqt');
  if(qt){const q=s.quantization||'4bit';for(let o of qt.options)if(o.value===q)o.selected=true;}
  _applyCC(s.clear_cache!==undefined?s.clear_cache:true);
  _applyGC(s.force_gc!==undefined?s.force_gc:true);
  _applyUL(s.unload_between_runs!==undefined?s.unload_between_runs:true);
  _applyRS(s.restore_runs!==undefined?s.restore_runs:false);
  _applyLP(s.low_priority!==undefined?s.low_priority:true);
  const mc=document.getElementById('mCur');
  if(mc)mc.textContent=(s.model_name||'(not set)')+'\\n'+(s.model_path||'');
  if(s.model_family&&s.model_size){
    const mf=document.getElementById('mFam');
    if(mf){for(let o of mf.options)if(o.value===s.model_family){o.selected=true;break;}}
    await loadSizes(s.model_family,s.model_size);}
  // Init HF token field on model page
  const hfEl=document.getElementById('amHf');
  const hfSt=document.getElementById('amHfStatus');
  if(hfEl&&s.hf_token){
    hfEl.dataset.raw=s.hf_token;
    hfEl.type='password';hfEl.value='\u25cf\u25cf\u25cf\u25cf\u25cf\u25cf\u25cf\u25cf';
    if(hfSt)hfSt.innerHTML='<span style="color:var(--go)">\u2713</span>';
  }
  // Init temperature chips from session
  if(Array.isArray(s.active_temps)&&s.active_temps.length>0){
    _activeTemps=s.active_temps;
    const sel=document.getElementById('iTempPreset');
    if(sel){
      let matched=false;
      for(const[k,v] of Object.entries(_TEMP_PRESETS)){
        if(JSON.stringify(v)===JSON.stringify(_activeTemps)){sel.value=k;matched=true;break;}
      }
      if(!matched)sel.value='custom';
    }
  }
  buildTempChips();
  }catch(e){}
}
async function saveSt(){
  const tempVal=document.getElementById('bcTempSel').value;
  const isAuto=(tempVal==='auto');
  const isAll=(tempVal==='all');
  const b={trials:Number(document.getElementById('itr').value||100),
    seed:Number(document.getElementById('ise').value||42)};
  // Auto mode: fetch first incomplete round and set grid to that temperature.
  if(isAuto){
    try{
      const ts=await(await fetch('/temp_status')).json();
      const fi=ts.first_incomplete;
      if(fi!=null){
        b.temperature=fi;
        const _tI=document.getElementById('tbTemp');
        if(_tI)_tI.textContent='T=Auto (next: '+fi.toFixed(1)+')';
      } else {
        // All complete
        const _tI=document.getElementById('tbTemp');
        if(_tI)_tI.textContent='T=Auto (all done)';
      }
    }catch(e){
      // Fallback: don't touch temperature
      const _tI=document.getElementById('tbTemp');
      if(_tI)_tI.textContent='T=Auto';
    }
  } else if(isAll){
    // "all" = run selected runs across every temperature with data.
    // Don't write temperature to session — the all-temps handler ignores it.
    const _tI=document.getElementById('tbTemp');
    if(_tI)_tI.textContent='T=All';
  } else {
    b.temperature=Number(tempVal||0);
  }
  try{const r=await fetch('/session',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(b)});const d=await r.json();
  if(d.ok){
    const f=document.getElementById('sfb');f.classList.add('sh');setTimeout(()=>f.classList.remove('sh'),2000);
    if(!isAuto&&!isAll){
      const _tI=document.getElementById('tbTemp');
      if(_tI)_tI.textContent='T='+(Number(tempVal)||0).toFixed(1);
    }
    // Refresh grid to show the correct temperature round
    setTimeout(doScan,300);
  }
  }catch(e){toast('Save failed: '+e,'er');}
}
async function saveAdv(){
  const qt=document.getElementById('iqt');
  const _hfEl=document.getElementById('amHf');
  const hf=_hfEl&&_hfEl.dataset.raw?_hfEl.dataset.raw:'';
  const b={quantization:qt?qt.value:'4bit',clear_cache:_CC};if(hf)b.hf_token=hf;
  try{const r=await fetch('/session',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(b)});const d=await r.json();if(d.ok)toast('Advanced settings saved','ok');}
  catch(e){toast('Save failed: '+e,'er');}
}
async function loadModelsData(){
  try{const r=await fetch('/models_data');_modelsData=await r.json();
  const mf=document.getElementById('mFam');if(!mf||!_modelsData.families)return;
  mf.innerHTML='';
  for(const[k,v] of Object.entries(_modelsData.families||{})){
    const o=document.createElement('option');o.value=k;o.textContent=v;mf.appendChild(o);}
  }catch(e){}
}
async function loadSizes(fam,selSz){
  const ms=document.getElementById('mSz');if(!ms)return;ms.innerHTML='';
  const szMap=(_modelsData.sizes||{})[fam]||{};
  for(const[sz,m] of Object.entries(szMap)){
    const o=document.createElement('option');o.value=sz;
    o.textContent=sz.toUpperCase()+' \u2014 '+m.name;
    o.dataset.path=m.path;o.dataset.variant=m.variant;o.dataset.name=m.name;ms.appendChild(o);}
  if(selSz){for(let o of ms.options)if(o.value===selSz){o.selected=true;break;}}
}
function onFamChange(){loadSizes(document.getElementById('mFam').value,'');}
function onSzChange(){const ms=document.getElementById('mSz');
  const sel=ms.options[ms.selectedIndex];
  if(sel&&sel.dataset.path){document.getElementById('mHfp').value='';
    document.getElementById('mCur').textContent=sel.textContent+'\\n'+sel.dataset.path;}}
async function saveModel(){
  const hfp=document.getElementById('mHfp').value.trim();let body={};
  if(hfp){body={model_path:hfp,model_name:hfp.split('/').pop(),
    model_family:'other',model_size:'other',model_variant:'unknown'};}
  else{const fam=document.getElementById('mFam').value;
    const ms=document.getElementById('mSz');const sel=ms.options[ms.selectedIndex];
    if(!sel||!sel.dataset.path){toast('Pick a model first','wa');return;}
    body={model_path:sel.dataset.path,
      model_name:sel.dataset.name||sel.textContent.split('\u2014').slice(1).join('\u2014').trim(),
      model_family:fam,model_size:sel.value,model_variant:sel.dataset.variant||'unknown'};}
  try{const r=await fetch('/session',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)});const d=await r.json();
  if(d.ok){const mc=document.getElementById('mCur');
    if(mc)mc.textContent=(d.session.model_name||'')+(d.session.model_path?'\\n'+d.session.model_path:'');
    const f=document.getElementById('mfb');f.classList.add('sh');setTimeout(()=>f.classList.remove('sh'),2000);
    toast('Model saved','ok');setTimeout(doScan,500);}
  else{toast(d.error||'Save failed','er');}
  }catch(e){toast('Save failed: '+e,'er');}
}
async function doStats(){
  
  try{const r=await fetch('/stats',{method:'POST'});const d=await r.json();
  if(d.ok){
    toast('Stats launched - see console','ok');
    addL({kind:'section',text:'\u25B6 Stats running\u2026'});
    updateLogRate(true); // 2s poll while stats runs
    let _pa=0;const _pd=[30000,60000,120000,180000,240000,300000];
    function _pp(){if(_pa>=_pd.length){updateLogRate(false);return;}
      const _d=_pd[_pa++];
      setTimeout(async()=>{try{const pr=await fetch('/outcomes');const pd=await pr.json();
        if(pd&&Object.keys(pd).length>0){_hypLoaded=false;loadHyp();updateLogRate(false);toast('Stats complete','ok');return;}}catch(e){}
        _pp();},_d);}
    _pp();
  } else{toast(d.error||'Stats failed','er');}
  }catch(e){toast('Stats failed: '+e,'er');}
}

function _showStatsDialog(){
  return new Promise(resolve=>{
    const bg=document.createElement('div');
    bg.style.cssText='position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:9999;display:flex;align-items:center;justify-content:center';
    const box=document.createElement('div');
    box.style.cssText='background:var(--bg2);border:1px solid var(--b2);border-radius:8px;padding:20px 24px;max-width:380px;width:90%';
    box.innerHTML='<div style="color:var(--t);font-size:13px;font-weight:600;margin-bottom:12px">All Stats + Report</div>'+
      '<div style="color:var(--t2);font-size:11px;margin-bottom:16px;line-height:1.5">'+
      'Run analysis (25,27,32,33,34,45,46) at each temperature, then cross-temp synthesis (47), pooled decomposition (40), and generate the report with figures.</div>'+
      '<div style="display:flex;flex-direction:column;gap:8px">'+
      '<button class="btn" style="padding:6px 12px;font-size:11px" data-mode="skip">Continue (skip existing)</button>'+
      '<button class="btn" style="padding:6px 12px;font-size:11px" data-mode="overwrite">Overwrite all</button>'+
      '<button class="btn" style="padding:6px 12px;font-size:11px" data-mode="backup">Backup then overwrite</button>'+
      '<button class="btn" style="padding:6px 12px;font-size:11px;opacity:.6" data-mode="">Cancel</button>'+
      '</div>';
    bg.appendChild(box);document.body.appendChild(bg);
    box.querySelectorAll('button').forEach(b=>{
      b.addEventListener('click',()=>{bg.remove();resolve(b.dataset.mode||null);});
    });
    bg.addEventListener('click',e=>{if(e.target===bg){bg.remove();resolve(null);}});
  });
}

// ── Tab switch ────────────────────────────────────────────────────────────────
function sw(n){
  ['C','H','N'].forEach(k=>{
    const tab=document.getElementById('t'+k);
    const pane=document.getElementById('p'+k);
    if(!tab||!pane)return;
    const active=k===n;
    tab.classList.toggle('on',active);
    tab.setAttribute('aria-selected',active?'true':'false');
    pane.classList.toggle('on',active);
  });
  if(n==='H'&&!_hypLoaded){loadHyp(true);}
  if(n==='N')loadSt();
}

// ── Launch / Abort / Pause ────────────────────────────────────────────────────
let _pruns='';
// v0.75.2.2: enable/disable run buttons during grid load
function _setRunBtnEnabled(en){
  // Main run buttons
  ['lBtn','lBtn2'].forEach(id=>{
    const e=document.getElementById(id);
    if(e){e.disabled=!en;e.style.opacity=en?'1':'0.4';e.style.pointerEvents=en?'auto':'none';}
  });
  // v0.75.2.2: also disable rdp popup run buttons and preset launch buttons
  document.querySelectorAll('#rdpActs button, .gc-btn.go, .set-action').forEach(b=>{
    b.disabled=!en;b.style.opacity=en?'1':'0.4';b.style.pointerEvents=en?'auto':'none';
  });
}
async function doLaunchOrQueue(){
  if(_gridBusy){toast('Grid still loading — wait','wa');return;}
  if(_prevRunning){ await doQueue(); } else { await doLaunch(); }
}
// v0.66.3.0: _getSelectedRuns reads from selR (the authoritative grid selection
// state), not ri.value (a display field that gets overwritten by loadSt on tab
// switch). Fixes bug where selecting Data preset then switching to Settings tab
// caused loadSt to overwrite ri.value with the stale session runs string,
// while selR (and the visible grid highlights) retained the correct selection.
function _getSelectedRuns(){
  if(selR.size>0) return [...selR].sort((a,b)=>a-b).join(',');
  const ri=document.getElementById('ri');
  return (ri?ri.value.trim():'')||'';
}
async function doQueue(){
  let runs=_getSelectedRuns();
  if(!runs){toast('Select runs to queue first','wa');return;}
  // Preserve all-temps prefix for analysis runs queued from layer 2
  if(_navLayer===2) runs='all-temps:'+runs;
  try{
    const r=await fetch('/queue',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({runs})});
    const d=await r.json();
    if(d.ok){toast('Queued: '+runs,'ok');}
    else{toast('Queue failed','er');}
  }catch(e){toast('Queue failed: '+e,'er');}
}
async function doClearQueue(){
  try{
    await fetch('/queue',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({runs:''})});
    toast('Queue cleared','ok');
  }catch(e){}
}
const _ANALYSIS_RUN_SET=new Set([41,42,43,44,45,46,47,48,49,50,51]);
function _parseRunNums(s){
  const nums=new Set();
  for(const p of s.split(',')){
    const t=p.trim();if(!t)continue;
    if(t.includes('-')){
      const[a,b]=t.split('-').map(Number);
      if(!isNaN(a)&&!isNaN(b))for(let i=a;i<=b;i++)nums.add(i);
    }else{const n=Number(t);if(!isNaN(n))nums.add(n);}
  }return nums;
}
async function doLaunch(force){
  if(_prevRunning){toast('Already running - use Queue button to stage runs','wa');return;}
  const runs=_getSelectedRuns();
  if(!runs&&!force){toast('Select runs first','wa');return;}
  // v0.75.2.2: Run 0056 intercept — show model selection popup whenever 55 is included
  const _parsed55=_parseRunNums(runs||'');
  console.log('[R55] doLaunch runs=',runs,'parsed=',_parsed55,'has55=',_parsed55.has(55));
  if(_parsed55.has(55)){
    // Store the full selection so other runs (50,51,52) launch alongside 55
    window._r55OtherRuns=[..._parsed55].filter(r=>r!==55).sort((a,b)=>a-b).join(',');
    console.log('[R55] Opening popup, otherRuns=',window._r55OtherRuns);
    openR55Pop();return;
  }
  let effectiveRuns=runs;
  // Layer 2 = all-temps, Layer 3 = per-temp
  if(_navLayer===2){
    if(!runs){toast('Select runs first','wa');return;}
    // v0.76.0.4: if user selected collection runs, route to auto-temp
    // so missing temperature directories get collected. all-temps-runs only
    // analyzes existing temps with data — silent no-op on fresh FP16 model.
    // v0.79.4.0: renumber. Collection runs are contiguous 1-40 in new numbering.
    const _sel2=_parseRunNums(runs);
    const _COLLECT=new Set([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
                            21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]);
    const _hasCollect=[..._sel2].some(r=>_COLLECT.has(r));
    effectiveRuns=_hasCollect ? 'auto-temp' : ('all-temps:'+runs);
  } else if(_navLayer===3){
    const _sel=_parseRunNums(runs);
  } else {
    const tempVal=document.getElementById('bcTempSel')?document.getElementById('bcTempSel').value:'';
    if(tempVal==='auto'){
      effectiveRuns='auto-temp';
    } else if(tempVal==='all'){
      effectiveRuns='all-temps:'+runs;
    }
  }
  _pruns=effectiveRuns;
  try{
    const s=await(await fetch('/session')).json();
    // Clear stale mode overrides — only popup launches should set these
    delete s.patch_modes_17;delete s.patch_modes_18;delete s.patch_modes_19;
    delete s.r3_cells;delete s.mc_conds;
    // v0.79.4.9: if the user navigated from the "All Models" synthetic
    // card, tell the server to fan out across every discovered model.
    // The server spawns start_here.py --all-models which iterates the
    // model list, dispatching effectiveRuns to each in sequence.
    const allModels=!!(_allModelsMode&&_allModelsMode.models);
    const body=Object.assign({},s,{runs:effectiveRuns,all_models:allModels});
    const r=await fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body)});
    const d=await r.json();
    if(!d.ok){
      if(r.status===409){ await doQueue(); return; }
      toast(d.error||'Launch failed','er');return;
    }
    toast(allModels?'Launched across all models':'Launched','ok');
  }catch(e){toast('Launch failed: '+e,'er');}
}
async function doPause(){
  try{const r=await fetch('/pause',{method:'POST'});const d=await r.json();
  PAU=d.paused;
  ['pBtnLbl','pBtnLbl2'].forEach(id=>{const e=document.getElementById(id);if(e)e.textContent=PAU?'Resume':'Pause';});
  }catch(e){toast('Pause failed','er');}
}
async function doAbort(){
  if(!_prevRunning){toast('No run active','wa');return;}
  try{const r=await fetch('/abort',{method:'POST'});const d=await r.json();
  if(d.ok)toast('Aborted','wa');else toast(d.error||'Abort failed','er');
  }catch(e){toast('Abort failed: '+e,'er');}
}
async function doResume(){
  try{const r=await fetch('/resume',{method:'POST'});const d=await r.json();
  if(d.ok){toast('Resuming','ok');doLaunch(true);}
  else toast(d.error||'Resume failed','er');
  }catch(e){toast('Resume failed: '+e,'er');}
}
async function checkResumeState(){
  try{const r=await fetch('/resume_state');const d=await r.json();
  ['rsBtn','rsBtn2'].forEach(id=>{const e=document.getElementById(id);if(e)e.style.display=(d.has_resume&&!_prevRunning)?'flex':'none';});
  }catch(e){}
}

// ── Sparklines ────────────────────────────────────────────────────────────────
function spk(k,v,ci,li,floor){const a=SD[k];a.push(v);if(a.length>MX)a.shift();
  const nums=a.filter(x=>typeof x==='number'&&isFinite(x));
  const avg=nums.length>0?nums.reduce((s,x)=>s+x,0)/nums.length:v;
  document.getElementById(li).textContent=typeof avg==='number'?avg.toFixed(4):v;dspk(ci,a,floor);}
function dspk(id,d,floor){const c=document.getElementById(id);if(!c)return;
  const ctx=c.getContext('2d'),W=c.offsetWidth||260,H=c.offsetHeight||60;
  c.width=W*devicePixelRatio;c.height=H*devicePixelRatio;
  ctx.scale(devicePixelRatio,devicePixelRatio);ctx.clearRect(0,0,W,H);
  if(d.length<2)return;
  const nums=d.filter(x=>typeof x==='number'&&isFinite(x));
  if(nums.length<2)return;
  let mn=Math.min(...nums),mx=Math.max(...nums);
  if(typeof floor==='number')mn=floor;
  let rng=mx-mn||0.001;
  const avg0=nums.reduce((s,x)=>s+x,0)/nums.length;
  const minRng=Math.abs(avg0)*0.1||0.01;
  if(rng<minRng){const mid=(mn+mx)/2;mn=mid-minRng/2;mx=mid+minRng/2;rng=minRng;}
  mn-=rng*0.1;mx+=rng*0.1;const fr=mx-mn;
  const pad=4,lm=36;
  const X=i=>(i/(MX-1))*(W-lm)+lm,Y=v=>H-pad-((v-mn)/fr)*(H-pad*2);
  // Avg line
  const avg=nums.reduce((s,x)=>s+x,0)/nums.length;
  ctx.setLineDash([2,3]);ctx.strokeStyle='rgba(255,255,255,.1)';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(lm,Y(avg));ctx.lineTo(W,Y(avg));ctx.stroke();ctx.setLineDash([]);
  // Y labels — adaptive precision based on data range
  ctx.font='7px system-ui';ctx.fillStyle='rgba(255,255,255,.2)';ctx.textAlign='right';
  const _yFmt=v=>{const a=Math.abs(fr);
    if(a>=1)return v.toFixed(1);if(a>=0.1)return v.toFixed(2);if(a>=0.01)return v.toFixed(3);
    if(a>=0.001)return v.toFixed(4);if(a>=0.0001)return v.toFixed(5);return v.toExponential(1);};
  ctx.fillText(_yFmt(mx),lm-3,pad+6);ctx.fillText(_yFmt(mn),lm-3,H-pad+1);
  // Fill
  const g=ctx.createLinearGradient(0,0,0,H);
  g.addColorStop(0,'rgba(0,212,170,.12)');g.addColorStop(1,'rgba(0,212,170,0)');
  ctx.beginPath();ctx.moveTo(X(0),H);let ok=false;
  d.forEach((v,i)=>{if(typeof v==='number'&&isFinite(v)){ctx.lineTo(X(i),Y(v));ok=true;}});
  if(ok){ctx.lineTo(X(d.length-1),H);ctx.closePath();ctx.fillStyle=g;ctx.fill();}
  // Line
  ctx.beginPath();let f=true;d.forEach((v,i)=>{if(typeof v!=='number'||!isFinite(v))return;
    if(f){ctx.moveTo(X(i),Y(v));f=false;}else ctx.lineTo(X(i),Y(v));});
  ctx.strokeStyle='rgba(0,212,170,.8)';ctx.lineWidth=1.5;ctx.stroke();
  const lv=d[d.length-1];if(typeof lv==='number'&&isFinite(lv)){
    ctx.beginPath();ctx.arc(X(d.length-1),Y(lv),2.5,0,Math.PI*2);ctx.fillStyle='#00d4aa';ctx.fill();}}

// ── Run detail popup ──────────────────────────────────────────────────────────
let _rdpRun=null,_rdpPollTimer=null;
function _startRdpPoll(){_rdpPollTimer=setInterval(()=>{
  if(_rdpRun==null)return;
  if(_navLayer===2)return; // all-temps popup is static — don't overwrite
  _refreshRdp(_rdpRun);
},8000);}
function _stopRdpPoll(){clearInterval(_rdpPollTimer);_rdpPollTimer=null;}
function closeRdp(){
  const pop=document.getElementById('rdpop');
  if(pop)pop.classList.remove('on');
  _rdpRun=null;_stopRdpPoll();
}
function _btn(label,cls,done,onclick,title){
  const dim=done?' dim':'';
  return '<button class="rdp-rbtn '+cls+dim+'" onclick="'+onclick+'" title="'+escH(title)+'">'+escH(label)+'</button>';
}
async function _refreshRdp(n){
  const body=document.getElementById('rdpBody');if(!body)return;
  try{
    const r=await fetch('/run_detail?run='+n);
    const d=await r.json();if(d.error){body.innerHTML='<span style="color:var(--er)">'+escH(d.error)+'</span>';return;}
    const st=d.status||'missing';
    const acts=document.getElementById('rdpActs');if(acts)acts.innerHTML='';
    let html='';
    const scol=st==='done'?'ok':st==='partial'?'wa':st==='needs_et'?'ye':st==='et_partial'?'wa':'er';
    const stlbl=st==='needs_et'?'needs E\\u209c pass':st==='et_partial'?'E\\u209c incomplete':
      st==='corrupt'?'corrupt \\u2014 check data':st;
    html+='<div class="rdp-row"><span class="rdp-lbl">Status</span>'+
      '<span class="rdp-val '+scol+'">'+escH(stlbl)+'</span></div>';
    const _pdescr=descs[String(n)]||'';
    if(_pdescr){html+='<div style="font-size:9px;color:var(--t2);margin:3px 0 2px;line-height:1.4">'+escH(_pdescr)+'</div>';}
    const _rhyps=Object.entries(_hypDeps||{}).filter(([k,v])=>(v.data_runs||[]).includes(n)).map(([k])=>k);
    if(_rhyps.length){html+='<div style="font-size:8px;color:var(--t3);margin-bottom:4px">Tests: '+escH(_rhyps.join(', '))+'</div>';}
    if(d.run1){
      const r19=d.run1;
      const nt=r19.n_trials||100;
      const passes=[
        {key:'abliterated', label:'abliterated', n:r19.abliterated||0},
        {key:'base',        label:'base',        n:r19.base||0},
        {key:'instruct',    label:'instruct',    n:r19.instruct||0},
        {key:'pass4',       label:'vectors',     n:r19.pass4?1:0, total:1},
      ];
      html+='<div class="rdp-row"><span class="rdp-lbl">Expected</span>'+
        '<span class="rdp-val">3 passes \u00d7 '+nt+' trials + vectors</span></div>';
      html+='<div style="margin-top:4px">';
      passes.forEach(p=>{
        const tot=p.total||nt;
        const done2=(p.n>=tot);
        const pct=Math.min(100,Math.round(p.n/tot*100));
        const cls2=done2?'ok':p.n>0?'wa':'er';
        html+='<div style="margin-bottom:4px">';
        html+='<div style="display:flex;align-items:center;gap:6px">'+
          '<span style="min-width:60px;color:var(--t2);font-size:8px;flex-shrink:0">'+escH(p.label)+'</span>'+
          '<div style="flex:1;height:5px;background:var(--b1);border-radius:3px">'+
            '<div style="height:5px;border-radius:3px;background:'+(done2?'var(--go)':p.n>0?'var(--wa)':'var(--b2)')+
          ';width:'+pct+'%"></div></div>'+
          '<span class="rdp-val '+cls2+'" style="min-width:44px;font-size:8px;flex-shrink:0">'+p.n+'/'+tot+'</span>'+
        '</div></div>';
      });
      html+='</div>';
      if(acts){
        const b=document.createElement('button');
        b.className='btn go';b.style.cssText='font-size:9px;padding:3px 9px';
        b.textContent='\\u25b6 Run';b.onclick=()=>rdpLaunchStd(1);
        acts.appendChild(b);
      }
    } else if(d.run3_grid){
      // Run 0003: 4×5 temperature grid
      const temps=d.run3_temps||[];
      const conds=d.run3_conds||[];
      const perCell=d.n_trials/Math.max(1,d.run3_cells_total)||100;
      html+='<div class="rdp-row"><span class="rdp-lbl">Grid</span>'+
        '<span class="rdp-val">'+conds.length+' cond \u00d7 '+temps.length+' temp \u00d7 '+Math.round(perCell)+' trials</span></div>';
      // Header row
      html+='<div style="margin-top:6px;display:grid;grid-template-columns:80px repeat('+temps.length+',1fr);gap:2px;font-size:8px">';
      html+='<div></div>';
      temps.forEach(t=>{ html+='<div style="text-align:center;color:var(--t3)">T='+t+'</div>'; });
      // Data rows
      d.run3_grid.forEach(row=>{
        html+='<div style="color:var(--t2);display:flex;align-items:center;font-size:8px">'+escH(row.cond)+'</div>';
        row.cells.forEach(cell=>{
          const col=cell.done?'var(--go)':cell.n>0?'var(--wa)':'var(--b2)';
          const txt=cell.done?'\u2713':cell.n;
          html+='<div style="text-align:center;background:'+col+';border-radius:3px;padding:2px 0;color:'+(cell.done?'var(--bg)':'var(--t1)')+'">'+txt+'</div>';
        });
      });
      html+='</div>';
      if(acts){const b=document.createElement('button');
        b.className='btn go';b.style.cssText='font-size:9px;padding:3px 9px';
        b.textContent='\u25b6 Run';b.onclick=()=>rdpLaunchStd(3);acts.appendChild(b);}
    } else if(d.mc_conditions){
      const perCond=d.mc_per_cond||100;
      const isEt=!!d.is_et_run;
      const mainDone=(st==='done'||st==='needs_et');
      const etDone=(st==='done');
      html+='<div class="rdp-row"><span class="rdp-lbl">Expected</span>'+
        '<span class="rdp-val">'+d.mc_conditions.length+' cond \\u00d7 '+perCond+' trials</span></div>';
      html+='<div style="margin-top:4px">';
      d.mc_conditions.forEach(c=>{
        const barW=c.pct;
        const cls2=c.done?'ok':c.n>0?'wa':'er';
        html+='<div style="margin-bottom:'+(isEt?'6':'3')+'px">';
        html+='<div style="display:flex;align-items:center;gap:6px">'+
          '<span style="min-width:46px;color:var(--t2);font-size:8px;flex-shrink:0">'+escH(c.condition)+'</span>'+
          '<div style="flex:1;height:5px;background:var(--b1);border-radius:3px">'+
            '<div style="height:5px;border-radius:3px;background:'+(c.done?'var(--go)':c.n>0?'var(--wa)':'var(--b2)')+
          ';width:'+barW+'%"></div></div>'+
          '<span class="rdp-val '+cls2+'" style="min-width:44px;font-size:8px;flex-shrink:0">'+c.n+'/'+perCond+'</span>';
        if(isEt){
          html+=_btn('A','go',mainDone,"rdpLaunchET("+n+",'a')",'Run abliterated generation');
          html+=_btn('E','ye',etDone,  "rdpLaunchET("+n+",'e')",'Base E\\u209c pass only');
          html+=_btn('B','wa',etDone,  "rdpLaunchETConfirm("+n+",'b')",'Generation + E\\u209c pass');
        }
        html+='</div></div>';
      });
      html+='</div>';
      if(acts){
        const b=document.createElement('button');
        if(isEt&&!etDone){
          b.className='btn ye';b.style.cssText='font-size:9px;padding:3px 9px';
          b.textContent='E-test \\u25b6';b.onclick=()=>rdpLaunchET(n,'e');
        } else {
          b.className='btn go';b.style.cssText='font-size:9px;padding:3px 9px';
          b.textContent='Run \\u25b6';b.onclick=()=>rdpLaunchStd(n);
        }
        acts.appendChild(b);
      }
    } else if(d.is_et_run){
      const comp=d.n_complete||0,tot=d.n_trials||100;
      const mainDone=(st==='done'||st==='needs_et');
      const etDone=(st==='done');
      html+='<div class="rdp-row"><span class="rdp-lbl">Complete</span>'+
        '<div class="rdp-rval">'+
        '<span class="rdp-val '+(comp>=tot?'ok':'wa')+'">'+comp+' / '+tot+'</span>'+
        _btn('A','go',mainDone,"rdpLaunchET("+n+",'a')",'Abliterated generation only')+
        _btn('E','ye',etDone,  "rdpLaunchET("+n+",'e')",'Base E\\u209c pass only')+
        _btn('B','wa',etDone,  "rdpLaunchETConfirm("+n+",'b')",'Generation + E\\u209c pass')+
        '</div></div>';
    } else if(d.is_patching_run){
      const nNone=d.n_complete||0,nt=d.n_trials||100,pt=d.patch_turns||13;
      // v0.79.4.0: isR17=activation (was R21), isR19=random (was R53), else isR18=layer iso (was R42)
      const isR17=(d.patch_col==='patch_mode');
      const isR19=(n===19);
      const patchSessKey=isR17?'patch_modes_17':(isR19?'patch_modes_19':'patch_modes_18');
      html+='<div class="rdp-row"><span class="rdp-lbl">None-mode trials</span>'+
        '<div class="rdp-rval">'+
        '<span class="rdp-val '+(nNone>=nt?'ok':'wa')+'">'+nNone+' / '+nt+
        ' <span style="font-weight:400;color:var(--t3)">('+pt+' turns each)</span></span>'+
        '</div></div>';
      html+='<div style="color:var(--t3);font-size:9px;margin:3px 0 5px 0">'+
        'Click to select \u00b7 Ctrl/Shift+click to multi-select \u00b7 then Run selected</div>';
      const modeInfo=isR17?
        {modes:['none','partial','full','random'],tips:{none:'Unpatched baseline.',partial:'Early layers grafted.',full:'All active layers grafted.',random:'Random noise baseline.'}}:
        {modes:(d.patch_modes||[]).map(m=>m.mode),tips:{none:'Unpatched baseline.',full:'All layers random noise.'}};
      const modeCounts=(d.mode_counts||{});
      // Expected rows per mode = n_trials * turns_per_trial
      const perModeExp=nt*pt;
      html+='<div id="patchModeRows_'+n+'">';
      modeInfo.modes.forEach(m=>{
        const cnt=modeCounts[m]||0;
        const trialEst=pt>0?Math.round(cnt/pt):0;
        const done2=(cnt>=perModeExp);
        const sel=(window._selPatchModes||{})[n]&&(window._selPatchModes[n]||new Set()).has(m);
        html+='<div class="rdp-row patch-mode-row'+(sel?' patch-sel':'')+'" '+
          'data-run="'+n+'" data-mode="'+escH(m)+'" onclick="patchModeClick(event,this)" style="cursor:pointer">'+
          '<span class="rdp-lbl" style="font-size:9px;min-width:52px">'+escH(m)+'</span>'+
          '<div class="rdp-rval">'+
          (done2
            ? '<span class="rdp-val ok" style="font-size:11px" title="'+nt+'/'+nt+' trials complete">\u2713</span>'
            : '<span class="rdp-val wa" style="font-size:9px" title="'+cnt+' rows / '+perModeExp+' expected">'+trialEst+'/'+nt+'</span>'
          )+
          '</div></div>';
      });
      html+='</div>';
      if(acts){
        const bs=document.createElement('button');
        bs.className='btn go';bs.style.cssText='font-size:9px;padding:3px 9px';
        bs.textContent='\\u25b6 Run selected';
        bs.onclick=()=>rdpLaunchPatch(n,patchSessKey);
        acts.appendChild(bs);
        const ba=document.createElement('button');
        ba.className='btn bl';ba.style.cssText='font-size:9px;padding:3px 9px';
        ba.textContent='Run all';ba.onclick=()=>rdpLaunchStd(n);
        acts.appendChild(ba);
      }
    } else if(d.status==='analysis'){
      const done2=(st==='done');
      html+='<div class="rdp-row"><span class="rdp-lbl">Type</span>'+
        '<span class="rdp-val">No-GPU analysis</span></div>';
      if(acts){const b=document.createElement('button');
        b.className='btn go';b.style.cssText='font-size:9px;padding:3px 9px';
        b.textContent='Run \\u25b6';b.onclick=()=>rdpLaunchStd(n);acts.appendChild(b);}
    } else {
      // Standard run
      const comp=d.n_complete||0,tot=d.n_trials||100;
      const et=d.expected_turns!=null?' ('+d.expected_turns+' turns/trial)':'';
      html+='<div class="rdp-row"><span class="rdp-lbl">Complete</span>'+
          '<div class="rdp-rval">'+
          '<span class="rdp-val '+(comp>=tot?'ok':'wa')+'">'+comp+' / '+tot+escH(et)+'</span>'+
          _btn('\\u25B6','go',(comp>=tot),'rdpLaunchStd('+n+')','Run')+
          '</div></div>';
      if(d.has_extra){
        html+='<div style="background:rgba(245,208,0,.12);border:1px solid var(--ye);'+
          'color:var(--ye);font-size:9px;padding:5px 8px;border-radius:3px;margin-top:4px">'+
          '\\u26a0 Extra trials detected \\u2014 run dedup_all_runs.py before analysis.</div>';
      }
      // v46.0.0: replace trial list with summary line
      if(!d.mc_conditions&&!d.is_patching_run){
        if(comp===0){
          html+='<div style="font-size:9px;color:var(--t3);margin-top:6px">Collection not started</div>';
        } else if(comp<tot){
          const maxTrial=d.partial&&d.partial.length?Math.max(...d.partial.map(p=>p.trial)):comp-1;
          html+='<div style="font-size:9px;color:var(--wa);margin-top:6px">Collection complete to trial '+maxTrial+'</div>';
        }
      }
      // Summary stats (mean ± std)
      if(d.summary&&Object.keys(d.summary).length){
        const _sf=(v,k)=>{
          if(v===undefined||v===null)return'—';
          const a=Math.abs(v);
          if(k==='disrupt'){if(a<0.001)return v.toExponential(1);return v.toFixed(4);}
          return v.toFixed(4);
        };
        html+='<div style="margin-top:6px;border-top:1px solid var(--b1);padding-top:5px">';
        const labels={sim_index:'Similarity',sim:'Similarity',disrupt:'Disruption'};
        ['sim_index','sim','disrupt'].forEach(k=>{
          const s=d.summary[k];
          if(!s)return;
          html+='<div style="display:flex;justify-content:space-between;font-size:9px;color:var(--t2);padding:1px 0">'+
            '<span>'+labels[k]+'</span><span>'+_sf(s.mean,k)+' \u00b1 '+_sf(s.std,k)+'</span></div>';
        });
        html+='</div>';
      }
    }
    body.innerHTML=html;
  }catch(e){body.innerHTML='<span style="color:var(--er)">'+escH(String(e))+'</span>';}
}
async function showRdp(n,cell){
  try{
  // Layer 2 (all-temps): show temperature breakdown instead of run detail
  // Pooled and cross-model runs don't have per-temp data — show standard detail
  const _POOLED_RUNS=new Set([50,51,52,53,54,55,56]);
  if(_navLayer===2&&_tempGrid[String(n)]&&!_POOLED_RUNS.has(n)){
    const tg=_tempGrid[String(n)];
    const _pdesc=descs[String(n)]||'';
    const pop=document.getElementById('rdpop');
    const title=document.getElementById('rdpTitle');
    const body=document.getElementById('rdpBody');
    const acts=document.getElementById('rdpActs');
    if(!pop||!body)return;
    _rdpRun=n;
    title.textContent='Run '+n+(_pdesc?' - '+_pdesc:'');
    if(acts)acts.innerHTML='';
    let html='<div style="margin-bottom:8px;font-size:10px;color:var(--t2)">Temperature breakdown:</div>';
    const temps=Object.keys(tg).map(Number).sort((a,b)=>a-b);
    temps.forEach(t=>{
      const ts=t.toFixed(1);
      const st=tg[ts]||'missing';
      const col=st==='done'?'var(--go)':st==='missing'?'var(--b2)':'var(--wa)';
      const label=st==='done'?'done':st==='missing'?'not started':st;
      html+='<div style="display:flex;align-items:center;gap:8px;padding:4px 0;cursor:pointer;border-radius:3px" '+
        'onclick="closeRdp();navTo(3,'+t+')" title="Click to view T='+t.toFixed(1)+' detail">'+
        '<div style="width:10px;height:10px;border-radius:2px;background:'+col+'"></div>'+
        '<span style="font-size:11px;font-weight:500;min-width:40px">T='+t.toFixed(1)+'</span>'+
        '<span style="font-size:9px;color:var(--t2)">'+escH(label)+'</span>'+
        '</div>';
    });
    body.innerHTML=html;
    // Run all-temps button
    if(acts){
      const b=document.createElement('button');b.className='btn go';
      b.style.cssText='font-size:9px;padding:3px 9px';
      b.textContent='Run all temps';
      b.onclick=()=>{closeRdp();selP(String(n),null);doLaunchOrQueue();};
      acts.appendChild(b);
    }
    pop.classList.add('on');
    const pw=300;
    const rect=cell.getBoundingClientRect();
    const lp=document.getElementById('left-col');
    const lpRight=lp?lp.getBoundingClientRect().right+12:rect.right+8;
    let left=Math.min(window.innerWidth-pw-8, Math.max(lpRight, rect.right+8));
    let top=Math.min(window.innerHeight-300, Math.max(8, rect.top));
    pop.style.left=left+'px';pop.style.top=top+'px';
    pop.style.width=pw+'px';
    return;
  }
  // If multiple runs selected and user clicks a selected cell, show all selected
  const runsToShow=(selR.size>1&&selR.has(n))?[...selR].sort((a,b)=>a-b):[n];
  _rdpRun=runsToShow.length===1?runsToShow[0]:runsToShow;
  const pop=document.getElementById('rdpop');
  const title=document.getElementById('rdpTitle');
  const body=document.getElementById('rdpBody');
  if(!pop||!body)return;
  if(runsToShow.length>1){
    title.textContent=runsToShow.length+' runs selected: '+runsToShow.join(', ');
  } else {
    const _pdesc=descs[String(n)]||'';
    title.textContent='Run '+n+(_pdesc?' \u2014 '+_pdesc:'');
  }
  body.innerHTML='<span style="color:var(--t3)">Loading\u2026</span>';
  pop.classList.add('on');
  const pw=340,ph=runsToShow.length>1?Math.min(520,160+runsToShow.length*80):320;
  const rect=cell.getBoundingClientRect();
  const lp=document.getElementById('left-col');
  const lpRight=lp?lp.getBoundingClientRect().right+12:rect.right+8;
  let left=Math.min(window.innerWidth-pw-8, Math.max(lpRight, rect.right+8));
  let top=Math.min(window.innerHeight-ph-8, Math.max(8, rect.top));
  pop.style.left=left+'px';pop.style.top=top+'px';
  pop.style.width=pw+'px';
  if(runsToShow.length>1){
    await _refreshRdpMulti(runsToShow);
  } else {
    await _refreshRdp(n);
  }
  if(_prevRunning)_startRdpPoll();
  }catch(e){console.error('showRdp error:',e);const b=document.getElementById('rdpBody');if(b)b.innerHTML='<span style="color:var(--er)">'+escH(String(e))+'</span>';}
}
async function _refreshRdpMulti(runs){
  const body=document.getElementById('rdpBody');if(!body)return;
  const acts=document.getElementById('rdpActs');if(acts)acts.innerHTML='';
  let html='';
  for(const n of runs){
    try{
      const r=await fetch('/run_detail?run='+n);
      const d=await r.json();
      if(d.error)continue;
      const st=d.status||'missing';
      const scol=st==='done'?'ok':st==='partial'?'wa':st==='needs_et'?'ye':'er';
      const stlbl=st==='needs_et'?'needs E\u209c':st;
      const comp=d.n_complete||0,tot=d.n_trials||100;
      const _pdescr=descs[String(n)]||'Run '+n;
      html+='<div style="border:1px solid var(--b2);border-radius:4px;padding:6px 8px;margin-bottom:6px">';
      html+='<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">';
      html+='<span style="font-size:10px;color:var(--t1);font-weight:600">Run '+n+'</span>';
      html+='<span class="rdp-val '+scol+'" style="font-size:9px">'+escH(stlbl)+'</span>';
      html+='</div>';
      html+='<div style="font-size:9px;color:var(--t2);margin-bottom:4px">'+escH(_pdescr)+'</div>';
      if(!d.is_patching_run&&!d.mc_conditions&&!d.run3_grid&&!d.run1){
        const pct=Math.min(100,Math.round(comp/Math.max(tot,1)*100));
        html+='<div style="display:flex;align-items:center;gap:6px">';
        html+='<div style="flex:1;height:4px;background:var(--b1);border-radius:2px">';
        html+='<div style="height:4px;border-radius:2px;background:'+(comp>=tot?'var(--go)':comp>0?'var(--wa)':'var(--b2)')+';width:'+pct+'%"></div></div>';
        html+='<span style="font-size:9px;color:var(--t2);min-width:44px">'+comp+'/'+tot+'</span>';
        html+='</div>';
      } else if(d.is_patching_run){
        const modeCounts=d.mode_counts||{};
        const pt=d.patch_turns||13;
        const modes=(d.patch_modes||[]).map(m=>m.mode);
        modes.forEach(m=>{
          const cnt=modeCounts[m]||0;
          const trialEst=pt>0?Math.round(cnt/pt):0;
          const done2=(trialEst>=(tot||100));
          html+='<div style="display:flex;gap:6px;align-items:center;font-size:8px;margin-top:2px">';
          html+='<span style="min-width:40px;color:var(--t3)">'+escH(m)+'</span>';
          html+=done2
            ? '<span class="ok" style="font-size:10px">\u2713</span>'
            : '<span class="wa">'+trialEst+'/'+(tot||100)+'</span>';
          html+='</div>';
        });
      }
      html+='</div>';
    }catch(e){}
  }
  body.innerHTML=html||'<span style="color:var(--t3)">No data</span>';
  if(acts){
    const b=document.createElement('button');
    b.className='btn go';b.style.cssText='font-size:9px;padding:3px 9px';
    b.textContent='\u25b6 Run all';
    b.onclick=()=>{closeRdp();selP(runs.join(','),null);doLaunchOrQueue();};
    acts.appendChild(b);
  }
}
document.addEventListener('click',e=>{
  const pop=document.getElementById('rdpop');
  if(!pop||!pop.classList.contains('on'))return;
  if(!pop.contains(e.target)&&!e.target.classList.contains('rc'))closeRdp();
});
// Drag — attach to both rdpop and etpop headers
(function(){
  function makeDraggable(popId){
    const pop=document.getElementById(popId);
    if(!pop)return;
    const hd=pop.querySelector('.rdp-hd');
    if(!hd)return;
    let ox=0,oy=0,mx=0,my=0;
    hd.addEventListener('mousedown',e=>{
      if(e.target.classList.contains('rdp-close'))return;
      e.preventDefault();
      ox=pop.offsetLeft; oy=pop.offsetTop;
      mx=e.clientX; my=e.clientY;
      function onMove(e){
        const dx=e.clientX-mx, dy=e.clientY-my;
        const nw=window.innerWidth, nh=window.innerHeight;
        const pw=pop.offsetWidth, ph=pop.offsetHeight;
        pop.style.left=Math.max(0,Math.min(nw-pw,ox+dx))+'px';
        pop.style.top=Math.max(0,Math.min(nh-ph,oy+dy))+'px';
      }
      function onUp(){
        document.removeEventListener('mousemove',onMove);
        document.removeEventListener('mouseup',onUp);
      }
      document.addEventListener('mousemove',onMove);
      document.addEventListener('mouseup',onUp);
    });
  }
  makeDraggable('rdpop');
  makeDraggable('etpop');
})();
async function showMergedRdp(cell){
  const runs=[...selR].sort((a,b)=>a-b);
  _rdpRun=runs;
  const pop=document.getElementById('rdpop');
  const title=document.getElementById('rdpTitle');
  const body=document.getElementById('rdpBody');
  if(!pop||!body)return;
  title.textContent=runs.length+' runs selected: '+runs.join(', ');
  body.innerHTML='<span style="color:var(--t3)">Loading\u2026</span>';
  pop.classList.add('on');
  const pw=340,ph=Math.min(540,180+runs.length*72);
  const rect=cell.getBoundingClientRect();
  const lp=document.getElementById('left-col');
  const lpRight=lp?lp.getBoundingClientRect().right+12:rect.right+8;
  let left=Math.min(window.innerWidth-pw-8,Math.max(lpRight,rect.right+8));
  let top=Math.min(window.innerHeight-ph-8,Math.max(8,rect.top));
  pop.style.left=left+'px';pop.style.top=top+'px';
  pop.style.width=pw+'px';
  await _refreshRdpMulti(runs);
  if(_prevRunning)_startRdpPoll();
}

// Patching / ET popup helpers (preserved from v45)
function patchModeClick(e,row){
  const n=parseInt(row.dataset.run);const m=row.dataset.mode;
  if(!window._selPatchModes)window._selPatchModes={};
  if(!window._selPatchModes[n])window._selPatchModes[n]=new Set();
  if(e.ctrlKey||e.shiftKey){
    if(window._selPatchModes[n].has(m))window._selPatchModes[n].delete(m);
    else window._selPatchModes[n].add(m);
  } else {
    const had=(window._selPatchModes[n].size===1&&window._selPatchModes[n].has(m));
    window._selPatchModes[n].clear();if(!had)window._selPatchModes[n].add(m);
  }
  document.querySelectorAll('#patchModeRows_'+n+' .patch-mode-row').forEach(r=>{
    r.classList.toggle('patch-sel',(window._selPatchModes[n]||new Set()).has(r.dataset.mode));
  });
}
async function rdpLaunchStd(n){
  closeRdp();selP(String(n),null);await doLaunchOrQueue();
}
async function rdpLaunchPatch(n,sessKey){
  const modes=[...((window._selPatchModes||{})[n]||new Set())];
  if(!modes.length){toast('Select at least one mode','wa');return;}
  closeRdp();
  try{
    const s=await(await fetch('/session')).json();
    s[sessKey]=modes;
    // Respect temperature dropdown — all-temps prefix if selected
    const tempVal=document.getElementById('bcTempSel').value;
    if(tempVal==='all'){
      s.runs='all-temps:'+String(n);
    } else if(tempVal==='auto'){
      toast('Full sweep ignores mode selection - use All temps','wa');return;
    } else {
      s.runs=String(n);
    }
    const r=await fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(s)});
    const d=await r.json();
    if(!d.ok){toast(d.error||'Launch failed','er');return;}
    toast('Launched','ok');
  }catch(e){toast('Launch failed: '+e,'er');}
}
async function rdpLaunchET(n,mode){
  closeRdp();
  try{
    const s=await(await fetch('/session')).json();
    s.et_mode=mode;s.runs=String(n);
    const r=await fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(s)});
    const d=await r.json();
    if(!d.ok){toast(d.error||'Launch failed','er');return;}
    toast('ET launch: mode '+mode,'ok');
  }catch(e){toast('Launch failed: '+e,'er');}
}
async function rdpLaunchETConfirm(n,mode){
  if(!confirm('Run '+n+': abliterated generation + E\\u209c pass (mode B). This overwrites existing abliterated data. Continue?'))return;
  rdpLaunchET(n,mode);
}

// ET batch popup
let _etSel={};
function openEtPop(){
  _etSel={};
  const pop=document.getElementById('etpop');if(!pop)return;
  const t=document.getElementById('etTable');if(!t)return;
  // v0.79.4.0: renumbered. Old [1,2,3,4,5,6,7,8,9,15,16,17,20] → new:
  // 1(was 20), 4(was 1), 5(was 2), 6(was 3), 7(was 4), 8(was 5),
  // 9(was 6), 10(was 7), 11(was 8), 12(was 9), 13(was 15), 14(was 16), 15(was 17)
  const ET_RUNS=[1,4,5,6,7,8,9,10,11,12,13,14,15];
  t.innerHTML='';
  ET_RUNS.forEach(n=>{
    const st=scanSt[String(n)]||'missing';
    const stcls={'done':'done','needs_et':'neet','et_partial':'etpt','partial':'part','missing':'miss'}[st]||'miss';
    const tr=document.createElement('tr');
    tr.innerHTML='<td><input type="checkbox" id="etc'+n+'" style="accent-color:var(--ac)" onchange="etChg('+n+',this.checked)"></td>'+
      '<td style="font-size:10px;color:var(--t2)">'+String(n).padStart(2,'0')+'</td>'+
      '<td style="font-size:9px;color:var(--t2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:150px">'+
        escH(descs[String(n)]||'')+'</td>'+
      '<td><span class="et-status '+stcls+'">'+escH(st)+'</span></td>'+
      '<td style="display:flex;gap:3px">'+
        '<button class="et-mode-btn'+(st!=='done'?' sel':'')+'" data-run="'+n+'" data-mode="a" onclick="etToggleMode(this)" title="Abliterated generation only">A</button>'+
        '<button class="et-mode-btn" data-run="'+n+'" data-mode="e" onclick="etToggleMode(this)" title="Base E\\u209c pass only">E</button>'+
        '<button class="et-mode-btn" data-run="'+n+'" data-mode="b" onclick="etToggleMode(this)" title="Both: generation + E\\u209c">B</button>'+
      '</td>';
    t.appendChild(tr);
    _etSel[n]={checked:false,mode:'a'};
  });
  pop.classList.add('on');
  pop.style.left='50%';pop.style.top='50%';
  pop.style.transform='translate(-50%,-50%)';
}
function etChg(n,v){if(_etSel[n])_etSel[n].checked=v;}
function etSelectAll(v){Object.keys(_etSel).forEach(n=>{_etSel[n].checked=v;const c=document.getElementById('etc'+n);if(c)c.checked=v;});}
function etSelectNeedsEt(){Object.keys(_etSel).forEach(n=>{const st=scanSt[String(n)]||'missing';const v=(st==='needs_et'||st==='et_partial');_etSel[n].checked=v;const c=document.getElementById('etc'+n);if(c)c.checked=v;});}
function etSetMode(m){Object.keys(_etSel).forEach(n=>{_etSel[n].mode=m;});
  document.querySelectorAll('.et-mode-btn').forEach(b=>{b.classList.toggle('sel',b.dataset.mode===m);});}
function etToggleMode(btn){const n=parseInt(btn.dataset.run);const m=btn.dataset.mode;
  if(_etSel[n])_etSel[n].mode=m;
  btn.closest('tr').querySelectorAll('.et-mode-btn').forEach(b=>b.classList.toggle('sel',b.dataset.mode===m));}
function closeEtPop(){const p=document.getElementById('etpop');if(p)p.classList.remove('on');}
async function launchEtBatch(){
  // v0.76.0.5: builds et_mode_overrides (dict keyed by str run num) — matches
  // the server contract at /run endpoint + _wsess whitelist + _run_session
  // override-read path. Prior version sent et_batch (list of objects) which
  // was dropped by the server; popup mode buttons were silently inert.
  const selected=Object.entries(_etSel).filter(([,v])=>v.checked);
  if(!selected.length){toast('Select at least one run','wa');return;}
  closeEtPop();
  try{
    const s=await(await fetch('/session')).json();
    s.et_mode_overrides=Object.fromEntries(selected.map(([n,v])=>[String(n),v.mode]));
    s.runs=selected.map(([n])=>n).join(',');
    delete s.et_batch; // legacy safety — ensure stale key doesn't persist
    const r=await fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(s)});
    const d=await r.json();
    if(!d.ok){toast(d.error||'Launch failed','er');return;}
    toast('ET batch launched','ok');
  }catch(e){toast('Launch failed: '+e,'er');}
}

// ── Run 0056: Cross-model paper assembly — model selection (v0.75.2.2) ─────
let _r55Models=[];
async function openR55Pop(){
  const pop=document.getElementById('r55pop');
  const ov=document.getElementById('r55overlay');
  const container=document.getElementById('r55Models');
  if(!pop||!container)return;
  container.innerHTML='<div style="color:var(--t3);padding:8px">Loading models...</div>';
  pop.style.display='block';if(ov)ov.style.display='block';
  try{
    const r=await fetch('/cross_model_status');const d=await r.json();
    _r55Models=d.models||[];
    if(!_r55Models.length){container.innerHTML='<div style="color:var(--er)">No models found.</div>';return;}
    let html='';
    for(const m of _r55Models){
      const ready=m.ready;
      const id='r55_'+m.family+'_'+m.size;
      html+='<label style="display:flex;align-items:center;gap:6px;padding:4px 0;'+
        (ready?'':'opacity:0.4;pointer-events:none;')+'">';
      html+='<input type="checkbox" id="'+id+'" '+(ready?'checked':'')+' '+(ready?'':'disabled')+' />';
      html+='<span style="font-weight:'+(ready?'600':'400')+'">'+m.label+'</span>';
      html+='<span style="color:var(--t3);font-size:8px;margin-left:auto">'+m.n_temps_analyzed+'/6 temps';
      if(m.has_r54_stamp)html+=' · R54 ✓';
      html+='</span></label>';
    }
    container.innerHTML=html;
  }catch(e){container.innerHTML='<div style="color:var(--er)">Failed: '+e+'</div>';}
}
function closeR55Pop(){
  const pop=document.getElementById('r55pop');if(pop)pop.style.display='none';
  const ov=document.getElementById('r55overlay');if(ov)ov.style.display='none';
}
async function launchR55(){
  const selected=_r55Models.filter(m=>{
    const cb=document.getElementById('r55_'+m.family+'_'+m.size);
    return cb&&cb.checked;
  });
  if(!selected.length){toast('Select at least one model','wa');return;}
  closeR55Pop();
  try{
    const s=await(await fetch('/session')).json();
    s.cross_model_include=selected.map(m=>m.label);
    // Include other runs (50,51,52) that were selected alongside 55
    const other=window._r55OtherRuns||'';
    const allRuns=other?other+',55':'55';
    s.runs=(_navLayer===2)?'all-temps:'+allRuns:allRuns;
    const r=await fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(s)});
    const d=await r.json();
    if(!d.ok){toast(d.error||'Launch failed','er');return;}
    toast('Launched '+allRuns+' ('+selected.length+' models)','ok');
  }catch(e){toast('Launch failed: '+e,'er');}
}

// Run 0003 helpers
function mcCondClick(e,row){
  const all=row.closest('#mcCondRows_'+parseInt(row.dataset.run));
  if(!all)return;
  if(!e.ctrlKey&&!e.shiftKey){all.querySelectorAll('.patch-mode-row').forEach(r=>r.classList.remove('patch-sel'));}
  row.classList.toggle('patch-sel');
}
async function mcLaunchSelected(n){
  const rows=document.querySelectorAll('#mcCondRows_'+n+' .patch-sel');
  const conds=[...rows].map(r=>r.dataset.cond);
  if(!conds.length){toast('Select conditions first','wa');return;}
  closeRdp();
  try{
    const s=await(await fetch('/session')).json();
    s.mc_conds=conds;s.runs=String(n);
    const r=await fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(s)});
    const d=await r.json();
    if(!d.ok){toast(d.error||'Launch failed','er');return;}
    toast('Launched (selected conditions)','ok');
  }catch(e){toast('Launch failed: '+e,'er');}
}
async function rdpLaunchMCCond(n,cond){
  closeRdp();
  try{
    const s=await(await fetch('/session')).json();
    s.mc_conds=[cond];s.runs=String(n);
    const r=await fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(s)});
    const d=await r.json();
    if(!d.ok){toast(d.error||'Launch failed','er');return;}
    toast('Launched condition '+cond,'ok');
  }catch(e){toast('Launch failed: '+e,'er');}
}

// ── Tiling panel drag (edge-redistributes space) ──────────────────────────────
(function(){
  function save(k,v){try{localStorage.setItem(k,String(v));}catch(e){}}
  function load(k,fb){try{const v=localStorage.getItem(k);return v!=null?parseFloat(v):fb;}catch(e){return fb;}}
  function setCSSVar(n,v){document.documentElement.style.setProperty(n,v);}

  // Restore saved values
  const lc=load('iota_lc',30);
  const lh=load('iota_lh',45);
  setCSSVar('--lc',Math.max(15,Math.min(75,lc))+'%');
  setCSSVar('--lh',Math.max(10,Math.min(90,lh))+'%');

  // Vertical splitter: redistributes % between left-col and right panel
  const vspl=document.getElementById('vspl');
  if(vspl){
    let drag=false,startX=0,startLc=40;
    vspl.addEventListener('mousedown',e=>{
      drag=true;startX=e.clientX;
      startLc=parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--lc'))||40;
      vspl.classList.add('drag');
      document.body.style.cursor='col-resize';document.body.style.userSelect='none';
      e.preventDefault();
    });
    document.addEventListener('mousemove',e=>{
      if(!drag)return;
      const total=document.getElementById('workspace').offsetWidth;
      if(!total)return;
      const delta=(e.clientX-startX)/total*100;
      const newLc=Math.max(15,Math.min(75,startLc+delta));
      setCSSVar('--lc',newLc+'%');
      save('iota_lc',newLc.toFixed(1));
    });
    document.addEventListener('mouseup',()=>{
      if(!drag)return;drag=false;
      vspl.classList.remove('drag');
      document.body.style.cursor='';document.body.style.userSelect='';
    });
    vspl.addEventListener('dblclick',()=>{setCSSVar('--lc','30%');save('iota_lc',30);});
  }

  // Horizontal splitter: redistributes % between grid and metrics (inside left col)
  const hspl=document.getElementById('hspl');
  if(hspl){
    let drag=false,startY=0,startLh=50;
    hspl.addEventListener('mousedown',e=>{
      drag=true;startY=e.clientY;
      startLh=parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--lh'))||50;
      hspl.classList.add('drag');
      document.body.style.cursor='row-resize';document.body.style.userSelect='none';
      e.preventDefault();
    });
    document.addEventListener('mousemove',e=>{
      if(!drag)return;
      const leftCol=document.getElementById('left-col');
      const total=leftCol?leftCol.offsetHeight:0;
      if(!total)return;
      const delta=(e.clientY-startY)/total*100;
      const newLh=Math.max(10,Math.min(90,startLh+delta));
      setCSSVar('--lh',newLh+'%');
      save('iota_lh',newLh.toFixed(1));
    });
    document.addEventListener('mouseup',()=>{
      if(!drag)return;drag=false;
      hspl.classList.remove('drag');
      document.body.style.cursor='';document.body.style.userSelect='';
    });
    hspl.addEventListener('dblclick',()=>{setCSSVar('--lh','45%');save('iota_lh',45);});
  }
})();

// ── Purge helpers ─────────────────────────────────────────────────────────────
async function purgeRun(n){
  if(_prevRunning){toast('Cannot purge while a run is active','er');return;}
  if(!confirm('Delete all data for Run '+n+'? This cannot be undone.'))return;
  try{
    const r=await fetch('/purge_run',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({run:n})});
    const d=await r.json();
    if(d.ok)toast('Run '+n+' data deleted','ok');
    else toast(d.error||'Purge failed','er');
    setTimeout(doScan,300);
  }catch(e){toast('Purge failed: '+e,'er');}
}

// ── Utilities ─────────────────────────────────────────────────────────────────
let tt;
function toast(msg,k='in'){const e=document.getElementById('tst');
  e.textContent=msg;e.className=k;clearTimeout(tt);tt=setTimeout(()=>{e.className='';e.textContent='';},4000);}
function S(id,v){const e=document.getElementById(id);if(e)e.textContent=v??'\\u2014';}
function f4(v){return v!=null?Number(v).toFixed(4):null;}
function p2(n){return String(n).padStart(2,'0');}
function bar(bi,pi,v,t){const _be=document.getElementById(bi);if(!_be)return;_be.style.width=Math.round(v/t*100)+'%';
  const _pe=document.getElementById(pi);if(_pe)_pe.textContent=v+'/'+t;}
function escH(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
function _fmtEta(sec){
  if(!sec||sec<0||!isFinite(sec))return'';
  if(sec<60)return Math.round(sec)+'s';
  if(sec<3600)return Math.round(sec/60)+'m';
  const h=Math.floor(sec/3600),m=Math.round((sec%3600)/60);
  return h+'h'+m+'m';
}

// ── Keyboard shortcuts ────────────────────────────────────────────────────────
document.addEventListener('keydown',e=>{
  const tag=e.target.tagName;const inField=tag==='INPUT'||tag==='TEXTAREA'||tag==='SELECT';
  if(e.ctrlKey&&e.key==='l'){e.preventDefault();clrC();return;}
  if(e.ctrlKey&&e.key==='k'){e.preventDefault();
    setTimeout(()=>document.getElementById('ri').focus(),50);return;}
  if(inField)return;
  if(e.code==='Space'){e.preventDefault();doPause();}
  else if(e.key==='r'||e.key==='R')doLaunchOrQueue();
  else if(e.key==='Escape'){
    const ep=document.getElementById('etpop');
    if(ep&&ep.classList.contains('on')){closeEtPop();return;}
    const p=document.getElementById('rdpop');
    if(p&&p.classList.contains('on'))closeRdp();else doAbort();}
  else if(e.key==='1')sw('C');else if(e.key==='2')sw('H');else if(e.key==='3')sw('N');
});

function updateClearBtn(){
  const scope=document.getElementById('clearScope').value;
  const btn=document.getElementById('nukeBtn');
  const desc=document.getElementById('clearDesc');
  const runs=_getSelectedRuns();
  const _ml=document.querySelector('.ml');
  const mn=_ml?_ml.textContent:'this model';
  const descs={
    'run_temp':'Deletes CSV + hidden states + analysis for the selected runs at the viewed temperature only. '+mn+' other temperatures and all other models are untouched.',
    'run_all':'Deletes CSV + hidden states + analysis for the selected runs at EVERY temperature. '+mn+' only. All other models are untouched.',
    'temp':'Deletes the entire temperature directory for '+mn+' \u2014 all runs, hidden states, analysis, visuals. Other temperatures and other models are untouched.',
    'all_temps':'Deletes ALL temperature directories + pooled results for '+mn+'. Calibration file is kept. All other models are untouched.',
    'delete_model':'Permanently deletes the ENTIRE '+mn+' directory \u2014 all temperatures, all data, calibration, everything. Other models are untouched.',
    'factory_reset':'\u2620 Deletes EVERYTHING for EVERY model. All data, logs, session, calibration, resume state. Only the Python source code survives. There is no undo.'
  };
  if(desc)desc.textContent=descs[scope]||'';
  if(!btn)return;
  if(scope==='factory_reset'){
    btn.className='nuke-btn mega';
    btn.innerHTML='\u2620 NUKE EVERYTHING \u2620';
  }else if(scope==='delete_model'){
    btn.className='nuke-btn mega';
    btn.innerHTML='&#x1f4a3; DELETE MODEL';
  }else if(scope==='all_temps'){
    btn.className='nuke-btn';
    btn.innerHTML='\u26a0\ufe0f WIPE ALL TEMPS';
  }else if(scope==='temp'){
    btn.className='nuke-btn';
    btn.innerHTML='&#x1f5d1; Clear Temp';
  }else if(scope==='run_all'){
    btn.className='nuke-btn';
    btn.innerHTML=runs?'&#x1f5d1; Clear '+runs.split(',').length+' runs \u00d7 all temps':'&#x1f5d1; Clear Runs';
  }else{
    btn.className='nuke-btn';
    btn.innerHTML=runs?'&#x1f5d1; Clear '+runs.split(',').length+' runs':'&#x1f5d1; Clear';
  }
}
async function clearDataPrompt(){
  if(_prevRunning){toast('Cannot clear while a run is active','er');return;}
  const scope=document.getElementById('clearScope').value;
  const runs=_getSelectedRuns();
  if((scope==='run_temp'||scope==='run_all')&&!runs){
    toast('Select runs in the grid first','wa');return;
  }
  if(scope==='factory_reset'){
    if(!confirm('\\u2620 FACTORY RESET \\u2620\\n\\nThis will delete:\\n\\u2022 ALL model data (every model, every temperature)\\n\\u2022 All log files\\n\\u2022 Session and resume files\\n\\u2022 Calibration data\\n\\u2022 Everything except the Python code\\n\\nThis cannot be undone.'))return;
    if(!confirm('Are you absolutely sure?\\n\\nType of destruction: TOTAL\\nRecovery options: NONE\\nRegret probability: HIGH'))return;
    if(!confirm('Final warning. Last chance to walk away.\\n\\n\\u2620 Deleting everything in 3... 2... 1...'))return;
    try{
      const r=await fetch('/factory_reset',{method:'POST'});
      const d=await r.json();
      if(d.ok){toast('\\u2620 Factory reset complete. '+d.deleted+' files destroyed.','ok');
        setTimeout(()=>location.reload(),1500);}
      else toast(d.error||'Reset failed','er');
    }catch(e){toast('Reset failed: '+e,'er');}
    return;
  }
  if(scope==='delete_model'){
    if(!confirm('Delete entire model directory?\\n\\nThis includes ALL data, analysis, calibration, and results for this model.\\n\\nThis cannot be undone.'))return;
    if(!confirm('Are you sure? Every file for this model will be permanently deleted.'))return;
    try{
      const s=await(await fetch('/session')).json();
      const r=await fetch('/delete_model',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({family:s.model_family,size:s.model_size,variant:s.model_variant})});
      const d=await r.json();
      if(d.ok){toast('Model deleted ('+d.deleted+' files)','ok');await buildModelCards();setTimeout(doScan,500);}
      else toast(d.error||'Delete failed','er');
    }catch(e){toast('Delete failed: '+e,'er');}
    return;
  }
  const labels={
    'run_temp':'selected run(s) at current temperature',
    'run_all':'selected run(s) at ALL temperatures',
    'temp':'ALL runs at current temperature',
    'all_temps':'ALL data at ALL temperatures'
  };
  let msg='Clear '+labels[scope];
  if(runs&&(scope==='run_temp'||scope==='run_all'))msg+='\\n\\nRuns: '+runs;
  if(scope==='all_temps')msg+='\\n\\nThis cannot be undone.';
  if(!confirm(msg+'\\n\\nAre you sure?'))return;
  try{
    const r=await fetch('/clear_data',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({scope:scope,runs:runs||''})});
    const d=await r.json();
    if(d.ok){toast('Cleared '+d.deleted+' files','ok');setTimeout(doScan,500);}
    else{toast(d.error||'Clear failed','er');}
  }catch(e){toast('Clear failed: '+e,'er');}
}

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded',()=>{
  const ri=document.getElementById('ri');
  if(ri){ri.addEventListener('input',()=>{const s=ri.value;selP(s,null);ri.value=s;});}
});
buildGrid();loadSt();loadModelsData();buildModelCards();initAddModelForm();loadTempGrid();loadHyp();updateClearBtn();
// Load run descriptions for tooltips and popups
fetch('/descs').then(r=>r.json()).then(d=>{
  Object.assign(descs,d);
  buildGrid();  // rebuild with descriptions now available
}).catch(()=>{});
// Load recent console history on page load
fetch('/log?tail=200').then(r=>r.json()).then(d=>{
  const con=document.getElementById('con');
  _addLBatch(d.lines||[],()=>{
    LLN=d.total||(d.lines||[]).length;
    LOG_START=Math.max(0,LLN-200);
    logInterval=setInterval(pollLog,10000);
    if(AS&&con)con.scrollTop=1e9;
    // Show scroll indicator if there's history above
    const ind=document.getElementById('con-load-indicator');
    if(ind&&LOG_START>0){ind.textContent='\\u25b2 scroll up to load history';ind.classList.add('on');}
  });
}).catch(()=>{logInterval=setInterval(pollLog,10000);});
setTimeout(doScan,800);updateScanRate(false);
checkResumeState();
</script>
</body>
</html>"""



# ── Run server ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--daemon', action='store_true')
    args, _ = parser.parse_known_args()
    # Clean orphaned .tmp files from prior hard crashes
    import glob as _g_tmp
    for _f in _g_tmp.glob(os.path.join(ROOT, '*.tmp')):
        try: os.remove(_f)
        except Exception: pass
    # Suppress ALL Flask/werkzeug/click output
    import logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    # Kill Flask's show_server_banner which prints "Serving Flask app" etc.
    import flask.cli
    flask.cli.show_server_banner = lambda *a, **kw: None
    # Start server in background thread
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000,
                     debug=False, threaded=True), daemon=True).start()
    print("\n" + "=" * 48)
    print("  IOTA Dashboard")
    print("  http://localhost:5000")
    print("=" * 48)
    if not args.daemon:
        import webbrowser
        import time as _tm
        _tm.sleep(0.5)  # let server start
        webbrowser.open("http://localhost:5000")
        while True:
            try:
                input("\n  Press Enter to reopen in browser... ")
                webbrowser.open("http://localhost:5000")
            except (EOFError, KeyboardInterrupt):
                print("\n  Shutting down.")
                break
    else:
        try:
            while True:
                import time; time.sleep(3600)
        except KeyboardInterrupt:
            print("\n  Shutting down.")

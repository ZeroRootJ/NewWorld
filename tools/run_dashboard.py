"""Local progress dashboard for a multi_truth_axes run (stdlib only).

Reads, never writes, the run folder produced by src/experiments/multi_truth_axes.py:
run_config.json (planned cells), run_registry.jsonl (completed tasks),
failures.jsonl, progress.log, and each completed SGS run's manifest.json
(for the sgsim near-singular warning count).

Usage (from the repo root):
    .venv/Scripts/python.exe tools/run_dashboard.py                 # latest run folder
    .venv/Scripts/python.exe tools/run_dashboard.py --run-dir results/raw/multi_truth_axes/<ts>
    then open http://127.0.0.1:8765

ETA: remaining tasks x per-task duration / workers. Per-task duration is the
observed mean for that (method, n_samples) once >= 3 such tasks finished,
otherwise the observed mean for the method, otherwise a prior (SGS: E2-E5
smoke-test seconds per realization x N; other methods 30 s). It assumes the
pool stays full, so it is optimistic near the end of the run.
"""

import argparse
import json
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_LABELS = {
    "E5": "E5 base case",
    "E2": "E2 range axis",
    "E3": "E3 nugget axis",
    "E4": "E4 density axis",
}
METHOD_ORDER = ["kriging", "sgs", "rbf_bootstrap", "gp_mle"]
# Smoke-test SGS seconds per realization by n_samples requested (coder report, 20 workers).
SGS_PRIOR_S_PER_REAL = {500: 13.58, 250: 13.51, 125: 12.97, 50: 9.89, 25: 7.28}
OTHER_PRIOR_S = 30.0
MIN_OBS = 3


def _read_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a line being written right now
    return rows


class RunState:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        self._warn_cache = {}

    def _sgs_warnings(self, rel_run_dir):
        if rel_run_dir in self._warn_cache:
            return self._warn_cache[rel_run_dir]
        n = None
        mpath = REPO_ROOT / rel_run_dir / "manifest.json"
        try:
            m = json.loads(mpath.read_text(encoding="utf-8"))
            n = m.get("params", {}).get("sgsim_warnings", {}).get("n_grid_node_warnings")
        except (OSError, json.JSONDecodeError):
            return None  # not cached: retry next poll
        self._warn_cache[rel_run_dir] = n
        return n

    def status(self):
        cfg = self.config
        methods = cfg["methods"]
        n_ens = cfg.get("n_ensemble_this_batch")
        cells = cfg["cells"]

        # config_key -> (n_samples, experiments using it)
        key_n = {}
        key_exps = defaultdict(set)
        for c in cells:
            key_n[c["config_key"]] = c["n_samples"]
            key_exps[c["config_key"]].add(c["experiment"])
        planned = {(k, m) for k in key_n for m in methods}

        reg = _read_jsonl(self.run_dir / "run_registry.jsonl")
        done = {}
        for r in reg:
            done[(r["config_key"], r["method"])] = r
        failures = _read_jsonl(self.run_dir / "failures.jsonl")
        failed_keys = {(f.get("config_key"), f.get("method")) for f in failures} - set(done)

        # per experiment x method
        exp_tbl = {}
        for exp in cfg["experiments"]:
            keys = [k for k in key_n if exp in key_exps[k]]
            row = {}
            for m in methods:
                tot = len(keys)
                d = sum(1 for k in keys if (k, m) in done)
                row[m] = {"done": d, "total": tot}
            exp_tbl[exp] = {
                "label": EXPERIMENT_LABELS.get(exp, exp),
                "methods": row,
                "done": sum(v["done"] for v in row.values()),
                "total": sum(v["total"] for v in row.values()),
            }

        # durations
        by_mn = defaultdict(list)
        by_m = defaultdict(list)
        for (k, m), r in done.items():
            s = r.get("total_seconds")
            if s is not None:
                by_mn[(m, key_n.get(k))].append(s)
                by_m[m].append(s)

        def est(m, n):
            if len(by_mn[(m, n)]) >= MIN_OBS:
                return sum(by_mn[(m, n)]) / len(by_mn[(m, n)]), "observed"
            if m != "sgs" and len(by_m[m]) >= MIN_OBS:
                return sum(by_m[m]) / len(by_m[m]), "observed"
            if m == "sgs":
                return SGS_PRIOR_S_PER_REAL.get(n, 13.0) * (n_ens or 50), "prior"
            return OTHER_PRIOR_S, "prior"

        remaining = [t for t in planned if t not in done]
        work = 0.0
        prior_used = 0
        for k, m in remaining:
            s, src = est(m, key_n[k])
            work += s
            prior_used += src == "prior"

        method_tbl = []
        for m in [x for x in METHOD_ORDER if x in methods]:
            ds = by_m[m]
            method_tbl.append({
                "method": m,
                "done": len(ds),
                "total": len(key_n),
                "mean_s": (sum(ds) / len(ds)) if ds else None,
                "max_s": max(ds) if ds else None,
            })

        # progress.log: session start, workers, last lines, freshness
        log_path = self.run_dir / "progress.log"
        lines = []
        if log_path.exists():
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        workers = 20
        session_start = None
        for ln in lines:
            if "session start" in ln:
                mw = re.search(r"workers=(\d+)", ln)
                if mw:
                    workers = int(mw.group(1))
                session_start = ln.split(" ", 1)[0]
        last_update_age = None
        if log_path.exists():
            last_update_age = time.time() - log_path.stat().st_mtime

        warn_total, warn_runs, warn_checked = 0, 0, 0
        for (k, m), r in done.items():
            if m == "sgs":
                n = self._sgs_warnings(r["run_dir"])
                if n is not None:
                    warn_checked += 1
                    warn_total += n
                    warn_runs += n > 0

        manifest_status = None
        try:
            manifest_status = json.loads((self.run_dir / "manifest.json").read_text(encoding="utf-8")).get("status")
        except (OSError, json.JSONDecodeError):
            pass

        return {
            "run_dir": str(self.run_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
            "batch": cfg.get("batch"),
            "n_ensemble": n_ens,
            "workers": workers,
            "now_utc": datetime.now(timezone.utc).isoformat(),
            "session_start_utc": session_start,
            "last_update_age_s": last_update_age,
            "manifest_status": manifest_status,
            "done": len([t for t in planned if t in done]),
            "total": len(planned),
            "n_failed_unresolved": len(failed_keys),
            "failures": failures[-20:],
            "eta_s": work / max(workers, 1),
            "eta_tasks_on_prior": prior_used,
            "experiments": exp_tbl,
            "methods": method_tbl,
            "sgs_warnings": {"total": warn_total, "runs_with_warnings": warn_runs, "runs_checked": warn_checked},
            "log_tail": lines[-14:],
        }


PAGE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Run Progress</title>
<style>
:root{--bg:#f5f6f8;--card:#fff;--fg:#1a2230;--muted:#5d6878;--line:#dde2e9;--accent:#2563a8;
--good:#1e7a4b;--warn:#a5640a;--bad:#b3261e;--track:#e6eaf0;
--mono:ui-monospace,Consolas,"IBM Plex Mono",monospace;--sans:system-ui,"Malgun Gothic",sans-serif}
@media (prefers-color-scheme:dark){:root{--bg:#12161c;--card:#1a2029;--fg:#e4e9f0;--muted:#9aa6b6;
--line:#2d3642;--accent:#7cb4e8;--good:#6fcf9b;--warn:#e8b860;--bad:#f08b84;--track:#2a323d}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 var(--sans)}
.wrap{max-width:1080px;margin:0 auto;padding:24px 16px 48px;display:flex;flex-direction:column;gap:16px}
h1{font-size:20px;margin:0;font-weight:600}
.sub{color:var(--muted);font-size:13px;font-family:var(--mono);overflow-wrap:anywhere}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}
.kpi .v{font-size:24px;font-weight:600;font-variant-numeric:tabular-nums}
.kpi .l{color:var(--muted);font-size:12.5px}
.bar{height:10px;background:var(--track);border-radius:5px;overflow:hidden}
.bar>i{display:block;height:100%;background:var(--accent);border-radius:5px;transition:width .4s}
.bar.big{height:14px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line);vertical-align:middle}
th{color:var(--muted);font-weight:500;font-size:12.5px}
tr:last-child td{border-bottom:0}
td.n{text-align:right;font-family:var(--mono);font-size:13px;white-space:nowrap}
.cellbar{display:flex;align-items:center;gap:8px;min-width:120px}
.cellbar .bar{flex:1}
.cellbar span{font-family:var(--mono);font-size:12px;color:var(--muted);white-space:nowrap}
.tblwrap{overflow-x:auto}
pre{margin:0;font:12px/1.55 var(--mono);white-space:pre-wrap;overflow-wrap:anywhere;color:var(--muted)}
.pill{display:inline-block;padding:1px 9px;border-radius:999px;font-size:12px;font-weight:500;border:1px solid currentColor}
.ok{color:var(--good)}.warn{color:var(--warn)}.bad{color:var(--bad)}
h2{font-size:14px;margin:0 0 10px;font-weight:600}
.row{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;justify-content:space-between}
</style></head><body><div class="wrap">
<div class="row"><div><h1>Multi-truth axes run</h1><div class="sub" id="rundir"></div></div>
<span class="pill" id="state">loading</span></div>
<div class="card"><div class="row" style="margin-bottom:8px"><strong id="pct"></strong><span class="sub" id="counts"></span></div>
<div class="bar big"><i id="mainbar" style="width:0"></i></div></div>
<div class="kpis">
<div class="card kpi"><div class="v" id="eta">-</div><div class="l" id="etal">남은 시간(추정)</div></div>
<div class="card kpi"><div class="v" id="finish">-</div><div class="l">예상 종료 시각</div></div>
<div class="card kpi"><div class="v" id="elapsed">-</div><div class="l">경과 시간</div></div>
<div class="card kpi"><div class="v" id="fail">-</div><div class="l">미해결 실패</div></div>
<div class="card kpi"><div class="v" id="warn">-</div><div class="l" id="warnl">SGS 경고 run</div></div>
</div>
<div class="card"><h2>실험 × 방법</h2><div class="tblwrap"><table id="exp"></table></div></div>
<div class="card"><h2>방법별 실행 시간</h2><div class="tblwrap"><table id="meth"></table></div></div>
<div class="card" id="failcard" hidden><h2>실패</h2><pre id="faillist"></pre></div>
<div class="card"><h2>progress.log (최근)</h2><pre id="log"></pre></div>
<div class="sub" id="foot"></div>
</div>
<script>
const M=["kriging","sgs","rbf_bootstrap","gp_mle"];
const fmt=s=>{if(s==null)return"-";s=Math.max(0,Math.round(s));const h=Math.floor(s/3600),m=Math.floor(s%3600/60);return h?`${h}h ${String(m).padStart(2,"0")}m`:`${m}m ${String(s%60).padStart(2,"0")}s`};
const bar=(d,t)=>`<div class="cellbar"><div class="bar"><i style="width:${t?100*d/t:0}%"></i></div><span>${d}/${t}</span></div>`;
async function tick(){
 let s;try{s=await (await fetch("/api/status",{cache:"no-store"})).json()}catch(e){document.getElementById("state").textContent="server offline";document.getElementById("state").className="pill bad";return}
 const $=id=>document.getElementById(id);
 $("rundir").textContent=`${s.run_dir} · batch ${s.batch} · N=${s.n_ensemble} · workers ${s.workers}`;
 const p=s.total?s.done/s.total:0;$("pct").textContent=(100*p).toFixed(1)+" %";
 $("counts").textContent=`${s.done} / ${s.total} tasks`;$("mainbar").style.width=(100*p)+"%";
 const complete=s.done>=s.total, stale=s.last_update_age_s!=null&&s.last_update_age_s>1800;
 $("state").textContent=complete?"complete":stale?`no log update ${fmt(s.last_update_age_s)}`:"running";
 $("state").className="pill "+(complete?"ok":stale?"warn":"ok");
 $("eta").textContent=complete?"0":fmt(s.eta_s);
 $("etal").textContent="남은 시간(추정)"+(s.eta_tasks_on_prior?` · ${s.eta_tasks_on_prior}개 사전값`:"");
 $("finish").textContent=complete?"-":new Date(Date.now()+s.eta_s*1000).toLocaleString([], {month:"numeric",day:"numeric",hour:"2-digit",minute:"2-digit"});
 $("elapsed").textContent=s.session_start_utc?fmt((Date.now()-Date.parse(s.session_start_utc))/1000):"-";
 $("fail").textContent=s.n_failed_unresolved;$("fail").className="v "+(s.n_failed_unresolved?"bad":"");
 const w=s.sgs_warnings;$("warn").textContent=w.runs_with_warnings;$("warn").className="v "+(w.runs_with_warnings?"warn":"");
 $("warnl").textContent=`SGS 경고 run (경고 ${w.total}개, ${w.runs_checked}개 확인)`;
 let h="<tr><th>실험</th>"+M.map(m=>`<th>${m}</th>`).join("")+"<th>전체</th></tr>";
 for(const [k,e] of Object.entries(s.experiments)){h+=`<tr><td>${e.label}</td>`+M.map(m=>e.methods[m]?`<td>${bar(e.methods[m].done,e.methods[m].total)}</td>`:"<td>-</td>").join("")+`<td>${bar(e.done,e.total)}</td></tr>`}
 $("exp").innerHTML=h;
 $("meth").innerHTML="<tr><th>방법</th><th>완료</th><th class='n'>평균</th><th class='n'>최대</th></tr>"+s.methods.map(m=>`<tr><td>${m.method}</td><td>${bar(m.done,m.total)}</td><td class="n">${fmt(m.mean_s)}</td><td class="n">${fmt(m.max_s)}</td></tr>`).join("");
 $("failcard").hidden=!s.failures.length;$("faillist").textContent=s.failures.map(f=>JSON.stringify(f)).join("\n");
 $("log").textContent=s.log_tail.join("\n");
 $("foot").textContent=`10초마다 갱신 · 마지막 로그 갱신 ${fmt(s.last_update_age_s)} 전 · ETA는 풀이 꽉 찬 상태를 가정하므로 막바지에는 낙관적입니다`;
}
tick();setInterval(tick,10000);
</script></body></html>"""


def make_handler(state: RunState):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/api/status"):
                body = json.dumps(state.status()).encode("utf-8")
                ctype = "application/json; charset=utf-8"
            elif self.path in ("/", "/index.html"):
                body = PAGE.encode("utf-8")
                ctype = "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", help="multi_truth_axes run folder (default: latest)")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = REPO_ROOT / run_dir
    else:
        cands = sorted((REPO_ROOT / "results/raw/multi_truth_axes").glob("*/run_config.json"))
        if not cands:
            raise SystemExit("no multi_truth_axes run folder found")
        run_dir = cands[-1].parent
    state = RunState(run_dir.resolve())
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state))
    print(f"dashboard for {run_dir} at http://127.0.0.1:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()

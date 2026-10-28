#!/usr/bin/env python3
"""Bounded CPU observations; wake the existing inspector, never control a GPU.

The registered 75-arm batch becoming ready is distinct from completion of the
user's post-batch request. This watcher never stops either monitoring timer.
"""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "Qwen_vl/outputs/audit_followup_20261008"
PRIVATE = BASE / "stall_watch_20261009"
CONFIG = PRIVATE / "config.json"
HOURLY_SERVICE = "codex-eadp-repair-monitor-20261008.service"
PHASE2 = BASE / "streamwait_continuation_20261008"
STATE = PHASE2 / "legacy_streamwait_continuation.state.json"
PLAN = PHASE2 / "legacy_streamwait_continuation.plan.json"
PUBLISH = BASE / "rerun_batch/group_publish/publish_state.json"
FOLLOWUP = BASE / "post_batch_followup_20261009/request.json"
ACTIVE = {"generating", "loading_reserved", "loading", "scoring", "validating"}
BAD = {"failed", "interrupted_preserved"}
STALL_SECONDS = 600
TRANSITION_SECONDS = 120
JSON_LIMIT = 64 * 1024 * 1024
ROW_LIMIT = 65536


def now_iso(epoch=None):
    return dt.datetime.fromtimestamp(time.time() if epoch is None else epoch,
                                    dt.timezone.utc).isoformat()


def read_json(path, limit=JSON_LIMIT):
    path = Path(path)
    with path.open("rb") as handle:
        if os.fstat(handle.fileno()).st_size > limit:
            raise ValueError("CPU snapshot input exceeds bounded limit: " + str(path))
        return json.load(handle)


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + str(os.getpid()) + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def proc(pid):
    result = {"pid": pid, "alive": False}
    if not isinstance(pid, int) or pid < 1:
        return result
    try:
        # comm may contain ')' or spaces; fields after the final ')' are stable.
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        argv = [x.decode(errors="replace") for x in
                (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0") if x]
        result.update(alive=fields[0] not in {"Z", "X"}, state=fields[0],
                      ppid=int(fields[1]), start_ticks=int(fields[19]), argv=argv)
    except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, ValueError):
        pass
    return result


def file_stat(path):
    try:
        value = Path(path).stat()
        return {"exists": True, "size": value.st_size, "mtime_ns": value.st_mtime_ns,
                "inode": value.st_ino}
    except FileNotFoundError:
        return {"exists": False, "size": 0, "mtime_ns": None, "inode": None}


def first_record(path):
    """One bounded line only; an incomplete tail cannot manufacture progress."""
    try:
        with Path(path).open("rb") as handle:
            row = handle.readline(ROW_LIMIT + 1)
        if len(row) > ROW_LIMIT or not row.endswith(b"\n"):
            return False
        return isinstance(json.loads(row), dict)
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return False


def epoch(text, fallback):
    try:
        return dt.datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except (AttributeError, ValueError):
        return fallback


def make_issue(kind, detail, job=None):
    return {"key": kind + (":" + job if job else ""), "kind": kind,
            "job": job, "detail": detail}


def identity_matches(actual, expected, argv):
    return (actual.get("alive") and actual.get("pid") == expected.get("pid")
            and actual.get("start_ticks") == expected.get("start_ticks")
            and actual.get("argv") == argv)


def generation_progress(files, previous, started, now):
    """Only observed writes reset timers; rereading a stale file does not."""
    result = {}
    for channel, keys in (("output", ("prediction", "runtime")), ("log", ("log",))):
        signature = {key: dict(files[key]) for key in keys}
        old = previous.get(channel, {})
        # First snapshot derives age from actual file timestamps, not watcher start.
        mtime = max([files[key]["mtime_ns"] / 1e9 for key in keys
                     if files[key].get("mtime_ns") is not None] or [started])
        if old and old.get("signature") == signature:
            changed = old["last_change_epoch"]
        elif old:
            changed = now
        else:
            changed = min(now, max(started, mtime))
        result[channel] = {"signature": signature, "last_change_epoch": changed}
    return result


def assess_job(job, previous, now):
    """Pure policy function, exercised on CPU boundary fixtures."""
    issues = []
    memo = {}
    if job["status"] in BAD or job.get("worker_status") in BAD:
        return [make_issue("worker_failed", "Worker/supervisor records a failed job", job["id"])], memo
    if job["status"] not in ACTIVE:
        return issues, memo
    identity = {k: job["expected_identity"].get(k) for k in ("pid", "start_ticks")}
    same_attempt = previous.get("identity") == identity
    previous = previous if same_attempt else {}
    memo["identity"] = identity
    actual = job["process"]
    if actual.get("alive") and not identity_matches(actual, identity, job["expected_argv"]):
        issues.append(make_issue("worker_identity_differs", "PID start ticks or exact argv differ", job["id"]))
        return issues, memo
    if not actual.get("alive"):
        # A child can exit while the live supervisor performs strict CPU scoring.
        # Wait two minutes for the canonical job transition, never act on that PID.
        missing_since = previous.get("missing_since_epoch", now)
        memo["missing_since_epoch"] = missing_since
        if now - missing_since >= TRANSITION_SECONDS:
            issues.append(make_issue("worker_missing", "Active child absent beyond scoring/transition grace", job["id"]))
        return issues, memo
    if actual.get("state") == "T":
        issues.append(make_issue("worker_stopped", "Active worker is stopped; inspector must check authorization", job["id"]))
    started = job["started_epoch"]
    memo["progress"] = generation_progress(job["files"], previous.get("progress", {}), started, now)
    has_record = job["first_record"]
    if not has_record and now - started >= STALL_SECONDS:
        issues.append(make_issue("loading_no_first_record", "No complete prediction/runtime record after ten minutes", job["id"]))
    elif has_record and all(now - memo["progress"][key]["last_change_epoch"] >= STALL_SECONDS
                            for key in ("output", "log")):
        issues.append(make_issue("generation_stalled", "Output and log both unchanged for ten minutes", job["id"]))
    return issues, memo


def batch_ready(view, expected_groups, expected_jobs):
    """Readiness for the original inspector, not independent score verification."""
    state = view["phase2"]
    if state.get("status") != "complete" or state.get("child_pids"):
        return False
    if set(state.get("jobs", {})) != set(expected_jobs):
        return False
    if any(job.get("status") != "complete" or job.get("returncode") != 0
           for job in state["jobs"].values()):
        return False
    if not all(view["dependencies"].values()) or not view.get("phase2_marker_success"):
        return False
    if view.get("live_registered_processes"):
        return False
    pub = view["publisher"]
    return (len(expected_groups) == 55 and set(pub.get("expected_groups", [])) == set(expected_groups)
            and set(pub.get("published_groups", {})) == set(expected_groups)
            and not pub.get("pending_commit") and not pub.get("pending_publication")
            and not pub.get("pending_groups") and not pub.get("error"))


def prepare_config():
    if CONFIG.exists():
        raise ValueError("Refuse to overwrite watcher registration")
    plan = read_json(PLAN)
    state = read_json(STATE)
    publication = read_json(PUBLISH)
    supervisor = proc(state["supervisor_pid"])
    if not supervisor["alive"] or len(plan["jobs"]) != 32:
        raise ValueError("Phase2 supervisor/32-job registration is not ready")
    target = str(PHASE2 / "schedule_legacy_streamwait_continuation.py")
    expected = ["/home/dell/miniconda3/envs/llava_pruner/bin/python",
                str(BASE / "threshold_triage_20261009/json_count_distribution_overlay.py"),
                "--target", target, "--", "--run", "--plan", str(PLAN)]
    if supervisor["argv"] != expected or len(publication["expected_groups"]) != 55:
        raise ValueError("Unexpected supervisor identity or publication scope")
    jobs = {}
    for key, job in plan["jobs"].items():
        outputs = job["output_paths"]
        if any(Path(value).parent != PHASE2 for value in outputs.values()):
            raise ValueError("Unexpected job output directory: " + key)
        jobs[key] = {"n": job["n"], "task": job["task"], "generate_command": job["generate_command"],
                     "output_paths": {x: outputs[x] for x in ("prediction", "runtime", "log", "state", "finished")}}
    PRIVATE.mkdir(mode=0o700, parents=True, exist_ok=True)
    result = {"created_utc": now_iso(), "registered_batch": {"arms": 75, "groups": 55},
              "scope": "Phase2 progress plus registered-batch readiness; post-batch hold remains active",
              "phase2_supervisor": {"pid": supervisor["pid"], "start_ticks": supervisor["start_ticks"],
                                    "argv": expected},
              "expected_groups": publication["expected_groups"], "jobs": jobs,
              "stall_seconds": STALL_SECONDS, "transition_seconds": TRANSITION_SECONDS,
              "hourly_service": HOURLY_SERVICE, "never_stop_monitoring": True}
    atomic(CONFIG, result)
    return result


def dependency_snapshot():
    result = {}
    pids = []
    for name in ("lane1_v15", "lane2_next"):
        state = read_json(BASE / ("rerun_batch/" + name + "_state.json"))
        result[name] = state.get("status") == "complete" and not state.get("child_pid")
        pids.append(state.get("supervisor_pid"))
    for name, stem, n in (("full_controls", BASE / "next_gap_diagnosis_20261008/stability/next_text_full_streamwait.controller", 3),
                          ("phase1", PHASE2 / "streamwait_repair_continuation", 12)):
        state = read_json(str(stem) + ".state.json")
        marker = read_json(str(stem) + ".finished.json")
        result[name] = (state.get("status") == "complete" and not state.get("child_pids")
                        and marker.get("success") is True and len(state.get("jobs", {})) == n
                        and all(x.get("status") == "complete" for x in state["jobs"].values()))
        pids.append(state.get("supervisor_pid"))
    pope_marker = read_json(BASE / "rerun_fast/pope_v15_AZ32.stream_control.finished.json")
    result["pope_control"] = pope_marker.get("success") is True
    return result, pids


def collect(config):
    start = time.monotonic()
    state = read_json(STATE)
    supervisor = proc(state.get("supervisor_pid"))
    jobs = []
    now = time.time()
    for key, row in state.get("jobs", {}).items():
        if row.get("status") not in ACTIVE | BAD:
            continue
        if key not in config["jobs"]:
            raise ValueError("Unregistered active job: " + key)
        spec = config["jobs"][key]
        paths = spec["output_paths"]
        wstate = read_json(paths["state"]) if Path(paths["state"]).exists() else {}
        actual = proc(row.get("pid"))
        files = {name: file_stat(paths[name]) for name in ("prediction", "runtime", "log")}
        expected = row.get("process_identity", {})
        jobs.append({"id": key, "status": row["status"], "n": spec["n"], "task": spec["task"],
                     "expected_identity": {"pid": row.get("pid"), "start_ticks": expected.get("start_ticks")},
                     "expected_argv": spec["generate_command"], "process": actual,
                     "started_epoch": epoch(row.get("started_utc"), now), "files": files,
                     "first_record": first_record(paths["prediction"]) or first_record(paths["runtime"]),
                     "worker_status": wstate.get("status"), "worker_generation_exit_code": wstate.get("generation_exit_code")})
    # An exit/new start during this short observation is normal: the canonical
    # transition or the bounded missing-child grace resolves it on the next tick.
    stable = file_stat(STATE)
    publication = read_json(PUBLISH)
    deps, dependency_pids = dependency_snapshot()
    marker_path = PHASE2 / "legacy_streamwait_continuation.finished.json"
    marker = read_json(marker_path) if marker_path.exists() else {}
    live = []
    if state.get("status") == "complete":
        for pid in dependency_pids:
            current = proc(pid)
            if current["alive"] and any(str(BASE) in arg for arg in current.get("argv", [])):
                live.append(pid)
        registered = config["phase2_supervisor"]
        if identity_matches(supervisor, registered, registered["argv"]):
            live.append(supervisor["pid"])
        for key, row in state.get("jobs", {}).items():
            current = proc(row.get("pid"))
            expected = {"pid": row.get("pid"), "start_ticks": row.get("process_identity", {}).get("start_ticks")}
            if identity_matches(current, expected, config["jobs"][key]["generate_command"]):
                live.append(row["pid"])
    followup = read_json(FOLLOWUP) if FOLLOWUP.exists() else {}
    view = {"phase2": state, "publisher": publication, "dependencies": deps,
            "phase2_marker_success": marker.get("success") is True,
            "live_registered_processes": sorted(set(live))}
    return {"observed_utc": now_iso(now), "observed_epoch": now, "phase2_status": state.get("status"),
            "phase2_updated_utc": state.get("updated_utc"), "phase2_file": stable,
            "phase2_jobs": {status: sum(row.get("status") == status for row in state.get("jobs", {}).values())
                            for status in sorted({row.get("status", "unknown") for row in state.get("jobs", {}).values()})},
            "supervisor": supervisor, "jobs": jobs, "dependencies": deps,
            "phase2_dependency": {k: state.get("dependency", {}).get(k) for k in ("ready", "reason")},
            "publisher": {"status": publication.get("status"), "published": len(publication.get("published_groups", {})),
                          "expected": len(publication.get("expected_groups", [])), "error": publication.get("error")},
            "registered_batch_ready_for_inspector": batch_ready(view, config["expected_groups"], config["jobs"]),
            "followup_status": followup.get("status"), "followup_hold": followup.get("status") != "complete",
            "read_duration_seconds": round(time.monotonic() - start, 4)}


def assess(snapshot, config, previous):
    issues = []
    status = snapshot["phase2_status"]
    supervisor = snapshot["supervisor"]
    if status in BAD:
        issues.append(make_issue("supervisor_failed", "Canonical phase2 status is failed; inherited error strings alone are ignored"))
    elif status != "complete":
        expected = config["phase2_supervisor"]
        if not supervisor.get("alive"):
            issues.append(make_issue("supervisor_missing", "Non-complete phase2 supervisor is absent"))
        elif not identity_matches(supervisor, expected, expected["argv"]):
            issues.append(make_issue("supervisor_identity_differs", "Pinned CPU supervisor start ticks/argv differ"))
        elif supervisor.get("state") == "T":
            issues.append(make_issue("supervisor_stopped", "CPU supervisor is stopped; check independent slot-reservation authorization"))
    memos = {}
    for job in snapshot["jobs"]:
        found, memo = assess_job(job, previous.get("jobs", {}).get(job["id"], {}), snapshot["observed_epoch"])
        issues.extend(found)
        memos[job["id"]] = memo
    no_active = not any(job["status"] in ACTIVE for job in snapshot["jobs"])
    ready_dependencies = bool(snapshot.get("dependencies")) and all(snapshot["dependencies"].values())
    if (status in {"running", "waiting_for_GPU_slot"} and no_active and ready_dependencies
            and snapshot.get("phase2_jobs", {}).get("pending", 0)):
        since = previous.get("jobs", {}).get("__queue_idle__", {}).get("since_epoch", snapshot["observed_epoch"])
        memos["__queue_idle__"] = {"since_epoch": since}
        if snapshot["observed_epoch"] - since >= STALL_SECONDS:
            issues.append(make_issue("pending_without_worker", "Pending jobs but no active worker for ten minutes; inspector must verify GPU/resource gates"))
    if snapshot["publisher"].get("error"):
        issues.append(make_issue("publisher_failed", str(snapshot["publisher"]["error"])))
    return issues, memos


def events(issues, ready, previous):
    current = {x["key"]: x for x in issues}
    old = previous.get("issues", {})
    result = []
    new = [current[k] for k in current.keys() - old.keys()]
    cleared = [old[k] for k in old.keys() - current.keys()]
    if new:
        result.append({"kind": "attention", "issues": new})
    if cleared:
        result.append({"kind": "recovered_observation", "issues": cleared})
    if ready and not previous.get("batch_ready_seen"):
        result.append({"kind": "registered_batch_ready"})
    return result, current


def invoke(argv, timeout=10):
    try:
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        return {"returncode": result.returncode, "output": (result.stdout + result.stderr)[-1000:].strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"returncode": -1, "output": str(error)}


def wake_inspector():
    status = invoke(["/usr/bin/systemctl", "--user", "is-active", HOURLY_SERVICE])
    if status["output"] in {"active", "activating", "reloading"}:
        return {"already_active": True, "status": status["output"]}
    return invoke(["/usr/bin/systemctl", "--user", "--no-block", "start", HOURLY_SERVICE])


def notify(event):
    kind = event["kind"]
    if kind == "attention":
        title = "EADP 实验巡检发现异常"
        body = "; ".join(x["key"] for x in event["issues"])
        body += "。已请求现有巡检；轻量 watcher 不重启 GPU。"
    elif kind == "recovered_observation":
        title = "EADP 实验异常观测已解除"
        body = "; ".join(x["key"] for x in event["issues"]) + "。继续监控并由现有巡检验证。"
    else:
        title = "EADP 登记批次已具备完成证据"
        body = "75 臂/55 组状态已齐，已请求完整巡检。后续审计任务仍由原监控继续。"
    return invoke(["/usr/bin/notify-send", "--app-name=EADP轻量巡检", "--urgency=" +
                   ("critical" if kind == "attention" else "normal"), title, body])


def run_once(dry_run=False):
    config = read_json(CONFIG, 1024 * 1024)
    old_path = PRIVATE / "watch_state.json"
    previous = read_json(old_path, 1024 * 1024) if old_path.exists() else {}
    try:
        snapshot = collect(config)
        issues, memos = assess(snapshot, config, previous)
    except Exception as error:
        snapshot = {"observed_utc": now_iso(), "observed_epoch": time.time(), "read_error": str(error),
                    "registered_batch_ready_for_inspector": False, "followup_hold": True}
        issues = [make_issue("watch_read_failed", str(error))]
        memos = previous.get("jobs", {})
    changed, issue_map = events(issues, snapshot["registered_batch_ready_for_inspector"], previous)
    now = snapshot["observed_epoch"]
    memo = {"updated_utc": now_iso(now), "jobs": memos, "issues": issue_map,
            "batch_ready_seen": previous.get("batch_ready_seen", False) or snapshot["registered_batch_ready_for_inspector"],
            "last_inspector_wake_epoch": previous.get("last_inspector_wake_epoch", 0)}
    snapshot.update(issues=issues, events=changed, actions=[])
    # Attention is retried at most every ten minutes if an earlier inspection
    # finished without resolving it. Notifications remain change-only.
    should_wake = (issues and now - memo["last_inspector_wake_epoch"] >= STALL_SECONDS) or any(
        event["kind"] == "registered_batch_ready" for event in changed)
    if not dry_run:
        if should_wake:
            snapshot["actions"].append({"inspector_wake": wake_inspector()})
            memo["last_inspector_wake_epoch"] = now
        for event in changed:
            snapshot["actions"].append({"notification": notify(event), "kind": event["kind"]})
        atomic(PRIVATE / "latest_snapshot.json", snapshot)
        atomic(old_path, memo)
        if changed:
            with (PRIVATE / "events.jsonl").open("a") as handle:
                handle.write(json.dumps({"utc": now_iso(now), "events": changed}, ensure_ascii=False) + "\n")
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Register only the existing 32 jobs and supervisor identity")
    parser.add_argument("--dry-run", action="store_true", help="Read/policy only; no files, service wake or notifications")
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps({"config": str(CONFIG), "jobs": len(prepare_config()["jobs"])}, ensure_ascii=False))
        return
    if args.dry_run:
        print(json.dumps(run_once(True), ensure_ascii=False, indent=2))
        return
    PRIVATE.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (PRIVATE / "watch.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        run_once()


if __name__ == "__main__":
    main()

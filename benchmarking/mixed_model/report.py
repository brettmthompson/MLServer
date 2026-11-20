"""Summarize a mixed-model run and correlate optional worker traces."""

import argparse
import json
import math
import sqlite3
from pathlib import Path


def summarize_window(
    db: sqlite3.Connection, model: str, version: str, start: float, end: float
) -> dict:
    selector = "FROM attempt WHERE phase='measure' AND model=? AND version=? "
    args = (model, version)
    cohort = selector + "AND due>=? AND due<? "
    params = (*args, start, end)
    counts = dict(
        db.execute("SELECT outcome,COUNT(*) " + cohort + "GROUP BY outcome", params)
    )
    elapsed = end - start
    arrivals = sum(counts.values())
    correct_in_window = db.execute(
        "SELECT COUNT(*) "
        + selector
        + "AND outcome='correct' AND finished>=? AND finished<?",
        params,
    ).fetchone()[0]
    timings = cohort + "AND finished IS NOT NULL "
    total = db.execute("SELECT COUNT(*) " + timings, params).fetchone()[0]
    latency = {}
    for pct in (50, 95, 99):
        if total:
            latency[f"p{pct}"] = db.execute(
                "SELECT 1000*(finished-submitted) "
                + timings
                + "ORDER BY 1 LIMIT 1 OFFSET ?",
                (*params, math.ceil(pct * total / 100) - 1),
            ).fetchone()[0]
    return dict(
        offered=arrivals,
        submitted=total,
        # A V2 inference response proves the call reached serving. A lost or
        # timed-out response says nothing about whether the worker ran it.
        accepted_lower_bound_from_replies=counts.get("correct", 0)
        + counts.get("wrong_output", 0),
        outcomes=counts,
        client_latency_ms=latency,
        correct_responses_in_window=correct_in_window,
        correct_responses_per_s=correct_in_window / elapsed if elapsed else None,
    )


def read_worker_log(db: sqlite3.Connection, path: Path, run_id: str) -> list[dict]:
    # Keep trace correlation off the serving path and on disk for long runs.
    db.execute("DROP TABLE IF EXISTS temp.worker_event")
    db.execute(
        "CREATE TEMP TABLE worker_event(id TEXT, message TEXT, pid INTEGER, "
        "event TEXT, ns INTEGER, PRIMARY KEY(message,pid,event))"
    )
    with path.open() as stream:
        for line in stream:
            _, marker, payload = line.partition("worker_execution ")
            if marker:
                event = json.loads(payload)
                if str(event["request"]).startswith(run_id + "."):
                    db.execute(
                        "INSERT OR IGNORE INTO worker_event VALUES(?,?,?,?,?)",
                        (
                            event["request"],
                            event["message"],
                            event["pid"],
                            event["event"],
                            event["ns"],
                        ),
                    )
    db.execute("CREATE INDEX temp.worker_id ON worker_event(id)")
    return [
        dict(
            model=model,
            version=version,
            client_outcome=outcome,
            worker_event=event,
            observations=n,
        )
        for model, version, outcome, event, n in db.execute(
            "SELECT a.model,a.version,a.outcome,e.event,COUNT(*) FROM worker_event e "
            "JOIN attempt a ON a.id=e.id WHERE a.phase='measure' GROUP BY 1,2,3,4"
        )
    ]


def summary(db: sqlite3.Connection, worker_log: Path | None = None) -> dict:
    run_id, address, config = db.execute("SELECT * FROM run").fetchone()
    config = json.loads(config)
    duration = config["duration_s"]
    windows = [("all", 0, duration)]
    actions = []
    for name, op, due, start, finish, status, expected in db.execute(
        "SELECT model,operation,due,submitted,finished,status,expected FROM lifecycle "
        "WHERE phase='measure' ORDER BY submitted"
    ):
        actions.append(
            dict(
                model=name,
                operation=op,
                due_s=due,
                start_s=start,
                end_s=finish,
                status=status,
                expected=expected,
            )
        )
        label = f"{len(actions)}:{op}:{name}"
        for part, left, right in (
            ("before", 0, start),
            ("during", start, finish),
            ("after", finish, duration),
        ):
            if left < min(right, duration):
                windows.append((f"{label}:{part}", left, min(right, duration)))
    result = dict(
        run=run_id,
        address=address,
        deployment=config["deployment"],
        actions=actions,
        models=[
            dict(
                model=m["name"],
                version=m["version"],
                windows={
                    name: summarize_window(db, m["name"], m["version"], a, b)
                    for name, a, b in windows
                },
            )
            for m in config["models"]
        ],
        memory=[
            dict(source=s, subject=i, min_bytes=lo, max_bytes=hi, samples=n)
            for s, i, lo, hi, n in db.execute(
                "SELECT source,subject,MIN(bytes),MAX(bytes),COUNT(*) "
                "FROM resource GROUP BY 1,2"
            )
        ],
    )
    if worker_log:
        result["observed_worker_events"] = read_worker_log(db, worker_log, run_id)
        result["worker_calls"] = [
            dict(
                model=name,
                version=version,
                started=started,
                observed_terminal=terminal,
                mean_worker_call_ms=mean_ms,
            )
            for name, version, started, terminal, mean_ms in db.execute(
                "SELECT a.model,a.version,COUNT(*),COUNT(t.ns),AVG(t.ns-s.ns)/1e6 "
                "FROM worker_event s JOIN attempt a ON a.id=s.id "
                "LEFT JOIN worker_event t ON t.message=s.message AND t.pid=s.pid "
                "AND t.event IN ('returned','raised') "
                "WHERE s.event='started' AND a.phase='measure' GROUP BY 1,2"
            )
        ]
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("database", type=Path)
    cli.add_argument("--worker-log", type=Path)
    args = cli.parse_args()
    with sqlite3.connect(
        f"{args.database.resolve().as_uri()}?mode=ro", uri=True
    ) as connection:
        print(json.dumps(summary(connection, args.worker_log), indent=2))

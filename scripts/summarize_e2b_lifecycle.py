#!/usr/bin/env python3
"""Summarize E2B lifecycle events with rollout JSONL as a fallback.

Backend logs are authoritative for create/close lifecycle events. Ray may keep
those logs on remote nodes, so completed tool exchanges embedded in rollout
JSONL are also counted. In the current E2B P0 implementation each completed
tool response implies that a sandbox was created and the command returned.
Sandbox close status is never inferred from rollout data.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    return parser.parse_args()


def read_lifecycle_log(run_root: Path) -> dict:
    path = run_root / "logs" / "e2b-lifecycle.log"
    text = path.read_text(errors="replace") if path.exists() else ""
    lines = text.splitlines()

    exit_codes = Counter(re.findall(r"exit_code=(None|-?\d+)", text))
    return {
        "create_success": sum("sandbox created backend=e2b" in line for line in lines),
        "create_failure": sum("E2B create failed" in line for line in lines),
        "execute_returned": sum("sandbox execute tool_call_id=" in line for line in lines),
        "execute_failure": sum("E2B execute failed" in line for line in lines),
        "execute_exit_0": exit_codes.get("0", 0),
        "execute_exit_nonzero": sum(
            count for code, count in exit_codes.items() if code not in {"0", "None"}
        ),
        "execute_exit_none": exit_codes.get("None", 0),
        "timeout_count": sum(re.search(r"timed_out=(True|1)", line) is not None for line in lines),
        "close_success": sum(
            "sandbox closed backend=e2b" in line
            and re.search(r"success=(1|True)", line) is not None
            for line in lines
        ),
        "close_failure": sum(
            "sandbox close failed" in line
            or (
                "sandbox closed backend=e2b" in line
                and re.search(r"success=(0|False)", line) is not None
            )
            for line in lines
        ),
        "exit_codes": dict(exit_codes),
        "log_file_present": path.exists(),
        "log_lines_matched": len(lines),
    }


def read_rollout_evidence(run_root: Path) -> dict:
    tool_calls = 0
    tool_responses = 0
    trajectories = 0
    trajectories_with_tool_calls = 0
    exit_codes: Counter[str] = Counter()
    timeout_count = 0
    malformed_json_lines = 0
    files = sorted((run_root / "rollouts").glob("*.jsonl"))

    for path in files:
        with path.open(errors="replace") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed_json_lines += 1
                    continue

                output = str(row.get("output", ""))
                calls = output.count("<tool_call>")
                responses = output.count("<tool_response>")

                trajectories += 1
                tool_calls += calls
                tool_responses += responses
                trajectories_with_tool_calls += int(calls > 0)
                exit_codes.update(re.findall(r"\[exit_code\]\s*(None|-?\d+)", output))
                timeout_count += len(re.findall(r"\[status\].*?timed out", output, re.IGNORECASE))

    return {
        "files": [str(path) for path in files],
        "trajectories": trajectories,
        "trajectories_with_tool_calls": trajectories_with_tool_calls,
        "tool_calls": tool_calls,
        "tool_responses": tool_responses,
        "unanswered_tool_calls": max(tool_calls - tool_responses, 0),
        "execute_exit_0": exit_codes.get("0", 0),
        "execute_exit_nonzero": sum(
            count for code, count in exit_codes.items() if code not in {"0", "None"}
        ),
        "execute_exit_none": exit_codes.get("None", 0),
        "timeout_count": timeout_count,
        "exit_codes": dict(exit_codes),
        "malformed_json_lines": malformed_json_lines,
    }


def build_summary(log: dict, rollout: dict) -> dict:
    # Do not add the two sources: the same execution may be present in both.
    # max() preserves authoritative log counts while filling gaps from rollout
    # evidence when Ray logs stayed on another physical node.
    execute_returned = max(log["execute_returned"], rollout["tool_responses"])
    create_success = max(log["create_success"], rollout["tool_responses"])

    summary = {
        "create_success": create_success,
        "create_failure": log["create_failure"],
        "execute_returned": execute_returned,
        "execute_failure": log["execute_failure"],
        "execute_exit_0": max(log["execute_exit_0"], rollout["execute_exit_0"]),
        "execute_exit_nonzero": max(log["execute_exit_nonzero"], rollout["execute_exit_nonzero"]),
        "execute_exit_none": max(log["execute_exit_none"], rollout["execute_exit_none"]),
        "timeout_count": max(log["timeout_count"], rollout["timeout_count"]),
        "close_success": log["close_success"],
        "close_failure": log["close_failure"],
        "close_status_observed": (log["close_success"] + log["close_failure"]) > 0,
        "create_success_inferred_from_rollouts": max(
            rollout["tool_responses"] - log["create_success"], 0
        ),
        "execute_returned_inferred_from_rollouts": max(
            rollout["tool_responses"] - log["execute_returned"], 0
        ),
        "rollout_tool_calls": rollout["tool_calls"],
        "rollout_unanswered_tool_calls": rollout["unanswered_tool_calls"],
        "lifecycle_log": log,
        "rollout_evidence": rollout,
    }

    create_total = summary["create_success"] + summary["create_failure"]
    execute_total = summary["execute_returned"] + summary["execute_failure"]
    summary["create_success_rate"] = (
        summary["create_success"] / create_total if create_total else None
    )
    summary["execute_success_rate"] = (
        summary["execute_exit_0"] / execute_total if execute_total else None
    )
    summary["close_completion_rate"] = (
        summary["close_success"] / summary["create_success"]
        if summary["create_success"] and summary["close_status_observed"]
        else None
    )
    return summary


def write_outputs(run_root: Path, summary: dict) -> None:
    metrics_dir = run_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    json_path = metrics_dir / "e2b-lifecycle-summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    env_values = {
        "E2B_CREATE_SUCCESS": summary["create_success"],
        "E2B_CREATE_FAILURE": summary["create_failure"],
        "E2B_EXECUTE_RETURNED": summary["execute_returned"],
        "E2B_EXECUTE_FAILURE": summary["execute_failure"],
        "E2B_EXECUTE_EXIT_0": summary["execute_exit_0"],
        "E2B_EXECUTE_EXIT_NONZERO": summary["execute_exit_nonzero"],
        "E2B_TIMEOUT_COUNT": summary["timeout_count"],
        "E2B_CLOSE_SUCCESS": summary["close_success"],
        "E2B_CLOSE_FAILURE": summary["close_failure"],
        "E2B_CLOSE_STATUS_OBSERVED": int(summary["close_status_observed"]),
        "E2B_ROLLOUT_TOOL_CALLS": summary["rollout_tool_calls"],
        "E2B_ROLLOUT_UNANSWERED_TOOL_CALLS": summary["rollout_unanswered_tool_calls"],
    }
    env_path = metrics_dir / "e2b-lifecycle-summary.env"
    env_path.write_text("".join(f"{key}={value}\n" for key, value in env_values.items()))


def main() -> None:
    args = parse_args()
    log = read_lifecycle_log(args.run_root)
    rollout = read_rollout_evidence(args.run_root)
    summary = build_summary(log, rollout)
    write_outputs(args.run_root, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

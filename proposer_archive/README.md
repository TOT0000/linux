# UAV Harness Logs Archive

This archive stores experiment outputs for UAV mission runs.

## Root files
- task_runs.xlsx: aggregated run summary
- task_runs_runtime_trace.jsonl: aggregated runtime snapshots
- task_runs_planning_trace.jsonl: aggregated planning traces
- task_runs_debug.jsonl: aggregated debug summaries

## Per-run files
Each run is stored under:
- runs/run_<id>/

Typical contents:
- run_<id>_runtime_trace.jsonl
- run_<id>_planning_trace.jsonl
- metadata.json

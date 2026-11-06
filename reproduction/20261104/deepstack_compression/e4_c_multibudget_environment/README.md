# 32-image timing: host background observations

This is a proposed low-frequency background observer, not yet run on the timing
job. `watch.py` runs `ps` every 60 seconds or more, with a 10-second query timeout,
records each query's UTC and monotonic boundaries plus the time namespace, and
fsyncs every event. It writes a second event after the first fsync; that second
event's write and fsync have no measured end. `ps` reports process lifetime CPU
percentages, not continuous utilization. External I/O, scheduler effects and
post-query residual effects are not captured.

Start it under the existing **timed** supervisor before starting the model run,
wait for the first complete snapshot/log pair, and stop it with a newly created
marker only after the model's supervisor and GPU container have exited. Its
duration cap is 7500 seconds, so coordinate start promptly and reject a run where
the observer expires before the model ends. The observer and model must use the
same time namespace. Record any manual checks separately with monotonic bounds;
the model supervisor's GPU health snapshots remain at least hourly. The timed
supervisor of the observer also takes health snapshots: keep and annotate those
extra windows rather than assuming they are free.

After the run, verify the observer supervisor completed successfully with no
process group, then check its frozen source, nonempty retained log and stop
marker. `validate.validate` rejects malformed, failed, truncated, wrong-namespace
or duration-expired observations. Require its first query to begin before the
model start and `watch_end.end_ns` to be at or after the model end. Once formal
and independent model acceptance permits reading the full journal,
`validate.overlaps` can annotate all 1520 retained jobs against both query and
query-plus-first-fsync windows; compare the observer supervisor's health windows
and model supervisor's hourly health windows separately. Missing/truncated logs,
an absent watch end, and namespace mismatches are unknown coverage, never zero
overlap. The second fsync and non-recorded effects remain unknown for every job;
no measurements are removed or treated as clean solely from no intersection.
The formal analysis source must bind these artifacts and both supervisor exits.

CPU synthetic tests: `python -m unittest -v test_validate`. The first local test
attempt failed at import due to a missing parenthesis; it was fixed before the
successful run. No real observer, GPU workload or model was started.

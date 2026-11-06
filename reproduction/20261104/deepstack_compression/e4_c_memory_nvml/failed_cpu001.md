`monitor_cpu001` failed before tests because naming the new file `sampler.py`
shadowed the existing `e4_c_memory/sampler.py` when `sampler_io` imported it,
causing a circular import. The failed source is retained as `sampler.py`;
the corrected candidate is `real_sampler.py` and imports the existing sampler
under its original name in a fresh process. No GPU or model was used.

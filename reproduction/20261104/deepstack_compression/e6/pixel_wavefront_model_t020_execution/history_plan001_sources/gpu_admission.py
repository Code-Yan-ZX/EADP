"""No GPU admission until execution plan, real preflight and independent review."""
def require_gpu():
 raise RuntimeError('GPU locked: six-arm execution plan, preflight and independent review missing')

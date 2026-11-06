"""Scoped C-SKIP primitive for Qwen3-VL's dedicated DeepStack mergers.

Only the three merger instances are patched. The visual forward, backbone
blocks, and main merger remain untouched. The stock vision forward still calls
the patched modules and returns a three-slot list containing None. An E4 caller
must use a DS-off prefill that never indexes or injects that list.

Patching module instances is process-local but not thread-safe. Do not share a
model between concurrent inferences while the context is active.
"""
from contextlib import AbstractContextManager


class SkipDeepStackMergers(AbstractContextManager):
    """Temporarily skip DS merger forward computation; restore on every exit."""

    _OWNER = '_eadp_e4_ds_skip_owner'

    def __init__(self, visual, *, audit_calls=False):
        self.visual = visual
        self.audit_calls = bool(audit_calls)
        self.skipped_calls = [0, 0, 0]
        self._saved = []
        self._entered = False

    def __enter__(self):
        if self._entered or self._saved:
            raise RuntimeError('skip context cannot be re-entered')
        mergers = list(self.visual.deepstack_merger_list)
        positions = list(self.visual.deepstack_visual_indexes)
        if len(mergers) != 3 or len({id(x) for x in mergers}) != 3:
            raise ValueError('expected three distinct DS merger modules')
        if len(positions) != 3 or len(set(positions)) != 3:
            raise ValueError('expected three distinct DS layer indexes')
        if any(not isinstance(i, int) or i < 0 or i >= len(self.visual.blocks) for i in positions):
            raise ValueError('DS layer index outside visual backbone')
        protected = [self.visual.merger, *self.visual.blocks]
        protected += [getattr(self.visual, name) for name in ('patch_embed', 'pos_embed', 'rotary_pos_emb')
                      if hasattr(self.visual, name)]
        protected_ids = set()
        for module in protected:
            protected_ids.add(id(module))
            if callable(getattr(module, 'modules', None)):
                protected_ids.update(id(child) for child in module.modules())
        if any(id(x) in protected_ids for x in mergers):
            raise ValueError('DS merger aliases visual backbone or main merger')
        if any(self._OWNER in m.__dict__ for m in mergers):
            raise RuntimeError('another DS skip context is active')
        if any(not callable(getattr(m, 'forward', None)) for m in mergers):
            raise ValueError('DS merger is not a callable module')
        try:
            for index, module in enumerate(mergers):
                had_instance_forward = 'forward' in module.__dict__
                old_instance_forward = module.__dict__.get('forward')
                self._saved.append((module, had_instance_forward, old_instance_forward))
                module.__dict__[self._OWNER] = self
                if self.audit_calls:
                    def skip(*args, _i=index, **kwargs):
                        self.skipped_calls[_i] += 1
                        return None
                else:
                    def skip(*args, **kwargs):
                        return None
                module.forward = skip
            self._entered = True
            return self
        except BaseException:
            self._restore()
            raise

    def _restore(self):
        problems = []
        while self._saved:
            module, had_instance_forward, old_instance_forward = self._saved.pop()
            try:
                if had_instance_forward:
                    module.forward = old_instance_forward
                elif 'forward' in module.__dict__:
                    del module.forward
            except BaseException as exc:
                problems.append(exc)
            finally:
                if module.__dict__.get(self._OWNER) is self:
                    del module.__dict__[self._OWNER]
        self._entered = False
        if problems:
            raise RuntimeError('failed to restore one or more DS merger forwards') from problems[0]

    def __exit__(self, exc_type, exc, traceback):
        self._restore()
        return False


def require_skip_visual_result(result):
    """Reject an incompatible vision return; return the untouched primary value."""
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise ValueError('expected (main_features, three DS slots)')
    primary, streams = result
    if primary is None or not isinstance(streams, (tuple, list)) or len(streams) != 3:
        raise ValueError('expected primary features and exactly three DS slots')
    if any(x is not None for x in streams):
        raise ValueError('DS merger unexpectedly returned a feature')
    return primary

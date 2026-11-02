"""Pass-through runtime observation for the author LLaVA model.

No model imports or device work at import time. Hooks retain references only;
reading/copying tensor contents is the caller's responsibility after generate.
"""

class Observer:
    names = ('encode_images', 'sim_cross', 'greed_select', 'prepare_inputs_labels_for_multimodal')

    def __init__(self, model):
        self.model = model
        self.original = {}
        self.handle = None
        self.reset()

    def reset(self):
        self.arrays = {}
        self.calls = dict(encode=0, cross=0, greedy=0, prepare=0, clip=0)
        self.errors = []
        self.encode_texts = []

    def note(self, label, fn):
        try:
            fn()
        except Exception as exc:
            self.errors.append('%s: %s: %s' % (label, type(exc).__name__, exc))

    def install(self):
        # Installation, including a failed partial installation, is reversible.
        try:
            import torch.nn.modules.module as torch_module
            tower = self.model.get_vision_tower().text_tower
            if torch_module._global_forward_pre_hooks or getattr(tower, '_forward_pre_hooks', {}):
                raise RuntimeError('text tower input may be modified by another forward pre-hook')
            for name in self.names:
                self.original[name] = getattr(self.model, name)
            native = self.original

            def encode(*args, **kwargs):
                self.calls['encode'] += 1
                self.note('encode arguments', lambda: self.encode_texts.append(kwargs.get('texts')))
                value = native['encode_images'](*args, **kwargs)
                self.note('encode return', lambda: self.arrays.update(projected_fp16=value[0], final_mask=value[1]))
                return value

            def cross(*args, **kwargs):
                self.calls['cross'] += 1
                self.note('cross arguments', lambda: self.arrays.update(text_global=args[0], text_dense=args[1], image_embeds=args[2]))
                value = native['sim_cross'](*args, **kwargs)
                self.note('cross return', lambda: self.arrays.update(global_sim=value[0], dense_sim=value[1]))
                return value

            def greedy(*args, **kwargs):
                self.calls['greedy'] += 1
                self.note('greedy arguments', lambda: self.arrays.update(importance=args[0], visual_sim=args[1]))
                value = native['greed_select'](*args, **kwargs)
                self.note('greedy return', lambda: self.arrays.update(internal_ids=value[0], quotas=value[1]))
                return value

            def prepare(*args, **kwargs):
                value = native['prepare_inputs_labels_for_multimodal'](*args, **kwargs)
                if isinstance(value, tuple) and len(value) == 7:
                    self.calls['prepare'] += 1
                    self.note('prepare return', lambda: self.arrays.update(actual_input_ids=args[0], prepared_embeddings_fp16=value[4]))
                return value

            for name, fn in zip(self.names, (encode, cross, greedy, prepare)):
                setattr(self.model, name, fn)

            def clip_hook(module, args, kwargs):
                self.calls['clip'] += 1
                self.note('CLIP text tower inputs', lambda: self.arrays.update(clip_input_ids=kwargs['input_ids'], clip_attention_mask=kwargs['attention_mask']))
                # Returning None leaves the actual invocation unchanged.
                return None

            self.handle = tower.register_forward_pre_hook(clip_hook, with_kwargs=True)
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        for name, original in self.original.items():
            setattr(self.model, name, original)
        self.original.clear()

    def __enter__(self):
        return self.install()

    def __exit__(self, *unused):
        self.close()

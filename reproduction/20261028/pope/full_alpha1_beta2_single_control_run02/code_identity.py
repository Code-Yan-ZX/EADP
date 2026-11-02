"""Stable structural identity for the original Python 3.10 eval_model code."""
import hashlib
import json
import types

FIELDS = ('co_argcount', 'co_posonlyargcount', 'co_kwonlyargcount', 'co_nlocals',
          'co_stacksize', 'co_flags', 'co_code', 'co_consts', 'co_names',
          'co_varnames', 'co_freevars', 'co_cellvars', 'co_filename', 'co_name',
          'co_firstlineno', 'co_linetable')


def canonical(value):
    if isinstance(value, types.CodeType):
        return ['code', [[name, canonical(getattr(value, name))] for name in FIELDS]]
    if value is None: return ['none']
    if type(value) is bool: return ['bool', value]
    if type(value) is int: return ['int', str(value)]
    if type(value) is str: return ['str', value]
    if type(value) is bytes: return ['bytes', value.hex()]
    if type(value) is tuple: return ['tuple', [canonical(x) for x in value]]
    if type(value) is frozenset:
        items = [canonical(x) for x in value]
        return ['frozenset', sorted(items, key=lambda x: json.dumps(x, ensure_ascii=True, separators=(',', ':')))]
    if type(value) is float: return ['float', value.hex()]
    if type(value) is complex: return ['complex', value.real.hex(), value.imag.hex()]
    if value is Ellipsis: return ['ellipsis']
    raise TypeError('Unsupported code constant: ' + repr(type(value)))


def fingerprint(code):
    encoded = json.dumps(canonical(code), ensure_ascii=True, separators=(',', ':')).encode('ascii')
    return hashlib.sha256(encoded).hexdigest()


def source_code(entry):
    module = compile(entry.read_bytes(), str(entry), 'exec', dont_inherit=True, optimize=0)
    candidates = [x for x in module.co_consts if isinstance(x, types.CodeType) and x.co_name == 'eval_model']
    assert len(candidates) == 1
    return candidates[0]

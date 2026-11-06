"""Count actual selector outputs using Python shape metadata; adds observer overhead.

No tensor contents are copied and this helper performs no explicit CUDA sync.
The underlying selector can still synchronize. This wrapper is part of the
measured candidate and is not claimed to be free or equivalent without review.
"""
from contextlib import contextmanager

@contextmanager
def count_calls(module,attribute,kind,expected_count):
    if kind not in ('pixel','eadp'):raise ValueError('unknown selector kind')
    original=getattr(module,attribute);counts=[]
    def observed(*args,**kwargs):
        result=original(*args,**kwargs)
        if kind=='pixel':
            if not isinstance(result,list) or len(result)!=1:raise ValueError('expected one image')
            value=len(result[0])
        else:
            if not isinstance(result,tuple) or len(result)!=2 or len(result[0].shape)!=2 or result[0].shape[0]!=1:raise ValueError('unexpected EADP selector output')
            value=result[0].shape[1]
        if type(value) is not int or not 1<=value<=1024:raise ValueError('invalid observed K')
        counts.append(value)
        return result
    setattr(module,attribute,observed)
    try:
        yield counts
        if len(counts)!=expected_count:raise ValueError('unexpected selector call count')
    finally:
        setattr(module,attribute,original)

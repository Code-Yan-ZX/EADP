"""CPU-only per-question and absolute-deadline stop contract.

The launcher must also enforce independent process/container hard timeouts;
cooperative checkpoints alone cannot interrupt an in-flight model call.
"""
import time
class Deadline:
    def __init__(self,absolute_end,question_seconds,*,wall=time.time,clock=time.monotonic):
        if type(absolute_end) not in (int,float) or type(question_seconds) not in (int,float) or question_seconds<=0:
            raise ValueError('explicit positive limits required')
        self.end=absolute_end;self.limit=question_seconds;self.wall=wall;self.clock=clock;self.start=clock()
    def check(self):
        if self.wall()>=self.end or self.clock()-self.start>=self.limit:
            raise TimeoutError('oracle deadline or per-question limit reached')

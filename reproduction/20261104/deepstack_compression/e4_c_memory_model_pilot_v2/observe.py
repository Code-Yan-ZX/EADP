"""Pure orchestration primitive, no launcher or GPU authorization.

Caller must have independently validated the pre-CUDA PID1 handshake and
same-image runtime before invoking this helper.
"""
def run(torch,load_model,generate,record,monotonic_ns):
 if torch.cuda.is_initialized():raise ValueError('measurement must begin before CUDA initialization')
 def mark(stage,**extra):record({'event':stage,'monotonic_ns':monotonic_ns(),**extra})
 try:
  mark('model_load_start');model=load_model();torch.cuda.synchronize();mark('model_load_end',allocated_peak_bytes=int(torch.cuda.max_memory_allocated()),reserved_peak_bytes=int(torch.cuda.max_memory_reserved()))
  torch.cuda.reset_peak_memory_stats();mark('request_start')
  # generate must persist returned raw IDs on return before any decoding or validation.
  response=generate(model);torch.cuda.synchronize();mark('request_end',allocated_peak_bytes=int(torch.cuda.max_memory_allocated()),reserved_peak_bytes=int(torch.cuda.max_memory_reserved()))
  return model,response
 except BaseException as e:
  mark('failure',error_type=type(e).__name__,error=str(e));raise

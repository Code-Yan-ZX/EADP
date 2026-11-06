"""Hard lock until a complete executable plan and independent review exist."""
def admit_gpu():
 raise RuntimeError('GPU locked: Pixel NVML executable plan and independent admission not frozen')

"""Pure window annotation. Inputs are normalized only after separate acceptance."""
KINDS=('observer_query','observer_first_fsync','run_supervisor_health','observer_supervisor_health','manual_check')
UNKNOWN=('observer_second_fsync','manual_log_fsync','unobserved_host_activity')

def annotate(start_ns,end_ns,request_namespace,request_clock,groups):
    if (type(start_ns) is not int or type(end_ns) is not int or start_ns<0 or end_ns<start_ns or
        not isinstance(request_namespace,str) or not request_namespace or not isinstance(request_clock,str) or not request_clock):
        raise ValueError('invalid request window or clock identity')
    if set(groups)!=set(KINDS):raise ValueError('required window groups')
    out={}
    for kind in KINDS:
        group=groups[kind]
        if group is None:
            out[kind]={'recorded_overlap':'unknown','reason':'window source unavailable'};continue
        if not isinstance(group,dict) or not isinstance(group.get('windows'),list):raise ValueError('window group schema')
        if group.get('time_namespace')!=request_namespace or group.get('clock')!=request_clock:
            out[kind]={'recorded_overlap':'unknown','reason':'clock or time namespace not verified equal'};continue
        windows=group['windows']
        if not windows:
            out[kind]={'recorded_overlap':'unknown','reason':'no recorded windows'};continue
        for pair in windows:
            if not isinstance(pair,(list,tuple)) or len(pair)!=2 or any(type(x) is not int for x in pair) or pair[0]<0 or pair[1]<pair[0]:raise ValueError('invalid recorded window')
        hits=[list(pair) for pair in windows if pair[0]<pair[1] and pair[0]<=end_ns and start_ns<pair[1]]
        if hits:
            out[kind]={'recorded_overlap':True,'overlapping_windows':hits,'scope':'recorded windows only'}
            continue
        coverage_start,coverage_end=group.get('coverage_start_ns'),group.get('coverage_end_ns')
        if (type(coverage_start) is not int or type(coverage_end) is not int or
            coverage_start>start_ns or coverage_end<end_ns or coverage_start>coverage_end):
            out[kind]={'recorded_overlap':'unknown','reason':'source coverage of complete request not established'}
        else:
            out[kind]={'recorded_overlap':False,'overlapping_windows':[],'scope':'recorded windows only'}
    for kind in UNKNOWN:out[kind]={'recorded_overlap':'unknown','reason':'window not recorded'}
    return out

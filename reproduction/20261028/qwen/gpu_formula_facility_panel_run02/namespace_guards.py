"""Explicit host/container bind aliases; payload-stat checks remain exact."""
from pathlib import Path


def expected_model_stat(plan, path, namespace):
    assert namespace in ('host', 'container')
    guard = plan['model_stat_guards'][path]
    expected = dict(guard['current_full_stat'])
    mapping = plan['model_bind_namespace']
    relative = Path(expected['resolved']).relative_to(mapping['host_resolved_root'])
    declared = str(Path(mapping['destination']) / relative)
    assert guard['container_resolved'] == declared
    assert Path(path).is_relative_to(mapping['destination'])
    if namespace == 'container': expected['resolved'] = declared
    return expected


def check_model_stats(plan, stat_function, namespace):
    for path in plan['model_stat_guards']:
        assert stat_function(path) == expected_model_stat(plan, path, namespace), (path, namespace)


def bindings(command):
    result = []
    for i, value in enumerate(command):
        if value != '--mount': continue
        flags = command[i + 1].split(',')
        fields = dict(f.split('=', 1) for f in flags if '=' in f)
        assert fields['type'] == 'bind'
        result.append(dict(source=fields['src'], destination=fields['dst'], readonly='readonly' in flags))
    return sorted(result, key=lambda item: item['destination'])


def verify_mounts(plan, inspected, gpu):
    assert inspected['Image'] == plan['container_image']
    actual = {item['Destination']: item for item in inspected['Mounts']}
    expected = plan['expected_bind_mounts']
    assert set(actual) == {item['destination'] for item in expected}
    records = []
    for mount in expected:
        value = actual[mount['destination']]
        assert value['Type'] == 'bind' and value['RW'] is (not mount['readonly'])
        assert Path(value['Source']).resolve() == Path(mount['source']).resolve()
        records.append(dict(requested=mount, actual=value, source_resolves_identically=True))
    devices = inspected['HostConfig'].get('Devices') or []
    if gpu:
        assert sorted(item['PathOnHost'] for item in devices) == sorted(
            ['/dev/nvidia0', '/dev/nvidiactl', '/dev/nvidia-uvm', '/dev/nvidia-uvm-tools'])
    else:
        assert devices == []
    return dict(all_requested_bind_sources_and_readonly_flags_verified=True,
                GPU_devices_requested=gpu, actual_devices=devices, bind_records=records)

"""Injectable lifecycle protocol for an eventual isolated memory controller.

This is not a Docker launcher and provides no GPU authority. A future reviewed
OS adapter must supply every observation and persist raw inspect/sampler logs.
"""
def run(ops,verify_ready,verify_samples):
 result={'status':'failed','phase':'initial','container_id':None,'error':None,'cleanup':None}
 container=None;worker_stopped=False;sampler=None;create_attempted=False
 try:
  result['phase']='create';create_attempted=True;container=ops.create_container();result['container_id']=container
  result['phase']='start';ops.start_container(container)
  result['phase']='identity';identity=ops.observe_running(container)
  if identity['container_id']!=container:raise ValueError('container identity mismatch')
  result['phase']='worker_ready';ready=ops.wait_worker_ready(identity)
  result['phase']='sampler';sampler=ops.start_sampler(identity);sample_ready=ops.wait_sampler_ready(sampler)
  result['phase']='release';release=verify_ready(identity,ready,sample_ready['started'],sample_ready['first_sample']);ops.release_worker(identity,release)
  result['phase']='worker_exit';worker_exit=ops.wait_worker(container)
  result['phase']='inspect';inspection=ops.inspect(container)
  if inspection.get('container_id')!=container or inspection.get('running') is not False or inspection.get('pid')!=0 or inspection.get('exit_code')!=0 or worker_exit.get('exit_code')!=0:raise ValueError('worker exit not independently stopped/zero')
  worker_stopped=True;end=ops.publish_stopped(identity,inspection)
  result['phase']='sampler_exit';samples,terminal=ops.wait_sampler(sampler)
  result['phase']='audit';audit=verify_samples(identity,release,samples,terminal,end)
  if audit.get('status')!='passed':raise ValueError('sample audit failed')
  result.update(status='passed',phase='completed',audit=audit)
 except BaseException as exc:
  result['error']={'type':type(exc).__name__,'message':str(exc)}
 finally:
  if container is None and create_attempted:
   try:
    container=ops.recover_owned_container();result['container_id']=container
   except BaseException as exc:result['create_recovery_error']=type(exc).__name__+': '+str(exc)
  if container is not None and not worker_stopped:
   cleanup={'stop_attempted':False,'kill_attempted':False,'stopped_verified':False,'errors':[]}
   try:
    cleanup['stop_attempted']=True;ops.stop_container(container)
   except BaseException as exc:cleanup['errors'].append('stop: '+type(exc).__name__+': '+str(exc))
   try:state=ops.inspect(container)
   except BaseException as exc:state=None;cleanup['errors'].append('inspect: '+type(exc).__name__+': '+str(exc))
   if not state or state.get('container_id')!=container or state.get('running') is not False or state.get('pid')!=0:
    try:cleanup['kill_attempted']=True;ops.kill_container(container)
    except BaseException as exc:cleanup['errors'].append('kill: '+type(exc).__name__+': '+str(exc))
    try:state=ops.inspect(container)
    except BaseException as exc:state=None;cleanup['errors'].append('final inspect: '+type(exc).__name__+': '+str(exc))
   cleanup['stopped_verified']=bool(state and state.get('container_id')==container and state.get('running') is False and state.get('pid')==0)
   result['cleanup']=cleanup
  if sampler is not None and result['status']!='passed':
   try:ops.stop_sampler(sampler)
   except BaseException as exc:result.setdefault('sampler_cleanup_errors',[]).append(type(exc).__name__+': '+str(exc))
  # The concrete adapter must write atomically and fsync, including unknowns.
  ops.persist_result(result)
 return result

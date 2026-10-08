"""CPU-only publication safety fixtures; no real remote or GPU access."""
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

spec=importlib.util.spec_from_file_location('publisher',Path(__file__).with_name('publish_repair_results.py'))
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


class PublisherSafety(unittest.TestCase):
    def setUp(self):
        for name,value in [('diagnostic_metadata',{}),('control_registration',None),
                           ('continuation_metadata',{}),('continuation_registration',None),('legacy_registration',None)]:
            patch=mock.patch.object(p,name,return_value=value)
            patch.start();self.addCleanup(patch.stop)

    def test_frozen_source_drift_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'source.py';path.write_text('original')
            protocol={'source_sha256':{str(path):p.sha(path)}}
            self.assertEqual(p.checked_source_maps(protocol)[str(path)],p.sha(path))
            path.write_text('changed')
            with self.assertRaises(ValueError):p.checked_source_maps(protocol)

    def test_symlink_archived_as_regular_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary);source=directory/'actual.json';source.write_text('{"official":true}')
            link=directory/'problems.json';link.symlink_to(source)
            files={}
            with mock.patch.object(p,'ROOT',directory),mock.patch.object(p,'LOCAL',directory/'publication'):
                p.add_files(files,[link]);package=p.bundle('fixture',files,[])
            with tarfile.open(package['blobs'][0]) as archive:
                member=archive.getmember('problems.json')
                self.assertTrue(member.isfile());self.assertFalse(member.issym())
                self.assertEqual(archive.extractfile(member).read(),source.read_bytes())

    def test_private_agent_events_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'agent.events.json';path.write_text('{}')
            with self.assertRaises(ValueError):p.add_files({},[path])

    def test_bundle_blobs_fit_limit_and_hash(self):
        with tempfile.TemporaryDirectory() as temporary,mock.patch.object(p,'LOCAL',Path(temporary)),mock.patch.object(p,'MAX_BLOB',40):
            package=p.bundle('fixture',{'evidence.json':b'{"complete":true}'},[])
            self.assertGreater(len(package['blobs']),1)
            for entry in package['manifest']['blobs']:
                path=Path(package['directory'])/entry['filename']
                self.assertEqual(p.sha(path),entry['sha256']);self.assertLessEqual(entry['bytes'],40)
            combined=b''.join(Path(path).read_bytes() for path in package['blobs'])
            self.assertEqual(p.digest(combined),package['manifest']['archive_sha256'])

    def test_all_published_skips_benchmark_scoring(self):
        groups=[('complete',None)]
        fake_summary={'rows':[{'output':'notneeded','status':'pending'} for _ in range(28)],'errors':[]}
        monitor=json.dumps({'needs_attention':False,'all_complete':False})
        with mock.patch.object(p,'command',return_value=monitor),mock.patch.object(p,'read',return_value=fake_summary),mock.patch.object(p,'module') as load,mock.patch.object(p,'group_definitions',return_value=groups),mock.patch.object(p,'add_files'),mock.patch.object(p,'validate_pope') as scorer:
            plan=p.prepare({'published_groups':{'complete':{}},'common_audit_published':True})
            self.assertEqual(plan['available_groups'],[]);scorer.assert_not_called()

    def test_attention_blocks_publication(self):
        with mock.patch.object(p,'command',return_value=json.dumps({'needs_attention':True})):
            with self.assertRaises(ValueError):p.prepare({})

    def test_successful_push_reply_loss_is_idempotent(self):
        state={'pending_commit':'new','pending_publication':{'parent_remote_head':'old','local_head':'local',
            'groups':[{'group_id':'pair','identity':'sha','scores':[]}]} }
        with mock.patch.object(p,'remote_head',return_value='new'),mock.patch.object(p,'command') as run,mock.patch.object(p,'atomic'):
            result=p.finish_pending(state)
            run.assert_not_called();self.assertEqual(result['published_groups']['pair']['commit'],'new')
            self.assertIsNone(result['pending_commit'])

    def test_pending_push_retries_same_commit_only(self):
        state={'pending_commit':'new','pending_publication':{'parent_remote_head':'old','local_head':'local','groups':[]}}
        with mock.patch.object(p,'remote_head',side_effect=['old','new']),mock.patch.object(p,'command') as run,mock.patch.object(p,'atomic'):
            p.finish_pending(state)
            self.assertEqual(run.call_args_list[1].args[0],['git','push','origin','new:'+p.REF])

    def test_unknown_remote_advance_is_rejected(self):
        state={'pending_commit':'new','pending_publication':{'parent_remote_head':'old','local_head':'local','groups':[]}}
        with mock.patch.object(p,'remote_head',return_value='foreign'),mock.patch.object(p,'command') as run:
            with self.assertRaises(ValueError):p.finish_pending(state)
            run.assert_not_called()

    def test_unchanged_noop_uses_no_git_and_clears_error(self):
        plan={'available_groups':[],'common_audit':None,'pending_groups':['waiting']}
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary);(directory/'source.py').write_text('same')
            with mock.patch.object(p,'ROOT',directory),mock.patch.object(p,'OUT',directory),mock.patch.object(p,'STATE',directory/'state.json'),mock.patch.object(p,'ALLOWLIST',['source.py']),mock.patch.object(p,'METADATA_ARTIFACTS',[]):
                fingerprint,_,_=p.metadata_snapshot(plan,{})
                state={'metadata_fingerprint':fingerprint,'status':'failed_retryable','error':'network timeout'}
                with mock.patch.object(p,'command') as run:
                    result=p.publish(plan,state)
                run.assert_not_called();self.assertEqual(result['status'],'published')
                self.assertNotIn('error',result);self.assertTrue(result['no_op'])

    def test_source_and_metadata_edits_request_git(self):
        plan={'available_groups':[],'common_audit':None,'pending_groups':[]}
        for name in ('source.py','evidence.json'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as temporary:
                directory=Path(temporary)
                for filename in ('source.py','evidence.json'):(directory/filename).write_text('original')
                with mock.patch.object(p,'ROOT',directory),mock.patch.object(p,'OUT',directory),mock.patch.object(p,'ALLOWLIST',['source.py']),mock.patch.object(p,'METADATA_ARTIFACTS',['evidence.json']):
                    fingerprint,_,_=p.metadata_snapshot(plan,{})
                    (directory/name).write_text('edited')
                    with mock.patch.object(p,'command',side_effect=RuntimeError('Git requested')) as run:
                        with self.assertRaisesRegex(RuntimeError,'Git requested'):
                            p.publish(plan,{'metadata_fingerprint':fingerprint})
                    run.assert_called_once()

    def test_new_group_cannot_use_noop(self):
        package={'group_id':'v15_sqa_FULL','identity':'new','manifest':{'scores':[{'accuracy':70.,'n':2017,'metric':'ScienceQA IMG'}]}}
        plan={'available_groups':[package],'common_audit':None,'pending_groups':[]}
        with mock.patch.object(p,'ALLOWLIST',[]),mock.patch.object(p,'METADATA_ARTIFACTS',[]):
            fingerprint,_,_=p.metadata_snapshot(plan,{})
            with mock.patch.object(p,'command',side_effect=RuntimeError('Git requested')) as run:
                with self.assertRaisesRegex(RuntimeError,'Git requested'):
                    p.publish(plan,{'metadata_fingerprint':fingerprint})
            run.assert_called_once()

    def test_fingerprint_captures_before_edit_not_after(self):
        plan={'available_groups':[],'common_audit':None,'pending_groups':[]}
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary);source=directory/'source.py';source.write_text('captured')
            with mock.patch.object(p,'ROOT',directory),mock.patch.object(p,'ALLOWLIST',['source.py']),mock.patch.object(p,'METADATA_ARTIFACTS',[]):
                before,files,_=p.metadata_snapshot(plan,{})
                source.write_text('later edit');after,_,_=p.metadata_snapshot(plan,{})
                self.assertEqual(files['source.py'],b'captured');self.assertNotEqual(before,after)

    def test_exact_allowlists_and_readable_scores_affect_fingerprint(self):
        plan={'available_groups':[],'common_audit':None,'pending_groups':[]}
        with mock.patch.object(p,'ALLOWLIST',[]),mock.patch.object(p,'METADATA_ARTIFACTS',[]):
            baseline,_,_=p.metadata_snapshot(plan,{})
            with mock.patch.object(p,'ALLOWLIST',['missing-allowlisted-source.py']):
                declared,_,_=p.metadata_snapshot(plan,{})
            with mock.patch.object(p,'METADATA_ARTIFACTS',['missing-artifact.json']):
                artifacts,_,_=p.metadata_snapshot(plan,{})
            score_state={'published_groups':{'v15_sqa_FULL':{'scores':[{'accuracy':70.,'n':2017,'metric':'ScienceQA IMG'}]}}}
            scored,_,_=p.metadata_snapshot(plan,score_state)
        self.assertEqual(len({baseline,declared,artifacts,scored}),4)


class DynamicPublication(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.out=self.root/'audit';self.directory=self.out/'next_gap_diagnosis_20261008'
        self.stability=self.directory/'stability';self.stability.mkdir(parents=True)
        for name,value in [('ROOT',self.root),('OUT',self.out),('ALLOWLIST',[]),('METADATA_ARTIFACTS',[])]:
            patch=mock.patch.object(p,name,value);patch.start();self.addCleanup(patch.stop)
        self.controller=self.directory/'schedule_next_full_controls.py';self.controller.write_text('frozen controller')
        self.worker=self.stability/'next_text_full_streamwait.py';self.worker.write_text('frozen worker')
        self.plan_path=self.stability/'next_text_full_streamwait.controller.plan.json'
        self.state_path=self.stability/'next_text_full_streamwait.controller.state.json'
        self.methods=['EADP_beta2','AZ_beta2','EADP_beta1']
        self.plan={'controller_sha256':p.sha(self.controller),'jobs':{arm:{'arm':arm} for arm in self.methods}}
        self.write(self.plan_path,self.plan)

    def write(self,path,value):
        Path(path).write_text(json.dumps(value))

    def registered(self,methods=None,**overrides):
        methods=self.methods if methods is None else methods
        state=dict(started_utc='2026-10-08T15:35:45+00:00',selected_arms=methods,
                   registered_methods=methods,plan_sha256=p.sha(self.plan_path),
                   publication_expected_groups=[group for group,arms in p.CONTROL_GROUPS.items() if set(arms)<=set(methods)])
        state.update(overrides);self.write(self.state_path,state)
        return p.control_registration()

    def test_prepare_only_does_not_extend_scope(self):
        self.write(self.state_path,{'started_utc':None,'selected_arms':[],'registered_methods':[]})
        self.assertIsNone(p.control_registration())
        self.assertTrue(p.publication_complete({'all_experiments_complete':True,'expected_groups':['original']},
                                               {'published_groups':{'original':{}}}))

    def test_started_three_arms_requires_nineteen_groups(self):
        registration=self.registered()
        original=['original'+str(index) for index in range(17)]
        expected=original+registration['expected_groups']
        self.assertEqual(len(expected),19)
        plan={'all_experiments_complete':True,'expected_groups':expected}
        state={'published_groups':dict.fromkeys(original,{})}
        self.assertFalse(p.publication_complete(plan,state))
        state['published_groups'][expected[-2]]={}
        self.assertFalse(p.publication_complete(plan,state))
        state['published_groups'][expected[-1]]={}
        self.assertTrue(p.publication_complete(plan,state))
        plan['all_experiments_complete']=False
        self.assertFalse(p.publication_complete(plan,state))

    def test_invalid_registration_and_changed_plan_rejected(self):
        for methods in [['EADP_beta2'],['unknown'],['EADP_beta1','EADP_beta1']]:
            with self.subTest(methods=methods),self.assertRaises(ValueError):self.registered(methods)
        with self.assertRaisesRegex(ValueError,'aliases'):
            self.registered(selected_arms=['EADP_beta1'])
        with self.assertRaisesRegex(ValueError,'plan hash'):
            self.registered(plan_sha256='wrong')
        self.controller.write_text('changed controller')
        with self.assertRaisesRegex(ValueError,'controller source'):
            self.registered()

    def test_mutable_controller_progress_does_not_create_metadata_commit(self):
        registration=self.registered(status='waiting',child_pids=[])
        plan={'available_groups':[],'control_registration':registration}
        with mock.patch.object(p,'diagnostic_metadata',return_value={}):
            before,files,_=p.metadata_snapshot(plan,{})
            registration2=self.registered(status='running',child_pids=[123],updated_utc='later')
            plan['control_registration']=registration2
            after,_,_=p.metadata_snapshot(plan,{})
        self.assertEqual(before,after)
        self.assertFalse(any('.state.' in path for path in files))

    def test_beta1_is_a_separate_eadp_control(self):
        history={
          'next_textvqa_K32_streamwait':{'scores':[{'method':'EADP','accuracy':52.,'n':5000,'metric':'TextVQA'},
            {'method':'AnchorZip','accuracy':53.,'n':5000,'metric':'TextVQA'}]},
          'next_textvqa_K32_default_beta1':{'scores':[{'method':'EADP','beta':1.,'accuracy':54.,'n':5000,'metric':'TextVQA'}]}}
        table,readme=p.readable_results(history)
        self.assertIn('next_textvqa_K32_streamwait',table)
        self.assertNotIn('default_beta1',table)
        self.assertIn('script-default sensitivity control',readme)
        self.assertIn('next_textvqa_K32_default_beta1/manifest.json',readme)
        history['next_textvqa_K32_default_beta1']['scores'][0]['method']='AnchorZip'
        with self.assertRaises(ValueError):p.readable_results(history)

    def test_partial_control_pair_does_not_score_or_package(self):
        registration=self.registered()
        finished={arm:self.stability/(arm+'.finished.json') for arm in self.methods}
        self.write(finished['EADP_beta2'],{'complete':True})
        with mock.patch.object(p,'control_paths',side_effect=lambda reg,arm:(None,{'finished':finished[arm]})):
            self.assertFalse(p.control_group_ready(registration,p.CONTROL_GROUPS['next_textvqa_K32_streamwait']))
            self.write(finished['AZ_beta2'],{'complete':True})
            self.assertTrue(p.control_group_ready(registration,p.CONTROL_GROUPS['next_textvqa_K32_streamwait']))

    def test_previously_registered_scope_cannot_disappear(self):
        summary={'rows':[{'output':'unused','status':'pending'} for _ in range(28)],'errors':[]}
        with mock.patch.object(p,'command',return_value=json.dumps({'needs_attention':False,'all_complete':False})),mock.patch.object(p,'read',return_value=summary),mock.patch.object(p,'module'),mock.patch.object(p,'group_definitions',return_value=[]),mock.patch.object(p,'control_registration',return_value=None):
            with self.assertRaisesRegex(ValueError,'disappeared'):
                p.prepare({'expected_groups':['next_textvqa_K32_streamwait']})

    def panel_fixture(self):
        worker=self.stability/'next_text_stream_panel.py';worker.write_text('frozen panel worker')
        stem=self.stability/'next_text_K32_EADP_stream_panel128'
        manifest=Path(str(stem)+'.manifest.json');self.write(manifest,{'n':128})
        records=Path(str(stem)+'.records.jsonl')
        records.write_text(''.join(json.dumps({'arms':{variant:{} for variant in ['as_is_1','as_is_2','wait_1','wait_2']}})+'\n' for _ in range(128)))
        protocol=Path(str(stem)+'.protocol.json')
        self.write(protocol,{'manifest':str(manifest),'output_paths':{'records':str(records)},'index_trace':False,'worker_sha256':p.sha(worker)})
        result=Path(str(stem)+'.result.json')
        self.write(result,{'success':True,'complete':True,'finished_utc':'2026-10-08T15:20:00+00:00',
          'protocol':str(protocol),'summary':{'n':128,'n_generations':512},
          'protocol_sha256':p.sha(protocol),'manifest_sha256':p.sha(manifest),'records_sha256':p.sha(records)})
        selected=[worker,manifest,records,protocol,result]
        report={'panel_complete':True,'files':[{'path':str(path.relative_to(self.root)),
                  'sha256':p.sha(path),'bytes':path.stat().st_size} for path in selected]}
        report_path=self.out/'next_gap_diagnosis_report.json';self.write(report_path,report)
        return report_path,report,result

    def test_complete_panel_selects_exact_files_and_report(self):
        report_path,report,_=self.panel_fixture()
        files=p.diagnostic_metadata()
        self.assertEqual(set(files),{entry['path'] for entry in report['files']}|{str(report_path.relative_to(self.root))})

    def test_partial_panel_is_not_published(self):
        report_path,report,_=self.panel_fixture();report['panel_complete']=False;self.write(report_path,report)
        self.assertEqual(p.diagnostic_metadata(),{})

    def test_panel_unsafe_or_drifted_selection_rejected(self):
        for mode in ['traversal','private','hash','partial']:
            with self.subTest(mode=mode):
                report_path,report,result=self.panel_fixture()
                if mode=='traversal':report['files'][0]['path']='../private.json'
                elif mode=='private':report['files'][0]['path']=str((self.directory/'private.json').relative_to(self.root))
                elif mode=='hash':report['files'][0]['sha256']='wrong'
                else:
                    data=p.read(result);data['complete']=False;self.write(result,data)
                    entry=report['files'][-1];entry['sha256']=p.sha(result);entry['bytes']=result.stat().st_size
                self.write(report_path,report)
                with self.assertRaises(ValueError):p.diagnostic_metadata()

    def control_fixture(self):
        arm='EADP_beta2';registration=self.registered()
        paths={name:self.stability/(name+'.json') for name in ['prediction','runtime','native_protocol','control_protocol','score','state','finished','log']}
        for path in paths.values():self.write(path,{})
        questions=self.stability/'questions.jsonl';questions.write_text('{}\n')
        images=self.stability/'images.json';self.write(images,{})
        parameters=dict(alpha=.5,beta=2.,visual_token_num=32,temperature=0,top_p=None,
                        num_beams=1,max_new_tokens=128,conv_mode='vicuna_v1',anchorzip=False,question_file=str(questions))
        protocol=dict(arm=arm,question_count=5000,attempt=1,index_trace=False,parameters=parameters,
            launcher_sha256=p.sha(self.worker),image_manifest=str(images),image_manifest_sha256=p.sha(images),
            AST_validation={'restoration_ast_identical':True,'record_stream_added':False,'inserted_nodes':[
              'caller_stream = torch.cuda.current_stream(device=self.device)',
              'image_stream.wait_stream(caller_stream)','text_stream.wait_stream(caller_stream)']})
        self.write(paths['control_protocol'],protocol)
        exact={'success':True,'n':5000,'accuracy_percent':52.123,'prediction_sha256':p.sha(paths['prediction']),
               'runtime_sha256':p.sha(paths['runtime'])}
        self.write(paths['score'],dict(exact,arm=arm,control_protocol_sha256=p.sha(paths['control_protocol']),image_manifest_sha256=p.sha(images)))
        self.write(paths['state'],{'generation_exit_code':0,'complete':True,'started_utc':'now'})
        finished={'success':True,'complete':True,'n':5000,'generation_exit_code':0,'arm':arm,
                 'prediction_sha256':p.sha(paths['prediction']),'score_sha256':p.sha(paths['score'])}
        self.write(paths['finished'],finished)
        registration['plan']['jobs'][arm]['attempt']=1
        worker=SimpleNamespace(__file__=str(self.worker),verify_images=mock.Mock(),validate_and_score=mock.Mock(return_value=exact))
        return registration,worker,paths,protocol,finished

    def test_complete_control_has_raw_runtime_protocol_score_inputs_and_hashes(self):
        registration,worker,paths,_,_=self.control_fixture()
        with mock.patch.object(p,'control_paths',return_value=(worker,paths)):
            files,score=p.validate_control(registration,'EADP_beta2')
        self.assertEqual(score['accuracy'],52.123);self.assertEqual(score['method'],'EADP')
        for name in ['prediction','runtime','native_protocol','control_protocol','score','state','finished']:
            self.assertIn(str(paths[name].relative_to(self.root)),files)
        self.assertIn(str(self.controller.relative_to(self.root)),files)
        worker.verify_images.assert_called_once();worker.validate_and_score.assert_called_once()

    def test_control_completion_and_official_score_hard_gates(self):
        for mode in ['missing_runtime','partial','exit','beta','score_drift','official_mismatch']:
            with self.subTest(mode=mode):
                registration,worker,paths,protocol,finished=self.control_fixture()
                if mode=='missing_runtime':paths['runtime'].unlink()
                elif mode=='partial':finished['n']=4999;self.write(paths['finished'],finished)
                elif mode=='exit':finished['generation_exit_code']=1;self.write(paths['finished'],finished)
                elif mode=='beta':protocol['parameters']['beta']=1.;self.write(paths['control_protocol'],protocol)
                elif mode=='score_drift':paths['score'].write_text('{}')
                else:worker.validate_and_score.return_value['accuracy_percent']=99.
                with mock.patch.object(p,'control_paths',return_value=(worker,paths)),self.assertRaises(ValueError):
                    p.validate_control(registration,'EADP_beta2')
                if mode!='official_mismatch':worker.validate_and_score.assert_not_called()


class ContinuationPublication(unittest.TestCase):
    write=DynamicPublication.write

    def setUp(self):
        DynamicPublication.setUp(self)
        self.continuation=p.continuation_directory();self.continuation.mkdir()
        self.cworker=self.continuation/'streamwait_repair_worker.py';self.cworker.write_text('frozen continuation worker')
        self.ccontroller=self.continuation/'schedule_streamwait_repair_continuation.py';self.ccontroller.write_text('frozen continuation controller')
        self.cplan_path=self.continuation/'streamwait_repair_continuation.plan.json'
        self.cstate_path=self.continuation/'streamwait_repair_continuation.state.json'
        self.input=self.continuation/'questions.jsonl';self.input.write_text('{}\n')
        self.images=self.continuation/'images.json';self.write(self.images,{})
        self.cgroups=[];self.cjobs={};self.cpaths={};self.validated={}
        for group_id,job_ids in p.CONTINUATION_GROUPS.items():
            parts=group_id.split('_');budget=0 if parts[2]=='FULL' else int(parts[2][1:])
            group={'group_id':group_id,'model':'next','task':parts[1],'budget':budget,
                   'protocol_label':'wait-only CUDA stream repair; original CQM-A','kind':'FULL' if len(job_ids)==1 else 'pair',
                   'arms':['FULL'] if len(job_ids)==1 else ['EADP','AnchorZip'],'job_ids':list(job_ids)}
            self.cgroups.append(group)
            for job_id in job_ids:
                paths={name:self.continuation/(job_id+'.'+name+'.json') for name in
                       ['prediction','runtime','native_protocol','control_protocol','score','finished','state']}
                for path in paths.values():self.write(path,{})
                protocol={'job_id':job_id,'launcher_sha256':p.sha(self.cworker),
                          'source_and_input_sha256':{str(self.cworker):p.sha(self.cworker),str(self.input):p.sha(self.input)},
                          'image_manifest':str(self.images),'output_paths':{name:str(path) for name,path in paths.items()}}
                self.write(paths['control_protocol'],protocol)
                exact={'accuracy_percent':50.,'n':5000 if parts[1]=='textvqa' else 2017,
                       'success':True,'method':'FULL' if len(job_ids)==1 else ('AnchorZip' if '_AZ' in job_id else 'EADP'),
                       'task':parts[1],'budget':budget}
                self.write(paths['score'],exact)
                finished={'success':True,'complete':True,'score_sha256':p.sha(paths['score']),
                          'prediction_sha256':p.sha(paths['prediction'])}
                self.write(paths['finished'],finished)
                self.cjobs[job_id]={'job_id':job_id,'attempt':1,'control_protocol':str(paths['control_protocol']),
                                   'control_protocol_sha256':p.sha(paths['control_protocol']),
                                   'output_paths':{name:str(path) for name,path in paths.items()}}
                self.cpaths[job_id]=paths
                public={str(path):p.sha(path) for name,path in paths.items() if name!='state'}
                public[str(self.input)]=p.sha(self.input);public[str(self.images)]=p.sha(self.images)
                self.validated[job_id]={'score':exact,'files':public,'finished':finished,'protocol':protocol}
        self.cplan={'groups':self.cgroups,'jobs':self.cjobs,'controller_sha256':p.sha(self.ccontroller),
                    'source_and_input_sha256':{str(self.cworker):p.sha(self.cworker)}}
        self.write(self.cplan_path,self.cplan)
        self.fakeworker=SimpleNamespace(__file__=str(self.cworker),
            paths=mock.Mock(side_effect=lambda job_id,attempt:self.cpaths[job_id]),
            validate_finished=mock.Mock(side_effect=lambda job_id,attempt:self.validated[job_id]))

    def cregistered(self,**overrides):
        state={'started_utc':'2026-10-09T00:00:00+08:00','registered_jobs':list(self.cjobs),
               'publication_expected_groups':[group['group_id'] for group in self.cgroups],
               'plan_sha256':p.sha(self.cplan_path)}
        state.update(overrides);self.write(self.cstate_path,state)
        return p.continuation_registration()

    def test_continuation_prepared_does_not_expand(self):
        self.write(self.cstate_path,{'started_utc':None,'registered_jobs':[],'publication_expected_groups':[]})
        self.assertIsNone(p.continuation_registration())

    def test_continuation_actual_registration_expands_to_twenty_six(self):
        registration=self.cregistered()
        expected=['old'+str(index) for index in range(19)]+registration['expected_groups']
        plan={'all_experiments_complete':True,'expected_groups':expected}
        state={'published_groups':dict.fromkeys(expected[:19],{})}
        self.assertEqual(len(expected),26);self.assertFalse(p.publication_complete(plan,state))
        state['published_groups']=dict.fromkeys(expected,{})
        self.assertTrue(p.publication_complete(plan,state))

    def test_continuation_scope_and_source_changes_rejected(self):
        for mode in ['job_missing','group_missing','plan_hash','controller','worker','group_kind']:
            with self.subTest(mode=mode):
                if mode=='job_missing':override={'registered_jobs':list(self.cjobs)[:-1]}
                elif mode=='group_missing':override={'publication_expected_groups':[]}
                elif mode=='plan_hash':override={'plan_sha256':'wrong'}
                elif mode=='controller':
                    self.ccontroller.write_text('changed');override={}
                elif mode=='worker':
                    self.cworker.write_text('changed');override={}
                else:
                    self.cplan['groups'][0]['kind']='FULL';self.write(self.cplan_path,self.cplan);override={}
                with self.assertRaises(ValueError):self.cregistered(**override)
                self.ccontroller.write_text('frozen continuation controller');self.cworker.write_text('frozen continuation worker')

    def test_continuation_scope_cannot_disappear(self):
        summary={'rows':[{'output':'unused','status':'pending'} for _ in range(28)],'errors':[]}
        with mock.patch.object(p,'command',return_value=json.dumps({'needs_attention':False,'all_complete':True})),mock.patch.object(p,'read',return_value=summary),mock.patch.object(p,'module'),mock.patch.object(p,'group_definitions',return_value=[]),mock.patch.object(p,'control_registration',return_value=None),mock.patch.object(p,'continuation_registration',return_value=None):
            with self.assertRaisesRegex(ValueError,'continuation scope disappeared'):
                p.prepare({'expected_groups':['next_sqa_FULL_streamwait']})

    def test_continuation_partial_pair_cannot_publish(self):
        registration=self.cregistered();group=self.cgroups[0]
        self.cpaths[group['job_ids'][1]]['finished'].unlink()
        with mock.patch.object(p,'load_continuation_worker',return_value=self.fakeworker):
            self.assertFalse(p.continuation_group_ready(registration,group))
        self.fakeworker.validate_finished.assert_not_called()

    def test_continuation_fixed_path_and_protocol_drift_rejected(self):
        registration=self.cregistered();job_id=self.cgroups[0]['job_ids'][0]
        with mock.patch.object(p,'load_continuation_worker',return_value=self.fakeworker):
            self.cpaths[job_id]['prediction']=self.root/'user.json'
            with self.assertRaisesRegex(ValueError,'outside authorized'):
                p.continuation_paths(registration,job_id)
            self.cpaths[job_id]['prediction']=self.continuation/(job_id+'.prediction.json')
            self.cpaths[job_id]['control_protocol'].write_text('{}')
            with self.assertRaisesRegex(ValueError,'protocol differs'):
                p.continuation_paths(registration,job_id)

    def test_complete_continuation_archives_all_public_evidence_no_state(self):
        registration=self.cregistered();group=self.cgroups[0];job_id=group['job_ids'][0]
        with mock.patch.object(p,'load_continuation_worker',return_value=self.fakeworker):
            files,score=p.validate_continuation_job(registration,group,job_id)
        self.assertEqual(score['accuracy'],50.);self.assertEqual(score['method'],'EADP')
        self.assertEqual(score['protocol'],group['protocol_label'])
        for name in ['prediction','runtime','native_protocol','control_protocol','score','finished']:
            self.assertIn(str(self.cpaths[job_id][name].relative_to(self.root)),files)
        self.assertIn(str(self.input.relative_to(self.root)),files)
        self.assertIn(str(self.images.relative_to(self.root)),files)
        self.assertFalse(any('.state.' in path for path in files))
        self.fakeworker.validate_finished.assert_called_once_with(job_id,attempt=1)

    def test_continuation_frozen_cpu_validator_failure_propagates(self):
        registration=self.cregistered();group=self.cgroups[0]
        self.fakeworker.validate_finished.side_effect=ValueError('official score mismatch')
        with mock.patch.object(p,'load_continuation_worker',return_value=self.fakeworker),self.assertRaisesRegex(ValueError,'official score mismatch'):
            p.validate_continuation_job(registration,group,group['job_ids'][0])

    def test_continuation_source_drift_rejected_before_validation(self):
        registration=self.cregistered();group=self.cgroups[0];self.input.write_text('changed')
        with mock.patch.object(p,'load_continuation_worker',return_value=self.fakeworker),self.assertRaisesRegex(ValueError,'SHA mismatch'):
            p.validate_continuation_job(registration,group,group['job_ids'][0])
        self.fakeworker.validate_finished.assert_not_called()

    def test_continuation_invalid_score_or_archive_selection_rejected(self):
        registration=self.cregistered();group=self.cgroups[0];job_id=group['job_ids'][0]
        for mode in ['accuracy','denominator','finished','missing_runtime','private_state','payload']:
            with self.subTest(mode=mode):
                validated=json.loads(json.dumps(self.validated[job_id]))
                if mode=='accuracy':validated['score']['accuracy_percent']=99.
                elif mode=='denominator':validated['score']['n']=4999
                elif mode=='finished':validated['finished']['complete']=False
                elif mode=='missing_runtime':validated['files'].pop(str(self.cpaths[job_id]['runtime']))
                elif mode=='private_state':validated['files'][str(self.cpaths[job_id]['state'])]=p.sha(self.cpaths[job_id]['state'])
                else:
                    payload=self.continuation/'model.safetensors';payload.write_text('payload')
                    validated['files'][str(payload)]=p.sha(payload)
                self.fakeworker.validate_finished.side_effect=None;self.fakeworker.validate_finished.return_value=validated
                with mock.patch.object(p,'load_continuation_worker',return_value=self.fakeworker),self.assertRaises(ValueError):
                    p.validate_continuation_job(registration,group,job_id)

    def test_continuation_full_and_pair_readable_preserve_original_scores(self):
        history={'next_textvqa_K128':{'scores':[{'accuracy':57.,'n':5000,'metric':'TextVQA'},
                                              {'accuracy':56.,'n':5000,'metric':'TextVQA'}]},
                 'next_textvqa_K128_streamwait':{'scores':[{'method':'EADP','accuracy':55.,'n':5000,'metric':'TextVQA','protocol':'wait-only'},
                    {'method':'AnchorZip','accuracy':54.,'n':5000,'metric':'TextVQA','protocol':'wait-only'}]},
                 'next_sqa_FULL_streamwait':{'scores':[{'method':'FULL','accuracy':67.,'n':2017,'metric':'SQA','group_kind':'FULL','protocol':'wait-only CQM-A'}]}}
        table,readme=p.readable_results(history)
        rows=list(__import__('csv').DictReader(io.StringIO(table)))
        self.assertEqual(len(rows),3)
        by={row['group_id']:row for row in rows}
        self.assertEqual(by['next_textvqa_K128']['EADP'],'57.0')
        self.assertEqual(by['next_textvqa_K128_streamwait']['EADP'],'55.0')
        self.assertEqual(by['next_sqa_FULL_streamwait']['FULL'],'67.0')
        self.assertEqual(by['next_sqa_FULL_streamwait']['AnchorZip'],'')
        self.assertEqual(by['next_sqa_FULL_streamwait']['protocol'],'wait-only CQM-A')
        self.assertIn('original scores remain separate',readme)

    def test_continuation_progress_does_not_change_metadata_fingerprint(self):
        registration=self.cregistered(status='waiting',child_pids=[])
        plan={'available_groups':[],'continuation_registration':registration}
        before,files,_=p.metadata_snapshot(plan,{})
        plan['continuation_registration']=self.cregistered(status='running',child_pids=[123])
        after,_,_=p.metadata_snapshot(plan,{})
        self.assertEqual(before,after)
        self.assertFalse(any('.state.' in path for path in files))

    def test_continuation_static_scope_is_exact_and_hashed(self):
        selected=[self.cworker,self.ccontroller,self.cplan_path]
        report={'files':[{'path':str(path.relative_to(self.root)),'sha256':p.sha(path),'bytes':path.stat().st_size} for path in selected]}
        report_path=self.out/'streamwait_continuation_scope.json';self.write(report_path,report)
        self.assertEqual(len(p.continuation_metadata()),4)
        self.cworker.write_text('changed')
        with self.assertRaisesRegex(ValueError,'hash/size'):p.continuation_metadata()

    def test_continuation_static_scope_excludes_private_and_partial(self):
        for suffix in ['.state.json','.finished.json','.jsonl','.safetensors']:
            with self.subTest(suffix=suffix):
                path=self.continuation/('partial'+suffix);path.write_text('{}')
                report={'files':[{'path':str(path.relative_to(self.root)),'sha256':p.sha(path),'bytes':path.stat().st_size}]}
                self.write(self.out/'streamwait_continuation_scope.json',report)
                with self.assertRaises(ValueError):p.continuation_metadata()


class LegacyPublication(unittest.TestCase):
    write=DynamicPublication.write

    def setUp(self):
        DynamicPublication.setUp(self)
        self.directory=p.continuation_directory();self.directory.mkdir()
        self.worker=self.directory/'legacy_streamwait_worker.py';self.worker.write_text('frozen legacy worker')
        self.controller=self.directory/'schedule_legacy_streamwait_continuation.py';self.controller.write_text('frozen legacy controller')
        self.plan_path=self.directory/'legacy_streamwait_continuation.plan.json'
        self.state_path=self.directory/'legacy_streamwait_continuation.state.json'
        self.input=self.directory/'questions.jsonl';self.input.write_text('{}\n')
        self.image=self.directory/'images.manifest.json';self.write(self.image,{'kind':'files'})
        self.identity=self.directory/'identities.manifest.json';self.write(self.identity,{'n':1})
        self.groups=[dict(value,group_id=key,protocol_label='legacy_native_caller_stream_wait_only_beta2')
                     for key,value in p.legacy_group_contracts().items()]
        self.jobs={};self.paths={};self.validated={}
        for group in self.groups:
            for job_id,method in zip(group['job_ids'],group['arms']):
                paths={name:self.directory/(job_id+'.'+name+'.json') for name in
                       ['prediction','runtime','native_protocol','control_protocol','score','finished','state']}
                for path in paths.values():self.write(path,{})
                protocol={'job_id':job_id,'launcher_sha256':p.sha(self.worker),'parameters':{'question_file':str(self.input),'lang':'en'},
                          'origin':{'score_data':[]},'identity_manifest':str(self.identity),'identity_manifest_sha256':p.sha(self.identity),
                          'image_manifest':str(self.image),'image_manifest_sha256':p.sha(self.image),
                          'source_and_input_sha256':{str(self.worker):p.sha(self.worker),str(self.input):p.sha(self.input)}}
                self.write(paths['control_protocol'],protocol)
                score={'success':True,'n':p.LEGACY_COUNTS[group['task']],'model':group['model'],'task':group['task'],
                       'method':method,'budget':group['budget'],'value':1429.4 if group['task']=='mme' else 65.1234,
                       'metric':p.LEGACY_METRICS[group['task']]}
                self.write(paths['score'],score)
                finished={'success':True,'complete':True,'n':score['n'],'job_id':job_id,'generation_exit_code':0,
                          'score_sha256':p.sha(paths['score']),'prediction_sha256':p.sha(paths['prediction'])}
                self.write(paths['finished'],finished)
                files={str(path):p.sha(path) for name,path in paths.items() if name!='state'}
                for path in [self.worker,self.input,self.identity,self.image]:files[str(path)]=p.sha(path)
                self.validated[job_id]={'score':score,'files':files,'finished':finished,'protocol':protocol}
                self.paths[job_id]=paths
                self.jobs[job_id]={'job_id':job_id,'model':group['model'],'task':group['task'],'method':method,
                    'budget':group['budget'],'n':score['n'],'attempt':1,'output_paths':{name:str(path) for name,path in paths.items()},
                    'control_protocol':str(paths['control_protocol']),'control_protocol_sha256':p.sha(paths['control_protocol'])}
        self.plan={'groups':self.groups,'jobs':self.jobs,'worker_path':str(self.worker),'worker_sha256':p.sha(self.worker),
                   'controller_sha256':p.sha(self.controller),'source_and_input_sha256':{str(self.worker):p.sha(self.worker)}}
        self.write(self.plan_path,self.plan)
        self.fakeworker=SimpleNamespace(__file__=str(self.worker),paths=mock.Mock(side_effect=lambda job,attempt:self.paths[job]),
                     validate_finished=mock.Mock(side_effect=lambda job,attempt:self.validated[job]))

    def registered(self,**overrides):
        state={'started_utc':'2026-10-09T01:00:00+08:00','registered_jobs':list(self.jobs),
               'publication_expected_groups':[group['group_id'] for group in self.groups],'plan_sha256':p.sha(self.plan_path)}
        state.update(overrides);self.write(self.state_path,state)
        return p.legacy_registration()

    def group(self,task,kind='single_audit'):
        return next(group for group in self.groups if group['task']==task and group['kind']==kind)

    def validate(self,group):
        registration=self.registered()
        with mock.patch.object(p,'load_legacy_worker',return_value=self.fakeworker):
            return p.validate_legacy_job(registration,group,group['job_ids'][0])

    def test_legacy_exact_counts_and_prepare_only_scope(self):
        self.assertEqual(len(self.groups),29);self.assertEqual(len(self.jobs),32)
        self.assertEqual(sum(job['n'] for job in self.jobs.values()),224199)
        self.assertEqual(sum(group['kind']=='single_audit' for group in self.groups),23)
        self.write(self.state_path,{'started_utc':None,'registered_jobs':[],'publication_expected_groups':[]})
        self.assertIsNone(p.legacy_registration())

    def test_legacy_registered_scope_requires_fifty_five_groups(self):
        registration=self.registered();old=['old'+str(index) for index in range(26)]
        plan={'all_experiments_complete':True,'expected_groups':old+registration['expected_groups']}
        state={'published_groups':dict.fromkeys(old,{})}
        self.assertEqual(len(plan['expected_groups']),55)
        self.assertFalse(p.publication_complete(plan,state))
        state['published_groups']=dict.fromkeys(plan['expected_groups'],{})
        self.assertTrue(p.publication_complete(plan,state))

    def test_legacy_missing_registration_or_changed_source_rejected(self):
        with self.assertRaises(ValueError):self.registered(registered_jobs=list(self.jobs)[:-1])
        with self.assertRaises(ValueError):self.registered(publication_expected_groups=[])
        with self.assertRaises(ValueError):self.registered(plan_sha256='wrong')
        self.worker.write_text('changed')
        with self.assertRaisesRegex(ValueError,'source changed'):self.registered()

    def test_legacy_registered_scope_cannot_disappear(self):
        summary={'rows':[{'output':'unused','status':'pending'} for _ in range(28)],'errors':[]}
        with mock.patch.object(p,'command',return_value=json.dumps({'needs_attention':False,'all_complete':True})),mock.patch.object(p,'read',return_value=summary),mock.patch.object(p,'module'),mock.patch.object(p,'group_definitions',return_value=[]),mock.patch.object(p,'control_registration',return_value=None),mock.patch.object(p,'continuation_registration',return_value=None),mock.patch.object(p,'legacy_registration',return_value=None):
            with self.assertRaisesRegex(ValueError,'legacy scope disappeared'):
                p.prepare({'expected_groups':[self.groups[0]['group_id']]})

    def test_partial_legacy_pair_never_scores(self):
        registration=self.registered();group=self.group('pope','pair')
        self.paths[group['job_ids'][1]]['finished'].unlink()
        with mock.patch.object(p,'load_legacy_worker',return_value=self.fakeworker):
            self.assertFalse(p.legacy_group_ready(registration,group))
        self.fakeworker.validate_finished.assert_not_called()

    def test_legacy_complete_raw_point_metric_and_evidence(self):
        group=self.group('mme','pair');files,score=self.validate(group)
        self.assertEqual(score['value'],1429.4);self.assertEqual(score['accuracy'],1429.4)
        self.assertEqual(score['metric'],p.LEGACY_METRICS['mme'])
        for name in ['prediction','runtime','native_protocol','control_protocol','score','finished']:
            self.assertIn(str(self.paths[group['job_ids'][0]][name].relative_to(self.root)),files)
        self.assertFalse(any('.state.' in key for key in files))

    def test_legacy_bad_finished_official_score_or_method_rejected(self):
        group=self.group('pope','pair');job=group['job_ids'][0];original=self.validated[job]
        for mode in ['exit','partial','score','metric','method','missing_runtime','hash']:
            with self.subTest(mode=mode):
                validated=json.loads(json.dumps(original))
                if mode=='exit':validated['finished']['generation_exit_code']=False
                elif mode=='partial':validated['finished']['n']-=1
                elif mode=='score':validated['score']['value']=99.
                elif mode=='metric':validated['score']['metric']='accuracy_percent'
                elif mode=='method':validated['score']['method']='AnchorZip'
                elif mode=='missing_runtime':validated['files'].pop(str(self.paths[job]['runtime']))
                else:validated['files'][str(self.identity)]='wrong'
                self.validated[job]=validated
                with self.assertRaises(ValueError):self.validate(group)
        self.validated[job]=original

    def test_legacy_frozen_cpu_validator_error_propagates(self):
        group=self.group('gqa','pair');self.fakeworker.validate_finished.side_effect=RuntimeError('official mismatch')
        with self.assertRaisesRegex(RuntimeError,'official mismatch'):self.validate(group)

    def test_legacy_user_private_and_model_payload_rejected(self):
        group=self.group('gqa','pair');job=group['job_ids'][0]
        for name in ['unknown.tsv','unknown.zip','model.safetensors','agent.events.json','user.state.json','llava_round2_driver.sh','llava_mmben_driver.sh','fig1_user.py']:
            with self.subTest(name=name):
                if name=='llava_round2_driver.sh':path=self.root/'Qwen_vl/scripts/stage1_roundtrip_pilot'/name;path.parent.mkdir(parents=True,exist_ok=True)
                else:path=self.directory/name
                path.write_text('{}');self.validated[job]['files'][str(path)]=p.sha(path)
                with self.assertRaises(ValueError):self.validate(group)
                self.validated[job]['files'].pop(str(path))

    def test_legacy_single_rows_never_make_fair_pair(self):
        group=self.group('mmbcn');_,score=self.validate(group)
        history={group['group_id']:{'scores':[score]}}
        paired,readme=p.readable_results(history)
        rows=list(__import__('csv').DictReader(io.StringIO(paired)))
        self.assertEqual(rows,[])
        legacy=list(__import__('csv').DictReader(io.StringIO(p.readable_legacy_results(history))))
        self.assertEqual(len(legacy),1);self.assertEqual(legacy[0]['method'],'AnchorZip')
        self.assertEqual(legacy[0]['paired_group'],'False');self.assertEqual(legacy[0]['guidance_language'],'en')
        self.assertNotIn('AZ_minus_EADP',legacy[0]);self.assertIn('lang=en',readme)

    def test_legacy_pair_and_full_use_explicit_model_task(self):
        history={}
        for kind in ['pair','FULL']:
            group=self.group('mme',kind);scores=[]
            for job in group['job_ids']:
                with mock.patch.object(p,'load_legacy_worker',return_value=self.fakeworker):
                    _,score=p.validate_legacy_job(self.registered(),group,job)
                scores.append(score)
            history[group['group_id']]={'scores':scores}
        table,_=p.readable_results(history);rows=list(__import__('csv').DictReader(io.StringIO(table)))
        self.assertEqual({row['model'] for row in rows},{'v15'})
        self.assertEqual({row['task'] for row in rows},{'mme'})
        self.assertEqual({row['budget'] for row in rows},{'K128','FULL'})
        full=next(row for row in rows if row['budget']=='FULL')
        self.assertEqual(full['FULL'],'1429.4');self.assertEqual(full['AnchorZip'],'')

    def test_legacy_progress_does_not_change_metadata_fingerprint(self):
        plan={'available_groups':[],'legacy_registration':self.registered(status='waiting')}
        before,files,_=p.metadata_snapshot(plan,{})
        plan['legacy_registration']=self.registered(status='running',child_pids=[123])
        after,_,_=p.metadata_snapshot(plan,{})
        self.assertEqual(before,after);self.assertFalse(any('.state.' in key for key in files))

    def test_mmbench_embedded_images_replaced_with_exact_labels(self):
        source=self.directory/'MMBench.tsv'
        source.write_text('index\tquestion\thint\tA\tB\tanswer\timage\n1\tquestion\thint\ta\tb\tB\tPRIVATE_BASE64_IMAGE\n')
        result=json.loads(p.mmbench_labels_without_images(source,p.sha(source),1))
        self.assertEqual(result['rows'][0]['answer'],'B');self.assertNotIn('image',result['columns'])
        self.assertNotIn('PRIVATE_BASE64_IMAGE',str(result));self.assertEqual(result['source_sha256'],p.sha(source))
        with self.assertRaises(ValueError):p.mmbench_labels_without_images(source,p.sha(source),2)

    def test_mme_archive_replaced_by_complete_canonical_labels(self):
        source=self.directory/'eval_tool.zip';source.write_bytes(b'original archive')
        gt={('existence',str(index),'question'): 'yes' for index in range(2374)}
        with mock.patch.object(p,'module',return_value=SimpleNamespace(canonical_gt=lambda path:gt)):
            result=json.loads(p.mme_gt_without_archive(source,p.sha(source)))
        self.assertEqual(result['n'],2374);self.assertEqual(result['source_sha256'],p.sha(source))
        self.assertTrue(result['original_archive_payload_excluded'])

    def test_mmb_unparsed_complete_wrong_answers_are_preserved(self):
        group=self.group('mmben');job=group['job_ids'][0]
        score=self.validated[job]['score'];score['unparsed_rows_counted_wrong']=7;score['failed_groups_counted_0']=2
        self.write(self.paths[job]['score'],score)
        self.validated[job]['finished']['score_sha256']=p.sha(self.paths[job]['score'])
        self.write(self.paths[job]['finished'],self.validated[job]['finished'])
        for name in ['score','finished']:self.validated[job]['files'][str(self.paths[job][name])]=p.sha(self.paths[job][name])
        _,published=self.validate(group)
        self.assertEqual(published['value'],65.1234)

    def attach_source(self,group,source,compressed_gt=False):
        job=group['job_ids'][0];validated=self.validated[job];protocol=validated['protocol']
        protocol['source_and_input_sha256'][str(source)]=p.sha(source)
        if compressed_gt:protocol['origin']['score_data']=[{'path':str(source),'sha256':p.sha(source)}]
        else:protocol['parameters']['question_file']=str(source)
        self.write(self.paths[job]['control_protocol'],protocol)
        self.jobs[job]['control_protocol_sha256']=p.sha(self.paths[job]['control_protocol'])
        validated['files'][str(self.paths[job]['control_protocol'])]=p.sha(self.paths[job]['control_protocol'])
        validated['files'][str(source)]=p.sha(source)
        self.write(self.plan_path,self.plan)

    def test_mmb_group_archive_explicitly_omits_embedded_image_source(self):
        group=self.group('mmben');source=self.directory/'MMBench.tsv'
        source.write_text('index\tquestion\thint\tA\tB\tanswer\timage\n'+''.join(
            f'{index}\tquestion\thint\ta\tb\tB\tPRIVATE_BASE64_IMAGE\n' for index in range(4876)))
        self.attach_source(group,source)
        files,score=self.validate(group)
        self.assertNotIn(str(source.relative_to(self.root)),files)
        labels=json.loads(files['derived_inputs/mmben_labels_without_images.json'])
        self.assertEqual(labels['n_rows'],4876)
        self.assertNotIn('PRIVATE_BASE64_IMAGE',str(labels))
        with mock.patch.object(p,'LOCAL',self.directory/'publication'):
            package=p.bundle(group['group_id'],files,[score])
        self.assertEqual(package['manifest']['omitted_source_payloads'][str(source)]['sha256'],p.sha(source))
        self.assertFalse(package['manifest']['contains_weights_or_images'])

    def test_mme_group_archive_uses_canonical_labels_and_original_zip_hash(self):
        group=self.group('mme','pair');source=self.directory/'eval_tool.zip';source.write_bytes(b'fixture original archive')
        self.attach_source(group,source,compressed_gt=True)
        gt={('existence',str(index),'question'): 'yes' for index in range(2374)}
        with mock.patch.object(p,'module',return_value=SimpleNamespace(canonical_gt=lambda path:gt)):
            files,_=self.validate(group)
        self.assertNotIn(str(source.relative_to(self.root)),files)
        labels=json.loads(files['derived_inputs/mme_canonical_ground_truth.json'])
        self.assertEqual(labels['n'],2374)
        references=json.loads(files['EXTERNAL_SOURCE_REFERENCES.json'])
        self.assertEqual(references[str(source)]['sha256'],p.sha(source))
        self.assertFalse(references[str(source)]['archived'])


if __name__=='__main__':unittest.main()

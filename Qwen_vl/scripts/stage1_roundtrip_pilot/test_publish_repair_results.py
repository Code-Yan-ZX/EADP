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
        for name,value in [('diagnostic_metadata',{}),('control_registration',None)]:
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


if __name__=='__main__':unittest.main()

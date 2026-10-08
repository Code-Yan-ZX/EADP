"""CPU-only publication safety fixtures; no real remote or GPU access."""
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

spec=importlib.util.spec_from_file_location('publisher',Path(__file__).with_name('publish_repair_results.py'))
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


class PublisherSafety(unittest.TestCase):
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


if __name__=='__main__':unittest.main()

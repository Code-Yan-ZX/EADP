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


if __name__=='__main__':unittest.main()

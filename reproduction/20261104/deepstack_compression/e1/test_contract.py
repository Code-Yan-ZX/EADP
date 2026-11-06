import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from contract import ARMS, TASKS, validate_panel, validate_quartet, validate_decode_trace, choose_legacy_dev
import runner


def sample_quartet():
    start, original = 2, 1029
    selected = list(range(0, 512, 2))
    retained = list(range(start)) + [start+i for i in selected] + list(range(start+1024, original))
    n = len(retained)
    full = [[[min(i,293) for i in range(original)]] for axis in range(3)]
    p1 = [[list(range(n))] for _ in range(3)]
    p3 = [[[axis[0][i] for i in retained]] for axis in full]
    mask = [[False]*start + [True]*256 + [False]*(n-start-256)]
    common = dict(task='DocVQA_VAL', qid='1', group_id='g', message_sha256='msg', processor={'input':'hash'},
                  image_start=start, original_length=original, ordered_ids=selected,
                  spatial_ids=selected, retained_ids=retained, original_position_ids=full, full_rope_delta=[[-735]],
                  visual_sha256='v', pruned_sha256='p', deepstack_full_sha256=['0','1','2'],
                  prefill_embed_sha256='emb', attention_mask_sha256='mask', output_ids=[7], answer='x')
    result=[]
    for arm in ARMS:
        on=arm.endswith('D1')
        result.append(dict(**copy.deepcopy(common), arm=arm, injected_mask=copy.deepcopy(mask) if on else None,
                           selected_deepstack=['a','b','c'], injected_deepstack=['a','b','c'] if on else None,
                           prefill_position_ids=copy.deepcopy(p1 if arm.startswith('P1') else p3),
                           rope_delta=[[0]] if arm.startswith('P1') else [[33]]))
    return result


class Tests(unittest.TestCase):
    def test_valid_four_arms(self):
        self.assertTrue(validate_quartet(sample_quartet()))

    def test_missing_arm_and_swapped_selection_rejected(self):
        records=sample_quartet()
        with self.assertRaises(ValueError): validate_quartet(records[:3])
        records[1]['spatial_ids'][0]=1
        with self.assertRaises(ValueError): validate_quartet(records)

    def test_bad_mask_missing_stream_and_bad_p3_rejected(self):
        for mutation in ('mask','stream','position'):
            records=sample_quartet()
            if mutation=='mask': records[1]['injected_mask'][0][0]=True
            elif mutation=='stream': records[3]['injected_deepstack'].pop()
            else: records[3]['prefill_position_ids'][1][0][4] += 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): validate_quartet(records)

    def test_missing_nonvisual_position_rejected(self):
        records=sample_quartet()
        for r in records: r['retained_ids'].pop()
        with self.assertRaises(ValueError): validate_quartet(records)

    def test_identical_but_wrong_p3_delta_rejected(self):
        records=sample_quartet()
        records[2]['rope_delta']=records[3]['rope_delta']=[[999999]]
        with self.assertRaises(ValueError): validate_quartet(records)
        records=sample_quartet()
        for r in records: r['full_rope_delta']=[[999231]]
        records[2]['rope_delta']=records[3]['rope_delta']=[[999999]]
        with self.assertRaises(ValueError): validate_quartet(records)

    def test_decode_trace_rejects_wrong_position_and_physical_cache(self):
        trace=[{}]+[{'cache_position':[100+i], 'position_ids':[[[107+i]]]*3,
                     'layer_cache_lengths':[100+i]*4} for i in range(32)]
        self.assertTrue(validate_decode_trace(trace,100,7,layers=4))
        with self.assertRaises(ValueError): validate_decode_trace(trace[:20],100,7,layers=4)
        trace[3]['layer_cache_lengths'][1] += 1
        with self.assertRaises(ValueError): validate_decode_trace(trace,100,7,layers=4)

    def test_panel_unique_and_complete(self):
        panel=[dict(task=t,qid=str(i),group_id=t+str(i),split='DEV') for t in TASKS for i in range(2)]
        self.assertEqual(len(validate_panel(panel,expected=2)),6)
        panel[-1]['group_id']=panel[0]['group_id']
        with self.assertRaises(ValueError): validate_panel(panel,expected=2)

    def test_legacy_anchor_groups_are_excluded(self):
        anchors=[{'index':str(i)} for i in range(16)]
        full=[{'task':'DocVQA_VAL','qid':str(i),'group_id':'old'+str(i)} for i in range(16)]
        records=[{'task':'DocVQA_VAL','qid':str(i),'group_id':'old'+str(i)} for i in range(16)]
        records += [{'task':'DocVQA_VAL','qid':'d'+str(i),'group_id':'dev'+str(i)} for i in range(20)]
        selected, old=choose_legacy_dev(records,full,anchors)
        self.assertEqual(len(selected),16)
        self.assertFalse({r['group_id'] for r in selected} & old)

    def test_image_must_match_audited_question_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'image'; path.write_bytes(b'img')
            image_sha=hashlib.sha256(b'img').hexdigest()
            message=[{'type':'image','value':str(path)}, {'type':'text','value':'q'}]
            digest=hashlib.sha256(json.dumps(message,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
            panel=[{'task':'DocVQA_VAL','qid':'1','group_id':'g'}]
            records=[dict(**panel[0], image_path=str(path), image_sha256=image_sha, message=message, message_sha256=digest)]
            mapping=[dict(**panel[0], split='DEV',status='found',expected_path=str(path),cache_byte_sha256=image_sha)]
            runner.validate_input_records(panel,records,mapping)
            mapping[0]['qid']='2'
            with self.assertRaises((ValueError,KeyError)): runner.validate_input_records(panel,records,mapping)

    def test_e0_proof_checks_current_adapter_and_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            e0=Path(tmp); (e0/'monitor_array16').mkdir()
            (e0/'monitor_verify_array16').mkdir()
            (e0/'monitor_verify_array16/terminal.json').write_text(json.dumps({'reason':'completed','worker_returncode':0,'remaining_process_group':[]}))
            filenames={'worker_sha256':'array_worker.py','adapter_sha256':'adapter.py','run_manifest_sha256':'array16_launch.json',
                       'verifier_sha256':'verify_arrays.py','supervisor_terminal_sha256':'monitor_array16/terminal.json','container_exit_sha256':'array16_exit.json',
                       'evidence_manifest_sha256':'anchor16_evidence_manifest.json','weight_postrun_sha256':'array16_weights_postrun.json',
                       'worker_midrun_sha256':'array16_worker_midrun.json','weight_midrun_sha256':'array16_weight_midrun.json'}
            for name in filenames.values(): (e0/name).write_text('{}')
            (e0/'monitor_array16/terminal.json').write_text(json.dumps({'reason':'completed','worker_returncode':0,'remaining_process_group':[]}))
            (e0/'array16_exit.json').write_text(json.dumps({'returncode':0,'timed_out':False,'gpu_pids_after':[]}))
            (e0/'array16_weights_postrun.json').write_text(json.dumps({'all_match':True}))
            (e0/'array16').mkdir()
            names=[e0/'array16'/f'{i:02d}_{arm}_{kind}.{ext}' for i in range(16) for arm in ('native','on','off','zero','k256_on') for kind in ('natural','forced') for ext in ('json','npz')]
            names += [e0/'array16'/f'{i:02d}_comparison_{kind}.json' for i in range(16) for kind in ('natural','forced')]
            for path in names: path.write_text('evidence')
            (e0/'anchor16_evidence_manifest.json').write_text(json.dumps({str(path):runner.sha(path) for path in names}))
            proof={'status':'passed','n_expected':16,'n_verified':16,'errors':[]}
            proof.update({key:runner.sha(e0/name) for key,name in filenames.items()})
            pp=e0/'proof.json'; pp.write_text(json.dumps(proof))
            previous=runner.E0; runner.E0=e0
            try:
                runner.validate_gate(pp)
                (e0/'adapter.py').write_text('changed')
                with self.assertRaises(ValueError): runner.validate_gate(pp)
            finally: runner.E0=previous

if __name__=='__main__': unittest.main()

"""Synthetic end-to-end checks of the diagnostic's actual decision rules."""
from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

from . import report
from .geometry import build_bank, uniform256_indices

SINGLE_FIRST = min(action["action_id"] for action in build_bank() if action["family"] == "single")
DUAL_IDS = sorted(action["action_id"] for action in build_bank() if action["family"] == "dual")
DUAL_FIRST, DUAL_SECOND = DUAL_IDS[:2]


def actions():
    return build_bank()


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def freeze(path):
    files = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (path/"manifest.json", path/"protocol.json", path/"source.py")}
    value = dict(files=files, schema=1)
    value["freeze_sha256"] = report._json_hash(value)
    _json(path/"freeze.json", value)
    return value["freeze_sha256"]


def generated(arm, bank, identity, answer="yes", cap=128):
    selected = DUAL_FIRST
    n = 1024 if arm == "full" else 256
    source = 256 if arm == "rres512" else 1024
    action = selected if arm in report.CONTROLS else arm
    dependent = arm in report.CONTROLS
    geometry = (report._controls(json.dumps(bank[selected],sort_keys=True))[arm]
                if dependent else bank.get(arm))
    keep = geometry["keep_idx"] if geometry is not None else (
        uniform256_indices() if arm == "uniform256" else list(range(n)))
    return dict(item_id=identity, arm=arm, text=answer, token_ids=[1], truncated=False,
        keep_idx=keep, actual_visual_count=n, source_visual_count=source,
        ds_lengths=[n]*3, positions_sha256="a"*64, gradients_enabled=False,
        prefill_length=n+10, cache_length=n+10, decode_positions=[],
        max_new_tokens=cap, selector_answer_dependent=dependent,
        geometry=geometry,
        selection_scope="offline_oracle_control" if dependent else "fixed_public_input",
        selected_dual_action=selected if dependent else None,
        audit=dict(native=True, positions_preserved=True, ds_streams=3, fresh_kv=True,
                   physical_visual_count=n, answer_in_selection=dependent))


def fixture(path, counts=(14, 14), strict_per_dataset=4):
    path.mkdir(parents=True)
    bank = {a["action_id"]: a for a in actions()}
    arms = [*report.BASELINES, *bank, *report.CONTROLS]
    datasets = {}
    for name, count in zip(("InfoVQA_TRAIN", "TextVQA_TRAIN"), counts):
        def item(identity):
            return dict(item_id=str(identity), parent_id=name+"_"+str(identity),
                question="Is it yes?", image_path="/unused.png",
                answers=["yes"]*(10 if name.startswith("Text") else 1))
        datasets[name] = dict(items=[item(i) for i in range(count)], dev=[item("dev_0"), item("dev_1")])
    manifest = dict(datasets=datasets)
    timing_arms = [*report.BASELINES, SINGLE_FIRST, DUAL_FIRST]
    protocol = dict(arms=arms, baseline_arms=list(report.BASELINES), action_bank=list(bank.values()),
        max_new_tokens=128, n=1024, k=256,
        timing=dict(arms=timing_arms, caps=[1,16], blocks=3,
            item_ids_by_dataset={name:[i["item_id"] for i in data["dev"]] for name,data in datasets.items()}))
    _json(path/"manifest.json", manifest)
    _json(path/"protocol.json", protocol)
    (path/"source.py").write_text("# frozen synthetic source\n")
    freeze_hash = freeze(path)
    (path/"evaluation").mkdir()
    connection = sqlite3.connect(path/"evaluation/predictions.sqlite3")
    connection.executescript("""
        CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);
        CREATE TABLE predictions(dataset TEXT,item_id TEXT,arm TEXT,payload TEXT,PRIMARY KEY(dataset,item_id,arm));
        CREATE TABLE timings(dataset TEXT,item_id TEXT,arm TEXT,cap INTEGER,block INTEGER,payload TEXT,PRIMARY KEY(dataset,item_id,arm,cap,block));
        CREATE TABLE events(id INTEGER PRIMARY KEY,timestamp TEXT,payload TEXT);
    """)
    connection.execute("INSERT INTO metadata VALUES('freeze_sha256',?)",(freeze_hash,))
    for name,data in datasets.items():
        for item in data["items"]:
            for arm in arms:
                success = arm == "full" or arm == DUAL_FIRST or (arm.startswith("single_") and int(item["item_id"]) >= strict_per_dataset)
                row = generated(arm, bank, item["item_id"], "yes" if success else "no")
                row.update(dataset=name, parent_id=item["parent_id"], freeze_sha256=freeze_hash)
                row["score"] = report.score_item(name,row["text"],item["answers"])
                connection.execute("INSERT INTO predictions VALUES(?,?,?,?)",(name,item["item_id"],arm,json.dumps(row)))
        for item in data["dev"]:
            for arm in timing_arms:
                for cap in (1,16):
                    for block in range(3):
                        row = generated(arm,bank,item["item_id"])
                        row.update(dataset=name,parent_id=item["parent_id"],freeze_sha256=freeze_hash,
                            cap=cap,block=block,token_ids=[1]*cap,generated_tokens=cap,elapsed_ms=100.+block)
                        connection.execute("INSERT INTO timings VALUES(?,?,?,?,?,?)",(name,item["item_id"],arm,cap,block,json.dumps(row)))
    connection.commit()
    connection.close()
    return protocol


def mutate(path, dataset, item_id, arm, update):
    connection = sqlite3.connect(path/"evaluation/predictions.sqlite3")
    where = (dataset,str(item_id),arm)
    value = json.loads(connection.execute("SELECT payload FROM predictions WHERE dataset=? AND item_id=? AND arm=?",where).fetchone()[0])
    update(value)
    connection.execute("UPDATE predictions SET payload=? WHERE dataset=? AND item_id=? AND arm=?",(json.dumps(value),*where))
    connection.commit()
    connection.close()


def test_complete_balanced_oracle_and_strict_controls_reach_only_diagnostic_go(tmp_path):
    path = tmp_path/"run"
    fixture(path)
    result = report.build_report(path,draws=256)
    assert result["status"] == "complete" and result["decision"] == "GO_DIAGNOSIS_ONLY"
    assert result["n_expected"] == result["n_observed"] == 28*37
    assert result["timing"]["n_observed"] == result["timing"]["n_expected"] == 216
    assert result["freeze"]["status"] == "verified" and not result["missing_keys"]
    for data in result["datasets"].values():
        assert data["n_strict"] == 4 and data["n_full_good_rres_bad"] == 14
        assert data["candidate_counts"] == {"single":12,"dual":12}
        assert data["dual_minus_single"]["delta"] == pytest.approx(4/14)
        assert data["full_minus_rres"]["delta"] == 1.
        assert data["finite_bank_oracle_means"]["dual"] == 1.
        assert data["evidence"][0]["chosen_dual_action"] == DUAL_FIRST
        assert data["evidence"][0]["raw_evidence"][report.CONTROLS[0]]["selector_answer_dependent"] is True
    assert "not a deployable" in result["timing"]["limitations"].lower()
    assert (path/"evaluation/report.json").exists() and "GO_DIAGNOSIS_ONLY" in (path/"evaluation/report.md").read_text()


def test_pool_uses_equal_dataset_weights_not_question_weights(tmp_path):
    path = tmp_path/"run"
    fixture(path,counts=(14,28))
    result = report.build_report(path,draws=32,write=False)
    assert result["equal_dataset_pool"]["delta"] == pytest.approx((4/14+4/28)/2)
    assert result["equal_dataset_pool"]["delta"] != pytest.approx(8/42)


def test_missing_control_keeps_original_denominator_and_blocks_decision(tmp_path):
    path = tmp_path/"run"
    fixture(path)
    connection = sqlite3.connect(path/"evaluation/predictions.sqlite3")
    connection.execute("DELETE FROM predictions WHERE dataset='InfoVQA_TRAIN' AND item_id='0' AND arm=?",(report.CONTROLS[0],))
    connection.commit()
    connection.close()
    result = report.build_report(path,draws=32,write=False)
    assert result["status"] == "incomplete" and result["decision"] == "INCOMPLETE"
    assert result["n_expected"] == 28*37 and result["n_observed"] == 28*37-1
    assert len(result["missing_keys"]) == 1
    assert result["datasets"]["InfoVQA_TRAIN"]["n_strict"] == 0


@pytest.mark.parametrize("change",[
    lambda r:r.update(selector_answer_dependent=False),
    lambda r:r["audit"].update(answer_in_selection=False),
    lambda r:r.update(selection_scope="online"),
    lambda r:r.update(keep_idx=list(range(256))),
    lambda r:r.update(ds_lengths=[256,256]),
    lambda r:r.update(cache_length=0),
    lambda r:r.update(score=1),
])
def test_answer_dependence_geometry_native_and_score_corruption_rejected(tmp_path,change):
    path = tmp_path/"run"
    fixture(path)
    mutate(path,"InfoVQA_TRAIN","0",report.CONTROLS[0],change)
    with pytest.raises(ValueError):
        report.build_report(path,draws=8,write=False)


def test_control_selection_obeys_deterministic_qa_ties(tmp_path):
    path = tmp_path/"run"
    fixture(path)
    # Both duals are correct; lexical first must still drive all controls.
    mutate(path,"InfoVQA_TRAIN","0",DUAL_SECOND,lambda r:r.update(text="yes",score=1.))
    result = report.build_report(path,draws=8,write=False)
    assert result["datasets"]["InfoVQA_TRAIN"]["evidence"][0]["chosen_dual_action"] == DUAL_FIRST
    def change(row):
        bank = {action["action_id"]:action for action in actions()}
        control = report._controls(json.dumps(bank[DUAL_SECOND],sort_keys=True))[report.CONTROLS[0]]
        row.update(selected_dual_action=DUAL_SECOND,keep_idx=control["keep_idx"],geometry=control)
    mutate(path,"InfoVQA_TRAIN","0",report.CONTROLS[0],change)
    with pytest.raises(ValueError,match="tie breaking"):
        report.build_report(path,draws=8,write=False)


def test_one_good_control_disqualifies_strict_case_but_truncations_stay(tmp_path):
    path = tmp_path/"run"
    fixture(path)
    mutate(path,"InfoVQA_TRAIN","0",report.CONTROLS[3],lambda r:r.update(text="yes",score=1.))
    mutate(path,"TextVQA_TRAIN","0",report.CONTROLS[0],lambda r:r.update(token_ids=[1]*128,truncated=True))
    result = report.build_report(path,draws=32,write=False)
    assert result["datasets"]["InfoVQA_TRAIN"]["n_strict"] == 3
    assert result["datasets"]["TextVQA_TRAIN"]["n_strict"] == 4
    assert result["n_truncated"] == 1 and result["n_expected"] == result["n_observed"]
    assert result["decision"] == "STOP_THIS_GEOMETRY"


def test_preregistered_gap_hold_and_no_general_multiregion_rejection(tmp_path):
    path = tmp_path/"run"
    fixture(path,counts=(11,14))
    result = report.build_report(path,draws=32,write=False)
    assert result["decision"] == "HOLD_INSUFFICIENT_GAP"
    assert "STOP applies only to these geometries" in result["interpretation"]


def test_freeze_corruption_is_detected_before_scores(tmp_path):
    path = tmp_path/"run"
    fixture(path)
    (path/"source.py").write_text("# changed\n")
    with pytest.raises(ValueError,match="Frozen source"):
        report.build_report(path,draws=8,write=False)


def test_exact_metrics_keep_anls_boundary_and_ten_vote_multiplicity():
    assert report.score_item("InfoVQA_TRAIN","ac",["ab"]) == .5
    assert report.score_item("TextVQA_TRAIN","yes",["yes"]*2+["no"]*8) == pytest.approx(.6)
    with pytest.raises(ValueError,match="ten"):
        report.score_item("TextVQA_TRAIN","yes",["yes"])


def test_bank_must_have_equal_candidate_opportunity():
    with pytest.raises(ValueError,match="equal banks"):
        report._bank(dict(action_bank=actions()[:-1]))

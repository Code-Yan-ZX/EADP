"""Join qualified EADP baselines to archived EADP paper tables; CPU only."""
from pathlib import Path
from decimal import Decimal
import csv
import hashlib
import html
import json

OUT = Path(__file__).resolve().parent
EVIDENCE = OUT.parent / "evidence.json"
SUPPLEMENT = OUT / "supplemental_provenance.json"
NAMES = {"v15": "LLaVA-1.5-7B", "next": "LLaVA-NeXT-7B", "qwen": "Qwen3-VL-8B"}
TASKS = {
    "v15": ["VQAv2", "GQA", "VizWiz", "SQA_IMG", "TextVQA", "POPE", "MME", "MMB_EN", "MMB_CN", "MMVet", "Avg9"],
    "next": ["VQAv2", "GQA", "VizWiz", "SQA_IMG", "TextVQA", "POPE", "MME", "MMB_EN", "MMB_CN", "MMVet", "Avg9"],
    "qwen": ["TextVQA", "ChartQA", "AI2D", "OCRBench", "HallusionBench", "MME", "MMB_EN", "MMB_CN", "DocVQA", "InfoVQA", "Avg10"],
}
BUDGETS = {"v15": [128, 64, 32], "next": [640, 320, 160], "qwen": [512, 256, 128]}


def read(path):
    return json.loads(path.read_text())


def key(record):
    budget = record.get("budget_nominal")
    if budget is None:
        budget = record["budget_cli"] * (5 if record["model"] == "next" else 1)
    return record["model"], budget, record["task"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    evidence = read(EVIDENCE)["records"]
    supplemental = read(SUPPLEMENT)
    paper = {key(r): r for r in evidence if r["origin"] == "eadp_paper_reported" and r["method"] == "EADP"}
    base = [r for r in evidence if r["origin"] == "local_experiment" and r["method"].split(" (")[0] == "EADP"]
    local = {}
    for origin, records in (("v2_initial_evidence", base), ("newly_verified_supplement", supplemental["records"])):
        for record in records:
            assert record["method"].split(" (")[0] == "EADP"
            k = key(record)
            assert k not in local, ("Duplicate local baseline", k)
            assert k in paper, ("Missing paper counterpart", k)
            local[k] = (record, origin)
    unresolved = {key(r): r for r in supplemental["unresolved"]}
    rows = []
    for model in NAMES:
        for task in TASKS[model]:
            for budget in BUDGETS[model]:
                k = model, budget, task
                p = paper[k]
                entry = local.get(k)
                r, origin = entry if entry else ({}, "")
                caveat = r.get("protocol_caveat", r.get("caveat", ""))
                if k in unresolved:
                    caveat = unresolved[k]["protocol_caveat"]
                delta = Decimal(str(r["value"])) - Decimal(str(p["value"])) if entry else None
                rows.append({
                    "model": NAMES[model], "model_key": model,
                    "budget_nominal": budget,
                    "budget_cli": budget // 5 if model == "next" else budget,
                    "task": task, "paper_split": p["split"],
                    "paper_value": p["value"], "local_value": r.get("value"),
                    "local_minus_paper": str(delta) if delta is not None else None,
                    "delta_unit": "raw_score_points" if task in ("MME", "OCRBench") else "percentage_points",
                    "status": "qualified_local" if entry else "unresolved_bank_identity" if k in unresolved else "no_qualified_local_in_scope",
                    "evidence_scope": origin, "local_evidence_id": r.get("id"),
                    "paper_evidence_id": p["id"], "paper_source": p["source"],
                    "local_score_source": r.get("score_source", r.get("source")),
                    "raw_prediction": r.get("raw_prediction"), "protocol_caveat": caveat,
                })
    assert len(paper) == len(rows) == 99
    assert len(local) == sum(r["status"] == "qualified_local" for r in rows) == 29
    assert all(r["local_value"] is None and r["local_minus_paper"] is None for r in rows if r["task"].startswith("Avg"))
    manifest = {
        "date_beijing": "2026-10-28", "definition": "difference = local reproduction - EADP paper",
        "paper_cells": len(rows), "qualified_local_pairs": len(local),
        "initial_v2_local_pairs": len(base), "newly_verified_supplemental_pairs": len(supplemental["records"]),
        "inputs": [{"path": str(p), "sha256": digest(p)} for p in (EVIDENCE, SUPPLEMENT)],
        "limits": [
            "Paper values are rounded as published; differences use unrounded stored local scores.",
            "This lists EADP baselines; AnchorZip results and native-restored Qwen runs are separate methods/settings.",
            "VizWiz local controls use alpha0/beta1/max128; do not imply identical unpublished paper argv.",
            "Qwen HallusionBench remains withheld because generation-time selection-bank identity is unresolved.",
            "Missing EADP local components prevent a complete local average; no available-task mean is substituted.",
            "Only the current table scope and verified supplements are joined; missing does not prove no historical experiment exists.",
            "No new GPU experiment or benchmark re-scoring was run for this comparison.",
        ], "rows": rows,
    }
    (OUT / "comparison.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    with (OUT / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in rows)
    notes = "每格为：论文 → 本机（本机−论文）；—表示本轮没有可采用的本机复现值。MME、OCRBench差值为原始分数，其余为百分点。论文值按原文精度，本机与差值保留三位小数。"
    page = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>EADP V2 论文与本机复现对比</title>',
            '<style>body{font:15px/1.6 system-ui,sans-serif;margin:32px;color:#202020}table{border-collapse:collapse;width:100%;margin:20px 0 32px}th,td{padding:8px 12px;border-bottom:1px solid #ddd;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#eee}small,p{max-width:1100px}a{color:#1651a0}</style>',
            '<h1>EADP V2：论文与本机复现对比</h1><p>' + notes + '</p>']
    compact = []
    for model in NAMES:
        page.append('<h2>' + NAMES[model] + '</h2><table><thead><tr><th>任务</th>' + ''.join('<th>' + str(b) + ' tokens</th>' for b in BUDGETS[model]) + '</tr></thead><tbody>')
        compact.append(NAMES[model] + '\n| 任务 | ' + ' | '.join(str(b) for b in BUDGETS[model]) + ' |\n|---|' + '---|' * 3)
        for task in TASKS[model]:
            cells = []
            for budget in BUDGETS[model]:
                r = next(r for r in rows if r['model_key'] == model and r['budget_nominal'] == budget and r['task'] == task)
                text = f"{r['paper_value']:g} → —"
                if r['local_value'] is not None:
                    mark = '†' if task == 'VizWiz' else ''
                    text = f"{r['paper_value']:g} → {r['local_value']:.3f}{mark} ({Decimal(r['local_minus_paper']):+.3f})"
                cells.append(text)
            page.append('<tr><td>' + task + '</td>' + ''.join('<td>' + html.escape(c) + '</td>' for c in cells) + '</tr>')
            compact.append('| ' + task + ' | ' + ' | '.join(cells) + ' |')
        page.append('</tbody></table>')
    page.append('<p>† VizWiz：4319题，EADP发布评分器；本机控制参数alpha=0、beta=1、max128。NeXT预算为名义总数，640/320/160对应CLI128/64/32。SQA使用本机已审计输入；POPE为实际8910题（random2910）。MMBench EN本机使用V11，未确认与论文题集版本完全一致。Qwen本机采用legacy（DeepStack关闭、1D位置），论文完整运行参数未核实。跨论文数值差不等于严格同配置误差的归因。</p>')
    page.append('<p>补充核验新增9项Qwen K256及2项真实EADP VizWiz。HallusionBench的预测/评分存在，但生成时剪枝bank身份未闭合，不纳入正式本机值。完整本机平均分保持缺失。</p>')
    page.append('<p><a href="comparison.csv">完整CSV</a> · <a href="comparison.json">逐项来源与比较说明</a> · <a href="supplemental_provenance.json">本次补充核验来源</a></p></html>')
    (OUT / "comparison.html").write_text('\n'.join(page) + '\n')
    print('\n\n'.join(compact))
    print(f"\nValidated {len(rows)} paper cells and {len(local)} qualified local pairs.")


if __name__ == "__main__":
    main()

"""CPU audit of currently referenced, valid-geometry binding diagnostics.

No diagnostic directory is searched. Image-level mechanisms are independent of
question correctness: the teacher only defines the registered accepted subset.
Missing diagnostics remain pending and never count as an absent mechanism.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

if __package__:
    from .analyze import correct, dataset_stratum
else:
    from analyze import correct, dataset_stratum


DEFAULT = Path(__file__).resolve().parents[2] / "outputs/binding_oracle"
GEOMETRY = "native_smart_resize_then_square_v2"
ORACLE = "field_oracle_ppe"
CONTROL = "field_anchor_control_ppe"
SIDES = ("original", "swapped")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def teacher_status(item, row):
    questions = []
    for side in SIDES:
        for field in (0, 1):
            entry = row.get("members", {}).get(f"{side}_f{field}", {}).get("teacher", {})
            if "prediction" not in entry:
                return "pending"
            questions.append(correct(entry["prediction"], item["fields"][field]["answers_" + side]))
    return "accepted" if all(questions) else "rejected"


def referenced_diag(row, side, arm, directory, image, cache):
    references = []
    for field in (0, 1):
        entry = row.get("members", {}).get(f"{side}_f{field}", {}).get(arm, {})
        path = (entry.get("selector") or {}).get("diagnostic_path")
        if path:
            path = Path(path)
            if not path.is_absolute():
                path = directory / path
            references.append((field, path.resolve()))
    require(references, f"no currently referenced {arm} diagnostic")
    paths = {path for _, path in references}
    require(len(paths) == 1, f"the two questions reference inconsistent {arm} diagnostics")
    path = next(iter(paths))
    if path not in cache:
        cache[path] = json.loads(path.read_text())
    diag = cache[path]
    require(diag.get("geometry_version") == GEOMETRY, f"{arm} diagnostic geometry is not {GEOMETRY}")
    require(Path(diag.get("image", "")).resolve() == Path(image).resolve(), f"{arm} diagnostic references another image")
    require(diag.get("valid") is True, f"{arm} diagnostic is not explicitly valid")
    return diag, dict(path=str(path), referenced_by_fields=[field for field, _ in references])


def unpack(diag):
    n, k = diag.get("N"), diag.get("K")
    require(n == 1024 and k == 256, "diagnostic does not use native 1024 -> 256 budget")
    dominant, anchors = diag.get("dominant_idx"), diag.get("anchor_idx")
    require(isinstance(dominant, list) and isinstance(anchors, list), "missing dominant/anchor indices")
    require(len(dominant) == 205 and len(anchors) == 51, "incorrect D205/C51 budget")
    keep = sorted(dominant + anchors)
    require(len(set(keep)) == k and min(keep) >= 0 and max(keep) < n, "nonunique or invalid kept indices")
    reps, rep_labels = diag.get("ppe_representatives"), diag.get("ppe_representative_field_labels")
    require(isinstance(reps, list) and isinstance(rep_labels, list)
            and len(reps) == len(rep_labels) == k, "missing aligned PPE representatives")
    require(all(len(ids) == len(labels) == 4 for ids, labels in zip(reps, rep_labels)), "PPE rows do not contain four slots")
    require(all(label in (0, 1, 2) for labels in rep_labels for label in labels), "invalid value-region labels")
    raw_clusters = diag.get("clusters")
    require(isinstance(raw_clusters, dict), "missing source cluster membership")
    clusters = {int(anchor): members for anchor, members in raw_clusters.items()}
    require(set(clusters) == set(anchors), "context cluster keys do not match anchor indices")
    owner = {token: token for token in dominant}
    for anchor, members in clusters.items():
        require(isinstance(members, list) and anchor in members and len(members) == len(set(members)), "invalid contextual cluster")
        for token in members:
            require(isinstance(token, int) and 0 <= token < n and token not in owner, "source member has multiple or invalid assignments")
            owner[token] = anchor
    require(set(owner) == set(range(n)), "not all 1024 source tokens are assigned once")
    require(diag.get("all_tokens_accounted_once") is True and diag.get("unassigned_count") == 0,
            "source-token accounting gate is missing or failed")
    by_token = {token: (ids, labels) for token, ids, labels in zip(keep, reps, rep_labels)}
    for token in dominant:
        require(by_token[token][0] == [token] * 4 and len(set(by_token[token][1])) == 1,
                "dominant token does not have a repeated singleton position")
    for anchor, members in clusters.items():
        require(all(token in members for token in by_token[anchor][0]), "PPE representative is not a source cluster member")
    return dict(dominant=dominant, anchors=anchors, keep=keep, clusters=clusters,
                owner=owner, by_token=by_token, reps=reps, rep_labels=rep_labels)


def oracle_mechanism(diag):
    require(diag.get("variant") == "field_oracle" and diag.get("constrained_assignment") is True,
            "referenced oracle is not field-constrained")
    require(diag.get("cross_label_assignees") == 0, "field oracle homogeneity gate is missing or failed")
    info = unpack(diag)
    labels = {}
    for token in info["dominant"]:
        labels[token] = info["by_token"][token][1][0]
    for anchor, members in info["clusters"].items():
        row_labels = info["by_token"][anchor][1]
        require(len(set(row_labels)) == 1, "field-oracle cluster has mixed representative labels")
        labels.update({token: row_labels[0] for token in members})
    observed = Counter(labels.values())
    require([observed[field] for field in (1, 2)] == diag.get("field_token_counts"),
            "reconstructed source value-token counts disagree with stored geometry counts")
    require([sum(labels[token] == field for token in info["keep"]) for field in (1, 2)]
            == diag.get("field_tokens_kept"), "reconstructed retained value-token counts disagree")
    require(all(observed[field] > 0 for field in (1, 2)), "a value region has no native token")
    dominant_set = set(info["dominant"])
    fields = []
    for field in (1, 2):
        tokens = {token for token, label in labels.items() if label == field}
        dominant = sorted(tokens & dominant_set)
        contextual = sorted(tokens - dominant_set)
        field_anchors = [anchor for anchor in info["anchors"] if labels[anchor] == field]
        merging = [anchor for anchor in field_anchors if len(info["clusters"][anchor]) > 1]
        fields.append(dict(field_label=field, source_tokens=len(tokens), source_token_ids=sorted(tokens),
                           dominant_singleton_tokens=len(dominant), dominant_token_ids=dominant,
                           contextual_tokens=len(contextual), contextual_token_ids=contextual,
                           contextual_anchor_tokens=len(field_anchors), contextual_anchor_ids=field_anchors,
                           contextual_non_anchor_tokens=len(contextual) - len(field_anchors),
                           contextual_clusters_requiring_merge=len(merging), merging_anchor_ids=merging,
                           all_tokens_dominant=not contextual,
                           has_contextual_tokens=bool(contextual), needs_contextual_merge=bool(merging)))
    result = dict(fields=fields, source_value_tokens=sum(row["source_tokens"] for row in fields),
                  dominant_value_tokens=sum(row["dominant_singleton_tokens"] for row in fields),
                  contextual_value_tokens=sum(row["contextual_tokens"] for row in fields),
                  non_anchor_contextual_value_tokens=sum(row["contextual_non_anchor_tokens"] for row in fields),
                  all_value_tokens_dominant=all(row["all_tokens_dominant"] for row in fields),
                  any_contextual_value_tokens=any(row["has_contextual_tokens"] for row in fields),
                  value_contextual_merge_required=any(row["needs_contextual_merge"] for row in fields))
    return result, info, labels


def position_coverage(info, labels, diag):
    for ids, values in zip(info["reps"], info["rep_labels"]):
        require(all(labels[token] == field for token, field in zip(ids, values)),
                "PPE representative field labels disagree across same-image diagnostics")
    counts = []
    for field in (1, 2):
        ids = {token for row in info["reps"] for token in row if labels[token] == field}
        slots = sum(value == field for row in info["rep_labels"] for value in row)
        counts.append(dict(field_label=field, position_slots=slots,
                           unique_position_ids=len(ids), position_ids=sorted(ids),
                           output_rows_with_field_position=sum(field in row for row in info["rep_labels"])))
    require([row["position_slots"] for row in counts] == diag.get("ppe_field_position_slots"),
            "recomputed PPE slot counts disagree with saved diagnostic")
    return dict(fields=counts, both_value_regions_covered=all(row["position_slots"] > 0 for row in counts))


def compare_assignments(oracle, control, labels, oracle_diag, control_diag):
    require(control_diag.get("variant") == "field_anchor_control" and control_diag.get("constrained_assignment") is False,
            "referenced control is not unrestricted field-anchor control")
    require(oracle["dominant"] == control["dominant"] and oracle["anchors"] == control["anchors"],
            "primary control and oracle have different dominant or anchor indices")
    changed = [token for token in range(1024) if oracle["owner"][token] != control["owner"][token]]
    before, after = position_coverage(control, labels, control_diag), position_coverage(oracle, labels, oracle_diag)
    fields = []
    for field, old, new in zip((1, 2), before["fields"], after["fields"]):
        fields.append(dict(field_label=field,
                           changed_assignment_value_tokens=sum(labels[token] == field for token in changed),
                           source_tokens_sent_outside_field_before=sum(
                               labels[token] == field and labels[anchor] != field for token, anchor in control["owner"].items()),
                           position_slots_before=old["position_slots"], position_slots_after=new["position_slots"],
                           position_slot_delta=new["position_slots"] - old["position_slots"],
                           unique_position_ids_before=old["unique_position_ids"], unique_position_ids_after=new["unique_position_ids"],
                           position_coverage_gained=old["position_slots"] == 0 < new["position_slots"],
                           position_coverage_lost=new["position_slots"] == 0 < old["position_slots"]))
    def mixed_clusters(info):
        return sum(len({labels[token] for token in members}) > 1 for members in info["clusters"].values())
    def background_into_value(info):
        return sum(labels[anchor] in (1, 2) and labels[token] == 0 for token, anchor in info["owner"].items())
    return dict(same_anchors=True, same_dominants=True, fields=fields,
                source_tokens_assignment_changed=len(changed), changed_source_token_ids=changed,
                value_tokens_assignment_changed=sum(labels[token] in (1, 2) for token in changed),
                background_tokens_assignment_changed=sum(labels[token] == 0 for token in changed),
                mixed_label_context_clusters_before=mixed_clusters(control),
                mixed_label_context_clusters_after=mixed_clusters(oracle),
                background_members_in_value_anchors_before=background_into_value(control),
                background_members_in_value_anchors_after=background_into_value(oracle),
                ppe_position_ids_changed=sum(a != b for old, new in zip(control["reps"], oracle["reps"]) for a, b in zip(old, new)),
                ppe_value_position_ids_changed=sum(a != b and (labels[a] in (1, 2) or labels[b] in (1, 2))
                                                   for old, new in zip(control["reps"], oracle["reps"]) for a, b in zip(old, new)),
                coverage_before=before, coverage_after=after)


def distribution(values):
    if not values:
        return dict(n=0, total=None, min=None, median=None, mean=None, max=None)
    return dict(n=len(values), total=sum(values), min=min(values), median=statistics.median(values),
                mean=statistics.mean(values), max=max(values))


def summarize(rows):
    oracle = [row["oracle"] for row in rows if row["oracle_status"] == "complete"]
    paired = [row["comparison"] for row in rows if row["comparison_status"] == "complete"]
    return dict(expected_images=len(rows), oracle_images_complete=len(oracle), oracle_images_pending=len(rows) - len(oracle),
                paired_diagnostics_complete=len(paired), paired_diagnostics_pending=len(rows) - len(paired),
                images_all_value_tokens_dominant=sum(row["all_value_tokens_dominant"] for row in oracle),
                images_with_contextual_value_tokens=sum(row["any_contextual_value_tokens"] for row in oracle),
                images_requiring_contextual_value_merge=sum(row["value_contextual_merge_required"] for row in oracle),
                source_value_tokens=distribution([row["source_value_tokens"] for row in oracle]),
                dominant_value_tokens=distribution([row["dominant_value_tokens"] for row in oracle]),
                contextual_value_tokens=distribution([row["contextual_value_tokens"] for row in oracle]),
                changed_source_tokens=distribution([row["source_tokens_assignment_changed"] for row in paired]),
                changed_value_tokens=distribution([row["value_tokens_assignment_changed"] for row in paired]),
                changed_background_tokens=distribution([row["background_tokens_assignment_changed"] for row in paired]),
                images_with_changed_value_assignments=sum(row["value_tokens_assignment_changed"] > 0 for row in paired),
                images_with_changed_value_positions=sum(row["ppe_value_position_ids_changed"] > 0 for row in paired),
                both_regions_have_ppe_position_before=sum(row["coverage_before"]["both_value_regions_covered"] for row in paired),
                both_regions_have_ppe_position_after=sum(row["coverage_after"]["both_value_regions_covered"] for row in paired),
                images_gaining_a_value_region_position=sum(any(field["position_coverage_gained"] for field in row["fields"]) for row in paired),
                images_losing_a_value_region_position=sum(any(field["position_coverage_lost"] for field in row["fields"]) for row in paired),
                source_tokens_outside_own_value_region_before=distribution([
                    sum(field["source_tokens_sent_outside_field_before"] for field in row["fields"]) for row in paired]),
                background_members_in_value_anchors_before=distribution([row["background_members_in_value_anchors_before"] for row in paired]),
                background_members_in_value_anchors_after=distribution([row["background_members_in_value_anchors_after"] for row in paired]))


def audit(predictions, manifest, directory):
    ids = [item["pair_id"] for item in manifest["records"]]
    require(len(ids) == len(set(ids)), "duplicate manifest pair IDs")
    active = set(predictions.get("active_pair_ids", ids))
    prediction_rows = [row for row in predictions.get("records", []) if row["pair_id"] in active]
    require(len(prediction_rows) == len({row["pair_id"] for row in prediction_rows}), "duplicate active prediction pair IDs")
    byid = {row["pair_id"]: row for row in prediction_rows}
    cache, records, screen = {}, [], Counter()
    screen_ids = {key: [] for key in ("accepted", "rejected", "pending")}
    for item in manifest["records"]:
        row = byid.get(item["pair_id"], {})
        status = teacher_status(item, row)
        screen[status] += 1
        screen_ids[status].append(item["pair_id"])
        if status != "accepted":
            continue
        for side in SIDES:
            result = dict(pair_id=item["pair_id"], side=side, image=item[side + "_image"],
                          source_document_id=item["source_document_id"], stratum=dataset_stratum(item),
                          oracle_status="pending", comparison_status="pending", pending=[])
            records.append(result)
            try:
                require(predictions.get("geometry_version") == GEOMETRY, "active predictions do not declare exact geometry v2")
                diag, reference = referenced_diag(row, side, ORACLE, directory, result["image"], cache)
                mechanism, info, labels = oracle_mechanism(diag)
                result.update(oracle_status="complete", oracle_reference=reference, oracle=mechanism)
            except (OSError, ValueError, TypeError, KeyError) as error:
                result["pending"].append(dict(stage="oracle", reason=str(error)))
                continue
            try:
                control_diag, reference = referenced_diag(row, side, CONTROL, directory, result["image"], cache)
                comparison = compare_assignments(info, unpack(control_diag), labels, diag, control_diag)
                result.update(comparison_status="complete", control_reference=reference, comparison=comparison)
            except (OSError, ValueError, TypeError, KeyError) as error:
                result["pending"].append(dict(stage="same_anchor_comparison", reason=str(error)))
    return dict(schema_version=1, generated_at=datetime.now(timezone.utc).isoformat(),
                snapshot_stage=predictions.get("stage"), required_geometry_version=GEOMETRY,
                active_geometry_version=predictions.get("geometry_version"),
                teacher_screen=dict(planned=len(ids), active=len(active),
                                    **{key: screen[key] for key in ("accepted", "rejected", "pending")},
                                    pair_ids=screen_ids),
                summary=summarize(records),
                strata={name: summarize([row for row in records if row["stratum"] == name])
                        for name in sorted({row["stratum"] for row in records})},
                records=records, referenced_diagnostic_files=len(cache),
                interpretation=[
                    "Each accepted page contributes two images; image counts are descriptive, not independent statistical samples.",
                    "Dominant PPE rows are singleton positions; contextual rows align to sorted dominant+anchor indices.",
                    "Non-representative source labels are inferred from field-oracle cluster homogeneity, conditional on cross_label_assignees=0; full counts are checked against saved geometry counts.",
                    "Missing or invalid currently referenced diagnostics remain pending; no directory glob or archived diagnostic is used.",
                    "Changes in aggregation and PPE representative positions are reported separately; this does not establish content-only causality or answer improvement.",
                ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=str(DEFAULT / "predictions.json"))
    parser.add_argument("--manifest", default=str(DEFAULT / "data/combined_manifest.json"))
    parser.add_argument("--out")
    args = parser.parse_args()
    prediction_path, manifest_path = Path(args.predictions), Path(args.manifest)
    prediction_bytes, manifest_bytes = prediction_path.read_bytes(), manifest_path.read_bytes()
    result = audit(json.loads(prediction_bytes), json.loads(manifest_bytes), prediction_path.parent)
    result["inputs"] = dict(predictions=str(prediction_path.resolve()), manifest=str(manifest_path.resolve()),
                            predictions_sha256=hashlib.sha256(prediction_bytes).hexdigest(),
                            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest())
    output = Path(args.out) if args.out else prediction_path.parent / "mechanism_audit.json"
    atomic_json(output, result)
    print(json.dumps(dict(output=str(output.resolve()), teacher_screen=result["teacher_screen"],
                          summary=result["summary"]), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

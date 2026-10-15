"""TEMP-ONLY hypothetical attestations/counts, NOT training/acceptance evidence.

Declared-real metadata exercises the happy-path software contracts using stub
bytes. No models are executed; fixture values/criteria are NOT customer SLA.
Never publish these fixtures as a trained candidate or use production registry.
"""

from datetime import datetime, timezone
import json

from backend.app.core.release_governance import ArtifactPin, ManualApproval, METRIC_DIRECTIONS
from backend.app.services import release_governance as governance

NOW = datetime(2026, 10, 15, tzinfo=timezone.utc)


def make_bundle(root, monkeypatch):
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(governance, "ROOT", root)
    counter = [0]
    def save(name, value):
        counter[0] += 1
        path = root / f"{counter[0]}_{name}"
        raw = governance.canonical_json(value) if not isinstance(value, bytes) else value
        path.write_bytes(raw)
        return {"path": str(path), "sha256": governance.digest(raw)}
    def sample(name, fresh, index=0):
        return {"sample_id": name, "fabric_id": "fixture_fabric", "source_id": name,
            "source_sha256": governance.digest((name + '_root').encode()),
            "ancestor_source_ids": [], "ancestor_source_sha256s": [],
            "lot_id": "fresh_lot" if fresh else "train_lot", "roll_id": "fresh_roll" if fresh else "train_roll",
            "camera_id": "fresh_camera" if fresh else "train_camera", "origin": "real",
            "annotation_source": "human_verified", "annotation_verified_by": "fixture_annotator",
            "partition": "fresh_locked" if fresh else "train",
            "image": save(name + '.image', (name + '_image_SOFTWARE_FIXTURE').encode()),
            "mask": save(name + '.mask', (name + '_mask_SOFTWARE_FIXTURE').encode()),
            "start_m": index * 50.0 if fresh else None, "end_m": (index + 1) * 50.0 if fresh else None}
    used = {"dataset_version": "fixture_used_v1", "purpose": "development_and_historical",
            "provenance_verified_by": "fixture_custodian", "complete_exposure_inventory": True,
            "samples": [sample("used", False)]}
    used_pin = save('used.json', used)
    model = save('model.bin', b'NOT_A_TRAINED_MODEL_SOFTWARE_FIXTURE')
    artifacts = {key: save(key, b'fixture_' + key.encode()) for key in ('reference_metadata','memory_bank','faiss_index')}
    artifacts['thresholds'] = save('thresholds.json', {"fixture_threshold": 0.25})
    artifacts['deployment_engine'] = save('engine.bin', b'NOT_AN_ENGINE_SOFTWARE_FIXTURE')
    artifacts['runtime_recipe'] = save('runtime.json', {"scope": "full_hybrid_pipeline", "precision": "FP32",
        "runtime_version": "fixture_runtime_v1", "model_sha256": model['sha256'], "fp16_fallback_enabled": False,
        "artifact_sha256": {role: pin['sha256'] for role,pin in artifacts.items()}})
    candidate = {"schema_version": 1, "candidate_id": "hypothetical_candidate", "fabric_id": "fixture_fabric",
        "model_version": "fixture_model_v1", "reference_id": "fixture_ref_v1", "reference_version": 1,
        "backbone_version": "fixture_backbone_v1", "runtime_version": "fixture_runtime_v1",
        "frozen_at": "2026-10-03T00:00:00Z", "origin": "real", "trained": True, "research_only": False,
        "precision": "FP32", "defect_classes": ["defect"], "segmentation_applicable": True,
        "model": model, "artifacts": artifacts,
        "training_report": save('training.json', {"trained": True, "synthetic": False, "model_sha256": model['sha256'],
            "model_version": "fixture_model_v1", "used_inventory_sha256": used_pin['sha256']})}
    candidate_pin = save('candidate.json', candidate)
    criteria = {name: {"direction": direction, "threshold": 0.5 if direction == 'min' else 100.0,
                       "applicable": True, "not_applicable_reason": None} for name, direction in METRIC_DIRECTIONS.items()}
    for name in ('underkill','false_positive_rate','overkill'):
        criteria[name]['threshold'] = 0.5
    policy = {"schema_version":1,"policy_version":"FIXTURE_ONLY_NOT_CUSTOMER_SLA", "approved_by":"fixture_policy_owner",
        "approved_at":"2026-10-02T00:00:00Z", "valid_until":"2027-10-02T00:00:00Z",
        "criteria":criteria,"per_class_recall":{"defect":0.5}}
    policy_pin = save('policy.json',policy)
    locked = {"lock_id":"fixture_lock_v1","locked_at":"2026-10-05T00:00:00Z","custodian":"fixture_custodian",
        "previously_opened":False,"historical_or_postmortem":False,"bound_candidate_sha256":candidate_pin['sha256'],
        "bound_policy_sha256":policy_pin['sha256'],"inventory":{"dataset_version":"fixture_fresh_v1","purpose":"fresh_locked",
            "provenance_verified_by":"fixture_custodian","complete_exposure_inventory":True,
            "samples":[sample('normal',True,0),sample('known',True,1),sample('unknown',True,2)]},
        "length_calibration":save('calibration.json',{"unit":"m","verified_by":"fixture_metrologist",
            "dataset_version":"fixture_fresh_v1","camera_ids":["fresh_camera"],"roll_ids":["fresh_roll"]})}
    locked_pin=save('lock.json',locked)
    validation={"candidate_sha256":candidate_pin['sha256'],"used_inventory_sha256":used_pin['sha256'],
        "evaluated_at":"2026-10-04T00:00:00Z","evaluator":"fixture_validator","synthetic":False,"passed":True,
        "functional_checks_passed":True,"numerical_parity_passed":True,"raw_result":save('validation.bin',b'validation software fixture')}
    evidence={"candidate_sha256":candidate_pin['sha256'],"policy_sha256":policy_pin['sha256'],"locked_test_sha256":locked_pin['sha256'],
        "evaluated_at":"2026-10-06T00:00:00Z","evaluator":"fixture_evaluator","synthetic":False,
        "scope":"full_hybrid_pipeline","raw_result":save('predictions.bin',b'prediction software fixture'),
        "event_matching_protocol":"FIXTURE one-to-one event ids; not real model matching",
        "observations":[
            dict(sample_id='normal',predicted_defective=False,defect_class=None,unknown_defect=False,truth_events=0,matched_events=0,false_alarm_events=0,pixel_tp=0,pixel_fp=0,pixel_fn=0),
            dict(sample_id='known',predicted_defective=True,defect_class='defect',unknown_defect=False,truth_events=2,matched_events=2,false_alarm_events=0,pixel_tp=8,pixel_fp=1,pixel_fn=1),
            dict(sample_id='unknown',predicted_defective=True,defect_class='unseen',unknown_defect=True,truth_events=1,matched_events=1,false_alarm_events=0,pixel_tp=2,pixel_fp=0,pixel_fn=0)],
        "latency_ms":[3.0,5.0,7.0],"timing_sample_ids":['normal','known','unknown'],"warmup_excluded":True,
        "cuda_synchronized_if_used":True,"throughput_images":3,"throughput_wall_seconds":0.1,"throughput_protocol":"FIXTURE serial wall window"}
    bundle={'candidate':candidate_pin,'policy':policy_pin,'used_inventory':used_pin,'locked_test':locked_pin,
            'validation':save('validation.json',validation),'evaluation':save('evaluation.json',evidence)}
    bundle_pin=save('bundle.json',bundle)
    return dict(root=root,save=save,bundle=bundle,bundle_pin=bundle_pin,candidate=candidate,policy=policy,
                used=used,locked=locked,validation=validation,evidence=evidence)


def rebind(fixture, role, value):
    """Update outer pins as a legitimate new declaration; inner mismatch tests stay visible."""
    fixture['bundle'][role]=fixture['save'](role+'.json',value)
    fixture['bundle_pin']=fixture['save']('bundle.json',fixture['bundle'])
    return ArtifactPin.model_validate(fixture['bundle_pin'])


def assess(fixture):
    return governance.assess_release(ArtifactPin.model_validate(fixture['bundle_pin']),trusted_local=True,now=NOW)


def registry_and_approval(fixture, report):
    c=fixture['candidate']
    record={"reference_id":c['reference_id'],"reference_version":c['reference_version'],"status":"CANDIDATE",
        "backbone_version":c['backbone_version'],"model_version":c['model_version'],"runtime_version":c['runtime_version'],
        "release_candidate_sha256":fixture['bundle']['candidate']['sha256'],"segmentation_model_path":c['model']['path'],
        "deployment_engine_path":c['artifacts']['deployment_engine']['path'],"runtime_recipe_path":c['artifacts']['runtime_recipe']['path'],
        "artifact_dir":str(fixture['root']),"metadata_path":c['artifacts']['reference_metadata']['path'],
        "memory_bank_path":c['artifacts']['memory_bank']['path'],"faiss_index_path":c['artifacts']['faiss_index']['path'],
        "thresholds":{"fixture_threshold":0.25}}
    payload={"schema_version":1,"shared_backbone":{"backbone_version":c['backbone_version']},
        "fabrics":{"fixture_fabric":{"status":"INACTIVE","active_reference":None,"references":[record]}}}
    registry_pin=fixture['save']('registry.json',payload)
    report_pin=ArtifactPin.model_validate(fixture['save']('report.json',report))
    approval=ManualApproval(candidate_sha256=fixture['bundle']['candidate']['sha256'],report_sha256=report_pin.sha256,
        expected_registry_sha256=registry_pin['sha256'],reviewer='fixture_independent_operator',reviewed_at=NOW,
        reason='TEMP software integration test only',confirm='ACTIVATE:hypothetical_candidate:'+report_pin.sha256)
    return registry_pin,report_pin,approval

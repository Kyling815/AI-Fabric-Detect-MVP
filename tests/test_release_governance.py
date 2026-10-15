"""Phase18 policy/arithmetic/failure tests. All values are SOFTWARE fixtures."""

import copy
from datetime import datetime
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.app.core.release_governance import (AcceptancePolicy, ArtifactPin, Candidate, EvaluationEvidence,
    METRIC_DIRECTIONS, SampleProvenance)
from backend.app.services import release_governance as gov
from tests.release_governance_fixtures import NOW, assess, make_bundle, rebind


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    return make_bundle(tmp_path,monkeypatch)


def test_hypothetical_happy_path_is_awaiting_manual_approval_not_active(bundle):
    report=assess(bundle)
    assert report['accepted'] and report['status']=='AWAITING_MANUAL_APPROVAL'
    assert not report['registry_modified'] and not report['automatic_activation']
    assert report['metrics']['defect_recall']==1
    assert report['metrics']['segmentation_iou']==pytest.approx(10/12)
    assert report['metrics']['segmentation_dice']==pytest.approx(20/22)
    assert report['metrics']['unknown_defect_recall']==1
    assert report['counts']['inspected_meters']==150
    assert report['metrics']['p50_latency_ms']==5
    assert report['metrics']['p95_latency_ms']==pytest.approx(6.8)
    assert report['metrics']['throughput_images_per_second']==30


@pytest.mark.parametrize('metric',list(METRIC_DIRECTIONS))
def test_each_required_criterion_missing_is_fail_closed(bundle,metric):
    bundle['policy']['criteria'][metric]['threshold']=None
    rebind(bundle,'policy',bundle['policy'])
    # Repin lock/evaluation to emulate a properly frozen but incomplete policy.
    bundle['locked']['bound_policy_sha256']=bundle['bundle']['policy']['sha256']
    rebind(bundle,'locked_test',bundle['locked'])
    bundle['evidence']['policy_sha256']=bundle['bundle']['policy']['sha256']
    bundle['evidence']['locked_test_sha256']=bundle['bundle']['locked_test']['sha256']
    rebind(bundle,'evaluation',bundle['evidence'])
    report=assess(bundle)
    assert not report['accepted'] and metric in report['missing']


@pytest.mark.parametrize('field,value',[('precision','FP16'),('precision','FP16_MIXED'),('origin','synthetic'),
    ('trained',False),('research_only',True),('training_report',None),('candidate_id','phase17_renamed_probe')])
def test_untrained_synthetic_phase17_and_fp16_never_promote(bundle,field,value):
    bundle['candidate'][field]=value
    rebind(bundle,'candidate',bundle['candidate'])
    report=assess(bundle)
    assert not report['accepted'] and all(v is None for v in report['metrics'].values())


def test_renamed_phase17_hash_is_still_denied(bundle):
    bundle['candidate']['model']['sha256']=next(iter(gov.PHASE17_RESEARCH_HASHES))
    # Pinned byte mismatch aborts even before the deny can be presented as data.
    rebind(bundle,'candidate',bundle['candidate'])
    with pytest.raises(ValueError,match='SHA'):
        assess(bundle)


@pytest.mark.parametrize('field',['source_id','source_sha256','lot_id','roll_id','camera_id','image'])
def test_source_lot_roll_camera_or_identical_image_leak_blocks(bundle,field):
    bundle['locked']['inventory']['samples'][0][field]=bundle['used']['samples'][0][field]
    rebind(bundle,'locked_test',bundle['locked'])
    report=assess(bundle)
    assert any('LEAK' in r for r in report['blocking_reasons'])


@pytest.mark.parametrize('field',['ancestor_source_ids','ancestor_source_sha256s'])
def test_ancestor_crop_lineage_leak_blocks(bundle,field):
    old=bundle['used']['samples'][0]
    bundle['locked']['inventory']['samples'][0][field]=[old['source_id'] if field.endswith('ids') else old['source_sha256']]
    rebind(bundle,'locked_test',bundle['locked'])
    assert any('LEAK' in r for r in assess(bundle)['blocking_reasons'])


@pytest.mark.parametrize('field,value',[('previously_opened',True),('historical_or_postmortem',True),
    ('bound_candidate_sha256','1'*64),('bound_policy_sha256','1'*64),('locked_at','2026-10-01T00:00:00Z')])
def test_historical_reopened_or_unbound_lock_rejected(bundle,field,value):
    bundle['locked'][field]=value
    rebind(bundle,'locked_test',bundle['locked'])
    assert not assess(bundle)['accepted']


@pytest.mark.parametrize('field,value',[('scope','segmentation_architecture_only'),('synthetic',True),
    ('candidate_sha256','1'*64),('policy_sha256','1'*64),('locked_test_sha256','1'*64),
    ('warmup_excluded',False),('cuda_synchronized_if_used',False),('evaluated_at','2026-10-01T00:00:00Z')])
def test_segmentation_only_or_invalid_evidence_is_not_factory_acceptance(bundle,field,value):
    bundle['evidence'][field]=value
    rebind(bundle,'evaluation',bundle['evidence'])
    assert not assess(bundle)['accepted']


@pytest.mark.parametrize('field',['approved_by','approved_at','valid_until'])
def test_unapproved_policy_blocks(bundle,field):
    bundle['policy'][field]=None
    rebind(bundle,'policy',bundle['policy'])
    assert not assess(bundle)['accepted']


def test_expired_policy_blocks_even_with_good_metrics(bundle):
    bundle['policy']['valid_until']='2026-10-10T00:00:00Z'
    rebind(bundle,'policy',bundle['policy'])
    assert not assess(bundle)['accepted']


@pytest.mark.parametrize('field,value',[('numerical_parity_passed',False),('functional_checks_passed',False),
    ('passed',False),('synthetic',True)])
def test_validation_stage_cannot_be_skipped(bundle,field,value):
    bundle['validation'][field]=value
    rebind(bundle,'validation',bundle['validation'])
    assert 'VALIDATION_STAGE_INVALID' in assess(bundle)['blocking_reasons']


def test_missing_per_class_criteria_blocks(bundle):
    bundle['policy']['per_class_recall']={}
    rebind(bundle,'policy',bundle['policy'])
    assert not assess(bundle)['accepted']


def test_missing_unknown_population_is_undefined_not_perfect(bundle):
    row=bundle['evidence']['observations'][2]
    row['unknown_defect']=False
    row['defect_class']='defect'
    rebind(bundle,'evaluation',bundle['evidence'])
    report=assess(bundle)
    assert report['metrics']['unknown_defect_recall'] is None and not report['accepted']


def test_missing_meter_evidence_is_undefined_not_zero(bundle):
    bundle['locked']['length_calibration']=None
    rebind(bundle,'locked_test',bundle['locked'])
    bundle['evidence']['locked_test_sha256']=bundle['bundle']['locked_test']['sha256']
    rebind(bundle,'evaluation',bundle['evidence'])
    report=assess(bundle)
    assert report['metrics']['false_alarms_per_100m'] is None
    assert not report['accepted']


def test_overlap_across_cameras_cannot_inflate_inspected_length(bundle):
    samples=bundle['locked']['inventory']['samples']
    samples[1]['start_m']=25.0
    samples[1]['camera_id']='another_fresh_camera'
    with pytest.raises(ValueError,match='Overlapping'):
        gov.inspected_length(tuple(SampleProvenance.model_validate_json(json.dumps(s)) for s in samples))


def test_all_normal_or_zero_predictions_does_not_invent_perfect_metrics(bundle):
    for row in bundle['evidence']['observations']:
        row.update(predicted_defective=False,truth_events=0,matched_events=0,false_alarm_events=0,
                   defect_class=None,unknown_defect=False,pixel_tp=0,pixel_fp=0,pixel_fn=0)
    rebind(bundle,'evaluation',bundle['evidence'])
    report=assess(bundle)
    assert report['metrics']['defect_recall'] is None
    assert report['metrics']['precision'] is None and report['metrics']['f1'] is None
    assert not report['accepted']


def test_failed_recall_and_underkill_metrics_rejects(bundle):
    bundle['evidence']['observations'][1].update(matched_events=0,predicted_defective=False)
    rebind(bundle,'evaluation',bundle['evidence'])
    report=assess(bundle)
    assert report['metrics']['false_negatives']==2
    assert report['metrics']['defect_recall']==pytest.approx(1/3)
    assert report['metrics']['missed_defects_per_100m']==pytest.approx(2/1.5)
    assert not report['accepted'] and 'defect_recall' in report['failed']


def test_legacy_known_history_bytes_deny_even_after_relabel(bundle):
    folder=bundle['root']/'storage/metadata/company_demo_manifests'
    folder.mkdir(parents=True)
    sample=bundle['locked']['inventory']['samples'][0]
    (folder/'fixture_all.csv').write_text('source_original,image_path\nold_source,'+sample['image']['path']+'\n',encoding='utf-8')
    report=assess(bundle)
    assert any('LEAK' in r for r in report['blocking_reasons'])


@pytest.mark.parametrize('value',[True,'0.8',float('nan'),float('inf'),-1.0])
def test_invalid_thresholds_rejected_not_coerced(bundle,value):
    bundle['policy']['criteria']['defect_recall']['threshold']=value
    with pytest.raises(ValidationError):
        AcceptancePolicy.model_validate_json(json.dumps(bundle['policy']))


@pytest.mark.parametrize('field',['lot_id','roll_id','camera_id','source_id','source_sha256'])
def test_missing_acquisition_provenance_not_inferred(bundle,field):
    sample=copy.deepcopy(bundle['used']['samples'][0])
    sample.pop(field)
    with pytest.raises(ValidationError):
        SampleProvenance.model_validate_json(json.dumps(sample))


def test_timing_counts_duplicates_or_zero_samples_rejected(bundle):
    data=bundle['evidence']
    data['latency_ms'][0]=0.0
    with pytest.raises(ValidationError):
        EvaluationEvidence.model_validate_json(json.dumps(data))


def test_explicit_trust_required_before_file_read(bundle):
    with pytest.raises(ValueError,match='trusted'):
        gov.assess_release(ArtifactPin.model_validate(bundle['bundle_pin']))


def test_unapproved_shipped_template_has_no_customer_numbers():
    root=Path(__file__).resolve().parents[1]
    policy=AcceptancePolicy.model_validate_json((root/'config/release_policy_unapproved_example.json').read_bytes())
    assert policy.approved_by is None and all(r.threshold is None for r in policy.criteria.values())


@pytest.mark.parametrize('role',['model','deployment_engine'])
def test_valid_hash_known_research_artifact_denied_even_if_declared_trained(bundle,monkeypatch,role):
    pin=bundle['candidate']['model'] if role=='model' else bundle['candidate']['artifacts'][role]
    monkeypatch.setattr(gov,'PHASE17_RESEARCH_HASHES',frozenset({pin['sha256']}))
    report=assess(bundle)
    assert 'PHASE17_RESEARCH_PROBE_PERMANENTLY_INELIGIBLE' in report['blocking_reasons']
    assert not report['accepted']


@pytest.mark.parametrize('source,reviewer',[('synthetic','fixture_annotator'),('unverified','fixture_annotator'),('human_verified',None)])
def test_synthetic_or_unreviewed_annotations_block_even_for_real_captures(bundle,source,reviewer):
    bundle['locked']['inventory']['samples'][0].update(annotation_source=source,annotation_verified_by=reviewer)
    rebind(bundle,'locked_test',bundle['locked'])
    assert any('HUMAN_VERIFIED' in r for r in assess(bundle)['blocking_reasons'])


def test_duplicate_locked_image_bytes_cannot_inflate_population(bundle):
    bundle['locked']['inventory']['samples'][1]['image']=bundle['locked']['inventory']['samples'][0]['image']
    rebind(bundle,'locked_test',bundle['locked'])
    assert 'DUPLICATE_LOCKED_IMAGE_BYTES' in assess(bundle)['blocking_reasons']


@pytest.mark.parametrize('value',[False,'5',float('nan'),float('inf'),0.0,-1.0])
def test_bad_latency_samples_never_accepted(bundle,value):
    bundle['evidence']['latency_ms'][0]=value
    with pytest.raises(ValidationError):
        EvaluationEvidence.model_validate_json(json.dumps(bundle['evidence']))


def test_relative_artifact_paths_rejected_before_read():
    with pytest.raises(ValidationError):
        ArtifactPin(path='model.onnx',sha256='1'*64)


def test_segmentation_na_cannot_bypass_required_segmentation_metrics(bundle):
    bundle['policy']['criteria']['segmentation_iou'].update(applicable=False,not_applicable_reason='fixture skip attempt')
    rebind(bundle,'policy',bundle['policy'])
    bundle['locked']['bound_policy_sha256']=bundle['bundle']['policy']['sha256']
    rebind(bundle,'locked_test',bundle['locked'])
    bundle['evidence'].update(policy_sha256=bundle['bundle']['policy']['sha256'],locked_test_sha256=bundle['bundle']['locked_test']['sha256'])
    rebind(bundle,'evaluation',bundle['evidence'])
    report=assess(bundle)
    assert not report['accepted'] and 'segmentation_iou' in report['failed']

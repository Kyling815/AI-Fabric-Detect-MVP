"""TEMP-only manual transition/CAS tests, never production activation/evidence."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from backend.app.core.release_governance import ArtifactPin
from backend.app.services import release_governance as gov
from tests.release_governance_fixtures import NOW, assess, make_bundle, registry_and_approval


@pytest.fixture
def setup(tmp_path,monkeypatch):
    fixture=make_bundle(tmp_path,monkeypatch)
    report=assess(fixture)
    registry,report_pin,approval=registry_and_approval(fixture,report)
    return fixture,registry,report_pin,approval


def test_manual_activation_rechecks_and_updates_only_temp_registry_with_backup(setup):
    fixture,registry,report_pin,approval=setup
    old=Path(registry['path']).read_bytes()
    result=gov.activate_manually(ArtifactPin.model_validate(fixture['bundle_pin']),report_pin,registry['path'],approval,
                                 trusted_local=True,now=NOW)
    current=json.loads(Path(registry['path']).read_bytes())
    assert current['fabrics']['fixture_fabric']['status']=='ACTIVE'
    assert current['fabrics']['fixture_fabric']['references'][0]['status']=='ACTIVE'
    assert Path(result['backup']).read_bytes()==old
    assert len(current['release_activation_audit'])==1
    assert not Path(registry['path']+'.phase18.lock').exists()


@pytest.mark.parametrize('change',['confirm','reviewer','candidate_sha256','report_sha256','expected_registry_sha256'])
def test_manual_approval_must_bind_exact_candidate_report_and_registry(setup,change):
    fixture,registry,report_pin,approval=setup
    values=approval.model_dump()
    values[change]='fixture_evaluator' if change=='reviewer' else ('1'*64 if 'sha256' in change else 'ACTIVATE')
    old=Path(registry['path']).read_bytes()
    with pytest.raises(ValueError):
        gov.activate_manually(ArtifactPin.model_validate(fixture['bundle_pin']),report_pin,registry['path'],values,trusted_local=True,now=NOW)
    assert Path(registry['path']).read_bytes()==old


@pytest.mark.parametrize('target',['model','evidence','report','thresholds','registry'])
def test_post_review_tamper_or_stale_registry_never_activates(setup,target):
    fixture,registry,report_pin,approval=setup
    target_pin=fixture['candidate']['model'] if target=='model' else fixture['bundle']['evaluation'] if target=='evidence' else (
        fixture['candidate']['artifacts']['thresholds'] if target=='thresholds' else registry if target=='registry' else report_pin.model_dump())
    path=Path(target_pin['path'])
    path.write_bytes(path.read_bytes()+b' ')
    before=Path(registry['path']).read_bytes()
    with pytest.raises(ValueError):
        gov.activate_manually(ArtifactPin.model_validate(fixture['bundle_pin']),report_pin,registry['path'],approval,trusted_local=True,now=NOW)
    assert Path(registry['path']).read_bytes()==before


def test_accepted_boolean_cannot_forge_report(setup):
    fixture,registry,report_pin,approval=setup
    forged=json.loads(gov.read_pin(report_pin))
    forged['metrics']['defect_recall']=0.1
    new_pin=ArtifactPin.model_validate(fixture['save']('forged_report.json',forged))
    values=approval.model_dump()
    values.update(report_sha256=new_pin.sha256,confirm='ACTIVATE:hypothetical_candidate:'+new_pin.sha256)
    with pytest.raises(ValueError,match='reproducible'):
        gov.activate_manually(ArtifactPin.model_validate(fixture['bundle_pin']),new_pin,registry['path'],values,trusted_local=True,now=NOW)


def test_existing_lock_refuses_concurrent_manual_writer(setup):
    fixture,registry,report_pin,approval=setup
    lock=Path(registry['path']+'.phase18.lock')
    lock.write_text('another operator',encoding='utf-8')
    before=Path(registry['path']).read_bytes()
    with pytest.raises(FileExistsError):
        gov.activate_manually(ArtifactPin.model_validate(fixture['bundle_pin']),report_pin,registry['path'],approval,trusted_local=True,now=NOW)
    assert Path(registry['path']).read_bytes()==before and lock.read_text()=='another operator'


def test_activation_replay_fails_expected_registry_pin(setup):
    fixture,registry,report_pin,approval=setup
    args=(ArtifactPin.model_validate(fixture['bundle_pin']),report_pin,registry['path'],approval)
    gov.activate_manually(*args,trusted_local=True,now=NOW)
    current=Path(registry['path']).read_bytes()
    with pytest.raises(ValueError):
        gov.activate_manually(*args,trusted_local=True,now=NOW)
    assert Path(registry['path']).read_bytes()==current


def test_legacy_direct_activator_is_disabled_without_writing():
    from scripts.evaluate_fabric_demo import activate
    with pytest.raises(RuntimeError,match='disabled'):
        activate([{'fabric_id':'fabric_001','reference_id':'fabric_001_ref_v1'}])


def test_legacy_activation_cli_stops_before_evaluation():
    root=Path(__file__).resolve().parents[1]
    completed=subprocess.run([sys.executable,str(root/'scripts/evaluate_fabric_demo.py'),'--activate-if-accepted'],
        capture_output=True,text=True,timeout=30)
    assert completed.returncode==2 and 'disabled' in completed.stderr


def test_current_readonly_audit_keeps_missing_metrics_null():
    root=Path(__file__).resolve().parents[1]
    completed=subprocess.run([sys.executable,str(root/'scripts/check_phase18_release.py'),'--audit-current'],
        capture_output=True,text=True,timeout=30)
    assert completed.returncode==0
    result=json.loads(completed.stdout)
    assert result['status']=='BLOCKED BY DATA / TRAINED CANDIDATE'
    assert not result['accepted'] and not result['registry_modified']
    assert all(v is None for v in result['metrics'].values())
    assert set(result['registry_fabric_statuses'].values())=={'INACTIVE'}


def test_atomic_publish_failure_leaves_registry_intact_and_unlocks(setup,monkeypatch):
    fixture,registry,report_pin,approval=setup
    before=Path(registry['path']).read_bytes()
    def fail_replace(*args):
        raise OSError('fixture filesystem failure')
    monkeypatch.setattr(gov.os,'replace',fail_replace)
    with pytest.raises(OSError):
        gov.activate_manually(ArtifactPin.model_validate(fixture['bundle_pin']),report_pin,registry['path'],approval,trusted_local=True,now=NOW)
    assert Path(registry['path']).read_bytes()==before
    assert not Path(registry['path']+'.phase18.lock').exists()
    assert not list(fixture['root'].glob('.phase18-*'))
    assert list(fixture['root'].glob('*.backup'))


def test_explicit_manual_cli_uses_only_temp_fixture_registry(setup):
    fixture,registry,report_pin,approval=setup
    request=fixture['save']('manual_request.json',{'report':report_pin.model_dump(),
        'registry_path':registry['path'],'approval':approval.model_dump(mode='json')})
    root=Path(__file__).resolve().parents[1]
    completed=subprocess.run([sys.executable,str(root/'scripts/check_phase18_release.py'),
        '--bundle',fixture['bundle_pin']['path'],'--bundle-sha256',fixture['bundle_pin']['sha256'],'--trusted-local',
        '--manual-activation-request',request['path'],'--request-sha256',request['sha256']],capture_output=True,text=True,timeout=30)
    assert completed.returncode==0,completed.stderr
    assert json.loads(completed.stdout)['activated']
    assert json.loads(Path(registry['path']).read_bytes())['fabrics']['fixture_fabric']['status']=='ACTIVE'


@pytest.mark.parametrize('key',['deployment_engine_path','runtime_recipe_path','reference_version','model_version','runtime_version','backbone_version','thresholds'])
def test_registry_provenance_cannot_be_rebound_to_wrong_artifacts_even_with_fresh_registry_pin(setup,key):
    fixture,registry,report_pin,approval=setup
    path=Path(registry['path'])
    payload=json.loads(path.read_bytes())
    row=payload['fabrics']['fixture_fabric']['references'][0]
    row[key]=2 if key=='reference_version' else {} if key=='thresholds' else 'wrong'
    new=gov.canonical_json(payload)
    path.write_bytes(new)
    values=approval.model_dump()
    values['expected_registry_sha256']=gov.digest(new)
    with pytest.raises(ValueError):
        gov.activate_manually(ArtifactPin.model_validate(fixture['bundle_pin']),report_pin,path,values,trusted_local=True,now=NOW)
    assert path.read_bytes()==new

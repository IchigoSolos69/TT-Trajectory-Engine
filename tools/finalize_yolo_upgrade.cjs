'use strict';
// Reconcile frozen evidence without fitting models, relabeling data or starting services.
const fs=require('node:fs');
const path=require('node:path');
const crypto=require('node:crypto');
const assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..');
const destination=process.argv[2];
if(!destination)throw Error('Usage: node tools/finalize_yolo_upgrade.cjs NEW_REPORT.json');
const output=path.resolve(destination);
if(fs.existsSync(output))throw Error('Choose a new report; historical evidence is preserved.');
const read=name=>JSON.parse(fs.readFileSync(path.join(root,name),'utf8'));
const hash=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const evidencePaths=[
  'artifacts/yolo_paired_training_plan_2026_10_11_v4.json',
  'artifacts/yolo_paired_training_blocked_fit_2026_10_11_v4.log',
  'artifacts/absence_interval_training_tests_2026_10_11_v4.log',
  'artifacts/full_absence_api_checks.json',
  'artifacts/training_sessions_api_checks.json',
  'artifacts/training_decoder_lock_checks.json',
  'artifacts/training_sessions_checks.json',
  'artifacts/yolo_upgrade_2026_10_10_v1/release_and_raw_final_tests.log',
  'artifacts/yolo_upgrade_2026_10_10_v1/desktop_smoke_v4.json',
  'artifacts/yolo_upgrade_2026_10_10_v1/frontend_build_v2.log',
  'artifacts/yolo_upgrade_2026_10_10_v1/desktop_unit_v2.log',
  'artifacts/yolo_upgrade_2026_10_10_v1/desktop_package_v2.log',
  'artifacts/yolo_upgrade_2026_10_10_v1/development_folder_split_v2.json',
  'artifacts/yolo_upgrade_2026_10_10_v1/raw_postprocess_reaudit.json',
  'artifacts/yolo_shape_native_2026_10_10_v1/paired_shape_regression.json',
];
const plan=read(evidencePaths[0]),smoke=read(evidencePaths[8]);
assert.equal(plan.ready_for_fitting,false);
assert.equal(plan.readiness.test_labels_consumed,false);
assert.equal(smoke.ok,true);
assert.equal(smoke.checks.length,19);
assert.equal(smoke.renderer_errors.length,0);
assert.equal(smoke.live.camera_active,false);
assert.equal(smoke.full_original_absence.fixture_only,true);
assert.equal(smoke.full_original_absence.source_frames,90);
assert.equal(smoke.full_original_absence.interval_frames,60);
assert.equal(smoke.full_original_absence.duration_s,2);
assert.equal(fs.existsSync(path.join(root,'artifacts/yolo_upgrade_2026_10_10_v1/blocked_fit_v4')),false);

const baseline=read('artifacts/yolo_upgrade_2026_10_10_v1/preservation_check.json');
const release='desktop/dist/TT Trajectory Engine paired training v2';
const releaseManifest=read(release+'/release-manifest.json');
const names={ball:'yolov11_ttgames.engine',scene:'scene.engine',blurball_experimental:'blurball_fp32.engine'};
for(const [model,name]of Object.entries(names)){
  assert.equal(hash(path.join(root,'models',name)),baseline.models[model]);
  assert.equal(hash(path.join(root,release,'resources/runtime/models',name)),baseline.models[model]);
  assert.equal(releaseManifest.models[model],baseline.models[model]);
}
assert.equal(hash(path.join(root,release,'resources/runtime/backend/tt_trajectory_engine.exe')),releaseManifest.source_backend_sha256);
assert.equal(hash(plan.inputs.weights),plan.weights_sha256);
for(const [name,sha]of Object.entries(releaseManifest.optional_training_tools)){
  assert.equal(hash(path.join(root,'python_tools',name)),sha);
  assert.equal(hash(path.join(root,release,'resources/runtime/training-tools',name)),sha);
}

const base=path.join(root,'training_data/live'),registry=read('training_data/live/training_sessions.json');
const stage=read('artifacts/reviewed_training_stage_2026_10_10_v1/manifest.json');
let preservedImages=0,preservedMetadata=0,sealedPreservationSkipped=0;
for(const [file,expected]of Object.entries(stage.source_snapshots)){
  const resolved=path.resolve(file),relative=path.relative(base,resolved);
  if(relative.startsWith('..')||path.isAbsolute(relative))continue;
  const clip=relative.split(path.sep)[0],assignment=registry.captures[clip],session=assignment?registry.sessions[assignment.session_id]:null;
  if(session?.sealed||session?.role==='test'){sealedPreservationSkipped++;continue;}
  assert.equal(hash(resolved),expected,'Earlier source/review snapshot changed: '+relative);
  if(/\.jpg$/i.test(relative))preservedImages++;else preservedMetadata++;
}
const sources=[
  'launch_desktop.bat','desktop/main.cjs','desktop/scripts/package.cjs','desktop/tests/in-app-smoke.cjs',
  'frontend/src/app/api/training-absence/route.ts','frontend/src/app/api/training-sessions/route.ts',
  'frontend/src/components/FullAbsenceIntervalReview.tsx','frontend/src/lib/trainingIntervalStorage.ts',
  'frontend/src/lib/trainingDecoderLock.ts','frontend/src/lib/trainingSessions.ts',
  'python_tools/yolo_ball_experiments.py','python_tools/yolo_paired_data.py',
  'python_tools/yolo_training_policy.py','python_tools/yolo_ultralytics_hooks.py',
  'python_tools/prepare_absence_interval.py','python_tools/recording_source.py',
  'python_tools/audit_absence_intervals.py','python_tools/ball_release_metrics.py',
  'python_tools/yolo_raw_diagnostics.py','backend_cpp/include/tensorrt_engine.hpp',
  'backend_cpp/src/tensorrt_engine.cpp','YOLO_UPGRADE_IMPLEMENTATION_2026_10_11.md',
  'NEXT_TRAINING_STEP.md','REQUIREMENTS_LEDGER.md','VERIFICATION.md',
];
const report={
  schema_version:1,date:'2026-10-11',timezone:'Asia/Calcutta',
  scope:'Verified paired-training foundation and local interval review; no fitted accuracy improvement or default model replacement.',
  application:path.join(root,release,'TT Trajectory Engine.exe'),launcher:path.join(root,'launch_desktop.bat'),
  whole_plan_complete:false,ready_for_fitting:false,promotion_approved:false,
  data_clarification:'Additional user clips are declared official World Cup match footage. No actual PC-camera recordings currently exist. Original iPhone clips remain one earlier session.',
  reviewed_camera_counts:plan.readiness.counts,training_blockers:plan.blockers,
  sealed_test_labels_consumed:false,live_accuracy_measured:false,live_60fps_measured:false,live_soak_completed:false,
  tests:{training_and_interval_cpu:62,release_metrics:19,raw_diagnostics:9,desktop_unit:6,
    decoder_api:read(evidencePaths[3]).checks,session_api:read(evidencePaths[4]).checks,
    decoder_ownership:read(evidencePaths[5]).checks,collection_readiness:read(evidencePaths[6]).checks,
    packaged_app:smoke.checks.length},
  preserved:{original_review_images:preservedImages,original_capture_index_annotation_files:preservedMetadata,
    sealed_preservation_reads_skipped:sealedPreservationSkipped,models:baseline.models,checkpoint_sha256:plan.weights_sha256},
  release_manifest_sha256:hash(path.join(root,release,'release-manifest.json')),
  source_sha256:Object.fromEntries(sources.map(name=>[name,hash(path.join(root,name))])),
  evidence_sha256:Object.fromEntries(evidencePaths.map(name=>[name,hash(path.join(root,name))])),
  remaining:['Independent paired camera sessions and reviewed labels','Controlled YOLO fitting and precision/recall gates',
    'Learned YOLO–BlurBall selector and student comparison','Independent sealed acceptance and actual live latency/soak',
    'Earlier feature acceptance and broader digital-twin/event/physics plan closure'],
};
fs.writeFileSync(output,JSON.stringify(report,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({report:output,tests:report.tests,preserved:report.preserved,ready_for_fitting:false,promotion_approved:false},null,2));

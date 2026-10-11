'use strict';
// Reconcile the actual resolver migration with preserved review snapshots.
// No video decode, sealed annotations, session assignments or model fitting.
const fs=require('node:fs'),fsp=fs.promises,path=require('node:path'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..');
const read=file=>JSON.parse(fs.readFileSync(path.join(root,file),'utf8'));
const hash=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
async function main(){
  const destination=process.argv[2];
  if(!destination||fs.existsSync(destination))throw Error('Choose a new verification report path');
  const auditPath='artifacts/yolo_upgrade_2026_10_10_v1/nested_folder_audit_v1.json';
  const splitPath='artifacts/yolo_upgrade_2026_10_10_v1/development_folder_split_v3.json';
  const planPath='artifacts/yolo_paired_training_plan_2026_10_11_v5.json';
  const testsPath='artifacts/recording_split_tests_2026_10_11_v2.log';
  const audit=read(auditPath),split=read(splitPath),plan=read(planPath),base=path.join(root,'training_data/live');
  assert.equal(split.inventory_sha256,hash(path.join(root,auditPath)));
  assert.equal(split.freshness_checked_by,'STREAM_SHA256_AND_STAT');
  assert.equal(split.independent_release_evidence,false);
  assert.equal(split.sealed_test_labels_consumed,false);
  assert.equal(audit.rows.length,split.sources.length);
  assert.equal(plan.ready_for_fitting,false);
  assert.equal(plan.readiness.test_labels_consumed,false);
  assert.equal(plan.readiness.original_source_issues.length,0);
  const previousPlan=read('artifacts/yolo_paired_training_plan_2026_10_11_v4.json');
  const registryFile=path.join(base,'training_sessions.json');
  assert.equal(hash(registryFile),previousPlan.dependencies[registryFile.replace(/\\/g,'/')],'Production session assignment changed');
  assert.equal(hash(audit.source_locations_path),audit.source_locations_sha256);
  assert.equal(hash(audit.source_locations_migration.previous_backup),audit.source_locations_migration.previous_sha256);
  const locations=JSON.parse(fs.readFileSync(audit.source_locations_path,'utf8'));
  let matchedImports=0;
  for(const row of split.sources){
    const actual=await fsp.stat(row.path);
    assert.equal(actual.size,row.bytes);
    assert.equal(actual.mtimeMs,row.source_mtime_ms,'Source changed since complete hash check');
    assert.equal((await fsp.realpath(row.path)).toLowerCase(),path.resolve(row.path).toLowerCase(),'Source alias after audit');
    assert.equal(row.independent_release_evidence,false);
    if(row.declared_domain==='match')assert.equal(row.mixed_match_highlights,true);
    if(row.known_session_ids.includes('iphone_same_venue_session_1'))assert.equal(row.effective_role,'train');
    if(row.matching_review_captures.length){
      assert.equal(locations.sources[row.sha256].paths[0],row.path);
      matchedImports+=row.matching_review_captures.length;
    }
  }
  const registry=read('training_data/live/training_sessions.json'),stage=read('artifacts/reviewed_training_stage_2026_10_10_v1/manifest.json');
  let metadata=0,images=0,sealedSkipped=0;
  for(const [file,expected]of Object.entries(stage.source_snapshots)){
    const absolute=path.resolve(file),relative=path.relative(base,absolute);
    if(relative.startsWith('..')||path.isAbsolute(relative))continue;
    const clip=relative.split(path.sep)[0],assignment=registry.captures[clip],session=assignment?registry.sessions[assignment.session_id]:null;
    if(session?.sealed||session?.role==='test'){sealedSkipped++;continue;}
    assert.equal(hash(absolute),expected,'Earlier review snapshot changed: '+relative);
    if(/\.jpg$/i.test(relative))images++;else metadata++;
  }
  const baseline=read('artifacts/yolo_upgrade_2026_10_10_v1/preservation_check.json');
  const models={ball:'yolov11_ttgames.engine',scene:'scene.engine',blurball_experimental:'blurball_fp32.engine'};
  for(const [model,name]of Object.entries(models))assert.equal(hash(path.join(root,'models',name)),baseline.models[model]);
  assert.equal(hash(plan.inputs.weights),plan.weights_sha256);
  const logs=fs.readFileSync(path.join(root,testsPath),'utf8');
  assert.match(logs,/pass 31/);assert.match(logs,/fail 0/);
  const sourcePaths=['tools/audit_recording_folders.cjs','tools/recording_folder_inventory.cjs','tools/prepare_development_folder_split.cjs',
    'tools/verify_recording_folder_migration.cjs','tools/tests/recording_folder_inventory.test.cjs','tools/tests/development_folder_split.test.cjs'];
  const report={schema_version:1,created_at:new Date().toISOString(),scope:'Source-only nested folder reconciliation; no fitted model accuracy or independent release acceptance',
    files:split.sources.length,folder_counts:split.folder_counts,reconnected_originals:audit.existing_originals_relocated,matched_review_imports:matchedImports,
    mixed_match_compilations:split.sources.filter(row=>row.mixed_match_highlights).length,exact_cross_split_duplicates:split.exact_cross_split_duplicates,
    same_session_cross_folder_sources:audit.known_session_split_conflicts,independent_release_evidence:false,
    production_partitions_modified:false,sealed_test_labels_consumed:false,ready_for_fitting:false,promotion_approved:false,
    camera_counts:plan.readiness.counts,training_blockers:plan.blockers,
    preserved:{original_capture_index_annotation_files:metadata,original_review_images:images,sealed_preservation_reads_skipped:sealedSkipped,models:baseline.models},
    checks:{inventory_fixtures:15,development_split_fixtures:16,live_camera_tested:false},
    source_resolver_sha256:audit.source_locations_sha256,
    evidence_sha256:Object.fromEntries([auditPath,splitPath,planPath,testsPath].map(file=>[file,hash(path.join(root,file))])),
    source_sha256:Object.fromEntries(sourcePaths.map(file=>[file,hash(path.join(root,file))])),
    limits:['Byte duplicate check does not detect overlapping highlight edits','No additional reviewed labels created','Independent paired camera validation/test remains missing','No live throughput/accuracy acceptance measured']};
  await fsp.writeFile(destination,JSON.stringify(report,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({files:report.files,reconnected_originals:report.reconnected_originals,checks:report.checks,preserved:report.preserved,ready_for_fitting:false},null,2));
}
main().catch(error=>{console.error(error.stack);process.exitCode=1;});

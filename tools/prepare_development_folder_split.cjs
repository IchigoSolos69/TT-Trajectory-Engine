'use strict';
// Development organization only. This cannot assign a production session or
// turn previously exposed/mixed-match footage into independent test evidence.
const fs=require('node:fs'),fsp=fs.promises,path=require('node:path'),crypto=require('node:crypto');
const hashPattern=/^[a-f0-9]{64}$/;
const roles=['train','val','test'];
const digest=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
async function sourceIdentity(row){
  const canonical=await fsp.realpath(row.path),samePath=(a,b)=>process.platform==='win32'?path.resolve(a).toLowerCase()===path.resolve(b).toLowerCase():path.resolve(a)===path.resolve(b);
  if(!samePath(canonical,row.path))throw Error('Recording alias is excluded: '+row.path);
  const before=await fsp.stat(row.path);
  if(!before.isFile()||before.size!==row.bytes)throw Error('Source moved/changed; repeat the full folder hash audit: '+row.path);
  if(row.source_mtime_ms!==undefined&&before.mtimeMs!==row.source_mtime_ms)throw Error('Source modification time changed since audit: '+row.path);
  const hash=crypto.createHash('sha256');
  for await(const bytes of fs.createReadStream(row.path))hash.update(bytes);
  const after=await fsp.stat(row.path);
  if(!samePath(await fsp.realpath(row.path),row.path))throw Error('Recording alias is excluded: '+row.path);
  if(before.size!==after.size||before.mtimeMs!==after.mtimeMs||hash.digest('hex')!==row.sha256)throw Error('Source bytes changed since audit: '+row.path);
  return after.mtimeMs;
}
function validateRow(row){
  if(!hashPattern.test(row.sha256)||!roles.includes(row.requested_role)||typeof row.path!=='string'||!path.isAbsolute(row.path)
    ||!Number.isSafeInteger(row.bytes)||row.bytes<0||!Array.isArray(row.known_session_ids)||row.known_session_ids.some(value=>typeof value!=='string')
    ||!Array.isArray(row.matching_imports)||row.matching_imports.some(item=>typeof item?.clip!=='string'))throw Error('Invalid audited source identity');
  if(row.folder_declared_domain!==undefined&&row.folder_declared_domain!==null&&!['iphone','match'].includes(row.folder_declared_domain))throw Error('Invalid folder-declared domain');
  if(row.effective_role!==undefined&&!['train','val','test','unassigned'].includes(row.effective_role))throw Error('Invalid effective role');
  if(row.source_mtime_ms!==undefined&&!Number.isFinite(row.source_mtime_ms))throw Error('Invalid source modification time');
  if(row.reason_codes!==undefined&&(!Array.isArray(row.reason_codes)||row.reason_codes.some(value=>typeof value!=='string')))throw Error('Invalid source reason codes');
  if(row.prior_development_evidence!==undefined&&(!Array.isArray(row.prior_development_evidence)
    ||row.prior_development_evidence.some(item=>typeof item?.report!=='string'||!hashPattern.test(item.report_sha256))))throw Error('Invalid prior exposure evidence');
  if(row.mixed_match_highlights!==undefined&&typeof row.mixed_match_highlights!=='boolean')throw Error('Invalid compilation declaration');
}
async function prepare({input,output,publicMatchSources=false}){
  if(!input||!output||fs.existsSync(output))throw Error('Usage: node tools/prepare_development_folder_split.cjs FOLDER_AUDIT.json NEW_MANIFEST.json [--public-match-sources]');
  const bytes=await fsp.readFile(input),audit=JSON.parse(bytes);
  if(audit.schema_version!==1||!Array.isArray(audit.rows)||audit.production_labels_modified!==false
    ||audit.dataset_partitions_modified===true)throw Error('Unmodified audited folder inventory required');
  // Validate all rows before reading sources. Do not infer independence from a
  // directory called seal; source/session provenance is a separate requirement.
  audit.rows.forEach(validateRow);
  const sessions=new Map(),duplicates=new Map(),sources=[];
  for(const row of audit.rows){
    const mtime=await sourceIdentity(row);
    for(const session of row.known_session_ids){const values=sessions.get(session)??new Set();values.add(row.requested_role);sessions.set(session,values);}
    const values=duplicates.get(row.sha256)??new Set();values.add(row.requested_role);duplicates.set(row.sha256,values);
    const knownIphone=row.known_session_ids.includes('iphone_same_venue_session_1');
    const domain=knownIphone?'iphone':row.folder_declared_domain??(publicMatchSources&&row.matching_imports.length===0?'match':null);
    const mixed=Boolean(row.mixed_match_highlights||(domain==='match'&&audit.mixed_match_highlights_declared));
    const reasonCodes=[...(row.reason_codes??[])];
    if(mixed&&!reasonCodes.includes('MIXED_MATCH_COMPILATION_NOT_INDEPENDENT'))reasonCodes.push('MIXED_MATCH_COMPILATION_NOT_INDEPENDENT');
    sources.push({path:row.path,relative_path:row.relative_path??null,sha256:row.sha256,bytes:row.bytes,source_mtime_ms:mtime,
      requested_split:row.requested_role,effective_role:knownIphone?'train':row.effective_role??'unassigned',
      known_session_ids:row.known_session_ids,matching_review_captures:row.matching_imports.map(item=>item.clip),
      session_group_status:row.known_session_ids.length?'KNOWN':'UNASSIGNED',folder_declared_domain:row.folder_declared_domain??null,
      declared_domain:domain,domain_provenance:knownIphone?'Existing original SHA and capture session metadata':
        row.folder_declared_domain?'User-organized footage-type folder; device/session provenance remains unverified':
          domain==='match'?'User clarified added clips are official match footage; exact original match/session grouping remains unverified':null,
      mixed_match_highlights:mixed,prior_development_evidence:row.prior_development_evidence??[],reason_codes:reasonCodes,reason:row.reason??null,
      independent_release_evidence:false});
  }
  const report={schema_version:1,purpose:'DEVELOPMENT_CLIP_HOLDOUT_ONLY',created_at:new Date().toISOString(),inventory_sha256:digest(bytes),
    freshness_checked_by:'STREAM_SHA256_AND_STAT',sources,
    shared_sessions:[...sessions].filter(([,values])=>values.size>1).map(([session,values])=>({session_id:session,splits:[...values].sort()})),
    exact_cross_split_duplicates:[...duplicates].filter(([,values])=>values.size>1).map(([sha256,values])=>({sha256,splits:[...values].sort()})),
    folder_counts:Object.fromEntries(roles.map(role=>[role,sources.filter(row=>row.requested_split===role).length])),
    independent_release_evidence:false,production_registry_modified:false,sealed_test_labels_consumed:false,contains_automatically_reviewed_ground_truth:false,
    pc_camera_sources_supplied:publicMatchSources?false:null,
    outstanding:['Mixed-match highlights remain development-only; original-source/overlap audit is required before training',
      'All new clips require explicitly reviewed ball targets and absences','Previously reviewed iPhone clips remain one training session regardless of folder',
      'Actual PC-camera and independent paired A/B/C session footage remains required for the agreed release']};
  await fsp.writeFile(output,JSON.stringify(report,null,2)+'\n',{flag:'wx'});
  return report;
}
async function main(){
  const [input,output,...options]=process.argv.slice(2);
  if(options.some(option=>option!=='--public-match-sources'))throw Error('Unknown folder split option');
  const report=await prepare({input,output,publicMatchSources:options.includes('--public-match-sources')});
  console.log(JSON.stringify({folder_counts:report.folder_counts,shared_sessions:report.shared_sessions,independent_release_evidence:false}));
}
module.exports={prepare,main};
if(require.main===module)main().catch(error=>{console.error(error.stack);process.exitCode=1;});

'use strict';
// Preserve the requested clip split for development comparisons, separately
// from the independent-session registry used by production fitting/release.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const [input,output,...options]=process.argv.slice(2);
if(options.some(option=>option!=='--public-match-sources'))throw Error('Unknown folder split option');
const publicMatchSources=options.includes('--public-match-sources');
if(!input||!output||fs.existsSync(output))throw Error('Usage: node tools/prepare_development_folder_split.cjs FOLDER_AUDIT.json NEW_MANIFEST.json');
const bytes=fs.readFileSync(input),audit=JSON.parse(bytes);
if(audit.schema_version!==1||!Array.isArray(audit.rows)||audit.production_labels_modified!==false)throw Error('Audited folder inventory required');
const sessions=new Map(),duplicates=new Map();
for(const row of audit.rows){
  if(!/^[a-f0-9]{64}$/.test(row.sha256)||!['train','val','test'].includes(row.requested_role)||!path.isAbsolute(row.path))throw Error('Invalid audited source identity');
  if(!fs.existsSync(row.path)||fs.statSync(row.path).size!==row.bytes)throw Error('Source moved/changed; repeat the full folder hash audit');
  for(const session of row.known_session_ids??[]){const roles=sessions.get(session)??new Set();roles.add(row.requested_role);sessions.set(session,roles);}
  const roles=duplicates.get(row.sha256)??new Set();roles.add(row.requested_role);duplicates.set(row.sha256,roles);
}
const report={schema_version:1,purpose:'DEVELOPMENT_CLIP_HOLDOUT_ONLY',created_at:new Date().toISOString(),inventory_sha256:crypto.createHash('sha256').update(bytes).digest('hex'),
  sources:audit.rows.map(row=>({path:row.path,sha256:row.sha256,bytes:row.bytes,requested_split:row.requested_role,known_session_ids:row.known_session_ids,matching_review_captures:row.matching_imports.map(item=>item.clip),session_group_status:row.known_session_ids.length?'KNOWN':'UNASSIGNED',declared_domain:publicMatchSources&&row.matching_imports.length===0?'match':null,domain_provenance:publicMatchSources&&row.matching_imports.length===0?'User clarified added clips are official match footage; exact original match/session grouping remains unverified':null})),
  shared_sessions:[...sessions].filter(([,roles])=>roles.size>1).map(([session,roles])=>({session_id:session,splits:[...roles].sort()})),
  exact_cross_split_duplicates:[...duplicates].filter(([,roles])=>roles.size>1).map(([digest,roles])=>({sha256:digest,splits:[...roles].sort()})),
  folder_counts:Object.fromEntries(['train','val','test'].map(role=>[role,audit.rows.filter(row=>row.requested_role===role).length])),
  independent_release_evidence:false,production_registry_modified:false,sealed_test_labels_consumed:false,contains_automatically_reviewed_ground_truth:false,
  pc_camera_sources_supplied:publicMatchSources?false:null,
  outstanding:['New public clips require actual match/session grouping and original-source/duplicate audit','All new clips require explicitly reviewed ball targets and absences','Previously reviewed iPhone clips are same-session development material','Actual PC-camera session A/B/C footage remains required for the agreed paired release']};
fs.writeFileSync(output,JSON.stringify(report,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({folder_counts:report.folder_counts,shared_sessions:report.shared_sessions,independent_release_evidence:false}));

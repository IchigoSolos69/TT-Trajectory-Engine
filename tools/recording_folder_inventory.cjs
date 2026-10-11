'use strict';
// Byte identity inventory only: no video decode, annotation edits or split assignment.
const fs=require('node:fs'),fsp=fs.promises,path=require('node:path'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'..');
const roles=[['training','train'],['validation','val'],['seal','test']];
const validCapture=name=>/^(?:live_[0-9]+_[0-9]+|phone_[0-9a-f]{16}_[0-9]+)$/.test(name);
const equalPath=(a,b)=>path.resolve(a).toLowerCase()===path.resolve(b).toLowerCase();
const digest=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
const sha=file=>new Promise((resolve,reject)=>{
  const hash=crypto.createHash('sha256'),stream=fs.createReadStream(file);
  stream.on('data',bytes=>hash.update(bytes));stream.on('error',reject);stream.on('end',()=>resolve(hash.digest('hex')));
});
async function ownedFile(directory,name,optional=false){
  const file=path.join(directory,name);
  try{if(!equalPath(await fsp.realpath(file),file))throw Error('Aliased metadata is excluded: '+file);return await fsp.readFile(file);}
  catch(error){if(optional&&error.code==='ENOENT')return null;throw error;}
}
async function knownSources(captures){
  const known=new Map(),registryRaw=await ownedFile(captures,'training_sessions.json',true);
  const registry=registryRaw?JSON.parse(registryRaw):{captures:{},sessions:{}};
  for(const entry of (await fsp.readdir(captures,{withFileTypes:true})).sort((a,b)=>a.name.localeCompare(b.name))){
    if(!validCapture(entry.name))continue;
    const assignment=registry.captures[entry.name],session=assignment?registry.sessions[assignment.session_id]:null;
    if(session?.sealed||session?.role==='test'){
      if(/^[0-9a-f]{64}$/.test(assignment.source_sha256??'')){
        const list=known.get(assignment.source_sha256)??[];list.push({clip:entry.name,session_id:assignment.session_id,sealed:true});known.set(assignment.source_sha256,list);
      }
      continue; // registry identities only; C evidence is never opened
    }
    if(entry.isSymbolicLink())throw Error('Capture directory alias is excluded: '+entry.name);
    if(!entry.isDirectory())continue;
    const directory=path.join(captures,entry.name);
    if(!equalPath(await fsp.realpath(directory),directory))throw Error('Capture directory alias is excluded: '+entry.name);
    const raw=await ownedFile(directory,'capture.json',true);if(!raw)continue;
    const value=JSON.parse(raw);
    if(/^[0-9a-f]{64}$/.test(value.original_sha256??'')){
      const list=known.get(value.original_sha256)??[];
      list.push({clip:entry.name,session_id:value.session_id,original_path:value.original_path});known.set(value.original_sha256,list);
    }
  }
  return known;
}
async function videoFiles(directory){
  if(!equalPath(await fsp.realpath(directory),directory))throw Error('Video folder alias is excluded: '+directory);
  const files=[];
  for(const entry of (await fsp.readdir(directory,{withFileTypes:true})).sort((a,b)=>a.name.localeCompare(b.name))){
    const file=path.join(directory,entry.name);
    if(entry.isSymbolicLink())throw Error('Recording symlink/junction is excluded: '+file);
    if(entry.isDirectory())files.push(...await videoFiles(file));
    else if(entry.isFile()&&/\.(mov|mp4|avi|mkv|webm|m4v)$/i.test(entry.name)){
      if(!equalPath(await fsp.realpath(file),file))throw Error('Recording alias is excluded: '+file);
      files.push(file);
    }
  }
  return files;
}
async function inventory({downloads,recordingsRoot,captures,mixedMatchHighlights=false,mixedMatchHashes,priorExposure=new Map(),onFile=()=>{}}){
  const scanRoot=recordingsRoot??downloads;
  if(typeof scanRoot!=='string'||!scanRoot.trim())throw Error('A recordings root is required');
  if(mixedMatchHashes!==undefined&&(!(mixedMatchHashes instanceof Set)||[...mixedMatchHashes].some(hash=>!/^[0-9a-f]{64}$/.test(hash))))throw Error('Invalid confirmed compilation SHA identities');
  if(mixedMatchHighlights&&mixedMatchHashes!==undefined)throw Error('Choose either global mixed-match classification or a confirmed compilation manifest');
  const known=await knownSources(captures),rows=[],sources={};
  for(const [folder,requested_role]of roles){
    const directory=path.join(scanRoot,folder);
    for(const file of await videoFiles(directory)){
      const before=await fsp.stat(file),hash=await sha(file),after=await fsp.stat(file);
      if(before.size!==after.size||before.mtimeMs!==after.mtimeMs)throw Error('File changed during hash audit: '+file);
      const imports=known.get(hash)??[],sessions=[...new Set(imports.map(item=>item.session_id).filter(Boolean))];
      const category=path.relative(directory,file).split(path.sep)[0];
      const folderDomain=category==='iphone'?'iphone':category==='world_cup'?'match':null;
      const mixedMatch=(folderDomain==='match'&&mixedMatchHighlights)||(mixedMatchHashes?.has(hash)??false);
      const knownDevelopment=sessions.includes('iphone_same_venue_session_1'),exposure=priorExposure.get(hash)??[];
      const reasons=[];
      if(knownDevelopment&&requested_role!=='train')reasons.push('SAME_IPHONE_SESSION_CROSS_FOLDER');
      if(mixedMatch)reasons.push('MIXED_MATCH_COMPILATION_NOT_INDEPENDENT');
      if(exposure.length&&requested_role==='test')reasons.push('PREVIOUSLY_EXPOSED_SOURCE_NOT_BLIND_TEST');
      if(imports.some(item=>item.sealed)&&requested_role!=='test')reasons.push('SEALED_SOURCE_DEVELOPMENT_ALIAS');
      if(knownDevelopment&&folderDomain==='match')reasons.push('FOLDER_DOMAIN_CONFLICTS_WITH_KNOWN_SESSION');
      rows.push({path:file,relative_path:path.relative(scanRoot,file),sha256:hash,bytes:after.size,source_mtime_ms:after.mtimeMs,
        requested_role,folder_declared_domain:folderDomain,matching_imports:imports,known_session_ids:sessions,
        effective_role:knownDevelopment?'train':'unassigned',independent_release_evidence:false,
        mixed_match_highlights:mixedMatch,prior_development_evidence:exposure,
        reason_codes:reasons,reason:knownDevelopment?'Previously reviewed same-session development footage; folder movement cannot establish independent validation/test':
          mixedMatch?'User-declared mixed-match highlights; review and duplicate/source audit required for training/development, excluded from independent validation/test':
          'Source/session/device provenance and reviewed labels still required'});
      if(imports.length)sources[hash]={paths:[...(sources[hash]?.paths??[]),file],reverified_at:new Date().toISOString()};
      onFile({file,relative_path:path.relative(scanRoot,file),known_original:imports.length>0});
    }
  }
  const byHash=new Map();for(const row of rows){const list=byHash.get(row.sha256)??[];list.push(row);byHash.set(row.sha256,list);}
  return {rows,sources,exact_cross_split_duplicates:[...byHash].filter(([,list])=>new Set(list.map(row=>row.requested_role)).size>1)
    .map(([sha256,list])=>({sha256,paths:list.map(row=>row.path),requested_roles:[...new Set(list.map(row=>row.requested_role))]}))};
}
const usage='Usage: node tools/audit_recording_folders.cjs NEW_REPORT.json [--recordings-root PATH] [--update-source-locations] [--mixed-match-highlights | --mixed-match-manifest FILE]';
function parseInventoryArgs(argv,{defaultRecordingsRoot=path.join(process.env.USERPROFILE??process.cwd(),'Downloads')}={}){
  const [destination,...args]=argv;
  if(typeof destination!=='string'||!destination.trim()||destination.startsWith('--'))throw Error(usage);
  const parsed={destination:path.resolve(destination),recordingsRoot:path.resolve(defaultRecordingsRoot),updateSourceLocations:false,mixedMatchHighlights:false,mixedMatchManifest:null},seen=new Set();
  for(let i=0;i<args.length;i++){
    const option=args[i];
    if(!['--recordings-root','--update-source-locations','--mixed-match-highlights','--mixed-match-manifest'].includes(option)||seen.has(option))throw Error(usage+'; unknown or repeated option: '+option);
    seen.add(option);
    if(option==='--update-source-locations')parsed.updateSourceLocations=true;
    else if(option==='--mixed-match-highlights')parsed.mixedMatchHighlights=true;
    else{
      const value=args[++i];
      if(typeof value!=='string'||!value.trim()||value.startsWith('--'))throw Error(usage+'; missing value for '+option);
      if(option==='--recordings-root')parsed.recordingsRoot=path.resolve(value);
      else parsed.mixedMatchManifest=path.resolve(value);
    }
  }
  if(parsed.mixedMatchHighlights&&parsed.mixedMatchManifest)throw Error(usage+'; classification options are mutually exclusive');
  return parsed;
}
function compilationHashesFromManifest(bytes){
  let value;try{value=JSON.parse(bytes);}catch(error){throw Error('Malformed compilation manifest JSON: '+error.message);}
  if(!value||value.schema_version!==1||!Array.isArray(value.rows))throw Error('Invalid compilation manifest: expected schema_version 1 and rows');
  const hashes=new Set(),declared=new Map();
  for(const row of value.rows){
    if(!row||typeof row!=='object'||Array.isArray(row)||!/^[0-9a-f]{64}$/.test(row.sha256??'')||(row.mixed_match_highlights!==undefined&&typeof row.mixed_match_highlights!=='boolean'))throw Error('Invalid compilation manifest row');
    if(row.mixed_match_highlights===undefined)continue;
    if(declared.has(row.sha256)&&declared.get(row.sha256)!==row.mixed_match_highlights)throw Error('Conflicting compilation declarations for the same SHA identity');
    declared.set(row.sha256,row.mixed_match_highlights);
    if(row.mixed_match_highlights===true)hashes.add(row.sha256);
  }
  return hashes;
}
async function loadMixedMatchManifest(file){
  const absolute=path.resolve(file),bytes=await ownedFile(path.dirname(absolute),path.basename(absolute));
  const hashes=compilationHashesFromManifest(bytes);
  return {hashes,provenance:{path:absolute,sha256:digest(bytes),confirmed_hash_count:hashes.size}};
}
function mergedLocations(previous,sources){
  if(previous&&(previous.schema_version!==1||!previous.sources||typeof previous.sources!=='object'||Array.isArray(previous.sources)))throw Error('Invalid existing source resolver');
  const result={...(previous??{}),schema_version:1,sources:{...(previous?.sources??{})}};
  for(const [hash,entry]of Object.entries(sources)){
    const old=result.sources[hash];
    if(old&&(!Array.isArray(old.paths)||old.paths.some(file=>typeof file!=='string'||!path.isAbsolute(file))))throw Error('Invalid original source locations');
    const unique=new Map();for(const file of [...entry.paths,...(old?.paths??[])])if(!unique.has(file.toLowerCase()))unique.set(file.toLowerCase(),file);
    result.sources[hash]={...(old??{}),...entry,paths:[...unique.values()]};
  }
  return result;
}
async function writeLocations({captures,sources,backup,update=false}){
  const resolver=path.join(captures,'source_locations.json'),lockPath=path.join(captures,'.source_locations.migration.lock');
  const lock=await fsp.open(lockPath,'wx');
  try{
    const old=await ownedFile(captures,'source_locations.json',true);
    if(old&&!update)throw Error('Preserve existing resolver; specify --update-source-locations for an audited migration.');
    const next=Buffer.from(JSON.stringify(mergedLocations(old?JSON.parse(old):null,sources),null,2)+'\n');
    if(old)await fsp.writeFile(backup,old,{flag:'wx'});
    const temp=resolver+'.'+crypto.randomUUID()+'.tmp';await fsp.writeFile(temp,next,{flag:'wx'});
    const current=await ownedFile(captures,'source_locations.json',true);
    if((old?digest(old):null)!==(current?digest(current):null))throw Error('Source resolver changed during migration');
    await fsp.rename(temp,resolver);
    return {path:resolver,sha256:digest(next),previous_sha256:old?digest(old):null,previous_backup:old?backup:null};
  }finally{await lock.close();await fsp.unlink(lockPath);}
}
async function priorExposure(){
  const result=new Map();
  for(const name of ['artifacts/blurball_research/001_comparison/report.json','artifacts/blurball_research/demo5_comparison/report.json','artifacts/analyses/1791703787994746_17624/manifest.json']){
    const file=path.join(root,name);if(!fs.existsSync(file))continue;
    const value=JSON.parse(await fsp.readFile(file));if(!/^[0-9a-f]{64}$/.test(value.source_sha256??''))throw Error('Invalid existing exposure source identity');
    const list=result.get(value.source_sha256)??[];list.push({report:name,report_sha256:await sha(file)});result.set(value.source_sha256,list);
  }
  return result;
}
async function main(){
  const options=parseInventoryArgs(process.argv.slice(2)),destination=options.destination;
  if(fs.existsSync(destination))throw Error('Inventory report already exists; preserve it and choose a new report path');
  const captures=path.join(root,'training_data/live'),compilations=options.mixedMatchManifest?await loadMixedMatchManifest(options.mixedMatchManifest):null;
  const {rows,sources,exact_cross_split_duplicates}=await inventory({recordingsRoot:options.recordingsRoot,captures,mixedMatchHighlights:options.mixedMatchHighlights,mixedMatchHashes:compilations?.hashes,priorExposure:await priorExposure(),onFile:row=>console.log(row.relative_path+': '+(row.known_original?'known original matched':'new/unassigned'))});
  const locations=await writeLocations({captures,sources,backup:destination.replace(/\.json$/i,'')+'.source_locations.before.json',update:options.updateSourceLocations});
  const report={schema_version:1,created_at:new Date().toISOString(),scope:'Recursive source-byte identity audit only; no video decoding, sealed labels/results, annotations or dataset assignment changed',
    rows,source_locations_path:locations.path,source_locations_sha256:locations.sha256,source_locations_migration:locations,
    existing_originals_relocated:Object.keys(sources).length,known_session_split_conflicts:rows.filter(row=>row.reason_codes.includes('SAME_IPHONE_SESSION_CROSS_FOLDER')).length,
    exact_cross_split_duplicates,folder_counts:Object.fromEntries(roles.map(([,role])=>[role,rows.filter(row=>row.requested_role===role).length])),
    recordings_root:options.recordingsRoot,mixed_match_compilation_manifest:compilations?.provenance??null,
    mixed_match_classification_mode:compilations?'confirmed_sha_manifest':options.mixedMatchHighlights?'all_folder_declared_match':'none',
    mixed_match_highlights_declared:options.mixedMatchHighlights,production_labels_modified:false,dataset_partitions_modified:false,ready_for_training:false};
  await fsp.writeFile(destination,JSON.stringify(report,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({files:rows.length,reconnected_originals:report.existing_originals_relocated,same_session_split_conflicts:report.known_session_split_conflicts,folder_counts:report.folder_counts,ready_for_training:false}));
}
module.exports={inventory,mergedLocations,writeLocations,videoFiles,parseInventoryArgs,compilationHashesFromManifest,loadMixedMatchManifest,main};
if(require.main===module)main().catch(error=>{console.error(error.stack);process.exitCode=1;});

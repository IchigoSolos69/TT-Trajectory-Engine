'use strict';
// Optional exact-byte hardlinks for immutable raw diagnostic blobs only. Dry run by default.
const fs=require('node:fs'),fsp=fs.promises,path=require('node:path'),crypto=require('node:crypto');
const actualRoot=path.resolve(__dirname,'..');
const allowedNames=['raw640','raw640_v2','raw960'];
const hashBytes=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
const equalPath=(a,b)=>process.platform==='win32'?path.resolve(a).toLowerCase()===path.resolve(b).toLowerCase():path.resolve(a)===path.resolve(b);
const inside=(root,file)=>{const relative=path.relative(root,file);return relative!==''&&!relative.startsWith('..'+path.sep)&&relative!=='..'&&!path.isAbsolute(relative);};
const id=stat=>stat.dev.toString()+':'+stat.ino.toString();
const snapshot=stat=>({file_id:id(stat),device:stat.dev.toString(),inode:stat.ino.toString(),links:stat.nlink.toString(),bytes:Number(stat.size),mtime_ns:stat.mtimeNs.toString(),mode:stat.mode.toString()});
const stable=(a,b)=>id(a)===id(b)&&a.size===b.size&&a.mtimeNs===b.mtimeNs&&a.mode===b.mode;
async function sha(file){const hash=crypto.createHash('sha256');for await(const bytes of fs.createReadStream(file))hash.update(bytes);return hash.digest('hex');}
function timestampNs(value){
  const match=typeof value==='string'&&/^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d{1,9}))?Z$/.exec(value);
  if(!match)throw Error('Invalid audited timestamp');
  const milliseconds=Date.parse(match[1]+'.000Z');if(!Number.isFinite(milliseconds)||new Date(milliseconds).toISOString().slice(0,19)!==match[1])throw Error('Invalid audited timestamp');
  return BigInt(milliseconds)*1000000n+BigInt((match[2]??'').padEnd(9,'0'));
}
async function noAliases(root,file,{directory=false}={}){
  const absolute=path.resolve(file);
  if(!inside(root,absolute))throw Error('Path outside intended repository root: '+file);
  if(!equalPath(await fsp.realpath(root),root))throw Error('Repository root alias is excluded');
  let current=root;
  for(const component of path.relative(root,absolute).split(path.sep)){
    current=path.join(current,component);const stat=await fsp.lstat(current,{bigint:true});
    if(stat.isSymbolicLink())throw Error('Symlink/junction alias is excluded: '+current);
  }
  if(!equalPath(await fsp.realpath(absolute),absolute))throw Error('Canonical path alias is excluded: '+absolute);
  const stat=await fsp.stat(absolute,{bigint:true});
  if(directory?!stat.isDirectory():!stat.isFile())throw Error('Unexpected path type: '+absolute);
  return stat;
}
function allowedBlob(root,file){
  if(typeof file!=='string'||!path.isAbsolute(file)||path.normalize(file)!==file)throw Error('Audited blob needs a canonical absolute path');
  const base=path.join(root,'artifacts','yolo_upgrade_2026_10_10_v1');
  if(!allowedNames.some(name=>inside(path.join(base,name),file))||!['.bgr8','.f32'].includes(path.extname(file)))throw Error('Only scoped raw tensor blobs may be deduplicated: '+file);
  return path.resolve(file);
}
function validateAudit(value,root){
  if(!value||value.schema_version!==1||!Array.isArray(value.files)||!Array.isArray(value.exact_duplicate_groups))throw Error('Invalid raw tensor audit schema');
  const files=new Map();
  for(const row of value.files){
    if(!row||!/^[0-9a-f]{64}$/.test(row.sha256??'')||!Number.isSafeInteger(row.bytes)||row.bytes<0)throw Error('Invalid audited blob identity');
    const file=allowedBlob(root,row.path);timestampNs(row.mtime_utc);
    const key=process.platform==='win32'?file.toLowerCase():file;
    if(files.has(key))throw Error('Duplicate audited pathname');files.set(key,{...row,path:file});
  }
  const used=new Set(),groups=[];
  for(const group of value.exact_duplicate_groups){
    if(!group||!/^[0-9a-f]{64}$/.test(group.sha256??'')||!Number.isSafeInteger(group.bytes)||group.bytes<0||!Array.isArray(group.paths)||group.paths.length<2||group.copies!==group.paths.length)throw Error('Invalid exact duplicate group');
    const rows=group.paths.map(file=>{
      file=allowedBlob(root,file);const key=process.platform==='win32'?file.toLowerCase():file,row=files.get(key);
      if(!row||row.sha256!==group.sha256||row.bytes!==group.bytes||used.has(key))throw Error('Duplicate group disagrees with audited files');used.add(key);return row;
    });groups.push({sha256:group.sha256,bytes:group.bytes,rows});
  }
  return {files:[...files.values()],groups};
}
async function verifyBlob(root,row){
  const before=await noAliases(root,row.path);
  if(before.size!==BigInt(row.bytes)||before.mtimeNs!==timestampNs(row.mtime_utc))throw Error('Audited size/timestamp changed: '+row.path);
  const hash=await sha(row.path),after=await noAliases(root,row.path);
  if(hash!==row.sha256||!stable(before,after))throw Error('Audited bytes or file identity changed: '+row.path);
  return after;
}
async function planDeduplication({audit,root=actualRoot}){
  root=path.resolve(root);const parsed=validateAudit(audit,root),verified=new Map();
  // Preflight every audited blob before creating any links.
  for(const row of parsed.files)verified.set(row.path,await verifyBlob(root,row));
  const actions=[],skipped=[];
  for(const group of parsed.groups){
    const canonical=group.rows[0],canonicalStat=verified.get(canonical.path);
    for(const target of group.rows.slice(1)){
      const targetStat=verified.get(target.path);let reason=null;
      if(id(targetStat)===id(canonicalStat))reason='ALREADY_HARDLINKED';
      else if(targetStat.dev!==canonicalStat.dev)reason='DIFFERENT_DEVICE';
      else if(targetStat.mtimeNs!==canonicalStat.mtimeNs)reason='DIFFERENT_TIMESTAMP';
      else if(targetStat.mode!==canonicalStat.mode)reason='DIFFERENT_MODE';
      if(reason)skipped.push({canonical:canonical.path,target:target.path,reason,bytes:target.bytes});
      else actions.push({canonical,target,before:snapshot(targetStat),canonical_before:snapshot(canonicalStat)});
    }
  }
  return {root,actions,skipped,verified_files:parsed.files.length};
}
async function replaceWithHardlink({root,action,operations=fsp}){
  const {canonical,target}=action;
  const canonicalBefore=await verifyBlob(root,canonical),targetBefore=await verifyBlob(root,target);
  if(id(canonicalBefore)===id(targetBefore))return {target:target.path,canonical:canonical.path,status:'ALREADY_HARDLINKED',reclaimed_unique_content_bytes:0};
  if(canonicalBefore.dev!==targetBefore.dev||canonicalBefore.mtimeNs!==targetBefore.mtimeNs||canonicalBefore.mode!==targetBefore.mode)throw Error('Hardlink metadata/device mismatch');
  if(id(targetBefore)!==action.before.file_id||id(canonicalBefore)!==action.canonical_before.file_id)throw Error('Planned file identity changed');
  const token=crypto.randomUUID(),temp=target.path+'.dedup-'+token+'.tmp',backup=target.path+'.dedup-'+token+'.rollback';
  let linked=false,backedUp=false,replaced=false;
  try{
    await operations.link(canonical.path,temp);linked=true;
    await operations.link(target.path,backup);backedUp=true;
    const lastTarget=await noAliases(root,target.path),lastCanonical=await noAliases(root,canonical.path),tempStat=await noAliases(root,temp);
    if(!stable(lastTarget,targetBefore)||!stable(lastCanonical,canonicalBefore)||id(tempStat)!==id(canonicalBefore))throw Error('File changed before atomic replacement');
    await operations.rename(temp,target.path);linked=false;replaced=true;
    const after=await verifyBlob(root,target),canonicalAfter=await verifyBlob(root,canonical);
    if(id(after)!==id(canonicalAfter)||id(after)!==id(canonicalBefore))throw Error('Hardlink verification failed');
    await operations.unlink(backup);backedUp=false;
    return {target:target.path,canonical:canonical.path,status:'DEDUPLICATED',sha256:target.sha256,bytes:target.bytes,before:snapshot(targetBefore),after:snapshot(after),reclaimed_unique_content_bytes:targetBefore.nlink===1n?target.bytes:0};
  }catch(error){
    if(replaced&&backedUp){
      try{await operations.rename(backup,target.path);backedUp=false;const restored=await verifyBlob(root,target);if(id(restored)!==id(targetBefore))throw Error('Rollback identity mismatch');}
      catch(rollbackError){throw new AggregateError([error,rollbackError],'Deduplication failed; rollback evidence retained at '+backup);}
    }
    throw error;
  }finally{
    // Exact owned temporary names only; original source/target paths are never unlinked.
    if(linked)await operations.unlink(temp);
    if(backedUp&&!replaced)await operations.unlink(backup);
  }
}
async function deduplicate({audit,root=actualRoot,apply=false,operations=fsp}){
  const plan=await planDeduplication({audit,root}),results=[];
  if(apply)for(const action of plan.actions)results.push(await replaceWithHardlink({root:plan.root,action,operations}));
  return {schema_version:1,created_at:new Date().toISOString(),scope:'Exact SHA/size/timestamp-identical immutable raw diagnostic blobs only; all pathnames and non-blob evidence preserved',applied:apply,verified_files:plan.verified_files,
    actions:plan.actions.map(action=>({canonical:action.canonical.path,target:action.target.path,sha256:action.target.sha256,bytes:action.target.bytes,expected_file_id:action.before.file_id})),skipped:plan.skipped,results,
    reclaimed_unique_content_bytes:results.reduce((total,row)=>total+row.reclaimed_unique_content_bytes,0),note:'Unique content bytes freed by removing the final redundant inode link; not a measured filesystem free-space delta'};
}
function parseArgs(argv){
  const [audit,report,...options]=argv;
  if(!audit||!report||audit.startsWith('--')||report.startsWith('--')||options.some(option=>option!=='--apply')||options.length>1)throw Error('Usage: node tools/deduplicate_raw_tensor_evidence.cjs AUDIT.json NEW_REPORT.json [--apply]');
  return {audit:path.resolve(audit),report:path.resolve(report),apply:options.includes('--apply')};
}
async function main(){
  const options=parseArgs(process.argv.slice(2));
  await noAliases(actualRoot,options.audit);
  if(!inside(actualRoot,options.report)||path.extname(options.report)!=='.json'||fs.existsSync(options.report))throw Error('Choose a new JSON report inside the repository');
  await noAliases(actualRoot,path.dirname(options.report),{directory:true});
  const auditBytes=await fsp.readFile(options.audit),audit=JSON.parse(auditBytes),lockPath=path.join(actualRoot,'artifacts','yolo_upgrade_2026_10_10_v1','.raw_tensor_dedup.lock');
  if(options.apply)await noAliases(actualRoot,path.dirname(lockPath),{directory:true});
  const lock=options.apply?await fsp.open(lockPath,'wx'):null;
  try{
    if(hashBytes(await fsp.readFile(options.audit))!==hashBytes(auditBytes))throw Error('Audit changed before deduplication');
    const report=await deduplicate({...options,audit,root:actualRoot});
    report.audit={path:options.audit,sha256:hashBytes(auditBytes)};
    await fsp.writeFile(options.report,JSON.stringify(report,null,2)+'\n',{flag:'wx'});
    console.log(JSON.stringify({applied:report.applied,verified_files:report.verified_files,eligible:report.actions.length,skipped:report.skipped.length,reclaimed_unique_content_bytes:report.reclaimed_unique_content_bytes}));
  }finally{if(lock){await lock.close();await fsp.unlink(lockPath);}}
}
module.exports={timestampNs,allowedBlob,validateAudit,planDeduplication,replaceWithHardlink,deduplicate,parseArgs};
if(require.main===module)main().catch(error=>{console.error(error.stack);process.exitCode=1;});

'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),fsp=fs.promises,path=require('node:path'),crypto=require('node:crypto');
const {timestampNs,allowedBlob,validateAudit,planDeduplication,replaceWithHardlink,deduplicate,parseArgs}=require('../deduplicate_raw_tensor_evidence.cjs');
const fixtureParent=path.resolve(__dirname,'../../artifacts'),fixedTime=new Date('2026-09-01T10:00:00.000Z');
const hash=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
async function fixture(t){
  const root=await fsp.mkdtemp(path.join(fixtureParent,'tensor-dedup-tests-'));
  t.after(async()=>{assert.equal(path.dirname(root),fixtureParent);assert.ok(path.basename(root).startsWith('tensor-dedup-tests-'));await fsp.rm(root,{recursive:true,force:true});});
  const base=path.join(root,'artifacts','yolo_upgrade_2026_10_10_v1');
  for(const name of ['raw640','raw640_v2','raw960'])await fsp.mkdir(path.join(base,name),{recursive:true});
  return {root,base};
}
async function blob(f,relative,bytes='same immutable bytes',time=fixedTime){
  const file=path.join(f.base,relative);await fsp.mkdir(path.dirname(file),{recursive:true});await fsp.writeFile(file,bytes);await fsp.utimes(file,time,time);
  return {path:file,bytes:Buffer.byteLength(bytes),mtime_utc:time.toISOString(),sha256:hash(bytes)};
}
const audit=(rows,groups=[rows])=>({schema_version:1,files:rows,exact_duplicate_groups:groups.map(items=>({sha256:items[0].sha256,bytes:items[0].bytes,copies:items.length,paths:items.map(item=>item.path)}))});
const fileId=stat=>stat.dev.toString()+':'+stat.ino.toString();
async function identity(file){return fileId(await fsp.stat(file,{bigint:true}));}
async function ownedTemps(f){const found=[];for(const name of ['raw640','raw640_v2','raw960'])for(const entry of await fsp.readdir(path.join(f.base,name),{recursive:true}))if(entry.includes('.dedup-'))found.push(entry);return found;}

test('default dry run hashes every blob and changes neither allocation identities nor adjacent contracts',async t=>{
  const f=await fixture(t),a=await blob(f,'raw640/case/input.f32'),b=await blob(f,'raw640_v2/case/input.f32');
  const contract=path.join(f.base,'raw640/case/contract.json');await fsp.writeFile(contract,'original contract');
  const originalIds=[await identity(a.path),await identity(b.path)],result=await deduplicate({...f,audit:audit([a,b])});
  assert.equal(result.applied,false);assert.equal(result.actions.length,1);assert.equal(result.results.length,0);assert.equal(result.reclaimed_unique_content_bytes,0);
  assert.deepEqual([await identity(a.path),await identity(b.path)],originalIds);assert.equal(await fsp.readFile(contract,'utf8'),'original contract');assert.deepEqual(await ownedTemps(f),[]);
});

test('atomic hardlink replacement preserves all paths bytes and timestamps with verified common file IDs',async t=>{
  const f=await fixture(t),a=await blob(f,'raw640/case/source.bgr8'),b=await blob(f,'raw960/case/source.bgr8');
  const before=await fsp.stat(b.path,{bigint:true}),originalCanonical=await identity(a.path),result=await deduplicate({...f,audit:audit([a,b]),apply:true});
  assert.equal(result.results[0].status,'DEDUPLICATED');assert.equal(await identity(a.path),originalCanonical);assert.equal(await identity(b.path),originalCanonical);
  assert.equal(hash(await fsp.readFile(b.path)),b.sha256);assert.equal((await fsp.stat(b.path,{bigint:true})).mtimeNs,before.mtimeNs);assert.equal(result.reclaimed_unique_content_bytes,b.bytes);assert.deepEqual(await ownedTemps(f),[]);
});

test('already linked duplicates reclaim zero bytes on repeated application',async t=>{
  const f=await fixture(t),a=await blob(f,'raw640/input.f32'),b={...a,path:path.join(f.base,'raw640_v2/input.f32')};await fsp.link(a.path,b.path);
  const result=await deduplicate({...f,audit:audit([a,b]),apply:true});assert.equal(result.actions.length,0);assert.equal(result.skipped[0].reason,'ALREADY_HARDLINKED');assert.equal(result.reclaimed_unique_content_bytes,0);assert.equal((await fsp.stat(a.path,{bigint:true})).nlink,2n);
});

test('outside links are preserved and do not inflate reclaimed unique content bytes',async t=>{
  const f=await fixture(t),a=await blob(f,'raw640/input.f32'),b=await blob(f,'raw640_v2/input.f32'),outside=path.join(f.root,'preserved external name.f32');await fsp.link(b.path,outside);
  const oldOutside=await identity(outside),result=await deduplicate({...f,audit:audit([a,b]),apply:true});
  assert.equal(result.reclaimed_unique_content_bytes,0);assert.equal(await identity(outside),oldOutside);assert.notEqual(await identity(outside),await identity(b.path));assert.equal(hash(await fsp.readFile(outside)),b.sha256);
});

test('identical bytes with differing timestamps are excluded without changing any mtime',async t=>{
  const f=await fixture(t),a=await blob(f,'raw640/input.f32'),b=await blob(f,'raw640_v2/input.f32','same immutable bytes',new Date(fixedTime.getTime()+1000));
  const before=[await identity(a.path),await identity(b.path)],result=await deduplicate({...f,audit:audit([a,b]),apply:true});assert.equal(result.actions.length,0);assert.equal(result.skipped[0].reason,'DIFFERENT_TIMESTAMP');assert.deepEqual([await identity(a.path),await identity(b.path)],before);assert.equal((await fsp.stat(b.path,{bigint:true})).mtimeNs,timestampNs(b.mtime_utc));
});

test('different current devices are excluded before any hardlink mutation',async t=>{
  const f=await fixture(t),a=await blob(f,'raw640/input.f32'),b=await blob(f,'raw640_v2/input.f32'),stat=fsp.stat;
  fsp.stat=async(file,...args)=>{const value=await stat(file,...args);if(file===b.path)value.dev+=1n;return value;};
  try{const result=await deduplicate({...f,audit:audit([a,b]),apply:true});assert.equal(result.actions.length,0);assert.equal(result.skipped[0].reason,'DIFFERENT_DEVICE');assert.equal(result.reclaimed_unique_content_bytes,0);}finally{fsp.stat=stat;}
  assert.notEqual(await identity(a.path),await identity(b.path));assert.deepEqual(await ownedTemps(f),[]);
});

test('current size hash or timestamp mismatch aborts before any mutation',async t=>{
  const f=await fixture(t),a=await blob(f,'raw640/input.f32'),b=await blob(f,'raw640_v2/input.f32'),idA=await identity(a.path),idB=await identity(b.path);
  for(const changes of [{bytes:b.bytes+1},{sha256:hash('wrong bytes')},{mtime_utc:new Date(fixedTime.getTime()+1000).toISOString()}])await assert.rejects(deduplicate({...f,audit:audit([a,{...b,...changes}],[[a,b]]),apply:true}),/disagrees|changed/);
  // Same-size replacement evades a size-only check but must fail SHA verification.
  await fsp.writeFile(b.path,'xxxxxxxxxxxxxxxxxxxx');await fsp.utimes(b.path,fixedTime,fixedTime);
  await assert.rejects(deduplicate({...f,audit:audit([a,b]),apply:true}),/bytes|size/);assert.equal(await identity(a.path),idA);assert.equal(await identity(b.path),idB);assert.deepEqual(await ownedTemps(f),[]);
});

test('whole-audit preflight prevents earlier valid groups being modified when a later blob is corrupt',async t=>{
  const f=await fixture(t),a=await blob(f,'raw640/input.f32'),b=await blob(f,'raw640_v2/input.f32'),c=await blob(f,'raw640/output.f32','different bytes'),d=await blob(f,'raw640_v2/output.f32','different bytes');
  const before=[await identity(a.path),await identity(b.path)];await fsp.writeFile(d.path,'corrupt');
  await assert.rejects(deduplicate({...f,audit:audit([a,b,c,d],[[a,b],[c,d]]),apply:true}),/changed/);assert.deepEqual([await identity(a.path),await identity(b.path)],before);
});

test('path scope and schema exclude models videos contracts traversal and unlisted duplicate group paths',async t=>{
  const f=await fixture(t),a=await blob(f,'raw640/input.f32'),b=await blob(f,'raw640_v2/input.f32');
  for(const file of [path.join(f.root,'models','model.f32'),path.join(f.base,'raw640','contract.json'),path.join(f.base,'raw640','source.mp4'),path.join(f.base,'raw640','..','escape.f32'),path.join(f.base,'raw640_evil','input.f32'),path.join(f.base,'raw640')+path.sep+'..'+path.sep+'raw640'+path.sep+'input.f32'])assert.throws(()=>allowedBlob(f.root,file),/scoped|canonical/);
  assert.throws(()=>validateAudit({schema_version:2,files:[],exact_duplicate_groups:[]},f.root),/schema/);
  assert.throws(()=>validateAudit(audit([a,b,a]),f.root),/Duplicate audited/);
  const invalid=audit([a,b]);invalid.exact_duplicate_groups[0].paths[1]=path.join(f.base,'raw960/unlisted.f32');assert.throws(()=>validateAudit(invalid,f.root),/disagrees/);
});

test('junction aliases are rejected even for exact bytes inside apparent allowed directories',async t=>{
  const f=await fixture(t),a=await blob(f,'raw640/input.f32'),outside=path.join(f.root,'outside directory');await fsp.mkdir(outside);const target=path.join(outside,'input.f32');await fsp.writeFile(target,'same immutable bytes');await fsp.utimes(target,fixedTime,fixedTime);
  const alias=path.join(f.base,'raw640_v2/alias');await fsp.symlink(outside,alias,process.platform==='win32'?'junction':'dir');const b={...a,path:path.join(alias,'input.f32')};
  await assert.rejects(deduplicate({...f,audit:audit([a,b]),apply:true}),/alias|junction/);assert.equal(hash(await fsp.readFile(target)),a.sha256);
});

test('atomic rename failure leaves the original target inode intact and cleans owned links',async t=>{
  const f=await fixture(t),a=await blob(f,'raw640/input.f32'),b=await blob(f,'raw640_v2/input.f32'),before=await identity(b.path);
  const operations={...fsp,rename:async()=>{throw Object.assign(Error('injected rename I/O failure'),{code:'EIO'});}};
  await assert.rejects(deduplicate({...f,audit:audit([a,b]),apply:true,operations}),/injected rename/);assert.equal(await identity(b.path),before);assert.equal(hash(await fsp.readFile(b.path)),b.sha256);assert.deepEqual(await ownedTemps(f),[]);assert.equal((await fsp.stat(a.path,{bigint:true})).nlink,1n);
});

test('hardlink creation failure leaves original target and no temporary paths',async t=>{
  const f=await fixture(t),a=await blob(f,'raw640/input.f32'),b=await blob(f,'raw640_v2/input.f32'),before=await identity(b.path);
  const operations={...fsp,link:async()=>{throw Object.assign(Error('injected link failure'),{code:'EPERM'});}};
  await assert.rejects(deduplicate({...f,audit:audit([a,b]),apply:true,operations}),/injected link/);assert.equal(await identity(b.path),before);assert.deepEqual(await ownedTemps(f),[]);
});

test('post-rename verification failure restores exact original inode through rollback link',async t=>{
  const f=await fixture(t),a=await blob(f,'raw640/input.f32'),b=await blob(f,'raw640_v2/input.f32'),before=await identity(b.path);
  let renamed=false;const operations={...fsp,rename:async(from,to)=>{if(from.endsWith('.tmp')){await fsp.rename(from,to);renamed=true;await fsp.unlink(to);await fsp.writeFile(to,'bad replacement');await fsp.utimes(to,fixedTime,fixedTime);}else await fsp.rename(from,to);}};
  await assert.rejects(deduplicate({...f,audit:audit([a,b]),apply:true,operations}),/changed/);assert.equal(renamed,true);assert.equal(await identity(b.path),before);assert.equal(hash(await fsp.readFile(b.path)),b.sha256);assert.deepEqual(await ownedTemps(f),[]);
});

test('rollback cleanup failure restores the original target rather than reporting freed content',async t=>{
  const f=await fixture(t),a=await blob(f,'raw640/input.f32'),b=await blob(f,'raw640_v2/input.f32'),before=await identity(b.path);
  const operations={...fsp,unlink:async(file)=>{if(file.endsWith('.rollback'))throw Error('injected rollback-link cleanup failure');return fsp.unlink(file);}};
  await assert.rejects(deduplicate({...f,audit:audit([a,b]),apply:true,operations}),/injected rollback-link cleanup/);assert.equal(await identity(b.path),before);assert.equal(hash(await fsp.readFile(b.path)),b.sha256);assert.deepEqual(await ownedTemps(f),[]);assert.equal((await fsp.stat(a.path,{bigint:true})).nlink,1n);
});

test('replaced planned inode is rejected even when byte hash size and timestamp match',async t=>{
  const f=await fixture(t),a=await blob(f,'raw640/input.f32'),b=await blob(f,'raw640_v2/input.f32'),plan=await planDeduplication({...f,audit:audit([a,b])});
  const replacement=b.path+'.replacement';await fsp.writeFile(replacement,'same immutable bytes');await fsp.utimes(replacement,fixedTime,fixedTime);await fsp.rename(replacement,b.path);
  await assert.rejects(replaceWithHardlink({root:f.root,action:plan.actions[0]}),/Planned file identity/);assert.notEqual(await identity(a.path),await identity(b.path));assert.deepEqual(await ownedTemps(f),[]);
});

test('timestamps retain 100 ns audit precision and CLI only accepts explicit optional apply',()=>{
  assert.equal(timestampNs('2026-09-01T10:00:00.1234567Z')-timestampNs('2026-09-01T10:00:00.000Z'),123456700n);
  assert.throws(()=>timestampNs('invalid'),/timestamp/);assert.throws(()=>timestampNs('2026-02-30T10:00:00.000Z'),/timestamp/);assert.equal(parseArgs(['audit.json','report.json']).apply,false);assert.equal(parseArgs(['audit.json','report.json','--apply']).apply,true);
  for(const args of [[],['audit.json'],['a','b','--force'],['a','b','--apply','--apply']])assert.throws(()=>parseArgs(args),/Usage/);
});

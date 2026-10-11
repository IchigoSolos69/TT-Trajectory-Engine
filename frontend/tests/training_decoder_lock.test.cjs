'use strict';
// Exercise actual ownership operations using isolated local files and simulated
// process liveness. No production locks or processes are touched.
const fs=require('node:fs'),fsp=require('node:fs/promises'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const ts=require('../node_modules/typescript');
const load=(file,deps,extra={})=>{const scope={exports:{},require:name=>deps[name],...extra};vm.runInNewContext(ts.transpileModule(fs.readFileSync(file,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020,esModuleInterop:true}}).outputText,scope);return scope.exports;};
async function main(){
 const directory=await fsp.mkdtemp(path.resolve(__dirname,'../../artifacts/decoder-lock-')),lockPath=path.join(directory,'.prepare.lock'),alive=new Set([100]);
 const interval=load(path.join(__dirname,'../src/lib/trainingAbsenceIntervals.ts'),{}),helper=load(path.join(__dirname,'../src/lib/trainingDecoderLock.ts'),{'node:fs/promises':fsp,'node:path':path,'./trainingAbsenceIntervals':interval},{process:{kill(pid){if(!alive.has(pid)){const error=Error('dead');error.code='ESRCH';throw error;}}}});
 let checks=0;const raw=(id,pid,owner)=>JSON.stringify({interval_id:id,process_id:pid,owner_pid:owner})+'\n',a='aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa',b='bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb';
 let owned=await helper.acquireDecoderLock(directory),saved=raw(a,200,100);await owned.lock.writeFile(saved);
 await assert.rejects(()=>helper.acquireDecoderLock(directory),/finishing its audit/);checks++;
 // A's worker died, but A's live frontend is still awaiting its delayed save.
 await owned.lock.close();await assert.rejects(()=>helper.acquireDecoderLock(directory),/finishing its audit/);assert.equal(await fsp.readFile(lockPath,'utf8'),saved);checks++;
 await fsp.unlink(lockPath);const oldHandle=await fsp.open(lockPath,'wx');await oldHandle.writeFile(saved);await fsp.unlink(lockPath);const later=await fsp.open(lockPath,'wx'),laterRaw=raw(b,201,101);await later.writeFile(laterRaw);
 await helper.releaseDecoderLock(oldHandle,lockPath,saved);assert.equal(await fsp.readFile(lockPath,'utf8'),laterRaw,'Delayed old finalizer must preserve B ownership');checks++;
 await helper.releaseDecoderLock(later,lockPath,laterRaw);assert(!fs.existsSync(lockPath));checks++;
 await fsp.writeFile(lockPath,raw(a,200,101));owned=await helper.acquireDecoderLock(directory);await owned.lock.writeFile(laterRaw);assert.equal(await fsp.readFile(lockPath,'utf8'),laterRaw);assert(!fs.existsSync(path.join(directory,'.prepare-recovery.lock')));checks++;
 await helper.releaseDecoderLock(owned.lock,lockPath,laterRaw);
 await fsp.writeFile(lockPath,raw(a,200,101));const concurrent=await Promise.allSettled([helper.acquireDecoderLock(directory),helper.acquireDecoderLock(directory)]);assert.equal(concurrent.filter(value=>value.status==='fulfilled').length,1,'Only one decoder may acquire ownership');assert.equal(concurrent.filter(value=>value.status==='rejected').length,1);checks++;
 const winner=concurrent.find(value=>value.status==='fulfilled').value;await winner.lock.writeFile(laterRaw);await helper.releaseDecoderLock(winner.lock,lockPath,laterRaw);
 owned=await helper.acquireDecoderLock(directory);await owned.lock.writeFile(saved);await fsp.unlink(lockPath);await helper.releaseDecoderLock(owned.lock,lockPath,saved);checks++;
 await fsp.writeFile(lockPath,'incomplete');await assert.rejects(()=>helper.acquireDecoderLock(directory),/incomplete/);assert.equal(await fsp.readFile(lockPath,'utf8'),'incomplete');checks++;
 const report={passed:true,checks,fixture_root:directory,scope:'Exclusive decoder ownership, dead-owner recovery, concurrent prepares and delayed finalizers; no processes killed'};await fsp.writeFile(path.resolve(__dirname,'../../artifacts/training_decoder_lock_checks.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
}
main().catch(error=>{console.error(error.stack??error);process.exitCode=1;});

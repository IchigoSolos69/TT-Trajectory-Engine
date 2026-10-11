import fs from 'node:fs/promises';
import path from 'node:path';
import {validIntervalId} from './trainingAbsenceIntervals';

const alive=(pid:number)=>{try{process.kill(pid,0);return true;}catch(error){if((error as NodeJS.ErrnoException).code==='ESRCH')return false;throw error;}};
export async function acquireDecoderLock(collection:string){
  const lockPath=path.join(collection,'.prepare.lock');
  try{return {lock:await fs.open(lockPath,'wx'),lockPath};}
  catch(error){if((error as NodeJS.ErrnoException).code!=='EEXIST')throw error;}
  const previous=await fs.readFile(lockPath,'utf8');let owner;
  try{owner=JSON.parse(previous);}catch{throw Error('A decoder is starting or its lock is incomplete. Wait and inspect the saved job log; no lock was removed.');}
  if(!validIntervalId(owner.interval_id)||!Number.isSafeInteger(owner.process_id)||owner.process_id<1||!Number.isSafeInteger(owner.owner_pid)||owner.owner_pid<1)throw Error('The previous decoder lock is incomplete. Inspect its job log before retrying.');
  // A dead worker with a live frontend may still be in its asynchronous save
  // finalizer. Its owner releases the lock; a second job cannot take it over.
  if(alive(owner.owner_pid)||alive(owner.process_id))throw Error('Another interval decoder is running or finishing its audit. Wait for its status before starting another job.');
  const recovery=path.join(collection,'.prepare-recovery.lock');let recoveryLock;
  try{recoveryLock=await fs.open(recovery,'wx');}catch(error){if((error as NodeJS.ErrnoException).code==='EEXIST')throw Error('Another frontend is recovering decoder ownership. Retry shortly.');throw error;}
  try{
    if(await fs.readFile(lockPath,'utf8')!==previous)throw Error('Decoder ownership changed. Refresh before preparing another interval.');
    await fs.unlink(lockPath);return {lock:await fs.open(lockPath,'wx'),lockPath};
  }finally{await recoveryLock.close();await fs.unlink(recovery);}
}
/** A delayed old finalizer must never remove a later worker's lock. */
export async function releaseDecoderLock(lock:fs.FileHandle,lockPath:string,expectedRaw:string){
  await lock.close();
  try{const current=await fs.readFile(lockPath,'utf8');if(current===expectedRaw)await fs.unlink(lockPath);}
  catch(error){if((error as NodeJS.ErrnoException).code!=='ENOENT')throw error;}
}

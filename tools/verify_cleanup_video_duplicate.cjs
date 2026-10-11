'use strict';
// Verify duplicate bytes before any deletion. No decoder or model is involved.
const fs=require('node:fs'),fsp=fs.promises,path=require('node:path'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'..');
async function fingerprint(relative){
  const file=path.join(root,relative),before=await fsp.stat(file);
  if((await fsp.realpath(file)).toLowerCase()!==file.toLowerCase())throw Error('Aliased file excluded');
  const hash=crypto.createHash('sha256');
  for await(const bytes of fs.createReadStream(file,{highWaterMark:4*1024*1024}))hash.update(bytes);
  const after=await fsp.stat(file);
  if(before.size!==after.size||before.mtimeMs!==after.mtimeMs)throw Error('Source changed during hash verification');
  return {path:file,bytes:after.size,mtime_ms:after.mtimeMs,sha256:hash.digest('hex')};
}
async function main(){
  const destination=process.argv[2];if(!destination||fs.existsSync(destination))throw Error('Choose a new duplicate verification report');
  const [canonical,duplicate]=await Promise.all([fingerprint('python_tools/dataset/raw_videos/training/world_cup/game_1.mp4'),fingerprint('frontend/public/game_1.mp4')]);
  const matches=canonical.bytes===duplicate.bytes&&canonical.sha256===duplicate.sha256;
  await fsp.writeFile(destination,JSON.stringify({schema_version:1,created_at:new Date().toISOString(),canonical,duplicate,matches,
    deletion_performed:false,scope:'Exact byte comparison of unused frontend copy against retained raw original'},null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({matches,bytes:duplicate.bytes,sha256:canonical.sha256}));
  if(!matches)throw Error('Copies differ; neither is eligible for deletion');
}
main().catch(error=>{console.error(error.stack);process.exitCode=1;});

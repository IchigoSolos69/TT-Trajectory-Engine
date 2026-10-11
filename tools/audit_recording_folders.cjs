'use strict';
// Inventory files and reconnect identical moved originals; no decoding, labels,
// training partition changes or sealed-test evaluation occur in this command.
const fs=require('node:fs'),fsp=fs.promises,path=require('node:path'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'..'),base=path.join(root,'training_data','live');
const sha=file=>new Promise((resolve,reject)=>{const digest=crypto.createHash('sha256'),stream=fs.createReadStream(file);stream.on('data',bytes=>digest.update(bytes));stream.on('error',reject);stream.on('end',()=>resolve(digest.digest('hex')));});
async function main(){
  const output=process.argv[2];if(!output||fs.existsSync(output))throw Error('Usage: node tools/audit_recording_folders.cjs NEW_OUTPUT.json');
  const known=new Map();
  for(const clip of fs.readdirSync(base)){
    const file=path.join(base,clip,'capture.json');if(!fs.existsSync(file))continue;
    const value=JSON.parse(fs.readFileSync(file,'utf8'));
    if(/^[0-9a-f]{64}$/.test(value.original_sha256??'')){
      const list=known.get(value.original_sha256)??[];list.push({clip,session_id:value.session_id,original_path:value.original_path});known.set(value.original_sha256,list);
    }
  }
  const downloads=path.join(process.env.USERPROFILE,'Downloads'),rows=[],sources={};
  for(const [folder,requested_role] of [['training','train'],['validation','val'],['seal','test']]){
    const directory=path.join(downloads,folder),entries=await fsp.readdir(directory,{withFileTypes:true});
    for(const entry of entries.sort((a,b)=>a.name.localeCompare(b.name))){
      if(!entry.isFile()||!/^.+\.(mov|mp4|avi|mkv|webm|m4v)$/i.test(entry.name))continue;
      const file=path.join(directory,entry.name),before=await fsp.stat(file),digest=await sha(file),after=await fsp.stat(file);
      if(before.size!==after.size||before.mtimeMs!==after.mtimeMs)throw Error('File changed during hash audit: '+file);
      const imports=known.get(digest)??[],session_ids=[...new Set(imports.map(item=>item.session_id).filter(Boolean))];
      const knownDevelopment=session_ids.includes('iphone_same_venue_session_1');
      rows.push({path:file,sha256:digest,bytes:after.size,requested_role,matching_imports:imports,known_session_ids:session_ids,effective_role:knownDevelopment?'train':'unassigned',reason:knownDevelopment?'Previously reviewed same-session development footage; folder move does not establish independent validation/test':'Unreviewed source; recording/session/device provenance and labels still required'});
      if(imports.length)sources[digest]={paths:[...(sources[digest]?.paths??[]),file],reverified_at:new Date().toISOString()};
      console.log(folder+'/'+entry.name+': '+(imports.length?'known original matched':'new/unassigned'));
    }
  }
  const resolver=path.join(base,'source_locations.json');
  if(fs.existsSync(resolver))throw Error('Preserve existing source resolver; review a migration instead of overwriting it');
  fs.writeFileSync(resolver,JSON.stringify({schema_version:1,sources},null,2)+'\n',{flag:'wx'});
  const report={schema_version:1,created_at:new Date().toISOString(),scope:'Source-byte identity audit only; no decoding or sealed label/result access',rows,source_locations_path:resolver,source_locations_sha256:await sha(resolver),existing_originals_relocated:Object.keys(sources).length,known_session_split_conflicts:rows.filter(row=>row.effective_role==='train'&&row.requested_role!=='train').length,production_labels_modified:false,dataset_partitions_modified:false,ready_for_training:false};
  fs.writeFileSync(output,JSON.stringify(report,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({files:rows.length,reconnected_originals:report.existing_originals_relocated,same_session_split_conflicts:report.known_session_split_conflicts}));
}
main().catch(error=>{console.error(error.stack);process.exitCode=1;});

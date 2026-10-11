'use strict';
// Apply the user's agreed existing-session training role without editing labels.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'..'),base=path.join(root,'training_data','live');
const destination=path.join(base,'training_sessions.json');
const hash=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
const ts=require('../frontend/node_modules/typescript'),vm=require('node:vm');
const load=(file,deps={})=>{const scope={exports:{},require:name=>deps[name]};vm.runInNewContext(ts.transpileModule(fs.readFileSync(file,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS}}).outputText,scope);return scope.exports;};
const lib=path.join(root,'frontend','src','lib');
const geometry=load(path.join(lib,'reviewGeometry.ts'));
const review=load(path.join(lib,'reviewAnnotations.ts'),{'./reviewGeometry':geometry});
const sessions=load(path.join(lib,'trainingSessions.ts'),{'./reviewAnnotations':review});
const output=process.argv[2];
if(!output)throw Error('Usage: node tools/register_existing_iphone_session.cjs NEW_AUDIT_FILE.json');
if(fs.existsSync(destination)||fs.existsSync(output))throw Error('Existing registry/audit preserved; use GUI assignment for subsequent captures');
let registry=sessions.emptyTrainingRegistry();const now=new Date().toISOString(),preserved={},assigned=[],unassigned=[];
for(const clip of fs.readdirSync(base).sort()){
  if(!sessions.validCaptureName(clip))continue;
  const directory=path.join(base,clip),manifest=path.join(directory,'capture.json');
  if(!fs.existsSync(manifest))continue;
  const capture=JSON.parse(fs.readFileSync(manifest,'utf8'));
  if(capture.source!=='recorded_camera'||capture.session_id!=='iphone_same_venue_session_1'||!/^IMG_\d+\.MOV$/i.test(path.basename(capture.original_path??''))){unassigned.push(clip);continue;}
  for(const name of ['capture.json','annotations.json','frames.jsonl']){
    const file=path.join(directory,name);if(fs.existsSync(file))preserved[file]=hash(fs.readFileSync(file));
  }
  registry=sessions.assignTrainingCapture(registry,{clip,session_id:capture.session_id,stage:'existing',domain:'iphone'},capture.session_id,now);
  assigned.push(clip);
}
if(!assigned.length)throw Error('No matching original iPhone session found');
sessions.validateTrainingRegistry(registry);
const bytes=JSON.stringify(registry,null,2)+'\n';
fs.writeFileSync(destination,bytes,{flag:'wx'});
for(const [file,digest] of Object.entries(preserved))if(hash(fs.readFileSync(file))!==digest)throw Error('Historical data changed: '+file);
const audit={schema_version:1,scope:'User-agreed existing iPhone session stays in training; no labels or originals altered',registry_path:destination,registry_sha256:hash(bytes),assigned,unassigned,preserved_sha256:preserved,preserved_reverified:true,new_sessions_created:false};
fs.writeFileSync(output,JSON.stringify(audit,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({assigned:assigned.length,unassigned:unassigned.length,preserved_reverified:true}));

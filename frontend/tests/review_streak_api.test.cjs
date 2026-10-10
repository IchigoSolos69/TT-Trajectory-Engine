'use strict';
// Executes the production route with actual NextRequest/NextResponse and a
// disposable capture. No existing recording or annotation is edited.
const fs=require('node:fs'),fsp=require('node:fs/promises'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const ts=require('../node_modules/typescript'),next=require('../node_modules/next/server');
const load=(file,deps,extra={})=>{const scope={exports:{},require:name=>deps[name],...extra};vm.runInNewContext(ts.transpileModule(fs.readFileSync(file,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,esModuleInterop:true}}).outputText,scope);return scope.exports;};
const lib=path.join(__dirname,'../src/lib'),geometry=load(path.join(lib,'reviewGeometry.ts'),{}),origin=load(path.join(lib,'reviewOrigin.ts'),{}, {URL});
async function main(){
  const workspace=path.resolve(__dirname,'../..'),artifacts=path.join(workspace,'artifacts');await fsp.mkdir(artifacts,{recursive:true});
  const root=await fsp.mkdtemp(path.join(artifacts,'review-streak-fixture-')),clip='live_100_0',directory=path.join(root,clip);await fsp.mkdir(directory);
  const file='frame_1.jpg',jpg=Buffer.from([0xff,0xd8,0xff,0xc0,0,8,8,4,56,7,128,3,0xff,0xd9]);
  await fsp.writeFile(path.join(directory,file),jpg);
  await fsp.writeFile(path.join(directory,'frames.jsonl'),JSON.stringify({file,source_frame:7,elapsed_ms:100})+'\n');
  const prior={reviewed:true,ball:[[.4,.5,.01,.02]],person:[[.1,.2,.2,.3]],table:[[.2,.4,.5,.2]],geometry:{...geometry.emptyGeometry(),visibility:'visible',landmarks:{corner_far_left:[.2,.4]},landmarks_reviewed:true,events:['strike'],events_reviewed:true}};
  const raw=JSON.stringify({[file]:prior});await fsp.writeFile(path.join(directory,'annotations.json'),raw);
  const errors=[];
  const route=load(path.join(__dirname,'../src/app/api/training-review/route.ts'),{
    'next/server':next,'node:fs/promises':fsp,'node:path':path,'node:crypto':crypto,
    '../../../lib/reviewGeometry':geometry,'../../../lib/reviewOrigin':origin,
  },{process:{env:{TT_TRAINING_ROOT:root},cwd:()=>workspace},console:{error(...args){errors.push(args.map(e=>e?.stack??String(e)).join(' '));}},Error});
  let revision=crypto.createHash('sha256').update(raw).digest('hex');
  const request=async(labels,reviewed=true,originHeader='http://127.0.0.1:3100')=>{
    const req=new next.NextRequest('http://127.0.0.1:3100/api/training-review',{method:'POST',headers:{host:'127.0.0.1:3100',origin:originHeader,'content-type':'application/json'},body:JSON.stringify({clip,file,labels,scope:'ball',reviewed,revision})});
    const response=await route.POST(req),body=await response.json();if(response.ok)revision=body.revision;return {status:response.status,body};
  };
  try{
    const streak=geometry.createBlurStreak([.2,.3],[.3,.4],{width:1920,height:1080});
    const label={ball:prior.ball,person:[],table:[],geometry:{...geometry.emptyGeometry(),visibility:'visible',blur_streak:streak}};
    const saved=await request(label);assert.equal(saved.status,200,JSON.stringify(saved.body)+' '+errors.join('\n'));assert(saved.body.annotation.geometry.blur_streak.reviewed);
    for(const kind of ['ball','person','table'])assert.deepEqual(saved.body.annotation[kind],prior[kind],'Streak review preserves existing boxes');
    assert.deepEqual(saved.body.annotation.geometry.landmarks,prior.geometry.landmarks);assert.deepEqual(saved.body.annotation.geometry.events,prior.geometry.events);
    assert.equal(saved.body.annotation.source_frame,7);assert.equal(saved.body.annotation.elapsed_ms,100);
    assert.equal(await fsp.readFile(path.join(directory,'annotation_history',crypto.createHash('sha256').update(raw).digest('hex')+'.json'),'utf8'),raw,'Prior revision is immutable');
    const snapshot=await fsp.readFile(path.join(directory,'annotations.json'),'utf8');
    const bad=await request({...label,geometry:{...label.geometry,blur_streak:{...streak,source_width:960}}});assert.equal(bad.status,400);assert.match(bad.body.error,/dimensions changed/);
    assert.equal(await fsp.readFile(path.join(directory,'annotations.json'),'utf8'),snapshot);
    for(const visibility of ['unknown','occluded','absent'])assert.equal((await request({...label,ball:[],geometry:{...label.geometry,visibility}})).status,400);
    assert.equal((await request(label,true,'http://localhost:3100')).status,403,'Loopback alias remains a different origin');
    assert.equal((await request(label,true,'https://untrusted.example')).status,403);
    const streakOnly=await request({...label,ball:[]});assert.equal(streakOnly.status,200);assert.equal(streakOnly.body.annotation.ball.length,0);assert.equal(streakOnly.body.annotation.reviewed_classes.ball,true);
    const skipped=await request({...label,ball:[]},false);assert.equal(skipped.status,200);assert.equal(skipped.body.annotation.geometry.blur_streak.reviewed,false);assert.equal(skipped.body.annotation.reviewed_classes.ball,false);
    const cleared=await request({...label,geometry:{...label.geometry,blur_streak:null}});assert.equal(cleared.status,200);assert.equal(cleared.body.annotation.geometry.blur_streak,null);
    const read=await route.GET(new next.NextRequest('http://127.0.0.1:3100/api/training-review?clip='+clip,{headers:{host:'127.0.0.1:3100'}}));
    assert.equal((await read.json()).annotations[file].geometry.blur_streak,null,'Cleared state survives reload');
    await fsp.writeFile(path.join(artifacts,'review_streak_api_checks.json'),JSON.stringify({passed:true,scope:'Production route execution with actual NextRequest and isolated fixture, not browser rendering',checks:['explicit review only','streak-only save','existing boxes/scene geometry preserved','source JPEG dimensions','unknown/occluded exclusion','exact origin guard','immutable history','skip and clear persistence']},null,2)+'\n');
    console.log('PASS: production review API explicit streak save/reload, source dimensions, legacy boxes, secure origin and immutable revision checks.');
  }finally{
    if(path.dirname(root)!==artifacts||!path.basename(root).startsWith('review-streak-fixture-'))throw Error('Refusing fixture cleanup outside artifacts');
    await fsp.rm(root,{recursive:true,force:true});
  }
}
main().catch(error=>{console.error(error);process.exitCode=1;});

'use strict';
// Execute production page handlers with controlled hooks. Browser rendering is
// a separate acceptance check; this test exercises actual controls/payloads.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const React=require('../node_modules/react'),ts=require('../node_modules/typescript');
const load=(file,deps,extra={})=>{const scope={exports:{},require:name=>deps[name],...extra};vm.runInNewContext(ts.transpileModule(fs.readFileSync(file,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,jsx:ts.JsxEmit.ReactJSX,esModuleInterop:true}}).outputText,scope);return scope.exports;};
const lib=path.join(__dirname,'../src/lib'),geometry=load(path.join(lib,'reviewGeometry.ts'),{}),review=load(path.join(lib,'reviewAnnotations.ts'),{'./reviewGeometry':geometry});
let state=[],cursor=0,effects=[],posts=[];
const hooks={...React,useState(initial){const i=cursor++;if(!(i in state))state[i]=initial;return[state[i],value=>state[i]=typeof value==='function'?value(state[i]):value];},useEffect(effect){effects.push(effect);}};
const annotation={reviewed:true,reviewed_classes:{ball:true,person:true,table:true},ball:[[.4,.5,.01,.02]],person:[[.1,.2,.2,.3]],table:[[.2,.4,.5,.2]],geometry:{...geometry.emptyGeometry(),visibility:'visible'}};
function ReviewFrameImage(){}
const api=async(url,options)=>({ok:true,json:async()=>{
  if(options?.method==='POST'){const body=JSON.parse(options.body);posts.push(body);return {revision:'next',annotation:{...body.labels,reviewed:body.reviewed,geometry:{...body.labels.geometry,blur_streak:body.labels.geometry.blur_streak?{...body.labels.geometry.blur_streak,reviewed:body.reviewed}:null}}};}
  if(url.includes('?clip='))return {frames:[{file:'frame_1.jpg',source_frame:1}],annotations:{'frame_1.jpg':annotation},revision:'original'};
  return {clips:['live_100_0']};
}});
const Page=load(path.join(__dirname,'../src/app/training-review/page.tsx'),{
 'react':hooks,'react/jsx-runtime':require('../node_modules/react/jsx-runtime'),'../../lib/reviewGeometry':geometry,
 '../../lib/reviewAnnotations':review,'../../components/ReviewFrameImage':{ReviewFrameImage},
 '../../components/TrainingSessionReadiness':{TrainingSessionReadiness(){return null;}},
 '../../components/BurstContinuityReview':{BurstContinuityReview(){return null;}},'../../components/FullAbsenceIntervalReview':{FullAbsenceIntervalReview(){return null;}},
},{fetch:api,window:{addEventListener(){},removeEventListener(){}},localStorage:{getItem(){return null;},setItem(){}}}).default;
const walk=node=>!node||typeof node!=='object'?[]:[node,...React.Children.toArray(node.props?.children).flatMap(walk)];
const render=()=>{cursor=0;effects=[];return walk(Page());};
const button=text=>render().find(node=>node.type==='button'&&node.props.children===text);
const image=()=>render().find(node=>node.type===ReviewFrameImage);
const flush=()=>new Promise(resolve=>setImmediate(resolve));
async function main(){
 render();for(const effect of effects)effect();await flush();
 await render().find(node=>node.type==='select'&&node.props['aria-label']==='Capture clip').props.onChange({target:{value:'live_100_0'}});await flush();
 const before=JSON.stringify(annotation);button('Blur streak · drag endpoints').props.onClick();
 image().props.onAnnotation([.2,.3],[.3,.4],{width:1920,height:1080});
 assert.equal(JSON.stringify(annotation),before,'Drawing edits drafts, not saved labels');
 assert.equal(button('Save this frame').props.disabled,false);
 assert(render().some(node=>node.type==='line'&&node.props.x1===20&&node.props.x2===30),'The endpoints render in source-normalized coordinates');
 await button('Save this frame').props.onClick();await flush();
 const posted=posts.at(-1);assert.equal(posted.reviewed,true);assert.equal(posted.scope,'ball');
 for(const kind of ['ball','person','table'])assert.equal(JSON.stringify(posted.labels[kind]),JSON.stringify(annotation[kind]),'Streak control does not replace boxes');
 assert.equal(posted.labels.geometry.blur_streak.source_width,1920);assert.equal(posted.labels.geometry.blur_streak.source_height,1080);
 assert.equal(posted.labels.geometry.blur_streak.reviewed,false,'Draft review is confirmed by the save operation, not by drawing');
 button('Clear streak').props.onClick();await button('Save this frame').props.onClick();await flush();assert.equal(posts.at(-1).labels.geometry.blur_streak,null);
 button('Blur streak · drag endpoints').props.onClick();image().props.onAnnotation([.5,.5],[.5,.5],{width:1920,height:1080});
 assert.equal(image().props.children.flat().filter(Boolean).length>0,true);
 await button('Skip / keep unreviewed').props.onClick();await flush();assert.equal(posts.at(-1).reviewed,false);assert.equal(posts.at(-1).labels.geometry.blur_streak.axis_rad,null);
 fs.writeFileSync(path.resolve(__dirname,'../../artifacts/review_streak_ui_checks.json'),JSON.stringify({passed:true,scope:'Production page handler execution, not browser rendering',checks:['optional mode','normalized endpoint overlay','draft-only drawing','explicit save payload','source dimensions','existing box preservation','clear','point axis unresolved','skip unreviewed']},null,2)+'\n');
 console.log('PASS: production training review controls create only optional streak drafts, preserve boxes and submit explicit review/clear/skip actions.');
}
main().catch(error=>{console.error(error);process.exitCode=1;});

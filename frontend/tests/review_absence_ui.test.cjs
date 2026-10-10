'use strict';
// Production page handlers with controlled hooks; not a browser paint test.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const React=require('../node_modules/react'),ts=require('../node_modules/typescript');
const load=(file,deps,extra={})=>{const scope={exports:{},require:name=>deps[name],...extra};vm.runInNewContext(ts.transpileModule(fs.readFileSync(file,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,jsx:ts.JsxEmit.ReactJSX,esModuleInterop:true}}).outputText,scope);return scope.exports;};
const lib=path.join(__dirname,'../src/lib'),geometry=load(path.join(lib,'reviewGeometry.ts'),{}),review=load(path.join(lib,'reviewAnnotations.ts'),{'./reviewGeometry':geometry});
let state=[],cursor=0,effects=[],posts=[];
const hooks={...React,useState(initial){const i=cursor++;if(!(i in state))state[i]=initial;return[state[i],value=>state[i]=typeof value==='function'?value(state[i]):value];},useEffect(effect){effects.push(effect);}};
const frames=[1,2,5,6].map((id,i)=>({file:`frame_${i+1}.jpg`,source_frame:id}));
const annotations=Object.fromEntries(frames.map(frame=>[frame.file,{...review.emptyLabels(),reviewed:false}]));
const api=async(url,options)=>({ok:true,json:async()=>{
 if(options?.method==='POST'){
  const body=JSON.parse(options.body);posts.push(body);
  if(body.action==='preview_ball_absence_interval')return {preview_token:'source-bound-token',affected_frames:4,frames,conflicts:[],session_id:'same_session'};
  if(body.action==='confirm_ball_absence_interval')return {revision:'saved',affected_frames:4,annotations:Object.fromEntries(frames.map(frame=>[frame.file,{...review.emptyLabels(),reviewed:true,reviewed_classes:{ball:true,person:false,table:false},geometry:{...geometry.emptyGeometry(),visibility:'absent'}}]))};
 }
 if(url.includes('?clip='))return {frames,annotations,revision:'original',capture:{source:'recorded_camera',full_source_decode_counted:true,source_frames:6,session_id:'same_session'}};
 return {clips:['phone_1234567890abcdef_1']};
}});
function ReviewFrameImage(){}
const Page=load(path.join(__dirname,'../src/app/training-review/page.tsx'),{'react':hooks,'react/jsx-runtime':require('../node_modules/react/jsx-runtime'),'../../lib/reviewGeometry':geometry,'../../lib/reviewAnnotations':review,'../../components/ReviewFrameImage':{ReviewFrameImage}},{fetch:api,window:{addEventListener(){},removeEventListener(){}},localStorage:{getItem(){return null;},setItem(){}}}).default;
const walk=node=>!node||typeof node!=='object'?[]:[node,...React.Children.toArray(node.props?.children).flatMap(walk)];
const render=()=>{cursor=0;effects=[];return walk(Page());};
const button=text=>render().find(node=>node.type==='button'&&node.props.children===text);
const flush=()=>new Promise(resolve=>setImmediate(resolve));
async function main(){
 render();for(const effect of effects)effect();await flush();
 const loadClip=async()=>{await render().find(node=>node.type==='select'&&node.props['aria-label']==='Capture clip').props.onChange({target:{value:'phone_1234567890abcdef_1'}});await flush();};
 await loadClip();assert.equal(button('Preview no-ball interval').props.disabled,false);
 await button('Preview no-ball interval').props.onClick();await flush();
 assert.equal(posts[0].action,'preview_ball_absence_interval');assert.equal(posts[0].start_source_frame,1);assert.equal(posts[0].end_source_frame,6);assert(!posts[0].confirm_absence);
 assert.equal(button('Confirm ball absence for selected images').props.disabled,true,'Preview never implies confirmation');
 assert.equal(render().filter(node=>node.type==='img').length,4,'Small selection previews every saved image');
 render().find(node=>node.type==='input'&&node.props['aria-label']==='Confirm no ball in selected images').props.onChange({target:{checked:true}});
 assert.equal(button('Confirm ball absence for selected images').props.disabled,false);
 await button('Confirm ball absence for selected images').props.onClick();await flush();
 const confirm=posts.at(-1);assert.equal(confirm.action,'confirm_ball_absence_interval');assert.equal(confirm.preview_token,'source-bound-token');assert.equal(confirm.confirm_absence,true);assert.equal(confirm.revision,'original');
 assert(!button('Confirm ball absence for selected images'),'Confirmed preview is discarded');
 assert(render().some(node=>node.type==='span'&&React.Children.toArray(node.props.children).join('').includes('BALL REVIEWED')));
 // A newly drawn draft must block bulk requests without clearing that draft.
 render().find(node=>node.type===ReviewFrameImage).props.onAnnotation([.4,.4],[.42,.42],{width:1920,height:1080});
 assert.equal(button('Preview no-ball interval').props.disabled,true);
 const before=posts.length;await button('Preview no-ball interval').props.onClick();assert.equal(posts.length,before);
 assert(render().some(node=>node.type==='p'&&String(node.props.children).includes('pending drafts')));
 fs.writeFileSync(path.resolve(__dirname,'../../artifacts/review_absence_ui_checks.json'),JSON.stringify({passed:true,scope:'Production page handlers with controlled hooks; browser paint remains separate',checks:['source range defaults','read-only preview request','sample thumbnails','explicit unchecked confirmation','token/revision confirmation payload','post-save reviewed state','draft conflict prevents request without discarding edits']},null,2)+'\n');
 console.log('PASS: no-ball interval UI preview, explicit confirmation, scoped payload, saved review and draft protection.');
}
main().catch(error=>{console.error(error);process.exitCode=1;});

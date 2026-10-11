// Execute the production components with explicit telemetry and controlled hooks.
// This never creates a WebSocket or opens/records a real camera.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const React=require('../node_modules/react'),ReactDOM=require('../node_modules/react-dom/server');
const ts=require('../node_modules/typescript');
let state=[],cursor=0;
const hooks={...React,useState(initial){const id=cursor++;if(!(id in state))state[id]=initial;return[state[id],next=>state[id]=typeof next==='function'?next(state[id]):next];}};
function moduleAt(file,extra={}){
 const scope={exports:{},require:name=>name==='react'?hooks:name==='react/jsx-runtime'?require('../node_modules/react/jsx-runtime'):extra[name]};
 vm.runInNewContext(ts.transpileModule(fs.readFileSync(path.join(__dirname,'../src/components',file),'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,jsx:ts.JsxEmit.ReactJSX}}).outputText,scope);
 return scope.exports;
}
const guide=moduleAt('TrainingRecordingGuide.tsx');
const sessions=moduleAt('TrainingSessionReadiness.tsx');
const controls=moduleAt('TrainingCaptureControls.tsx',{'./TrainingRecordingGuide':guide,'./TrainingSessionReadiness':sessions});
const walk=node=>!node||typeof node!=='object'?[]:[node,...React.Children.toArray(node.props?.children).flatMap(walk)];
const commands=[];
const frame={mode:'live',camera_active:false,camera_index:0,source_id:1,training_recording:{active:false,failed:false,saved_frames:0,dropped_frames:0,clip:''}};
const tree=()=>controls.TrainingCaptureControls({frame,sendCommand:command=>commands.push(command)});
const buttons=()=>walk(tree()).filter(node=>node.type==='button');
assert(buttons().every(b=>b.props.disabled),'All capture modes must be disabled without camera authorization');
frame.camera_active=true;
buttons().find(b=>b.props.children==='Record training clip').props.onClick();
buttons().find(b=>b.props.children==='Record temporal bursts').props.onClick();
assert(buttons().find(b=>b.props.children==='Capture difficult examples').props.disabled);
frame.intelligence={};
buttons().find(b=>b.props.children==='Capture difficult examples').props.onClick();
frame.training_recording.active=true;
assert.equal(buttons().length,1,'An active capture exposes stop without competing start modes');
buttons()[0].props.onClick();
assert.deepEqual(commands,['START_TRAINING_RECORDING','START_TEMPORAL_RECORDING','START_ACTIVE_LEARNING','STOP_TRAINING_RECORDING']);
frame.training_recording.dropped_frames=3;frame.training_recording.failed=true;
cursor=0;
const html=ReactDOM.renderToStaticMarkup(React.createElement(controls.TrainingCaptureControls,{frame,sendCommand(){}}));
assert.match(html,/Some images were not saved/);assert.match(html,/Recording failed/);
assert.match(html,/Camera settings need your check/);assert.match(html,/Sampled images, not continuous match video/);
assert.match(html,/ball hidden behind a player is occluded/);assert.match(html,/validation and test sessions/);
assert.match(html,/After recording/);assert.match(html,/Nothing is uploaded/);
assert.match(html,/Session C is reserved for final acceptance/);assert.match(html,/three separate gameplay sessions/);assert.match(html,/Brightness is not a cure/);
state=[];cursor=0;
let checks=walk(guide.TrainingRecordingGuide()).filter(node=>node.type==='input');
assert.equal(checks.length,4);assert(checks.every(c=>!c.props.checked));
checks[0].props.onChange({target:{checked:true}});cursor=0;
checks=walk(guide.TrainingRecordingGuide()).filter(node=>node.type==='input');
assert.equal(checks[0].props.checked,true);assert.equal(checks[1].props.checked,false);
checks[0].props.onChange({target:{checked:false}});cursor=0;
assert(walk(guide.TrainingRecordingGuide()).filter(node=>node.type==='input').every(c=>!c.props.checked));
fs.writeFileSync('artifacts/training_capture_guide_checks.json',JSON.stringify({passed:true,real_camera_opened:false,scope:'Production component execution and server rendering; not a desktop browser visual test.',commands,checks:['camera authorization gating','mode command dispatch','active stop control','capture failure/drop warnings','mode limits and labeling/split guidance','optional checklist toggle']},null,2));
console.log('PASS: training guide component rendering, optional checklist, capture command wiring and warnings; no real camera opened.');

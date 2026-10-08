// Execute the production annotation component with controlled hooks and image
// rectangles. Tests source-coordinate integrity; not a browser layout test.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const React=require('../node_modules/react'),ts=require('../node_modules/typescript');
let state=[],cursor=0;
const hooks={...React,
 useState(initial){const id=cursor++;if(!(id in state))state[id]=initial;return[state[id],value=>state[id]=typeof value==='function'?value(state[id]):value];},
 useRef(initial){const id=cursor++;if(!(id in state))state[id]={current:initial};return state[id];}
};
const scope={exports:{},require:name=>name==='react'?hooks:require('../node_modules/'+name)};
vm.runInNewContext(ts.transpileModule(fs.readFileSync(path.join(__dirname,'../src/components/ReviewFrameImage.tsx'),'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,jsx:ts.JsxEmit.ReactJSX}}).outputText,scope);
const walk=node=>!node||typeof node!=='object'?[]:[node,...React.Children.toArray(node.props?.children).flatMap(walk)];
const annotations=[],props={src:'frame_1.jpg',enabled:true,children:React.createElement('rect',{x:50,y:50,width:1,height:1}),onAnnotation:(...value)=>annotations.push(value)};
const tree=()=>{cursor=0;return scope.exports.ReviewFrameImage(props);};
const nodes=()=>walk(tree());
const image=()=>nodes().find(n=>n.type==='img');
const wrapper=()=>nodes().find(n=>n.props?.onPointerDown);
const button=name=>nodes().find(n=>n.type==='button'&&n.props.children===name);
const load=()=>image().props.onLoad({currentTarget:{naturalWidth:1920,naturalHeight:1080}});
const event=(rect,x,y)=>({pointerId:1,clientX:rect.left+x*rect.width,clientY:rect.top+y*rect.height,currentTarget:{getBoundingClientRect:()=>rect,setPointerCapture(){}}});
const drag=(rect)=>{const node=wrapper();node.props.onPointerDown(event(rect,.56,.53));node.props.onPointerUp(event(rect,.575,.54));};
drag({left:0,top:0,width:960,height:540});assert.equal(annotations.length,0,'Cannot annotate an unloaded image');
assert(button('100%').props.disabled);
load();
const cases=[{label:'Fit',rect:{left:20,top:80,width:960,height:540}},
 {label:'100%',rect:{left:-300,top:-250,width:1920,height:1080},width:1920},
 {label:'200%',rect:{left:-1800,top:-700,width:3840,height:2160},width:3840}];
for(const test of cases){
 button(test.label).props.onClick();drag(test.rect);
 const [a,b,size]=annotations.at(-1);
 for(const [actual,expected] of [[a[0],.56],[a[1],.53],[b[0],.575],[b[1],.54]])assert(Math.abs(actual-expected)<1e-12);
 assert.equal(size.width,1920);assert.equal(size.height,1080);
 if(test.width)assert.equal(image().props.style.width,test.width);
 assert.equal(nodes().find(n=>n.type==='svg').props.viewBox,'0 0 100 100');
}
const before=annotations.length,oldImage=image(),rect=cases[2].rect;
wrapper().props.onPointerDown(event(rect,.56,.53));props.src='frame_2.jpg';wrapper().props.onPointerUp(event(rect,.575,.54));
assert.equal(annotations.length,before,'Source switches invalidate in-flight annotations');
load();oldImage.props.onError();assert(!button('100%').props.disabled,'Late failures from an old image cannot disable the new image');
wrapper().props.onPointerDown(event(rect,.56,.53));button('Fit').props.onClick();wrapper().props.onPointerUp(event(rect,.575,.54));
assert.equal(annotations.length,before,'Zoom switches invalidate in-flight annotations');
props.enabled=false;drag(cases[0].rect);assert.equal(annotations.length,before);
props.enabled=true;image().props.onError();drag(cases[0].rect);assert.equal(annotations.length,before,'Failed images cannot be labeled');
assert(nodes().some(n=>n.props?.role==='alert'));
load();const current=wrapper();current.props.onPointerDown(event(cases[0].rect,.56,.53));current.props.onPointerCancel();current.props.onPointerUp(event(cases[0].rect,.575,.54));assert.equal(annotations.length,before);
fs.writeFileSync('artifacts/review_frame_image_checks.json',JSON.stringify({passed:true,scope:'Production component execution; source-coordinate and stale-image checks, not browser layout/render evidence',checks:['Fit/native/2x scrolled coordinate equivalence','native dimension propagation','overlay coordinate basis','unloaded/failed/busy image guards','source/zoom/cancel invalidation','late old-image failure isolation']},null,2)+'\n');
console.log('PASS: annotation coordinates match at Fit/100%/200%, including scrolled images and stale-source guards.');

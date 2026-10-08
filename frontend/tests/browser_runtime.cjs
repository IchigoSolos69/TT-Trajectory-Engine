// Isolated Chrome DevTools verification; never attaches to the user's browser profile.
const fs=require('node:fs');
const assert=require('node:assert/strict');
async function main(){
  const tabs=await (await fetch('http://localhost:9227/json')).json();
  const target=tabs.find(t=>t.type==='page');
  if(!target) throw new Error('No isolated test page');
  const ws=new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j;});
  let id=0;const calls=new Map(),events=[];
  ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const p=calls.get(m.id);calls.delete(m.id);m.error?p.reject(new Error(JSON.stringify(m.error))):p.resolve(m.result);}else if(['Runtime.exceptionThrown','Log.entryAdded','Network.webSocketFrameReceived'].includes(m.method)){
    if(m.method==='Network.webSocketFrameReceived'){
      try{const f=JSON.parse(m.params.response.payloadData);events.push({method:m.method,frame_id:f.frame_id,mode:f.mode,image_bytes:f.preview_jpeg?.length,scene_count:f.scene_detections?.length,table_corners:f.table_corners?.length,playback_rate:f.playback_rate,paused:f.paused});}catch{}
    }else events.push(m);
  }};
  const call=(method,params={})=>new Promise((resolve,reject)=>{calls.set(++id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
  const evaluate=async expression=>{const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
  const wait=async expression=>{for(let i=0;i<100;i++){if(await evaluate(expression))return;await new Promise(r=>setTimeout(r,100));}throw new Error('UI condition timed out: '+expression);};
  await call('Page.enable');await call('Runtime.enable');await call('Log.enable');await call('Network.enable');
  await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
  await call('Page.navigate',{url:'http://localhost:3000'});
  await wait("document.body.innerText.includes('PHYSICS')");
  await wait("[...document.querySelectorAll('button')].some(b=>Object.keys(b).some(k=>k.startsWith('__reactProps')))");
  const click=async text=>evaluate(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.innerText.trim()===${JSON.stringify(text)});if(!b)throw Error('Missing button');b.click();return true;})()`);
  const screenshot=async name=>{await new Promise(r=>setTimeout(r,700));const r=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});fs.writeFileSync(`artifacts/${name}.png`,Buffer.from(r.data,'base64'));};
  await click('PHYSICS');await wait("document.body.innerText.includes('TABLE BOUNCES')");
  const physicsText=await evaluate('document.body.innerText');assert.match(physicsText,/2 TABLE BOUNCES/);
  await screenshot('physics_runtime');
  await click('3D VOLUMETRIC');await wait("document.body.innerText.includes('3D TABLE VIEW')");await screenshot('3d_runtime');
  await click('Top');await screenshot('3d_top_runtime');
  await click('2D RADAR');await screenshot('radar_runtime');
  await click('VISUALIZER');
  if(process.argv.includes('--stream')){
    await wait("!!document.querySelector('canvas[data-frame-id]')");
    await click('Restart');
    await wait("Number(document.querySelector('canvas').dataset.frameId)<300");
    const before=await evaluate("Number(document.querySelector('canvas').dataset.frameId)");
    await screenshot('footage_runtime');
    await click('Pause');await wait("[...document.querySelectorAll('button')].some(b=>b.innerText.trim()==='Play')");
    await new Promise(r=>setTimeout(r,250));
    const paused=await evaluate("Number(document.querySelector('canvas').dataset.frameId)");
    await new Promise(r=>setTimeout(r,500));
    assert.equal(await evaluate("Number(document.querySelector('canvas').dataset.frameId)"),paused,'paused frame stays fixed');
    await evaluate("(()=>{const s=document.querySelector('select[aria-label=\"Playback speed\"]');s.value='0.5';s.dispatchEvent(new Event('change',{bubbles:true}));})()");
    await wait("document.querySelector('select').value==='0.5'");
    await click('Play');await wait(`Number(document.querySelector('canvas').dataset.frameId)>${paused}`);
    assert(events.some(e=>e.image_bytes>1000&&e.scene_count>0),'real imagery and scene detections arrived');
    assert(events.some(e=>e.playback_rate===.5),'backend acknowledged playback rate');
    events.push({check:'canvas_progress_and_pause',before,paused});
  }
  const exceptions=events.filter(e=>e.method==='Runtime.exceptionThrown');
  fs.writeFileSync('artifacts/browser_runtime.json',JSON.stringify({passed:exceptions.length===0,events},null,2));
  assert.equal(exceptions.length,0,JSON.stringify(exceptions));
  console.log('PASS: browser views, second bounce display, screenshots, no uncaught JS exceptions'+(process.argv.includes('--stream')?', synchronized footage and playback controls':''));ws.close();
}
main().catch(e=>{console.error(e);process.exit(1);});

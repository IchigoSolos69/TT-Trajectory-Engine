// UI command wiring against an explicit fixture in the isolated test browser.
// The fixture replaces WebSocket before page code, so it cannot open a camera.
const fs=require('node:fs');
const assert=require('node:assert/strict');
async function main(){
  const target=await(await fetch('http://localhost:9227/json/new?about:blank',{method:'PUT'})).json();
  if(!target)throw new Error('No isolated test tab');
  const ws=new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject;});
  let id=0;const pending=new Map(),exceptions=[];
  ws.onmessage=e=>{const message=JSON.parse(e.data);if(message.id){const p=pending.get(message.id);pending.delete(message.id);if(p)message.error?p.reject(new Error(JSON.stringify(message.error))):p.resolve(message.result);}else if(message.method==='Runtime.exceptionThrown')exceptions.push(message.params);};
  const call=(method,params={})=>new Promise((resolve,reject)=>{pending.set(++id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
  await call('Page.enable');await call('Runtime.enable');
  const fixture=await call('Page.addScriptToEvaluateOnNewDocument',{source:`
    window.trainingTestCommands=[];
    const NativeWebSocket=window.WebSocket;
    window.WebSocket=class {
      static OPEN=1;static CONNECTING=0;static CLOSING=2;static CLOSED=3;readyState=1;
      frame={mode:'live',frame_id:0,ball:null,players:[],table_corners:[],camera_active:false,preview_jpeg:'',fps:0,rally:{},training_recording:{active:false,failed:false,saved_frames:0,dropped_frames:0,clip:''}};
      constructor(url,protocols){if(new URL(url).port!=='9002')return new NativeWebSocket(url,protocols);setTimeout(()=>{if(this.readyState===1){this.onopen?.();this.emit()}},100)}
      emit(){this.onmessage?.({data:JSON.stringify(this.frame)})}
      send(command){window.trainingTestCommands.push(command);if(command==='START_CAMERA')this.frame.camera_active=true;
        if(command==='START_TRAINING_RECORDING')this.frame.training_recording.active=true;
        if(command==='STOP_TRAINING_RECORDING')this.frame.training_recording.active=false;
        this.emit()}
      close(){this.readyState=3}
    };
  `});
  await call('Page.navigate',{url:'http://localhost:3000'});
  let result;
  for(let i=0;i<150;i++){
    const response=await call('Runtime.evaluate',{expression:`(()=>{const panel=document.querySelector('[aria-label="Training capture controls"]');if(!panel)return null;const button=panel.querySelector('button');return {text:panel.innerText,button:button.innerText,disabled:button.disabled}})()`,returnByValue:true});
    result=response.result.value;if(result)break;
    await new Promise(r=>setTimeout(r,100));
  }
  if(!result){
    const dom=await call('Runtime.evaluate',{expression:'document.body.innerText',returnByValue:true});
    fs.writeFileSync('artifacts/browser_training_failure.json',JSON.stringify({body:dom.result.value,exceptions},null,2));
    await call('Page.removeScriptToEvaluateOnNewDocument',{identifier:fixture.identifier});
    ws.close();throw new Error('Live controls missing: '+JSON.stringify({body:dom.result.value,exceptions:exceptions.map(e=>e.exceptionDetails.exception?.description)}));
  }
  assert.match(result.button,/Record training clip|Stop recording/);
  assert.equal(result.disabled,true,'Recording is disabled before camera authorization');
  assert.match(result.text,/Nothing is uploaded/);
  const click=async label=>{
    await call('Runtime.evaluate',{expression:`[...document.querySelectorAll('button')].find(b=>b.innerText.trim()===${JSON.stringify(label)}).click()`});
    await new Promise(r=>setTimeout(r,200));
  };
  await click('AUTHORIZE WEBCAM');await click('Record training clip');
  const active=await call('Runtime.evaluate',{expression:`document.querySelector('[aria-label="Training capture controls"] button').innerText`,returnByValue:true});
  assert.equal(active.result.value,'Stop recording');await click('Stop recording');
  const commands=await call('Runtime.evaluate',{expression:'window.trainingTestCommands',returnByValue:true});
  assert.deepEqual(commands.result.value,['START_CAMERA','START_TRAINING_RECORDING','STOP_TRAINING_RECORDING']);
  assert.equal(exceptions.length,0,JSON.stringify(exceptions));
  fs.writeFileSync('artifacts/browser_training_controls.json',JSON.stringify({passed:true,fixture:true,real_camera_opened:false,commands:commands.result.value,...result,exceptions},null,2));
  console.log('PASS: live UI fixture, recording disabled until camera authorization, start/stop command wiring, no uncaught JS exceptions. No real camera opened.');
  await call('Page.removeScriptToEvaluateOnNewDocument',{identifier:fixture.identifier});
  // This is the dedicated verification browser, never a user's browsing profile.
  await call('Browser.close');
}
main().catch(e=>{console.error(e);process.exitCode=1;});

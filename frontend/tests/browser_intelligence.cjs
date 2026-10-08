const fs=require('node:fs'),assert=require('node:assert/strict');
async function main(){
 const tab=await(await fetch('http://localhost:9227/json/new?about:blank',{method:'PUT'})).json();
 const ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j;});
 let id=0;const calls=new Map(),errors=[];
 ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const p=calls.get(m.id);calls.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params);};
 const call=(method,params={})=>new Promise((resolve,reject)=>{calls.set(++id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
 const evaluate=async expression=>{const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
 const wait=async expression=>{for(let i=0;i<150;i++){if(await evaluate('Boolean('+expression+')'))return;await new Promise(r=>setTimeout(r,100));}throw Error('Timed out '+expression);};
 const click=async text=>evaluate('(()=>{const b=[...document.querySelectorAll("button")].find(b=>b.textContent.trim()==='+JSON.stringify(text)+');if(!b)throw Error("button missing");b.click();})()');
 await call('Page.enable');await call('Runtime.enable');await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1200,deviceScaleFactor:1,mobile:false});
 // An actual recorded candidate payload, explicitly a UI fixture. No camera or
 // production WebSocket is controlled by this browser test.
 const sample=JSON.parse(fs.readFileSync('artifacts/intelligence_ui_sample.json','utf8'));
 await call('Page.addScriptToEvaluateOnNewDocument',{source:'window.WebSocket=class {static OPEN=1;readyState=1;constructor(){setTimeout(()=>{this.onopen?.();this.onmessage?.({data:'+JSON.stringify(JSON.stringify(sample))+'});},200)}send(){}close(){this.readyState=3}};'});
 await call('Page.navigate',{url:'http://localhost:3100'});
 await wait('document.querySelector("[aria-label=\'Measurement provenance\']") && document.querySelector("canvas[data-frame-id]")');
 const body=await evaluate('document.body.innerText');assert(body.includes('coverage has not been validated'));assert(body.includes('Rally event timeline'));
 await click('2D RADAR');await wait('document.querySelector("[aria-label=\'2D table plane projection\']") && document.querySelector("canvas[data-frame-id]")');
 let alignment=await evaluate('({view:document.querySelector("[aria-label=\'2D table plane projection\']").dataset.frameId,image:document.querySelector("canvas[data-frame-id]").dataset.frameId})');assert.equal(alignment.view,alignment.image);
 await click('3D VOLUMETRIC');await wait('document.querySelector("[aria-label=\'3D table plane projection\']")');await new Promise(r=>setTimeout(r,500));
 let shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});fs.writeFileSync('artifacts/intelligence_3d_ui.png',Buffer.from(shot.data,'base64'));
 await call('Page.navigate',{url:'http://localhost:3100/training-review'});
 await wait('document.querySelector("select")?.options.length===2');
 await evaluate('(()=>{const s=document.querySelector("select");s.value="live_0_0";s.dispatchEvent(new Event("change",{bubbles:true}));})()');
 await wait('document.querySelector("img")?.naturalWidth>0');
 const rect=await evaluate('(()=>{const r=document.querySelector("img").getBoundingClientRect();return {x:r.left+r.width*.4,y:r.top+r.height*.4}})()');
 await call('Input.dispatchMouseEvent',{type:'mousePressed',button:'left',clickCount:1,...rect});
 await call('Input.dispatchMouseEvent',{type:'mouseReleased',button:'left',clickCount:1,...rect});
 await wait('document.body.innerText.includes("1 frame(s) have unsaved edits")');
 const boxCount=await evaluate('document.querySelectorAll("svg rect").length');
 await click('Next');await wait('document.body.innerText.includes("source frame 2")');
 await click('Previous');await wait('document.body.innerText.includes("source frame 1")');
 assert.equal(await evaluate('document.querySelectorAll("svg rect").length'),boxCount,'frame navigation must retain annotation drafts');
 await click('Save reviewed labels');await wait('document.body.innerText.includes("Reviewed labels saved locally.")');
 shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});fs.writeFileSync('artifacts/training_review_ui.png',Buffer.from(shot.data,'base64'));
 assert.equal(errors.length,0,JSON.stringify(errors));
 fs.writeFileSync('artifacts/browser_intelligence.json',JSON.stringify({passed:true,fixture:'recorded native candidate + isolated review fixtures',alignment,review_click_save:true,drafts_survive_navigation:true,errors},null,2));
 await call('Page.close');ws.close();console.log('PASS: candidate provenance, covariance views, source-frame agreement, review click-to-label and local save; no JS exceptions.');
}
main().catch(e=>{console.error(e);process.exit(1);});

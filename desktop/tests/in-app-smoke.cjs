'use strict';
const fs=require('node:fs');
const path=require('node:path');
const assert=require('node:assert/strict');
const {enginePacket,freePort}=require('../services.cjs');
const wait=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function eventually(check,timeout=60000){const start=Date.now();let error;while(Date.now()-start<timeout){try{return await check();}catch(e){error=e;}await wait(200);}throw error||Error('Test timeout');}
exports.run=async({app,window,engine,startSession,endSession,openRecording,dialog,dataRoot,getOrigin,getContext})=>{
  const report={started_at:new Date().toISOString(),packaged:app.isPackaged,electron:process.versions.electron,checks:[],renderer_errors:[]};
  const output=process.env.TT_DESKTOP_SMOKE_REPORT;
    window.webContents.on('console-message',event=>{if(event.level==='error')report.renderer_errors.push(event.message);});
  window.webContents.on('render-process-gone',(_event,details)=>report.renderer_errors.push(details.reason));
  try{
    assert.ok(output&&process.env.TT_DESKTOP_SMOKE_VIDEO,'Explicit smoke fixture and report required');
    const security=await window.webContents.executeJavaScript(`({node:typeof require,bridge:typeof window.ttDesktop?.startLive,isolated:typeof process,launcher:!!document.getElementById('live')})`);
    assert.deepEqual(security,{node:'undefined',bridge:'function',isolated:'undefined',launcher:true});report.security=security;
    report.checks.push('Sandboxed native window with isolated, typed preload bridge');
    // Exercise the actual launcher button -> preload -> IPC -> C++ startup.
    await window.webContents.executeJavaScript(`document.getElementById('live').click()`);
    await eventually(async()=>{assert.equal(getContext().phase,'ready');assert.equal(getContext().mode,'live');});
    const first=await enginePacket(engine.url,null,engine.child.pid);
    assert.equal(first.mode,'live');assert.equal(first.camera_active,false);
    report.live={pid:engine.child.pid,url:engine.url,camera_active:first.camera_active};
    await eventually(async()=>{const html=await window.webContents.executeJavaScript('document.body.innerText');assert.ok(html.includes('AUTHORIZE WEBCAM'));assert.ok(html.includes('LOCAL APP'));});
    report.checks.push('Launcher live control starts only live mode; camera remains unauthorized');
    const livePid=engine.child.pid;
    await endSession();assert.equal(engine.child,null);
    assert.throws(()=>process.kill(livePid,0));report.checks.push('End session releases its backend process');
    // Automated native dialog plumbing: supply a Unicode/space-containing fixture
    // through the real openRecording handler; no camera or external files mutated.
    const originalDialog=dialog.showOpenDialog;
    dialog.showOpenDialog=async(_window,options)=>{assert.equal(options.properties[0],'openFile');assert.ok(options.filters[0].extensions.includes('mov'));return {canceled:false,filePaths:[process.env.TT_DESKTOP_SMOKE_VIDEO]};};
    try{
      await window.webContents.executeJavaScript(`document.getElementById('recording').click()`);
      await eventually(async()=>{assert.equal(getContext().phase,'ready');assert.equal(getContext().mode,'simulation');});
    }finally{dialog.showOpenDialog=originalDialog;}
    const simulationPid=engine.child.pid;
    let packet=await eventually(async()=>{const p=await enginePacket(engine.url,null,simulationPid);assert.equal(p.analysis?.status,'COMPLETE');assert.ok(p.frame_id>0);return p;},120000);
    assert.equal(packet.analysis.analyzed_frames,packet.analysis.total_frames);assert.equal(packet.mode,'simulation');
    report.recording={pid:simulationPid,url:engine.url,analysis:packet.analysis,source:packet.video_file||packet.source_name,source_id:packet.source_id};
    await enginePacket(engine.url,'PAUSE',simulationPid);await enginePacket(engine.url,'SEEK_FRAME:120',simulationPid);
    await eventually(async()=>{const text=await window.webContents.executeJavaScript('document.body.innerText');assert.ok(text.includes('FRAME 120'));});
    report.checks.push('Native recording selection bridge handles Unicode/spaces and analyzes every frame');
    const endpoints=['/','/training-review','/api/training-review','/favicon.ico'];
    report.http=[];
    for(const endpoint of endpoints){const response=await fetch(getOrigin()+endpoint);assert.equal(response.status,200);report.http.push({endpoint,status:response.status});}
    const snapshot=await window.webContents.executeJavaScript(`({toolbar:!!document.querySelector('[aria-label="Desktop session controls"]'),canvas:!!document.querySelector('canvas'),frame:document.body.innerText.match(/FRAME 120/)?120:null,buttons:[...document.querySelectorAll('button')].map(b=>b.textContent).filter(t=>/Open recording|Live camera|End session/.test(t))})`);
    assert.equal(snapshot.toolbar,true);assert.equal(snapshot.frame,120);report.rendered=snapshot;
    await window.webContents.executeJavaScript('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))');
    const painted=await window.webContents.executeJavaScript(`document.querySelector('canvas[aria-label="Source footage with synchronized ball, person and table detections"]').dataset.frameId`);
    assert.equal(painted,'120');report.rendered.painted_frame=Number(painted);
    fs.writeFileSync(output.replace(/\.json$/,'.png'),(await window.webContents.capturePage()).toPNG());
    report.checks.push('Desktop renders footage and native controls; review API, current assets and favicon load');
    await window.webContents.executeJavaScript(`[...document.querySelectorAll('button')].find(button=>button.textContent.includes('2D RADAR')).click()`);
    const radar=await eventually(async()=>{
      const value=await window.webContents.executeJavaScript(`(()=>{const element=document.querySelector('[aria-label="2D table plane projection"]');return element?{frame:element.dataset.frameId,x:element.dataset.ballX,y:element.dataset.ballY}:null})()`);
      assert.equal(value?.frame,'120');return value;
    });
    await window.webContents.executeJavaScript(`[...document.querySelectorAll('button')].find(button=>button.textContent.includes('3D VOLUMETRIC')).click()`);
    const volumetric=await eventually(async()=>{
      const value=await window.webContents.executeJavaScript(`(()=>{const element=document.querySelector('[aria-label="3D table plane projection"]');return element?{frame:element.dataset.frameId,x:element.dataset.ballX,y:element.dataset.ballY}:null})()`);
      assert.equal(value?.frame,'120');return value;
    });
    assert.deepEqual(volumetric,radar);report.spatial_views={radar,volumetric};
    report.checks.push('Footage, 2D and 3D retain the same paused frame and authoritative table coordinates');
    await window.loadURL(getOrigin()+'/training-review');
    await eventually(async()=>{assert.ok((await window.webContents.executeJavaScript('document.body.innerText')).includes('Training review'));});
    await window.loadURL(getOrigin()+'/');
    const draft={ball:[[.2,.3,.01,.01]],person:[],table:[]},draftKey='phone_0123456789abcdef_1/frame_000001.jpg';
    await window.webContents.executeJavaScript(`window.ttDesktop.saveReviewDrafts(${JSON.stringify({[draftKey]:draft})})`);
    const saved=await window.webContents.executeJavaScript('window.ttDesktop.getReviewDrafts()');assert.deepEqual(saved[draftKey],draft);
    report.checks.push('Review drafts persist in an application-owned file independent of the HTTP port');
    // Invalid recordings preserve the previous source, just as in browser mode.
    const bad=path.join(dataRoot,'invalid recording.mov');fs.writeFileSync(bad,'not a video');
    const beforeInvalid=await enginePacket(engine.url,null,simulationPid);
    await enginePacket(engine.url,`LOAD_VIDEO:${bad}`,simulationPid);
    packet=await eventually(async()=>{const p=await enginePacket(engine.url,null,simulationPid);assert.ok(p.video_error);return p;});
    assert.equal(packet.source_id,beforeInvalid.source_id);assert.equal(packet.analysis_id,beforeInvalid.analysis_id);assert.equal(packet.frame_id,beforeInvalid.frame_id);report.checks.push('Invalid video preserves the active recording');
    const cancelledDialog=dialog.showOpenDialog;
    dialog.showOpenDialog=async()=>({canceled:true,filePaths:[]});
    try{await openRecording();assert.equal(engine.child.pid,simulationPid);}finally{dialog.showOpenDialog=cancelledDialog;}
    report.checks.push('Canceling recording selection preserves the session');
    assert.equal(report.renderer_errors.length,0,JSON.stringify(report.renderer_errors));
    report.frontend_origin=getOrigin();report.data_root=dataRoot;report.ok=true;
    // The outer runner checks all children exited after the actual window closes.
    window.close();
  }catch(error){report.ok=false;report.error=error.stack;console.error(error);app.quit();}
  finally{report.finished_at=new Date().toISOString();if(output)fs.writeFileSync(output,JSON.stringify(report,null,2));}
};

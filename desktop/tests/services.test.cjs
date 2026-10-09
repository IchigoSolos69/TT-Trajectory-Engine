const {test}=require('node:test');
const assert=require('node:assert/strict');
const {spawn}=require('node:child_process');
const net=require('node:net');
const {once}=require('node:events');
const {freePort,stopChild,EngineSession}=require('../services.cjs');
const {allowedPage}=require('../security.cjs');
const {pathToFileURL}=require('node:url');
const path=require('node:path');

test('desktop chooses a free loopback port while other services retain ownership',async()=>{
  const unrelated=net.createServer();unrelated.listen(0,'127.0.0.1');await once(unrelated,'listening');
  const occupied=unrelated.address().port,selected=await freePort();
  assert.notEqual(selected,occupied);assert.equal(unrelated.listening,true);
  await new Promise(resolve=>unrelated.close(resolve));
});
test('cleanup only stops the supplied child, preserving an unrelated process',async()=>{
  const own=spawn(process.execPath,['-e','setInterval(()=>{},1000)'],{windowsHide:true});
  const unrelated=spawn(process.execPath,['-e','setInterval(()=>{},1000)'],{windowsHide:true});
  await Promise.all([once(own,'spawn'),once(unrelated,'spawn')]);
  try{await stopChild(own);assert.ok(own.exitCode!==null||own.signalCode!==null);assert.throws(()=>process.kill(own.pid,0));assert.equal(unrelated.exitCode,null);}
  finally{await stopChild(unrelated);}
});
test('IPC navigation trust rejects external content and unintended local paths',()=>{
  const launcher=path.resolve(__dirname,'../launcher.html'),origin='http://127.0.0.1:45678';
  assert.equal(allowedPage(origin+'/',origin,launcher),true);
  assert.equal(allowedPage(origin+'/training-review',origin,launcher),true);
  assert.equal(allowedPage(pathToFileURL(launcher).href,origin,launcher),true);
  for(const url of ['https://example.com/',origin+'/api/training-review','http://localhost:45678/','http://127.0.0.1:45679/',pathToFileURL(__filename).href])assert.equal(allowedPage(url,origin,launcher),false);
});
test('live source never substitutes a recording',async()=>{
  const engine=new EngineSession({});await assert.rejects(engine.start('live','clip.mov'),/cannot load/);assert.equal(engine.child,null);
});

const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs/promises'),path=require('node:path'),os=require('node:os');
const {DraftStore}=require('../draft-store.cjs');
test('draft revisions survive a fresh instance and concurrent saves stay ordered',async()=>{
  const root=await fs.mkdtemp(path.join(os.tmpdir(),'tt-desktop-drafts-')),store=new DraftStore(root);
  assert.deepEqual(await store.read(),{});
  const key='phone_0123456789abcdef_1/frame_000001.jpg';
  await Promise.all([store.save({[key]:{ball:[],person:[],table:[]}}),store.save({[key]:{ball:[[.2,.3,.01,.01]],person:[],table:[]}})]);
  assert.equal((await new DraftStore(root).read())[key].ball.length,1);
  await assert.rejects(store.save({'../escape':{}}),/Invalid/);
  await fs.writeFile(store.file,'corrupt json');await assert.rejects(store.read());
  assert.equal(await fs.readFile(store.file,'utf8'),'corrupt json','Read error must not overwrite drafts');
});

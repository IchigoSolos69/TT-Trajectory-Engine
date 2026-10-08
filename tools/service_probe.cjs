// Local launcher health probe; does not authorize a camera or record footage.
const kind=process.argv[2],port=Number(process.argv[3]);
async function main(){
 if(kind==='frontend'){
  const response=await fetch(`http://127.0.0.1:${port}/api/health`,{signal:AbortSignal.timeout(4000)});
  if(!response.ok)throw Error('Dashboard health endpoint unavailable');
  console.log(JSON.stringify(await response.json()));return;
 }
 await new Promise((resolve,reject)=>{
  const socket=new WebSocket(`ws://127.0.0.1:${port}`);
  const timeout=setTimeout(()=>{socket.close();reject(Error('Engine health timeout'));},5000);
  socket.onerror=()=>{clearTimeout(timeout);reject(Error('Engine connection failed'));};
  socket.onmessage=event=>{try{const frame=JSON.parse(event.data);if(!frame.service)throw Error('Engine identity unavailable');
   if(kind==='shutdown')socket.send('SHUTDOWN');
   console.log(JSON.stringify(frame.service));clearTimeout(timeout);socket.close();resolve();
  }catch(error){clearTimeout(timeout);socket.close();reject(error);}};
 });
}
// Bound process lifetime even when an unrelated TCP listener never completes
// the WebSocket handshake or acknowledges close. Allow queued shutdown bytes
// to flush before exiting a successful local probe.
main().then(()=>setTimeout(()=>process.exit(0),100)).catch(error=>{console.error(error.message);process.exit(1);});

'use strict';
// Read-only logical-byte inventory. Never follows reparse points or touches Git.
const fs=require('node:fs'),path=require('node:path');
const root=path.resolve(__dirname,'..'),destination=process.argv[2],options=process.argv.slice(3);
if(options.length>1||options.some(option=>option!=='--summary-only'))throw Error('Usage: node tools/audit_repository_storage.cjs NEW_REPORT.json [--summary-only]');
const summaryOnly=options.includes('--summary-only');
if(!destination||fs.existsSync(destination))throw Error('Choose a new storage audit JSON');
const output=path.resolve(destination);
if(!output.startsWith(root+path.sep))throw Error('Report must stay in this workspace');
const files=[],directories=[],excluded=[],errors=[];
function visit(relative){
  const directory=path.join(root,relative);let entries;
  try{entries=fs.readdirSync(directory,{withFileTypes:true});}catch(error){errors.push({path:relative,error:error.stack});return {bytes:0,files:0};}
  let bytes=0,count=0;
  for(const entry of entries){
    const child=path.join(relative,entry.name),absolute=path.join(root,child);
    if(relative===''&&['.git','.aws','.codex','.agents'].includes(entry.name)){excluded.push({path:child,reason:'Protected metadata'});continue;}
    if(entry.isSymbolicLink()){excluded.push({path:child,reason:'Symlink/junction not followed'});continue;}
    if(entry.isDirectory()){const result=visit(child);bytes+=result.bytes;count+=result.files;}
    else if(entry.isFile()){
      try{const stat=fs.lstatSync(absolute);if(stat.isSymbolicLink()){excluded.push({path:child,reason:'Changed to alias during audit'});continue;}
        files.push({path:child,bytes:stat.size,mtime_ms:stat.mtimeMs});bytes+=stat.size;count++;}
      catch(error){errors.push({path:child,error:error.stack});}
    }
  }
  directories.push({path:relative||'.',bytes,files:count});return {bytes,files:count};
}
const total=visit('');directories.sort((a,b)=>b.bytes-a.bytes);files.sort((a,b)=>b.bytes-a.bytes);
fs.mkdirSync(path.dirname(output),{recursive:true});
const report={schema_version:1,created_at:new Date().toISOString(),root,logical_bytes:total.bytes,file_count:total.files,complete:errors.length===0,
  summary_only:summaryOnly,excluded,errors,
  directories:summaryOnly?directories.filter(row=>row.path.split(path.sep).length<=3):directories,
  files:summaryOnly?files.slice(0,100):files,
  limits:['Logical sizes, not allocated/compressed disk usage','Protected Git/configuration metadata and reparse targets excluded','No source/video/model content read or changed',...(summaryOnly?['Report lists only shallow directory totals and the 100 largest files; totals include every scanned file']:[])]};
fs.writeFileSync(output,JSON.stringify(report)+'\n',{flag:'wx'});
console.log(JSON.stringify({report:output,logical_gib:total.bytes/1024**3,file_count:total.files,complete:report.complete,
  largest_directories:directories.filter(row=>row.path.split(path.sep).length<=2).slice(0,30),largest_files:files.slice(0,15),excluded:excluded.length,errors},null,2));

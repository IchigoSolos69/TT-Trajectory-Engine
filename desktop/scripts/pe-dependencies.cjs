'use strict';
const fs = require('node:fs');
function imports(file) {
  const data = fs.readFileSync(file);
  if (data.length < 64 || data.toString('ascii',0,2) !== 'MZ') throw Error(`Not a Windows executable: ${file}`);
  const pe = data.readUInt32LE(60);
  if (data.toString('ascii',pe,pe+4) !== 'PE\0\0') throw Error(`Invalid PE header: ${file}`);
  const sections = data.readUInt16LE(pe+6), optionalSize = data.readUInt16LE(pe+20), optional = pe+24;
  const magic = data.readUInt16LE(optional);
  if (![0x10b,0x20b].includes(magic)) throw Error(`Unsupported PE: ${file}`);
  const directories = optional+(magic===0x20b?112:96);
  const sectionStart = optional+optionalSize;
  const offset = rva => {
    if (!rva) return 0;
    for (let i=0;i<sections;i++) {
      const section=sectionStart+40*i, address=data.readUInt32LE(section+12), size=Math.max(data.readUInt32LE(section+8),data.readUInt32LE(section+16));
      if (rva>=address && rva<address+size) return data.readUInt32LE(section+20)+rva-address;
    }
    throw Error(`Unmapped PE address ${rva} in ${file}`);
  };
  const string = rva => { const start=offset(rva), end=data.indexOf(0,start); if(end<0)throw Error('Unterminated PE import'); return data.toString('ascii',start,end); };
  const result = new Set();
  const importRva=data.readUInt32LE(directories+8);
  if(importRva)for(let pos=offset(importRva);data.readUInt32LE(pos+12);pos+=20)result.add(string(data.readUInt32LE(pos+12)));
  const delayRva=data.readUInt32LE(directories+13*8);
  if(delayRva)for(let pos=offset(delayRva);data.readUInt32LE(pos+4);pos+=32){
    if(!(data.readUInt32LE(pos)&1))throw Error(`Unsupported absolute delayed import in ${file}`);
    result.add(string(data.readUInt32LE(pos+4)));
  }
  return [...result];
}
module.exports={imports};

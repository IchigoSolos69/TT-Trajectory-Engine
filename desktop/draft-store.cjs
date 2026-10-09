'use strict';
const fs=require('node:fs/promises'),path=require('node:path');
class DraftStore{
  constructor(root){this.file=path.join(root,'artifacts','desktop','review-drafts.json');this.pending=Promise.resolve();}
  async read(){
    try{return JSON.parse(await fs.readFile(this.file,'utf8'));}
    catch(error){if(error.code==='ENOENT')return {};throw error;}
  }
  save(value){
    if(!value||typeof value!=='object'||Array.isArray(value)||Object.keys(value).some(key=>! /^(?:live_[0-9]+_[0-9]+|phone_[0-9a-f]{16}_[0-9]+)\/frame_[0-9]+\.jpg$/.test(key)))return Promise.reject(Error('Invalid review draft collection'));
    const text=JSON.stringify(value);if(Buffer.byteLength(text)>5_000_000)return Promise.reject(Error('Review drafts exceed local storage limit'));
    const write=async()=>{await fs.mkdir(path.dirname(this.file),{recursive:true});const temporary=this.file+'.tmp';await fs.writeFile(temporary,text);await fs.rename(temporary,this.file);};
    const operation=this.pending.then(write);
    // Preserve a rejected caller-visible write while allowing a later retry.
    this.pending=operation.catch(error=>console.error('[desktop] Draft write failed:',error));
    return operation;
  }
}
module.exports={DraftStore};

// Rasterize our code-defined icon without an image service or external assets.
const fs=require('node:fs'),path=require('node:path'),zlib=require('node:zlib');
const size=128,data=Buffer.alloc((size*4+1)*size);
const distance=(x,y,ax,ay,bx,by)=>{const t=Math.max(0,Math.min(1,((x-ax)*(bx-ax)+(y-ay)*(by-ay))/((bx-ax)**2+(by-ay)**2)));return Math.hypot(x-ax-t*(bx-ax),y-ay-t*(by-ay));};
for(let y=0;y<size;y++)for(let x=0;x<size;x++){
 let color=[8,51,68,255];
 const corner=Math.hypot(Math.max(0,30-x,x-97),Math.max(0,30-y,y-97));if(corner>30)color=[0,0,0,0];
 if(y>=56&&y<=86&&x>=38-(y-56)*16/30&&x<=90+(y-56)*16/30)color=[22,78,99,255];
 if(Math.min(distance(x,y,22,86,38,56),distance(x,y,38,56,90,56),distance(x,y,90,56,106,86),distance(x,y,106,86,22,86))<2)color=[34,211,238,255];
 if(distance(x,y,64,57,64,86)<1.5)color=[226,232,240,255];
 for(let i=0;i<50;i++){const t=i/49,px=29*(1-t)**2+116*t*(1-t)+90*t*t,py=43*(1-t)**2+26*t*(1-t)+44*t*t;if(Math.hypot(x-px,y-py)<2.5)color=[110,231,183,255];}
 if(Math.hypot(x-91,y-44)<8)color=[251,191,36,255];
 const offset=y*(size*4+1)+1+x*4;for(let c=0;c<4;c++)data[offset+c]=color[c];
}
function crc(buffer){let v=0xffffffff;for(const b of buffer){v^=b;for(let i=0;i<8;i++)v=(v>>>1)^((v&1)?0xedb88320:0);}return (v^0xffffffff)>>>0;}
function chunk(name,bytes){const body=Buffer.concat([Buffer.from(name),bytes]),out=Buffer.alloc(bytes.length+12);out.writeUInt32BE(bytes.length);body.copy(out,4);out.writeUInt32BE(crc(body),out.length-4);return out;}
const header=Buffer.alloc(13);header.writeUInt32BE(size);header.writeUInt32BE(size,4);header[8]=8;header[9]=6;
const png=Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]),chunk('IHDR',header),chunk('IDAT',zlib.deflateSync(data)),chunk('IEND',Buffer.alloc(0))]);
const root=path.resolve(__dirname,'..');fs.writeFileSync(path.join(root,'icon.png'),png);
const ico=Buffer.alloc(22);ico.writeUInt16LE(1,2);ico.writeUInt16LE(1,4);ico[6]=size;ico[7]=size;ico.writeUInt16LE(1,10);ico.writeUInt16LE(32,12);ico.writeUInt32LE(png.length,14);ico.writeUInt32LE(22,18);fs.writeFileSync(path.join(root,'icon.ico'),Buffer.concat([ico,png]));

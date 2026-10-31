"""Self-contained offline 3D viewer (no external libraries)."""
import json

TEMPLATE = r'''<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>AI MoCap Viewer</title><style>
:root{--bg:#0b0f14;--fg:#e8eef4;--mut:#8a98a8;--ln:#24313d;--raw:#ff5d5d;--l:#ff9a3c;--r:#4f8cff;--c:#37e0e8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px system-ui,Segoe UI,Tahoma,Arial;display:flex;flex-direction:column;height:100vh}
header{padding:10px 16px;border-bottom:1px solid var(--ln);display:flex;gap:14px;align-items:center;flex-wrap:wrap}
h1{font-size:15px;margin:0}small{color:var(--mut)}label{display:flex;gap:6px;align-items:center;color:var(--mut);cursor:pointer}
canvas{flex:1;width:100%;min-height:0;display:block;cursor:grab}
.bar{direction:ltr;display:flex;gap:10px;align-items:center;padding:10px 16px;border-top:1px solid var(--ln)}
button,select{background:#18222d;color:var(--fg);border:1px solid var(--ln);border-radius:8px;padding:6px 12px;cursor:pointer}
input[type=range]{flex:1}#t{min-width:150px;color:var(--mut);font-variant-numeric:tabular-nums}
</style></head><body>
<header><h1>عارض الحركة — خام مقابل مصحَّح</h1>
<label><input type="checkbox" id="cC" checked>المصحَّح</label>
<label><input type="checkbox" id="cR">الخام</label>
<label><input type="checkbox" id="cM" checked>مركز الثقل</label>
<label><input type="checkbox" id="cF" checked>تلامس القدم</label>
<small>اسحب للتدوير · العجلة للتقريب · برتقالي = يسار الشخصية، أزرق = يمين</small></header>
<canvas id="c"></canvas>
<div class="bar"><button id="pp">⏸</button><select id="sp"><option>0.25</option><option>0.5</option><option selected>1</option><option>2</option></select>
<input id="s" type="range" min="0" max="0" value="0"><span id="t"></span></div>
<script>
const D=__DATA__;const N=D.joints.length,T=D.corrected.length,I=Object.fromEntries(D.joints.map((n,i)=>[n,i]));
const col=n=>n.startsWith('left')?'#ff9a3c':n.startsWith('right')?'#4f8cff':'#37e0e8';
const cv=document.getElementById('c'),g=cv.getContext('2d');let yaw=.7,pit=.18,zoom=1.4,fi=0,play=true,acc=0,last=0;
const $=id=>document.getElementById(id),s=$('s');s.max=T-1;
function fit(){const r=devicePixelRatio||1;cv.width=cv.clientWidth*r;cv.height=cv.clientHeight*r}addEventListener('resize',fit);fit();
let drag=null;cv.onpointerdown=e=>{drag=[e.clientX,e.clientY];cv.setPointerCapture(e.pointerId)};cv.onpointerup=()=>drag=null;
cv.onpointermove=e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.008;pit=Math.max(-1.3,Math.min(1.3,pit+(e.clientY-drag[1])*.008));drag=[e.clientX,e.clientY]};
cv.onwheel=e=>{e.preventDefault();zoom=Math.max(.3,Math.min(5,zoom*(e.deltaY<0?1.1:.9)))};
$('pp').onclick=()=>{play=!play;$('pp').textContent=play?'⏸':'▶'};s.oninput=()=>{fi=+s.value;play=false;$('pp').textContent='▶'};
let cx0=0,cz0=0;
function P(p){const x=p[0]-cx0,y=p[1]-.9,z=p[2]-cz0;const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pit),sp=Math.sin(pit);
 let X=x*cy+z*sy,Z=-x*sy+z*cy;let Y=y*cp-Z*sp;Z=y*sp+Z*cp;const d=4.5,f=Math.min(cv.width,cv.height)*.9*zoom,k=f/(d+Z);return[cv.width/2+X*k,cv.height/2-Y*k,k]}
function line(a,b,c,w){g.strokeStyle=c;g.lineWidth=w;g.beginPath();g.moveTo(a[0],a[1]);g.lineTo(b[0],b[1]);g.stroke()}
function skel(F,alpha,raw,q){const pts=F.map(P);g.globalAlpha=alpha;g.lineCap='round';
 for(const[a,b]of D.bones){const A=pts[I[a]],B=pts[I[b]];line(A,B,raw?'#ff5d5d':col(b),Math.max(2,A[2]*.03))}
 D.joints.forEach((n,i)=>{const c=q?q[i]:1;g.fillStyle=raw?'#ff5d5d':col(n);g.globalAlpha=alpha*(.35+.65*c);g.beginPath();g.arc(pts[i][0],pts[i][1],Math.max(3,pts[i][2]*.022),0,7);g.fill()});g.globalAlpha=1}
function draw(){const W=cv.width,H=cv.height;g.clearRect(0,0,W,H);const C=D.corrected[fi];cx0=C[I.hip][0];cz0=C[I.hip][2];
 g.strokeStyle='#1b2a36';g.lineWidth=1;const gx=Math.round(cx0*2)/2,gz=Math.round(cz0*2)/2;
 for(let i=-8;i<=8;i++){line(P([gx+i*.25,0,gz-2]),P([gx+i*.25,0,gz+2]),'#1b2a36',1);line(P([gx-2,0,gz+i*.25]),P([gx+2,0,gz+i*.25]),'#1b2a36',1)}
 if($('cR').checked&&D.raw)skel(D.raw[fi],.55,true);
 if($('cC').checked)skel(C,1,false,D.quality?D.quality[fi]:null);
 if($('cM').checked&&D.com){g.strokeStyle='#e8eef4';g.globalAlpha=.5;g.beginPath();for(let k=Math.max(0,fi-60);k<=fi;k++){const p=P([D.com[k][0],0.002,D.com[k][2]]);k==Math.max(0,fi-60)?g.moveTo(p[0],p[1]):g.lineTo(p[0],p[1])}g.stroke();g.globalAlpha=1;
  const c=P(D.com[fi]),c0=P([D.com[fi][0],0,D.com[fi][2]]);line(c,c0,'#e8eef4',1);g.fillStyle='#fff';g.beginPath();g.arc(c[0],c[1],6,0,7);g.fill()}
 if($('cF').checked&&D.contacts){for(const s of['left','right'])if(D.contacts[s][fi]){const p=P([C[I[s+'_heel']][0],0,C[I[s+'_heel']][2]]);g.strokeStyle='#7CFC9A';g.lineWidth=3;g.beginPath();g.ellipse(p[0],p[1],p[2]*.08,p[2]*.03,0,0,7);g.stroke()}}
 $('t').textContent=`frame ${fi+1}/${T} · ${(fi/D.fps).toFixed(2)}s`}
function loop(t){const dt=t-last;last=t;if(play){acc+=dt*(+$('sp').value);const step=1000/D.fps;while(acc>step){fi=(fi+1)%T;acc-=step}s.value=fi}draw();requestAnimationFrame(loop)}requestAnimationFrame(loop);
</script></body></html>'''


def write_viewer(data, path):
    open(path, 'w', encoding='utf-8').write(TEMPLATE.replace('__DATA__', json.dumps(data, separators=(',', ':'))))

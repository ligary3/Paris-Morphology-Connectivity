import fs from 'node:fs/promises';
import path from 'node:path';
const {Presentation,PresentationFile}=await import(process.env.ARTIFACT_TOOL_MODULE || '@oai/artifact-tool');
import {fileURLToPath} from 'node:url';
const root=fileURLToPath(new URL('../../',import.meta.url)),out=path.join(root,'outputs/figures'),source='', assets=path.join(root,'images/workflow_assets');
await fs.mkdir(out,{recursive:true});
const p=Presentation.create({slideSize:{width:1526.4,height:980}}),s=p.slides.add(); s.background.fill='#FFFFFF';
const ink='#141414',sub='#242424',slate='#292E30',teal='#153B2F',blue='#193547',line='#C8D6D9',white='#FFFFFF';
const flowGreen='#348348',flowBlue='#246BB0';
let id=0;
function rect(x,y,w,h,fill='none',edge='none',lw=1,dash=false){if(x>=1060){x=1060+.8*(x-1060);w*=.8}return s.shapes.add({name:'shape-'+(++id),geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:edge,width:lw,style:dash?'dashed':'solid'}})}
function tx(str,x,y,w,h,size=21,bold=false,color=ink,align='left',face='Times New Roman'){const z=rect(x,y,w,h);z.name=str.slice(0,70);z.text=str;z.text.style={typeface:'Times New Roman',fontSize:size,bold,color,alignment:align,verticalAlignment:'middle',autoFit:'none',insets:{left:0,right:0,top:0,bottom:0}};return z}
let imageIndex=0;
const assetNames=['geometry','real','homo','circuit_real','circuit_homo','wrf'];
import {readFileSync} from 'node:fs';
function pic(src,x,y,w,h,alt,crop){const data=readFileSync(path.join(assets,assetNames[imageIndex++]+'.png'));s.images.add({blob:new Uint8Array(data),contentType:'image/png',position:{left:x,top:y,width:w,height:h},fit:'contain',alt})}
function arrow(x,y,X,Y,color=flowBlue){const h=y===Y;if(x>=1060){x=1060+.8*(x-1060);X=1060+.8*(X-1060)}else if(X>=1060){X=1060+.8*(X-1060)}return s.shapes.add({name:'flow-'+(++id),geometry:h?'rightArrow':'downArrow',position:h?{left:x,top:y-3.3,width:X-x,height:6.6}:{left:x-3.3,top:y,width:6.6,height:Y-y},fill:color,line:{fill:'none',width:0}})}
function col(x,w,n,title,col){const [base,header,stroke]=n==='01'?['#F4F3F0','#DBDEDB','#879492']:n==='02'?['#F0F5EE','#CBDEC9','#809D84']:['#EFF5F7','#C8DDE6','#799AA9'];rect(x,12,w,954,base,stroke,1.5,true);rect(x+4,16,w-8,58,header);rect(x+4,73,w-8,1.7,col);tx(Number(n)+'   '+title,x+12,15,w-22,56,28,true,ink,'left','Times New Roman')}
function head(txt,x,y,w,col){tx(txt,x,y,w,36,24,true,col,'left','Times New Roman')}
function band(str,x,y,w,h,col,bg,sz=21){rect(x,y,w,h,bg);tx(str,x+7,y+2,w-14,h-4,sz,true,col,'center')}
function question(n,str,x,y,w,col){tx(n+'.',x,y,40,53,20,true,col,'left','Times New Roman');tx(str,x+43,y,w-45,53,19,false,sub,'left','Times New Roman')}
col(12,272,'01','Data sources',slate);col(302,738,'02','REAL–HOMO experiments',teal);col(1060,568,'03','Network value and air quality',blue);
// Only scientific modules have a frame; data labels and individual operations stay unboxed.
function moduleBox(x,y,w,h,edge){rect(x,y,w,h,white,edge,1.1)}
for(const [y,h] of [[87,366],[462,250],[720,235]])moduleBox(20,y,256,h,'#C7CFCA');
moduleBox(311,87,278,810,'#BDCFBD');
moduleBox(607,87,423,495,'#BDCFBD');
moduleBox(607,590,423,307,'#BDCFBD');
moduleBox(1069,87,549,566,'#BDCED6');
moduleBox(1069,660,549,56,'#BDCED6');
moduleBox(1069,725,549,230,'#BDCED6');
head('Urban geometry',26,96,244,slate);
pic(source,28,153,240,177,'Paris building footprints and land cover',[36,152,284,346]);
tx('BD TOPO 2022',28,337,240,32,22,true);
tx('Footprints and heights\nWater, parks and urban extent',28,375,240,75,19,false,sub);
head('Meteorology',26,469,244,slate);
tx('ERA5 + static geography',28,518,235,55,21,true);
tx('WRF forcing',28,580,235,43,19,false,sub);
tx('7 surface stations and\nTrappes radiosonde',28,634,235,68,20,false,sub);
head('LCZ and air quality',26,731,244,slate);
tx('Cerema LCZ 2022',28,790,237,35,21,true);
tx('Airparif',28,836,237,35,21,true);
tx('NO₂, O₃ and Ox\nParis, July 2022',28,877,235,68,20,false,sub);
head('Paired scenarios',318,96,259,teal);tx('100 m building morphology',318,148,259,40,21,true);
tx('Height, coverage,\n16-direction FAI',318,194,259,62,20,false,sub);
tx('REAL',318,273,259,35,23,true,teal,'center','Times New Roman');
const figs='manuscript_paris/figures/';
pic(figs+'REAL_concept_no_text.png',321,313,253,162,'Realistic morphology schematic');
tx('Observed heterogeneity',318,476,259,44,19,false,sub,'center');
tx('HOMO',318,531,259,35,23,true,teal,'center','Times New Roman');
pic(figs+'HOMO_concept_no_text.png',321,571,253,162,'Homogenized morphology schematic');
tx('Uniform built-area parameters',318,734,259,42,19,false,sub,'center');
tx('Held constant',318,793,260,28,19,true,teal);
tx('Urban extent, water, parks,\nperiphery and morphology totals',318,826,270,59,18,false,sub);
head('Circuit connectivity',618,96,404,teal);
tx('Directional FAI resistance',618,149,403,32,22,true);
tx('Rᵢ(θ) = 1 + FAIᵢ(θ) / q',620,190,400,35,24,false,ink);
tx('Edge current   Cᵢⱼ = (φᵢ − φⱼ) / rᵢⱼ',620,231,400,35,22,false,ink);
tx('REAL',618,280,198,25,18,true,teal,'center');tx('HOMO',823,280,197,25,18,true,teal,'center');
pic(source,619,312,195,107,'REAL connectivity map',[815,352,1039,476]);pic(source,823,312,196,107,'HOMO connectivity map',[1071,352,1295,476]);
function delta(symbol,tag,y,h){
  rect(618,y,404,h,'#DBECE6');
  const x=tag?691:714;
  tx('Δ'+symbol,x,y,43,h,25,true,teal);
  if(tag)tx(tag,x+34,y+12,27,h-12,15,true,teal);
  const q=x+(tag?65:42);
  tx('= '+symbol,q,y,47,h,25,true,teal);
  tx('REAL',q+40,y+12,45,h-12,15,true,teal);
  tx('− '+symbol,q+90,y,47,h,25,true,teal);
  tx('HOMO',q+130,y+12,52,h-12,15,true,teal);
}
delta('C','3D',432,45);
tx('LCZ benchmark',618,486,404,28,19,true,teal);
tx('Paris class-median FAI\nSame resistance settings and HOMO reference',618,519,408,57,18,false,sub);
head('Paired WRF response',618,597,404,teal);
tx('WRF + SLUCM   •   9 / 3 / 1 km',618,649,404,38,22,true);
tx('Shared forcing and static geography',618,691,404,32,19,false,sub);
pic(source,618,737,136,103,'WRF paired airflow response',[841,835,1010,991]);
tx('H, λₚ, scalar FAI, z₀ and d',764,730,259,33,18,false,sub);
tx('10–1000 m layer-mean\nalong-wind velocity',764,769,259,65,21,false,ink);
delta('U','',850,42);
question('Q1','Connectivity and airflow redistribution',320,902,690,teal);
arrow(281,214,316,214,flowGreen);arrow(578,215,613,215,flowGreen);arrow(578,668,613,668,flowGreen);
head('Network added-value test',1078,96,532,blue);
tx('Direction-matched inputs',1078,149,532,36,22,true);
function inputRow(y,label,value){
 rect(1080,y,529,38,'#FFFFFF','#9FB7C1');
 rect(1081,y+1,140,36,'#E6F0F4');
 tx(label,1087,y+1,128,36,19,true,blue,'center');
 tx(value,1231,y+1,372,36,20,false,ink,'center');
}
inputRow(196,'Predictors','Building- or LCZ-derived ΔC');
inputRow(240,'WRF target','Mean ΔU within each inflow class');
tx('Baseline morphology X',1079,306,532,36,23,true);tx('Local, neighborhood and upwind',1079,346,532,32,20,false,sub);
band('X vs. X + ΔC (3D)',1080,397,250,55,blue,'#E6F0F4');band('X vs. X + ΔC (LCZ)',1345,397,264,55,blue,'#E6F0F4');
arrow(1205,454,1205,477);arrow(1477,454,1477,477);
rect(1080,480,529,46,'#FFFFFF','#ADC3CD');tx('Repeated spatial block CV',1087,483,515,40,21,false,ink,'center');
arrow(1344,528,1344,546);
tx('Held-out RMSE reduction (%)',1080,548,529,44,24,true,blue,'center','Times New Roman');
question('Q2','Added value of 3D and LCZ networks',1080,604,531,blue);
tx('Robustness',1080,670,120,28,19,true,blue);
tx('Vertical range, nonlinear model,\npermutation and resistance scale',1207,659,401,55,19,false,sub);
head('Air-quality association',1078,734,532,blue);
band('Hourly WRF ΔU',1080,778,207,44,blue,'#E6F0F4',20);
tx('+',1292,778,31,44,21,true,blue,'center');
band('Airparif background stations',1328,778,281,44,blue,'#E6F0F4',19);
tx('Matched by station and hour',1080,827,529,25,19,true,ink,'center');
arrow(1344,852,1344,866);
rect(1080,868,529,57,white,'#ADC3CD');
tx('Two-way demeaning by station and hour',1084,868,521,28,19,false,ink,'center');
tx('Spearman ρ with time-block bootstrap',1084,896,521,27,20,false,ink,'center');
tx('Q3.  Associations with NO₂, O₃ and Ox',1080,927,529,26,19,true,blue,'center');
// Cross-column routes terminate at the relevant input rows.
const rule=(x,y,w,h,c)=>rect(x,y,w,h,c);
rule(1022,452,27,1.6,flowBlue);rule(1048,215,1.6,239,flowBlue);arrow(1048,215,1080,215);
rule(1022,870,32,1.6,flowBlue);rule(1053,259,1.6,613,flowBlue);arrow(1053,259,1080,259);arrow(1053,800,1080,800);
s.speakerNotes.textFrame.setText('Paired REAL–HOMO study in Paris. The original schematic illustrations are conceptual. Two independent added-value comparisons share the baseline and target. LCZ connectivity uses the same HOMO reference. The Airparif analysis uses hourly responses.');
const dst=path.join(out,'Fig01_workflow_v19.pptx');
await(await PresentationFile.exportPptx(p)).save(dst);
await fs.writeFile(path.join(out,'Fig01_workflow_v19.png'),new Uint8Array(await(await p.export({slide:s,format:'png',scale:2.5})).arrayBuffer()));

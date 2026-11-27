/* Receives lexical `data` from comparison_script. Uses no network or physics. */
if (!window.Plotly) throw new Error('The page Plotly bundle is unavailable');
const graph = document.getElementById('field-comparison-3d');
if (!graph) throw new Error('The comparison plot container is missing');
window.rebotComparisonData = data;
const nativePose = document.getElementById('recorded-pose');
const ownPose = document.getElementById('comparison-pose');
const poseLabel = document.getElementById('comparison-pose-label');
const metrics = document.getElementById('comparison-pose-metrics');
const status = document.getElementById('comparison-status');
const activeMarginLabel = document.getElementById('comparison-active-margin');
const offsetSelector=document.getElementById('comparison-active-offset');
const sceneSelector=document.getElementById('comparison-scene-layout');
let activeMargin=data.default_margin_m;
const initial = data.frames[0];
const dynamicMeshes = [], dynamicLines = [], environmentIndices = [], marginIndices = [];
const commandLines=[],commandHeads=[];
const derivationTraces=[];
let commandPayload=null,graphReady=false;
let guidancePayload=null,guidedMode='poisson',lastFit=null;
window.addEventListener('rebot-derivation-step',event=>{
  guidancePayload=event.detail;guidedMode=guidancePayload.mode||'poisson';
  if(graphReady){fitComparison();renderGuidance(guidancePayload).catch(error=>console.error('Derivation overlay',error));}
});
const traces = [];
const clone = value => JSON.parse(JSON.stringify(value));
function coordinates(vertices) {
  return {x:vertices.map(p=>p[0]), y:vertices.map(p=>p[1]), z:vertices.map(p=>p[2])};
}
function triangleCoordinates(mesh) {
  return {...coordinates(mesh.vertices), i:mesh.faces.map(f=>f[0]),
          j:mesh.faces.map(f=>f[1]), k:mesh.faces.map(f=>f[2])};
}
function meshTrace(mesh, attributes) {
  return {type:'mesh3d', ...triangleCoordinates(mesh), flatshading:true,
          lighting:{ambient:.8,diffuse:.55,specular:.05},
          hoverinfo:'name',showlegend:false,...attributes};
}
function edges(vertices) {
  const x=[],y=[],z=[];
  for(const edge of data.arm_edges){
    for(const index of edge){x.push(vertices[index][0]);y.push(vertices[index][1]);z.push(vertices[index][2]);}
    x.push(null);y.push(null);z.push(null);
  }
  return {x,y,z};
}
for(const scene of ['scene','scene2']){
  if(scene==='scene') for(const mesh of data.environment){
    environmentIndices.push(traces.length);
    traces.push(meshTrace(mesh,{scene,color:mesh.color,opacity:.11,
                               name:`Environment h = ${mesh.level_m2} m²`}));
  }
  for(const obstacle of data.obstacles){
    traces.push(meshTrace(obstacle,{scene,color:'#B85042',opacity:.68,
                                   name:`Physical obstacle: ${obstacle.name}`}));
  }
  dynamicMeshes.push({index:traces.length,kind:'arm',scene});
  traces.push(meshTrace({vertices:initial.arm_vertices,faces:data.arm_faces},
                        {scene,facecolor:data.arm_face_colors,opacity:.96,name:'Articulated oriented arm boxes'}));
  dynamicLines.push({index:traces.length,kind:'edges',scene});
  traces.push({type:'scatter3d',...edges(initial.arm_vertices),scene,mode:'lines',
               line:{color:'#223648',width:1.3},showlegend:false,hoverinfo:'skip'});
  dynamicLines.push({index:traces.length,kind:'nearest',scene});
  traces.push({type:'scatter3d',...coordinates(initial.nearest.points_m),scene,mode:'lines+markers',
               line:{color:'#AA3377',width:5,dash:'dash'},marker:{color:'#AA3377',size:3},
               name:'Selected surface pair',showlegend:false,hoverinfo:'name'});
  if(scene==='scene2') for(let index=0;index<data.margins_m.length;index++){
    const shell=initial.shells[index],margin=data.margins_m[index];
    marginIndices.push(traces.length);
    dynamicMeshes.push({index:traces.length,kind:'shell',marginIndex:index,scene});
    traces.push(meshTrace(shell,{scene,color:shell.color,opacity:.24,
                                 visible:margin===data.default_margin_m,
                                 name:`Robot exterior distance = ${margin} m`}));
  }
  const commandColor=scene==='scene'?'#0072B2':'#009E73';
  commandLines.push({index:traces.length,scene});
  traces.push({type:'scatter3d',x:[],y:[],z:[],scene,mode:'lines',
               line:{color:commandColor,width:7},name:'Instantaneous filtered TCP velocity × 0.25 s',
               showlegend:false,hoverinfo:'name'});
  commandHeads.push({index:traces.length,scene});
  traces.push({type:'cone',x:[],y:[],z:[],u:[],v:[],w:[],scene,anchor:'tip',
               sizemode:'absolute',sizeref:.022,showscale:false,
               colorscale:[[0,commandColor],[1,commandColor]],
               name:'Instantaneous TCP velocity arrowhead',showlegend:false,hoverinfo:'name'});
  for(const kind of ['outline','points','direction']){
    derivationTraces.push({index:traces.length,scene,kind});
    traces.push({type:'scatter3d',x:[],y:[],z:[],text:[],scene,
      mode:kind==='outline'?'lines':kind==='points'?'markers+text':'lines+text',
      marker:{size:7,color:'#D55E00'},line:{color:'#D55E00',width:kind==='outline'?5:7},
      textposition:'top center',textfont:{size:14,color:'#8A3500'},
      name:kind==='direction'?'Selected gradient / normal direction':'Selected mathematical geometry',showlegend:false,hoverinfo:'text'});
  }
  derivationTraces.push({index:traces.length,scene,kind:'cone'});
  traces.push({type:'cone',x:[],y:[],z:[],u:[],v:[],w:[],scene,anchor:'tip',sizemode:'absolute',sizeref:.025,
    colorscale:[[0,'#D55E00'],[1,'#D55E00']],showscale:false,showlegend:false,name:'Gradient or normal; not a force',hoverinfo:'name'});
}
const bounds=data.display_bounds_m;
const camera={eye:{x:1.5,y:-1.75,z:1.15},center:{x:0,y:0,z:0},up:{x:0,y:0,z:1}};
const axis={showbackground:true,backgroundcolor:'#F6F7F9',gridcolor:'#D9DEE5',zeroline:false,
            showspikes:false,tickfont:{size:11}};
function sceneLayout(){
  return {xaxis:{...axis,title:{text:'x (m)'},range:[bounds[0][0],bounds[1][0]]},
          yaxis:{...axis,title:{text:'y (m)'},range:[bounds[0][1],bounds[1][1]]},
          zaxis:{...axis,title:{text:'z (m)'},range:[bounds[0][2],bounds[1][2]]},
          aspectmode:'data',camera:clone(camera)};
}
const layout={height:700,margin:{l:0,r:0,t:55,b:15},paper_bgcolor:'white',
              font:{family:'STIX Two Text, Times New Roman, serif',size:15,color:'#273340'},
              scene:sceneLayout(),scene2:sceneLayout(),uirevision:'comparison-camera',showlegend:false};
await Plotly.newPlot(graph,traces,layout,{displaylogo:false,responsive:true,
  toImageButtonOptions:{format:'png',filename:'rebot-field-comparison',scale:2}});
graphReady=true;
let cameraBusy=false;
graph.on('plotly_relayout',event=>{
  if(cameraBusy)return;
  const keys=Object.keys(event);
  const source=keys.some(k=>k==='scene2.camera'||k.startsWith('scene2.camera.'))?'scene2':
               keys.some(k=>k==='scene.camera'||k.startsWith('scene.camera.'))?'scene':null;
  if(!source)return;
  const target=source==='scene'?'scene2':'scene';
  const next=clone(graph.layout[source].camera);
  cameraBusy=true;
  Promise.resolve(Plotly.relayout(graph,{[`${target}.camera`]:next}))
    .finally(()=>{cameraBusy=false;});
});
function fitComparison(){
  const width=graph.parentElement.getBoundingClientRect().width;
  const requested=sceneSelector?.value||'auto';
  const view=requested==='auto'?(width>=920?'both':guidedMode):requested;
  const narrow=width<650;
  const key=view+'|'+narrow;
  if(key===lastFit)return;
  lastFit=key;
  const annotations=view!=='both'?
    [{text:view==='poisson'?'Environment Poisson · H (m²)':'Articulated arm envelope · dR (m)',x:.5,y:1.025,xref:'paper',yref:'paper',showarrow:false}]:narrow?
    [{text:'Environment Poisson · h (m²)',x:.5,y:1.01,xref:'paper',yref:'paper',showarrow:false},
     {text:'Robot-centred exterior distance · dR (m)',x:.5,y:.485,xref:'paper',yref:'paper',showarrow:false}]:
    [{text:'Environment Poisson · h (m²)',x:.235,y:1.035,xref:'paper',yref:'paper',showarrow:false},
     {text:'Robot-centred exterior distance · dR (m)',x:.765,y:1.035,xref:'paper',yref:'paper',showarrow:false}];
  const full={x:[0,1],y:[0,1]},hidden={x:[0,1],y:[0,.000001]};
  const settings={'scene.domain':view==='both'?(narrow?{x:[0,1],y:[.54,1]}:{x:[0,.48],y:[0,1]}):(view==='poisson'?full:hidden),
    'scene2.domain':view==='both'?(narrow?{x:[0,1],y:[0,.46]}:{x:[.52,1],y:[0,1]}):(view==='distance'?full:hidden),
    height:view==='both'&&narrow?950:560,annotations};
  for(const scene of ['scene','scene2'])for(const axis of ['xaxis','yaxis','zaxis'])
    settings[scene+'.'+axis+'.visible']=view==='both'||(view==='poisson'?scene==='scene':scene==='scene2');
  Plotly.relayout(graph,settings);
  graph.dataset.sceneView=view;
}
fitComparison();window.addEventListener('resize',fitComparison);
if(sceneSelector)sceneSelector.addEventListener('change',fitComparison);
if(window.ResizeObserver)new ResizeObserver(()=>{fitComparison();Plotly.Plots.resize(graph);}).observe(graph.parentElement);
for(const checkbox of document.querySelectorAll('.comparison-poisson-toggle')){
  checkbox.addEventListener('change',()=>Plotly.restyle(graph,{visible:checkbox.checked},
                             [environmentIndices[Number(checkbox.dataset.index)]]));
}
for(const checkbox of document.querySelectorAll('.comparison-margin-toggle')){
  checkbox.addEventListener('change',()=>{
    const checked=[...document.querySelectorAll('.comparison-margin-toggle')].filter(input=>input.checked);
    if(!checked.length){checkbox.checked=true;return;}
    if(checkbox.checked)activeMargin=data.margins_m[Number(checkbox.dataset.index)];
    else if(activeMargin===data.margins_m[Number(checkbox.dataset.index)])
      activeMargin=data.margins_m[Number(checked[0].dataset.index)];
    Plotly.restyle(graph,{visible:checkbox.checked},[marginIndices[Number(checkbox.dataset.index)]]);
    publishComparisonState();
  });
}
if(offsetSelector)offsetSelector.addEventListener('change',()=>{
  activeMargin=Number(offsetSelector.value);
  const visible=[];
  for(const input of document.querySelectorAll('.comparison-margin-toggle')){
    input.checked=data.margins_m[Number(input.dataset.index)]===activeMargin;
    visible.push(input.checked);
  }
  Plotly.restyle(graph,{visible},marginIndices);publishComparisonState();
});
function publishComparisonState(){
  const visible=[...document.querySelectorAll('.comparison-margin-toggle')]
    .filter(input=>input.checked).map(input=>data.margins_m[Number(input.dataset.index)]);
  activeMarginLabel.textContent=`${(100*activeMargin).toFixed(1).replace(/\.0$/,'')} cm`;
  if(offsetSelector)offsetSelector.value=String(activeMargin);
  const detail={pose_index:current,margin_m:activeMargin,visible_margins_m:visible,
                additional_visible_margins_m:visible.filter(m=>m!==activeMargin)};
  window.rebotComparisonState=detail;
  window.dispatchEvent(new CustomEvent('rebot-comparison-change',{detail}));
}
function updateDescription(index){
  const frame=data.frames[index];
  ownPose.value=String(index);
  poseLabel.textContent=`${index===0?'Initial pose · ':''}${frame.time_s.toFixed(2)} s · snapshot ${index+1}/${data.frames.length}`;
  const nearest=frame.nearest;
  const clearance=nearest.clearance_m===null?'not defined':`${(100*nearest.clearance_m).toFixed(2)} cm`;
  metrics.textContent=`Whole-arm minimum clearance: ${clearance}. `+
    `Lowest sampled H: ${frame.min_h_m2.toExponential(3)} m². Orange: the geometry used in this lesson.`;
  graph.dataset.poseIndex=String(index);
}
let pending=null,updating=false,current=0;
async function applyPose(index){
  index=Math.max(0,Math.min(data.frames.length-1,Math.round(Number(index))));
  if(!Number.isFinite(index))return;
  pending=index;
  if(updating)return;
  updating=true;
  try{
    while(pending!==null){
      const next=pending;pending=null;
      const frame=data.frames[next];
      const updates={x:[],y:[],z:[],i:[],j:[],k:[]};
      for(const item of dynamicMeshes){
        const mesh=item.kind==='arm'?{vertices:frame.arm_vertices,faces:data.arm_faces}:frame.shells[item.marginIndex];
        const payload=triangleCoordinates(mesh);
        for(const key of Object.keys(updates))updates[key].push(payload[key]);
      }
      await Plotly.restyle(graph,updates,dynamicMeshes.map(item=>item.index));
      const lines={x:[],y:[],z:[]};
      for(const item of dynamicLines){
        const payload=item.kind==='edges'?edges(frame.arm_vertices):coordinates(frame.nearest.points_m);
        for(const key of Object.keys(lines))lines[key].push(payload[key]);
      }
      await Plotly.restyle(graph,lines,dynamicLines.map(item=>item.index));
      current=next;updateDescription(next);
      publishComparisonState();
    }
  }finally{updating=false;}
}
if(nativePose)nativePose.addEventListener('input',()=>applyPose(nativePose.value));
ownPose.addEventListener('input',()=>{
  window.rebotComparisonPose(ownPose.value);
});
window.rebotSetComparisonPose=applyPose;
window.rebotComparisonPose=index=>{
  index=Math.max(0,Math.min(data.frames.length-1,Math.round(Number(index))));
  if(!Number.isFinite(index))return Promise.resolve();
  if(!nativePose)return applyPose(index);
  // The native control owns the detailed field/load panels too. Route public
  // example buttons and our slider through it, instead of moving only this
  // comparison. Resolve after our queued pose geometry has caught up.
  const finished=new Promise(resolve=>{
    const onChange=event=>{
      if(event.detail.pose_index!==index)return;
      window.removeEventListener('rebot-comparison-change',onChange);
      resolve(index);
    };
    window.addEventListener('rebot-comparison-change',onChange);
  });
  nativePose.value=String(index);
  nativePose.dispatchEvent(new Event('input',{bubbles:true}));
  return finished;
};
window.rebotComparisonGraph=graph;
async function renderGuidance(payload){
  if(!payload||Number(payload.pose_index)!==current)return;
  graph.dataset.guidance=JSON.stringify({stage:payload.stage,mode:payload.mode,body:payload.body,pose_index:payload.pose_index});
  const frame=data.frames[current],target=payload.mode==='distance'?'scene2':'scene';
  const point=payload.point_m,other=payload.obstacle_point_m;
  const valid=p=>Array.isArray(p)&&p.length===3&&p.every(Number.isFinite);
  const body=(data.body_names||[]).indexOf(payload.body),stage=Number(payload.stage);
  const direction=payload.direction_world||[0,0,0],length=Math.hypot(...direction);
  const end=valid(point)&&length>1e-12?point.map((p,i)=>p+.13*direction[i]/length):null;
  const updates={x:[],y:[],z:[],text:[]},indices=[];
  for(const item of derivationTraces.filter(t=>t.kind!=='cone')){
    let vertices=[],labels=[];
    if(item.kind==='outline'&&body>=0){
      for(const edge of data.arm_edges){
        if(Math.floor(edge[0]/8)!==body)continue;
        vertices.push(frame.arm_vertices[edge[0]],frame.arm_vertices[edge[1]],null);
      }
    }else if(item.kind==='points'&&item.scene===target&&valid(point)){
      vertices=[point];labels=[payload.mode==='distance'?'p_R':'pᵢ'];
      if(payload.mode==='distance'&&valid(other)){vertices.push(other);labels.push('p_O');}
      if(stage===1)labels[0]+=' · b = '+Number(payload.barrier_value).toExponential(2)+' '+(payload.barrier_units||'');
    }else if(item.kind==='direction'&&item.scene===target&&end&&stage>=2&&stage<=3){
      vertices=[point,end];labels=['',payload.mode==='distance'?'n':'∇H'];
    }
    for(const [axis,key]of ['x','y','z'].entries())updates[key].push(vertices.map(v=>v===null?null:v[axis]));
    updates.text.push(labels);indices.push(item.index);
  }
  for(const item of dynamicLines.filter(t=>t.kind==='nearest')){
    const vertices=payload.mode==='distance'&&valid(point)&&valid(other)?[point,other]:[];
    for(const [axis,key]of ['x','y','z'].entries())updates[key].push(vertices.map(v=>v[axis]));
    updates.text.push([]);indices.push(item.index);
  }
  await Plotly.restyle(graph,updates,indices);
  const cones=derivationTraces.filter(t=>t.kind==='cone'),c={x:[],y:[],z:[],u:[],v:[],w:[]};
  for(const item of cones){
    const shown=item.scene===target&&end&&stage>=2&&stage<=3;
    for(const [axis,key]of ['x','y','z'].entries())c[key].push(shown?[end[axis]]:[]);
    for(const [axis,key]of ['u','v','w'].entries())c[key].push(shown?[direction[axis]/length]:[]);
  }
  await Plotly.restyle(graph,c,cones.map(t=>t.index));
  const annotations=[];
  const labelAt=(p,text,ax,ay)=>({x:p[0],y:p[1],z:p[2],text,showarrow:true,arrowhead:2,
    ax,ay,arrowcolor:'#D55E00',arrowsize:1,arrowwidth:2,font:{size:15,color:'#8A3500'},
    bgcolor:'rgba(255,255,255,.96)',bordercolor:'#D55E00',borderpad:4});
  if(valid(point))annotations.push(labelAt(point,payload.mode==='distance'?'p_R · '+payload.body:'pᵢ · '+payload.body,-50,-45));
  if(payload.mode==='distance'&&valid(other))annotations.push(labelAt(other,'p_O · '+payload.obstacle,65,35));
  if(end&&stage>=2&&stage<=3)annotations.push(labelAt(end,payload.mode==='distance'?'n · outward normal':'∇H · increasing field',-55,25));
  await Plotly.relayout(graph,{'scene.annotations':target==='scene'?annotations:[],
    'scene2.annotations':target==='scene2'?annotations:[]});
  graph.dataset.guidanceRendered=String(stage);
  if(commandPayload)await renderCommandArrows(commandPayload);
}
async function renderCommandArrows(payload){
  commandPayload=payload;
  if(!graphReady||Number(payload.pose_index)!==current)return;
  const start=data.frames[current].tcp_m;
  const lineUpdates={x:[],y:[],z:[]},headUpdates={x:[],y:[],z:[],u:[],v:[],w:[]};
  for(const item of commandLines){
    const velocity=item.scene==='scene'?payload.poisson:payload.distance;
    const accepted=item.scene==='scene'?payload.poisson_success:payload.distance_success;
    const valid=(!guidancePayload||guidancePayload.stage===4)&&accepted!==false&&Array.isArray(velocity)&&velocity.length===3&&velocity.every(Number.isFinite);
    const end=valid?start.map((value,index)=>value+.25*velocity[index]):start;
    for(const [axis,key] of ['x','y','z'].entries())lineUpdates[key].push(valid?[start[axis],end[axis]]:[]);
    const magnitude=valid?Math.hypot(...velocity):0;
    for(const [axis,key] of ['x','y','z'].entries())headUpdates[key].push(magnitude>1e-8?[end[axis]]:[]);
    for(const [axis,key] of ['u','v','w'].entries())headUpdates[key].push(magnitude>1e-8?[velocity[axis]/magnitude]:[]);
  }
  await Plotly.restyle(graph,lineUpdates,commandLines.map(item=>item.index));
  await Plotly.restyle(graph,headUpdates,commandHeads.map(item=>item.index));
}
window.addEventListener('rebot-comparison-command',event=>{
  renderCommandArrows(event.detail).catch(error=>console.error('Comparison command arrow',error));
});
updateDescription(nativePose?Number(nativePose.value):0);
if(nativePose&&Number(nativePose.value)!==0)await applyPose(nativePose.value);
publishComparisonState();
if(guidancePayload)renderGuidance(guidancePayload);
status.textContent=`${data.frames.length} recorded poses · displayed commands are independent queries at each pose`;

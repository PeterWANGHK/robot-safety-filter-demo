// Reuse the verified local meshes/FK payload; no duplicate pose geometry.
(function(){'use strict';
const plot=document.getElementById('safety-field-3d'),panel=document.getElementById('environment-detail');
if(!plot||!panel)return;
let mounted=false,mounting=null,indices=[],lastPose=-1;
let dirty=false,updateLoop=null;
const value=id=>document.getElementById(id);
function points(mesh,t){return mesh.vertices.map(p=>t.rotation_world_from_link.map((row,i)=>row.reduce((s,r,j)=>s+r*p[j],t.translation_world_m[i])));}
async function update(){
 if(!panel.open)return;
 const model=(window.rebotComparisonData||{}).visual_model;
 if(!model||!model.meshes.length)return;
 const pose=Number(value('recorded-pose').value),frame=model.frames[pose];if(!frame)return;
 if(!mounted){
  if(mounting){await mounting;}else{
   const start=plot.data.length;
   mounting=Plotly.addTraces(plot,model.meshes.map((m,i)=>{const p=points(m,frame.transforms[i]);return {type:'mesh3d',x:p.map(v=>v[0]),y:p.map(v=>v[1]),z:p.map(v=>v[2]),i:m.faces.map(f=>f[0]),j:m.faces.map(f=>f[1]),k:m.faces.map(f=>f[2]),color:m.color,opacity:m.opacity,name:'Official URDF visual meshes',legendgroup:'official-visuals',showlegend:i===0,hovertemplate:m.body_name+' · '+m.material_name+'<extra>URDF visual mesh</extra>',flatshading:false,lighting:{ambient:.65,diffuse:.85,specular:.3,roughness:.45,fresnel:.08},lightposition:{x:2,y:-3,z:4}};})).then(()=>{indices=model.meshes.map((_,i)=>start+i);mounted=true;lastPose=pose;plot.dataset.officialVisualParts=String(indices.length);});
   await mounting;
  }
 }
 const selected=Number(value('recorded-pose').value),current=model.frames[selected];
 if(lastPose!==selected){const xyz=model.meshes.map((m,i)=>points(m,current.transforms[i]));await Plotly.restyle(plot,{x:xyz.map(p=>p.map(v=>v[0])),y:xyz.map(p=>p.map(v=>v[1])),z:xyz.map(p=>p.map(v=>v[2]))},indices);lastPose=selected;}
 const show=value('comparison-show-visual'),proxy=value('comparison-show-proxies'),opacity=value('comparison-mesh-opacity');
 await Plotly.restyle(plot,{visible:!show||show.checked,opacity:model.meshes.map(m=>m.opacity*(opacity?Number(opacity.value):1))},indices);
 const proxyIndex=plot.data.findIndex(t=>t.name==='Arm collision proxies');if(proxyIndex>=0)await Plotly.restyle(plot,{visible:!!proxy&&proxy.checked},[proxyIndex]);
 plot.dataset.visualPose=String(selected);
}
// Plotly restyles are asynchronous. Coalesce requests, then read the latest
// controls again after each completed update. A rapid return to the previous
// pose must not be skipped while another pose is still being rendered.
function safeUpdate(){
 dirty=true;
 if(updateLoop)return updateLoop;
 updateLoop=(async()=>{
  try{
   while(dirty){dirty=false;await update();}
  }catch(e){console.error('Detailed visual model',e);}
  finally{updateLoop=null;if(dirty)safeUpdate();}
 })();
 return updateLoop;
}
window.rebotUpdateDetailedVisuals=safeUpdate;
panel.addEventListener('toggle',safeUpdate);
window.addEventListener('rebot-comparison-change',safeUpdate);
['comparison-show-visual','comparison-show-proxies','comparison-mesh-opacity'].forEach(id=>{const e=value(id);if(e)e.addEventListener('input',safeUpdate);});
if(value('recorded-pose'))value('recorded-pose').addEventListener('input',safeUpdate);
safeUpdate();
})();

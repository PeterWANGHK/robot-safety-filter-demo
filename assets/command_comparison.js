(function(){
  'use strict';
  const payload='__COMMAND_COMPARISON_GZIP__';
  let data=null;
  const command=document.getElementById('comparison-command');
  const summary=document.getElementById('command-comparison-summary');
  const diagnostic=document.getElementById('comparison-command-diagnostics');
  const tbody=document.querySelector('#comparison-command-table tbody');
  function fixed(x,n=4){return Number.isFinite(x)?x.toFixed(n):'unavailable';}
  function update(){
    if(!data)return;
    const s=window.rebotComparisonState||{pose_index:Number(document.getElementById('recorded-pose')?.value||0),margin_m:.05};
    const frame=data.frames[Math.min(data.frames.length-1,Math.max(0,s.pose_index))];
    const m=data.margins_m.findIndex(x=>Math.abs(x-s.margin_m)<1e-9);
    const c=frame.commands[Number(command.value)||0], p=c.poisson, d=c.distance[m<0?0:m];
    tbody.replaceChildren();
    for(let j=0;j<6;j++){
      const row=document.createElement('tr');
      ['J'+(j+1),fixed(c.nominal_rad_s[j]),fixed(p.velocity_rad_s[j]),fixed(d.velocity_rad_s[j])].forEach(text=>{
        const cell=document.createElement('td');cell.textContent=text;cell.style.cssText='padding:6px 10px;border-bottom:1px solid #d9dee5;text-align:right';row.append(cell);
      });tbody.append(row);
    }
    summary.textContent='Pose '+s.pose_index+' · actual minimum proxy clearance '+fixed(frame.clearance_m,3)+' m · arm margin '+fixed(s.margin_m,3)+' m. '+
      'Command correction: Poisson '+fixed(p.correction_norm_rad_s,3)+'; distance '+fixed(d.correction_norm_rad_s,3)+' rad/s.';
    diagnostic.textContent='Poisson: '+p.status+' ('+p.all_rows+' body-sample constraints); distance: '+d.status+' ('+d.all_rows+' box pairs). '+
      'Solver-checked commands: '+(p.success&&d.success?'both accepted. ':'a query was rejected; its zero output is not a safe-braking certificate. ')+
      'Initial distance-margin condition: '+(d.initial_margin_feasible?'satisfied.':'violated; a recovery command does not establish invariance.');
    window.rebotComparisonCommands={pose_index:s.pose_index,margin_m:s.margin_m,nominal:c.nominal_tcp_m_s,poisson:p.tcp_velocity_m_s,distance:d.tcp_velocity_m_s};
    window.dispatchEvent(new CustomEvent('rebot-comparison-command',{detail:window.rebotComparisonCommands}));
  }
  async function start(){
    try{
      if(typeof DecompressionStream==='undefined')throw Error('This browser lacks DecompressionStream. Open in a recent Chromium or Firefox browser.');
      const bytes=Uint8Array.from(atob(payload),x=>x.charCodeAt(0));
      data=await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).json();
      data.command_names.forEach((name,i)=>{const option=document.createElement('option');option.value=i;option.textContent=name;command.append(option);});
      command.addEventListener('change',update);window.addEventListener('rebot-comparison-change',update);
      document.getElementById('comparison-example').addEventListener('click',()=>{
        const state=window.rebotComparisonState||{margin_m:.05};
        const mi=Math.max(0,data.margins_m.findIndex(x=>Math.abs(x-state.margin_m)<1e-9));
        let best={score:-1,pose:0,command:0};
        data.frames.forEach((f,pi)=>f.commands.forEach((c,ci)=>{
          const p=c.poisson,d=c.distance[mi];
          if(!p.success||!d.success)return;
          const score=Math.max(p.correction_norm_rad_s,d.correction_norm_rad_s);
          if(score>best.score)best={score,pose:pi,command:ci};
        }));
        command.value=best.command;
        if(window.rebotComparisonPose)window.rebotComparisonPose(best.pose);
        update();
      });
      update();window.rebotCommandComparisonReady=true;
    }catch(error){summary.textContent='Command comparison unavailable: '+error.message;console.error(error);}
  }
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',start,{once:true});else start();
}());

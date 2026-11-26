(function(){
  const settings=JSON.parse(document.getElementById('joint-layer-data').textContent);
  const data=settings.data,traces=settings.traces,plot=document.getElementById('safety-field-3d');
  const pose=document.getElementById('recorded-pose'),region=document.getElementById('body-region');
  const mass=document.getElementById('payload-mass'),budget=document.getElementById('torque-budget');
  const offset=document.getElementById('payload-com');
  const offsetAxis=document.getElementById('payload-com-axis');
  const number=(x,n=3)=>Math.abs(x)<.5*10**(-n)?(0).toFixed(n):x.toFixed(n);
  function update(){
    const index=pose?Number(pose.value):0,frame=data.frames[index],key=region.value;
    const m=Number(mass.value),fraction=Number(budget.value)/100,d=Number(offset.value)/100;
    const load=RebotLoadMath.evaluate(frame,data.model_caps_nm,m,fraction,d,Number(offsetAxis.value)),selected=frame.regions[key];
    document.getElementById('payload-value').textContent=m.toFixed(2)+' kg';
    document.getElementById('budget-value').textContent=budget.value+'%';
    document.getElementById('com-value').textContent=offset.value+' cm';
    const cap=load.capacity;
    let capacityText='no nonnegative mass satisfies this torque budget';
    if(cap.feasible)capacityText=cap.unbounded?'unbounded in this static torque model':
      `${cap.minimumKg.toFixed(2)}–${cap.maximumKg.toFixed(2)} kg`;
    const limiting=data.joint_labels[load.bottleneck];
    document.getElementById('load-summary').textContent=
      `Payload weight: ${(m*9.81).toFixed(2)} N. ${load.feasible?'Within':'Exceeds'} the selected static torque budget. `+
      `Most used: ${limiting}, ${(100*load.ratio[load.bottleneck]).toFixed(1)}%. `+
      `Static torque-only payload interval at this pose: ${capacityText}.`+
      (cap.feasible&&!cap.zeroFeasible?' This is a counterweight-only interval; the unloaded pose already exceeds budget.':'');
    document.getElementById('load-summary').className=load.feasible?'':'over-budget';
    document.getElementById('region-summary').textContent=
      `${data.regions[key]}: ${selected.sample_count} surface samples; minimum h=${selected.h_min_m2.toFixed(6)} m². `+
      `Whole-arm physical box clearance: ${(100*frame.whole_arm_clearance_m).toFixed(1)} cm (payload geometry excluded).`;
    const rows=document.getElementById('joint-load-rows');rows.replaceChildren();
    data.joint_labels.forEach((label,i)=>{
      const tr=document.createElement('tr');
      const values=[label,number(frame.q_rad[i]*180/Math.PI,1),number(frame.tau_robot_nm[i]),
                    number(load.payload[i]),number(load.total[i]),number(load.caps[i],2),
                    (100*load.ratio[i]).toFixed(1)+'%'];
      values.forEach((value,j)=>{const td=document.createElement('td');td.textContent=value;
        if(j===6&&load.ratio[i]>1)td.className='over-budget';tr.appendChild(td);});
      rows.appendChild(tr);
    });
    document.getElementById('critical-summary').textContent=
      `Critical sample ${selected.critical_sample} on ${selected.critical_body}; `+
      `world position (${selected.critical_point_m.map(v=>number(v)).join(', ')}) m. `+
      `The same environmental field is evaluated on every connected link surface.`+
      (selected.critical_body==='base_link'?' This point is on the fixed base, so its joint sensitivity is zero. Select elbow or wrist to inspect a moving link.':'');
    const sensitivity=document.getElementById('sensitivity-row');sensitivity.replaceChildren();
    ['∂H/∂q (m²/rad)',...selected.barrier_joint_sensitivity_m2_per_rad.map(v=>number(v,5))].forEach(value=>{
      const td=document.createElement('td');td.textContent=value;sensitivity.appendChild(td);
    });
    const points=selected.displayed_points_m;
    Plotly.restyle(plot,{x:[points.map(p=>p[0])],y:[points.map(p=>p[1])],z:[points.map(p=>p[2])],
                         'marker.color':[selected.displayed_h_m2],customdata:[selected.displayed_h_m2]},[traces.body]);
    const selectedBodies=data.region_bodies[key];
    const faceColors=data.proxy_bodies.flatMap(body=>Array(12).fill(
      !selectedBodies||selectedBodies.includes(body)?'#374553':'#b7c1cb'));
    Plotly.restyle(plot,{facecolor:[faceColors]},[traces.arm]);
    Plotly.restyle(plot,{x:[[selected.critical_point_m[0]]],y:[[selected.critical_point_m[1]]],
                         z:[[selected.critical_point_m[2]]]},[traces.critical]);
    const colors=load.ratio.map(v=>v>1?'#C44E36':v>.8?'#CC9900':'#228833');
    const hover=data.joint_labels.map((label,i)=>`${label}<br>Static hold: ${number(load.total[i])} N·m`+
      `<br>Budget used: ${(100*load.ratio[i]).toFixed(1)}%<br>World axis: `+
      frame.joint_axes_world[i].map(v=>number(v)).join(', '));
    Plotly.restyle(plot,{'marker.color':[colors],x:[frame.joint_origins_m.map(p=>p[0])],
                         y:[frame.joint_origins_m.map(p=>p[1])],z:[frame.joint_origins_m.map(p=>p[2])],
                         hovertext:[hover]},[traces.joints]);
    const p=load.com;
    Plotly.restyle(plot,{x:[[p[0],p[0]]],y:[[p[1],p[1]]],z:[[p[2],p[2]-.10]],
                         text:[[`Payload COM<br>${(m*9.81).toFixed(2)} N`, '']],
                         visible:m>0},[traces.weight]);
  }
  [region,mass,budget,offset,offsetAxis].forEach(control=>control.addEventListener('input',update));
  window.rebotUpdateJointLayer=update;
  update();
})();

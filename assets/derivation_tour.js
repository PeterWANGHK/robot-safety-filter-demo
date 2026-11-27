(function(){
  'use strict';
  const packed='__DERIVATION_TOUR_GZIP__';
  let data=null,equations=null,scheduled=false;
  const state={stage:0,mode:'poisson',region:'elbow',pose_index:0,margin_m:.05};
  const root=document.getElementById('derivation-tour');
  if(!root)return;
  const explanation=document.getElementById('derivation-explanation');
  const context=document.getElementById('derivation-context');
  const equation=document.getElementById('derivation-equation');
  const numbers=document.getElementById('derivation-numbers');
  const symbols=document.getElementById('derivation-symbols');
  const region=document.getElementById('derivation-region');
  const previous=document.getElementById('derivation-previous');
  const next=document.getElementById('derivation-next');
  const scope=document.getElementById('derivation-scope');
  const finite=Number.isFinite;
  const dot=(a,b)=>a.reduce((sum,value,index)=>sum+value*b[index],0);
  const vector6=value=>Array.isArray(value)&&value.length===6&&value.every(finite);
  const number=value=>!finite(value)?'unavailable':value===0?'0':
    (Math.abs(value)<.001?value.toExponential(2):value.toFixed(4).replace(/0+$/,'').replace(/\.$/,''));
  const vector=value=>Array.isArray(value)?'('+value.map(number).join(', ')+')':'unavailable';
  function appendNumber(label,value){
    const row=document.createElement('div');row.className='dt-number';
    const key=document.createElement('span');key.className='dt-number-label';key.textContent=label;
    const text=document.createElement('span');text.textContent=value;
    row.append(key,text);numbers.append(row);
  }
  function result(text,kind=''){
    const box=document.createElement('div');box.className='dt-result'+(kind?' dt-'+kind:'');box.textContent=text;numbers.append(box);
  }
  function equationAt(mode,stage){
    const item=equations[mode][stage];equation.dataset.tex=item.tex;equation.textContent=item.fallback;
    if(window.katex?.render){
      try{window.katex.render(item.tex,equation,{displayMode:true,throwOnError:true,trust:false,output:'htmlAndMathml'});}
      catch(error){equation.textContent=item.fallback;console.error('Derivation equation',error);}
    }
  }
  function drawSymbols(mode,stage){
    const jointBounds=['ℓ / u','Lower/upper joint-velocity bounds combine ±0.45 rad/s speed limits with joint-position limits and a 0.015 rad joint margin.'];
    const definitions=stage===4?[
      ['vnom','Requested six-joint velocity before safety filtering, in rad/s.'],
      ['v*','Accepted six-joint velocity closest to the request, in rad/s; this display does not execute it.'],
      ['Aall','All original geometry-derivative rows: 798 body-sample rows or 50 whole-box pairs.'],
      ['βall','Right sides −αb for every original geometry row. Min row residual is minᵢ(Aᵢv*+αbᵢ), excluding joint bounds.'],jointBounds
    ]:stage===3?[
      ['α','Barrier decay rate: 2 s⁻¹ for these cached projections.'],
      ['v','Six commanded joint velocities, in rad/s.'],
      ['A','The selected barrier’s derivative with respect to the six joint angles; A·v is its instantaneous rate.'],
      ['b','Selected remaining barrier margin: H−H₀ in m², or d−δ in m.']
    ]:mode==='poisson'?[
      [['O / c / s','Occupied region O is the union of the five red axis-aligned obstacle boxes. Each c is its world centre and each s its world-axis half-sizes, in metres.'],['p / r','p is the chosen surface point in world metres; r is its fixed location in the selected link frame, in metres.'],['R / t','R rotates the link frame into world coordinates; t is that frame’s position in world metres.'],['q','Six joint angles in radians.']],
      [['H','Environment Poisson scalar, in m²; it is not distance or force.'],['Δ','Laplacian: sum of the three second spatial derivatives. A finite-difference grid solves ΔH=−1.'],['Ω','Free voxel domain; its boundary includes inflated obstacles and the outer rectangular wall.'],['H₀','Small Poisson reserve: 0.000020 m² for the current controller.'],['b','Remaining Poisson margin H−H₀, in m².']],
      [['∇H','Spatial derivative of H, in metres; this geometric direction is not a force.'],['J','Point Jacobian in m/rad; converts joint velocity into point velocity.'],['A','Joint derivative ∇HᵀJ, in m²/rad.'],['ṗ / ḃ','Time derivatives of world point position (m/s) and barrier value (m²/s).']]
    ][stage]:[
      [['pR','Closest robot-box witness point, in world metres; changes with pose.'],['pO','Closest obstacle-box witness, in world metres; can move along a static obstacle as the robot pose changes.'],['d','Positive minimum separation between the selected two full boxes, in metres.'],['q','Six joint angles in radians.']],
      [['d','Positive robot-box to obstacle-box separation, in metres.'],['δ','Selected required separation margin, in metres.'],['b','Remaining distance margin d−δ, in metres.']],
      [['n','Unit normal from obstacle to robot; not a force. The derivative applies locally at positive separation and differentiable closest features.'],['JR','Rigid-body point Jacobian at the current robot witness, in m/rad. Closest-feature switches can make the distance derivative nonsmooth.'],['A','Joint derivative nᵀJR, in m/rad.'],['ḃ','Instantaneous change of the distance margin, in m/s.']]
    ][stage];
    symbols.replaceChildren();
    for(const [label,title] of definitions){
      const item=document.createElement('abbr');item.textContent=label;item.title=title;item.tabIndex=0;symbols.append(item);
    }
  }
  function selectedRow(frame){
    const selection=frame.regions[state.region]||frame.regions.whole;
    const reserve=data.controller.poisson_h_reserve_m2;
    if(state.mode==='poisson'){
      const point=selection.poisson;
      return {point_m:point.point_world_m,body:point.body_name,a:point.a_m2_per_rad,
        direction_world:point.gradient_world_m,raw:point.h_m2,b:point.h_m2-reserve,
        valid:point.field_valid,units:'m²',rate_units:'m²/s',a_units:'m²/rad',
        count:frame.poisson_controller_row_count,index:point.sample_index,
        selection:point.selection,reserve};
    }
    const index=selection.pair_index,pairs=frame.pairs;
    if(index===null||index===undefined)return null;
    return {point_m:pairs.robot_witness_world_m[index],obstacle_point_m:pairs.obstacle_witness_world_m[index],
      body:pairs.robot_bodies[index],obstacle:pairs.obstacle_names[index],a:pairs.gradients_q[index],
      direction_world:pairs.normal_obstacle_to_robot_world[index],raw:pairs.distances_m[index],
      b:pairs.distances_m[index]-state.margin_m,valid:pairs.pair_valid[index],
      units:'m',rate_units:'m/s',a_units:'m/rad',count:frame.pair_row_count,index};
  }
  function currentCommand(frame){
    const command=window.rebotComparisonCommands;
    if(!command||Number(command.pose_index)!==state.pose_index)return null;
    if(state.mode==='distance'&&Math.abs(Number(command.margin_m)-state.margin_m)>1e-9)return null;
    const index=Number.isInteger(command.command_index)?command.command_index:
      Number(document.getElementById('comparison-command')?.value||0);
    const residuals=frame.command_residuals?.[index];
    const marginIndex=data.margins_m.findIndex(value=>Math.abs(value-state.margin_m)<1e-9);
    const nominal=command.nominal_rad_s;
    const filtered=state.mode==='poisson'?command.poisson_rad_s:command.distance_rad_s;
    const accepted=state.mode==='poisson'?
      (command.poisson_success??residuals?.poisson_command_accepted):
      (command.distance_success??residuals?.distance_commands_accepted?.[marginIndex]);
    const minimum=state.mode==='poisson'?residuals?.poisson_min_m2_s:residuals?.distance_min_m_s?.[marginIndex];
    const nominalMinimum=state.mode==='poisson'?residuals?.nominal_poisson_min_m2_s:residuals?.nominal_pair_min_m_s?.[marginIndex];
    return {index,nominal,filtered,accepted,minimum,nominalMinimum};
  }
  function emitOverlay(row){
    const label=state.stage===0?'Chosen '+row.body+' surface':state.mode==='poisson'?'H sample / ∇H direction':'Closest pair / unit normal n';
    const detail={stage:state.stage,mode:state.mode,region:state.region,pose_index:state.pose_index,
      point_m:row.point_m,direction_world:row.direction_world,body:row.body,label,
      barrier_value:row.b,barrier_units:row.units,raw_field_value:row.raw,
      selected_row_index:row.index,margin_m:state.margin_m};
    if(row.obstacle_point_m)detail.obstacle_point_m=row.obstacle_point_m;
    if(row.obstacle)detail.obstacle=row.obstacle;
    window.dispatchEvent(new CustomEvent('rebot-derivation-step',{detail}));
  }
  function render(){
    if(!data)return;
    const shared=window.rebotComparisonState||state;
    state.pose_index=Math.max(0,Math.min(data.frames.length-1,Number(shared.pose_index)||0));
    state.margin_m=finite(shared.margin_m)?shared.margin_m:state.margin_m;
    const frame=data.frames[state.pose_index],row=selectedRow(frame);
    if(!row){explanation.textContent='No valid geometric row is available for this region.';return;}
    const command=currentCommand(frame),alpha=data.controller.alpha_per_s,rhs=-alpha*row.b;
    const nominalLhs=command&&vector6(command.nominal)?dot(row.a,command.nominal):null;
    const filteredLhs=command?.accepted&&vector6(command.filtered)?dot(row.a,command.filtered):null;
    const tolerance=2e-6*Math.max(Math.hypot(...row.a),1e-3);
    context.textContent=state.mode==='poisson'?`${row.body} · one of ${row.count} surface rows`:
      `${row.body} ↔ ${row.obstacle} · one of ${row.count} box pairs`;
    context.title=state.mode==='poisson'?
      'This display chooses the smallest positive valid H among moving samples in the selected region when available, not necessarily the global minimum or active QP row. All original rows remain checked.':
      'This display chooses the closest full-box pair in the selected region. Its obstacle witness may change with pose. All original pairs remain checked.';
    document.getElementById('derivation-count').textContent=`${state.stage+1} / 5`;
    for(const button of root.querySelectorAll('[data-derivation-stage]')){
      if(Number(button.dataset.derivationStage)===state.stage)button.setAttribute('aria-current','step');
      else button.removeAttribute('aria-current');
    }
    for(const button of root.querySelectorAll('[data-derivation-mode]'))
      button.setAttribute('aria-pressed',String(button.dataset.derivationMode===state.mode));
    previous.disabled=state.stage===0;next.disabled=state.stage===4;
    equationAt(state.mode,state.stage);numbers.replaceChildren();
    if(state.stage===0){
      explanation.textContent=state.mode==='poisson'?
        'Red boxes form the occupied region; the orange point follows the selected arm link.':
        'Follow the shortest segment between these two full boxes. Its endpoints are the robot and obstacle witnesses.';
      appendNumber(state.mode==='poisson'?'p =':'pR =',vector(row.point_m)+' m');
      if(state.mode==='distance')appendNumber('pO =',vector(row.obstacle_point_m)+' m');
      else appendNumber('q =',vector(frame.q_rad)+' rad');
    }else if(state.stage===1){
      explanation.textContent=state.mode==='poisson'?
        'Solve the Poisson equation in free space, with zero values on obstacles and the outer boundary. Subtract the reserve at this sample.':
        'Subtract the selected margin from this pair’s gap. Positive b means this pair meets that margin.';
      const barrierNumber=value=>state.mode==='poisson'&&finite(value)?value.toFixed(6):number(value);
      appendNumber(state.mode==='poisson'?'H =':'d =',barrierNumber(row.raw)+' '+row.units);
      appendNumber(state.mode==='poisson'?'H₀ =':'δ =',barrierNumber(state.mode==='poisson'?row.reserve:state.margin_m)+' '+row.units);
      appendNumber('b =',state.mode==='poisson'?barrierNumber(row.raw)+' − '+barrierNumber(row.reserve)+' = '+barrierNumber(row.b)+' '+row.units:barrierNumber(row.b)+' '+row.units);
      result(!row.valid?'Geometry is invalid for this derivative.':row.b>=0?'Selected barrier is nonnegative.':'Selected barrier is negative; initial margin is not met.',!row.valid?'unavailable':row.b>=0?'':'correction');
    }else if(state.stage===2){
      explanation.textContent=state.mode==='poisson'?
        'The Jacobian maps joint motion to point motion. The field gradient converts that motion into a change in H.':
        'The normal points from obstacle to robot. Projecting the witness Jacobian onto it gives each joint’s effect on the gap.';
      appendNumber(state.mode==='poisson'?'∇H =':'n =',vector(row.direction_world)+(state.mode==='poisson'?' m':' (unit vector)'));
      appendNumber('A =',vector(row.a)+' '+row.a_units);
    }else if(state.stage===3){
      explanation.textContent='This inequality limits how fast the selected barrier may decrease. It is only one row; other rows can still require a correction.';
      appendNumber('α =',number(alpha)+' s⁻¹');
      appendNumber('Required −αb =',number(rhs)+' '+row.rate_units);
      appendNumber('Nominal A·v =',number(nominalLhs)+' '+row.rate_units);
      if(finite(nominalLhs)){
        const passes=nominalLhs-rhs>=-tolerance;
        const allTolerance=state.mode==='poisson'?frame.numeric_tolerance?.poisson_raw_min_m2_s:frame.numeric_tolerance?.pair_raw_min_m_s;
        const otherNeedsCorrection=passes&&finite(command?.nominalMinimum)&&command.nominalMinimum<-(allTolerance??tolerance);
        result(!passes?'Requested motion shrinks this margin too quickly.':otherNeedsCorrection?
          (state.mode==='poisson'?'This point passes; another body point needs correction.':'This pair passes; another box pair needs correction.'):
          'This selected row permits the requested motion.',!passes||otherNeedsCorrection?'correction':'');
      }
      else result('Choose a cached nominal command above the graph.','unavailable');
    }else{
      explanation.textContent='The QP finds the closest joint velocity satisfying all rows and joint bounds. The numbers below evaluate its cached result at this frozen pose.';
      appendNumber('v* =',command?.accepted&&vector6(command.filtered)?vector(command.filtered)+' rad/s':'no accepted command');
      appendNumber('Selected A·v* ≥ −αb:',finite(filteredLhs)?number(filteredLhs)+' ≥ '+number(rhs)+' '+row.rate_units:'unavailable');
      appendNumber('Min '+row.count+' row residuals:',number(command?.minimum)+' '+row.rate_units);
      if(command?.accepted){
        const rawTolerance=state.mode==='poisson'?frame.numeric_tolerance?.poisson_raw_min_m2_s:frame.numeric_tolerance?.pair_raw_min_m_s;
        if(!finite(command.minimum))result('Cached solver accepted; raw-row residual is unavailable.','unavailable');
        else if(command.minimum<0&&(!finite(rawTolerance)||command.minimum < -rawTolerance))
          result('Raw residual exceeds the displayed tolerance bound; this result is unverified.','unavailable');
        else result(command.minimum<0?'Accepted numerical result; negative residual is within the numerical tolerance bound.':'Accepted numerical result; joint bounds checked separately.');
      }
      else result('This query was not accepted; a zero output is not a braking guarantee.','unavailable');
    }
    drawSymbols(state.mode,state.stage);
    scope.textContent='Frozen recorded pose · selected row explains; all '+row.count+' rows remain checked · not forces.';
    window.rebotDerivationState={...state,selected_row:{body:row.body,barrier_value:row.b,a:row.a,
      units:row.units,rhs,nominal_lhs:nominalLhs,filtered_lhs:filteredLhs,all_row_minimum:command?.minimum??null,
      accepted:command?.accepted??false}};
    emitOverlay(row);
  }
  function schedule(){
    if(scheduled)return;scheduled=true;
    Promise.resolve().then(()=>{scheduled=false;render();});
  }
  async function start(){
    try{
      if(typeof DecompressionStream==='undefined')throw Error('Open in a recent browser with offline gzip support.');
      const bytes=Uint8Array.from(atob(packed),value=>value.charCodeAt(0));
      const payload=await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).json();
      data=payload.diagnostics;equations=payload.equations;
      state.region=data.defaults?.region||'elbow';region.value=state.region;
      state.margin_m=data.defaults?.margin_m||.05;
      for(const button of root.querySelectorAll('[data-derivation-stage]'))
        button.addEventListener('click',()=>{state.stage=Number(button.dataset.derivationStage);render();});
      for(const button of root.querySelectorAll('[data-derivation-mode]'))
        button.addEventListener('click',()=>{state.mode=button.dataset.derivationMode;render();});
      previous.addEventListener('click',()=>{state.stage=Math.max(0,state.stage-1);render();});
      next.addEventListener('click',()=>{state.stage=Math.min(4,state.stage+1);render();});
      region.addEventListener('change',()=>{state.region=region.value;render();});
      window.addEventListener('rebot-comparison-change',schedule);
      window.addEventListener('rebot-comparison-command',schedule);
      window.rebotSetDerivationStage=stage=>{state.stage=Math.max(0,Math.min(4,Math.round(Number(stage))||0));render();};
      window.rebotSetDerivationMode=mode=>{if(mode==='poisson'||mode==='distance'){state.mode=mode;render();}};
      render();window.rebotDerivationReady=true;
    }catch(error){explanation.textContent='The guided derivation could not load: '+error.message;console.error('Derivation tour',error);}
  }
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',start,{once:true});else start();
}());

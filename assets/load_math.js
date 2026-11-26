/* Affine inverse-statics only. No controller commands or probability model. */
(function(root){
  function interval(gravity, perKg, caps){
    let lower=0, upper=Infinity, upperJoint=null, lowerJoint=null, impossible=false;
    const zeroFeasible=gravity.every((g,i)=>Math.abs(g)<=caps[i]);
    gravity.forEach((g,i)=>{
      const a=perKg[i], cap=caps[i];
      if(a===0){if(Math.abs(g)>cap)impossible=true;return;}
      const x=(-cap-g)/a,y=(cap-g)/a,lo=Math.min(x,y),hi=Math.max(x,y);
      if(lo>lower){lower=lo;lowerJoint=i;}
      if(hi<upper){upper=hi;upperJoint=i;}
    });
    const feasible=!impossible && upper>=lower;
    return {feasible,minimumKg:feasible?lower:null,maximumKg:feasible&&Number.isFinite(upper)?upper:null,
            unbounded:feasible&&!Number.isFinite(upper),zeroFeasible,upperJoint,lowerJoint};
  }
  function evaluate(frame, modelCaps, mass, fraction, offset, axis=2){
    const perKg=frame.per_kg_tcp_nm.map((v,i)=>v+offset*frame.per_kg_per_m_tool_xyz_nm[axis][i]);
    const payload=perKg.map(v=>mass*v),total=payload.map((v,i)=>v+frame.tau_robot_nm[i]);
    const caps=modelCaps.map(v=>fraction*v),ratio=total.map((v,i)=>Math.abs(v)/caps[i]);
    const bottleneck=ratio.indexOf(Math.max(...ratio));
    return {perKg,payload,total,caps,ratio,bottleneck,feasible:ratio.every(v=>v<=1),
            capacity:interval(frame.tau_robot_nm,perKg,caps),
            com:frame.tcp_m.map((v,i)=>v+offset*frame.tool_axes_world[axis][i])};
  }
  const api={interval,evaluate};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
  else root.RebotLoadMath=api;
})(typeof window==='undefined'?globalThis:window);

(function(){'use strict';
const data=JSON.parse(document.getElementById('roadmap-data').textContent);
const byId=id=>document.getElementById(id);
function stage(index){
 const s=data.stages[index];
 byId('stage-label').textContent=s.label;byId('stage-goal').textContent=s.goal;
 byId('stage-equation').dataset.tex=s.tex;
 if(window.katex)window.katex.render(s.tex,byId('stage-equation'),{displayMode:true,throwOnError:true,trust:false});
 byId('stage-symbols').textContent=s.symbols;byId('stage-practice').textContent=s.practice;byId('stage-check').textContent=s.check;
 byId('stage-source').textContent=s.source_label;byId('stage-source').href=s.source;
 document.querySelectorAll('#pipeline-steps button').forEach((b,i)=>b.setAttribute('aria-pressed',String(i===index)));
 byId('stage-card').dataset.stage=String(index);
}
data.stages.forEach((s,i)=>{const b=document.createElement('button');b.type='button';b.textContent=s.name;b.setAttribute('aria-pressed','false');b.addEventListener('click',()=>stage(i));byId('pipeline-steps').append(b);});
function dimensions(){const n=Number(byId('state-robot').value);byId('state-count').textContent=n===6?'6 joint angles + 6 joint velocities = 12 state variables':'9 configuration + 9 velocity coordinates, before wheel constraints';}
byId('state-robot').addEventListener('change',dimensions);dimensions();
function val(record,...keys){for(const k of keys){if(record[k]!==undefined&&record[k]!==null)return typeof record[k]==='string'?record[k]:Array.isArray(record[k])?record[k].join('; '):JSON.stringify(record[k]);}return '';}
const methodRoles={voxblox:['mapping'],nvblox:['mapping'],rdf:['mapping'],ramp:['planning'],ncsb:['mapping','filter','planning'],safer_splat:['mapping','filter'],oscbf:['filter'],rm4d:['reachability']};
function role(record){return methodRoles[record.id]||record.roles||[];}
function addText(parent,tag,text,cls){if(!text)return;const e=document.createElement(tag);e.textContent=text;if(cls)e.className=cls;parent.append(e);return e;}
function link(parent,label,url){if(!/^https:\/\//.test(url))return;const a=document.createElement('a');a.textContent=label;a.href=url;parent.append(a);}
function render(){
 const search=byId('paper-search').value.trim().toLowerCase(),selected=byId('paper-role').value;
 const list=data.methods.filter(p=>(!search||JSON.stringify(p).toLowerCase().includes(search))&&(selected==='all'||role(p).includes(selected)));
 byId('paper-count').textContent=list.length+' / '+data.methods.length+' methods';const host=byId('method-cards');host.replaceChildren();
 for(const p of list){const a=document.createElement('article');addText(a,'p',[val(p,'venue','publication'),val(p,'year')].filter(Boolean).join(' · '),'tag');addText(a,'h3',val(p,'title','name'));
 const dl=document.createElement('dl');for(const [label,keys]of [['Representation',['representation','method']],['Input → output',['input_output']],['Input',['input']],['Output',['output']],['Use for B601',['applicability','rebot_applicability','relevance']]]){const value=val(p,...keys);if(value){addText(dl,'dt',label);addText(dl,'dd',value);}}a.append(dl);
 addText(a,'p',val(p,'limitations','limitation'),'limit');
 const links=document.createElement('p');link(links,'Paper',val(p,'paper_url','paper'));const code=val(p,'code_url','code');if(code){links.append(' · ');link(links,'Source code',code);}a.append(links);
 addText(a,'small',val(p,'license','code_status'));
 const details=document.createElement('details');addText(details,'summary','Validation and portability');addText(details,'p',val(p,'publication_status'));addText(details,'p',val(p,'validation'));addText(details,'p',val(p,'jetson_evidence'));link(details,'Licence source',val(p,'license_url'));a.append(details);host.append(a);}
 if(!list.length)addText(host,'p','No match. Try another role or a broader term.','empty');
}
byId('paper-search').addEventListener('input',render);byId('paper-role').addEventListener('change',render);
stage(0);render();
})();

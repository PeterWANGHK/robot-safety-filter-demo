"""Execute the detailed-view adapter against an asynchronous Plotly double.

This checks observable final geometry and control state, including a delayed
render followed by a return to the old pose. It advances no simulation.
"""
from pathlib import Path
import shutil
import subprocess
import unittest


class DetailedVisualRuntimeTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Node.js is needed for the browser adapter harness')
    def test_world_transform_and_latest_pose_win_during_pending_plotly_update(self):
        source = Path(__file__).resolve().parents[1]/'assets'/'detailed_visual.js'
        harness = r"""
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const elements={};
function element(id,extra={}){
  const e={dataset:{},listeners:{},addEventListener(name,handler){
    (this.listeners[name]??=[]).push(handler);
  },...extra};
  elements[id]=e;return e;
}
const plot=element('safety-field-3d',{data:[{name:'Arm collision proxies',visible:true}]});
const panel=element('environment-detail',{open:false});
const slider=element('recorded-pose',{value:'0'});
const visual=element('comparison-show-visual',{checked:true});
const proxy=element('comparison-show-proxies',{checked:false});
const opacity=element('comparison-mesh-opacity',{value:'.6'});
const mesh={vertices:[[1,2,3],[0,0,0],[1,0,0]],faces:[[0,1,2]],color:'#abcdef',opacity:.8,
            body_name:'test_link',material_name:'test_material'};
const frames=[0,1,2].map(pose=>({transforms:[{
  rotation_world_from_link:[[0,-1,0],[1,0,0],[0,0,1]],translation_world_m:[pose,10,20]
}]}));
const errors=[];
const window={rebotComparisonData:{visual_model:{meshes:[mesh],frames}},addEventListener(){}};
let addCalls=0,delayNextGeometry=false,releaseGeometry=null;
const clone=value=>JSON.parse(JSON.stringify(value));
const Plotly={
  async addTraces(target,traces){addCalls++;target.data.push(...clone(traces));},
  restyle(target,updates,indices){
    const apply=()=>{
      for(const [position,index]of indices.entries()){
        for(const [key,value]of Object.entries(updates)){
          target.data[index][key]=clone(Array.isArray(value)?value[position]:value);
        }
      }
    };
    if(updates.x&&delayNextGeometry){
      delayNextGeometry=false;
      return new Promise(resolve=>{releaseGeometry=()=>{apply();resolve();};});
    }
    apply();return Promise.resolve();
  }
};
vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'),{
  document:{getElementById:id=>elements[id]},window,Plotly,
  console:{error:(...args)=>errors.push(args.map(String).join(' '))}
});
(async()=>{
  await window.rebotUpdateDetailedVisuals();
  assert.equal(addCalls,0,'closed detail panel must not tessellate/mount visuals');
  panel.open=true;
  await window.rebotUpdateDetailedVisuals();
  assert.equal(addCalls,1);
  assert.deepEqual(plot.data[1].x,[-2,0,0]);
  assert.deepEqual(plot.data[1].y,[11,10,11]);
  assert.deepEqual(plot.data[1].z,[23,20,20]);
  assert.equal(plot.data[1].color,'#abcdef');
  assert.equal(plot.data[1].opacity,.48);
  assert.equal(plot.data[0].visible,false);

  // The second request returns to the last completed pose while pose 2's
  // restyle is pending. The final rendered coordinates must follow the slider.
  delayNextGeometry=true;slider.value='2';
  const first=window.rebotUpdateDetailedVisuals();
  assert.equal(typeof releaseGeometry,'function');
  slider.value='0';
  const second=window.rebotUpdateDetailedVisuals();
  visual.checked=false;proxy.checked=true;opacity.value='.5';
  releaseGeometry();
  await Promise.all([first,second]);
  assert.deepEqual(plot.data[1].x,[-2,0,0],'a pending stale pose must not win');
  assert.equal(plot.dataset.visualPose,'0');
  assert.equal(plot.data[1].visible,false);
  assert.equal(plot.data[1].opacity,.4);
  assert.equal(plot.data[0].visible,true);
  assert.equal(addCalls,1,'coalesced requests must not mount duplicate traces');
  assert.deepEqual(errors,[]);
  console.log('detailed visual runtime: world transform, deferred reverse pose, controls passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
        result = subprocess.run(['node', '-e', harness, str(source)], capture_output=True,
                                text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn('controls passed', result.stdout)


if __name__ == '__main__':
    unittest.main()

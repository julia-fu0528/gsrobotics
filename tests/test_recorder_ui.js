const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function setup(initialState) {
  const elements = new Map();
  const context2d = {scale(){},fillRect(){}};
  function element(id) {
    if (!elements.has(id)) elements.set(id, {
      value:'test session', hidden:false, disabled:false, textContent:'',
      checked:false, children:[], replaceChildren(){this.children=[];this.value='';}, appendChild(child){this.children.push(child);},
      classList:{toggle(){}}, setAttribute(){}, addEventListener(){},
      getContext(){return context2d;},
      getBoundingClientRect(){return {width:500,height:300};}
    });
    return elements.get(id);
  }
  const calls = [];
  let state = initialState, failure = false, scheduled;
  const context = vm.createContext({
    document:{getElementById:element,createElement(){return {value:'',textContent:''};}},
    window:{devicePixelRatio:1,addEventListener(){}},
    setTimeout(fn){scheduled=fn;},
    async fetch(url, options) {
      calls.push({url,options});
      if (failure) throw new Error('Network disconnected');
      return {ok:true,json:async()=>url==='/api/state' ? state : url==='/api/cameras' ? {cameras:[{device:'/dev/v4l/by-id/mini',path:'/dev/video4',label:'GelSight Mini',gelsight:true},{device:'/dev/video0',path:'/dev/video0',label:'Webcam',gelsight:false}],demo_device:'__demo__'} : {ok:true}};
    }
  });
  vm.runInContext(fs.readFileSync('tactile_recorder/ui/app.js','utf8'),context);
  return {element,calls,setState(value){state=value;},disconnect(){failure=true;},async poll(){await scheduled();}};
}
const settle = () => new Promise(resolve=>setImmediate(resolve));
const ready = {ready:true,connected:true,recording:false,mode:'Demo · synthetic data',fps:5,
  source_frame:7,saved_frames:0,calibration_frames:2,calibration_total:2,depth_available:true,
  preview:{images:{raw:'raw',deformation:'difference',gradient:'gradient',hog:'hog',depth_preview:'depth'},points:[[0,0,0]]}};

test('camera picker connects selected source, sets suitable depth default, and locks during recording',async()=>{
  const ui=setup({...ready,source_device:'/dev/video4',depth_enabled:true}); await settle();
  assert.equal(ui.element('camera').value,'/dev/v4l/by-id/mini');
  assert.equal(ui.element('cameraDepth').checked,true);
  ui.element('camera').value='/dev/video0'; ui.element('camera').onchange();
  assert.equal(ui.element('cameraDepth').checked,false);
  await ui.element('connectCamera').onclick();
  assert.equal(ui.calls.at(-1).url,'/api/source');
  assert.deepEqual(JSON.parse(ui.calls.at(-1).options.body),{device:'/dev/video0',depth_enabled:false});
  ui.setState({...ready,recording:true}); await ui.poll();
  assert.equal(ui.element('camera').disabled,true);
  assert.equal(ui.element('connectCamera').disabled,true);
});

test('calibration, start/stop controls, final count and downloads follow server state',async()=>{
  const ui=setup({...ready,ready:false,depth_available:false});
  await settle();
  assert.equal(ui.element('record').disabled,true);
  assert.match(ui.element('status').textContent,/Keep the sensor untouched/);
  ui.setState(ready); await ui.poll();
  assert.equal(ui.element('record').disabled,false);
  assert.equal(ui.element('raw').hidden,false);
  assert.equal(ui.element('deformation').hidden,false);
  assert.equal(ui.element('deformation').src,'data:image/jpeg;base64,difference');
  await ui.element('record').onclick();
  assert.equal(ui.calls.at(-1).url,'/api/start');
  assert.equal(JSON.parse(ui.calls.at(-1).options.body).label,'test session');
  ui.setState({...ready,recording:true,saved_frames:6}); await ui.poll();
  assert.equal(ui.element('stop').disabled,false);
  assert.equal(ui.element('calibrate').disabled,true);
  assert.equal(ui.element('record').disabled,true);
  ui.setState({...ready,saved_frames:7}); await ui.poll();
  assert.equal(ui.element('count').textContent,7);
  assert.equal(ui.element('cloudDownload').tabIndex,0);
});

test('connection loss hides stale frames and disables recording and downloads',async()=>{
  const ui=setup(ready); await settle();
  ui.disconnect(); await ui.poll();
  assert.equal(ui.element('raw').hidden,true);
  assert.equal(ui.element('deformation').hidden,true);
  assert.equal(ui.element('record').disabled,true);
  assert.equal(ui.element('cloudDownload').tabIndex,-1);
  assert.match(ui.element('status').textContent,/Recording status is unknown/);
  assert.equal(ui.element('error').hidden,false);
});

const $ = id => document.getElementById(id);
let pending = false, lastState = null;
let cameras = [], refreshingCameras = false, cameraListTime = 0, cameraSelectionInitialized = false;
let cameraError = '';
function sourceControls(state) {
  const busy = !!(pending || state.recording || state.switching);
  $('camera').disabled = busy;
  $('cameraDepth').disabled = busy;
  $('connectCamera').disabled = busy || !$('camera').value;
  $('disconnectCamera').disabled = busy || !state.source_device;
  $('refreshCameras').disabled = refreshingCameras;
}
function depthForSelectedCamera() {
  const selected=cameras.find(camera=>camera.device===$('camera').value);
  $('cameraDepth').checked=!!selected?.gelsight || $('camera').value==='__demo__';
  sourceControls(lastState || {});
}
async function refreshCameras() {
  if (refreshingCameras) return;
  refreshingCameras=true;
  sourceControls(lastState || {});
  try {
    const response=await fetch('/api/cameras');
    if (!response.ok) throw new Error('Could not refresh cameras');
    const data=await response.json();
    cameras=data.cameras;
    let selected=$('camera').value;
    if (!cameraSelectionInitialized && lastState?.source_device) selected=lastState.source_device;
    const choices=[{device:'',label:'Choose a camera…'},...cameras,{device:data.demo_device,label:'Demo — synthetic data'}];
    $('camera').replaceChildren();
    for (const camera of choices) {
      const option=document.createElement('option');
      option.value=camera.device; option.textContent=camera.label;
      $('camera').appendChild(option);
    }
    // CLI numeric/device paths may have a stable by-id alias in the list.
    const current=cameras.find(camera=>camera.device===selected || camera.path===selected || camera.path===`/dev/video${selected}`);
    $('camera').value=current?.device || (choices.some(camera=>camera.device===selected) ? selected : '');
    if (!cameraSelectionInitialized) {
      if (lastState?.source_device) $('cameraDepth').checked=!!lastState.depth_enabled;
      else depthForSelectedCamera();
      cameraSelectionInitialized=!!lastState;
    }
    cameraError='';
    cameraListTime=Date.now();
  } catch (error) {
    cameraError=error.message;
    cameraListTime=Date.now();
  } finally {
    refreshingCameras=false;
    sourceControls(lastState || {});
    updateCameraHint();
  }
}
function updateCameraHint() {
  $('cameraHint').textContent=cameraError || (lastState?.recording ? 'Stop recording before changing cameras.' : lastState?.switching ? 'Connecting camera and resetting the reference…' : cameras.length ? 'Select a source and click Connect. Keep the sensor untouched while the reference is collected.' : 'No cameras detected. Plug in a sensor and click Refresh cameras, or select Demo.');
}
$('camera').onchange=depthForSelectedCamera;
$('refreshCameras').onclick=refreshCameras;
$('connectCamera').onclick=()=>command('source',{device:$('camera').value,depth_enabled:$('cameraDepth').checked});
$('disconnectCamera').onclick=()=>command('source',{device:'',depth_enabled:false});
function enableLink(id, enabled) {
  $(id).classList.toggle('disabled', !enabled);
  $(id).setAttribute('aria-disabled', String(!enabled));
  $(id).tabIndex = enabled ? 0 : -1;
}
function controls(state) {
  sourceControls(state);
  $('record').disabled = !!(pending || !state.ready || state.recording || state.switching);
  $('stop').disabled = pending || !state.recording;
  $('calibrate').disabled = !!(pending || !state.connected || state.recording || state.switching);
  $('label').disabled = state.recording;
}
async function command(action, payload = {label:$('label').value}) {
  pending = true;
  if (lastState) controls(lastState);
  try {
    const response = await fetch(`/api/${action}`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error);
    cameraError='';
  } catch (error) {
    $('error').hidden = false;
    $('error').textContent = error.message;
    if (action==='source') {cameraError=error.message; updateCameraHint();}
  } finally { pending = false; }
}
$('record').onclick = () => command('start');
$('stop').onclick = () => command('stop');
$('calibrate').onclick = () => command('calibrate');
const canvas = $('cloud'), ctx = canvas.getContext('2d');
let points = [], yaw = -.4, pitch = .8, zoom = 1, drag = null;
function drawCloud() {
  const rect = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width*dpr); canvas.height = Math.round(rect.height*dpr);
  ctx.scale(dpr,dpr);
  let scale = Math.min(rect.width/420, rect.height/270)*zoom;
  const projected = points.map(([x,y,z]) => {
    x -= 159.5; y -= 119.5;
    const u = x*Math.cos(yaw)-y*Math.sin(yaw), v = x*Math.sin(yaw)+y*Math.cos(yaw);
    return [u, v*Math.cos(pitch)-z*Math.sin(pitch), v*Math.sin(pitch)+z*Math.cos(pitch), z];
  }).sort((a,b) => a[2]-b[2]);
  let cx=0, cy=0;
  if (projected.length) {
    const xs=projected.map(p=>p[0]), ys=projected.map(p=>p[1]);
    const xmin=Math.min(...xs), xmax=Math.max(...xs), ymin=Math.min(...ys), ymax=Math.max(...ys);
    cx=(xmin+xmax)/2; cy=(ymin+ymax)/2;
    scale=Math.min(rect.width*.88/Math.max(320,xmax-xmin),rect.height*.82/Math.max(180,ymax-ymin))*zoom;
  }
  for (const [u,v,,z] of projected) {
    const x=rect.width/2+(u-cx)*scale, y=rect.height/2+(v-cy)*scale;
    ctx.fillStyle = `hsl(${Math.max(0,Math.min(240,160+z*3))} 65% 65%)`;
    ctx.fillRect(x,y,Math.max(1.5,2*scale),Math.max(1.5,2*scale));
  }
}
canvas.onpointerdown = event => { drag = [event.clientX,event.clientY]; canvas.setPointerCapture(event.pointerId); };
canvas.onpointermove = event => {
  if (!drag) return;
  yaw += (event.clientX-drag[0])*.008;
  pitch = Math.max(-1.5,Math.min(1.5,pitch+(event.clientY-drag[1])*.008));
  drag = [event.clientX,event.clientY]; drawCloud();
};
canvas.onpointerup = canvas.onpointercancel = () => { drag = null; };
canvas.addEventListener('wheel', event => { event.preventDefault(); zoom = Math.max(.3,Math.min(4,zoom*Math.exp(-event.deltaY*.001))); drawCloud(); }, {passive:false});
$('resetView').onclick = () => { yaw=-.4; pitch=.8; zoom=1; drawCloud(); };
window.addEventListener('resize',drawCloud);
async function poll() {
  try {
    const response = await fetch('/api/state');
    if (!response.ok) throw new Error(`Server error ${response.status}`);
    const state = await response.json(); lastState = state;
    controls(state);
    updateCameraHint();
    if (Date.now()-cameraListTime>3000) await refreshCameras();
    $('source').textContent = state.mode;
    $('connection').textContent = state.recording ? '● RECORDING' : state.switching ? '● CONNECTING' : state.connected ? '● LIVE' : '● OFFLINE';
    $('connection').className = `badge ${state.recording ? 'recording' : state.connected ? 'live' : ''}`;
    $('fps').textContent = state.connected ? state.fps.toFixed(1) : '—';
    $('count').textContent = state.saved_frames;
    $('status').textContent = state.switching ? 'Connecting camera…' : !state.connected ? 'Select a camera above and click Connect. Refresh the list after plugging in a sensor.' : !state.ready ? `Setting reference ${state.calibration_frames}/${state.calibration_total} · Keep the sensor untouched` : state.recording ? 'Recording synchronized images and numeric data' : 'Reference ready · Live preview';
    $('frame').textContent = `FRAME ${String(state.source_frame).padStart(6,'0')}`;
    $('error').hidden = !(state.error || state.depth_error);
    $('error').textContent = [state.error, state.depth_error ? `Depth unavailable: ${state.depth_error}` : ''].filter(Boolean).join(' · ');
    if (state.session) $('sessionPath').textContent = state.session;
    for (const key of ['raw','deformation','gradient','hog']) {
      const value = state.connected && state.preview?.images[key];
      $(key).hidden = !value;
      if (value) $(key).src = `data:image/jpeg;base64,${value}`;
    }
    if (state.preview?.depth_range) {
      const [lo,hi]=state.preview.depth_range;
      const repaired=100*(state.preview.invalid_normal_fraction || 0);
      $('depthScale').textContent = `Scale ±${state.preview.depth_limit.toFixed(2)} · range ${lo.toFixed(2)} to ${hi.toFixed(2)} model units` + (repaired > 0 ? ` · ${repaired.toFixed(2)}% normals repaired` : '');
    }
    points = state.connected ? state.preview?.points || [] : [];
    $('cloudEmpty').hidden = points.length > 0;
    drawCloud();
    enableLink('depthDownload',state.depth_available);
    enableLink('cloudDownload',state.depth_available);
    enableLink('frameDownload',state.ready);
  } catch (error) {
    $('connection').textContent = '● DISCONNECTED'; $('connection').className = 'badge';
    $('status').textContent = 'Connection lost. Recording status is unknown; reconnect to check.';
    $('error').hidden = false; $('error').textContent = error.message;
    controls({ready:false,recording:false,connected:false});
    for (const id of ['depthDownload','cloudDownload','frameDownload']) enableLink(id,false);
    for (const id of ['raw','deformation','gradient','hog']) $(id).hidden=true;
    points=[]; drawCloud(); $('cloudEmpty').hidden=false;
  } finally { setTimeout(poll,200); }
}
poll();

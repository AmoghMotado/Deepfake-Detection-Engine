function showPanel(name){
  document.querySelectorAll('.panel').forEach(el=>el.classList.add('hidden'));
  document.getElementById('panel-'+name).classList.remove('hidden');
}
async function postForm(url, formData, progressEl){
  progressEl.classList.remove('hidden');
  const res = await fetch(url,{method:'POST', body: formData});
  progressEl.classList.add('hidden');
  return res.json();
}
async function predictImage(){
  const f = document.getElementById('photoFile').files[0]; if(!f) return;
  const fd = new FormData(); fd.append('file', f);
  const out = await postForm('/api/predict-image', fd, document.getElementById('photoProgress'));
  document.getElementById('photoResult').textContent = JSON.stringify(out,null,2);
}
async function predictVideo(){
  const f = document.getElementById('videoFile').files[0]; if(!f) return;
  const fd = new FormData(); fd.append('file', f);
  const out = await postForm('/api/predict-video', fd, document.getElementById('videoProgress'));
  document.getElementById('videoResult').textContent = JSON.stringify(out,null,2);
}
async function compareImages(){
  const f1 = document.getElementById('photo1').files[0];
  const f2 = document.getElementById('photo2').files[0];
  if(!f1||!f2) return;
  const fd = new FormData(); fd.append('file1', f1); fd.append('file2', f2);
  const out = await postForm('/api/compare-images', fd, document.getElementById('cpProgress'));
  document.getElementById('cpResult').textContent = JSON.stringify(out,null,2);
}
async function compareVideos(){
  const f1 = document.getElementById('video1').files[0];
  const f2 = document.getElementById('video2').files[0];
  if(!f1||!f2) return;
  const fd = new FormData(); fd.append('file1', f1); fd.append('file2', f2);
  const out = await postForm('/api/compare-videos', fd, document.getElementById('cvProgress'));
  document.getElementById('cvResult').textContent = JSON.stringify(out,null,2);
}
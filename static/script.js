'use strict';
const $ = id => document.getElementById(id);
const canvas = $('drawCanvas');
const ctx = canvas.getContext('2d', {willReadFrequently:true});
let mode = 'digits', drawing = false, hasInk = false, run = null, generation = 0;
const percent = (n, digits=2) => `${(n*100).toFixed(digits)}%`;
const escapeHtml = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const shapeNames = {circle:'Hình tròn',rectangle:'Hình chữ nhật',triangle:'Tam giác',ellipse:'Ellipse',background:'Nền / chi tiết phụ'};

function invalidate(){
  generation++;
  if(run) run.abort();
  run = null;
  $('predictBtn').disabled = false;
  $('predictBtn').innerHTML = 'Nhận dạng <span>→</span>';
  $('resultBody').classList.remove('result-loading');
}

function emptyResult(){
  $('resultBody').innerHTML = `<div class="empty-result"><div class="empty-art">${mode==='digits'?'7':'○'}</div><h3>Sẵn sàng đọc nét vẽ</h3><p>${mode==='digits'?'Vẽ hoặc chọn một chữ số mẫu, rồi bấm Nhận dạng.':'Vẽ các hình tách rời nhau, hoặc thử một ảnh mẫu bên trái.'}</p></div>`;
  $('pipelineSection').hidden = true;
}

function resetCanvas(){
  invalidate();
  drawing = false;
  canvas.width=mode==='digits'?280:600;canvas.height=mode==='digits'?280:360;
  $('drawingArea').style.aspectRatio=mode==='shapes'?'5 / 3':'';
  ctx.fillStyle = '#fff';ctx.fillRect(0,0,canvas.width,canvas.height);
  ctx.lineCap='round';ctx.lineJoin='round';ctx.strokeStyle='#172822';
  hasInk=false;$('canvasPlaceholder').hidden=false;$('inputError').hidden=true;
  $('imageUpload').value='';
  $('realContext').hidden=true;
  emptyResult();
}

function showError(message){$('inputError').textContent=message;$('inputError').hidden=false;}

function setMode(next){
  invalidate();mode=next;
  for(const button of document.querySelectorAll('[data-mode]')){
    const active=button.dataset.mode===mode;
    button.classList.toggle('active',active);
    if(active)button.setAttribute('aria-current','page');else button.removeAttribute('aria-current');
  }
  $('workspace').hidden=mode==='metrics'||mode==='real';$('metricsSection').hidden=mode!=='metrics';$('realSection').hidden=mode!=='real';$('pipelineSection').hidden=true;
  if(mode==='real'){
    $('pageTitle').innerHTML='Ảnh chụp thực tế<span>.</span>';
    $('pageDescription').textContent='Ảnh có nguồn và nhãn thủ công, tách riêng development, validation và test đã sử dụng.';
    renderReal();return;
  }
  if(mode==='metrics'){
    $('pageTitle').innerHTML='Kiểm thử & kết quả<span>.</span>';
    $('pageDescription').textContent='Kết quả chạy thật trên các tập kiểm thử riêng, cùng những trường hợp mô hình còn sai.';
    renderMetrics();return;
  }
  const digits=mode==='digits';
  $('pageTitle').innerHTML=`${digits?'Chữ số viết tay':'Hình học cơ bản'}<span>.</span>`;
  $('pageDescription').textContent=digits?'Vẽ một chữ số, xem dự đoán và cách CNN nhìn nét vẽ của bạn.':'Tìm hình tròn và chữ nhật trong ảnh, rồi quan sát cách từng vùng được phân loại.';
  $('inputType').textContent=digits?'MỘT CHỮ SỐ':'NHIỀU HÌNH TÁCH RỜI';
  $('modelTag').textContent=digits?'CNN · MNIST':'CNN · HÌNH HỌC';
  $('drawingArea').classList.toggle('shape-area',!digits);
  canvas.width=digits?280:600;canvas.height=digits?280:360;
  canvas.setAttribute('aria-label',digits?'Khung vẽ một chữ số':'Khung vẽ nhiều hình học tách rời');
  $('brushSize').min=digits?8:3;$('brushSize').max=digits?34:20;$('brushSize').value=digits?22:6;
  $('brushValue').textContent=$('brushSize').value;
  $('inputHint').textContent=digits?'Viết một chữ số 0–9 thật rõ. Không dùng chế độ này để đọc cả dãy số.':'Dùng nền đồng nhất, hình kín và tách rời nhau. Hình vuông được tính là chữ nhật. Hình chồng nhau có thể bị gộp.';
  $('sampleNote').textContent=digits?'Từ tập MNIST test':'Ảnh tổng hợp có nhãn';
  $('sampleGrid').classList.toggle('shape-samples',!digits);
  const choices=digits?Array.from({length:10},(_,i)=>({name:`digit_${i}`,label:String(i)})):
    [{name:'shapes_mixed',label:'Nhiều hình'},{name:'shapes_dark',label:'Nền tối'},{name:'shapes_noise',label:'Có nhiễu'},{name:'shapes_touching',label:'Hình dính nhau'}];
  $('sampleGrid').innerHTML=choices.map(c=>`<button class="sample-button" data-sample="${c.name}" title="Thử ${c.label}"><img src="/examples/${c.name}.png" alt="${c.label}"><span>${c.label}</span></button>`).join('');
  resetCanvas();
}

function point(event){const r=canvas.getBoundingClientRect();return {x:(event.clientX-r.left)*canvas.width/r.width,y:(event.clientY-r.top)*canvas.height/r.height};}
canvas.addEventListener('pointerdown',event=>{
  if(drawing||event.button>0)return;
  event.preventDefault();invalidate();emptyResult();drawing=true;canvas.setPointerCapture(event.pointerId);
  const p=point(event);ctx.lineWidth=Number($('brushSize').value);ctx.strokeStyle='#172822';ctx.fillStyle='#172822';
  ctx.beginPath();ctx.arc(p.x,p.y,ctx.lineWidth/2,0,Math.PI*2);ctx.fill();
  ctx.beginPath();ctx.moveTo(p.x,p.y);hasInk=true;$('canvasPlaceholder').hidden=true;$('inputError').hidden=true;
});
canvas.addEventListener('pointermove',event=>{if(!drawing)return;const p=point(event);ctx.lineTo(p.x,p.y);ctx.stroke();});
for(const name of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(name,()=>{drawing=false;ctx.closePath();});
$('brushSize').addEventListener('input',()=>$('brushValue').textContent=$('brushSize').value);
$('clearBtn').addEventListener('click',resetCanvas);
document.querySelectorAll('[data-mode]').forEach(button=>button.addEventListener('click',()=>setMode(button.dataset.mode)));

function loadImage(url, token=generation){
  return new Promise((resolve,reject)=>{
    const image=new Image();
    image.onload=()=>{
      if(token!==generation)return resolve(false);
      if(Math.max(image.naturalWidth,image.naturalHeight)>4096||image.naturalWidth*image.naturalHeight>12000000)return reject(new Error('Ảnh quá lớn. Hãy dùng ảnh có mỗi cạnh tối đa 4096 pixel, không quá 12 megapixel.'));
      if(mode==='shapes'){
        canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;
        $('drawingArea').style.aspectRatio=`${canvas.width} / ${canvas.height}`;
        ctx.lineCap='round';ctx.lineJoin='round';
      }
      ctx.fillStyle='#fff';ctx.fillRect(0,0,canvas.width,canvas.height);
      const scale=Math.min(canvas.width/image.naturalWidth,canvas.height/image.naturalHeight);
      const w=image.naturalWidth*scale,h=image.naturalHeight*scale;
      ctx.drawImage(image,(canvas.width-w)/2,(canvas.height-h)/2,w,h);
      hasInk=true;$('canvasPlaceholder').hidden=true;$('inputError').hidden=true;$('realContext').hidden=true;emptyResult();resolve(true);
    };
    image.onerror=()=>reject(new Error('Không đọc được ảnh. Hãy thử tệp PNG hoặc JPG khác.'));
    image.src=url;
  });
}

$('sampleGrid').addEventListener('click',async event=>{
  const button=event.target.closest('[data-sample]');if(!button)return;
  invalidate();const token=generation;
  try{await loadImage(`/examples/${button.dataset.sample}.png`,token);}catch(error){showError(error.message);}
});

$('imageUpload').addEventListener('change',event=>{
  const file=event.target.files[0];if(!file)return;
  if(file.size>8*1024*1024){showError('Tệp quá lớn. Vui lòng chọn ảnh nhỏ hơn 8 MB.');return;}
  if(!/\.(png|jpe?g|webp|bmp)$/i.test(file.name)){showError('Chỉ hỗ trợ PNG, JPG, WebP hoặc BMP.');return;}
  invalidate();const token=generation;
  const reader=new FileReader();
  reader.onload=async()=>{try{await loadImage(reader.result,token);}catch(error){showError(error.message);}};
  reader.onerror=()=>showError('Không đọc được tệp.');reader.readAsDataURL(file);
});

function warningsBlock(items){return items?.length?`<div class="warnings">${items.map(w=>`<p>${escapeHtml(w)}</p>`).join('')}</div>`:'';}
function pipeline(data){
  const isDigit=mode==='digits';
  const stages=isDigit?[
    ['gray','Ảnh mức xám','Tự tính độ sáng từ RGB và xử lý kênh trong suốt.'],
    data.foreground_method==='background_subtraction'
      ?['binary','Trừ nền rồi tách nét bằng Otsu',`Nền không đều được ước lượng bằng phép đóng/mở ảnh xám rồi trừ đi; ngưỡng Otsu T = ${data.threshold}.`]
      :['binary','Tách nét bằng Otsu',`Ngưỡng tự động T = ${data.threshold}. Loại các vùng nhiễu rất nhỏ.`],
    ['normalized','Chuẩn hóa 28 × 28','Giữ tỉ lệ, nét nằm trong 20 × 20; căn theo trọng tâm và chia 255.']
  ]:[['gray','Ảnh mức xám','Chuyển RGB sang cường độ sáng.'],['denoised','Lọc median 3 × 3','Giảm nhiễu trước khi phân đoạn.'],['binary','Ngưỡng Otsu đại diện',`T = ${data.threshold}. Các nhánh sáng/tối được xem riêng bên dưới.`],['closed','Vùng ứng viên sau lọc nền','Chỉ hợp những vùng đã qua lọc nền và vùng quá nhỏ; giữ các nhánh tách nền riêng để đối chiếu.'],['edges','Biên Sobel','Minh họa biên ảnh; CNN nhận các vùng đã chuẩn hóa.']];
  $('pipelineSteps').classList.toggle('five',!isDigit);
  $('pipelineSteps').innerHTML=stages.map(([key,title,description],i)=>`<div class="pipeline-step"><div class="pipeline-step-image"><img src="${data.pipeline[key]}" alt="${title}"></div><p>${String(i+1).padStart(2,'0')} · ${title}</p><small>${description}</small></div>`).join('');
  const notes=isDigit&&data.preprocessing_notes?.length?' · '+data.preprocessing_notes.join(' '):'';
  $('pipelineMeta').textContent=(data.polarity==='both_polarities'?'Xét hai cực tính · lọc vùng nền, chi tiết trong và khung trùng':data.polarity==='dark_on_light'?'Nét tối / nền sáng':'Nét sáng / nền tối')+notes;
  $('featureMaps').hidden=!isDigit;
  if(isDigit)$('featureMaps').innerHTML=`<div class="feature-map-strip">${data.feature_maps.map((src,i)=>`<img src="${src}" alt="Bản đồ kích hoạt ${i+1}">`).join('')}</div><div class="feature-caption"><strong>8 bản đồ kích hoạt từ lớp tích chập đầu tiên</strong>Các bộ lọc được học từ dữ liệu, không gán ý nghĩa thủ công cho từng bộ lọc.</div>`;
  const branchPanel=$('pipelineBranches');
  branchPanel.hidden=isDigit||!data.pipeline_branches?.length;
  if(!branchPanel.hidden){
    const names={global_dark:'Otsu · vùng tối',global_light:'Otsu · vùng sáng',local_dark:'Trừ nền · vùng tối',local_light:'Trừ nền · vùng sáng',edges_otsu:'Biên Sobel · Otsu',edges_low:'Biên Sobel · ngưỡng thấp'};
    const name=value=>value.endsWith('_hull')?(names[value.slice(0,-5)]||value.slice(0,-5).replace('color_','Kênh màu '))+' · bao lồi':names[value]||value.replace('color_','Kênh màu ').replace('_dark',' · tối').replace('_light',' · sáng');
    branchPanel.innerHTML=`<div class="branch-heading"><div><h3>Đối chiếu từng nhánh tách nền</h3><p>Vùng sáng và tối được xử lý riêng. Ảnh “sau lọc nền” cho thấy vùng thực sự chuyển tới CNN.</p></div><label>Chọn nhánh <select id="pipelineBranchSelect">${data.pipeline_branches.map((b,i)=>`<option value="${i}">${escapeHtml(name(b.name))}</option>`).join('')}</select></label></div><div id="pipelineBranchImages" class="branch-images"></div>`;
    const renderBranch=()=>{
      const b=data.pipeline_branches[Number($('pipelineBranchSelect').value)];
      $('pipelineBranchImages').innerHTML=[['binary','Mặt nạ riêng của nhánh'],['closed','Sau đóng hình thái'],['candidates','Sau lọc nền / vùng nhỏ']].map(([key,label])=>`<div><img src="${b[key]}" alt="${escapeHtml(label+' — '+name(b.name))}"><p>${label}</p></div>`).join('')+`<div><img src="${data.pipeline_retained}" alt="Các vùng giữ lại sau CNN và lọc trùng"><p>Giữ lại sau CNN / lọc trùng</p></div>`;
    };
    $('pipelineBranchSelect').addEventListener('change',renderBranch);renderBranch();
  }
  $('pipelineSection').hidden=false;
}

function renderDigit(data){
  const ok=data.confident!==false, state=ok?'confident':'uncertain';
  const status=ok?`Điểm dự đoán <strong>${percent(data.confidence)}</strong>`:`<strong>Chưa chắc chắn</strong> · điểm ${percent(data.confidence)} dưới ngưỡng ${percent(data.min_confidence,0)}`;
  const merged=data.merged_parts>1?`<p class="digit-note uncertain">Có ${data.merged_parts} cụm nét tách rời đã được gộp lại và đọc như MỘT chữ số (chế độ này không đọc chuỗi số).</p>`:'';
  const ignored=data.ignored_parts?`<p class="digit-note">${data.ignored_parts} vết nhỏ (dưới 18% cụm nét lớn nhất) bị bỏ qua, tô màu đỏ.</p>`:'';
  const image=data.annotated?`<div class="digit-annotated"><img class="annotated-image" src="${data.annotated}" alt="Ảnh kết quả: nét được đọc tô ${ok?'xanh':'đỏ'}"><div class="shape-legend"><span class="legend-item target">Nét được đọc · chắc chắn</span><span class="legend-item unknown">Chưa chắc chắn / vết bị bỏ qua</span></div>${merged}${ignored}</div>`:merged+ignored;
  $('resultBody').innerHTML=`<div class="result-top ${state}"><div class="predicted-digit">${data.prediction}</div><div><div class="eyebrow">DỰ ĐOÁN CỦA MÔ HÌNH</div><h3>Chữ số ${data.prediction}${ok?'':' <span class="uncertain-tag">chưa chắc chắn</span>'}</h3><div class="confidence-label">${status}</div></div></div>${image}<div class="softmax-note">Điểm softmax thể hiện mức ưu tiên của mô hình giữa 10 lớp; chưa được hiệu chỉnh thành xác suất đúng.</div><div class="tiny-heading">PHÂN BỐ ĐIỂM CHO 10 CHỮ SỐ</div><div class="probability-list ${state}">${data.probabilities.map((p,i)=>`<div class="probability-row ${i===data.prediction?'selected':''}"><span>${i}</span><div class="bar-track"><span style="width:${p*100}%"></span></div><span>${percent(p,1)}</span></div>`).join('')}</div>${warningsBlock(data.warnings)}`;
}

function shapeKind(o){return o.class==='unknown'?'unknown':(o.target?'target':'other');}

function renderShapes(data){
  const known=data.objects.length-data.unknown_count;
  $('resultBody').innerHTML=`<div class="shape-counts"><div class="shape-count"><span>○ Hình tròn</span><strong>${data.counts.circle}</strong></div><div class="shape-count"><span>▭ Chữ nhật</span><strong>${data.counts.rectangle}</strong></div>${data.unknown_count?`<div class="shape-count unknown"><span>? Chưa xác định</span><strong>${data.unknown_count}</strong></div>`:''}</div><img class="annotated-image" id="annotatedImage" src="${data.annotated}" alt="Ảnh kết quả: mỗi vùng được tô màu theo nhãn"><div class="shape-legend"><span class="legend-item target">Tròn / chữ nhật</span><span class="legend-item other">Hình khác (ngoài phạm vi)</span><span class="legend-item unknown">Chưa xác định</span>${data.annotated_known?'<label class="legend-toggle"><input type="checkbox" id="hideUnknown"> Ẩn vùng chưa xác định</label>':''}</div><p class="table-note">Đã giữ ${data.segmentation.retained_count}/${data.segmentation.candidate_count} vùng ứng viên sau lọc nền, chi tiết trong và khung trùng: ${known} vùng đã xác định, ${data.unknown_count} vùng chưa xác định.</p><div class="object-list">${data.objects.map(o=>objectRow(o)).join('')}</div>${warningsBlock(data.warnings)}<p class="subtle" style="line-height:1.8;margin-bottom:0">Tam giác, ellipse và đa giác là ví dụ ngoài hai lớp mục tiêu. Điểm dưới ${percent(data.min_confidence,0)} hoặc vùng nhỏ hơn 12 px được ghi là “Chưa xác định” (tô đỏ); hình ngoài dữ liệu học vẫn có thể bị gán sai.</p>`;
  const toggle=$('hideUnknown');
  if(toggle){toggle.addEventListener('change',()=>{$('annotatedImage').src=toggle.checked?data.annotated_known:data.annotated;document.querySelectorAll('.object-row.unknown').forEach(r=>{r.hidden=toggle.checked;});});}
}

function objectRow(o){
  const f=o.features, kind=shapeKind(o);
  const unknownNote=kind!=='unknown'?'':`<p class="unknown-note">${o.unknown_reason==='too_small'?'Vùng nhỏ hơn 12 px nên không phân loại.':'Điểm CNN cao nhất dưới ngưỡng tin cậy nên không gán nhãn.'} Điểm cao nhất: ${percent(o.confidence,1)}.</p>`;
  const score=kind==='unknown'?(o.unknown_reason==='too_small'?'quá nhỏ':'dưới ngưỡng'):percent(o.confidence,1);
  const feats=f.perimeter===undefined?`Diện tích: ${f.area} px`:`Diện tích: ${f.area} px · Độ lấp đầy: ${percent(f.extent,1)}<br>Chu vi (mã chuỗi Freeman): ${f.perimeter} px<br>Độ tròn 4πA/p²: ${f.circularity.toFixed(3)} <small>(tròn ≈ 0,9–1 · vuông ≈ 0,79)</small><br>Độ lệch tâm: ${f.eccentricity.toFixed(3)} · Số góc trên biên dạng r(θ): ${f.corners}`;
  return `<details class="object-row ${kind}"><summary><span class="object-id">${o.id}</span><strong>${escapeHtml(o.label)}</strong><span>${score}</span></summary><div class="object-details">${o.crop?`<img src="${o.crop}" alt="Vùng hình chuẩn hóa 32 × 32">`:''}${unknownNote}<p>Khung (x, y, rộng, cao): ${o.bbox.join(', ')} px<br>${feats}</p><p>Đặc trưng hình học (Chương 7) để quan sát; nhãn do CNN quyết định, riêng nhãn “Đa giác” là do biên dạng có từ 5 góc trở lên.</p></div></details>`;
}

async function recognize(){
  if(!hasInk){showError('Hãy vẽ hoặc chọn một ảnh mẫu trước khi nhận dạng.');return;}
  invalidate();const token=generation;const controller=new AbortController();run=controller;
  $('inputError').hidden=true;$('predictBtn').disabled=true;$('predictBtn').innerHTML='<span class="loading-dot"></span> Đang xử lý';$('resultBody').classList.add('result-loading');
  try{
    const response=await fetch(`/api/${mode}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:canvas.toDataURL('image/png')}),signal:controller.signal});
    const data=await response.json();if(!response.ok)throw new Error(data.error||'Không thể nhận dạng ảnh.');
    if(token!==generation)return;
    if(mode==='digits')renderDigit(data);else renderShapes(data);pipeline(data);
  }catch(error){if(error.name!=='AbortError'&&token===generation)showError(error.message);}
  finally{if(token===generation){run=null;$('predictBtn').disabled=false;$('predictBtn').innerHTML='Nhận dạng <span>→</span>';$('resultBody').classList.remove('result-loading');}}
}
$('predictBtn').addEventListener('click',recognize);

let metricsCache=null;
function metricTable(headers,rows){return `<div class="table-wrap"><table><thead><tr>${headers.map(h=>`<th>${h}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(v=>`<td>${v}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;}
async function renderMetrics(){
  const section=$('metricsSection');section.innerHTML='<div class="metric-notice"><span class="loading-dot"></span> Đang đọc kết quả kiểm thử…</div>';
  try{
    {const response=await fetch('/api/metrics',{cache:'no-store'});const data=await response.json();if(!response.ok)throw new Error(data.error);metricsCache=data;}
    const m=metricsCache,d=m.digits,s=m.shapes;
    const cm=d.cnn_raw.confusion_matrix;
    const conditions={clean:'Nền sáng, hình tách rời',gaussian_and_impulse_noise:'Nhiễu Gaussian + xung',dark_background:'Nền tối'};
    section.innerHTML=`<div class="metric-hero"><div class="metric-box"><span>CHỮ SỐ · MNIST TEST</span><strong>${percent(d.cnn_raw.accuracy)}</strong><small>10.000 ảnh gốc · CNN, chưa qua pipeline tải ảnh</small></div><div class="metric-box"><span>CHỮ SỐ · QUA PIPELINE</span><strong>${percent(d.cnn_pipeline.accuracy)}</strong><small>Cùng 10.000 ảnh qua tiền xử lý của ứng dụng</small></div><div class="metric-box"><span>PHÂN LOẠI HÌNH · TỔNG HỢP</span><strong>${percent(s.classification.accuracy)}</strong><small>${s.classification.samples} vùng test cố định · 4 lớp hình; CNN có thêm lớp nền</small></div></div>
    <div class="metric-notice">Huấn luyện chữ số trên 55.000 ảnh MNIST, chọn mô hình bằng 5.000 ảnh validation, rồi đánh giá trên 10.000 ảnh test riêng. Hình học đã khử trùng ảnh chuẩn hóa trong và giữa train / validation / test. Ảnh chụp thực được đánh giá riêng ở mục Ảnh chụp thực tế; không gộp với kết quả tổng hợp.</div>
    <article class="card"><div class="card-heading"><div><span class="step-number">01</span><h2>So sánh nhận dạng chữ số</h2></div><span class="pill">CÙNG 10.000 ẢNH TEST</span></div>${metricTable(['Mô hình / cách đưa ảnh vào','Accuracy','Macro F1','Số ảnh sai'],[['ANN gốc được cung cấp · ảnh MNIST gốc',percent(d.original_ann_raw.accuracy),d.original_ann_raw.macro_f1.toFixed(4),10000-d.original_ann_raw.correct],['CNN mới · ảnh MNIST gốc',percent(d.cnn_raw.accuracy),d.cnn_raw.macro_f1.toFixed(4),10000-d.cnn_raw.correct],['CNN mới · qua tiền xử lý ứng dụng',percent(d.cnn_pipeline.accuracy),d.cnn_pipeline.macro_f1.toFixed(4),10000-d.cnn_pipeline.correct]])}<p class="table-note">ANN gốc giữ nguyên trọng số từ dự án đầu vào. Hai mô hình có quy trình huấn luyện khác nhau; bảng này so sánh hành vi của hai checkpoint, chưa cô lập tác động của kiến trúc.</p></article>
    <div class="metric-cols"><article class="card"><div class="card-heading"><div><span class="step-number">02</span><h2>Ma trận nhầm lẫn · CNN</h2></div></div><div class="confusion-wrap"><table class="confusion-table"><thead><tr><th>Thật ↓<br>Đoán →</th>${Array.from({length:10},(_,i)=>`<th>${i}</th>`).join('')}</tr></thead><tbody>${cm.map((row,i)=>`<tr><th>${i}</th>${row.map(value=>`<td style="background:rgba(81,145,68,${.035+.65*value/Math.max(...row)})">${value}</td>`).join('')}</tr>`).join('')}</tbody></table></div><p class="table-note">Ảnh MNIST test gốc. Ô trên đường chéo là dự đoán đúng.</p></article><article class="card"><div class="card-heading"><div><span class="step-number">03</span><h2>Những chữ số CNN đoán sai</h2></div></div><img class="failure-image" src="/results/digit_failures.png" alt="Các ảnh MNIST CNN đoán sai, có nhãn thật và dự đoán"><p class="table-note">24 ca sai đầu tiên theo thứ tự tập test, không chọn riêng những ca có lợi. Xem thêm lỗi sau tiền xử lý trong <a href="/results/pipeline_failures.png" target="_blank" rel="noopener" style="text-decoration:underline">ảnh lỗi pipeline</a>.</p></article></div>
    <article class="card"><div class="card-heading"><div><span class="step-number">D</span><h2>Kiểm tra trùng dữ liệu hình học</h2></div><span class="pill green">${s.deduplication.passed?'ĐÃ QUA':'CẦN KIỂM TRA'}</span></div>${metricTable(['Tập dữ liệu','Số ảnh','Ảnh duy nhất','Ảnh trùng trong tập'],Object.entries(s.deduplication.splits).map(([name,v])=>[{train:'Huấn luyện',validation:'Validation',test:'Kiểm thử'}[name],v.samples,v.unique_normalized_uint8,v.duplicates_within_split]))}<p class="table-note">Kiểm tra trên đầu vào CNN 32 × 32 sau lượng tử hóa 8-bit, không dùng nhãn để phân biệt ảnh. Trùng giữa train–validation: ${s.deduplication.cross_split_overlap.train__validation}; train–test: ${s.deduplication.cross_split_overlap.train__test}; validation–test: ${s.deduplication.cross_split_overlap.validation__test}. Đây là kiểm tra trùng pixel, chưa bảo đảm loại mọi hình gần giống nhau về cảm nhận. CNN hình hiện học ${m.protocol.geometry_training_v4.train_samples.toLocaleString('vi-VN')} mẫu sau bổ sung development và lớp nền; ${m.protocol.geometry_training_v4.real_validation_samples.toLocaleString('vi-VN')} ứng viên validation giữ riêng, không tăng cường hoặc dùng để train.</p></article>
    <article class="card"><div class="card-heading"><div><span class="step-number">04</span><h2>Phát hiện hình trong ảnh có nhiều đối tượng</h2></div><span class="pill">IOU ≥ 0,5</span></div>${metricTable(['Điều kiện ảnh tổng hợp','Số ảnh','Hình mục tiêu','Precision','Recall','F1'],Object.entries(s.detection.conditions).map(([key,v])=>[conditions[key]||key,v.scenes,v.ground_truth_targets,percent(v.precision),percent(v.recall),v.f1.toFixed(4)]))}<p class="table-note">${s.detection.scenes} ảnh chứa tổng cộng ${s.detection.ground_truth_all_classes} đối tượng, trong đó ${s.detection.ground_truth_targets} hình tròn/chữ nhật cần tìm. IoU trung bình trên các phát hiện đúng: ${s.detection.mean_iou_on_true_positives.toFixed(4)}. Tam giác, ellipse và vùng nền được dùng làm lớp ngoài mục tiêu; hình vuông thuộc lớp chữ nhật.</p></article>
    <article class="card"><div class="card-heading"><div><span class="step-number">05</span><h2>Ca khó & giới hạn đã quan sát</h2></div><span class="pill">CẦN CẢI THIỆN</span></div><ul class="limits-list"><li>Ảnh có hình chạm hoặc chồng nhau: recall ${percent(s.stress_touching.recall)} trên ${s.stress_touching.scenes} ảnh stress test. Các vùng liên thông bị gộp, CNN nhận một hình ghép thay vì từng vật riêng.</li><li>Pipeline tiền xử lý làm thay đổi một số nét và vị trí chữ: có ${10000-d.cnn_pipeline.correct} ảnh MNIST test bị đoán sai sau pipeline, so với ${10000-d.cnn_raw.correct} ảnh khi dùng đầu vào gốc.</li><li>Đã xử lý: ánh sáng lệch, giấy kẻ dòng, nét bút mảnh (chữ số); nét viền hở, viền 1 px, hình quá nhỏ, lục giác (hình học). Còn sai: hình chạm nhau, bát giác (quá gần hình tròn), ảnh bị cắt mép và nền nhiều chi tiết.</li><li>Chưa hỗ trợ chữ cái, chữ tiếng Việt, cả dãy số hoặc ảnh hình học bất kỳ. Đã có bộ ảnh chụp thực chia theo nguồn thành development / validation / test; chưa đủ để suy ra độ chính xác trên mọi ảnh thực tế.</li></ul></article>`;
  }catch(error){section.innerHTML=`<div class="metric-notice">${escapeHtml(error.message)}</div>`;}
}

let realCache=null;
const safeUrl=value=>/^https?:\/\//.test(String(value))?escapeHtml(value):'#';
const difficultyName=value=>value==='basic'?'Ảnh cơ bản':value==='negative'?'Không có mục tiêu':'Ca khó';

const roleNames={development:'Phát triển',validation:'Validation',test:'Test (nguồn riêng)'};
async function renderReal(){
  const section=$('realSection');
  section.innerHTML='<div class="metric-notice"><span class="loading-dot"></span>Đang đọc kết quả ảnh thực…</div>';
  try{
    const response=await fetch('/api/real',{cache:'no-store'});const data=await response.json();if(!response.ok)throw new Error(data.error);realCache=data;
    const r=realCache,d=r.detection_by_role.test;
    if(!d)throw new Error('Chưa có đánh giá test sau khi khóa cấu hình. Chạy scripts/evaluate.py với cấu hình đã khóa.');
    const rows=Object.entries(r.detection_by_role).map(([key,v])=>[roleNames[key],v.scenes,v.ground_truth_targets,v.tp,v.fp,v.fn,percent(v.precision),percent(v.recall),v.f1.toFixed(4)]);
    const compare=r.baseline_comparison;
    const seen=r.protocol.test_previously_seen_in_phase;
    const testName=seen?`TEST (ĐÃ MỞ Ở GIAI ĐOẠN ${seen})`:'TEST MỚI';
    section.innerHTML=`<div class="metric-hero"><div class="metric-box"><span>${testName} · PRECISION</span><strong>${percent(d.precision)}</strong><small>${d.tp} đúng / ${d.tp+d.fp} dự đoán mục tiêu</small></div><div class="metric-box"><span>${testName} · RECALL</span><strong>${percent(d.recall)}</strong><small>${d.tp} tìm được / ${d.ground_truth_targets} vật thể có nhãn</small></div><div class="metric-box"><span>${testName} · F1</span><strong>${d.f1.toFixed(4)}</strong><small>${d.scenes} ảnh test · IoU ≥ 0,5</small></div></div>
    <div class="metric-notice">${r.audit.case_count} ảnh thực, ${Object.values(r.audit.target_counts).reduce((a,b)=>a+b,0)} vật thể có nhãn. Các ảnh test giai đoạn trước đã chuyển sang development. CNN học thêm từ development, có lớp nền; chọn mô hình và pipeline bằng validation có cả hai lớp. Điểm test chỉ tính nhóm nguồn mới sau khi khóa cấu hình. Đây là tập thử nhỏ, có cả ca bị cắt, che khuất và nền phức tạp.${seen?` <strong>Lưu ý:</strong> ảnh test đã được mở lần đầu ở giai đoạn ${seen}. Giai đoạn ${r.protocol.protocol_phase} chỉ sửa lỗi xử lý ảnh bằng ảnh tổng hợp và development/validation, không chạy ảnh test khi sửa, nhưng điểm test không còn là đánh giá trên ảnh chưa từng thấy.`:''}</div>
    <article class="card"><div class="card-heading"><div><span class="step-number">01</span><h2>Kết quả theo vai trò dữ liệu</h2></div><span class="pill">${r.protocol.original_lock_verified?`CẤU HÌNH ĐÃ KHÓA · GIAI ĐOẠN ${r.protocol.protocol_phase||4}`:'THỬ KHÁM PHÁ'}</span></div>${metricTable(['Tập ảnh','Ảnh','Vật thể','TP','FP','FN','Precision','Recall','F1'],rows)}<p class="table-note">Ghép một dự đoán với tối đa một nhãn, đúng lớp và IoU ≥ 0,5. Nhãn chỉ là đường bao ngoài vật thể; chữ, trục, lỗ và bóng đổ không phải mục tiêu riêng. Hình vuông thuộc chữ nhật; góc nhìn xiên và vật bị che/cắt được ghi là ca khó.</p></article>
    ${compare?`<article class="card"><div class="card-heading"><div><span class="step-number">02</span><h2>Trước và sau tách nền / lọc vùng</h2></div></div>${metricTable(['Tập ảnh','Pipeline','TP','FP','FN','F1'],['development','validation','test'].flatMap(role=>['legacy_v3','new_regions_old_cnn','phase4','phase5'].map(mode=>{const v=compare[mode]?.[role];return v?[roleNames[role],({legacy_v3:'Giai đoạn 3',new_regions_old_cnn:'Tách vùng giai đoạn 4 + CNN giai đoạn 3',phase4:compare.phase5?'Giai đoạn 4 (trước sửa lỗi)':'Tách vùng mới + CNN mới',phase5:'Giai đoạn 5 (hiện tại)'})[mode],v.tp,v.fp,v.fn,v.f1.toFixed(4)]:[];})))}<p class="table-note">Cùng ảnh và nhãn. ${compare.phase5?'Giai đoạn 4 và 5 dùng chung trọng số CNN, nên chênh lệch giữa hai dòng là tác động của các sửa lỗi xử lý ảnh.':'Bảng tách riêng tác động của tách vùng và CNN.'} Giữ cả kết quả tăng và giảm.</p></article>`:''}
    <article class="card"><div class="card-heading"><div><h2>Thư viện ảnh có nhãn</h2></div><label>Lọc tập ảnh <select id="realRoleFilter"><option value="all">Tất cả (${r.cases.length})</option>${Object.entries(r.audit.case_counts_by_role).map(([role,n])=>`<option value="${role}">${roleNames[role]} (${n})</option>`).join('')}</select></label></div><p class="table-note">Mở “Đối chiếu nhãn” để xem lỗi từng vật thể, hoặc “Thử ảnh này” để chạy pipeline hiện tại.</p></article>
    <div class="real-gallery">${r.cases.map(c=>`<article class="card real-case" data-real-case="${escapeHtml(c.id)}" data-real-role="${c.split}"><img class="real-photo" loading="lazy" src="${c.image_url}" alt="${escapeHtml(c.title)}"><div class="real-case-content"><div class="real-case-heading"><h2>${escapeHtml(c.title)}</h2><span class="pill ${c.difficulty==='basic'?'green':''}">${difficultyName(c.difficulty)}</span></div><p class="real-case-score">${roleNames[c.split]} · ${c.ground_truth.length} nhãn · <strong>${c.detection.tp} đúng</strong> · ${c.detection.fp} dư · ${c.detection.fn} bỏ sót</p><p class="real-case-note">${escapeHtml(c.notes)}</p><div class="real-attribution">${escapeHtml(c.source.author)} · ${escapeHtml(c.source.license)}<br><a href="${safeUrl(c.source.source_page)}" target="_blank" rel="noopener">Trang nguồn</a> · <a href="${safeUrl(c.source.license_url)}" target="_blank" rel="noopener">Giấy phép</a></div><div class="real-case-actions"><button class="button secondary" data-open-real="${escapeHtml(c.id)}">Thử ảnh này ↗</button><button class="button primary" data-inspect-real="${escapeHtml(c.id)}">Đối chiếu nhãn →</button></div></div></article>`).join('')}</div>
    <article class="card"><div class="card-heading"><div><h2>Lỗi và chẩn đoán</h2></div></div>${metricTable(['Đối chiếu vật thể trên tất cả các tập','Số lượng'],[['Khớp đúng',r.failure_counts.matched||0],['Không có vùng khớp',r.failure_counts.no_matching_region_after_segmentation||0],['Sai lớp / điểm thấp',r.failure_counts.region_found_but_wrong_class_or_low_score||0],['Có ứng viên nhưng bị CNN/lọc vùng loại',r.failure_counts.candidate_removed_by_filter_or_background_cnn||0],['Vùng khớp đã gán cho vật thể khác',r.failure_counts.matching_region_already_assigned||0]])}<p class="table-note">Khi cho sẵn bbox thật, CNN đúng ${r.roi_diagnostic.correct}/${r.roi_diagnostic.samples} vùng. Đây là chẩn đoán có trợ giúp vị trí, không phải độ chính xác phát hiện toàn ảnh. Hình chồng nhau, vỏ trong suốt và nền có họa tiết vẫn có thể sai.</p></article>`;
    $('realRoleFilter').addEventListener('change',event=>{for(const card of section.querySelectorAll('[data-real-role]'))card.hidden=event.target.value!=='all'&&card.dataset.realRole!==event.target.value;});
  }catch(error){section.innerHTML=`<div class="metric-notice">${escapeHtml(error.message)}</div>`;}
}

$('realSection').addEventListener('click',async event=>{
  const open=event.target.closest('[data-open-real]');
  const inspect=event.target.closest('[data-inspect-real]');
  const id=open?.dataset.openReal||inspect?.dataset.inspectReal;
  if(!id||!realCache)return;
  const c=realCache.cases.find(c=>c.id===id);if(!c)return;
  if(open){
    setMode('shapes');const token=generation;
    try{
      const loaded=await loadImage(c.image_url,token);if(!loaded)return;
      $('realContext').innerHTML=`<strong>${escapeHtml(c.title)}</strong><br>Ảnh giữ nguyên tỉ lệ và độ phân giải khi đưa vào backend. ${escapeHtml(c.source.author)} · ${escapeHtml(c.source.license)} · <a href="${safeUrl(c.source.source_page)}" target="_blank" rel="noopener">Nguồn</a>`;
      $('realContext').hidden=false;
      await recognize();
      $('workspace').scrollIntoView({behavior:'smooth',block:'start'});
    }catch(error){showError(error.message);}
  }else{
    const reasons={matched:'Khớp đúng',no_matching_region_after_segmentation:'Không có vùng khớp',region_found_but_wrong_class_or_low_score:'Sai lớp / điểm thấp',matching_region_already_assigned:'Vùng đã gán cho vật thể khác',candidate_removed_by_filter_or_background_cnn:'Có ứng viên, bị CNN/lọc vùng loại'};
    $('realDialogBody').innerHTML=`<div class="dialog-heading"><div><h2>${escapeHtml(c.title)}</h2><p>${difficultyName(c.difficulty)} · ${c.detection.tp} đúng, ${c.detection.fp} dư, ${c.detection.fn} bỏ sót</p></div><button class="button secondary" id="closeRealDialog">Đóng ×</button></div><img class="comparison-photo" src="/results/${encodeURIComponent(c.comparison_image)}" alt="Bên trái: nhãn thủ công. Bên phải: dự đoán mô hình"><p class="table-note">Bên trái là bbox thủ công màu đỏ. Bên phải là dự đoán: xanh cho tròn/chữ nhật, cam cho lớp khác. ${escapeHtml(c.notes)}</p>${metricTable(['Vật thể','Lớp thật','Chẩn đoán toàn ảnh','CNN khi biết sẵn vị trí'],c.roi_diagnostics.map(v=>[v.annotation_id,shapeNames[v.true_class],reasons[v.failure_category],shapeNames[v.roi_top1_class]||v.roi_top1_class]))}<p class="table-note">Nguồn: ${escapeHtml(c.source.author)} · ${escapeHtml(c.source.license)}. Ảnh đã thu nhỏ/đổi định dạng; không sinh bằng AI.</p>`;
    $('realDialog').showModal();$('closeRealDialog').addEventListener('click',()=>$('realDialog').close());
  }
});

async function checkModels(){
  try{const response=await fetch('/api/health');const data=await response.json();if(!response.ok)throw new Error(data.error);$('modelStatus').innerHTML='<span class="online-dot"></span>Hai mô hình đã sẵn sàng';}
  catch(error){$('modelStatus').textContent='Chưa nạp được mô hình';showError(error.message);}
}
setMode('digits');checkModels();

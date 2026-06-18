
const API = 'http://127.0.0.1:8180/detect';

// ═══════════════════════════════════════════════════════
//  TAB SWITCHING
// ═══════════════════════════════════════════════════════
function switchTab(mode) {
  document.querySelectorAll('.tab-btn').forEach((b, i) => {
    b.classList.toggle('active', (i === 0) === (mode === 'image'));
  });
  document.getElementById('panel-image').classList.toggle('active', mode === 'image');
  document.getElementById('panel-video').classList.toggle('active', mode === 'video');

  // stop video when switching away
  if (mode === 'image') stopDetection();
}

// ═══════════════════════════════════════════════════════
//  IMAGE MODE
// ═══════════════════════════════════════════════════════
const dropZone    = document.getElementById('dropZone');
const imgInput    = document.getElementById('imgFileInput');
const imgPreviewW = document.getElementById('img-preview-wrap');
const imgPreview  = document.getElementById('img-preview');
const imgDetBtn   = document.getElementById('imgDetectBtn');
const imgSpinner  = document.getElementById('imgSpinner');
const imgBtnLabel = document.getElementById('imgBtnLabel');
const imgError    = document.getElementById('imgError');
const imgResultSec= document.getElementById('img-result-section');

imgInput.addEventListener('change', () => {
  const f = imgInput.files[0];
  if (f) showImgPreview(f);
});

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) {
    imgInput.files = e.dataTransfer.files;
    showImgPreview(f);
  }
});

function showImgPreview(file) {
  imgPreview.src = URL.createObjectURL(file);
  imgPreviewW.style.display = 'block';
  imgDetBtn.disabled = false;
  imgError.style.display = 'none';
  imgResultSec.style.display = 'none';
}

async function detectImage() {
  const file = imgInput.files[0];
  if (!file) return;

  imgDetBtn.disabled = true;
  imgSpinner.style.display = 'block';
  imgBtnLabel.textContent = 'กำลังวิเคราะห์…';
  imgError.style.display = 'none';

  try {
    const data = await callDetect(file);
    renderImageResult(data);
    imgResultSec.style.display = 'block';
    imgResultSec.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (err) {
    imgError.textContent = '⚠️  ' + err.message;
    imgError.style.display = 'block';
  } finally {
    imgDetBtn.disabled = false;
    imgSpinner.style.display = 'none';
    imgBtnLabel.textContent = '🔍\u00A0\u00A0ตรวจจับทะเบียน';
  }
}

// ═══════════════════════════════════════════════════════
//  IMAGE RESULT RENDERER
// ═══════════════════════════════════════════════════════
function renderImageResult(data) {
  const vehicles = data.vehicles || [];
  const rejected = data.rejected || [];
  const listEl   = document.getElementById('imgVehicleList');
  const imgEl    = document.getElementById('imgResult');

  // count badge
  const countEl = document.getElementById('imgVehicleCount');
  if (countEl) countEl.textContent = vehicles.length;

  // annotated image
  imgEl.src = data.image ? 'data:image/jpeg;base64,' + data.image : '';

  // ── valid vehicles ────────────────────────────────────────
  listEl.innerHTML = '';

  if (vehicles.length === 0) {
    listEl.innerHTML = `
      <div class="no-vehicle">
        <span style="font-size:1.8rem">🔍</span>
        <span>ไม่พบทะเบียนที่สมบูรณ์</span>
      </div>`;
  } else {
    vehicles.forEach((v, i) => {
      const card = document.createElement('div');
      card.className = 'vehicle-row';
      card.innerHTML = `
        <div class="vehicle-index">#${i + 1}</div>
        <div class="vehicle-data">
          <div class="plate-visual" style="display:inline-block;">
            <span class="plate-text" style="font-size:1.4rem;letter-spacing:4px;">${v.license_plate || '—'}</span>
          </div>
        </div>
        <div class="province-pill" style="align-self:center;">
          <span class="dot"></span>
          <span>${v.province || '—'}</span>
        </div>
      `;
      listEl.appendChild(card);
    });
  }

  // ── rejected (debug) ─────────────────────────────────────
  const rejectedWrap    = document.getElementById('imgRejectedWrap');
  const rejectedList    = document.getElementById('imgRejectedList');
  const rejectedCountEl = document.getElementById('imgRejectedCount');

  if (rejected.length > 0) {
    rejectedWrap.style.display = 'block';
    rejectedCountEl.textContent = rejected.length;
    rejectedList.innerHTML = '';
    rejected.forEach(r => {
      const row = document.createElement('div');
      row.className = 'rejected-row';
      row.innerHTML = `
        <span style="font-size:1rem">⚠️</span>
        <span class="rejected-raw">${r.raw_text || '—'}</span>
        <span class="rejected-reason">${r.reason || ''}</span>
      `;
      rejectedList.appendChild(row);
    });
  } else {
    rejectedWrap.style.display = 'none';
  }
}

function toggleRejected() {
  const list = document.getElementById('imgRejectedList');
  const icon = document.getElementById('rejectedToggleIcon');
  const open = list.style.display === 'flex';
  list.style.display = open ? 'none' : 'flex';
  icon.textContent = open ? '▼ แสดง' : '▲ ซ่อน';
}


let videoSource   = 'webcam';   // 'webcam' | 'file'
let stream        = null;
let detectionLoop = null;
let detectInterval= 1500;       // ms
let frameCount    = 0;
let isDetecting   = false;
let lastPlate     = '';         // for dedup

const videoEl      = document.getElementById('videoEl');
const videoBox     = document.getElementById('videoBox');
const canvas       = document.getElementById('captureCanvas');
const startBtn     = document.getElementById('startBtn');
const stopBtn      = document.getElementById('stopBtn');
const scanLine     = document.getElementById('scanLine');
const scanCorners  = document.getElementById('scanCorners');
const statusDot    = document.getElementById('statusDot');
const statusText   = document.getElementById('statusText');
const frameCounter = document.getElementById('frameCounter');
const feedList     = document.getElementById('feedList');
const feedEmpty    = document.getElementById('feedEmpty');
const feedCount    = document.getElementById('feedCount');

// Video source toggle
function setVideoSource(src) {
  videoSource = src;
  stopDetection();

  document.getElementById('srcWebcam').classList.toggle('active', src === 'webcam');
  document.getElementById('srcFile').classList.toggle('active', src === 'file');

  const webcamPH    = document.getElementById('webcamPlaceholder');
  const fileZone    = document.getElementById('fileUploadZone');

  webcamPH.style.display = src === 'webcam' ? 'flex' : 'none';
  fileZone.style.display = src === 'file'   ? 'flex' : 'none';
  videoEl.style.display  = 'none';

  startBtn.textContent = src === 'webcam' ? '▶ เริ่มกล้อง' : '▶ เริ่มวิดีโอ';
}

// File video input
document.getElementById('vidFileInput').addEventListener('change', function() {
  const f = this.files[0];
  if (!f) return;
  const url = URL.createObjectURL(f);
  videoEl.src = url;
  videoEl.style.display = 'block';
  document.getElementById('fileUploadZone').style.display = 'none';
  videoEl.play();
});

// Start button
async function startVideo() {
  if (videoSource === 'webcam') {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
      videoEl.srcObject = stream;
      videoEl.style.display = 'block';
      document.getElementById('webcamPlaceholder').style.display = 'none';
    } catch (e) {
      setStatus('ไม่สามารถเปิดกล้องได้: ' + e.message, false);
      return;
    }
  } else {
    if (!videoEl.src && !videoEl.srcObject) {
      setStatus('กรุณาเลือกไฟล์วิดีโอก่อน', false);
      return;
    }
    videoEl.play();
  }

  beginDetectionLoop();
}

function beginDetectionLoop() {
  isDetecting = true;
  startBtn.disabled = true;
  stopBtn.disabled  = false;
  scanLine.style.display   = 'block';
  scanCorners.style.display= 'block';
  statusDot.classList.add('running');
  setStatus('กำลังตรวจจับ…', true);
  lastPlate = '';

  detectionLoop = setInterval(captureAndDetect, detectInterval);
}

function stopDetection() {
  isDetecting = false;
  clearInterval(detectionLoop);
  detectionLoop = null;

  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
    videoEl.srcObject = null;
    videoEl.style.display = 'none';
    document.getElementById('webcamPlaceholder').style.display = 'flex';
  }

  startBtn.disabled = false;
  stopBtn.disabled  = true;
  scanLine.style.display    = 'none';
  scanCorners.style.display = 'none';
  statusDot.classList.remove('running');
  setStatus('หยุดทำงาน', false);
}

// Capture frame → call API → add to feed
async function captureAndDetect() {
  if (!isDetecting || videoEl.readyState < 2) return;

  // Draw current frame to hidden canvas
  canvas.width  = videoEl.videoWidth  || 640;
  canvas.height = videoEl.videoHeight || 360;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);

  frameCount++;
  frameCounter.textContent = `Frame: ${frameCount}`;

  canvas.toBlob(async (blob) => {
    if (!blob) return;
    try {
      const data = await callDetect(blob);
      const vehicles = data.vehicles || [];
      vehicles.forEach(v => {
        if (v.license_plate && v.license_plate.trim()) {
          addFeedItem(v.license_plate, v.province, data.image);
        }
      });
    } catch (e) {
      // silent fail on individual frames
      console.warn('Frame detect error:', e.message);
    }
  }, 'image/jpeg', 0.85);
}

// Interval slider
function updateInterval(val) {
  detectInterval = parseInt(val);
  document.getElementById('intervalLabel').textContent = (detectInterval / 1000).toFixed(1) + 's';
  if (isDetecting) {
    clearInterval(detectionLoop);
    detectionLoop = setInterval(captureAndDetect, detectInterval);
  }
}

// ═══════════════════════════════════════════════════════
//  FEED
// ═══════════════════════════════════════════════════════
let feedItems = [];

function addFeedItem(plate, province, imgB64) {
  // Dedup: if same plate as last entry, just update its count + time
  if (feedItems.length > 0 && feedItems[0].plate === plate) {
    feedItems[0].count++;
    feedItems[0].time = now();
    feedItems[0].img  = imgB64 || feedItems[0].img;
    renderFeed();
    return;
  }

  feedItems.unshift({ plate, province, img: imgB64, time: now(), count: 1 });
  if (feedItems.length > 100) feedItems.pop(); // cap at 100

  updateFeedCount();
  renderFeed();
}

function renderFeed() {
  if (feedItems.length === 0) {
    feedEmpty.style.display = 'flex';
    // remove all items except empty
    [...feedList.children].forEach(c => { if (c !== feedEmpty) c.remove(); });
    return;
  }

  feedEmpty.style.display = 'none';

  // Rebuild DOM from feedItems array
  const existing = [...feedList.querySelectorAll('.feed-item')];

  // Remove extras
  while (feedList.querySelectorAll('.feed-item').length > feedItems.length) {
    const last = feedList.querySelector('.feed-item:last-child');
    if (last) last.remove();
  }

  feedItems.forEach((item, i) => {
    let el = feedList.querySelectorAll('.feed-item')[i];
    if (!el) {
      el = document.createElement('div');
      el.className = 'feed-item';
      feedList.insertBefore(el, feedList.children[i] || null);
    }

    el.onclick = () => openModal(item);
    el.innerHTML = `
      <img class="feed-thumb" src="${item.img ? 'data:image/jpeg;base64,' + item.img : ''}" alt=""/>
      <div class="feed-info">
        <div class="feed-plate">${item.plate}</div>
        <div class="feed-province">📍 ${item.province || '—'}</div>
      </div>
      <div class="feed-meta">
        <span class="feed-time">${item.time}</span>
        ${item.count > 1 ? `<span class="feed-badge">×${item.count}</span>` : ''}
      </div>
    `;
  });
}

function clearFeed() {
  feedItems = [];
  feedCount.textContent = '0';
  feedEmpty.style.display = 'flex';
  [...feedList.querySelectorAll('.feed-item')].forEach(el => el.remove());
}

function updateFeedCount() {
  feedCount.textContent = feedItems.length;
}

// ═══════════════════════════════════════════════════════
//  MODAL
// ═══════════════════════════════════════════════════════
function openModal(item) {
  document.getElementById('modalImg').src       = item.img ? 'data:image/jpeg;base64,' + item.img : '';
  document.getElementById('modalPlate').textContent    = item.plate;
  document.getElementById('modalProvince').textContent = item.province || '—';
  document.getElementById('modal').classList.add('open');
}

function closeModal() {
  document.getElementById('modal').classList.remove('open');
}

document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

// ═══════════════════════════════════════════════════════
//  SHARED API CALL
// ═══════════════════════════════════════════════════════
async function callDetect(fileOrBlob) {
  const fd = new FormData();
  fd.append('file', fileOrBlob instanceof Blob && !(fileOrBlob instanceof File)
    ? new File([fileOrBlob], 'frame.jpg', { type: 'image/jpeg' })
    : fileOrBlob
  );

  const res = await fetch(API, { method: 'POST', body: fd });
  if (!res.ok) throw new Error(`Server error ${res.status}`);
  return await res.json();
}

// ═══════════════════════════════════════════════════════
//  UTILS
// ═══════════════════════════════════════════════════════
function setStatus(msg, running) {
  statusText.textContent = msg;
  statusDot.classList.toggle('running', running);
}

function now() {
  return new Date().toLocaleTimeString('th-TH', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}
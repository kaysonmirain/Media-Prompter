/**
 * Media Prompter — Frontend Application Logic
 * Handles file upload, WebSocket progress, and rich results rendering.
 */

const API_BASE = window.location.origin;
const WS_BASE  = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}`;

// ── State ─────────────────────────────────────────────────────────────────
let currentTaskId   = null;
let currentWs       = null;
let fullResultData  = null;
let currentTab      = 'detections';

// ── DOM References ────────────────────────────────────────────────────────
const dropZone         = document.getElementById('dropZone');
const fileInput        = document.getElementById('fileInput');
const browseBtn        = document.getElementById('browseBtn');
const heroSection      = document.getElementById('heroSection');
const uploadSection    = document.getElementById('uploadSection');
const processingSection= document.getElementById('processingSection');
const resultsSection   = document.getElementById('resultsSection');
const imagePreview     = document.getElementById('imagePreview');
const videoPreview     = document.getElementById('videoPreview');
const previewBadge     = document.getElementById('previewBadge');
const progressFill     = document.getElementById('progressFill');
const progressLabel    = document.getElementById('progressLabel');
const captionText      = document.getElementById('captionText');
const statsRow         = document.getElementById('statsRow');
const tabsNav          = document.getElementById('tabsNav');
const detectionsGrid   = document.getElementById('detectionsGrid');
const noDetections     = document.getElementById('noDetections');
const sceneBars        = document.getElementById('sceneBars');
const jsonViewer       = document.getElementById('jsonViewer');
const videoPanel       = document.getElementById('videoPanel');
const resetBtn         = document.getElementById('resetBtn');
const copyJsonBtn      = document.getElementById('copyJsonBtn');
const downloadJsonBtn  = document.getElementById('downloadJsonBtn');
const moodContent      = document.getElementById('moodContent');
const tagsContent      = document.getElementById('tagsContent');
const colorsContent    = document.getElementById('colorsContent');
const sceneClipContent = document.getElementById('sceneClipContent');
const resultsImagePreview = document.getElementById('resultsImagePreview');
const resultsVideoPreview = document.getElementById('resultsVideoPreview');
const resultsPreviewBadge = document.getElementById('resultsPreviewBadge');

// ── Health Check ──────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(5000) });
  } catch {
    // Server offline; errors surface when the user uploads a file.
  }
}

// ── Drag & Drop ───────────────────────────────────────────────────────────
dropZone.addEventListener('click', (e) => {
  if (e.target !== browseBtn && !browseBtn.contains(e.target)) {
    fileInput.click();
  }
});
browseBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  fileInput.click();
});

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (file) handleFile(file);
});

resetBtn.addEventListener('click', resetUI);

// ── File Handling ─────────────────────────────────────────────────────────
function handleFile(file) {
  const isImage = file.type.startsWith('image/');
  const isVideo = file.type.startsWith('video/');

  if (!isImage && !isVideo) {
    alert('Unsupported file type. Please upload an image or video.');
    return;
  }

  // Show preview
  showProcessingUI(file, isImage);

  // Upload and analyze
  uploadFile(file);
}

function showProcessingUI(file, isImage) {
  if (heroSection) heroSection.classList.add('hidden');
  uploadSection.classList.add('hidden');
  processingSection.classList.remove('hidden');
  resultsSection.classList.add('hidden');

  previewBadge.textContent = isImage ? 'IMAGE' : 'VIDEO';

  if (isImage) {
    imagePreview.src = URL.createObjectURL(file);
    imagePreview.classList.remove('hidden');
    videoPreview.classList.add('hidden');
  } else {
    videoPreview.src = URL.createObjectURL(file);
    videoPreview.classList.remove('hidden');
    imagePreview.classList.add('hidden');
  }

  resetProgress();
}

function resetProgress() {
  progressFill.style.width = '0%';
  progressLabel.textContent = 'Uploading file...';
}

// ── Upload ────────────────────────────────────────────────────────────────
async function uploadFile(file) {
  const formData = new FormData();
  formData.append('file', file);

  updateProgress('Analyzing... (ETA ~15 seconds)', 30);

  try {
    const res = await fetch(`${API_BASE}/analyze`, {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Upload failed');
    }

    const data = await res.json();
    handleResults(data);
  } catch (err) {
    handleError(err.message);
  }
}

// ── Progress Updates ──────────────────────────────────────────────────────
function updateProgress(step, percent) {
  progressLabel.textContent = step;
  progressFill.style.width = `${percent}%`;
}

// ── Results ───────────────────────────────────────────────────────────────
function handleResults(data) {
  fullResultData = data;
  progressFill.style.width = '100%';
  progressLabel.textContent = 'Analysis complete';

  setTimeout(() => {
    processingSection.classList.add('hidden');
    showResultsPreview();
    renderResults(data);
    resultsSection.classList.remove('hidden');
  }, 600);
}

function showResultsPreview() {
  const isImage = !imagePreview.classList.contains('hidden');
  resultsPreviewBadge.textContent = previewBadge.textContent;

  if (isImage) {
    resultsImagePreview.src = imagePreview.src;
    resultsImagePreview.classList.remove('hidden');
    resultsVideoPreview.classList.add('hidden');
    resultsVideoPreview.removeAttribute('src');
    resultsVideoPreview.load();
  } else {
    resultsVideoPreview.src = videoPreview.src;
    resultsVideoPreview.classList.remove('hidden');
    resultsImagePreview.classList.add('hidden');
    resultsImagePreview.removeAttribute('src');
  }
}

function renderResults(data) {
  const isVideo = data.type === 'video';

  // Caption
  const caption = isVideo
    ? (data.frame_results?.[0]?.caption?.text ?? 'Video analyzed successfully.')
    : (data.caption?.text ?? 'N/A');
  captionText.textContent = formatCaption(caption);

  // Stats
  renderStats(data, isVideo);

  // Tab: show/hide video tab
  const videoTab = document.getElementById('tab-video');
  if (isVideo) {
    videoTab.classList.remove('hidden');
  } else {
    videoTab.classList.add('hidden');
  }

  // Detections
  renderDetections(isVideo ? aggregateDetections(data) : data.detections || []);

  // Semantic
  const semanticSrc = isVideo
    ? (data.frame_results?.find(f => f.semantic_tags)  ?? {})
    : data;
  renderSemantic(semanticSrc);

  // Scene
  const sceneSrc = isVideo
    ? (data.frame_results?.find(f => f.scene_classifications)?.scene_classifications ?? [])
    : (data.scene_classifications ?? []);
  renderSceneBars(sceneSrc);

  // Video panel
  if (isVideo) renderVideoPanel(data);

  // JSON (strip internal model metadata)
  jsonViewer.textContent = JSON.stringify(sanitizeForExport(data), null, 2);

  // Switch to detections tab
  switchTab('detections');
}

function renderStats(data, isVideo) {
  statsRow.innerHTML = '';
  const stats = [];

  if (isVideo) {
    const s = data.summary ?? {};
    stats.push(
      { value: s.frames_analyzed ?? 0, label: 'Frames Analyzed' },
      { value: s.total_detections ?? 0, label: 'Total Detections' },
      { value: s.unique_objects_seen ?? 0, label: 'Unique Objects' },
      { value: formatDuration(data.metadata?.duration_seconds), label: 'Video Duration' },
      { value: data.metadata?.resolution ?? 'N/A', label: 'Resolution' },
      { value: s.motion_level?.split(' ')[0] ?? 'N/A', label: 'Motion Level' }
    );
  } else {
    const s = data.summary ?? {};
    stats.push(
      { value: s.total_objects_detected ?? 0, label: 'Objects Detected' },
      { value: s.unique_object_types ?? 0, label: 'Unique Types' },
      { value: (data.scene_classifications?.[0]?.label ?? 'N/A').split(' ').slice(0,2).join(' '), label: 'Top Scene' },
      { value: data.mood ?? 'N/A', label: 'Mood' },
      { value: `${s.analysis_time_seconds ?? '?'}s`, label: 'Analysis Time' },
      { value: `${data.scene_classifications?.[0]?.confidence ?? 0}%`, label: 'Scene Confidence' }
    );
  }

  stats.forEach(s => {
    const card = document.createElement('div');
    const text = String(s.value ?? 'N/A');
    const len = text.length;
    let sizeClass = '';
    let valueHtml = escHtml(text);

    const resMatch = text.match(/^(\d+)\s*[x×]\s*(\d+)$/i);
    if (resMatch) {
      sizeClass = 'stat-card--resolution';
      valueHtml = `<span>${resMatch[1]}</span><span class="stat-res-x">×</span><span>${resMatch[2]}</span>`;
    } else if (len >= 9) {
      sizeClass = 'stat-card--lg';
    } else if (len > 5) {
      sizeClass = 'stat-card--md';
    }

    card.className = `stat-card${sizeClass ? ` ${sizeClass}` : ''}`;
    card.innerHTML = `
      <div class="stat-value" title="${escHtml(text)}">${valueHtml}</div>
      <div class="stat-label">${escHtml(s.label)}</div>
    `;
    statsRow.appendChild(card);
    fitStatValue(card);
  });

  requestAnimationFrame(() => {
    statsRow.querySelectorAll('.stat-card').forEach(fitStatValue);
  });
}

function fitStatValue(card) {
  const valueEl = card.querySelector('.stat-value');
  if (!valueEl || card.classList.contains('stat-card--resolution')) return;

  const maxW = card.clientWidth - 8;
  if (maxW <= 0) return;

  let size = card.classList.contains('stat-card--lg') ? 16
    : card.classList.contains('stat-card--md') ? 20
    : 28;

  valueEl.style.fontSize = `${size}px`;
  let guard = 0;
  while (size > 11 && valueEl.scrollWidth > maxW && guard < 20) {
    size -= 1;
    valueEl.style.fontSize = `${size}px`;
    guard++;
  }
}

function aggregateDetections(videoData) {
  const seen = new Map();
  for (const frame of videoData.frame_results ?? []) {
    for (const det of frame.detections ?? []) {
      const existing = seen.get(det.label);
      if (!existing || det.confidence > existing.confidence) {
        seen.set(det.label, det);
      }
    }
  }
  return [...seen.values()].sort((a, b) => b.confidence - a.confidence);
}

function renderDetections(detections) {
  detectionsGrid.innerHTML = '';

  if (!detections || detections.length === 0) {
    noDetections.classList.remove('hidden');
    return;
  }
  noDetections.classList.add('hidden');

  detections.forEach((det, i) => {
    const card = document.createElement('div');
    card.className = 'detection-card';
    card.style.animationDelay = `${i * 0.04}s`;

    const confColor = det.confidence >= 80 ? 'var(--emerald)'
                    : det.confidence >= 50 ? 'var(--cyan)'
                    : 'var(--purple)';

    card.innerHTML = `
      <div class="det-header">
        <div class="det-label-row">
          <div class="det-label">${escHtml(det.label)}</div>
          <div class="det-category">${escHtml(det.category ?? 'Object')}</div>
        </div>
        <div class="det-conf" style="color:${confColor};border-color:${confColor}33;background:${confColor}18">
          ${det.confidence}%
        </div>
      </div>
      <div class="det-info">${escHtml(det.info ?? '')}</div>
      <div class="det-meta">
        ${det.position ? `<span class="det-tag">${escHtml(det.position)}</span>` : ''}
        ${det.area_percent != null ? `<span class="det-tag">${det.area_percent}% of frame</span>` : ''}
        ${det.bbox ? `<span class="det-tag">${det.bbox.x2 - det.bbox.x1}×${det.bbox.y2 - det.bbox.y1}px</span>` : ''}
      </div>
    `;
    detectionsGrid.appendChild(card);
  });
}

function renderSemantic(data) {
  // Mood
  renderSemItems(moodContent, data.mood_options ?? (data.mood ? [{ label: data.mood, confidence: 100 }] : []));

  // Tags
  renderSemItems(tagsContent, data.semantic_tags ?? []);

  // Colors
  renderSemItems(colorsContent, data.dominant_colors_approx ?? []);

  // CLIP Scene
  renderSemItems(sceneClipContent, data.scene_options ?? (data.scene_type ? [{ label: data.scene_type, confidence: 100 }] : []));
}

function renderSemItems(container, items) {
  container.innerHTML = '';
  if (!items || items.length === 0) {
    container.innerHTML = '<div style="font-size:12px;color:var(--text-3)">No data available</div>';
    return;
  }
  items.forEach(item => {
    const pct = Math.round(item.confidence ?? 0);
    const div = document.createElement('div');
    div.className = 'sem-item';
    div.innerHTML = `
      <div class="sem-label">${escHtml(item.label ?? item)}</div>
      <div class="sem-bar-wrap"><div class="sem-bar" style="width:${pct}%"></div></div>
      <div class="sem-pct">${pct}%</div>
    `;
    container.appendChild(div);
  });
}

function renderSceneBars(classifications) {
  sceneBars.innerHTML = '';
  if (!classifications || classifications.length === 0) {
    sceneBars.innerHTML = '<div style="color:var(--text-3);font-size:13px">No scene data available</div>';
    return;
  }
  const maxConf = classifications[0]?.confidence ?? 100;

  classifications.forEach((cls, i) => {
    const row = document.createElement('div');
    row.className = 'scene-bar-row';
    const barWidth = Math.round((cls.confidence / maxConf) * 100);
    row.innerHTML = `
      <div class="scene-bar-label" title="${escHtml(cls.label)}">${i + 1}. ${escHtml(cls.label)}</div>
      <div class="scene-bar-track">
        <div class="scene-bar-fill" style="width:0%" data-target="${barWidth}"></div>
      </div>
      <div class="scene-bar-pct">${cls.confidence}%</div>
    `;
    sceneBars.appendChild(row);
  });

  // Animate bars
  requestAnimationFrame(() => {
    document.querySelectorAll('.scene-bar-fill').forEach(bar => {
      bar.style.width = bar.dataset.target + '%';
    });
  });
}

function renderVideoPanel(data) {
  videoPanel.innerHTML = '';

  const s = data.summary ?? {};
  const meta = data.metadata ?? {};

  // Summary grid
  const summaryGrid = document.createElement('div');
  summaryGrid.className = 'video-summary-grid';

  const vsItems = [
    { label: 'Duration', value: formatDuration(meta.duration_seconds) },
    { label: 'FPS', value: meta.fps ?? 'N/A' },
    { label: 'Resolution', value: meta.resolution ?? 'N/A' },
    { label: 'Frames Analyzed', value: s.frames_analyzed ?? 0 },
    { label: 'Dominant Scene', value: (s.dominant_scene ?? 'N/A').split(' ').slice(0,3).join(' ') },
    { label: 'Dominant Mood', value: (s.dominant_mood ?? 'N/A').split(' ').slice(0,2).join(' ') },
    { label: 'Motion Level', value: s.motion_level ?? 'N/A' },
    { label: 'Motion Score', value: s.motion_score ?? 'N/A' },
  ];

  vsItems.forEach(item => {
    const card = document.createElement('div');
    card.className = 'vs-card';
    card.innerHTML = `<div class="vs-label">${item.label}</div><div class="vs-value">${item.value}</div>`;
    summaryGrid.appendChild(card);
  });
  videoPanel.appendChild(summaryGrid);

  // Top objects
  if (s.top_objects?.length) {
    const title = document.createElement('div');
    title.className = 'video-section-title';
    title.textContent = 'Most Detected Objects';
    videoPanel.appendChild(title);

    const grid = document.createElement('div');
    grid.className = 'top-objects-grid';
    s.top_objects.slice(0, 10).forEach(obj => {
      const card = document.createElement('div');
      card.className = 'top-obj-card';
      card.innerHTML = `
        <div class="top-obj-count">${obj.count}</div>
        <div class="top-obj-label">${escHtml(obj.label)}</div>
        <div style="font-size:10px;color:var(--purple);margin-top:3px">${obj.frequency_percent}% of frames</div>
      `;
      grid.appendChild(card);
    });
    videoPanel.appendChild(grid);
  }

  // Timeline
  const timeline = data.timeline ?? [];
  if (timeline.length) {
    const title2 = document.createElement('div');
    title2.className = 'video-section-title';
    title2.textContent = 'Detection Timeline';
    videoPanel.appendChild(title2);

    const list = document.createElement('div');
    list.className = 'timeline-list';
    timeline.slice(0, 20).forEach(event => {
      const item = document.createElement('div');
      item.className = 'timeline-item';
      item.innerHTML = `
        <div class="tl-time">${event.timestamp}s</div>
        <div class="tl-event">${event.event.replace(/_/g, ' ')}</div>
        <div class="tl-objects">${(event.objects ?? []).join(', ')}</div>
      `;
      list.appendChild(item);
    });
    videoPanel.appendChild(list);
  }
}

// ── Tab Switching ─────────────────────────────────────────────────────────
tabsNav.addEventListener('click', (e) => {
  const btn = e.target.closest('.tab-btn');
  if (!btn) return;
  switchTab(btn.dataset.tab);
});

function switchTab(tab) {
  currentTab = tab;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === `panel-${tab}`));

  // Animate scene bars when switching to scene tab
  if (tab === 'scene') {
    document.querySelectorAll('.scene-bar-fill').forEach(bar => {
      bar.style.width = bar.dataset.target + '%';
    });
  }
}

// ── JSON Actions ──────────────────────────────────────────────────────────
copyJsonBtn.addEventListener('click', () => {
  if (!fullResultData) return;
  navigator.clipboard.writeText(JSON.stringify(sanitizeForExport(fullResultData), null, 2))
    .then(() => {
      copyJsonBtn.textContent = 'Copied';
      setTimeout(() => { copyJsonBtn.textContent = 'Copy JSON'; }, 2000);
    });
});

downloadJsonBtn.addEventListener('click', () => {
  if (!fullResultData) return;
  const blob = new Blob([JSON.stringify(sanitizeForExport(fullResultData), null, 2)], { type: 'application/json' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url;
  a.download = `media_prompter_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
});

// ── Error Handling ────────────────────────────────────────────────────────
function handleError(message) {
  progressLabel.textContent = `Error: ${message}`;
  progressFill.style.background = 'linear-gradient(90deg, #ef4444, #f97316)';
  progressFill.style.width = '100%';

  setTimeout(() => {
    alert(`Analysis failed: ${message}\n\nMake sure the backend server is running:\n  cd backend && python main.py`);
    resetUI();
  }, 1500);
}

// ── Reset ─────────────────────────────────────────────────────────────────
function resetUI() {
  if (currentWs) { currentWs.close(); currentWs = null; }
  currentTaskId = null;
  fullResultData = null;

  fileInput.value = '';
  imagePreview.src = '';
  videoPreview.src = '';
  imagePreview.classList.add('hidden');
  videoPreview.classList.add('hidden');
  resultsImagePreview.src = '';
  resultsVideoPreview.src = '';
  resultsImagePreview.classList.add('hidden');
  resultsVideoPreview.classList.add('hidden');

  progressFill.style.width = '0%';
  progressFill.style.background = 'linear-gradient(90deg, var(--violet), var(--cyan))';

  if (heroSection) heroSection.classList.remove('hidden');
  uploadSection.classList.remove('hidden');
  processingSection.classList.add('hidden');
  resultsSection.classList.add('hidden');
}

// ── Helpers ───────────────────────────────────────────────────────────────
function formatCaption(text) {
  if (!text || typeof text !== 'string') return text ?? 'N/A';
  let t = text.trim().replace(/\s+/g, ' ');
  t = t.replace(/(\w)\s+'\s*([a-z]{1,2})\b/gi, "$1'$2");
  t = t.replace(/\s+'\s*([a-z]{1,2})\b/gi, "'$1");
  t = t.replace(/\s+([.,!?;:])/g, '$1');
  if (t.length) t = t.charAt(0).toUpperCase() + t.slice(1);
  t = t.replace(/([.!?]\s+)([a-z])/g, (_, end, letter) => end + letter.toUpperCase());
  return t.trim();
}

function sanitizeForExport(data) {
  const copy = JSON.parse(JSON.stringify(data));
  if (copy.summary) {
    delete copy.summary.models_used;
    delete copy.summary.device_used;
  }
  if (copy.caption?.model) delete copy.caption.model;
  for (const frame of copy.frame_results ?? []) {
    if (frame.summary) {
      delete frame.summary.models_used;
      delete frame.summary.device_used;
    }
    if (frame.caption?.model) delete frame.caption.model;
  }
  return copy;
}

function escHtml(str) {
  if (str == null) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function formatDuration(sec) {
  if (sec == null || isNaN(sec)) return 'N/A';
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

// ── Init ──────────────────────────────────────────────────────────────────
checkHealth();

// Re-check health every 30s
setInterval(checkHealth, 30000);

if (typeof ResizeObserver !== 'undefined') {
  new ResizeObserver(() => {
    document.querySelectorAll('.stat-card').forEach(fitStatValue);
  }).observe(statsRow);
}

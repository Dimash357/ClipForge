// Cartoon Studio (3D): create page + project page.
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const csrf = () => decodeURIComponent((document.cookie.split('; ').find(x => x.startsWith('csrftoken=')) || '').split('=')[1] || '');
  const post = async (url, body) => {
    const r = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json', 'X-CSRFToken': csrf()}, body: JSON.stringify(body || {})});
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);
    return d;
  };
  const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
  const fillSettings = (form, st) => {
    for (const [k, v] of Object.entries(st || {})) {
      const el = form.elements[k];
      if (!el) continue;
      if (el.type === 'checkbox') el.checked = !!v; else el.value = v;
    }
  };
  const readSettings = form => {
    const out = {};
    for (const k of ['aspect', 'resolution', 'quality', 'fps', 'transition', 'motion', 'music', 'music_volume', 'music_dir', 'sfx_dir', 'sfx', 'captions', 'voice']) {
      const el = form.elements[k];
      if (el) out[k] = el.type === 'checkbox' ? el.checked : (k === 'fps' ? +el.value : k === 'music_volume' ? +el.value : el.value);
    }
    return out;
  };

  // ---------------------------------------------------------------- create page
  const form = $('#cs-form');
  if (form) {
    const def = JSON.parse(($('#cs-defaults') || {}).textContent || '{}');
    fillSettings(form, def);
    let mode = 'oneclick';
    form.querySelectorAll('button[type=submit]').forEach(b => b.addEventListener('click', () => { mode = b.dataset.mode; }));
    form.addEventListener('submit', async e => {
      e.preventDefault();
      const f = form.elements;
      $('#cs-msg').textContent = 'Создаём…';
      try {
        const d = await post('/api/studio/create/', {idea: f.idea.value, format: f.format.value, language: f.language.value,
          duration: +f.duration.value, style: f.style.value, audience: f.audience.value, mode, settings: readSettings(form)});
        location.href = d.url;
      } catch (err) { $('#cs-msg').textContent = err.message; }
    });
    return;
  }

  // ---------------------------------------------------------------- project page
  const ID = window.CS_ID;
  if (!ID) return;
  const ICON = {done: '✓', pending: '○', stale: '↻', running: '◌', error: '✕', skipped: '–'};
  const ST_RU = {done: 'готово', pending: 'ожидает', stale: 'устарело', running: 'идёт', error: 'ошибка', skipped: 'выключено'};
  let data = null, settingsLoaded = false, metaLoaded = false, timer = null;
  const busy = () => data && ((data.job && ['queued', 'running', 'paused'].includes(data.job.status)) || data.media_job);

  const run = async body => {
    $('#cs-msg').textContent = '';
    try { await post(`/api/studio/${ID}/run/`, body); } catch (err) { $('#cs-msg').textContent = err.message; }
    load();
  };
  const media = async body => {
    try { await post(`/api/scripts/${ID}/media/`, body); } catch (err) { $('#cs-msg').textContent = err.message; }
    load();
  };

  function render() {
    const d = data;
    $('#cs-title').textContent = d.title || d.idea.slice(0, 80);
    $('#cs-sub').textContent = `${d.format || ''} · ${d.scenes} сцен · ${d.settings.aspect} ${d.settings.resolution}p` +
      (d.media.frames ? ` · кадров ${d.media.images}/${d.media.frames} · реплик ${d.media.voiced}/${d.media.lines}` : '');
    const b = busy();
    $('#cs-all').disabled = !!b;
    $('#cs-cancel').hidden = !(d.job && ['queued', 'running', 'paused'].includes(d.job.status));
    const j = d.job;
    $('#cs-bar').style.width = (j && ['queued', 'running', 'paused'].includes(j.status) ? j.progress : (d.output_url && d.stages.render.status === 'done' ? 100 : 0)) + '%';
    $('#cs-job').textContent = d.media_job ? `Медиа: ${d.media_job.message}` : j ? (['queued', 'running', 'paused'].includes(j.status) ?
      `${j.stage || ''} — ${j.message || ''}` : (d.error ? `Ошибка: ${d.error}` : '')) : '';
    $('#cs-stages').innerHTML = d.order.map(k => {
      const s = d.stages[k];
      const btn = s.status === 'done' ? `<button class="sm" data-redo="${k}">Переделать</button>` :
        s.status === 'skipped' ? '' : `<button class="sm" data-only="${k}">Запустить</button>`;
      return `<div class="cs-stage st-${s.status}"><span class="cs-ic">${ICON[s.status] || '?'}</span><div><b>${esc(d.labels[k])}</b>
        <small>${ST_RU[s.status] || s.status}${s.message ? ' — ' + esc(s.message) : ''}</small></div>${b ? '' : btn}</div>`;
    }).join('');
    const out = d.output_url;
    const v = $('#cs-video');
    if (out && v.dataset.src !== out + d.output.at) {
      v.dataset.src = out + d.output.at;
      v.innerHTML = `<video controls preload="metadata" src="${out}?v=${d.output.at}"></video>
        <div class="row"><a class="btn sm" href="${out}" download>Скачать MP4</a><span class="hint">${esc(d.output.size)} · ${(d.output.bytes / 1048576).toFixed(1)} МБ · ${esc(d.output.encoder)}${d.stages.render.status !== 'done' ? ' · <b>устарело — нажмите «Продолжить»</b>' : ''}</span></div>`;
    }
    if (!settingsLoaded) { fillSettings($('#cs-settings'), d.settings); settingsLoaded = true; }
    if (!metaLoaded && d.meta.title) {
      const f = $('#cs-meta').elements;
      f.title.value = d.meta.title; f.description.value = d.meta.description || ''; f.hashtags.value = (d.meta.hashtags || []).join(' ');
      metaLoaded = true;
    }
    $('#cs-frames').innerHTML = d.frames.map((f, i) => {
      const img = f.image || {};
      const pic = img.path && !img.error ? `<img src="${d.media_url}${img.path}" loading="lazy">` : `<span class="muted">${img.error ? '✕ ' + esc(img.error).slice(0, 90) : 'нет изображения'}</span>`;
      return `<div class="cs-frame"><div class="cs-pic">${pic}</div><small>${i + 1}. ${esc(f.shot || '')} · ${esc(f.camera || '')}${img.manual ? ' · своё' : ''}</small>
        ${b ? '' : `<button class="sm" data-frame="${f.id}">↻ Перегенерировать</button>`}</div>`;
    }).join('') || '<span class="muted">Появятся после раскадровки.</span>';
    $('#cs-lines').innerHTML = d.lines.map(l => {
      const v = l.voice || {};
      const au = v.path && !v.error ? `<audio controls preload="none" src="${d.media_url}${v.path}"></audio><small>${v.duration}s · ${esc(v.voice || '')}</small>` : `<small class="muted">${v.error ? '✕ ' + esc(v.error).slice(0, 90) : 'не озвучено'}</small>`;
      return `<div class="cs-line"><div><b>${esc(l.speaker)}</b>: ${esc(l.text)}</div><div class="row">${au}${b ? '' : `<button class="sm" data-line="${l.scene}|${l.li}">↻ Переозвучить</button>`}</div></div>`;
    }).join('') || '<span class="muted">Появится после сценария.</span>';
  }

  async function load() {
    clearTimeout(timer);
    try {
      data = await (await fetch(`/api/studio/${ID}/`)).json();
      render();
    } catch (err) { $('#cs-msg').textContent = err.message; }
    timer = setTimeout(load, busy() ? 1500 : 8000);
  }

  $('#cs-all').addEventListener('click', () => run({}));
  $('#cs-cancel').addEventListener('click', async () => { if (data.job) await post(`/api/jobs/${data.job.id}/cancel/`); load(); });
  document.addEventListener('click', e => {
    const t = e.target.closest('button');
    if (!t) return;
    if (t.dataset.only) run({only: [t.dataset.only]});
    else if (t.dataset.redo) { if (confirm('Переделать этап? Следующие этапы потом пересоберутся.')) run({only: [t.dataset.redo], force: [t.dataset.redo]}); }
    else if (t.dataset.frame) media({what: 'frames', ids: [t.dataset.frame], force: true});
    else if (t.dataset.line) { const [s, li] = t.dataset.line.split('|'); media({what: 'lines', ids: [[s, +li]], force: true}); }
  });
  $('#cs-settings').addEventListener('submit', async e => {
    e.preventDefault();
    try { await post(`/api/studio/${ID}/settings/`, {settings: readSettings(e.target)}); $('#cs-msg').textContent = 'Настройки сохранены.'; } catch (err) { $('#cs-msg').textContent = err.message; }
    load();
  });
  $('#cs-meta').addEventListener('submit', async e => {
    e.preventDefault();
    const f = e.target.elements;
    try { await post(`/api/studio/${ID}/meta/`, {title: f.title.value, description: f.description.value, hashtags: f.hashtags.value}); $('#cs-meta-msg').textContent = 'Сохранено.'; } catch (err) { $('#cs-meta-msg').textContent = err.message; }
  });
  $('#cs-meta-regen').addEventListener('click', () => { metaLoaded = false; run({only: ['meta'], force: ['meta']}); });
  load();
})();

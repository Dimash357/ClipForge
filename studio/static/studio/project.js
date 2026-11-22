(() => {
const PID = window.CF.pid;
const $ = id => document.getElementById(id);
const csrf = () => decodeURIComponent((document.cookie.split('; ').find(x => x.startsWith('csrftoken=')) || '').split('=')[1] || '');
const post = (url, body) => fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json', 'X-CSRFToken': csrf()}, body: JSON.stringify(body || {})});
function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === 'class') e.className = v; else if (k === 'text') e.textContent = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v); else if (v !== false && v != null) e.setAttribute(k, v === true ? '' : v);
  }
  for (const c of kids.flat()) if (c != null) e.append(c.nodeType ? c : document.createTextNode(c));
  return e;
}
const TYPES = {reaction_cam: 'Вебка + игра', action: 'Игра (за действием)', fullframe: 'Весь кадр', gameplay: 'Игра', game_cam: 'Игра + камера', talking: 'Автор', screen: 'Экран', smart: 'Smart', center: 'Центр'};
const MODES = {auto: 'AI Auto', reaction: 'Reaction Priority', fullframe: 'Без обрезки', director: 'Auto Scene Director', smart: 'Smart Reframe', gameplay_facecam: 'Gameplay Priority', gameplay: 'Gameplay', center: 'Центр'};
const STATUS = {done: 'Готово', error: 'Ошибка', cancelled: 'Остановлено', interrupted: 'Прервано', review: 'Ждёт подтверждения', processing: 'В работе', fetching: 'Загрузка', queued: 'В очереди'};
let data = null, timer = null;
const cards = new Map(), cand = new Map();

async function poll() {
  clearTimeout(timer);
  try {
    const r = await fetch(`/api/projects/${PID}/status/`); data = await r.json(); render(data);
    timer = setTimeout(poll, data.active ? 1200 : (['processing', 'fetching', 'queued'].includes(data.status) ? 2000 : 6000));
  } catch (e) { timer = setTimeout(poll, 4000); }
}

function render(d) {
  const badge = $('statusbadge'); badge.textContent = STATUS[d.status] || d.status; badge.className = 'badge b-' + d.status;
  $('message').textContent = d.message || STATUS[d.status] || d.status;
  for (const s of d.stages) {
    const el = document.querySelector(`.step[data-key="${s.key}"]`); if (!el) continue;
    el.className = 'step ' + s.state;
    const bar = el.querySelector('.bar');
    if (s.state === 'active') {
      const known = d.stage_progress != null; el.classList.toggle('indet', !known); bar.style.width = known ? Math.round(d.stage_progress * 100) + '%' : '';
    } else { el.classList.remove('indet'); bar.style.width = s.state === 'done' ? '100%' : '0'; }
  }
  const act = d.status === 'processing' || d.status === 'fetching' || d.status === 'queued';
  $('submsg').textContent = act && d.stage_progress != null ? `Текущий этап: ${Math.round(d.stage_progress * 100)}%` : (act ? 'Идёт этап без точного прогресса — индикатор показывает активность.' : '');
  const err = $('error'); err.style.display = d.error ? 'block' : 'none'; err.textContent = d.error || '';
  const actions = $('actions'); actions.replaceChildren();
  if (act) actions.append(h('button', {class: 'danger', text: d.cancel_requested ? 'Останавливаем…' : '■ Остановить', disabled: d.cancel_requested, onclick: cancel}));
  else {
    if (['error', 'cancelled', 'interrupted'].includes(d.status)) actions.append(h('button', {class: 'primary', text: '↻ Продолжить (без повторной загрузки и транскрибации)', onclick: () => resume({})}));
    if (d.status === 'done') actions.append(h('button', {text: '↻ Подобрать моменты заново', onclick: () => resume({reprocess: true, skip_review: true})}));
    if (d.transcript_meta && d.transcript_meta.device && !act) actions.append(h('span', {class: 'hint', text: `Whisper: ${d.transcript_meta.model} · ${String(d.transcript_meta.device).toUpperCase()}`}));
    const a = d.analysis;
    if (a && a.probe && !act) actions.append(h('span', {class: 'hint', text: `Видео: ${a.probe.width}×${a.probe.height}, ${Math.round(a.probe.fps)} FPS, ${Math.round(a.probe.duration)} с · язык: ${a.language || 'авто'} · вебка: ${a.cam ? a.cam.replace('_', ' ') : (a.cam_checked ? 'не найдена' : 'не проверялась')}`}));
  }
  $('progresspanel').style.display = (d.status === 'done' && !d.error && d.clips.length) ? 'none' : 'block';
  if (d.status === 'done' && d.clips.some(c => c.status === 'error')) $('progresspanel').style.display = 'block';
  renderReview(d); renderClips(d);
}

async function cancel() { await post(`/api/projects/${PID}/cancel/`); poll(); }
async function resume(body) { const r = await post(`/api/projects/${PID}/resume/`, body); if (!r.ok) alert((await r.json()).error || 'Не удалось запустить'); poll(); }

// ---------------------------------------------------------------- review (candidates)
function renderReview(d) {
  const panel = $('reviewpanel'), list = $('candidates');
  const cs = d.clips.filter(c => c.status === 'candidate');
  panel.style.display = d.status === 'review' && cs.length ? 'block' : 'none';
  if (panel.style.display === 'none') { cand.clear(); list.replaceChildren(); return; }
  for (const c of cs) { if (!cand.has(c.id)) { const el = candCard(c, d); cand.set(c.id, el); list.append(el); } }
}
function scoreBars(sc) {
  if (!sc || sc.hook == null) return null;
  const names = {hook: 'Hook', context: 'Context', engagement: 'Engagement', completeness: 'Completeness', reaction: 'Реакция', audio: 'Звук/эмоции', action: 'Динамика'};
  const chips = Object.entries(names).filter(([k]) => sc[k] != null).map(([k, n]) => h('span', {class: 'chip', text: `${n} ${sc[k]}`}));
  if (sc.interest != null) chips.unshift(h('span', {class: 'chip ' + (sc.interest >= 70 ? 'q-ok' : sc.interest >= 50 ? 'q-warn' : ''), text: `Интересность ${sc.interest}/100`}));
  return h('div', {class: 'row'}, chips);
}
// bounds editor: number inputs, +/- nudges, "from player", restore the detected bounds
function boundsEditor(c, player) {
  const s = h('input', {type: 'number', step: '0.1', min: '0', value: c.start.toFixed(1)}), e = h('input', {type: 'number', step: '0.1', min: '0', value: c.end.toFixed(1)});
  const nudge = (inp, d) => { inp.value = Math.max(0, parseFloat(inp.value || '0') + d).toFixed(1); upd(); };
  const len = h('span', {class: 'hint'});
  const upd = () => { len.textContent = `Длина: ${(parseFloat(e.value) - parseFloat(s.value)).toFixed(1)} с`; };
  s.addEventListener('input', upd); e.addEventListener('input', upd); upd();
  const auto = (c.scores && c.scores.auto_bounds) || null;
  const snap = h('input', {type: 'checkbox', checked: true});
  const box = h('div', {},
    h('div', {class: 'grid2'},
      h('div', {}, h('label', {text: 'Начало, с'}), s, h('div', {class: 'row'},
        h('button', {class: 'sm', type: 'button', text: '−1', onclick: () => nudge(s, -1)}), h('button', {class: 'sm', type: 'button', text: '+1', onclick: () => nudge(s, 1)}),
        h('button', {class: 'sm', type: 'button', text: '⇤ из плеера', onclick: () => { s.value = player().currentTime.toFixed(1); upd(); }}))),
      h('div', {}, h('label', {text: 'Конец, с'}), e, h('div', {class: 'row'},
        h('button', {class: 'sm', type: 'button', text: '−1', onclick: () => nudge(e, -1)}), h('button', {class: 'sm', type: 'button', text: '+1', onclick: () => nudge(e, 1)}),
        h('button', {class: 'sm', type: 'button', text: '⇥ из плеера', onclick: () => { e.value = player().currentTime.toFixed(1); upd(); }})))),
    h('div', {class: 'row'}, len,
      auto ? h('button', {class: 'sm', type: 'button', text: `↺ Найденные границы (${auto[0].toFixed(1)}–${auto[1].toFixed(1)})`, onclick: () => { s.value = auto[0].toFixed(1); e.value = auto[1].toFixed(1); upd(); }}) : null,
      h('label', {class: 'check'}, snap, ' не резать слова')));
  box._get = () => ({start: parseFloat(s.value), end: parseFloat(e.value), snap: snap.checked});
  box._s = s; box._e = e;
  return box;
}
function candCard(c, d) {
  const v = h('video', {controls: true, preload: 'metadata', src: `${d.source_url}#t=${c.start},${c.end}`});
  const be = boundsEditor(c, () => v);
  const chk = h('input', {type: 'checkbox', checked: true});
  v.addEventListener('timeupdate', () => { const end = parseFloat(be._e.value); if (v.currentTime >= end) { v.pause(); v.currentTime = parseFloat(be._s.value); } });
  const el = h('div', {class: 'clip', 'data-id': c.id}, v,
    h('label', {class: 'check'}, chk, ` Рендерить · Score ${c.score}`), be,
    scoreBars(c.scores), h('div', {class: 'why', text: c.reason}), h('div', {class: 'why', text: c.text}));
  el._get = () => ({id: c.id, ...be._get(), on: chk.checked});
  return el;
}
$('rv_go').addEventListener('click', async () => {
  const picked = [...cand.values()].map(x => x._get()).filter(x => x.on);
  if (!picked.length) { alert('Отметьте хотя бы один фрагмент.'); return; }
  const r = await post(`/api/projects/${PID}/render/`, {clips: picked, discard_others: true, snap: picked.some(x => x.snap), mode: $('rv_mode').value, overrides: {subtitle_style: $('rv_style').value}});
  if (!r.ok) alert((await r.json()).error); cand.clear(); $('candidates').replaceChildren(); poll();
});

// ---------------------------------------------------------------- results
function timeline(c) {
  if (!c.segments || c.segments.length < 1) return null;
  const total = c.segments[c.segments.length - 1].end - c.segments[0].start || 1;
  const bar = h('div', {class: 'tl'}, c.segments.map(s => h('span', {class: 't-' + s.type, style: `flex:${(s.end - s.start) / total}`,
    title: `${TYPES[s.type] || s.type} · ${s.start.toFixed(1)}–${s.end.toFixed(1)} с` + (s.confidence != null ? ` · уверенность ${Math.round(s.confidence * 100)}%` : '') + (s.fallback ? ' · запасной вариант' : '')})));
  const used = [...new Set(c.segments.map(s => s.type))];
  return h('div', {}, bar, h('div', {class: 'legend'}, used.map(t => h('span', {}, h('span', {class: 't-' + t, style: 'display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:4px'}), TYPES[t] || t))));
}
function qualityChip(q) {
  if (!q || q.score == null) return h('span', {class: 'chip', text: 'Проверка…'});
  return h('span', {class: 'chip ' + (q.score >= 85 ? 'q-ok' : q.score >= 60 ? 'q-warn' : 'q-bad'), text: `Качество ${q.score}/100`});
}
async function copyText(text) {
  try { await navigator.clipboard.writeText(text); return true; } catch (e) {
    const ta = h('textarea', {style: 'position:fixed;left:-9999px'}); ta.value = text; document.body.append(ta); ta.select();
    let ok = false; try { ok = document.execCommand('copy'); } catch (e2) {} ta.remove(); return ok;
  }
}
function copyBtn(text, label) {
  return h('button', {class: 'secondary', type: 'button', text: label || text, title: 'Копировать', onclick: async ev => {
    const b = ev.currentTarget; const ok = await copyText(text); b.dataset.o = b.dataset.o || b.textContent;
    b.textContent = ok ? '✓ Скопировано' : 'Не удалось скопировать'; setTimeout(() => b.textContent = b.dataset.o, 1200); }});
}
function clipCard(c) {
  const media = c.url ? h('video', {controls: true, preload: 'metadata', poster: c.poster || false, src: c.url}) :
    h('div', {class: 'ph', text: c.status === 'error' ? 'Ошибка рендера' : c.status === 'rendering' ? 'Рендерится…' : c.status === 'queued' ? 'В очереди' : 'Ещё не отрендерен'});
  const meta = c.meta || {};
  const kids = [media,
    h('div', {class: 'row'}, h('span', {class: 'chip', text: `${c.duration} с`}),
      h('span', {class: 'chip', title: c.auto ? c.auto.reason : '', text: c.auto ? `AI Auto → ${c.auto.label}` : (MODES[c.mode] || c.mode)}), c.quality ? qualityChip(c.quality) : null,
      c.fallback_renderer ? h('span', {class: 'chip q-warn', text: 'резервный рендер v4'}) : null, h('span', {class: 'chip', text: `Score ${c.score}`})),
    h('h3', {text: c.title})];
  if (c.status === 'ready') kids.push(shortMeta(c));
  if (c.error) kids.push(h('div', {class: 'error', text: c.error}));
  for (const n of (c.notes || [])) kids.push(h('div', {class: 'notice', text: 'ℹ ' + n}));
  const tl = timeline(c); if (tl) kids.push(tl);
  if (c.reason) kids.push(h('div', {class: 'why', text: c.reason}));
  if (c.scores && c.scores.hook != null) kids.push(scoreBars(c.scores));
  if (meta.titles && meta.titles.length) {
    kids.push(h('details', {}, h('summary', {text: 'Названия, описание, хештеги'}),
      h('div', {class: 'titles'}, meta.titles.map(t => copyBtn(t))),
      meta.titles_en && meta.lang !== 'en' ? h('div', {class: 'titles', style: 'margin-top:8px'}, h('small', {text: 'English'}), meta.titles_en.map(t => copyBtn(t))) : null,
      meta.titles_ru && meta.lang !== 'ru' ? h('div', {class: 'titles', style: 'margin-top:8px'}, h('small', {text: 'Русский'}), meta.titles_ru.map(t => copyBtn(t))) : null,
      !meta.bilingual ? h('p', {class: 'hint', text: 'Варианты составлены из речи клипа на языке видео. Двуязычные варианты доступны при включённом USE_OLLAMA=1.'}) : null,
      meta.description ? h('div', {class: 'titles', style: 'margin-top:8px'}, copyBtn(meta.description, '📝 ' + meta.description)) : null,
      meta.hashtags ? h('div', {class: 'tags', text: meta.hashtags.join(' ')}) : null));
  }
  if (c.quality && c.quality.checks && c.quality.checks.length) {
    kids.push(h('details', {}, h('summary', {text: 'Автопроверка качества'}),
      h('div', {class: 'checks'}, c.quality.checks.map(k => h('div', {class: k.status, text: `${k.label}${k.detail ? ' — ' + k.detail : ''}`})))));
  }
  const row = h('div', {class: 'row'});
  if (c.url) row.append(h('a', {class: 'btn', href: c.download, download: '', text: '↓ Скачать MP4'}));
  row.append(h('button', {type: 'button', text: '✎ Перемонтировать', onclick: () => openDialog(c)}), h('button', {type: 'button', text: 'Aa Субтитры', onclick: () => openDialog(c, true)}));
  kids.push(row);
  if (c.encoder) kids.push(h('div', {class: 'hint', text: `Кодирование: ${c.encoder}`}));
  return h('div', {class: 'clip'}, kids);
}
// ---------------------------------------------------------------- Shorts title + description
function shortMeta(c) {
  const s = c.short;
  const box = h('div', {class: 'shortmeta'});
  const err = h('div', {class: 'error', style: 'display:none'});
  const gen = async (btn) => {
    btn.disabled = true; const old = btn.textContent; btn.textContent = '… генерируем';
    try {
      const r = await post(`/api/clips/${c.id}/short-meta/`, {regenerate: true});
      const j = await r.json().catch(() => ({error: `Ошибка сервера (${r.status}). Обновите страницу.`}));
      if (!r.ok) { err.textContent = j.error || 'Ошибка'; err.style.display = 'block'; return; }
      c.short = j.short; c.title = j.title; const card = cards.get(c.id);
      const el = clipCard(c); if (card) { card.el.replaceWith(el); card.el = el; }
    } catch (e) { err.textContent = 'Сервер недоступен'; err.style.display = 'block'; }
    finally { btn.disabled = false; btn.textContent = old; }
  };
  if (!s) {
    box.append(h('div', {class: 'hint', text: 'Название и описание для Shorts ещё не созданы.'}),
      h('div', {class: 'row'}, h('button', {type: 'button', text: '✨ Сгенерировать название и описание', onclick: ev => gen(ev.target)})), err);
    return box;
  }
  box.append(
    h('label', {text: 'Название для YouTube Shorts'}), h('div', {class: 'sm-title', text: s.title}),
    h('label', {text: 'Описание'}), h('div', {class: 'sm-desc', text: s.text || s.description}),
    h('div', {class: 'row'}, copyBtn(s.title, '📋 Скопировать название'), copyBtn(s.text || s.description, '📋 Скопировать описание'),
      h('button', {type: 'button', text: '↻ Сгенерировать заново', onclick: ev => gen(ev.target)})),
    h('div', {class: 'hint', text: s.provider === 'local' ? 'Составлено локально из речи и событий клипа (без API)' : `Сгенерировано AI: ${s.provider}`}), err);
  return box;
}
function renderClips(d) {
  const wrap = $('clips'); const list = d.clips.filter(c => c.status !== 'candidate');
  if (!list.length) return;
  if (wrap.querySelector('p.muted')) wrap.replaceChildren();
  const ids = new Set(list.map(c => c.id));
  for (const [id, v] of cards) if (!ids.has(id)) { v.el.remove(); cards.delete(id); }
  for (const c of list) {
    const sig = JSON.stringify([c.status, c.url, c.title, c.error, c.quality && c.quality.score, c.poster, (c.meta || {}).titles, c.short && c.short.title, c.short && c.short.text, c.auto && c.auto.mode]);
    const cur = cards.get(c.id);
    if (cur && cur.sig === sig) continue;
    const el = clipCard(c);
    if (cur) cur.el.replaceWith(el); else wrap.append(el);
    cards.set(c.id, {sig, el});
  }
}

// ---------------------------------------------------------------- re-render dialog
const dlg = $('dlg'); let dlgClip = null;
function openDialog(c, focusStyle) {
  dlgClip = c; const o = c.opts || {};
  $('dlgtitle').textContent = `Перемонтировать: ${c.title}`;
  const v = $('dlgvideo'); v.src = `${data.source_url}#t=${c.start},${c.end}`;
  $('d_start').value = c.start.toFixed(1); $('d_end').value = c.end.toFixed(1); $('d_mode').value = c.mode;
  const auto = (c.scores && c.scores.auto_bounds) || null; const rb = $('d_restore');
  rb.style.display = auto ? '' : 'none'; if (auto) rb.textContent = `↺ Найденные границы (${auto[0].toFixed(1)}–${auto[1].toFixed(1)})`;
  $('d_corner').value = 'auto'; $('d_style').value = o.subtitle_style; $('d_words').value = o.caption_words || 0;
  $('d_hl').value = o.caption_highlight == null ? 'auto' : (o.caption_highlight ? '1' : '0'); $('d_caps').value = o.captions ? '1' : '0';
  $('d_preset').value = o.preset; $('d_quality').value = o.quality; $('d_bg').value = o.bg_fill; $('d_trans').value = o.transition;
  $('dlgerr').style.display = 'none'; dlg.showModal(); if (focusStyle) $('d_style').focus();
}
for (const [id, inp, d] of [['d_s_m', 'd_start', -1], ['d_s_p', 'd_start', 1], ['d_e_m', 'd_end', -1], ['d_e_p', 'd_end', 1]])
  $(id).onclick = () => { $(inp).value = Math.max(0, parseFloat($(inp).value || '0') + d).toFixed(1); };
$('d_restore').onclick = () => { const a = dlgClip && dlgClip.scores && dlgClip.scores.auto_bounds; if (a) { $('d_start').value = a[0].toFixed(1); $('d_end').value = a[1].toFixed(1); } };
$('d_setstart').onclick = () => $('d_start').value = $('dlgvideo').currentTime.toFixed(1);
$('d_setend').onclick = () => $('d_end').value = $('dlgvideo').currentTime.toFixed(1);
$('d_go').onclick = async () => {
  const body = {start: parseFloat($('d_start').value), end: parseFloat($('d_end').value), snap: $('d_snap').checked, mode: $('d_mode').value, corner: $('d_corner').value,
    overrides: {subtitle_style: $('d_style').value, caption_words: parseInt($('d_words').value || '0', 10), caption_highlight: $('d_hl').value === 'auto' ? null : $('d_hl').value === '1',
      captions: $('d_caps').value === '1', preset: $('d_preset').value, quality: $('d_quality').value, bg_fill: $('d_bg').value, transition: $('d_trans').value}};
  const r = await post(`/api/clips/${dlgClip.id}/rerender/`, body);
  if (!r.ok) { const e = $('dlgerr'); e.textContent = (await r.json()).error || 'Ошибка'; e.style.display = 'block'; return; }
  dlg.close(); $('progresspanel').style.display = 'block'; poll();
};
poll();
})();

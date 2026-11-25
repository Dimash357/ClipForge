// Clip Selection Studio (2D): cards, sort / filter / bulk select, compare, one editor (bounds, Hook, Smart
// Editing, layout, captions) with draft preview, undo, restore, bulk apply and ZIP export. Uses the existing APIs.
(function () {
  const PID = window.CF.pid;
  const $ = id => document.getElementById(id);
  const csrf = () => decodeURIComponent((document.cookie.split('; ').find(x => x.startsWith('csrftoken=')) || '').split('=')[1] || '');
  const post = (url, body) => fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json', 'X-CSRFToken': csrf()}, body: JSON.stringify(body || {})});
  function h(tag, a, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(a || {})) {
      if (k === 'text') e.textContent = v; else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
      else if (v !== false && v != null) e.setAttribute(k, v === true ? '' : v);
    }
    e.append(...kids.filter(k => k != null && k !== false)); return e;
  }
  const MODES = Object.fromEntries(JSON.parse($('cs-modes').textContent));
  const STYLES = JSON.parse($('cs-styles').textContent);
  const FONTS = JSON.parse($('cs-fonts').textContent);
  const HOOK_KIND = {original: 'как есть', start: 'другое начало', teaser: 'тизер в начале'};
  const EDIT = {off: 'Без монтажа', natural: 'Natural', dynamic: 'Dynamic', aggressive: 'Aggressive'};
  const REASON = {pause: 'пауза', silence: 'тишина', dead_air: 'пустой промежуток', repeat: 'повтор', filler: 'запинка', boring: 'неинтересно'};
  const STATUS = {ready: 'Готов', rendering: 'Рендер…', queued: 'В очереди', error: 'Ошибка', candidate: 'Кандидат'};
  const fmt = s => s == null ? '—' : `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, '0')}`;
  let data = null, timer = null;
  const selected = new Set();
  const cards = new Map();

  // ------------------------------------------------------------------ data helpers
  const viral = c => (c.meta && c.meta.viral) || null;
  const hookSel = c => { const hk = c.meta && c.meta.hooks; if (!hk) return null; return (hk.variants || []).find(v => v.id === (hk.selected || 'orig')) || null; };
  const duration = c => c.output_duration || c.duration;
  const keyOf = {viral: c => -(viral(c) ? viral(c).score : -1), hook: c => -(hookSel(c) ? hookSel(c).score : -1), position: c => c.position,
    duration: c => duration(c), score: c => -c.score};

  function visible() {
    const q = $('cs-search').value.trim().toLowerCase(), f = $('cs-filter').value;
    return data.clips.filter(c => c.status !== 'candidate').filter(c => {
      if (q && !(`${c.title} ${c.text}`.toLowerCase().includes(q))) return false;
      if (f === 'ready') return c.status === 'ready';
      if (f === 'work') return c.status === 'rendering' || c.status === 'queued';
      if (f === 'error') return c.status === 'error' || !!c.edit_error;
      if (f === 'unique') return !(viral(c) && viral(c).duplicate_of);
      if (f === 'top') return viral(c) && viral(c).score >= 60;
      return true;
    }).sort((a, b) => keyOf[$('cs-sort').value](a) - keyOf[$('cs-sort').value](b));
  }

  // ------------------------------------------------------------------ status strip
  function renderStatus(d) {
    const box = $('cs-status');
    const act = ['processing', 'fetching', 'queued'].includes(d.status);
    const msg = h('div', {class: 'row', style: 'justify-content:space-between'},
      h('span', {text: d.message || ''}), h('span', {class: `badge b-${d.status}`, text: d.status}));
    const parts = [msg];
    if (d.error) parts.push(h('div', {class: 'error', text: d.error}));
    if (window.renderLive && d.job && (act || d.job.status === 'running')) parts.push(renderLive(d.job, d.load));
    const errs = d.clips.filter(c => c.status === 'error' || c.edit_error);
    if (errs.length) parts.push(h('small', {class: 'livewarn', text: `Проблемы: ${errs.map(c => `${c.title}: ${c.error || c.edit_error}`).join(' · ').slice(0, 400)}`}));
    parts.push(h('small', {class: 'muted'}, 'Очередь и ресурсы: ', h('a', {href: '/queue/', text: 'страница «Очередь задач»'})));
    box.replaceChildren(...parts);
  }

  // ------------------------------------------------------------------ cards
  function badge(cls, text, title) { return h('span', {class: `cs-badge ${cls}`, text, title: title || false}); }
  function scoreCls(x) { return x == null ? '' : x >= 70 ? 'good' : x >= 50 ? 'mid' : 'low'; }
  function card(c) {
    const v = viral(c), hk = hookSel(c);
    const cb = h('input', {type: 'checkbox', class: 'cs-pick'}); cb.checked = selected.has(c.id);
    cb.onchange = () => { if (cb.checked) selected.add(c.id); else selected.delete(c.id); toolbar(); };
    const media = c.url ? h('video', {src: c.url, poster: c.poster || false, muted: true, loop: true, playsinline: true, preload: 'none'})
      : h('div', {class: 'cs-ph', text: STATUS[c.status] || c.status});
    if (c.url) { media.onmouseenter = () => media.play().catch(() => {}); media.onmouseleave = () => media.pause(); media.onclick = () => { media.muted = !media.muted; }; }
    const busy = c.status === 'rendering' || c.status === 'queued';
    return h('div', {class: 'cs-card' + (selected.has(c.id) ? ' sel' : '') + (busy ? ' busy' : '')},
      h('div', {class: 'cs-media'}, media, cb,
        h('div', {class: 'cs-badges'}, v ? badge(scoreCls(v.score), `Viral ${Math.round(v.score)}`, 'Viral Score — оценка сборки Shorts, не прогноз просмотров') : null,
          hk ? badge(scoreCls(hk.score), `Hook ${Math.round(hk.score)}`) : null, v && v.rank ? badge('rank', `#${v.rank}`) : null,
          v && v.duplicate_of ? badge('low', 'дубль') : null),
        h('div', {class: 'cs-dur', text: fmt(duration(c))})),
      h('div', {class: 'cs-info'},
        h('b', {text: c.title, title: c.title}),
        h('small', {class: 'muted', text: [STATUS[c.status] || c.status, c.meta && c.meta.edit && c.meta.edit.mode && c.meta.edit.mode !== 'off' ? `монтаж ${EDIT[c.meta.edit.mode]}` : null,
          hk && hk.kind !== 'original' ? HOOK_KIND[hk.kind] : null, MODES[c.mode] ? MODES[c.mode].split(' (')[0] : c.mode].filter(Boolean).join(' · ')}),
        c.error ? h('small', {class: 'error', text: c.error.slice(0, 160)}) : null,
        h('div', {class: 'row'}, h('button', {class: 'sm', type: 'button', text: '✎ Редактировать', onclick: () => openEditor(c)}),
          c.download ? h('a', {class: 'btn sm', href: c.download, download: '', text: '⬇'}) : null)));
  }
  function renderGrid() {
    const list = visible(), grid = $('cs-grid');
    const ids = new Set(list.map(c => c.id));
    for (const [id, v] of cards) if (!ids.has(id)) { v.el.remove(); cards.delete(id); }
    let prev = null;
    for (const c of list) {
      const sig = JSON.stringify([c.status, c.url, c.title, c.error, c.edit_error, viral(c), c.meta && c.meta.hooks && c.meta.hooks.selected, c.meta && c.meta.edit, c.mode, selected.has(c.id), c.output_duration]);
      let cur = cards.get(c.id);
      if (!cur || cur.sig !== sig) { const el = card(c); if (cur) cur.el.replaceWith(el); cur = {sig, el}; cards.set(c.id, cur); }
      if (prev ? prev.nextSibling !== cur.el : grid.firstChild !== cur.el) grid.insertBefore(cur.el, prev ? prev.nextSibling : grid.firstChild);
      prev = cur.el;
    }
    if (!list.length) grid.replaceChildren(h('p', {class: 'muted', text: 'Клипов по этому фильтру нет.'})); else if (grid.querySelector('p.muted')) grid.querySelector('p.muted').remove();
    toolbar();
  }
  function toolbar() {
    const n = selected.size, vis = visible();
    $('cs-count').textContent = `${vis.length} клип(ов)` + (n ? ` · выбрано ${n}` : '');
    $('cs-compare').disabled = n < 2 || n > 4;
    $('cs-best').disabled = !n || [...selected].every(id => { const c = data.clips.find(x => x.id === id); return !c || !viral(c) || viral(c).applied || !(viral(c).gain > 0); });
    $('cs-export').disabled = !n || ![...selected].some(id => { const c = data.clips.find(x => x.id === id); return c && c.status === 'ready' && c.url; });
    $('cs-all').checked = vis.length > 0 && vis.every(c => selected.has(c.id));
  }

  // ------------------------------------------------------------------ polling
  async function poll() {
    clearTimeout(timer);
    try {
      const r = await fetch(`/api/projects/${PID}/status/`); data = await r.json();
      for (const id of [...selected]) if (!data.clips.some(c => c.id === id)) selected.delete(id);
      renderStatus(data); renderGrid();
      const busy = data.active || data.clips.some(c => c.status === 'rendering' || c.status === 'queued');
      timer = setTimeout(poll, busy ? 1500 : 6000);
    } catch (e) { timer = setTimeout(poll, 4000); }
  }

  // ------------------------------------------------------------------ toolbar actions
  $('cs-sort').onchange = renderGrid; $('cs-filter').onchange = renderGrid; $('cs-search').oninput = renderGrid;
  $('cs-all').onchange = () => { for (const c of visible()) if ($('cs-all').checked) selected.add(c.id); else selected.delete(c.id); for (const v of cards.values()) v.sig = ''; renderGrid(); };
  $('cs-export').onclick = async () => {
    const ids = [...selected].join(',');
    window.location.href = `/api/projects/${PID}/export/?ids=${ids}`;
  };
  $('cs-best').onclick = async () => {
    const items = [];
    for (const id of selected) {
      const c = data.clips.find(x => x.id === id); const v = c && viral(c);
      if (!v || v.applied || !(v.gain > 0)) continue;
      const hk = c.meta.hooks, var_ = hk && (hk.variants || []).find(x => x.id === v.combo.hook);
      items.push({id, hook: v.combo.hook, edit: {mode: v.combo.edit, restore: []},
        ...(var_ && var_.kind === 'start' ? {start: var_.start, end: var_.end} : (hk && hk.clip ? {start: hk.clip[0], end: hk.clip[1]} : {}))});
    }
    if (!items.length) return;
    const r = await post(`/api/projects/${PID}/apply/`, {clips: items});
    if (!r.ok) alert((await r.json()).error || 'Не удалось'); poll();
  };
  $('cs-compare').onclick = () => {
    const list = [...selected].map(id => data.clips.find(c => c.id === id)).filter(Boolean);
    const rows = [['Viral', c => viral(c) ? Math.round(viral(c).score) : '—'], ['Hook', c => hookSel(c) ? Math.round(hookSel(c).score) : '—'],
      ['Длительность', c => fmt(duration(c))], ['Монтаж', c => EDIT[(c.meta.edit || {}).mode || 'off'] || '—'],
      ['Начало', c => hookSel(c) ? HOOK_KIND[hookSel(c).kind] : '—'], ['Раскладка', c => (MODES[c.mode] || c.mode || '').split(' (')[0]],
      ...['hook', 'emotion', 'dynamics', 'coherence', 'originality', 'retention'].map(k => [{hook: 'начало', emotion: 'эмоции', dynamics: 'динамика', coherence: 'связность', originality: 'оригинальность', retention: 'удержание'}[k], c => viral(c) ? viral(c).parts[k] : '—'])];
    const best = {};
    for (const [name, f] of rows) { const vals = list.map(f).map(Number); best[name] = Math.max(...vals.filter(x => !isNaN(x))); }
    $('cs-compare-body').replaceChildren(
      h('div', {class: 'cs-cmp-videos'}, ...list.map(c => h('div', {}, c.url ? h('video', {src: c.url, controls: true, playsinline: true, preload: 'metadata'}) : h('div', {class: 'cs-ph', text: 'нет рендера'}), h('b', {text: c.title})))),
      h('table', {class: 'cs-cmp'}, ...rows.map(([name, f]) => h('tr', {}, h('th', {text: name}), ...list.map(c => { const v = f(c); return h('td', {class: Number(v) === best[name] ? 'best' : '', text: String(v)}); })))));
    $('cs-compare-dlg').showModal();
  };

  // ------------------------------------------------------------------ editor
  let ed = null;            // {clip, base: state, draft: state, undo: [states], tab, plan}
  const clone = x => JSON.parse(JSON.stringify(x));
  function stateOf(c) {
    const hk = c.meta && c.meta.hooks, e = (c.meta && c.meta.edit) || {};
    return {start: c.start, end: c.end, mode: c.mode, overrides: {...(c.opts ? pickOv(c.opts) : {})}, hook: hk ? (hk.selected || 'orig') : null,
      edit: {mode: c.edit_mode || e.mode || 'off', restore: [...(e.restore || [])], ...editSwitches(e)}};
  }
  function editSwitches(e) { const o = {}; for (const k of ['silence', 'zoom']) if (typeof e[k] === 'boolean') o[k] = e[k]; if (e.overrides) o.overrides = {...e.overrides}; return o; }
  function editQuery(e) {
    const q = new URLSearchParams({mode: e.mode, restore: (e.restore || []).join(',')});
    for (const k of ['silence', 'zoom']) if (typeof e[k] === 'boolean') q.set(k, e[k] ? '1' : '0');
    for (const [k, v] of Object.entries(e.overrides || {})) q.set(k, v);
    return q.toString();
  }
  function pickOv(o) { const out = {}; for (const k of ['subtitle_style', 'caption_words', 'caption_highlight', 'captions', 'caption_font', 'caption_censor', 'bg_fill', 'cam_position', 'transition']) out[k] = o[k]; return out; }
  function payload() {                // only what really differs from the clip's current state
    const d = ed.draft, b = ed.base, out = {id: ed.clip.id, start: d.start, end: d.end, mode: d.mode, hook: d.hook, edit: d.edit, overrides: {}};
    for (const [k, v] of Object.entries(d.overrides)) if (JSON.stringify(v) !== JSON.stringify(b.overrides[k])) out.overrides[k] = v;
    return out;
  }
  function change(fn) { ed.undo.push(clone(ed.draft)); fn(ed.draft); drawEditor(); }
  function openEditor(c) {
    ed = {clip: c, base: stateOf(c), draft: stateOf(c), undo: [], tab: 'bounds', plan: null};
    $('cs-ed-title').textContent = c.title;
    const v = $('cs-ed-video'); v.src = c.url || ''; v.poster = c.poster || '';
    $('cs-ed-pinfo').textContent = c.url ? 'Показан текущий рендер.' : 'Рендера ещё нет — нажмите «Предпросмотр».';
    $('cs-ed-err').style.display = 'none';
    drawEditor(); $('cs-editor').showModal();
  }
  function diff() {
    const d = ed.draft, b = ed.base, out = [];
    if (Math.abs(d.start - b.start) > 0.01 || Math.abs(d.end - b.end) > 0.01) out.push(`границы ${d.start.toFixed(1)}–${d.end.toFixed(1)} с`);
    if (d.hook !== b.hook) out.push('Hook');
    if (JSON.stringify(d.edit) !== JSON.stringify(b.edit)) out.push(`монтаж ${EDIT[d.edit.mode]}`);
    if (d.mode !== b.mode) out.push('раскладка');
    if (JSON.stringify(d.overrides) !== JSON.stringify(b.overrides)) out.push('субтитры/вид');
    return out;
  }
  function sel(options, value, onchange) {
    const s = h('select', {onchange: e => onchange(e.target.value)}, ...options.map(([k, l]) => h('option', {value: k, text: l})));
    s.value = String(value); return s;
  }
  function field(label, el) { return h('label', {class: 'cs-field'}, h('span', {text: label}), el); }
  async function loadPlan() {
    const d = ed.draft;
    if (d.edit.mode === 'off') { ed.plan = null; drawEditor(); return; }
    const r = await fetch(`/api/clips/${ed.clip.id}/edit/?${editQuery(d.edit)}`);
    const j = r.ok ? await r.json() : null;
    ed.plan = j ? j.plan : null; ed.planInfo = j; drawEditor();
  }
  const TABS = {
    bounds() {
      const d = ed.draft;
      const num = (k) => { const i = h('input', {type: 'number', step: '0.1', value: d[k].toFixed(1)}); i.onchange = () => change(s => { s[k] = parseFloat(i.value) || s[k]; }); return i; };
      const nudge = (k, x) => h('button', {class: 'sm', type: 'button', text: (x > 0 ? '+' : '') + x, onclick: () => change(s => { s[k] = Math.max(0, +(s[k] + x).toFixed(2)); })});
      const src = h('video', {src: `${data.source_url}#t=${d.start}`, controls: true, preload: 'metadata', class: 'cs-src'});
      const auto = ed.clip.scores && ed.clip.scores.auto_bounds;
      return [h('div', {class: 'row'}, field('Начало, с', num('start')), nudge('start', -1), nudge('start', 1), field('Конец, с', num('end')), nudge('end', -1), nudge('end', 1)),
        h('small', {class: 'muted', text: `Длина ${(d.end - d.start).toFixed(1)} с`}), src,
        h('div', {class: 'row'}, h('button', {class: 'sm', type: 'button', text: 'Начало = позиция', onclick: () => change(s => { s.start = +src.currentTime.toFixed(2); })}),
          h('button', {class: 'sm', type: 'button', text: 'Конец = позиция', onclick: () => change(s => { s.end = +src.currentTime.toFixed(2); })}),
          auto ? h('button', {class: 'sm', type: 'button', text: `Найденные ${auto[0].toFixed(1)}–${auto[1].toFixed(1)}`, onclick: () => change(s => { s.start = auto[0]; s.end = auto[1]; })}) : null)];
    },
    hook() {
      const hk = ed.clip.meta && ed.clip.meta.hooks;
      if (!hk) return [h('p', {class: 'muted', text: 'Варианты начала ещё не найдены (Hook Engine выключен или клип старый) — откройте блок «Hook» на странице проекта.'})];
      return (hk.variants || []).map(v => h('label', {class: 'cs-opt' + (ed.draft.hook === v.id ? ' on' : '')},
        h('input', {type: 'radio', name: 'hk', checked: ed.draft.hook === v.id, onchange: () => change(s => { s.hook = v.id; if (v.kind === 'start') { s.start = v.start; s.end = v.end; } else if (hk.clip) { s.start = hk.clip[0]; s.end = hk.clip[1]; } })}),
        h('div', {}, h('b', {text: `${Math.round(v.score)}/100 · ${HOOK_KIND[v.kind]}` + (v.id === hk.recommended ? ' · рекомендуем' : '')}),
          v.text ? h('div', {class: 'hint', text: `«${v.text}»`}) : null,
          h('small', {class: 'muted', text: (v.reasons || []).slice(0, 4).map(r => `${r.points > 0 ? '+' : ''}${r.points} ${r.text}`).join(' · ')}))));
    },
    edit() {
      const d = ed.draft;
      const out = [h('div', {class: 'row'}, ...Object.entries(EDIT).map(([k, l]) => h('button', {class: 'sm' + (d.edit.mode === k ? ' primary' : ''), type: 'button', text: l,
        onclick: () => { change(s => { s.edit = {mode: k, restore: [], ...editSwitches(s.edit)}; }); loadPlan(); }})))];
      if (d.edit.mode !== 'off') {
        const info = ed.planInfo || {}, ch = info.choice || {}, def = info.defaults || {}, lim = info.limits || {};
        const sw = (k, label) => { const cb = h('input', {type: 'checkbox'}); cb.checked = ch[k] !== false; cb.onchange = () => { change(st => { st.edit[k] = cb.checked; }); loadPlan(); }; return h('label', {class: 'hint'}, cb, ' ' + label); };
        const sl = (k, label, step) => {
          const v = (d.edit.overrides || {})[k] ?? (ch.overrides || {})[k] ?? def[k];
          const i = h('input', {type: 'range', min: (lim[k] || [0, 1])[0], max: (lim[k] || [0, 1])[1], step, value: v});
          const out = h('small', {class: 'muted', text: ` ${(+v).toFixed(2)}`});
          i.oninput = () => { out.textContent = ` ${(+i.value).toFixed(2)}`; };
          i.onchange = () => { change(st => { st.edit.overrides = {...(st.edit.overrides || {}), [k]: +i.value}; }); loadPlan(); };
          return h('label', {class: 'cs-field'}, h('span', {text: label}), i, out);
        };
        out.push(h('div', {class: 'row'}, sw('silence', 'Удалять паузы и тишину'), sw('zoom', 'Авто-зум')),
          h('div', {class: 'row'}, sl('max', 'Пауза длиннее, с', 0.05), sl('zoom_strength', 'Сила зума', 0.05), sl('zoom_rate', 'Зумов на 10 с', 0.1)));
      }
      if (d.edit.mode !== 'off' && !ed.plan) { out.push(h('small', {class: 'muted', text: 'Считаем план…'})); loadPlan(); return out; }
      if (ed.plan) {
        const s = ed.plan.stats;
        out.push(h('small', {class: 'muted', text: `${s.original.toFixed(1)} → ${s.result.toFixed(1)} с, вырезано ${s.removed.toFixed(1)} с · пауз удалено: ${s.pauses ?? 0} · зумов: ${s.zooms ?? 0}`}));
        if ((ed.plan.zooms || []).length) out.push(h('small', {class: 'hint', text: 'Зум: ' + ed.plan.zooms.map(z => `${z.t.toFixed(1)} с ×${z.level.toFixed(2)} (${z.label})`).join(' · ')}));
        out.push(h('div', {class: 'editcuts'}, ...ed.plan.cuts.map(x => { const cb = h('input', {type: 'checkbox'}); cb.checked = x.applied !== false && !d.edit.restore.includes(x.id); cb.disabled = !!x.skipped;
          cb.onchange = () => { change(st => { st.edit.restore = cb.checked ? st.edit.restore.filter(i => i !== x.id) : [...st.edit.restore, x.id]; }); loadPlan(); };
          return h('label', {class: 'hint', title: x.skipped || false}, cb, ` ${x.start.toFixed(1)}–${x.end.toFixed(1)} с · ${REASON[x.reason]} · ${x.seconds.toFixed(1)} с`); })));
      }
      return out;
    },
    layout() {
      const d = ed.draft, o = d.overrides;
      return [field('Раскладка', sel(Object.entries(MODES), d.mode, v => change(s => { s.mode = v; }))),
        field('Фон', sel([['blur', 'Размытый'], ['solid', 'Однотонный'], ['focus', 'Приближение']], o.bg_fill, v => change(s => { s.overrides.bg_fill = v; }))),
        field('Вебка', sel([['bottom', 'Снизу'], ['top', 'Сверху']], o.cam_position, v => change(s => { s.overrides.cam_position = v; }))),
        field('Переходы', sel([['none', 'Без переходов'], ['fade', 'Затухание']], o.transition, v => change(s => { s.overrides.transition = v; })))];
    },
    caps() {
      const o = ed.draft.overrides;
      return [field('Субтитры', sel([['true', 'Включены'], ['false', 'Выключены']], o.captions, v => change(s => { s.overrides.captions = v === 'true'; }))),
        field('Стиль', sel(STYLES, o.subtitle_style, v => change(s => { s.overrides.subtitle_style = v; }))),
        field('Шрифт', sel(FONTS.map(f => [f, f === 'style' ? 'Как в стиле' : f]), o.caption_font, v => change(s => { s.overrides.caption_font = v; }))),
        field('Слов на экране', sel([['0', 'Как в стиле'], ['1', '1'], ['2', '2'], ['3', '3'], ['4', '4']], o.caption_words || 0, v => change(s => { s.overrides.caption_words = parseInt(v, 10); }))),
        field('Подсветка слов', sel([['null', 'Как в стиле'], ['true', 'Да'], ['false', 'Нет']], o.caption_highlight, v => change(s => { s.overrides.caption_highlight = v === 'null' ? null : v === 'true'; }))),
        field('Цензура мата', sel([['false', 'Нет'], ['true', 'Да']], o.caption_censor, v => change(s => { s.overrides.caption_censor = v === 'true'; })))];
    },
  };
  function drawEditor() {
    for (const b of $('cs-tabs').children) b.classList.toggle('on', b.dataset.tab === ed.tab);
    $('cs-tab').replaceChildren(...TABS[ed.tab]());
    const ch = diff();
    $('cs-changes').textContent = ch.length ? `Не применено: ${ch.join(', ')}` : 'Изменений нет.';
    $('cs-ed-undo').disabled = !ed.undo.length; $('cs-ed-reset').disabled = !ch.length; $('cs-ed-apply').disabled = !ch.length;
    $('cs-ed-prev').disabled = !ed.clip.history; $('cs-ed-orig').disabled = !ed.clip.edited;
  }
  for (const b of $('cs-tabs').children) b.onclick = () => { ed.tab = b.dataset.tab; drawEditor(); };
  const err = (t) => { const e = $('cs-ed-err'); e.style.display = t ? 'block' : 'none'; e.textContent = t || ''; };
  $('cs-ed-close').onclick = () => $('cs-editor').close();
  $('cs-ed-undo').onclick = () => { if (ed.undo.length) { ed.draft = ed.undo.pop(); ed.plan = null; drawEditor(); } };
  $('cs-ed-reset').onclick = () => { ed.undo.push(clone(ed.draft)); ed.draft = clone(ed.base); ed.plan = null; drawEditor(); };
  $('cs-ed-preview').onclick = async () => {
    const btn = $('cs-ed-preview'); btn.disabled = true; btn.textContent = '… собираем черновик'; err('');
    try {
      const r = await post(`/api/clips/${ed.clip.id}/preview/`, payload()); const d = await r.json();
      if (!r.ok) { err(d.error || 'Не удалось'); return; }
      const v = $('cs-ed-video'); v.src = d.url; v.play().catch(() => {});
      $('cs-ed-pinfo').textContent = `Черновик 360p (${(d.info.duration || 0).toFixed(1)} с)${d.info.cached ? ', из кэша' : `, собран за ${d.info.seconds} с`}. Итоговый рендер — в полном качестве.`;
    } finally { btn.disabled = false; btn.textContent = '▶ Предпросмотр монтажа'; }
  };
  $('cs-ed-apply').onclick = async () => {
    const r = await post(`/api/projects/${PID}/apply/`, {clips: [payload()]});
    if (!r.ok) { err((await r.json()).error || 'Не удалось'); return; }
    $('cs-editor').close(); poll();
  };
  const restore = (to) => async () => {
    if (!confirm(to === 'original' ? 'Вернуть исходный клип (найденные границы, без Hook и монтажа)?' : 'Вернуть версию до последнего применения?')) return;
    const r = await post(`/api/clips/${ed.clip.id}/restore/`, {to});
    if (!r.ok) { err((await r.json()).error || 'Не удалось'); return; }
    $('cs-editor').close(); poll();
  };
  $('cs-ed-prev').onclick = restore('previous'); $('cs-ed-orig').onclick = restore('original');
  poll();
})();

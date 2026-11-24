// AI Script Generator (3A): structure by part, scene cards with lines / visual / timing, manual edits, one-scene
// regeneration, full regeneration, export of the script JSON and its Timeline IR.
(function () {
  const ID = window.SG.id, PARTS = JSON.parse(document.getElementById('sg-parts').textContent);
  const $ = id => document.getElementById(id);
  const csrf = () => decodeURIComponent((document.cookie.split('; ').find(x => x.startsWith('csrftoken=')) || '').split('=')[1] || '');
  const post = (url, body) => fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json', 'X-CSRFToken': csrf()}, body: JSON.stringify(body || {})});
  function h(tag, a, ...kids) { const e = document.createElement(tag); for (const [k, v] of Object.entries(a || {})) { if (k === 'text') e.textContent = v; else if (k.startsWith('on')) e.addEventListener(k.slice(2), v); else if (v !== false && v != null) e.setAttribute(k, v === true ? '' : v); } e.append(...kids.filter(x => x != null && x !== false)); return e; }
  const fmt = s => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, '0')}`;
  let S = null, timer = null, sig = '', filled = false;

  async function load() {
    clearTimeout(timer);
    try { S = await (await fetch(`/api/scripts/${ID}/`)).json(); render(); } catch (e) {}
    timer = setTimeout(load, S && S.status === 'generating' ? 1500 : 8000);
  }
  function err(t) { const box = $('sg-status'); box.append(h('div', {class: 'error', text: t})); }
  function render() {
    const st = $('sg-status');
    const d = S.data || {};
    st.replaceChildren(...[h('div', {class: 'row', style: 'justify-content:space-between'},
      h('span', {text: S.status === 'generating' ? 'Генерируется…' : S.status === 'error' ? 'Ошибка генерации' : d.scenes ? `${d.scenes.length} сцен · ${fmt(d.total || 0)} из ${fmt(S.duration)}` : 'Черновик'}),
      h('span', {class: `badge b-${S.status}`, text: S.status})),
      d.provider ? h('small', {class: 'muted', text: `Текст: ${d.provider === 'template' ? 'шаблон (LLM недоступна)' : d.provider}` + (d.provider_last ? ` · последняя сцена: ${d.provider_last}` : '')}) : null,
      d.note ? h('small', {class: 'livewarn', text: d.note}) : null,
      S.error ? h('div', {class: 'error', text: S.error}) : null,
      S.job && S.job.status === 'running' && window.renderLive ? renderLive(S.job, null) : null,
      d.scenes ? h('div', {class: 'row'}, h('a', {class: 'btn sm', href: `/api/scripts/${ID}/export/script/`, text: '⬇ Сценарий (JSON)'}),
        h('a', {class: 'btn sm', href: `/api/scripts/${ID}/export/timeline/`, text: '⬇ Timeline IR'})) : null].filter(Boolean));
    if (!filled) { filled = true; $('sg-idea').value = S.idea; $('sg-duration').value = S.duration; $('sg-genre').value = S.genre; $('sg-style').value = S.style; $('sg-audience').value = S.audience; }
    if (d.title) $('sg-title').textContent = d.title;
    const s2 = JSON.stringify([S.status, d.generated_at, (d.scenes || []).map(x => [x.id, x.revision, x.duration, x.locked])]);
    if (s2 === sig) return; sig = s2;
    const body = $('sg-body'); body.replaceChildren();
    if (!d.scenes) return;
    if (d.logline) body.append(h('p', {class: 'hint', text: d.logline}));
    if ((d.characters || []).length) body.append(h('div', {class: 'panel', style: 'padding:10px'}, h('b', {text: 'Персонажи: '}),
      ...d.characters.map(c => h('span', {class: 'chip', title: `${c.description} · ${c.voice}`, text: c.name}))));
    const names = Object.fromEntries((d.characters || []).map(c => [c.id, c.name]));
    for (const part of Object.keys(PARTS)) {
      const list = d.scenes.filter(x => x.part === part); if (!list.length) continue;
      body.append(h('h2', {style: 'font-size:17px;margin-top:18px', text: PARTS[part]}), ...list.map(x => sceneCard(x, names, d)));
    }
  }
  function sceneCard(x, names, d) {
    const busy = S.status === 'generating';
    const lines = h('textarea', {rows: Math.max(2, x.lines.length + 1), class: 'sg-lines'});
    lines.value = x.lines.map(l => `${l.speaker === 'narrator' ? 'Рассказчик' : names[l.speaker] || l.speaker}: ${l.text}`).join('\n');
    const visual = h('textarea', {rows: 2}); visual.value = x.visual;
    const dur = h('input', {type: 'number', step: '0.5', min: '2.5', max: '15', value: x.duration});
    const ost = h('input', {value: x.on_screen_text || '', placeholder: 'Текст на экране'});
    const instr = h('input', {placeholder: 'Пожелание к новой версии: смешнее, короче, с вопросом…'});
    const rev = new Map([['рассказчик', 'narrator'], ...Object.entries(names).map(([id, n]) => [n.toLowerCase(), id])]);
    const save = async () => {
      const ls = lines.value.split('\n').map(t => t.trim()).filter(Boolean).map(t => { const m = t.match(/^([^:]{1,40}):\s*(.+)$/); const sp = m && rev.get(m[1].trim().toLowerCase()); return sp ? {speaker: sp, text: m[2]} : {speaker: 'narrator', text: t}; });
      const r = await post(`/api/scripts/${ID}/scenes/${x.id}/`, {lines: ls, visual: visual.value, duration: parseFloat(dur.value), on_screen_text: ost.value});
      if (!r.ok) { err((await r.json()).error || 'Не удалось сохранить'); return; } sig = ''; load();
    };
    const lock = async () => { await post(`/api/scripts/${ID}/scenes/${x.id}/`, {locked: !x.locked}); sig = ''; load(); };
    const regen = async () => { const r = await post(`/api/scripts/${ID}/scenes/${x.id}/regenerate/`, {instruction: instr.value}); if (!r.ok) err((await r.json()).error || 'Не удалось'); load(); };
    return h('div', {class: 'panel sg-scene' + (x.locked ? ' locked' : ''), style: 'padding:12px'},
      h('div', {class: 'row', style: 'justify-content:space-between'}, h('b', {text: `${fmt(x.start)}–${fmt(x.start + x.duration)} · ${x.title}`}),
        h('small', {class: 'muted', text: `версия ${x.revision}` + (x.locked ? ' · закреплена' : '')})),
      h('label', {class: 'cs-field'}, h('span', {text: 'Реплики (Имя: текст)'}), lines),
      h('label', {class: 'cs-field'}, h('span', {text: 'Визуал (для раскадровки)'}), visual),
      h('div', {class: 'row'}, h('label', {class: 'cs-field'}, h('span', {text: 'Длительность, с'}), dur), h('label', {class: 'cs-field', style: 'flex:1'}, h('span', {text: 'Текст на экране'}), ost)),
      h('small', {class: 'muted', text: [x.shot && `Камера: ${x.shot}`, x.mood && `Настроение: ${x.mood}`, x.sfx && `Звуки: ${x.sfx}`, x.music && `Музыка: ${x.music}`].filter(Boolean).join(' · ')}),
      h('div', {class: 'row'}, h('button', {class: 'sm primary', type: 'button', text: 'Сохранить', disabled: busy, onclick: save}),
        h('button', {class: 'sm', type: 'button', text: x.locked ? 'Открепить' : 'Закрепить', disabled: busy, onclick: lock}),
        instr, h('button', {class: 'sm', type: 'button', text: '↻ Перегенерировать сцену', disabled: busy || x.locked, onclick: regen})));
  }
  $('sg-regen').onclick = async () => {
    if (S && (S.data || {}).scenes && !confirm('Сгенерировать весь сценарий заново? Текущие сцены будут заменены.')) return;
    const r = await post(`/api/scripts/${ID}/generate/`, {idea: $('sg-idea').value, duration: parseInt($('sg-duration').value, 10), genre: $('sg-genre').value, style: $('sg-style').value, audience: $('sg-audience').value});
    if (!r.ok) err((await r.json()).error || 'Не удалось'); sig = ''; load();
  };
  load();
})();

// AI Storyboard (3B): Style / Character Bible, frames per scene (shot, composition, action, camera, characters,
// prompts), reorder, lock, edit, one-frame regeneration, (re)build in the job queue, Timeline IR export.
(function () {
  const ID = window.SB.id;
  const SHOTS = JSON.parse(document.getElementById('sb-shots').textContent);
  const MOVES = JSON.parse(document.getElementById('sb-moves').textContent);
  const COMPS = JSON.parse(document.getElementById('sb-comps').textContent);
  const PART = {hook: 'Хук', development: 'Развитие', climax: 'Кульминация', ending: 'Завершение'};
  const $ = id => document.getElementById(id);
  const csrf = () => decodeURIComponent((document.cookie.split('; ').find(x => x.startsWith('csrftoken=')) || '').split('=')[1] || '');
  const post = (url, body) => fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json', 'X-CSRFToken': csrf()}, body: JSON.stringify(body || {})});
  function h(tag, a, ...kids) { const e = document.createElement(tag); for (const [k, v] of Object.entries(a || {})) { if (k === 'text') e.textContent = v; else if (k.startsWith('on')) e.addEventListener(k.slice(2), v); else if (v !== false && v != null) e.setAttribute(k, v === true ? '' : v); } e.append(...kids.filter(x => x != null && x !== false)); return e; }
  const fmt = s => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, '0')}`;
  const field = (label, el) => h('label', {class: 'cs-field'}, h('span', {text: label}), el);
  const sel = (opts, v) => { const s = h('select', {}, ...[...new Set([v, ...opts])].filter(Boolean).map(o => h('option', {value: o, text: o}))); s.value = v; return s; };
  const area = (v, rows) => { const t = h('textarea', {rows: rows || 2}); t.value = v || ''; return t; };
  let S = null, timer = null, sig = '', bibleSig = '';

  async function load() {
    clearTimeout(timer);
    try { S = await (await fetch(`/api/scripts/${ID}/storyboard/`)).json(); render(); } catch (e) {}
    timer = setTimeout(load, S && S.status === 'generating' ? 1500 : 8000);
  }
  const fail = async r => { if (!r.ok) { const d = await r.json().catch(() => ({})); alert(d.error || 'Не удалось'); } sig = ''; bibleSig = ''; load(); };
  const build = scenes => post(`/api/scripts/${ID}/storyboard/`, {scenes}).then(fail);

  function render() {
    const d = S.data || {}, sb = d.storyboard;
    const busy = S.status === 'generating';
    const n = sb ? Object.values(sb.scenes).reduce((a, x) => a + x.frames.length, 0) : 0;
    $('sb-status').replaceChildren(...[
      h('div', {class: 'row', style: 'justify-content:space-between'}, h('span', {text: busy ? 'Генерируется…' : sb ? `${n} кадров в ${Object.keys(sb.scenes).length} сценах` : (d.scenes ? 'Раскадровки ещё нет' : 'Сначала создайте сценарий')}), h('span', {class: `badge b-${S.status}`, text: S.status})),
      sb ? h('small', {class: 'muted', text: `Кадры: ${sb.provider === 'rules' ? 'по правилам (LLM недоступна)' : sb.provider}` + (sb.provider_last ? ` · последний кадр: ${sb.provider_last}` : '')}) : null,
      sb && sb.note ? h('small', {class: 'livewarn', text: sb.note}) : null,
      S.error ? h('div', {class: 'error', text: S.error}) : null,
      S.job && S.job.status === 'running' && window.renderLive ? renderLive(S.job, null) : null,
      d.scenes ? h('div', {class: 'row'},
        h('button', {class: 'sm primary', type: 'button', disabled: busy, text: sb ? '↻ Пересобрать раскадровку (закреплённое останется)' : '▦ Создать раскадровку', onclick: () => build([])}),
        S.stale && S.stale.length && sb ? h('button', {class: 'sm', type: 'button', disabled: busy, text: `Обновить изменённые сцены (${S.stale.length})`, onclick: () => build(S.stale)}) : null,
        sb ? h('a', {class: 'btn sm', href: `/api/scripts/${ID}/export/timeline/`, text: '⬇ Timeline IR (для 3C)'}) : null,
        h('a', {class: 'btn sm', href: `/api/scripts/${ID}/export/script/`, text: '⬇ JSON'})) : null].filter(Boolean));
    if (!sb) { $('sb-body').replaceChildren(); return; }
    const bs = JSON.stringify([sb.style, sb.characters, busy]);
    if (bs !== bibleSig) { bibleSig = bs; renderBibles(sb, d, busy); }
    const s2 = JSON.stringify([busy, sb.generated_at, S.stale, Object.values(sb.scenes).map(x => x.frames.map(f => [f.id, f.revision, f.locked, f.start, f.duration]))]);
    if (s2 === sig) return; sig = s2;
    const names = Object.fromEntries(Object.entries(sb.characters).map(([k, c]) => [k, c.name]));
    $('sb-body').replaceChildren(...d.scenes.map(s => sceneBlock(s, sb.scenes[s.id], sb, names, busy)));
  }

  function renderBibles(sb, d, busy) {
    const st = sb.style;
    const f = {look: area(st.look), lighting: area(st.lighting, 1), environment: area(st.environment, 1), camera_language: area(st.camera_language, 1), negative: area(st.negative)};
    const pal = h('input', {value: (st.palette || []).join(' ')});
    const sw = h('div', {class: 'sb-pal'}, ...(st.palette || []).map(c => h('i', {style: `background:${c}`, title: c})));
    $('sb-style-body').replaceChildren(sw, field('Вид / техника', f.look), field('Палитра (#hex через пробел)', pal), field('Освещение', f.lighting),
      field('Окружение', f.environment), field('Язык камеры', f.camera_language), field('Negative', f.negative),
      h('button', {class: 'sm primary', type: 'button', disabled: busy, text: 'Сохранить стиль и обновить промпты', onclick: () => post(`/api/scripts/${ID}/storyboard/bible/`, {style: {
        ...Object.fromEntries(Object.entries(f).map(([k, el]) => [k, el.value])), palette: pal.value.split(/[\s,]+/).filter(Boolean)}}).then(fail)}));
    const chars = Object.entries(sb.characters);
    $('sb-chars-body').replaceChildren(...(chars.length ? chars.map(([id, c]) => {
      const g = {appearance: area(c.appearance), clothing: area(c.clothing, 1), features: area(c.features, 1), colors: h('input', {value: c.colors || ''}), voice: h('input', {value: c.voice || ''})};
      return h('div', {class: 'sb-char'}, h('b', {text: `${c.name} · seed ${c.seed}`}), field('Внешность', g.appearance), field('Одежда (постоянная)', g.clothing),
        field('Особенности', g.features), h('div', {class: 'row'}, field('Цвета', g.colors), field('Голос', g.voice)),
        h('details', {}, h('summary', {class: 'hint', text: 'Промпт листа персонажа'}), h('small', {class: 'muted', text: c.ref_prompt})),
        h('button', {class: 'sm', type: 'button', disabled: busy, text: 'Сохранить персонажа', onclick: () => post(`/api/scripts/${ID}/storyboard/bible/`, {characters: {[id]: Object.fromEntries(Object.entries(g).map(([k, el]) => [k, el.value]))}}).then(fail)}));
    }) : [h('p', {class: 'muted', text: 'В сценарии нет постоянных персонажей.'})]));
  }

  function sceneBlock(s, ssb, sb, names, busy) {
    const stale = (S.stale || []).includes(s.id);
    const head = h('div', {class: 'row', style: 'justify-content:space-between'},
      h('b', {text: `${PART[s.part]} · ${fmt(s.start)}–${fmt(s.start + s.duration)} · ${s.title}`}),
      stale ? h('button', {class: 'sm', type: 'button', disabled: busy, text: 'Сцена изменилась — обновить кадры', onclick: () => build([s.id])}) : null);
    const lines = h('small', {class: 'muted', text: s.lines.map(l => `${l.speaker === 'narrator' ? 'Рассказчик' : names[l.speaker] || l.speaker}: ${l.text}`).join(' · ')});
    if (!ssb) return h('div', {class: 'panel sb-scene', style: 'padding:12px'}, head, lines, h('p', {class: 'muted', text: 'Кадров ещё нет.'}));
    return h('div', {class: 'panel sb-scene', style: 'padding:12px'}, head, lines,
      h('small', {class: 'muted', text: `Место: ${ssb.location}`}),
      h('div', {class: 'sb-frames'}, ...ssb.frames.map((f, i) => frameCard(f, i, ssb.frames.length, sb, names, busy))));
  }

  function frameCard(f, i, n, sb, names, busy) {
    const shot = sel(SHOTS, f.shot), cam = sel(MOVES, f.camera), comp = sel(COMPS, f.composition);
    const action = area(f.action, 3), dur = h('input', {type: 'number', step: '0.1', min: '0.6', value: f.duration});
    const who = Object.entries(names).map(([id, nm]) => { const cb = h('input', {type: 'checkbox'}); cb.checked = f.characters.includes(id); cb.dataset.id = id; return h('label', {class: 'hint'}, cb, ' ' + nm); });
    const prompt = area(f.prompt, 4), neg = area(f.negative, 2), instr = h('input', {placeholder: 'Пожелание: крупнее, драматичнее…'});
    const pal = (sb.style.palette || ['#334', '#556']);
    const save = (extra) => post(`/api/scripts/${ID}/storyboard/frames/${f.id}/`, Object.assign({shot: shot.value, camera: cam.value, composition: comp.value, action: action.value,
      duration: parseFloat(dur.value), characters: who.map(l => l.querySelector('input')).filter(c => c.checked).map(c => c.dataset.id)}, extra || {})).then(fail);
    return h('div', {class: 'sb-frame' + (f.locked ? ' locked' : '')},
      h('div', {class: 'sb-thumb', style: `background:linear-gradient(160deg,${pal[0]},${pal[1] || pal[0]})`},
        h('span', {text: `${i + 1}`}), h('small', {text: `${f.shot}\n${f.camera}`}), h('em', {text: `${f.start.toFixed(1)}–${(f.start + f.duration).toFixed(1)} с`})),
      h('div', {class: 'row'}, field('План', shot), field('Камера', cam)), field('Композиция', comp), field('Действие в кадре', action),
      h('div', {class: 'row'}, field('Длительность', dur), ...who),
      h('details', {}, h('summary', {class: 'hint', text: `Промпты${f.manual_prompt ? ' (написаны вручную)' : ''}`}), field('Positive', prompt), field('Negative', neg),
        h('button', {class: 'sm', type: 'button', disabled: busy, text: 'Сохранить промпты как есть', onclick: () => save({prompt: prompt.value, negative: neg.value})})),
      h('div', {class: 'row'},
        h('button', {class: 'sm', type: 'button', disabled: busy || i === 0, text: '←', title: 'Раньше', onclick: () => post(`/api/scripts/${ID}/storyboard/frames/${f.id}/move/`, {to: i - 1}).then(fail)}),
        h('button', {class: 'sm', type: 'button', disabled: busy || i === n - 1, text: '→', title: 'Позже', onclick: () => post(`/api/scripts/${ID}/storyboard/frames/${f.id}/move/`, {to: i + 1}).then(fail)}),
        h('button', {class: 'sm primary', type: 'button', disabled: busy, text: 'Сохранить', onclick: () => save()}),
        h('button', {class: 'sm', type: 'button', disabled: busy, text: f.locked ? 'Открепить' : 'Закрепить', onclick: () => post(`/api/scripts/${ID}/storyboard/frames/${f.id}/`, {locked: !f.locked}).then(fail)})),
      h('div', {class: 'row'}, instr, h('button', {class: 'sm', type: 'button', disabled: busy || f.locked, text: '↻ Новый вариант', onclick: () => post(`/api/scripts/${ID}/storyboard/frames/${f.id}/regenerate/`, {instruction: instr.value}).then(fail)})),
      h('small', {class: 'muted', text: `id ${f.id} · версия ${f.revision} · seed ${f.seed}`}));
  }
  load();
})();

const csrf = () => decodeURIComponent((document.cookie.split('; ').find(x => x.startsWith('csrftoken=')) || '').split('=')[1] || '');
const post = (url, body) => fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json', 'X-CSRFToken': csrf()}, body: JSON.stringify(body || {})});
function h(tag, a, ...kids) { const e = document.createElement(tag); for (const [k, v] of Object.entries(a || {})) { if (k === 'text') e.textContent = v; else if (k.startsWith('on')) e.addEventListener(k.slice(2), v); else if (v !== false && v != null) e.setAttribute(k, v === true ? '' : v); } e.append(...kids.filter(Boolean)); return e; }
const dur = s => { s = Math.round(s); const m = Math.floor(s / 60), x = s % 60; return m >= 60 ? `${Math.floor(m / 60)} ч ${m % 60} мин` : (m ? `${m} мин ${x} с` : `${x} с`); };
let timer = null;

async function act(id, action, body) {
  const r = await post(`/api/jobs/${id}/${action}/`, body);
  if (!r.ok) alert((await r.json()).error || 'Не удалось');
  load();
}

function card(j) {
  const live = ['queued', 'running', 'paused'].includes(j.status);
  const pct = j.status === 'running' ? (j.stage_progress != null ? `${j.stage_label || j.stage}: ${Math.round(j.stage_progress * 100)}%` : (j.stage_label || j.stage || '')) : '';
  const btns = h('div', {class: 'row'});
  if (j.status === 'running') btns.append(h('button', {class: 'sm', text: j.pause_requested ? 'Пауза после шага…' : '⏸ Пауза', disabled: j.pause_requested, onclick: () => act(j.id, 'pause')}));
  if (j.status === 'queued') btns.append(h('button', {class: 'sm', text: '⏸ Пауза', onclick: () => act(j.id, 'pause')}));
  if (j.status === 'paused') btns.append(h('button', {class: 'sm primary', text: '▶ Продолжить', onclick: () => act(j.id, 'resume')}));
  if (['failed', 'interrupted', 'cancelled'].includes(j.status)) btns.append(h('button', {class: 'sm primary', text: '↻ Продолжить с контрольной точки', onclick: () => act(j.id, 'retry')}));
  if (j.status === 'completed') btns.append(h('button', {class: 'sm', text: '↻ Запустить заново', onclick: () => act(j.id, 'retry')}));
  if (live) {
    btns.append(h('button', {class: 'sm', text: '▲', title: 'Повысить приоритет', onclick: () => act(j.id, 'priority', {priority: j.priority + 1})}));
    btns.append(h('button', {class: 'sm', text: '▼', title: 'Понизить приоритет', onclick: () => act(j.id, 'priority', {priority: j.priority - 1})}));
    btns.append(h('button', {class: 'sm danger', text: j.cancel_requested ? 'Отменяем…' : '■ Отменить', disabled: j.cancel_requested, onclick: () => confirm('Отменить задачу?') && act(j.id, 'cancel')}));
  }
  const title = j.project ? h('a', {href: `/projects/${j.project}/`, text: j.title || `Проект #${j.project}`, style: 'font-weight:700'}) : h('b', {text: j.title || j.label});
  const bar = h('div', {style: 'height:6px;background:#222a40;border-radius:4px;margin:8px 0;overflow:hidden'},
    h('div', {style: `height:100%;width:${j.status === 'completed' ? 100 : j.progress}%;background:linear-gradient(90deg,var(--acc),var(--acc2))`}));
  return h('div', {class: 'panel', style: 'padding:14px'},
    h('div', {class: 'row', style: 'justify-content:space-between'}, title, h('span', {class: `badge b-${j.status}`, text: j.status_label})),
    h('small', {class: 'muted', text: `#${j.id} · ${j.label} · приоритет ${j.priority} · попытка ${j.attempts}/${j.max_attempts} · время ${dur(j.elapsed)} · создана ${new Date(j.created_at).toLocaleString()}`}),
    live || j.status === 'completed' ? bar : null,
    h('div', {text: [pct, j.message].filter(Boolean).join(' — ')}),
    j.checkpoints.length ? h('small', {class: 'muted', text: 'Готовые шаги: ' + j.checkpoints.join(', ')}) : null,
    j.error ? h('pre', {class: 'error', style: 'white-space:pre-wrap;max-height:120px;overflow:auto', text: j.error}) : null,
    btns);
}

async function load() {
  clearTimeout(timer);
  try {
    const d = await (await fetch('/api/jobs/')).json();
    document.getElementById('live').replaceChildren(...(d.live.length ? d.live.map(card) : [h('p', {class: 'muted', text: 'Очередь пуста.'})]));
    document.getElementById('recent').replaceChildren(...d.recent.map(card));
    timer = setTimeout(load, d.live.some(j => j.status === 'running') ? 1500 : 5000);
  } catch (e) { timer = setTimeout(load, 5000); }
}
load();

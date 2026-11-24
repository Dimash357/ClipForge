// Resource manager panel (queue page): mode switch, capacity vs use, leases, waiting requests, loaded models.
(function () {
  const csrf = () => decodeURIComponent((document.cookie.split('; ').find(x => x.startsWith('csrftoken=')) || '').split('=')[1] || '');
  const post = (url, body) => fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json', 'X-CSRFToken': csrf()}, body: JSON.stringify(body || {})});
  const el = (tag, attrs, ...kids) => { const e = document.createElement(tag); Object.assign(e, attrs || {}); e.append(...kids.filter(k => k != null)); return e; };
  const NAMES = {cpu: 'CPU (слоты)', ram_mb: 'RAM, МБ', vram_mb: 'VRAM, МБ', nvenc: 'NVENC', nvdec: 'NVDEC'};
  const req = r => Object.entries(r).map(([k, v]) => `${NAMES[k] || k} ${v}`).join(', ');
  let timer = null;

  function bar(k, used, cap) {
    const pct = cap ? Math.min(100, Math.round(100 * used / cap)) : 0;
    return el('div', {className: 'liveop'}, el('div', {className: 'liveop-h'}, el('span', {textContent: NAMES[k]}),
      el('small', {textContent: cap ? `${used} / ${cap}` : 'нет на этом компьютере'})),
      el('div', {className: 'livebar'}, el('i', {style: `width:${pct}%`})));
  }

  function render(d) {
    const box = document.getElementById('resources');
    const modes = el('div', {className: 'row'});
    for (const [k, m] of Object.entries(d.modes)) {
      modes.append(el('button', {className: 'sm' + (k === d.mode ? ' primary' : ''), textContent: m.label, title: m.note,
        disabled: d.mode_locked, onclick: async () => { const r = await post('/api/resources/mode/', {mode: k}); if (!r.ok) alert((await r.json()).error); load(); }}));
    }
    const m = d.machine;
    const head = el('small', {className: 'muted', textContent: `${d.modes[d.mode].note}. ` +
      `${m.cores} ядер · RAM ${m.ram_total ? (m.ram_total / 1024).toFixed(0) + ' ГБ' : '?'} · ` +
      (m.gpu ? `${m.gpu}, VRAM ${(m.vram_total / 1024).toFixed(1)} ГБ` : 'GPU NVIDIA не найден — всё выполняется на CPU') +
      (d.mode_locked ? ' · режим задан CLIPFORGE_RESOURCE_MODE' : '')});
    const bars = el('div', {className: 'live'}, ...Object.keys(NAMES).map(k => bar(k, d.used[k], d.capacity[k])));
    const lists = [];
    if (d.leases.length) lists.push(el('small', {textContent: 'Сейчас заняты: ' + d.leases.map(l => `${l.label} (${req(l.req)}, ${Math.round(l.held)} с)`).join(' · ')}));
    if (d.waiting.length) lists.push(el('small', {className: 'livewarn', textContent: 'Ждут ресурсов: ' + d.waiting.map(w => `задача ${w.job ?? '—'} (${req(w.req)}, ${Math.round(w.waited)} с)`).join(' · ')}));
    lists.push(el('small', {className: 'muted', textContent: d.models.length ? 'Модели в памяти: ' + d.models.map(x => `${x.key} [${x.device}${x.vram_mb ? ', ' + x.vram_mb + ' МБ VRAM' : ''}${x.ram_mb ? ', ' + x.ram_mb + ' МБ RAM' : ''}${x.in_use ? ', используется' : ', простаивает ' + x.idle + ' с'}]`).join(' · ') : 'Моделей в памяти нет (загружаются по требованию).'}));
    if (d.models.length) lists.push(el('button', {className: 'sm', textContent: 'Выгрузить простаивающие модели', onclick: async () => { await post('/api/resources/unload/'); load(); }}));
    const ev = (d.events || []).slice(-3).map(e => `${new Date(e.at * 1000).toLocaleTimeString()} ${e.what === 'oom' ? 'нехватка памяти' : 'переход на CPU'}: ${e.where}`);
    if (ev.length) lists.push(el('small', {className: 'muted', textContent: 'События: ' + ev.join(' · ')}));
    box.replaceChildren(el('div', {className: 'row', style: 'justify-content:space-between'}, el('b', {textContent: 'Ресурсы и режим'}), modes), head, bars, ...lists);
  }

  async function load() {
    clearTimeout(timer);
    try { render(await (await fetch('/api/resources/')).json()); } catch (e) {}
    timer = setTimeout(load, 3000);
  }
  load();
})();

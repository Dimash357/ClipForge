// 3C settings: providers, availability checks, recommended models.
(function () {
  const $ = id => document.getElementById(id);
  const csrf = () => decodeURIComponent((document.cookie.split('; ').find(x => x.startsWith('csrftoken=')) || '').split('=')[1] || '');
  function h(tag, a, ...kids) { const e = document.createElement(tag); for (const [k, v] of Object.entries(a || {})) { if (k === 'text') e.textContent = v; else if (v !== false && v != null) e.setAttribute(k, v === true ? '' : v); } e.append(...kids.filter(x => x != null && x !== false)); return e; }
  const form = $('gm-form');
  const fill = (sel, opts, value, empty) => { sel.replaceChildren(...(empty ? [h('option', {value: '', text: empty})] : []), ...[...new Set([...(value ? [value] : []), ...opts])].map(o => h('option', {value: o, text: o}))); sel.value = value || ''; };
  const ok = (good, text) => h('div', {class: good ? 'gm-ok' : 'gm-bad', text: (good ? '✓ ' : '✗ ') + text});
  async function check() {
    $('gm-status').replaceChildren(h('span', {class: 'muted', text: 'Проверяем…'}));
    const d = await (await fetch('/api/media/status/')).json();
    const c = d.comfy, t = d.tts, cfg = d.config;
    for (const [k, v] of Object.entries(cfg)) { const el = form.elements[k]; if (!el || el.tagName === 'SELECT' && ['checkpoint', 'voice_ru', 'voice_en', 'ip_preset'].includes(k)) continue; if (el.type === 'checkbox') el.checked = !!v; else el.value = v; }
    fill($('gm-ckpt'), c.checkpoints || [], cfg.checkpoint, '— выберите —');
    fill($('gm-ipp'), c.ip_presets || [cfg.ip_preset], cfg.ip_preset);
    const voices = Object.keys(t.voices || {});
    fill($('gm-vru'), voices.filter(v => /^ru|^uk|any/.test((t.voices[v] || {}).language || v) || t.provider !== 'piper'), cfg.voice_ru);
    fill($('gm-ven'), voices.filter(v => /^en|any/.test((t.voices[v] || {}).language || v) || t.provider !== 'piper'), cfg.voice_en);
    $('gm-status').replaceChildren(
      ok(c.ok, c.ok ? `ComfyUI ${c.version || ''} — ${c.device || ''}` + (c.vram_total ? `, VRAM ${(c.vram_total / 2 ** 30).toFixed(1)} ГБ` : '') : `ComfyUI недоступен (${cfg.comfy_url}): ${c.error || ''}`),
      c.ok ? ok(c.checkpoint_ok, c.checkpoint_ok ? `Модель: ${cfg.checkpoint}` : `Модель не выбрана или не найдена (в ComfyUI: ${(c.checkpoints || []).length} шт.)`) : null,
      c.ok ? ok(c.ip_adapter, c.ip_adapter ? 'IP-Adapter установлен' : 'IP-Adapter не установлен — внешность держится описанием и seed') : null,
      ok(t.ok, t.ok ? `Озвучка: ${t.provider}, голосов: ${voices.length}` : `Озвучка: ${t.error || 'недоступна'}`));
    $('gm-rec').replaceChildren(h('table', {class: 'cs-cmp'}, h('tr', {}, h('th', {text: 'Изображения'}), h('th', {text: 'Файл'}), h('th', {text: 'Размер'}), h('th', {text: 'VRAM'}), h('th', {text: 'Для чего'})),
      ...d.recommended.images.map(m => h('tr', {}, h('td', {text: m.name}), h('td', {text: m.file}), h('td', {text: m.size}), h('td', {text: m.vram}), h('td', {text: m.for})))),
      h('table', {class: 'cs-cmp'}, h('tr', {}, h('th', {text: 'Голоса Piper'}), h('th', {text: 'Для чего'}), h('th', {text: 'Размер'})),
        ...d.recommended.voices.map(v => h('tr', {}, h('td', {text: v.name}), h('td', {text: v.for}), h('td', {text: v.size})))));
  }
  form.onsubmit = async (e) => {
    e.preventDefault();
    const body = {};
    for (const el of form.elements) if (el.name) body[el.name] = el.type === 'checkbox' ? el.checked : el.value;
    const r = await fetch('/api/media/config/', {method: 'POST', headers: {'Content-Type': 'application/json', 'X-CSRFToken': csrf()}, body: JSON.stringify(body)});
    $('gm-saved').textContent = r.ok ? 'Сохранено.' : 'Не удалось сохранить.'; check();
  };
  $('gm-check').onclick = check;
  check();
})();

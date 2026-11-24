// Live operations of a running job: progress, elapsed / remaining time, signs of life, CPU/GPU load.
(function () {
  const fmt = s => { if (s == null) return '—'; s = Math.max(0, Math.round(s)); const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60), x = s % 60; return h ? `${h}:${String(m).padStart(2, '0')}:${String(x).padStart(2, '0')}` : `${m}:${String(x).padStart(2, '0')}`; };
  function el(tag, attrs, ...kids) { const e = document.createElement(tag); Object.assign(e, attrs || {}); e.append(...kids.filter(k => k != null)); return e; }
  function opRow(o) {
    const pct = o.frac != null ? Math.round(o.frac * 100) : null;
    const bar = el('div', {className: 'livebar' + (pct == null ? ' indet' : '')}, el('i', {style: pct == null ? '' : `width:${pct}%`}));
    const times = `прошло ${fmt(o.elapsed)}` + (o.eta != null ? ` · осталось ~${fmt(o.eta)}` : (pct == null ? ' · время неизвестно' : ''));
    return el('div', {className: 'liveop'}, el('div', {className: 'liveop-h'}, el('span', {textContent: o.label}), el('small', {textContent: times})), bar);
  }
  function loadLine(l) {
    if (!l) return null;
    const parts = [];
    if (l.cpu != null) parts.push(`CPU ${Math.round(l.cpu)}% (${l.cores} ядер)`);
    if (l.gpu) {
      const g = l.gpu;
      parts.push(`GPU ${g.util ?? '—'}%` + (g.decoder != null ? ` · NVDEC ${g.decoder}%` : '') + (g.encoder != null ? ` · NVENC ${g.encoder}%` : ''));
      if (g.mem_total) parts.push(`VRAM ${(g.mem_used / 1024).toFixed(1)}/${(g.mem_total / 1024).toFixed(1)} ГБ`);
    } else parts.push('GPU: нет данных (nvidia-smi недоступен)');
    return el('div', {className: 'liveload', textContent: parts.join(' · ')});
  }
  window.renderLive = function (job, load) {
    const box = el('div', {className: 'live'});
    if (!job) return box;
    if (job.status === 'running') {
      const ok = job.alive;
      box.append(el('div', {className: 'livepulse ' + (ok ? 'ok' : 'bad'), textContent: ok ? 'обработка активна' : `нет признаков жизни ${fmt(job.heartbeat_age)}`}));
      if (job.stalled_for) box.append(el('div', {className: 'livewarn', textContent: `Прогресс не менялся ${fmt(job.stalled_for)}. Если так продолжится, процесс будет остановлен автоматически, а обработку можно продолжить из кэша.`}));
      for (const o of job.ops || []) box.append(opRow(o));
      box.append(loadLine(load));
    }
    const t = job.timings || {};
    const keys = Object.keys(t);
    if (keys.length && job.status !== 'running') box.append(el('small', {className: 'muted', textContent: 'Время этапов: ' + keys.map(k => `${k} ${fmt(t[k])}`).join(' · ')}));
    return box;
  };
})();

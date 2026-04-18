/**
 * Beat Saber map canvas preview — 3D perspective + audio sync
 *
 * Usage:  initMapPreview(containerElementId, jobId)
 */
(function (global) {
    'use strict';

    // ── Constants ─────────────────────────────────────────────────────────────
    const COLORS      = ['#ff3344', '#3377ff'];
    const LOOK_AHEAD  = 4.5;    // beats visible ahead of hit line
    const LOOK_BACK   = 0.18;   // beats shown after hit (fade out)
    const VP_Y_FRAC   = 0.08;   // vanishing point (fraction of canvas height)
    const HIT_Y_FRAC  = 0.82;   // hit line (fraction of canvas height)
    const MIN_SCALE   = 0.07;   // note scale at maximum distance
    const TRACK_FRAC  = 0.80;   // track width at near end (fraction of canvas width)

    // cut-direction → rotation in radians
    // 0=up 1=down 2=left 3=right 4=upLeft 5=upRight 6=downLeft 7=downRight 8=dot
    const DIR_ROT = [
        0, Math.PI, -Math.PI / 2, Math.PI / 2,
        -Math.PI / 4, Math.PI / 4, -3 * Math.PI / 4, 3 * Math.PI / 4
    ];

    // ── Helpers ───────────────────────────────────────────────────────────────
    function rrect(ctx, x, y, w, h, r) {
        if (ctx.roundRect) { ctx.roundRect(x, y, w, h, r); return; }
        const d = Math.min(r, w / 2, h / 2);
        ctx.moveTo(x + d, y); ctx.lineTo(x + w - d, y);
        ctx.quadraticCurveTo(x + w, y, x + w, y + d);
        ctx.lineTo(x + w, y + h - d);
        ctx.quadraticCurveTo(x + w, y + h, x + w - d, y + h);
        ctx.lineTo(x + d, y + h);
        ctx.quadraticCurveTo(x, y + h, x, y + h - d);
        ctx.lineTo(x, y + d);
        ctx.quadraticCurveTo(x, y, x + d, y);
        ctx.closePath();
    }

    function lighten(hex, a) {
        const c = parseInt(hex.replace('#', ''), 16);
        const r = (c >> 16) & 0xff, g = (c >> 8) & 0xff, b = c & 0xff;
        const m = v => Math.min(255, Math.round(v + (255 - v) * a));
        return `rgb(${m(r)},${m(g)},${m(b)})`;
    }
    function darken(hex, a) {
        const c = parseInt(hex.replace('#', ''), 16);
        const r = (c >> 16) & 0xff, g = (c >> 8) & 0xff, b = c & 0xff;
        const m = v => Math.round(v * (1 - a));
        return `rgb(${m(r)},${m(g)},${m(b)})`;
    }
    function fmtTime(s) {
        return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
    }

    // ── MapPreview class ──────────────────────────────────────────────────────
    class MapPreview {
        constructor(container, jobId) {
            this._c      = container;
            this._jobId  = jobId;
            this._data   = null;
            this._diffIdx = 0;
            this._beat   = 0;
            this._playing = false;
            this._rafId  = null;
            this._lastTs = null;
            this._totalBeats = 0;
            this._audio  = null;

            this._c.innerHTML = '<p class="mp-loading">Loading preview\u2026</p>';
            this._load();
        }

        async _load() {
            try {
                const res = await fetch(`/api/preview/${this._jobId}`);
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                this._data = await res.json();
                this._buildUI();
            } catch (e) {
                this._c.innerHTML =
                    `<p class="mp-loading mp-error">\u26a0\ufe0f Preview unavailable: ${e.message}</p>`;
            }
        }

        _buildUI() {
            const d = this._data;
            this._totalBeats = d.duration * d.bpm / 60;
            this._diffIdx    = d.difficulties.length - 1;

            const opts = d.difficulties
                .map((x, i) => `<option value="${i}"${i === this._diffIdx ? ' selected' : ''}>${x.name}</option>`)
                .join('');

            this._c.innerHTML = `
                <canvas class="mp-canvas"></canvas>
                <div class="mp-bar">
                    <button class="mp-btn" id="mp-play">\u25b6</button>
                    <button class="mp-btn mp-secondary" id="mp-reset">\u23ee</button>
                    <input  class="mp-seek" type="range" min="0" max="1000" value="0" id="mp-seek">
                    <span   class="mp-time" id="mp-time">0:00 / ${fmtTime(d.duration)}</span>
                    <select class="mp-diff" id="mp-diff">${opts}</select>
                </div>`;

            this._canvas = this._c.querySelector('canvas');
            this._ctx    = this._canvas.getContext('2d');

            // Hidden audio element for playback sync
            this._audio = new Audio(`/api/audio/${this._jobId}`);
            this._audio.preload = 'auto';

            this._c.querySelector('#mp-play') .addEventListener('click',  () => this._togglePlay());
            this._c.querySelector('#mp-reset').addEventListener('click',  () => this._reset());
            this._c.querySelector('#mp-diff') .addEventListener('change', e  => { this._diffIdx = +e.target.value; this._drawFrame(); });
            this._c.querySelector('#mp-seek') .addEventListener('input',  e  => {
                const frac = +e.target.value / 1000;
                this._beat = frac * this._totalBeats;
                if (this._audio) this._audio.currentTime = frac * d.duration;
                if (!this._playing) this._drawFrame();
            });

            this._resize();
            window.addEventListener('resize', () => this._resize());
            this._drawFrame();
        }

        _resize() {
            const w = this._c.clientWidth || 640;
            this._canvas.width  = w;
            this._canvas.height = Math.round(w * 0.54);
            this._drawFrame();
        }

        // ── Playback ──────────────────────────────────────────────────────────
        _togglePlay() {
            if (this._playing) {
                this._playing = false;
                cancelAnimationFrame(this._rafId);
                this._audio?.pause();
                this._c.querySelector('#mp-play').textContent = '\u25b6';
            } else {
                if (this._beat >= this._totalBeats) this._beat = 0;
                this._playing = true;
                this._lastTs  = null;
                if (this._audio) {
                    this._audio.currentTime = this._beat * 60 / this._data.bpm;
                    this._audio.play().catch(() => {/* autoplay policy: silently fall back to timer */});
                }
                this._c.querySelector('#mp-play').textContent = '\u23f8';
                this._rafId = requestAnimationFrame(ts => this._tick(ts));
            }
        }

        _reset() {
            this._playing = false;
            cancelAnimationFrame(this._rafId);
            this._beat = 0;
            if (this._audio) { this._audio.pause(); this._audio.currentTime = 0; }
            const btn = this._c.querySelector('#mp-play');
            if (btn) btn.textContent = '\u25b6';
            this._updateControls();
            this._drawFrame();
        }

        _tick(ts) {
            if (!this._playing) return;

            const audio = this._audio;
            if (audio && !audio.paused && !audio.ended && audio.readyState >= 3) {
                // Audio-driven time — most accurate
                this._beat = audio.currentTime * this._data.bpm / 60;
            } else if (this._lastTs !== null) {
                // Fallback: manual RAF timer
                this._beat = Math.min(
                    this._beat + (ts - this._lastTs) / 1000 * this._data.bpm / 60,
                    this._totalBeats);
            }
            this._lastTs = ts;

            this._updateControls();
            this._drawFrame();

            if (this._beat < this._totalBeats) {
                this._rafId = requestAnimationFrame(t => this._tick(t));
            } else {
                this._playing = false;
                audio?.pause();
                const btn = this._c.querySelector('#mp-play');
                if (btn) btn.textContent = '\u25b6';
            }
        }

        _updateControls() {
            const frac = this._totalBeats > 0 ? this._beat / this._totalBeats : 0;
            const seek = this._c.querySelector('#mp-seek');
            if (seek && document.activeElement !== seek) seek.value = Math.round(frac * 1000);
            const timeEl = this._c.querySelector('#mp-time');
            if (timeEl) timeEl.textContent =
                `${fmtTime(this._beat * 60 / this._data.bpm)} / ${fmtTime(this._data.duration)}`;
        }

        // ── Perspective projection ─────────────────────────────────────────────
        // lane 0-3, row 0-2, beatAhead (0 = at hit line, LOOK_AHEAD = far away)
        _project(lane, row, beatAhead, CW, CH) {
            const hitY = CH * HIT_Y_FRAC;
            const vpY  = CH * VP_Y_FRAC;
            const t     = Math.max(0, Math.min(1, beatAhead / LOOK_AHEAD));
            const scale = MIN_SCALE + (1 - MIN_SCALE) * (1 - t);

            const trackHalfW = CW * TRACK_FRAC * 0.5;
            const laneUnit   = trackHalfW * scale * 2 / 4;        // lane width at this depth
            const noteSize   = Math.min(CW / 4, hitY - vpY) * 0.20 * scale / MIN_SCALE * MIN_SCALE;
            // Simpler note size: use a fixed base scaled by perspective
            const ns = Math.min(CW * 0.18, (hitY - vpY) * 0.22) * scale;
            const rowH = ns * 1.18;

            const x = CW / 2 + (lane - 1.5) * laneUnit;
            const y = (hitY + (vpY - hitY) * t) - (row - 1) * rowH;

            return { x, y, ns, scale, t };
        }

        // ── Drawing ────────────────────────────────────────────────────────────
        _drawFrame() {
            if (!this._ctx || !this._data) return;
            const ctx  = this._ctx;
            const CW   = this._canvas.width;
            const CH   = this._canvas.height;
            const beat = this._beat;
            const hitY = CH * HIT_Y_FRAC;
            const vpY  = CH * VP_Y_FRAC;

            // Background gradient
            ctx.clearRect(0, 0, CW, CH);
            const bg = ctx.createLinearGradient(0, 0, 0, CH);
            bg.addColorStop(0, '#04040e');
            bg.addColorStop(1, '#080820');
            ctx.fillStyle = bg;
            ctx.fillRect(0, 0, CW, CH);

            this._drawTrack(ctx, CW, CH, vpY, hitY, beat);
            this._drawHitLine(ctx, CW, hitY);

            // Notes — sorted far-to-near (painter's algorithm)
            const diff  = this._data.difficulties[this._diffIdx];
            const notes = diff?.notes ?? [];
            const visible = notes
                .filter(n => n.b >= beat - LOOK_BACK && n.b <= beat + LOOK_AHEAD)
                .sort((a, b) => b.b - a.b);

            for (const n of visible) {
                const ahead = n.b - beat;
                if (ahead < 0) {
                    // Just passed — show at hit line fading out
                    const alpha = Math.max(0, 1 + ahead / LOOK_BACK);
                    ctx.save();
                    ctx.globalAlpha = alpha;
                    this._drawNote(ctx, this._project(n.x, n.y, 0, CW, CH), n.c, n.d);
                    ctx.restore();
                } else {
                    this._drawNote(ctx, this._project(n.x, n.y, ahead, CW, CH), n.c, n.d);
                }
            }

            // Song progress strip
            ctx.fillStyle = '#ffffff10';
            ctx.fillRect(0, 0, CW, 3);
            ctx.fillStyle = '#3377ffcc';
            ctx.fillRect(0, 0, CW * Math.min(1, this._totalBeats > 0 ? beat / this._totalBeats : 0), 3);
        }

        _drawTrack(ctx, CW, CH, vpY, hitY, beat) {
            const hw  = CW * TRACK_FRAC * 0.5;
            const hwF = hw * MIN_SCALE;

            // Track floor fill
            ctx.save();
            ctx.beginPath();
            ctx.moveTo(CW / 2 - hw,  hitY);
            ctx.lineTo(CW / 2 + hw,  hitY);
            ctx.lineTo(CW / 2 + hwF, vpY);
            ctx.lineTo(CW / 2 - hwF, vpY);
            ctx.closePath();
            const floor = ctx.createLinearGradient(0, hitY, 0, vpY);
            floor.addColorStop(0, '#0c1248');
            floor.addColorStop(1, '#06081a');
            ctx.fillStyle = floor;
            ctx.fill();
            ctx.restore();

            // Lane dividers (5 converging lines)
            for (let i = 0; i <= 4; i++) {
                const nx = CW / 2 + (i - 2) * hw * 2 / 4;
                const fx = CW / 2 + (i - 2) * hwF * 2 / 4;
                const lg = ctx.createLinearGradient(0, hitY, 0, vpY);
                lg.addColorStop(0, i === 0 || i === 4 ? '#3355cc66' : '#2244aa44');
                lg.addColorStop(1, '#2244aa11');
                ctx.strokeStyle = lg;
                ctx.lineWidth   = i === 0 || i === 4 ? 1.5 : 0.7;
                ctx.beginPath(); ctx.moveTo(nx, hitY); ctx.lineTo(fx, vpY); ctx.stroke();
            }

            // Beat grid lines receding into perspective
            const frac = beat % 1;
            for (let i = 0; i <= Math.ceil(LOOK_AHEAD); i++) {
                const off = i - frac;
                if (off < 0 || off > LOOK_AHEAD) continue;
                const t  = off / LOOK_AHEAD;
                const y  = hitY + (vpY - hitY) * t;
                const hw2 = hw * (MIN_SCALE + (1 - MIN_SCALE) * (1 - t));
                const bar = Math.round(beat + off) % 4 === 0;
                ctx.strokeStyle = bar ? '#3355cc44' : '#2244aa22';
                ctx.lineWidth   = bar ? 1.2 : 0.6;
                ctx.beginPath(); ctx.moveTo(CW / 2 - hw2, y); ctx.lineTo(CW / 2 + hw2, y); ctx.stroke();
            }
        }

        _drawHitLine(ctx, CW, hitY) {
            const hw = CW * TRACK_FRAC * 0.5;

            // Glow under the hit line
            const glow = ctx.createLinearGradient(0, hitY - 4, 0, hitY + 28);
            glow.addColorStop(0, '#4488ff55');
            glow.addColorStop(1, 'transparent');
            ctx.fillStyle = glow;
            ctx.fillRect(CW / 2 - hw, hitY - 4, hw * 2, 32);

            // Hit line itself
            ctx.strokeStyle = '#5599ffbb';
            ctx.lineWidth   = 1.5;
            ctx.beginPath(); ctx.moveTo(CW / 2 - hw, hitY); ctx.lineTo(CW / 2 + hw, hitY); ctx.stroke();

            // Target rings per lane
            const laneW = hw * 2 / 4;
            for (let i = 0; i < 4; i++) {
                const cx = CW / 2 - hw + i * laneW + laneW / 2;
                ctx.strokeStyle = '#4488ff33';
                ctx.lineWidth   = 1;
                ctx.beginPath(); ctx.arc(cx, hitY, laneW * 0.34, 0, Math.PI * 2); ctx.stroke();
            }
        }

        _drawNote(ctx, { x, y, ns: sz, scale }, color, dir) {
            const col   = COLORS[color] ?? '#aaaaaa';
            const r     = Math.max(2, 5 * scale);
            const bevel = sz * 0.11;

            ctx.save();

            // ── 3D depth shadow (offset face) ─────────────────────────────────
            ctx.shadowColor = 'transparent';
            ctx.beginPath();
            rrect(ctx, x - sz / 2 + bevel, y - sz / 2 + bevel, sz, sz, r);
            ctx.fillStyle = darken(col, 0.60);
            ctx.fill();

            // ── Main face with glow ───────────────────────────────────────────
            ctx.shadowColor = col;
            ctx.shadowBlur  = sz * 1.0;
            ctx.beginPath();
            rrect(ctx, x - sz / 2, y - sz / 2, sz, sz, r);
            const grad = ctx.createLinearGradient(x - sz / 2, y - sz / 2, x + sz / 2, y + sz / 2);
            grad.addColorStop(0, lighten(col, 0.38));
            grad.addColorStop(1, col);
            ctx.fillStyle = grad;
            ctx.fill();

            // ── Border ────────────────────────────────────────────────────────
            ctx.shadowBlur  = 0;
            ctx.strokeStyle = lighten(col, 0.55);
            ctx.lineWidth   = Math.max(0.5, 1.5 * scale);
            ctx.stroke();

            // ── Top-left highlight strip (sells the 3D face) ──────────────────
            ctx.beginPath();
            rrect(ctx, x - sz / 2 + 2, y - sz / 2 + 2, sz * 0.45, sz * 0.12, r * 0.5);
            ctx.fillStyle = 'rgba(255,255,255,0.22)';
            ctx.fill();

            // ── Direction arrow or dot ────────────────────────────────────────
            if (dir === 8) {
                ctx.beginPath();
                ctx.arc(x, y, sz * 0.14, 0, Math.PI * 2);
                ctx.fillStyle = '#ffffffee';
                ctx.fill();
            } else {
                const a = sz * 0.27;
                ctx.translate(x, y);
                ctx.rotate(DIR_ROT[dir] ?? 0);
                ctx.beginPath();
                ctx.moveTo(0, -a);
                ctx.lineTo(a * 0.65, a * 0.38);
                ctx.lineTo(0, 0);
                ctx.lineTo(-a * 0.65, a * 0.38);
                ctx.closePath();
                ctx.fillStyle = '#ffffffee';
                ctx.fill();
            }

            ctx.restore();
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────
    global.initMapPreview = function (containerId, jobId) {
        const el = document.getElementById(containerId);
        if (!el) { console.warn('mapPreview: container not found:', containerId); return; }
        new MapPreview(el, jobId);
    };

}(window));
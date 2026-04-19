/**
 * Beat Saber preview with two useful views:
 * - Runner: stronger 3D tunnel/player camera
 * - Timing: explicit beat-arrival strip for judging sync
 * - Split: both at once
 */
(function (global) {
    'use strict';

    const COLORS = ['#ff3855', '#3b82ff'];
    const LOOK_AHEAD = 4.75;
    const LOOK_BACK = 0.22;
    const RUNNER_VP_Y = 0.07;
    const RUNNER_HIT_Y = 0.83;
    const RUNNER_TRACK = 0.76;
    const MIN_SCALE = 0.06;
    const TIMING_WINDOW = 1.5;
    const TIMING_PAST = 0.45;
    const DIR_ROT = [
        0, Math.PI, -Math.PI / 2, Math.PI / 2,
        -Math.PI / 4, Math.PI / 4, -3 * Math.PI / 4, 3 * Math.PI / 4
    ];

    function fmtTime(s) {
        return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
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

    function roundedRect(ctx, x, y, w, h, r) {
        if (ctx.roundRect) {
            ctx.roundRect(x, y, w, h, r);
            return;
        }
        const d = Math.min(r, w / 2, h / 2);
        ctx.moveTo(x + d, y);
        ctx.lineTo(x + w - d, y);
        ctx.quadraticCurveTo(x + w, y, x + w, y + d);
        ctx.lineTo(x + w, y + h - d);
        ctx.quadraticCurveTo(x + w, y + h, x + w - d, y + h);
        ctx.lineTo(x + d, y + h);
        ctx.quadraticCurveTo(x, y + h, x, y + h - d);
        ctx.lineTo(x, y + d);
        ctx.quadraticCurveTo(x, y, x + d, y);
        ctx.closePath();
    }

    class MapPreview {
        constructor(container, jobId) {
            this._c = container;
            this._jobId = jobId;
            this._data = null;
            this._diffIdx = 0;
            this._beat = 0;
            this._playing = false;
            this._rafId = null;
            this._lastTs = null;
            this._totalBeats = 0;
            this._audio = null;
            this._mode = 'split';

            this._c.innerHTML = '<p class="mp-loading">Loading preview…</p>';
            this._load();
        }

        async _load() {
            try {
                const res = await fetch(`/api/preview/${this._jobId}`);
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                this._data = await res.json();
                this._buildUI();
            } catch (e) {
                this._c.innerHTML = `<p class="mp-loading mp-error">Preview unavailable: ${e.message}</p>`;
            }
        }

        _currentDifficulty() {
            return this._data?.difficulties?.[this._diffIdx] ?? null;
        }

        _currentTimingPoints() {
            return this._currentDifficulty()?.timingPoints ?? [{ b: 0, t: 0, m: this._data?.bpm ?? 120 }];
        }

        _currentDuration() {
            return this._currentDifficulty()?.duration ?? this._data?.duration ?? 60;
        }

        _beatToSeconds(beat) {
            const timing = this._currentTimingPoints();
            let tp = timing[0];
            for (const point of timing) {
                if (point.b <= beat) tp = point;
                else break;
            }
            return tp.t + (beat - tp.b) * 60 / tp.m;
        }

        _secondsToBeat(seconds) {
            const timing = this._currentTimingPoints();
            let tp = timing[0];
            for (const point of timing) {
                if (point.t <= seconds) tp = point;
                else break;
            }
            return tp.b + (seconds - tp.t) * tp.m / 60;
        }

        _buildUI() {
            const d = this._data;
            this._diffIdx = d.difficulties.length - 1;

            const opts = d.difficulties
                .map((x, i) => `<option value="${i}"${i === this._diffIdx ? ' selected' : ''}>${x.name}</option>`)
                .join('');

            this._c.innerHTML = `
                <div class="mp-stage-wrap mp-mode-split">
                    <canvas class="mp-canvas mp-canvas-runner"></canvas>
                    <canvas class="mp-canvas mp-canvas-timing"></canvas>
                </div>
                <div class="mp-bar">
                    <button class="mp-btn" id="mp-play">▶</button>
                    <button class="mp-btn mp-secondary" id="mp-reset">⏮</button>
                    <input class="mp-seek" type="range" min="0" max="1000" value="0" id="mp-seek">
                    <span class="mp-time" id="mp-time">0:00 / ${fmtTime(d.duration)}</span>
                    <div class="mp-modes">
                        <button class="mp-mode is-active" data-mode="split">Split</button>
                        <button class="mp-mode" data-mode="runner">3D</button>
                        <button class="mp-mode" data-mode="timing">Timing</button>
                    </div>
                    <select class="mp-diff" id="mp-diff">${opts}</select>
                </div>`;

            this._runnerCanvas = this._c.querySelector('.mp-canvas-runner');
            this._timingCanvas = this._c.querySelector('.mp-canvas-timing');
            this._runnerCtx = this._runnerCanvas.getContext('2d');
            this._timingCtx = this._timingCanvas.getContext('2d');
            this._stageWrap = this._c.querySelector('.mp-stage-wrap');

            this._audio = new Audio(`/api/audio/${this._jobId}`);
            this._audio.preload = 'auto';

            this._c.querySelector('#mp-play').addEventListener('click', () => this._togglePlay());
            this._c.querySelector('#mp-reset').addEventListener('click', () => this._reset());
            this._c.querySelector('#mp-diff').addEventListener('change', e => {
                this._diffIdx = +e.target.value;
                const duration = this._currentDuration();
                const clampedSeconds = Math.min(this._beatToSeconds(this._beat), duration);
                if (this._audio) this._audio.currentTime = clampedSeconds;
                this._beat = this._secondsToBeat(clampedSeconds);
                this._updateControls();
                this._drawFrame();
            });
            this._c.querySelector('#mp-seek').addEventListener('input', e => {
                const frac = +e.target.value / 1000;
                const seconds = frac * this._currentDuration();
                this._beat = this._secondsToBeat(seconds);
                if (this._audio) this._audio.currentTime = seconds;
                if (!this._playing) this._drawFrame();
            });
            this._c.querySelectorAll('.mp-mode').forEach(btn => btn.addEventListener('click', () => {
                this._mode = btn.dataset.mode;
                this._c.querySelectorAll('.mp-mode').forEach(x => x.classList.toggle('is-active', x === btn));
                this._stageWrap.className = `mp-stage-wrap mp-mode-${this._mode}`;
                this._resize();
            }));

            this._resize();
            window.addEventListener('resize', () => this._resize());
            this._drawFrame();
        }

        _resize() {
            const w = this._c.clientWidth || 760;
            const runnerH = Math.round(w * 0.60);
            const timingH = this._mode === 'timing'
                ? Math.max(260, Math.round(w * 0.40))
                : Math.max(190, Math.round(w * 0.28));
            this._runnerCanvas.width = w;
            this._runnerCanvas.height = runnerH;
            this._timingCanvas.width = w;
            this._timingCanvas.height = timingH;
            this._drawFrame();
        }

        _togglePlay() {
            if (this._playing) {
                this._playing = false;
                cancelAnimationFrame(this._rafId);
                this._audio?.pause();
                this._c.querySelector('#mp-play').textContent = '▶';
                return;
            }

            const duration = this._currentDuration();
            if (this._beatToSeconds(this._beat) >= duration) this._beat = 0;
            this._playing = true;
            this._lastTs = null;
            if (this._audio) {
                this._audio.currentTime = this._beatToSeconds(this._beat);
                this._audio.play().catch(() => {});
            }
            this._c.querySelector('#mp-play').textContent = '⏸';
            this._rafId = requestAnimationFrame(ts => this._tick(ts));
        }

        _reset() {
            this._playing = false;
            cancelAnimationFrame(this._rafId);
            this._beat = 0;
            if (this._audio) {
                this._audio.pause();
                this._audio.currentTime = 0;
            }
            this._c.querySelector('#mp-play').textContent = '▶';
            this._updateControls();
            this._drawFrame();
        }

        _tick(ts) {
            if (!this._playing) return;

            const audio = this._audio;
            const duration = this._currentDuration();
            if (audio && !audio.paused && !audio.ended && audio.readyState >= 3) {
                this._beat = this._secondsToBeat(audio.currentTime);
            } else if (this._lastTs !== null) {
                const nextSeconds = Math.min(this._beatToSeconds(this._beat) + (ts - this._lastTs) / 1000, duration);
                this._beat = this._secondsToBeat(nextSeconds);
            }
            this._lastTs = ts;

            this._updateControls();
            this._drawFrame();

            if (this._beatToSeconds(this._beat) < duration) {
                this._rafId = requestAnimationFrame(t => this._tick(t));
            } else {
                this._playing = false;
                audio?.pause();
                this._c.querySelector('#mp-play').textContent = '▶';
            }
        }

        _updateControls() {
            const currentSeconds = this._beatToSeconds(this._beat);
            const duration = this._currentDuration();
            const frac = duration > 0 ? currentSeconds / duration : 0;
            const seek = this._c.querySelector('#mp-seek');
            if (seek && document.activeElement !== seek) seek.value = Math.round(frac * 1000);
            this._c.querySelector('#mp-time').textContent =
                `${fmtTime(currentSeconds)} / ${fmtTime(duration)}`;
        }

        _projectRunner(lane, row, beatAhead, width, height) {
            const hitY = height * RUNNER_HIT_Y;
            const vpY = height * RUNNER_VP_Y;
            const t = Math.max(0, Math.min(1, beatAhead / LOOK_AHEAD));
            const scale = MIN_SCALE + (1 - MIN_SCALE) * (1 - t);
            const halfW = width * RUNNER_TRACK * 0.5;
            const laneUnit = halfW * scale * 2 / 4;
            const size = Math.min(width * 0.16, (hitY - vpY) * 0.20) * scale;
            const perspectiveHeight = Math.max(28, (hitY - vpY) * (0.16 + 0.10 * (1 - t)));
            const rowH = perspectiveHeight / 3;
            const x = width / 2 + (lane - 1.5) * laneUnit;
            const y = (hitY + (vpY - hitY) * t) - (row - 1) * rowH;
            return { x, y, size, scale, t, hitY, vpY, halfW, rowH };
        }

        _drawFrame() {
            if (!this._data) return;
            this._drawRunnerFrame();
            this._drawTimingFrame();
        }

        _drawRunnerFrame() {
            const ctx = this._runnerCtx;
            const width = this._runnerCanvas.width;
            const height = this._runnerCanvas.height;
            const beat = this._beat;
            const hitY = height * RUNNER_HIT_Y;
            const vpY = height * RUNNER_VP_Y;
            const halfW = width * RUNNER_TRACK * 0.5;
            const beatPulse = 1 - Math.min(1, Math.abs((beat % 1) - 0) * 4);

            ctx.clearRect(0, 0, width, height);

            const bg = ctx.createLinearGradient(0, 0, 0, height);
            bg.addColorStop(0, '#04050c');
            bg.addColorStop(0.55, '#091124');
            bg.addColorStop(1, '#05070f');
            ctx.fillStyle = bg;
            ctx.fillRect(0, 0, width, height);

            // Tunnel walls
            this._fillQuad(ctx,
                width / 2 - halfW, hitY,
                width / 2 - halfW * 0.20, vpY,
                0, height,
                0, 0,
                '#081425', '#02050b');
            this._fillQuad(ctx,
                width / 2 + halfW, hitY,
                width / 2 + halfW * 0.20, vpY,
                width, height,
                width, 0,
                '#081425', '#02050b');

            // Floor
            const floor = ctx.createLinearGradient(0, hitY, 0, vpY);
            floor.addColorStop(0, '#101a34');
            floor.addColorStop(1, '#060913');
            ctx.beginPath();
            ctx.moveTo(width / 2 - halfW, hitY);
            ctx.lineTo(width / 2 + halfW, hitY);
            ctx.lineTo(width / 2 + halfW * MIN_SCALE, vpY);
            ctx.lineTo(width / 2 - halfW * MIN_SCALE, vpY);
            ctx.closePath();
            ctx.fillStyle = floor;
            ctx.fill();

            // Side sabers / lane glow
            this._drawLaneRails(ctx, width, hitY, vpY, halfW);
            this._drawRunnerGrid(ctx, width, hitY, vpY, halfW, beat, beatPulse);
            this._drawRunnerHitZone(ctx, width, height, hitY, halfW, beatPulse);

            const notes = (this._data.difficulties[this._diffIdx]?.notes ?? [])
                .filter(n => n.b >= beat - LOOK_BACK && n.b <= beat + LOOK_AHEAD)
                .sort((a, b) => b.b - a.b);

            for (const note of notes) {
                const ahead = note.b - beat;
                const proj = this._projectRunner(note.x, note.y, Math.max(0, ahead), width, height);
                if (ahead < 0) {
                    ctx.save();
                    ctx.globalAlpha = Math.max(0, 1 + ahead / LOOK_BACK);
                    this._drawRunnerNote(ctx, proj, note.c, note.d);
                    ctx.restore();
                } else {
                    this._drawRunnerNote(ctx, proj, note.c, note.d);
                }
            }
        }

        _drawLaneRails(ctx, width, hitY, vpY, halfW) {
            const near = halfW;
            const far = halfW * MIN_SCALE;
            for (let lane = 0; lane <= 4; lane++) {
                const nx = width / 2 + (lane - 2) * near * 2 / 4;
                const fx = width / 2 + (lane - 2) * far * 2 / 4;
                const rail = ctx.createLinearGradient(0, hitY, 0, vpY);
                rail.addColorStop(0, lane <= 2 ? '#ff375533' : '#3b82ff33');
                rail.addColorStop(1, '#7dd3fc12');
                ctx.strokeStyle = rail;
                ctx.lineWidth = lane === 0 || lane === 4 ? 2 : 1;
                ctx.beginPath();
                ctx.moveTo(nx, hitY);
                ctx.lineTo(fx, vpY);
                ctx.stroke();
            }
        }

        _drawRunnerGrid(ctx, width, hitY, vpY, halfW, beat, beatPulse) {
            const frac = beat % 1;
            for (let i = 0; i <= Math.ceil(LOOK_AHEAD) + 1; i++) {
                const off = i - frac;
                if (off < 0 || off > LOOK_AHEAD) continue;
                const t = off / LOOK_AHEAD;
                const y = hitY + (vpY - hitY) * t;
                const hw = halfW * (MIN_SCALE + (1 - MIN_SCALE) * (1 - t));
                const bar = Math.round(beat + off) % 4 === 0;
                ctx.strokeStyle = bar ? '#8bd8ff55' : '#4b78d822';
                ctx.lineWidth = bar ? 1.4 : 0.7;
                ctx.beginPath();
                ctx.moveTo(width / 2 - hw, y);
                ctx.lineTo(width / 2 + hw, y);
                ctx.stroke();

                const rowStep = Math.max(10, (hitY - vpY) * (0.16 + 0.10 * (1 - t)) / 3);
                ctx.strokeStyle = bar ? 'rgba(139,216,255,0.20)' : 'rgba(75,120,216,0.12)';
                ctx.lineWidth = 0.8;
                for (let row = 1; row <= 2; row++) {
                    const rowY = y - row * rowStep;
                    ctx.beginPath();
                    ctx.moveTo(width / 2 - hw, rowY);
                    ctx.lineTo(width / 2 + hw, rowY);
                    ctx.stroke();
                }
            }

            ctx.fillStyle = `rgba(123,216,255,${0.04 + beatPulse * 0.08})`;
            ctx.fillRect(width / 2 - halfW, hitY - 24, halfW * 2, 48);
        }

        _drawRunnerHitZone(ctx, width, height, hitY, halfW, beatPulse) {
            const glow = ctx.createLinearGradient(0, hitY - 8, 0, hitY + 40);
            glow.addColorStop(0, `rgba(140,210,255,${0.32 + beatPulse * 0.18})`);
            glow.addColorStop(1, 'transparent');
            ctx.fillStyle = glow;
            ctx.fillRect(width / 2 - halfW, hitY - 8, halfW * 2, 42);

            ctx.strokeStyle = `rgba(150,225,255,${0.75 + beatPulse * 0.2})`;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(width / 2 - halfW, hitY);
            ctx.lineTo(width / 2 + halfW, hitY);
            ctx.stroke();

            const laneW = halfW * 2 / 4;
            for (let i = 0; i < 4; i++) {
                const cx = width / 2 - halfW + i * laneW + laneW / 2;
                ctx.strokeStyle = 'rgba(150,225,255,0.22)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.arc(cx, hitY, laneW * 0.30, 0, Math.PI * 2);
                ctx.stroke();
            }

            ctx.strokeStyle = 'rgba(150,225,255,0.16)';
            ctx.lineWidth = 1;
            for (let row = 1; row <= 2; row++) {
                const rowY = hitY - row * (height * 0.08);
                ctx.beginPath();
                ctx.moveTo(width / 2 - halfW, rowY);
                ctx.lineTo(width / 2 + halfW, rowY);
                ctx.stroke();
            }
        }

        _drawRunnerNote(ctx, proj, colorIdx, dir) {
            const color = COLORS[colorIdx] ?? '#b8c1d9';
            const size = proj.size;
            const bevel = size * 0.11;
            const radius = Math.max(2, 5 * proj.scale);

            ctx.save();
            ctx.beginPath();
            roundedRect(ctx, proj.x - size / 2 + bevel, proj.y - size / 2 + bevel, size, size, radius);
            ctx.fillStyle = darken(color, 0.62);
            ctx.fill();

            ctx.shadowColor = color;
            ctx.shadowBlur = size * 0.9;
            ctx.beginPath();
            roundedRect(ctx, proj.x - size / 2, proj.y - size / 2, size, size, radius);
            const face = ctx.createLinearGradient(proj.x - size / 2, proj.y - size / 2, proj.x + size / 2, proj.y + size / 2);
            face.addColorStop(0, lighten(color, 0.40));
            face.addColorStop(1, color);
            ctx.fillStyle = face;
            ctx.fill();
            ctx.shadowBlur = 0;
            ctx.strokeStyle = lighten(color, 0.55);
            ctx.lineWidth = Math.max(0.8, 1.8 * proj.scale);
            ctx.stroke();

            ctx.beginPath();
            roundedRect(ctx, proj.x - size / 2 + 2, proj.y - size / 2 + 2, size * 0.44, size * 0.12, radius * 0.6);
            ctx.fillStyle = 'rgba(255,255,255,0.22)';
            ctx.fill();

            if (dir === 8) {
                ctx.beginPath();
                ctx.arc(proj.x, proj.y, size * 0.14, 0, Math.PI * 2);
                ctx.fillStyle = '#ffffffee';
                ctx.fill();
            } else {
                const a = size * 0.28;
                ctx.translate(proj.x, proj.y);
                ctx.rotate(DIR_ROT[dir] ?? 0);
                ctx.beginPath();
                ctx.moveTo(0, -a);
                ctx.lineTo(a * 0.68, a * 0.40);
                ctx.lineTo(0, 0);
                ctx.lineTo(-a * 0.68, a * 0.40);
                ctx.closePath();
                ctx.fillStyle = '#ffffffef';
                ctx.fill();
            }
            ctx.restore();
        }

        _drawTimingFrame() {
            const ctx = this._timingCtx;
            const width = this._timingCanvas.width;
            const height = this._timingCanvas.height;
            const beat = this._beat;
            const centerX = width * 0.5;
            const gridTop = 16;
            const gridBottom = height - 18;
            const gridHeight = gridBottom - gridTop;
            const laneHeight = gridHeight / 4;
            const rowHeight = laneHeight / 3;
            const laneLabelX = 26;
            const noteTravelWidth = width * 0.42;
            const beatPhase = 1 - Math.min(1, Math.abs((beat % 1) - 0) * 4);

            ctx.clearRect(0, 0, width, height);
            const bg = ctx.createLinearGradient(0, 0, 0, height);
            bg.addColorStop(0, '#07101c');
            bg.addColorStop(1, '#05070f');
            ctx.fillStyle = bg;
            ctx.fillRect(0, 0, width, height);

            for (let lane = 0; lane < 4; lane++) {
                const laneY = gridTop + lane * laneHeight;
                const laneBg = ctx.createLinearGradient(0, laneY, width, laneY + laneHeight);
                laneBg.addColorStop(0, lane % 2 === 0 ? 'rgba(26,42,72,0.30)' : 'rgba(18,30,52,0.22)');
                laneBg.addColorStop(1, 'rgba(8,12,22,0.08)');
                ctx.fillStyle = laneBg;
                ctx.fillRect(0, laneY, width, laneHeight);

                ctx.fillStyle = 'rgba(160,205,255,0.58)';
                ctx.font = '600 12px Rajdhani, sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(`${lane + 1}`, laneLabelX, laneY + laneHeight / 2);

                for (let row = 0; row < 3; row++) {
                    const rowTop = laneY + row * rowHeight;
                    const alpha = row === 1 ? 0.18 : 0.12;
                    ctx.fillStyle = `rgba(80,140,210,${alpha})`;
                    ctx.fillRect(44, rowTop + 1, width - 52, rowHeight - 2);
                }

                ctx.strokeStyle = 'rgba(120,170,255,0.24)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(44, laneY);
                ctx.lineTo(width - 8, laneY);
                ctx.moveTo(44, laneY + laneHeight);
                ctx.lineTo(width - 8, laneY + laneHeight);
                ctx.stroke();

                ctx.strokeStyle = 'rgba(120,170,255,0.14)';
                for (let row = 1; row < 3; row++) {
                    const y = laneY + row * rowHeight;
                    ctx.beginPath();
                    ctx.moveTo(44, y);
                    ctx.lineTo(width - 8, y);
                    ctx.stroke();
                }
            }

            ctx.fillStyle = 'rgba(135,220,255,0.07)';
            ctx.fillRect(centerX - 10, 0, 20, height);
            ctx.strokeStyle = `rgba(135,220,255,${0.65 + beatPhase * 0.25})`;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(centerX, 8);
            ctx.lineTo(centerX, height - 8);
            ctx.stroke();

            const diff = this._data.difficulties[this._diffIdx];
            const notes = diff?.notes ?? [];
            const visible = notes.filter(n => n.b >= beat - TIMING_PAST && n.b <= beat + TIMING_WINDOW);

            for (const note of visible) {
                const delta = note.b - beat;
                const x = centerX + (delta / TIMING_WINDOW) * noteTravelWidth;
                const laneY = gridTop + note.x * laneHeight;
                const y = laneY + (2 - note.y + 0.5) * rowHeight;
                const w = Math.min(24, rowHeight * 0.78);
                const h = Math.min(24, rowHeight * 0.78);
                ctx.save();
                if (delta < 0) ctx.globalAlpha = Math.max(0, 1 + delta / TIMING_PAST);
                ctx.translate(x, y);
                ctx.rotate((DIR_ROT[note.d] ?? 0) * 0.12);
                ctx.fillStyle = COLORS[note.c] ?? '#b8c1d9';
                ctx.shadowColor = ctx.fillStyle;
                ctx.shadowBlur = 10;
                ctx.fillRect(-w / 2, -h / 2, w, h);
                ctx.shadowBlur = 0;
                ctx.strokeStyle = 'rgba(255,255,255,0.72)';
                ctx.lineWidth = 1;
                ctx.strokeRect(-w / 2, -h / 2, w, h);
                ctx.restore();
            }

            for (let i = -1; i <= 4; i++) {
                const x = centerX + (i / TIMING_WINDOW) * noteTravelWidth;
                const isBeat = i >= 0;
                ctx.strokeStyle = isBeat ? 'rgba(120,170,255,0.26)' : 'rgba(255,130,130,0.20)';
                ctx.lineWidth = isBeat && i % 1 === 0 ? 1.2 : 0.6;
                ctx.beginPath();
                ctx.moveTo(x, height - 18);
                ctx.lineTo(x, height - 2);
                ctx.stroke();
            }

            ctx.fillStyle = 'rgba(180,230,255,0.70)';
            ctx.font = '700 11px Orbitron, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.fillText('NOW', centerX, 6);
        }

        _fillQuad(ctx, x1, y1, x2, y2, x3, y3, x4, y4, c1, c2) {
            const g = ctx.createLinearGradient(0, y1, 0, y3);
            g.addColorStop(0, c1);
            g.addColorStop(1, c2);
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.lineTo(x4, y4);
            ctx.lineTo(x3, y3);
            ctx.closePath();
            ctx.fillStyle = g;
            ctx.fill();
        }
    }

    global.initMapPreview = function (containerId, jobId) {
        const el = document.getElementById(containerId);
        if (!el) return;
        new MapPreview(el, jobId);
    };
}(window));

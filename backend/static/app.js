/* img2vec Web UI - Alpine.js aplikacija (bez build koraka). */

function app() {
  return {
    // --- stanje ---
    health: null,
    image: null,
    result: null,
    svgMarkup: '',
    loading: false,
    error: '',
    notice: '',
    tab: 'vector',
    layers: { engrave: true, cut: true },
    zoom: 1,
    panX: 0,
    panY: 0,
    panning: false,
    panStart: null,
    dragging: false,
    renderTimer: null,
    presets: [],
    presetKey: '',
    presetDescription: '',
    testImages: [],
    job: null,
    jobTimer: null,

    modes: [
      { key: 'stencil', label: 'Puna silueta', hint: 'Fill / gravura + Cut' },
      { key: 'centerline', label: 'Jednostruke linije', hint: 'Line / Score' },
      { key: 'mesh', label: '2D low-poly', hint: 'Faza 2 - mreza' },
      { key: 'wireframe', label: '3D wireframe', hint: 'Faza 3 - dubina + HLR' }
    ],

    params: {
      mode: 'stencil',
      target_width_mm: 150,
      target_height_mm: null,
      quality: 3,
      remove_background: true,
      bg_model: 'isnet-general-use',
      invert: false,
      min_island_area_px: 30,
      threshold_method: 'auto',
      threshold_block: 31,
      threshold_c: 10,
      smooth_radius: 2,
      turd_size: 2,
      opt_tolerance: 0.2,
      line_threshold_method: 'auto',
      min_spur_px: 12,
      min_path_length_mm: 1,
      merge_tolerance_mm: 0.2,
      simplify_tolerance_mm: 0.05,
      add_cut: false,
      cut_offset_mm: 3,
      cut_simplify_mm: 0.05,
      mesh_spacing_mm: 3,
      mesh_edge_threshold: 0.62,
      mesh_max_edge_mm: 14,
      depth_levels: 18,
      depth_parallax: 0.35,
      depth_smoothing: 9
    },

    // --- init ---
    async init() {
      await this.loadHealth();
      await this.loadPresets();
      await this.loadTestImages();
      this.$watch('params', () => this.scheduleRender(), { deep: true });
      // deep link za brzu proveru/demo: /?demo=03_logo_bw.png
      const demo = new URLSearchParams(window.location.search).get('demo');
      if (demo) {
        await this.loadTestImage(demo);
      }
    },

    async api(path, options) {
      const opts = options || {};
      const response = await fetch(path, opts);
      if (!response.ok) {
        let detail = response.status + ' ' + response.statusText;
        try {
          const data = await response.json();
          if (data && data.detail) {
            detail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
          }
        } catch (ignored) { /* nije JSON */ }
        throw new Error(detail);
      }
      const contentType = response.headers.get('content-type') || '';
      if (contentType.indexOf('application/json') >= 0) {
        return response.json();
      }
      return response.text();
    },

    showError(message) {
      this.error = String(message);
      window.setTimeout(() => { if (this.error === String(message)) this.error = ''; }, 9000);
    },

    showNotice(message) {
      this.notice = message;
      window.setTimeout(() => { if (this.notice === message) this.notice = ''; }, 4000);
    },

    async loadHealth() {
      try {
        this.health = await this.api('/api/health');
      } catch (error) {
        this.showError('Backend nije dostupan: ' + error.message);
      }
    },

    async loadPresets() {
      try {
        const data = await this.api('/api/presets');
        this.presets = data.presets || [];
      } catch (error) {
        this.showError(error.message);
      }
    },

    async loadTestImages() {
      try {
        const data = await this.api('/api/test-images');
        this.testImages = data.images || [];
      } catch (error) {
        this.testImages = [];
      }
    },

    // --- ucitavanje slike ---
    onDrop(event) {
      this.dragging = false;
      const files = event.dataTransfer && event.dataTransfer.files;
      if (files && files.length) this.uploadFile(files[0]);
    },

    onFile(event) {
      const files = event.target.files;
      if (files && files.length) this.uploadFile(files[0]);
      event.target.value = '';
    },

    async uploadFile(file) {
      this.error = '';
      if (file.size > 10 * 1024 * 1024) {
        this.showError('Fajl je prevelik (maksimum 10 MB).');
        return;
      }
      const form = new FormData();
      form.append('file', file);
      this.loading = true;
      try {
        const data = await this.api('/api/images', { method: 'POST', body: form });
        this.image = data;
        this.result = null;
        this.svgMarkup = '';
        this.job = null;
        this.tab = 'vector';
        this.resetView();
        await this.renderNow();
      } catch (error) {
        this.showError(error.message);
      } finally {
        this.loading = false;
      }
    },

    async loadTestImage(name) {
      try {
        const response = await fetch('/api/test-images/' + encodeURIComponent(name));
        if (!response.ok) throw new Error('Test slika nije dostupna');
        const blob = await response.blob();
        const file = new File([blob], name, { type: 'image/png' });
        await this.uploadFile(file);
      } catch (error) {
        this.showError(error.message);
      }
    },

    clearImage() {
      this.image = null;
      this.result = null;
      this.svgMarkup = '';
      this.job = null;
      this.error = '';
    },

    setMode(key) {
      this.params.mode = key;
    },

    // --- preseti ---
    applyPreset() {
      const preset = this.presets.find((item) => item.key === this.presetKey);
      if (!preset) return;
      this.params = Object.assign({}, this.params, preset.params);
      this.presetDescription = preset.description || '';
      this.showNotice('Primenjen preset: ' + preset.name);
    },

    async savePreset() {
      const key = window.prompt('Kljuc preseta (mala slova, cifre, - i _):', 'moj-preset');
      if (!key) return;
      const name = window.prompt('Naziv preseta:', key) || key;
      try {
        await this.api('/api/presets', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            key: key, name: name, description: 'Timski preset', params: this.params
          })
        });
        await this.loadPresets();
        this.presetKey = key;
        this.showNotice('Preset "' + key + '" je sacuvan u presets/ i spreman za git.');
      } catch (error) {
        this.showError(error.message);
      }
    },

    // --- render ---
    scheduleRender() {
      if (!this.image || this.params.mode === 'wireframe') return;
      window.clearTimeout(this.renderTimer);
      this.renderTimer = window.setTimeout(() => this.renderNow(), 450);
    },

    async renderNow() {
      if (!this.image) return;
      if (this.params.mode === 'wireframe') {
        this.runWireframe();
        return;
      }
      this.loading = true;
      try {
        const meta = await this.api('/api/render', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image_id: this.image.image_id, params: this.params })
        });
        this.result = meta;
        const svg = await fetch(meta.svg_url).then((response) => response.text());
        this.svgMarkup = svg;
        this.$nextTick(() => this.applyLayers());
      } catch (error) {
        this.showError(error.message);
      } finally {
        this.loading = false;
      }
    },

    runWireframe() {
      if (!this.image) return;
      if (this.job && this.job.status === 'running') return;
      this.params.mode = 'wireframe';
      this.loading = true;
      this.job = { status: 'queued', progress: 0, message: 'Saljem posao...' };
      this.api('/api/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_id: this.image.image_id, params: this.params })
      }).then((job) => {
        this.job = job;
        this.pollJob();
      }).catch((error) => {
        this.loading = false;
        this.job = null;
        this.showError(error.message);
      });
    },

    pollJob() {
      window.clearTimeout(this.jobTimer);
      if (!this.job || !this.job.job_id) return;
      this.jobTimer = window.setTimeout(async () => {
        try {
          const job = await this.api('/api/jobs/' + this.job.job_id);
          this.job = job;
          if (job.status === 'done') {
            this.loading = false;
            this.result = job.result;
            const svg = await fetch(job.result.svg_url).then((response) => response.text());
            this.svgMarkup = svg;
            this.$nextTick(() => this.applyLayers());
            this.showNotice('3D wireframe je gotov.');
          } else if (job.status === 'error') {
            this.loading = false;
            this.showError(job.message);
          } else {
            this.pollJob();
          }
        } catch (error) {
          this.loading = false;
          this.showError(error.message);
        }
      }, 700);
    },

    // --- slojevi i zum ---
    applyLayers() {
      const host = document.getElementById('svg-host');
      if (!host) return;
      host.querySelectorAll('path').forEach((element) => {
        const fill = (element.getAttribute('fill') || '').toUpperCase();
        const stroke = (element.getAttribute('stroke') || '').toUpperCase();
        const isCut = fill === '#FF0000' || stroke === '#FF0000';
        if (isCut) {
          element.style.display = this.layers.cut ? '' : 'none';
        } else {
          element.style.display = this.layers.engrave ? '' : 'none';
        }
      });
    },

    paperStyle() {
      return 'transform: translate(' + this.panX + 'px, ' + this.panY + 'px) scale(' + this.zoom + ');';
    },

    resetView() {
      this.zoom = 1;
      this.panX = 0;
      this.panY = 0;
    },

    onWheel(event) {
      const factor = event.deltaY > 0 ? 0.9 : 1.1;
      this.zoom = Math.min(12, Math.max(0.15, this.zoom * factor));
    },

    onPanStart(event) {
      if (event.button !== 0) return;
      this.panning = true;
      this.panStart = { x: event.clientX - this.panX, y: event.clientY - this.panY };
    },

    onPanMove(event) {
      if (!this.panning || !this.panStart) return;
      this.panX = event.clientX - this.panStart.x;
      this.panY = event.clientY - this.panStart.y;
    },

    onPanEnd() {
      this.panning = false;
      this.panStart = null;
    },

    // --- izvoz ---
    download(kind) {
      if (!this.image || !this.result) return;
      const base = '/api/images/' + this.image.image_id;
      let path = base + '/download/svg';
      if (kind === 'dxf') path = base + '/download/dxf';
      if (kind === 'png') path = base + '/preview.png';
      window.location.href = path;
    }
  };
}
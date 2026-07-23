/* Slide navigation + fragment stepping + responsive scaling.
 * No dependencies. Edit index.html to add/remove <section class="slide">. */

(() => {
  const SLIDE_W = 1280;
  const SLIDE_H = 720;

  const stage  = document.getElementById('stage');
  const hud    = document.getElementById('hud');
  const help   = document.getElementById('help');
  const slides = Array.from(document.querySelectorAll('.slide'));

  let current = 0;
  let fragStep = 0;   // how many fragments are visible on the current slide

  // ── atlas iframe stage tracking ────────────────────────────────────────
  const ATLAS_STAGES = 6;
  let atlasStage = 1;

  function getAtlasIframe() {
    return slides[current]?.querySelector('iframe[src*="atlas.html"]') ?? null;
  }

  function postAtlasStage(n) {
    const iframe = getAtlasIframe();
    iframe?.contentWindow?.postMessage({ type: 'atlas-stage', stage: n }, '*');
  }

  /* ---------- scaling: keep 1280x720 aspect, fit to viewport ----------
     We don't transform the stage itself — instead we publish a scale
     factor via the `--slide-scale` CSS variable and let each .slide
     apply `translate(-50%,-50%) scale(var(--slide-scale))`. That way
     the slide stays geometrically centered no matter the viewport. */
  function rescale() {
    const sx = window.innerWidth  / SLIDE_W;
    const sy = window.innerHeight / SLIDE_H;
    const s  = Math.min(sx, sy);
    document.documentElement.style.setProperty('--slide-scale', s);
  }
  window.addEventListener('resize', rescale);

  /* ---------- show a slide ---------- */
  function fragmentsOn(slide) {
    return Array.from(slide.querySelectorAll('.fragment'));
  }

  function applyFragments(slide) {
    fragmentsOn(slide).forEach((el, i) => {
      el.classList.toggle('visible', i < fragStep);
    });
  }

  function progressFor(i) {
    /* Count non-title slides for the progress bar, like Beamer. */
    const totalCounted = slides.filter(s => !s.classList.contains('no-count')).length;
    const counted = slides.slice(0, i + 1).filter(s => !s.classList.contains('no-count')).length;
    return { counted, totalCounted };
  }

  function show(i, opts = {}) {
    const movingForward = i >= current;
    i = Math.max(0, Math.min(slides.length - 1, i));
    current = i;
    fragStep = opts.fragStep ?? 0;

    slides.forEach((s, j) => s.classList.toggle('active', j === i));
    const s = slides[i];
    applyFragments(s);

    /* update the per-slide progress bar (if the slide has one) */
    const bar = s.querySelector('.progress-fill');
    const num = s.querySelector('.progress-num');
    if (bar) {
      const { counted, totalCounted } = progressFor(i);
      bar.style.setProperty('--progress', `${(counted / totalCounted) * 100}%`);
      bar.style.width = `${(counted / totalCounted) * 100}%`;
      if (num) num.textContent = `${counted} / ${totalCounted}`;
    }

    hud.textContent = `${i + 1} / ${slides.length}`;
    history.replaceState(null, '', `#${i + 1}`);

    if (window.MathJax?.typesetPromise) {
      MathJax.typesetPromise([s]).catch(() => {});
    }

    /* reset atlas stage when entering/leaving the atlas slide */
    const atlasIframe = s.querySelector('iframe[src*="atlas.html"]');
    if (atlasIframe) {
      atlasStage = movingForward ? 1 : ATLAS_STAGES;
      // give the iframe a moment to be ready (especially on first visit)
      setTimeout(() => {
        atlasIframe.contentWindow?.postMessage({ type: 'atlas-stage', stage: atlasStage }, '*');
      }, 80);
    }
  }

  /* ---------- navigation ---------- */
  function next() {
    const total = fragmentsOn(slides[current]).length;
    if (fragStep < total) { fragStep++; applyFragments(slides[current]); return; }
    if (current < slides.length - 1) show(current + 1, { fragStep: 0 });
  }
  function prev() {
    if (fragStep > 0) { fragStep--; applyFragments(slides[current]); return; }
    if (current > 0) {
      /* land on the new slide with all its fragments revealed */
      const target = current - 1;
      const fragCount = fragmentsOn(slides[target]).length;
      show(target, { fragStep: fragCount });
    }
  }

  document.addEventListener('keydown', (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    switch (e.key) {
      case 'ArrowRight':
      case 'ArrowDown':
      case 'Enter':
      case 'PageDown':
      case ' ': {
        e.preventDefault();
        const iframe = getAtlasIframe();
        if (iframe && atlasStage < ATLAS_STAGES) {
          atlasStage++;
          postAtlasStage(atlasStage);
        } else {
          next();
        }
        break;
      }
      case 'ArrowLeft':
      case 'ArrowUp':
      case 'PageUp': {
        e.preventDefault();
        const iframe = getAtlasIframe();
        if (iframe && atlasStage > 1) {
          atlasStage--;
          postAtlasStage(atlasStage);
        } else {
          prev();
        }
        break;
      }
      case 'Home':
        e.preventDefault(); show(0); break;
      case 'End':
        e.preventDefault(); show(slides.length - 1); break;
      case '?':
        help.classList.toggle('show'); break;
      case 'Escape':
        help.classList.remove('show'); break;
      case 'f':
        if (!document.fullscreenElement) document.documentElement.requestFullscreen();
        else document.exitFullscreen();
        break;
    }
  });

  /* click to advance (skip clicks on links/buttons/videos) */
  stage.addEventListener('click', (e) => {
    if (e.target.closest('a, button, input, textarea, video')) return;
    next();
  });

  /* show native video controls only while hovered */
  document.querySelectorAll('.slide video').forEach((v) => {
    v.addEventListener('mouseenter', () => v.setAttribute('controls', ''));
    v.addEventListener('mouseleave', () => v.removeAttribute('controls'));
  });

  /* ---------- boot ---------- */
  const initial = Math.max(1, parseInt(location.hash.slice(1), 10) || 1) - 1;
  rescale();
  show(initial);
})();

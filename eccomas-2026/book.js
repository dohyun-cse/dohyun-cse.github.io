/* Book template: view-mode toggle + chapter navigation + ToC.
 *
 *   - body[data-view="long"]  : continuous scroll
 *   - body[data-view="short"] : one chapter at a time
 *
 * State persists via localStorage; the URL hash tracks the current
 * chapter so deep links work in either mode.
 */

(async () => {
  const LS_VIEW = 'book.view';

  const body        = document.body;
  const tocList     = document.getElementById('toc-list');
  const drawerList  = document.getElementById('drawer-list');
  const drawer      = document.getElementById('toc-drawer');
  const backdrop    = document.getElementById('toc-backdrop');
  const btnHamb     = document.getElementById('btn-hamburger');
  const btnClose    = document.getElementById('btn-close-drawer');
  const btnPrev     = document.getElementById('btn-prev');
  const btnNext     = document.getElementById('btn-next');
  const btnView     = document.getElementById('btn-view');
  const headerTitle = document.getElementById('header-title');
  const bookTitle   = document.querySelector('.cover .title')?.textContent?.trim() || 'Document';

  /* ── markdown rendering (marked) ──────────────────────────────────────
   * Configure marked with:
   *   - callout extension for theorem-likes and blocks (> [!Kind^label] Title)
   *   - math passthrough extensions ($...$, $$...$$, \(...\), \[...\],
   *     and \begin{...}...\end{...}) so MathJax sees the originals
   *   - heading shift: `# Title` → <h2>, `## Sub` → <h3>, etc.,
   *     since each chapter is wrapped in a <section class="chapter">.
   */
  const THEOREM_KINDS = new Set([
    'theorem', 'lemma', 'definition', 'proposition',
    'corollary', 'remark', 'proof',
  ]);
  const BLOCK_VARIANTS = { block: '', alert: 'alert', example: 'example' };

  function escapeAttr(s) {
    return String(s).replace(/&/g, '&amp;').replace(/"/g, '&quot;')
      .replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  function escapeMath(s) {
    /* Math text gets embedded in HTML output; & and < must be entity-encoded
     * so the browser doesn't interpret them as markup. MathJax decodes
     * entities when it reads textContent, so the rendered math is correct. */
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function configureMarked() {
    if (typeof marked === 'undefined' || marked.__bookConfigured) return;
    marked.__bookConfigured = true;

    marked.use({
      gfm: true,
      breaks: false,
      extensions: [
        {
          /* > [!Theorem^thm-foo] Optional title
             > body line 1
             > body line 2                                         */
          name: 'callout',
          level: 'block',
          start(src) { return src.match(/^>\s*\[!/)?.index; },
          tokenizer(src) {
            const m = src.match(
              /^>\s*\[!([A-Za-z]+)(?:\^([^\]\s]+))?\]([^\n]*)((?:\n>[^\n]*)*)/
            );
            if (!m) return;
            const kindRaw = m[1].toLowerCase();
            if (!THEOREM_KINDS.has(kindRaw) && !(kindRaw in BLOCK_VARIANTS)) return;
            const label = m[2] || '';
            const title = m[3].trim();
            const bodyRaw = m[4]
              .split('\n')
              .map((l) => l.replace(/^>\s?/, ''))
              .join('\n')
              .replace(/^\n+|\n+$/g, '');
            return {
              type: 'callout',
              raw: m[0],
              kindRaw, label, title,
              tokens: this.lexer.blockTokens(bodyRaw, []),
            };
          },
          renderer(t) {
            const inner = this.parser.parse(t.tokens);
            if (t.kindRaw in BLOCK_VARIANTS) {
              const variant = BLOCK_VARIANTS[t.kindRaw];
              const cls = variant ? `block ${variant}` : 'block';
              const idAttr = t.label ? ` id="${escapeAttr(t.label)}"` : '';
              const titleHtml = t.title
                ? `<div class="block-title">${escapeAttr(t.title)}</div>`
                : '';
              return `<div class="${cls}"${idAttr}>${titleHtml}` +
                     `<div class="block-body">${inner}</div></div>\n`;
            }
            const attrs = [`class="${t.kindRaw}"`];
            if (t.label) attrs.push(`id="${escapeAttr(t.label)}"`);
            if (t.title) attrs.push(`data-name="${escapeAttr(t.title)}"`);
            return `<div ${attrs.join(' ')}>${inner}</div>\n`;
          },
        },
        {
          name: 'mathDisplay', level: 'block',
          start(src) { return src.match(/\$\$/)?.index; },
          tokenizer(src) {
            const m = src.match(/^\$\$([\s\S]+?)\$\$/);
            if (m) return { type: 'mathDisplay', raw: m[0], text: m[1] };
          },
          renderer(t) { return `$$${escapeMath(t.text)}$$\n`; },
        },
        {
          name: 'mathEnv', level: 'block',
          start(src) { return src.match(/\\begin\{/)?.index; },
          tokenizer(src) {
            const m = src.match(/^\\begin\{(\w+\*?)\}([\s\S]+?)\\end\{\1\}/);
            if (m) return { type: 'mathEnv', raw: m[0], env: m[1], text: m[2] };
          },
          renderer(t) {
            return `\\begin{${t.env}}${escapeMath(t.text)}\\end{${t.env}}\n`;
          },
        },
        {
          name: 'mathBracket', level: 'block',
          start(src) { return src.match(/\\\[/)?.index; },
          tokenizer(src) {
            const m = src.match(/^\\\[([\s\S]+?)\\\]/);
            if (m) return { type: 'mathBracket', raw: m[0], text: m[1] };
          },
          renderer(t) { return `\\[${escapeMath(t.text)}\\]\n`; },
        },
        {
          name: 'mathParen', level: 'inline',
          start(src) { return src.match(/\\\(/)?.index; },
          tokenizer(src) {
            const m = src.match(/^\\\(([\s\S]+?)\\\)/);
            if (m) return { type: 'mathParen', raw: m[0], text: m[1] };
          },
          renderer(t) { return `\\(${escapeMath(t.text)}\\)`; },
        },
        {
          name: 'mathInline', level: 'inline',
          start(src) {
            const m = src.match(/(?<!\$)\$(?!\$)/);
            return m?.index;
          },
          tokenizer(src) {
            const m = src.match(/^\$(?!\$)((?:\\.|[^\n$])+?)\$(?!\$)/);
            if (m) return { type: 'mathInline', raw: m[0], text: m[1] };
          },
          renderer(t) { return `$${escapeMath(t.text)}$`; },
        },
      ],
      renderer: {
        heading(token) {
          const text = this.parser.parseInline(token.tokens);
          const depth = Math.min(6, token.depth + 1);
          return `<h${depth}>${text}</h${depth}>\n`;
        },
      },
    });
  }

  /* ── load external chapter files ──────────────────────────────────────
   * <div data-chapter-src="chapters/foo.md"></div>  →  Markdown, wrapped
   *     in <section class="chapter" id="{filename}">  (filename minus
   *     any leading NN- prefix).
   * <div data-chapter-src="chapters/foo.html"></div> →  HTML, injected as-is
   *     (must already contain its own <section class="chapter">).
   * Requires HTTP (not file://). Inline chapters still work — placeholders
   * and inline <section class="chapter"> may coexist. */
  function chapterIdFromSrc(src) {
    const stem = src.match(/([^/]+)\.[^.]+$/)?.[1] || 'unknown';
    return stem.replace(/^\d+[-_]/, '');
  }

  /* ── wiki-style references (preprocessed before marked) ──────────────
   * Forms (file portion is optional — labels are globally unique, so the
   * file is just a source-readability hint):
   *
   *   [[label]]                       → <a class="ref" data-ref="label">
   *   [[file^label]]                  → same as above (file ignored)
   *   [[@bib-key]]                    → <a class="cite" data-cite="bib-key">
   *   [[file^@bib-key1,bib-key2]]     → multi-cite, file ignored
   *
   * Run as a string substitution BEFORE marked, so it also reaches
   * content inside block HTML (e.g., <figcaption>) which marked
   * otherwise passes through unprocessed.  We first protect math and
   * code spans so [[…]] inside those is left alone. */
  function preprocessWikiRefs(md) {
    const protected_ = [];
    const stash = (m) => `\x00PH${protected_.push(m) - 1}\x00`;
    md = md
      .replace(/```[\s\S]*?```/g, stash)
      .replace(/~~~[\s\S]*?~~~/g, stash)
      .replace(/\\begin\{(\w+\*?)\}[\s\S]+?\\end\{\1\}/g, stash)
      .replace(/\$\$[\s\S]+?\$\$/g, stash)
      .replace(/\\\[[\s\S]+?\\\]/g, stash)
      .replace(/\\\([\s\S]+?\\\)/g, stash)
      .replace(/\$(?!\$)(?:\\.|[^\n$])+?\$/g, stash)
      .replace(/`[^`\n]+`/g, stash);
    md = md.replace(/\[\[([^\]\n]+)\]\]/g, (full, payload) => {
      payload = payload.trim();
      const caret = payload.lastIndexOf('^');
      if (caret >= 0) payload = payload.slice(caret + 1).trim();
      if (!payload) return full;
      if (payload.startsWith('@')) {
        const key = payload.slice(1).trim();
        if (!key) return full;
        return `<a class="cite" data-cite="${escapeAttr(key)}"></a>`;
      }
      return `<a class="ref" data-ref="${escapeAttr(payload)}"></a>`;
    });
    /* Iterative restore so nested placeholders (e.g., \begin{…} stashed
       inside an already-stashed $$…$$) all get unwound. */
    let prev = null;
    while (prev !== md) {
      prev = md;
      md = md.replace(/\x00PH(\d+)\x00/g, (_, i) => protected_[+i]);
    }
    return md;
  }

  async function loadChapters() {
    const placeholders = document.querySelectorAll('[data-chapter-src]');
    if (!placeholders.length) return;
    configureMarked();
    await Promise.all(Array.from(placeholders).map(async (el) => {
      const src = el.dataset.chapterSrc;
      try {
        const res = await fetch(src);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const text = await res.text();
        let html;
        if (/\.md$/i.test(src)) {
          if (typeof marked === 'undefined') {
            throw new Error('marked library is not loaded');
          }
          const id = chapterIdFromSrc(src);
          const body = marked.parse(preprocessWikiRefs(text));
          html = `<section class="chapter" id="${escapeAttr(id)}">${body}</section>`;
        } else {
          html = text;
        }
        const tpl = document.createElement('template');
        tpl.innerHTML = html.trim();
        el.replaceWith(tpl.content);
      } catch (err) {
        const errSection = document.createElement('section');
        errSection.className = 'chapter';
        errSection.id = 'load-error-' + chapterIdFromSrc(src);
        errSection.innerHTML =
          `<h2>Could not load <code>${escapeAttr(src)}</code></h2>` +
          `<p><span class="red">${escapeAttr(err.message)}.</span> ` +
          `Serve over HTTP (e.g. <code>python3 -m http.server</code>); ` +
          `<code>file://</code> URLs can't fetch.</p>`;
        el.replaceWith(errSection);
      }
    }));
  }
  await loadChapters();

  /* Pages, in display order. A "page" is either the .front-matter wrapper
     (cover+abstract+ToC) or a .chapter. In short-form view we show exactly
     one at a time; in long-form view we show all of them stacked. */
  const frontMatter = document.querySelector('.front-matter');
  const chapters    = Array.from(document.querySelectorAll('main > .chapter'));
  const pages       = frontMatter ? [frontMatter, ...chapters] : chapters.slice();

  /* ── ToC generation ────────────────────────────────────────────────── */
  function buildToc() {
    if (!tocList && !drawerList) return;
    const frags = chapters.map((ch, i) => {
      const h2 = ch.querySelector(':scope > h2');
      const id = ch.id || (ch.id = `ch-${i + 1}`);
      const num = i + 1;
      const title = (h2?.textContent || `Chapter ${num}`).replace(/^\s*\d+\.?\s*/, '');
      const subs = Array.from(ch.querySelectorAll(':scope > h3')).map((h3, j) => {
        if (!h3.id) h3.id = `${id}-s${j + 1}`;
        return { id: h3.id, title: h3.textContent.trim(), num: `${num}.${j + 1}` };
      });
      // also write the canonical "N. Title" prefix back into the h2 if not present
      if (h2 && !/^\d+\./.test(h2.textContent)) {
        h2.textContent = `${num}. ${title}`;
      }
      return { id, num, title, subs };
    });

    function render(target, isDrawer) {
      target.innerHTML = '';
      frags.forEach((ch) => {
        const li = document.createElement('li');
        const a = document.createElement('a');
        a.href = `#${ch.id}`;
        a.dataset.target = ch.id;
        if (isDrawer) {
          a.innerHTML = `<span class="toc-num">${ch.num}.</span>${ch.title}`;
        } else {
          a.innerHTML =
            `<span class="toc-num">${ch.num}.</span>` +
            `<span class="toc-ttl">${ch.title}</span>` +
            `<span class="toc-fill"></span>`;
        }
        li.appendChild(a);
        target.appendChild(li);

        ch.subs.forEach((s) => {
          const sli = document.createElement('li');
          sli.className = 'sub';
          const sa = document.createElement('a');
          sa.href = `#${s.id}`;
          sa.dataset.target = ch.id;            // jump to the chapter that owns it
          sa.dataset.subTarget = s.id;
          if (isDrawer) {
            sa.innerHTML = `<span class="toc-num">${s.num}</span>${s.title}`;
          } else {
            sa.innerHTML =
              `<span class="toc-num">${s.num}</span>` +
              `<span class="toc-ttl">${s.title}</span>` +
              `<span class="toc-fill"></span>`;
          }
          sli.appendChild(sa);
          target.appendChild(sli);
        });
      });
    }
    if (tocList)    render(tocList, false);
    if (drawerList) render(drawerList, true);

    return frags;
  }

  const tocFrags = buildToc();

  /* ── numbered environments + cross-refs + citations ────────────────── */
  /*
   * Conventions:
   *   <div class="theorem" id="thm-foo">     →  Theorem N.k
   *   <div class="definition" id="def-foo" data-name="Bregman divergence">
   *                                         →  Definition N.k (Bregman divergence)
   *   <figure id="fig-foo">                  →  Figure N.k (label injected into <figcaption>)
   *   <a class="ref" data-ref="thm-foo"></a> →  text becomes "Theorem N.k", href "#thm-foo"
   *   <ol class="bib"><li id="bib-key">…</li></ol>
   *                                          →  Numbered [1], [2], …
   *   <a class="cite" data-cite="bib-a,bib-b"></a>  →  "[1, 3]"
   *
   * Equations use MathJax's native machinery (\label{eq:foo} + \eqref{eq:foo})
   * because mathjax-config.js sets `tags: 'ams'`.
   */
  const ENV_NAMES = {
    theorem:     'Theorem',
    definition:  'Definition',
    proposition: 'Proposition',
    lemma:       'Lemma',
    corollary:   'Corollary',
    remark:      'Remark',
  };
  const ENV_SELECTOR = Object.keys(ENV_NAMES).map((c) => '.' + c).join(',');

  function prependInline(el, html) {
    /* Inject `html` at the start of `el`. If `el`'s first child is a
       <p> (the common case for markdown-rendered content), insert into
       that <p> so the inline label sits next to the body text instead
       of on its own line. */
    const first = el.firstElementChild;
    if (first && first.tagName === 'P') {
      first.insertAdjacentHTML('afterbegin', html);
    } else {
      el.insertAdjacentHTML('afterbegin', html);
    }
  }

  function appendInline(el, html) {
    /* Mirror of prependInline for trailing content (e.g., proof QED). */
    const last = el.lastElementChild;
    if (last && last.tagName === 'P') {
      last.insertAdjacentHTML('beforeend', html);
    } else {
      el.insertAdjacentHTML('beforeend', html);
    }
  }

  function buildNumbering() {
    const labels = {};

    chapters.forEach((ch, i) => {
      const chNum = i + 1;
      let envCounter = 0;
      let figCounter = 0;

      // walk theorem-likes and figures in document order
      const nodes = ch.querySelectorAll(`${ENV_SELECTOR}, figure`);
      nodes.forEach((el) => {
        if (el.dataset.numbered) return;

        if (el.tagName === 'FIGURE') {
          // only number figures that have an id (i.e. are referenced) or
          // a figcaption — skip decorative ones
          if (!el.querySelector('figcaption')) return;
          figCounter++;
          const num = `${chNum}.${figCounter}`;
          const label = `Figure ${num}`;
          const cap = el.querySelector('figcaption');
          cap.insertAdjacentHTML('afterbegin',
            `<span class="fig-label">${label}.</span> `);
          el.dataset.numbered = '1';
          if (el.id) labels[el.id] = { label, kind: 'figure', name: null, href: `#${el.id}` };
        } else {
          envCounter++;
          const num = `${chNum}.${envCounter}`;
          const klass = Object.keys(ENV_NAMES).find((c) => el.classList.contains(c));
          const name = ENV_NAMES[klass];
          const label = `${name} ${num}`;
          const named = el.dataset.name ? ` <em class="env-named">(${el.dataset.name})</em>` : '';
          prependInline(el, `<span class="env-label">${label}${named}.</span> `);
          el.dataset.numbered = '1';
          if (el.id) labels[el.id] = {
            label,
            kind: klass,
            name: el.dataset.name || null,
            href: `#${el.id}`,
          };
        }
      });
    });

    /* Proof: prepend "Proof." and append "□" inline, working for both
       direct text bodies and markdown-rendered <p>-wrapped bodies. */
    document.querySelectorAll('.proof').forEach((p) => {
      if (p.dataset.labeled) return;
      prependInline(p, '<span class="proof-label">Proof.</span> ');
      appendInline(p, ' <span class="proof-qed">□</span>');
      p.dataset.labeled = '1';
    });

    return labels;
  }

  function buildBibliography() {
    const bibMap = {};
    document.querySelectorAll('ol.bib > li').forEach((li, i) => {
      const n = i + 1;
      li.dataset.bibNum = n;
      if (li.id) bibMap[li.id] = n;
    });
    return bibMap;
  }

  function annotateNavigation(el, targetId) {
    /* If the target lives inside a chapter, set data-target/data-subTarget
       so the existing click handler can route via the short-mode page
       switcher. */
    const target = document.getElementById(targetId);
    if (!target) return;
    const ch = target.closest('.chapter');
    if (ch) {
      el.dataset.target = ch.id;
      el.dataset.subTarget = targetId;
    } else if (target.closest('.front-matter')) {
      el.dataset.target = 'front-matter';
      el.dataset.subTarget = targetId;
    }
  }

  function resolveRefs(labels, bibMap) {
    document.querySelectorAll('.ref').forEach((el) => {
      const id = el.dataset.ref;
      const entry = id && labels[id];
      if (!entry) {
        el.textContent = `[?${id ? ':' + id : ''}]`;
        el.classList.add('ref-broken');
        return;
      }
      el.textContent = entry.label;
      if (el.tagName === 'A') el.setAttribute('href', entry.href);
      annotateNavigation(el, id);
    });

    document.querySelectorAll('.cite').forEach((el) => {
      const keys = (el.dataset.cite || '')
        .split(',').map((s) => s.trim()).filter(Boolean);
      const resolved = keys.map((k) => ({ k, n: bibMap[k] }));
      const missing  = resolved.filter((r) => r.n == null);
      if (resolved.length === 0 || missing.length === resolved.length) {
        el.textContent = '[?]';
        el.classList.add('ref-broken');
        return;
      }
      const nums = resolved.map((r) => r.n != null ? r.n : '?');
      el.textContent = `[${nums.join(', ')}]`;
      const firstKey = (resolved.find((r) => r.n != null) || resolved[0]).k;
      if (el.tagName === 'A') el.setAttribute('href', `#${firstKey}`);
      annotateNavigation(el, firstKey);
    });
  }

  function buildListOfEnvs(labels) {
    /* Populate any <ol class="lone">…</ol> with one entry per numbered
       environment, in document order:
           Theorem 3.1   (three-point identity)   [thm-three-point]
       Each entry is a link that navigates in either view mode. */
    const lists = document.querySelectorAll('ol.lone');
    if (!lists.length) return;
    /* `labels` is built in chapter / document order, so Object.entries
       preserves that. */
    const items = Object.entries(labels);
    lists.forEach((list) => {
      const kindFilter = (list.dataset.kind || '').split(',').map(s => s.trim()).filter(Boolean);
      list.innerHTML = '';
      items.forEach(([id, entry]) => {
        if (kindFilter.length && !kindFilter.includes(entry.kind)) return;
        const li = document.createElement('li');
        const a = document.createElement('a');
        a.href = entry.href;
        a.dataset.ref = id;          // not used, but handy for inspection
        annotateNavigation(a, id);
        const nameText = entry.name ? `(${entry.name})` : '';
        a.innerHTML =
          `<span class="lone-label">${entry.label}</span>` +
          `<span class="lone-name">${nameText}</span>` +
          `<span class="lone-key">[${id}]</span>`;
        li.appendChild(a);
        list.appendChild(li);
      });
      if (!list.children.length) {
        const li = document.createElement('li');
        li.className = 'lone-empty';
        li.textContent = 'No numbered environments yet.';
        list.appendChild(li);
      }
    });
  }

  const envLabels = buildNumbering();
  const bibNumbers = buildBibliography();
  resolveRefs(envLabels, bibNumbers);
  buildListOfEnvs(envLabels);

  /* ── per-chapter prev/next footers ─────────────────────────────────── */
  function buildChapterFooters() {
    chapters.forEach((ch, i) => {
      if (ch.querySelector(':scope > .chapter-nav')) return;
      const nav = document.createElement('nav');
      nav.className = 'chapter-nav';

      const prev = chapters[i - 1];
      const next = chapters[i + 1];

      if (prev) {
        const a = document.createElement('a');
        a.href = `#${prev.id}`;
        a.dataset.target = prev.id;
        a.className = 'prev';
        a.innerHTML = `<span class="label">← Previous</span><span class="ttl">${(prev.querySelector('h2')?.textContent || '').trim()}</span>`;
        nav.appendChild(a);
      } else {
        const s = document.createElement('span');
        s.className = 'spacer';
        nav.appendChild(s);
      }

      if (next) {
        const a = document.createElement('a');
        a.href = `#${next.id}`;
        a.dataset.target = next.id;
        a.className = 'next';
        a.innerHTML = `<span class="label">Next →</span><span class="ttl">${(next.querySelector('h2')?.textContent || '').trim()}</span>`;
        nav.appendChild(a);
      } else {
        const s = document.createElement('span');
        s.className = 'spacer';
        nav.appendChild(s);
      }

      ch.appendChild(nav);
    });
  }
  buildChapterFooters();

  /* ── current page state ────────────────────────────────────────────── */
  /* index 0 = front matter (if it exists), then chapters */
  let currentIdx = 0;

  function pageIdOf(p) {
    if (!p) return null;
    return p.classList.contains('front-matter') ? 'front-matter' : p.id;
  }
  function pageByHash(hash) {
    const h = (hash || '').replace(/^#/, '');
    if (!h) return 0;
    if (h === 'front-matter') return frontMatter ? 0 : 0;
    // a sub-id might be inside a chapter (e.g. "ch-2-s1") — find its parent
    const target = document.getElementById(h);
    if (!target) return 0;
    const ch = target.closest('.chapter');
    if (ch) return pages.indexOf(ch);
    const fm = target.closest('.front-matter');
    if (fm) return 0;
    return 0;
  }

  function setHeaderTitle() {
    const view = body.dataset.view;
    if (view === 'long') {
      headerTitle.textContent = bookTitle;
      return;
    }
    const p = pages[currentIdx];
    if (!p) return;
    if (p.classList.contains('front-matter')) {
      headerTitle.textContent = bookTitle;
    } else {
      const h2 = p.querySelector(':scope > h2');
      headerTitle.textContent = h2 ? h2.textContent.trim() : bookTitle;
    }
  }

  function setActiveTocLink() {
    const p = pages[currentIdx];
    const id = pageIdOf(p);
    document.querySelectorAll('#toc-drawer a, #toc-list a').forEach((a) => {
      a.classList.toggle('current', a.dataset.target === id);
    });
  }

  function updateNavButtons() {
    btnPrev.disabled = currentIdx <= 0;
    btnNext.disabled = currentIdx >= pages.length - 1;
  }

  function applyCurrent({ scrollToTop = true, subTarget = null } = {}) {
    /* mark .current on the active page; CSS hides the others in short mode */
    pages.forEach((p, i) => p.classList.toggle('current', i === currentIdx));
    setHeaderTitle();
    setActiveTocLink();
    updateNavButtons();

    if (body.dataset.view === 'short') {
      if (scrollToTop) window.scrollTo({ top: 0, behavior: 'instant' in window ? 'instant' : 'auto' });
      if (subTarget) {
        const el = document.getElementById(subTarget);
        if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    } else if (scrollToTop && subTarget) {
      const el = document.getElementById(subTarget);
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    if (window.MathJax?.typesetPromise) {
      MathJax.typesetPromise([pages[currentIdx]]).catch(() => {});
    }
  }

  function goTo(idx, opts = {}) {
    idx = Math.max(0, Math.min(pages.length - 1, idx));
    if (idx === currentIdx && !opts.force) {
      applyCurrent(opts);
      return;
    }
    currentIdx = idx;
    const id = pageIdOf(pages[idx]) || '';
    if (id) history.replaceState(null, '', `#${opts.subTarget || id}`);
    applyCurrent(opts);
  }

  /* ── view-mode toggle ──────────────────────────────────────────────── */
  function setView(view) {
    body.dataset.view = view;
    try { localStorage.setItem(LS_VIEW, view); } catch (e) {}
    btnView.textContent = view === 'long' ? '⇆' : '⇆';
    btnView.title = view === 'long'
      ? 'Switch to short view (one chapter at a time)'
      : 'Switch to long view (continuous)';
    setHeaderTitle();
    updateNavButtons();

    if (view === 'long') {
      // long mode shows everything; scroll to the current chapter so the
      // user doesn't lose their place
      const target = pages[currentIdx];
      if (target) {
        setTimeout(() => target.scrollIntoView({ behavior: 'auto', block: 'start' }), 0);
      }
    } else {
      applyCurrent({ scrollToTop: true });
    }
  }

  /* ── ToC drawer ────────────────────────────────────────────────────── */
  function openDrawer() {
    drawer.classList.add('open');
    backdrop.classList.add('open');
  }
  function closeDrawer() {
    drawer.classList.remove('open');
    backdrop.classList.remove('open');
  }

  /* ── intersection observer: in long mode, update header title + ToC
        highlight as the user scrolls past each chapter ─────────────── */
  function installScrollSpy() {
    if (!('IntersectionObserver' in window)) return;
    const io = new IntersectionObserver((entries) => {
      if (body.dataset.view !== 'long') return;
      // pick the entry whose top is closest to the header
      const visible = entries
        .filter((e) => e.isIntersecting)
        .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
      if (!visible.length) return;
      const top = visible[0].target;
      const idx = pages.indexOf(top);
      if (idx >= 0 && idx !== currentIdx) {
        currentIdx = idx;
        setHeaderTitle();
        setActiveTocLink();
        const id = pageIdOf(top);
        if (id) history.replaceState(null, '', `#${id}`);
      }
    }, { rootMargin: '-72px 0px -60% 0px', threshold: 0 });
    pages.forEach((p) => io.observe(p));
  }

  /* ── wiring ───────────────────────────────────────────────────────── */
  btnHamb.addEventListener('click', openDrawer);
  btnClose.addEventListener('click', closeDrawer);
  backdrop.addEventListener('click', closeDrawer);

  btnPrev.addEventListener('click', () => goTo(currentIdx - 1));
  btnNext.addEventListener('click', () => goTo(currentIdx + 1));

  btnView.addEventListener('click', () => {
    setView(body.dataset.view === 'long' ? 'short' : 'long');
  });

  document.addEventListener('click', (e) => {
    const a = e.target.closest('a[data-target]');
    if (!a) return;
    e.preventDefault();
    const targetId = a.dataset.target;
    const subTarget = a.dataset.subTarget || null;
    let idx;
    if (targetId === 'front-matter' && frontMatter) {
      idx = 0;
    } else {
      const el = document.getElementById(targetId);
      idx = el ? pages.indexOf(el) : currentIdx;
    }
    closeDrawer();
    if (body.dataset.view === 'long') {
      const el = document.getElementById(subTarget || targetId);
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      // scrollspy will catch up the header
    } else {
      goTo(idx, { subTarget });
    }
  });

  document.addEventListener('keydown', (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.target.matches('input, textarea, [contenteditable]')) return;
    if (body.dataset.view !== 'short') return;
    if (e.key === 'ArrowRight' || e.key === 'PageDown') {
      e.preventDefault();
      goTo(currentIdx + 1);
    } else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
      e.preventDefault();
      goTo(currentIdx - 1);
    } else if (e.key === 'Escape') {
      closeDrawer();
    }
  });

  window.addEventListener('hashchange', () => {
    const idx = pageByHash(location.hash);
    if (body.dataset.view === 'short') goTo(idx);
    else {
      currentIdx = idx;
      setHeaderTitle();
      setActiveTocLink();
    }
  });

  /* ── boot ─────────────────────────────────────────────────────────── */
  let view = 'long';
  try { view = localStorage.getItem(LS_VIEW) || 'long'; } catch (e) {}
  body.dataset.view = view;

  currentIdx = pageByHash(location.hash);
  setView(view);
  applyCurrent({ scrollToTop: false });
  installScrollSpy();

  /* MathJax may have done its initial pass before the fetched chapters
     arrived. Retypeset the whole document once MathJax is ready. */
  if (window.MathJax) {
    const startup = window.MathJax.startup?.promise || Promise.resolve();
    startup.then(() => {
      if (window.MathJax.typesetPromise) {
        return window.MathJax.typesetPromise([document.querySelector('main')]);
      }
    }).catch(() => {});
  }
})();

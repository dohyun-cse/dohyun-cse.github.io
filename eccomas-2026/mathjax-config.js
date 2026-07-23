/* MathJax configuration — must run BEFORE the MathJax library script
 * (loaded in index.html). Add or edit macros here. */

window.MathJax = {
  loader: { load: ['[tex]/color'] },
  tex: {
    tags: 'ams',
    packages: { '[+]': ['color'] },
    inlineMath:  [['$',  '$'],  ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    macros: {
      /* RGB tuples — MathJax's color extension supports the 'RGB'
         model (0–255 components) but not xcolor's 'HTML' model,
         and a bare '#hex' clashes with macro '#N' params. */
      RED:    ['{\\color[RGB]{250,45,25}#1}',  1],   /* #FA2D19 */
      BLUE:   ['{\\color[RGB]{0,86,179}#1}',   1],   /* #0056B3 */
      GREEN:  ['{\\color[RGB]{30,132,73}#1}',  1],   /* #1E8449 */
      ORANGE: ['{\\color[RGB]{224,123,0}#1}',  1],   /* #E07B00 */

      dd:     '\\mathrm{d}',
      coloneqq: '\\mathrel{:=}',
      norm:   ['\\left\\lVert #1 \\right\\rVert', 1],
      argmin: '\\operatorname*{arg\\,min}',
      essinf: '\\operatorname*{ess\\,inf}',
      esssup: '\\operatorname*{ess\\,sup}'
    }
  },
  svg: { fontCache: 'global' }
};

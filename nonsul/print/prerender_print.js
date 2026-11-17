// pandoc 의 span.math 를 KaTeX HTML(MathML 제외)로 치환. 사용: node prerender_print.js <katex_dist> <file.html>
const fs = require('fs');
const katex = require(process.argv[2] + '/katex.js');
const f = process.argv[3];
let h = fs.readFileSync(f, 'utf8');
const un = s => s.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&amp;/g, '&');
let n = 0; const bad = [];
h = h.replace(/<span class="math (inline|display)">([\s\S]*?)<\/span>/g, (m, k, t) => {
  t = un(t).replace(/^\\\(|\\\)$/g, '').replace(/^\\\[|\\\]$/g, '');
  n++;
  const o = { displayMode: k === 'display', output: 'html', strict: false };
  try { return katex.renderToString(t, { ...o, throwOnError: true }); }
  catch (e) { bad.push(t.slice(0, 60) + ' :: ' + e.message.slice(0, 50)); return katex.renderToString(t, { ...o, throwOnError: false }); }
});
fs.writeFileSync(f, h);
console.log('math', n, 'errors', bad.length);
if (bad.length) console.log(bad.slice(0, 10).join('\n'));

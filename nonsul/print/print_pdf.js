// HTML -> A4 PDF (Chromium). 사용: node print_pdf.js <in.html> <out.pdf> "<푸터 문구>"
const { chromium } = require('playwright');
(async () => {
  const [inp, out, label] = process.argv.slice(2);
  const b = await chromium.launch();
  const p = await b.newPage();
  await p.goto('file://' + inp, { waitUntil: 'load', timeout: 300000 });
  await p.evaluate(() => document.fonts.ready);
  await p.pdf({
    path: out, format: 'A4', printBackground: true, displayHeaderFooter: true,
    margin: { top: '17mm', bottom: '19mm', left: '17mm', right: '17mm' },
    headerTemplate: '<div></div>',
    footerTemplate: `<div style="width:100%;font-size:8px;text-align:center;color:#444;font-family:'WenQuanYi Zen Hei',sans-serif">${label} · <span class="pageNumber"></span></div>`,
    timeout: 600000,
  });
  await b.close();
})();

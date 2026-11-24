const {chromium}=require('playwright');
(async()=>{const b=await chromium.launch();const p=await b.newPage({viewport:{width:800,height:1000}});const d='file://'+process.cwd()+'/';
await p.goto(d+'cover_clean.html');await p.evaluate(()=>document.fonts.ready);
await p.pdf({path:'cover_clean.pdf',printBackground:true,preferCSSPageSize:true});
await p.goto(d+(process.env.BODY||'body_clean.html'));await p.emulateMedia({media:'print'});await p.evaluate(()=>document.fonts.ready);
await p.pdf({path:'body_clean.pdf',printBackground:true,preferCSSPageSize:true,displayHeaderFooter:true,headerTemplate:'<span></span>',
footerTemplate:'<div style="width:100%;font-size:8px;color:#5A534A;text-align:center;font-family:Liberation Sans,sans-serif">— <span class="pageNumber"></span> —</div>'});
await b.close();})();

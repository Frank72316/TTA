// Render the editable layout directly to PDF using the cached assets.
const fs = require('node:fs');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const { chromium } = require('playwright');
(async () => {
  const python = process.env.PYTHON || (process.platform === 'win32' ? 'py' : 'python3');
  const layout = execFileSync(python, ['-B', path.join(__dirname, 'compose.py'), '--print-layout'], {
    encoding: 'utf8', maxBuffer: 32 * 1024 * 1024,
  });
  const pdf = path.join(__dirname, 'TTA阶段交接报告.pdf');
  const exe = process.env.CHROME_PATH || ['C:/Program Files/Google/Chrome/Application/chrome.exe','/usr/bin/chromium','/usr/bin/google-chrome'].find(fs.existsSync);
  const browser = await chromium.launch({ executablePath:exe, headless:true });
  try {
    const page = await browser.newPage();
    await page.emulateMedia({media:'print'});
    await page.setContent(layout, {waitUntil:'load'});
    await page.evaluate(() => document.fonts.ready);
    const pages = await page.evaluate(() => [...document.querySelectorAll('.page')].map((p,i) => {
      const footer = p.querySelector('footer').getBoundingClientRect();
      const body = p.querySelector('main').getBoundingClientRect();
      const imgs = [...p.querySelectorAll('img')];
      if (imgs.some(x=>!x.complete||x.naturalWidth!==224||!x.src.startsWith('data:image/png'))) throw Error('Image validation failed on page '+(i+1));
      return {page:i+1,free_mm:+((footer.top-body.bottom)*25.4/96).toFixed(1),imgs:imgs.length};
    }));
    console.table(pages);
    if (pages.length!==8 || pages.some(p=>p.free_mm < 1)) throw Error('Content intersects footer');
    await page.pdf({path:pdf,preferCSSPageSize:true,printBackground:true,displayHeaderFooter:false,tagged:true});
  } finally { await browser.close(); }
})().catch(e=>{console.error(e);process.exitCode=1;});


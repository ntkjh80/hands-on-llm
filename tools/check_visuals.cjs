#!/usr/bin/env node
// Optional browser QA. Install Playwright separately or set PLAYWRIGHT_MODULE.
const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');
const { createHash } = require('crypto');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(__dirname, '..');
const output = path.join(root, 'output', 'qa');
fs.mkdirSync(output, { recursive: true });

async function main() {
  const launch = { headless: true };
  if (process.env.CHROME_EXECUTABLE) launch.executablePath = process.env.CHROME_EXECUTABLE;
  const browser = await chromium.launch(launch);
  const results = [];
  const requested = process.argv.slice(2).map(Number);
  const numbers = requested.length ? requested : Array.from({ length: 12 }, (_, i) => i + 1);
  if (numbers.some(n => !Number.isInteger(n) || n < 1 || n > 12)) throw new Error('Use chapter numbers 1 through 12');
  for (const n of numbers) {
    const chapter = `ch${String(n).padStart(2, '0')}`;
    const file = path.join(root, 'chapters', chapter, 'visuals.html');
    if (!fs.existsSync(file)) { results.push({ chapter, missing: true }); continue; }
    const page = await browser.newPage({ viewport: { width: 1100, height: 850 } });
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
    await page.goto(pathToFileURL(file).href, { waitUntil: 'load' });
    await page.screenshot({ path: path.join(output, `${chapter}-desktop.png`), fullPage: true });
    const interactions = [];
    const controls = await page.locator('input[type=range], select, input[type=number]').all();
    for (const control of controls) {
      if (!(await control.isVisible()) || !(await control.isEnabled())) continue;
      const before = await page.locator('body').innerText();
      const beforeMarkup = await page.locator('body').innerHTML();
      const info = await control.evaluate(el => ({ tag: el.tagName, type: el.type, id: el.id,
        value: el.value, min: el.min, max: el.max, step: el.step,
        options: el.options ? [...el.options].map(x => x.value) : [] }));
      let altered = false;
      if (info.tag === 'SELECT' && info.options.length > 1) {
        const next = info.options.find(x => x !== info.value);
        await control.selectOption(next);
        altered = true;
      } else if (info.type === 'range' || info.type === 'number') {
        const min = info.min === '' ? 0 : Number(info.min);
        const max = info.max === '' ? Number(info.value) + 1 : Number(info.max);
        const next = Number(info.value) === max ? min : max;
        await control.fill(String(next));
        await control.dispatchEvent('input');
        await control.dispatchEvent('change');
        altered = true;
      }
      if (altered) {
        await page.waitForTimeout(80);
        interactions.push({ id: info.id, type: info.type,
          textChanged: (await page.locator('body').innerText()) !== before,
          markupChanged: (await page.locator('body').innerHTML()) !== beforeMarkup });
        if (info.tag === 'SELECT') await control.selectOption(info.value);
        else { await control.fill(info.value); await control.dispatchEvent('input'); await control.dispatchEvent('change'); }
      }
    }
    await page.setViewportSize({ width: 390, height: 844 });
    await page.screenshot({ path: path.join(output, `${chapter}-mobile.png`), fullPage: true });
    const overflow = await page.evaluate(() => ({
      document: document.documentElement.scrollWidth,
      viewport: window.innerWidth,
      elements: [...document.querySelectorAll('body *')]
        .filter(el => el.getBoundingClientRect().right > window.innerWidth + 2 &&
          el.getBoundingClientRect().width > 0)
        .slice(0, 8).map(el => ({ tag: el.tagName, id: el.id, className: String(el.className) }))
    }));
    const row = { chapter, sha256: createHash('sha256').update(fs.readFileSync(file)).digest('hex'), errors, interactions, overflow };
    results.push(row);
    console.log(JSON.stringify(row));
    await page.close();
  }
  await browser.close();
  const resultPath = path.join(output, 'visual-checks.json');
  const previous = fs.existsSync(resultPath) ? JSON.parse(fs.readFileSync(resultPath, 'utf8')) : [];
  const merged = new Map(previous.map(row => [row.chapter, row]));
  for (const row of results) merged.set(row.chapter, row);
  fs.writeFileSync(resultPath, JSON.stringify([...merged.values()].sort((a,b) => a.chapter.localeCompare(b.chapter)), null, 2));
  if (results.some(x => x.missing || x.errors?.length)) process.exitCode = 1;
}
main().catch(e => { console.error(e); process.exitCode = 1; });

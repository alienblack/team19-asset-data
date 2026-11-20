/** Browser checks for a running local workbench. Uses synthetic example data only. */
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const args = process.argv.slice(2);
const option = (name, fallback) => args.includes(name) ? args[args.indexOf(name) + 1] : fallback;
const root = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const base = option('--url', 'http://127.0.0.1:8765');
const out = resolve(option('--out', resolve(root, 'outputs/ui-checks', new Date().toISOString().replace(/[:.]/g, '-'))));
await mkdir(out, { recursive: true });
const checks = [], runs = [], pageErrors = [];
const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({ viewport: { width: 1512, height: 1050 }, acceptDownloads: true });
context.on('page', page => page.on('pageerror', error => pageErrors.push(error.message)));
const page = await context.newPage();
const check = async (name, fn) => { await fn(); checks.push(name); console.log('PASS · ' + name); };
const isHidden = async locator => !(await locator.isVisible());
const row = name => page.locator('#schema tbody tr').filter({ has: page.locator('strong', { hasText: new RegExp('^' + name + '$') }) });
async function openDetails(label) {
  const summary = page.locator('summary').filter({ hasText: new RegExp('^' + label + '$') });
  if (!(await summary.evaluate(el => el.parentElement.open))) await summary.click();
}
async function fastSettings() {
  await openDetails('Evaluation settings');
  await page.locator('#folds').fill('2');
  await page.locator('#evaluation-rows').fill('80');
  await openDetails('Model settings');
  await page.locator('#trees').fill('8');
  await page.locator('#passes').fill('1');
  await page.locator('#em-steps').fill('5');
}
async function runStudy(name) {
  await page.locator('#report-title').fill(name);
  const responsePromise = page.waitForResponse(r => r.url().endsWith('/api/run') && r.request().method() === 'POST');
  await page.locator('#run-button').click();
  const response = await responsePromise;
  assert.equal(response.status(), 200, await response.text());
  const { job } = await response.json();
  const sent = response.request().postDataJSON().config;
  await page.locator('#result').waitFor({ state: 'visible', timeout: 180000 });
  const state = await (await context.request.get(base + '/api/jobs/' + job)).json();
  assert.equal(state.status, 'success', JSON.stringify(state));
  assert.equal(await page.locator('#job-badge').innerText(), 'Complete');
  const reportUrl = new URL(await page.locator('#report-link').getAttribute('href'), base).href;
  const summary = await (await context.request.get(new URL('summary.json', reportUrl).href)).json();
  assert.equal(summary.failures.length, 0);
  assert.equal(summary.rows, 240);
  runs.push({ name, job, report: reportUrl, score_reference: summary.score_reference, config: sent });
  return { reportUrl, summary, sent, job };
}
try {
  await page.goto(base);
  await check('Initial screen and missing-upload validation', async () => {
    assert.equal(await page.locator('h1').innerText(), 'Fill missing asset data');
    assert(await page.locator('#run-button').isDisabled());
    await page.locator('#inspect-button').click();
    assert.match(await page.locator('#error').innerText(), /Choose a dataset CSV first/);
    assert(await page.locator('#error').isVisible());
  });
  await check('Example profile, types and progressive settings', async () => {
    await page.locator('#example-button').click();
    await page.waitForFunction(() => document.getElementById('summary-rows').textContent === '240');
    assert.equal(await page.locator('#summary-columns').innerText(), '7');
    assert.equal(await page.locator('#summary-missing').innerText(), '207');
    assert.equal(await row('asset_id').locator('[data-key=type]').inputValue(), 'ignore');
    assert.equal(await row('material').locator('[data-key=type]').inputValue(), 'categorical');
    assert(await isHidden(row('span_m').locator('[data-key=group]')));
    await page.locator('#field-options').click();
    assert.equal(await row('span_m').locator('[data-key=group]').inputValue(), 'location');
    await page.locator('#field-options').click();
    assert(await isHidden(page.locator('[name=method][value=ridge]')));
    await openDetails('More methods');
    assert(await page.locator('[name=method][value=ridge]').isVisible());
    assert.equal(await page.locator('[name=method]').count(), 12);
    await page.screenshot({ path: resolve(out, 'workbench-desktop.png'), fullPage: true });
  });
  await fastSettings();
  let first;
  await check('Fresh scattered run without a truth file', async () => {
    first = await runStudy('UI check · scattered');
    assert.match(first.summary.score_reference, /Recorded/);
    assert.equal(first.sent.masking, 'scattered');
  });
  const report = await context.newPage();
  await report.goto(first.reportUrl);
  await check('Report, filters, chart enlargement and PNG download', async () => {
    assert.equal(await report.locator('#metrics tbody tr').count(), 18);
    await report.locator('#method-filter').selectOption('gaussian');
    assert.equal(await report.locator('#metrics tbody tr:not(.hidden)').count(), 6);
    await report.locator('#target-filter').selectOption('span_m');
    assert.equal(await report.locator('#metrics tbody tr:not(.hidden)').count(), 1);
    await report.locator('.chart-image').first().click();
    assert(await report.locator('#chart-dialog').isVisible());
    await report.locator('#close-dialog').click();
    const download = report.waitForEvent('download');
    await report.locator('.download-chart').first().click();
    const file = await download;
    assert.match(file.suggestedFilename(), /\.png$/);
    await file.saveAs(resolve(out, 'downloaded-chart.png'));
  });
  await check('Completed CSV and configuration downloads', async () => {
    for (const name of ['completed_median.csv', 'completed_forest.csv', 'completed_gaussian.csv', 'config.json']) {
      const download = report.waitForEvent('download');
      await report.locator('a[download][href="' + name + '"]').first().click();
      await (await download).saveAs(resolve(out, name));
    }
    const csv = await readFile(resolve(out, 'completed_gaussian.csv'), 'utf8');
    assert.equal(csv.trimEnd().split('\n').length, 241);
  });
  await check('Print includes all rows and decoded figures, then restores filters', async () => {
    await report.evaluate(() => { window.print = () => { window.printCheck = { rows: document.querySelectorAll('#metrics tbody tr:not(.hidden)').length, imagesReady: [...document.querySelectorAll('.chart img')].every(i => i.complete && i.naturalWidth > 0) }; }; });
    await report.evaluate(() => window.printReport());
    const printed = await report.evaluate(() => window.printCheck);
    assert.equal(printed.rows, 18);
    assert(printed.imagesReady);
    assert.equal(await report.locator('#metrics tbody tr:not(.hidden)').count(), 1);
    await report.locator('#method-filter').selectOption('');
    await report.locator('#target-filter').selectOption('');
    await report.pdf({ path: resolve(out, 'browser-print.pdf'), format: 'A4', printBackground: true });
    await report.screenshot({ path: resolve(out, 'report-desktop.png'), fullPage: true });
  });
  await check('CSV upload with aligned truth and selection invalidation', async () => {
    await page.locator('#dataset-file').setInputFiles(resolve(root, 'examples/example_assets.csv'));
    assert(await page.locator('#run-button').isDisabled());
    assert(await isHidden(page.locator('#schema')));
    await openDetails('Reference data \\(optional\\)');
    await page.locator('#truth-file').setInputFiles(resolve(root, 'examples/example_truth.csv'));
    await page.locator('#inspect-button').click();
    await page.waitForFunction(() => document.getElementById('dataset-message').textContent.includes('matching truth supplied'));
    const result = await runStudy('UI check · uploaded truth');
    assert.match(result.summary.score_reference, /Supplied truth/);
  });
  await check('Grouped gaps reveal groups; selected measurement noise is submitted', async () => {
    await page.locator('#example-button').click();
    await page.waitForFunction(() => document.getElementById('summary-rows').textContent === '240');
    await page.waitForFunction(() => [...document.querySelectorAll('#schema tbody tr')].some(tr => tr.querySelector('strong').textContent === 'coastal_index' && tr.querySelector('[data-key=group]').value === 'location'));
    await page.locator('#masking').selectOption('grouped');
    assert(await row('span_m').locator('[data-key=group]').isVisible());
    assert.equal(await row('coastal_index').locator('[data-key=group]').inputValue(), 'location');
    await openDetails('Optional measurement-noise stress test');
    await page.locator('#noise').fill('5');
    for (const name of ['span_m', 'coastal_index']) await page.locator('#noise-columns input[value="' + name + '"]').check();
    const result = await runStudy('UI check · grouped with noise');
    assert.equal(result.sent.masking, 'grouped');
    assert.equal(result.sent.noise, .05);
    assert.deepEqual(result.sent.noise_columns.sort(), ['coastal_index', 'span_m']);
    assert.notEqual(runs[0].job, result.job);
  });
  await check('Bad CSV headers produce a visible validation error', async () => {
    await page.locator('#truth-file').setInputFiles([]);
    await page.locator('#dataset-file').setInputFiles({ name: 'invalid.csv', mimeType: 'text/csv', buffer: Buffer.from('same,same\n' + '1,2\n'.repeat(12)) });
    await page.locator('#inspect-button').click();
    await page.locator('#error').waitFor({ state: 'visible' });
    assert.match(await page.locator('#error').innerText(), /unique|duplicate/i);
    assert(await page.locator('#run-button').isDisabled());
  });
  await check('Invalid grouped and noise settings do not show a false running state', async () => {
    await page.locator('#example-button').click();
    await page.waitForFunction(() => document.getElementById('summary-rows').textContent === '240');
    await page.locator('#masking').selectOption('grouped');
    for (const input of await page.locator('#schema [data-key=group]').all()) await input.fill('');
    await page.locator('#noise').fill('0');
    await page.locator('#run-button').click();
    await page.locator('#error').waitFor({ state: 'visible' });
    assert.match(await page.locator('#error').innerText(), /group/i);
    assert(['Not started', 'Failed'].includes(await page.locator('#job-badge').innerText()));
    assert(await isHidden(page.locator('#result')));
    await page.locator('#masking').selectOption('scattered');
    await page.locator('#noise').fill('5');
    for (const box of await page.locator('#noise-columns input').all()) await box.uncheck();
    await page.locator('#run-button').click();
    await page.waitForFunction(() => document.getElementById('error').textContent.toLowerCase().includes('noise'));
    assert(['Not started', 'Failed'].includes(await page.locator('#job-badge').innerText()));
  });
  await check('Desktop and mobile have no page-level horizontal overflow', async () => {
    for (const width of [1512, 390]) {
      await page.setViewportSize({ width, height: width === 390 ? 844 : 1050 });
      await page.goto(base);
      await page.locator('#example-button').click();
      await page.waitForFunction(() => document.getElementById('summary-rows').textContent === '240');
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
      await report.setViewportSize({ width, height: 844 });
      assert(await report.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
      if (width === 390) await page.screenshot({ path: resolve(out, 'workbench-mobile.png'), fullPage: true });
    }
  });
  await check('No uncaught browser JavaScript errors', async () => assert.deepEqual(pageErrors, []));
  await writeFile(resolve(out, 'ui-results.json'), JSON.stringify({ passed: true, checks, runs, pageErrors, tested_utc: new Date().toISOString() }, null, 2) + '\n');
  console.log('Evidence: ' + out);
} catch (error) {
  await page.screenshot({ path: resolve(out, 'failure.png'), fullPage: true }).catch(() => {});
  await writeFile(resolve(out, 'ui-results.json'), JSON.stringify({ passed: false, checks, runs, pageErrors, error: String(error.stack) }, null, 2) + '\n');
  throw error;
} finally { await browser.close(); }

// =====================================================================
// GoldMind frontend logic
// Talks to the Flask API (/api/summary, /api/year-forecast, /api/history,
// /api/predict) and renders the numbers + Chart.js charts.
// Supports Arabic (RTL) / English (LTR) via a lightweight i18n layer,
// and a manual refresh button that force-bypasses the server cache.
// =====================================================================

// ---------------------------- i18n ----------------------------------
const TRANSLATIONS = {
  ar: {
    dir: 'rtl',
    locale: 'ar-EG',
    currency: 'ج.م',
    pageTitle: 'GoldMind — توقعات سعر الذهب',
    tagline: 'نظام تنبؤ أسعار الذهب — عيار 21 مصر',
    loading: 'جاري التحميل…',
    live: 'مباشر',
    partialLive: 'مباشر (جزئي)',
    estimatedLive: 'تقدير لحظي',
    cachedPrice: 'آخر سعر مسجل',
    updatedToday: (d) => `تحديث اليوم ${d}`,
    dataThroughShort: (d) => `بيانات حتى ${d}`,
    refreshTitle: 'تحديث الأسعار والتوقعات الآن',
    themeTitle: 'تبديل الوضع الليلي/النهاري',
    currentPriceLabel: 'السعر الحالي · جرام 21',
    tomorrowLabel: 'توقع الموديل · بكرة',
    weekLabel: 'توقع الموديل · الأسبوع القادم',
    onDate: (d) => `يوم ${d}`,
    closesOn: (d) => `يغلق يوم ${d}`,
    expectedRange: (lo, hi, cur) => `المدى المتوقع: ${lo} – ${hi} ${cur}`,
    weekChartTitle: 'توقعات الأسبوع القادم',
    pastYearTitle: 'السنة الماضية',
    pastMonthTitle: 'الشهر الماضي',
    nextYearTitle: 'توقعات السنة القادمة',
    bandDisclaimer: 'شريط اللون حوالين الخط بيمثل مدى عدم اليقين (10٪–90٪) — كل ما التاريخ أبعد، المدى بيكبر طبيعيًا.',
    lookupTitle: 'اسأل عن أي تاريخ',
    lookupCopy: 'اختار أي تاريخ في السنة القادمة، والموديل هيتوقعلك السعر المتوقع في اليوم ده.',
    lookupButton: 'توقّع السعر',
    lookupCalculating: 'بيتوقع…',
    lookupResultDate: (d) => `التاريخ: ${d}`,
    lookupResultCurrent: (p, cur) => `السعر الحالي: ${p} ${cur}`,
    lookupResultRange: (lo, hi, cur) => `المدى المتوقع: ${lo} – ${hi} ${cur}`,
    footerDisclaimer: 'البيانات مبنية على أسعار السوق والدولار وGold World وغيرها من المؤشرات الاقتصادية. التوقعات تقديرية وليست نصيحة استثمارية.',
    dataThroughFooter: (d) => `آخر تحديث لبيانات التدريب: ${d}`,
    tooltipPrice: (v) => ` ${v} ج.م`,
    genericError: 'حصل خطأ، حاول تاني',
  },
  en: {
    dir: 'ltr',
    locale: 'en-US',
    currency: 'EGP',
    pageTitle: 'GoldMind — Gold Price Forecast',
    tagline: 'Gold Price Forecasting System — Egypt 21K',
    loading: 'Loading…',
    live: 'Live',
    partialLive: 'Live (partial)',
    estimatedLive: 'Live estimate',
    cachedPrice: 'Last recorded price',
    updatedToday: (d) => `Updated today ${d}`,
    dataThroughShort: (d) => `Data through ${d}`,
    refreshTitle: 'Refresh prices and forecasts now',
    themeTitle: 'Toggle dark/light mode',
    currentPriceLabel: 'Current Price · 21K/gram',
    tomorrowLabel: "Model Forecast · Tomorrow",
    weekLabel: 'Model Forecast · Next Week',
    onDate: (d) => `On ${d}`,
    closesOn: (d) => `Closes ${d}`,
    expectedRange: (lo, hi, cur) => `Expected range: ${lo} – ${hi} ${cur}`,
    weekChartTitle: 'Next Week Forecast',
    pastYearTitle: 'Past Year',
    pastMonthTitle: 'Past Month',
    nextYearTitle: 'Next Year Forecast',
    bandDisclaimer: 'The shaded band represents the uncertainty range (10%–90%) — it naturally widens the further out the date is.',
    lookupTitle: 'Ask About Any Date',
    lookupCopy: "Pick any date within the coming year, and the model will predict the expected price for that day.",
    lookupButton: 'Predict Price',
    lookupCalculating: 'Calculating…',
    lookupResultDate: (d) => `Date: ${d}`,
    lookupResultCurrent: (p, cur) => `Current price: ${p} ${cur}`,
    lookupResultRange: (lo, hi, cur) => `Expected range: ${lo} – ${hi} ${cur}`,
    footerDisclaimer: 'Data is based on market prices, USD/EGP, world gold prices, and other economic indicators. Forecasts are estimates, not investment advice.',
    dataThroughFooter: (d) => `Training data last updated: ${d}`,
    tooltipPrice: (v) => ` ${v} EGP`,
    genericError: 'Something went wrong, please try again',
  },
};

let currentLang = localStorage.getItem('goldmind-lang') || 'ar';
const t = () => TRANSLATIONS[currentLang];
const fmt = (n) => Number(n).toLocaleString(t().locale, { maximumFractionDigits: 0 });

function applyLanguage(lang) {
  currentLang = lang;
  localStorage.setItem('goldmind-lang', lang);

  document.documentElement.lang = lang;
  document.documentElement.dir = t().dir;
  document.getElementById('pageTitle').textContent = t().pageTitle;
  document.getElementById('langLabel').textContent = lang === 'ar' ? 'EN' : 'AR';

  document.querySelectorAll('[data-i18n]').forEach((el) => {
    const key = el.getAttribute('data-i18n');
    const val = t()[key];
    if (typeof val === 'string') el.textContent = val;
  });
  document.querySelectorAll('[data-i18n-title]').forEach((el) => {
    const key = el.getAttribute('data-i18n-title');
    const val = t()[key];
    if (typeof val === 'string') { el.title = val; el.setAttribute('aria-label', val); }
  });

  document.querySelectorAll('.currency').forEach((el) => { el.textContent = t().currency; });

  // Re-render anything already loaded so it picks up the new language,
  // instead of waiting for the next refresh.
  if (lastSummaryData) renderSummary(lastSummaryData);
  if (lastWeekData) renderWeekChart(lastWeekData);
  if (lastYearForecastData) renderYearForecastChart(lastYearForecastData);
  if (lastYearActualData) renderActualChart(lastYearActualData, 'yearActualChart', 'yearActual');
  if (lastMonthActualData) renderActualChart(lastMonthActualData, 'monthActualChart', 'monthActual');
}

document.getElementById('langToggle').addEventListener('click', () => {
  applyLanguage(currentLang === 'ar' ? 'en' : 'ar');
});

// ---------------------------- theme toggle ----------------------------
const root = document.documentElement;
const themeToggle = document.getElementById('themeToggle');

function applyStoredTheme() {
  const stored = localStorage.getItem('goldmind-theme');
  if (stored) root.setAttribute('data-theme', stored);
}
applyStoredTheme();

themeToggle.addEventListener('click', () => {
  const next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  root.setAttribute('data-theme', next);
  localStorage.setItem('goldmind-theme', next);
  updateChartTheme();
});

// ---------------------------- chart color helpers ----------------------
function chartColors() {
  const dark = root.getAttribute('data-theme') === 'dark';
  return {
    gold: dark ? '#d4af6a' : '#9a6b1f',
    goldFill: dark ? 'rgba(212,175,106,0.15)' : 'rgba(154,107,31,0.12)',
    text: dark ? '#9a978f' : '#6b6355',
    grid: dark ? 'rgba(212,175,106,0.08)' : 'rgba(120,90,30,0.10)',
    band: dark ? 'rgba(212,175,106,0.10)' : 'rgba(154,107,31,0.08)',
    up: dark ? '#6fbf8f' : '#2f8a5b',
    upFill: dark ? 'rgba(111,191,143,0.14)' : 'rgba(47,138,91,0.10)',
    down: dark ? '#d97a6c' : '#b0492f',
    downFill: dark ? 'rgba(217,122,108,0.14)' : 'rgba(176,73,47,0.10)',
  };
}

// TradingView-style segment coloring: each little piece of the line is
// green when it rises from the previous point and red when it falls,
// instead of one flat color for the whole line.
function directionalSegment(c) {
  return {
    borderColor: (ctx) => (ctx.p0.parsed.y <= ctx.p1.parsed.y ? c.up : c.down),
  };
}

// Colors a single value (e.g. a KPI card's predicted price) green/red
// relative to a reference price - used next to the tomorrow/week numbers.
function directionColor(value, reference, c) {
  return value >= reference ? c.up : c.down;
}

const charts = {};

function baseOptions() {
  const c = chartColors();
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (ctx) => t().tooltipPrice(fmt(ctx.parsed.y)),
        },
      },
    },
    scales: {
      x: { ticks: { color: c.text, font: { family: 'JetBrains Mono', size: 11 } }, grid: { color: 'transparent' } },
      y: { ticks: { color: c.text, font: { family: 'JetBrains Mono', size: 11 } }, grid: { color: c.grid } },
    },
  };
}

function updateChartTheme() {
  Object.values(charts).forEach((ch) => {
    if (!ch) return;
    const opts = baseOptions();
    ch.options.scales = opts.scales;
    ch.data.datasets.forEach((ds) => {
      if (ds._role === 'line') ds.borderColor = chartColors().gold;
      if (ds._role === 'fill') ds.backgroundColor = chartColors().goldFill;
      if (ds._role === 'band') ds.backgroundColor = chartColors().band;
    });
    ch.update();
  });
}

// Destroys an existing Chart.js instance bound to a canvas before making
// a new one - Chart.js throws "Canvas is already in use" otherwise, which
// matters now that charts can be re-rendered (refresh button, language
// switch) instead of only being created once at page load.
function destroyChart(key) {
  if (charts[key]) {
    charts[key].destroy();
    charts[key] = null;
  }
}

// ---------------------------- API calls ----------------------------
async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error((await res.json()).error || t().genericError);
  return res.json();
}

// Cache of the last successfully loaded payload for each section, so a
// language switch can instantly re-render with the new strings without
// waiting on a fresh network round trip.
let lastSummaryData = null, lastWeekData = null, lastYearForecastData = null;
let lastYearActualData = null, lastMonthActualData = null;

function renderSummary(data) {
  document.getElementById('currentPrice').textContent = fmt(data.current_price);
  const pill = document.getElementById('livePill');
  const label = document.getElementById('liveLabel');
  if (data.price_source === 'live') {
    pill.classList.add('is-live');
    label.textContent = t().live;
  } else if (data.price_source === 'partial_live') {
    pill.classList.add('is-live');
    label.textContent = t().partialLive;
  } else if (data.price_source === 'estimated_live') {
    pill.classList.add('is-live');
    label.textContent = t().estimatedLive;
  } else {
    pill.classList.remove('is-live');
    label.textContent = t().cachedPrice;
  }
  // 'partial_live' and 'estimated_live' both still mean today's price is
  // fresh (the actual local quote hasn't ticked yet today for
  // 'estimated_live' - see predictor.py - but the number shown IS
  // recalculated live from world gold/USD movement, not frozen) - only
  // pure 'cached' (no live data at all) should say "data through <date>".
  document.getElementById('currentSourceNote').textContent =
    (data.price_source === 'live' || data.price_source === 'partial_live' || data.price_source === 'estimated_live')
      ? t().updatedToday(data.as_of)
      : t().dataThroughShort(data.model_data_through);

  const tmr = data.tomorrow;
  document.getElementById('tomorrowPrice').textContent = fmt(tmr.predicted_price);
  document.getElementById('tomorrowDate').textContent = t().onDate(tmr.date);
  document.getElementById('tomorrowRange').textContent =
    t().expectedRange(fmt(tmr.predicted_price_low), fmt(tmr.predicted_price_high), t().currency);
  renderDeltaBadge('tomorrowDelta', tmr.predicted_price, data.current_price);

  const weekLast = data.next_7_days[data.next_7_days.length - 1];
  document.getElementById('weekPrice').textContent = fmt(weekLast.predicted_price);
  document.getElementById('weekDate').textContent = t().closesOn(weekLast.date);
  document.getElementById('weekRange').textContent =
    t().expectedRange(fmt(weekLast.predicted_price_low), fmt(weekLast.predicted_price_high), t().currency);
  renderDeltaBadge('weekDelta', weekLast.predicted_price, data.current_price);

  document.getElementById('dataThrough').textContent = t().dataThroughFooter(data.model_data_through);
}

// TradingView-style colored "+12 (+0.2%)" / "-8 (-0.1%)" badge next to a
// predicted price, relative to today's current price.
function renderDeltaBadge(elId, predicted, current) {
  const el = document.getElementById(elId);
  if (!el || current == null) return;
  const diff = predicted - current;
  const pct = current !== 0 ? (diff / current) * 100 : 0;
  const up = diff >= 0;
  const sign = up ? '+' : '';
  el.textContent = `${sign}${fmt(diff)} (${sign}${pct.toFixed(1)}%)`;
  el.classList.toggle('is-up', up);
  el.classList.toggle('is-down', !up);
}

async function loadSummary(force) {
  const data = await getJSON(`/api/summary${force ? '?force=1' : ''}`);
  lastSummaryData = data;
  lastWeekData = data.next_7_days;
  renderSummary(data);
  renderWeekChart(data.next_7_days);
}

function renderWeekChart(weekData) {
  destroyChart('week');
  const c = chartColors();
  const ctx = document.getElementById('weekChart');
  charts.week = new Chart(ctx, {
    type: 'line',
    data: {
      labels: weekData.map((d) => d.date.slice(5)),
      datasets: [
        {
          data: weekData.map((d) => d.predicted_price_high),
          borderColor: 'transparent',
          backgroundColor: c.band,
          fill: '+1',
          pointRadius: 0,
          _role: 'band',
        },
        {
          data: weekData.map((d) => d.predicted_price_low),
          borderColor: 'transparent',
          backgroundColor: 'transparent',
          fill: false,
          pointRadius: 0,
        },
        {
          data: weekData.map((d) => d.predicted_price),
          borderColor: c.gold,
          segment: directionalSegment(c),
          backgroundColor: c.goldFill,
          borderWidth: 2.5,
          tension: 0.35,
          fill: false,
          pointRadius: 4,
          pointBackgroundColor: (ctx) => {
            const i = ctx.dataIndex;
            const arr = ctx.dataset.data;
            const prev = i > 0 ? arr[i - 1] : arr[i];
            return arr[i] >= prev ? c.up : c.down;
          },
          _role: 'line',
        },
      ],
    },
    options: baseOptions(),
  });
}

async function loadYearForecast(force) {
  const data = await getJSON(`/api/year-forecast${force ? '?force=1' : ''}`);
  lastYearForecastData = data.forecast;
  renderYearForecastChart(data.forecast);
}

function renderYearForecastChart(forecast) {
  destroyChart('yearForecast');
  const c = chartColors();
  const ctx = document.getElementById('yearForecastChart');
  charts.yearForecast = new Chart(ctx, {
    type: 'line',
    data: {
      labels: forecast.map((d) => d.date),
      datasets: [
        {
          data: forecast.map((d) => d.predicted_price_high),
          borderColor: 'transparent',
          backgroundColor: c.band,
          fill: '+1',
          pointRadius: 0,
          _role: 'band',
        },
        {
          data: forecast.map((d) => d.predicted_price_low),
          borderColor: 'transparent',
          backgroundColor: 'transparent',
          fill: false,
          pointRadius: 0,
        },
        {
          data: forecast.map((d) => d.predicted_price),
          borderColor: c.gold,
          segment: directionalSegment(c),
          backgroundColor: c.goldFill,
          borderWidth: 2.5,
          tension: 0.3,
          fill: false,
          pointRadius: 0,
          _role: 'line',
        },
      ],
    },
    options: {
      ...baseOptions(),
      scales: {
        ...baseOptions().scales,
        x: { ...baseOptions().scales.x, ticks: { ...baseOptions().scales.x.ticks, maxTicksLimit: 8 } },
      },
    },
  });
}

async function loadActualChart(range, canvasId, key, force) {
  const data = await getJSON(`/api/history?range=${range}${force ? '&force=1' : ''}`);
  if (key === 'yearActual') lastYearActualData = data.history;
  if (key === 'monthActual') lastMonthActualData = data.history;
  renderActualChart(data.history, canvasId, key);
}

function renderActualChart(history, canvasId, key) {
  destroyChart(key);
  const c = chartColors();
  const ctx = document.getElementById(canvasId);
  charts[key] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: history.map((d) => d.date.slice(5)),
      datasets: [
        {
          data: history.map((d) => d.price),
          borderColor: c.gold,
          segment: directionalSegment(c),
          backgroundColor: c.goldFill,
          borderWidth: 2,
          tension: 0.25,
          fill: true,
          pointRadius: 0,
          _role: 'line',
        },
      ],
    },
    options: {
      ...baseOptions(),
      scales: {
        ...baseOptions().scales,
        x: { ...baseOptions().scales.x, ticks: { ...baseOptions().scales.x.ticks, maxTicksLimit: 8 } },
      },
    },
  });
}

// ---------------------------- load everything (used by boot + refresh) --
async function loadAll(force) {
  await Promise.all([
    loadSummary(force).catch((e) => console.error('summary failed', e)),
    loadYearForecast(force).catch((e) => console.error('year forecast failed', e)),
    loadActualChart('1y', 'yearActualChart', 'yearActual', force).catch((e) => console.error(e)),
    loadActualChart('1m', 'monthActualChart', 'monthActual', force).catch((e) => console.error(e)),
  ]);
}

// ---------------------------- refresh button ----------------------------
const refreshBtn = document.getElementById('refreshBtn');
const refreshIcon = document.getElementById('refreshIcon');

refreshBtn.addEventListener('click', async () => {
  refreshBtn.disabled = true;
  refreshIcon.classList.add('is-spinning');
  try {
    await loadAll(true);
  } finally {
    refreshBtn.disabled = false;
    refreshIcon.classList.remove('is-spinning');
  }
});

// ---------------------------- custom date lookup ----------------------
document.getElementById('lookupForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const dateVal = document.getElementById('lookupDate').value;
  const resultBox = document.getElementById('lookupResult');
  resultBox.innerHTML = `<span class="result-meta">${t().lookupCalculating}</span>`;

  try {
    const data = await getJSON(`/api/predict?date=${dateVal}`);
    const currentLine = lastSummaryData
      ? `<div class="result-line">${t().lookupResultCurrent(fmt(lastSummaryData.current_price), t().currency)}</div>`
      : '';
    resultBox.innerHTML = `
      <div class="result-price">${fmt(data.predicted_price)} ${t().currency}</div>
      <div class="result-line">${t().lookupResultDate(data.date)}</div>
      ${currentLine}
      <div class="result-line">${t().lookupResultRange(fmt(data.predicted_price_low), fmt(data.predicted_price_high), t().currency)}</div>`;
  } catch (err) {
    resultBox.innerHTML = `<span class="result-error">${err.message}</span>`;
  }
});

// set date input bounds (tomorrow -> +364 days)
(function setDateBounds() {
  const input = document.getElementById('lookupDate');
  const today = new Date();
  const min = new Date(today); min.setDate(min.getDate() + 1);
  const max = new Date(today); max.setDate(max.getDate() + 364);
  input.min = min.toISOString().slice(0, 10);
  input.max = max.toISOString().slice(0, 10);
})();

// ---------------------------- boot ----------------------------
applyLanguage(currentLang);
loadAll(false);

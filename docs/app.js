// -------------------- Globals --------------------
let MODEL, META, VOCAB, IDF, SVD, OHE;

// Модель предсказывает P(OTHER). Для UI нужна P(GOOD) → инвертируем.
const INVERT_PROB = true;  // при желании можно читать из META, если добавите поле

// -------------------- Init -----------------------
async function loadAll() {
  const [meta, vocab, idf, svd, ohe] = await Promise.all([
    fetch('artifacts/meta.json').then(r => r.json()),
    fetch('artifacts/tfidf_vocabulary.json').then(r => r.json()),
    fetch('artifacts/tfidf_idf.json').then(r => r.json()),
    fetch('artifacts/svd_components.json').then(r => r.json()),
    fetch('artifacts/ohe_categories.json').then(r => r.json()),
  ]);
  META = meta; VOCAB = vocab; IDF = idf; SVD = svd; OHE = ohe;

  // версионирование по желанию
  MODEL = await tf.loadLayersModel('model_tfjs/model.json?v=6');

  const [catsWine, catsFood, catsCuisine] = OHE;
  fillSelect('wine_category', catsWine);
  fillSelect('food_category', catsFood);
  fillSelect('cuisine',      catsCuisine);

  // В META порог сохранён для исходного класса. Пересчитываем под GOOD.
  const thrRaw  = META.best_threshold ?? 0.5;
  const thrGood = INVERT_PROB ? (1 - thrRaw) : thrRaw;
  document.getElementById('thr').textContent = thrGood.toFixed(4);
}

function fillSelect(id, arr) {
  const el = document.getElementById(id);
  el.innerHTML = arr.map(v => `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`).join('');
}

function escapeHtml(s){
  return String(s).replace(/[&<>"']/g, m => ({
    "&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#039;"
  }[m]));
}

// -------------------- TF-IDF (ngram 1..2, L2) --------------------
function tokenize(str) {
  // как в sklearn: token_pattern=r'(?u)\b\w\w+\b' → минимум 2 символа
  const uni = (String(str).toLowerCase().match(/\b[a-z0-9]{2,}\b/g)) || [];

  // биграммы
  const bi = [];
  for (let i = 0; i + 1 < uni.length; i++) bi.push(uni[i] + ' ' + uni[i + 1]);

  // диапазон n-грамм из META
  const [n1, n2] = META.ngram_range || [1, 1];
  let out = [];
  if (n1 === 1) out = out.concat(uni);
  if (n2 >= 2) out = out.concat(bi);
  return out;
}

function tfidfVector(text) {
  const nFeatures = Object.keys(VOCAB).length;
  const vec = new Float32Array(nFeatures);
  const counts = {};
  for (const t of tokenize(text)) counts[t] = (counts[t] || 0) + 1;

  for (const [tok, cnt] of Object.entries(counts)) {
    const idx = VOCAB[tok];
    if (idx !== undefined) vec[idx] = cnt;
  }
  // умножение на IDF
  for (let j = 0; j < vec.length; j++) vec[j] *= IDF[j];

  // L2-нормализация
  let norm = 0;
  for (let j = 0; j < vec.length; j++) norm += vec[j] * vec[j];
  norm = Math.sqrt(norm) || 1;
  for (let j = 0; j < vec.length; j++) vec[j] /= norm;

  return vec;
}

// -------------------- SVD projection --------------------
function svdProject(vec) {
  const nComp = SVD.length;
  const out = new Float32Array(nComp);
  for (let k = 0; k < nComp; k++) {
    const row = SVD[k];
    let s = 0;
    for (let j = 0; j < vec.length; j++) s += vec[j] * row[j];
    out[k] = s;
  }
  return out;
}

// -------------------- OHE --------------------
function oheVector(value, cats) {
  const out = new Float32Array(cats.length);
  const idx = cats.indexOf(value);
  if (idx >= 0) out[idx] = 1;
  return out;
}

// -------------------- Features build --------------------
function buildFeatures() {
  const desc    = document.getElementById('desc').value.trim();
  const wineCat = document.getElementById('wine_category').value;
  const foodCat = document.getElementById('food_category').value;
  const cuisine = document.getElementById('cuisine').value;

  const vTfidf = tfidfVector(desc);
  const vSvd   = svdProject(vTfidf);

  const [catsWine, catsFood, catsCuisine] = OHE;
  const vWine = oheVector(wineCat, catsWine);
  const vFood = oheVector(foodCat, catsFood);
  const vCuis = oheVector(cuisine, catsCuisine);

  const full = new Float32Array(vSvd.length + vWine.length + vFood.length + vCuis.length);
  let p = 0;
  full.set(vSvd, p);   p += vSvd.length;
  full.set(vWine, p);  p += vWine.length;
  full.set(vFood, p);  p += vFood.length;
  full.set(vCuis, p);

  return { full, desc, wineCat, foodCat, cuisine };
}

// -------------------- Predict --------------------
async function predict() {
  const { full, desc, wineCat, foodCat, cuisine } = buildFeatures();

  const t = tf.tensor2d(full, [1, full.length]);
  const pRaw = (await MODEL.predict(t).data())[0]; // вероятность исходного класса (скорее всего OTHER)
  t.dispose();

  // Преобразуем в вероятность GOOD (если включена инверсия)
  const pGood = INVERT_PROB ? (1 - pRaw) : pRaw;

  // Порог под тот же класс, что и pGood
  const thrRaw  = META.best_threshold ?? 0.5;
  const thrGood = INVERT_PROB ? (1 - thrRaw) : thrRaw;

  const isGood = pGood >= thrGood;

  document.getElementById('prob').textContent = pGood.toFixed(3);
  document.getElementById('res').textContent  = isGood ? 'GOOD MATCH ✅' : 'OTHER / NOT IDEAL ◻️';
  document.getElementById('details').textContent =
    `wine_category=${wineCat} · food_category=${foodCat} · cuisine=${cuisine} · ` +
    `text="${desc.slice(0,120)}${desc.length>120 ? '…' : ''}"`;
}

// -------------------- Hook up --------------------
document.getElementById('btn').addEventListener('click', predict);
loadAll();

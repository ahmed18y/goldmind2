# GoldMind — Gold Price Predictor (Egypt 21K)

مشروع ML كامل بيتعلم من بيانات تاريخية (ذهب عالمي، دولار/جنيه، فوائد أمريكا،
بترول، فضة، S&P500، VIX، ومؤشرات مخاطر سياسية/اقتصادية) عشان يتوقع سعر
الذهب عيار 21 في مصر - بكرة، الأسبوع الجاي، وأي تاريخ لحد سنة قدام.

**مميزات الواجهة:** دعم كامل عربي/إنجليزي (زرار تبديل يغيّر النصوص +
اتجاه الصفحة RTL/LTR + تنسيق الأرقام)، وضع ليلي/نهاري، وزرار تحديث
يدوي بيتجاوز الـ cache ويجيب أحدث سعر وتوقعات لحظيًا وقت الضغط عليه.

## هيكل المشروع

```
GoldMind/
├── app.py                     # Flask backend (API + يعرض الصفحة)
├── requirements.txt            # Runtime فقط (اللي Vercel بيثبته)
├── requirements-dev.txt        # + yfinance/openpyxl/xlrd (pipeline محلي/CI فقط)
├── vercel.json                 # إعدادات النشر على Vercel (zero-config)
├── .gitattributes               # توحيد line endings (LF) في كل الريبو
│
├── feature_utils.py            # معادلات الـ features (مشتركة بين التدريب والتوقع اللحظي)
│
├── data_pipeline/
│   ├── update_gold_project.py # تحميل الداتا الخام (Yahoo/FRED/GPR/GoldRate24)
│   └── build_dataset.py       # دمج كل المصادر + feature engineering
│
├── model/
│   ├── train_model.py         # تدريب الموديل + تقييم walk-forward (يُشغّل يدويًا أو عن طريق CI)
│   ├── predictor.py           # منطق التوقع + التوقع اللحظي (بيستخدمه app.py وقت الطلب)
│   ├── live_price.py          # جلب سعر الذهب اللحظي من GoldRate24
│   └── live_quotes.py         # جلب أسعار باقي المؤشرات اللحظية بالتوازي (Yahoo)
│
├── models/                    # الموديلات المدرّبة (.joblib) + metadata + تقرير التقييم
├── data/
│   ├── raw/                   # كل CSV زي ما هو من المصدر - من غير تعديل
│   └── processed/             # merged_dataset.csv (تدريب فقط، مش متتبّع في git) + price_history.csv + recent_market_window.csv (يستخدمهم الموقع)
│
├── templates/index.html
├── public/css/style.css        # (كان static/ - Vercel بيقدّم public/ فقط، مش static/)
├── public/js/main.js
├── public/vendor/chart.umd.min.js
│
└── .github/workflows/daily_update.yml   # تحديث + إعادة تدريب تلقائي يوميًا (بيوقف لو مصدر أساسي فشل)
```

## طريقة عمل التوقع (مهم تفهمها)

الموديل مدرّب بفكرة **"الأفق الزمني كـ feature"**: بدل ما نعمل موديل منفصل
لكل مدة (بكرة، أسبوع، سنة...) اللي كان هيبقى صعب الصيانة، عندنا موديل واحد
بياخد "بعد كام يوم عايز تتوقع" (`horizon`) كـ input زي أي feature تاني.
كده تقدر تتوقع أي مدى - يوم، أسبوع، أو 365 يوم - من نفس الموديل.

كمان بندرّب 3 نسخ من الموديل (quantile 10% / 50% / 90%) عشان نديك **مدى
ثقة** حوالين التوقع مش رقم جامد واحد - وده مهم جدًا خصوصًا في التوقعات
البعيدة (سنة قدام) اللي طبيعي يكون فيها عدم يقين كبير.

**تنويه مهم / محدودية معروفة:** التوقعات المستقبلية (بكرة، الأسبوع، السنة)
بتستخدم آخر حالة معروفة لباقي المؤشرات (الدولار، البترول، إلخ) كنقطة انطلاق،
لأننا مش عندنا طريقة نعرف بيها قيمهم الفعلية في المستقبل. ده افتراض شائع
في نماذج التوقع قصيرة/متوسطة المدى، بس معناه إن التوقعات بتاعة سنة قدام
تحديدًا تقديرية بشكل أكبر - عشان كده الموديل بيديك مدى (low-high) مش رقم
واحد بس.

## تشغيل المشروع محليًا

```bash
# للتشغيل المحلي بس (السيرفر + إعادة التدريب): يشمل requirements.txt تلقائي
pip install -r requirements-dev.txt

# 1) (اختياري) تحديث الداتا الخام لآخر سعر متاح
python data_pipeline/update_gold_project.py

# 2) دمج الداتا + بناء الـ features
python data_pipeline/build_dataset.py

# 3) تدريب الموديل (+ تقييم walk-forward تلقائي)
python model/train_model.py

# 4) تشغيل السيرفر
python app.py
# افتح http://127.0.0.1:5000
```

ملاحظة: `requirements.txt` (من غير `-dev`) هو الملف اللي **Vercel بيثبته
وقت الـ deployment** - مقصود يفضل صغير (Flask + pandas/numpy/scikit-learn/
joblib بس)، من غير `yfinance`/`openpyxl`/`xlrd` اللي مستخدمين في
الـ pipeline المحلي/CI بس ومش محتاجهم التطبيق المنشور خالص.

الموديلات المدرّبة والداتا المدموجة موجودين جاهزين في الـ zip، فمش لازم
تعمل الخطوات 1-3 أول مرة - ينفع تشغّل `python app.py` على طول.

## إضافة GPR / EPU (مؤشرات الأحداث السياسية والاقتصادية)

`update_gold_project.py` بقى فيه كود تحميل جاهز لـ:
- **EPU (Economic Policy Uncertainty)** - من FRED، مباشر (سلسلة `USEPUINDXD` يومية و`GEPUCURRENT` شهرية عالمية)
- **GPR (Geopolitical Risk Index)** - من موقع الباحثين اللي ابتكروه (Caldara & Iacoviello, Federal Reserve)

الملفين دول محتاجين إنترنت مباشر وقت التشغيل (السكريبت بيتصل بـ fred.stlouisfed.org
و matteoiacoviello.com)، فشغّلهم من جهازك أو خلي الـ GitHub Action يشغلهم
(موجودة أوتوماتيك). GPR بالذات ملفه بصيغة `.xls` قديمة ومحتاج مكتبة
`xlrd` (موجودة في `requirements-dev.txt`) - لو التنزيل فشل، السكريبت
بيطبع تحذير واضح ويكمل باقي البيانات عادي (GPR مصنّف "optional" لأنه
مش داخل في الـ features بعد). أما EPU/Yahoo/GoldRate24 فمصنّفين
"required" - لو أي واحد فيهم فشل، السكريبت بيوقف بالكامل (exit code 1)
قبل ما يعمل build/train/commit على بيانات ناقصة.

بعد ما الملفات تتحمّل في `data/raw/06_EVENTS/`، شغّل `build_dataset.py`
تاني - هو بيكتشف أي ملف موجود في الفولدر ده تلقائيًا ويضمّه للـ features
من غير أي تعديل تاني في الكود.

## التحديث التلقائي (Sync مع السوق)

- **السعر الحالي في الصفحة**: بيتجاب مباشرة (live) من نفس API بتاعك
  (elomla.com) وقت ما حد يفتح الصفحة - مفيش تخزين وسيط.
- **باقي المؤشرات (الدولار، DXY، البترول، إلخ)**: بتتجاب لحظيًا هي كمان
  وقت كل طلب (من Yahoo Finance عن طريق `model/live_quotes.py`، السبعة
  مصادر بيتجابوا بالتوازي مش واحد ورا التاني) ويتم إعادة حساب الـ
  features بتاعت "النهاردة" على أساسها فورًا - مش بننتظر الـ batch
  اليومي عشان الموديل "يحس" بتحرك السوق. لو مؤشر معين فشل جلبه لحظيًا،
  بيرجع تلقائي لآخر قيمة معروفة من غير ما يبوّظ الطلب (شوف
  `PROJECT_EXPLANATION.md` قسم 7 للتفاصيل الكاملة).
- **بيانات التدريب + الموديل**: بيتحدثوا مرة كل يوم أوتوماتيك عن طريق
  `.github/workflows/daily_update.yml` (GitHub Actions) اللي بيشغّل
  السكريبتات التلاتة بالترتيب وبيعمل commit للنتائج (لو أي مصدر بيانات
  أساسي فشل، الـ workflow بيوقف بالكامل بدل ما يكمل على بيانات ناقصة)،
  وده بيخلي Vercel يعمل redeploy تلقائي كل مرة. من غير احتياج لأي
  database منفصل. تقدر كمان تشغّله يدويًا من تبويب Actions في GitHub
  (زرار "Run workflow").

  **تنويه مهم:** GitHub بيوقف تشغيل أي scheduled workflow تلقائيًا لو
  الريبو فضل من غير أي نشاط (commit) لمدة **60 يوم متواصلة**. الـ Action
  نفسه بيعمل commit يومي طول ما فيه بيانات جديدة، فده مش هيحصل عادةً -
  بس لو لأي سبب وقفت البيانات تتحدث فترة طويلة، افتح تبويب **Actions**
  وتأكد إن الـ workflow لسه "Active" ومش متعطل.

## الرفع على GitHub ثم Vercel

### لو الريبو والـ Vercel project موجودين بالفعل (حالتك الحالية)

الريبو والـ Vercel deployment أصلًا موجودين ومربوطين ببعض، فمش محتاج تعمل
`git init`/`New Project` تاني - كل اللي محتاجه هو تعمل commit + push
للتعديلات دي، و**Vercel هيسمع تلقائي على طول** لأنه مربوط بالـ GitHub repo
بتاعك (Git Integration)، وأي push جديد لـ `main` بيعمل deployment جديد
أوتوماتيك من غير أي تدخل يدوي أو زرار تضغطه على Vercel:

```bash
# فك الضغط عن الملف اللي هبعتهولك فوق نسختك المحلية من الريبو (استبدل كل حاجة)
git add -A
git commit -m "fix: serve static assets from public/, fix GPR download, workflow fail-fast, staircase in forecasts, model evaluation"
git push
```

بعد الـ push مباشرة، روح تبويب **Deployments** في Vercel dashboard - هتلاقي
deployment جديد بدأ لوحده (Building) في خلال ثواني. لما يخلص (Ready)،
افتح الرابط وجرّب بالترتيب ده بالظبط (نفس الفحص اللي ChatGPT اقترحه، وهو
صح):

1. `https://YOUR-DOMAIN.vercel.app/` → الصفحة تظهر **بالتصميم كامل** (لو
   لسه من غير CSS، امسح الـ deployment القديم من "..." → Redeploy، أو
   تأكد إن الـ push فعلًا وصل للـ branch اللي Vercel بيتابعه).
2. `https://YOUR-DOMAIN.vercel.app/css/style.css` → لازم يرجع محتوى CSS
   فعلي (مش 404).
3. `https://YOUR-DOMAIN.vercel.app/api/health` → `{"status": "ok"}`.

### تأكيد إعدادات Vercel dashboard (مرة واحدة بس)

افتح Project Settings → General وتأكد إن:
- **Framework Preset**: Other
- **Root Directory**: `.` (الجذر)
- **Build Command / Output Directory**: فاضيين (خليهم Default)
- **Install Command**: Default (`pip install -r requirements.txt` تلقائي)

### مهم: `merged_dataset.csv` اللي كان متتبّع في git قبل كده

لو كنت عملت commit للملف ده قبل ما تضيفه لـ `.gitignore` (غالبًا الحالة
عندك، بما إنه كان بيتعمله commit يوميًا)، إضافته لـ `.gitignore` دلوقتي
**مش هتوقف تتبعه تلقائيًا** - git هيفضل يتابع أي تغيير فيه لحد ما تعمل:

```bash
git rm --cached data/processed/merged_dataset.csv
git commit -m "chore: stop tracking merged_dataset.csv (regenerable, was bloating repo history)"
git push
```

الملف هيفضل موجود على جهازك عادي (`--cached` بس بيشيله من git، مش من
الديسك)، وهيتم توليده تاني تلقائيًا كل مرة الـ workflow يشتغل.

### أول مرة (لو مفيش ريبو أصلًا)

```bash
git init
git add .
git commit -m "Initial commit: gold price predictor"
git branch -M main
git remote add origin <رابط الريبو بتاعك>
git push -u origin main
```

بعدها من [vercel.com](https://vercel.com): New Project → Import من الريبو
ده → نفس إعدادات الـ Dashboard فوق → Deploy.

## ليه الـ refresh مش بيعيد تدريب الموديل (وده قرار مقصود)

فكرة إن زرار الـ refresh "يجمع بيانات جديدة من الـ API وبعدين يدرّب
الموديلات من تاني على طول" اتفحصت، ومش هتتنفذ زي ما هي، للأسباب دي
(كل واحد فيهم اتقاس فعليًا مش بس نظريًا):

1. **الوقت**: تدريب الـ 3 موديلات فعليًا بياخد **حوالي 3 دقايق** (على
   معالج واحد - قياس فعلي، مش تقدير). حد مش هيقف يستنى 3 دقايق كل ما
   يدوس refresh على موقع.
2. **Vercel serverless functions** ليها حد أقصى للوقت (`maxDuration` -
   حاطينه 30 ثانية في `vercel.json`)، ومفيهاش قرص دائم تقدر تحفظ عليه
   موديل جديد بشكل يفضل موجود للطلبات الجاية - أي حاجة تتحفظ جوه طلب
   HTTP واحد بتروح لما الطلب يخلص. يعني حتى لو التدريب خلص في وقت
   قصير، مفيش مكان تحفظ فيه النتيجة بشكل دائم من جوه الموقع نفسه.
3. **يوم واحد زيادة من البيانات مش هيغيّر حاجة فعليًا**: الموديل مدرّب
   على 273,699 صف (بعد التوسيع لكل الآفاق) - يوم واحد إضافي هو جزء من
   آلاف، مفيش أي فايدة إحصائية حقيقية من إعادة التدريب الكامل عشانه.

**البديل الصح (وهو أصلًا اللي شغال ومُختبر فعليًا)**:
- **Refresh = جلب بيانات لحظية + inference بس** (مش تدريب). لما تدوس
  refresh، السعر الحالي وكل المؤشرات التانية (دولار، DXY، بترول...)
  بتتجاب لحظيًا من الـ API فورًا، ويتم حساب الـ features بتاعت
  "النهاردة" منها، وبعدين الموديل **الجاهز أصلًا** (اللي اتدرب قبل كده)
  بيدي توقع جديد بناءً على البيانات الجديدة دي. ده سريع (ثواني، مش
  دقايق) وده اللي فعليًا بيخلي الأرقام على الموقع "حية".
- **إعادة التدريب الفعلية** (اللي بتضيف بيانات جديدة لمعرفة الموديل
  نفسها) بتحصل **مرة كل يوم أوتوماتيك** عن طريق GitHub Action (شوف
  `.github/workflows/daily_update.yml`) - ده التوقيت المنطقي لأن سعر
  الذهب بياخد قيمة جديدة "رسمية" مرة واحدة في اليوم أصلًا مش أكتر.

**اتأكد فعليًا** (مش افتراض): جربت سيناريوهين بسعر لحظي مختلف تمامًا
(6330 مقابل 6800 جنيه)، وأكدت إن السعر المعروض وكل التوقعات
(الغد/الأسبوع/السنة) بتتغير فعلًا مع كل قيمة - يعني الـ "live" حقيقي
مش شكلي. وأكدت كمان إن الـ cache (20 ثانية) بيمنع تكرار الطلب على
الـ API لو حد دوس refresh أكتر من مرة بسرعة، وإن زرار refresh
(`force=1`) بيتخطى الـ cache ده ويجيب بيانات جديدة فعلًا في كل مرة.

## ملاحظات تقنية

- الموديل: `HistGradientBoostingRegressor` من scikit-learn (مش LightGBM/XGBoost)
  عشان يفضل الـ deployment على Vercel بسيط وخفيف من غير أي مكتبات native
  محتاجة compilation.
- **Chart.js مُضمّن جوه المشروع** (`public/vendor/chart.umd.min.js`) بدل ما يتجاب
  من CDN خارجي وقت التشغيل - ده بيضمن إن الـ charts تشتغل دايمًا (محليًا
  أو على Vercel) حتى لو في مشكلة شبكة أو إضافة متصفح بتمنع الاتصال
  بمواقع الـ CDN زي cdnjs.cloudflare.com.
- كل الملفات الخام في `data/raw/` بتفضل زي ما هي - من غير تعديل - بنفس
  فلسفة سكريبتك الأصلي، والمعالجة كلها بتحصل في `build_dataset.py`.
- `data/processed/merged_dataset.csv` (~19MB) مستخدم في التدريب بس، مش
  بيترفع لـ Vercel (مستبعد في `.vercelignore`)، **ومش متتبّع في git كمان**
  (`.gitignore`) لأنه قابل لإعادة البناء 100% من `data/raw/` - الموقع
  بيستخدم بديل خفيف `price_history.csv` لعرض charts السعر الفعلي.
- **دقة الموديل - بصراحة**: `models/meta.json["evaluation"]` فيه تقييم
  walk-forward حقيقي (held-out) مقارنةً بـ baseline بسيط لكل الآفاق (1،
  7، 30، 90، 180، 365 يوم). بعد إضافة معايرة (`blend_alpha` +
  `band_scale` - شوف `train_model.py`)، الموديل بقى **يتفوق على الـ
  baseline في كل الآفاق** (كان قبل المعايرة بيخسر في 1-180 يوم). لسه
  فيه نقطة صادقة لازم تعرفها: تغطية الـ uncertainty band (النطاق
  المعروض) في آفاق 90+ يوم لسه أقل من الـ 80% المستهدفة - يعني النطاق
  المعروض متفائل شوية عن المخاطرة الحقيقية في التوقعات البعيدة. شوف
  `PROJECT_EXPLANATION.md` للتفاصيل والأرقام كاملة ولسبب القرار ده.

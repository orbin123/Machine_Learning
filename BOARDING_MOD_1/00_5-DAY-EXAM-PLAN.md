# 🎯 5-Day Exam Battle Plan — Boarding Mod 1

> **Reality check:** You cannot *master* all 11 subjects in 5 days. You **can** revise the
> high-yield 20% that shows up in 80% of questions, and be able to *talk* confidently about
> the rest. This plan is built for **coverage + confidence**, not perfection.

---

## 📋 How to use this file

- Each day mixes **subject types** (a bit of coding, a bit of math, a bit of ML) so your brain
  never gets bored or fatigued on one thing.
- Every day has three tiers:
  - 🔴 **MUST complete** — non-negotiable. If you only do this, you'll survive.
  - 🟡 **SHOULD do** — do it if MUST is done and time remains.
  - 🟢 **NICE to have** — stretch goals / buffer if you're ahead.
- Tick the `[ ]` boxes as you go. In Obsidian these render as clickable checkboxes.
- Each day is **one division**. If you get extra time later, split any division into two.

### 🧠 Study tips that apply every single day
- [ ] **Active recall > re-reading.** After each topic, close the file and explain it out loud (or write it from memory). If you can't, you don't know it yet.
- [ ] **Theory → then immediately practice.** Read `theory.md`, then open `practical.md` and *type* the code yourself. Don't copy-paste.
- [ ] **Keep a "One-Liner Sheet"** (a single new note). For every topic write ONE sentence you'd say in an interview. This becomes your Day-5 revision.
- [ ] **Pomodoro:** 50 min focus + 10 min break. Aim for 6–8 blocks/day.
- [ ] **Prioritise interview checklists.** Most subject files have an "Interview Preparation Checklist" at the bottom — those are literally the exam questions. Start there when short on time.

---

## 🗺️ The 5 Divisions at a glance

| Day | Theme | Subjects mixed |
|-----|-------|----------------|
| **1** | Programming foundations | Python 🐍 + Data Structures 🧱 + Maths (Linear Algebra) ➗ |
| **2** | Data handling & ML intro | NumPy/Pandas 📊 + Machine Learning basics 🤖 + Maths (Calculus + Stats) ➗ |
| **3** | Predictive modelling & data querying | Supervised ML 📈 + SQL 🗄️ |
| **4** | Patterns & the web | Unsupervised ML 🔍 + Web Dev (Django/DRF) 🌐 |
| **5** | Lock it in | Statistics finish 📐 + Project Presentation 🚀 + Full revision & mock 🔁 |

---

# 📅 DAY 1 — Programming Foundations
### 🐍 Python + 🧱 Data Structures + ➗ Linear Algebra
> **Why together:** Python is your tool for everything else, DSA is pure logic, and Linear Algebra
> is short & mechanical. A coding-heavy day with a light math finish.

**Suggested order:** Python (morning, freshest brain) → DSA (afternoon) → Linear Algebra (evening, low energy OK).

### 🔴 MUST complete
- [ ] **Python core** — `Python/theory.md` §Fundamentals + OOP
  - [ ] Data types, mutability (list vs tuple vs set vs dict — know when to use each)
  - [ ] The **4 pillars of OOP** (encapsulation, abstraction, inheritance, polymorphism) — 1 example each
  - [ ] `__init__`, `__str__`, `__repr__`, and why dunder methods matter
  - [ ] Exception handling: `try / except / else / finally`
- [ ] **DSA essentials** — `Data Structures/01theory.md`
  - [ ] Big-O: time & space complexity of array, linked list, hash table, BST operations (make a tiny table)
  - [ ] Arrays vs Linked Lists (when to use which)
  - [ ] Stacks vs Queues (LIFO vs FIFO) + one real-world use each
  - [ ] Binary Search (be able to write it from memory)
- [ ] **Linear Algebra** — `Maths/theory.md` Week 1
  - [ ] Scalars / vectors / matrices + basic operations
  - [ ] Matrix multiplication rule (why dimensions must match)
  - [ ] What eigenvalues & eigenvectors *are* (conceptually) and their link to PCA

### 🟡 SHOULD do
- [ ] Python: list/dict comprehensions, `map`/`filter`/`lambda`, decorators (just the concept + one example)
- [ ] Python: generators & `yield` (lazy evaluation — one-line why it saves memory)
- [ ] DSA: implement a Linked List (insert at head/tail, delete) — from `02practical.md`
- [ ] DSA: Recursion — write factorial + fibonacci, identify base case
- [ ] Solve **3 easy coding problems** (arrays/strings) on any platform

### 🟢 NICE to have
- [ ] Python: iterators vs iterables, context managers (`with`), type hints
- [ ] DSA: Tree traversals (inorder/preorder/postorder), BFS vs DFS idea
- [ ] Linear Algebra: matrix inverse & determinant (2×2 by hand)

### ✅ End-of-Day 1 you should be able to say:
> "I can write clean Python with OOP, explain Big-O of the core data structures, code binary search, and explain what a matrix and an eigenvector are."

---

# 📅 DAY 2 — Data Handling & ML Intro
### 📊 NumPy/Pandas + 🤖 Machine Learning basics + ➗ Calculus & Stats
> **Why together:** NumPy/Pandas is *how* you do ML, ML fundamentals is the *what/why*, and
> Calculus (gradients) + Stats is the *math underneath*. This is the conceptual heart of the whole exam.

**Suggested order:** NumPy/Pandas (hands-on, morning) → ML fundamentals (afternoon) → Calculus + Stats (evening).

### 🔴 MUST complete
- [ ] **NumPy & Pandas** — `Numpy & Pandas/theory.md`
  - [ ] NumPy: arrays, indexing/slicing, broadcasting (know what broadcasting means)
  - [ ] Pandas: `DataFrame`, selecting/filtering rows & columns, `groupby`, handling missing values (`fillna`/`dropna`)
  - [ ] Reading a CSV → basic EDA (`.head()`, `.info()`, `.describe()`)
- [ ] **Machine Learning fundamentals** — `Machine Learning/theory.md`
  - [ ] Supervised vs Unsupervised vs Reinforcement (definition + 2 examples each)
  - [ ] The **ML workflow** (data → preprocess → train → evaluate → deploy) — memorise the pipeline
  - [ ] Data preprocessing: handling missing values, outliers (Z-score & IQR), scaling (Normalization vs Standardization)
  - [ ] Encoding categoricals: One-Hot vs Label encoding
- [ ] **Calculus + Stats core** — `Maths/theory.md` Weeks 2–3
  - [ ] Derivative = rate of change; **gradient descent** in plain words (how a model "learns")
  - [ ] Mean / median / mode, variance, standard deviation
  - [ ] Normal distribution + what skewness means

### 🟡 SHOULD do
- [ ] Pandas: `merge`/`join`, aggregation, `apply()`
- [ ] Matplotlib/Seaborn: line, bar, histogram, box plot, **heatmap for correlation**
- [ ] ML: Feature selection (filter/wrapper/embedded — one line each)
- [ ] Stats: Correlation vs Covariance (what's the difference?)
- [ ] Stats: Probability basics + Bayes' theorem (the formula + intuition)

### 🟢 NICE to have
- [ ] Calculus: chain rule, partial derivatives (why needed for backprop/gradient)
- [ ] Stats: Hypothesis testing — H₀ vs H₁, p-value, Type I vs Type II error
- [ ] Stats: Z-test / T-test / Chi-square / ANOVA (just *when* to use each)

### ✅ End-of-Day 2 you should be able to say:
> "I can wrangle a dataset in Pandas, explain the full ML pipeline, describe how gradient descent minimises a loss, and interpret basic statistics of a dataset."

---

# 📅 DAY 3 — Predictive Modelling & Data Querying
### 📈 Supervised ML + 🗄️ SQL
> **Why together:** Supervised ML is the biggest, highest-value topic — pair it with SQL, a
> different muscle (query logic), so you alternate between conceptual ML and hands-on querying.

**Suggested order:** Supervised ML (morning + early afternoon, it's heavy) → SQL (late afternoon + evening, practice-driven).

### 🔴 MUST complete
- [ ] **Supervised ML — Regression** — `Supervised ML/theory.md` Week 1
  - [ ] Linear Regression: the idea, cost function (MSE), what R² tells you
  - [ ] Regularization: **Ridge (L2) vs Lasso (L1)** — and why Lasso does feature selection
  - [ ] Overfitting vs Underfitting + Bias-Variance tradeoff (the #1 interview question)
- [ ] **Supervised ML — Classification** — Week 2
  - [ ] Logistic Regression: sigmoid, decision boundary
  - [ ] **Confusion matrix** → Precision, Recall, F1 (know the formulas & when each matters)
  - [ ] Decision Trees (Gini vs Entropy) + KNN (one line each)
- [ ] **SQL core** — `SQL/theory.md` Weeks 1–2
  - [ ] CRUD (`INSERT`/`SELECT`/`UPDATE`/`DELETE`)
  - [ ] `WHERE`, `ORDER BY`, `GROUP BY` + `HAVING` (WHERE vs HAVING!)
  - [ ] Aggregates: `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`
  - [ ] **JOINs**: INNER vs LEFT vs RIGHT vs FULL (draw them!)

### 🟡 SHOULD do
- [ ] Supervised ML: SVM (hyperplane, kernel trick — concept), ROC-AUC curve
- [ ] Supervised ML: **Ensembles** — Bagging vs Boosting, Random Forest vs XGBoost (when/why)
- [ ] Supervised ML: Cross-validation (K-Fold), GridSearchCV for tuning
- [ ] SQL: Subqueries + CTEs (`WITH`), `DISTINCT`, `LIMIT`/`OFFSET`
- [ ] SQL: Write **5 practice queries** from `SQL/practical.md`

### 🟢 NICE to have
- [ ] Supervised ML: Handling imbalanced data (SMOTE, class weights)
- [ ] SQL: **Window functions** (`ROW_NUMBER`, `RANK`, `LAG`/`LEAD`, `PARTITION BY`)
- [ ] SQL: Indexing (B-Tree), ACID & transactions, normalization (1NF→3NF)

### ✅ End-of-Day 3 you should be able to say:
> "I can explain regression vs classification, pick the right evaluation metric, describe bias-variance, and write multi-table SQL joins with aggregation."

---

# 📅 DAY 4 — Patterns & The Web
### 🔍 Unsupervised ML + 🌐 Web Dev (Django/DRF)
> **Why together:** Unsupervised ML is smaller and conceptual; Web Dev is practical & systems-y.
> One "data science" block, one "software engineering" block — nicely balanced.

**Suggested order:** Unsupervised ML (morning) → Web Dev (afternoon + evening).

### 🔴 MUST complete
- [ ] **Unsupervised ML** — `Unsupervised ML/theory.md`
  - [ ] Clustering vs Dimensionality Reduction (the two big families)
  - [ ] **K-Means**: the 5 algorithm steps + choosing K (Elbow method, Silhouette score)
  - [ ] **PCA**: what it does (reduce dimensions, keep variance) + its link to eigenvectors
  - [ ] Why feature scaling matters *before* clustering
- [ ] **Django fundamentals** — `Web Dev/theory.md` Week 1
  - [ ] **MVT architecture** (vs MVC) — explain each layer
  - [ ] Models & the ORM (CRUD, relationships: 1-1, 1-many, many-many), migrations
  - [ ] Views (FBV vs CBV) + URL routing
- [ ] **REST / DRF basics** — Week 3
  - [ ] What REST is + HTTP methods (GET/POST/PUT/PATCH/DELETE) + **PUT vs PATCH**
  - [ ] Common status codes (200/201/400/401/403/404/500)
  - [ ] Serializers (what they do) + Authentication (Session vs Token vs **JWT**)

### 🟡 SHOULD do
- [ ] Unsupervised: Hierarchical clustering (dendrogram) & DBSCAN (core/border/noise points)
- [ ] Unsupervised: Distance metrics (Euclidean, Manhattan, Cosine — when each)
- [ ] Unsupervised: PCA vs t-SNE vs SVD (compare)
- [ ] Django: Authentication/permissions, Forms vs ModelForms vs DRF Serializers
- [ ] Django: `select_related()` vs `prefetch_related()` (the N+1 problem)

### 🟢 NICE to have
- [ ] Unsupervised: Cluster evaluation (Davies-Bouldin, Calinski-Harabasz)
- [ ] Django: Middleware & request lifecycle, Signals, Custom User model
- [ ] Django: Deployment story (Gunicorn + Nginx, static vs media, AWS EC2/Beanstalk)
- [ ] Security: CSRF, XSS, SQL injection prevention

### ✅ End-of-Day 4 you should be able to say:
> "I can explain K-Means and PCA, describe Django's MVT flow and the ORM, and design a REST API with JWT auth."

---

# 📅 DAY 5 — Lock It In
### 📐 Finish Stats + 🚀 Project Presentation + 🔁 Full Revision & Mock
> **Why:** Day 5 is NOT for new material. It's for closing gaps, nailing your project pitch
> (you built LingosAI — this is your strongest card), and rapid-fire self-testing.

**Suggested order:** Weak-spot patching (morning) → Project prep (midday) → Mock/rapid revision (afternoon/evening) → light review + sleep.

### 🔴 MUST complete
- [ ] **Patch your weak spots** — go through your "One-Liner Sheet" and re-study anything you can't recall
- [ ] **Project Presentation** — `Project Presentation/Project Presentation.md`
  - [ ] Be able to give a **60-second pitch** of LingosAI (problem → solution → impact)
  - [ ] Explain the **agent architecture** (Teacher, Planner, Task Gen, Evaluator, Feedback)
  - [ ] Explain **RAG** (why Pinecone / vector DB — recall past mistakes) in plain English
  - [ ] Explain the **deterministic scoring engine** (why *no* LLM → reproducible/explainable)
  - [ ] Draw/describe the **architecture**: Next.js → FastAPI (Routes→Service→Repository) → PostgreSQL, Redis, S3
  - [ ] Know your **CI/CD** story (GitHub Actions → ECR → ECS rolling deploy → smoke test → rollback)
- [ ] **Rapid revision** — read every subject's **Interview Preparation Checklist** out loud and answer each

### 🟡 SHOULD do
- [ ] **Mock exam:** self-test 3–5 questions per subject (no notes), then grade yourself
- [ ] Stats finish: Hypothesis testing + which statistical test to use when (from Day 2 backlog)
- [ ] Re-write the trickiest concepts from scratch on paper (bias-variance, JOINs, K-Means, gradient descent)

### 🟢 NICE to have
- [ ] Anticipate cross-topic questions ("How does PCA use eigenvectors?", "How would you deploy an ML model with Django?")
- [ ] Prepare 2–3 smart questions to ask *them* (shows engagement)
- [ ] Get a full night's sleep — a rested brain beats one more hour of cramming

### ✅ End-of-Day 5 you should be able to say:
> "I can pitch my project end-to-end, defend every architectural choice, and give a one-line answer to any interview-checklist question across all subjects."

---

## 🧭 Emergency triage (if you fall badly behind)
If a day runs over, **don't** try to finish everything — protect these absolute-highest-yield items:
1. **Bias-Variance / Overfitting** (Day 3) — asked almost everywhere
2. **SQL JOINs + GROUP BY** (Day 3) — pure recall, easy marks
3. **Python OOP 4 pillars** (Day 1) — guaranteed question
4. **ML pipeline + preprocessing** (Day 2) — the backbone of every ML discussion
5. **Your project pitch** (Day 5) — the one thing nobody can out-prepare you on
6. **K-Means + PCA** (Day 4) — the two must-know unsupervised methods

## 📊 Progress tracker
- [ ] Day 1 complete
- [ ] Day 2 complete
- [ ] Day 3 complete
- [ ] Day 4 complete
- [ ] Day 5 complete

> You've got this. Coverage beats perfection. Tick boxes, move forward, don't spiral. 🚀

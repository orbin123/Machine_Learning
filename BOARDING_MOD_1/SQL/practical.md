# SQL — Practical & Coding Assessment Guide

> Hands-on SQL problems modeled on real coding rounds, practical lab exams, and take-home assessments. Each question gives difficulty, time, concepts tested, an approach, a production-quality solution, alternatives, variations, and follow-ups. For the underlying concepts, see [[BROTOTYPE/BOARDING MOD 1/SQL/theory]].

## How to Use This Guide

- All questions run against **one shared schema** (defined in **Part 1 — Sample Schema & Seed Data**). Load it once into PostgreSQL and every query below is runnable.
- Each question is structured as: **Difficulty / Estimated Time / Concepts Tested → Problem → Example Input → Example Output → Approach → SQL Implementation → Alternative Solution → Interview Variations → Common Follow-up Questions**.
- Practice by reading only the **Problem Statement** first, attempting it yourself, then checking the solution.

> Examples are **PostgreSQL-flavored**, with MySQL differences noted where relevant.

---

## Table of Contents

**Part 1 — Schema & CRUD**
- Sample Schema & Seed Data
- CRUD — INSERT
- CRUD — SELECT, Filtering & Sorting
- CRUD — Pagination (LIMIT / OFFSET)
- CRUD — UPDATE
- CRUD — DELETE / TRUNCATE

**Part 2 — Functions, Aggregation & Grouping**
- String / Numeric / Date Functions & NULL Handling
- Aggregate Functions (COUNT / SUM / AVG / MIN / MAX)
- GROUP BY
- HAVING

**Part 3 — Joins, Subqueries & CTEs**
- Joins (INNER / LEFT / RIGHT / FULL / SELF / CROSS)
- Subqueries (single-row, multi-row, correlated, EXISTS)
- Common Table Expressions (CTEs) & Recursive CTEs

**Part 4 — Window Functions, Views & Optimization**
- Window Functions (ranking, LEAD / LAG, running totals)
- Views & Materialized Views
- Indexing, EXPLAIN & Query Optimization
- Quick Reference — Cheatsheet

**Part 5 — Transactions, Procedures & Design**
- Transactions & Concurrency (BEGIN/COMMIT/ROLLBACK/SAVEPOINT, isolation, locking)
- Stored Procedures & Functions
- Normalization (Practical)
- Database Design & ER Modeling (Practical)

---


---

# Sample Schema & Seed Data

This guide is built around a single, self-contained **e-commerce + company** dataset. Every practical question in this document runs against the tables defined below. Create the schema once (PostgreSQL flavor), seed it, and you can execute every solution end-to-end.

> **How to use this section:** Run the whole block once in a scratch database (`createdb sql_prep && psql sql_prep -f schema.sql`). The `DROP TABLE ... CASCADE` prelude makes the script idempotent, so you can re-run it whenever you want a clean slate between exercises.

```sql
-- ============================================================
--  RESET (idempotent) — drop in dependency-safe order
-- ============================================================
DROP TABLE IF EXISTS order_items CASCADE;
DROP TABLE IF EXISTS orders      CASCADE;
DROP TABLE IF EXISTS products    CASCADE;
DROP TABLE IF EXISTS customers   CASCADE;
DROP TABLE IF EXISTS employees   CASCADE;
DROP TABLE IF EXISTS departments CASCADE;

-- ============================================================
--  DEPARTMENTS
-- ============================================================
CREATE TABLE departments (
    id       INTEGER      PRIMARY KEY,
    name     VARCHAR(60)  NOT NULL UNIQUE,
    location VARCHAR(60)  NOT NULL
);

-- ============================================================
--  EMPLOYEES  (self-referencing manager_id, FK to departments)
-- ============================================================
CREATE TABLE employees (
    id            INTEGER      PRIMARY KEY,
    name          VARCHAR(80)  NOT NULL,
    department_id INTEGER      REFERENCES departments(id),
    manager_id    INTEGER      REFERENCES employees(id),
    salary        NUMERIC(10,2) NOT NULL CHECK (salary >= 0),
    hire_date     DATE         NOT NULL,
    email         VARCHAR(120) NOT NULL UNIQUE
);

-- ============================================================
--  CUSTOMERS
-- ============================================================
CREATE TABLE customers (
    id         INTEGER      PRIMARY KEY,
    name       VARCHAR(80)  NOT NULL,
    email      VARCHAR(120) NOT NULL UNIQUE,
    city       VARCHAR(60),
    created_at TIMESTAMP    NOT NULL DEFAULT now()
);

-- ============================================================
--  PRODUCTS
-- ============================================================
CREATE TABLE products (
    id       INTEGER       PRIMARY KEY,
    name     VARCHAR(100)  NOT NULL,
    category VARCHAR(50)   NOT NULL,
    price    NUMERIC(10,2) NOT NULL CHECK (price >= 0),
    stock    INTEGER       NOT NULL DEFAULT 0 CHECK (stock >= 0)
);

-- ============================================================
--  ORDERS
-- ============================================================
CREATE TABLE orders (
    id          INTEGER       PRIMARY KEY,
    customer_id INTEGER       NOT NULL REFERENCES customers(id),
    order_date  DATE          NOT NULL,
    status      VARCHAR(20)   NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','paid','shipped','delivered','cancelled')),
    total       NUMERIC(12,2) NOT NULL DEFAULT 0 CHECK (total >= 0)
);

-- ============================================================
--  ORDER_ITEMS  (line items; FK to orders + products)
-- ============================================================
CREATE TABLE order_items (
    id         INTEGER       PRIMARY KEY,
    order_id   INTEGER       NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    product_id INTEGER       NOT NULL REFERENCES products(id),
    quantity   INTEGER       NOT NULL CHECK (quantity > 0),
    unit_price NUMERIC(10,2) NOT NULL CHECK (unit_price >= 0)
);

-- ============================================================
--  SEED DATA
-- ============================================================
INSERT INTO departments (id, name, location) VALUES
    (1, 'Engineering', 'San Francisco'),
    (2, 'Sales',       'New York'),
    (3, 'Marketing',   'Austin'),
    (4, 'Finance',     'Chicago'),
    (5, 'Support',     'Remote');

-- managers are inserted first (manager_id NULL), then reports
INSERT INTO employees (id, name, department_id, manager_id, salary, hire_date, email) VALUES
    (1,  'Alice Johnson',   1, NULL, 185000, '2015-03-01', 'alice.johnson@corp.com'),
    (2,  'Bob Smith',       2, NULL, 160000, '2016-07-15', 'bob.smith@corp.com'),
    (3,  'Carol Williams',  1, 1,    142000, '2018-01-20', 'carol.williams@corp.com'),
    (4,  'David Brown',     1, 1,    128000, '2019-06-10', 'david.brown@corp.com'),
    (5,  'Eve Davis',       2, 2,     98000, '2020-02-05', 'eve.davis@corp.com'),
    (6,  'Frank Miller',    3, NULL,  115000, '2017-11-30', 'frank.miller@corp.com'),
    (7,  'Grace Wilson',    3, 6,     87000, '2021-04-12', 'grace.wilson@corp.com'),
    (8,  'Henry Moore',     4, NULL,  120000, '2018-09-01', 'henry.moore@corp.com'),
    (9,  'Ivy Taylor',      1, 3,     99000, '2022-08-19', 'ivy.taylor@corp.com'),
    (10, 'Jack Anderson',   5, NULL,  72000, '2023-01-09', 'jack.anderson@corp.com'),
    (11, 'Karen Thomas',    5, 10,    68000, '2023-05-22', 'karen.thomas@corp.com'),
    (12, 'Leo Martinez',    2, 2,     91000, '2021-10-03', 'leo.martinez@corp.com');

INSERT INTO customers (id, name, email, city, created_at) VALUES
    (1, 'Nina Patel',     'nina.patel@example.com',   'Austin',        '2023-01-15 09:20:00'),
    (2, 'Omar Haddad',    'omar.haddad@example.com',  'New York',      '2023-02-02 14:05:00'),
    (3, 'Priya Nair',     'priya.nair@example.com',   'San Francisco', '2023-02-18 11:45:00'),
    (4, 'Quentin Roy',    'quentin.roy@example.com',  'Chicago',       '2023-03-10 18:30:00'),
    (5, 'Rosa Lopez',     'rosa.lopez@example.com',   'Austin',        '2023-04-01 08:15:00'),
    (6, 'Sam Turner',     'sam.turner@example.com',   'New York',      '2023-04-25 16:50:00'),
    (7, 'Tara Singh',     'tara.singh@example.com',    NULL,           '2023-05-30 12:00:00'),
    (8, 'Umar Farooq',    'umar.farooq@example.com',  'San Francisco', '2023-06-14 10:10:00');

INSERT INTO products (id, name, category, price, stock) VALUES
    (1, 'Wireless Mouse',        'Electronics', 24.99,  150),
    (2, 'Mechanical Keyboard',   'Electronics', 79.99,   60),
    (3, 'USB-C Hub',             'Electronics', 39.50,    0),
    (4, '27-inch Monitor',       'Electronics', 289.00,  25),
    (5, 'Office Chair',          'Furniture',   199.00,  40),
    (6, 'Standing Desk',         'Furniture',   449.00,  12),
    (7, 'Notebook (Pack of 3)',  'Stationery',  12.49,  500),
    (8, 'Gel Pens (Pack of 10)', 'Stationery',   8.99,  320),
    (9, 'Desk Lamp',             'Furniture',    34.95,  75),
    (10,'Webcam 1080p',          'Electronics',  59.00,  90);

INSERT INTO orders (id, customer_id, order_date, status, total) VALUES
    (1, 1, '2023-05-01', 'delivered', 104.98),
    (2, 1, '2023-06-12', 'shipped',   289.00),
    (3, 2, '2023-06-15', 'paid',      88.98),
    (4, 3, '2023-06-20', 'delivered', 647.99),
    (5, 4, '2023-07-02', 'pending',   34.95),
    (6, 5, '2023-07-08', 'cancelled', 199.00),
    (7, 2, '2023-07-19', 'delivered', 24.99),
    (8, 6, '2023-08-01', 'paid',      537.98),
    (9, 3, '2023-08-11', 'shipped',   12.49),
    (10,8, '2023-08-25', 'pending',   118.00);

INSERT INTO order_items (id, order_id, product_id, quantity, unit_price) VALUES
    (1,  1, 1, 2, 24.99),
    (2,  1, 8, 1,  8.99),   -- note: seeded totals are illustrative, not strictly summed
    (3,  2, 4, 1, 289.00),
    (4,  3, 2, 1, 79.99),
    (5,  3, 8, 1,  8.99),
    (6,  4, 6, 1, 449.00),
    (7,  4, 5, 1, 199.00),
    (8,  5, 9, 1, 34.95),
    (9,  6, 5, 1, 199.00),
    (10, 7, 1, 1, 24.99),
    (11, 8, 4, 1, 289.00),
    (12, 8, 5, 1, 199.00),
    (13, 8, 10,1, 59.00),
    (14, 9, 7, 1, 12.49),
    (15, 10,10,2, 59.00);
```

> **Note on the `id` columns:** they are declared as plain `INTEGER PRIMARY KEY` with explicit values so the seed data is deterministic and every example output below is reproducible. In a real schema you would use `GENERATED ALWAYS AS IDENTITY` (PostgreSQL) or `AUTO_INCREMENT` (MySQL) — several questions below explore exactly that.

## CRUD — INSERT

### Practical Question 1: Register a New Customer
- **Difficulty:** Easy
- **Estimated Time:** 4 min
- **Concepts Tested:** Basic `INSERT`, column list, `DEFAULT` values, `RETURNING`

**Problem Statement**
A new customer, *Wei Chen* (`wei.chen@example.com`) from *Seattle*, just signed up. Insert them into `customers`. Let the database stamp `created_at` automatically and return the newly assigned row.

**Example Input** — `customers` (relevant existing max id is 8):

| id | name | email | city |
|----|------|-------|------|
| 8 | Umar Farooq | umar.farooq@example.com | San Francisco |

**Example Output** — the `RETURNING` clause emits:

| id | name | email | city | created_at |
|----|------|-------|------|------------|
| 9 | Wei Chen | wei.chen@example.com | Seattle | 2026-07-09 10:00:00 |

**Approach**
1. Target only the columns you are supplying; omit `created_at` so its `DEFAULT now()` fires.
2. Supply an explicit `id` of 9 (our seed schema uses manual ids).
3. Use `RETURNING *` to confirm the insert without a second round-trip.

#### SQL Implementation
```sql
INSERT INTO customers (id, name, email, city)
VALUES (9, 'Wei Chen', 'wei.chen@example.com', 'Seattle')
RETURNING *;
```
- Naming the columns explicitly (rather than relying on positional `VALUES`) is the professional default: the statement keeps working if someone later reorders or adds columns.
- Because `created_at` is omitted, its `DEFAULT now()` supplies the timestamp — never hard-code "now" from the app layer when the DB can do it consistently.
- `RETURNING *` is a PostgreSQL feature that hands back the full inserted row (including defaults and identity values) in the same statement — invaluable when the DB generates the key.
- **Perf/index note:** the `email UNIQUE` constraint is backed by a unique index; the insert incurs one index maintenance write and will raise `23505` if the email already exists.

#### Alternative Solution
In a production schema you would not pass `id` at all — you'd declare it `GENERATED ALWAYS AS IDENTITY` and let the DB assign it:
```sql
-- assuming: id INTEGER GENERATED ALWAYS AS IDENTITY
INSERT INTO customers (name, email, city)
VALUES ('Wei Chen', 'wei.chen@example.com', 'Seattle')
RETURNING id;
```
This is canonical because manually managing surrogate keys invites race conditions. We pass explicit ids in this guide only to keep example outputs deterministic.

#### Interview Variations
1. Insert the customer **only if** the email does not already exist (upsert / `ON CONFLICT DO NOTHING`).
2. Insert and, in the same transaction, create their first (empty) order.
3. Return only the generated `id` instead of the whole row.

#### Common Follow-up Questions
- *What happens if you omit a `NOT NULL` column with no default?* The insert fails with `23502 not_null_violation`.
- *Why prefer `RETURNING` over `SELECT ... currval()`?* It is atomic, concurrency-safe, and one round-trip; `currval()` is session-scoped and easy to misuse.
- *Does MySQL support `RETURNING`?* Classic MySQL does not (use `LAST_INSERT_ID()`); MariaDB 10.5+ does.

### Practical Question 2: Bulk-Load Several Products
- **Difficulty:** Easy
- **Estimated Time:** 5 min
- **Concepts Tested:** Multi-row `INSERT`, single-statement batching, default columns

**Problem Statement**
The catalog team hands you three new products to add in one shot: a *Laptop Stand* (Furniture, 45.00, stock 30), a *HDMI Cable* (Electronics, 9.99, stock 200), and a *Whiteboard Marker* (Stationery, 3.50, stock 0). Insert all three in a single statement.

**Example Output** — `products` after insert (new rows):

| id | name | category | price | stock |
|----|------|----------|-------|-------|
| 11 | Laptop Stand | Furniture | 45.00 | 30 |
| 12 | HDMI Cable | Electronics | 9.99 | 200 |
| 13 | Whiteboard Marker | Stationery | 3.50 | 0 |

**Approach**
1. Use one `INSERT` with several comma-separated tuples — one server round-trip, one transaction.
2. Keep the column order consistent across every tuple.
3. Optionally `RETURNING id, name` to log what was created.

#### SQL Implementation
```sql
INSERT INTO products (id, name, category, price, stock)
VALUES
    (11, 'Laptop Stand',      'Furniture',   45.00,  30),
    (12, 'HDMI Cable',        'Electronics',  9.99, 200),
    (13, 'Whiteboard Marker', 'Stationery',   3.50,   0)
RETURNING id, name;
```
- A **multi-row `VALUES`** list is dramatically faster than N separate `INSERT`s: one parse, one plan, one WAL/commit cycle instead of N. For thousands of rows prefer `COPY` (Postgres) / `LOAD DATA` (MySQL).
- All tuples must have identical arity and compatible types, evaluated as one atomic unit — if row 3 violates a `CHECK`, none of the three are inserted.
- **Perf note:** wrapping large batches in an explicit transaction and inserting in chunks of ~1–10k rows balances lock duration against commit overhead.

#### Alternative Solution
`INSERT ... SELECT` when the data is *derived* rather than literal — e.g., seeding from a staging table:
```sql
INSERT INTO products (id, name, category, price, stock)
SELECT id, name, category, price, stock FROM products_staging;
```
Use literal multi-row `VALUES` for hand-supplied constants; use `INSERT ... SELECT` when the source is another query.

#### Interview Variations
1. Load 10,000 rows efficiently — what changes? (`COPY`, batching, deferring indexes.)
2. Skip rows whose `name` already exists.
3. Insert and capture all generated ids into an array in the app.

#### Common Follow-up Questions
- *Is a multi-row insert atomic?* Yes — it's a single statement; any failure rolls back the whole statement.
- *`COPY` vs multi-row `INSERT`?* `COPY` bypasses per-row SQL parsing and is the fastest bulk path; `INSERT` is fine for modest counts and supports `ON CONFLICT`.
- *How to get all inserted ids?* `RETURNING id` streams one row per inserted tuple.

### Practical Question 3: Idempotent Upsert of a Customer
- **Difficulty:** Medium
- **Estimated Time:** 7 min
- **Concepts Tested:** `INSERT ... ON CONFLICT` (upsert), `EXCLUDED`, conflict targets

**Problem Statement**
A sync job re-imports customers and may re-send someone who already exists (matched by unique `email`). Insert *Nina Patel* (`nina.patel@example.com`); if that email already exists, update the stored `city` and `name` to the incoming values instead of erroring.

**Example Input** — existing `customers`:

| id | name | email | city |
|----|------|-------|------|
| 1 | Nina Patel | nina.patel@example.com | Austin |

Incoming payload: name `Nina Patel`, city `Portland`.

**Example Output** — row 1 after upsert:

| id | name | email | city |
|----|------|-------|------|
| 1 | Nina Patel | nina.patel@example.com | Portland |

**Approach**
1. Attempt the `INSERT` as normal.
2. Declare the conflict target — the unique column(s) that define "already exists": `(email)`.
3. On conflict, `DO UPDATE SET ...` pulling incoming values from the special `EXCLUDED` row.

#### SQL Implementation
```sql
INSERT INTO customers (id, name, email, city)
VALUES (1, 'Nina Patel', 'nina.patel@example.com', 'Portland')
ON CONFLICT (email)
DO UPDATE SET
    name = EXCLUDED.name,
    city = EXCLUDED.city
RETURNING id, name, city;
```
- `ON CONFLICT (email)` names the **conflict target** — it must correspond to a unique or exclusion constraint. `EXCLUDED` is a virtual row holding the values you *tried* to insert.
- This is a true **upsert**: exactly one row exists afterward, with no read-modify-write race — the whole thing is one atomic statement, safe under concurrency.
- Note we do **not** overwrite `id` or `created_at`, preserving the original identity and signup time.
- **Perf note:** the conflict target must be backed by a unique index (here the `email` unique constraint); that index is what the engine probes to detect the collision.

#### Alternative Solution
`ON CONFLICT DO NOTHING` when you only want insert-if-absent semantics:
```sql
INSERT INTO customers (id, name, email, city)
VALUES (1, 'Nina Patel', 'nina.patel@example.com', 'Portland')
ON CONFLICT (email) DO NOTHING;
```
Choose `DO NOTHING` for "first write wins" and `DO UPDATE` for "last write wins." The read-then-`INSERT`/`UPDATE` pattern in app code is the wrong alternative — it has a check-then-act race.

#### Interview Variations
1. Only overwrite `city` when the incoming value is non-null (`COALESCE(EXCLUDED.city, customers.city)`).
2. Upsert on a **composite** unique key.
3. Track how many rows were inserted vs updated (`xmax = 0` trick, or `RETURNING (xmax = 0) AS inserted`).

#### Common Follow-up Questions
- *What is `EXCLUDED`?* A pseudo-table exposing the values proposed by the failed insert.
- *MySQL equivalent?* `INSERT ... ON DUPLICATE KEY UPDATE col = VALUES(col)` (or `col = new.col` in MySQL 8.0.19+).
- *Can you reference the existing row?* Yes — the target table name (`customers.city`) refers to the current stored row inside `DO UPDATE`.

### Practical Question 4: Insert an Order and Its Line Items Atomically
- **Difficulty:** Hard
- **Estimated Time:** 10 min
- **Concepts Tested:** Multi-table insert, transactions, CTE-chained `INSERT ... RETURNING`, referential integrity

**Problem Statement**
Customer 4 (Quentin Roy) places a new order dated `2026-07-09` for 2× *Wireless Mouse* (product 1) and 1× *Desk Lamp* (product 9). Create the `orders` row and both `order_items` rows so that either **all** rows commit or **none** do, and compute the order `total` from the line items.

**Example Output** — new `orders` row:

| id | customer_id | order_date | status | total |
|----|-------------|-----------|--------|-------|
| 11 | 4 | 2026-07-09 | pending | 84.93 |

new `order_items` rows:

| id | order_id | product_id | quantity | unit_price |
|----|----------|-----------|----------|------------|
| 16 | 11 | 1 | 2 | 24.99 |
| 17 | 11 | 9 | 1 | 34.95 |

**Approach**
1. Insert the parent `orders` row first (child rows need its `id` via FK).
2. Insert both `order_items` referencing that order id.
3. Do it in one transaction so a failure anywhere rolls everything back. A writable CTE lets us chain the inserts in a single statement.

#### SQL Implementation
```sql
WITH new_order AS (
    INSERT INTO orders (id, customer_id, order_date, status, total)
    VALUES (11, 4, DATE '2026-07-09', 'pending', 2*24.99 + 1*34.95)
    RETURNING id
),
items AS (
    INSERT INTO order_items (id, order_id, product_id, quantity, unit_price)
    SELECT 16, id, 1, 2, 24.99 FROM new_order
    UNION ALL
    SELECT 17, id, 9, 1, 34.95 FROM new_order
    RETURNING order_id
)
SELECT * FROM new_order;
```
- A **writable CTE** (data-modifying `WITH`) runs both inserts inside one statement — hence one atomic unit. The child insert reads the parent's `id` straight from the `new_order` CTE, so you never hard-code a key you don't yet know.
- The FK `order_items.order_id → orders.id` guarantees line items can't point at a nonexistent order; doing parent-then-child order respects that.
- If you prefer explicit control, wrap plain statements in `BEGIN; ... COMMIT;` — same atomicity, more readable for many rows.

#### Alternative Solution
Classic explicit transaction, ideal when logic branches between steps:
```sql
BEGIN;
INSERT INTO orders (id, customer_id, order_date, status, total)
VALUES (11, 4, DATE '2026-07-09', 'pending', 84.93);

INSERT INTO order_items (id, order_id, product_id, quantity, unit_price) VALUES
    (16, 11, 1, 2, 24.99),
    (17, 11, 9, 1, 34.95);
COMMIT;
```
The CTE form is elegant for a fixed shape; the `BEGIN/COMMIT` form is clearer when you need application logic, error handling, or `SAVEPOINT`s between steps.

#### Interview Variations
1. Also decrement `products.stock` for each item in the same transaction.
2. Roll back the whole thing if any product is out of stock (`stock >= quantity` guard).
3. Compute `total` with a trigger instead of inline arithmetic.

#### Common Follow-up Questions
- *Why insert the parent before children?* The FK requires the referenced `orders.id` to exist (unless the constraint is `DEFERRABLE`).
- *What isolation level do you need?* `READ COMMITTED` suffices here; stock-guard variants may want `REPEATABLE READ` or explicit row locks (`SELECT ... FOR UPDATE`).
- *Are writable CTEs guaranteed atomic?* Yes — the entire statement (all CTEs) commits or aborts together.
- *Can I nest transactions?* Not truly; use `SAVEPOINT` for partial rollback within one transaction.

## CRUD — SELECT, Filtering & Sorting

### Practical Question 5: Find Well-Paid Engineers, Highest First
- **Difficulty:** Easy
- **Estimated Time:** 4 min
- **Concepts Tested:** `WHERE` with multiple predicates, `ORDER BY`, column projection

**Problem Statement**
List the `name`, `salary`, and `hire_date` of employees in department 1 (Engineering) earning more than 100,000, ordered from highest salary to lowest.

**Example Input** — `employees` (Engineering rows):

| id | name | department_id | salary |
|----|------|---------------|--------|
| 1 | Alice Johnson | 1 | 185000 |
| 3 | Carol Williams | 1 | 142000 |
| 4 | David Brown | 1 | 128000 |
| 9 | Ivy Taylor | 1 | 99000 |

**Example Output**

| name | salary | hire_date |
|------|--------|-----------|
| Alice Johnson | 185000.00 | 2015-03-01 |
| Carol Williams | 142000.00 | 2018-01-20 |
| David Brown | 128000.00 | 2019-06-10 |

**Approach**
1. Filter with two `AND`-ed predicates: `department_id = 1` and `salary > 100000`.
2. Project only the three requested columns.
3. Sort descending with `ORDER BY salary DESC`.

#### SQL Implementation
```sql
SELECT name, salary, hire_date
FROM employees
WHERE department_id = 1
  AND salary > 100000
ORDER BY salary DESC;
```
- Predicates in `WHERE` run **before** projection and ordering; only rows passing every `AND` survive. Ivy (99,000) is filtered out.
- Selecting explicit columns instead of `SELECT *` reduces I/O and makes the contract with callers stable — a senior habit.
- **Perf/index note:** a composite index `(department_id, salary DESC)` would let the planner satisfy both the filter and the sort from the index (an index-only scan if just those columns are needed), avoiding a sort step entirely.

#### Alternative Solution
Same result, ordering by an ordinal or expression is possible but discouraged for readability:
```sql
SELECT name, salary, hire_date
FROM employees
WHERE department_id = 1 AND salary > 100000
ORDER BY 2 DESC;   -- 2 = salary; fragile if columns change
```
Named columns in `ORDER BY` are the canonical choice; positional ordinals are terse but break silently when the select list is edited.

#### Interview Variations
1. Break ties by earliest `hire_date` (`ORDER BY salary DESC, hire_date ASC`).
2. Return only the top 3 (add `LIMIT 3` — see the pagination section).
3. Filter by department **name** instead of id (requires a join).

#### Common Follow-up Questions
- *Is `salary > 100000` inclusive?* No — use `>=` for inclusive.
- *Does `ORDER BY` run before or after `WHERE`?* After filtering; logical order is `FROM → WHERE → SELECT → ORDER BY`.
- *Default sort direction?* `ASC`. In Postgres you can also control `NULLS FIRST/LAST`.

### Practical Question 6: Customers in a Set of Cities, with NULL Handling
- **Difficulty:** Medium
- **Estimated Time:** 6 min
- **Concepts Tested:** `IN`, `IS NULL`, `OR` precedence, three-valued logic

**Problem Statement**
Return all customers located in *Austin* or *New York*, **plus** any customer whose `city` is unknown (`NULL`). Sort by city then name, keeping the NULL-city customers last.

**Example Input** — `customers`:

| id | name | city |
|----|------|------|
| 1 | Nina Patel | Austin |
| 2 | Omar Haddad | New York |
| 5 | Rosa Lopez | Austin |
| 6 | Sam Turner | New York |
| 7 | Tara Singh | NULL |

**Example Output**

| id | name | city |
|----|------|------|
| 1 | Nina Patel | Austin |
| 5 | Rosa Lopez | Austin |
| 2 | Omar Haddad | New York |
| 6 | Sam Turner | New York |
| 7 | Tara Singh | NULL |

**Approach**
1. Use `city IN ('Austin','New York')` for the set membership.
2. `OR city IS NULL` — you cannot match NULL with `=` or `IN`; NULL comparisons yield `UNKNOWN`, never `TRUE`.
3. Sort with `NULLS LAST` so unknown cities sink to the bottom.

#### SQL Implementation
```sql
SELECT id, name, city
FROM customers
WHERE city IN ('Austin', 'New York')
   OR city IS NULL
ORDER BY city ASC NULLS LAST, name ASC;
```
- **Three-valued logic gotcha:** `city IN ('Austin','New York')` returns `UNKNOWN` for a NULL city, which is *not* `TRUE`, so NULL rows would be excluded without the explicit `OR city IS NULL`.
- `ORDER BY ... NULLS LAST` is a PostgreSQL nicety; by default Postgres sorts NULLs *last* for `ASC` anyway, but stating it makes intent unmissable.
- **Perf note:** `IN` on a short constant list compiles to a series of OR'd equalities; a B-tree index on `city` can be used, but the `OR city IS NULL` branch may push the planner toward a bitmap OR or a seq scan on small tables.

#### Alternative Solution
Coalesce the NULL into a sentinel to fold the logic into the set test — usually **not** recommended because it defeats index usage:
```sql
SELECT id, name, city
FROM customers
WHERE COALESCE(city, '(unknown)') IN ('Austin', 'New York', '(unknown)')
ORDER BY city NULLS LAST, name;
```
The explicit `IS NULL` version is canonical: clearer intent and index-friendly.

#### Interview Variations
1. Exclude NULLs instead (just drop the `OR` branch).
2. Case-insensitive city match (`lower(city) IN ('austin','new york')`, ideally with a functional index).
3. Cities matching a pattern (`city LIKE 'New%'`).

#### Common Follow-up Questions
- *Why doesn't `city = NULL` work?* Any comparison with NULL is `UNKNOWN`; use `IS NULL`.
- *Does `NOT IN` handle NULLs safely?* Dangerously no — `NOT IN (subquery with NULLs)` can return zero rows unexpectedly; prefer `NOT EXISTS`.
- *MySQL `NULLS LAST`?* Not supported directly; emulate with `ORDER BY (city IS NULL), city`.

### Practical Question 7: Fuzzy Product Search by Name and Price Band
- **Difficulty:** Medium
- **Estimated Time:** 6 min
- **Concepts Tested:** `LIKE`/`ILIKE`, `BETWEEN`, `DISTINCT`, combining ranges with pattern match

**Problem Statement**
Support wants every product whose name contains the word "desk" (case-insensitive) OR that falls in the mid-price band 30–80 inclusive. Return distinct categories affected, and separately the matching products ordered by price ascending.

**Example Output** — matching products:

| id | name | category | price |
|----|------|----------|-------|
| 9 | Desk Lamp | Furniture | 34.95 |
| 3 | USB-C Hub | Electronics | 39.50 |
| 6 | Standing Desk | Furniture | 449.00 |
| 10 | Webcam 1080p | Electronics | 59.00 |
| 2 | Mechanical Keyboard | Electronics | 79.99 |

**Approach**
1. Case-insensitive substring match with `ILIKE '%desk%'` (Postgres) — catches "Desk Lamp" and "Standing Desk".
2. Price band with `price BETWEEN 30 AND 80` (inclusive on both ends).
3. Combine with `OR`; order the product list by `price`.

#### SQL Implementation
```sql
-- Matching products
SELECT id, name, category, price
FROM products
WHERE name ILIKE '%desk%'
   OR price BETWEEN 30 AND 80
ORDER BY price ASC;

-- Distinct categories affected
SELECT DISTINCT category
FROM products
WHERE name ILIKE '%desk%'
   OR price BETWEEN 30 AND 80
ORDER BY category;
```
- `ILIKE` is PostgreSQL's case-insensitive `LIKE`; `%` matches any run of characters. In standard SQL/MySQL use `LOWER(name) LIKE '%desk%'`.
- `BETWEEN a AND b` is inclusive — equivalent to `price >= 30 AND price <= 80`. Standing Desk (449) is included via the name match, not the price band.
- `DISTINCT` collapses duplicate category values; it applies to the **whole** select list, not one column.
- **Perf note:** leading-wildcard `%desk%` cannot use a plain B-tree index — it forces a scan. For real search, use a trigram index (`CREATE INDEX ... USING gin (name gin_trgm_ops)`) or full-text search.

#### Alternative Solution
For anchored (prefix) searches, drop the leading `%` to enable index use:
```sql
WHERE name ILIKE 'desk%'   -- can use a B-tree index on lower(name)
```
Canonical guidance: prefix matches are index-friendly; infix/`%x%` matches need trigram/FTS. Choose based on whether users search by prefix or substring.

#### Interview Variations
1. Exclude out-of-stock items (`AND stock > 0`).
2. Rank by relevance using `similarity()` from `pg_trgm`.
3. Escape a literal `%` or `_` in the search term (`LIKE ... ESCAPE '\'`).

#### Common Follow-up Questions
- *`LIKE` vs `ILIKE` vs `~*`?* `LIKE` is case-sensitive globbing; `ILIKE` is case-insensitive; `~*` is a case-insensitive POSIX regex.
- *Is `BETWEEN` inclusive?* Yes, both bounds.
- *Why is `%term%` slow?* The leading wildcard prevents B-tree seeking; the engine must scan every row.

### Practical Question 8: Categorize Employees by Tenure with CASE
- **Difficulty:** Hard
- **Estimated Time:** 9 min
- **Concepts Tested:** Computed columns, `CASE`, date arithmetic, `ORDER BY` on an alias/expression

**Problem Statement**
Produce a report of every employee with a derived `tenure_years` (whole years since `hire_date` as of 2026-07-09) and a `seniority_band`: `Veteran` (≥ 8 yrs), `Established` (4–7 yrs), or `Newcomer` (< 4 yrs). Sort by tenure descending.

**Example Output** (abridged)

| name | hire_date | tenure_years | seniority_band |
|------|-----------|--------------|----------------|
| Alice Johnson | 2015-03-01 | 11 | Veteran |
| Bob Smith | 2016-07-15 | 9 | Veteran |
| Frank Miller | 2017-11-30 | 8 | Veteran |
| Henry Moore | 2018-09-01 | 7 | Established |
| Karen Thomas | 2023-05-22 | 3 | Newcomer |

**Approach**
1. Compute tenure with age/date math: `EXTRACT(YEAR FROM age(DATE '2026-07-09', hire_date))`.
2. Map the numeric tenure to a band with a searched `CASE`.
3. Sort by the underlying expression (aliases aren't visible in `WHERE` but are usable in `ORDER BY`).

#### SQL Implementation
```sql
SELECT
    name,
    hire_date,
    EXTRACT(YEAR FROM age(DATE '2026-07-09', hire_date))::int AS tenure_years,
    CASE
        WHEN age(DATE '2026-07-09', hire_date) >= INTERVAL '8 years' THEN 'Veteran'
        WHEN age(DATE '2026-07-09', hire_date) >= INTERVAL '4 years' THEN 'Established'
        ELSE 'Newcomer'
    END AS seniority_band
FROM employees
ORDER BY tenure_years DESC, hire_date ASC;
```
- `age(a, b)` returns a symbolic interval; `EXTRACT(YEAR FROM age(...))` gives *completed* whole years — more accurate than `(current_date - hire_date)/365`, which drifts on leap years.
- A **searched `CASE`** evaluates `WHEN` clauses top-to-bottom and returns on the first `TRUE`; ordering the bands from strictest to loosest is what makes the overlapping thresholds correct.
- `ORDER BY tenure_years` can reference the select-list **alias** — a special allowance in `ORDER BY` (and `GROUP BY` in Postgres), unlike `WHERE`, which runs before aliases exist.
- **Perf note:** the `CASE`/`age` expressions are computed per row and cannot use an index; for large tables you'd store `hire_date` and filter on raw date ranges, computing bands only for the returned page.

#### Alternative Solution
Compute tenure once in a CTE/subquery, then classify — avoids repeating the `age()` call and keeps thresholds in one place:
```sql
SELECT name, hire_date, tenure_years,
       CASE WHEN tenure_years >= 8 THEN 'Veteran'
            WHEN tenure_years >= 4 THEN 'Established'
            ELSE 'Newcomer' END AS seniority_band
FROM (
    SELECT name, hire_date,
           EXTRACT(YEAR FROM age(DATE '2026-07-09', hire_date))::int AS tenure_years
    FROM employees
) t
ORDER BY tenure_years DESC, hire_date;
```
This is the more maintainable canonical form: the expensive expression is written once, and the `CASE` reads cleanly off the derived integer.

#### Interview Variations
1. Use `CURRENT_DATE` instead of a hard-coded date for a live report.
2. Add a fourth band and make thresholds data-driven from a `bands` table (join on range).
3. Count employees per band (`GROUP BY seniority_band`).

#### Common Follow-up Questions
- *Why not `(current_date - hire_date)/365`?* Integer/day math ignores leap years and gives fractional drift; `age()` is calendar-correct.
- *Can I use the alias in `WHERE`?* No — `WHERE` is evaluated before the select list; repeat the expression or wrap in a subquery/CTE.
- *`CASE` vs `COALESCE`/`NULLIF`?* `CASE` is the general conditional; `COALESCE`/`NULLIF` are shorthands for specific null patterns.
- *MySQL date diff?* Use `TIMESTAMPDIFF(YEAR, hire_date, '2026-07-09')`.

## CRUD — Pagination (LIMIT/OFFSET)

### Practical Question 9: First Page of Products by Price
- **Difficulty:** Easy
- **Estimated Time:** 4 min
- **Concepts Tested:** `LIMIT`, `ORDER BY`, deterministic ordering

**Problem Statement**
Show "page 1" of the product catalog: the 3 cheapest products, ordered by price ascending. Return `id`, `name`, `price`.

**Example Output**

| id | name | price |
|----|------|-------|
| 8 | Gel Pens (Pack of 10) | 8.99 |
| 7 | Notebook (Pack of 3) | 12.49 |
| 1 | Wireless Mouse | 24.99 |

**Approach**
1. Order by `price ASC` — pagination without an `ORDER BY` is meaningless (row order is undefined).
2. Add a tiebreaker (`id`) so ties are deterministic across pages.
3. `LIMIT 3` to take the first page.

#### SQL Implementation
```sql
SELECT id, name, price
FROM products
ORDER BY price ASC, id ASC
LIMIT 3;
```
- **`LIMIT` without `ORDER BY` is a bug**, not a shortcut: the engine may return any 3 rows, and "any" can change between runs. Always pair them.
- The secondary `id` key makes the sort **total** — critical so that two products with the same price don't swap places between page 1 and page 2.
- **Perf note:** with an index on `(price, id)` this is an index range scan reading only the first 3 entries — O(page size), not O(table).

#### Alternative Solution
`FETCH FIRST` is the SQL-standard spelling and behaves identically:
```sql
SELECT id, name, price
FROM products
ORDER BY price ASC, id ASC
FETCH FIRST 3 ROWS ONLY;
```
`LIMIT` is ubiquitous (Postgres/MySQL/SQLite); `FETCH FIRST n ROWS ONLY` is ANSI-standard and preferred if portability to Oracle/DB2/SQL Server is a concern.

#### Interview Variations
1. Cheapest 3 *in stock* (`WHERE stock > 0`).
2. Most expensive 3 (`ORDER BY price DESC`).
3. Return the single cheapest per category (needs `DISTINCT ON` or window functions).

#### Common Follow-up Questions
- *What if two rows tie on price?* Add a unique tiebreaker to `ORDER BY`, else order is nondeterministic.
- *Is `LIMIT` applied before or after `ORDER BY`?* After — the sort happens, then the top N are taken.
- *MySQL/SQL Server?* MySQL uses `LIMIT`; SQL Server uses `TOP` or `OFFSET ... FETCH`.

### Practical Question 10: Nth Page with OFFSET
- **Difficulty:** Medium
- **Estimated Time:** 6 min
- **Concepts Tested:** `LIMIT ... OFFSET`, page-number math, stable ordering

**Problem Statement**
The catalog UI shows 3 products per page ordered by price ascending. Return **page 3** (the 7th–9th cheapest products).

**Example Output**

| id | name | price |
|----|------|-------|
| 10 | Webcam 1080p | 59.00 |
| 2 | Mechanical Keyboard | 79.99 |
| 5 | Office Chair | 199.00 |

**Approach**
1. Page size = 3, page number = 3 → skip `(3 - 1) * 3 = 6` rows.
2. `OFFSET 6` skips them; `LIMIT 3` takes the page.
3. Keep the same total ordering (`price, id`) used on every page.

#### SQL Implementation
```sql
SELECT id, name, price
FROM products
ORDER BY price ASC, id ASC
LIMIT 3 OFFSET 6;          -- page 3, size 3  ->  OFFSET (page-1)*size
```
- `OFFSET n` **discards** the first *n* ordered rows before `LIMIT` takes over. The general formula: `OFFSET (page_number - 1) * page_size`.
- The ordering must be identical and total on every page, or rows can appear on two pages or be skipped entirely.
- **Perf pitfall:** `OFFSET` still *reads and throws away* those n rows. `OFFSET 6` is trivial, but `OFFSET 1000000` scans a million rows every time — pagination that degrades linearly with page depth.

#### Alternative Solution — Keyset ("seek") pagination
Instead of counting from the start, remember the last row seen and seek past it:
```sql
-- last row on page 2 had (price, id) = (39.50, 3); fetch the next 3
SELECT id, name, price
FROM products
WHERE (price, id) > (39.50, 3)
ORDER BY price ASC, id ASC
LIMIT 3;
```
This is the **canonical** approach for deep pagination and infinite scroll: it uses the index to jump straight to the cursor position, giving constant-time pages regardless of depth. `OFFSET` is fine for shallow, page-numbered UIs; keyset wins for large offsets and stable "next" links.

#### Interview Variations
1. Return page 3 **and** the total page count (`COUNT(*)` in a separate/parallel query).
2. Make page size a bind parameter.
3. Convert to keyset pagination for an "infinite scroll" feed.

#### Common Follow-up Questions
- *Why is deep `OFFSET` slow?* The DB must generate and discard every skipped row; cost grows with the offset.
- *Does keyset pagination support jumping to page 500?* No — it's inherently next/previous; page-number jumps need `OFFSET` or precomputed boundaries.
- *What breaks pagination?* A non-total or changing `ORDER BY`, or rows being inserted/deleted between page fetches.

### Practical Question 11: Top-N-per-Group — Two Most Recent Orders per Customer
- **Difficulty:** Hard
- **Estimated Time:** 10 min
- **Concepts Tested:** Window functions (`ROW_NUMBER`), `PARTITION BY`, per-group limiting, `DISTINCT ON`

**Problem Statement**
For each customer who has orders, return their **two most recent** orders (by `order_date`, newest first). A global `LIMIT` can't express "N per group" — you need windowing.

**Example Input** — `orders` for customers 1, 2, 3:

| id | customer_id | order_date |
|----|-------------|-----------|
| 1 | 1 | 2023-05-01 |
| 2 | 1 | 2023-06-12 |
| 3 | 2 | 2023-06-15 |
| 7 | 2 | 2023-07-19 |
| 4 | 3 | 2023-06-20 |
| 9 | 3 | 2023-08-11 |

**Example Output**

| customer_id | id | order_date | rn |
|-------------|----|-----------|----|
| 1 | 2 | 2023-06-12 | 1 |
| 1 | 1 | 2023-05-01 | 2 |
| 2 | 7 | 2023-07-19 | 1 |
| 2 | 3 | 2023-06-15 | 2 |
| 3 | 9 | 2023-08-11 | 1 |
| 3 | 4 | 2023-06-20 | 2 |

**Approach**
1. Number each customer's orders newest-first with `ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC)`.
2. Wrap in a subquery/CTE (window functions can't live in `WHERE`).
3. Keep rows where `rn <= 2`.

#### SQL Implementation
```sql
SELECT customer_id, id, order_date, rn
FROM (
    SELECT
        customer_id,
        id,
        order_date,
        ROW_NUMBER() OVER (
            PARTITION BY customer_id
            ORDER BY order_date DESC, id DESC
        ) AS rn
    FROM orders
) ranked
WHERE rn <= 2
ORDER BY customer_id, rn;
```
- `PARTITION BY customer_id` restarts the numbering for each customer — think "a separate `ORDER BY` + counter per group." `ORDER BY order_date DESC` makes rn=1 the newest.
- Window functions are computed **after** `WHERE`/`GROUP BY`, so you can't filter `rn` in the same query level — hence the subquery. This "rank then filter" is the standard top-N-per-group pattern.
- The `id DESC` tiebreaker keeps results deterministic when two orders share a date.
- **Perf note:** an index on `(customer_id, order_date DESC)` lets the engine feed rows to the window in order, avoiding a large sort.

#### Alternative Solution — Postgres `DISTINCT ON`
For "top 1 per group" Postgres offers a concise idiom; for top-2 you'd still window, but it's worth knowing:
```sql
-- most recent (top 1) order per customer
SELECT DISTINCT ON (customer_id) customer_id, id, order_date
FROM orders
ORDER BY customer_id, order_date DESC, id DESC;
```
`DISTINCT ON` is elegant and fast for **N=1**. For N≥2, `ROW_NUMBER` is the general, portable canonical tool. Use `RANK`/`DENSE_RANK` instead if ties should all qualify.

#### Interview Variations
1. Use `RANK()` so tied dates both count toward the "top 2".
2. Return the top *spend* orders per customer instead of most recent (`ORDER BY total DESC`).
3. Include customers with zero orders (`LEFT JOIN` from `customers`).

#### Common Follow-up Questions
- *`ROW_NUMBER` vs `RANK` vs `DENSE_RANK`?* `ROW_NUMBER` is always unique (1,2,3); `RANK` leaves gaps after ties (1,1,3); `DENSE_RANK` doesn't (1,1,2).
- *Why can't I use `rn` in `WHERE`?* Window functions are evaluated after `WHERE`; wrap in a subquery/CTE or use `QUALIFY` (in engines that support it, e.g. Snowflake/DuckDB).
- *Does MySQL support this?* Yes, 8.0+ has window functions; older MySQL needs a correlated-subquery or variable trick.

## CRUD — UPDATE

### Practical Question 12: Give One Employee a Raise
- **Difficulty:** Easy
- **Estimated Time:** 4 min
- **Concepts Tested:** Single-row `UPDATE`, `WHERE` on primary key, `RETURNING`

**Problem Statement**
Employee 5 (Eve Davis) is promoted; raise her salary to 110,000. Update exactly that row and return the new value to confirm.

**Example Input** — `employees` row 5:

| id | name | salary |
|----|------|--------|
| 5 | Eve Davis | 98000.00 |

**Example Output**

| id | name | salary |
|----|------|--------|
| 5 | Eve Davis | 110000.00 |

**Approach**
1. `SET salary = 110000`.
2. Scope with `WHERE id = 5` — the primary key guarantees exactly one row.
3. `RETURNING` to verify without a follow-up `SELECT`.

#### SQL Implementation
```sql
UPDATE employees
SET salary = 110000
WHERE id = 5
RETURNING id, name, salary;
```
- The `WHERE id = 5` clause is the safety belt: **an `UPDATE` with no `WHERE` rewrites every row.** Always confirm the predicate before running.
- Filtering on the primary key uses the PK index for an instant, single-row locate-and-update.
- `RETURNING` echoes the post-update state — cleaner and race-free versus a separate `SELECT`.

#### Alternative Solution
Run it inside a transaction so you can inspect and abort if the count looks wrong:
```sql
BEGIN;
UPDATE employees SET salary = 110000 WHERE id = 5;
-- check "UPDATE 1"; if it said UPDATE 0 or UPDATE 5, ROLLBACK
COMMIT;
```
For a keyed single-row change the bare statement is fine; the transactional wrapper is the canonical safe habit for anything you're unsure about.

#### Interview Variations
1. Raise by a percentage instead (`SET salary = salary * 1.10`).
2. Only raise if current salary < 100,000 (`AND salary < 100000`).
3. Also stamp an `updated_at` column.

#### Common Follow-up Questions
- *What if `WHERE` matches nothing?* Zero rows updated, no error; the reported count is `UPDATE 0`.
- *How do I know how many rows changed?* The command tag (`UPDATE n`) or `RETURNING` row count.
- *Is the update atomic?* Yes — a single statement either fully applies or rolls back.

### Practical Question 13: Percentage Raise for a Whole Department
- **Difficulty:** Medium
- **Estimated Time:** 6 min
- **Concepts Tested:** Set-based `UPDATE`, expression in `SET`, filtering the target set, rounding

**Problem Statement**
Marketing (department 3) gets a 5% cost-of-living raise, but cap any single raise so no salary exceeds 130,000 after the increase. Round to 2 decimals.

**Example Input** — Marketing employees:

| id | name | salary |
|----|------|--------|
| 6 | Frank Miller | 115000.00 |
| 7 | Grace Wilson | 87000.00 |

**Example Output**

| id | name | salary |
|----|------|--------|
| 6 | Frank Miller | 120750.00 |
| 7 | Grace Wilson | 91350.00 |

**Approach**
1. Compute the new salary as `salary * 1.05`, then `LEAST(..., 130000)` to apply the cap.
2. `ROUND(..., 2)` for currency precision.
3. Restrict with `WHERE department_id = 3`.

#### SQL Implementation
```sql
UPDATE employees
SET salary = ROUND(LEAST(salary * 1.05, 130000), 2)
WHERE department_id = 3
RETURNING id, name, salary;
```
- The whole update is **set-based**: one statement recomputes every matching row's salary in a single pass — vastly better than looping row-by-row in application code.
- `LEAST(a, b)` returns the smaller argument, implementing the cap inline; `GREATEST` would be its floor counterpart.
- `ROUND(x, 2)` keeps monetary values clean; with `NUMERIC` the arithmetic is exact (no float drift).
- **Perf note:** an index on `department_id` locates the target rows; the update itself must rewrite each matched row and maintain any index on `salary`.

#### Alternative Solution
Encapsulate the cap in a `CASE` when the rule is more than a simple min/max:
```sql
UPDATE employees
SET salary = CASE
                WHEN salary * 1.05 > 130000 THEN 130000
                ELSE ROUND(salary * 1.05, 2)
             END
WHERE department_id = 3;
```
`LEAST` is the concise canonical form for a pure cap; `CASE` is preferable once the branching logic grows (e.g., different caps by role).

#### Interview Variations
1. Raise only employees hired before 2020 (`AND hire_date < DATE '2020-01-01'`).
2. Different percentages per department using a join to a `raise_rules` table.
3. Log every change to an audit table in the same transaction.

#### Common Follow-up Questions
- *Can I reference the old value in `SET`?* Yes — the right-hand side sees pre-update values (`salary * 1.05`).
- *Order of evaluation across rows?* Undefined and irrelevant for set-based updates; each row is computed from its own pre-image.
- *Float vs numeric for money?* Use `NUMERIC`/`DECIMAL`; floating point introduces rounding errors.

### Practical Question 14: Correlated UPDATE from Another Table
- **Difficulty:** Medium
- **Estimated Time:** 8 min
- **Concepts Tested:** `UPDATE ... FROM` (Postgres), correlated update, recomputing derived columns

**Problem Statement**
The seeded `orders.total` values are approximate. Recompute each order's `total` as the true sum of its `order_items` (`SUM(quantity * unit_price)`). Update all orders.

**Example Input** — order 8's items:

| order_id | product_id | quantity | unit_price |
|----------|-----------|----------|------------|
| 8 | 4 | 1 | 289.00 |
| 8 | 5 | 1 | 199.00 |
| 8 | 10 | 1 | 59.00 |

**Example Output** — order 8 after update:

| id | total |
|----|-------|
| 8 | 547.00 |

**Approach**
1. Aggregate `order_items` by `order_id` to get each order's true total.
2. Join that aggregate back to `orders` and assign it.
3. Use `UPDATE ... FROM` (Postgres) or a correlated subquery (portable).

#### SQL Implementation
```sql
UPDATE orders o
SET total = t.computed_total
FROM (
    SELECT order_id, SUM(quantity * unit_price) AS computed_total
    FROM order_items
    GROUP BY order_id
) t
WHERE o.id = t.order_id
RETURNING o.id, o.total;
```
- **`UPDATE ... FROM`** is PostgreSQL's join-in-update: the derived table `t` supplies the new value, and `WHERE o.id = t.order_id` correlates each order to its own aggregate.
- Precomputing the sums once in a grouped subquery is far more efficient than a per-row correlated subquery that re-scans `order_items` for every order.
- Orders with **no** items won't appear in `t`, so they're left untouched here — a deliberate choice (see variations for the `COALESCE`-to-zero alternative).
- **Perf note:** an index on `order_items(order_id)` accelerates the `GROUP BY`; the update rewrites one row per order.

#### Alternative Solution — portable correlated subquery
Works on engines without `UPDATE ... FROM` (e.g., older MySQL uses a multi-table `UPDATE ... JOIN` instead):
```sql
UPDATE orders o
SET total = COALESCE((
    SELECT SUM(oi.quantity * oi.unit_price)
    FROM order_items oi
    WHERE oi.order_id = o.id
), 0);
```
The correlated-subquery form is the most portable and also handles item-less orders (via `COALESCE(...,0)`); `UPDATE ... FROM` is typically faster because the aggregate is computed once, not per row.

#### Interview Variations
1. Only recompute orders in `('pending','paid')` status.
2. Set item-less orders' totals to 0 explicitly.
3. Maintain `total` automatically going forward with a trigger.

#### Common Follow-up Questions
- *Why can a correlated subquery be slow?* It may re-execute per outer row; `UPDATE ... FROM` with a grouped subquery evaluates the aggregate once.
- *MySQL syntax?* `UPDATE orders o JOIN (...) t ON o.id = t.order_id SET o.total = t.computed_total;`
- *What about orders with no items in the `FROM` version?* They aren't matched, so they keep their old total — use the `COALESCE` subquery to force 0.

### Practical Question 15: Conditional Bulk Status Transition
- **Difficulty:** Hard
- **Estimated Time:** 9 min
- **Concepts Tested:** Multi-column `SET`, guarded state transitions, `WHERE` on multiple conditions, concurrency safety

**Problem Statement**
A nightly job promotes stale orders: every order still in `'paid'` status whose `order_date` is more than 30 days before 2026-07-09 should transition to `'shipped'`. Update `status` and return how many/which orders moved. Ensure the job is safe to re-run.

**Example Input** — `orders` (paid):

| id | order_date | status |
|----|-----------|--------|
| 3 | 2023-06-15 | paid |
| 8 | 2023-08-01 | paid |

**Example Output** — rows transitioned:

| id | order_date | status |
|----|-----------|--------|
| 3 | 2023-06-15 | shipped |
| 8 | 2023-08-01 | shipped |

**Approach**
1. Filter to the exact source state: `status = 'paid'` (this makes the job idempotent — re-runs match nothing new).
2. Add the staleness predicate `order_date < DATE '2026-07-09' - INTERVAL '30 days'`.
3. `SET status = 'shipped'` and `RETURNING` the affected rows for the job log.

#### SQL Implementation
```sql
UPDATE orders
SET status = 'shipped'
WHERE status = 'paid'
  AND order_date < DATE '2026-07-09' - INTERVAL '30 days'
RETURNING id, order_date, status;
```
- Guarding on the **current** state (`status = 'paid'`) is what makes this a safe, **idempotent** transition: run it twice and the second run finds nothing to do, because the rows are now `'shipped'`.
- The date predicate uses interval arithmetic; keeping `order_date` bare on the left (rather than wrapping it in a function) preserves the ability to use an index on `order_date`.
- `RETURNING` gives the job an exact manifest of what changed — ideal for logging and downstream events.
- **Concurrency note:** under default `READ COMMITTED`, if two workers run this simultaneously, row locks serialize them and each row transitions once; the state guard prevents double-processing.

#### Alternative Solution
When the transition must emit domain events or update multiple tables, wrap it and capture ids first:
```sql
BEGIN;
WITH moved AS (
    UPDATE orders
    SET status = 'shipped'
    WHERE status = 'paid'
      AND order_date < DATE '2026-07-09' - INTERVAL '30 days'
    RETURNING id
)
INSERT INTO order_events (order_id, event, created_at)
SELECT id, 'auto_shipped', now() FROM moved;
COMMIT;
```
The single `UPDATE ... RETURNING` is canonical for a pure status flip; the writable-CTE version is the pattern when the state change must fan out to other tables atomically.

#### Interview Variations
1. Also require the order to have at least one `order_item` (`AND EXISTS (...)`).
2. Transition `pending → cancelled` for orders older than 90 days.
3. Prevent illegal transitions with a `CHECK`/trigger state machine.

#### Common Follow-up Questions
- *Why filter on the source status at all?* Idempotency and correctness — you only advance rows actually in the expected state.
- *Is `RETURNING` evaluated before or after the update?* After — it reflects the new (post-update) values.
- *How to avoid double-processing under concurrency?* The status guard plus row locking; or claim rows with `SELECT ... FOR UPDATE SKIP LOCKED` for worker queues.

## CRUD — DELETE / TRUNCATE

### Practical Question 16: Delete a Single Cancelled Order
- **Difficulty:** Easy
- **Estimated Time:** 4 min
- **Concepts Tested:** Single-row `DELETE`, `WHERE` safety, `RETURNING`, cascade behavior

**Problem Statement**
Order 6 was cancelled and should be removed entirely. Delete it and return the deleted row for the audit log. Note what happens to its `order_items`.

**Example Input** — `orders` row 6 and its item:

| id | customer_id | status | total |
|----|-------------|--------|-------|
| 6 | 5 | cancelled | 199.00 |

`order_items`: one row (id 9) with `order_id = 6`.

**Example Output** — deleted row:

| id | customer_id | status | total |
|----|-------------|--------|-------|
| 6 | 5 | cancelled | 199.00 |

**Approach**
1. Target the exact row with `WHERE id = 6`.
2. `RETURNING *` to capture what was removed.
3. Rely on `ON DELETE CASCADE` (defined on `order_items.order_id`) to remove child line items automatically.

#### SQL Implementation
```sql
DELETE FROM orders
WHERE id = 6
RETURNING *;
```
- Just like `UPDATE`, **`DELETE` without a `WHERE` empties the table.** The keyed predicate is mandatory discipline.
- Because `order_items.order_id` was declared `REFERENCES orders(id) ON DELETE CASCADE`, deleting order 6 also removes its line item (id 9) in the same operation — no orphan rows, no manual cleanup.
- `RETURNING *` hands back the deleted row so you can archive it; once committed, it's gone otherwise.
- **Perf note:** the PK index locates the row instantly; the cascade triggers a keyed delete on `order_items` via its `order_id` FK (index that FK column to keep cascades fast).

#### Alternative Solution — soft delete
In many production systems you never hard-delete; you flag instead:
```sql
-- requires a deleted_at TIMESTAMP column
UPDATE orders SET deleted_at = now() WHERE id = 6;
-- queries then add:  WHERE deleted_at IS NULL
```
Hard `DELETE` is canonical when the row is truly disposable; **soft delete** is preferred when you need audit history, referential safety, or "undo." Choose based on retention requirements.

#### Interview Variations
1. Delete only if it's still `cancelled` (`AND status = 'cancelled'`) to avoid racing a re-activation.
2. Delete and archive to a `deleted_orders` table in one transaction.
3. What if the FK had `ON DELETE RESTRICT`? (The delete would fail while items exist.)

#### Common Follow-up Questions
- *What happens to `order_items` here?* They're cascade-deleted due to `ON DELETE CASCADE`.
- *Without cascade?* The delete would raise `23503 foreign_key_violation`; delete children first or use `RESTRICT`/`SET NULL` as designed.
- *Can I undo a committed delete?* No — restore from backup/WAL or use soft deletes to make "undo" trivial.

### Practical Question 17: Delete Rows Matching a Subquery
- **Difficulty:** Medium
- **Estimated Time:** 7 min
- **Concepts Tested:** `DELETE ... WHERE ... IN (subquery)` / `EXISTS`, deleting based on related data

**Problem Statement**
Purge every order belonging to customers located in *Chicago*. The city lives on `customers`, not `orders`, so you must correlate. Return the deleted order ids.

**Example Input** — Quentin Roy (customer 4) is in Chicago and owns order 5.

**Example Output** — deleted:

| id | customer_id |
|----|-------------|
| 5 | 4 |

**Approach**
1. Identify the target customers via `customers.city = 'Chicago'`.
2. Delete orders whose `customer_id` is in that set — using `IN (subquery)` or, better, `EXISTS`.
3. `RETURNING` the removed ids.

#### SQL Implementation
```sql
DELETE FROM orders o
WHERE EXISTS (
    SELECT 1
    FROM customers c
    WHERE c.id = o.customer_id
      AND c.city = 'Chicago'
)
RETURNING o.id, o.customer_id;
```
- `EXISTS` is the robust choice for correlated deletes: it short-circuits on the first match and, crucially, is **NULL-safe** — unlike `IN`, it never misbehaves when the subquery can yield NULLs.
- The subquery references the outer `orders` row (`c.id = o.customer_id`), so each order is tested against its own customer's city.
- Cascade again removes any child `order_items` for the deleted orders.
- **Perf note:** an index on `customers(city)` and on `orders(customer_id)` makes this a cheap semi-join delete rather than a nested scan.

#### Alternative Solution — `IN` list
Readable and fine when the subquery cannot produce NULLs (here `customer_id` is `NOT NULL`):
```sql
DELETE FROM orders
WHERE customer_id IN (
    SELECT id FROM customers WHERE city = 'Chicago'
);
```
`IN` reads naturally for simple membership; prefer `NOT EXISTS` over `NOT IN` whenever the "anti" case is involved, because `NOT IN` with a single NULL in the subquery silently deletes nothing.

#### Interview Variations
1. Delete only *cancelled* orders for those customers (`AND o.status = 'cancelled'`).
2. Delete the *customers* too, in dependency-safe order (children first) or via cascade.
3. Use `USING` (Postgres join-delete) instead of a subquery.

#### Common Follow-up Questions
- *`IN` vs `EXISTS` for deletes?* Semantically similar; `EXISTS` is NULL-safe and often better-optimized for correlated cases.
- *Why is `NOT IN` dangerous?* A NULL in the subquery makes the whole predicate `UNKNOWN`, so zero rows match — a classic silent bug.
- *Postgres join-delete syntax?* `DELETE FROM orders o USING customers c WHERE c.id = o.customer_id AND c.city = 'Chicago';`

### Practical Question 18: DELETE vs TRUNCATE — Emptying a Table
- **Difficulty:** Medium
- **Estimated Time:** 6 min
- **Concepts Tested:** `TRUNCATE`, `DELETE` without `WHERE`, DDL vs DML, identity reset, FK constraints

**Problem Statement**
You need to wipe **all** rows from `order_items` before reloading a fresh export. Compare emptying it with `DELETE` versus `TRUNCATE`, and pick the right tool for a full reload.

**Example Output** — after either operation:

| (order_items is empty) |
|------------------------|

**Approach**
1. `DELETE FROM order_items` removes every row as row-by-row DML (logged, fires triggers, MVCC dead tuples left for `VACUUM`).
2. `TRUNCATE order_items` deallocates the table's storage in one metadata operation — far faster for full wipes.
3. Consider FKs: `order_items` is referenced by nothing, so `TRUNCATE` is clean here.

#### SQL Implementation
```sql
-- Fast full wipe + reset any identity sequence, all in one shot:
TRUNCATE TABLE order_items RESTART IDENTITY;

-- Equivalent (slower) DML form:
DELETE FROM order_items;
```
- `TRUNCATE` is effectively **DDL**: it drops and recreates the table's data files instead of scanning and marking each row dead, so it's near-instant regardless of row count and leaves no bloat to `VACUUM`.
- `RESTART IDENTITY` also resets the table's identity/serial counter back to its start — something `DELETE` does *not* do.
- `TRUNCATE` in PostgreSQL **is transactional** (you can `ROLLBACK` it) — a common misconception; in MySQL/Oracle it implicitly commits and cannot be rolled back.
- **Perf note:** for emptying a large table entirely, `TRUNCATE` is O(1)-ish metadata work; `DELETE` is O(rows) plus WAL and index maintenance. Use `DELETE` only when you need a `WHERE`, triggers to fire, or per-row logging.

#### Alternative Solution
When child tables reference this one, truncate the whole related set together:
```sql
TRUNCATE TABLE orders, order_items RESTART IDENTITY CASCADE;
```
`DELETE FROM ... WHERE ...` remains the right tool for *partial* clears; `TRUNCATE ... CASCADE` is canonical for wiping a table plus everything that FKs into it in one atomic step.

#### Interview Variations
1. Empty `orders` (which *is* referenced by `order_items`) — why does bare `TRUNCATE orders` fail, and how does `CASCADE` fix it?
2. Reset identity so the next insert starts at 1.
3. Keep the table but reclaim space after a huge `DELETE` (`VACUUM FULL` / rebuild).

#### Common Follow-up Questions
- *Does `TRUNCATE` fire `DELETE` triggers?* No — row-level `DELETE` triggers do not fire (Postgres has separate `TRUNCATE` triggers); this is a key correctness difference.
- *Can `TRUNCATE` be rolled back?* In PostgreSQL yes (it's transactional); in MySQL/Oracle no (implicit commit).
- *Does `DELETE` reset auto-increment?* No; the sequence keeps climbing. `TRUNCATE ... RESTART IDENTITY` resets it.
- *Which reclaims disk space immediately?* `TRUNCATE`; `DELETE` leaves dead tuples until `VACUUM`.

### Practical Question 19: Safely Delete Duplicates, Keeping One
- **Difficulty:** Hard
- **Estimated Time:** 10 min
- **Concepts Tested:** Deduplication, `ctid`/`ROW_NUMBER`, self-referential delete, keeping the canonical row

**Problem Statement**
Suppose a bad import left duplicate `customers` sharing the same `email` (imagine two rows with `nina.patel@example.com`). Delete the duplicates but **keep the earliest** (lowest `id`) row of each email group. Write a delete that generalizes to any number of duplicates.

**Example Input** — hypothetical dup rows:

| id | name | email |
|----|------|-------|
| 1 | Nina Patel | nina.patel@example.com |
| 20 | Nina P. | nina.patel@example.com |
| 21 | N. Patel | nina.patel@example.com |

**Example Output** — surviving row for that email:

| id | name | email |
|----|------|-------|
| 1 | Nina Patel | nina.patel@example.com |

**Approach**
1. Within each `email` group, rank rows by `id` ascending; the rank-1 row is the keeper.
2. Delete every row whose rank > 1.
3. Use a window function in a CTE, then delete by the identified ids (or use `ctid` for tables without a unique key).

#### SQL Implementation
```sql
WITH ranked AS (
    SELECT id,
           ROW_NUMBER() OVER (
               PARTITION BY email
               ORDER BY id ASC
           ) AS rn
    FROM customers
)
DELETE FROM customers c
USING ranked r
WHERE c.id = r.id
  AND r.rn > 1;
```
- `ROW_NUMBER() ... PARTITION BY email ORDER BY id` numbers each email group; `rn = 1` is the earliest (kept), `rn > 1` are the duplicates to remove — a clean, deterministic "keep one" rule.
- The `DELETE ... USING` correlates the target table to the CTE by `id`, so only the flagged duplicate rows are removed.
- If the table had **no** unique key at all, you'd partition/order by the system column `ctid` instead of `id` to distinguish otherwise-identical rows.
- **Perf note:** this scans the table once to rank; an index on `email` helps the partitioning. After dedup, add a `UNIQUE (email)` constraint to *prevent* recurrence — fixing the cause, not just the symptom.

#### Alternative Solution — self-join keeping MIN(id)
Portable and index-friendly when a natural key exists:
```sql
DELETE FROM customers c
USING customers keep
WHERE c.email = keep.email
  AND c.id   > keep.id;   -- delete any row that has a smaller-id twin
```
The self-join form is concise for the "keep the min" rule; the `ROW_NUMBER` form is the canonical general tool because it extends to arbitrary "keep the newest / highest-scoring / etc." tiebreakers just by changing the `ORDER BY`.

#### Interview Variations
1. Keep the **most recently** created duplicate instead (`ORDER BY created_at DESC`).
2. Dedup on a composite key (`PARTITION BY name, email`).
3. Move deleted duplicates into a quarantine table first.

#### Common Follow-up Questions
- *How do you prevent duplicates recurring?* Add a `UNIQUE` constraint after cleanup; the DB then rejects future dups.
- *What's `ctid`?* PostgreSQL's physical row identifier — useful for dedup when there's no unique/primary key.
- *Why `DELETE ... USING` over `DELETE ... WHERE id IN (subquery)`?* Both work; `USING` (a join-delete) is often clearer and lets the planner use a join strategy.
- *Is this safe under concurrency?* Run it in a transaction; ideally lock the table or add the `UNIQUE` constraint in the same transaction so new dups can't sneak in mid-cleanup.

---


> Hands-on practice built on the **shared schema** (`departments`, `employees`, `customers`, `products`, `orders`, `order_items`). The schema is defined elsewhere — here we only query it.
>
> Flavor: **PostgreSQL**. Where a construct is Postgres-only (e.g. `FILTER`, `::` casts, `date_trunc`), I call out the MySQL equivalent so you don't get burned in a cross-engine assessment.

---

## String / Numeric / Date Functions & NULL Handling

This group is where most candidates quietly lose points. The logic is trivial; the *edge cases* are not. NULL is not a value — it's the absence of one — so every function and comparison behaves differently around it. Master `COALESCE`, `NULLIF`, three-valued logic, and the date/string built-ins and you've covered a huge fraction of "easy" screener questions.

### Practical Question 1: Normalize and Split Customer Emails
- **Difficulty:** Easy
- **Estimated Time:** 6 min
- **Concepts Tested:** `LOWER`, `TRIM`, `SPLIT_PART`/`SUBSTRING`, string functions, data hygiene

**Problem Statement**
The `customers.email` column was populated by several intake forms and contains inconsistent casing and stray whitespace. Return each customer's `id`, a cleaned email (lower-cased, trimmed), and the **email domain** (everything after the `@`). Order by `id`.

**Example Input** — `customers`

| id | name    | email                    |
|----|---------|--------------------------|
| 1  | Alice   | `  Alice@Gmail.com `     |
| 2  | Bob     | `BOB@company.CO `        |
| 3  | Carol   | `carol@shop.io`          |

**Example Output**

| id | clean_email       | domain      |
|----|-------------------|-------------|
| 1  | alice@gmail.com   | gmail.com   |
| 2  | bob@company.co    | company.co  |
| 3  | carol@shop.io     | shop.io     |

**Approach**
1. `TRIM` the surrounding whitespace, then `LOWER` to normalize case.
2. Extract the domain by taking the substring after the `@`.
3. In Postgres, `SPLIT_PART(str, '@', 2)` is the cleanest way to grab the second token.

#### SQL Implementation
```sql
SELECT
    id,
    LOWER(TRIM(email))                        AS clean_email,
    SPLIT_PART(LOWER(TRIM(email)), '@', 2)     AS domain
FROM customers
ORDER BY id;
```

- `TRIM` removes leading/trailing spaces; wrap it *inside* `LOWER` so both the stored value and the extracted domain are normalized identically.
- `SPLIT_PART(text, delimiter, n)` returns the n-th field (1-indexed). Passing `2` returns everything after the first `@`. It returns `''` (not NULL) when the delimiter is absent — worth knowing for validation.
- **Perf note:** This is a projection over a full scan; no index helps here. If you frequently filter by domain, materialize it as a generated column and index that: `ALTER TABLE customers ADD COLUMN domain text GENERATED ALWAYS AS (split_part(lower(email),'@',2)) STORED;`

#### Alternative Solution
Portable ANSI version using `SUBSTRING` + `POSITION` (works on MySQL too, though MySQL calls it `SUBSTRING_INDEX`):
```sql
-- ANSI / broadly portable
SELECT
    id,
    LOWER(TRIM(email)) AS clean_email,
    SUBSTRING(LOWER(TRIM(email)) FROM POSITION('@' IN email) + 1) AS domain
FROM customers
ORDER BY id;

-- MySQL idiom
-- SUBSTRING_INDEX(LOWER(TRIM(email)), '@', -1) AS domain
```

#### Interview Variations
1. Return the **local part** (before the `@`) instead of the domain.
2. Flag rows whose email is missing an `@` as `'INVALID'`.
3. Count how many customers share each domain (leads into `GROUP BY`).

#### Common Follow-up Questions
- **What does `SPLIT_PART` return if there's no `@`?** The whole string for index 1, and an empty string for index 2 — never NULL. So domain-absence shows up as `''`, which you test with `= ''`, not `IS NULL`.
- **Why `TRIM` inside `LOWER` rather than the reverse?** Order is irrelevant for correctness (both are pure functions), but nesting `TRIM` innermost keeps the pattern readable and lets you reuse the trimmed value.
- **How would you validate emails properly?** A regex check (`email ~ '^[^@]+@[^@]+\.[^@]+$'` in Postgres); but real validation is best done at the application layer or via `CHECK` constraints on insert.

### Practical Question 2: Fill Missing Order Totals with COALESCE
- **Difficulty:** Easy
- **Estimated Time:** 6 min
- **Concepts Tested:** `COALESCE`, NULL handling, arithmetic with NULL, `ROUND`

**Problem Statement**
Some legacy `orders` rows have a NULL `total` (the value was never backfilled). For reporting, treat a missing total as `0`. Return each order's `id`, `status`, and a `safe_total` that is the total when present and `0` otherwise, rounded to 2 decimals.

**Example Input** — `orders`

| id | status    | total   |
|----|-----------|---------|
| 10 | paid      | 149.5   |
| 11 | pending   | NULL    |
| 12 | cancelled | 0.00    |
| 13 | paid      | NULL    |

**Example Output**

| id | status    | safe_total |
|----|-----------|------------|
| 10 | paid      | 149.50     |
| 11 | pending   | 0.00       |
| 12 | cancelled | 0.00       |
| 13 | paid      | 0.00       |

**Approach**
1. Wrap `total` in `COALESCE(total, 0)` so NULL collapses to `0`.
2. `ROUND(..., 2)` to enforce two-decimal presentation.
3. Note the semantic difference between a *missing* total (`NULL`) and an *explicit* zero (order 12) — both map to `0.00` here, which the business has to be OK with.

#### SQL Implementation
```sql
SELECT
    id,
    status,
    ROUND(COALESCE(total, 0), 2) AS safe_total
FROM orders
ORDER BY id;
```

- `COALESCE(a, b, c, …)` returns the first non-NULL argument. It's ANSI-standard and short-circuits, so `COALESCE(total, 0)` never evaluates `0` unless `total` is NULL.
- Beware silent NULL propagation: `total + 5` is NULL when `total` is NULL. Any arithmetic on a nullable column should defend with `COALESCE` *before* the operation, e.g. `COALESCE(total,0) + 5`.
- **Perf note:** `COALESCE` on a projected column is free-ish, but it makes a predicate non-sargable. `WHERE COALESCE(total,0) = 0` cannot use an index on `total`; prefer `WHERE total = 0 OR total IS NULL` if you need the index.

#### Alternative Solution
`CASE` makes the intent explicit and lets you distinguish "missing" from "zero" if the business later cares:
```sql
SELECT
    id,
    status,
    CASE WHEN total IS NULL THEN 0 ELSE ROUND(total, 2) END AS safe_total
FROM orders
ORDER BY id;
```
Prefer `COALESCE` for the simple two-way case (less code, clearer intent); reach for `CASE` when the mapping is more than "NULL → default".

#### Interview Variations
1. Distinguish missing vs. zero by also returning a boolean `was_null` flag.
2. Coalesce against a *computed* fallback (e.g. sum of `order_items`) instead of a literal.
3. Replace NULL `status` with `'unknown'` in the same query.

#### Common Follow-up Questions
- **`COALESCE` vs. `ISNULL`/`IFNULL`?** `COALESCE` is ANSI-standard and n-ary; `IFNULL` (MySQL) and `ISNULL` (SQL Server) are two-argument vendor extensions. Prefer `COALESCE` for portability.
- **`COALESCE` vs. `NVL`?** `NVL` is Oracle's two-arg version. Same idea, less portable.
- **Does `COALESCE` short-circuit type-wise?** All arguments must be of a common type; Postgres will try to coerce. `COALESCE(total, 'n/a')` fails because text and numeric don't unify — cast first.

### Practical Question 3: Employee Tenure in Whole Years
- **Difficulty:** Medium
- **Estimated Time:** 8 min
- **Concepts Tested:** Date arithmetic, `AGE`, `EXTRACT`, `date_trunc`, `CURRENT_DATE`, NULL-safe defaults

**Problem Statement**
For each employee, compute completed years of tenure as of today (`hire_date` → today). Return `id`, `name`, `hire_date`, and `years_tenure` (an integer of *completed* years, not rounded up). Employees hired in the future (bad data) should show `0`, never a negative number.

**Example Input** — `employees` (assume today = 2026-07-09)

| id | name    | hire_date  |
|----|---------|------------|
| 1  | Alice   | 2020-01-15 |
| 2  | Bob     | 2025-08-01 |
| 3  | Carol   | 2026-12-01 |

**Example Output**

| id | name  | hire_date  | years_tenure |
|----|-------|------------|--------------|
| 1  | Alice | 2020-01-15 | 6            |
| 2  | Bob   | 2025-08-01 | 0            |
| 3  | Carol | 2026-12-01 | 0            |

**Approach**
1. Use `AGE(CURRENT_DATE, hire_date)` to get a calendar-accurate interval (handles leap years and month lengths).
2. `EXTRACT(YEAR FROM …)` pulls the whole-year component.
3. Guard against future hire dates with `GREATEST(0, …)` so anomalies floor at 0.

#### SQL Implementation
```sql
SELECT
    id,
    name,
    hire_date,
    GREATEST(
        0,
        EXTRACT(YEAR FROM AGE(CURRENT_DATE, hire_date))::int
    ) AS years_tenure
FROM employees
ORDER BY id;
```

- `AGE(a, b)` returns an `interval` like `6 years 5 mons 24 days`; `EXTRACT(YEAR FROM …)` on that interval gives *completed* years — exactly the "anniversary" semantics people expect, unlike a raw day-count division.
- `GREATEST(0, x)` clamps negatives from future-dated rows to zero. `LEAST`/`GREATEST` skip NULL arguments, so if `hire_date` is NULL the whole expression is NULL — decide whether you want that or a `COALESCE(hire_date, CURRENT_DATE)` guard.
- **Perf note:** `CURRENT_DATE` is stable within a statement, so the planner evaluates it once. Computing tenure per-row is cheap; only worry if you filter on it (`WHERE years_tenure > 5` is non-sargable — filter on `hire_date <= CURRENT_DATE - INTERVAL '5 years'` instead to use an index).

#### Alternative Solution
Avoid `AGE` (which MySQL lacks) with a portable month-difference trick. In MySQL: `TIMESTAMPDIFF(YEAR, hire_date, CURDATE())` already returns completed years:
```sql
-- MySQL
SELECT id, name, hire_date,
       GREATEST(0, TIMESTAMPDIFF(YEAR, hire_date, CURDATE())) AS years_tenure
FROM employees;
```
A common *wrong* approach is `(CURRENT_DATE - hire_date)/365` — it drifts on leap years and miscounts around anniversaries. Prefer calendar-aware functions.

#### Interview Variations
1. Return tenure in years *and* months (e.g. "6y 5m").
2. Bucket employees into tenure bands (`<1y`, `1-3y`, `3+y`) using `CASE`.
3. Compute tenure as of the order date rather than today (correlated to another table).

#### Common Follow-up Questions
- **Why not divide day-differences by 365.25?** It's approximately right but can be off by a day around anniversaries and leap boundaries; `AGE`/`TIMESTAMPDIFF` are exact.
- **What happens with a NULL `hire_date`?** `AGE` returns NULL, so `years_tenure` is NULL. Wrap `hire_date` in `COALESCE` or filter those rows out explicitly.
- **`EXTRACT` vs. `date_part`?** Identical in Postgres; `EXTRACT` is the SQL-standard spelling, `date_part` is the function form.

### Practical Question 4: Mask Emails and Handle Blank-vs-NULL
- **Difficulty:** Medium
- **Estimated Time:** 9 min
- **Concepts Tested:** `NULLIF`, `COALESCE`, `CONCAT`, `LEFT`/`REPEAT`, empty-string vs NULL, three-valued logic

**Problem Statement**
Produce a privacy-safe employee directory. For each employee return `id`, `name`, and a `masked_email` where everything before the `@` is replaced by its first character followed by `***`. Some rows have `email = ''` (empty string from a bad import) — treat those the same as NULL and output `'no-email'`.

**Example Input** — `employees`

| id | name  | email               |
|----|-------|---------------------|
| 1  | Alice | alice@corp.com      |
| 2  | Bob   | (empty string)      |
| 3  | Carol | NULL                |

**Example Output**

| id | name  | masked_email    |
|----|-------|-----------------|
| 1  | Alice | a***@corp.com   |
| 2  | Bob   | no-email        |
| 3  | Carol | no-email        |

**Approach**
1. Collapse empty strings to NULL first: `NULLIF(email, '')`. Now "missing" is uniformly NULL.
2. `COALESCE(..., NULL)` stays NULL for missing; build the mask only for present emails.
3. Compose the mask: first character + `'***@'` + domain. Use `COALESCE` at the very end to substitute `'no-email'`.

#### SQL Implementation
```sql
SELECT
    id,
    name,
    COALESCE(
        LEFT(NULLIF(email, ''), 1) || '***@' ||
            SPLIT_PART(NULLIF(email, ''), '@', 2),
        'no-email'
    ) AS masked_email
FROM employees
ORDER BY id;
```

- `NULLIF(email, '')` returns NULL when `email` is the empty string, unifying the two "missing" representations. This is the canonical fix for the empty-string-vs-NULL trap.
- The `||` concatenation operator is **NULL-propagating**: if any operand is NULL, the whole expression is NULL. That's exactly what we want — a NULL email makes the entire mask NULL, and the outer `COALESCE` then supplies `'no-email'`.
- **Perf note:** Pure projection; no index relevance. If you did this a lot, wrap it in a view rather than repeating the expression.

#### Alternative Solution
Some engines (MySQL default) treat `CONCAT` as NULL-*ignoring*, which would break the guard. Be explicit with `CASE` for cross-engine safety:
```sql
SELECT
    id,
    name,
    CASE
        WHEN COALESCE(email, '') = '' THEN 'no-email'
        ELSE CONCAT(LEFT(email, 1), '***@', SUBSTRING_INDEX(email, '@', -1))
    END AS masked_email
FROM employees;
```
`COALESCE(email,'') = ''` is a neat idiom that catches *both* NULL and empty string in one predicate.

#### Interview Variations
1. Mask the domain too, keeping only the TLD (`a***@***.com`).
2. Preserve the last character of the local part as well (`a***e@…`).
3. Only mask when the viewer lacks a `pii_access` flag (parameterized).

#### Common Follow-up Questions
- **Why does `'' = NULL` not work to find blanks?** Any comparison to NULL yields *unknown*, not true — so `email = NULL` never matches. Use `IS NULL`, and normalize blanks with `NULLIF` first.
- **`NULLIF(a, b)` — what exactly does it return?** NULL if `a = b`, otherwise `a`. It's shorthand for `CASE WHEN a = b THEN NULL ELSE a END`.
- **Is `||` or `CONCAT` safer around NULLs?** In Postgres `||` propagates NULL and `CONCAT` ignores it; in MySQL `CONCAT` ignores NULL and `||` means logical OR. Know your engine — the difference silently changes results.

## Aggregate Functions (COUNT/SUM/AVG/MIN/MAX)

Aggregates collapse a set of rows into a single value. The recurring traps: `COUNT(*)` vs. `COUNT(col)` vs. `COUNT(DISTINCT col)`, how every aggregate **silently skips NULLs** (except `COUNT(*)`), integer-division surprises in `AVG`, and the difference between "no matching rows" (aggregate returns NULL, or 0 for COUNT) and a matching row with a NULL value. Conditional aggregation with `FILTER`/`CASE` also lives here and shows up constantly.

### Practical Question 5: Company-Wide Salary Statistics
- **Difficulty:** Easy
- **Estimated Time:** 5 min
- **Concepts Tested:** `COUNT`, `AVG`, `MIN`, `MAX`, `SUM`, `ROUND`, NULL skipping

**Problem Statement**
Return a one-row summary of the `employees` table: total headcount, total payroll, average salary (rounded to 2 decimals), and the minimum and maximum salary.

**Example Input** — `employees`

| id | name  | salary  |
|----|-------|---------|
| 1  | Alice | 90000   |
| 2  | Bob   | 60000   |
| 3  | Carol | 75000   |
| 4  | Dan   | NULL    |

**Example Output**

| headcount | total_payroll | avg_salary | min_salary | max_salary |
|-----------|---------------|------------|------------|------------|
| 4         | 225000        | 75000.00   | 60000      | 90000      |

**Approach**
1. `COUNT(*)` counts all rows including Dan (headcount is about people, not salaries).
2. `SUM`, `AVG`, `MIN`, `MAX` all ignore Dan's NULL salary automatically.
3. Note `AVG` divides by 3 (non-NULL salaries), not 4 — that's usually what you want, but it's a classic gotcha.

#### SQL Implementation
```sql
SELECT
    COUNT(*)              AS headcount,
    SUM(salary)           AS total_payroll,
    ROUND(AVG(salary), 2) AS avg_salary,
    MIN(salary)           AS min_salary,
    MAX(salary)           AS max_salary
FROM employees;
```

- `COUNT(*)` counts rows; `COUNT(salary)` would return 3 (skips the NULL). Choose deliberately based on whether you're counting *entities* or *non-null values*.
- `AVG(salary)` = `SUM(salary) / COUNT(salary)` = `225000 / 3` = `75000`. If you actually want to treat missing salary as 0, use `AVG(COALESCE(salary, 0))`, which divides by 4 and yields `56250`.
- **Perf note:** These are full-table aggregates — a sequential scan is expected. On huge tables, a covering index on `salary` lets Postgres do an index-only scan; `MIN`/`MAX` alone can use a btree index endpoint in O(log n).

#### Alternative Solution
If the business wants NULL salaries counted as zero in the average, be explicit:
```sql
SELECT
    COUNT(*)                          AS headcount,
    COALESCE(SUM(salary), 0)          AS total_payroll,
    ROUND(AVG(COALESCE(salary, 0)),2) AS avg_salary,
    MIN(salary)                       AS min_salary,
    MAX(salary)                       AS max_salary
FROM employees;
```
The `COALESCE(SUM(...), 0)` guard matters when the table might be empty — see the follow-up below.

#### Interview Variations
1. Add a `salary_range` column = `MAX - MIN`.
2. Return the same stats but only for employees hired in the last 2 years.
3. Compute the median salary (Postgres `PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary)`).

#### Common Follow-up Questions
- **What do aggregates return on an empty table?** `COUNT` returns `0`; every other aggregate returns `NULL`. That's why report totals are wrapped in `COALESCE(SUM(x), 0)`.
- **`COUNT(*)` vs. `COUNT(1)` — performance difference?** None in modern Postgres/MySQL; the planner treats them identically. `COUNT(*)` is the idiomatic spelling.
- **Does `AVG` on integers truncate?** In Postgres `AVG` of an `integer` column returns `numeric`, so no truncation. In some engines `SUM/COUNT` of ints does integer division — cast to a decimal to be safe.

### Practical Question 6: Distinct Customers Who Placed Orders
- **Difficulty:** Easy
- **Estimated Time:** 5 min
- **Concepts Tested:** `COUNT(DISTINCT …)`, `COUNT(*)` vs `COUNT(col)`, deduplication

**Problem Statement**
From the `orders` table, report three numbers: the total number of orders, the number of **distinct** customers who have ordered, and the number of distinct order statuses in use.

**Example Input** — `orders`

| id | customer_id | status  |
|----|-------------|---------|
| 1  | 100         | paid    |
| 2  | 100         | paid    |
| 3  | 101         | pending |
| 4  | 102         | paid    |

**Example Output**

| total_orders | distinct_customers | distinct_statuses |
|--------------|--------------------|-------------------|
| 4            | 3                  | 2                 |

**Approach**
1. `COUNT(*)` for total orders (every row is an order).
2. `COUNT(DISTINCT customer_id)` dedups repeat buyers — customer 100 counts once.
3. `COUNT(DISTINCT status)` counts the distinct status vocabulary actually present.

#### SQL Implementation
```sql
SELECT
    COUNT(*)                       AS total_orders,
    COUNT(DISTINCT customer_id)    AS distinct_customers,
    COUNT(DISTINCT status)         AS distinct_statuses
FROM orders;
```

- `COUNT(DISTINCT col)` first removes duplicates *and* NULLs, then counts. So a customer with 5 orders contributes 1; a NULL `customer_id` contributes 0.
- Distinguish the three forms clearly: `COUNT(*)` = rows, `COUNT(col)` = non-null values, `COUNT(DISTINCT col)` = distinct non-null values. Assessments love probing this.
- **Perf note:** `COUNT(DISTINCT …)` requires a sort or hash to dedup and can be expensive on large tables. An index on `customer_id` helps; for approximate answers at scale, Postgres offers `HLL`-style extensions or you can pre-aggregate.

#### Alternative Solution
`COUNT(DISTINCT …)` over many columns in one pass can be slow; an equivalent is to dedup in a subquery:
```sql
SELECT COUNT(*) AS distinct_customers
FROM (SELECT DISTINCT customer_id FROM orders WHERE customer_id IS NOT NULL) d;
```
For a single distinct count this is usually the same plan; the subquery form shines when you need `DISTINCT` across a *combination* of columns, e.g. `COUNT(DISTINCT (customer_id, status))` which older MySQL can't do inline.

#### Interview Variations
1. Count distinct customers who placed a **paid** order only.
2. Report the average number of orders per customer (`COUNT(*) * 1.0 / COUNT(DISTINCT customer_id)`).
3. Find customers in `customers` who have *never* ordered (anti-join / `NOT EXISTS`).

#### Common Follow-up Questions
- **Does `COUNT(DISTINCT col)` count NULLs?** No. NULLs are excluded before deduping, just like plain `COUNT(col)`.
- **Why is `COUNT(DISTINCT …)` slower than `COUNT(*)`?** It must materialize and dedup the value set (sort/hash), whereas `COUNT(*)` just tallies rows.
- **Can I `COUNT(DISTINCT a, b)` in Postgres?** Not directly, but `COUNT(DISTINCT (a, b))` works using a row constructor; MySQL supports `COUNT(DISTINCT a, b)` with its own syntax.

### Practical Question 7: Order Status Breakdown with Conditional Aggregation
- **Difficulty:** Medium
- **Estimated Time:** 9 min
- **Concepts Tested:** `FILTER` clause, `CASE` inside aggregates, pivoting, `SUM(...) FILTER`

**Problem Statement**
Produce a single-row dashboard tile over `orders`: the count of paid orders, count of pending orders, count of cancelled orders, and the **total revenue from paid orders only**. Do it in one pass over the table (no self-joins, no multiple subqueries).

**Example Input** — `orders`

| id | status    | total  |
|----|-----------|--------|
| 1  | paid      | 100    |
| 2  | pending   | 50     |
| 3  | paid      | 200    |
| 4  | cancelled | 80     |

**Example Output**

| paid_cnt | pending_cnt | cancelled_cnt | paid_revenue |
|----------|-------------|---------------|--------------|
| 2        | 1           | 1             | 300          |

**Approach**
1. This is a *pivot*: turn row values (`status`) into columns. Conditional aggregation is the tool.
2. In Postgres, `COUNT(*) FILTER (WHERE status = 'paid')` reads cleanly and is the idiomatic form.
3. For revenue, `SUM(total) FILTER (WHERE status = 'paid')` restricts the sum to paid rows only.

#### SQL Implementation
```sql
SELECT
    COUNT(*) FILTER (WHERE status = 'paid')       AS paid_cnt,
    COUNT(*) FILTER (WHERE status = 'pending')    AS pending_cnt,
    COUNT(*) FILTER (WHERE status = 'cancelled')  AS cancelled_cnt,
    COALESCE(SUM(total) FILTER (WHERE status = 'paid'), 0) AS paid_revenue
FROM orders;
```

- The `FILTER (WHERE …)` clause is SQL-standard and Postgres-native: it applies a per-aggregate predicate so different columns can aggregate different subsets in the *same* scan. It's more readable than the `CASE` trick and lets the planner optimize.
- Wrap the filtered `SUM` in `COALESCE(..., 0)` because if there are zero paid orders the `SUM` is NULL, not 0.
- **Perf note:** One sequential scan computes all four metrics — far better than four separate `WHERE`-filtered queries. **MySQL lacks `FILTER`**; use the `CASE` form shown below.

#### Alternative Solution
The portable `CASE`-inside-aggregate idiom (works everywhere, including MySQL):
```sql
SELECT
    COUNT(CASE WHEN status = 'paid'      THEN 1 END) AS paid_cnt,
    COUNT(CASE WHEN status = 'pending'   THEN 1 END) AS pending_cnt,
    COUNT(CASE WHEN status = 'cancelled' THEN 1 END) AS cancelled_cnt,
    COALESCE(SUM(CASE WHEN status = 'paid' THEN total END), 0) AS paid_revenue
FROM orders;
```
Key subtlety: the `CASE` has **no `ELSE`**, so non-matching rows produce NULL, and `COUNT`/`SUM` skip NULLs — giving exactly the filtered count/sum. Writing `ELSE 0` would break the `COUNT` (it would count every row).

#### Interview Variations
1. Add `avg_paid_order = AVG(total) FILTER (WHERE status='paid')`.
2. Compute paid revenue as a **percentage** of total revenue.
3. Break the same metrics down per month (moves into `GROUP BY`).

#### Common Follow-up Questions
- **Why `COUNT(CASE … THEN 1 END)` and not `SUM(CASE … THEN 1 ELSE 0 END)`?** Both work for counting. `SUM(... ELSE 0)` is the classic portable form; `COUNT(... no ELSE)` relies on NULL-skipping. Just don't mix an `ELSE 0` into a `COUNT`.
- **Is `FILTER` faster than `CASE`?** Same single-scan complexity; `FILTER` is mainly a readability/standards win and occasionally enables better planning.
- **What if `status` can be NULL?** Neither branch matches, so a NULL-status row is excluded from all buckets — which is usually correct. Add an explicit `WHERE status IS NULL` bucket if you need it.

### Practical Question 8: Highest and Lowest Priced Product per Category Value
- **Difficulty:** Medium
- **Estimated Time:** 8 min
- **Concepts Tested:** `MIN`/`MAX`, `SUM` of a product, revenue = qty × price, aggregate over a join

**Problem Statement**
Across all `order_items`, compute the total gross revenue (`SUM(quantity * unit_price)`), the single most expensive line item's value, the cheapest line item's value, and the average line-item value. Round money to 2 decimals.

**Example Input** — `order_items`

| id | order_id | quantity | unit_price |
|----|----------|----------|------------|
| 1  | 10       | 2        | 25.00      |
| 2  | 10       | 1        | 100.00     |
| 3  | 11       | 5        | 10.00      |

**Example Output**

| gross_revenue | max_line | min_line | avg_line |
|---------------|----------|----------|----------|
| 200.00        | 100.00   | 50.00    | 66.67    |

**Approach**
1. The per-line value is a computed expression `quantity * unit_price` — aggregate over the *expression*, not a stored column.
2. `SUM`, `MAX`, `MIN`, `AVG` all take that expression. Line values are 50, 100, 50.
3. Round the money columns; `AVG` = 200/3 = 66.67.

#### SQL Implementation
```sql
SELECT
    ROUND(SUM(quantity * unit_price), 2) AS gross_revenue,
    ROUND(MAX(quantity * unit_price), 2) AS max_line,
    ROUND(MIN(quantity * unit_price), 2) AS min_line,
    ROUND(AVG(quantity * unit_price), 2) AS avg_line
FROM order_items;
```

- Aggregates accept arbitrary scalar expressions, not just bare columns. `SUM(quantity * unit_price)` is the canonical "revenue" pattern — never store what you can derive.
- If either `quantity` or `unit_price` can be NULL, the product is NULL and that line is skipped by every aggregate. Guard with `COALESCE(quantity,0) * COALESCE(unit_price,0)` if a missing value should mean zero.
- **Perf note:** Full scan of `order_items`. Because the aggregated value is computed, no plain index helps; an *expression index* on `(quantity * unit_price)` could, but is rarely worth it for a full-table sum.

#### Alternative Solution
If you also need revenue *net* of a per-line discount stored elsewhere, push the arithmetic into a derived table first for clarity:
```sql
SELECT ROUND(SUM(line_value), 2) AS gross_revenue,
       ROUND(MAX(line_value), 2) AS max_line,
       ROUND(MIN(line_value), 2) AS min_line,
       ROUND(AVG(line_value), 2) AS avg_line
FROM (
    SELECT quantity * unit_price AS line_value
    FROM order_items
) t;
```
Same result; the CTE/subquery version keeps the expression DRY when it appears in many aggregates.

#### Interview Variations
1. Restrict to line items belonging to `paid` orders (join to `orders`).
2. Return revenue per `product_id` (moves into `GROUP BY`).
3. Weight the average by quantity (weighted average price).

#### Common Follow-up Questions
- **Does `AVG(quantity*unit_price)` weight by quantity?** No — it averages the *line totals* equally. A quantity-weighted unit price would be `SUM(quantity*unit_price)/SUM(quantity)`.
- **Why round at the end, not per row?** Rounding each line then summing accumulates rounding error; aggregate first, round once for reporting.
- **What if unit_price differs from the products table price?** `order_items.unit_price` is the historical price at sale time — always aggregate the line's own `unit_price`, not the current `products.price`.

## GROUP BY

`GROUP BY` partitions rows into buckets and runs the aggregates once per bucket. The rules that trip people up: every column in the `SELECT` list must be either inside an aggregate or in the `GROUP BY` (Postgres enforces this; MySQL historically didn't, producing nondeterministic results); NULL forms its own group; and grouping by a joined dimension changes the grain of your result. This section covers the bread-and-butter reporting queries: per-department, per-city, per-month rollups.

### Practical Question 9: Average Salary per Department
- **Difficulty:** Easy
- **Estimated Time:** 6 min
- **Concepts Tested:** `GROUP BY`, `AVG`, `COUNT`, join to a dimension, `ROUND`

**Problem Statement**
For every department, report the department name, the number of employees, and their average salary (rounded to 2 decimals). Sort by average salary descending.

**Example Input** — `employees`

| id | name  | department_id | salary |
|----|-------|---------------|--------|
| 1  | Alice | 1             | 90000  |
| 2  | Bob   | 1             | 70000  |
| 3  | Carol | 2             | 60000  |

`departments`

| id | name         |
|----|--------------|
| 1  | Engineering  |
| 2  | Sales        |

**Example Output**

| department  | headcount | avg_salary |
|-------------|-----------|------------|
| Engineering | 2         | 80000.00   |
| Sales       | 1         | 60000.00   |

**Approach**
1. Join `employees` to `departments` so we can show the readable name.
2. `GROUP BY` the department (id and/or name) to form one bucket per department.
3. `AVG(salary)` and `COUNT(*)` per bucket, then `ORDER BY avg_salary DESC`.

#### SQL Implementation
```sql
SELECT
    d.name                     AS department,
    COUNT(*)                   AS headcount,
    ROUND(AVG(e.salary), 2)    AS avg_salary
FROM employees e
JOIN departments d ON d.id = e.department_id
GROUP BY d.id, d.name
ORDER BY avg_salary DESC;
```

- Group by `d.id` (the primary key) and include `d.name` too. Grouping by the PK is the safe habit: two departments could share a name, and Postgres lets you `SELECT d.name` when you've grouped by `d.id` because name is functionally dependent on the PK.
- `AVG(e.salary)` ignores NULL salaries within each group. If a department's employees all have NULL salary, `AVG` is NULL for that group — decide whether to `COALESCE` it.
- **Perf note:** An index on `employees(department_id)` speeds the join/grouping; for large tables Postgres may pick a HashAggregate. Adding `salary` to that index (`(department_id, salary)`) enables an index-only scan.

#### Alternative Solution
If you want to include departments that currently have **no employees** (headcount 0), switch to a `LEFT JOIN` from the departments side:
```sql
SELECT
    d.name AS department,
    COUNT(e.id) AS headcount,                 -- COUNT(e.id), not COUNT(*)
    ROUND(AVG(e.salary), 2) AS avg_salary
FROM departments d
LEFT JOIN employees e ON e.department_id = d.id
GROUP BY d.id, d.name
ORDER BY avg_salary DESC NULLS LAST;
```
Critical detail: with a `LEFT JOIN`, empty departments produce one all-NULL employee row, so you must use `COUNT(e.id)` (yields 0) rather than `COUNT(*)` (would wrongly yield 1).

#### Interview Variations
1. Include only departments with more than N employees (leads into `HAVING`).
2. Add the min and max salary per department.
3. Show each department's average as a percentage of the company-wide average (window or subquery).

#### Common Follow-up Questions
- **Why must every non-aggregated SELECT column be in `GROUP BY`?** Otherwise the value is ambiguous — which row's value would it show? Postgres rejects it; MySQL (pre-`ONLY_FULL_GROUP_BY`) silently picked an arbitrary row, a notorious source of bugs.
- **Can I `GROUP BY` a column alias or position?** Postgres allows `GROUP BY 1` (ordinal) and aliases from the select list in some cases; it's terser but less readable — prefer explicit expressions in production.
- **Where do NULL department_ids go?** They form a single NULL group under an inner join only if the join matches — with an inner join they're dropped; with a left/right join a NULL bucket can appear.

### Practical Question 10: Count Customers per City
- **Difficulty:** Easy
- **Estimated Time:** 5 min
- **Concepts Tested:** `GROUP BY`, `COUNT`, NULL grouping, ordering aggregates

**Problem Statement**
Report how many customers are registered in each city, most populous city first. Cities are stored in `customers.city`. Some customers have a NULL city — surface them as a distinct group labelled `'(unknown)'`.

**Example Input** — `customers`

| id | name  | city    |
|----|-------|---------|
| 1  | Alice | Austin  |
| 2  | Bob   | Austin  |
| 3  | Carol | Dallas  |
| 4  | Dan   | NULL    |

**Example Output**

| city      | customer_count |
|-----------|----------------|
| Austin    | 2              |
| Dallas    | 1              |
| (unknown) | 1              |

**Approach**
1. `GROUP BY city` — NULL cities collapse into their own single group.
2. `COUNT(*)` per group gives the population.
3. `COALESCE(city, '(unknown)')` for a friendly label; order by count descending.

#### SQL Implementation
```sql
SELECT
    COALESCE(city, '(unknown)') AS city,
    COUNT(*)                    AS customer_count
FROM customers
GROUP BY city
ORDER BY customer_count DESC, city;
```

- `GROUP BY city` treats **all NULLs as one group** (SQL's grouping semantics differ from `=`, which never matches NULL). That's why the unknown bucket aggregates correctly.
- Group by the raw `city`, not `COALESCE(city,'(unknown)')` — grouping by the bare column lets an index on `city` be used; the `COALESCE` is only for display in the select list. (Grouping by the expression also works and is fine if there's no index.)
- The tiebreaker `, city` in the `ORDER BY` makes output deterministic when two cities have equal counts.
- **Perf note:** An index on `customers(city)` enables a fast grouped aggregate; without it, expect a scan + HashAggregate.

#### Alternative Solution
If you must group by the display expression (e.g. the engine can't group by a column not in select), that's equally correct:
```sql
SELECT
    COALESCE(city, '(unknown)') AS city,
    COUNT(*) AS customer_count
FROM customers
GROUP BY COALESCE(city, '(unknown)')
ORDER BY customer_count DESC;
```
Prefer grouping by the bare column when an index exists; group by the expression when you need the NULL and any empty-string `''` to fold into the same bucket (then use `COALESCE(NULLIF(city,''), '(unknown)')`).

#### Interview Variations
1. Only show cities with at least 10 customers (`HAVING`).
2. Count *distinct* email domains per city instead of raw customers.
3. Show the newest signup date per city (`MAX(created_at)`).

#### Common Follow-up Questions
- **Does `GROUP BY` put all NULLs together?** Yes — grouping uses "not distinct" semantics, so every NULL lands in one group, unlike equality comparison which never matches NULL.
- **How do empty strings interact with NULL here?** `''` and NULL are different groups. If bad data mixes them, normalize with `NULLIF(city,'')` inside the group key so blanks fold into unknown.
- **Why order by the aggregate alias — is that allowed?** `ORDER BY` runs after `SELECT`, so it can reference output aliases like `customer_count`. (`WHERE`/`GROUP BY` cannot, because they run earlier.)

### Practical Question 11: Monthly Revenue Report
- **Difficulty:** Medium
- **Estimated Time:** 10 min
- **Concepts Tested:** `date_trunc`/`EXTRACT`, `GROUP BY` a derived column, `SUM`, time-series rollup, `COALESCE`

**Problem Statement**
Produce a monthly revenue report from paid orders: for each calendar month, show the month (as the first day of that month) and the total revenue. Only include orders with `status = 'paid'`. Sort chronologically.

**Example Input** — `orders`

| id | order_date | status | total |
|----|------------|--------|-------|
| 1  | 2026-01-05 | paid   | 100   |
| 2  | 2026-01-20 | paid   | 150   |
| 3  | 2026-02-02 | paid   | 200   |
| 4  | 2026-02-10 | pending| 999   |

**Example Output**

| month      | revenue |
|------------|---------|
| 2026-01-01 | 250     |
| 2026-02-01 | 200     |

**Approach**
1. Bucket each order into a month with `date_trunc('month', order_date)`.
2. Filter to `status = 'paid'` in `WHERE` (before aggregation — cheaper and correct).
3. `GROUP BY` the truncated month, `SUM(total)`, order by month.

#### SQL Implementation
```sql
SELECT
    date_trunc('month', order_date)::date AS month,
    COALESCE(SUM(total), 0)               AS revenue
FROM orders
WHERE status = 'paid'
GROUP BY date_trunc('month', order_date)
ORDER BY month;
```

- `date_trunc('month', ts)` floors a timestamp/date to the first instant of its month — the canonical way to bucket a time series. Cast `::date` for a clean `YYYY-MM-01` label.
- Put `status = 'paid'` in `WHERE`, not `HAVING`: it's a row-level filter that should run *before* grouping, shrinking the input and (if indexed) using an index. `HAVING` is only for predicates on the aggregated groups.
- **Perf note:** A composite index on `orders(status, order_date)` lets Postgres filter to paid rows and read them in date order, enabling an efficient grouped aggregate. For recurring dashboards, consider a materialized view refreshed nightly.

#### Alternative Solution
Portable/MySQL version using `EXTRACT` or `DATE_FORMAT` to build a `YYYY-MM` key:
```sql
-- Postgres, textual month key
SELECT to_char(order_date, 'YYYY-MM') AS month, SUM(total) AS revenue
FROM orders WHERE status = 'paid'
GROUP BY to_char(order_date, 'YYYY-MM')
ORDER BY month;

-- MySQL
-- SELECT DATE_FORMAT(order_date, '%Y-%m') AS month, SUM(total) AS revenue
-- FROM orders WHERE status='paid' GROUP BY 1 ORDER BY 1;
```
`date_trunc` keeps the value as a real date (better for sorting/joining to a calendar); `to_char`/`DATE_FORMAT` produce a string label. Use the date form when you'll join against a month dimension or fill gaps.

#### Interview Variations
1. Fill missing months (zero-revenue months) using a `generate_series` calendar `LEFT JOIN`.
2. Add a running cumulative revenue column (`SUM(...) OVER (ORDER BY month)`).
3. Break revenue down by month **and** status (multi-column `GROUP BY`).

#### Common Follow-up Questions
- **Why doesn't a month with zero paid orders appear?** Grouping only produces rows for data that exists. To show zero months, generate a calendar with `generate_series` and `LEFT JOIN` the aggregated revenue, wrapping `SUM` in `COALESCE(...,0)`.
- **Should the `status` filter be in `WHERE` or `HAVING`?** `WHERE` — it's per-row and runs before aggregation. `HAVING` would work but scans/aggregates more rows than necessary and is semantically wrong for a non-aggregate condition.
- **Timezone concerns with `date_trunc`?** For `timestamptz`, `date_trunc` uses the session timezone; specify one explicitly (`date_trunc('month', order_date AT TIME ZONE 'UTC')`) to avoid month-boundary drift across regions.

### Practical Question 12: Revenue per Product Category
- **Difficulty:** Medium
- **Estimated Time:** 10 min
- **Concepts Tested:** multi-table join, `GROUP BY` on a joined dimension, `SUM` of qty×price, grain awareness

**Problem Statement**
For each product category, compute total units sold and total revenue (`quantity * unit_price`) across all order line items. Include only line items from `paid` orders. Sort by revenue descending.

**Example Input**

`products`

| id | name    | category    | price |
|----|---------|-------------|-------|
| 1  | Widget  | Hardware    | 25    |
| 2  | Gadget  | Hardware    | 40    |
| 3  | eBook   | Digital     | 10    |

`order_items`

| id | order_id | product_id | quantity | unit_price |
|----|----------|------------|----------|------------|
| 1  | 100      | 1          | 2        | 25         |
| 2  | 100      | 3          | 1        | 10         |
| 3  | 101      | 2          | 3        | 40         |

`orders`: order 100 = paid, order 101 = paid.

**Example Output**

| category | units_sold | revenue |
|----------|------------|---------|
| Hardware | 5          | 170     |
| Digital  | 1          | 10      |

**Approach**
1. Join `order_items` → `products` (for category) and → `orders` (to filter paid).
2. Filter `orders.status = 'paid'` in `WHERE`.
3. `GROUP BY p.category`; `SUM(quantity)` for units and `SUM(quantity * unit_price)` for revenue.

#### SQL Implementation
```sql
SELECT
    p.category,
    SUM(oi.quantity)                    AS units_sold,
    SUM(oi.quantity * oi.unit_price)    AS revenue
FROM order_items oi
JOIN products p ON p.id = oi.product_id
JOIN orders   o ON o.id = oi.order_id
WHERE o.status = 'paid'
GROUP BY p.category
ORDER BY revenue DESC;
```

- Be conscious of **grain**: `order_items` is the finest grain (one row per line), so we aggregate there and pull `category` from the joined `products`. Grouping at the wrong grain (e.g. after a fan-out join) double-counts.
- Filtering `o.status = 'paid'` in `WHERE` prunes non-paid line items before aggregation. The join to `orders` exists solely for this predicate.
- Revenue uses `oi.unit_price` (historical sale price), *not* `p.price` (current catalog price) — a classic correctness trap in reporting.
- **Perf note:** Indexes on `order_items(order_id)`, `order_items(product_id)`, and `orders(id, status)` support the joins and filter. Expect hash joins + HashAggregate on large data.

#### Alternative Solution
Pre-filter paid orders in a CTE to make the intent explicit and let the optimizer prune early:
```sql
WITH paid AS (
    SELECT id FROM orders WHERE status = 'paid'
)
SELECT p.category,
       SUM(oi.quantity)                 AS units_sold,
       SUM(oi.quantity * oi.unit_price) AS revenue
FROM order_items oi
JOIN paid     ON paid.id = oi.order_id
JOIN products p ON p.id = oi.product_id
GROUP BY p.category
ORDER BY revenue DESC;
```
Functionally identical; the CTE reads better when the paid-order logic grows more complex. In modern Postgres CTEs are inlined, so there's no performance penalty here.

#### Interview Variations
1. Add average selling price per category (`SUM(qty*price)/SUM(qty)`).
2. Show categories with zero paid sales (`LEFT JOIN` from products).
3. Rank categories and keep only the top 3 (`ORDER BY … LIMIT` or window `RANK`).

#### Common Follow-up Questions
- **Why join `orders` at all if we only need the status?** Because the paid/pending flag lives on the order, not the line item. The join is a filter bridge; nothing from `orders` appears in the output.
- **Could this double-count units?** Only if a join fans out. Each `order_item` matches exactly one product and one order, so the grain stays at one row per line — no double counting.
- **What if a product's category is NULL?** Those rows form a NULL group. `COALESCE(p.category,'Uncategorized')` gives it a label if desired.

## HAVING

`HAVING` filters *groups* after aggregation, the way `WHERE` filters rows before it. The single most important mental model: `WHERE` → `GROUP BY` → `HAVING` → `SELECT` → `ORDER BY`. If a predicate references an aggregate (`COUNT(*) > 5`, `SUM(total) > 1000`), it belongs in `HAVING`; if it references a raw row value, it belongs in `WHERE`. Putting row filters in `HAVING` still works but scans more than necessary; putting aggregate filters in `WHERE` is a hard error.

### Practical Question 13: Departments with More Than N Employees
- **Difficulty:** Easy
- **Estimated Time:** 6 min
- **Concepts Tested:** `HAVING`, `COUNT`, group filtering, `WHERE` vs `HAVING`

**Problem Statement**
List departments that have **more than 5 employees**. Return the department name and its headcount, largest first.

**Example Input** — `employees` (department_id shown)

| id | department_id |
|----|---------------|
| …  | 1 (×7 rows)   |
| …  | 2 (×3 rows)   |
| …  | 3 (×6 rows)   |

**Example Output**

| department  | headcount |
|-------------|-----------|
| Engineering | 7         |
| Support     | 6         |

**Approach**
1. Group employees by department and `COUNT(*)` each group.
2. Keep only groups whose count exceeds 5 using `HAVING COUNT(*) > 5`.
3. Order by headcount descending.

#### SQL Implementation
```sql
SELECT
    d.name       AS department,
    COUNT(*)     AS headcount
FROM employees e
JOIN departments d ON d.id = e.department_id
GROUP BY d.id, d.name
HAVING COUNT(*) > 5
ORDER BY headcount DESC;
```

- `HAVING COUNT(*) > 5` filters *after* the groups are formed — you cannot express this in `WHERE` because the count doesn't exist until aggregation happens.
- You can repeat the aggregate in `HAVING` (`COUNT(*)`) even though the alias `headcount` isn't visible there — `HAVING` runs before the `SELECT` list's aliases are assigned. (Postgres tolerates the alias in some cases, but repeating the expression is the portable habit.)
- **Perf note:** `HAVING` doesn't reduce the work of grouping — every group is still computed, then filtered. It can't use an index to skip groups. Push any row-level predicates into `WHERE` to shrink the input first.

#### Alternative Solution
Parameterize N and combine with a row-level filter to show the correct ordering of the pipeline:
```sql
SELECT d.name AS department, COUNT(*) AS headcount
FROM employees e
JOIN departments d ON d.id = e.department_id
WHERE e.salary IS NOT NULL          -- row filter: runs first
GROUP BY d.id, d.name
HAVING COUNT(*) > 5                  -- group filter: runs after aggregation
ORDER BY headcount DESC;
```
Note the two filters do different jobs: `WHERE` removes ineligible *rows* pre-aggregation; `HAVING` removes ineligible *groups* post-aggregation.

#### Interview Variations
1. Departments with more than 5 employees **and** average salary above 80k (compound `HAVING`).
2. Departments with between 3 and 10 employees (`HAVING COUNT(*) BETWEEN 3 AND 10`).
3. The single largest department only (`ORDER BY … LIMIT 1`).

#### Common Follow-up Questions
- **Why can't `COUNT(*) > 5` go in `WHERE`?** `WHERE` is evaluated per row before any grouping, so the aggregate doesn't exist yet. `HAVING` exists precisely to filter on aggregates.
- **Can `HAVING` reference a column not in `GROUP BY`?** Only inside an aggregate. A bare non-grouped column in `HAVING` is the same error as in `SELECT`.
- **Is `HAVING` without `GROUP BY` valid?** Yes — it treats the whole table as one group, e.g. `SELECT COUNT(*) FROM orders HAVING COUNT(*) > 100` returns a row only if the condition holds.

### Practical Question 14: Repeat Customers (More Than One Order)
- **Difficulty:** Easy
- **Estimated Time:** 6 min
- **Concepts Tested:** `HAVING COUNT(*) > 1`, grouping by a foreign key, duplicate detection

**Problem Statement**
Identify repeat customers — those who have placed more than one order. Return the customer id and their order count, most frequent first.

**Example Input** — `orders`

| id | customer_id |
|----|-------------|
| 1  | 100         |
| 2  | 100         |
| 3  | 101         |
| 4  | 100         |

**Example Output**

| customer_id | order_count |
|-------------|-------------|
| 100         | 3           |

**Approach**
1. Group `orders` by `customer_id`.
2. `COUNT(*)` orders per customer.
3. Keep groups with `COUNT(*) > 1`.

#### SQL Implementation
```sql
SELECT
    customer_id,
    COUNT(*) AS order_count
FROM orders
WHERE customer_id IS NOT NULL
GROUP BY customer_id
HAVING COUNT(*) > 1
ORDER BY order_count DESC;
```

- This is the canonical **duplicate/repeat detection** pattern: `GROUP BY key HAVING COUNT(*) > 1`. Memorize it — it appears in "find duplicate emails", "find repeat buyers", "detect double-booked slots", etc.
- The `WHERE customer_id IS NOT NULL` keeps the NULL bucket (orders with no customer) out of the result — a NULL group could otherwise show up with a misleading count.
- **Perf note:** An index on `orders(customer_id)` makes the grouped aggregate efficient. To also fetch customer names, join to `customers` *after* this aggregation (or wrap it in a CTE) so you don't inflate the grouping.

#### Alternative Solution
Join to `customers` to return names, keeping the aggregation clean via a CTE:
```sql
WITH repeat_customers AS (
    SELECT customer_id, COUNT(*) AS order_count
    FROM orders
    WHERE customer_id IS NOT NULL
    GROUP BY customer_id
    HAVING COUNT(*) > 1
)
SELECT c.id, c.name, rc.order_count
FROM repeat_customers rc
JOIN customers c ON c.id = rc.customer_id
ORDER BY rc.order_count DESC;
```
Aggregate first, decorate with names second — this keeps the grouping grain correct and avoids grouping by extra customer columns.

#### Interview Variations
1. Repeat customers among **paid** orders only (add `WHERE status='paid'`).
2. Customers whose *total* spend exceeds a threshold (`HAVING SUM(total) > 1000`).
3. Customers with orders in more than one distinct month (`HAVING COUNT(DISTINCT date_trunc('month', order_date)) > 1`).

#### Common Follow-up Questions
- **`> 1` vs `>= 2` — any difference?** None; they're equivalent for integer counts. Use whichever reads clearer to you.
- **How would you find one-time customers instead?** `HAVING COUNT(*) = 1`. And customers with *zero* orders need an anti-join (`NOT EXISTS`) against `customers`, since they have no rows in `orders` to group.
- **Why not `SELECT customer_id, COUNT(*) … WHERE COUNT(*)>1`?** Illegal — the aggregate isn't available in `WHERE`. This is the textbook reason `HAVING` exists.

### Practical Question 15: High-Value Product Categories
- **Difficulty:** Medium
- **Estimated Time:** 9 min
- **Concepts Tested:** compound `HAVING`, `SUM` + `AVG` in one filter, `WHERE`/`HAVING` interplay

**Problem Statement**
Find product categories that are both **popular and premium**: total revenue over 10,000 **and** an average line-item value above 100, considering only line items from paid orders. Return category, revenue, and average line value. Sort by revenue descending.

**Example Input** — conceptual aggregates per category (from `order_items` joined to paid `orders` and `products`)

| category | revenue | avg_line |
|----------|---------|----------|
| Hardware | 25000   | 150      |
| Digital  | 12000   | 40       |
| Luxury   | 8000    | 400      |

**Example Output**

| category | revenue | avg_line |
|----------|---------|----------|
| Hardware | 25000   | 150.00   |

**Approach**
1. Join line items to products (category) and orders (paid filter).
2. Filter paid orders in `WHERE`.
3. Aggregate revenue and average line value per category, then apply **both** conditions in a compound `HAVING`.

#### SQL Implementation
```sql
SELECT
    p.category,
    SUM(oi.quantity * oi.unit_price)          AS revenue,
    ROUND(AVG(oi.quantity * oi.unit_price), 2) AS avg_line
FROM order_items oi
JOIN products p ON p.id = oi.product_id
JOIN orders   o ON o.id = oi.order_id
WHERE o.status = 'paid'
GROUP BY p.category
HAVING SUM(oi.quantity * oi.unit_price) > 10000
   AND AVG(oi.quantity * oi.unit_price) > 100
ORDER BY revenue DESC;
```

- `HAVING` can combine multiple aggregate predicates with `AND`/`OR`, each independently evaluated per group. Only groups satisfying *both* survive.
- Digital fails the `avg_line > 100` test; Luxury fails the `revenue > 10000` test; only Hardware clears both — showing how compound `HAVING` narrows on multiple dimensions at once.
- You must repeat the aggregate expressions in `HAVING` (aliases from `SELECT` aren't guaranteed available there). Postgres allows the alias in `HAVING` in practice, but repeating is portable.
- **Perf note:** All groups are computed then filtered; `HAVING` can't prune early. Keep the `WHERE o.status='paid'` filter to minimize rows entering the aggregation. Indexes on the join keys drive the plan.

#### Alternative Solution
Aggregate once in a subquery, then filter — sometimes clearer and avoids repeating expressions:
```sql
SELECT category, revenue, ROUND(avg_line, 2) AS avg_line
FROM (
    SELECT p.category,
           SUM(oi.quantity * oi.unit_price) AS revenue,
           AVG(oi.quantity * oi.unit_price) AS avg_line
    FROM order_items oi
    JOIN products p ON p.id = oi.product_id
    JOIN orders   o ON o.id = oi.order_id
    WHERE o.status = 'paid'
    GROUP BY p.category
) g
WHERE revenue > 10000 AND avg_line > 100
ORDER BY revenue DESC;
```
The outer `WHERE` here filters *rows of the already-aggregated result*, so it's legal and equivalent to `HAVING`. Handy when you want to reference the computed aliases without repeating the expressions.

#### Interview Variations
1. Use `OR` instead of `AND` (popular *or* premium) and observe how the result set grows.
2. Add a minimum distinct-product count per category (`HAVING COUNT(DISTINCT product_id) >= 3`).
3. Return the top category by revenue among those passing the filter (`LIMIT 1`).

#### Common Follow-up Questions
- **Can I filter on an aggregate alias in `HAVING`?** Portably, no — repeat the expression. The subquery-then-`WHERE` pattern is the clean way to filter by alias.
- **Does `HAVING` with `AND` short-circuit / index?** No index help; both aggregates are computed for every group regardless, then the boolean is evaluated.
- **When prefer the subquery form over `HAVING`?** When expressions are long/repeated, or you need to further join/rank the filtered aggregates. Functionally they're the same; readability decides.

### Practical Question 16: Customers Whose Total Spend Exceeds the Average
- **Difficulty:** Hard
- **Estimated Time:** 12 min
- **Concepts Tested:** `HAVING` with a scalar subquery, aggregate over aggregates, `SUM`, comparison to a global average

**Problem Statement**
Find "whale" customers whose **total lifetime spend** (sum of paid order totals) exceeds the **average total spend across all customers**. Return customer id and their total spend, biggest spender first.

**Example Input** — per-customer paid spend

| customer_id | total_spend |
|-------------|-------------|
| 100         | 900         |
| 101         | 300         |
| 102         | 150         |

Average spend = (900 + 300 + 150) / 3 = 450.

**Example Output**

| customer_id | total_spend |
|-------------|-------------|
| 100         | 900         |

**Approach**
1. Per customer, `SUM(total)` over paid orders → each customer's spend.
2. Separately compute the *average of those per-customer sums* — an aggregate over an aggregate, which needs a subquery.
3. In `HAVING`, compare each group's `SUM(total)` against that scalar subquery.

#### SQL Implementation
```sql
SELECT
    customer_id,
    SUM(total) AS total_spend
FROM orders
WHERE status = 'paid'
GROUP BY customer_id
HAVING SUM(total) > (
    -- average of per-customer totals (avg of sums, not avg of rows)
    SELECT AVG(cust_total)
    FROM (
        SELECT SUM(total) AS cust_total
        FROM orders
        WHERE status = 'paid'
        GROUP BY customer_id
    ) per_customer
)
ORDER BY total_spend DESC;
```

- The trap here is **"average of sums" ≠ "average of rows"**. `AVG(total)` over raw orders averages *order* values; we need the average of each *customer's* total, which requires first summing per customer (the inner derived table) and then averaging those sums.
- The scalar subquery in `HAVING` is uncorrelated — it computes one number for the whole query. Postgres evaluates it once (an `InitPlan`), not per group.
- **Perf note:** The per-customer aggregation runs twice (once in the outer query, once in the subquery). For big tables, compute it once in a CTE and self-reference, or use a window function to avoid the double scan.

#### Alternative Solution
Compute per-customer totals once in a CTE, then compare to the average via a window function — single aggregation pass:
```sql
WITH spend AS (
    SELECT customer_id, SUM(total) AS total_spend
    FROM orders
    WHERE status = 'paid'
    GROUP BY customer_id
)
SELECT customer_id, total_spend
FROM (
    SELECT customer_id,
           total_spend,
           AVG(total_spend) OVER () AS avg_spend
    FROM spend
) s
WHERE total_spend > avg_spend
ORDER BY total_spend DESC;
```
`AVG(total_spend) OVER ()` computes the global average of the per-customer totals alongside each row, so we filter without re-aggregating. This is the more scalable pattern and the one I'd ship.

#### Interview Variations
1. Compare each customer to the average within their **city** instead of globally (`PARTITION BY city`).
2. Return the top 10% of spenders (`NTILE(10)` or `PERCENT_RANK`).
3. Use `> ALL (...)` or a threshold multiple of the average (e.g. 2× average).

#### Common Follow-up Questions
- **Why is `HAVING SUM(total) > AVG(total)` wrong?** Both aggregates there are computed *within the same group*, so it compares a customer's sum to that same customer's average order — meaningless for "above the population average". You need a subquery/window for the global figure.
- **Correlated vs. uncorrelated subquery here?** Uncorrelated — the average doesn't depend on the outer group, so it's evaluated once. A correlated version would recompute per group and be far slower.
- **`HAVING` subquery vs. CTE+window — which in production?** The CTE+window: it aggregates once and reads clearly. The `HAVING`-subquery form is fine for small data or when a window isn't available.


---

## Joins (INNER/LEFT/RIGHT/FULL/SELF/CROSS)

Joins are the backbone of relational querying. Before we dive in, internalize the mental model I give every junior engineer: a join is a filtered Cartesian product. The database conceptually pairs every row of the left table with every row of the right table, then keeps only the pairs the `ON` predicate accepts. The *type* of join decides what happens to rows that found **no** partner:

- `INNER JOIN` — drop unmatched rows on both sides.
- `LEFT JOIN` — keep all left rows; pad the right side with `NULL` when there is no match.
- `RIGHT JOIN` — mirror image of LEFT.
- `FULL OUTER JOIN` — keep unmatched rows from *both* sides.
- `SELF JOIN` — a table joined to itself (aliased twice); the classic tool for hierarchies.
- `CROSS JOIN` — the raw Cartesian product with no `ON` at all.

The single most common bug I see in assessments is putting a filter on the *right* table of a `LEFT JOIN` in the `WHERE` clause instead of the `ON` clause. That silently converts your outer join back into an inner join. Watch for it below.

---

### Practical Question 1: Employees with Their Department Name
- **Difficulty:** Easy
- **Estimated Time:** 5 min
- **Concepts Tested:** INNER JOIN, LEFT JOIN, foreign-key navigation, unmatched-row handling

**Problem Statement**
List every employee's name alongside the name of the department they belong to. Then produce a second version that *also* includes employees who have not yet been assigned to a department (`department_id IS NULL`).

**Example Input**

`employees`

| id | name    | department_id | salary |
|----|---------|---------------|--------|
| 1  | Alice   | 10            | 90000  |
| 2  | Bob     | 20            | 70000  |
| 3  | Carol   | NULL          | 65000  |

`departments`

| id | name        |
|----|-------------|
| 10 | Engineering |
| 20 | Sales       |
| 30 | Marketing   |

**Example Output** (LEFT JOIN version)

| name  | department  |
|-------|-------------|
| Alice | Engineering |
| Bob   | Sales       |
| Carol | NULL        |

**Approach**
1. Identify the linking columns: `employees.department_id` references `departments.id`.
2. For "only assigned employees", use `INNER JOIN` — unmatched rows disappear.
3. To keep employees with no department, switch to `LEFT JOIN` so Carol survives with `NULL` in the department column.
4. Alias the department name to a friendly column so it doesn't collide with `employees.name`.

#### SQL Implementation
```sql
-- INNER JOIN: only employees who HAVE a department
SELECT e.name AS employee, d.name AS department
FROM employees AS e
INNER JOIN departments AS d
    ON e.department_id = d.id
ORDER BY e.name;

-- LEFT JOIN: keep everyone, even the un-assigned
SELECT e.name AS employee, d.name AS department
FROM employees AS e
LEFT JOIN departments AS d
    ON e.department_id = d.id
ORDER BY e.name;
```

The `INNER JOIN` drops Carol because her `department_id` is `NULL` and `NULL = anything` is never true. The `LEFT JOIN` preserves her, padding `d.name` with `NULL`. Note that the `ORDER BY` references the aliased base-table column, which is legal because `e.name` is unambiguous.

**Performance/index considerations:** Ensure a B-tree index exists on `employees(department_id)`. Because `departments.id` is the primary key it is already indexed, so PostgreSQL can drive the join efficiently (typically a hash join for large sets, or an index nested-loop when one side is small).

#### Alternative Solution
You can express the same result with a correlated scalar subquery, though it is generally slower and cannot easily return multiple department columns:
```sql
SELECT e.name AS employee,
       (SELECT d.name FROM departments d WHERE d.id = e.department_id) AS department
FROM employees e;
```
This scalar subquery naturally behaves like a `LEFT JOIN` (it returns `NULL` when no match). Prefer the join for readability and when you need more than one column from `departments`.

#### Interview Variations
- Return the *count of employees per department* instead — introduces `GROUP BY`.
- Show departments that currently have **zero** employees (reverse the outer join direction).
- Include only employees whose department name starts with `'E'` — where does the filter go, `ON` or `WHERE`?

#### Common Follow-up Questions
- **"What's the difference between putting a condition in `ON` versus `WHERE` for a LEFT JOIN?"** In `ON`, the condition affects *which rows match* but still keeps all left rows. In `WHERE`, it filters the final result and can eliminate the `NULL`-padded left rows, effectively making it an inner join.
- **"Why did Carol disappear in the INNER JOIN?"** `NULL = d.id` evaluates to `UNKNOWN`, which the join treats as non-matching.
- **"How would you default the department to 'Unassigned' instead of NULL?"** Wrap it: `COALESCE(d.name, 'Unassigned')`.

---

### Practical Question 2: Employees With Their Manager's Name (Self Join)
- **Difficulty:** Medium
- **Estimated Time:** 8 min
- **Concepts Tested:** SELF JOIN, LEFT JOIN, table aliasing, hierarchical relationships within one table

**Problem Statement**
The `employees` table references itself: `manager_id` points at another row's `id`. Produce a report showing each employee's name next to their manager's name. Include the CEO (whose `manager_id` is `NULL`) and label their manager as `'— (top of org)'`.

**Example Input**

`employees`

| id | name    | manager_id |
|----|---------|------------|
| 1  | Alice   | NULL       |
| 2  | Bob     | 1          |
| 3  | Carol   | 1          |
| 4  | Dan     | 2          |

**Example Output**

| employee | manager        |
|----------|----------------|
| Alice    | — (top of org) |
| Bob      | Alice          |
| Carol    | Alice          |
| Dan      | Bob            |

**Approach**
1. Recognize this is a **self join**: the same table plays two roles — "the employee" and "the manager".
2. Alias the table twice: `e` for the employee row, `m` for the manager row.
3. Join `e.manager_id = m.id`.
4. Use a `LEFT JOIN` so Alice (no manager) is not dropped.
5. Wrap the manager name in `COALESCE` to produce the friendly top-of-org label.

#### SQL Implementation
```sql
SELECT e.name                              AS employee,
       COALESCE(m.name, '— (top of org)')  AS manager
FROM employees AS e
LEFT JOIN employees AS m
    ON e.manager_id = m.id
ORDER BY e.id;
```

This is *the* self-join pattern every assessment loves. The two aliases `e` and `m` are logically two independent copies of one physical table. The `LEFT JOIN` is essential: with an `INNER JOIN`, Alice (whose `manager_id` is `NULL`) would vanish from the report — a classic mistake. The `COALESCE` converts the padded `NULL` into a readable label.

**Performance/index considerations:** An index on `employees(manager_id)` speeds the join for large orgs. The self join reads the table twice logically but the planner scans the physical heap/index once per alias.

#### Alternative Solution
A correlated scalar subquery avoids the second alias but is less flexible:
```sql
SELECT e.name AS employee,
       COALESCE(
           (SELECT m.name FROM employees m WHERE m.id = e.manager_id),
           '— (top of org)'
       ) AS manager
FROM employees e
ORDER BY e.id;
```
The self join wins when you need more than one manager column (e.g., manager's email and salary too) because the subquery would need one lookup per column.

#### Interview Variations
- Also show the manager's *manager* (grandmanager) — chain a third alias `mm`.
- Filter to employees who earn **more** than their manager (`e.salary > m.salary`) — a favorite "who out-earns the boss" question.
- List managers alongside the *count* of their direct reports.

#### Common Follow-up Questions
- **"Why two aliases?"** SQL needs to distinguish the employee copy from the manager copy of the same table; without aliases every column reference is ambiguous.
- **"How do you find employees who earn more than their manager?"** `INNER JOIN employees m ON e.manager_id = m.id WHERE e.salary > m.salary`.
- **"What if the hierarchy is deeper than one level and you want the whole chain?"** That's a recursive CTE — see Question 12.

---

### Practical Question 3: Customers With No Orders (LEFT JOIN … IS NULL)
- **Difficulty:** Medium
- **Estimated Time:** 8 min
- **Concepts Tested:** LEFT JOIN anti-join, IS NULL filtering, NOT EXISTS, NOT IN + NULL trap

**Problem Statement**
Find every customer who has never placed an order. Return their `id`, `name`, and `email`.

**Example Input**

`customers`

| id | name    | email             |
|----|---------|-------------------|
| 1  | Alice   | alice@mail.com    |
| 2  | Bob     | bob@mail.com      |
| 3  | Carol   | carol@mail.com    |

`orders`

| id  | customer_id | total |
|-----|-------------|-------|
| 100 | 1           | 250   |
| 101 | 1           | 90    |

**Example Output**

| id | name  | email          |
|----|-------|----------------|
| 2  | Bob   | bob@mail.com   |
| 3  | Carol | carol@mail.com |

**Approach**
1. This is an **anti-join**: we want customers with *no* matching order.
2. `LEFT JOIN` customers to orders, then keep only rows where the order side is `NULL` (`o.customer_id IS NULL`).
3. Because it's a LEFT JOIN anti-pattern, put the "no match" test in `WHERE`, and it must reference a column guaranteed non-null on matched rows (the join key or PK).

#### SQL Implementation
```sql
SELECT c.id, c.name, c.email
FROM customers AS c
LEFT JOIN orders AS o
    ON o.customer_id = c.id
WHERE o.id IS NULL;          -- no matching order row survived
ORDER BY c.id;
```

The mechanism: the `LEFT JOIN` keeps every customer; matched customers get real order rows, unmatched customers get a single `NULL`-padded row. Filtering `o.id IS NULL` isolates exactly the unmatched customers. Test against the order's **primary key** (`o.id`), never a nullable column, so a genuine order with a `NULL` in some other field can't fool the filter.

#### Alternative Solution
The `NOT EXISTS` form is my preferred production idiom — it reads like the requirement and is null-safe:
```sql
SELECT c.id, c.name, c.email
FROM customers c
WHERE NOT EXISTS (
    SELECT 1
    FROM orders o
    WHERE o.customer_id = c.id
)
ORDER BY c.id;
```

**The NOT IN + NULL trap (read this twice).** You might be tempted to write:
```sql
-- DANGEROUS if orders.customer_id can be NULL
SELECT * FROM customers
WHERE id NOT IN (SELECT customer_id FROM orders);
```
If **any** row in `orders.customer_id` is `NULL`, the entire `NOT IN` returns *no rows at all*. Why? `id NOT IN (1, NULL)` expands to `id <> 1 AND id <> NULL`, and `id <> NULL` is `UNKNOWN`, so the whole `AND` can never be `TRUE`. `NOT EXISTS` and `LEFT JOIN … IS NULL` do not suffer this — they handle `NULL` correctly. **Rule of thumb: prefer `NOT EXISTS` for anti-joins; only use `NOT IN` when the subquery column is provably `NOT NULL`.**

Performance-wise, on modern PostgreSQL `NOT EXISTS` and `LEFT JOIN … IS NULL` typically both compile to an efficient *anti-join* execution node, so pick the one that reads clearest — usually `NOT EXISTS`.

#### Interview Variations
- Customers with no orders in the **last 90 days** (add `AND o.order_date >= CURRENT_DATE - INTERVAL '90 days'` inside the `EXISTS`, or in the `ON`).
- Products never sold (anti-join `products` against `order_items`).
- Customers whose *only* orders are `status = 'cancelled'`.

#### Common Follow-up Questions
- **"Why can `NOT IN` return zero rows unexpectedly?"** A `NULL` in the subquery result poisons the comparison to `UNKNOWN`. Covered above.
- **"Which is faster, `NOT EXISTS` or `LEFT JOIN … IS NULL`?"** Usually equivalent — the planner picks an anti-join for both. Measure with `EXPLAIN ANALYZE` on your data.
- **"How do you count how many customers never ordered?"** Wrap either query in `SELECT COUNT(*) FROM (…) t`.

---

### Practical Question 4: Sales Rep × Region Assignment Matrix (CROSS JOIN)
- **Difficulty:** Easy
- **Estimated Time:** 6 min
- **Concepts Tested:** CROSS JOIN, Cartesian product, generating combinations, LEFT JOIN to a generated grid

**Problem Statement**
The sales team wants a coverage matrix: every combination of *department* × *distinct customer city*, so they can see which department-city pairs currently have zero customers. First generate all pairs, then flag which pairs are empty.

**Example Input**

`departments`

| id | name  |
|----|-------|
| 20 | Sales |
| 30 | Mktg  |

Distinct `customers.city`: `Austin`, `Denver`

**Example Output** (the generated grid)

| department | city   |
|------------|--------|
| Sales      | Austin |
| Sales      | Denver |
| Mktg       | Austin |
| Mktg       | Denver |

**Approach**
1. A `CROSS JOIN` produces every combination — no `ON` clause.
2. Generate the distinct cities in a subquery/CTE, then cross join to departments.
3. (Extension) `LEFT JOIN` the real customers to find empty cells.

#### SQL Implementation
```sql
SELECT d.name AS department, c.city
FROM departments AS d
CROSS JOIN (SELECT DISTINCT city FROM customers) AS c
ORDER BY d.name, c.city;
```

`CROSS JOIN` intentionally has no join predicate — it returns `rows(departments) × rows(cities)`. This is the one join where an explosion of rows is the *goal*, not a bug. Guard against accidental cross joins elsewhere: a missing `ON` in older comma-join syntax (`FROM a, b`) silently produces one.

#### Alternative Solution
The legacy comma syntax yields the same Cartesian product but is discouraged because it hides intent:
```sql
SELECT d.name AS department, c.city
FROM departments d, (SELECT DISTINCT city FROM customers) c;
```
For calendar/number grids, combine `CROSS JOIN` with `generate_series()`:
```sql
SELECT d.name, g.day::date
FROM departments d
CROSS JOIN generate_series('2026-01-01', '2026-01-07', INTERVAL '1 day') AS g(day);
```

#### Interview Variations
- Build a date-spine: cross join every product with every day in a month for an inventory report.
- Show only the *empty* pairs by left-joining the grid back to `customers`.
- Cross join two small dimension tables to seed a reporting fact table.

#### Common Follow-up Questions
- **"When is a CROSS JOIN legitimate vs. a mistake?"** Legitimate when you deliberately need all combinations (grids, spines). A mistake when you forgot the `ON` clause.
- **"How large is the result?"** Exactly the product of the two row counts — be careful with big tables.
- **"How do you find combinations that don't yet exist in data?"** Cross join the dimensions, then `LEFT JOIN … IS NULL` against the fact table.

---

### Practical Question 5: Reconciling Two Sides With FULL OUTER JOIN
- **Difficulty:** Hard
- **Estimated Time:** 12 min
- **Concepts Tested:** FULL OUTER JOIN, COALESCE, reconciliation, detecting orphans on both sides

**Problem Statement**
Imagine a data-quality audit. You want a single result that shows, per `customer_id`, whether it exists in `customers`, in `orders`, or both — surfacing orphaned orders (a `customer_id` in `orders` with no matching customer) *and* customers who never ordered, in one pass.

**Example Input**

`customers`: ids `{1, 2, 3}`
`orders.customer_id`: `{1, 1, 99}` (99 is an orphaned order — no such customer)

**Example Output**

| customer_id | customer_name | order_count | status              |
|-------------|---------------|-------------|---------------------|
| 1           | Alice         | 2           | matched             |
| 2           | Bob           | 0           | customer_no_orders  |
| 3           | Carol         | 0           | customer_no_orders  |
| 99          | NULL          | 1           | orphaned_order      |

**Approach**
1. Aggregate orders per `customer_id` first (so counts are clean).
2. `FULL OUTER JOIN` customers to that aggregate on the id.
3. Use `COALESCE` to pick the id from whichever side is present.
4. Derive a `status` with `CASE` based on which side is `NULL`.

#### SQL Implementation
```sql
SELECT
    COALESCE(c.id, o.customer_id)            AS customer_id,
    c.name                                   AS customer_name,
    COALESCE(o.order_count, 0)               AS order_count,
    CASE
        WHEN c.id IS NULL              THEN 'orphaned_order'
        WHEN o.customer_id IS NULL     THEN 'customer_no_orders'
        ELSE                                'matched'
    END                                      AS status
FROM customers AS c
FULL OUTER JOIN (
    SELECT customer_id, COUNT(*) AS order_count
    FROM orders
    GROUP BY customer_id
) AS o
    ON c.id = o.customer_id
ORDER BY customer_id;
```

The `FULL OUTER JOIN` keeps rows that exist on *only* the left (customers who never ordered → `o.*` is `NULL`) **and** rows that exist on *only* the right (orphaned order ids → `c.*` is `NULL`). `COALESCE(c.id, o.customer_id)` grabs the id from whichever side supplied it. The `CASE` reads the null-ness of each side to classify the row. Pre-aggregating orders in the derived table keeps the count correct and avoids fan-out.

**Performance/index considerations:** `FULL OUTER JOIN` cannot use a plain nested-loop with an index the way inner joins can; PostgreSQL typically uses a hash or merge join. Aggregating first shrinks the right side, which helps.

#### Alternative Solution
Databases lacking `FULL OUTER JOIN` (e.g., older MySQL) emulate it by `UNION`-ing a `LEFT JOIN` with the anti-half of a `RIGHT JOIN`:
```sql
SELECT c.id AS customer_id, c.name, ...
FROM customers c
LEFT JOIN order_agg o ON c.id = o.customer_id
UNION
SELECT o.customer_id, NULL, ...
FROM order_agg o
LEFT JOIN customers c ON c.id = o.customer_id
WHERE c.id IS NULL;   -- only the orphans the LEFT JOIN missed
```
PostgreSQL supports `FULL OUTER JOIN` natively, so prefer it for clarity.

#### Interview Variations
- Reconcile `products` vs `order_items` to find products never ordered *and* order_items pointing at deleted products.
- Add a monetary column: total spend per side, flagging mismatches.
- Convert `status` counts into a summary with an outer `GROUP BY status`.

#### Common Follow-up Questions
- **"When would you reach for FULL OUTER JOIN in practice?"** Reconciliation / data-quality audits where either side may have rows the other lacks.
- **"Why COALESCE the key column?"** Because on unmatched rows one side's key is `NULL`; COALESCE yields the present value.
- **"How do you find only the orphans?"** Keep the FULL JOIN but filter `WHERE c.id IS NULL`.

---

## Subqueries (single-row, multi-row, correlated, EXISTS)

A subquery is a `SELECT` nested inside another statement. Classify them by two axes:

- **By cardinality of what they return:** *scalar* (one row, one column — usable anywhere a value is expected), *multi-row* (a column of values — used with `IN`, `ANY`, `ALL`), or *table* (used in `FROM` as a derived table).
- **By dependency:** *non-correlated* (runs once, independent of the outer query) vs *correlated* (references a column from the outer query, so it conceptually re-runs per outer row).

`EXISTS` deserves special mention: it's a correlated boolean test that short-circuits on the first matching row, making it ideal for semi-joins and anti-joins. Now, the classics.

---

### Practical Question 6: Employees Earning Above the Company Average (Scalar Subquery)
- **Difficulty:** Easy
- **Estimated Time:** 5 min
- **Concepts Tested:** Single-row (scalar) subquery, aggregate in a subquery, WHERE filtering

**Problem Statement**
Return all employees whose salary is strictly greater than the *company-wide* average salary. Show name and salary, highest first.

**Example Input**

`employees`

| id | name  | salary |
|----|-------|--------|
| 1  | Alice | 90000  |
| 2  | Bob   | 70000  |
| 3  | Carol | 65000  |
| 4  | Dan   | 50000  |

(Average = 68750)

**Example Output**

| name  | salary |
|-------|--------|
| Alice | 90000  |
| Bob   | 70000  |

**Approach**
1. Compute the average salary with a **scalar subquery** — it returns exactly one value.
2. Compare each row's salary against it in the `WHERE`.
3. Order descending.

#### SQL Implementation
```sql
SELECT name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees)
ORDER BY salary DESC;
```

The subquery `(SELECT AVG(salary) FROM employees)` is *non-correlated* — it runs **once**, produces a single number, and PostgreSQL substitutes it into the comparison. This is the cleanest use of a scalar subquery. If it accidentally returned more than one row, PostgreSQL would raise `more than one row returned by a subquery used as an expression`.

#### Alternative Solution
A window function avoids scanning the table twice and lets you show the average alongside each row:
```sql
SELECT name, salary
FROM (
    SELECT name, salary, AVG(salary) OVER () AS avg_sal
    FROM employees
) t
WHERE salary > avg_sal
ORDER BY salary DESC;
```
The window version reads the table once (one scan) versus the subquery's two logical passes — often faster on large tables, and it can expose `avg_sal` in the output.

#### Interview Variations
- Above the average of *their own department* (turns it correlated — see Q9).
- Above the *median* rather than the mean (`PERCENTILE_CONT(0.5)`).
- Within 10% of the max salary.

#### Common Follow-up Questions
- **"What makes this scalar?"** It returns one row and one column, so it's usable as a single value.
- **"Does the subquery run per row?"** No — it's non-correlated, so it runs once and is cached.
- **"How to include the boundary (>=)?"** Change `>` to `>=`.

---

### Practical Question 7: Second-Highest Salary (Classic Subquery)
- **Difficulty:** Medium
- **Estimated Time:** 8 min
- **Concepts Tested:** Multi-row subquery, MAX-of-filtered-set, handling duplicates and NULLs, DISTINCT

**Problem Statement**
Find the second-highest *distinct* salary in the company. Return a single value. Handle ties correctly (if the top salary appears twice, the second-highest distinct salary is the next lower value, not a repeat of the top).

**Example Input**

`employees.salary`: `90000, 90000, 70000, 65000`

**Example Output**

| second_highest |
|----------------|
| 70000          |

**Approach**
1. The second-highest distinct salary is the *maximum salary that is strictly less than the overall maximum*.
2. Inner subquery: `MAX(salary)`.
3. Outer: `MAX(salary) WHERE salary < that`.

#### SQL Implementation
```sql
SELECT MAX(salary) AS second_highest
FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);
```

This "max below the max" trick is robust to duplicate top salaries because `WHERE salary < top` excludes *all* copies of the maximum. Because we use `MAX`, `DISTINCT` is implied — duplicates don't matter. If there is no second salary (only one distinct value), this returns `NULL` rather than an error, which is usually the desired behavior.

#### Alternative Solution
The general "N-th highest" pattern uses window functions, which scale to any rank:
```sql
-- DENSE_RANK treats ties as one rank -> true "distinct" ranking
SELECT DISTINCT salary AS second_highest
FROM (
    SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS rnk
    FROM employees
) t
WHERE rnk = 2;
```
Or the compact `OFFSET` approach:
```sql
SELECT DISTINCT salary
FROM employees
ORDER BY salary DESC
OFFSET 1 LIMIT 1;
```
Use `DENSE_RANK` when you need the *N-th distinct* value generically; use `ROW_NUMBER` if you want the literal N-th row including ties; use the `MAX`-below-`MAX` trick when N=2 and you want the simplest possible query.

**Trap:** `LIMIT 1 OFFSET 1` returns the second *row*, which may be a duplicate of the top salary if the top appears twice. Add `DISTINCT` (as above) or aggregate first to get the second *distinct* salary.

#### Interview Variations
- Generalize to the N-th highest (parameterize `rnk = :n` or `OFFSET :n-1`).
- Second-highest salary *per department* (`PARTITION BY department_id`).
- Return the *employee(s)* earning the second-highest salary, not just the number.

#### Common Follow-up Questions
- **"Why not `LIMIT 1 OFFSET 1`?"** It returns the second row, not the second *distinct* salary, so ties on the top salary break it unless you add `DISTINCT`.
- **"ROW_NUMBER vs RANK vs DENSE_RANK?"** `ROW_NUMBER` = unique sequential; `RANK` = ties share a rank then skip; `DENSE_RANK` = ties share a rank, no gaps. For "N-th distinct value" use `DENSE_RANK`.
- **"What if the table has one salary?"** The `MAX`-below-`MAX` query returns `NULL`; decide whether that's acceptable or should raise.

---

### Practical Question 8: Orders Containing a Specific Product (EXISTS / IN)
- **Difficulty:** Medium
- **Estimated Time:** 8 min
- **Concepts Tested:** EXISTS semi-join, IN multi-row subquery, correlation, when EXISTS beats IN

**Problem Statement**
Find all orders that include at least one product from the `'Electronics'` category. Return the order id and order date, each order only once.

**Example Input**

`products`

| id | name    | category    |
|----|---------|-------------|
| 5  | Phone   | Electronics |
| 6  | Cable   | Electronics |
| 7  | Mug     | Kitchen     |

`order_items`

| order_id | product_id |
|----------|------------|
| 100      | 5          |
| 100      | 7          |
| 101      | 7          |

**Example Output**

| id  | order_date |
|-----|------------|
| 100 | 2026-06-01 |

**Approach**
1. We need orders having *at least one* electronics item — a **semi-join**.
2. `EXISTS` with a correlated subquery is the natural fit: for each order, check if any of its items is electronics.
3. Join `order_items` to `products` inside the `EXISTS` to test the category.

#### SQL Implementation
```sql
SELECT o.id, o.order_date
FROM orders AS o
WHERE EXISTS (
    SELECT 1
    FROM order_items AS oi
    JOIN products    AS p ON p.id = oi.product_id
    WHERE oi.order_id = o.id          -- correlation to the outer order
      AND p.category  = 'Electronics'
)
ORDER BY o.id;
```

`EXISTS` returns `TRUE` as soon as the subquery finds its **first** matching row, then stops — that short-circuit makes it efficient. It also naturally deduplicates: an order with three electronics items still appears once, because `EXISTS` is a boolean, not a join that fans out. The correlation `oi.order_id = o.id` is what ties the inner query to the current outer order.

#### Alternative Solution
The `IN` form is equivalent here and reads well because `order_items.order_id` is `NOT NULL`:
```sql
SELECT o.id, o.order_date
FROM orders o
WHERE o.id IN (
    SELECT oi.order_id
    FROM order_items oi
    JOIN products p ON p.id = oi.product_id
    WHERE p.category = 'Electronics'
)
ORDER BY o.id;
```
A plain `JOIN` also works but needs `DISTINCT` to avoid duplicate orders:
```sql
SELECT DISTINCT o.id, o.order_date
FROM orders o
JOIN order_items oi ON oi.order_id = o.id
JOIN products p ON p.id = oi.product_id
WHERE p.category = 'Electronics';
```
Prefer `EXISTS` for semi-joins: it avoids fan-out (no `DISTINCT` needed) and short-circuits. Reserve `IN` for when the subquery is small and provably non-null; recall the `NOT IN` + `NULL` trap from Q3 if you ever negate it.

#### Interview Variations
- Orders that contain electronics **but no** kitchen items (`EXISTS` + `NOT EXISTS`).
- Orders where *every* item is electronics (universal quantification — `NOT EXISTS` of a non-electronics item).
- Customers who have ever bought electronics.

#### Common Follow-up Questions
- **"EXISTS vs IN — performance?"** Often the planner treats them identically (both semi-joins). `EXISTS` is safer with correlation and nulls; `IN` can be cleaner for simple static lists.
- **"Why `SELECT 1` in EXISTS?"** The projected value is irrelevant; `EXISTS` only cares whether a row exists. `SELECT 1`, `SELECT *`, or `SELECT NULL` are equivalent.
- **"How to express 'every item is electronics'?"** `NOT EXISTS (… WHERE category <> 'Electronics')` — double negation for universal quantification.

---

### Practical Question 9: Employees Above Their Department's Average (Correlated Subquery)
- **Difficulty:** Hard
- **Estimated Time:** 12 min
- **Concepts Tested:** Correlated subquery, per-group comparison, correlation vs window functions

**Problem Statement**
Return employees who earn more than the average salary *of their own department*. Show name, department_id, and salary. Also produce the list of departments whose *average salary* exceeds the company-wide average.

**Example Input**

`employees`

| id | name  | department_id | salary |
|----|-------|---------------|--------|
| 1  | Alice | 10            | 90000  |
| 2  | Bob   | 10            | 60000  |
| 3  | Carol | 20            | 80000  |
| 4  | Dan   | 20            | 40000  |

Dept 10 avg = 75000; Dept 20 avg = 60000.

**Example Output** (part 1)

| name  | department_id | salary |
|-------|---------------|--------|
| Alice | 10            | 90000  |
| Carol | 20            | 80000  |

**Approach**
1. For each employee, compare their salary to the average of *their* department.
2. That average depends on the outer row's `department_id` → a **correlated subquery**.
3. The subquery references `e.department_id` from the outer query, so it re-evaluates per department group.

#### SQL Implementation
```sql
-- Part 1: employees above their OWN department's average
SELECT e.name, e.department_id, e.salary
FROM employees AS e
WHERE e.salary > (
    SELECT AVG(e2.salary)
    FROM employees AS e2
    WHERE e2.department_id = e.department_id   -- correlation
)
ORDER BY e.department_id, e.salary DESC;
```

The inner query is **correlated**: `e2.department_id = e.department_id` ties it to the current outer employee's department. Conceptually it computes a fresh departmental average for each outer row (the optimizer often rewrites this to run once per department, not once per row). Contrast with Q6, where the average was global and non-correlated.

```sql
-- Part 2: departments whose average salary beats the company average
SELECT d.id, d.name, AVG(e.salary) AS dept_avg
FROM departments AS d
JOIN employees AS e ON e.department_id = d.id
GROUP BY d.id, d.name
HAVING AVG(e.salary) > (SELECT AVG(salary) FROM employees)
ORDER BY dept_avg DESC;
```
Here the company average is a *non-correlated* scalar subquery inside `HAVING`, compared against each group's aggregated average.

**Performance/index considerations:** Correlated subqueries can degrade to O(n × groups) if the planner doesn't rewrite them. Index `employees(department_id, salary)` to make the per-department aggregate cheap. For large datasets, the window-function alternative below is usually faster because it needs a single pass.

#### Alternative Solution
A window function computes each department's average in one scan and avoids re-execution:
```sql
SELECT name, department_id, salary
FROM (
    SELECT name, department_id, salary,
           AVG(salary) OVER (PARTITION BY department_id) AS dept_avg
    FROM employees
) t
WHERE salary > dept_avg
ORDER BY department_id, salary DESC;
```
The `PARTITION BY department_id` gives every row its department's average without a correlated re-scan. Prefer this for large tables; the correlated subquery is fine for small ones and reads more declaratively.

#### Interview Variations
- Employees above their *manager's* salary (correlate on `manager_id`).
- Departments where the *max* salary is below the company average (struggling teams).
- Employees in the top salary quartile of their department (`NTILE(4)`).

#### Common Follow-up Questions
- **"How do I know it's correlated?"** The inner query references a column from the outer query (`e.department_id`). Remove that and it becomes independent.
- **"Correlated subquery vs window function — which to use?"** Window function for performance and to expose the computed value; correlated subquery when it reads more naturally or the DB lacks window support.
- **"Why `HAVING` and not `WHERE` in part 2?"** `WHERE` filters rows *before* aggregation; `HAVING` filters *after*, and we're comparing an aggregate (`AVG`).

---

## Common Table Expressions (CTEs) & Recursive CTEs

A CTE (`WITH name AS (…)`) is a named, inline result set that exists for the duration of one statement. Reach for CTEs to:

- **Name and sequence** logical steps so a complex query reads top-to-bottom instead of inside-out.
- **Reuse** a subresult referenced multiple times.
- **Recurse** over hierarchical or graph data (org charts, bill-of-materials, category trees) with `WITH RECURSIVE`.

A note on PostgreSQL semantics: since v12, non-recursive CTEs are *not* an optimization fence by default — the planner may inline them (you can force materialization with `WITH … AS MATERIALIZED`). A recursive CTE has two arms joined by `UNION [ALL]`: the **anchor** (base case, runs once) and the **recursive term** (references the CTE itself, runs repeatedly until it produces no new rows).

---

### Practical Question 10: Top 3 Customers by Spend Using a CTE
- **Difficulty:** Easy
- **Estimated Time:** 6 min
- **Concepts Tested:** Basic CTE, aggregation, readability, ORDER BY + LIMIT

**Problem Statement**
Compute each customer's total order spend, then return the top 3 spenders with their name and total. Use a CTE to make the two logical steps (aggregate, then rank/limit) explicit.

**Example Input**

`orders`

| id  | customer_id | total |
|-----|-------------|-------|
| 100 | 1           | 250   |
| 101 | 1           | 90    |
| 102 | 2           | 400   |
| 103 | 3           | 120   |

**Example Output**

| name  | total_spend |
|-------|-------------|
| Bob   | 400         |
| Alice | 340         |
| Carol | 120         |

**Approach**
1. First CTE: sum `orders.total` per `customer_id`.
2. Main query: join back to `customers` for the name, order by spend, `LIMIT 3`.

#### SQL Implementation
```sql
WITH customer_spend AS (
    SELECT customer_id, SUM(total) AS total_spend
    FROM orders
    GROUP BY customer_id
)
SELECT c.name, cs.total_spend
FROM customer_spend AS cs
JOIN customers AS c ON c.id = cs.customer_id
ORDER BY cs.total_spend DESC
LIMIT 3;
```

The CTE `customer_spend` names the aggregation step, so the final `SELECT` reads as plain English: take the spend, attach the name, rank, keep three. Functionally this equals a derived table in the `FROM`, but the `WITH` form scales better as steps multiply. Only `SUM`med, `GROUP BY`ed customers with at least one order appear — customers with no orders are absent (they'd need a `LEFT JOIN` from `customers`).

#### Alternative Solution
Equivalent inline derived table (no CTE):
```sql
SELECT c.name, cs.total_spend
FROM (
    SELECT customer_id, SUM(total) AS total_spend
    FROM orders GROUP BY customer_id
) AS cs
JOIN customers c ON c.id = cs.customer_id
ORDER BY cs.total_spend DESC
LIMIT 3;
```
The CTE and derived table produce identical plans in modern PostgreSQL; choose the CTE when the query has several stages or the subresult is reused.

#### Interview Variations
- Include customers with zero spend (LEFT JOIN customers → CTE, `COALESCE(total_spend, 0)`).
- Top 3 *per city* (window `RANK() OVER (PARTITION BY city ORDER BY spend DESC)`).
- Restrict to `status = 'completed'` orders only.

#### Common Follow-up Questions
- **"CTE vs subquery — any performance difference?"** In PostgreSQL 12+, generally no — non-recursive CTEs can be inlined. Pre-12 they were an optimization fence.
- **"How to force materialization?"** `WITH customer_spend AS MATERIALIZED (…)`.
- **"Why are some customers missing?"** They have no orders; the inner aggregation only sees customers present in `orders`.

---

### Practical Question 11: Running Total of Sales per Category (CTE + Window)
- **Difficulty:** Medium
- **Estimated Time:** 10 min
- **Concepts Tested:** CTE staging, window function running total, PARTITION BY, ordered cumulative sums

**Problem Statement**
For each product category, show a running (cumulative) total of revenue ordered by month. Revenue = `SUM(order_items.quantity * order_items.unit_price)`. Stage the monthly-per-category revenue in a CTE, then compute the running total with a window function.

**Example Input** (already joined/derived for illustration)

| category    | month   | monthly_revenue |
|-------------|---------|-----------------|
| Electronics | 2026-01 | 1000            |
| Electronics | 2026-02 | 500             |
| Kitchen     | 2026-01 | 300             |

**Example Output**

| category    | month   | monthly_revenue | running_total |
|-------------|---------|-----------------|---------------|
| Electronics | 2026-01 | 1000            | 1000          |
| Electronics | 2026-02 | 500             | 1500          |
| Kitchen     | 2026-01 | 300             | 300           |

**Approach**
1. CTE `monthly`: join `order_items` → `products` → `orders`, group by category and month, sum revenue.
2. Main query: `SUM(...) OVER (PARTITION BY category ORDER BY month)` produces the cumulative running total, restarting per category.

#### SQL Implementation
```sql
WITH monthly AS (
    SELECT
        p.category,
        date_trunc('month', o.order_date)::date AS month,
        SUM(oi.quantity * oi.unit_price)        AS monthly_revenue
    FROM order_items AS oi
    JOIN products    AS p ON p.id = oi.product_id
    JOIN orders      AS o ON o.id = oi.order_id
    GROUP BY p.category, date_trunc('month', o.order_date)
)
SELECT
    category,
    month,
    monthly_revenue,
    SUM(monthly_revenue) OVER (
        PARTITION BY category
        ORDER BY month
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_total
FROM monthly
ORDER BY category, month;
```

The CTE stages a clean, one-row-per-(category, month) table. The window `SUM(...) OVER (PARTITION BY category ORDER BY month …)` accumulates revenue *within* each category and *resets* at each new category, thanks to `PARTITION BY`. The explicit frame `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` states "sum from the first row of this partition up to the current row" — this is a genuine running total. Be aware: the *default* frame for an ordered window is `RANGE`, which lumps together **peer rows with equal `ORDER BY` values**; specifying `ROWS` makes the running total unambiguous when months could tie.

**Performance/index considerations:** Indexes on `order_items(product_id)`, `order_items(order_id)`, and `orders(order_date)` help the joins and grouping. The window computation itself needs the data sorted by `(category, month)`, which the planner arranges via a sort.

#### Alternative Solution
A correlated subquery can emulate the running total but is O(n²) and far slower:
```sql
SELECT m1.category, m1.month, m1.monthly_revenue,
       (SELECT SUM(m2.monthly_revenue)
        FROM monthly m2
        WHERE m2.category = m1.category
          AND m2.month   <= m1.month) AS running_total
FROM monthly m1;
```
Always prefer the window function — one ordered pass versus a re-scan per row.

#### Interview Variations
- 3-month moving average (`ROWS BETWEEN 2 PRECEDING AND CURRENT ROW` with `AVG`).
- Percent-of-category-total per month (`monthly_revenue / SUM(...) OVER (PARTITION BY category)`).
- Year-to-date reset each January (partition by `category, year`).

#### Common Follow-up Questions
- **"ROWS vs RANGE in the frame?"** `ROWS` counts physical rows; `RANGE` groups rows with equal `ORDER BY` values into the same frame. For running totals over possibly-tied keys, use `ROWS`.
- **"Why PARTITION BY category?"** To restart the cumulative sum for each category instead of running across all of them.
- **"Can I do this without a CTE?"** Yes, but staging the aggregation first keeps the window logic readable.

---

### Practical Question 12: Recursive Org Chart / Employee Hierarchy
- **Difficulty:** Hard
- **Estimated Time:** 15 min
- **Concepts Tested:** WITH RECURSIVE, anchor + recursive term, depth tracking, path building, cycle safety

**Problem Statement**
Given the self-referencing `employees(id, name, manager_id)`, produce the full reporting hierarchy starting from the CEO (`manager_id IS NULL`). For each employee show their `level` (CEO = 1), and a breadcrumb `path` like `Alice > Bob > Dan`.

**Example Input**

`employees`

| id | name  | manager_id |
|----|-------|------------|
| 1  | Alice | NULL       |
| 2  | Bob   | 1          |
| 3  | Carol | 1          |
| 4  | Dan   | 2          |

**Example Output**

| id | name  | level | path                |
|----|-------|-------|---------------------|
| 1  | Alice | 1     | Alice               |
| 2  | Bob   | 2     | Alice > Bob         |
| 3  | Carol | 2     | Alice > Carol       |
| 4  | Dan   | 3     | Alice > Bob > Dan   |

**Approach**
1. **Anchor:** select the root(s) — employees with `manager_id IS NULL` — at `level = 1` with `path = name`.
2. **Recursive term:** join `employees` to the CTE where `employees.manager_id = cte.id`, incrementing level and appending to the path.
3. `UNION ALL` combines anchor + each recursion round until no new rows appear.
4. Track and break cycles defensively.

#### SQL Implementation
```sql
WITH RECURSIVE org_chart AS (
    -- Anchor: the top of the hierarchy
    SELECT
        e.id,
        e.name,
        e.manager_id,
        1                       AS level,
        e.name::text            AS path
    FROM employees AS e
    WHERE e.manager_id IS NULL

    UNION ALL

    -- Recursive term: attach each direct report to its manager
    SELECT
        e.id,
        e.name,
        e.manager_id,
        oc.level + 1            AS level,
        oc.path || ' > ' || e.name AS path
    FROM employees AS e
    JOIN org_chart AS oc ON e.manager_id = oc.id
)
SELECT id, name, level, path
FROM org_chart
ORDER BY path;
```

How it executes: the **anchor** runs once and seeds `org_chart` with Alice (level 1). The **recursive term** then repeatedly joins the *newly added* rows back to `employees` — round 1 finds Bob and Carol (Alice's reports), round 2 finds Dan (Bob's report), round 3 finds nothing, so recursion stops. `UNION ALL` (not `UNION`) is standard here: it avoids a needless dedup pass, and in a tree there are no duplicates anyway. The `path` string accumulates the breadcrumb by concatenation; casting the anchor's `name` to `text` keeps the column types consistent across both arms.

**Cycle safety:** real-world data sometimes has corrupt loops (A manages B manages A), which would recurse forever. Guard it by carrying the visited path as an array and refusing to revisit:
```sql
WITH RECURSIVE org_chart AS (
    SELECT e.id, e.name, e.manager_id, 1 AS level, ARRAY[e.id] AS visited
    FROM employees e
    WHERE e.manager_id IS NULL
    UNION ALL
    SELECT e.id, e.name, e.manager_id, oc.level + 1, oc.visited || e.id
    FROM employees e
    JOIN org_chart oc ON e.manager_id = oc.id
    WHERE NOT e.id = ANY(oc.visited)      -- stop if we've seen this node
)
SELECT * FROM org_chart;
```
PostgreSQL 14+ also offers native `UNION ALL … CYCLE id SET is_cycle USING cyclepath` syntax for the same protection.

**Performance/index considerations:** Index `employees(manager_id)` so each recursion round's join is cheap. Deep hierarchies cost one join round per level; breadth doesn't multiply rounds.

#### Alternative Solution
Without recursion you're limited to a *fixed* depth via repeated self joins — only viable if you know the max levels:
```sql
SELECT e1.name AS lvl1, e2.name AS lvl2, e3.name AS lvl3
FROM employees e1
LEFT JOIN employees e2 ON e2.manager_id = e1.id
LEFT JOIN employees e3 ON e3.manager_id = e2.id
WHERE e1.manager_id IS NULL;
```
This breaks the moment the org grows a fourth level. The recursive CTE handles arbitrary depth — always prefer it for true hierarchies.

#### Interview Variations
- Return only a given manager's subtree (change the anchor to `WHERE id = :manager_id`).
- Compute each manager's *total* headcount (all descendants), via recursion then `GROUP BY` the root.
- Bill-of-materials: recurse a `product_components(parent_id, child_id, qty)` table to explode assemblies.

#### Common Follow-up Questions
- **"UNION vs UNION ALL in a recursive CTE?"** `UNION ALL` is standard and faster; use `UNION` only if the graph can produce duplicate rows you must collapse. For trees, `UNION ALL` is correct.
- **"How do you prevent infinite loops?"** Track visited nodes in an array (`NOT id = ANY(visited)`) or use the `CYCLE` clause (PG 14+).
- **"What are the two required parts?"** The anchor (base case) and the recursive term, combined by `UNION [ALL]`.
- **"How do you limit depth?"** Add `WHERE oc.level < :max` in the recursive term.

---

### Practical Question 13: Filling Date Gaps with a Recursive Date Series (CTE)
- **Difficulty:** Medium
- **Estimated Time:** 10 min
- **Concepts Tested:** Recursive CTE for sequence generation, LEFT JOIN to a spine, gap filling, COALESCE

**Problem Statement**
Sales reporting needs a continuous daily series even for days with **no** orders (showing `0`). Generate every date in a range using a recursive CTE (the classic "date spine"), then LEFT JOIN daily order totals so gap days appear with zero revenue.

**Example Input**

`orders`

| order_date | total |
|------------|-------|
| 2026-06-01 | 100   |
| 2026-06-03 | 50    |

Range: 2026-06-01 → 2026-06-03.

**Example Output**

| day        | daily_total |
|------------|-------------|
| 2026-06-01 | 100         |
| 2026-06-02 | 0           |
| 2026-06-03 | 50          |

**Approach**
1. **Anchor:** start at the range's first date.
2. **Recursive term:** add one day until reaching the end date.
3. `LEFT JOIN` the generated spine to per-day order sums; `COALESCE` missing days to 0.

#### SQL Implementation
```sql
WITH RECURSIVE calendar AS (
    SELECT DATE '2026-06-01' AS day          -- anchor: first day
    UNION ALL
    SELECT day + INTERVAL '1 day'
    FROM calendar
    WHERE day < DATE '2026-06-03'            -- stop condition
),
daily AS (
    SELECT order_date::date AS day, SUM(total) AS daily_total
    FROM orders
    GROUP BY order_date::date
)
SELECT
    cal.day::date,
    COALESCE(d.daily_total, 0) AS daily_total
FROM calendar AS cal
LEFT JOIN daily AS d ON d.day = cal.day::date
ORDER BY cal.day;
```

The recursive `calendar` CTE emits one row per day: the anchor seeds the first date, and each recursion adds a day until the `WHERE day < end` stop condition fails. We then `LEFT JOIN` the real `daily` aggregates onto this complete spine so days with no orders survive, and `COALESCE(..., 0)` turns their `NULL` revenue into `0`. Without the spine, June 2 would simply be missing from the output.

**Note:** In PostgreSQL you'd normally reach for the built-in `generate_series('2026-06-01'::date, '2026-06-03', '1 day')` instead of hand-rolling recursion — it's simpler and faster. The recursive version is shown because assessments often ask you to demonstrate `WITH RECURSIVE`, and because some engines lack `generate_series`.

#### Alternative Solution
The idiomatic PostgreSQL one-liner spine:
```sql
SELECT g.day::date, COALESCE(d.daily_total, 0) AS daily_total
FROM generate_series(DATE '2026-06-01', DATE '2026-06-03', INTERVAL '1 day') AS g(day)
LEFT JOIN (
    SELECT order_date::date AS day, SUM(total) AS daily_total
    FROM orders GROUP BY order_date::date
) d ON d.day = g.day::date
ORDER BY g.day;
```
Prefer `generate_series` in production PostgreSQL; use the recursive CTE when the engine lacks a series generator or the exam explicitly wants recursion.

#### Interview Variations
- Generate a series of *numbers* 1..N recursively (running counter).
- Weekly buckets instead of daily (`+ INTERVAL '1 week'`).
- Fill gaps *per category* (cross join the spine with distinct categories, then LEFT JOIN).

#### Common Follow-up Questions
- **"Why do gap days need a spine?"** Aggregating `orders` only yields days that *have* orders; the spine supplies the missing calendar days.
- **"Recursive CTE vs `generate_series`?"** `generate_series` is simpler and faster in PostgreSQL; recursion is the portable fallback and what many assessments want to see.
- **"Where does the `0` come from?"** `COALESCE(daily_total, 0)` replaces the `NULL` produced by the LEFT JOIN on gap days.

---

### Practical Question 14: Layered Analytics — Chained CTEs for Category Leaderboard
- **Difficulty:** Hard
- **Estimated Time:** 14 min
- **Concepts Tested:** Multiple chained CTEs, referencing one CTE from another, window ranking, staged readability

**Problem Statement**
Build a category leaderboard in stages: (1) compute revenue per product, (2) roll it up to revenue per category and each category's share of grand-total revenue, (3) rank categories and return the top category per… well, return all categories ranked with their revenue, share %, and rank. Demonstrate chaining multiple CTEs where each references the previous.

**Example Input** (`order_items` joined to `products`)

| category    | product | line_revenue |
|-------------|---------|--------------|
| Electronics | Phone   | 1000         |
| Electronics | Cable   | 200          |
| Kitchen     | Mug     | 300          |

**Example Output**

| rank | category    | category_revenue | pct_of_total |
|------|-------------|------------------|--------------|
| 1    | Electronics | 1200             | 80.0         |
| 2    | Kitchen     | 300              | 20.0         |

**Approach**
1. CTE `product_rev`: revenue per product = `SUM(quantity * unit_price)`.
2. CTE `category_rev`: roll `product_rev` up to category totals (references the first CTE).
3. Final `SELECT`: rank categories with `RANK() OVER (ORDER BY revenue DESC)` and compute each category's share of the grand total using a window `SUM(...) OVER ()`.

#### SQL Implementation
```sql
WITH product_rev AS (
    SELECT
        p.id           AS product_id,
        p.category,
        SUM(oi.quantity * oi.unit_price) AS product_revenue
    FROM order_items AS oi
    JOIN products    AS p ON p.id = oi.product_id
    GROUP BY p.id, p.category
),
category_rev AS (
    SELECT
        category,
        SUM(product_revenue) AS category_revenue
    FROM product_rev                      -- references the CTE above
    GROUP BY category
)
SELECT
    RANK() OVER (ORDER BY category_revenue DESC)            AS rank,
    category,
    category_revenue,
    ROUND(
        100.0 * category_revenue
              / SUM(category_revenue) OVER (),              -- grand total
        1
    ) AS pct_of_total
FROM category_rev
ORDER BY rank;
```

Watch how the CTEs **chain**: `category_rev` reads from `product_rev`, and the final query reads from `category_rev`. Each stage has one clear job, so the whole pipeline reads top-to-bottom like a data-flow diagram — this is the real superpower of CTEs on complex reports. In the final query, `SUM(category_revenue) OVER ()` (empty `OVER()`) computes the *grand total across all rows* without collapsing them, letting us derive each category's percentage share in the same pass. `RANK()` orders categories by revenue; use `DENSE_RANK` if you want no gaps after ties.

**Performance/index considerations:** The heavy lifting is the initial aggregation over `order_items`; index `order_items(product_id)`. The category roll-up and window ranking operate on a small, already-reduced set, so they're cheap.

#### Alternative Solution
You *could* collapse this into nested subqueries, but it becomes an inside-out tangle:
```sql
SELECT RANK() OVER (ORDER BY category_revenue DESC) AS rank,
       category, category_revenue,
       ROUND(100.0 * category_revenue / SUM(category_revenue) OVER (), 1) AS pct
FROM (
    SELECT category, SUM(product_revenue) AS category_revenue
    FROM (
        SELECT p.category, SUM(oi.quantity * oi.unit_price) AS product_revenue
        FROM order_items oi JOIN products p ON p.id = oi.product_id
        GROUP BY p.id, p.category
    ) pr
    GROUP BY category
) cr
ORDER BY rank;
```
Same result, worse readability. Chained CTEs are strictly clearer once you exceed two stages — favor them for maintainability.

#### Interview Variations
- Add a *cumulative* share column (running sum of `pct_of_total`) to find the categories making up 80% of revenue (Pareto/ABC analysis).
- Top-selling *product* within each category (window `RANK() PARTITION BY category`).
- Compare each category's revenue to the prior period using a second staged CTE.

#### Common Follow-up Questions
- **"Can one CTE reference another?"** Yes — later CTEs in the same `WITH` list can reference earlier ones (but not vice-versa, unless recursive).
- **"What does `SUM(x) OVER ()` with an empty window do?"** Computes the aggregate over *all* rows of the result while keeping every row — perfect for share-of-total.
- **"CTE chain vs nested subqueries — performance?"** Usually identical plans in modern PostgreSQL; the difference is readability and maintainability.
- **"RANK vs DENSE_RANK for the leaderboard?"** `RANK` leaves gaps after ties (1,1,3); `DENSE_RANK` doesn't (1,1,2). Pick based on how you want tied categories numbered.


---


> Advanced querying and performance. This guide uses the **shared schema** defined in the earlier parts (`departments`, `employees(id, name, department_id, manager_id, salary, hire_date, email)`, `customers(id, name, email, city, created_at)`, `products(id, name, category, price, stock)`, `orders(id, customer_id, order_date, status, total)`, `order_items(id, order_id, product_id, quantity, unit_price)`). All examples are **PostgreSQL-flavored**.
>
> Mindset: in a live assessment, narrate your reasoning. Interviewers care less about the perfect keystroke and more about *why* you reach for a window function instead of a self-join, or *why* an index does or does not get used.

---

## Window Functions (ranking, LEAD/LAG, running totals)

Window functions compute a value **across a set of rows related to the current row** without collapsing them into a group. That is the key mental model: `GROUP BY` returns one row per group; a window function returns *every* row, annotated with an aggregate/rank computed over its `OVER (...)` frame. Master `PARTITION BY` (reset the calculation per group), `ORDER BY` (define order within the partition — critical for ranking, `LAG`/`LEAD`, and running totals), and the **frame clause** (`ROWS`/`RANGE BETWEEN ...`) which decides exactly which rows feed a running aggregate.

### Practical Question 1: Nth-Highest Salary Per Department
- **Difficulty:** Medium
- **Estimated Time:** 12 min
- **Concepts Tested:** `DENSE_RANK`, `PARTITION BY`, handling ties, filtering on a window result via a subquery/CTE

**Problem Statement**
For each department, find the employee(s) earning the **2nd-highest salary**. Ties must all be returned (two people tied for 2nd both count). Return the department name, employee name, and salary, ordered by department.

**Example Input** — `employees` (subset):

| id | name    | department_id | salary |
|----|---------|---------------|--------|
| 1  | Alice   | 10            | 9000   |
| 2  | Bob     | 10            | 8000   |
| 3  | Carol   | 10            | 8000   |
| 4  | Dan     | 20            | 7000   |
| 5  | Eve     | 20            | 6000   |

**Example Output** (with `departments` 10 = 'Sales', 20 = 'Eng'):

| department | name  | salary |
|------------|-------|--------|
| Eng        | Eve   | 6000   |
| Sales      | Bob   | 8000   |
| Sales      | Carol | 8000   |

**Approach**
1. Rank employees *within each department* by salary descending.
2. Use `DENSE_RANK`, not `ROW_NUMBER` or `RANK`: `DENSE_RANK` gives tied salaries the same rank **and** does not skip the next rank, so "2nd-highest salary" means the second *distinct* salary value — exactly the business intent.
3. You cannot filter on a window function in `WHERE` (windows are evaluated after `WHERE`), so wrap the ranking in a CTE and filter `rnk = 2` in the outer query.
4. Join to `departments` for the readable name and order the output.

#### SQL Implementation
```sql
WITH ranked AS (
    SELECT
        e.department_id,
        e.name,
        e.salary,
        DENSE_RANK() OVER (
            PARTITION BY e.department_id
            ORDER BY e.salary DESC
        ) AS rnk
    FROM employees e
)
SELECT d.name AS department, r.name, r.salary
FROM ranked r
JOIN departments d ON d.id = r.department_id
WHERE r.rnk = 2
ORDER BY d.name, r.name;
```
Clause-by-clause: `PARTITION BY e.department_id` restarts the ranking for every department; `ORDER BY e.salary DESC` makes rank 1 the top earner. `DENSE_RANK` is deliberate — with the `{9000, 8000, 8000}` Sales set, both 8000-earners get `rnk = 2` and are returned. Had we used `RANK`, they'd still be 2, but the *next* salary would jump to rank 4; had we used `ROW_NUMBER`, only one arbitrary 8000-row would survive `rnk = 2`, silently dropping a valid answer.

Index consideration: a composite index on `employees(department_id, salary DESC)` lets Postgres feed the window's `PARTITION BY`/`ORDER BY` directly from index order, avoiding a sort. On small tables it won't matter; on millions of rows it's the difference between a Sort node and an Index Scan.

#### Alternative Solution
A correlated subquery counting distinct higher salaries — portable to engines lacking window functions, but O(n) subquery executions and much harder to extend to "Nth":
```sql
SELECT d.name AS department, e.name, e.salary
FROM employees e
JOIN departments d ON d.id = e.department_id
WHERE (
    SELECT COUNT(DISTINCT e2.salary)
    FROM employees e2
    WHERE e2.department_id = e.department_id
      AND e2.salary > e.salary
) = 1   -- exactly one distinct salary is higher => 2nd highest
ORDER BY d.name, e.name;
```
Tradeoff: the window version is a single pass with one sort; the correlated version re-scans `employees` per row and typically loses on anything non-trivial. Prefer the window function unless you're on a legacy engine.

#### Interview Variations
- Generalize to the **Nth**-highest by parameterizing `WHERE rnk = :n`.
- Return the **top N** salaries per department instead of exactly the Nth (`WHERE rnk <= :n`).
- Break ties deterministically (e.g., earliest `hire_date` wins) by switching to `ROW_NUMBER` with a tiebreaker in `ORDER BY`.

#### Common Follow-up Questions
- **Why not `LIMIT 1 OFFSET 1`?** That works for a single global 2nd-highest, but can't produce a *per-department* result in one query, and it can't return ties.
- **`RANK` vs `DENSE_RANK` vs `ROW_NUMBER`?** `ROW_NUMBER` = unique sequential (no ties); `RANK` = ties share a number, gaps follow; `DENSE_RANK` = ties share a number, no gaps.
- **What if a department has fewer than 2 distinct salaries?** It simply produces no `rnk = 2` row — correct behavior; mention it proactively.

### Practical Question 2: Cumulative (Running) Revenue by Day
- **Difficulty:** Easy
- **Estimated Time:** 10 min
- **Concepts Tested:** `SUM() OVER (ORDER BY ...)`, running totals, window frames, ordered aggregation

**Problem Statement**
Produce a daily revenue report: for each `order_date`, show that day's total order revenue and the **cumulative revenue** from the first day through that day (a running total). Consider only `status = 'completed'` orders.

**Example Input** — `orders`:

| id | order_date | status    | total |
|----|------------|-----------|-------|
| 1  | 2026-01-01 | completed | 100   |
| 2  | 2026-01-01 | completed | 50    |
| 3  | 2026-01-02 | completed | 200   |
| 4  | 2026-01-03 | cancelled | 999   |
| 5  | 2026-01-03 | completed | 75    |

**Example Output**

| order_date | daily_revenue | cumulative_revenue |
|------------|---------------|--------------------|
| 2026-01-01 | 150           | 150                |
| 2026-01-02 | 200           | 350                |
| 2026-01-03 | 75            | 425                |

**Approach**
1. First collapse to one row per day with `GROUP BY order_date` and `SUM(total)` filtered to completed orders.
2. Then apply a window `SUM(...) OVER (ORDER BY order_date)` over that daily aggregate to accumulate.
3. Because the two operations happen at different granularities, run the grouping in a CTE and the window in the outer query — mixing a plain aggregate and a window aggregate over its result in one level gets confusing fast.
4. Understand the default frame: `SUM() OVER (ORDER BY x)` implies `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`, which is exactly a running total.

#### SQL Implementation
```sql
WITH daily AS (
    SELECT
        order_date,
        SUM(total) AS daily_revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY order_date
)
SELECT
    order_date,
    daily_revenue,
    SUM(daily_revenue) OVER (
        ORDER BY order_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_revenue
FROM daily
ORDER BY order_date;
```
I wrote the frame explicitly (`ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`) even though the default is nearly equivalent — being explicit signals you understand frames, and it sidesteps a real gotcha: the *default* frame is `RANGE`, which treats **peer rows** (equal `order_date`) as a single group and returns the same running total for all of them. After the `GROUP BY` here dates are unique, so `RANGE` and `ROWS` agree — but on un-aggregated data they differ, and interviewers love that distinction.

#### Alternative Solution
A self-join accumulation (pre-window-function style), shown to contrast cost:
```sql
SELECT d1.order_date, d1.daily_revenue,
       SUM(d2.daily_revenue) AS cumulative_revenue
FROM daily d1
JOIN daily d2 ON d2.order_date <= d1.order_date
GROUP BY d1.order_date, d1.daily_revenue
ORDER BY d1.order_date;
```
This is O(n²) — each day re-sums all prior days — and needs the same `daily` CTE anyway. The window version is a single ordered pass. Only reach for the self-join on engines without window support.

#### Interview Variations
- **Running total per customer**: add `PARTITION BY customer_id` so each customer's total resets.
- **7-day moving average** instead of cumulative: `ROWS BETWEEN 6 PRECEDING AND CURRENT ROW` with `AVG`.
- **Percent of grand total**: divide each day by `SUM(daily_revenue) OVER ()` (empty `OVER` = whole partition).

#### Common Follow-up Questions
- **What frame does `ORDER BY` alone give you?** `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` — a running total, but with peer-row grouping semantics.
- **How do you fill gaps for days with zero orders?** `LEFT JOIN` against a `generate_series` of dates so the running total carries across empty days.
- **Does the cumulative column force a sort?** Yes, on `order_date`; an index providing that order (or the upstream `GROUP BY`'s sort) can satisfy it.

### Practical Question 3: Month-over-Month Revenue Growth
- **Difficulty:** Medium
- **Estimated Time:** 15 min
- **Concepts Tested:** `LAG`, date truncation/grouping, growth-rate arithmetic, `NULL` handling for the first period

**Problem Statement**
Report monthly completed-order revenue and the **month-over-month growth percentage** versus the previous month. The first month has no prior period, so its growth should be `NULL`.

**Example Input** — `orders` (completed only):

| order_date | total |
|------------|-------|
| 2026-01-15 | 1000  |
| 2026-01-20 | 500   |
| 2026-02-10 | 1800  |
| 2026-03-01 | 900   |

**Example Output**

| month    | revenue | prev_revenue | growth_pct |
|----------|---------|--------------|------------|
| 2026-01  | 1500    | NULL         | NULL       |
| 2026-02  | 1800    | 1500         | 20.00      |
| 2026-03  | 900     | 1800         | -50.00     |

**Approach**
1. Bucket orders into months with `date_trunc('month', order_date)` and sum revenue per month.
2. Pull the previous month's revenue onto each row with `LAG(revenue) OVER (ORDER BY month)`.
3. Compute growth as `(revenue - prev) / prev * 100`. Cast to numeric so you don't get integer-division truncation.
4. Guard the first row (no previous month) and any month where the previous revenue is 0 to avoid divide-by-zero — `NULLIF(prev, 0)` makes that division yield `NULL` cleanly.

#### SQL Implementation
```sql
WITH monthly AS (
    SELECT
        date_trunc('month', order_date)::date AS month,
        SUM(total) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY date_trunc('month', order_date)
)
SELECT
    to_char(month, 'YYYY-MM') AS month,
    revenue,
    LAG(revenue) OVER (ORDER BY month) AS prev_revenue,
    ROUND(
        (revenue - LAG(revenue) OVER (ORDER BY month))
        / NULLIF(LAG(revenue) OVER (ORDER BY month), 0) * 100,
        2
    ) AS growth_pct
FROM monthly
ORDER BY month;
```
Why `LAG` over a self-join to "the previous month": `LAG` is order-aware and handles calendar gaps by *row position*, not by literal `month - 1` arithmetic — so if March had zero orders, `LAG` compares the next populated month to February, which is usually what stakeholders expect. If you instead need strict calendar adjacency (treating a missing month as a real gap), generate a complete month series and `LEFT JOIN`. `NULLIF(prev, 0)` protects against division by zero; the first month's `LAG` is `NULL`, so the whole expression is `NULL` — matching the spec.

Note: I repeat the `LAG(...)` expression for readability; Postgres recognizes the identical window and computes it once, so there's no performance penalty. If it bothers you, wrap the `LAG` in an inner CTE and reference the column.

#### Alternative Solution
Materialize `prev_revenue` first, then do arithmetic on a clean column — often more readable and easier to debug:
```sql
WITH monthly AS (
    SELECT date_trunc('month', order_date)::date AS month,
           SUM(total) AS revenue
    FROM orders WHERE status = 'completed'
    GROUP BY 1
),
with_prev AS (
    SELECT month, revenue,
           LAG(revenue) OVER (ORDER BY month) AS prev_revenue
    FROM monthly
)
SELECT to_char(month, 'YYYY-MM') AS month, revenue, prev_revenue,
       ROUND((revenue - prev_revenue) / NULLIF(prev_revenue, 0) * 100, 2) AS growth_pct
FROM with_prev
ORDER BY month;
```
Tradeoff: an extra CTE level for clarity vs. the single-level version's compactness. Same plan in Postgres — choose whichever reads better under interview pressure.

#### Interview Variations
- **Year-over-year**: partition by month-of-year and `LAG` over years, or use `LAG(revenue, 12)`.
- **Per-category MoM**: `PARTITION BY category ORDER BY month` so each category's growth is independent.
- **Fill missing months with 0** via `generate_series` before applying `LAG` so gaps count as real declines.

#### Common Follow-up Questions
- **Why `NULLIF(prev, 0)`?** Prevents a divide-by-zero error when a prior month had zero revenue; the row's growth becomes `NULL` instead of crashing.
- **`LAG` vs `LEAD`?** `LAG` looks backward (previous row), `LEAD` looks forward (next row); both take an optional offset and default value: `LAG(revenue, 1, 0)`.
- **Integer division trap?** If `revenue`/`total` are integers, `(a-b)/b` truncates. Cast to `numeric` (or multiply by `100.0`) before dividing.

### Practical Question 4: Top-3 Products Per Category + Dedupe with ROW_NUMBER
- **Difficulty:** Hard
- **Estimated Time:** 18 min
- **Concepts Tested:** `ROW_NUMBER`, top-N-per-group, deduplication, `PARTITION BY` with tiebreakers, joining aggregated sales

**Problem Statement**
Two-part, real-world task. (a) Rank products **within each category** by total units sold and return the **top 3 per category**. (b) The `products` table has accidental duplicate rows (same `name` + `category` inserted twice); show how to keep only the **first** occurrence per `(name, category)` and delete the rest.

**Example Input** — `products` + `order_items` units sold:

| product | category    | units_sold |
|---------|-------------|------------|
| A       | Electronics | 120        |
| B       | Electronics | 90         |
| C       | Electronics | 90         |
| D       | Electronics | 40         |
| E       | Books       | 200        |

**Example Output** (top-3 per category):

| category    | product | units_sold | rn |
|-------------|---------|------------|----|
| Books       | E       | 200        | 1  |
| Electronics | A       | 120        | 1  |
| Electronics | B       | 90         | 2  |
| Electronics | C       | 90         | 3  |

**Approach**
1. Aggregate `order_items` up to units sold per product, then join to `products` for category.
2. `ROW_NUMBER() OVER (PARTITION BY category ORDER BY units_sold DESC, product_id)` — `ROW_NUMBER` (not `RANK`) because "top 3" means *at most 3 rows*, and the `product_id` tiebreaker makes it deterministic when units are equal.
3. Filter `rn <= 3` in an outer query (again, can't filter a window in `WHERE`).
4. For dedupe: number duplicates with `ROW_NUMBER() OVER (PARTITION BY name, category ORDER BY id)` and delete every row with `rn > 1`.

#### SQL Implementation
```sql
-- Part (a): top-3 products per category by units sold
WITH product_sales AS (
    SELECT
        p.id AS product_id,
        p.name AS product,
        p.category,
        COALESCE(SUM(oi.quantity), 0) AS units_sold
    FROM products p
    LEFT JOIN order_items oi ON oi.product_id = p.id
    GROUP BY p.id, p.name, p.category
),
ranked AS (
    SELECT
        category, product, units_sold,
        ROW_NUMBER() OVER (
            PARTITION BY category
            ORDER BY units_sold DESC, product_id
        ) AS rn
    FROM product_sales
)
SELECT category, product, units_sold, rn
FROM ranked
WHERE rn <= 3
ORDER BY category, rn;
```
`LEFT JOIN` + `COALESCE` keeps never-sold products in the ranking with 0 units (drop to `INNER JOIN` if you only want products that actually sold). The `product_id` in `ORDER BY` is the deterministic tiebreaker — without it, tied products (B and C at 90) could swap between runs, and worse, which one falls off at the rank-3/rank-4 boundary becomes nondeterministic.

```sql
-- Part (b): delete duplicate products, keeping the lowest id per (name, category)
WITH dups AS (
    SELECT id,
           ROW_NUMBER() OVER (
               PARTITION BY name, category
               ORDER BY id
           ) AS rn
    FROM products
)
DELETE FROM products
WHERE id IN (SELECT id FROM dups WHERE rn > 1);
```
This is the canonical dedupe pattern: partition by the "sameness" key, order so the survivor is deterministic (lowest `id` = first inserted), and delete everything ranked > 1. In Postgres you can also use `ctid` if there's truly no unique column.

#### Alternative Solution
For top-N-per-group, a `LATERAL` join fetches each category's top rows with `LIMIT` — and can be dramatically faster when a supporting index exists (index-driven "loose index scan"):
```sql
SELECT c.category, t.product, t.units_sold
FROM (SELECT DISTINCT category FROM products) c
CROSS JOIN LATERAL (
    SELECT p.name AS product, SUM(oi.quantity) AS units_sold
    FROM products p
    LEFT JOIN order_items oi ON oi.product_id = p.id
    WHERE p.category = c.category
    GROUP BY p.name
    ORDER BY units_sold DESC NULLS LAST
    LIMIT 3
) t
ORDER BY c.category;
```
Tradeoff: `LATERAL` shines when categories are few and each has many products (it does one bounded top-3 lookup per category rather than ranking every row). `ROW_NUMBER` is simpler, single-pass, and easier to read; `LATERAL` wins on very large, well-indexed tables.

#### Interview Variations
- **Top-N with ties included** (could return >3 rows): switch to `RANK`/`DENSE_RANK` and filter `<= 3`.
- **Top product only per category**: `DISTINCT ON (category) ... ORDER BY category, units_sold DESC` — a concise Postgres idiom.
- **Dedupe but keep the most recent** row: order the partition by `created_at DESC` (or `id DESC`) so the newest survives.

#### Common Follow-up Questions
- **`ROW_NUMBER` vs `RANK` for top-N?** `ROW_NUMBER` guarantees exactly N rows; `RANK`/`DENSE_RANK` may return more when there are ties at the cutoff. Pick based on whether ties should be kept.
- **How do you prevent duplicates recurring after cleanup?** Add a `UNIQUE (name, category)` constraint (or unique index) so future inserts fail fast.
- **Is `DELETE ... WHERE id IN (subquery)` safe on large tables?** It can be heavy; batch it (`DELETE ... LIMIT` loops) or use `USING` self-join. Always run the `SELECT` first to preview what would be deleted.
- **What if there's no unique id to order by?** Use `ctid` (physical row locator) as the tiebreaker to distinguish otherwise-identical rows.

## Views & Materialized Views

A **view** is a stored query — a named virtual table that runs its underlying `SELECT` every time you reference it. It costs nothing to store, always reflects live data, and is your primary tool for encapsulating join/filter logic, exposing a stable interface over a changing schema, and enforcing row/column-level access. A **materialized view** stores the *result set* on disk: reads are as fast as a plain table, but the data is a snapshot that goes stale until you `REFRESH`. The core interview judgment is knowing which one to reach for — freshness vs. speed, and how to refresh without locking readers.

### Practical Question 5: A Reusable Reporting View
- **Difficulty:** Easy
- **Estimated Time:** 10 min
- **Concepts Tested:** `CREATE VIEW`, encapsulating joins/aggregation, when views help, view vs. base-table semantics

**Problem Statement**
Analysts keep rewriting the same customer-orders summary. Create a reusable **view** `customer_order_summary` exposing, per customer: name, city, number of completed orders, total completed spend, and the most recent order date. Then show how an analyst queries it.

**Example Input** — `customers` + `orders`:

| customer | city   | order_date | status    | total |
|----------|--------|------------|-----------|-------|
| Alice    | NYC    | 2026-01-02 | completed | 100   |
| Alice    | NYC    | 2026-02-10 | completed | 250   |
| Bob      | LA     | 2026-01-05 | cancelled | 99    |

**Example Output** — `SELECT * FROM customer_order_summary`:

| customer | city | order_count | total_spend | last_order_date |
|----------|------|-------------|-------------|-----------------|
| Alice    | NYC  | 2           | 350         | 2026-02-10      |
| Bob      | LA   | 0           | 0           | NULL            |

**Approach**
1. Write the aggregation `SELECT` first and verify it standalone — a view is only as good as its query.
2. `LEFT JOIN` customers to orders so customers with zero completed orders still appear (Bob).
3. Push the `status = 'completed'` filter into the join's `ON` clause (not `WHERE`), otherwise the `LEFT JOIN` degrades to an inner join and drops Bob.
4. Wrap it in `CREATE OR REPLACE VIEW`.

#### SQL Implementation
```sql
CREATE OR REPLACE VIEW customer_order_summary AS
SELECT
    c.id           AS customer_id,
    c.name         AS customer,
    c.city,
    COUNT(o.id)                       AS order_count,
    COALESCE(SUM(o.total), 0)         AS total_spend,
    MAX(o.order_date)                 AS last_order_date
FROM customers c
LEFT JOIN orders o
       ON o.customer_id = c.id
      AND o.status = 'completed'   -- filter in the JOIN, not WHERE
GROUP BY c.id, c.name, c.city;

-- Analyst usage: the view composes like any table
SELECT * FROM customer_order_summary
WHERE total_spend > 200
ORDER BY total_spend DESC;
```
The subtle-but-critical point is the filter placement. `AND o.status = 'completed'` inside `ON` means "only match completed orders during the join"; unmatched customers still get a row with `NULL`s, which `COUNT(o.id)` renders as 0 and `COALESCE` cleans up. Move that predicate to a `WHERE` and you'd null-filter Bob out entirely. Also note `COUNT(o.id)` (not `COUNT(*)`) — `COUNT(*)` would count the single all-`NULL` outer-join row as 1 for Bob.

The view stores no data; every `SELECT * FROM customer_order_summary` re-executes the join against live tables, so it's always current. The analyst's outer `WHERE total_spend > 200` is transparently pushed into the view's execution by the planner.

#### Alternative Solution
Same logic as a **CTE inline** in each query instead of a shared view:
```sql
WITH customer_order_summary AS ( /* ...same SELECT... */ )
SELECT * FROM customer_order_summary WHERE total_spend > 200;
```
Tradeoff: a CTE isn't reusable across sessions/queries and every analyst must copy it (drift risk). A view centralizes the definition — fix a bug once, everyone benefits. Use a CTE for one-off analysis, a view for shared, stable logic.

#### Interview Variations
- **Column privacy**: expose a view that omits `email`, then grant analysts access to the view but not the base table.
- **Updatable view**: a simple single-table view can be `INSERT`/`UPDATE`-able; this aggregated one cannot (explain why: no 1:1 row mapping).
- **Parameterize by city**: views can't take parameters — use a function returning `TABLE(...)` or push the filter to the caller.

#### Common Follow-up Questions
- **Does a view store data?** No — it's a stored query, re-run on each reference. Only a *materialized* view stores rows.
- **Is a view slower than the raw query?** No; the planner inlines the view definition and optimizes the combined query, so it's equivalent.
- **Can you index a view?** Not a regular view directly (index the base tables). You'd need a materialized view to build indexes on the stored result.
- **`CREATE OR REPLACE` limitation?** You can't change/reorder existing output columns' names/types with `REPLACE` — you must `DROP` and recreate.

### Practical Question 6: Materialized View for a Dashboard
- **Difficulty:** Medium
- **Estimated Time:** 14 min
- **Concepts Tested:** `CREATE MATERIALIZED VIEW`, `REFRESH ... CONCURRENTLY`, staleness tradeoffs, indexing a matview

**Problem Statement**
An executive dashboard shows daily revenue KPIs and is hit hundreds of times a minute, but the underlying `orders`/`order_items` tables are huge and the live aggregation is too slow for interactive use. Build a **materialized view** `daily_sales_dashboard` and describe how to keep it fresh without blocking dashboard readers.

**Example Output** — `SELECT * FROM daily_sales_dashboard ORDER BY sales_day DESC LIMIT 2`:

| sales_day  | order_count | gross_revenue | units_sold | avg_order_value |
|------------|-------------|---------------|------------|-----------------|
| 2026-07-08 | 412         | 51830.00      | 1290       | 125.80          |
| 2026-07-07 | 388         | 47210.50      | 1177       | 121.68          |

**Approach**
1. Write the heavy aggregation once, materialize its result to disk with `CREATE MATERIALIZED VIEW`.
2. Add a **unique index** on the grain (`sales_day`) — this is mandatory for `REFRESH CONCURRENTLY` and speeds dashboard lookups.
3. Refresh on a schedule (cron / `pg_cron`) using `REFRESH MATERIALIZED VIEW CONCURRENTLY` so readers keep seeing the old snapshot while the new one builds — no exclusive lock.
4. Explicitly document the staleness window (e.g., "data as of last refresh, up to N minutes old").

#### SQL Implementation
```sql
CREATE MATERIALIZED VIEW daily_sales_dashboard AS
SELECT
    o.order_date                              AS sales_day,
    COUNT(DISTINCT o.id)                      AS order_count,
    SUM(o.total)                              AS gross_revenue,
    SUM(oi.quantity)                          AS units_sold,
    ROUND(SUM(o.total) / NULLIF(COUNT(DISTINCT o.id), 0), 2) AS avg_order_value
FROM orders o
JOIN order_items oi ON oi.order_id = o.id
WHERE o.status = 'completed'
GROUP BY o.order_date
WITH DATA;

-- REQUIRED for REFRESH ... CONCURRENTLY and fast point lookups
CREATE UNIQUE INDEX ux_daily_sales_dashboard_day
    ON daily_sales_dashboard (sales_day);

-- Non-blocking refresh (run on a schedule)
REFRESH MATERIALIZED VIEW CONCURRENTLY daily_sales_dashboard;
```
`WITH DATA` populates immediately (`WITH NO DATA` creates it empty and unqueryable until the first refresh). The unique index is doing double duty: dashboards filter/sort by `sales_day` so it accelerates reads, and `REFRESH ... CONCURRENTLY` *requires* a unique index to diff old vs. new rows. Without `CONCURRENTLY`, `REFRESH` takes an `ACCESS EXCLUSIVE` lock and every dashboard query blocks until it finishes — unacceptable for a hot dashboard. With it, Postgres builds the new data in the background and swaps rows transactionally; the tradeoff is `CONCURRENTLY` is slower and needs more temp space.

Staleness note: readers see the *last refreshed* snapshot. If the business needs "no more than 5 minutes stale," schedule the refresh every 5 minutes and surface a `last_refreshed_at` timestamp on the dashboard.

#### Alternative Solution
A plain **summary table** maintained incrementally by triggers or an ETL job:
```sql
-- On each new/updated completed order, upsert that day's aggregate
INSERT INTO daily_sales_rollup (sales_day, gross_revenue, order_count)
VALUES (:day, :amount, 1)
ON CONFLICT (sales_day)
DO UPDATE SET gross_revenue = daily_sales_rollup.gross_revenue + EXCLUDED.gross_revenue,
              order_count   = daily_sales_rollup.order_count + 1;
```
Tradeoff: a matview is trivial to define and always internally consistent, but a **full** refresh re-computes everything (expensive at scale, and stale between refreshes). An incremental rollup table is always current and cheap per-event, but you own the maintenance logic and it's error-prone (late-arriving updates, cancellations, backfills). Choose matview for simplicity and tolerable staleness; incremental table when you need real-time and can invest in the plumbing.

#### Interview Variations
- **Partial/partitioned refresh**: matviews can't refresh a slice; simulate with a rollup table partitioned by month, refreshing only recent partitions.
- **Concurrent refresh prerequisite**: interviewer asks "why did `REFRESH CONCURRENTLY` error?" → missing unique index.
- **Chained matviews**: one matview built on another; discuss refresh ordering.

#### Common Follow-up Questions
- **View vs. materialized view — when each?** Regular view for always-fresh, cheap-to-compute logic; materialized view when the query is expensive and slight staleness is acceptable.
- **Does a matview auto-update when base data changes?** No — it's a snapshot; you must `REFRESH` (manually, via cron/`pg_cron`, or triggers).
- **Why does `REFRESH CONCURRENTLY` need a unique index?** It diffs the new result against existing rows to update in place without an exclusive lock; the unique key identifies matching rows.
- **Can you index a materialized view?** Yes — it's physically stored, so you can add as many indexes as the query patterns need.

### Practical Question 7: Updatable View with WITH CHECK OPTION
- **Difficulty:** Medium
- **Estimated Time:** 13 min
- **Concepts Tested:** updatable views, `WITH CHECK OPTION`, security barriers, when a view is/ isn't updatable

**Problem Statement**
The support team should only see and modify **active NYC customers**. Create a view `nyc_customers` scoped to `city = 'NYC'`, make it updatable, and ensure that an `INSERT`/`UPDATE` through the view **cannot** create or move a row out of that scope (e.g., cannot set `city = 'LA'`).

**Example Input** — `customers`:

| id | name  | city | email           |
|----|-------|------|-----------------|
| 1  | Alice | NYC  | alice@ex.com    |
| 2  | Bob   | LA   | bob@ex.com      |

**Example Output** — `SELECT * FROM nyc_customers`:

| id | name  | city | email        |
|----|-------|------|--------------|
| 1  | Alice | NYC  | alice@ex.com |

An `UPDATE nyc_customers SET city = 'LA' WHERE id = 1;` must **fail**.

**Approach**
1. A view over a **single table** with no aggregation/`DISTINCT`/`GROUP BY` is automatically updatable in Postgres.
2. Add `WITH CHECK OPTION` so any row written through the view must still satisfy the view's `WHERE` predicate.
3. This both scopes visibility (rows) and enforces the invariant on writes — a lightweight row-level access pattern.

#### SQL Implementation
```sql
CREATE OR REPLACE VIEW nyc_customers AS
SELECT id, name, city, email
FROM customers
WHERE city = 'NYC'
WITH CHECK OPTION;

-- Works: stays within scope
UPDATE nyc_customers SET name = 'Alice B.' WHERE id = 1;

-- Fails: WITH CHECK OPTION rejects rows that would leave the view
UPDATE nyc_customers SET city = 'LA' WHERE id = 1;
-- ERROR: new row violates check option for view "nyc_customers"

-- Fails: can't insert an out-of-scope row through the view
INSERT INTO nyc_customers (name, city, email)
VALUES ('Carl', 'LA', 'carl@ex.com');
-- ERROR: new row violates check option
```
`WITH CHECK OPTION` is the star here: without it, the view is still updatable but an `UPDATE ... SET city = 'LA'` would succeed and the row would silently **vanish from the view** (it no longer matches `WHERE city = 'NYC'`) — a classic footgun. The check option guarantees writes preserve the view's defining predicate. For genuine security (preventing a hostile user from inferring hidden rows via side-effecting functions), also add `WITH (security_barrier = true)`.

Why this view is updatable at all: it selects from one base table, every non-selected column is nullable or has a default, and there's no `GROUP BY`, `DISTINCT`, window function, or set operation. Break any of those rules and you'd need an `INSTEAD OF` trigger to make writes work.

#### Alternative Solution
For a non-trivial (multi-table or aggregated) view, use an `INSTEAD OF` trigger to define custom write behavior:
```sql
CREATE FUNCTION nyc_customers_upd() RETURNS trigger AS $$
BEGIN
    IF NEW.city <> 'NYC' THEN
        RAISE EXCEPTION 'out of scope city: %', NEW.city;
    END IF;
    UPDATE customers SET name = NEW.name, email = NEW.email WHERE id = NEW.id;
    RETURN NEW;
END; $$ LANGUAGE plpgsql;

CREATE TRIGGER trg_nyc_upd INSTEAD OF UPDATE ON nyc_customers
FOR EACH ROW EXECUTE FUNCTION nyc_customers_upd();
```
Tradeoff: `WITH CHECK OPTION` is declarative and free for simple views; `INSTEAD OF` triggers are the only option for complex views but add procedural code you must maintain and test.

#### Interview Variations
- **`LOCAL` vs `CASCADED` check option**: `CASCADED` (default) enforces underlying views' conditions too; `LOCAL` checks only this view.
- **Column-level security**: omit `email` from the view to hide it, then grant on the view only.
- **Combine with RLS**: contrast view-based scoping vs. PostgreSQL Row-Level Security policies for multi-tenant apps.

#### Common Follow-up Questions
- **When is a view auto-updatable?** Single base table (or another updatable view), no `DISTINCT`/`GROUP BY`/`HAVING`/window/aggregate/set-op, and the target columns map directly to base columns.
- **What does `WITH CHECK OPTION` actually prevent?** Writes that would produce a row not visible through the view — stops rows from silently escaping the view's filter.
- **`security_barrier` — why?** Stops the planner from pushing user-supplied functions below the view's filter, which could leak filtered-out rows. Important for security-sensitive views.

### Practical Question 8: Diagnosing and Fixing a Stale Materialized View
- **Difficulty:** Hard
- **Estimated Time:** 16 min
- **Concepts Tested:** matview freshness tracking, refresh strategy, `pg_cron` scheduling, monitoring staleness, refresh dependency ordering

**Problem Statement**
Users report the dashboard from Q6 (`daily_sales_dashboard`) "shows yesterday's numbers." Design a robust freshness strategy: track when the matview was last refreshed, expose that to consumers, automate refreshes, and handle a chain where a second matview depends on the first.

**Example Output** — a freshness check:

| matview               | last_refreshed_at   | age         | is_stale |
|-----------------------|---------------------|-------------|----------|
| daily_sales_dashboard | 2026-07-09 08:55:00 | 00:04:32    | false    |

**Approach**
1. Matviews don't record their own refresh time — maintain it yourself in a small `matview_refresh_log` table, updated by the refresh routine.
2. Wrap the refresh in a function that `REFRESH ... CONCURRENTLY`s and then upserts the log timestamp atomically.
3. Schedule it with `pg_cron` (or an external scheduler); compute staleness as `now() - last_refreshed_at`.
4. For dependency chains, refresh base matviews **before** dependents in a single ordered routine.

#### SQL Implementation
```sql
-- 1. Freshness bookkeeping table
CREATE TABLE IF NOT EXISTS matview_refresh_log (
    matview          text PRIMARY KEY,
    last_refreshed_at timestamptz NOT NULL DEFAULT now()
);

-- 2. Refresh routine that also records the timestamp
CREATE OR REPLACE FUNCTION refresh_dashboards() RETURNS void AS $$
BEGIN
    -- refresh base matview first, then any dependents, in order
    REFRESH MATERIALIZED VIEW CONCURRENTLY daily_sales_dashboard;
    INSERT INTO matview_refresh_log (matview, last_refreshed_at)
    VALUES ('daily_sales_dashboard', now())
    ON CONFLICT (matview)
    DO UPDATE SET last_refreshed_at = EXCLUDED.last_refreshed_at;

    -- e.g. a rollup built on top of the daily matview
    REFRESH MATERIALIZED VIEW CONCURRENTLY weekly_sales_dashboard;
    INSERT INTO matview_refresh_log (matview, last_refreshed_at)
    VALUES ('weekly_sales_dashboard', now())
    ON CONFLICT (matview) DO UPDATE SET last_refreshed_at = EXCLUDED.last_refreshed_at;
END; $$ LANGUAGE plpgsql;

-- 3. Schedule every 5 minutes with pg_cron
SELECT cron.schedule('refresh-dashboards', '*/5 * * * *', 'SELECT refresh_dashboards()');

-- 4. Staleness monitor
SELECT matview,
       last_refreshed_at,
       now() - last_refreshed_at              AS age,
       now() - last_refreshed_at > interval '10 minutes' AS is_stale
FROM matview_refresh_log;
```
The crux: Postgres gives you no built-in `last_refreshed` for matviews, so the "stale dashboard" complaint is unactionable until you *measure* it. The log table plus the monitor query turn "it feels old" into "it's 4m32s old, threshold 10m, not stale" — or a paging alert when it is. Doing the `REFRESH` and the log upsert in the same function keeps them consistent; ordering the refreshes (base before dependent) ensures `weekly_sales_dashboard` is computed from freshly-refreshed daily data, not the previous snapshot.

Failure handling note: if a `REFRESH` throws, the function aborts and the log timestamp isn't updated — so the monitor correctly keeps showing the *old* time and staleness climbs, tripping the alert. That's the desired behavior; don't swallow the exception.

#### Alternative Solution
Event-driven refresh via a trigger on the base tables instead of a fixed schedule:
```sql
CREATE OR REPLACE FUNCTION mark_dashboard_dirty() RETURNS trigger AS $$
BEGIN
    -- enqueue a refresh request rather than refreshing inline (too slow in a trigger)
    INSERT INTO refresh_queue (matview, requested_at) VALUES ('daily_sales_dashboard', now())
    ON CONFLICT (matview) DO UPDATE SET requested_at = now();
    RETURN NULL;
END; $$ LANGUAGE plpgsql;
-- a worker drains refresh_queue and runs REFRESH CONCURRENTLY
```
Tradeoff: scheduled refresh is dead simple and predictable but wastes work when data is idle and lags when data is busy. Event-driven refresh is fresher and skips idle periods, but you must never `REFRESH` inline in a trigger (it's slow and would serialize writes) — you enqueue and let a background worker do it. Most teams start with cron and graduate to event-driven only when freshness SLAs demand it.

#### Interview Variations
- **SLA-based alerting**: page when `age > interval '15 minutes'`; wire the monitor query into your metrics system.
- **Refresh dependency graph**: with many chained matviews, topologically sort them and refresh in order.
- **Zero-downtime full rebuild**: build a new matview under a temp name, then `ALTER ... RENAME` swap.

#### Common Follow-up Questions
- **Does Postgres track matview refresh time?** No native column; you maintain it yourself (log table) or read `pg_stat_user_tables` heuristics.
- **Why refresh base matviews before dependents?** So dependents compute from current data; otherwise you propagate a stale snapshot one layer down.
- **Cron vs. event-driven refresh?** Cron: simple, predictable, wasteful/laggy at extremes. Event-driven: fresher, efficient, more moving parts. Match to the freshness SLA.
- **Can a failed `REFRESH CONCURRENTLY` leave partial data?** No — it's transactional; on failure the old snapshot remains intact, which is exactly why the monitor keeps showing the old timestamp.

## Indexing, EXPLAIN & Query Optimization

This is where SQL meets systems thinking. An index is a sorted, auxiliary data structure (Postgres default: a B-tree) that trades write cost and disk space for dramatically faster reads on selective predicates, joins, and ordered scans. The skill an interviewer probes is diagnostic: read an `EXPLAIN (ANALYZE, BUFFERS)` plan, find the expensive node (a `Seq Scan` over millions of rows, a `Sort` spilling to disk, a bad row estimate), and add — or fix — exactly the right index. Just as important is knowing when an index **won't** help (or won't be used) so you don't cargo-cult indexes onto every column.

> How to read `EXPLAIN ANALYZE`: `cost=start..total` are the planner's estimates (arbitrary units); `actual time=start..total` is real milliseconds; `rows` is per-loop row count; `loops` multiplies it. Compare **estimated vs actual rows** — big gaps mean stale statistics and bad plans. `BUFFERS` shows pages read (`shared hit` = cache, `read` = disk). Read plans **inside-out**: the most-indented node runs first.

### Practical Question 9: This Query Is Slow — Add the Right Index
- **Difficulty:** Medium
- **Estimated Time:** 15 min
- **Concepts Tested:** `EXPLAIN ANALYZE`, Seq Scan vs Index Scan, choosing an index for a selective predicate, before/after measurement

**Problem Statement**
This customer-facing query looks up a user's recent orders and has become slow as `orders` grew to millions of rows:
```sql
SELECT id, order_date, total
FROM orders
WHERE customer_id = 4242
ORDER BY order_date DESC
LIMIT 20;
```
Diagnose it with `EXPLAIN ANALYZE` and add the right index. Show before/after plans.

**Example Input** — `orders`: ~5,000,000 rows, ~50k distinct customers (so ~100 orders each). No index on `customer_id`.

**Example Output** — the query returns 20 rows; the goal is to change *how* they're fetched.

**Approach**
1. Run `EXPLAIN (ANALYZE, BUFFERS)` on the current query to confirm the bottleneck.
2. Recognize the pattern: an **equality filter** (`customer_id = 4242`) plus an **`ORDER BY ... LIMIT`**. The ideal index matches the equality column first, then the sort column.
3. Create a composite index `(customer_id, order_date DESC)` so Postgres can seek to the customer and walk pre-sorted rows, stopping after 20.
4. Re-run `EXPLAIN ANALYZE` and compare.

#### SQL Implementation
```sql
-- BEFORE: no supporting index
EXPLAIN (ANALYZE, BUFFERS)
SELECT id, order_date, total
FROM orders
WHERE customer_id = 4242
ORDER BY order_date DESC
LIMIT 20;
```
```text
Limit  (cost=189234.12..189234.17 rows=20 width=20)
       (actual time=812.443..812.449 rows=20 loops=1)
  Buffers: shared hit=1204 read=44231
  ->  Sort  (cost=189234.12..189484.63 rows=100204 width=20)
            (actual time=812.441..812.444 rows=20 loops=1)
        Sort Key: order_date DESC
        Sort Method: top-N heapsort  Memory: 27kB
        ->  Seq Scan on orders  (cost=0.00..186567.00 rows=100 width=20)
                 (actual time=0.021..798.113 rows=98 loops=1)
              Filter: (customer_id = 4242)
              Rows Removed by Filter: 4999902
Planning Time: 0.12 ms
Execution Time: 812.5 ms
```
Read it: the `Seq Scan` reads all 5M rows and throws away 4,999,902 (`Rows Removed by Filter`) to find ~98 matches — that's the 800ms, and `read=44231` disk pages confirm it's hammering storage. The `top-N heapsort` on top is cheap; the scan is the villain.

```sql
-- FIX: composite index matching (equality col, then sort col)
CREATE INDEX idx_orders_customer_date
    ON orders (customer_id, order_date DESC);
```
```sql
-- AFTER: same query
EXPLAIN (ANALYZE, BUFFERS)
SELECT id, order_date, total
FROM orders
WHERE customer_id = 4242
ORDER BY order_date DESC
LIMIT 20;
```
```text
Limit  (cost=0.43..8.71 rows=20 width=20)
       (actual time=0.028..0.061 rows=20 loops=1)
  Buffers: shared hit=23
  ->  Index Scan using idx_orders_customer_date on orders
            (cost=0.43..41.02 rows=100 width=20)
            (actual time=0.026..0.055 rows=20 loops=1)
        Index Cond: (customer_id = 4242)
Planning Time: 0.15 ms
Execution Time: 0.08 ms
```
Now it's an `Index Scan` with `Index Cond: (customer_id = 4242)` — Postgres seeks directly to that customer and, because `order_date DESC` is the second index column, the rows come out **already sorted**, so there's **no Sort node at all**. It reads 23 buffers instead of 45k and stops after 20 rows. ~812ms → ~0.08ms, roughly a 10,000x improvement. That `LIMIT 20` is why the ordered index is so powerful: the engine never touches row 21.

Index-order matters: I put `order_date DESC` to match the query's sort direction. A B-tree can scan backwards, so plain `order_date` usually works too, but matching the direction avoids any ambiguity and helps mixed-direction multicolumn sorts.

#### Alternative Solution
A **covering index** with `INCLUDE` to get an index-only scan (no heap fetch for `total`):
```sql
CREATE INDEX idx_orders_customer_date_cover
    ON orders (customer_id, order_date DESC) INCLUDE (id, total);
```
Now all three selected columns (`id`, `order_date`, `total`) live in the index, so Postgres can return results without visiting the table heap at all (`Index Only Scan`), shaving the heap lookups. Tradeoff: a wider index costs more disk and slows writes; worth it for a hot, read-heavy endpoint, overkill for a rarely-run query.

#### Interview Variations
- **Add a status filter** (`WHERE customer_id = ? AND status = 'completed'`): does the index still help? Consider `(customer_id, status, order_date DESC)`.
- **Pagination with keyset** instead of `OFFSET`: `WHERE customer_id = ? AND order_date < :last_seen` rides the same index efficiently.
- **What if selectivity is poor** (customer has 4M of 5M orders)? The planner may correctly prefer a Seq Scan.

#### Common Follow-up Questions
- **Why did the Sort disappear?** The index already stores rows in `order_date` order within each `customer_id`, so the sort is satisfied for free.
- **Column order in the index — why `customer_id` first?** Equality/most-selective predicate leads; the sort column follows so ordered retrieval works. Reverse it and the equality can't be a clean seek.
- **Cost of the index?** Slower `INSERT`/`UPDATE`/`DELETE` on `orders`, more disk; you're trading write throughput for read latency.
- **Do I need `ANALYZE` after creating it?** `CREATE INDEX` updates stats for the index, but run `ANALYZE orders` if table stats are stale so the planner estimates correctly.

### Practical Question 10: Composite Index & the Leftmost-Prefix Rule
- **Difficulty:** Medium
- **Estimated Time:** 14 min
- **Concepts Tested:** multicolumn index column ordering, leftmost-prefix rule, which queries an index can serve, avoiding redundant indexes

**Problem Statement**
You have `CREATE INDEX idx_emp ON employees (department_id, salary, hire_date);`. For each of the following queries, state whether `idx_emp` can be used and why, then recommend the correct index set:
1. `WHERE department_id = 5`
2. `WHERE department_id = 5 AND salary > 100000`
3. `WHERE salary > 100000`
4. `WHERE department_id = 5 ORDER BY salary`
5. `WHERE department_id = 5 AND hire_date > '2020-01-01'`

**Example Output** — usage verdict:

| query | uses idx_emp? | reason |
|-------|---------------|--------|
| 1     | Yes           | uses leftmost column |
| 2     | Yes (full)    | equality on col1 + range on col2 |
| 3     | No (poorly)   | skips leftmost column |
| 4     | Yes           | filter col1, ordered by col2 |
| 5     | Partial       | col1 seek, then filter (skips salary) |

**Approach**
1. Internalize the **leftmost-prefix rule**: a composite B-tree on `(a, b, c)` can serve predicates on `a`, `(a, b)`, `(a, b, c)` — i.e., any *prefix* — but not `b` or `c` alone, because the index is sorted by `a` first.
2. A **range** (`>`, `<`, `BETWEEN`) on a column consumes the index's ordering from that point on; columns *after* the range column can't be used for further seeking, only filtering.
3. Ordering (`ORDER BY b`) can be satisfied by the index once `a` is pinned by equality.
4. Recommend indexes that cover real access patterns without redundancy.

#### SQL Implementation
```sql
-- Given:
CREATE INDEX idx_emp ON employees (department_id, salary, hire_date);

-- 1) WHERE department_id = 5
--    USES IT: department_id is the leftmost column -> clean index seek.

-- 2) WHERE department_id = 5 AND salary > 100000
--    USES IT FULLY: equality on col1 positions the scan, range on col2
--    walks the sorted salaries. Optimal case for this index.

-- 3) WHERE salary > 100000
--    CANNOT use it as a seek: salary is the 2nd column, so the index
--    isn't sorted by salary globally. Planner may do a full Index-Only
--    Scan if the index is cheaper than the heap, but no efficient seek.
--    FIX: a dedicated index.
CREATE INDEX idx_emp_salary ON employees (salary);

-- 4) WHERE department_id = 5 ORDER BY salary
--    USES IT: department_id equality pins the prefix, and within that
--    the index is already ordered by salary -> no Sort node needed.

-- 5) WHERE department_id = 5 AND hire_date > '2020-01-01'
--    PARTIAL: seeks on department_id, but hire_date is the 3rd column and
--    salary (the 2nd) is unconstrained, so hire_date can't be a seek key --
--    it's applied as a filter after fetching all of dept 5's rows.
--    FIX if this is hot:
CREATE INDEX idx_emp_dept_hire ON employees (department_id, hire_date);
```
The mental model: a `(department_id, salary, hire_date)` index is like a phone book sorted by last name, then first name, then middle name. You can find "everyone named Smith" (col1), or "Smith, John" (col1+col2), but you **cannot** efficiently find "everyone whose first name is John" (col2 alone) — you'd scan the whole book. Query 5 is subtle: because `salary` (the middle column) is skipped, the index can locate dept 5's block but `hire_date` sits *behind* the unconstrained salary, so it degrades to filtering each of dept 5's rows rather than a range seek.

Redundancy note: with `idx_emp` present, a separate index on `(department_id)` or `(department_id, salary)` would be **redundant** — `idx_emp`'s leftmost prefixes already cover them. Don't create those; you'd only add write overhead.

#### Alternative Solution
Reorder columns by selectivity/usage, or use **partial indexes** for skewed access:
```sql
-- If most queries filter active, high earners, a partial index is smaller & faster
CREATE INDEX idx_emp_high_earners
    ON employees (department_id, hire_date)
    WHERE salary > 100000;
```
Tradeoff: a partial index is tiny and lightning-fast for the matching subset but only helps queries whose `WHERE` implies the same predicate. General composite indexes serve more queries; partial indexes optimize a specific hot path. Column ordering is the other lever — lead with the column used for equality in the most queries.

#### Interview Variations
- **Reverse the column order** to `(salary, department_id)` and re-answer all five — flips which queries are served.
- **Index skip scan**: some engines (and newer Postgres) can "skip scan" a leading column with few distinct values; discuss its limits.
- **Covering an `ORDER BY ... LIMIT`** across two columns: which composite serves it without a sort?

#### Common Follow-up Questions
- **State the leftmost-prefix rule in one sentence.** A composite index can serve any query that constrains a *contiguous prefix* of its columns starting from the first.
- **Why can't col2 alone use the index efficiently?** Rows are globally ordered by col1 first; col2 is only sorted *within* each col1 value, so there's no global col2 ordering to seek on.
- **Does a range on col1 let col2 seek?** No — once you hit a range predicate, subsequent columns can only be used as filters, not seek keys.
- **Is `(a)` redundant if `(a, b)` exists?** Yes for lookups on `a`; the `(a,b)` index's leftmost prefix covers it. Keep the narrower one only if index size for `a`-only scans matters a lot.

### Practical Question 11: Why Isn't My Index Being Used?
- **Difficulty:** Hard
- **Estimated Time:** 18 min
- **Concepts Tested:** non-sargable predicates (function on column), low selectivity, expression/functional indexes, type mismatch, stale statistics

**Problem Statement**
Three "the index exists but the query still does a Seq Scan" scenarios. Diagnose *why* each ignores its index and fix it:
1. `WHERE lower(email) = 'alice@ex.com'` with an index on `email`.
2. `WHERE status = 'completed'` with an index on `orders(status)`, but 95% of rows are `'completed'`.
3. `WHERE EXTRACT(YEAR FROM order_date) = 2026` with an index on `order_date`.

**Example Input** — `customers` with `idx_email ON customers(email)`; `orders` with `idx_status ON orders(status)` and `idx_order_date ON orders(order_date)`.

**Example Output** — the fixes turn Seq Scans into Index Scans (except case 2, where a Seq Scan is *correct*).

**Approach**
1. Learn the term **sargable** (Search-ARGument-able): a predicate is sargable if the indexed column appears *bare* on one side so the B-tree can seek. Wrapping the column in a function (`lower(col)`, `EXTRACT(...)`) makes it **non-sargable** — the engine must compute the function per row, defeating the index.
2. For low-selectivity predicates, a Seq Scan is genuinely cheaper than an index scan + heap fetches — the planner is right to skip the index.
3. Fixes: build an **expression index** matching the predicate, rewrite to a **sargable range**, or accept the Seq Scan.

#### SQL Implementation
```sql
-- CASE 1: function on the column makes idx_email unusable
EXPLAIN ANALYZE
SELECT * FROM customers WHERE lower(email) = 'alice@ex.com';
```
```text
Seq Scan on customers  (cost=0.00..21930.00 rows=5000 width=...)
   Filter: (lower(email) = 'alice@ex.com')
   Rows Removed by Filter: 999999
```
The index is on `email`, but the query searches `lower(email)` — a *different* value the plain index doesn't store sorted. Fix with an **expression (functional) index** that indexes exactly what you query:
```sql
CREATE INDEX idx_email_lower ON customers (lower(email));
-- Now the same query does an Index Scan using idx_email_lower.
-- (Alternatively, store email already-normalized and index it plain,
--  or use a case-insensitive citext column.)
```

```sql
-- CASE 2: low selectivity -- the index is skipped ON PURPOSE
EXPLAIN ANALYZE
SELECT * FROM orders WHERE status = 'completed';   -- 95% of rows match
```
```text
Seq Scan on orders  (cost=0.00..96500.00 rows=4750000 width=...)
   Filter: (status = 'completed')
```
This is **correct**, not a bug. Reading 95% of the table via an index means ~4.75M random heap lookups — far slower than one sequential sweep. The planner's cost model chooses the Seq Scan deliberately. Don't "fix" it. If you frequently need the *rare* statuses, index those with a **partial index**:
```sql
CREATE INDEX idx_orders_active ON orders (order_date)
    WHERE status IN ('pending', 'processing');   -- indexes only the 5% minority
```

```sql
-- CASE 3: EXTRACT() wraps the column -> non-sargable
EXPLAIN ANALYZE
SELECT * FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2026;
```
```text
Seq Scan on orders  (cost=0.00..111000.00 ...)
   Filter: (EXTRACT(year FROM order_date) = 2026::numeric)
```
Rewrite to a **sargable range** so the bare `order_date` can use `idx_order_date`:
```sql
SELECT * FROM orders
WHERE order_date >= '2026-01-01'
  AND order_date <  '2027-01-01';   -- half-open range, index-friendly
```
```text
Index Scan using idx_order_date on orders
   Index Cond: (order_date >= '2026-01-01' AND order_date < '2027-01-01')
```
Same result set, but now it's a range seek. The rule of thumb: **never wrap an indexed column in a function or arithmetic in `WHERE`** — transform the *constant* side instead, or build an expression index.

A fourth common culprit worth naming: **type mismatch**. `WHERE varchar_col = 123` (int literal) or comparing `bigint` to `int` can force an implicit cast on the column and skip the index; and **stale statistics** (after a big bulk load, before `ANALYZE`) make the planner mis-estimate selectivity and choose a Seq Scan — always `ANALYZE` after large data changes.

#### Alternative Solution
Instead of an expression index for case 1, **normalize on write**: store `email` lowercased (or use the `citext` extension for case-insensitive comparison) so the plain index works and every query is naturally sargable:
```sql
-- citext: comparisons are case-insensitive, plain index seeks work
ALTER TABLE customers ALTER COLUMN email TYPE citext;
CREATE INDEX idx_email ON customers (email);
SELECT * FROM customers WHERE email = 'Alice@Ex.com';  -- uses the index
```
Tradeoff: expression indexes keep the raw column intact and are surgical, but you must remember to match the exact expression in every query. Normalizing/`citext` makes correctness the default but changes the stored data model and affects all consumers.

#### Interview Variations
- **`LIKE 'abc%'` vs `LIKE '%abc'`**: prefix match is sargable (uses a B-tree, esp. with `text_pattern_ops`); leading-wildcard needs a trigram (`pg_trgm`) GIN index.
- **`OR` across columns** preventing index use: rewrite as `UNION` of two indexable branches.
- **Implicit cast** on a join key silently disabling an index — how to spot it in the plan (`Filter` showing a `::cast`).

#### Common Follow-up Questions
- **What does "sargable" mean?** A predicate the engine can satisfy by seeking an index — the indexed column appears un-wrapped so the B-tree ordering applies.
- **Is a Seq Scan always bad?** No. For low-selectivity predicates or small tables it's the *optimal* choice; forcing an index would be slower.
- **How do you index `lower(email)`?** An expression index: `CREATE INDEX ... ON customers (lower(email))`, then always query `WHERE lower(email) = ...`.
- **Query was fast, got slow after a bulk load — why?** Likely stale statistics; run `ANALYZE` (or `VACUUM ANALYZE`) so the planner re-estimates and picks the index again.
- **How do I force an index to test?** `SET enable_seqscan = off;` for diagnosis only — it proves whether the index *can* help; never leave it off in production.

### Practical Question 12: End-to-End Optimization of a Slow Reporting Query
- **Difficulty:** Hard
- **Estimated Time:** 20 min
- **Concepts Tested:** reading a full `EXPLAIN ANALYZE` plan, join strategy, multiple index decisions, row-estimate errors, holistic tuning

**Problem Statement**
This monthly revenue-by-city report times out. Optimize it end-to-end: interpret the plan, decide on indexes, and rewrite if needed.
```sql
SELECT c.city,
       COUNT(DISTINCT o.id)  AS orders,
       SUM(oi.quantity * oi.unit_price) AS revenue
FROM customers c
JOIN orders o       ON o.customer_id = c.id
JOIN order_items oi ON oi.order_id = o.id
WHERE o.status = 'completed'
  AND o.order_date >= '2026-06-01'
  AND o.order_date <  '2026-07-01'
GROUP BY c.city
ORDER BY revenue DESC;
```

**Example Output**

| city | orders | revenue  |
|------|--------|----------|
| NYC  | 1820   | 254300.0 |
| LA   | 1533   | 219880.5 |

**Approach**
1. Get the baseline plan with `EXPLAIN (ANALYZE, BUFFERS)`; find the dominant node and any estimate/actual row mismatches.
2. The `orders` filter (`status` + `order_date` range) is the driving predicate — index it so the scan starts small.
3. Ensure the joins ride indexes on the foreign keys (`orders.customer_id`, `order_items.order_id`).
4. Re-measure; verify the join strategy (hash vs nested loop) is sane given the now-smaller driving set.

#### SQL Implementation
```sql
-- BASELINE
EXPLAIN (ANALYZE, BUFFERS)
SELECT c.city, COUNT(DISTINCT o.id) AS orders,
       SUM(oi.quantity * oi.unit_price) AS revenue
FROM customers c
JOIN orders o ON o.customer_id = c.id
JOIN order_items oi ON oi.order_id = o.id
WHERE o.status = 'completed'
  AND o.order_date >= '2026-06-01' AND o.order_date < '2026-07-01'
GROUP BY c.city
ORDER BY revenue DESC;
```
```text
Sort  (actual time=4210.6..4210.7 rows=2 loops=1)
  Sort Key: (sum(oi.quantity * oi.unit_price)) DESC
  ->  HashAggregate  (actual time=4198.1..4210.4 rows=2 loops=1)
        Group Key: c.city
        ->  Hash Join  (actual time=980.2..3900.5 rows=210400 loops=1)
              Hash Cond: (oi.order_id = o.id)
              ->  Seq Scan on order_items oi  (actual time=0.01..1400.9 rows=8000000 loops=1)
              ->  Hash  (actual time=978.4..978.4 rows=52600 loops=1)
                    ->  Hash Join  (actual time=120.3..950.1 rows=52600 loops=1)
                          Hash Cond: (o.customer_id = c.id)
                          ->  Seq Scan on orders o  (actual time=95.1..820.4 rows=52600 loops=1)
                                Filter: (status = 'completed' AND order_date >= ... AND order_date < ...)
                                Rows Removed by Filter: 4947400
                          ->  Hash  (actual time=25.0..25.0 rows=50000 loops=1)
                                ->  Seq Scan on customers c (rows=50000)
Execution Time: 4215.0 ms
```
Two villains stand out. First, the `Seq Scan on orders` reads all 5M rows and filters out 4,947,400 to keep 52,600 — the driving predicate has no index. Second, the `Seq Scan on order_items` reads all **8M** rows to feed the hash join, even though only ~210k are relevant. Fix both:
```sql
-- Index the driving predicate: equality (status) then range (order_date)
CREATE INDEX idx_orders_status_date
    ON orders (status, order_date)
    INCLUDE (customer_id, id);      -- covers the join keys too

-- Index the join key on the large child table
CREATE INDEX idx_order_items_order_id
    ON order_items (order_id);
```
```text
-- AFTER
Sort  (actual time=88.3..88.3 rows=2 loops=1)
  ->  HashAggregate  (actual time=86.0..88.1 rows=2 loops=1)
        ->  Nested Loop  (actual time=0.20..70.4 rows=210400 loops=1)
              ->  Nested Loop  (actual time=0.10..30.2 rows=52600 loops=1)
                    ->  Index Scan using idx_orders_status_date on orders o
                          Index Cond: (status='completed' AND order_date>='2026-06-01'
                                       AND order_date<'2026-07-01')  (rows=52600)
                    ->  Index Scan using customers_pkey on customers c
                          Index Cond: (id = o.customer_id)
              ->  Index Scan using idx_order_items_order_id on order_items oi
                    Index Cond: (order_id = o.id)  (rows≈4 per order)
Execution Time: 92.0 ms
```
~4215ms → ~92ms. The `orders` Seq Scan became an `Index Scan` that fetches only June's completed orders directly; because the index `INCLUDE`s `customer_id` and `id`, the join keys come from the index without heap visits. The `order_items` full scan became a per-order `Index Scan` on `order_id`. The planner also flipped from hash joins over huge inputs to **nested loops** over the now-small driving set (52k orders, ~4 items each) — the right strategy once the inputs are small and indexed. Watch the estimated-vs-actual rows: if they'd diverged wildly, I'd `ANALYZE` the tables so the planner costs the joins correctly.

#### Alternative Solution
Pre-filter `orders` in a CTE (or reduce granularity first) to shrink the join inputs explicitly, which can help the planner and readability:
```sql
WITH recent_orders AS (
    SELECT id, customer_id
    FROM orders
    WHERE status = 'completed'
      AND order_date >= '2026-06-01' AND order_date < '2026-07-01'
)
SELECT c.city, COUNT(DISTINCT ro.id) AS orders,
       SUM(oi.quantity * oi.unit_price) AS revenue
FROM recent_orders ro
JOIN customers c ON c.id = ro.customer_id
JOIN order_items oi ON oi.order_id = ro.id
GROUP BY c.city
ORDER BY revenue DESC;
```
Tradeoff: with the indexes in place the planner already achieves this shape, so the CTE is mostly cosmetic in modern Postgres (CTEs are no longer an optimization fence by default). But it documents intent and, if this report runs constantly, the real endgame is a **materialized view** (tying back to Q6) refreshed nightly — turning a 92ms live query into a sub-millisecond lookup.

#### Interview Variations
- **Add a per-city index or pre-aggregate** if `GROUP BY city` becomes the bottleneck at higher cardinality.
- **Very large date range** (a full year): does the index still win, or does the driving set get big enough that a Seq Scan returns? Discuss the tipping point.
- **`COUNT(DISTINCT o.id)` is expensive**: since the grain guarantees distinct order ids per group, could you drop `DISTINCT`? Analyze the join fan-out first.

#### Common Follow-up Questions
- **How did you pick which index to add first?** Start with the node removing the most rows / consuming the most time — here the `orders` Seq Scan filtering out 4.9M rows.
- **Why did the join switch from Hash to Nested Loop?** Once the driving side is small and the inner side is indexed, repeated index lookups (nested loop) beat building big hash tables; the planner re-costs and switches automatically.
- **What does `INCLUDE` buy here?** It stores the join keys in the index so the scan is index-only for those columns — no heap fetch — without making them part of the B-tree search key.
- **When do you stop optimizing?** When the query meets its latency SLA. If live tuning can't get there, escalate to a materialized view or a denormalized rollup.

---

## Quick Reference — Cheatsheet

**Window functions**
- Ranking: `ROW_NUMBER` (unique), `RANK` (ties + gaps), `DENSE_RANK` (ties, no gaps).
- Offset: `LAG(col, n, default)` / `LEAD(...)` for period-over-period.
- Running total: `SUM(x) OVER (ORDER BY t ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)`.
- Can't filter a window in `WHERE` — wrap in a CTE/subquery and filter outside.
- `ROWS` (physical rows) vs `RANGE` (peer values) frames differ on ties — be explicit.

**Views / matviews**
- View = stored query, always fresh, no storage; matview = stored result, fast reads, goes stale.
- `WITH CHECK OPTION` stops writes from escaping a view's filter.
- Matview needs a **unique index** for `REFRESH ... CONCURRENTLY` (non-blocking).
- Postgres doesn't track matview refresh time — log it yourself.

**Indexing / EXPLAIN**
- Read `EXPLAIN (ANALYZE, BUFFERS)` inside-out; compare estimated vs actual rows.
- Composite index: **leftmost-prefix** rule; equality columns first, then the range/sort column.
- `Seq Scan` + high `Rows Removed by Filter` ⇒ missing/unused index (or low selectivity — then it's correct).
- Keep predicates **sargable**: never wrap the indexed column in a function; transform the constant instead, or build an expression index.
- `INCLUDE` columns enable index-only scans without bloating the search key.
- After bulk loads, `ANALYZE` so the planner has fresh statistics.


---


> PostgreSQL-flavored, hands-on coding-assessment prep. This part moves past single-query tricks into the machinery that separates a junior from a senior: **correctness under concurrency**, **server-side logic**, and **schema design**. Every question is written the way I'd coach a teammate before an on-site: understand the failure mode first, then write the code that survives it.

## Shared Schema

These questions reference the shared e-commerce/company schema defined in earlier parts:

- `departments(id, name, ...)`
- `employees(id, name, department_id, salary, ...)`
- `customers(id, name, email, ...)`
- `products(id, name, price, stock_quantity, ...)`
- `orders(id, customer_id, order_date, status, total, ...)`
- `order_items(id, order_id, product_id, quantity, unit_price, ...)`

For transaction examples we also use a simple bank-style table (defined inline where needed):

```sql
CREATE TABLE accounts (
    id      BIGINT PRIMARY KEY,
    owner   TEXT        NOT NULL,
    balance NUMERIC(14,2) NOT NULL DEFAULT 0 CHECK (balance >= 0)
);
```

The `CHECK (balance >= 0)` is not decoration — it is a last line of defense that turns a logic bug into a loud, transaction-aborting error instead of a silently overdrawn account.

---

## Transactions & Concurrency

The mental model I want you to carry into the interview: a transaction is a bubble of "all-or-nothing" work, and **isolation levels are a dial that trades throughput for how much of other people's in-flight work you're allowed to see**. Most bugs in this space are not syntax errors — they are two correct-looking transactions interleaving in an order the author never pictured. So for every problem below, sketch the interleaving as a two-column timeline before you trust the code.

### Practical Question 1: Safe Money Transfer With `SELECT ... FOR UPDATE`

- **Difficulty:** Medium
- **Estimated Time:** 15 min
- **Concepts Tested:** ACID atomicity, pessimistic row locking, `FOR UPDATE`, lost-update anomaly, deadlock avoidance

**Problem Statement**
Write a transaction that transfers a given amount from one account to another. It must be atomic (never debit without the matching credit), must never allow the source account to go negative, and must be safe when many transfers run concurrently — including two transfers touching the same pair of accounts at the same time.

**Example Input**

```sql
INSERT INTO accounts (id, owner, balance) VALUES
    (1, 'Alice', 500.00),
    (2, 'Bob',   100.00);
-- Transfer 200.00 from account 1 -> account 2
```

**Example Output**

```
 id | owner | balance
----+-------+---------
  1 | Alice | 300.00
  2 | Bob   | 300.00
```
Concurrent transfers must leave the sum of balances unchanged (conservation of money) and never produce a negative balance.

**Approach**
1. Open a transaction with `BEGIN`.
2. Lock **both** rows with `SELECT ... FOR UPDATE` so no concurrent transaction can read-then-write the same rows underneath us (prevents the lost-update anomaly).
3. Always lock rows in a **deterministic order** (e.g. ascending `id`) to avoid deadlocks between transfers going in opposite directions.
4. Verify the source has sufficient funds; if not, abort.
5. Debit source, credit destination.
6. `COMMIT`.

#### SQL Implementation

```sql
BEGIN;

-- Lock rows in a fixed order (ascending id) to avoid deadlocks.
-- FOR UPDATE takes a row-level exclusive lock: any other txn that
-- also runs FOR UPDATE on these rows will BLOCK here until we commit.
SELECT id, balance
FROM   accounts
WHERE  id IN (1, 2)
ORDER  BY id
FOR    UPDATE;

-- Application/procedure checks that account 1 has >= 200 before proceeding.
-- The CHECK constraint is a backstop, but checking here gives a clean error.

UPDATE accounts SET balance = balance - 200.00 WHERE id = 1;
UPDATE accounts SET balance = balance + 200.00 WHERE id = 2;

COMMIT;
```

**Why the lock matters — the lost-update anomaly.** Without `FOR UPDATE`, two concurrent transfers can both read `balance = 500`, both compute `500 - 200 = 300`, and both write `300`. One debit vanishes. Here is the interleaving that `FOR UPDATE` prevents:

| Txn A (transfer 200 from acct 1) | Txn B (transfer 100 from acct 1) |
|---|---|
| `BEGIN` | `BEGIN` |
| `SELECT balance FROM accounts WHERE id=1;` → 500 | |
| | `SELECT balance FROM accounts WHERE id=1;` → 500 |
| `UPDATE ... balance = 500 - 200 = 300` | |
| | `UPDATE ... balance = 500 - 100 = 400` (overwrites!) |
| `COMMIT` | `COMMIT` → final balance 400, should be 200 |

With `FOR UPDATE`, Txn B's `SELECT ... FOR UPDATE` blocks until Txn A commits, then re-reads the *committed* 300 and computes 200. Money is conserved.

**Why the fixed lock order matters — deadlocks.** If transfer A locks row 1 then row 2, while transfer B (going the other direction) locks row 2 then row 1, they can deadlock. Postgres will detect it and kill one with a `deadlock detected` error, but that's an avoidable failure. Locking by ascending `id` guarantees all transfers grab rows in the same order, so one simply waits instead of deadlocking.

#### Alternative Solution

**Optimistic concurrency** (no `FOR UPDATE`), using a version/guard in the `WHERE` clause and checking the affected row count:

```sql
BEGIN;
-- Debit only if funds still sufficient; the WHERE clause is the guard.
UPDATE accounts SET balance = balance - 200.00
WHERE  id = 1 AND balance >= 200.00;
-- Application checks: if row count = 0, ROLLBACK (insufficient funds or race lost).
UPDATE accounts SET balance = balance + 200.00 WHERE id = 2;
COMMIT;
```

Because `balance = balance - 200` is computed **inside** the database under a row lock held for the duration of the `UPDATE`, this specific form is actually safe from lost updates too. Tradeoff: it's less explicit about intent, and it can't easily "read then decide" across multiple rows before writing. Use `FOR UPDATE` when you need to read several rows, make a decision, then write; use the guarded `UPDATE` for simple single-row conditional writes with lower lock contention.

#### Interview Variations
- **Cross-currency transfer:** debit in USD, credit in EUR at a rate looked up mid-transaction — now you must also lock/snapshot the rate.
- **Transfer with an audit ledger:** insert a row into a `transfers` log in the same transaction, so the ledger and balances can never disagree.
- **Batch payroll:** debit one company account and credit 10,000 employees atomically — discuss lock ordering and holding one hot lock on the company row.

#### Common Follow-up Questions
- *What lock does `FOR UPDATE` take, and what does it block?* A row-level exclusive lock. It blocks other `FOR UPDATE`/`UPDATE`/`DELETE` on those rows, but plain `SELECT` (without `FOR ...`) still reads the last committed snapshot and is not blocked.
- *`FOR UPDATE` vs `FOR NO KEY UPDATE` vs `FOR SHARE`?* `FOR SHARE` is a shared lock (multiple readers, blocks writers). `FOR NO KEY UPDATE` is weaker than `FOR UPDATE` and doesn't block concurrent foreign-key checks that only need the key. Use the weakest lock that's still correct.
- *What if account 2 doesn't exist?* The credit `UPDATE` affects 0 rows silently — money disappears. Check row counts, or add an FK/existence check, and abort if the destination is missing.
- *Why not `SERIALIZABLE` instead of explicit locks?* You can — it would abort one transfer with a serialization failure that you retry. `FOR UPDATE` is pessimistic (block and proceed); `SERIALIZABLE` is optimistic (proceed and maybe retry). Both are correct; pick based on contention.

### Practical Question 2: Savepoint Rollback — Partial Undo Inside a Transaction

- **Difficulty:** Easy
- **Estimated Time:** 10 min
- **Concepts Tested:** `SAVEPOINT`, `ROLLBACK TO SAVEPOINT`, `RELEASE SAVEPOINT`, nested error handling within one transaction

**Problem Statement**
You are importing a batch of account adjustments inside a single transaction. If one adjustment is invalid, you want to undo **only that adjustment** and continue with the rest — without throwing away the whole batch. Demonstrate savepoints to achieve this partial rollback.

**Example Input**

```sql
INSERT INTO accounts (id, owner, balance) VALUES
    (1, 'Alice', 500.00),
    (2, 'Bob',   100.00);
-- Batch: credit Alice +50 (valid), then attempt to debit Bob -999 (would go negative → invalid)
```

**Example Output**

```
 id | owner | balance
----+-------+---------
  1 | Alice | 550.00   -- kept
  2 | Bob   | 100.00   -- bad adjustment rolled back, original preserved
```

**Approach**
1. `BEGIN` the transaction.
2. Apply the first valid adjustment.
3. Set a `SAVEPOINT` before the risky adjustment.
4. Attempt the risky adjustment. If it errors (or you detect it's invalid), `ROLLBACK TO SAVEPOINT` — this undoes work back to the savepoint but keeps everything before it.
5. `COMMIT` the surviving work.

#### SQL Implementation

```sql
BEGIN;

-- Valid adjustment: keep this no matter what happens next.
UPDATE accounts SET balance = balance + 50.00 WHERE id = 1;

SAVEPOINT before_bob;

-- Risky adjustment. The CHECK (balance >= 0) constraint will raise an
-- error here because 100 - 999 < 0.
UPDATE accounts SET balance = balance - 999.00 WHERE id = 2;
-- ERROR: new row for relation "accounts" violates check constraint

-- After the error, the transaction is in an aborted state ONLY back to
-- the savepoint. Undo just the failed statement:
ROLLBACK TO SAVEPOINT before_bob;

-- We can optionally release the savepoint (frees resources) and continue.
RELEASE SAVEPOINT before_bob;

-- Alice's +50 survives; commit it.
COMMIT;
```

**The key mental model:** a `SAVEPOINT` is a bookmark inside a transaction. When a statement raises an error, everything since the last savepoint is rolled back and the transaction becomes usable again after `ROLLBACK TO SAVEPOINT` — you are *not* forced to abort the entire transaction. Without the savepoint, the failed `UPDATE` would poison the whole transaction (`current transaction is aborted, commands ignored until end of transaction block`), and Alice's valid +50 would be lost on the mandatory `ROLLBACK`.

Note: `psql` won't literally continue past an error in a plain script unless you use `ON_ERROR_ROLLBACK` — in real code this pattern lives inside a driver or a PL/pgSQL `BEGIN ... EXCEPTION` block (which uses an implicit savepoint under the hood). That's exactly how application frameworks give you "try one row, skip if bad, keep the rest."

#### Alternative Solution

Inside PL/pgSQL, the same partial-undo is expressed with a `BEGIN ... EXCEPTION` sub-block, which Postgres implements as an implicit savepoint:

```sql
DO $$
BEGIN
    UPDATE accounts SET balance = balance + 50.00 WHERE id = 1;

    BEGIN
        UPDATE accounts SET balance = balance - 999.00 WHERE id = 2;
    EXCEPTION WHEN check_violation THEN
        RAISE NOTICE 'Skipping invalid debit for account 2';
        -- exception block auto-rolled back to the implicit savepoint
    END;
END $$;
```

Tradeoff: cleaner and self-contained, but every `EXCEPTION` block has a small cost (it establishes a savepoint), so avoid wrapping millions of rows each in its own block on a hot path.

#### Interview Variations
- **Nested savepoints:** set `sp1`, then `sp2`; show that `ROLLBACK TO sp1` also discards `sp2`.
- **Retry pattern:** roll back to a savepoint and re-attempt the same statement with adjusted values (e.g. clamp the debit).
- **Savepoint after partial success in a loop:** import 1,000 rows, savepoint per row, skip only the bad ones, report how many succeeded.

#### Common Follow-up Questions
- *Does `RELEASE SAVEPOINT` commit anything?* No. It just discards the bookmark (and merges its work into the enclosing transaction). Nothing is durable until `COMMIT`.
- *What happens to a savepoint after you roll back to it — can you reuse the name?* `ROLLBACK TO` keeps the savepoint alive so you can retry; it's destroyed by `RELEASE` or by rolling back to an earlier savepoint.
- *Are savepoints visible to other sessions?* No — they're internal to one transaction and invisible outside it.
- *Do savepoints survive a `COMMIT`?* No; commit ends the transaction and all its savepoints.

### Practical Question 3: Choosing an Isolation Level to Prevent an Anomaly

- **Difficulty:** Hard
- **Estimated Time:** 18 min
- **Concepts Tested:** SQL isolation levels (READ COMMITTED, REPEATABLE READ, SERIALIZABLE), dirty read / non-repeatable read / phantom read / write skew, MVCC in PostgreSQL

**Problem Statement**
An on-call system enforces the rule: **at least one engineer must remain on-call at all times.** Two engineers each try to remove themselves from the on-call rotation at the same moment. Each transaction reads "how many are on-call?" (sees 2), concludes "it's safe, one will remain," and removes itself. Result: **zero** on-call engineers — the rule is violated even though each transaction individually looked correct. This is the classic **write-skew** anomaly. Choose and justify an isolation level that prevents it, and show the code.

**Example Input**

```sql
CREATE TABLE oncall (
    engineer   TEXT PRIMARY KEY,
    is_oncall  BOOLEAN NOT NULL
);
INSERT INTO oncall VALUES ('Alice', true), ('Bob', true);
-- Both Alice and Bob concurrently try to go off-call.
```

**Example Output**
The rule holds: exactly one of the two transactions succeeds, the other is aborted with a serialization failure (to be retried), leaving **one** engineer on-call.

```
 engineer | is_oncall
----------+-----------
 Alice    | f
 Bob      | t          -- one remains; second txn was rolled back
```

**Approach**
1. Identify the anomaly. Both transactions read an overlapping set (the on-call engineers) and write to *different rows* based on that read. No row is updated by both, so row locks and `REPEATABLE READ` do **not** catch it — that's what makes write skew special.
2. `READ COMMITTED` (Postgres default): each statement sees the latest committed data, but the two reads happen before either write commits → anomaly occurs.
3. `REPEATABLE READ` in Postgres uses snapshot isolation: each transaction keeps its start-time snapshot, both still see 2, both write different rows → **write skew still occurs** (snapshot isolation does not prevent it).
4. `SERIALIZABLE`: Postgres uses Serializable Snapshot Isolation (SSI), which tracks read/write dependencies and **aborts** one transaction with a serialization failure when the interleaving couldn't have happened in any serial order. This prevents write skew.
5. Wrap the logic in `SERIALIZABLE` and add application retry-on-`40001`.

#### SQL Implementation

```sql
-- Each engineer runs this. Under SERIALIZABLE, one will be aborted.
BEGIN ISOLATION LEVEL SERIALIZABLE;

-- Read the invariant-relevant set.
SELECT count(*) FROM oncall WHERE is_oncall;   -- both see 2

-- Application logic: only go off-call if more than one remains.
-- (count > 1 is true for both under their snapshots)
UPDATE oncall SET is_oncall = false WHERE engineer = 'Alice';  -- Txn A
-- UPDATE oncall SET is_oncall = false WHERE engineer = 'Bob'; -- Txn B

COMMIT;
-- One COMMIT succeeds. The other fails:
-- ERROR: could not serialize access due to read/write dependencies
--        among transactions (SQLSTATE 40001)  → retry it.
```

The interleaving and why only `SERIALIZABLE` saves us:

| Txn A (Alice goes off-call) | Txn B (Bob goes off-call) |
|---|---|
| `BEGIN ISOLATION LEVEL SERIALIZABLE` | `BEGIN ISOLATION LEVEL SERIALIZABLE` |
| `SELECT count(*) WHERE is_oncall` → 2 | |
| | `SELECT count(*) WHERE is_oncall` → 2 |
| `UPDATE oncall SET is_oncall=false WHERE engineer='Alice'` | |
| | `UPDATE oncall SET is_oncall=false WHERE engineer='Bob'` |
| `COMMIT` ✅ (one on-call remains) | |
| | `COMMIT` ❌ `40001` — SSI detects the read/write dependency cycle, aborts, retry re-reads count=1 and declines |

On retry, Txn B begins a fresh snapshot, reads `count = 1`, its own guard (`count > 1`) is now false, and it correctly refuses to go off-call. Invariant preserved.

**Anomaly → minimum isolation level cheat sheet (PostgreSQL):**

| Anomaly | READ COMMITTED | REPEATABLE READ | SERIALIZABLE |
|---|---|---|---|
| Dirty read | prevented | prevented | prevented |
| Non-repeatable read | **possible** | prevented | prevented |
| Phantom read | **possible** | prevented* | prevented |
| Write skew | **possible** | **possible** | prevented |

\*Standard SQL allows phantoms at REPEATABLE READ; Postgres's snapshot-based implementation actually prevents them, but it still does **not** prevent write skew — that gap is the whole point of this question. Note Postgres has no true "READ UNCOMMITTED" (it maps to READ COMMITTED), so dirty reads never happen.

#### Alternative Solution

**Materialize the conflict into a real lock** so a weaker isolation level suffices. If both transactions must touch the *same* row, ordinary locking catches them:

```sql
BEGIN;  -- READ COMMITTED is enough now
-- Lock all on-call rows; the two txns now contend on the same rows.
SELECT count(*) FROM oncall WHERE is_oncall FOR UPDATE;
-- Second txn blocks until first commits, then re-reads count = 1 and declines.
UPDATE oncall SET is_oncall = false WHERE engineer = 'Alice';
COMMIT;
```

Tradeoff: `SELECT ... FOR UPDATE` (pessimistic) blocks instead of aborting, so there's no retry loop, but it serializes access to the whole on-call set and reduces concurrency. `SERIALIZABLE` (optimistic) allows more concurrency but requires you to handle `40001` retries. Rule of thumb: high contention → pessimistic locks; low contention with rare conflicts → `SERIALIZABLE`.

#### Interview Variations
- **Bank overdraft across two accounts:** "total across A+B must stay ≥ 0" is write skew too — same fix.
- **Double-booking a meeting room:** two bookings read "no overlap" then both insert → phantom/write skew; show `SERIALIZABLE` vs an exclusion constraint.
- **Set the level per-session** with `SET SESSION CHARACTERISTICS AS TRANSACTION ISOLATION LEVEL SERIALIZABLE` and discuss the blast radius.

#### Common Follow-up Questions
- *Why doesn't REPEATABLE READ catch write skew?* Because the two transactions write **different** rows, there's no direct write-write conflict; snapshot isolation only detects that when the *same* row is updated by both (first-updater-wins → `40001`). Write skew is a read/write dependency, which only SSI (`SERIALIZABLE`) tracks.
- *What's the cost of SERIALIZABLE?* Extra bookkeeping (predicate locks / SIReadLocks) and the need to retry aborted transactions. Under low conflict it's cheap; under high conflict the retry rate can hurt.
- *Is `SERIALIZABLE` in Postgres the same as locking everything?* No — it's optimistic. Transactions run concurrently against snapshots and only get aborted if a genuine non-serializable cycle is detected.
- *How would you make retries robust?* Wrap the transaction in application code that catches SQLSTATE `40001` (and `40P01` deadlock) and retries with backoff, capping attempts. Keep transactions short to lower conflict probability.

---

## Stored Procedures & Functions

The distinction that trips people up in interviews: in PostgreSQL a **`FUNCTION`** returns a value (scalar, row, or table) and runs *inside* the calling query's transaction — it **cannot** issue `COMMIT`/`ROLLBACK`. A **`PROCEDURE`** (added in PG 11) returns nothing, is invoked with `CALL`, and **can** manage transactions internally. Reach for a function when you want reusable query logic; reach for a procedure when you need to orchestrate multi-step, transaction-controlling work like "place an order." Get that framing right and half of these questions answer themselves.

### Practical Question 4: PL/pgSQL Function Returning a Table

- **Difficulty:** Medium
- **Estimated Time:** 15 min
- **Concepts Tested:** `RETURNS TABLE`, set-returning functions, PL/pgSQL vs SQL functions, parameterized reporting, `STABLE`/`LANGUAGE sql`

**Problem Statement**
Write a reusable function `top_customers_by_spend(since DATE, min_orders INT)` that returns each customer's id, name, order count, and total spend, considering only orders on/after `since`, and only customers with at least `min_orders` orders in that window, sorted by total spend descending. It should be callable like a table: `SELECT * FROM top_customers_by_spend('2026-01-01', 3);`.

**Example Input**

```sql
-- orders / customers as per shared schema; call:
SELECT * FROM top_customers_by_spend(DATE '2026-01-01', 2);
```

**Example Output**

```
 customer_id | customer_name | order_count | total_spend
-------------+---------------+-------------+-------------
          42 | Priya Nair    |           5 |     4820.00
          17 | Sam Okafor    |           3 |     2110.50
```

**Approach**
1. Declare the function with `RETURNS TABLE(...)` naming each output column and type — this defines the result shape.
2. Use `RETURN QUERY` to stream the result of a single aggregate query (efficient — no row-by-row loop).
3. Filter by `order_date >= since` in the `WHERE`, aggregate per customer, and enforce `min_orders` in the `HAVING` clause.
4. Mark it `STABLE` (reads data, no writes, same result within a statement) so the planner can optimize calls.

#### SQL Implementation

```sql
CREATE OR REPLACE FUNCTION top_customers_by_spend(
    since       DATE,
    min_orders  INT
)
RETURNS TABLE (
    customer_id    BIGINT,
    customer_name  TEXT,
    order_count    BIGINT,
    total_spend    NUMERIC
)
LANGUAGE plpgsql
STABLE
AS $$
BEGIN
    RETURN QUERY
    SELECT  c.id,
            c.name,
            count(o.id)        AS order_count,
            sum(o.total)       AS total_spend
    FROM    customers c
    JOIN    orders    o ON o.customer_id = c.id
    WHERE   o.order_date >= since          -- parameter shadows nothing; see note
    GROUP   BY c.id, c.name
    HAVING  count(o.id) >= min_orders
    ORDER   BY total_spend DESC;
END;
$$;
```

**Important parts explained.**
- `RETURNS TABLE(...)` is syntactic sugar for a set of `OUT` parameters; the named columns (`customer_id`, etc.) are in scope inside the body, so **qualify your table columns** (`c.id`, `o.id`) to avoid ambiguity between the output column `customer_id` and any table column. Here I aliased explicitly and selected `c.id` into the first output slot positionally.
- `RETURN QUERY` appends the whole result set at once — far better than looping with `RETURN NEXT`, which materializes row-by-row.
- `STABLE` tells the planner the function won't modify the database and returns consistent results within a single statement; this permits optimizations like caching the call in a scan. Use `IMMUTABLE` only for pure functions of their inputs (not this one — it reads tables), and `VOLATILE` (the default) if it writes or is nondeterministic.

#### Alternative Solution

For a pure query with no procedural logic, a **`LANGUAGE sql`** function is leaner and can be inlined by the planner (often faster than PL/pgSQL):

```sql
CREATE OR REPLACE FUNCTION top_customers_by_spend(since DATE, min_orders INT)
RETURNS TABLE (customer_id BIGINT, customer_name TEXT,
               order_count BIGINT, total_spend NUMERIC)
LANGUAGE sql STABLE AS $$
    SELECT c.id, c.name, count(o.id), sum(o.total)
    FROM   customers c
    JOIN   orders o ON o.customer_id = c.id
    WHERE  o.order_date >= since
    GROUP  BY c.id, c.name
    HAVING count(o.id) >= min_orders
    ORDER  BY sum(o.total) DESC;
$$;
```

Tradeoff: `LANGUAGE sql` can't do conditionals, loops, or exception handling — but when you don't need them, it's simpler and the optimizer can fold it into the outer query. Use PL/pgSQL only when you genuinely need procedural control flow.

#### Interview Variations
- **Dynamic sort:** add a `sort_by TEXT` parameter and build the query with `EXECUTE format(...)` (discuss SQL-injection safety via `%I`/`quote_ident`).
- **Return a composite/`SETOF customers`** instead of a bespoke table shape.
- **Cursor version:** return a `refcursor` so a client can fetch lazily from a huge result.

#### Common Follow-up Questions
- *`RETURNS TABLE` vs `RETURNS SETOF record`?* `RETURNS TABLE` names and types the columns up front; `SETOF record` forces the caller to supply a column definition list (`AS t(a int, b text)`). Prefer `RETURNS TABLE` for ergonomics.
- *Why qualify column names?* Output columns declared in `RETURNS TABLE` share the namespace with query columns; unqualified references can be ambiguous and error at creation time.
- *Is the result streamed or materialized?* `RETURN QUERY` streams from a single query; `RETURN NEXT` in a loop materializes. For big sets, prefer `RETURN QUERY`.
- *Can I call it in a JOIN?* Yes — set-returning functions can appear in `FROM`, including `LATERAL` joins to parameterize per outer row.

### Practical Question 5: Stored Procedure to Place an Order and Decrement Stock Atomically

- **Difficulty:** Hard
- **Estimated Time:** 22 min
- **Concepts Tested:** `CREATE PROCEDURE`, transaction control in procedures, row locking, exception handling, atomic multi-table writes, overselling prevention

**Problem Statement**
Write a stored procedure `place_order(p_customer_id BIGINT, p_product_id BIGINT, p_qty INT)` that, atomically: verifies stock, decrements `products.stock_quantity`, creates an `orders` row, and inserts the matching `order_items` line. It must never oversell under concurrency (two buyers racing for the last unit), and if anything fails, nothing should persist.

**Example Input**

```sql
-- products.stock_quantity for product 7 = 1 (last unit)
CALL place_order(42, 7, 1);   -- Buyer A
CALL place_order(99, 7, 1);   -- Buyer B, concurrent
```

**Example Output**
Exactly one `CALL` succeeds and creates an order; the other raises `insufficient stock` and writes nothing. Final `stock_quantity = 0`, one order + one order_item created.

**Approach**
1. `CREATE PROCEDURE` (not function) so we own the transaction semantics.
2. **Lock the product row first** with `SELECT ... FOR UPDATE` — this serializes concurrent buyers of the same product so the stock check and decrement are one indivisible step (prevents overselling).
3. If stock is insufficient, `RAISE EXCEPTION` — this rolls back the whole transaction.
4. Decrement stock, insert the order (capturing its id with `RETURNING ... INTO`), insert the order item.
5. `COMMIT` at the end of the procedure.

#### SQL Implementation

```sql
CREATE OR REPLACE PROCEDURE place_order(
    p_customer_id BIGINT,
    p_product_id  BIGINT,
    p_qty         INT
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_stock      INT;
    v_price      NUMERIC(12,2);
    v_order_id   BIGINT;
BEGIN
    IF p_qty <= 0 THEN
        RAISE EXCEPTION 'quantity must be positive, got %', p_qty;
    END IF;

    -- (1) Lock the product row. Any concurrent place_order() for the same
    --     product blocks here until we commit — this is what prevents
    --     overselling the last unit.
    SELECT stock_quantity, price
      INTO v_stock, v_price
      FROM products
     WHERE id = p_product_id
     FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'product % does not exist', p_product_id;
    END IF;

    -- (2) Check-then-act is now safe because we hold the row lock.
    IF v_stock < p_qty THEN
        RAISE EXCEPTION 'insufficient stock for product %: have %, need %',
            p_product_id, v_stock, p_qty;
    END IF;

    -- (3) Decrement stock.
    UPDATE products
       SET stock_quantity = stock_quantity - p_qty
     WHERE id = p_product_id;

    -- (4) Create the order header, grab its generated id.
    INSERT INTO orders (customer_id, order_date, status, total)
    VALUES (p_customer_id, CURRENT_DATE, 'PLACED', v_price * p_qty)
    RETURNING id INTO v_order_id;

    -- (5) Create the line item.
    INSERT INTO order_items (order_id, product_id, quantity, unit_price)
    VALUES (v_order_id, p_product_id, p_qty, v_price);

    -- (6) Make it durable. Because a procedure controls its own txn,
    --     everything above commits together or not at all.
    COMMIT;
END;
$$;
```

**Concurrency walk-through — racing for the last unit:**

| Buyer A: `CALL place_order(42,7,1)` | Buyer B: `CALL place_order(99,7,1)` |
|---|---|
| `SELECT ... FOR UPDATE` on product 7 → locks row, stock=1 | |
| | `SELECT ... FOR UPDATE` on product 7 → **BLOCKS** (A holds lock) |
| stock(1) ≥ qty(1) ✅ decrement → 0 | |
| insert order + item, `COMMIT` (lock released) | |
| | unblocks, re-reads stock = **0** |
| | 0 < 1 → `RAISE EXCEPTION 'insufficient stock'`, rolls back |

Without `FOR UPDATE`, both buyers could read stock=1, both pass the check, both decrement → stock = -1 and two orders for one unit. The row lock is the whole ballgame.

**Why a procedure, not a function?** Only a procedure can `COMMIT`/`ROLLBACK` inside its body. A function runs within the caller's transaction, so if you want the caller to just `CALL place_order(...)` and have it be a complete unit of work (including commit), a procedure is the right tool. Any unhandled `RAISE EXCEPTION` inside aborts the current transaction, undoing the decrement and inserts — atomicity for free.

#### Alternative Solution

**Guarded `UPDATE` with `RETURNING`, no explicit `FOR UPDATE`** — let the decrement itself be the concurrency guard:

```sql
CREATE OR REPLACE PROCEDURE place_order(p_customer_id BIGINT, p_product_id BIGINT, p_qty INT)
LANGUAGE plpgsql AS $$
DECLARE v_price NUMERIC(12,2); v_order_id BIGINT; v_ok BOOLEAN;
BEGIN
    -- Atomically decrement only if enough stock; the row lock is held for
    -- the duration of this UPDATE, so no separate SELECT ... FOR UPDATE needed.
    UPDATE products
       SET stock_quantity = stock_quantity - p_qty
     WHERE id = p_product_id AND stock_quantity >= p_qty
    RETURNING price INTO v_price;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'insufficient stock or missing product %', p_product_id;
    END IF;

    INSERT INTO orders (customer_id, order_date, status, total)
    VALUES (p_customer_id, CURRENT_DATE, 'PLACED', v_price * p_qty)
    RETURNING id INTO v_order_id;

    INSERT INTO order_items (order_id, product_id, quantity, unit_price)
    VALUES (v_order_id, p_product_id, p_qty, v_price);
    COMMIT;
END;
$$;
```

Tradeoff: fewer statements and slightly less lock-hold time, and the `WHERE stock_quantity >= p_qty` makes the check and decrement a single atomic operation. Downside: you can't distinguish "product missing" from "out of stock" without a second lookup, and multi-product orders (several line items) are cleaner with explicit locking in a defined order. Use the guarded `UPDATE` for single-item hot paths; use explicit `FOR UPDATE` when you must read/validate several rows before writing.

#### Interview Variations
- **Multi-line cart:** accept arrays of product ids/quantities; lock all product rows **in id order** to avoid deadlocks, then decrement each.
- **Reserve-then-confirm:** split into `reserve_stock` (hold) and `confirm_order` with a timeout that releases reservations.
- **Idempotency:** accept a client-supplied `idempotency_key` and make repeated `CALL`s with the same key a no-op (unique constraint + `ON CONFLICT DO NOTHING`).

#### Common Follow-up Questions
- *Why can't a function `COMMIT`?* A function executes inside the transaction of the statement that called it; committing mid-statement would break that statement's atomicity. Procedures are invoked standalone via `CALL`, so they may manage transactions.
- *What happens on `RAISE EXCEPTION` after a partial write?* The whole transaction is rolled back to before the `CALL` (or to an enclosing savepoint), so the stock decrement and any inserts are undone — nothing persists.
- *Could a `BEFORE UPDATE` trigger or `CHECK (stock_quantity >= 0)` help?* Yes — add `CHECK (stock_quantity >= 0)` as a backstop so even a logic bug can't drive stock negative; it turns the error into a constraint violation.
- *How do you prevent deadlocks with multi-item orders?* Sort the product ids and lock in a consistent order across all callers, so no two transactions acquire the same locks in opposite sequences.

### Practical Question 6: Trigger Function to Keep a Denormalized Total in Sync

- **Difficulty:** Medium
- **Estimated Time:** 16 min
- **Concepts Tested:** `CREATE TRIGGER`, trigger functions, `RETURNS TRIGGER`, `NEW`/`OLD`, `TG_OP`, maintaining derived columns, `AFTER` vs `BEFORE`

**Problem Statement**
`orders.total` is a stored (denormalized) sum of its `order_items` lines. Whenever an order item is inserted, updated, or deleted, the parent order's `total` must be recomputed automatically so the two never drift. Write a trigger and its trigger function.

**Example Input**

```sql
INSERT INTO order_items (order_id, product_id, quantity, unit_price)
VALUES (100, 7, 2, 25.00);   -- +50.00 to order 100's total
UPDATE order_items SET quantity = 3 WHERE id = <that row>;  -- now +75.00
DELETE FROM order_items WHERE id = <that row>;              -- back to prior total
```

**Example Output**
After each statement, `SELECT total FROM orders WHERE id = 100;` reflects the exact sum of that order's current line items — no manual recompute needed.

**Approach**
1. Write a `RETURNS TRIGGER` function that recomputes the total for the affected order.
2. Determine which order is affected: on `DELETE`, use `OLD.order_id`; otherwise `NEW.order_id`. Handle the case where an `UPDATE` moves a line to a different order (recompute both).
3. Recompute with a single aggregate `UPDATE ... SET total = (SELECT coalesce(sum(...),0) ...)`.
4. Attach it as an `AFTER INSERT OR UPDATE OR DELETE ... FOR EACH ROW` trigger (AFTER, so the changed row is already visible to the aggregate).

#### SQL Implementation

```sql
CREATE OR REPLACE FUNCTION sync_order_total()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_order_id BIGINT;
BEGIN
    -- Recompute for the new/updated order (or the old one on DELETE).
    v_order_id := COALESCE(NEW.order_id, OLD.order_id);

    UPDATE orders o
       SET total = COALESCE(
               (SELECT sum(oi.quantity * oi.unit_price)
                  FROM order_items oi
                 WHERE oi.order_id = v_order_id), 0)
     WHERE o.id = v_order_id;

    -- If an UPDATE moved the line to a DIFFERENT order, fix the old order too.
    IF TG_OP = 'UPDATE' AND NEW.order_id IS DISTINCT FROM OLD.order_id THEN
        UPDATE orders o
           SET total = COALESCE(
                   (SELECT sum(oi.quantity * oi.unit_price)
                      FROM order_items oi
                     WHERE oi.order_id = OLD.order_id), 0)
         WHERE o.id = OLD.order_id;
    END IF;

    -- AFTER triggers ignore the return value, but returning NULL is conventional.
    RETURN NULL;
END;
$$;

CREATE TRIGGER trg_sync_order_total
    AFTER INSERT OR UPDATE OR DELETE ON order_items
    FOR EACH ROW
    EXECUTE FUNCTION sync_order_total();
```

**Important parts explained.**
- **`AFTER`, not `BEFORE`:** we run the aggregate *after* the row change is applied so `sum(...)` already includes an inserted/updated row and excludes a deleted one. A `BEFORE` trigger would sum stale data.
- **`TG_OP`** tells us the operation (`INSERT`/`UPDATE`/`DELETE`), letting one function handle all three. `NEW` is null on `DELETE`; `OLD` is null on `INSERT` — hence `COALESCE(NEW.order_id, OLD.order_id)`.
- **`IS DISTINCT FROM`** is null-safe equality; it correctly detects a moved line even if one side were null.
- Returning `NULL` from an `AFTER ... FOR EACH ROW` trigger is fine — the return value is only meaningful for `BEFORE` row triggers (where returning `NULL` would skip the operation).

#### Alternative Solution

**Statement-level trigger with transition tables** (PG 10+) recomputes once per statement instead of once per row — far better for bulk `INSERT`/`DELETE` of many lines:

```sql
CREATE OR REPLACE FUNCTION sync_order_total_stmt()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    -- Collect every order touched by this statement from both transition tables.
    WITH affected AS (
        SELECT order_id FROM new_rows
        UNION
        SELECT order_id FROM old_rows
    )
    UPDATE orders o
       SET total = COALESCE((SELECT sum(oi.quantity*oi.unit_price)
                             FROM order_items oi WHERE oi.order_id = o.id), 0)
      FROM affected a
     WHERE o.id = a.order_id;
    RETURN NULL;
END;
$$;

CREATE TRIGGER trg_sync_order_total_stmt
    AFTER INSERT OR UPDATE OR DELETE ON order_items
    REFERENCING NEW TABLE AS new_rows OLD TABLE AS old_rows
    FOR EACH STATEMENT
    EXECUTE FUNCTION sync_order_total_stmt();
```

Tradeoff: statement-level fires once even for a 10,000-row insert (one recompute per affected order vs 10,000 row-trigger invocations) — much faster in bulk. Downside: slightly more machinery and PG 10+ only. For OLTP single-row edits either is fine; for batch loads, prefer the statement-level version. (An even simpler design: drop the stored column entirely and make `total` a view/generated expression — denormalize only when reads demand it.)

#### Interview Variations
- **Guard against recursion:** if the trigger also updated `order_items`, you'd need a recursion guard or `pg_trigger_depth()`.
- **Only recompute when relevant columns change:** add `WHEN (OLD.quantity IS DISTINCT FROM NEW.quantity OR OLD.unit_price IS DISTINCT FROM NEW.unit_price)` to the trigger to skip no-op updates.
- **Audit trail:** extend the function to also insert a row into `order_total_history`.

#### Common Follow-up Questions
- *`BEFORE` vs `AFTER` trigger — when each?* `BEFORE` to validate/modify the incoming row (`NEW`) or to cancel the operation by returning `NULL`; `AFTER` for actions that need the final committed-in-statement state, like maintaining derived data or cascading writes.
- *Row-level vs statement-level?* Row-level fires per affected row (access to `NEW`/`OLD`); statement-level fires once (access to transition tables). Bulk ops favor statement-level.
- *What are the risks of denormalizing `total` with a trigger?* Drift if any path bypasses the trigger (e.g. `COPY` with triggers disabled, or direct `total` updates). Consider a periodic reconciliation job or a generated column.
- *Do triggers fire inside the same transaction?* Yes — trigger effects are part of the triggering statement's transaction and roll back with it.

---

## Normalization (Practical)

Normalization is not academic ceremony — it's how you stop the same fact from living in ten places where nine of them can rot. The three rules I recite in interviews: **1NF** — atomic values, no repeating groups; **2NF** — no non-key column depends on only *part* of a composite key; **3NF** — no non-key column depends on another *non-key* column (no transitive dependency). Or the classic mnemonic: every non-key attribute must depend on *the key, the whole key, and nothing but the key.*

### Practical Question 7: Normalize an Un-normalized Orders Spreadsheet to 3NF

- **Difficulty:** Medium
- **Estimated Time:** 22 min
- **Concepts Tested:** 1NF/2NF/3NF, functional dependencies, update/insert/delete anomalies, decomposition, redesigned DDL with keys and FKs

**Problem Statement**
A team has been running the business off one flat spreadsheet loaded into a single table. Identify its anomalies, walk it through 1NF → 2NF → 3NF, and produce the redesigned DDL.

**Example Input** — the "spreadsheet" table (everything crammed together):

```
orders_flat
------------------------------------------------------------------------------------------
order_id | order_date | customer_name | customer_email      | customer_city | products
---------+------------+---------------+---------------------+---------------+-------------------
 1001    | 2026-03-01 | Priya Nair    | priya@example.com   | Kochi         | "Mouse x2 @10.00,
         |            |               |                     |               |  Keyboard x1 @40.00"
 1002    | 2026-03-02 | Sam Okafor    | sam@example.com     | Lagos         | "Mouse x1 @10.00"
------------------------------------------------------------------------------------------
```

Problems visible by eye:
- **Not 1NF:** `products` is a repeating group stuffed into one cell (multiple items, quantities, prices in a single string).
- **Redundancy / update anomaly:** if Priya changes her email, you must edit every order row she has, and they can disagree.
- **Insert anomaly:** you can't record a new customer until they place an order (no place to put them).
- **Delete anomaly:** deleting Priya's only order erases the fact that Priya (and her city) ever existed.

**Example Output** — target 3NF schema (four tables):

```
customers(customer_id PK, name, email, city)
products (product_id PK, name, unit_price)
orders   (order_id PK, order_date, customer_id FK→customers)
order_items(order_id FK, product_id FK, quantity, PK(order_id, product_id))
```

**Approach**
1. **1NF:** eliminate the repeating group. Break the `products` cell into one row per item → an `order_items` table keyed by `(order_id, product_id)`. Values are now atomic.
2. **2NF:** the `order_items` composite key is `(order_id, product_id)`. Attributes like product name/unit price depend on `product_id` **alone** (partial dependency), and order_date/customer depend on `order_id` alone. Move each to the table of the key it fully depends on → split out `products` and `orders`.
3. **3NF:** in `orders`, customer_name/email/city depend on the customer, not directly on `order_id` — that's a transitive dependency (`order_id → customer_id → email`). Extract a `customers` table so orders only stores `customer_id`.
4. Add primary keys, foreign keys, and constraints.

#### SQL Implementation

```sql
-- 3NF redesign. Each fact lives in exactly one place.

CREATE TABLE customers (
    customer_id  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name         TEXT NOT NULL,
    email        TEXT NOT NULL UNIQUE,          -- a customer's email: stored once
    city         TEXT
);

CREATE TABLE products (
    product_id   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name         TEXT NOT NULL,
    unit_price   NUMERIC(12,2) NOT NULL CHECK (unit_price >= 0)
);

CREATE TABLE orders (
    order_id     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    order_date   DATE NOT NULL,
    customer_id  BIGINT NOT NULL REFERENCES customers(customer_id)
);

CREATE TABLE order_items (
    order_id     BIGINT NOT NULL REFERENCES orders(order_id) ON DELETE CASCADE,
    product_id   BIGINT NOT NULL REFERENCES products(product_id),
    quantity     INT    NOT NULL CHECK (quantity > 0),
    -- unit_price captured at sale time; see note on why we still store it here
    unit_price   NUMERIC(12,2) NOT NULL CHECK (unit_price >= 0),
    PRIMARY KEY (order_id, product_id)
);
```

**How each anomaly is fixed.**
- **Update anomaly gone:** Priya's email exists in exactly one `customers` row — change it once, done.
- **Insert anomaly gone:** you can add a customer or a product without any order (their rows are independent).
- **Delete anomaly gone:** deleting an order (and cascading its items) doesn't erase the customer or the product catalog.
- **1NF restored:** each `order_items` row holds one atomic product/quantity/price.

**A senior nuance — the deliberate denormalization of `unit_price`.** I keep `unit_price` in `order_items` *even though* `products.unit_price` exists. This is **not** a 3NF violation: an order line's price is a historical fact captured at purchase time, functionally dependent on the line's key, not on the current catalog price. If the catalog price later changes, past invoices must not change with it. Normalization is about eliminating *redundant storage of the same fact*, not about forbidding a snapshot of a value that legitimately differs over time.

#### Alternative Solution

If the business genuinely never re-prices products, you could drop `order_items.unit_price` and always join to `products.unit_price` — a stricter normalization that removes the "duplicate" column:

```sql
CREATE TABLE order_items (
    order_id   BIGINT NOT NULL REFERENCES orders(order_id) ON DELETE CASCADE,
    product_id BIGINT NOT NULL REFERENCES products(product_id),
    quantity   INT NOT NULL CHECK (quantity > 0),
    PRIMARY KEY (order_id, product_id)
);
-- line total := quantity * products.unit_price (computed at read time)
```

Tradeoff: maximally normalized and no price duplication, but you **lose price history** — old orders silently reflect today's price. Almost every real commerce system chooses to store the sale-time price. This is the practical lesson: 3NF is the default, and you denormalize *consciously* only where a business fact demands it.

#### Interview Variations
- **Add discounts/tax:** where do line discounts live — on `order_items` or a separate `promotions` table? Watch for new transitive dependencies.
- **Multiple addresses per customer:** shipping vs billing → an `addresses` table (1:N), resolving another repeating group.
- **Show the anomaly concretely:** ask them to write the `UPDATE` that would corrupt data in the flat table, then show it's impossible in the 3NF design.

#### Common Follow-up Questions
- *What's the difference between 2NF and 3NF violations?* 2NF: a non-key attribute depends on *part* of a composite key (partial dependency). 3NF: a non-key attribute depends on *another non-key* attribute (transitive dependency). Both are fixed by pulling the offending attributes into their own table.
- *When would you intentionally denormalize?* For read performance (precomputed totals, reporting tables, caches) where the write-time cost of keeping copies in sync is worth the read speedup — always a conscious trade, ideally with a trigger or job to prevent drift.
- *Is a surrogate key (`GENERATED ... IDENTITY`) required for 3NF?* No — 3NF is about dependencies, not key style. Surrogate keys are a pragmatic choice to keep FKs small and stable when natural keys are wide or mutable.
- *How do you verify a schema is in 3NF?* List the functional dependencies; ensure every determinant is a candidate key (or the dependent attribute is prime). If a non-key determines a non-key, you have a 3NF violation.

### Practical Question 8: Diagnose the Normal Form and Fix a Partial Dependency (2NF)

- **Difficulty:** Easy
- **Estimated Time:** 12 min
- **Concepts Tested:** identifying functional dependencies, composite keys, partial dependency, 2NF decomposition

**Problem Statement**
Given the table below, state the highest normal form it satisfies, identify the specific violation, and decompose it to fix it.

**Example Input**

```
enrollments(student_id, course_id, student_name, course_title, grade)
PRIMARY KEY (student_id, course_id)

Functional dependencies:
  student_id           -> student_name
  course_id            -> course_title
  (student_id, course_id) -> grade
```

**Example Output**
Diagnosis: it's in **1NF** but **not 2NF** — `student_name` depends on only `student_id` (part of the key) and `course_title` on only `course_id` (part of the key): both are **partial dependencies**. Fix by splitting into three tables.

**Approach**
1. Check 1NF: all values atomic, single-valued → yes, it's 1NF.
2. Check 2NF: with a composite key `(student_id, course_id)`, look for non-key attributes that depend on only part of it. `student_name` (← student_id) and `course_title` (← course_id) both do → 2NF violated.
3. Decompose so each non-key attribute sits with the whole key it depends on: `students`, `courses`, and a pure junction `enrollments` holding only `grade`.

#### SQL Implementation

```sql
CREATE TABLE students (
    student_id   BIGINT PRIMARY KEY,
    student_name TEXT NOT NULL
);

CREATE TABLE courses (
    course_id    BIGINT PRIMARY KEY,
    course_title TEXT NOT NULL
);

CREATE TABLE enrollments (
    student_id BIGINT NOT NULL REFERENCES students(student_id),
    course_id  BIGINT NOT NULL REFERENCES courses(course_id),
    grade      TEXT,                       -- depends on the WHOLE key
    PRIMARY KEY (student_id, course_id)
);
```

Now every non-key attribute depends on the **whole** key of its table: `grade` genuinely needs both `student_id` and `course_id`, while `student_name`/`course_title` live where their single-column keys are. No partial dependencies remain, and the design is also in 3NF (no non-key determines another non-key).

#### Alternative Solution

If `student_id → student_name` were the *only* concern and courses were trivial, you might argue for keeping a single `courses`-less structure — but that reintroduces redundancy (a student's name repeated on every enrollment). There's no good non-decomposition fix; the point of the question is to recognize that **partial dependencies force decomposition**. The only real variation is key style (surrogate `enrollment_id` vs the natural composite key), which doesn't change the normalization.

#### Interview Variations
- **Add `instructor` and `department`:** introduce a transitive dependency (`course_id → instructor → department`) to push them into 3NF territory.
- **Grade history:** a student retakes a course → the composite PK no longer holds; add `term` to the key.
- **Give only sample data, not the FDs:** ask them to infer the functional dependencies from the rows.

#### Common Follow-up Questions
- *Why is 2NF only interesting with composite keys?* Partial dependency means "depends on part of the key." With a single-column key there's no proper subset to depend on, so a 1NF table with a single-attribute key is automatically in 2NF.
- *How did you find the FDs?* From business meaning: a student's name is a property of the student, not of the student-course pair. FDs come from domain knowledge, then you verify against data.
- *Does decomposition ever lose information?* A proper (lossless-join) decomposition doesn't — you can reconstruct the original by joining. Splitting on a shared key (here the single-column keys) guarantees losslessness.

### Practical Question 9: BCNF — When 3NF Isn't Enough

- **Difficulty:** Hard
- **Estimated Time:** 18 min
- **Concepts Tested:** Boyce-Codd Normal Form, candidate keys, determinants, overlapping candidate keys, lossless decomposition

**Problem Statement**
Explain, with a concrete schema, a case that is in 3NF but violates BCNF, why it still allows an anomaly, and how to decompose it. Then note the tradeoff BCNF can introduce.

**Example Input**

```
Consider room bookings where each time slot in a room holds one class,
and each class meets in exactly one room:

bookings(room, time_slot, class)

FDs:
  (room, time_slot) -> class     -- a room at a time has one class
  class             -> room      -- each class is tied to one fixed room

Candidate keys: (room, time_slot) AND (class, time_slot)
```

**Example Output**
This is in 3NF but **not BCNF**, because the determinant `class` (in `class → room`) is **not a candidate key**, yet the dependent attribute `room` *is* part of a candidate key (a "prime" attribute), which is exactly the loophole 3NF permits. Decompose into `class_room(class, room)` and `bookings(class, time_slot)`.

**Approach**
1. Find all candidate keys: `(room, time_slot)` and `(class, time_slot)` both uniquely identify a row.
2. Check BCNF: **every determinant must be a candidate key.** `class → room` has determinant `class`, which is not a candidate key → BCNF violated.
3. Check why 3NF passed anyway: 3NF forgives a violation if the dependent attribute is *prime* (part of some candidate key). `room` is prime, so 3NF is satisfied — but the redundancy is real.
4. Decompose losslessly on the offending determinant `class`.

#### SQL Implementation

```sql
-- BCNF decomposition: pull out the class -> room fact.
CREATE TABLE class_room (
    class TEXT PRIMARY KEY,          -- class determines its room
    room  TEXT NOT NULL
);

CREATE TABLE class_schedule (
    class     TEXT NOT NULL REFERENCES class_room(class),
    time_slot TEXT NOT NULL,
    PRIMARY KEY (class, time_slot)
);

-- The rule "one class per room per time slot" is now an emergent constraint.
-- To still enforce it, add a uniqueness guard across the join:
CREATE UNIQUE INDEX one_class_per_room_slot
    ON class_schedule (time_slot, class);   -- see tradeoff note below
```

**The anomaly BCNF removes.** In the original single table, the fact "class CS101 is in room A" is repeated on every time slot CS101 meets. Change CS101's room and you must update many rows (update anomaly); the `class_room` split stores it once.

**The tradeoff — dependency preservation.** BCNF decomposition is always lossless, but it is **not always dependency-preserving**. Here the constraint `(room, time_slot) → class` now spans two tables and can't be enforced by a single-table key anymore — you'd need a cross-table check (a trigger or a carefully chosen unique index) to stop two different classes being scheduled in the same room at the same time. This is the canonical reason engineers sometimes **stop at 3NF**: 3NF can always be achieved losslessly *and* dependency-preservingly, whereas BCNF occasionally forces you to give up dependency preservation. Know the tradeoff and choose deliberately.

#### Alternative Solution

**Stay in 3NF** and accept the small redundancy, enforcing the `class → room` consistency with a trigger or a `CHECK` via a lookup, keeping `(room, time_slot) → class` enforceable as a table-level unique key:

```sql
CREATE TABLE bookings (
    room      TEXT NOT NULL,
    time_slot TEXT NOT NULL,
    class     TEXT NOT NULL,
    PRIMARY KEY (room, time_slot),
    UNIQUE (class, time_slot)      -- enforces the second candidate key
);
-- Redundant class->room is guarded by a trigger checking consistency.
```

Tradeoff: keeps both dependencies enforceable in one table (dependency-preserving) at the cost of the redundancy BCNF would remove. For a small, low-churn dataset like room assignments, this is often the pragmatic choice; for high-churn data where the update anomaly bites, decompose to BCNF and accept the cross-table constraint.

#### Interview Variations
- **Three overlapping candidate keys:** extend to `(room, time_slot, building)` to make the candidate-key analysis harder.
- **Prove losslessness:** show the decomposition satisfies the lossless-join condition (the shared attribute `class` is a key of `class_room`).
- **4NF/multivalued dependencies:** push further into "a class has many required textbooks and many meeting times" to motivate 4NF.

#### Common Follow-up Questions
- *What exactly is the difference between 3NF and BCNF?* Both forbid non-key determinants — but 3NF grants an exception when the dependent attribute is prime (part of a candidate key). BCNF grants no exception: **every** determinant must be a candidate key.
- *Is every 3NF relation in BCNF?* No. The counterexample requires overlapping candidate keys, as above. If the relation has a single candidate key, 3NF and BCNF coincide.
- *Why might you stop at 3NF?* Because 3NF is always achievable with a lossless *and* dependency-preserving decomposition; BCNF may force you to sacrifice dependency preservation, pushing a constraint into application/trigger logic.
- *Is BCNF decomposition always lossless?* Yes — decomposing on a determinant guarantees the lossless-join property. It's dependency preservation (not losslessness) that BCNF can cost you.

---

## Database Design & ER Modeling (Practical)

Design questions are where interviewers watch you *think*, not recall syntax. My process every time: (1) list the **entities** (nouns), (2) find the **relationships** and their **cardinality** (1:1, 1:N, M:N), (3) resolve every M:N into a **junction table**, (4) pick keys and enforce the business rules with constraints. Draw the ER diagram first; the DDL then writes itself.

### Practical Question 10: Design a Library Management Schema (ER Diagram + DDL)

- **Difficulty:** Hard
- **Estimated Time:** 25 min
- **Concepts Tested:** entity/relationship modeling, cardinality, ASCII ER diagrams, PK/FK design, M:N resolution, the book-vs-copy distinction, constraint design

**Problem Statement**
Design a normalized schema for a library. Requirements: the library owns **physical copies** of **books** (a title can have many copies); books have one or more **authors** (and an author writes many books); **members** borrow copies via **loans** with a due date and return date; a copy can be on loan to at most one member at a time, and the same copy can be loaned many times over its life. Produce an ASCII ER diagram and the DDL.

**Example Input** — requirements as above. Key subtlety to model correctly:
- "The Pragmatic Programmer" is one **book** (title, ISBN), but the library holds **3 copies** — loans are against a *copy*, not the title.
- Books ↔ Authors is **many-to-many**.

**Example Output** — the entities and relationships:

```
authors ──< book_authors >── books ──< copies ──< loans >── members
   (M:N via book_authors)        (1:N)      (1:N via loans)  (M:N over time)
```

**Approach**
1. **Entities:** `authors`, `books`, `copies`, `members`, `loans`.
2. **Relationships & cardinality:**
   - `books` ↔ `authors`: **M:N** → junction `book_authors`.
   - `books` → `copies`: **1:N** (a title has many physical copies).
   - `copies` ↔ `members`: **M:N over time** through `loans` (a copy is borrowed by many members across its life; a member borrows many copies) → the `loans` table *is* the resolving entity, enriched with due/return dates.
3. **Constraint:** a copy can be on loan to only one member *at a time* → enforce "at most one open loan per copy."
4. Choose surrogate keys, add FKs and business-rule constraints.

**ASCII ER diagram:**

```
 +-----------+        +---------------+        +-----------+
 |  authors  |        | book_authors  |        |   books   |
 +-----------+        +---------------+        +-----------+
 | author_id |PK   ┌─<| author_id  FK |>─┐  ┌─<| book_id   |PK
 | name      |     │  | book_id    FK |  │  │  | isbn      |UQ
 +-----------+     │  +---------------+  │  │  | title     |
        │          │   (junction: M:N)   │  │  | published |
        └──────────┘                     └──┘  +-----------+
                                                     │ 1
                                                     │
                                                     │ N
                                               +-----------+
                                               |  copies   |
                                               +-----------+
                                               | copy_id   |PK
                                               | book_id FK|>── books
                                               | barcode   |UQ
                                               | status    |
                                               +-----------+
                                                     │ 1
                                                     │
                                                     │ N
 +-----------+                                 +-----------+
 |  members  |                                 |   loans   |
 +-----------+                                 +-----------+
 | member_id |PK ──< (N)                    (N)| loan_id   |PK
 | name      |         └──────────────────────| copy_id FK|>── copies
 | email  UQ |                                 | member_id |FK>── members
 | joined_on |                                 | loaned_on |
 +-----------+                                 | due_on    |
                                               | returned_on (nullable)
                                               +-----------+
```

#### SQL Implementation

```sql
CREATE TABLE authors (
    author_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name      TEXT NOT NULL
);

CREATE TABLE books (
    book_id   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    isbn      TEXT UNIQUE,                 -- a title-level identifier
    title     TEXT NOT NULL,
    published DATE
);

-- M:N resolution between books and authors.
CREATE TABLE book_authors (
    book_id   BIGINT NOT NULL REFERENCES books(book_id)   ON DELETE CASCADE,
    author_id BIGINT NOT NULL REFERENCES authors(author_id) ON DELETE CASCADE,
    author_order SMALLINT,                 -- 1 = primary author, etc.
    PRIMARY KEY (book_id, author_id)       -- composite key = no duplicate pairing
);

-- The crucial book-vs-copy split: loans reference a physical copy.
CREATE TABLE copies (
    copy_id  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    book_id  BIGINT NOT NULL REFERENCES books(book_id),
    barcode  TEXT UNIQUE NOT NULL,
    status   TEXT NOT NULL DEFAULT 'available'
             CHECK (status IN ('available','on_loan','lost','retired'))
);

CREATE TABLE members (
    member_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name      TEXT NOT NULL,
    email     TEXT UNIQUE NOT NULL,
    joined_on DATE NOT NULL DEFAULT CURRENT_DATE
);

CREATE TABLE loans (
    loan_id     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    copy_id     BIGINT NOT NULL REFERENCES copies(copy_id),
    member_id   BIGINT NOT NULL REFERENCES members(member_id),
    loaned_on   DATE NOT NULL DEFAULT CURRENT_DATE,
    due_on      DATE NOT NULL,
    returned_on DATE,                       -- NULL = still out
    CHECK (due_on >= loaned_on),
    CHECK (returned_on IS NULL OR returned_on >= loaned_on)
);

-- Enforce "at most one OPEN loan per copy" (a copy can't be lent twice at once).
-- A partial unique index: uniqueness applies only to rows not yet returned.
CREATE UNIQUE INDEX one_open_loan_per_copy
    ON loans (copy_id)
    WHERE returned_on IS NULL;
```

**Design decisions worth defending aloud.**
- **Books vs copies is the make-or-break modeling call.** Loans point at `copies`, not `books`, so the library can lend one of three identical titles and track *which* physical item is out, lost, or retired. Beginners collapse these into one table and then can't answer "how many copies are available right now?"
- **The partial unique index** (`WHERE returned_on IS NULL`) is the elegant way to say "a copy may have many historical loans but only one open loan." It's a real, DB-enforced business rule — not application hope.
- **`book_authors` composite PK** `(book_id, author_id)` makes a duplicate author-on-book pairing physically impossible.
- Loans are kept forever (history), with `returned_on` distinguishing open from closed — no data is deleted to "return" a book.

#### Alternative Solution

Model **reservations/holds** and **fines** as first-class entities if the domain needs them, and consider whether `copies.status` should be *derived* rather than stored:

```sql
-- 'status' can drift from the loans table. Alternative: drop copies.status and
-- derive availability from the loans partial index.
-- A copy is "on loan" iff EXISTS an open loan; "available" otherwise.
CREATE VIEW copy_availability AS
SELECT c.copy_id,
       CASE WHEN l.loan_id IS NOT NULL THEN 'on_loan' ELSE 'available' END AS status
FROM   copies c
LEFT   JOIN loans l ON l.copy_id = c.copy_id AND l.returned_on IS NULL;
```

Tradeoff: deriving status via a view guarantees it can never disagree with the loans table (single source of truth), at the cost of a join on every read. Storing `copies.status` is faster to query but must be kept in sync (a trigger on `loans`). Choose derived-by-default; denormalize the status column only if availability queries prove hot.

#### Interview Variations
- **Overdue fines:** add a `fines` table (1:N from loans) or compute fines on the fly from `due_on` vs `returned_on`.
- **Reservations/holds queue:** members reserve a title (not a copy) and get the next returned copy → a `holds` table with position ordering.
- **Multiple branches:** add a `branches` entity; copies belong to a branch, and inter-branch transfers become their own relationship.

#### Common Follow-up Questions
- *Why separate `books` and `copies`?* Because a loan is against a physical item, and availability, condition, and loss are per-copy facts. One title, many copies is a 1:N relationship that a single table can't represent without redundancy.
- *How do you stop a copy being loaned twice simultaneously?* The partial unique index on `(copy_id) WHERE returned_on IS NULL`. It's enforced by the database regardless of application bugs.
- *Where do author names live to avoid duplication?* Only in `authors`; `book_authors` stores just the FK pair, so renaming an author touches one row.
- *How would you query "available copies of a given title"?* Count copies of the book minus copies with an open loan (or filter the `copy_availability` view). The book-vs-copy split makes this a clean aggregate.

### Practical Question 11: Resolve an M:N Relationship With a Junction Table

- **Difficulty:** Easy
- **Estimated Time:** 12 min
- **Concepts Tested:** many-to-many resolution, junction/associative tables, composite keys, relationship attributes, referential integrity

**Problem Statement**
Students enroll in courses: a student takes many courses and a course has many students. A relational table can't store an M:N relationship directly. Model it correctly, and show how to attach an attribute (the enrollment date and grade) *to the relationship itself*.

**Example Input**

```
students(student_id, name)   -- many
courses (course_id, title)   -- many
"a student enrolls in many courses; a course has many students"
```

**Example Output** — a junction table linking the two, carrying relationship attributes:

```
students ──< enrollments >── courses
  1        N            N     1
enrollments(student_id FK, course_id FK, enrolled_on, grade, PK(student_id,course_id))
```

**Approach**
1. Recognize the M:N: neither side can hold a single FK to the other without duplicating rows.
2. Create a **junction table** whose primary key is the **composite** of both foreign keys — this both links the entities and prevents duplicate pairings.
3. Put any attribute that describes *the pairing* (not either entity alone) on the junction table: `enrolled_on`, `grade`.
4. Add FKs with sensible `ON DELETE` behavior.

#### SQL Implementation

```sql
CREATE TABLE students (
    student_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name       TEXT NOT NULL
);

CREATE TABLE courses (
    course_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    title     TEXT NOT NULL
);

-- Junction table: one row per (student, course) pairing.
CREATE TABLE enrollments (
    student_id  BIGINT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    course_id   BIGINT NOT NULL REFERENCES courses(course_id)   ON DELETE CASCADE,
    enrolled_on DATE NOT NULL DEFAULT CURRENT_DATE,  -- attribute OF the relationship
    grade       TEXT,
    PRIMARY KEY (student_id, course_id)              -- no duplicate enrollment
);

-- Helpful for "all students in a course" queries (the PK already covers
-- "all courses for a student" via its leading column).
CREATE INDEX idx_enrollments_course ON enrollments (course_id);
```

**Why this is the canonical M:N pattern.**
- The **composite primary key** `(student_id, course_id)` does two jobs: it uniquely identifies the pairing and makes enrolling the same student in the same course twice impossible.
- **Relationship attributes** like `grade` and `enrolled_on` belong here, not on `students` or `courses`, because a grade is meaningless without *both* a student and a course.
- The **extra index on `course_id`** matters: the composite PK indexes `(student_id, course_id)` so lookups by `student_id` (leading column) are fast, but "who's in course X?" needs a separate index on `course_id`.

#### Alternative Solution

Add a **surrogate key** to the junction table when it's itself referenced by other tables (e.g. an `assignments` table points at a specific enrollment):

```sql
CREATE TABLE enrollments (
    enrollment_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    student_id    BIGINT NOT NULL REFERENCES students(student_id),
    course_id     BIGINT NOT NULL REFERENCES courses(course_id),
    enrolled_on   DATE NOT NULL DEFAULT CURRENT_DATE,
    grade         TEXT,
    UNIQUE (student_id, course_id)     -- still forbid duplicate pairings
);
```

Tradeoff: a surrogate `enrollment_id` gives child tables a single, stable column to reference (cleaner FKs than a two-column composite), at the cost of an extra column and needing a separate `UNIQUE` to keep the natural key's guarantee. Use the composite PK when the junction is a leaf; add a surrogate when other tables must reference individual rows of it.

#### Interview Variations
- **Ternary relationship:** add `term_id` so a student can take the same course in different terms → the key becomes `(student_id, course_id, term_id)`.
- **Self-referential M:N:** model "prerequisites" (courses ↔ courses) with a junction `prerequisites(course_id, prereq_course_id)`.
- **Query drills:** "students in ≥ 3 courses", "courses with no enrollments" — exercise the junction with `GROUP BY`/`HAVING` and anti-joins.

#### Common Follow-up Questions
- *Why can't you store M:N without a junction table?* A single FK column can reference only one row on the other side (that's 1:N). Representing "many on both sides" requires a separate table with one row per pair.
- *Where do relationship attributes go?* On the junction table — any attribute that depends on *both* participants (grade, enrolled_on, role) belongs to the relationship, not to either entity.
- *Composite PK vs surrogate PK on the junction?* Composite when nothing references individual pairings; surrogate when other tables must FK to a specific junction row. Either way, keep a uniqueness constraint on the natural pair.
- *What `ON DELETE` behavior is right?* Usually `CASCADE` (deleting a student removes their enrollments) or `RESTRICT` (block deletion while enrollments exist), depending on whether the pairing should vanish with its parent.

### Practical Question 12: Design a Ride-Sharing Schema (Roles, Trips, and a Rating Relationship)

- **Difficulty:** Medium
- **Estimated Time:** 22 min
- **Concepts Tested:** ER modeling, role modeling (a person as rider and driver), 1:N vs M:N, status lifecycle, ASCII ER diagram, DDL with constraints

**Problem Statement**
Design a schema for a ride-sharing service. A **user** can be a **rider**, a **driver**, or both. A **driver** operates one **vehicle** at a time (owns possibly several). A **trip** connects one rider and one driver, has pickup/dropoff, a status lifecycle, and a fare. After a trip, both parties can **rate** each other. Produce an ASCII ER diagram and DDL.

**Example Input** — requirements as above. Modeling subtleties:
- The same person can be both rider and driver → don't create two disjoint user tables; use roles.
- A trip references the user twice (as rider and as driver) → two FKs to the same table.
- Ratings are per-trip and directional (rider→driver and driver→rider).

**Example Output** — entities and relationships:

```
users ──1:1?── driver_profiles ──1:N── vehicles
  │                                        │
  │ (rider_id)          (driver_id)        │ (vehicle_id)
  └──────< trips >───────────┘─────────────┘
              │ 1:N
              └──< ratings >   (each trip → up to 2 directional ratings)
```

**Approach**
1. **One `users` table** with contact/auth fields; being a driver is an *added role*, modeled as `driver_profiles` (1:1 optional with users) holding license info. This avoids duplicating a person who is both.
2. **`vehicles`** belong to a driver: 1:N (`driver_id` FK).
3. **`trips`** carry two FKs to `users` — `rider_id` and `driver_id` — plus the operating `vehicle_id`, timestamps, locations, `status`, and `fare`.
4. **`ratings`** hang off a trip; make them directional with a `rater_id`/`ratee_id` and a per-direction uniqueness so each side rates once.

**ASCII ER diagram:**

```
 +-------------+          +------------------+         +-------------+
 |    users    |          | driver_profiles  |         |  vehicles   |
 +-------------+          +------------------+         +-------------+
 | user_id  PK |1───────0..1| user_id  PK/FK  |1──────< | vehicle_id PK
 | name        |          | license_no   UQ  |    N    | driver_id FK|>─┐
 | email    UQ |          | rating_avg       |         | plate    UQ |  │
 | phone    UQ |          +------------------+         | model       |  │
 +-------------+                                       +-------------+  │
     │  │                                                     ▲         │
     │  │ rider_id (N)                    driver_id (N)        │ vehicle │
     │  └────────────────< trips >───────────────┘────────────┘         │
     │                       │                                          │
     │                       │ 1                                        │
     │                       │ N                                        │
     │                 +-----------+                                    │
     └── rater/ratee ──|  ratings  |                                    │
                       +-----------+                                    │
                       | trip_id FK|>── trips                           │
                       | rater_id  |>── users                           │
                       | ratee_id  |>── users                           │
                       | stars     |                                    │
                       +-----------+                                    │
                                                                         (driver owns vehicles)
```

#### SQL Implementation

```sql
CREATE TABLE users (
    user_id  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name     TEXT NOT NULL,
    email    TEXT UNIQUE NOT NULL,
    phone    TEXT UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Being a driver is a ROLE layered on a user (optional 1:1). A user who is
-- only a rider simply has no driver_profiles row.
CREATE TABLE driver_profiles (
    user_id     BIGINT PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
    license_no  TEXT UNIQUE NOT NULL,
    verified_at TIMESTAMPTZ
);

CREATE TABLE vehicles (
    vehicle_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    driver_id  BIGINT NOT NULL REFERENCES driver_profiles(user_id) ON DELETE CASCADE,
    plate      TEXT UNIQUE NOT NULL,
    model      TEXT
);

CREATE TABLE trips (
    trip_id      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    rider_id     BIGINT NOT NULL REFERENCES users(user_id),
    driver_id    BIGINT NOT NULL REFERENCES driver_profiles(user_id),
    vehicle_id   BIGINT REFERENCES vehicles(vehicle_id),
    pickup_addr  TEXT NOT NULL,
    dropoff_addr TEXT,
    requested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    status       TEXT NOT NULL DEFAULT 'requested'
                 CHECK (status IN ('requested','accepted','in_progress','completed','cancelled')),
    fare         NUMERIC(10,2) CHECK (fare >= 0),
    CHECK (rider_id <> driver_id)          -- you can't drive yourself
);

-- Directional per-trip ratings: rider rates driver and vice-versa (each once).
CREATE TABLE ratings (
    trip_id  BIGINT NOT NULL REFERENCES trips(trip_id) ON DELETE CASCADE,
    rater_id BIGINT NOT NULL REFERENCES users(user_id),
    ratee_id BIGINT NOT NULL REFERENCES users(user_id),
    stars    SMALLINT NOT NULL CHECK (stars BETWEEN 1 AND 5),
    comment  TEXT,
    PRIMARY KEY (trip_id, rater_id),       -- one rating per rater per trip
    CHECK (rater_id <> ratee_id)
);
```

**Design decisions worth defending aloud.**
- **Role modeling over duplicate tables.** A person who both rides and drives is one `users` row plus an optional `driver_profiles` row. Splitting into separate `riders`/`drivers` tables would duplicate that person and their contact info — a normalization and integrity nightmare.
- **Two FKs to `users` from one table.** `trips.rider_id` and `trips.driver_id` both reference users; the `CHECK (rider_id <> driver_id)` encodes the rule that they must differ. This "self-referential via two roles" pattern comes up constantly (sender/receiver, manager/report, home/away team).
- **`ratings` PK `(trip_id, rater_id)`** guarantees each participant rates a given trip at most once, while allowing both directions (two rows, two different `rater_id`s).
- **Status as a `CHECK`-constrained enum-ish column** documents the lifecycle in the schema itself; a fuller design might use a real `ENUM` type or a `trip_status_history` table for auditing transitions.

#### Alternative Solution

Model the **rider/driver roles as an explicit roles table** instead of a dedicated `driver_profiles`, when there may be more roles later (admin, fleet-manager):

```sql
CREATE TABLE user_roles (
    user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    role    TEXT NOT NULL CHECK (role IN ('rider','driver','admin')),
    PRIMARY KEY (user_id, role)
);
-- Driver-specific fields (license) would then move to a driver_details table
-- keyed by user_id, populated only for users with the 'driver' role.
```

Tradeoff: an M:N `user_roles` table is more extensible (adding a role is a new enum value, not a new table) and cleaner if roles multiply — but it can't *by itself* enforce "only drivers have a license," so you still need a driver-specific detail table plus application checks. Use the dedicated `driver_profiles` when "driver" is a rich, first-class role with its own attributes; use a generic `user_roles` table when roles are many and lightweight.

#### Interview Variations
- **Surge pricing / fare breakdown:** split `fare` into base, distance, time, surge → a `fare_components` child table.
- **Live location:** a high-write `trip_locations(trip_id, ts, lat, lng)` stream — discuss partitioning and retention.
- **Driver availability / matching:** add a `driver_status` table (online/offline, current location) and discuss how you'd find the nearest available driver (spatial index / PostGIS).

#### Common Follow-up Questions
- *Why not separate `riders` and `drivers` tables?* Because one person can be both; separate tables duplicate the person and force you to keep two identities in sync. Model the shared identity once and layer roles on top.
- *How do two FKs to the same table work?* Perfectly fine — each FK column independently references `users(user_id)`; you distinguish them by column name (`rider_id`, `driver_id`) and enforce they differ with a `CHECK`.
- *How would you compute a driver's average rating?* Aggregate `ratings.stars` where `ratee_id` = the driver (across all their trips); optionally cache it in `driver_profiles.rating_avg` via a trigger for fast reads.
- *How do you model a trip's status changes over time?* Either a single `status` column (current state only) or a `trip_status_history(trip_id, status, changed_at)` table when you need the full audit trail of transitions.
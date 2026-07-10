# SQL — Complete Theory & Interview Preparation Guide

> A first-principles study guide covering the entire SQL syllabus (fundamentals → advanced → database design). Written for technical interviews, viva, written assessments, and practical lab exams. For hands-on coding questions, see [[practical]].

## How to Use This Guide

Every topic follows the same structure so you can study consistently:

- **What is it?** — the concept in plain language, and *why it exists*.
- **Why is it needed?** — the real-world problem it solves.
- **How does it work?** — step-by-step mechanics with SQL examples.
- **Internal Working** — what happens behind the scenes in the engine.
- **Advantages / Limitations** — trade-offs an interviewer expects you to know.
- **Real-world Applications** — where companies actually use it.
- **Interview Questions** — graded Beginner → Advanced, plus scenario/why/comparison questions.
- **Model Answers** — full reasoned answers (not one-liners).
- **Common Mistakes** — the traps beginners fall into.
- **Related Concepts** — what to study next.

> All examples are **PostgreSQL-flavored**, with MySQL differences noted where they matter.

---

## Table of Contents

**Week 1 — Fundamentals & CRUD**
1. Introduction to SQL
2. Database Fundamentals
3. SQL Data Types
4. CRUD — CREATE / INSERT
5. CRUD — READ / SELECT (Filtering, Sorting, Limiting)
6. CRUD — UPDATE
7. CRUD — DELETE (vs TRUNCATE vs DROP)
8. SQL Functions — String, Numeric, Date & NULL Handling

**Week 2 — Querying & Data Analysis**
9. Aggregate Functions
10. GROUP BY
11. HAVING
12. Joins (INNER, LEFT, RIGHT, FULL, SELF, CROSS)
13. Subqueries
14. Common Table Expressions (CTEs)

**Week 3 — Advanced SQL**
15. Window Functions
16. Views
17. Indexing
18. Stored Procedures & Functions
19. Performance Optimization (EXPLAIN / Execution Plans)

**Week 4 — Database Design & Transactions**
20. Transactions & ACID
21. Concurrency Control (Isolation Levels, Locking)
22. Deadlocks
23. Database Normalization (1NF → BCNF)
24. Database Design Principles
25. Entity Relationship (ER) Modeling

---


---

# Introduction to SQL

## What is it?

SQL (Structured Query Language) is a **declarative, domain-specific language** designed for managing and manipulating data held in a **relational database management system (RDBMS)**. When you write SQL, you are talking to a database engine — telling it *what* data you want, not *how* to physically go and get it.

Think of it this way. If a database is a giant, highly-organized warehouse of information, SQL is the language you use to talk to the warehouse manager. You don't walk the aisles yourself; you say "bring me every customer from Kerala who spent more than 10,000 last month," and the manager (the query engine) figures out the fastest route through the shelves.

SQL is standardized by both ANSI (American National Standards Institute) and ISO (International Organization for Standardization). The first standard appeared in 1986 (SQL-86), and it has been revised many times since (SQL-92, SQL:1999, SQL:2003, SQL:2008, SQL:2011, SQL:2016, SQL:2023). Every major database — PostgreSQL, MySQL, Oracle, SQL Server, SQLite — implements a *dialect* of this standard, meaning they share a large common core but each adds its own extensions and quirks.

SQL is broadly divided into sub-languages, and understanding this taxonomy early will save you confusion later:

| Sub-language | Full name | Purpose | Example statements |
|---|---|---|---|
| **DDL** | Data Definition Language | Define/alter database structure | `CREATE`, `ALTER`, `DROP`, `TRUNCATE` |
| **DML** | Data Manipulation Language | Modify the data itself | `INSERT`, `UPDATE`, `DELETE` |
| **DQL** | Data Query Language | Retrieve data | `SELECT` |
| **DCL** | Data Control Language | Permissions and access | `GRANT`, `REVOKE` |
| **TCL** | Transaction Control Language | Manage transactions | `COMMIT`, `ROLLBACK`, `SAVEPOINT` |

```sql
-- DDL: defining structure
CREATE TABLE customers (
    id       SERIAL PRIMARY KEY,
    name     VARCHAR(100) NOT NULL,
    city     VARCHAR(50),
    spend    NUMERIC(10, 2)
);

-- DML: putting data in
INSERT INTO customers (name, city, spend)
VALUES ('Aisha', 'Kochi', 12500.00);

-- DQL: getting data out
SELECT name, spend
FROM customers
WHERE city = 'Kochi' AND spend > 10000;
```

## Why is it needed?

Before relational databases and SQL, data was stored in **flat files** or **hierarchical/navigational databases** where the application code had to know the exact physical layout of the data on disk. If you moved a field or reorganized a file, every program that touched it broke. Retrieving related data meant writing procedural loops by hand, pointer-chasing through records.

SQL, built on E.F. Codd's **relational model** (1970), solved several deep problems at once:

1. **Data independence.** Your query describes the *logical* result you want. The physical storage — indexes, page layout, file organization — can change completely underneath, and your SQL keeps working. This decoupling is arguably the single most important idea in database engineering.

2. **A single, universal language.** Instead of every application inventing its own data-access code, SQL gives analysts, backend engineers, data scientists, and DBAs a *shared vocabulary*. A business analyst and a senior engineer can read the same query.

3. **Set-based thinking.** SQL operates on *sets of rows* at once, not one record at a time. This lets the engine parallelize, optimize, and use decades of research into query planning. A single `UPDATE` can modify a million rows.

4. **Declarative optimization.** Because you say *what* not *how*, the database's **query optimizer** is free to choose the best execution strategy based on current data statistics, available indexes, and memory. As your data grows from 1,000 to 100 million rows, the *same query* can silently switch strategies.

5. **Data integrity guarantees.** Through constraints and transactions (ACID properties), SQL databases prevent your data from ever entering a corrupt or contradictory state — something almost impossible to guarantee with hand-written file code.

In short: SQL is needed because it turns data management from a fragile, hand-coded, application-specific chore into a robust, standardized, mathematically-grounded discipline.

## How does it work?

At a high level, when you submit a SQL statement, the database engine runs it through a pipeline. Understanding this pipeline is what separates someone who *writes* SQL from someone who *reasons* about SQL.

```
   Your SQL text
        |
        v
  +-----------+     +-----------+     +-------------+     +-----------+     +-----------+
  |  PARSER   | --> | REWRITER/ | --> |  OPTIMIZER  | --> | EXECUTOR  | --> |  STORAGE  |
  | (syntax + |     | BINDER    |     | (chooses    |     | (runs the |     |  ENGINE   |
  |  grammar) |     | (resolve  |     |  the plan)  |     |  plan)    |     | (reads    |
  |           |     |  names)   |     |             |     |           |     |  pages)   |
  +-----------+     +-----------+     +-------------+     +-----------+     +-----------+
        |                 |                  |                  |                 |
   checks that      confirms tables/    generates and     iterates over     fetches actual
   the statement    columns exist,      compares many     the chosen         rows from disk
   is grammatically resolves types      possible          execution tree     or memory/cache
   valid                                execution plans    node by node
```

The crucial mental model for a beginner is the **logical order of evaluation** of a `SELECT`. People write clauses in one order but the database *logically* processes them in another. This explains almost every "why can't I use my column alias in WHERE?" question a junior ever asks.

```
Written order:            Logical evaluation order:
  SELECT      (5)           1. FROM / JOIN     -> build the source rows
  FROM        (1)           2. WHERE           -> filter rows
  WHERE       (2)           3. GROUP BY        -> form groups
  GROUP BY    (3)           4. HAVING          -> filter groups
  HAVING      (4)           5. SELECT          -> pick/compute columns (aliases born here)
  ORDER BY    (6)           6. ORDER BY        -> sort the result
  LIMIT       (7)           7. LIMIT / OFFSET  -> trim the result
```

Because `SELECT` (where aliases are defined) is evaluated at step 5, but `WHERE` runs at step 2, you *cannot* reference a `SELECT` alias in a `WHERE` clause — the alias doesn't exist yet at that point. `ORDER BY` (step 6) *can* use aliases because it runs after `SELECT`.

```sql
-- This FAILS: alias 'total' does not exist during WHERE evaluation
SELECT price * quantity AS total
FROM orders
WHERE total > 1000;          -- ERROR: column "total" does not exist

-- This WORKS: repeat the expression, or use a subquery/CTE
SELECT price * quantity AS total
FROM orders
WHERE price * quantity > 1000
ORDER BY total DESC;          -- alias is fine here
```

## Internal Working

Let's go one level deeper than most tutorials. When the optimizer produces a **query plan**, that plan is a tree of physical operators. You can *see* it. This is the most powerful diagnostic skill in SQL.

```sql
EXPLAIN ANALYZE
SELECT c.name, SUM(o.amount)
FROM customers c
JOIN orders o ON o.customer_id = c.id
WHERE c.city = 'Kochi'
GROUP BY c.name;
```

`EXPLAIN` shows the plan the optimizer *chose*; `ANALYZE` actually *runs* it and shows real timings and row counts. A simplified plan tree looks like:

```
GroupAggregate  (cost=... rows=...)          <- computes SUM per group
  -> Sort (on c.name)
       -> Hash Join  (o.customer_id = c.id)  <- combines the two tables
            -> Seq Scan on orders o          <- reads all orders
            -> Hash
                 -> Index Scan on customers c <- uses an index because of WHERE city
                      Filter: city = 'Kochi'
```

Key internal concepts every senior should internalize:

- **Access methods.** The engine can read a table by a full **Sequential Scan** (read every page — good when most rows match) or an **Index Scan** (jump via a B-tree index — good when few rows match). The optimizer picks based on **statistics** (histograms of column value distributions) collected by `ANALYZE`/`VACUUM ANALYZE`.

- **Join algorithms.** There are three canonical ones:
  - **Nested Loop Join** — for each row on the left, probe the right. Great for small inputs or when an index exists on the join key.
  - **Hash Join** — build a hash table on the smaller side, probe with the larger. Great for large, unsorted, equality joins.
  - **Merge Join** — sort both sides on the join key, then walk them in lockstep. Great when inputs are already sorted.

- **The cost model.** The optimizer assigns an estimated *cost* (an abstract number combining CPU and I/O) to each candidate plan and picks the cheapest. It relies on cardinality estimates — its guess of how many rows each step produces. Bad statistics → bad estimates → bad plans. This is why "the same query is suddenly slow" is often a statistics problem, not a code problem.

- **The buffer cache / buffer pool.** Databases don't read disk on every query. They keep frequently-used pages (typically 8 KB in PostgreSQL, 16 KB default in InnoDB/MySQL) in a shared in-memory cache. The Write-Ahead Log (WAL / redo log) ensures durability without writing every change to the main data files immediately.

## Advantages

- **Declarative & concise.** Complex data operations that would take hundreds of lines of procedural code become a handful of readable clauses.
- **Portable knowledge.** The 80% core is identical everywhere; learning SQL once lets you work against nearly any relational database.
- **Battle-tested optimization.** You inherit 50+ years of query-optimizer research for free.
- **Strong data integrity.** Constraints, transactions, and ACID guarantees are built in, not bolted on.
- **Scales with data, not code.** The same query adapts as the dataset grows; the engine changes strategy, you don't change SQL.
- **Rich ecosystem.** Tooling, ORMs, BI platforms, and drivers exist for every language.
- **Ad-hoc analysis.** Non-programmers can answer business questions directly.

## Limitations

- **Impedance mismatch with application code.** Object-oriented languages think in objects and pointers; SQL thinks in sets and relations. Bridging the two (via ORMs) introduces friction and sometimes inefficiency.
- **Dialect fragmentation.** Despite the standard, `LIMIT` vs `TOP` vs `FETCH FIRST`, string functions, date handling, and `AUTO_INCREMENT` vs `SERIAL` differ across vendors, hurting true portability.
- **Weak for hierarchical/graph traversal.** Recursive relationships and deep graph traversals are awkward in SQL (recursive CTEs help but are clunky compared to a graph database).
- **Not procedural by default.** Loops, branching, and complex control flow require stored-procedure extensions (PL/pgSQL, T-SQL, PL/SQL) that *are* vendor-specific.
- **NULL semantics are subtle.** Three-valued logic (`TRUE`/`FALSE`/`UNKNOWN`) trips up nearly every developer at some point.
- **Horizontal scaling is hard.** Traditional single-node RDBMS scale up more easily than out; sharding a relational database is non-trivial (part of why NoSQL emerged).

## Real-world Applications

- **Transactional systems (OLTP):** banking, e-commerce checkout, ticket booking — anywhere correctness and concurrency matter.
- **Analytics & reporting (OLAP):** dashboards, business intelligence, financial reporting over data warehouses (often SQL-compatible engines like Snowflake, BigQuery, Redshift).
- **Backend of virtually every web/mobile app:** user accounts, content, orders, messages typically live in an RDBMS.
- **Data engineering pipelines:** SQL is the lingua franca of ETL/ELT; tools like dbt are essentially "SQL as software engineering."
- **Embedded storage:** SQLite (SQL in a single file) powers phones, browsers, and desktop apps.
- **Data science:** feature extraction, cohort analysis, and exploratory analysis frequently start with SQL before data ever reaches Python/R.

## Setting Up an Environment

**PostgreSQL** (open-source, standards-leading, feature-rich):

```bash
# macOS (Homebrew)
brew install postgresql@16
brew services start postgresql@16

# Ubuntu/Debian
sudo apt update && sudo apt install postgresql

# Connect via the psql client
psql -U postgres -d postgres

# Inside psql
CREATE DATABASE shop;
\c shop            -- connect to the 'shop' database
\dt                -- list tables
\d customers       -- describe a table
\q                 -- quit
```

**MySQL** (extremely popular, especially in the web/LAMP world):

```bash
# macOS
brew install mysql
brew services start mysql

# Ubuntu/Debian
sudo apt install mysql-server

# Connect via the mysql client
mysql -u root -p

# Inside the mysql shell
CREATE DATABASE shop;
USE shop;          -- MySQL's equivalent of \c
SHOW TABLES;
DESCRIBE customers;
```

Key differences to note from day one:

| Aspect | PostgreSQL | MySQL |
|---|---|---|
| Auto-increment key | `SERIAL` / `GENERATED ... AS IDENTITY` | `AUTO_INCREMENT` |
| String comparison | Case-sensitive by default | Case-insensitive by default (depends on collation) |
| Default quoting of identifiers | Double quotes `"col"` | Backticks `` `col` `` |
| Boolean type | Native `BOOLEAN` | `TINYINT(1)` under the hood |
| Full standard compliance | Very high | Historically looser (improving) |

## SQL Dialects and Standards

The ANSI/ISO SQL standard defines the *core* language, but no vendor implements 100% of it, and every vendor adds extensions. Think of the standard as the "constitution" and each dialect as a "local law."

```
                +---------------------------+
                |   ANSI/ISO SQL Standard   |   <- common core (SELECT, JOIN,
                |   (SQL-92, :1999, :2016)  |      constraints, transactions...)
                +---------------------------+
                  /       |         |       \
                 /        |         |        \
          PostgreSQL    MySQL   SQL Server   Oracle
          (PL/pgSQL,  (AUTO_    (T-SQL,      (PL/SQL,
           SERIAL,     INCREMENT, TOP,        ROWNUM,
           arrays,     backticks) MERGE)      sequences)
           JSONB)
```

Practical portability advice: write to the standard when you can, isolate vendor-specific bits (identity columns, `LIMIT`/`TOP`, upsert syntax, string/date functions), and never assume a query tested on MySQL behaves identically on PostgreSQL — especially around `NULL` handling, `GROUP BY` strictness, and implicit type casting.

# Database Fundamentals

## What is it?

This topic covers the *conceptual building blocks* of the relational model — the vocabulary and structural rules that everything else in SQL rests on. If Topic 1 was "what is the language," this is "what are the nouns the language talks about."

Let's define each term precisely, because interviewers probe whether you know the difference between a *logical* concept and its *physical* implementation.

- **Database:** A structured, organized collection of related data managed by a DBMS. It is the top-level container. One PostgreSQL/MySQL *server instance* can host many databases (e.g., `shop`, `analytics`, `auth`), each isolated from the others.

- **Table (Relation):** A two-dimensional structure of rows and columns that stores data about a single kind of entity (customers, orders, products). In formal relational theory a table is a *relation* — a set of tuples. The word "set" matters: relational theory says a table has no inherent row order and no duplicate rows (though SQL tables, unlike pure relations, *do* permit duplicates unless a key forbids them).

- **Row (Record / Tuple):** A single, complete entry in a table — one customer, one order. It is a horizontal slice representing one instance of the entity.

- **Column (Field / Attribute):** A named, typed vertical slice — one property of the entity, like `email` or `created_at`. Every value in a column shares the same data type (this is *domain integrity*).

- **Primary Key (PK):** One or more columns whose values *uniquely identify* each row. A PK must be **unique** and **not null**. Each table has at most one primary key (which may be *composite* — spanning multiple columns).

- **Foreign Key (FK):** One or more columns in a table that *reference* the primary key (or a unique key) of another table, establishing a relationship and enforcing **referential integrity** — you cannot reference a parent row that doesn't exist.

- **Constraint:** A rule the database enforces on data to guarantee validity: `NOT NULL`, `UNIQUE`, `CHECK`, `DEFAULT`, `PRIMARY KEY`, `FOREIGN KEY`.

- **Schema:** An overloaded word. (1) In the general sense, "schema" means the overall *design/structure* of a database — the tables, columns, types, and relationships. (2) In PostgreSQL specifically, a **schema** is also a *namespace* inside a database — a folder that groups tables (`public.customers`, `sales.orders`). MySQL treats "schema" and "database" as synonyms.

```
DATABASE: shop
│
├── SCHEMA: public
│     │
│     ├── TABLE: customers
│     │     ┌────┬──────────┬──────────┐
│     │     │ id │  name    │  city    │   <- columns (attributes)
│     │     ├────┼──────────┼──────────┤
│     │     │ 1  │  Aisha   │  Kochi   │   <- row (tuple / record)
│     │     │ 2  │  Ravi    │  Delhi   │
│     │     └────┴──────────┴──────────┘
│     │        ▲
│     │        └─ PRIMARY KEY (id)
│     │
│     └── TABLE: orders
│           ┌────┬─────────────┬────────┐
│           │ id │ customer_id │ amount │
│           ├────┼─────────────┼────────┤
│           │ 10 │     1       │ 12500  │   customer_id ──┐
│           └────┴─────────────┴────────┘                 │
│                       └── FOREIGN KEY (customer_id) ─────┘
│                            REFERENCES customers(id)
```

## Why is it needed?

These fundamentals exist to solve the problems of **redundancy, inconsistency, and integrity** that plague unstructured data.

Imagine storing orders in a single spreadsheet where each order row *also* holds the customer's full name, address, and phone. If Aisha places 50 orders, her address is copied 50 times. Now she moves house — you must update 50 rows, and if you miss one, your data is *inconsistent*. This is called an **update anomaly**. Related evils are **insertion anomalies** (you can't record a customer until they place an order) and **deletion anomalies** (deleting the last order erases the customer entirely).

The relational fundamentals fix this:

- **Tables + normalization** let each fact live in exactly one place. Customer data in `customers`, orders in `orders`.
- **Primary keys** give every entity a stable, unique identity so it can be referenced without ambiguity.
- **Foreign keys** connect tables reliably, so "which customer placed this order" is a guaranteed, enforced link — not a fragile text-matching guess.
- **Constraints** push data-validity rules *down into the database*, where they are enforced consistently no matter which application, script, or human writes the data. A rule enforced only in application code is a rule waiting to be bypassed.
- **Schemas** provide organization and access control at scale, so a large database of hundreds of tables stays navigable and securable.

The deep principle: **the database should be the last line of defense for data correctness.** Application bugs come and go; a `CHECK (price >= 0)` constraint protects you forever.

## How does it work?

You declare all of this with DDL, and the engine enforces it on every write.

```sql
-- Parent table with a primary key
CREATE TABLE customers (
    id      SERIAL      PRIMARY KEY,            -- PK: unique + not null, auto-generated
    email   VARCHAR(255) UNIQUE NOT NULL,       -- must exist and be unique
    name    VARCHAR(100) NOT NULL,
    country CHAR(2)      NOT NULL DEFAULT 'IN',  -- default when unspecified
    age     INT          CHECK (age >= 18)      -- domain rule
);

-- Child table with a foreign key
CREATE TABLE orders (
    id          SERIAL PRIMARY KEY,
    customer_id INT    NOT NULL,
    amount      NUMERIC(10,2) NOT NULL CHECK (amount > 0),
    created_at  TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_customer
        FOREIGN KEY (customer_id)
        REFERENCES customers(id)
        ON DELETE RESTRICT      -- block deleting a customer who has orders
        ON UPDATE CASCADE       -- if the customer's id changes, propagate it
);
```

**Composite primary key** (a key made of multiple columns — common in junction tables for many-to-many relationships):

```sql
CREATE TABLE student_courses (
    student_id INT REFERENCES students(id),
    course_id  INT REFERENCES courses(id),
    enrolled   DATE NOT NULL DEFAULT CURRENT_DATE,
    PRIMARY KEY (student_id, course_id)   -- the *pair* must be unique
);
```

Now if you try to insert an order referencing a non-existent customer, or a second customer with a duplicate email, or a negative amount, the engine *rejects the write* with a constraint-violation error. The bad data never lands.

## Internal Working

Under the hood, these logical concepts map to concrete physical structures:

- **Primary keys and unique constraints are almost always backed by an index** — typically a **B-tree**. When you declare `PRIMARY KEY`, the engine silently creates a unique B-tree index on that column. This is *how* it enforces uniqueness efficiently: to check whether a value already exists, it walks the B-tree in O(log n) time instead of scanning the whole table.

```
        A B-tree index on customers(id)

                    [ 50 ]
                   /      \
            [20, 35]      [70, 90]
            /   |   \      /   |   \
        ...  ...  ...   ...  ...  ...   <- leaf nodes point to row locations

  Lookup of id=70: root -> right child -> found. ~O(log n) page reads.
```

- **Foreign keys are enforced via triggers/checks at write time.** When you insert/update a child row, the engine looks up the referenced key in the parent's unique index. When you delete/update a parent row, it must check children — which is why **indexing your foreign key columns matters for performance** (the child side is not auto-indexed in PostgreSQL, a classic performance gotcha).

- **Referential actions** (`ON DELETE`/`ON UPDATE`) define what happens to children when the parent changes:

| Action | Behavior on parent delete/update |
|---|---|
| `NO ACTION` | Reject if children exist (checked at end of statement) — default |
| `RESTRICT` | Reject immediately if children exist |
| `CASCADE` | Delete/update the children too |
| `SET NULL` | Set the child FK column to `NULL` |
| `SET DEFAULT` | Set the child FK column to its default |

- **Constraints are metadata stored in the system catalog** (`pg_constraint` in PostgreSQL, `information_schema.table_constraints` in the standard). The engine consults them during query planning and enforces them during execution.

- **Rows are physically stored in pages** (fixed-size blocks). A table is a heap of pages; indexes are separate structures pointing into that heap. This separation is *why* you can have many indexes on one table.

## Advantages

- **Data integrity by construction.** Invalid states become literally impossible to store.
- **Eliminates redundancy.** Each fact stored once → no update anomalies.
- **Self-documenting structure.** The schema, keys, and constraints describe the business rules; a new engineer can read the DDL and understand the domain.
- **Reliable relationships.** Foreign keys guarantee referential integrity across tables.
- **Efficient identity lookups.** PK/unique indexes make finding and joining rows fast.
- **Centralized enforcement.** Rules live in one place (the DB), not scattered across every app.

## Limitations

- **Rigidity.** A strict schema resists change; adding/removing columns on huge tables can require careful migrations and locking.
- **Constraint-checking overhead.** Every FK and CHECK adds work on writes; extremely high-throughput ingest pipelines sometimes disable/defer constraints for speed.
- **Foreign keys complicate sharding.** Cross-node referential integrity is hard, so massively distributed systems often drop FK enforcement.
- **Composite keys can be cumbersome** to reference from many child tables (a reason some teams prefer surrogate keys).
- **Schema-per-namespace management** adds administrative complexity in large multi-tenant systems.

## Real-world Applications

- **E-commerce:** `customers`, `orders`, `order_items`, `products` linked by keys; FKs ensure no order references a deleted product.
- **Banking:** `accounts`, `transactions` with CHECK constraints guaranteeing balances and non-negative amounts.
- **Multi-tenant SaaS:** PostgreSQL schemas isolate each tenant's tables within one database.
- **Reference/lookup tables:** `countries`, `currencies`, `roles` referenced by FK across the system for consistency.
- **Join/junction tables:** many-to-many relationships (students↔courses, users↔roles) implemented via composite-key tables.

## Interview Questions

**Beginner**
1. What is the difference between a row and a column?
2. What is a primary key, and can a table have more than one?
3. What does the `NOT NULL` constraint do?
4. What is the difference between a `PRIMARY KEY` and a `UNIQUE` constraint?
5. What is a foreign key?

**Intermediate**
6. What is referential integrity, and how do foreign keys enforce it?
7. Explain the difference between `ON DELETE CASCADE` and `ON DELETE RESTRICT`.
8. What is a composite primary key and when would you use one?
9. What is the difference between a `CHECK` constraint and a `DEFAULT`?
10. In PostgreSQL, what is the difference between a database and a schema?

**Advanced**
11. Why is a primary key usually backed by an index, and what type of index?
12. Why should you index foreign key columns even though the constraint works without an index?
13. What are update, insertion, and deletion anomalies, and how do keys/normalization prevent them?
14. How does `NULL` interact with `UNIQUE` constraints across different databases?

**Scenario-based**
15. You must delete a customer who has 200 orders. What are your options and their trade-offs?
16. A table has millions of rows and you need to add a `NOT NULL` column. What problems arise and how do you handle it?

**"Why" questions**
17. Why enforce data rules in the database rather than only in application code?
18. Why can a foreign key reference a `UNIQUE` column and not just a `PRIMARY KEY`?

**Comparison questions**
19. Surrogate key vs natural key — trade-offs?
20. `NO ACTION` vs `RESTRICT` — is there a real difference?

## Model Answers

**1. Row vs column.** A *row* (record/tuple) is a single horizontal entry in a table representing one instance of the entity the table models — for example, one specific customer with all their attributes filled in. A *column* (field/attribute) is a vertical, named, typed slice representing a single property shared by every row — for example, `email`. The key conceptual point is that a column enforces *domain integrity*: every value in it must be of the same declared type and obey the same constraints, whereas a row bundles one value from each column into a complete, meaningful record. Rows are about *entities*; columns are about *attributes*.

**2. Primary key, and can there be more than one.** A primary key is the column or set of columns whose values uniquely identify each row in the table. It carries two guarantees automatically: the values are *unique* (no two rows share them) and *not null* (every row must have one). A table can have **at most one primary key**, because a primary key expresses *the* canonical identity of a row and there can only be one canonical identity. However, that single primary key may be *composite* — spanning multiple columns — and a table may additionally have several `UNIQUE` constraints that also enforce uniqueness on other columns. So "one primary key, but possibly many unique keys" is the precise answer. The primary key is also, by convention, the natural target for foreign keys from other tables.

**3. NOT NULL.** The `NOT NULL` constraint forbids a column from containing the special marker `NULL`, which represents "unknown" or "absent" data. Declaring a column `NOT NULL` means every insert or update *must* supply a real value for it; otherwise the write is rejected. This matters because `NULL` propagates surprising three-valued-logic behavior through queries (comparisons with `NULL` yield `UNKNOWN`, not `TRUE`/`FALSE`), and because certain columns are conceptually mandatory — a customer without a name, or an order without an amount, is meaningless. Making mandatory columns `NOT NULL` pushes that business rule into the database so no code path can accidentally create a half-formed record.

**4. PRIMARY KEY vs UNIQUE.** Both enforce uniqueness of values, and both are typically backed by a unique index. The differences are: (a) a `PRIMARY KEY` also implies `NOT NULL`, whereas a `UNIQUE` column *can* hold null values; (b) a table may have only one primary key but many unique constraints; (c) the primary key is the semantically-preferred identity of the row and the default reference target for foreign keys. A subtle and interview-favorite point is how nulls behave under `UNIQUE`: in standard SQL and PostgreSQL, `NULL` is never equal to `NULL`, so a unique column may contain *multiple* null rows. (PostgreSQL 15+ adds `NULLS NOT DISTINCT` to change this.) So conceptually: primary key = "the identity, mandatory and singular"; unique = "an alternate key, optional-nullable and repeatable-as-a-constraint."

**5. Foreign key.** A foreign key is a column (or set of columns) in one table whose values are required to match a primary key or unique key in another (or the same) table. It formalizes a *relationship* — for instance, `orders.customer_id` referencing `customers.id` states "every order belongs to an existing customer." Its purpose is **referential integrity**: the database will refuse to insert an order for a customer that doesn't exist and (depending on referential actions) will refuse to delete a customer who still has orders. This transforms a relationship from a fragile, hope-based convention into a hard, enforced guarantee.

**6. Referential integrity.** Referential integrity is the property that every foreign-key value in a child table corresponds to an actually-existing key in the parent table — there are no "dangling" references pointing at nothing. Foreign keys enforce this on *both* sides of every write. On the child side: when you insert or update a row, the engine looks up the referenced value in the parent's unique/primary-key index and rejects the operation if it's absent. On the parent side: when you delete or update the referenced key, the engine checks whether any child rows still point to it and applies the declared referential action (`RESTRICT`, `CASCADE`, `SET NULL`, etc.). Because these checks are performed by the engine within transactions, the relationship can never be silently broken, even under concurrent access.

**7. CASCADE vs RESTRICT on delete.** `ON DELETE CASCADE` means that when a parent row is deleted, all child rows referencing it are *automatically deleted too* — deleting a customer wipes out all their orders in the same transaction. `ON DELETE RESTRICT` means the delete of the parent is *blocked* if any child rows still reference it — you must delete or reassign the orders first. The trade-off is safety versus convenience: `CASCADE` is powerful but dangerous, because a single delete can silently remove huge subtrees of data; `RESTRICT` is safer because it forces an explicit decision, but it makes cleanup more work. A senior engineer chooses `CASCADE` only when the child truly cannot exist without the parent (e.g., `order_items` under an `order`), and `RESTRICT`/`NO ACTION` when children have independent value (e.g., you'd never want deleting a `category` to vaporize every `product` in it).

**8. Composite primary key.** A composite primary key is a primary key composed of two or more columns, where the *combination* must be unique even though each column individually may repeat. The classic use case is a **junction/join table** modeling a many-to-many relationship — for example `student_courses(student_id, course_id)`: a student appears in many rows, a course appears in many rows, but each (student, course) *pair* appears at most once, correctly expressing "a student can enroll in a course only once." You'd use a composite key when the natural identity of a row genuinely depends on multiple attributes together. The trade-off is that other tables referencing this row must carry all the key columns as their foreign key, which is why many teams instead add a single surrogate `id` and demote the composite to a `UNIQUE` constraint.

**9. CHECK vs DEFAULT.** These solve different problems. A `CHECK` constraint is a *validation rule* — a boolean expression that every row must satisfy, such as `CHECK (age >= 18)` or `CHECK (amount > 0)`; if a write would violate it, the write is rejected. A `DEFAULT` is a *value-supplying* mechanism — it specifies what value a column receives when an `INSERT` omits it, such as `DEFAULT NOW()` or `DEFAULT 'IN'`; it never rejects anything, it fills in blanks. In short, `CHECK` guards the *gate* (is this value allowed?), while `DEFAULT` stocks the *shelf* (what value do we use when none is given?). They frequently coexist on the same column.

**10. Database vs schema in PostgreSQL.** In PostgreSQL these are two distinct levels of containment. A *database* is a top-level, fully-isolated container; you connect to exactly one database per session, and (importantly) you *cannot* run an ordinary query that joins tables across two different databases in the same statement. A *schema* is a namespace *inside* a database — a logical folder that groups tables, views, and functions, allowing qualified names like `sales.orders` and `hr.employees` to coexist without collision. Schemas also serve as a unit of permission granting and are heavily used for multi-tenancy and for separating application modules. (Note the vocabulary trap: MySQL uses "schema" and "database" as synonyms, so this distinction is PostgreSQL/standard-specific.)

**11. Why a PK is backed by an index, and which type.** To *enforce* uniqueness, the engine must, on every insert, answer "does this key already exist?" quickly. Scanning the whole table for each insert would be O(n) and catastrophic at scale, so the engine automatically maintains a **unique index** — almost always a **B-tree** — on the primary-key columns. A B-tree gives O(log n) existence checks and, as a bonus, makes point lookups and range scans on the key fast, which is exactly what joins and `WHERE id = ?` queries need. So the index is not an optional optimization here; it is the *mechanism* by which the uniqueness guarantee is made affordable. (Hash indexes could enforce uniqueness for equality too, but B-trees are the default because they also support ordering and ranges.)

**12. Why index foreign key columns.** The foreign-key *constraint* works correctly without an index on the child column — the parent side is indexed (it's a PK/unique key), which is what the constraint check on inserts needs. The reason to *also* index the child column is **performance on parent-side operations and joins**. When you delete or update a parent row, the engine must find all children referencing it; without an index on the child's FK column, that becomes a full sequential scan of the child table for every such operation — devastating on large tables. Additionally, virtually every query that joins parent to child filters on that FK column, so the index accelerates normal read traffic. PostgreSQL notably does *not* auto-create this index (MySQL/InnoDB does), making "forgot to index the FK" one of the most common real-world performance bugs.

**13. The three anomalies.** These are the data-corruption hazards that arise when data is stored redundantly in a single un-normalized table. An **update anomaly** occurs when a fact is duplicated across many rows (e.g., a customer's address stored on every one of their orders): updating it requires changing every copy, and missing one leaves the data inconsistent. An **insertion anomaly** occurs when you cannot record one fact without another being present (e.g., you can't add a new customer until they place an order, because customer data only lives in the orders table). A **deletion anomaly** occurs when removing one fact unintentionally destroys another (e.g., deleting a customer's last order erases the only record of that customer). Splitting data into properly-keyed tables — normalization — ensures each fact lives in exactly one place, so each of these anomalies simply cannot occur: you update the address once, you insert a customer independently of orders, and deleting an order leaves the customer intact.

**14. NULL and UNIQUE across databases.** Because SQL's three-valued logic treats `NULL = NULL` as `UNKNOWN` (not `TRUE`), most databases consider two null values *distinct* for uniqueness purposes, so a `UNIQUE` column may contain multiple null rows. This is the standard/PostgreSQL/Oracle/MySQL behavior. The notable historical outlier is Microsoft SQL Server, whose unique *indexes* allow only a *single* null. PostgreSQL 15 introduced `UNIQUE NULLS NOT DISTINCT` to opt into treating nulls as equal (thus permitting only one). The practical lesson: never rely on a `UNIQUE` constraint to prevent duplicate *nulls* — if a column must be both unique and always-present, combine `UNIQUE` with `NOT NULL`.

**15. Deleting a customer with 200 orders.** Your options depend on the foreign-key referential action and the business meaning. (a) If orders should die with the customer, `ON DELETE CASCADE` deletes all 200 automatically in one transaction — simplest, but destructive and irreversible, and it can be surprisingly expensive if the cascade fans out further. (b) If orders must be preserved for audit/financial reasons, use `ON DELETE RESTRICT`/`NO ACTION` and instead *soft-delete* the customer (set an `is_deleted`/`deleted_at` flag) so the orders keep their reference intact. (c) `ON DELETE SET NULL` orphans the orders by nulling `customer_id` — appropriate only if an order can meaningfully exist without a customer, which is rare. The senior instinct is that hard-deleting entities with financial history is usually wrong; soft deletion preserves referential integrity and auditability, and you almost never want a stray `CASCADE` silently erasing transactional records.

**16. Adding a NOT NULL column to a huge table.** The core problem is twofold: you must supply a value for existing rows (a `NOT NULL` column can't be left empty), and historically the operation rewrites/locks the table, blocking traffic. The safe pattern is: (1) add the column as *nullable* first (fast, metadata-only in modern PostgreSQL if no default, or with a constant default which PG 11+ handles without a rewrite); (2) backfill existing rows in *batches* to avoid long locks and huge transactions; (3) once every row has a value, add the `NOT NULL` constraint — in recent PostgreSQL you can add a `CHECK (col IS NOT NULL) NOT VALID`, validate it concurrently, then promote, minimizing lock time. The anti-pattern is `ALTER TABLE ... ADD COLUMN ... NOT NULL DEFAULT <volatile>` on an old engine, which takes a long exclusive lock while rewriting every page. Always test the migration's locking behavior against your specific version.

**17. Why enforce rules in the DB, not just the app.** Because the database is the single, final chokepoint through which *all* writes pass, regardless of which application, microservice, admin script, migration, or human-with-a-SQL-console produced them. Application-level validation protects only the code paths you remembered to guard; the moment a second service, a bulk import, or a hotfix bypasses that code, invalid data slips in — and bad data is far more expensive to fix after the fact than to prevent. Database constraints (NOT NULL, CHECK, UNIQUE, FK) are declarative, always-on, and immune to being forgotten. The best practice is *defense in depth*: validate in the app for good UX and fast feedback, but back it with database constraints as the guarantee of last resort. Trusting application code alone is trusting that every future developer never makes a mistake.

**18. Why an FK can reference a UNIQUE column.** A foreign key's job is to point at a row that can be *uniquely identified* by the referenced column(s). What it actually requires is that the target column has a uniqueness guarantee so that "the referenced row" is unambiguous — and both `PRIMARY KEY` and `UNIQUE` provide exactly that guarantee (both are backed by a unique index the engine can probe). The primary key is merely the *conventional* target, but the SQL standard permits referencing any unique key. This is useful when you want to relate tables via a natural unique attribute (say, `products.sku`) rather than the surrogate primary key. The one thing you cannot reference is a non-unique column, because then "the parent row" wouldn't be well-defined.

**19. Surrogate vs natural key.** A *natural key* uses real-world data that is already unique (email, SSN, ISBN, country code) as the primary key. A *surrogate key* is a system-generated, meaningless identifier (an auto-increment integer or UUID) used solely for identity. Natural keys are meaningful and can eliminate a join in some cases, but they have serious drawbacks: real-world "unique" values change (people change emails, countries change codes), they may be large (hurting index and FK size), and they can leak sensitive data into every referencing table. Surrogate keys are stable (they never change), compact, and decoupled from business meaning, which is why they're the common default — at the cost of an extra column and the need to also put a `UNIQUE` constraint on the natural attribute to prevent duplicates. Many senior designs use a surrogate PK *plus* a unique natural key, getting stability and integrity together.

**20. NO ACTION vs RESTRICT.** Both prevent you from deleting/updating a parent row that still has referencing children, and in casual use they look identical. The real difference is *timing*. `RESTRICT` checks *immediately* when the row is affected and fails right away. `NO ACTION` (the SQL-standard default) defers the check to the *end of the statement*, which means if the constraint is `DEFERRABLE`, the violation can even be postponed to the end of the transaction — giving you a window in which the referential integrity is temporarily inconsistent as long as it's resolved before commit. So `NO ACTION` is the more flexible, standard-compliant behavior (compatible with deferred constraints), while `RESTRICT` is stricter and immediate. For most everyday schemas the practical effect is the same, but the distinction matters when you need to reorder operations within a transaction.

## Common Mistakes

- **Forgetting to index foreign key columns**, then wondering why parent deletes and joins are slow.
- **Assuming `UNIQUE` prevents duplicate NULLs** — it usually doesn't; combine with `NOT NULL`.
- **Using natural keys that later change** (email as PK), causing painful cascading updates.
- **Reflexively adding `ON DELETE CASCADE`** and later discovering a single delete wiped out an entire subtree of important data.
- **Confusing "schema" the design with "schema" the PostgreSQL namespace** in conversation and code.
- **Enforcing rules only in the app** and trusting that no other writer will ever bypass them.
- **Declaring everything nullable** "to be safe," which pushes NULL-handling complexity into every downstream query.
- **Creating tables without a primary key at all**, making rows impossible to reference or deduplicate reliably.

## Related Concepts

- **Normalization (1NF–BCNF)** — the formal process that motivates splitting data into keyed tables.
- **Indexes (B-tree, hash)** — the physical structures that make keys and constraints efficient.
- **ACID transactions** — the guarantee under which constraint enforcement remains consistent concurrently.
- **ER modeling** — entity-relationship diagrams that precede and inform table/key design.
- **Referential actions** — CASCADE/RESTRICT/SET NULL behavior on FKs.
- **Surrogate vs natural keys** — key-selection strategy.
- **System catalog / information_schema** — where constraint and schema metadata lives.

# SQL Data Types

## What is it?

A **data type** is the declaration of *what kind of value* a column may hold and *how the database physically stores and interprets those bits*. When you write `age INT` or `price NUMERIC(10,2)`, you are telling the engine three things at once: the set of legal values (the *domain*), the storage layout on disk, and the operations that make sense (you can `SUM` an integer but not concatenate it like text).

Choosing types is not a formality — it is one of the highest-leverage design decisions you make. The right type gives you correctness (a date behaves like a date), efficiency (compact storage, fast comparisons), and safety (the engine rejects nonsense like `'abc'` in an integer column). The wrong type invites rounding bugs in money, silent truncation of text, timezone chaos, and bloated indexes.

SQL types fall into a few broad families:

| Family | Examples (PostgreSQL) | Stores |
|---|---|---|
| **Numeric — exact** | `SMALLINT`, `INTEGER`, `BIGINT`, `NUMERIC`/`DECIMAL` | whole numbers, exact decimals |
| **Numeric — approximate** | `REAL`, `DOUBLE PRECISION` (`FLOAT`) | floating-point reals |
| **Character/String** | `CHAR(n)`, `VARCHAR(n)`, `TEXT` | text |
| **Date/Time** | `DATE`, `TIME`, `TIMESTAMP`, `TIMESTAMPTZ`, `INTERVAL` | temporal values |
| **Boolean** | `BOOLEAN` | true/false/unknown |
| **Others** | `UUID`, `JSONB`, `BYTEA`, `ARRAY`, `ENUM` | specialized data |

## Why is it needed?

Types are needed because *bits are meaningless without interpretation*. The same 32 bits can be an integer, a float, or four characters — the type tells the engine which. Beyond that basic role, well-chosen types deliver several concrete benefits:

1. **Correctness and domain integrity.** A `DATE` column can't hold "banana"; a `NUMERIC` column won't silently round your money. The type is the first and cheapest constraint on your data.

2. **Storage efficiency.** An `INTEGER` uses 4 bytes; a `BIGINT` 8; a `SMALLINT` 2. Multiply by billions of rows and the difference is gigabytes of disk, RAM (buffer cache), and backup size. Smaller rows mean more rows per page, means fewer I/O operations, means faster queries.

3. **Performance.** Comparisons and arithmetic on fixed-width numeric types are single CPU instructions; comparing text is far slower. Indexes on compact types are smaller and faster to traverse.

4. **Correct operations and sorting.** Store a number as text and `'100' < '9'` sorts wrong (lexicographically). Store a date as text and range queries and date math break. The type unlocks the *right* semantics for `<`, `SUM`, `ORDER BY`, etc.

5. **Self-documentation.** `is_active BOOLEAN` and `created_at TIMESTAMPTZ` tell the next engineer exactly what the data means.

The overarching principle: **model the data as what it actually is.** Money is exact decimal, not float. A timestamp is a moment in time, not a string. Choosing the truthful type prevents an entire category of bugs before it can exist.

## How does it work?

You attach a type to each column in DDL, and the engine enforces and stores accordingly.

```sql
CREATE TABLE products (
    id          BIGINT       GENERATED ALWAYS AS IDENTITY,  -- exact integer, 8 bytes
    sku         VARCHAR(32)  NOT NULL,                       -- variable text, capped
    name        TEXT         NOT NULL,                       -- unbounded text
    description TEXT,
    price       NUMERIC(10,2) NOT NULL CHECK (price >= 0),   -- exact money: 10 digits, 2 after point
    weight_kg   REAL,                                        -- approximate float, fine for physical measures
    in_stock    BOOLEAN      NOT NULL DEFAULT TRUE,
    launch_date DATE,                                        -- calendar day, no time
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()          -- absolute instant, tz-aware
);
```

**Numeric — exact vs approximate.** This is the single most important distinction. `NUMERIC(p, s)` (a.k.a. `DECIMAL`) stores numbers *exactly* using base-10 digits — `p` total significant digits, `s` after the decimal point. `REAL`/`DOUBLE PRECISION` (IEEE 754 floating point) store *approximations* in base-2, which cannot represent many decimal fractions precisely.

```sql
-- The classic float trap
SELECT 0.1::DOUBLE PRECISION + 0.2::DOUBLE PRECISION;   -- 0.30000000000000004  (!)
SELECT 0.1::NUMERIC        + 0.2::NUMERIC;              -- 0.3                  (exact)

-- NEVER store money as float:
--   balance REAL   -> rounding errors accumulate, audits fail
--   balance NUMERIC(12,2) -> exact to the cent
```

**Strings.** `CHAR(n)` is fixed-length and *blank-padded* to `n`; `VARCHAR(n)` is variable-length with a max of `n`; `TEXT` is unbounded variable-length.

```sql
SELECT 'hi'::CHAR(5);      -- 'hi   '  (padded to 5)
SELECT length('hi'::CHAR(5));  -- in PostgreSQL, trailing blanks are trimmed on read -> 2
```

**Date/Time.** `DATE` = calendar day. `TIME` = time of day. `TIMESTAMP` = date + time *without* zone. `TIMESTAMPTZ` = an absolute instant, stored in UTC and converted to the session timezone on display.

```sql
SELECT CURRENT_DATE;                    -- 2026-07-09
SELECT NOW();                           -- 2026-07-09 14:32:10.123+05:30 (timestamptz)
SELECT NOW() + INTERVAL '7 days';       -- date/time arithmetic via INTERVAL
```

**Boolean.**

```sql
SELECT TRUE, FALSE, NULL::BOOLEAN;      -- three-valued logic
-- PostgreSQL accepts 'true'/'t'/'yes'/'1' and 'false'/'f'/'no'/'0' as literals
```

## Internal Working

- **Fixed-width numerics** (`SMALLINT` 2B, `INTEGER` 4B, `BIGINT` 8B, `REAL` 4B, `DOUBLE` 8B) are stored as raw machine values and compared/added with native CPU instructions — extremely fast.

- **`NUMERIC` is stored as a variable-length sequence of base-10000 "digits"** plus a sign, weight, and scale. It is exact but *software-implemented* arithmetic — noticeably slower and larger than hardware floats. That's the deliberate trade: exactness for speed.

- **IEEE 754 floats** store a sign, exponent, and mantissa in base-2. Because 0.1 has no finite base-2 representation (like 1/3 has no finite base-10 representation), floats are *approximate*. This is not a bug; it's the format. It's why float equality (`x = 0.3`) is unreliable and why money must never be float.

```
 NUMERIC(10,2) value 12345.67 :  [sign][weight][ digit groups: 1 2345 6700 ]  <- exact base-10000
 DOUBLE 12345.67              :  [sign|  exponent  |        mantissa (base-2)  ] <- nearest representable
```

- **`VARCHAR(n)`/`TEXT` in PostgreSQL share the same underlying `varlena` storage** — a 1- or 4-byte length header followed by the bytes. Crucially, in PostgreSQL `VARCHAR(n)`, `VARCHAR`, and `TEXT` have **essentially identical performance**; the `n` only adds a length check. (This is *not* true in all engines — in some, `CHAR`/`VARCHAR` behave very differently.) Long values are transparently compressed and/or moved to out-of-line **TOAST** storage.

- **`CHAR(n)` is blank-padded** to the full length on disk, so it wastes space for short values and is rarely worth using except for truly fixed-width codes.

- **Dates/timestamps are stored as integers**: PostgreSQL stores a timestamp as microseconds since 2000-01-01. `TIMESTAMPTZ` stores the same integer *normalized to UTC*; there is **no timezone stored in the column** — the zone is applied only on input/output using the session's `TimeZone` setting. `TIMESTAMP` (without tz) stores the wall-clock number you gave it with no conversion, which is why it's ambiguous across zones.

- **`BOOLEAN` occupies 1 byte** (not 1 bit) in PostgreSQL. MySQL has no true boolean — `BOOLEAN` is an alias for `TINYINT(1)`, storing 0/1.

## Advantages

- **Correctness:** the right type makes illegal values unstorable and gives operations the right semantics.
- **Space efficiency:** compact types shrink tables, indexes, RAM footprint, and backups.
- **Speed:** fixed-width numeric comparisons and arithmetic are single CPU instructions.
- **Exactness where it matters:** `NUMERIC` guarantees to-the-cent financial accuracy.
- **Proper sorting/ranges:** typed dates and numbers sort and range-filter correctly, unlike text look-alikes.
- **Self-documenting schema:** types communicate intent to every future reader.
- **Timezone correctness:** `TIMESTAMPTZ` handles global users without manual conversion.

## Limitations

- **`NUMERIC` is slower and larger** than floats — a real cost in heavy numeric/scientific workloads where tiny approximation is acceptable.
- **Floats cannot represent most decimals exactly** — disqualifying for money and any exact requirement.
- **`CHAR(n)` wastes space** via padding and rarely offers benefits over `VARCHAR`/`TEXT`.
- **Over-tight limits cause truncation or migrations:** picking `VARCHAR(20)` for something that grows forces painful `ALTER`s later.
- **`TIMESTAMP` without tz is dangerously ambiguous** across regions and DST changes.
- **Type differences across engines** (MySQL boolean, `NUMBER` in Oracle, integer sizes) hurt portability.
- **Under-sized integers overflow:** an `INTEGER` id column caps at ~2.1 billion — a real production outage cause.

## Real-world Applications

- **Money & finance:** `NUMERIC(19,4)` for balances, prices, tax — never float.
- **Identifiers:** `BIGINT` or `UUID` primary keys sized to never overflow at scale.
- **User content:** `TEXT` for comments/descriptions of unbounded length; `VARCHAR(320)` for emails.
- **Event timestamps:** `TIMESTAMPTZ DEFAULT NOW()` for `created_at`/`updated_at` in globally-distributed apps.
- **Scientific/analytics data:** `DOUBLE PRECISION` for sensor readings, coordinates, ML features where approximation is fine and speed matters.
- **Flags & states:** `BOOLEAN` for `is_active`, `is_deleted`, feature toggles.
- **Calendar data:** `DATE` for birthdays, invoice dates where time-of-day is irrelevant.

## Interview Questions

**Beginner**
1. What is the difference between `CHAR` and `VARCHAR`?
2. What is the difference between `INT` and `BIGINT`?
3. What does `NUMERIC(10,2)` mean?
4. What type would you use to store true/false?
5. What is the difference between `DATE` and `TIMESTAMP`?

**Intermediate**
6. Why should you never store money in a `FLOAT`/`REAL` column?
7. In PostgreSQL, is there a performance difference between `VARCHAR(n)` and `TEXT`?
8. What is the difference between `TIMESTAMP` and `TIMESTAMPTZ`?
9. When would you choose `SMALLINT` over `INTEGER`?
10. What is the difference between `DECIMAL`/`NUMERIC` and `FLOAT`/`DOUBLE PRECISION` internally?

**Advanced**
11. Explain how a `TIMESTAMPTZ` is actually stored and displayed. Is the timezone kept in the column?
12. Why does `0.1 + 0.2` not equal `0.3` in floating point?
13. What happens when an auto-increment `INTEGER` primary key reaches its maximum? How do you prevent it?
14. How does PostgreSQL physically store a `VARCHAR`, and what is TOAST?

**Scenario-based**
15. You're designing a payments table storing amounts up to 10 million with 4 decimal places for currency conversion. What type and why?
16. Users worldwide book appointments; you must store both "the exact instant" and "the local wall-clock time they chose." How do you model the time columns?

**"Why" questions**
17. Why does choosing a smaller integer type improve query performance, not just save disk?
18. Why is `TEXT` often preferred over `VARCHAR(n)` in PostgreSQL?

**Comparison questions**
19. `CHAR(n)` vs `VARCHAR(n)` vs `TEXT` — full comparison.
20. Exact numeric (`NUMERIC`) vs approximate numeric (`DOUBLE`) — when to use each?

## Model Answers

**1. CHAR vs VARCHAR.** `CHAR(n)` is a *fixed-length* type: every value is stored padded with trailing spaces to exactly `n` characters, so `'hi'` in a `CHAR(5)` occupies five characters (`'hi   '`). `VARCHAR(n)` is *variable-length* with an upper bound of `n`: `'hi'` occupies just two characters plus a small length header, and values up to `n` are allowed. The practical consequences are that `CHAR` wastes space for values shorter than `n` and can introduce surprising trailing-space behavior in comparisons, while `VARCHAR` stores only what you put in. `CHAR` is justified only for genuinely fixed-width codes (a 2-letter country code, a fixed-length hash), and even then most engineers reach for `VARCHAR`/`TEXT`. In PostgreSQL specifically there's no performance win from `CHAR`, so it's rarely used.

**2. INT vs BIGINT.** Both store exact whole numbers; they differ in *range and storage size*. `INTEGER` (`INT`) uses 4 bytes and spans roughly −2.1 billion to +2.1 billion. `BIGINT` uses 8 bytes and spans about ±9.2 quintillion. You pick based on the maximum value the column will ever need: `INT` is plenty for most counts and small ids, but for the primary key of a table that could exceed ~2 billion rows over its lifetime — or that burns through ids via deletes/gaps — `BIGINT` is the safe choice, because exhausting an `INT` key in production is a genuine, painful outage. The cost of `BIGINT` is 4 extra bytes per value, which matters when multiplied across billions of rows and their indexes, so it's a real (if usually worthwhile) trade-off for large-scale keys.

**3. NUMERIC(10,2).** This declares an *exact* decimal number with **precision 10** and **scale 2**: at most 10 significant digits in total, of which 2 are to the right of the decimal point. That means up to 8 digits before the point and 2 after — a maximum magnitude of 99,999,999.99. Values are stored and computed exactly in base-10, so there is no rounding drift, which is precisely why this shape is the go-to for money. If you try to insert a number with more than 2 decimal places it is rounded to scale; more than 10 total significant digits raises an error rather than silently corrupting the value.

**4. Storing true/false.** The proper type is `BOOLEAN`, which models a truth value and, in SQL, actually has *three* states: `TRUE`, `FALSE`, and `NULL` (unknown). In PostgreSQL `BOOLEAN` is a first-class type occupying one byte and accepting literals like `TRUE`/`'t'`/`'yes'`/`1`. A subtlety worth mentioning in an interview is that MySQL has no native boolean — `BOOLEAN`/`BOOL` is just an alias for `TINYINT(1)` storing 0 or 1 — so behavior and aggregation differ slightly across engines. Using a real boolean (rather than a `CHAR(1)` 'Y'/'N' or an integer flag) is best practice because it's self-documenting and integrates with SQL's logical operators.

**5. DATE vs TIMESTAMP.** A `DATE` stores only a calendar day — year, month, day — with no time component (e.g., `2026-07-09`), and is the right choice for birthdays, invoice dates, or anything where time-of-day is irrelevant. A `TIMESTAMP` stores both a date *and* a time-of-day down to sub-second precision (e.g., `2026-07-09 14:32:10.123`). The important follow-up is that plain `TIMESTAMP` carries *no timezone* information, so it represents an ambiguous wall-clock reading; for absolute moments in a global system you generally want `TIMESTAMPTZ` instead. So: `DATE` for "which day," `TIMESTAMP`/`TIMESTAMPTZ` for "which precise moment."

**6. Why not store money in FLOAT.** Because floating-point types are *approximate*: they represent numbers in base-2, and most decimal fractions (including 0.10, 0.20, 0.01) have no exact finite base-2 representation, so they're stored as the nearest representable value. Those tiny errors accumulate across additions and multiplications, so a running balance computed in float will eventually disagree with the exact figure by fractions of a cent — and in finance, "close" is unacceptable: audits fail, ledgers don't reconcile, and `WHERE balance = 100.00` may not match a row that should be exactly 100.00. The correct type is `NUMERIC`/`DECIMAL`, which stores and computes in exact base-10, guaranteeing to-the-cent accuracy. The rule is absolute: money is always exact numeric, never float.

**7. VARCHAR(n) vs TEXT performance in PostgreSQL.** There is *no meaningful performance difference*. In PostgreSQL, `CHAR`, `VARCHAR`, and `TEXT` are implemented over the same variable-length (`varlena`) storage machinery, and long values are handled identically via compression and TOAST. The only thing `VARCHAR(n)` adds is a length-limit check on write; it does not store or scan faster than `TEXT`. Consequently, many PostgreSQL practitioners use `TEXT` (or unbounded `VARCHAR`) by default and enforce any real length limits with a `CHECK` constraint, which is easier to change later than altering a type's declared length. Note this equivalence is a PostgreSQL characteristic — other databases may store and index `CHAR`/`VARCHAR`/`TEXT` quite differently.

**8. TIMESTAMP vs TIMESTAMPTZ.** `TIMESTAMP` (without time zone) stores the literal date-and-time you give it with *no* conversion and *no* recorded zone — it's a bare wall-clock reading, ambiguous unless you separately know which zone it refers to. `TIMESTAMPTZ` (with time zone) represents an *absolute instant*: on input, PostgreSQL converts the supplied value to UTC using the session's timezone and stores that UTC integer; on output, it converts back to the session's timezone for display. Critically, `TIMESTAMPTZ` does *not* store the original zone — it stores a normalized instant. For almost all "when did this happen" columns (`created_at`, event logs) in systems with users or servers in different zones, `TIMESTAMPTZ` is correct because it pins an unambiguous moment; plain `TIMESTAMP` should be reserved for cases where you truly mean a zone-less local wall-clock value.

**9. SMALLINT over INTEGER.** You choose `SMALLINT` (2 bytes, range ±32,767) when you're certain the values stay small and you're storing enormous numbers of rows, so the 2-byte-per-value saving versus `INTEGER` (4 bytes) meaningfully reduces table and index size, memory footprint, and I/O. Good candidates are small enumerations, ages, quantities, or status codes. The caution is that the saving is only worth it at large scale and only if you're confident about the bound — outgrowing a `SMALLINT` forces a type-widening migration, and for the sake of a couple of bytes on a small table it's usually not worth the risk. So: `SMALLINT` for provably-small columns on very large tables; `INTEGER` as the safe general default.

**10. NUMERIC vs FLOAT internally.** `NUMERIC`/`DECIMAL` stores numbers as an exact base-10 value — internally a sign plus a sequence of decimal digit groups plus a scale — and performs arithmetic in software, so results are *exact* to the declared scale but slower and larger. `FLOAT`/`DOUBLE PRECISION` uses the IEEE 754 binary format (sign, exponent, mantissa) processed directly by the CPU's floating-point unit, so it's *fast and compact* but *approximate*, since binary can't finitely represent most decimal fractions. The essence: `NUMERIC` trades speed and space for exactness; floats trade exactness for speed and space. Use `NUMERIC` when every digit must be right (money, quantities that must reconcile) and floats when you're modeling continuous physical quantities where a minuscule approximation is irrelevant and performance matters.

**11. How TIMESTAMPTZ is stored and displayed.** Despite the name, a `TIMESTAMPTZ` does *not* store a timezone in the column. Internally it stores a single integer — in PostgreSQL, microseconds since 2000-01-01 UTC — representing one absolute instant normalized to UTC. On *input*, the value you provide is converted to UTC using the session's `TimeZone` setting (or an explicit offset in the literal). On *output*, that UTC instant is converted back into the session's current timezone for display. So two clients in Kolkata and New York querying the same row see different wall-clock strings, yet both refer to the identical moment. This design is exactly why `TIMESTAMPTZ` is the right choice for recording events: it's unambiguous and timezone-agnostic in storage, with presentation handled per-session. If you genuinely need to remember the *original* zone the user was in, you store that separately (e.g., a companion `TIMESTAMP` plus a zone-name column).

**12. Why 0.1 + 0.2 != 0.3.** Floating-point numbers are stored in base-2, and 0.1, 0.2, and 0.3 are all *repeating* fractions in binary (just as 1/3 is 0.333... and never terminates in base-10). The hardware must round each to the nearest value representable in 53 bits of mantissa, so the stored 0.1 and 0.2 are each very slightly off; when added, their rounding errors combine to produce 0.30000000000000004, which differs from the separately-rounded stored value of 0.3. This isn't a database bug — it's inherent to IEEE 754 and identical in C, Java, Python, and JavaScript. The practical takeaways are: never test floats for exact equality (compare within a tolerance), and never use floats where exactness is required — use `NUMERIC`, whose base-10 storage represents 0.1, 0.2, 0.3 exactly, so the sum is precisely 0.3.

**13. INTEGER primary key overflow.** An auto-increment `INTEGER` key can generate values only up to 2,147,483,647; when the sequence tries to exceed that, the next insert fails with an integer-out-of-range error — and because it's the primary key, *all new inserts stop*, a serious production outage. Note this happens based on how many ids have been *generated*, not how many rows currently exist, so heavy insert/delete churn can exhaust it even below 2 billion live rows. Prevention: use `BIGINT` (or `UUID`) for keys on any table that could plausibly approach that scale — the extra 4 bytes are cheap insurance. Remediation on a live system is painful: you must widen the column type (a potentially long, locking operation on a huge table, often done with careful online-migration tooling) and widen every foreign key referencing it. The senior lesson is to default important keys to `BIGINT` from the start rather than fix this under fire.

**14. VARCHAR storage and TOAST in PostgreSQL.** PostgreSQL stores `VARCHAR`/`TEXT` as a `varlena` (variable-length) structure: a length header (1 byte for short values, 4 for longer) followed by the actual bytes, so only the real content plus a tiny overhead is stored — there's no padding. Rows live in fixed-size 8 KB pages, but a single value can't span pages, so when a row (or a field) is too large, PostgreSQL uses **TOAST** (The Oversized-Attribute Storage Technique): it transparently compresses large field values and, if still too big, breaks them into chunks stored in a separate associated "TOAST table," leaving just a pointer in the main row. This is entirely automatic and invisible to your SQL. Its benefit is that big text/JSON/bytea values don't bloat the main table's pages, keeping scans over the small columns fast, and TOASTed data is only fetched when actually referenced.

**15. Payments table type choice.** Amounts must be *exact* (it's money) and need up to 10 million with 4 decimal places (for currency-conversion precision), so the correct type is `NUMERIC`/`DECIMAL` with enough precision and a scale of 4 — for example `NUMERIC(19,4)`. I'd size the precision generously (19 total digits is a common financial standard, comfortably covering millions with 4 decimals plus headroom for multiplication during conversions) so intermediate calculations don't overflow. I would *never* use `FLOAT`/`DOUBLE` here because base-2 approximation would introduce sub-cent drift that breaks reconciliation and audits. I'd also add `CHECK (amount >= 0)` if negatives are illegal, and store the currency code alongside so the scale is interpreted correctly. Scale 4 (rather than 2) is chosen deliberately because FX conversion needs more decimal places than final display, and you round to 2 only at presentation.

**16. Modeling appointment times worldwide.** These are two genuinely different pieces of information, so they need two columns. For "the exact instant" the appointment occurs, use `TIMESTAMPTZ` — it stores an unambiguous UTC moment, so reminders, ordering, and overlap checks work correctly regardless of where the server or other users are. For "the local wall-clock time the user chose" plus the ability to reproduce it and survive DST rule changes, store the user's *IANA timezone name* (e.g., `'Asia/Kolkata'`) in a separate `TEXT`/`VARCHAR` column, and optionally the local `TIMESTAMP` (without tz) they picked. From instant + zone name you can always recompute the local presentation, and keeping the zone *name* (not a fixed offset) is important because offsets change with daylight-saving rules. The anti-pattern is storing only a plain `TIMESTAMP` and hoping everyone shares a zone, or storing only a UTC instant and losing the user's intended local context.

**17. Why smaller integers speed up queries.** Beyond saving disk, a smaller type makes *every layer* faster. Databases read and cache data in fixed-size pages; narrower columns mean each row is smaller, so *more rows fit per page*, so a scan of N rows touches fewer pages — fewer disk reads and fewer buffer-cache pages, i.e., less I/O, the usual bottleneck. Indexes on smaller types are physically smaller, so their B-trees are shallower and traverse in fewer page reads, and more of the index fits in RAM. Comparisons and arithmetic on narrow fixed-width types are cheap CPU operations, and smaller rows improve cache locality. So the space saving compounds into I/O, memory, and CPU savings — which is why type sizing is a performance decision, not merely a storage one, at scale.

**18. Why TEXT is often preferred over VARCHAR(n) in PostgreSQL.** Because in PostgreSQL they perform identically (same underlying storage, same TOAST handling), the `n` in `VARCHAR(n)` buys you only a length check — and that length limit is frequently an *arbitrary guess* that later proves too small, forcing a schema migration to widen it. By using `TEXT` (or unbounded `VARCHAR`) and, where a real limit exists, expressing it as a `CHECK (length(col) <= n)` constraint, you get the same validation but can change the limit far more easily and expressively than altering the column type. There's no storage or speed penalty, so the default recommendation in the PostgreSQL community is: reach for `TEXT` unless a specific external requirement dictates a hard `VARCHAR(n)`. (In other engines that optimize fixed limits, this advice may not hold.)

**19. CHAR(n) vs VARCHAR(n) vs TEXT.**

| Aspect | `CHAR(n)` | `VARCHAR(n)` | `TEXT` |
|---|---|---|---|
| Length | Fixed, blank-padded to `n` | Variable, max `n` | Variable, unbounded |
| Storage | Always `n` chars | Actual length + header | Actual length + header |
| Best for | Truly fixed-width codes | Bounded text with a real limit | Free-form / unbounded text |
| PostgreSQL perf | No advantage | Same as TEXT | Baseline |

`CHAR(n)` fixes every value at `n` characters via space-padding, wasting space for shorter values and occasionally causing trailing-space comparison surprises; it's justified only for genuinely fixed-length codes. `VARCHAR(n)` stores only the actual bytes but caps length at `n`, useful when a real business limit exists (e.g., a 2-char state code). `TEXT` is unbounded variable-length for free-form content. In PostgreSQL all three share the same varlena storage and TOAST, so there's no performance reason to prefer `CHAR`/`VARCHAR` over `TEXT`; choose based on whether you need a length limit, and prefer `TEXT` + a `CHECK` when you want flexibility. (Note: in some other databases, `CHAR` and `VARCHAR` have very different storage/index behavior, so this equivalence is PostgreSQL-specific.)

**20. Exact NUMERIC vs approximate DOUBLE.** Use *exact* `NUMERIC`/`DECIMAL` whenever every digit must be correct and reproducible: money, tax, quantities that must reconcile, anything audited or compared for equality — you accept slower, larger storage in exchange for guaranteed precision. Use *approximate* `DOUBLE PRECISION`/`REAL` for continuous, measured, or scientific quantities — sensor readings, geographic coordinates, physics simulations, ML features, statistical aggregates — where the values are inherently imprecise anyway, a rounding error in the 15th digit is irrelevant, and you want maximum arithmetic speed and compactness. The decisive questions are: "Must this be exact and equality-comparable?" (→ `NUMERIC`) versus "Is this a continuous measurement where speed matters and tiny approximation is fine?" (→ `DOUBLE`). Misapplying the first rule to money is the classic, costly mistake.

## Common Mistakes

- **Storing money in `FLOAT`/`REAL`**, causing rounding drift and failed reconciliation.
- **Using `INTEGER` for a high-growth primary key** and later hitting the ~2.1 billion overflow wall.
- **Using plain `TIMESTAMP` instead of `TIMESTAMPTZ`** for event times, creating timezone ambiguity and DST bugs.
- **Comparing floats for exact equality** (`WHERE x = 0.3`) and getting inconsistent matches.
- **Over-constraining `VARCHAR(n)`** with an arbitrary small limit, forcing later migrations.
- **Overusing `CHAR(n)`**, wasting space on padding for no benefit.
- **Storing numbers, dates, or booleans as text**, breaking sorting, range queries, and arithmetic.
- **Storing a fixed UTC offset instead of an IANA zone name**, which breaks when DST rules change.
- **Assuming MySQL `BOOLEAN` is a real boolean** — it's `TINYINT(1)`.

## Related Concepts

- **Type casting & coercion** (`CAST`, `::`) — converting between types and the pitfalls of implicit casts.
- **IEEE 754 floating point** — the standard behind approximate numerics.
- **TOAST / out-of-line storage** — how large values are stored in PostgreSQL.
- **Constraints (`CHECK`, `NOT NULL`)** — layered on top of types to further restrict domains.
- **Collations & character encodings (UTF-8)** — how text is compared and sorted.
- **`INTERVAL` and date/time arithmetic** — operating on temporal types.
- **Sequences / IDENTITY / UUID** — strategies for generating key values and avoiding overflow.
- **`ENUM`, `JSONB`, `ARRAY`** — richer PostgreSQL types beyond the SQL-standard core.

---

# CRUD — CREATE / INSERT

## What is it?

The **C** in CRUD stands for **Create**, and in SQL the primary tool for creating rows of data is the `INSERT` statement. `INSERT` is a **Data Manipulation Language (DML)** command whose sole job is to add new rows (tuples) into an existing table.

It's important to separate two ideas that beginners often conflate:

- `CREATE TABLE` is a **DDL (Data Definition Language)** command — it creates the *structure* (the schema, the empty container).
- `INSERT INTO` is a **DML** command — it puts *data* into that container.

When people say "CRUD — Create", in the day-to-day operational sense they almost always mean `INSERT`. That is what this topic covers.

```sql
INSERT INTO employees (first_name, last_name, salary)
VALUES ('Ada', 'Lovelace', 95000);
```

## Why is it needed?

A database with no rows is like a spreadsheet with headers but no content — structurally valid but useless. Every application that persists state needs a way to *materialize* new facts about the world: a new user signs up, an order is placed, a sensor emits a reading. `INSERT` is the doorway through which all data enters a relational system.

From first principles, ask: **why not just append to a file?** Because a relational table gives you three guarantees that a raw file does not:

1. **Schema enforcement** — the row must conform to declared column types and constraints (`NOT NULL`, `CHECK`, `FOREIGN KEY`).
2. **Atomicity** — an `INSERT` either fully happens or does not (ACID); no half-written rows.
3. **Concurrency safety** — many clients can insert simultaneously and the engine serializes access to shared structures (indexes, sequences).

`INSERT` is the command that hooks new data into all of that machinery in one shot.

## How does it work?

There are four patterns you must have at your fingertips.

**1. Single-row insert (explicit columns — always prefer this):**

```sql
INSERT INTO employees (first_name, last_name, salary, department_id)
VALUES ('Grace', 'Hopper', 105000, 3);
```

Listing columns explicitly is the professional default. It survives schema changes (adding a new nullable column won't break the statement) and makes intent unambiguous.

**2. Single-row insert (implicit / positional — avoid in production code):**

```sql
-- Fragile: depends on physical column order
INSERT INTO employees
VALUES (DEFAULT, 'Grace', 'Hopper', 105000, 3);
```

If someone reorders columns or adds one, this silently breaks or misassigns values.

**3. Multiple rows in one statement (multi-row VALUES):**

```sql
INSERT INTO employees (first_name, last_name, salary)
VALUES
  ('Alan',   'Turing',  110000),
  ('Edsger', 'Dijkstra', 98000),
  ('Donald', 'Knuth',   120000);
```

This is dramatically faster than three separate statements — one parse, one plan, one round trip, and typically one transaction/one WAL flush.

**4. `INSERT ... SELECT` (insert the result of a query):**

```sql
-- Archive high earners into a separate table
INSERT INTO high_earners (first_name, last_name, salary)
SELECT first_name, last_name, salary
FROM employees
WHERE salary > 100000;
```

This copies/derives data from existing tables. No `VALUES` clause is used — the rows come from the `SELECT`. It's the backbone of ETL, table copies, and derived/aggregated tables.

**5. `RETURNING` (PostgreSQL) — get back what was inserted:**

```sql
INSERT INTO employees (first_name, last_name, salary)
VALUES ('Katherine', 'Johnson', 99000)
RETURNING id, created_at;
```

`RETURNING` hands you server-generated values (auto-increment IDs, `DEFAULT` timestamps, computed columns) in the *same round trip*, avoiding a follow-up `SELECT`.

> **MySQL note:** MySQL does **not** support `RETURNING` (as of 8.0). Instead you use `LAST_INSERT_ID()` after the insert to fetch the last auto-increment value. Postgres and MariaDB (10.5+) support `RETURNING`.

## Internal Working

Understanding what the engine *does* on an `INSERT` demystifies performance and locking behavior.

```
  INSERT statement
        |
        v
  [1] Parse & plan  -> validate syntax, resolve column list
        |
        v
  [2] Constraint check -> types, NOT NULL, CHECK, UNIQUE, FK lookups
        |
        v
  [3] Sequence / default resolution -> nextval() for SERIAL, now() for defaults
        |
        v
  [4] Write row to a heap page (Postgres) / clustered index (InnoDB)
        |
        v
  [5] Update every index on the table (B-tree insert per index)
        |
        v
  [6] Write WAL / redo log record  -> durability
        |
        v
  [7] COMMIT -> flush WAL, release locks, make visible (MVCC)
```

Key internals to understand:

- **Indexes cost you on write.** Every secondary index must be updated per inserted row. A table with 6 indexes does roughly 6x the index work per insert. This is why bulk-loading pipelines often *drop indexes, load, then rebuild*.
- **Sequences are non-transactional.** In Postgres, `nextval()` on a `SERIAL`/`IDENTITY` column increments even if the transaction later rolls back — that's why auto-increment IDs have gaps. This is intentional: making sequences transactional would serialize all inserts.
- **MVCC append.** In Postgres an insert writes a brand-new tuple version stamped with the transaction's `xmin`; it becomes visible to others only at commit. No in-place overwrite happens.
- **Multi-row batching wins** because steps [1] and [6] are amortized: one parse, and WAL can be flushed once for the whole batch instead of per row.

## Advantages

- **Set-based multi-row inserts** are far more efficient than row-by-row loops — fewer round trips, fewer WAL flushes, one plan.
- **`INSERT ... SELECT`** lets you move/transform data server-side without shipping it to the client and back.
- **`RETURNING`** eliminates the classic "insert then select the ID" race and round trip.
- **Constraint enforcement at insert time** guarantees data integrity — bad data is rejected at the door, not discovered later.
- **`DEFAULT` and generated columns** let the database fill in values (timestamps, UUIDs, computed columns), keeping application code simpler.

## Limitations

- **Write amplification from indexes** — heavily indexed tables insert slowly.
- **Bulk inserts via `INSERT` are still slower** than dedicated bulk loaders (`COPY` in Postgres, `LOAD DATA INFILE` in MySQL) for millions of rows.
- **Sequence gaps** — auto-increment IDs are not gap-free; never assume contiguity.
- **Positional (column-less) inserts are brittle** and a frequent source of production bugs after schema changes.
- **Very large multi-row `VALUES`** lists can hit parameter/packet limits (e.g., Postgres 65535 bind parameters, MySQL `max_allowed_packet`).
- **`RETURNING` portability** — not available in MySQL, so cross-database code can't rely on it.

## Real-world Applications

- **User registration / sign-up flows** — one `INSERT ... RETURNING id` to create the account and immediately get its ID for the session.
- **Order placement** — insert an order header, get its ID via `RETURNING`, then batch-insert line items.
- **ETL / data warehousing** — `INSERT ... SELECT` to populate staging, dimension, and fact tables from raw sources.
- **Audit logging** — triggers `INSERT` a row into an audit table on every change.
- **Seeding / migrations** — multi-row `INSERT` statements to populate lookup/reference tables (countries, statuses, roles).
- **Materialized snapshots** — `INSERT ... SELECT` to periodically capture aggregated reporting tables.

## Interview Questions

**Beginner**
1. What is the difference between `CREATE TABLE` and `INSERT INTO`?
2. Write an `INSERT` statement that adds one row to a `products` table.

**Intermediate**
3. How do you insert multiple rows in a single statement, and why is it better than multiple single-row inserts?
4. What does `INSERT ... SELECT` do and when would you use it?

**Advanced**
5. Explain why auto-increment / `SERIAL` IDs can have gaps.
6. How does the number of indexes on a table affect insert performance, and what do bulk-load pipelines do about it?

**Scenario-based**
7. You need to create an order and its 20 line items and return the new order ID to the app. How do you structure the inserts?

**"Why" questions**
8. Why should you always list column names explicitly in an `INSERT`?
9. Why are sequences non-transactional in PostgreSQL?

**Comparison questions**
10. Compare `RETURNING` (Postgres) vs `LAST_INSERT_ID()` (MySQL) for retrieving generated keys.

## Model Answers

**1. `CREATE TABLE` vs `INSERT INTO`.**
`CREATE TABLE` is a DDL statement that defines *structure* — it creates an empty table with named, typed columns and constraints. It runs once when you design the schema. `INSERT INTO` is a DML statement that adds *data* — actual rows — into that already-existing structure, and it runs continuously during the life of the application. A useful mental model: `CREATE TABLE` builds the filing cabinet and labels the drawers; `INSERT` puts documents into the drawers. They also differ in transactional/locking behavior and in permissions — a role may be allowed to `INSERT` data but not to `CREATE`/alter tables.

**2. Simple insert.**
```sql
INSERT INTO products (name, price, in_stock)
VALUES ('Mechanical Keyboard', 79.99, true);
```
Note I listed the columns explicitly and omitted `id` (assuming it's a `SERIAL`/`IDENTITY` that the database fills in). Any column not listed takes its `DEFAULT`, or `NULL` if no default and the column is nullable.

**3. Multi-row insert.**
```sql
INSERT INTO products (name, price) VALUES
  ('Mouse', 25.00),
  ('Mat',    9.00),
  ('Cable',  4.50);
```
It's better than three separate statements because the server parses and plans once instead of three times, the client makes one network round trip instead of three, and — critically — the engine can group the writes into a single WAL/redo flush and a single transaction. Under load, batching inserts is one of the highest-leverage performance wins available, often 10x or more versus a naive per-row loop.

**4. `INSERT ... SELECT`.**
It inserts rows produced by a `SELECT` query rather than by a literal `VALUES` list. You use it whenever the source data already lives in the database: copying a table, archiving rows that match a condition, denormalizing/aggregating into a reporting table, or moving data between staging and production tables in ETL. Because the whole operation runs inside the server, no data is shipped to the client and back — it's both faster and atomic.
```sql
INSERT INTO orders_archive
SELECT * FROM orders WHERE created_at < '2024-01-01';
```

**5. Why gaps in auto-increment IDs.**
Sequences (Postgres `SERIAL`/`IDENTITY`, MySQL `AUTO_INCREMENT`) are deliberately *non-transactional*. When a transaction calls `nextval()`, the counter advances immediately and is **not** rolled back if the transaction aborts. If it were rolled back, every concurrent insert would have to wait for others to commit before knowing the next value — serializing all inserts and destroying throughput. So the engine trades gap-free numbering for concurrency. Failed transactions, rolled-back inserts, `ON CONFLICT` skips, and cached sequence blocks all produce gaps. The correct takeaway: IDs are unique and monotonically increasing, but never assume they're contiguous, and never derive business meaning (like "we have N rows") from the max ID.

**6. Indexes and insert performance.**
Every index on a table is a separate data structure (usually a B-tree) that must be kept consistent. When you insert a row, the engine writes the row to the heap/clustered index *and then* inserts a corresponding entry into every secondary index — each is an independent B-tree insertion that may trigger page splits and its own WAL records. So a table with six indexes does on the order of six times the index maintenance per row compared to an unindexed table. This is why high-volume bulk-load pipelines commonly drop non-essential indexes, load the data, and rebuild the indexes at the end — building an index once over sorted data is far cheaper than maintaining it incrementally through millions of random inserts.

**7. Order + line items scenario.**
Do it in one transaction. Insert the order header first with `RETURNING` to capture the generated ID, then batch-insert all line items in a single multi-row insert referencing that ID:
```sql
BEGIN;
INSERT INTO orders (customer_id, status)
VALUES (42, 'pending')
RETURNING id;   -- say it returns 1001

INSERT INTO order_items (order_id, product_id, qty) VALUES
  (1001, 7, 2),
  (1001, 9, 1),
  ... ;
COMMIT;
```
Wrapping both in a transaction guarantees you never end up with an order that has no items or items with no order. In MySQL you'd replace `RETURNING` with a read of `LAST_INSERT_ID()`. In application code, many ORMs and drivers let you use a CTE (`WITH ins AS (INSERT ... RETURNING id) ...`) to do it in a single statement.

**8. Why list columns explicitly.**
Explicit column lists decouple your statement from the table's physical column order. If a colleague adds a new nullable column or reorders columns, a column-less `INSERT ... VALUES (...)` will either break loudly or — worse — silently insert values into the wrong columns. Explicit lists are self-documenting, make code reviews easier, and are the only safe choice in migrations and long-lived application code. The tiny extra typing is cheap insurance against a whole category of data-corruption bugs.

**9. Why sequences are non-transactional.**
Because gap-free, transactional numbering is fundamentally at odds with concurrency. If sequence values rolled back with their transactions, the database would have to hold the "next value" decision until commit, forcing every inserting transaction to queue behind every other — a global serialization point. By making `nextval()` advance immediately and irreversibly, Postgres lets thousands of transactions grab distinct IDs in parallel without blocking. The cost is gaps; the benefit is scalability. This is a deliberate, principled engineering trade-off, not a bug.

**10. `RETURNING` vs `LAST_INSERT_ID()`.**
`RETURNING` (Postgres, MariaDB 10.5+) is a clause appended to `INSERT`/`UPDATE`/`DELETE` that returns *arbitrary columns from the affected rows* — including multiple rows and computed/default values — in the same statement and round trip. `LAST_INSERT_ID()` (MySQL) is a session-scoped function returning only the single most recent auto-increment value for the current connection. `RETURNING` is strictly more powerful: it works for multi-row inserts, can return any column (not just the auto-increment key), and is race-free by construction. `LAST_INSERT_ID()` is limited to one generated key and requires care with multi-row inserts (it returns the *first* generated ID of the batch in MySQL). If you need the generated ID of one insert on one connection, both work; for anything richer, `RETURNING` wins.

## Common Mistakes

- **Column-less inserts in application code** — `INSERT INTO t VALUES (...)` breaks silently after schema changes. Always list columns.
- **Looping single-row inserts** in application code instead of batching — orders of magnitude slower.
- **Assuming contiguous IDs** — building logic on "no gaps" in a serial column.
- **Forgetting a transaction** around related inserts (order + items), risking orphaned/partial data.
- **Inserting into a `SERIAL` column manually** and then hitting duplicate-key errors later because the sequence wasn't advanced (`setval` needed after manual inserts).
- **Giant single `VALUES` list** exceeding bind-parameter or packet limits — chunk large batches (e.g., 1,000 rows at a time).
- **Relying on `RETURNING` in MySQL** — it doesn't exist there.

## Related Concepts

- **`COPY` (Postgres) / `LOAD DATA INFILE` (MySQL)** — high-speed bulk loading, far faster than `INSERT` for large volumes.
- **`ON CONFLICT` / `UPSERT`** — `INSERT ... ON CONFLICT DO UPDATE` (Postgres) and `INSERT ... ON DUPLICATE KEY UPDATE` (MySQL).
- **Sequences / `IDENTITY` / `AUTO_INCREMENT`** — generated key mechanisms.
- **Transactions & ACID** — the atomicity guarantee behind multi-statement inserts.
- **MVCC** — how Postgres makes new rows visible only on commit.
- **Constraints** — `NOT NULL`, `CHECK`, `UNIQUE`, `FOREIGN KEY` enforced at insert time.
- **CTEs with `RETURNING`** — writable CTEs to chain inserts in one statement.

# CRUD — READ / SELECT

## What is it?

The **R** in CRUD is **Read**, and the `SELECT` statement is how you read data out of a relational database. `SELECT` is the most-used, most-optimized, and most-nuanced statement in all of SQL. Every reporting dashboard, API response, search box, and analytics query ultimately bottoms out in a `SELECT`.

A `SELECT` is a *declarative* description of the result you want — you describe **what** rows and columns you need, not **how** to fetch them. The engine's query planner decides the "how".

The clauses, in the order you *write* them:

```sql
SELECT   [DISTINCT] columns
FROM     table(s)
WHERE    row_filter
GROUP BY grouping
HAVING   group_filter
ORDER BY sort
LIMIT    n OFFSET m;
```

This topic focuses on the single-table read essentials: projecting columns, `DISTINCT`, aliases, **filtering** (`WHERE` and its operators), **sorting** (`ORDER BY`), and **limiting/pagination** (`LIMIT`/`OFFSET`).

## Why is it needed?

Storing data is pointless if you can't retrieve exactly the slice you need. Applications almost never want "all the data" — they want *this user's unread messages, newest first, 20 at a time*. `SELECT` with `WHERE`, `ORDER BY`, and `LIMIT` is the language for expressing precisely that.

From first principles, the value of a declarative `SELECT` is **separation of intent from execution**. You state the result set; the optimizer chooses indexes, join algorithms, and scan strategies. The same query can run against a 100-row table (sequential scan) or a 100-million-row table (index scan) without you rewriting it. That's the power of the relational model: physical data independence.

## How does it work?

**Projecting columns.**

```sql
SELECT * FROM employees;                       -- all columns (avoid in prod)
SELECT first_name, last_name FROM employees;   -- specific columns (preferred)
```

Prefer explicit column lists: `SELECT *` fetches unneeded data, breaks when columns change, and prevents some index-only scans.

**`DISTINCT` — remove duplicate rows:**

```sql
SELECT DISTINCT department_id FROM employees;
SELECT DISTINCT department_id, job_title FROM employees;  -- distinct on the combo
```

`DISTINCT` dedups on *the entire selected row*, not just the first column.

**`AS` aliases — rename columns/expressions and tables:**

```sql
SELECT first_name AS fname,
       salary * 12 AS annual_salary
FROM employees AS e;
```

Aliases rename output columns (great for computed expressions) and give tables short handles. The `AS` keyword is optional but improves readability.

**Filtering with `WHERE`.**

```sql
SELECT * FROM employees WHERE salary > 90000;
```

Comparison operators: `=`, `<>` (or `!=`), `<`, `<=`, `>`, `>=`.
Logical operators: `AND`, `OR`, `NOT` — mind precedence (`NOT` > `AND` > `OR`); use parentheses.

```sql
SELECT * FROM employees
WHERE department_id = 3
  AND (salary > 100000 OR job_title = 'Manager');
```

**`BETWEEN` — inclusive range:**

```sql
SELECT * FROM employees WHERE salary BETWEEN 80000 AND 100000;
-- equivalent to: salary >= 80000 AND salary <= 100000
```

**`IN` — membership in a set:**

```sql
SELECT * FROM employees WHERE department_id IN (1, 3, 5);
```

**`LIKE` / `ILIKE` — pattern matching:**

```sql
SELECT * FROM employees WHERE last_name LIKE 'S%';    -- starts with S
SELECT * FROM employees WHERE email LIKE '%@corp.com';-- ends with
SELECT * FROM employees WHERE first_name LIKE '_a%';  -- 2nd char is 'a'
SELECT * FROM employees WHERE last_name ILIKE 's%';   -- case-insensitive (Postgres)
```

`%` matches any sequence of characters; `_` matches exactly one. `ILIKE` is Postgres's case-insensitive `LIKE`. In MySQL, `LIKE` is case-insensitive by default for typical collations, so there's no separate `ILIKE`.

**`IS NULL` / `IS NOT NULL` — the only correct way to test for NULL:**

```sql
SELECT * FROM employees WHERE manager_id IS NULL;
SELECT * FROM employees WHERE manager_id IS NOT NULL;
```

You **cannot** use `= NULL` — see the three-valued logic discussion below.

**Sorting with `ORDER BY`.**

```sql
SELECT * FROM employees ORDER BY salary DESC;
SELECT * FROM employees ORDER BY department_id ASC, salary DESC;  -- multi-key
```

`ASC` (default) is ascending; `DESC` is descending. You can sort by multiple keys — ties on the first are broken by the second, and so on.

**NULL ordering:**

```sql
SELECT * FROM employees ORDER BY commission DESC NULLS LAST;
```

In Postgres, NULLs sort as *larger* than any value by default, so `ASC` puts them last and `DESC` puts them first. Use `NULLS FIRST` / `NULLS LAST` to control this explicitly. (MySQL treats NULLs as smallest and lacks `NULLS FIRST/LAST`; you emulate with `ORDER BY col IS NULL, col`.)

**Limiting & pagination — `LIMIT` / `OFFSET`.**

```sql
SELECT * FROM employees ORDER BY id LIMIT 20;             -- first 20
SELECT * FROM employees ORDER BY id LIMIT 20 OFFSET 40;   -- rows 41–60 (page 3)
```

`LIMIT n` caps the row count; `OFFSET m` skips the first `m` rows. **A `LIMIT` without `ORDER BY` returns an arbitrary, non-deterministic set** — always pair them.

## Internal Working

The clauses are *written* in one order but *evaluated* by the engine in a different **logical order**. Understanding this explains dozens of "why can't I…" questions:

```
Logical evaluation order:
  1. FROM      -> pick the source table(s), form the row source
  2. WHERE     -> filter individual rows (runs BEFORE grouping)
  3. GROUP BY  -> collapse rows into groups
  4. HAVING    -> filter groups (runs AFTER grouping)
  5. SELECT    -> compute expressions, apply aliases, DISTINCT
  6. ORDER BY  -> sort the final rows
  7. LIMIT/OFFSET -> slice the sorted result
```

Consequences that trip people up:

- **You can't reference a `SELECT` alias in `WHERE`**, because `WHERE` (step 2) runs before `SELECT` (step 5) computes the alias. You *can* reference it in `ORDER BY` (step 6, which runs after `SELECT`).
- **`WHERE` filters rows; `HAVING` filters groups.** `WHERE salary > 100000` removes individual employees before grouping; `HAVING AVG(salary) > 100000` removes whole departments after grouping. Put a condition in `WHERE` when it doesn't involve an aggregate — it's cheaper because fewer rows reach the grouping step.

**How the planner executes filtering & sorting:**

- `WHERE col = x` on an indexed column → the planner may do an **index scan** (walk the B-tree to matching rows) instead of a **sequential scan** (read every row). This is the single biggest performance lever.
- `LIKE 'abc%'` (prefix, anchored left) **can** use a B-tree index; `LIKE '%abc'` (leading wildcard) **cannot** — it forces a full scan or requires a trigram/GIN index.
- `ORDER BY` may be satisfied "for free" if an index already stores rows in that order; otherwise the engine performs an explicit **sort** (in memory, or spilling to disk for large sets).

**The three-valued logic (3VL) — the deepest idea in this topic.**

SQL's `NULL` means "unknown", and comparisons with unknown yield **UNKNOWN**, a third truth value beside TRUE and FALSE:

```
   x = NULL      -> UNKNOWN   (NOT FALSE — this is why = NULL never matches)
   NULL = NULL   -> UNKNOWN
   5 > NULL      -> UNKNOWN
   NULL IS NULL  -> TRUE      (IS NULL is a special predicate, not a comparison)
```

`WHERE` keeps a row **only when the predicate is TRUE** — UNKNOWN rows are dropped, just like FALSE. That's why `WHERE manager_id = NULL` returns *nothing*, and you must write `IS NULL`.

Truth tables:

```
 AND | T | F | U        OR  | T | F | U        NOT
 ----+---+---+---        ----+---+---+---       ----
  T  | T | F | U          T  | T | T | T        T -> F
  F  | F | F | F          F  | T | F | U        F -> T
  U  | U | F | U          U  | T | U | U        U -> U
```

A subtle trap: `WHERE col NOT IN (1, 2, NULL)` can return **no rows**, because `col <> NULL` is UNKNOWN, so the whole `NOT IN` can never be TRUE. This bites people constantly with `NOT IN` over subqueries that contain NULLs.

## Advantages

- **Declarative** — describe the result; the optimizer finds an efficient plan.
- **Composable** — filtering, sorting, and limiting combine cleanly and can be layered with joins, subqueries, and CTEs.
- **Index-accelerated** — `WHERE`/`ORDER BY` on indexed columns can turn full scans into fast lookups.
- **Physical data independence** — the same query works as tables grow and indexes change.
- **Expressive predicates** — `BETWEEN`, `IN`, `LIKE`, ranges, and boolean logic cover most real filtering needs concisely.

## Limitations

- **`SELECT *` is a code smell** in production — over-fetches, breaks on schema change, defeats index-only scans.
- **`OFFSET` pagination degrades on deep pages** — `OFFSET 1000000` still scans and discards a million rows. Keyset/seek pagination is preferred at scale.
- **Leading-wildcard `LIKE '%x'`** can't use a normal B-tree index — full scan.
- **NULL/3VL surprises** — `= NULL`, `NOT IN` with NULLs, and NULL sort order routinely cause bugs.
- **`LIMIT` without `ORDER BY`** is non-deterministic — no stable ordering guarantee.
- **`DISTINCT` can be expensive** — it forces a sort or hash to dedupe; often a sign a join fan-out should be fixed instead.

## Real-world Applications

- **API list endpoints** — `WHERE tenant_id = ? ORDER BY created_at DESC LIMIT 20 OFFSET ?` for paginated feeds.
- **Search boxes** — `WHERE name ILIKE '%term%'` (or full-text/trigram indexes for scale).
- **Dashboards & reports** — filtered, sorted extracts of transactional data.
- **Deduplication checks** — `SELECT DISTINCT` or `SELECT ..., COUNT(*)` to find duplicates.
- **Data validation** — `WHERE required_col IS NULL` to find records missing mandatory fields.
- **Top-N queries** — `ORDER BY score DESC LIMIT 10` for leaderboards and "most recent" widgets.

## Interview Questions

**Beginner**
1. What's the difference between `SELECT *` and selecting specific columns?
2. How do you filter rows where salary is greater than 50,000?
3. What does `DISTINCT` do?

**Intermediate**
4. Explain `BETWEEN`, `IN`, and `LIKE` with examples.
5. How does `ORDER BY` handle multiple columns and NULLs?
6. How do you implement pagination with `LIMIT` and `OFFSET`?

**Advanced**
7. Why can't you reference a `SELECT` alias in the `WHERE` clause but you can in `ORDER BY`?
8. Explain SQL's three-valued logic and why `WHERE col = NULL` returns nothing.

**Scenario-based**
9. A `NOT IN (subquery)` suddenly returns zero rows in production. What's the likely cause and how do you fix it?

**"Why" questions**
10. Why does deep `OFFSET` pagination get slow, and what's the alternative?
11. Why can `LIKE 'abc%'` use an index but `LIKE '%abc'` cannot?

**Comparison questions**
12. `WHERE` vs `HAVING` — what's the difference and when do you use each?
13. `LIKE` vs `ILIKE` (and MySQL's default behavior).

## Model Answers

**1. `SELECT *` vs specific columns.**
`SELECT *` returns every column of every row. It's convenient for ad-hoc exploration but poor for production: it transfers columns you don't need (wasting network and memory), it silently changes shape when someone alters the table (breaking downstream code that assumes a column order/count), and it can prevent *index-only scans* — where the database answers a query entirely from an index without touching the table — because the index rarely covers every column. Selecting exactly the columns you need is faster, more stable, and self-documenting.

**2. Filtering by salary.**
```sql
SELECT * FROM employees WHERE salary > 50000;
```
`WHERE` evaluates the predicate per row and keeps rows where it is TRUE. If `salary` is indexed and the condition is selective, the planner can use an index range scan instead of reading the whole table.

**3. `DISTINCT`.**
`DISTINCT` removes duplicate rows from the result, deduplicating on the *entire* set of selected columns — not just the first. So `SELECT DISTINCT dept, title` returns each unique (dept, title) *pair*. Internally the engine must sort or hash the rows to find duplicates, which has a cost; if you're reaching for `DISTINCT` to paper over duplicates created by a join, it's usually better to fix the join.

**4. `BETWEEN`, `IN`, `LIKE`.**
`BETWEEN a AND b` is an inclusive range test — `salary BETWEEN 80000 AND 100000` is shorthand for `salary >= 80000 AND salary <= 100000` (both endpoints included). `IN (v1, v2, ...)` tests set membership — `dept IN (1,3,5)` is a compact form of `dept = 1 OR dept = 3 OR dept = 5`. `LIKE` does pattern matching where `%` matches any run of characters and `_` matches exactly one: `LIKE 'S%'` = starts with S, `LIKE '%son'` = ends with son, `LIKE '_a%'` = second character is 'a'. A key caveat: `BETWEEN` and `IN` behave surprisingly around NULL, and `IN`/`NOT IN` with a NULL in the list can yield UNKNOWN.

**5. `ORDER BY` with multiple columns and NULLs.**
With multiple sort keys, the engine sorts by the first key; rows that tie on it are then ordered by the second key, and so on — like sorting a spreadsheet by column A then column B. Each key has its own direction: `ORDER BY dept ASC, salary DESC`. For NULLs, Postgres treats NULL as larger than any non-NULL, so ascending order puts NULLs last and descending puts them first; you override with `NULLS FIRST`/`NULLS LAST`. MySQL treats NULL as smallest and has no `NULLS FIRST/LAST` clause, so you emulate it with an expression like `ORDER BY (col IS NULL), col`.

**6. Pagination with `LIMIT`/`OFFSET`.**
`LIMIT n OFFSET m` returns `n` rows after skipping the first `m`, so page `p` (1-based, page size `s`) is `LIMIT s OFFSET (p-1)*s`. Crucially it must be paired with a deterministic `ORDER BY`, otherwise "page 2" is meaningless because row order isn't guaranteed:
```sql
SELECT * FROM articles ORDER BY published_at DESC, id DESC
LIMIT 20 OFFSET 40;   -- page 3, 20 per page
```
Including a unique tiebreaker (`id`) in the sort prevents rows from jumping between pages when the primary sort key has ties.

**7. Alias in `WHERE` vs `ORDER BY`.**
It comes down to *logical evaluation order*. The engine processes `FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT`. Column aliases are created in the `SELECT` step. `WHERE` runs *before* `SELECT`, so at that point the alias doesn't exist yet — hence you must repeat the full expression in `WHERE` (or wrap the query in a subquery/CTE). `ORDER BY` runs *after* `SELECT`, so the alias is already defined and you can sort by it. This isn't an arbitrary rule; it's a direct consequence of the pipeline.

**8. Three-valued logic and `= NULL`.**
SQL's `NULL` represents an *unknown* value, so any comparison involving it returns a third truth value, UNKNOWN, rather than TRUE or FALSE. `col = NULL` evaluates to UNKNOWN because you can't know whether an unknown equals anything. `WHERE` retains a row only when its predicate is TRUE — both FALSE and UNKNOWN rows are discarded. Therefore `WHERE col = NULL` matches nothing, ever. To test for null you use the dedicated predicate `IS NULL`, which returns a real boolean. This three-valued logic ripples through `AND`/`OR`/`NOT` and is the single most common source of subtle SQL bugs.

**9. `NOT IN (subquery)` returning zero rows.**
The classic cause is a **NULL in the subquery result**. `x NOT IN (a, b, NULL)` expands to `x <> a AND x <> b AND x <> NULL`. That last term is UNKNOWN, and `something AND UNKNOWN` can never be TRUE — so the whole predicate is at best UNKNOWN and the row is dropped. The result: as soon as the subquery yields even one NULL, `NOT IN` returns no rows. Fixes: filter NULLs out of the subquery (`WHERE col IS NOT NULL`), or better, rewrite using `NOT EXISTS`, which uses row-existence semantics and is immune to this trap:
```sql
SELECT e.* FROM employees e
WHERE NOT EXISTS (
  SELECT 1 FROM terminated t WHERE t.emp_id = e.id
);
```

**10. Deep `OFFSET` pagination.**
`OFFSET m` doesn't magically jump to row `m` — the database must generate, order, and then *discard* the first `m` rows before returning the next page. At `OFFSET 1000000` it does a million rows of work to hand you 20. Cost grows linearly with page depth, so late pages get progressively slower. The scalable alternative is **keyset (seek) pagination**: remember the last row's sort key and fetch the next page with a `WHERE` on it:
```sql
SELECT * FROM articles
WHERE (published_at, id) < ('2026-01-01', 500)
ORDER BY published_at DESC, id DESC
LIMIT 20;
```
This uses the index to jump straight to the boundary in roughly constant time regardless of depth. The trade-off is you can't jump to an arbitrary page number, only "next/previous".

**11. Anchored vs leading-wildcard `LIKE`.**
A B-tree index stores values in sorted order. `LIKE 'abc%'` is a *prefix* match — all matches share the prefix "abc", which forms a contiguous range in the sorted index, so the engine can seek to "abc" and scan the range: index-friendly. `LIKE '%abc'` has a *leading wildcard* — matching values can start with anything, so they're scattered throughout the index with no contiguous range to seek. The B-tree is useless and the engine must scan every row. To index suffix/substring searches you need a different structure: a trigram GIN index (`pg_trgm`) in Postgres, a reversed-string functional index, or a full-text index.

**12. `WHERE` vs `HAVING`.**
`WHERE` filters *individual rows* before grouping; `HAVING` filters *groups* after `GROUP BY` has aggregated them. Because `WHERE` runs first, it cannot reference aggregate functions (there are no groups yet), while `HAVING` exists precisely to test aggregates like `COUNT(*)` or `AVG(salary)`. Rule of thumb: if a condition is about a raw column value, put it in `WHERE` (it's cheaper — it shrinks the data before grouping); if it's about an aggregate over a group, it must go in `HAVING`.
```sql
SELECT dept, AVG(salary)
FROM employees
WHERE active = true          -- per-row filter
GROUP BY dept
HAVING AVG(salary) > 90000;  -- per-group filter
```

**13. `LIKE` vs `ILIKE`.**
`LIKE` is case-*sensitive* pattern matching in PostgreSQL. `ILIKE` is Postgres's case-*insensitive* variant, so `'smith' ILIKE 'S%'` is TRUE. Under the hood `ILIKE` effectively compares case-folded values, which usually can't use a plain B-tree index (you'd add a functional index on `lower(col)` and query `lower(col) LIKE lower(?)` for index support). MySQL has no `ILIKE`: its `LIKE` is case-insensitive by default for common `_ci` collations, and case-sensitive only with a binary/`_bin`/`_cs` collation. So the same `LIKE` query can behave differently across engines depending on collation — a portability gotcha worth knowing.

## Common Mistakes

- **Using `= NULL` / `<> NULL`** instead of `IS NULL` / `IS NOT NULL`.
- **`NOT IN` over a subquery that can contain NULLs** — silently returns nothing; use `NOT EXISTS`.
- **`LIMIT` without `ORDER BY`** — non-deterministic paging; add a stable, unique sort key.
- **Deep `OFFSET` pagination** on large tables — switch to keyset pagination.
- **`SELECT *` in application/production queries** — over-fetch and fragility.
- **Confusing `WHERE` and `HAVING`** — putting non-aggregate filters in `HAVING` (works but wasteful) or trying to use aggregates in `WHERE` (error).
- **Forgetting operator precedence** — `WHERE a OR b AND c` binds as `a OR (b AND c)`; parenthesize.
- **Assuming NULL sort position** — it differs between Postgres and MySQL.
- **Leading-wildcard `LIKE '%x'`** on big tables expecting index speed.

## Related Concepts

- **`GROUP BY` / aggregates / `HAVING`** — grouping and group-level filtering.
- **Joins** — reading across multiple tables.
- **Indexes (B-tree, GIN, trigram)** — what makes `WHERE`/`ORDER BY`/`LIKE` fast.
- **Keyset / seek pagination** — scalable alternative to `OFFSET`.
- **Full-text search** — for substring/relevance search beyond `LIKE`.
- **Window functions** — `ROW_NUMBER()`, `RANK()` for advanced top-N-per-group.
- **`EXPLAIN` / query plans** — inspecting scan vs index-scan vs sort choices.
- **Three-valued logic & NULL semantics** — foundational to correct predicates.

---

# CRUD — UPDATE

## What is it?

The **U** in CRUD is **Update**, and the `UPDATE` statement modifies the values of *existing* rows in a table. It does not create rows and (in its pure form) does not delete them — it changes column values in place, for the rows that match a condition.

The anatomy:

```sql
UPDATE table
SET    col1 = value1,
       col2 = value2
WHERE  condition;      -- which rows to change
```

The `SET` clause says *what to change*; the `WHERE` clause says *which rows*. Leave off the `WHERE` and you change **every row in the table** — one of the most dangerous mistakes in SQL.

## Why is it needed?

The world changes: prices rise, users edit their profiles, orders move from "pending" to "shipped", account balances go up and down. Without `UPDATE`, the only way to reflect a change would be to delete the old row and insert a new one — losing the row's identity, its foreign-key references, and its place in indexes. `UPDATE` preserves the row's identity (its primary key and all relationships) while changing its attributes.

From first principles, `UPDATE` is the statement that keeps stored data **consistent with reality over time**. It's also where a lot of correctness and concurrency subtlety lives, because two users updating the same row at the same time raises the classic *lost update* problem, which the database's locking and isolation levels must handle.

## How does it work?

**Single column:**

```sql
UPDATE employees
SET salary = 100000
WHERE id = 42;
```

**Multiple columns (one statement, comma-separated assignments):**

```sql
UPDATE employees
SET salary = 100000,
    job_title = 'Senior Engineer',
    updated_at = now()
WHERE id = 42;
```

All assignments happen together for each matched row — you don't need (or want) separate statements.

**Expressions referencing the current value:**

```sql
-- Give everyone in dept 3 a 10% raise
UPDATE employees
SET salary = salary * 1.10
WHERE department_id = 3;
```

The right-hand side sees the row's *pre-update* value, so `salary = salary * 1.10` is well-defined.

**Conditional updates with `CASE`:**

```sql
UPDATE employees
SET bonus = CASE
              WHEN rating >= 5 THEN salary * 0.20
              WHEN rating >= 3 THEN salary * 0.10
              ELSE 0
            END
WHERE active = true;
```

One statement applies different logic per row based on its data.

**`UPDATE ... FROM` — update using another table (PostgreSQL):**

```sql
-- Set each employee's dept_name from the departments table
UPDATE employees e
SET department_name = d.name
FROM departments d
WHERE e.department_id = d.id;
```

The `FROM` clause joins in other tables so their columns can drive the new values. This is the idiomatic Postgres way to do a join-based update.

> **MySQL note:** MySQL uses a different syntax — the join goes in the `UPDATE` clause itself:
> ```sql
> UPDATE employees e
> JOIN departments d ON e.department_id = d.id
> SET e.department_name = d.name;
> ```

**Update with a correlated subquery (portable across engines):**

```sql
UPDATE employees e
SET department_name = (
  SELECT d.name FROM departments d WHERE d.id = e.department_id
)
WHERE EXISTS (
  SELECT 1 FROM departments d WHERE d.id = e.department_id
);
```

The subquery computes the new value per row. The `WHERE EXISTS` guard matters: without it, employees with no matching department would get their `department_name` set to `NULL` (an empty subquery returns NULL).

**`RETURNING` after update (Postgres):**

```sql
UPDATE accounts
SET balance = balance - 100
WHERE id = 7
RETURNING id, balance;   -- see the new balance in the same round trip
```

## Internal Working

Contrary to the name, in an MVCC database an `UPDATE` does **not** overwrite bytes in place. Here's what actually happens in PostgreSQL:

```
  UPDATE finds matching rows (via WHERE, using index or seq scan)
        |
        v
  For each matched row:
     [1] Create a NEW tuple version with the new column values
     [2] Stamp old tuple's xmax = current txid (mark it "deleted for the future")
     [3] Stamp new tuple's xmin = current txid
     [4] Update indexes to point at the new tuple (HOT optimization may skip
         indexes if no indexed column changed and space is on the same page)
     [5] Write WAL record
        |
        v
  COMMIT -> new version becomes visible; old version is dead, later reclaimed by VACUUM
```

Key internals:

- **UPDATE = delete-old + insert-new (logically).** The old row version lingers as a "dead tuple" until `VACUUM` reclaims it. Heavy update workloads therefore produce **bloat** and depend on autovacuum to stay healthy.
- **HOT (Heap-Only Tuples).** If no *indexed* column changed and the new version fits on the same page, Postgres avoids updating the indexes — a big win. Updating an indexed column forfeits this.
- **Locking.** An `UPDATE` takes a **row-level exclusive lock** on each matched row for the duration of the transaction. A second transaction trying to update the same row **blocks** until the first commits or rolls back. This is how the engine prevents *lost updates* at READ COMMITTED.
- **MySQL/InnoDB** updates the clustered index in place (with undo logs for MVCC/rollback) and updates secondary indexes as needed; it also takes row locks. The MVCC mechanism differs (undo logs vs. new tuple versions) but the *lost update* protection via row locks is the same idea.
- **The lost-update problem.** Two transactions read balance=100, both compute 100-10, both write 90 — one update is lost. Solutions: `UPDATE ... SET balance = balance - 10` (atomic, avoids the read), pessimistic `SELECT ... FOR UPDATE`, or optimistic concurrency with a `version` column checked in the `WHERE`.

## Advantages

- **Preserves row identity** — primary key, foreign keys, and index positions survive; no dangling references.
- **Set-based** — one statement can update millions of rows with a single logical operation and plan.
- **Atomic per statement** — all matched rows change or none do (within the transaction's guarantees).
- **Expressive** — `CASE`, expressions over current values, and joins/subqueries let complex transformations happen server-side.
- **`RETURNING`** surfaces the post-update values without a second query.
- **Concurrency-safe** with proper use of atomic expressions or locking.

## Limitations

- **Forgetting `WHERE` updates every row** — catastrophic and easy to do.
- **Large updates cause bloat & long locks** — a single `UPDATE` over 50M rows holds locks, generates huge WAL, and bloats the table; often better chunked in batches.
- **Lost-update hazards** under concurrency if you read-then-write in application code instead of using atomic SQL.
- **HOT forfeited** when indexed columns change, raising write cost.
- **Syntax divergence** — `UPDATE ... FROM` (Postgres) vs `UPDATE ... JOIN` (MySQL); subquery form is the portable fallback.
- **Triggers & cascades** can make an "innocent" update do far more work than it appears (audit rows, `ON UPDATE CASCADE`, etc.).

## Real-world Applications

- **Profile edits** — user changes email/name/preferences.
- **State machines** — moving an order through `pending → paid → shipped → delivered`.
- **Balances & counters** — `SET balance = balance - amount`, `SET view_count = view_count + 1` (atomic).
- **Bulk corrections / backfills** — `UPDATE ... FROM` to repair or enrich data from a reference table.
- **Soft deletes** — `UPDATE ... SET deleted_at = now()` instead of a physical `DELETE`.
- **Denormalization maintenance** — keeping a cached `department_name` or `comment_count` in sync.
- **Optimistic locking** — `UPDATE ... WHERE id = ? AND version = ?` then checking the affected-row count.

## Interview Questions

**Beginner**
1. Write an `UPDATE` that changes one employee's salary.
2. What happens if you run an `UPDATE` without a `WHERE` clause?

**Intermediate**
3. How do you update multiple columns in a single statement?
4. How do you give every employee in a department a 10% raise in one statement?

**Advanced**
5. Explain how PostgreSQL physically performs an `UPDATE` (MVCC). What is bloat and what is HOT?
6. Describe the lost-update problem and three ways to prevent it.

**Scenario-based**
7. You must copy `department_name` into `employees` from a `departments` table. Show the Postgres and MySQL approaches, and a portable one.
8. You need to update 100 million rows without taking the site down. How do you approach it?

**"Why" questions**
9. Why is `UPDATE ... SET count = count + 1` safer under concurrency than reading the value in the app and writing back?
10. Why can a large `UPDATE` slow down other queries even after it finishes?

**Comparison questions**
11. `UPDATE ... FROM` vs a correlated-subquery update — trade-offs?
12. Update-in-place (MySQL/InnoDB) vs new-tuple-versioning (Postgres MVCC).

## Model Answers

**1. Update one salary.**
```sql
UPDATE employees SET salary = 100000 WHERE id = 42;
```
The `WHERE id = 42` targets exactly one row by primary key — the safest possible predicate. Always update by a unique, selective key unless you deliberately intend a bulk change.

**2. `UPDATE` without `WHERE`.**
It updates **every row in the table** — the `SET` is applied unconditionally. `UPDATE employees SET salary = 0;` zeroes out every salary. This is one of the most infamous production incidents in the industry. Mitigations: always write the `WHERE` first, run inside a transaction so you can `ROLLBACK`, preview with a `SELECT` using the same predicate, and enable "safe update mode" in MySQL (`sql_safe_updates`) which refuses `UPDATE`/`DELETE` without a key-based `WHERE`.

**3. Multiple columns.**
List comma-separated assignments in one `SET`:
```sql
UPDATE employees
SET salary = 100000, job_title = 'Senior Engineer', updated_at = now()
WHERE id = 42;
```
All assignments apply atomically to each matched row in a single pass — more efficient and correct than three separate statements (which would also mean three separate lock/WAL cycles).

**4. 10% raise for a department.**
```sql
UPDATE employees SET salary = salary * 1.10 WHERE department_id = 3;
```
This is set-based: one statement updates all matching rows. The right-hand side references the row's current `salary`, and the engine uses the pre-update value for each row, so the arithmetic is well-defined even though many rows change at once.

**5. MVCC, bloat, and HOT.**
In Postgres an `UPDATE` doesn't overwrite the existing row. It writes a *new* version of the tuple with the changed values, marks the old version as expired (sets its `xmax`), and points current transactions at the appropriate version based on visibility rules. The old, now-dead version remains on disk until `VACUUM` reclaims its space. **Bloat** is the accumulation of these dead tuples (and empty space) when updates/deletes outpace vacuuming — it grows the table and its indexes, slowing scans. **HOT (Heap-Only Tuple)** is an optimization: if the update changes no indexed column and the new version fits on the same heap page, Postgres links the new version to the old within the page and skips updating every index — dramatically cheaper. Updating an indexed column disables HOT for that row, so indexing volatile columns raises write cost.

**6. Lost-update problem and prevention.**
The lost update occurs when two transactions read the same value, each computes a new value from it, and each writes back — the second write silently overwrites the first, and one update is "lost". Classic with balances/counters. Three preventions: (a) **Atomic SQL** — do the arithmetic in the database: `UPDATE accounts SET balance = balance - 10 WHERE id = ?`; the engine reads and writes under a row lock, so no interleaving. (b) **Pessimistic locking** — `SELECT ... FOR UPDATE` locks the row before you read it in application code, forcing other writers to wait. (c) **Optimistic concurrency** — keep a `version` column; update with `WHERE id = ? AND version = ?` and `SET version = version + 1`; if zero rows were affected, someone else changed it first and you retry. Choose atomic SQL when possible, optimistic for low-contention web flows, pessimistic for high-contention critical sections.

**7. Join-based update (copy department_name).**
Postgres:
```sql
UPDATE employees e
SET department_name = d.name
FROM departments d
WHERE e.department_id = d.id;
```
MySQL:
```sql
UPDATE employees e
JOIN departments d ON e.department_id = d.id
SET e.department_name = d.name;
```
Portable (correlated subquery):
```sql
UPDATE employees e
SET department_name = (SELECT d.name FROM departments d WHERE d.id = e.department_id)
WHERE EXISTS (SELECT 1 FROM departments d WHERE d.id = e.department_id);
```
The `WHERE EXISTS` guard is important — without it, unmatched employees would have `department_name` overwritten with NULL, because a subquery with no match returns NULL.

**8. Updating 100 million rows safely.**
Don't do it in one statement — that single transaction would hold locks, generate enormous WAL, bloat the table massively, and block autovacuum from cleaning up until it finishes. Instead **batch** it: loop over chunks of, say, 10,000–50,000 rows keyed by primary key ranges or a `LIMIT`-driven cursor, committing each batch so locks release and vacuum can keep up:
```sql
-- repeat until zero rows affected
UPDATE big_table
SET status = 'migrated'
WHERE id IN (
  SELECT id FROM big_table WHERE status = 'old' ORDER BY id LIMIT 10000
);
COMMIT;
```
Add a brief sleep between batches to reduce I/O pressure, run during low traffic, monitor replication lag, and ensure the filter column is indexed so each batch is cheap. This trades one giant lock for many tiny ones and keeps the site responsive.

**9. Why atomic increment is safer.**
`UPDATE ... SET count = count + 1` performs the read and the write as a single indivisible operation *inside the database*, under a row-level lock. Two concurrent executions are serialized by that lock, so both increments count. If instead the application reads `count` (say 100), adds 1, and writes 101, two concurrent requests can both read 100 and both write 101 — a lost update; the true answer should be 102. Pushing the arithmetic into the SQL statement removes the read-modify-write window in application space and lets the engine's locking guarantee correctness.

**10. Why a large UPDATE slows things afterward.**
Because it leaves debris. In Postgres, every updated row leaves a dead tuple behind; a 100M-row update can double the table's physical size overnight. Until autovacuum reclaims that space, sequential scans read more pages, indexes are larger, and cache hit rates drop — so unrelated queries slow down. The huge WAL burst also stresses I/O and replication. Even in MySQL/InnoDB, the undo logs and secondary-index churn plus a bloated buffer pool footprint degrade neighbors. The fix is proactive vacuuming/`OPTIMIZE`, batching the update, and monitoring bloat.

**11. `UPDATE ... FROM` vs correlated subquery.**
`UPDATE ... FROM` (or MySQL's `UPDATE ... JOIN`) expresses a set-based join once and lets the planner choose an efficient join strategy (hash/merge join) — typically much faster for large updates and cleaner to read. A correlated subquery re-evaluates the inner query *per outer row*, which can be slower and risks the NULL-overwrite pitfall if you forget the existence guard. The subquery form's one advantage is portability — it works the same across engines. Rule: prefer `UPDATE ... FROM`/`JOIN` for performance and clarity on your target engine; fall back to the subquery only for cross-database code, and always guard it with `WHERE EXISTS`.

**12. In-place vs new-version updates.**
InnoDB (MySQL) updates the row in place within the clustered index and records the prior image in **undo logs**, which serve rollback and MVCC reads; secondary indexes are updated only for changed indexed columns. Postgres never updates in place — it writes a **new tuple version** and marks the old one dead, relying on `VACUUM` to reclaim space. Consequences: Postgres updates can be cheaper to *write* (append-like) but generate bloat and depend heavily on autovacuum; InnoDB avoids bloat from versioning (undo is reclaimed by purge) but in-place updates of indexed columns and page reorganizations have their own costs. Both use row locks to prevent lost updates. Understanding your engine's model guides indexing and maintenance decisions.

## Common Mistakes

- **Omitting `WHERE`** and updating the whole table.
- **Read-modify-write in app code** instead of atomic `SET col = col + n`, causing lost updates.
- **Forgetting the `WHERE EXISTS` guard** in subquery updates, nulling out unmatched rows.
- **One massive `UPDATE`** instead of batching — locks, bloat, WAL blowups, replication lag.
- **Updating hot/indexed columns needlessly**, forfeiting HOT and inflating index maintenance.
- **Not wrapping risky updates in a transaction** so they can't be rolled back.
- **Ignoring triggers/cascades** that turn a small update into a large hidden workload.
- **Assuming the affected-row count is 1** in optimistic locking without checking it.

## Related Concepts

- **Transactions & isolation levels** — READ COMMITTED, REPEATABLE READ, SERIALIZABLE and their effect on concurrent updates.
- **`SELECT ... FOR UPDATE`** — pessimistic row locking.
- **Optimistic concurrency / version columns** — lock-free conflict detection.
- **MVCC, `VACUUM`, autovacuum, bloat** — Postgres update mechanics.
- **HOT updates** — index-avoiding optimization.
- **`UPSERT` (`ON CONFLICT` / `ON DUPLICATE KEY`)** — insert-or-update in one statement.
- **Triggers & `ON UPDATE CASCADE`** — side effects of updates.
- **Soft delete** — using `UPDATE` to mark rather than remove.

---

# CRUD — DELETE

## What is it?

The **D** in CRUD is **Delete**, and the `DELETE` statement removes existing rows from a table. Like `UPDATE`, it is targeted by a `WHERE` clause: the rows matching the condition are removed; the rest stay.

```sql
DELETE FROM table
WHERE  condition;
```

Omit the `WHERE` and you delete **every row in the table** — though, importantly, the table itself and its structure remain. This topic also compares `DELETE` with its two often-confused cousins, `TRUNCATE` and `DROP`, which operate at very different levels.

## Why is it needed?

Data outlives its usefulness. Users close accounts, carts are abandoned, logs age out, GDPR "right to be forgotten" requests arrive, test data must be purged. `DELETE` is how you remove specific rows while keeping the table and everything else intact.

From first principles, deletion is deceptively subtle because rows are rarely isolated — they're referenced by **foreign keys**. Deleting a `customer` who still has `orders` must either be blocked, cascade to the orders, or null out the reference. So `DELETE` is tightly bound up with **referential integrity**, and choosing the right behavior (`RESTRICT`, `CASCADE`, `SET NULL`) is a core design decision. There's also a recurring real-world tension: **hard delete** (physically remove) vs **soft delete** (mark as deleted with a flag/timestamp), each with different trade-offs for auditability and recovery.

## How does it work?

**Conditional delete:**

```sql
DELETE FROM employees WHERE id = 42;
DELETE FROM sessions WHERE last_seen < now() - interval '30 days';
```

**Delete based on another table (Postgres `USING`):**

```sql
DELETE FROM order_items oi
USING orders o
WHERE oi.order_id = o.id
  AND o.status = 'cancelled';
```

> **MySQL** uses a `JOIN` form: `DELETE oi FROM order_items oi JOIN orders o ON oi.order_id = o.id WHERE o.status = 'cancelled';`

**Delete with a subquery (portable):**

```sql
DELETE FROM order_items
WHERE order_id IN (SELECT id FROM orders WHERE status = 'cancelled');
```

**`RETURNING` deleted rows (Postgres):**

```sql
DELETE FROM employees WHERE department_id = 9
RETURNING id, last_name;   -- capture what was removed, e.g. to archive
```

**Foreign-key behavior when deleting a parent:**

```sql
-- Defined on the child table's FK:
FOREIGN KEY (customer_id) REFERENCES customers(id) ON DELETE CASCADE;   -- delete children too
FOREIGN KEY (customer_id) REFERENCES customers(id) ON DELETE RESTRICT;  -- block if children exist
FOREIGN KEY (customer_id) REFERENCES customers(id) ON DELETE SET NULL;  -- orphan the child, null the FK
```

**Soft delete (an `UPDATE`, not a `DELETE`):**

```sql
UPDATE users SET deleted_at = now() WHERE id = 42;
-- and then always query: WHERE deleted_at IS NULL
```

## Internal Working

**`DELETE` internals (Postgres MVCC):**

```
  DELETE matches rows via WHERE (index or seq scan)
        |
        v
  For each matched row:
     [1] Mark the tuple's xmax = current txid  (does NOT physically erase it)
     [2] Write WAL record
     [3] Fire row-level triggers / FK cascade actions
        |
        v
  COMMIT -> row becomes invisible to new transactions
        |
        v
  VACUUM (later) -> physically reclaims the dead tuple's space
```

So a `DELETE`, like an `UPDATE`, doesn't immediately shrink the file — it marks tuples dead and leaves the space for `VACUUM` to reclaim. Deleting a billion rows can leave the table just as large on disk until vacuumed (and even then space is reused, not necessarily returned to the OS without `VACUUM FULL`).

**`TRUNCATE` internals:**

`TRUNCATE` is a **DDL** operation. Instead of visiting each row, it essentially discards the table's underlying data files and creates fresh empty ones (in Postgres, it assigns a new relfilenode). It does **not** scan rows, does **not** fire per-row triggers, does **not** generate per-row WAL, and cannot use a `WHERE`. That's why it's near-instant regardless of table size.

**`DROP` internals:**

`DROP TABLE` removes the table *definition itself* from the system catalogs and frees all its storage — schema, data, indexes, constraints, triggers. After a `DROP`, the table no longer exists.

**The comparison — this is a classic interview table:**

```
                | DELETE              | TRUNCATE            | DROP
----------------+---------------------+---------------------+----------------------
Category (DDL/  | DML                 | DDL                 | DDL
  DML)          |                     |                     |
Removes         | selected rows       | all rows            | rows + table itself
WHERE clause    | Yes                 | No (all or nothing) | No
Structure kept? | Yes                 | Yes                 | No (table gone)
Speed on big    | Slow (row-by-row,   | Very fast (drops    | Very fast
  tables        | logged, triggers)   | data files)         |
Per-row triggers| Fire (BEFORE/AFTER  | Do NOT fire         | N/A
                | DELETE)             | (TRUNCATE trigger   |
                |                     | only, Postgres)     |
WAL / logging   | Per-row (heavy)     | Minimal (metadata)  | Minimal (metadata)
Resets identity/| No (by default)     | Optional (RESTART   | N/A
  AUTO_INCREMENT|                     | IDENTITY / MySQL    |
                |                     | resets to 0)        |
Transactional / | Yes (rollback OK)   | Postgres: yes,      | Postgres: yes;
  rollback      |                     | rollback-able.      | MySQL: implicit
                |                     | MySQL: implicit     | commit, no rollback
                |                     | commit, no rollback |
FK handling     | Honors ON DELETE    | Blocked if FK refs  | Blocked/ CASCADE
                | CASCADE/RESTRICT    | exist (unless       | dependent objects
                |                     | CASCADE)            |
Reclaims space  | Needs VACUUM        | Frees immediately   | Frees immediately
Typical use     | Remove some rows    | Empty a table fast  | Delete the table
```

Two engine caveats worth memorizing:
- **MySQL:** `TRUNCATE` and `DROP` cause an **implicit commit** and **cannot be rolled back**. `DELETE` in a transaction can.
- **PostgreSQL:** `TRUNCATE` and `DROP` are transactional (`ROLLBACK`-able if inside a `BEGIN`), which surprises people coming from MySQL/Oracle.

## Advantages

- **`DELETE` is precise** — remove exactly the rows a `WHERE` selects; it's the only one of the three that filters.
- **`DELETE` is transactional and reversible** (before commit) and fires triggers/cascades, so integrity logic runs.
- **`TRUNCATE` is dramatically faster** for emptying a whole table — no per-row work.
- **`DROP` fully reclaims** all storage and removes the object cleanly.
- **`RETURNING`** lets you archive deleted rows in the same statement.
- **FK actions (`CASCADE`/`SET NULL`)** automate consistent multi-table cleanup.

## Limitations

- **`DELETE` on huge tables is slow and bloating** — per-row logging, trigger firing, dead tuples needing VACUUM.
- **Forgetting `WHERE`** deletes all rows (though structure survives).
- **`TRUNCATE` can't filter** and (MySQL) can't be rolled back — irreversible in that engine.
- **`DROP` is destructive** — the table and all dependent data vanish; recoverable only from backups.
- **FK constraints can block deletes** (`RESTRICT`/`NO ACTION`) or cause surprising wide cascades.
- **`DELETE` doesn't reset AUTO_INCREMENT/identity**; new rows continue from the old high-water mark.
- **Space isn't returned to the OS** after `DELETE` without aggressive vacuuming.

## Real-world Applications

- **GDPR / account deletion** — hard-delete a user and cascade to their data, or anonymize.
- **Expiring data** — `DELETE FROM sessions WHERE expires_at < now()` (often batched or via partition drop).
- **Emptying staging tables** between ETL runs — `TRUNCATE` for speed.
- **Test/dev resets** — `TRUNCATE ... RESTART IDENTITY` to get clean, id-from-1 tables.
- **Schema teardown / migrations** — `DROP TABLE` for obsolete tables.
- **Cancelled-order cleanup** — join-based `DELETE` across parent/child tables.
- **Partition management** — dropping an old time-partition is far cheaper than deleting its rows.

## Interview Questions

**Beginner**
1. How do you delete a specific row? What happens if you forget the `WHERE`?
2. Does `DELETE` remove the table itself?

**Intermediate**
3. Explain the differences between `DELETE`, `TRUNCATE`, and `DROP`.
4. What are `ON DELETE CASCADE`, `RESTRICT`, and `SET NULL`?

**Advanced**
5. Why is `TRUNCATE` so much faster than `DELETE` on a large table?
6. In PostgreSQL, why doesn't a `DELETE` immediately free disk space?

**Scenario-based**
7. You must delete 200 million expired rows from a live 500M-row table without downtime. How?
8. A `DELETE FROM customers WHERE id = 5` fails with a foreign-key error. Why, and what are your options?

**"Why" questions**
9. Why might you choose a soft delete over a hard delete?
10. Why can `TRUNCATE` be rolled back in Postgres but not in MySQL?

**Comparison questions**
11. `DELETE` vs `TRUNCATE` — full comparison including triggers, identity reset, and rollback.
12. Hard delete vs soft delete — trade-offs.

## Model Answers

**1. Delete a row / forgetting WHERE.**
```sql
DELETE FROM employees WHERE id = 42;
```
The `WHERE` restricts the operation to matching rows. If you omit it — `DELETE FROM employees;` — the statement removes **every row** in the table. The table structure, indexes, and constraints remain (it becomes an empty table), but all data is gone. Always write the `WHERE` first, preview with a `SELECT` using the same predicate, and run inside a transaction so a mistake can be `ROLLBACK`ed.

**2. Does DELETE remove the table?**
No. `DELETE` only removes *rows*. The table definition — columns, indexes, constraints, triggers, permissions — stays intact and ready to receive new rows. To remove the table itself you use `DROP TABLE`.

**3. DELETE vs TRUNCATE vs DROP.**
`DELETE` (DML) removes selected rows via a `WHERE`, row by row, firing triggers and honoring FK actions, fully logged and transactional. `TRUNCATE` (DDL) removes *all* rows at once by discarding the table's data files — no `WHERE`, no per-row triggers, minimal logging, near-instant regardless of size, and it keeps the table structure. `DROP` (DDL) removes the *entire table* — data plus definition — so the table ceases to exist. Order of destructiveness: `DELETE` (some/all rows) < `TRUNCATE` (all rows, keep table) < `DROP` (table gone). Use `DELETE` for selective removal, `TRUNCATE` to quickly empty a table, `DROP` to eliminate a table entirely.

**4. FK on-delete actions.**
These are declared on the child table's foreign key and decide what happens when the referenced parent row is deleted. `ON DELETE CASCADE` deletes the child rows along with the parent — deleting a customer deletes their orders. `ON DELETE RESTRICT` (and the similar `NO ACTION`) *blocks* the parent delete if any child rows exist, forcing you to remove them first — the default protective behavior. `ON DELETE SET NULL` keeps the child rows but sets their foreign-key column to NULL, orphaning them (requires the column to be nullable). The choice encodes business rules: cascade when children can't exist without the parent (order items), restrict when accidental mass deletion must be prevented, set-null when the relationship is optional.

**5. Why TRUNCATE beats DELETE.**
`DELETE` must locate and process each row individually: mark it dead, write a WAL/redo record for it, fire any per-row triggers, and check/apply FK cascades — all proportional to the number of rows. `TRUNCATE` sidesteps all of that; it doesn't touch rows at all. It simply throws away the table's underlying data files and gives it fresh empty ones, a constant-time metadata operation. No row scan, no per-row logging, no triggers. That's why `TRUNCATE` on a billion-row table completes in milliseconds while `DELETE` could run for hours — but it's also why `TRUNCATE` can't be selective or fire row triggers.

**6. Why DELETE doesn't free space immediately (Postgres).**
Because of MVCC. A `DELETE` doesn't erase the row; it marks the tuple as expired by setting its `xmax`, so transactions that started before the commit can still see it (snapshot isolation) and so the change can be rolled back. The dead tuple physically remains until `VACUUM` runs and reclaims its space for reuse. Even then, ordinary `VACUUM` returns the space to the table's free space map, not to the operating system — the file typically doesn't shrink. To actually return space to the OS you need `VACUUM FULL` (which rewrites the table and takes an exclusive lock) or table rewriting tools. This is why mass deletes are notorious for leaving bloated tables.

**7. Deleting 200M rows on a live table.**
Never in one statement — it would hold locks, generate massive WAL, bloat the table, and stall autovacuum. **Batch** it: delete in chunks (e.g., 10k–50k rows) keyed by an indexed column, committing between batches so locks release and vacuum keeps pace:
```sql
-- repeat until 0 rows affected
DELETE FROM events
WHERE id IN (
  SELECT id FROM events WHERE created_at < now() - interval '90 days'
  ORDER BY id LIMIT 10000
);
COMMIT;
```
Pace the loop with brief sleeps, run in off-peak hours, watch replication lag, and vacuum periodically. Even better, if the data is time-based, **partition the table by time and `DROP`/`DETACH` the old partition** — an instant metadata operation that avoids row-by-row deletion entirely. Partition-drop is the gold standard for large time-series purges.

**8. FK error on DELETE.**
The delete fails because other rows reference the customer via a foreign key whose `ON DELETE` action is `RESTRICT`/`NO ACTION` (the default) — the database refuses to create dangling references (e.g., orders pointing at a nonexistent customer). Options: (a) delete or reassign the child rows first, then the parent; (b) if the schema intends cascading removal, redefine the FK with `ON DELETE CASCADE` so children go automatically; (c) if the relationship is optional, use `ON DELETE SET NULL` to orphan the children; or (d) soft-delete the customer instead of physically removing them. The right choice depends on business rules — but silently dropping the constraint just to make the error go away risks orphaned, inconsistent data.

**9. Soft vs hard delete — why soft.**
A soft delete marks a row as deleted (`deleted_at`/`is_deleted`) instead of physically removing it, and queries filter on `deleted_at IS NULL`. You choose it when you need **auditability, recoverability, or referential history**: undelete/undo features, regulatory retention, preserving foreign-key references so historical orders still resolve their (now "deleted") customer, and analytics that must count churned records. The costs are real: every query must remember the `deleted_at IS NULL` filter (a footgun; often solved with views), unique constraints get complicated (a "deleted" email may still collide), tables grow with data that's logically gone, and privacy regulations may actually *require* hard deletion. So soft delete is great for business objects you might restore or must audit; hard delete (or anonymization) is right for true erasure requirements.

**10. TRUNCATE rollback: Postgres vs MySQL.**
It comes down to how each engine treats DDL in transactions. PostgreSQL has **transactional DDL** — `TRUNCATE`, `DROP`, `CREATE`, etc. participate in transactions, take the necessary locks, and can be rolled back if issued inside a `BEGIN`. MySQL (with InnoDB) does **not** have fully transactional DDL; statements like `TRUNCATE` and `DROP` trigger an **implicit commit** — they end the current transaction and take effect immediately, so there's nothing to roll back. Practically: in Postgres you can wrap a `TRUNCATE` in a transaction and `ROLLBACK` if something looks wrong; in MySQL, once you `TRUNCATE`, it's gone. This is a frequent cross-engine gotcha.

**11. DELETE vs TRUNCATE (full).**
`DELETE` is DML: it removes selected rows (or all, without `WHERE`), row by row, firing `BEFORE/AFTER DELETE` triggers, honoring FK cascade/restrict actions, writing per-row WAL, and it's fully transactional and rollback-able. It does **not** reset auto-increment/identity, and it leaves dead tuples that need vacuuming. `TRUNCATE` is DDL: it removes *all* rows by dropping the data files — no `WHERE`, no per-row triggers (Postgres supports a statement-level `TRUNCATE` trigger only), minimal logging, and it's near-instant on any size. It can **reset identity** (`RESTART IDENTITY` in Postgres; MySQL resets `AUTO_INCREMENT` to 1) and frees space immediately. Rollback differs by engine (Postgres: yes; MySQL: no, implicit commit). In short: reach for `DELETE` when you need selectivity, triggers, or rollback; reach for `TRUNCATE` to empty a whole table as fast as possible.

**12. Hard vs soft delete trade-offs.**
Hard delete physically removes the row: storage is reclaimed (after vacuum), unique constraints stay simple, and privacy "erasure" is genuinely satisfied — but the data is gone (recoverable only from backups), foreign-key references may break or need cascading, and you lose audit history. Soft delete keeps the row with a deleted marker: it's recoverable, preserves history and references, and supports undo — but it complicates every query (must filter deleted rows), muddies unique constraints, grows tables with logically-dead data, and can violate regulations that mandate true deletion. Many mature systems use a **hybrid**: soft-delete for user-facing recoverability and audit, then a background job hard-deletes (or anonymizes) after a retention window to satisfy privacy and reclaim space.

## Common Mistakes

- **Forgetting `WHERE`** — wiping the whole table's data.
- **Using `DELETE` to empty a large table** when `TRUNCATE` (or partition drop) is the right, fast tool.
- **Confusing `TRUNCATE`/`DROP` rollback behavior** across engines — irreversible in MySQL.
- **One giant `DELETE`** on a live table instead of batching — locks, bloat, WAL/replication storms.
- **Not anticipating FK cascades** — a small delete triggering a huge cascade, or being blocked by `RESTRICT`.
- **Expecting disk to shrink** after `DELETE` without `VACUUM FULL`.
- **Assuming `DELETE` resets identity** — it does not; use `TRUNCATE ... RESTART IDENTITY`.
- **Soft-delete without a consistent filter** — forgetting `WHERE deleted_at IS NULL` and leaking "deleted" rows.

## Related Concepts

- **Foreign keys & referential actions** — `CASCADE`, `RESTRICT`, `NO ACTION`, `SET NULL`, `SET DEFAULT`.
- **`TRUNCATE`, `DROP`, DDL vs DML** — categories and transactional behavior.
- **MVCC, dead tuples, `VACUUM`/`VACUUM FULL`** — space reclamation after deletes.
- **Table partitioning** — `DROP`/`DETACH PARTITION` for cheap bulk purges.
- **Soft delete pattern** — `deleted_at`, partial unique indexes, filtered views.
- **Cascade triggers & audit tables** — capturing what was deleted.
- **Transactions & isolation** — rollback and concurrent-delete visibility.
- **`RETURNING`** — archiving deleted rows in one statement.

---

# SQL Functions — String, Numeric, Date & NULL Handling

## What is it?

SQL **built-in functions** are engine-provided routines that transform or compute values inside a query. Rather than pulling raw data to the application and processing it there, you push the computation into the database where the data already lives. This topic covers four families that appear constantly in real work and interviews:

- **String functions** — `CONCAT`, `LENGTH`, `LOWER`, `UPPER`, `TRIM`, `SUBSTRING`, `REPLACE`.
- **Numeric functions** — `ROUND`, `CEIL`, `FLOOR`, `ABS`.
- **Date/time functions** — `CURRENT_DATE`, `CURRENT_TIMESTAMP`, `AGE`, `EXTRACT`.
- **NULL-handling functions** — `COALESCE`, `NULLIF`.

These are **scalar functions**: they take one row's values in and return one value out (as opposed to *aggregate* functions like `SUM` that collapse many rows into one).

```sql
SELECT UPPER(first_name), ROUND(salary/12, 2), EXTRACT(YEAR FROM hire_date),
       COALESCE(nickname, first_name) AS display_name
FROM employees;
```

## Why is it needed?

Two first-principles reasons.

**1. Push computation to the data.** Filtering, formatting, and aggregating at the database avoids shipping large result sets over the network to be processed client-side. `WHERE lower(email) = lower(?)` lets the engine filter with an index; doing it in the app means fetching every row first.

**2. Correctness and consistency.** Date arithmetic, rounding rules, and NULL semantics are subtle. Using the engine's tested functions gives consistent, well-defined behavior across every client and language that touches the database. A `COALESCE` in the query guarantees the same default no matter which app reads it.

NULL handling deserves special emphasis: because of three-valued logic, arithmetic and concatenation involving NULL often *propagate* NULL (`5 + NULL = NULL`, `'a' || NULL = NULL` in standard SQL). `COALESCE` and `NULLIF` are the surgical tools for taming that.

## How does it work?

**String functions.**

```sql
SELECT CONCAT(first_name, ' ', last_name) AS full_name FROM employees;
-- Postgres also: first_name || ' ' || last_name   (|| is the SQL standard operator)

SELECT LENGTH('hello');          -- 5   (character count; Postgres CHAR length)
SELECT LOWER('HELLO'), UPPER('hi');   -- 'hello', 'HI'
SELECT TRIM('  hi  ');           -- 'hi'  (strips leading & trailing spaces)
SELECT TRIM(BOTH 'x' FROM 'xxhixx');  -- 'hi'  (strip a specific character)
SELECT SUBSTRING('database' FROM 1 FOR 4);  -- 'data'  (start, length; 1-based)
-- MySQL: SUBSTRING('database', 1, 4)
SELECT REPLACE('a-b-c', '-', '/');    -- 'a/b/c'
```

Notes: `CONCAT` treats NULL as an empty string in both Postgres and MySQL, whereas the `||` operator returns NULL if any operand is NULL (Postgres). `LENGTH` counts characters in Postgres; in MySQL `LENGTH` counts *bytes* and `CHAR_LENGTH` counts characters — a classic gotcha with multibyte text.

**Numeric functions.**

```sql
SELECT ROUND(3.14159, 2);   -- 3.14  (round to 2 decimals)
SELECT ROUND(2.5);          -- 3     (rounds half away from zero for numeric)
SELECT CEIL(4.1);           -- 5     (smallest integer >= value)
SELECT FLOOR(4.9);          -- 4     (largest integer <= value)
SELECT ABS(-17);            -- 17    (absolute value)
```

Rounding subtlety: `ROUND` on `numeric` in Postgres rounds half *away from zero*; on `double precision` it may use banker's rounding (round-half-to-even) because that's how IEEE floating point behaves. For money, always use `numeric`/`decimal`, never floats.

**Date/time functions.**

```sql
SELECT CURRENT_DATE;        -- 2026-07-09           (today's date)
SELECT CURRENT_TIMESTAMP;   -- 2026-07-09 14:33:07+00  (date + time + tz, now)
SELECT now();               -- Postgres synonym for CURRENT_TIMESTAMP

SELECT AGE(TIMESTAMP '2026-07-09', TIMESTAMP '1990-01-15');
-- 36 years 5 mons 24 days   (Postgres: symbolic interval between two timestamps)
SELECT AGE(birth_date);     -- interval from birth_date to CURRENT_DATE

SELECT EXTRACT(YEAR  FROM hire_date);   -- 2019   (pull a field out)
SELECT EXTRACT(MONTH FROM CURRENT_DATE);
SELECT EXTRACT(DOW   FROM order_ts);    -- day of week (0=Sunday in Postgres)
SELECT EXTRACT(EPOCH FROM (end_ts - start_ts));  -- duration in seconds
```

> **MySQL notes:** `AGE` doesn't exist — use `TIMESTAMPDIFF(YEAR, birth, CURDATE())` for whole years, or `DATEDIFF` for day counts. `CURRENT_DATE`/`CURRENT_TIMESTAMP` exist (also `CURDATE()`/`NOW()`). `EXTRACT(YEAR FROM d)` works, and `YEAR(d)`/`MONTH(d)` are shorthands.

**NULL-handling functions.**

```sql
SELECT COALESCE(nickname, first_name, 'Unknown') FROM users;
-- returns the FIRST non-NULL argument, left to right

SELECT COALESCE(discount, 0) * price FROM line_items;
-- guard arithmetic: NULL discount becomes 0 instead of nulling the product

SELECT NULLIF(a, b);
-- returns NULL if a = b, else returns a
SELECT total / NULLIF(count, 0);   -- avoid divide-by-zero: 0 becomes NULL -> result NULL
```

`COALESCE` = "give me the first thing that isn't null." `NULLIF(x, y)` = "turn x into null when it equals y" — most famously to neutralize a divide-by-zero.

## Internal Working

- **Scalar functions are evaluated per row** during query execution, in the `SELECT`/`WHERE` projection step. They're generally cheap CPU operations, but calling one on an indexed column *in a `WHERE`* can defeat the index: `WHERE lower(email) = ?` can't use a plain index on `email` — the engine would have to compute `lower()` for every row. The fix is a **functional/expression index**: `CREATE INDEX ON users (lower(email))`, after which the predicate is index-sargable.

- **`COALESCE` and `NULLIF` are short-circuit `CASE` expressions under the hood.** `COALESCE(a, b, c)` is literally defined as `CASE WHEN a IS NOT NULL THEN a WHEN b IS NOT NULL THEN b ELSE c END`, and it **stops evaluating at the first non-NULL** — so a side-effecting or expensive later argument isn't evaluated if an earlier one is non-null. `NULLIF(a, b)` is `CASE WHEN a = b THEN NULL ELSE a END`. Knowing this desugaring explains their behavior exactly.

- **NULL propagation.** Most scalar functions and operators return NULL if any input is NULL (`ABS(NULL)`, `UPPER(NULL)`, `1 + NULL`, `'x' || NULL` all yield NULL). The important exceptions are `CONCAT` (treats NULL as `''`), the NULL-handling functions themselves, and `IS [NOT] DISTINCT FROM`. This is a direct consequence of NULL meaning "unknown": a function of an unknown input is unknown.

- **Data types matter.** Rounding behavior, `LENGTH` bytes-vs-chars, and interval arithmetic all depend on the column's declared type. `EXTRACT` returns a `double precision`/`numeric`; `AGE` returns an `interval` (a symbolic year/month/day value, not a fixed number of seconds — because months have varying lengths). This is why `AGE` gives "36 years 5 mons" rather than a raw day count.

- **`CURRENT_TIMESTAMP` is fixed for the whole transaction** in Postgres — it returns the transaction start time, so multiple calls within one transaction agree. `clock_timestamp()` gives the actual wall-clock instant if you need it to advance. This trips people up when timing things inside a transaction.

## Advantages

- **Server-side computation** reduces data transfer and centralizes logic.
- **Correctness** — battle-tested implementations of tricky rounding, date math, and NULL logic.
- **Indexable via expression indexes** — `lower(col)`, `(col::date)` etc. keep function-based filters fast.
- **`COALESCE`/`NULLIF` make queries robust** to missing data and division-by-zero without verbose `CASE`.
- **Consistency** — every client sees the same computed/defaulted values.
- **Composability** — functions nest and combine (`ROUND(EXTRACT(EPOCH FROM age)/86400)`).

## Limitations

- **Functions on columns in `WHERE`/`JOIN` can disable index usage** unless a matching expression index exists.
- **Engine divergence** — `AGE`, `||` vs `CONCAT`, `LENGTH` bytes-vs-chars, `SUBSTRING` syntax, and rounding modes differ between Postgres and MySQL. Portability requires care.
- **Floating-point rounding** is surprising; money must use `numeric`/`decimal`.
- **NULL propagation** silently turns computed columns to NULL if you forget to `COALESCE`.
- **Locale/collation** affect `UPPER`/`LOWER`/`TRIM` on non-ASCII text.
- **`AGE` returns an interval**, not a scalar — comparisons and formatting need care.

## Real-world Applications

- **Case-insensitive login/search** — `WHERE lower(email) = lower(?)` with a `lower(email)` index.
- **Display formatting** — `CONCAT`/`INITCAP` for full names, `ROUND(price, 2)` for money display.
- **Data cleaning** — `TRIM` stray whitespace, `REPLACE` bad characters, `LOWER` to normalize.
- **Reporting by period** — `EXTRACT(YEAR/MONTH ...)` or `date_trunc` to group sales by month.
- **Age / tenure calculations** — `AGE(birth_date)` or `EXTRACT(YEAR FROM AGE(...))`.
- **Safe division** — `total / NULLIF(count, 0)` in KPI/ratio queries.
- **Default values in output** — `COALESCE(phone, 'N/A')`, `COALESCE(discount, 0)`.
- **Deduping "empty means null"** — `NULLIF(trim(input), '')` to convert blank strings to NULL.

## Interview Questions

**Beginner**
1. Name a string, a numeric, and a date function and say what each does.
2. What does `LENGTH('hello')` return?

**Intermediate**
3. What's the difference between `CEIL`, `FLOOR`, and `ROUND`?
4. What does `COALESCE` do, and give a practical use.
5. How do you extract the year from a date column?

**Advanced**
6. Why can `WHERE lower(email) = ?` be slow, and how do you fix it?
7. Explain how `COALESCE` and `NULLIF` are defined in terms of `CASE`, and why short-circuiting matters.
8. Why does `AGE` return "years/months/days" instead of a number, and how would you get whole years?

**Scenario-based**
9. A ratio query occasionally errors with "division by zero". How do you fix it in SQL?
10. Users complain that "José" and "JOSÉ" are treated as different on login. How do you make login case-insensitive and index-efficient?

**"Why" questions**
11. Why does `'abc' || NULL` return NULL but `CONCAT('abc', NULL)` return 'abc'?
12. Why should money never be stored/rounded as floating point?

**Comparison questions**
13. `COALESCE` vs `NULLIF` — what's the difference?
14. `CONCAT` vs the `||` operator; `LENGTH` vs `CHAR_LENGTH` (Postgres vs MySQL).

## Model Answers

**1. One of each.**
String: `UPPER(s)` uppercases text. Numeric: `ABS(n)` returns the absolute value (magnitude, always non-negative). Date: `CURRENT_DATE` returns today's date. Each is a scalar function — one input value per row, one output value — evaluated during query execution so the transformation happens inside the database rather than in application code.

**2. `LENGTH('hello')`.**
It returns `5` — the number of characters. Caveat for portability: in PostgreSQL `LENGTH` counts *characters*, but in MySQL `LENGTH` counts *bytes*, so a multibyte string like `'café'` returns 5 bytes in MySQL's `LENGTH` but 4 with `CHAR_LENGTH`. When working with Unicode text in MySQL, prefer `CHAR_LENGTH` for a true character count.

**3. `CEIL` vs `FLOOR` vs `ROUND`.**
`CEIL(x)` rounds *up* to the smallest integer ≥ x (`CEIL(4.1)=5`, `CEIL(-4.9)=-4`). `FLOOR(x)` rounds *down* to the largest integer ≤ x (`FLOOR(4.9)=4`, `FLOOR(-4.1)=-5`). `ROUND(x)` rounds to the *nearest* integer (or to a given number of decimals with a second argument), using round-half-away-from-zero for `numeric`. So they differ in direction: ceil always up, floor always down, round to nearest. Note ceil/floor of negatives surprises people — "up" means toward positive infinity, not toward larger magnitude.

**4. `COALESCE`.**
`COALESCE(a, b, c, ...)` returns the first non-NULL argument scanning left to right, or NULL if all are NULL. It's the standard way to supply defaults for missing data. Practical uses: `COALESCE(nickname, full_name, 'Guest')` to pick a display name, and — critically — guarding arithmetic: `price * COALESCE(quantity, 0)` so a NULL quantity yields 0 instead of nulling the whole expression (since `price * NULL` is NULL). It's cleaner and standard-portable compared to writing an explicit `CASE`.

**5. Extract the year.**
```sql
SELECT EXTRACT(YEAR FROM hire_date) FROM employees;   -- portable-ish
-- Postgres also: date_part('year', hire_date)
-- MySQL shorthand: YEAR(hire_date)
```
`EXTRACT(field FROM source)` pulls a single component (YEAR, MONTH, DAY, HOUR, DOW, EPOCH, ...) out of a date/timestamp/interval and returns it as a number. It's the standard, cross-engine way; `date_part` (Postgres) and `YEAR()` (MySQL) are equivalents/shorthands.

**6. Slow `lower(email)` filter.**
A plain B-tree index on `email` stores the *original* values, so a predicate on `lower(email)` doesn't match the index — the engine must compute `lower()` for every row (a full scan) before comparing. The fix is an **expression (functional) index** that stores the computed value: `CREATE INDEX idx_users_lower_email ON users (lower(email));`. After that, `WHERE lower(email) = lower($1)` is sargable and uses the index directly. The general rule: any function you apply to a column in a `WHERE`/`JOIN` predicate needs a matching expression index (or a rewrite that leaves the column bare) to stay fast. In Postgres you might alternatively use the `citext` type or an `ILIKE`/collation approach, but the expression index is the most direct.

**7. `COALESCE`/`NULLIF` as `CASE`.**
Both are syntactic sugar over `CASE`. `COALESCE(a, b, c)` is exactly `CASE WHEN a IS NOT NULL THEN a WHEN b IS NOT NULL THEN b ELSE c END`, and `NULLIF(a, b)` is `CASE WHEN a = b THEN NULL ELSE a END`. Short-circuiting matters because `CASE` evaluates conditions in order and stops at the first match — so `COALESCE` does not evaluate later arguments once it finds a non-NULL one. That means `COALESCE(cheap_col, expensive_function())` skips the expensive call whenever `cheap_col` is non-null, and it's safe to put a potentially error-raising or costly expression later in the list. It also means each argument should be independently valid, since any of them *might* be evaluated.

**8. Why `AGE` returns an interval.**
`AGE` computes a *calendar* difference and returns it as a symbolic `interval` of years, months, and days — e.g., "36 years 5 mons 24 days" — because the gap between two dates isn't a fixed number of seconds: months have 28–31 days and years include leap days. Expressing it as years/months/days is the human-meaningful, calendar-correct answer. If you need a plain integer number of whole years (e.g., legal age), extract it: `EXTRACT(YEAR FROM AGE(birth_date))`. For a raw day count use `CURRENT_DATE - birth_date` (Postgres date subtraction yields integer days), and for seconds use `EXTRACT(EPOCH FROM interval)`. In MySQL, use `TIMESTAMPDIFF(YEAR, birth_date, CURDATE())` for whole years.

**9. Division-by-zero fix.**
Wrap the divisor in `NULLIF(divisor, 0)`. `NULLIF(count, 0)` returns NULL when `count` is 0, and dividing by NULL yields NULL instead of raising an error:
```sql
SELECT total_revenue / NULLIF(order_count, 0) AS avg_order_value FROM stats;
```
If you'd rather show 0 (or another default) than NULL when the denominator is zero, combine both functions: `COALESCE(total / NULLIF(count, 0), 0)`. This is the idiomatic, branch-free way to make ratio queries robust — no `CASE`, no application-side guard.

**10. Case-insensitive login.**
Normalize both sides and back it with an expression index. Query with `WHERE lower(email) = lower($1)` and create `CREATE INDEX ON users (lower(email))` so the lookup stays index-fast. For accent-insensitivity ("José" vs "JOSE"), you additionally need to strip diacritics — e.g., Postgres `unaccent(lower(email))` with a matching functional index, or store a normalized column. A cleaner Postgres option is the `citext` extension, which gives a case-insensitive text type so plain `=` and a normal index behave case-insensitively. The key point interviewers look for: normalize consistently on write and read, and index the *same expression* you filter on.

**11. `||` vs `CONCAT` with NULL.**
The SQL-standard concatenation operator `||` propagates NULL: if any operand is NULL the result is NULL, because concatenating with an unknown value yields an unknown string. `'abc' || NULL` is therefore NULL in Postgres. `CONCAT`, by design/convention, treats NULL arguments as empty strings, so `CONCAT('abc', NULL)` returns `'abc'`. This is a deliberate ergonomic choice — `CONCAT` is forgiving about missing pieces, which is usually what you want when building display strings, whereas `||` is strict. If you want `||`-style building without NULL wiping the result, wrap operands in `COALESCE(col, '')`.

**12. Why not float for money.**
Binary floating point (`float`/`double`, IEEE 754) cannot represent most decimal fractions exactly — `0.1 + 0.2` famously isn't `0.3`. Accumulated rounding errors corrupt sums, and float rounding uses round-half-to-even, which can produce results that don't match human/accounting expectations. Money requires exact decimal arithmetic and predictable rounding, so you store it as `NUMERIC`/`DECIMAL` (arbitrary-precision, exact) — or as integer minor units (cents). Then `ROUND(amount, 2)` behaves deterministically. Using floats for currency is a classic bug that surfaces as pennies that don't reconcile.

**13. `COALESCE` vs `NULLIF`.**
They're near-opposites. `COALESCE` *removes* NULLs from the output — it returns the first non-NULL of its arguments, so it turns NULL into a real value (a default). `NULLIF` *introduces* NULL — `NULLIF(a, b)` returns NULL when `a` equals `b`, otherwise returns `a`, so it converts a specific sentinel value into NULL. You use `COALESCE` to supply defaults for missing data; you use `NULLIF` to neutralize a special value, most commonly `NULLIF(divisor, 0)` to dodge division-by-zero, or `NULLIF(trim(text), '')` to treat empty strings as NULL. They're frequently combined: `COALESCE(x / NULLIF(y, 0), 0)`.

**14. `CONCAT` vs `||`; `LENGTH` vs `CHAR_LENGTH`.**
`CONCAT(a, b, ...)` is a function available in both Postgres and MySQL and treats NULL as empty string. `||` is the SQL-standard concatenation operator (Postgres, Oracle) that propagates NULL; in MySQL, `||` by default means logical OR (unless `PIPES_AS_CONCAT` SQL mode is set), so portable code prefers `CONCAT`. For length: in PostgreSQL, `LENGTH` and `CHAR_LENGTH` both count *characters*. In MySQL, `LENGTH` counts *bytes* while `CHAR_LENGTH` (a.k.a. `CHARACTER_LENGTH`) counts characters — so for multibyte/UTF-8 text they diverge, and `CHAR_LENGTH` is the safe choice for a true character count. These small divergences are exactly the kind of portability detail senior engineers watch for.

## Common Mistakes

- **Applying a function to an indexed column in `WHERE`** without an expression index — silently forces full scans.
- **Assuming `LENGTH` = character count in MySQL** — it's bytes; use `CHAR_LENGTH`.
- **Forgetting NULL propagation** — `price * quantity` becomes NULL when `quantity` is NULL; wrap with `COALESCE`.
- **Using floats for money** and being surprised by rounding drift.
- **`CEIL`/`FLOOR` on negatives** — misjudging direction ("up" = toward +infinity).
- **Relying on `AGE`/`||`/`ILIKE` in MySQL** where they don't exist or behave differently.
- **Not guarding division** — omitting `NULLIF(divisor, 0)` and hitting divide-by-zero.
- **Expecting `CURRENT_TIMESTAMP` to advance** within a transaction in Postgres — it's frozen at transaction start.
- **Treating blank strings and NULL as the same** without `NULLIF(trim(x), '')`.

## Related Concepts

- **Three-valued logic & NULL semantics** — the foundation for `COALESCE`/`NULLIF` and propagation rules.
- **Expression / functional indexes** — keeping function-based predicates fast.
- **Aggregate vs scalar functions** — `SUM`/`AVG`/`COUNT` vs per-row functions.
- **`CASE` expressions** — the general conditional that `COALESCE`/`NULLIF` desugar to.
- **`date_trunc`, `to_char`, `INTERVAL` arithmetic** — richer date handling beyond `EXTRACT`/`AGE`.
- **`NUMERIC`/`DECIMAL` vs floating point** — exact money math.
- **Collations & `citext`/`unaccent`** — case- and accent-insensitive text.
- **Window functions** — combining scalar functions with `OVER (...)` analytics.


---

# Aggregate Functions

## What is it?

An **aggregate function** takes a *set* of rows as input and collapses (aggregates) them into a **single scalar value**. Where a scalar function like `UPPER(name)` runs once per row and returns one value per row, an aggregate function runs once per *group of rows* and returns one value for the entire group.

The core SQL aggregate functions defined by the ANSI standard are:

| Function | Purpose | Ignores NULL? |
|---|---|---|
| `COUNT(*)` | Counts rows | No — counts every row |
| `COUNT(col)` | Counts non-NULL values in `col` | Yes |
| `COUNT(DISTINCT col)` | Counts distinct non-NULL values | Yes |
| `SUM(col)` | Adds numeric values | Yes |
| `AVG(col)` | Arithmetic mean of numeric values | Yes |
| `MIN(col)` | Smallest value | Yes |
| `MAX(col)` | Largest value | Yes |

When used with no `GROUP BY`, an aggregate treats the *whole table* (after `WHERE`) as one single group and returns exactly one row.

```sql
-- One group = the entire employees table
SELECT COUNT(*)      AS total_rows,
       COUNT(bonus)  AS rows_with_bonus,
       SUM(salary)   AS payroll,
       AVG(salary)   AS avg_salary,
       MIN(salary)   AS lowest,
       MAX(salary)   AS highest
FROM   employees;
```

## Why is it needed?

From first principles: a relational table is a *set*, and questions about data are very often questions about *properties of a set* rather than individual members — "how many customers do we have?", "what is our total revenue?", "what is the average order value?". Without aggregation you would have to pull every row out of the database and compute these numbers in application code, which means:

1. **Network cost** — shipping a million rows to compute a single number is enormously wasteful.
2. **Correctness** — the database engine computes aggregates with well-defined NULL and overflow semantics; ad-hoc application loops drift and diverge.
3. **Optimizability** — the engine can satisfy `MAX(indexed_col)` by reading one entry of a B-tree index instead of scanning the table.

Aggregates are the foundation of all reporting, analytics, dashboards, and business intelligence. They are the "reduce" half of the "map/reduce" you constantly do over relational data.

## How does it work?

Logically, an aggregate query is processed like this:

1. `FROM` produces the working set of rows.
2. `WHERE` filters rows (this happens **before** aggregation — a row removed by `WHERE` never reaches the aggregate).
3. Rows are partitioned into groups (one group total if there's no `GROUP BY`).
4. The aggregate function is applied to each group, folding many rows into one value.
5. `HAVING` filters the resulting groups.
6. `SELECT` projects the final columns.

The three flavors of `COUNT` are the classic interview trap, so let's be precise. Given this table:

```
id | dept | bonus
---+------+------
 1 | A    | 100
 2 | A    | NULL
 3 | B    | 100
 4 | B    | 200
 5 | B    | NULL
```

```sql
SELECT COUNT(*)              -- 5  : every row
     , COUNT(bonus)          -- 3  : non-NULL bonus values (rows 1,3,4)
     , COUNT(DISTINCT bonus) -- 2  : distinct non-NULL values (100, 200)
FROM t;
```

- `COUNT(*)` counts **rows**. It does not look at any column value, so NULLs are irrelevant. This is the correct choice for "how many rows".
- `COUNT(bonus)` counts **non-NULL values** in that column. Rows 2 and 5 are skipped.
- `COUNT(DISTINCT bonus)` first removes NULLs, then de-duplicates, giving 2.

**NULL handling** is the unifying theme. With the sole exception of `COUNT(*)`, every aggregate *silently ignores NULLs*. This has important consequences:

```sql
-- Column: bonus = (100, NULL, 200)
SELECT SUM(bonus),        -- 300   (NULL skipped, not treated as 0)
       AVG(bonus),        -- 150   (300 / 2, NOT 300 / 3!)
       COUNT(bonus),      -- 2
       MIN(bonus),        -- 100
       MAX(bonus)         -- 200
FROM t;
```

Note `AVG` divides by the count of **non-NULL** values (2), not the row count (3). If you semantically want NULLs to count as zero, you must be explicit:

```sql
AVG(COALESCE(bonus, 0))   -- 100  (300 / 3) -- treats missing bonus as 0
```

**Empty-set behavior** is another edge case: over zero rows, `COUNT` returns `0`, but `SUM`, `AVG`, `MIN`, and `MAX` all return `NULL` (not 0 and not an error).

```sql
SELECT SUM(salary), COUNT(*) FROM employees WHERE 1=0;
-- SUM => NULL, COUNT => 0
```

## Internal Working

Under the hood, aggregation is a *stateful fold* over the input rows. Each aggregate maintains a small accumulator (its "transition state") that is updated once per row, then finalized.

- `COUNT(*)`: state is an integer counter, `+1` per row.
- `COUNT(col)`: `+1` per row **where the value is not NULL**.
- `SUM`: a running total; NULL inputs skip the update.
- `AVG`: usually stored as a pair `(running_sum, running_count)`; the final value is `running_sum / running_count`. This is why AVG naturally ignores NULLs — they never increment either accumulator.
- `MIN` / `MAX`: a running extreme value, updated with a comparison.

PostgreSQL exposes this model directly: an aggregate is defined by a state type, an `sfunc` (state transition function) called per row, and an optional `finalfunc`. `AVG` for `bigint`, for instance, accumulates into an internal `{sum, count}` structure.

**Two execution strategies** the planner chooses between:

1. **Hash aggregation** — build an in-memory hash table keyed by the grouping columns; each entry holds the accumulator. One pass over the data, no sort required. Great when the number of distinct groups fits in `work_mem`.

2. **Sorted / grouped aggregation** — sort (or read a pre-sorted index) so that identical group keys are adjacent, then aggregate each run in a streaming fashion. Cheap if a suitable index already provides the order, and it uses bounded memory.

**Index optimizations** matter here:

```sql
-- With a B-tree index on salary, MAX(salary) reads ONE index entry:
SELECT MAX(salary) FROM employees;   -- O(log n), no table scan
```

Postgres rewrites unqualified `MIN`/`MAX` on an indexed column into an `ORDER BY ... LIMIT 1` index probe. `COUNT(*)` on the whole table, by contrast, generally still needs to visit rows (in Postgres, due to MVCC visibility; in MySQL's InnoDB likewise). MySQL's older MyISAM engine cached the row count, making `COUNT(*)` O(1) — a historically famous difference.

## Advantages

- **Massive data reduction at the source** — compute a single number from billions of rows without moving them across the network.
- **Declarative and optimizable** — you state *what* you want; the planner picks hash vs. sort, uses indexes, and can parallelize.
- **Correct, standardized NULL semantics** — consistent behavior instead of hand-rolled loops.
- **Composability** — aggregates combine with `GROUP BY`, `HAVING`, window functions, and subqueries to express rich analytics.

## Limitations

- **NULL semantics surprise people** — `AVG` ignoring NULLs vs. treating them as 0 is a frequent, silent bug.
- **`COUNT(DISTINCT ...)` is expensive** — it must materialize or sort distinct values; on huge cardinalities it is slow and memory-hungry. Approximate alternatives (`APPROX_COUNT_DISTINCT`, HyperLogLog) exist in analytics engines.
- **Loss of row-level detail** — aggregation collapses rows; you cannot see the underlying members without window functions or a self-join.
- **Overflow** — `SUM` over many large integers can overflow the column type; use a wider type or `SUM(col::bigint)`.
- **You cannot mix aggregated and non-aggregated columns freely** (covered under GROUP BY).

## Real-world Applications

- **Financial reporting** — `SUM(amount)` for revenue, `AVG(order_total)` for average order value.
- **Dashboards / KPIs** — daily active users via `COUNT(DISTINCT user_id)`.
- **Data quality checks** — `COUNT(*) - COUNT(email)` reveals how many rows have a missing email.
- **Leaderboards / extremes** — `MAX(score)`, `MIN(response_time)`.
- **Capacity planning** — `MAX(concurrent_sessions)` per hour.

## Interview Questions

**Beginner**
1. What is the difference between `COUNT(*)` and `COUNT(column)`?
2. Do aggregate functions include NULL values in their calculation?

**Intermediate**
3. What does `AVG(col)` return if some values in `col` are NULL, and how would you make NULLs count as zero?
4. What do `SUM`, `COUNT`, and `MAX` each return when run over a set of zero rows?

**Advanced**
5. How does a database compute `MAX(indexed_column)` efficiently, and why can `COUNT(*)` be more expensive than you'd expect?

**Scenario-based**
6. A report shows the "average bonus" as 150 but finance insists it should be 100. Explain what's happening and how to fix it.

**"Why" questions**
7. Why does `COUNT(*)` include NULL-bearing rows while `COUNT(col)` does not?

**Comparison questions**
8. Compare `COUNT(DISTINCT col)` against `COUNT(col)` in both meaning and performance.

## Model Answers

**1. `COUNT(*)` vs `COUNT(column)`.**
`COUNT(*)` counts rows. It is defined over the row itself, not any particular column value, so it never inspects data and never skips anything — if a group has five rows, `COUNT(*)` is five. `COUNT(column)` counts the number of rows in which `column` is **not NULL**. So if two of those five rows have a NULL in that column, `COUNT(column)` returns three. The practical rule: use `COUNT(*)` when the question is "how many records", and `COUNT(column)` when the question is "how many records actually have a value for this attribute". A common real use is `COUNT(*) - COUNT(phone_number)` to measure missing data.

**2. Do aggregates include NULLs?**
With one exception, no. `SUM`, `AVG`, `MIN`, `MAX`, and `COUNT(col)` all ignore NULLs entirely — a NULL input simply doesn't participate in the fold. The single exception is `COUNT(*)`, which counts rows regardless of their contents. This is by design: NULL means "unknown", and it would be wrong to, say, treat an unknown salary as zero when summing payroll. The consequence you must internalize is that `AVG` divides by the count of *non-NULL* values, so ignoring NULLs can change the denominator.

**3. `AVG` with NULLs.**
`AVG(col)` computes `SUM(col) / COUNT(col)`, and both of those ignore NULLs, so the NULL rows are excluded from *both* numerator and denominator. If the values are `(100, NULL, 200)`, `AVG` is `300/2 = 150`, not `300/3 = 100`. If your business definition says a missing value should count as zero, make it explicit with `AVG(COALESCE(col, 0))`, which yields `300/3 = 100`. The lesson is that "average" is ambiguous until you decide what a missing value means, and SQL's default is "exclude it".

**4. Aggregates over zero rows.**
This trips up many engineers. `COUNT` (any form) returns `0` because counting nothing is legitimately zero. But `SUM`, `AVG`, `MIN`, and `MAX` all return `NULL` — the sum of an empty set is undefined in SQL, not zero. This matters when a `WHERE` clause happens to match no rows: `SELECT SUM(amount) FROM sales WHERE region='Mars'` gives NULL, and if you then do arithmetic on it you'll propagate NULL. Guard with `COALESCE(SUM(amount), 0)` when you need a numeric zero.

**5. `MAX(indexed_column)` vs `COUNT(*)`.**
If there is a B-tree index on the column, `MAX` (or `MIN`) does not need to scan the table at all: a B-tree keeps keys in sorted order, so the maximum is simply the right-most leaf entry. Postgres rewrites this into an index probe equivalent to `ORDER BY col DESC LIMIT 1`, which is O(log n). `COUNT(*)`, however, must determine how many rows are *currently visible* to your transaction. Under MVCC (Postgres, InnoDB) different transactions see different row versions, so the engine generally has to walk the rows or an index to check visibility — it cannot trust a single cached counter. That's why `COUNT(*)` on a large table can be surprisingly costly, whereas `MAX(indexed_col)` is nearly free. (MyISAM historically cached an exact row count because it lacks MVCC, making its `COUNT(*)` O(1).)

**6. "Average bonus" scenario.**
The report is computing `AVG(bonus)`, and some employees have `NULL` bonus (they received none, recorded as NULL). Because `AVG` ignores NULLs, it's averaging only over employees who *have* a bonus, inflating the number to 150. Finance wants the average across *all* employees, treating no-bonus as 0. The fix is `AVG(COALESCE(bonus, 0))`, or equivalently `SUM(bonus) / COUNT(*)`. This is fundamentally a semantic disagreement about what NULL means — "not applicable" vs. "zero" — and the SQL must encode the intended meaning explicitly.

**7. Why `COUNT(*)` counts NULL rows but `COUNT(col)` doesn't.**
`COUNT(*)` is defined as the cardinality of the group — it answers "how many rows", a property of the set that is independent of any column's contents, so NULLs are irrelevant. `COUNT(col)` is defined as "how many rows have a known (non-NULL) value for `col`". Since NULL means "no known value", such rows correctly do not add to that count. They answer different questions: existence of a *row* versus existence of a *value*. Conflating them causes bugs, which is exactly why interviewers ask.

**8. `COUNT(DISTINCT col)` vs `COUNT(col)`.**
`COUNT(col)` counts non-NULL values including duplicates; `COUNT(DISTINCT col)` counts distinct non-NULL values. Meaning: if `col` is `(A, A, B, NULL)`, `COUNT(col) = 3` and `COUNT(DISTINCT col) = 2`. Performance-wise they are very different. `COUNT(col)` is a simple counter incremented per row — cheap, streamable, O(n) with O(1) memory. `COUNT(DISTINCT col)` must eliminate duplicates, which requires either sorting the values or building a hash set of everything seen; that is more CPU and can consume memory proportional to the number of distinct values. On high-cardinality columns over billions of rows it becomes a real bottleneck, which is why analytics systems offer approximate distinct counts (HyperLogLog) that trade a small error for constant memory.

## Common Mistakes

- **Assuming `AVG` divides by row count.** It divides by the count of non-NULL values. Use `COALESCE` if you want NULLs as zero.
- **Expecting `SUM`/`MAX` to return 0 over an empty set.** They return NULL; wrap in `COALESCE(..., 0)`.
- **Using `COUNT(col)` when you meant `COUNT(*)`** (or vice versa) and silently dropping NULL rows from your totals.
- **`COUNT(DISTINCT a, b)` portability** — supported in MySQL, but in Postgres you write `COUNT(DISTINCT (a, b))` using a row constructor.
- **Integer overflow in `SUM`** — cast to a wider type when summing many/large integers.
- **Mixing aggregates with bare columns** without a proper `GROUP BY` (see next topic) — an error in standard SQL, silently wrong in old MySQL.

## Related Concepts

- **GROUP BY** — partitions rows so aggregates apply per group.
- **HAVING** — filters the groups aggregates produce.
- **Window functions** (`SUM() OVER (...)`) — aggregate *without* collapsing rows.
- **`FILTER (WHERE ...)` clause** (Postgres) — conditional aggregation, e.g. `COUNT(*) FILTER (WHERE status='paid')`.
- **`DISTINCT`** — de-duplication, the mechanism behind `COUNT(DISTINCT)`.
- **NULL / three-valued logic** — the semantic backbone of aggregate behavior.
- **Approximate aggregation (HyperLogLog)** — scalable distinct counts.

---

# GROUP BY

## What is it?

`GROUP BY` partitions the rows of a table into **buckets (groups)** that share the same value(s) in the grouping column(s), and then applies aggregate functions **once per bucket** instead of once for the whole table. It is the clause that turns "give me one number for everything" into "give me one number *per category*".

```sql
-- Without GROUP BY: one row for the whole table
SELECT COUNT(*) FROM employees;                       -- 1 row

-- With GROUP BY: one row per department
SELECT dept, COUNT(*) FROM employees GROUP BY dept;   -- N rows, one per dept
```

Conceptually the output has exactly **one row per distinct group key**. If you group by `dept`, you get one row per department; if you group by `(dept, job_title)`, you get one row per distinct combination of the two.

## Why is it needed?

Aggregates alone answer questions about the *entire* set. But almost every real analytical question is "per something": revenue *per month*, orders *per customer*, error rate *per service*, average grade *per class*. `GROUP BY` is the mechanism that expresses "per".

From first principles, `GROUP BY` is the SQL realization of a **partition-then-reduce** operation. You take one big set, split it into disjoint subsets by a key (the partition), and reduce each subset to a summary row. Without it you'd have to run one query per category (`WHERE dept='A'`, `WHERE dept='B'`, …), which is unknown in advance, unbounded, and hopelessly inefficient. `GROUP BY` does all partitions in a single pass.

## How does it work?

Multi-column grouping groups by the **tuple** of values:

```sql
SELECT dept, job_title, COUNT(*) AS headcount, AVG(salary) AS avg_pay
FROM   employees
GROUP  BY dept, job_title;
```

This produces one row for each *distinct combination* of `(dept, job_title)` that actually appears in the data. `(Sales, Manager)` and `(Sales, Rep)` are different groups.

**The fundamental rule:** every column in the `SELECT` list must be either (a) part of the `GROUP BY`, or (b) wrapped in an aggregate function. Anything else is illegal because it is *ambiguous*.

```sql
-- ILLEGAL in standard SQL / PostgreSQL:
SELECT dept, name, COUNT(*)
FROM   employees
GROUP  BY dept;
-- ERROR: column "name" must appear in the GROUP BY clause
--        or be used in an aggregate function
```

Why the error? After grouping by `dept`, the group "Sales" is a single output row but contains *many* names (Alice, Bob, Carol). The engine cannot decide which single `name` to put in that one row — the value is not functionally determined by the group key. It's ambiguous, so standard SQL rejects it.

> **PostgreSQL nuance:** if you group by a table's primary key, Postgres *does* let you select any other column of that table without wrapping it, because the PK functionally determines every other column. `GROUP BY orders.id` lets you `SELECT orders.customer_name` legally.

> **MySQL nuance:** historically MySQL allowed the illegal query above and returned an *arbitrary* row's `name` — a notorious source of silent bugs. Modern MySQL enables `ONLY_FULL_GROUP_BY` by default, bringing it in line with the standard.

**GROUPING SETS / ROLLUP / CUBE (brief).** Sometimes you want multiple levels of grouping (subtotals and grand totals) in one query. These extensions produce several grouping levels at once:

```sql
-- ROLLUP: hierarchical subtotals + grand total
SELECT dept, job_title, SUM(salary)
FROM   employees
GROUP  BY ROLLUP (dept, job_title);
-- rows for each (dept, job_title), plus a subtotal per dept
-- (job_title = NULL), plus a grand total (dept = NULL, job_title = NULL)
```

- **`ROLLUP (a, b)`** generates the grouping sets `(a,b)`, `(a)`, and `()` — a hierarchy, good for drill-down subtotals.
- **`CUBE (a, b)`** generates *all* combinations: `(a,b)`, `(a)`, `(b)`, `()` — every possible subtotal.
- **`GROUPING SETS ((a), (b))`** lets you specify exactly which grouping levels you want.

The "extra" total rows carry `NULL` in the columns that were rolled up; the `GROUPING(col)` function distinguishes a real NULL from a subtotal-marker NULL.

## Internal Working

The engine has the same two strategies as aggregation, now keyed by the group columns:

1. **Hash Aggregate.** Build a hash table keyed by the grouping tuple. For each input row, hash its key, find or create the bucket, and update that bucket's accumulators. One pass, no ordering of output. Chosen when the number of distinct groups fits in `work_mem`. If it overflows, Postgres (v13+) spills batches to disk.

```
row (dept=A) -> hash(A) -> bucket A: {count+1, sum+=salary}
row (dept=B) -> hash(B) -> bucket B: {count+1, sum+=salary}
row (dept=A) -> hash(A) -> bucket A: {count+1, sum+=salary}
```

2. **Sorted (GroupAggregate).** Sort the rows by the grouping key so identical keys are adjacent, then stream through, emitting a result row every time the key changes. Uses bounded memory and produces sorted output for free. Especially cheap when an index already supplies the required order, so no explicit sort is needed.

The planner costs both and picks the cheaper. `EXPLAIN` will show `HashAggregate` or `GroupAggregate`. Grouping by a leading indexed column often favors the streaming sort-based plan; grouping by an unindexed column with few distinct values favors the hash plan.

For `ROLLUP`/`CUBE`, the engine can compute the finest grouping first and then aggregate those partial results upward, reusing work rather than rescanning the base table for each level.

## Advantages

- **Expresses "per-category" analytics** in a single declarative statement.
- **Single pass** over the data for all groups, instead of one query per category.
- **Optimizable** — hash vs. sort, index-assisted grouping, parallel aggregation, disk spill for huge cardinalities.
- **ROLLUP/CUBE** compute multi-level subtotals server-side, powering OLAP and pivot-style reports.

## Limitations

- **The GROUP BY rule is strict** — every non-aggregated selected column must be grouped, which can feel verbose for wide tables (mitigated by PK-based functional dependency in Postgres).
- **High-cardinality grouping is expensive** — millions of distinct keys means a large hash table or a large sort.
- **NULLs form their own group** — all NULL keys collapse into a single group, which may or may not be what you want.
- **You lose row detail** — like all aggregation, individual rows disappear into summaries (use window functions if you need both).
- **`CUBE` explodes combinatorially** — n columns yield 2^n grouping sets.

## Real-world Applications

- **Sales by region/month:** `GROUP BY region, date_trunc('month', sold_at)`.
- **Per-user activity:** `GROUP BY user_id` to count sessions or sum spend.
- **Error monitoring:** `GROUP BY service, status_code` to build error-rate dashboards.
- **Financial statements with subtotals:** `ROLLUP (region, product)` for region subtotals and a grand total.
- **Cohort analysis:** grouping by signup week to compare cohorts.

## Interview Questions

**Beginner**
1. What does `GROUP BY` do, and how many rows does it output?
2. Can you select a column that is neither in the `GROUP BY` nor inside an aggregate? Why or why not?

**Intermediate**
3. What does grouping by multiple columns mean, and how many groups result?
4. How are NULL values treated by `GROUP BY`?

**Advanced**
5. Explain the difference between a HashAggregate and a GroupAggregate plan, and when each is chosen.

**Scenario-based**
6. You need per-department subtotals *and* a company-wide grand total in one result set. How do you do it?

**"Why" questions**
7. Why does standard SQL forbid selecting a non-grouped, non-aggregated column, while MySQL historically allowed it?

**Comparison questions**
8. Compare `ROLLUP`, `CUBE`, and `GROUPING SETS`.

## Model Answers

**1. What `GROUP BY` does.**
`GROUP BY` partitions rows into groups that share the same values in the specified column(s) and produces exactly one output row per distinct group. So `GROUP BY dept` over a table with three departments yields three rows. Within each group, aggregate functions like `COUNT` or `SUM` are computed over just that group's rows. It is the SQL way of saying "compute this metric *per* this category". The number of output rows equals the number of distinct grouping-key values present in the data (after `WHERE` filtering).

**2. Selecting a non-grouped, non-aggregated column.**
No — in standard SQL and PostgreSQL it is an error. After grouping by `dept`, a single group like "Sales" corresponds to one output row but many underlying rows with many different `name` values. The engine has no principled way to pick one `name` to represent the group, so the value is undefined. The rule is that every selected column must be *functionally determined by the group* — either it's a grouping key (same for all rows in the group) or it's aggregated (collapsed to one value). The one exception: if you group by a primary key, all other columns of that table are functionally determined and Postgres permits selecting them.

**3. Multi-column grouping.**
Grouping by `(a, b)` groups rows by the *combination* of their `a` and `b` values — the group key is the tuple. The number of groups equals the number of distinct `(a, b)` pairs that actually occur in the data, which is at most `distinct(a) × distinct(b)` but usually far fewer because not all combinations exist. For example `GROUP BY country, city` gives one row per city (since each city belongs to one country), not one per arbitrary country-city cross product.

**4. NULLs and `GROUP BY`.**
For grouping purposes, all NULLs are treated as equal and collapse into a **single group**. This is different from normal SQL equality, where `NULL = NULL` is unknown, not true. So if 10 rows have a NULL `dept`, they form one group of 10, and its key displays as NULL. Be aware this can merge semantically distinct "missing" rows together, and remember the resulting NULL in output can be confused with the subtotal NULLs produced by `ROLLUP`/`CUBE` (use `GROUPING()` to disambiguate).

**5. HashAggregate vs GroupAggregate.**
A **HashAggregate** builds an in-memory hash table keyed by the grouping columns; it scans the input once, updating each group's accumulators, and emits results in no particular order. It's ideal when the number of distinct groups is small enough to fit in `work_mem`. A **GroupAggregate** requires the input to be *sorted* by the grouping key (via an explicit sort or, better, a pre-sorted index); it then streams through, closing out a group each time the key changes. GroupAggregate uses bounded memory and yields ordered output, so the planner prefers it when an index already provides the order or when grouping cardinality is huge (avoiding a giant hash table). The optimizer estimates group count and memory and picks the cheaper plan; `EXPLAIN` reveals which.

**6. Subtotals plus grand total.**
Use `GROUP BY ROLLUP(dept)`:

```sql
SELECT dept, SUM(salary)
FROM   employees
GROUP  BY ROLLUP (dept);
```

This returns one row per department *plus* one extra row where `dept` is NULL holding the company-wide total. `ROLLUP` generates the grouping sets `(dept)` and `()` (the empty set = grand total) in a single pass. To label the total row nicely, use `GROUPING(dept)` — it returns 1 for the super-aggregate row — e.g. `CASE WHEN GROUPING(dept)=1 THEN 'ALL' ELSE dept END`.

**7. Why the standard forbids it.**
The relational model requires that a query's result be well-defined and deterministic. If you select `name` while grouping only by `dept`, the result depends on *which* of many names the engine happens to grab — that's nondeterministic and violates the principle that SQL is declarative, not dependent on physical row order. So the standard rejects it as ambiguous. MySQL historically prioritized convenience and returned an arbitrary value, which produced subtly wrong reports (the displayed name might not even correspond to the aggregated figures). Recognizing the hazard, modern MySQL turns on `ONLY_FULL_GROUP_BY` by default, matching the standard.

**8. ROLLUP vs CUBE vs GROUPING SETS.**
All three produce multiple grouping levels in one query. `GROUPING SETS ((a),(b),())` is the general, explicit form — you list precisely the grouping combinations you want. `ROLLUP(a,b)` is shorthand for a *hierarchy*: it yields `(a,b)`, `(a)`, and `()`, which suits ordered drill-downs like year → quarter → month with subtotals at each level and a grand total. `CUBE(a,b)` yields *every* subset: `(a,b)`, `(a)`, `(b)`, `()` — all possible subtotals, useful for cross-tab/OLAP where you want totals along every dimension. `CUBE` grows as 2^n grouping sets, so it's powerful but can be expensive; `ROLLUP` grows linearly (n+1 sets). Choose `ROLLUP` for hierarchies, `CUBE` for full cross-tabulation, and `GROUPING SETS` when you want an exact custom subset.

## Common Mistakes

- **Selecting an ungrouped column** and expecting a sensible value — an error in standard SQL, an arbitrary value in legacy MySQL.
- **Forgetting that NULLs collapse into one group**, unexpectedly merging "unknown" rows.
- **Using `WHERE` to filter on an aggregate** (`WHERE COUNT(*) > 5`) — that's what `HAVING` is for; `WHERE` runs before grouping.
- **Grouping by a column but ordering by an aggregate alias** in dialects that don't allow it — mind alias scoping.
- **Confusing subtotal NULLs (ROLLUP) with real NULLs** — use `GROUPING()`.
- **Assuming output is sorted** — `GROUP BY` does not guarantee order; add `ORDER BY` explicitly.

## Related Concepts

- **Aggregate functions** — the reducers applied per group.
- **HAVING** — filters groups after aggregation.
- **Window functions with `PARTITION BY`** — group-wise computation that *keeps* individual rows.
- **`DISTINCT`** — `SELECT DISTINCT a` is essentially `GROUP BY a` with no aggregates.
- **ROLLUP / CUBE / GROUPING SETS / GROUPING()** — multi-level and OLAP grouping.
- **`work_mem` and hash spill** — the memory mechanics behind hash aggregation.

---

# HAVING

## What is it?

`HAVING` is a filter that applies to **groups** produced by `GROUP BY`, *after* aggregation has happened. Where `WHERE` filters individual **rows** before they are grouped, `HAVING` filters the aggregated **result rows** based on the value of aggregate functions.

```sql
-- "Show only departments with more than 10 employees"
SELECT dept, COUNT(*) AS headcount
FROM   employees
GROUP  BY dept
HAVING COUNT(*) > 10;
```

The mental model: `WHERE` decides *which rows go into the groups*; `HAVING` decides *which groups survive*.

## Why is it needed?

You cannot put an aggregate in a `WHERE` clause:

```sql
-- ILLEGAL:
SELECT dept, COUNT(*) FROM employees WHERE COUNT(*) > 10 GROUP BY dept;
-- ERROR: aggregate functions are not allowed in WHERE
```

This is not an arbitrary restriction — it's a consequence of *when* things happen. `WHERE` is evaluated **before** rows are grouped, so at that moment `COUNT(*)` for a group does not yet exist; there are no groups yet. There has to be a clause that runs *after* grouping to filter on aggregate results, and that clause is `HAVING`. Without it, "give me only the categories whose total exceeds X" would be inexpressible in a single query.

## How does it work?

The key is the **logical order of execution** of a `SELECT` statement:

```
1. FROM      (+ JOINs)     -- assemble the source rows
2. WHERE                   -- filter individual rows
3. GROUP BY                -- partition rows into groups
4. HAVING                  -- filter the groups
5. SELECT                  -- compute output columns / aggregates
6. ORDER BY                -- sort the result
7. LIMIT / OFFSET          -- take a slice
```

`WHERE` (step 2) happens before `GROUP BY` (step 3), so it can only see raw column values, never aggregates. `HAVING` (step 4) happens after `GROUP BY`, so it *can* see aggregates — the groups and their `COUNT`/`SUM`/etc. already exist.

A query can use both, and idiomatically should:

```sql
SELECT   dept, AVG(salary) AS avg_pay
FROM     employees
WHERE    hire_date >= '2020-01-01'   -- pre-filter rows (uses index, cheap)
GROUP BY dept
HAVING   AVG(salary) > 50000         -- post-filter groups (needs the aggregate)
ORDER BY avg_pay DESC;
```

Read it in execution order: first keep only employees hired since 2020 (`WHERE`), then group those by department, compute each department's average salary, then keep only departments whose average exceeds 50k (`HAVING`), then sort.

**Performance principle:** push as much filtering as possible into `WHERE`, because it runs *first* and shrinks the data before the expensive grouping. Only put a condition in `HAVING` if it genuinely depends on an aggregate. Filtering a non-aggregate in `HAVING` still works but is wasteful — you'd group rows only to throw whole groups away that `WHERE` could have eliminated row-by-row (often using an index) beforehand.

```sql
-- WORKS but SUBOPTIMAL: dept filter belongs in WHERE
SELECT dept, COUNT(*) FROM employees
GROUP BY dept HAVING dept <> 'Temp';

-- BETTER: filter rows before grouping (can use an index on dept)
SELECT dept, COUNT(*) FROM employees
WHERE dept <> 'Temp' GROUP BY dept;
```

## Internal Working

`HAVING` is not a separate scan of the table — it is a predicate applied to the *stream of aggregated group rows* as they emerge from the aggregate operator. In an execution plan you'll see the `HashAggregate`/`GroupAggregate` node produce group rows, and a `Filter` condition on that node (or a filter node just above it) applies the `HAVING` predicate, discarding groups that fail.

Because the aggregate values are already computed at that point, evaluating `HAVING COUNT(*) > 10` is just an integer comparison per group — cheap. The cost of aggregation itself (building the hash table or sorting) is unavoidable; `HAVING` only decides which of the finished groups to emit.

Some optimizers perform a useful rewrite: if a `HAVING` predicate references *only* grouping columns and no aggregates (e.g. `HAVING dept <> 'Temp'`), the planner may push it down into `WHERE`, evaluating it before grouping to save work. You should not rely on this — write it in `WHERE` yourself for clarity and portability. Conversely, a genuine aggregate predicate (`HAVING SUM(x) > 100`) cannot be pushed down, since the aggregate doesn't exist until after grouping.

An edge case worth knowing: you can use `HAVING` **without** `GROUP BY`. Then the whole table is one implicit group, and `HAVING` filters that single group — the query returns either one row or zero rows:

```sql
SELECT SUM(amount) FROM sales HAVING SUM(amount) > 1000000;
-- returns the total only if it exceeds a million; otherwise no rows
```

## Advantages

- **Enables aggregate-based filtering** — the only standard way to filter on `COUNT`/`SUM`/`AVG` results.
- **Composes cleanly** with `WHERE`, letting you filter both rows (before) and groups (after) in one query.
- **Expressive** — thresholds like "customers with more than 5 orders" or "products whose total revenue exceeds X" become trivial.

## Limitations

- **Runs after grouping**, so it cannot reduce the work of the aggregation itself — misusing it for row filters wastes effort.
- **Cannot use `SELECT`-list aliases in standard SQL** — `HAVING` is logically evaluated before `SELECT`, so `HAVING total > 10` (where `total` is a SELECT alias) is not portable. (MySQL and Postgres allow it as an extension; Oracle/SQL Server generally require repeating the aggregate expression.)
- **Easy to misuse** — putting non-aggregate conditions in `HAVING` "works" but signals a misunderstanding and can hurt performance.

## Real-world Applications

- **"Power users":** `GROUP BY user_id HAVING COUNT(*) > 100` to find highly active accounts.
- **Duplicate detection:** `GROUP BY email HAVING COUNT(*) > 1` to find duplicate emails.
- **Revenue thresholds:** `GROUP BY product HAVING SUM(revenue) > 1000000` for top products.
- **Data-quality gates:** flag groups whose average or max exceeds a limit.
- **Fraud/anomaly rules:** `GROUP BY card_id HAVING COUNT(*) > 5 AND SUM(amount) > 10000` within a time window.

## Interview Questions

**Beginner**
1. What is the difference between `WHERE` and `HAVING`?
2. Can you use an aggregate function in a `WHERE` clause?

**Intermediate**
3. Given a query that filters both rows and groups, which conditions go in `WHERE` and which in `HAVING`, and why?
4. Can you use `HAVING` without `GROUP BY`? What does it mean?

**Advanced**
5. Explain, using logical query processing order, why `WHERE` cannot reference aggregates but `HAVING` can, and whether you can reference a `SELECT` alias in `HAVING`.

**Scenario-based**
6. Find all email addresses that appear more than once in a `users` table.

**"Why" questions**
7. Why is it a performance anti-pattern to put a non-aggregate filter in `HAVING`?

**Comparison questions**
8. Compare `WHERE` and `HAVING` across: what they filter, when they run, what they can reference, and index usage.

## Model Answers

**1. `WHERE` vs `HAVING`.**
`WHERE` filters individual rows *before* they are grouped; it can reference raw columns but not aggregates. `HAVING` filters groups *after* aggregation; it can reference aggregate results like `COUNT(*)` or `SUM(x)`. In short: `WHERE` decides which rows enter the groups, `HAVING` decides which groups appear in the output. A query often uses both — `WHERE` to cut the data down cheaply first, `HAVING` to apply the aggregate threshold afterward.

**2. Aggregate in `WHERE`?**
No. Aggregate functions are not allowed in `WHERE` because `WHERE` is evaluated before grouping occurs — at that stage the groups, and therefore their `COUNT`/`SUM`, don't exist yet. To filter on an aggregate you must use `HAVING`, which runs after `GROUP BY` when the aggregate values are available.

**3. Which conditions go where.**
Conditions on *raw row values* (dates, statuses, IDs) belong in `WHERE` so they run first and shrink the input before grouping, ideally using an index. Conditions on *aggregate results* (`COUNT(*) > 10`, `SUM(amount) > 1000`) must go in `HAVING` because those values only exist after grouping. Rule of thumb: if the predicate can be evaluated looking at a single row, put it in `WHERE`; if it needs the whole group, put it in `HAVING`. Example: `WHERE status='paid'` (row-level) combined with `HAVING SUM(amount) > 500` (group-level).

**4. `HAVING` without `GROUP BY`.**
Yes. When there's no `GROUP BY`, the entire table (after `WHERE`) is treated as a single implicit group, and `HAVING` filters that one group. The query returns one row if the group satisfies the predicate, or zero rows if it doesn't. For instance `SELECT SUM(amount) FROM sales HAVING SUM(amount) > 1000000` returns the grand total only when it exceeds a million, otherwise an empty result. It's an all-or-nothing filter on the aggregate of the whole table.

**5. Logical order and aliases.**
The logical processing order is `FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT`. Because `WHERE` (step 2) executes before `GROUP BY` (step 3), no groups and hence no aggregates exist when `WHERE` runs, so it cannot reference them. `HAVING` (step 4) runs after grouping, so aggregates are available. This same ordering explains the alias question: `SELECT` (step 5) runs *after* `HAVING`, so column aliases defined in the `SELECT` list don't formally exist yet when `HAVING` is evaluated. Standard SQL therefore requires you to repeat the aggregate expression in `HAVING` rather than use its alias. In practice PostgreSQL and MySQL relax this and let you use the alias, but for portability (and to match Oracle/SQL Server) you should repeat `HAVING SUM(amount) > 100` rather than `HAVING total > 100`. Interestingly, `ORDER BY` (step 6) *can* use `SELECT` aliases because it runs after `SELECT`.

**6. Duplicate emails.**

```sql
SELECT email, COUNT(*) AS occurrences
FROM   users
GROUP  BY email
HAVING COUNT(*) > 1;
```

Group all rows by `email`, count how many rows fall into each email's group, and keep only groups whose count exceeds one — those are the duplicated addresses. This is the canonical duplicate-detection pattern; add more columns to the `GROUP BY` to detect duplicates across a composite key.

**7. Non-aggregate filter in `HAVING` anti-pattern.**
Because `HAVING` runs *after* grouping, a condition placed there is applied only once the engine has already spent effort assembling every group — building the hash table or sorting, and computing aggregates for groups you're about to discard. If that same condition is on a raw column (e.g. `dept <> 'Temp'`), putting it in `WHERE` lets the engine drop those rows *before* grouping, often via an index, so they never inflate the aggregation at all. Same result, less work. Some optimizers auto-push such predicates down, but you shouldn't depend on that; write row filters in `WHERE`.

**8. `WHERE` vs `HAVING` comparison.**

| Aspect | `WHERE` | `HAVING` |
|---|---|---|
| Filters | Individual rows | Groups (aggregated rows) |
| Runs (logical order) | Before `GROUP BY` | After `GROUP BY` |
| Can reference aggregates? | No | Yes |
| Can reference raw columns? | Yes | Only if grouped (or via aggregate) |
| Can use indexes to filter? | Yes, typically | No — operates on computed groups |
| Requires `GROUP BY`? | No | No (whole table = one group) |

The essence: `WHERE` is the cheap, early, row-level, index-friendly filter; `HAVING` is the later, group-level filter that exists specifically so you can threshold on aggregates. Use both, each for its purpose.

## Common Mistakes

- **Putting an aggregate in `WHERE`** — illegal; use `HAVING`.
- **Putting a row-level filter in `HAVING`** — works but wastes effort; move it to `WHERE`.
- **Relying on `SELECT` aliases in `HAVING`** — non-portable; repeat the aggregate expression.
- **Forgetting `HAVING` needs the aggregate spelled out**, e.g. writing `HAVING count > 1` where `count` is an alias in strict dialects.
- **Assuming `HAVING` improves performance** — it filters *after* the expensive aggregation, not before.
- **Confusing "filter rows" with "filter groups"** — the single most common conceptual error.

## Related Concepts

- **`WHERE`** — the row-level counterpart that runs first.
- **`GROUP BY`** — produces the groups `HAVING` filters.
- **Aggregate functions** — what `HAVING` predicates typically reference.
- **Logical query processing order** — the framework that explains all of the above.
- **Window functions** — filtering on ranked results uses a subquery/CTE instead of `HAVING`, since window functions also run after `WHERE`.
- **CTEs / subqueries** — used to filter on aliases or window results that `HAVING` cannot reach.

---

# Joins

## What is it?

A **join** combines rows from two (or more) tables into a single result set, matching rows based on a related column — typically a **key relationship**. It is the operation that reconstitutes the connected picture from data that normalization deliberately split apart.

Relational databases are *normalized*: you don't store the customer's name on every order row; you store a `customer_id` on the order and the name once in a `customers` table. That `customer_id` is a **foreign key** referencing the `customers` table's **primary key** `id`. A join follows that relationship to bring the pieces back together:

```sql
SELECT o.id, o.total, c.name
FROM   orders o
JOIN   customers c ON c.id = o.customer_id;
```

Keys and relationships in one glance:

- **Primary key (PK):** uniquely identifies a row (`customers.id`).
- **Foreign key (FK):** a column that references another table's PK (`orders.customer_id`).
- **Relationship cardinality:** one-to-many (one customer, many orders), one-to-one, or many-to-many (resolved via a junction table with two FKs).

The join predicate (`ON c.id = o.customer_id`) is the condition that decides which rows from each side are "related" and should be combined.

## Why is it needed?

From first principles, normalization removes redundancy to prevent update anomalies — you change a customer's name in exactly one place. But queries need the *denormalized* view: "orders with customer names". Joins are the bridge between the *storage* model (normalized, non-redundant) and the *query* model (combined, human-meaningful).

Without joins you would either:

1. **Denormalize** and store everything redundantly (leading to inconsistency and bloat), or
2. **Query each table separately** and stitch results in application code (many round-trips, N+1 query problems, hand-written matching logic that the database does far better).

Joins let the engine do the matching efficiently, using indexes and choosing among specialized algorithms, all expressed declaratively.

## How does it work?

The join type determines *what happens to rows that have no match on the other side*. This is the entire conceptual axis. Consider two tables:

```
employees                 departments
emp | name  | dept_id     dept_id | dept_name
----+-------+--------     --------+----------
 1  | Alice |   10          10    | Sales
 2  | Bob   |   20          20    | Eng
 3  | Carol |  NULL         30    | HR      <- no employees
```

Alice/Bob match a department; Carol has no dept; HR has no employees. Watch how each join type treats the unmatched Carol and unmatched HR.

### INNER JOIN — only matches

Returns only row pairs that satisfy the predicate. Unmatched rows on *either* side are dropped.

```
   A ∩ B
  ┌───────┐
  │  ▓▓▓  │      only the overlap
  └───────┘
```

```sql
SELECT e.name, d.dept_name
FROM   employees e
INNER JOIN departments d ON d.dept_id = e.dept_id;
-- Alice|Sales, Bob|Eng     (Carol dropped: no dept; HR dropped: no employees)
```

### LEFT (OUTER) JOIN — all left rows, matched right

Returns every row from the **left** table; where there's no right match, right columns are `NULL`.

```
   A (all) + matches from B
  ┌───────┬───┐
  │ ▓▓▓▓▓ │▓▓ │    all of A, overlap filled from B
  └───────┴───┘
```

```sql
SELECT e.name, d.dept_name
FROM   employees e
LEFT JOIN departments d ON d.dept_id = e.dept_id;
-- Alice|Sales, Bob|Eng, Carol|NULL     (Carol kept; HR still absent)
```

The classic **"find rows with no match"** idiom uses `LEFT JOIN ... WHERE right.key IS NULL` — the anti-join:

```sql
-- Employees not assigned to any department
SELECT e.name
FROM   employees e
LEFT JOIN departments d ON d.dept_id = e.dept_id
WHERE  d.dept_id IS NULL;               -- Carol
```

### RIGHT (OUTER) JOIN — all right rows, matched left

The mirror image: every row from the **right** table, left columns NULL where unmatched. `A RIGHT JOIN B` is identical to `B LEFT JOIN A`. Most engineers standardize on LEFT for readability.

```sql
SELECT e.name, d.dept_name
FROM   employees e
RIGHT JOIN departments d ON d.dept_id = e.dept_id;
-- Alice|Sales, Bob|Eng, NULL|HR         (HR kept; Carol absent)
```

### FULL OUTER JOIN — everything, matched where possible

Every row from **both** tables; NULLs fill wherever a side has no match. It's the union of LEFT and RIGHT.

```
   A ∪ B
  ┌───┬───────┬───┐
  │▓▓ │ ▓▓▓▓▓ │▓▓ │   all of A + all of B
  └───┴───────┴───┘
```

```sql
SELECT e.name, d.dept_name
FROM   employees e
FULL OUTER JOIN departments d ON d.dept_id = e.dept_id;
-- Alice|Sales, Bob|Eng, Carol|NULL, NULL|HR
```

> **MySQL note:** MySQL has no `FULL OUTER JOIN`. Emulate it with `LEFT JOIN ... UNION ... RIGHT JOIN`.

### SELF JOIN — a table joined to itself

Nothing special syntactically — you just reference the same table twice with different aliases. Used for hierarchical/adjacency data (employee → manager) or comparing rows within one table.

```sql
-- Each employee with their manager's name (both live in `employees`)
SELECT e.name AS employee, m.name AS manager
FROM   employees e
LEFT JOIN employees m ON m.emp = e.manager_id;
```

The `LEFT` join here keeps the top boss (whose `manager_id` is NULL) in the result with a NULL manager.

### CROSS JOIN — Cartesian product

Every row of A paired with every row of B, no predicate. Result size is `|A| × |B|`.

```sql
SELECT s.size, c.color
FROM   sizes s
CROSS JOIN colors c;      -- all size/color combinations
```

Useful for generating combinations (e.g., all size×color variants), calendars (dates × stores), or test data. Dangerous by accident: an INNER JOIN with a missing/incorrect `ON` degenerates into a cross join and explodes row counts.

### Logical query processing order (where joins fit)

```
1. FROM / JOIN   -- build the combined row set (ON predicate applied here)
2. WHERE         -- filter combined rows
3. GROUP BY
4. HAVING
5. SELECT
6. ORDER BY
7. LIMIT / OFFSET
```

A subtle but crucial point: for **outer** joins, the `ON` predicate and the `WHERE` predicate behave differently. `ON` is applied *while* matching (step 1) and preserves unmatched rows as NULL-extended; `WHERE` runs *afterward* (step 2) and will *discard* those NULL-extended rows if it references the outer side. This is why putting `d.active = true` in `WHERE` silently turns a LEFT JOIN into an INNER JOIN, whereas putting it in `ON` keeps the unmatched left rows.

## Internal Working

The join *type* is a logical specification; the join *algorithm* is how the engine physically computes it. The optimizer picks among three, based on table sizes, indexes, sortedness, and memory. Understanding them is the difference between a junior and a senior engineer.

### 1. Nested Loop Join

```
for each row r in OUTER table:
    for each row s in INNER table:
        if predicate(r, s): emit (r, s)
```

The straightforward double loop. Naively it is `O(N × M)`. But when the inner side has an **index** on the join column, the inner "scan" becomes an index lookup, making it `O(N × log M)` — the *indexed nested loop*, which is extremely fast when the outer side is small.

- **Best when:** one side is small (few outer rows) and the inner side has an index on the join key. Also the only option for non-equality joins (`<`, `>`, ranges) and cross joins.
- **Worst when:** both sides are large and unindexed — the `N × M` blowup.

### 2. Hash Join

```
-- Build phase: hash the smaller ("build") input by join key
for each row s in BUILD table:
    hashtable[ key(s) ].append(s)
-- Probe phase: scan the larger ("probe") input
for each row r in PROBE table:
    for each s in hashtable[ key(r) ]:
        emit (r, s)
```

Build an in-memory hash table on the smaller input keyed by the join column, then stream the larger input, probing the hash table. Roughly `O(N + M)` — linear.

- **Best when:** large, unindexed inputs joined on **equality**. The workhorse for big analytical joins.
- **Requires:** an equi-join (`=`), and memory for the build side (spills to disk in batches — "grace hash join" — if it exceeds `work_mem`).
- **Cannot do:** inequality join predicates.

### 3. Merge Join (Sort-Merge)

```
-- Both inputs sorted by the join key, then advance in lock-step
i, j = 0, 0
while i < |A| and j < |B|:
    if A[i].key == B[j].key: emit matches (handle duplicate runs)
    elif A[i].key <  B[j].key: i++       -- advance the smaller side
    else:                      j++
```

Sort both inputs on the join key (or read them pre-sorted from an index), then walk both in a single synchronized pass like a zipper. Cost is dominated by the sort: `O(N log N + M log M)`, or just `O(N + M)` if both inputs are already sorted (e.g., both come from index scans on the join column).

- **Best when:** inputs are already sorted on the join key, or are huge and the output is needed sorted anyway; also handles large-to-large equi-joins where a hash table wouldn't fit well.
- **Handles:** equality naturally; range/`<`,`>` predicates in some engines.
- **Cost:** the sorts, if not already provided by indexes.

### Choosing between them (mental table)

| Algorithm | Time | Needs | Sweet spot |
|---|---|---|---|
| Nested loop (indexed) | O(N · log M) | index on inner | small outer + indexed inner; any predicate |
| Hash join | O(N + M) | equi-join, memory | large unindexed inputs, equality |
| Merge join | O(N + M) if pre-sorted | sorted inputs | already-sorted / index-ordered inputs, huge equi-joins |

`EXPLAIN` (Postgres) or `EXPLAIN FORMAT=JSON` / `EXPLAIN ANALYZE` (MySQL 8+, which added hash joins) reveals which was chosen. The optimizer estimates row counts from statistics; stale statistics are a leading cause of a bad algorithm choice (e.g. a nested loop over millions of rows). Keeping stats fresh (`ANALYZE`) and having the right indexes is how you steer it.

## Advantages

- **Normalization without pain** — store data once, query it combined; no redundancy, no update anomalies.
- **Declarative and optimized** — you specify the relationship; the engine picks nested-loop/hash/merge, uses indexes, parallelizes, and reorders joins.
- **Expressive relationships** — one-to-many, many-to-many (via junction tables), and self-referential hierarchies all expressible.
- **Set-based, single round trip** — replaces application-side N+1 loops with one efficient server-side operation.

## Limitations

- **Performance cliffs** — a missing index or stale statistics can turn a fast indexed nested loop into a catastrophic `N × M` scan.
- **Accidental Cartesian products** — a forgotten or wrong `ON` clause explodes row counts.
- **Outer-join semantics are subtle** — the `ON` vs `WHERE` distinction silently changes results; NULLs from outer joins complicate downstream logic and aggregation.
- **Many-way joins are hard to optimize** — join order is a combinatorial search; the planner may pick poorly on complex queries.
- **Fan-out / duplication** — joining a one-to-many relationship multiplies rows, inflating `SUM`/`COUNT` if you're not careful (join-then-aggregate traps).

## Real-world Applications

- **Orders + customers + products** — the canonical e-commerce query joining across the normalized schema.
- **Anti-joins for gaps** — `LEFT JOIN ... IS NULL` to find users with no orders, products never sold, records missing a related row.
- **Org charts / bill-of-materials** — self-joins (or recursive CTEs) over adjacency lists.
- **Many-to-many** — students↔courses via an `enrollments` junction table, joined on both sides.
- **Data reconciliation** — `FULL OUTER JOIN` two sources to find rows present in one but not the other.
- **Combinations / scheduling** — `CROSS JOIN` of dates × resources to build a complete grid, then LEFT JOIN actuals onto it.

## Interview Questions

**Beginner**
1. What is a join, and what is the difference between `INNER JOIN` and `LEFT JOIN`?
2. What is a `CROSS JOIN` and when would you use it?

**Intermediate**
3. Explain `RIGHT JOIN` and `FULL OUTER JOIN`, and how you'd emulate FULL OUTER in MySQL.
4. What is a self join and give a realistic use case.

**Advanced**
5. Describe nested-loop, hash, and merge join algorithms, their costs, and when the optimizer picks each.

**Scenario-based**
6. You wrote a `LEFT JOIN` but a condition in the `WHERE` clause on the right table made unmatched rows disappear. What happened and how do you fix it?

**"Why" questions**
7. Why do we normalize and then join, instead of just storing everything in one wide table?

**Comparison questions**
8. Compare `INNER`, `LEFT`, `RIGHT`, and `FULL OUTER` joins in terms of which unmatched rows they preserve.

## Model Answers

**1. Join, and INNER vs LEFT.**
A join combines rows from two tables based on a related column, usually a foreign-key/primary-key relationship, producing a single result set. An `INNER JOIN` returns only the rows that have a match on both sides — unmatched rows from either table are dropped. A `LEFT JOIN` (left outer join) returns *all* rows from the left table, and for those with no match on the right it fills the right-side columns with NULL. So if you join `employees` to `departments`, INNER gives only employees who have a valid department, while LEFT gives every employee including those whose department is NULL or unmatched (with NULL department columns). The choice hinges on whether you want to keep unmatched left rows.

**2. CROSS JOIN.**
A `CROSS JOIN` produces the Cartesian product — every row of the first table paired with every row of the second, with no join predicate. If A has 10 rows and B has 5, the result has 50. You use it deliberately to generate all combinations: sizes × colors for product variants, dates × stores to build a reporting grid, or numbers/calendars for gap-filling. It's also what you accidentally get when you forget the `ON` clause, which is why unexpected row-count explosions often trace back to an unintended cross join.

**3. RIGHT and FULL OUTER, and MySQL emulation.**
A `RIGHT JOIN` keeps all rows from the right table, NULL-filling the left where there's no match — it's the mirror of LEFT, and `A RIGHT JOIN B` equals `B LEFT JOIN A`. A `FULL OUTER JOIN` keeps all rows from *both* tables, NULL-filling either side where a match is missing; it's the union of the LEFT and RIGHT results. MySQL doesn't support FULL OUTER JOIN, so you emulate it by unioning a LEFT and a RIGHT join: `SELECT ... A LEFT JOIN B ON ... UNION SELECT ... A RIGHT JOIN B ON ...`. The `UNION` (not `UNION ALL`) deduplicates the overlapping matched rows so they appear once.

**4. Self join.**
A self join is a table joined to itself, using two different aliases so the engine treats them as two logical tables. The classic case is an adjacency-list hierarchy: an `employees` table where each row has a `manager_id` pointing at another row's `emp` id. Joining `employees e` to `employees m ON m.emp = e.manager_id` pairs each employee with their manager, letting you select `e.name AS employee, m.name AS manager`. Using a LEFT self join keeps the top executive (whose `manager_id` is NULL) with a NULL manager. Self joins also compare rows within a table, e.g. finding pairs of products with the same price.

**5. Join algorithms.**
There are three physical algorithms. A **nested loop** iterates the outer table and, for each outer row, scans the inner table for matches; naively O(N×M), but if the inner side has an index on the join key it becomes an index probe per outer row, roughly O(N·log M) — excellent when the outer input is small and the inner is indexed, and it's the only choice for inequality/range predicates and cross joins. A **hash join** builds an in-memory hash table on the smaller input keyed by the join column, then streams the larger input probing that table; it's about O(N+M) and is the go-to for large, unindexed inputs joined on equality, though it needs memory (spilling to disk as a grace hash join if it overflows) and only supports equi-joins. A **merge join** sorts both inputs on the join key (or reads them already-sorted from indexes) and walks them in lock-step like a zipper; it's O(N log N + M log M) dominated by the sorts, or O(N+M) if the inputs arrive pre-sorted, and shines for huge equi-joins or when the output is needed in sorted order. The optimizer estimates row counts from table statistics and available indexes and picks the cheapest; keeping statistics fresh with `ANALYZE` and providing appropriate indexes is how you get it to choose well.

**6. LEFT JOIN turned INNER by WHERE.**
This is the `ON` vs `WHERE` trap. In logical processing, the join and its `ON` predicate run first, producing left rows with NULL-filled right columns for the unmatched ones. Then `WHERE` runs on that combined set. If your `WHERE` references a right-table column — say `WHERE d.active = true` — the unmatched left rows have `d.active = NULL`, `NULL = true` is not true, so those rows are filtered out, effectively converting your LEFT JOIN into an INNER JOIN. The fix depends on intent: if the condition is meant to *restrict which right rows are eligible to match* while still keeping unmatched left rows, move it into the `ON` clause: `LEFT JOIN departments d ON d.dept_id = e.dept_id AND d.active = true`. If you genuinely want only active departments and don't care about unmatched employees, then an INNER JOIN was what you wanted anyway. The rule: predicates on the outer (nullable) side of an outer join usually belong in `ON`, not `WHERE`.

**7. Why normalize then join.**
Storing everything in one wide table creates redundancy: a customer's name would be repeated on every one of their orders. That causes update anomalies (change the name in one row, forget another, and the data is now inconsistent), wastes storage, and risks insertion/deletion anomalies (you can't record a customer with no orders, or you lose the customer when their last order is deleted). Normalization stores each fact exactly once — the customer's name lives in one `customers` row — eliminating those anomalies and keeping data consistent. The cost is that answering "orders with customer names" now requires recombining tables, which is exactly what joins do, efficiently and on demand, using indexes. So we separate concerns: normalized *storage* for integrity, joins for *query-time* recombination. It's the right trade-off for transactional systems; analytical/warehouse systems sometimes deliberately denormalize for read speed, accepting the redundancy.

**8. INNER/LEFT/RIGHT/FULL comparison.**

| Join | Unmatched LEFT rows | Unmatched RIGHT rows |
|---|---|---|
| `INNER` | dropped | dropped |
| `LEFT OUTER` | kept (right cols NULL) | dropped |
| `RIGHT OUTER` | dropped | kept (left cols NULL) |
| `FULL OUTER` | kept (right cols NULL) | kept (left cols NULL) |

All four return the matched pairs identically; they differ *only* in how they treat rows with no match on the other side. INNER keeps none of the orphans, LEFT keeps left orphans, RIGHT keeps right orphans, and FULL keeps both. Choosing correctly is about which side's unmatched rows carry meaning for your question — e.g., LEFT JOIN from `customers` to `orders` keeps customers who have never ordered (right columns NULL), which INNER would silently hide.

## Common Mistakes

- **Filtering the outer side in `WHERE`** instead of `ON`, silently converting a LEFT JOIN into an INNER JOIN.
- **Forgetting the `ON` clause**, producing an accidental Cartesian product.
- **Fan-out double counting** — joining a one-to-many relationship then `SUM`-ing multiplies values; aggregate before joining, or count distinct.
- **Assuming `RIGHT JOIN` is needed** — nearly always rewritable (and clearer) as a `LEFT JOIN` with the tables swapped.
- **Using FULL OUTER JOIN in MySQL** — unsupported; emulate with `LEFT ... UNION ... RIGHT`.
- **Not indexing join keys**, forcing a slow nested-loop or an unnecessary hash/sort on large tables.
- **Ambiguous column references** in multi-table joins — always alias tables and qualify columns.
- **Comparing to NULL with `=`** in join/anti-join logic instead of `IS NULL`.

## Related Concepts

- **Primary keys, foreign keys, and referential integrity** — the relationships joins traverse.
- **Normalization / denormalization** — the design decisions that make joins necessary or avoidable.
- **Indexes (B-tree)** — what makes indexed nested-loop and merge joins fast.
- **Query optimizer, statistics, `EXPLAIN`/`ANALYZE`** — how the physical join algorithm is chosen and diagnosed.
- **`work_mem` / hash spill (grace hash join)** — memory mechanics of hash joins.
- **Anti-join / semi-join** — `NOT EXISTS`, `LEFT JOIN ... IS NULL`, `IN`/`EXISTS` patterns.
- **Recursive CTEs (`WITH RECURSIVE`)** — traversing hierarchies beyond a single self join.
- **Junction (bridge) tables** — resolving many-to-many relationships.
- **Logical query processing order** — explains the `ON` vs `WHERE` behavior for outer joins.


---

# Subqueries

## What is it?

A **subquery** (also called an *inner query* or *nested query*) is a `SELECT` statement embedded inside another SQL statement. The outer statement is called the **outer query** (or *main query*). The subquery runs and produces a result, and that result is consumed by the outer query.

Think of it as composition of functions in math: `f(g(x))`. The inner function `g(x)` produces something that the outer function `f` operates on. In SQL, the subquery is `g`, the outer query is `f`.

Subqueries can appear in almost every clause:

- In `WHERE` — to filter based on a computed set or value.
- In `SELECT` — to compute a per-row scalar value.
- In `FROM` — as a derived table (an inline, throwaway table).
- In `HAVING` — to filter groups based on a computed value.

There are several ways to categorize subqueries, and understanding the categories is what separates someone who *uses* subqueries from someone who *understands* them:

**By the shape of what they return:**

- **Scalar subquery** — returns exactly one row and one column (a single value). Example: `(SELECT MAX(salary) FROM employees)`.
- **Single-row subquery** — returns one row, possibly multiple columns. Compared with `=`, `<`, `>`, etc.
- **Multiple-row subquery** — returns many rows (one column). Must be compared with set operators: `IN`, `ANY`, `ALL`, or used with `EXISTS`.
- **Multiple-column subquery** — returns multiple columns; used with row constructors or in `FROM`.
- **Table subquery** — returns a full result set used as a derived table in `FROM`.

**By dependency on the outer query:**

- **Non-correlated (self-contained) subquery** — can run independently of the outer query. It's evaluated once, and its result is reused.
- **Correlated subquery** — references columns from the outer query, so it *cannot* run on its own. Conceptually it is re-evaluated for each row the outer query considers.

## Why is it needed?

At first principles, SQL is a declarative language for asking questions about sets of data. Many real questions are *multi-step*: "Find the employees who earn more than the average salary" requires you to (1) compute the average, then (2) compare each employee against it. You cannot write `WHERE salary > AVG(salary)` directly, because an aggregate can't sit in a `WHERE` clause that filters individual rows — the aggregate needs its own scope. The subquery gives you that separate scope.

Reasons subqueries exist:

1. **Comparison against a computed value.** "Above average", "the most recent order", "more than the department max" — all require computing an aggregate or value first.
2. **Filtering by membership in a derived set.** "Customers who have placed an order", "products never sold" — you need a set of keys computed from another table.
3. **Existence checks.** "Departments that have at least one employee" — you don't care about the values, only whether a matching row exists.
4. **Building intermediate tables inline** without creating a permanent or temp table (derived tables in `FROM`).
5. **Correlating rows across a comparison** — "each employee compared to their own department's average" — where the comparison value changes per row.

Without subqueries you'd be forced to either run multiple round-trips from the application (compute the average, then send a second query with a literal value baked in — which is racy and chatty) or resort to joins that don't always express the intent cleanly (especially existence and "not exists" semantics).

## How does it work?

Let's ground everything in a small schema you can picture:

```sql
CREATE TABLE departments (
    dept_id   INT PRIMARY KEY,
    dept_name TEXT
);

CREATE TABLE employees (
    emp_id   INT PRIMARY KEY,
    emp_name TEXT,
    dept_id  INT REFERENCES departments(dept_id),
    salary   NUMERIC,
    mgr_id   INT
);
```

### Scalar subquery (single value)

Returns one row, one column. Usable anywhere a single value is allowed:

```sql
-- Employees earning more than the company-wide average
SELECT emp_name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);
```

The inner query yields one number; the outer query compares each row's `salary` against it. If a scalar subquery accidentally returns more than one row, the database raises an error (`more than one row returned by a subquery used as an expression` in PostgreSQL).

Scalar subqueries also work in `SELECT`:

```sql
SELECT emp_name,
       salary,
       (SELECT AVG(salary) FROM employees) AS company_avg,
       salary - (SELECT AVG(salary) FROM employees) AS diff_from_avg
FROM employees;
```

### Single-row subquery with comparison operators

```sql
-- The employee(s) with the highest salary
SELECT emp_name, salary
FROM employees
WHERE salary = (SELECT MAX(salary) FROM employees);
```

`=`, `<>`, `<`, `<=`, `>`, `>=` require the subquery to return at most one row.

### Multiple-row subquery: IN, ANY, ALL

When the subquery returns a *set* of rows, you must use a set operator.

**IN** — true if the outer value matches any value in the set:

```sql
-- Employees who work in departments located via some rule (here: dept 1 or 2)
SELECT emp_name
FROM employees
WHERE dept_id IN (SELECT dept_id FROM departments WHERE dept_name IN ('Sales','Engineering'));
```

**ANY / SOME** — compares the outer value against each element with a chosen operator; true if it holds for at least one:

```sql
-- Employees earning more than AT LEAST ONE employee in dept 3
-- i.e. more than the minimum salary in dept 3
SELECT emp_name, salary
FROM employees
WHERE salary > ANY (SELECT salary FROM employees WHERE dept_id = 3);
```

`> ANY (…)` is equivalent to `> MIN(…)`. `= ANY (…)` is exactly equivalent to `IN (…)`.

**ALL** — true if the comparison holds against every element of the set:

```sql
-- Employees earning more than EVERY employee in dept 3
-- i.e. more than the maximum salary in dept 3
SELECT emp_name, salary
FROM employees
WHERE salary > ALL (SELECT salary FROM employees WHERE dept_id = 3);
```

`> ALL (…)` is equivalent to `> MAX(…)`. `<> ALL (…)` is equivalent to `NOT IN (…)`.

Mnemonic: **ANY = "at least one" (OR-like)**, **ALL = "every one" (AND-like)**.

### Subquery in FROM (derived table)

Here the subquery acts as a table. It **must** be given an alias.

```sql
-- Average salary per department, then filter to well-paid departments
SELECT d.dept_name, x.avg_sal
FROM (
    SELECT dept_id, AVG(salary) AS avg_sal
    FROM employees
    GROUP BY dept_id
) AS x
JOIN departments d ON d.dept_id = x.dept_id
WHERE x.avg_sal > 60000;
```

### Correlated subquery

The inner query references a column from the outer query (`e1.dept_id` below), so it can't stand alone:

```sql
-- Employees who earn more than the average salary OF THEIR OWN department
SELECT e1.emp_name, e1.salary, e1.dept_id
FROM employees e1
WHERE e1.salary > (
    SELECT AVG(e2.salary)
    FROM employees e2
    WHERE e2.dept_id = e1.dept_id   -- correlation: depends on outer row
);
```

Conceptually, for each row `e1` the database plugs in that row's `dept_id`, computes the departmental average, and compares. (In practice the optimizer often rewrites this into a join or a windowed computation — see Internal Working.)

### EXISTS vs IN

**EXISTS** takes a correlated subquery and returns true as soon as the subquery produces *at least one row*. It doesn't care about the values selected — `SELECT 1` is idiomatic.

```sql
-- Departments that have at least one employee
SELECT d.dept_name
FROM departments d
WHERE EXISTS (
    SELECT 1 FROM employees e WHERE e.dept_id = d.dept_id
);
```

**IN** with a subquery checks membership of a value in a returned set:

```sql
SELECT d.dept_name
FROM departments d
WHERE d.dept_id IN (SELECT dept_id FROM employees);
```

Both answer the same question here. Differences that matter:

- `EXISTS` short-circuits: it stops at the first match, which can be faster for existence checks over large child sets.
- `IN` builds/streams the whole set of values.
- **NULL handling differs dramatically** with the negated forms (next section).
- Modern optimizers (PostgreSQL included) frequently transform both into **semi-joins**, so raw performance is often the same. Correctness of the NULL semantics, however, is *not* something the optimizer will "fix" for you.

### The NOT IN + NULL trap

This is one of the most notorious footguns in SQL and a favorite interview question.

`NOT IN` is defined in terms of `<>` comparisons combined with `AND`. If the subquery's result set contains **even a single NULL**, the whole `NOT IN` predicate can never return TRUE — it collapses to NULL (unknown) for every row, so you get **zero rows back**.

Why? `x NOT IN (a, b, NULL)` expands to `x <> a AND x <> b AND x <> NULL`. The last term `x <> NULL` is **UNKNOWN**, and `TRUE AND UNKNOWN = UNKNOWN`, `FALSE AND UNKNOWN = FALSE`. So the predicate is either FALSE or UNKNOWN — never TRUE. No row passes.

```sql
-- DANGEROUS: if any employee has a NULL dept_id, this returns NOTHING
SELECT d.dept_name
FROM departments d
WHERE d.dept_id NOT IN (SELECT dept_id FROM employees);
```

If just one employee row has `dept_id = NULL`, this query silently returns an empty set — even for departments that genuinely have no employees. It's silent, it's wrong, and it passes code review because the SQL "looks fine".

Fixes:

```sql
-- Option A: exclude NULLs explicitly
WHERE d.dept_id NOT IN (SELECT dept_id FROM employees WHERE dept_id IS NOT NULL);

-- Option B (preferred): use NOT EXISTS, which is NULL-safe
SELECT d.dept_name
FROM departments d
WHERE NOT EXISTS (
    SELECT 1 FROM employees e WHERE e.dept_id = d.dept_id
);
```

`NOT EXISTS` uses row existence, not value comparison, so NULLs in the child don't sabotage it. **Rule of thumb: prefer `NOT EXISTS` over `NOT IN` unless you are certain the subquery column is NOT NULL.**

## Internal Working

To reason about performance you need a mental model of how the planner treats subqueries.

**Non-correlated scalar subqueries** are typically evaluated **once**. In PostgreSQL, a subquery like `(SELECT AVG(salary) FROM employees)` becomes an `InitPlan` — computed a single time and cached as a constant reused for every outer row. So the "average salary" is not recomputed per employee, despite how the query reads.

**Correlated subqueries** conceptually run once per outer row (a nested-loop pattern). If written naively, a correlated subquery over a large outer table can be O(N×M). *However*, the optimizer frequently **de-correlates** them: it rewrites `WHERE EXISTS (…)` into a **semi-join**, and `WHERE NOT EXISTS (…)` into an **anti-join**. A semi-join returns each outer row at most once when a match exists (unlike a regular join, which would multiply rows); an anti-join returns outer rows with no match. These can then be executed with hash or merge strategies, turning an apparent O(N×M) into roughly O(N+M).

**`IN (subquery)`** is also typically converted to a semi-join. **`NOT IN (subquery)`** is harder for the optimizer because of the NULL semantics — it often *cannot* be simplified to a clean anti-join when the column is nullable, which is a second reason (beyond correctness) to prefer `NOT EXISTS`.

**Derived tables (`FROM (…)`)** are usually either *inlined/flattened* into the parent query (subquery flattening) or materialized into an intermediate result. PostgreSQL flattens simple subqueries automatically; ones containing `GROUP BY`, `DISTINCT`, `LIMIT`, or aggregation may be materialized.

**`InitPlan` vs `SubPlan`** (PostgreSQL vocabulary you can drop in an interview): an *InitPlan* is evaluated once before the main plan runs (non-correlated); a *SubPlan* is evaluated repeatedly, parameterized by the outer row (correlated). Reading `EXPLAIN` output, seeing a `SubPlan` on a big table is a red flag worth investigating.

Practical takeaways:

- Use `EXPLAIN (ANALYZE, BUFFERS)` to see whether a subquery became a semi-join or stayed a per-row SubPlan.
- Don't assume a correlated subquery is slow — check the plan; the optimizer may have de-correlated it.
- `NOT IN` on a nullable column is both a correctness bug and an optimization blocker.

## Advantages

- **Expressiveness.** They let you phrase multi-step, "compare against a computed value" questions directly in one declarative statement.
- **Readability for existence/membership** logic — `EXISTS`, `IN`, `NOT EXISTS` read close to natural language.
- **No temporary objects.** Derived tables give you an intermediate result without `CREATE TABLE`, permissions, or cleanup.
- **Encapsulation of intent.** The inner question is visibly separated from the outer question.
- **Correlated subqueries** express per-row relationships (each row vs its own group) cleanly.
- **Single round-trip.** Everything computes server-side; no chatty app-side "compute then re-query" logic that could go stale.

## Limitations

- **Readability collapses when nested deeply.** Three or four levels of nesting is very hard to follow — CTEs (next topic) are usually better.
- **No reuse.** A subquery used in two places must be written twice; a CTE or view can be named once and reused.
- **The NOT IN / NULL trap.** Silent wrong results on nullable columns.
- **Scalar-subquery-in-SELECT can be slow** if not de-correlated — it may run per output row.
- **Scalar subqueries must return ≤ 1 row**; a data change that yields two rows turns a working query into a runtime error.
- **Harder to debug** than CTEs, which you can `SELECT * FROM` piece by piece.
- **Optimizer dependence.** Whether a correlated subquery is fast depends on de-correlation, which varies by engine and version.

## Real-world Applications

- **"Above/below average" analytics** — salaries above department average, orders above customer's historical average.
- **Latest-record-per-group** — the most recent order per customer via a correlated `= (SELECT MAX(order_date) …)` (though window functions often do this better).
- **Data-quality / gap reports** — "customers with no orders", "products never sold" via `NOT EXISTS`.
- **Access control filters** — "rows in projects the current user is a member of" via `WHERE project_id IN (SELECT … )`.
- **Deduplication and existence checks** during ETL — "insert only rows that don't already exist" via `WHERE NOT EXISTS (…)`.
- **Threshold filtering on aggregates** — "departments whose average salary exceeds X" via a derived table in `FROM`.

## Interview Questions

**Beginner**

1. What is a subquery, and where can it appear in a SQL statement?
2. What is the difference between a scalar subquery and a multiple-row subquery?
3. Why can't you use `=` with a subquery that returns many rows, and what should you use instead?

**Intermediate**

4. Explain the difference between `IN`, `ANY`, and `ALL`. Give an equivalent aggregate for `> ANY` and `> ALL`.
5. What is a correlated subquery? How does it differ from a non-correlated one?
6. What must you always provide for a subquery used in the `FROM` clause, and why?

**Advanced**

7. Explain the `NOT IN` + NULL trap. Why does it return no rows, and how do you fix it?
8. How does a database optimizer typically execute `WHERE EXISTS (…)` and `WHERE NOT EXISTS (…)`? Define semi-join and anti-join.
9. In PostgreSQL, what is the difference between an `InitPlan` and a `SubPlan`, and why does it matter?

**Scenario-based**

10. You have `orders(customer_id)` and `customers(customer_id)`. Write a query to find customers who have never ordered. Then explain why `NOT IN` might silently break here.
11. You need each employee's salary compared to their own department's average, on every row. How would you write it, and what's a more efficient alternative?

**"Why" questions**

12. Why might a correlated subquery not actually be slow in practice?
13. Why is `NOT EXISTS` generally preferred over `NOT IN`?

**Comparison questions**

14. `EXISTS` vs `IN` — when are they equivalent and when do they differ?
15. Subquery vs join — when would you reach for each?

## Model Answers

**1. What is a subquery, and where can it appear?**
A subquery is a `SELECT` statement nested inside another SQL statement; its result feeds the outer (main) query. It can appear in the `WHERE` clause (to filter against a computed value or set), in the `SELECT` list (as a per-row scalar value), in the `FROM` clause (as a derived table that behaves like a temporary table), and in `HAVING` (to filter groups). The key mental model is function composition: the inner query produces something — a scalar, a set, or a table — and the outer query consumes it. The category of the subquery is defined both by *what shape* it returns (scalar, single-row, multi-row, table) and by *whether it depends on the outer query* (correlated vs non-correlated).

**2. Scalar vs multiple-row subquery.**
A scalar subquery returns exactly one row and one column — a single value — so it can be used anywhere a single value is legal, including comparison with `=`, `>`, `<`, or in the `SELECT` list. Example: `WHERE salary > (SELECT AVG(salary) FROM employees)`. A multiple-row subquery returns a set of values (one column, many rows) and therefore cannot be compared with a plain scalar operator; it must be used with a set-aware construct such as `IN`, `ANY`, `ALL`, or `EXISTS`. If a scalar subquery accidentally returns more than one row at runtime, the database raises an error rather than silently picking one, which is why "single-row" is a runtime guarantee you must design for.

**3. Why not `=` with a many-row subquery?**
Because `=` is a scalar comparison: it compares one value against one value and produces a single true/false. If the right side is a set of several values, `= set` is undefined — the engine can't decide which element to compare against. Semantically you must state your intent: do you mean "equal to *any* element" (that's `IN`, or `= ANY`), or "equal to *every* element" (that's `= ALL`, which is only true when the set is a single repeated value)? So instead of `=` you use `IN` / `= ANY` for membership, or restructure the subquery (e.g. add an aggregate like `MAX`) so it genuinely returns one row.

**4. IN vs ANY vs ALL.**
All three compare an outer value against a set produced by a subquery. `IN` tests membership and is exactly equivalent to `= ANY`. `ANY` (synonym `SOME`) applies your chosen operator and is true if it holds for *at least one* element — so `x > ANY (set)` is true when `x` exceeds the smallest element, i.e. equivalent to `x > MIN(set)`. `ALL` requires the operator to hold for *every* element — so `x > ALL (set)` is true only when `x` exceeds the largest, i.e. `x > MAX(set)`. The mnemonic: ANY behaves like OR across the set ("at least one"), ALL behaves like AND across the set ("all of them"). Also note `<> ALL (set)` equals `NOT IN (set)` — and inherits the same NULL trap.

**5. Correlated vs non-correlated.**
A non-correlated (self-contained) subquery references only its own tables; it can be executed standalone and, being independent of the outer row, is typically evaluated once and its result reused (an `InitPlan` in PostgreSQL). A correlated subquery references one or more columns from the outer query, so it is logically re-evaluated for each candidate outer row — the outer row supplies parameters to the inner query. Example of correlation: `WHERE e1.salary > (SELECT AVG(e2.salary) FROM employees e2 WHERE e2.dept_id = e1.dept_id)` — the inner average depends on `e1.dept_id`. Conceptually correlated subqueries are nested loops, but optimizers frequently de-correlate them into joins, so "correlated" describes the *semantics*, not necessarily the *execution*.

**6. Alias for a FROM subquery.**
A subquery in `FROM` produces a derived table, and every table reference in SQL needs a name so its columns can be qualified and referenced. Without an alias there's no way to write `x.avg_sal` in the outer `SELECT` or to join it. PostgreSQL enforces this and will error `subquery in FROM must have an alias`. So you always write `FROM (SELECT …) AS x`. It's good practice to alias the derived columns too (via `AS`) so their names are stable and explicit.

**7. The NOT IN + NULL trap.**
`x NOT IN (subquery)` is defined as `x <> v1 AND x <> v2 AND … ` for each value the subquery returns. If any `vi` is NULL, then `x <> NULL` evaluates to UNKNOWN (three-valued logic: any comparison with NULL is unknown). Now the whole AND-chain can be at most UNKNOWN — because `TRUE AND UNKNOWN = UNKNOWN` — and can never become TRUE. Since a row only passes a `WHERE` clause when the predicate is TRUE, *no row passes*, and the query silently returns zero rows even when there are legitimate non-matching rows. The fix is either to filter NULLs out of the subquery (`WHERE col IS NOT NULL`) or, far better, to rewrite as `NOT EXISTS`, which is based on row existence and is unaffected by NULL values in the child. This is why NOT EXISTS is the safe default.

**8. How EXISTS / NOT EXISTS execute; semi-join and anti-join.**
`WHERE EXISTS (correlated subquery)` is typically implemented as a **semi-join**: for each outer row, the engine checks whether at least one matching inner row exists and, if so, emits the outer row *once* (a regular join would emit it once per match, multiplying rows — the semi-join deliberately does not). `WHERE NOT EXISTS (…)` becomes an **anti-join**: emit the outer row only when *no* matching inner row exists. Both can be executed with efficient hash or merge algorithms rather than literal nested loops, so despite reading like per-row checks they often run in roughly linear time. `EXISTS` also short-circuits at the first match, which helps when the child set is large. This transformation is why existence-based predicates scale well.

**9. InitPlan vs SubPlan (PostgreSQL).**
In PostgreSQL's executor, an **InitPlan** is a sub-plan evaluated *once*, before (or independently of) the main plan, because it does not depend on the current outer row — for example a non-correlated scalar subquery like `(SELECT MAX(salary) FROM employees)`. Its single result is cached and treated as a constant. A **SubPlan** is a sub-plan evaluated *repeatedly*, parameterized by values from each outer row — the executor's representation of a correlated subquery that the planner did not de-correlate. This matters because a SubPlan running on a large outer relation implies per-row execution (potentially O(N×M)); seeing one in `EXPLAIN` output over a big table tells you to consider rewriting the query so the optimizer can turn it into a semi-/anti-join.

**10. Customers who never ordered.**
```sql
-- Safe version
SELECT c.customer_id, c.name
FROM customers c
WHERE NOT EXISTS (
    SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id
);
```
The naive `WHERE c.customer_id NOT IN (SELECT customer_id FROM orders)` looks equivalent, but `orders.customer_id` may contain NULLs (e.g. an order not yet assigned to a customer, or a soft-deleted link). A single NULL in that column makes the `NOT IN` predicate UNKNOWN for every customer, so the query returns *no* customers at all — silently wrong. `NOT EXISTS` compares by row existence keyed on `o.customer_id = c.customer_id`; NULL child rows simply never match any customer and don't poison the result, so it's both correct and typically executed as an efficient anti-join.

**11. Each employee vs their department average.**
The classic correlated form:
```sql
SELECT e.emp_name, e.salary,
       (SELECT AVG(x.salary) FROM employees x WHERE x.dept_id = e.dept_id) AS dept_avg
FROM employees e;
```
This is clear but may run the inner aggregate per row if not de-correlated. The more efficient and idiomatic alternative is a **window function**, which computes the departmental average in a single pass without a self-join or per-row subquery:
```sql
SELECT emp_name, salary,
       AVG(salary) OVER (PARTITION BY dept_id) AS dept_avg
FROM employees;
```
The window version reads better, avoids repeated scans, and keeps every original row — exactly the "compute across a group without collapsing it" use case window functions were built for.

**12. Why a correlated subquery may not be slow.**
Because "correlated" describes semantics, not execution. Cost-based optimizers routinely **de-correlate** these queries — rewriting `EXISTS`/`IN` into semi-joins and `NOT EXISTS` into anti-joins — and then execute them with hash or merge algorithms that touch each table roughly once. So a query that reads like an O(N×M) nested loop can actually run in near-linear time. The only way to know is to read the execution plan: if you see a hash semi-join, it's fine; if you see a `SubPlan` re-executed per row on a large table, then it really is per-row and worth rewriting.

**13. Why NOT EXISTS over NOT IN.**
Two reasons: correctness and optimizability. Correctness — `NOT IN` breaks (returns zero rows) if the subquery yields any NULL, due to three-valued logic; `NOT EXISTS` is immune because it tests row existence, not value equality. Optimizability — because of those NULL semantics the planner often cannot convert `NOT IN` on a nullable column into a clean anti-join, whereas `NOT EXISTS` maps directly onto an anti-join. So `NOT EXISTS` is safer *and* usually at least as fast. Reserve `NOT IN` for cases where the column is provably `NOT NULL` (e.g. a primary key) and readability wins.

**14. EXISTS vs IN.**
They're logically equivalent for the positive case when the join column has no NULLs: "departments that have employees" works with either. Differences: `IN` compares a value against a materialized/streamed set of values, while `EXISTS` checks for the presence of a correlated row and short-circuits at the first hit. For very large child sets, `EXISTS` can be cheaper because of short-circuiting; for small constant lists, `IN` is simpler. The big divergence is in the negated forms — `NOT IN` is NULL-sensitive and `NOT EXISTS` is not. Modern optimizers turn both positive forms into semi-joins, so the practical performance is frequently identical, and the choice comes down to correctness (NULLs) and readability.

**15. Subquery vs join.**
Use a **join** when you need columns from both tables in the result, or when you're combining rows to enrich them. Use a **subquery** when you only need to *filter* the outer query by a computed value or by existence/membership and don't want the extra columns — especially existence (`EXISTS`) and non-existence (`NOT EXISTS`) checks, which express intent more clearly than an outer join with an `IS NULL` filter. A subtle correctness point: a plain inner join to a child table can *duplicate* outer rows when multiple children match, whereas a semi-join via `EXISTS` returns each outer row once. So "does a related row exist?" is better as `EXISTS`; "give me the matching related data" is better as a join. Under the hood they often compile to similar plans, so pick the form that most clearly states your intent and avoids row multiplication.

## Common Mistakes

- **Using `NOT IN` on a nullable column** — silent empty results. Prefer `NOT EXISTS`.
- **Assuming a scalar subquery will always return one row** — data growth makes it return two and the query errors in production. Guard with an aggregate or `LIMIT 1` where appropriate.
- **Forgetting the alias on a `FROM` subquery** — a hard syntax error in PostgreSQL.
- **Using `IN` with a giant subquery result** where `EXISTS` would short-circuit — occasionally a performance smell (though optimizers often equalize them).
- **Deep nesting** (subquery in subquery in subquery) that nobody can read — refactor to CTEs.
- **Re-writing the same subquery in multiple places** instead of naming it once with a CTE or view.
- **Confusing `ANY` and `ALL`** — writing `> ALL` when you meant `> ANY` (or vice versa) inverts the logic.
- **Expecting `= ALL (empty set)`** semantics — `ALL` over an empty set is TRUE and `ANY` over an empty set is FALSE, which surprises people.
- **Correlated subquery in `SELECT` over a huge table** without checking the plan — can silently become per-row execution.

## Related Concepts

- **Common Table Expressions (CTEs)** — named subqueries; the readable, reusable alternative (next topic).
- **Joins** (inner, outer, semi, anti) — the relational-algebra operations subqueries frequently compile into.
- **Window functions** — often replace correlated aggregate subqueries with a single-pass computation.
- **Three-valued logic (TRUE/FALSE/UNKNOWN)** — the root cause of the NOT IN/NULL trap.
- **Query planner / EXPLAIN** — how to verify whether a subquery de-correlated.
- **Derived tables and lateral joins (`LATERAL`)** — a correlated subquery in `FROM`, letting the derived table reference outer columns.
- **Views and materialized views** — persistently named queries built on the same idea.

# Common Table Expressions (CTEs)

## What is it?

A **Common Table Expression (CTE)** is a named, temporary result set defined at the top of a query using the `WITH` keyword, which you can then reference in the main query (and, for recursive CTEs, within itself). You can think of it as a **named subquery** whose scope is a single statement.

```sql
WITH high_earners AS (
    SELECT emp_id, emp_name, salary
    FROM employees
    WHERE salary > 100000
)
SELECT * FROM high_earners WHERE emp_name LIKE 'A%';
```

Here `high_earners` is a CTE. It exists only for the duration of that one statement — it's not a table, not a view, and it disappears when the statement finishes. The mental shift from a subquery is *naming and position*: instead of burying the derived query inside the `FROM` clause, you lift it to the top, give it a name, and refer to it like a table. This inversion — define first, use later, reading top-to-bottom — is the whole ergonomic point.

There are two flavors:

- **Non-recursive CTE** — an ordinary named subquery. You can define several, chained, each able to reference the ones before it.
- **Recursive CTE** (`WITH RECURSIVE`) — a CTE that references itself, used to walk hierarchies (org charts, bill-of-materials, folder trees) and to generate sequences. This is genuinely more powerful than plain subqueries: it lets SQL do iteration/graph traversal, which is otherwise impossible in a single standard query.

## Why is it needed?

First principles: complex questions are built from simpler ones, and humans read top-to-bottom. Subqueries force you to read *inside-out* and *right-to-left* — you find the deepest nested `SELECT`, understand it, then work outward. That's cognitively expensive and error-prone past two levels. CTEs let you **decompose a problem into named steps** that read like a recipe: first compute A, then use A to compute B, then combine.

Concrete drivers:

1. **Readability & decomposition.** Break a monster query into labeled stages with meaningful names (`monthly_totals`, `ranked_customers`), so the query documents itself.
2. **Reuse within one statement.** A subquery referenced twice must be written twice; a CTE is defined once and referenced many times by name.
3. **Recursion.** Some problems — traversing a parent/child hierarchy of unknown depth, generating a series of numbers or dates — cannot be expressed with plain `SELECT`/join. Recursive CTEs are the standard-SQL answer.
4. **Chaining transformations.** Multi-stage pipelines (filter → aggregate → rank → filter again) map naturally onto a sequence of CTEs, each feeding the next.
5. **Debuggability.** You can select from any intermediate CTE to inspect it (by temporarily making it the final query), which you cannot easily do with a deeply nested subquery.

Without CTEs, teams often reach for temporary tables or views just to name an intermediate result — heavier tools with more lifecycle and permission overhead than the problem warrants.

## How does it work?

### Basic single CTE

```sql
WITH dept_avg AS (
    SELECT dept_id, AVG(salary) AS avg_sal
    FROM employees
    GROUP BY dept_id
)
SELECT e.emp_name, e.salary, d.avg_sal
FROM employees e
JOIN dept_avg d ON d.dept_id = e.dept_id
WHERE e.salary > d.avg_sal;
```

The CTE `dept_avg` is defined once and joined like a table. Compare this to writing the aggregate as a derived table inside `FROM` — same result, but the CTE version separates "compute the averages" from "use the averages".

### Multiple CTEs

Separate definitions with commas. Later CTEs may reference earlier ones (but not vice versa — definitions are top-down):

```sql
WITH
monthly_sales AS (
    SELECT date_trunc('month', order_date) AS month,
           SUM(amount) AS total
    FROM orders
    GROUP BY 1
),
with_growth AS (
    SELECT month,
           total,
           LAG(total) OVER (ORDER BY month) AS prev_total
    FROM monthly_sales
)
SELECT month,
       total,
       prev_total,
       ROUND(100.0 * (total - prev_total) / prev_total, 1) AS pct_growth
FROM with_growth
ORDER BY month;
```

This reads as a clean pipeline: aggregate by month → attach previous month → compute growth. Each stage is named and independently understandable.

### Recursive CTE — the anatomy

A recursive CTE has a strict shape:

```sql
WITH RECURSIVE cte_name AS (
    -- 1. ANCHOR member: the base case, runs once
    SELECT ...
    UNION [ALL]
    -- 2. RECURSIVE member: references cte_name, runs repeatedly
    SELECT ... FROM cte_name JOIN ... WHERE ...
)
SELECT * FROM cte_name;
```

The engine executes it iteratively:
1. Evaluate the **anchor** member → this seeds a "working table".
2. Evaluate the **recursive** member using the current working table's rows as input → produces a new set of rows.
3. Those new rows become the working table for the next iteration, and are appended to the result.
4. Repeat until an iteration produces **zero new rows**, then stop.

**Worked example 1 — number series (generate 1..10):**

```sql
WITH RECURSIVE nums AS (
    SELECT 1 AS n              -- anchor
    UNION ALL
    SELECT n + 1 FROM nums     -- recursive
    WHERE n < 10               -- termination condition
)
SELECT n FROM nums;
-- 1,2,3,4,5,6,7,8,9,10
```

Trace it: anchor gives `{1}`. Iteration 1 takes `n=1`, produces `2`. Iteration 2 takes `n=2`, produces `3`. … Iteration where `n=10` fails the `WHERE n < 10`, produces nothing, so recursion stops. Without the `WHERE`, this is an infinite loop.

**Worked example 2 — org hierarchy (walk the management chain):**

```sql
-- employees(emp_id, emp_name, mgr_id) ; CEO has mgr_id = NULL
WITH RECURSIVE org_chart AS (
    -- anchor: top of the tree (the CEO)
    SELECT emp_id, emp_name, mgr_id, 1 AS level,
           emp_name::text AS path
    FROM employees
    WHERE mgr_id IS NULL

    UNION ALL

    -- recursive: everyone who reports to someone already in org_chart
    SELECT e.emp_id, e.emp_name, e.mgr_id, oc.level + 1,
           oc.path || ' > ' || e.emp_name
    FROM employees e
    JOIN org_chart oc ON e.mgr_id = oc.emp_id
)
SELECT level, emp_name, path
FROM org_chart
ORDER BY path;
```

Sample output:

| level | emp_name | path |
|-------|----------|-----------------------------|
| 1 | Alice | Alice |
| 2 | Bob | Alice > Bob |
| 3 | Carol | Alice > Bob > Carol |
| 2 | Dan | Alice > Dan |

The `level` column tracks depth; the `path` column builds a breadcrumb showing the chain of command. This is the canonical pattern: anchor selects the roots, the recursive member joins the table back to the CTE on the parent/child key, accumulating depth and a path.

### CTE vs subquery

Functionally, a non-recursive CTE and an equivalent `FROM`-subquery usually produce the same plan. The choice is about *communication*: CTEs win on readability, naming, reuse within a statement, and debuggability. Subqueries can be more concise for a trivial one-off filter. Recursion is the one place where a CTE does something a subquery simply cannot.

## Internal Working

**Materialization — the crucial nuance.** Historically, PostgreSQL treated every CTE as an **optimization fence**: it was always *materialized* — computed once into a temporary in-memory (or spilled) result, then read back — and the planner would **not** push predicates from the outer query down into the CTE, nor inline it. This guaranteed a CTE ran exactly once (useful when it has side effects or when you reference it multiple times) but could hurt performance when you only needed a filtered slice, because the whole CTE was computed regardless.

**PostgreSQL 12 changed this.** Since v12, a CTE that is (a) referenced only **once**, (b) **not recursive**, and (c) **side-effect-free** (no data-modifying `INSERT/UPDATE/DELETE ... RETURNING`) is **inlined** by default — treated like a subquery, so the optimizer can push predicates in and flatten it. You can override the decision explicitly:

- `WITH cte AS MATERIALIZED (…)` — force materialization (old behavior; compute once, fence).
- `WITH cte AS NOT MATERIALIZED (…)` — force inlining even when referenced multiple times.

Why you'd force **MATERIALIZED**: when the CTE is expensive and referenced several times, computing it once and reusing beats recomputing; or when inlining would cause the planner to run an expensive function repeatedly; or to deliberately stop predicate pushdown.

Why you'd force **NOT MATERIALIZED**: when a single-use CTE is being materialized (e.g. because you reference it twice but the recomputation is cheap and pushdown would help far more).

**Recursive CTE execution** is the iterative working-table algorithm described above. Key performance notes:
- The recursive member is re-run each iteration on only the *newly produced* rows, not the entire accumulated result (that's why it's called the "working table").
- **Cycles** (e.g. a graph with a loop, or bad data where A reports to B and B reports to A) cause infinite recursion. Guard against it with a depth limit (`WHERE level < N`), by tracking a visited-path array and checking membership, or with the SQL-standard `CYCLE` clause (PostgreSQL 14+): `... CYCLE emp_id SET is_cycle USING path_col`.
- `UNION` (vs `UNION ALL`) de-duplicates each iteration, which adds cost but can help terminate some graphs; `UNION ALL` is faster and usual for trees.

**Other engines differ:** SQL Server and Oracle generally inline CTEs (no fence); MySQL 8+ supports CTEs and recursive CTEs and may materialize or merge. So "CTEs are always materialized" is an outdated, PostgreSQL-<12-specific claim — say "it depends on the engine and version" in an interview.

## Advantages

- **Readability & self-documentation** — named stages read top-to-bottom like a recipe.
- **Decomposition** of complex logic into manageable, testable steps.
- **Reuse within a statement** — define once, reference by name multiple times.
- **Recursion** — hierarchies, graphs, and series that are otherwise impossible in one query.
- **Debuggability** — inspect any stage by selecting from it.
- **No lifecycle overhead** — unlike temp tables (no `CREATE`, no cleanup, no separate permissions).
- **Optional materialization control** (PostgreSQL) — `MATERIALIZED` / `NOT MATERIALIZED` gives you a performance lever.

## Limitations

- **Single-statement scope** — a CTE vanishes after the statement; you can't reuse it across queries (that's what a view is for).
- **Possible materialization cost** — a fenced/materialized CTE that you only partially need may compute far more than necessary (mitigated in PG 12+ inlining, but still relevant when forced or on older versions).
- **No indexes on a CTE** — you can't add an index to an intermediate CTE result; if you need one, a temp table may be better.
- **Recursion pitfalls** — infinite loops on cyclic data without a guard; can be memory-heavy on large/deep graphs.
- **Optimizer opacity across engines** — behavior (inline vs materialize) varies by database and version, so performance isn't perfectly portable.
- **Not always faster** — a CTE is a readability tool; it doesn't inherently speed anything up and can occasionally be slower than an equivalent join if materialized.

## Real-world Applications

- **Reporting pipelines** — multi-stage aggregations (filter → group → rank → filter) expressed as chained CTEs.
- **Org charts / reporting hierarchies** — "all employees under manager X", management depth, breadcrumb paths (recursive).
- **Bill of materials / parts explosion** — "all sub-components of product P" (recursive over a parts graph).
- **Folder / category trees** — file systems, nested product categories, threaded comments (recursive).
- **Graph traversal** — shortest-ish paths, transitive closure ("all users reachable from X in a follow graph").
- **Generating series** — dense date spines for time-series reports (a row per day even when no data), running sequences.
- **Deduplication + ranking** — a CTE computing `ROW_NUMBER()` then the outer query keeping rank = 1.
- **Readable refactors** — turning an unreadable four-level nested subquery into named stages during code review.

## Interview Questions

**Beginner**

1. What is a CTE and what keyword introduces it?
2. How do you define multiple CTEs in one query?
3. What is the scope/lifetime of a CTE?

**Intermediate**

4. What are the two required parts of a recursive CTE, and what does each do?
5. Write a recursive CTE that generates the numbers 1 to 5.
6. How does a later CTE reference an earlier one, and can an earlier CTE reference a later one?

**Advanced**

7. Explain CTE materialization in PostgreSQL, including the change introduced in version 12.
8. When would you force `MATERIALIZED` vs `NOT MATERIALIZED`, and why?
9. How does the recursive CTE execution algorithm (working table) actually run, and how do you prevent infinite recursion on cyclic data?

**Scenario-based**

10. Given `employees(emp_id, emp_name, mgr_id)`, return every employee under the CEO with their depth in the hierarchy.
11. You have a query with a subquery repeated three times in the `WHERE` and `SELECT`. How would CTEs improve it, and would it be faster?

**"Why" questions**

12. Why do CTEs improve readability compared to nested subqueries?
13. Why might a CTE be *slower* than an equivalent subquery in some cases?

**Comparison questions**

14. CTE vs subquery — when does it matter which you use?
15. CTE vs temporary table vs view — compare scope, reuse, indexing, and lifecycle.

## Model Answers

**1. What is a CTE and what introduces it?**
A Common Table Expression is a named temporary result set, defined with the `WITH` keyword, that exists only for the duration of the single statement it precedes. You define it once at the top — `WITH name AS (SELECT …)` — and then reference `name` in the main query as if it were a table. Conceptually it's a named subquery: it doesn't create any persistent object, holds no data after the statement completes, and primarily serves to make complex queries readable by letting you decompose them into labeled steps that read top-to-bottom instead of inside-out.

**2. Multiple CTEs.**
You write a single `WITH` and separate the CTE definitions with commas: `WITH a AS (…), b AS (…), c AS (…) SELECT … FROM c`. Each CTE can reference any CTE defined *before* it, which lets you build a pipeline of transformations — for example `monthly_sales`, then `with_growth` that reads from `monthly_sales`, then a final query reading from `with_growth`. Only the first definition follows `WITH`; the rest just follow commas (you do not repeat `WITH`).

**3. Scope and lifetime.**
A CTE's scope is the single SQL statement in which it is defined; its lifetime ends when that statement finishes executing. It is not visible to other statements, not stored, and not shared across sessions — contrast this with a view (a persistent, reusable named query) or a temporary table (persists for the session/transaction and can be referenced by multiple statements). This single-statement scope is exactly why you reach for a view or temp table when you need to reuse the intermediate result across several queries.

**4. Two parts of a recursive CTE.**
A recursive CTE requires an **anchor member** and a **recursive member**, combined with `UNION` or `UNION ALL`. The anchor member is the base case: a non-recursive `SELECT` that runs once and seeds the result (e.g. the root of a hierarchy, or the starting number). The recursive member references the CTE's own name and runs repeatedly, each iteration consuming the rows produced by the previous iteration (the "working table") and producing new rows, until an iteration yields no new rows and recursion stops. You also need a termination condition (in the recursive member's `WHERE`) to ensure it eventually produces nothing — otherwise it loops forever.

**5. Numbers 1 to 5.**
```sql
WITH RECURSIVE nums AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM nums WHERE n < 5
)
SELECT n FROM nums;
```
The anchor produces `1`. Each recursive step adds one to the last value, and the `WHERE n < 5` stops it: when `n = 5` the recursive member's filter excludes it, no new row appears, and recursion terminates with `1,2,3,4,5`.

**6. Ordering of references.**
A CTE may reference any CTE that appears *earlier* in the same `WITH` clause, because definitions are read and made available top-down. An earlier CTE cannot reference a later one (there's no forward reference) — the exception being a recursive CTE, which references *itself* via the `RECURSIVE` keyword. This top-down rule is what makes a chain of CTEs behave like a readable, ordered pipeline of steps.

**7. CTE materialization in PostgreSQL and the v12 change.**
Before PostgreSQL 12, every CTE was an **optimization fence**: it was always materialized into a temporary result computed exactly once, and the planner would not push outer-query predicates into it or flatten it into the main query. That guaranteed single evaluation but often hurt performance, since even a CTE you only needed a filtered slice of was computed in full. From version 12, PostgreSQL **inlines** a CTE by default when it is non-recursive, side-effect-free, and referenced only once — treating it like an ordinary subquery so predicates can be pushed down and the query flattened. CTEs that are recursive, referenced multiple times, or that modify data are still materialized. You can override the automatic decision with the `MATERIALIZED` and `NOT MATERIALIZED` keywords.

**8. Forcing MATERIALIZED vs NOT MATERIALIZED.**
Force `MATERIALIZED` when the CTE is expensive to compute and referenced multiple times — you want it computed once and reused rather than recomputed per reference — or when it contains a volatile/expensive function you don't want re-evaluated, or when you deliberately want to prevent predicate pushdown (an optimization fence used as a tuning tool). Force `NOT MATERIALIZED` when the planner is materializing a CTE you'd rather have inlined — typically a cheap CTE where pushing the outer `WHERE` filter down into it would let it read far fewer rows via an index, which outweighs recomputation. In short: MATERIALIZED trades recomputation for reuse and a fence; NOT MATERIALIZED trades the fence for pushdown and flattening.

**9. Working-table algorithm and cycle prevention.**
The engine first evaluates the anchor member, placing its rows into a "working table" and into the accumulated result. It then repeatedly evaluates the recursive member using *only the current working table* as its input, appends the newly produced rows to the result, and replaces the working table with just those new rows for the next round. It halts when an iteration produces zero new rows. Because only newly produced rows feed each step, it's efficient for trees. On cyclic data (e.g. a graph with a loop or bad hierarchy data), it never produces zero new rows and loops forever; you prevent this by adding a depth guard (`WHERE level < N`), by carrying an array of visited keys and excluding rows whose key is already in the path, or by using the SQL-standard `CYCLE` clause (PostgreSQL 14+) which detects repeats automatically and marks them.

**10. Employees under the CEO with depth.**
```sql
WITH RECURSIVE org AS (
    SELECT emp_id, emp_name, mgr_id, 1 AS depth
    FROM employees
    WHERE mgr_id IS NULL          -- CEO / root
    UNION ALL
    SELECT e.emp_id, e.emp_name, e.mgr_id, o.depth + 1
    FROM employees e
    JOIN org o ON e.mgr_id = o.emp_id
)
SELECT emp_id, emp_name, depth
FROM org
WHERE depth > 1                   -- exclude the CEO if "under" is strict
ORDER BY depth, emp_name;
```
The anchor picks the CEO (no manager). The recursive member joins the employees table to the accumulating `org` set on `e.mgr_id = o.emp_id` — i.e. "anyone whose manager is already known to be in the tree" — incrementing `depth` each level. It naturally terminates when it reaches leaf employees who manage no one.

**11. Repeated subquery refactor.**
Lifting the repeated subquery into a single CTE removes the duplication: you define it once (`WITH s AS (…)`) and reference `s` in the `SELECT` and `WHERE`, so there's one source of truth and the intent is clear. Whether it's *faster* depends on the engine and version. On PostgreSQL 12+, a CTE referenced multiple times is materialized by default, so it may be computed once and reused — potentially faster than a subquery evaluated three times — but if the original subqueries were themselves de-correlated or cached by the optimizer, the plans could be identical. The unambiguous win is maintainability and correctness (no risk of the three copies drifting apart); performance is a "measure it" question, and you can force `MATERIALIZED` if you specifically want single evaluation.

**12. Why CTEs improve readability.**
Because they let you read a query the way humans reason — top to bottom, as a sequence of named steps — instead of the inside-out, right-to-left decoding that nested subqueries demand. Each CTE has a meaningful name that documents what that stage produces (`monthly_totals`, `ranked_customers`), so the query becomes self-describing. Complex logic decomposes into small, individually understandable units, and you can verify each stage in isolation. The cognitive load of holding several nesting levels in your head at once is replaced by following a linear pipeline.

**13. Why a CTE might be slower.**
Chiefly because of materialization. When a CTE is materialized (forced, or by default in older PostgreSQL, or when referenced multiple times), it is computed in full into a temporary result and acts as an optimization fence — the planner can't push the outer query's filters into it or flatten it into a larger join. So if you only need a small filtered slice, a materialized CTE may compute the entire intermediate set first, whereas an equivalent inline subquery would let the optimizer push the predicate down and read far less. Also, a materialized CTE has no indexes, so downstream joins against it may be less efficient than against a real (indexed) table. This is why on older PG versions people sometimes rewrote CTEs as subqueries for speed.

**14. CTE vs subquery.**
For non-recursive logic they are usually functionally equivalent and, on modern engines, often produce the same plan — so the choice is mostly about communication. Prefer a CTE when the query is complex, when an intermediate result is used more than once, or when you want named, debuggable stages. Prefer a plain subquery for a small, one-off filter where naming it adds ceremony without clarity. The decision genuinely *matters* in two cases: recursion (only a CTE can do it) and materialization behavior (a forced/older-version materialized CTE can perform differently from an inlined subquery, so if you're performance-tuning, know your engine's rules).

**15. CTE vs temp table vs view.**
A **CTE** is scoped to a single statement, holds no persistent data, cannot be indexed, and has zero lifecycle management — ideal for readability within one query. A **temporary table** persists for the session or transaction, can be referenced by many statements, *can* be indexed and analyzed (so it's better when you reuse a heavy intermediate result across queries or need index-backed joins on it), but costs you `CREATE`/populate/drop and statistics. A **view** is a permanent, named, reusable query definition stored in the schema — it's shared across sessions and users and re-executes its underlying query each time it's referenced (unless it's a materialized view, which stores results and must be refreshed). Rule of thumb: CTE for in-statement decomposition, temp table for multi-statement reuse or when you need an index, view for a durable named abstraction shared across queries and users.

## Common Mistakes

- **Forgetting `RECURSIVE`** — a self-referencing CTE without the `RECURSIVE` keyword errors (in PostgreSQL, `WITH RECURSIVE` is required once and covers all recursive CTEs in the clause).
- **No termination condition** in a recursive CTE → infinite loop / runaway query. Always bound it.
- **Not guarding against cycles** on graph data — depth limit, visited-path array, or `CYCLE` clause.
- **Assuming CTEs are always materialized** (or always inlined) — behavior depends on engine and version.
- **Assuming a CTE is faster** just because it looks tidy — it's a readability tool, not a speedup.
- **Expecting to index a CTE** — you can't; use a temp table if you need an index.
- **Referencing a later CTE from an earlier one** — forward references aren't allowed.
- **Using `UNION` when `UNION ALL` was intended** in recursion — `UNION` de-duplicates every iteration, adding cost; use `UNION ALL` for trees unless you specifically need dedup.
- **Trying to reuse a CTE in a different statement** — it doesn't exist outside its one query.

## Related Concepts

- **Subqueries / derived tables** — the un-named equivalent a CTE improves upon.
- **Recursion & graph traversal** — transitive closure, hierarchical queries, `CONNECT BY` (Oracle's alternative).
- **Window functions** — frequently paired with CTEs in reporting pipelines.
- **Views & materialized views** — the persistent, cross-statement counterparts.
- **Temporary tables** — for reusable, indexable intermediate results across statements.
- **Optimization fence / predicate pushdown / query flattening** — the planner behaviors behind CTE materialization.
- **`CYCLE` and `SEARCH` clauses (SQL:1999 / PostgreSQL 14+)** — standard cycle detection and traversal ordering for recursive CTEs.

---

# Window Functions

## What is it?

A **window function** performs a calculation across a set of rows that are related to the current row — *without collapsing those rows into a single output row*. This is the defining property, and the whole reason they exist.

Contrast with a regular aggregate:

- `GROUP BY` + `SUM(salary)` **collapses** each group to one row. You lose the individual rows.
- `SUM(salary) OVER (PARTITION BY dept_id)` computes the same per-department sum but **attaches it to every original row**. You keep all the detail *and* get the aggregate alongside.

That's the magic: you compute something over a group (or an ordered range of neighbors) yet still emit one row per input row. The `OVER` clause is the marker that turns an ordinary function into a window function — it defines the "window" of rows the function sees.

```sql
SELECT emp_name, dept_id, salary,
       AVG(salary) OVER (PARTITION BY dept_id) AS dept_avg,
       salary - AVG(salary) OVER (PARTITION BY dept_id) AS diff
FROM employees;
```

Every employee row is preserved, and each carries its department's average and the difference — impossible with `GROUP BY` alone (which would force you to drop the per-employee detail).

Categories of window functions:

- **Ranking functions** — `ROW_NUMBER()`, `RANK()`, `DENSE_RANK()`, `NTILE(n)`, `PERCENT_RANK()`, `CUME_DIST()`.
- **Value / analytical functions** — `LEAD()`, `LAG()`, `FIRST_VALUE()`, `LAST_VALUE()`, `NTH_VALUE()`.
- **Aggregate functions used as windows** — `SUM()`, `AVG()`, `COUNT()`, `MIN()`, `MAX()` with `OVER`.

## Why is it needed?

First principles: an enormous class of analytics questions are **relative** — they compare a row to its neighbors or to its group:

- "Rank each salesperson within their region."
- "What's each month's sales *and* the running total to date?"
- "How does this month compare to last month (difference / growth)?"
- "What percentage of the department's payroll does each person represent?"
- "Give me the top 3 products per category."

To answer these with only `GROUP BY` and joins, you'd write correlated subqueries or self-joins — one per metric — that are verbose, slow (often O(N²)), and hard to read. A "running total" via self-join joins each row to all prior rows and re-sums them; a "rank" via subquery counts how many rows are greater for each row. Window functions replace all of that with a single, declarative `OVER (…)` clause evaluated in essentially one ordered pass.

They exist because the relational model's aggregation (`GROUP BY`) was *lossy* — it destroyed detail — and analysts constantly need detail and aggregate *together*. Window functions (SQL:2003) fill exactly that gap. They also make previously-impossible-in-one-query things (running totals, moving averages, gap-and-island analysis, top-N-per-group, period-over-period comparisons) both possible and efficient.

## How does it work?

The general syntax:

```sql
function() OVER (
    [PARTITION BY  col1, col2, ...]     -- divide rows into groups (windows)
    [ORDER BY      col3, ...]           -- order rows within each partition
    [frame_clause]                      -- ROWS/RANGE BETWEEN ... (which neighbors)
)
```

- **PARTITION BY** splits the rows into independent partitions; the function restarts for each partition. Omit it and the whole result set is one partition. (It's the windowing analog of `GROUP BY`, but it doesn't collapse rows.)
- **ORDER BY** defines the ordering *within* each partition — essential for ranking, `LEAD`/`LAG`, running totals, and any "position-dependent" function.
- **frame clause** narrows the window to a subset of the partition *relative to the current row* (e.g. "all rows from the start up to the current row", or "the 3 rows before and after").

### Ranking functions — the crucial distinctions

Consider salaries within one department, ordered descending. Watch how each function handles the **tie** at 5000:

**Input (ordered by salary DESC):**

| emp_name | salary |
|----------|--------|
| Alice | 9000 |
| Bob | 7000 |
| Carol | 5000 |
| Dan | 5000 |
| Eve | 4000 |

**Output of the three ranking functions:**

| emp_name | salary | ROW_NUMBER | RANK | DENSE_RANK |
|----------|--------|:---------:|:----:|:----------:|
| Alice | 9000 | 1 | 1 | 1 |
| Bob | 7000 | 2 | 2 | 2 |
| Carol | 5000 | 3 | 3 | 3 |
| Dan | 5000 | 4 | 3 | 3 |
| Eve | 4000 | 5 | 5 | 4 |

```sql
SELECT emp_name, salary,
       ROW_NUMBER() OVER (ORDER BY salary DESC) AS row_number,
       RANK()       OVER (ORDER BY salary DESC) AS rank,
       DENSE_RANK() OVER (ORDER BY salary DESC) AS dense_rank
FROM employees;
```

- **ROW_NUMBER()** — a unique sequential integer, *no ties*. Carol and Dan (both 5000) get 3 and 4 arbitrarily (the tie is broken nondeterministically unless you add a tiebreaker to `ORDER BY`). Use it for "pick exactly one row per group" (deduplication, pagination).
- **RANK()** — ties share the same rank, then the next rank **skips** (a "gap"). Carol and Dan are both 3; Eve is 5 (4 is skipped). This mirrors sports standings: two silver medalists means no bronze. Use when the gap is meaningful ("Olympic ranking").
- **DENSE_RANK()** — ties share the rank, and the next rank does **not** skip (no gap). Carol and Dan are 3; Eve is 4. Use when you want *distinct rank levels* without holes (e.g. "the 4th distinct highest salary").

The one-line summary you should be able to recite: **ROW_NUMBER never ties; RANK ties and skips; DENSE_RANK ties and does not skip.**

**NTILE(n)** divides the ordered partition into `n` roughly equal buckets and labels each row with its bucket number — used for quartiles, deciles, percentiles:

```sql
SELECT emp_name, salary,
       NTILE(4) OVER (ORDER BY salary DESC) AS quartile
FROM employees;
-- quartile 1 = top 25% earners, 4 = bottom 25%
```

If rows don't divide evenly, earlier buckets get one extra row.

### Analytical value functions: LEAD, LAG, FIRST_VALUE, LAST_VALUE

**LAG(col, offset, default)** returns a column value from a *previous* row; **LEAD** from a *following* row. Perfect for period-over-period comparisons:

**Input:**

| month | revenue |
|-------|---------|
| Jan | 100 |
| Feb | 130 |
| Mar | 120 |
| Apr | 160 |

```sql
SELECT month, revenue,
       LAG(revenue)  OVER (ORDER BY month) AS prev_month,
       LEAD(revenue) OVER (ORDER BY month) AS next_month,
       revenue - LAG(revenue) OVER (ORDER BY month) AS mom_change
FROM monthly;
```

**Output:**

| month | revenue | prev_month | next_month | mom_change |
|-------|---------|:----------:|:----------:|:----------:|
| Jan | 100 | NULL | 130 | NULL |
| Feb | 130 | 100 | 120 | 30 |
| Mar | 120 | 130 | 160 | -10 |
| Apr | 160 | 120 | NULL | 40 |

The first row's `LAG` is NULL (nothing before it); the last row's `LEAD` is NULL. Supply a third argument as the default to replace NULL, e.g. `LAG(revenue, 1, 0)`.

**FIRST_VALUE(col)** / **LAST_VALUE(col)** return the value from the first / last row of the *frame*. `FIRST_VALUE` is intuitive; `LAST_VALUE` has a famous pitfall (next section).

### Aggregate windows: SUM/AVG/COUNT OVER — running totals

Add `ORDER BY` inside `OVER` and an aggregate becomes *cumulative* (a running total), because the default frame becomes "start of partition to current row":

**Input & output:**

| day | sales | running_total |
|-----|-------|:-------------:|
| 1 | 100 | 100 |
| 2 | 50 | 150 |
| 3 | 80 | 230 |
| 4 | 40 | 270 |

```sql
SELECT day, sales,
       SUM(sales) OVER (ORDER BY day) AS running_total
FROM daily_sales;
```

Without `ORDER BY`, `SUM(sales) OVER ()` gives the *grand total* on every row (whole partition, no ordering). With `PARTITION BY region ORDER BY day`, you get a running total that restarts each region. This single distinction — presence or absence of `ORDER BY` in the window — flips an aggregate between "total of the whole partition" and "running total up to this row", and understanding *why* requires the frame clause.

### The frame clause: ROWS vs RANGE

The frame clause defines, *within the ordered partition*, exactly which rows the function sees relative to the current row:

```
ROWS  BETWEEN <start> AND <end>
RANGE BETWEEN <start> AND <end>
```

Bounds: `UNBOUNDED PRECEDING`, `N PRECEDING`, `CURRENT ROW`, `N FOLLOWING`, `UNBOUNDED FOLLOWING`.

- **ROWS** counts *physical rows* — "the 3 rows before this one".
- **RANGE** counts *logical value ranges* — all rows whose `ORDER BY` value falls within the range, so **all rows tied with the current row are treated together** (peers).

Examples:

```sql
-- Moving average of the current row and the 2 preceding rows (3-row window)
AVG(sales) OVER (ORDER BY day ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)

-- Running total, explicit frame
SUM(sales) OVER (ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
```

**The default frame** (when you specify `ORDER BY` but *no* explicit frame) is:
`RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`.
This default is the source of the two biggest surprises:

1. Because it's `RANGE`, tied rows are all included at once — a running `SUM` over rows tied on the `ORDER BY` key gives all of them the same cumulative total (the total *through* the last peer), not a per-row incremental value.
2. It powers the `LAST_VALUE` pitfall below.

### The LAST_VALUE default-frame pitfall

You'd expect `LAST_VALUE` to give the last value of the partition. It usually doesn't, because of the default frame:

```sql
-- WRONG (surprising) result:
SELECT emp_name, salary,
       LAST_VALUE(salary) OVER (ORDER BY salary) AS last_val
FROM employees;
```

With the default frame `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`, the window for each row ends *at the current row*. So `LAST_VALUE` returns the value of the **current row**, not the partition's last row! On an ascending order it effectively returns each row's own salary.

To get the true last value of the whole partition, you must extend the frame to the end:

```sql
-- CORRECT:
SELECT emp_name, salary,
       LAST_VALUE(salary) OVER (
           ORDER BY salary
           ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
       ) AS last_val
FROM employees;
```

`FIRST_VALUE` doesn't suffer this because the frame *starts* at `UNBOUNDED PRECEDING` by default, so the first row is always in view. A common alternative to fixing the frame is to flip the `ORDER BY` and use `FIRST_VALUE` instead. This is one of the most-tested window-function traps in interviews.

## Internal Working

**Logical processing position.** Window functions are evaluated *after* `FROM`, `WHERE`, `GROUP BY`, and `HAVING`, but *before* `ORDER BY`, `DISTINCT`, and `LIMIT`. This has two critical consequences:

1. **You cannot reference a window function in `WHERE` or `GROUP BY`** — it doesn't exist yet at that stage. To filter on a window result (e.g. "keep only rank = 1"), you must wrap the query in a subquery or CTE and filter in the outer query:
   ```sql
   SELECT * FROM (
       SELECT *, ROW_NUMBER() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rn
       FROM employees
   ) t
   WHERE rn = 1;   -- top earner per department
   ```
2. Because they run after `WHERE`, window functions see only the rows that survived filtering — the partition is over the *filtered* set, not the raw table.

**Execution.** The planner introduces a **WindowAgg** node (PostgreSQL). It typically:
1. Sorts the input by `PARTITION BY` then `ORDER BY` columns (a sort is often required — hence window functions benefit from indexes matching the partition/order keys).
2. Streams through the sorted rows, maintaining state per partition (a running accumulator for `SUM`, a counter for `ROW_NUMBER`, a peer-group tracker for `RANK`, a buffer for framed functions like moving averages).
3. Emits one output row per input row with the computed value.

Multiple window functions that share the *same* `OVER` specification can reuse a single sort/WindowAgg; different specifications may each add a sort. So aligning window definitions (and providing supporting indexes) is a real optimization.

**Frames and buffering.** Simple running aggregates need only a running accumulator (cheap). Framed functions like `AVG(...) OVER (... ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)` need a small buffer of rows; `... UNBOUNDED FOLLOWING` frames need to look ahead, requiring the whole partition to be materialized. `RANGE` frames with ties do extra work grouping peers.

**RANGE vs ROWS cost & correctness.** `ROWS` is generally cheaper and unambiguous (physical rows). `RANGE` must identify peer groups (rows with equal `ORDER BY` values) and can produce different results when ties exist. Use `ROWS` unless you specifically need peer semantics.

**`WINDOW` clause for reuse.** You can name a window spec once and reuse it:
```sql
SELECT emp_name,
       RANK()       OVER w,
       DENSE_RANK() OVER w
FROM employees
WINDOW w AS (PARTITION BY dept_id ORDER BY salary DESC);
```

## Advantages

- **Detail + aggregate together** — the core benefit; keep every row and annotate it with group-level or neighbor-relative values.
- **Replaces slow self-joins/correlated subqueries** — running totals, ranks, and moving averages that would be O(N²) become a single ordered pass.
- **Expressive & declarative** — complex analytics (top-N-per-group, period-over-period, percentiles) in a compact clause.
- **Composable** — many window functions in one `SELECT`, sharing sorts via a `WINDOW` clause.
- **Standardized** (SQL:2003) and widely supported (PostgreSQL, SQL Server, Oracle, MySQL 8+, BigQuery, Snowflake…).
- **Performance-friendly** with matching indexes on partition/order keys.

## Limitations

- **Cannot be used in `WHERE`/`GROUP BY`/`HAVING`** — must wrap in a subquery/CTE to filter on the result.
- **Often require a sort** — can be expensive on large unindexed data.
- **Frame semantics are subtle** — the default-frame / `LAST_VALUE` pitfall bites the unwary.
- **No cross-partition awareness** — a function can't see other partitions (by design, but sometimes limiting).
- **Memory for wide frames** — `UNBOUNDED FOLLOWING` and large frames may buffer whole partitions.
- **Not for reducing row count** — they don't aggregate-and-collapse; if you truly want fewer rows, use `GROUP BY`.
- **Readability for newcomers** — the `OVER (…)` mental model takes time to internalize.

## Real-world Applications

- **Leaderboards & rankings** — top salesperson per region, top-N products per category (`ROW_NUMBER`/`RANK` + outer filter).
- **Running totals & cumulative metrics** — account balances, cumulative revenue, burn-down (`SUM OVER (ORDER BY …)`).
- **Moving averages** — smoothing time series, 7-day rolling averages (`AVG OVER (… ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)`).
- **Period-over-period analysis** — month-over-month growth, day-over-day deltas (`LAG`/`LEAD`).
- **Percentiles & bucketing** — quartiles of customers by spend, deciles for cohorts (`NTILE`, `PERCENT_RANK`).
- **Deduplication** — keep the latest row per key (`ROW_NUMBER() … ORDER BY updated_at DESC`, keep `rn = 1`).
- **Gap-and-island analysis** — finding consecutive streaks (logins, uptime) via ranking differences.
- **First/last touch attribution** — first and last event per user session (`FIRST_VALUE`/`LAST_VALUE`).
- **Share-of-total** — each row's percent of its group (`amount / SUM(amount) OVER (PARTITION BY grp)`).

## Interview Questions

**Beginner**

1. What is a window function and how does it differ from `GROUP BY`?
2. What does the `OVER` clause do, and what are `PARTITION BY` and `ORDER BY` within it?
3. Name the three ranking functions and what each returns.

**Intermediate**

4. Explain the difference between `ROW_NUMBER`, `RANK`, and `DENSE_RANK` using a tie example.
5. What do `LAG` and `LEAD` do? What value do they return at the partition boundary?
6. How do you turn `SUM()` into a running total, and what happens if you omit `ORDER BY`?

**Advanced**

7. Explain the frame clause. What is the difference between `ROWS` and `RANGE`?
8. What is the default frame when `ORDER BY` is present but no frame is specified, and why does it matter?
9. Explain the `LAST_VALUE` pitfall and two ways to fix it.
10. Where do window functions sit in the logical query processing order, and why can't you filter on them in `WHERE`?

**Scenario-based**

11. Return the top 3 highest-paid employees per department.
12. Compute each month's revenue, the previous month's revenue, and the month-over-month growth percentage.
13. From a table with duplicate rows per `user_id`, keep only the most recently updated row for each user.

**"Why" questions**

14. Why do window functions exist when we already have `GROUP BY`?
15. Why would you use `ROWS` instead of `RANGE` (or vice versa)?

**Comparison questions**

16. Window function vs `GROUP BY` aggregate — when to use each.
17. `RANK` vs `DENSE_RANK` vs `ROW_NUMBER` — give a concrete use case for each.

## Model Answers

**1. Window function vs GROUP BY.**
A window function computes a value across a set of related rows (a "window") but returns one output row per input row — it does not collapse the rows. `GROUP BY` collapses each group into a single summary row, discarding the individual detail. So `SUM(salary)` with `GROUP BY dept_id` gives one total row per department, whereas `SUM(salary) OVER (PARTITION BY dept_id)` attaches each department's total to *every* employee row while keeping all employees visible. The fundamental difference is that window functions let you have detail and aggregate simultaneously; `GROUP BY` forces you to choose the aggregate and lose the detail.

**2. The OVER clause.**
`OVER` is what turns an ordinary function into a window function — it defines the window of rows the function operates over. Inside it, `PARTITION BY` divides the rows into independent groups and the function restarts for each group (analogous to `GROUP BY` but without collapsing); omit it and the entire result set is one partition. `ORDER BY` defines the ordering of rows *within* each partition, which is essential for any position-dependent function — ranking, `LAG`/`LEAD`, running totals — and also, by establishing an order, activates the default frame that makes aggregates cumulative. An optional frame clause further narrows the window to rows relative to the current row.

**3. The three ranking functions.**
`ROW_NUMBER()` assigns a unique, gap-free sequential integer to each row in the ordered partition, breaking ties arbitrarily. `RANK()` gives tied rows the same rank and then leaves a gap — the next rank jumps ahead by the number of tied rows. `DENSE_RANK()` gives tied rows the same rank but leaves no gap — the next rank is the immediately following integer. In one sentence: ROW_NUMBER never ties, RANK ties-and-skips, DENSE_RANK ties-without-skipping.

**4. ROW_NUMBER vs RANK vs DENSE_RANK with a tie.**
Take salaries 9000, 7000, 5000, 5000, 4000 ordered descending. ROW_NUMBER yields 1, 2, 3, 4, 5 — the two 5000s get distinct numbers (3 and 4) chosen arbitrarily. RANK yields 1, 2, 3, 3, 5 — both 5000s are rank 3, and because two rows occupied rank 3, the next value (4000) is rank 5, skipping 4. DENSE_RANK yields 1, 2, 3, 3, 4 — both 5000s are rank 3 and 4000 is rank 4, with no gap. The tie is the whole point: ROW_NUMBER forces uniqueness, RANK preserves positional meaning (like Olympic medals), DENSE_RANK counts distinct value levels.

**5. LAG and LEAD.**
`LAG(col, offset, default)` returns the value of `col` from a row `offset` positions *before* the current row within the ordered partition; `LEAD` returns it from a row `offset` positions *after*. The offset defaults to 1. They're the standard tool for comparing a row to its neighbors — previous vs current month, etc. At the partition boundary there is no neighbor: `LAG` on the first row and `LEAD` on the last row return NULL, unless you supply the third `default` argument (e.g. `LAG(x, 1, 0)`), which is returned instead of NULL. They also restart at each partition boundary when `PARTITION BY` is used, so the previous partition's last row never leaks into the next partition's first.

**6. Running total with SUM, and omitting ORDER BY.**
You turn `SUM()` into a running total by adding `ORDER BY` inside the `OVER` clause: `SUM(sales) OVER (ORDER BY day)`. The `ORDER BY` establishes an order and, via the default frame (`RANGE UNBOUNDED PRECEDING` to `CURRENT ROW`), makes the sum cumulative up to the current row. If you *omit* `ORDER BY` — `SUM(sales) OVER ()` — there is no ordering and the frame is the entire partition, so every row receives the same grand total rather than a running one. Add `PARTITION BY region ORDER BY day` and you get a running total that resets for each region. The presence or absence of `ORDER BY` in the window is exactly what flips between "grand total" and "running total".

**7. The frame clause; ROWS vs RANGE.**
The frame clause restricts the window, for each row, to a subset of its partition defined relative to the current row — using bounds like `UNBOUNDED PRECEDING`, `N PRECEDING`, `CURRENT ROW`, `N FOLLOWING`, `UNBOUNDED FOLLOWING`. `ROWS` interprets these as *physical row counts* — `2 PRECEDING` means literally the two rows before this one. `RANGE` interprets them as *logical value ranges* over the `ORDER BY` expression — so all rows whose ordering value is tied with the current row are treated as a single peer group and included together. Practically, `ROWS` is precise and cheaper and is what you want for moving windows of exactly N rows; `RANGE` is what you want when tied values should be grouped as peers (and it's the default), but it can produce surprising results and extra cost when ties exist.

**8. The default frame and why it matters.**
When you specify `ORDER BY` in the window but no explicit frame, the default is `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`. It matters for two reasons. First, it's what makes aggregates cumulative (a running total ending at the current row) rather than partition-wide. Second, because it's `RANGE` and ends at `CURRENT ROW`, it produces the `LAST_VALUE` pitfall and makes running sums over tied rows behave by peer group (all tied rows share the cumulative total through the last peer). Knowing the default frame explains a whole cluster of "why is my window result surprising?" bugs, so it's essential knowledge, not trivia.

**9. The LAST_VALUE pitfall and fixes.**
`LAST_VALUE(col) OVER (ORDER BY …)` surprises people by returning the current row's value instead of the partition's final value. The cause is the default frame `RANGE … CURRENT ROW`: the window ends at the current row, so the "last" row in view is always the current one. Fix 1: extend the frame to cover the whole partition — `LAST_VALUE(col) OVER (ORDER BY … ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)`. Fix 2: reverse the ordering and use `FIRST_VALUE` instead — `FIRST_VALUE(col) OVER (ORDER BY … DESC)` — since `FIRST_VALUE` is safe (its frame starts at `UNBOUNDED PRECEDING`, so the first row is always visible). `FIRST_VALUE` doesn't need the fix precisely because the default frame's *start* is unbounded while its *end* is the current row.

**10. Logical position and why not in WHERE.**
Window functions are evaluated late in logical query processing: after `FROM`, `WHERE`, `GROUP BY`, and `HAVING`, but before `SELECT DISTINCT`, `ORDER BY`, and `LIMIT`. Because `WHERE`, `GROUP BY`, and `HAVING` are processed *before* the window functions are computed, the window results simply don't exist yet at those stages — so you can't reference them there. To filter on a window result (e.g. keep only `rank = 1`), you compute it in an inner query (subquery or CTE) and apply the filter in an outer query where the value now exists as an ordinary column. A useful corollary: since windows run after `WHERE`, they operate only over rows that survived filtering, so the partition is over the filtered set.

**11. Top 3 per department.**
```sql
SELECT dept_id, emp_name, salary
FROM (
    SELECT dept_id, emp_name, salary,
           DENSE_RANK() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rnk
    FROM employees
) t
WHERE rnk <= 3
ORDER BY dept_id, salary DESC;
```
Partition by department, order by salary descending, and rank. Filtering must happen in the outer query because window results aren't available in `WHERE`. I chose `DENSE_RANK` so that ties at the third salary all qualify as "top 3 pay levels"; use `ROW_NUMBER` instead if you must return *exactly* three rows regardless of ties, and `RANK` if you want gap semantics.

**12. Month-over-month growth.**
```sql
SELECT month, revenue,
       LAG(revenue) OVER (ORDER BY month) AS prev_revenue,
       ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY month))
             / LAG(revenue) OVER (ORDER BY month), 1) AS mom_growth_pct
FROM monthly_revenue
ORDER BY month;
```
`LAG(revenue)` pulls the previous month's figure; the growth percent is `(current - previous) / previous * 100`. The first month's `prev_revenue` is NULL, so its growth is NULL — legitimately, since there's no prior period. If you'd rather show 0 or handle the division-by-zero of a zero previous month, wrap it with `COALESCE`/`NULLIF`. Using a CTE to compute `prev_revenue` once is cleaner than repeating the `LAG` three times.

**13. Keep latest row per user (dedup).**
```sql
SELECT user_id, /* other columns */ ...
FROM (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY updated_at DESC) AS rn
    FROM user_events
) t
WHERE rn = 1;
```
`ROW_NUMBER` is the right choice here (not `RANK`/`DENSE_RANK`) because you want *exactly one* row per user even if two rows share the same `updated_at` — `ROW_NUMBER` breaks ties and guarantees a single `rn = 1`. Partition by the dedup key (`user_id`), order by recency descending, and keep the first row of each partition. Add a tiebreaker to `ORDER BY` (e.g. `, id DESC`) to make the choice deterministic when timestamps tie.

**14. Why window functions exist alongside GROUP BY.**
Because `GROUP BY` is lossy: it aggregates and collapses, so you can't keep individual rows and their group aggregate at the same time — yet an enormous class of analytics questions needs exactly that ("show me each sale *and* its share of the daily total"). Before window functions, people simulated this with self-joins and correlated subqueries — one per metric — which were verbose and frequently O(N²). Window functions (SQL:2003) provide the missing capability: relative, per-row calculations over ordered groups (ranks, running totals, moving averages, period-over-period) computed efficiently in essentially one ordered pass, while preserving every row. They complement `GROUP BY` rather than replace it.

**15. ROWS vs RANGE choice.**
Use `ROWS` when you want a precise count of physical neighboring rows — a 3-row moving average, "the previous 6 rows", etc. — because it's unambiguous and cheaper, and it doesn't special-case ties. Use `RANGE` when tied `ORDER BY` values should be treated as a single peer group — for instance a running total where all rows sharing the same date should reflect the same cumulative value, or when you need value-based bounds (e.g. `RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW` for a true time-window rather than a row-count window). In short: `ROWS` for row-count logic, `RANGE` for value/peer logic. When ties are present the two can give different answers, so choose deliberately.

**16. Window vs GROUP BY aggregate.**
Use a `GROUP BY` aggregate when your desired output has *fewer* rows than the input — one summary per group — and you don't need the underlying detail (a report of total sales per region). Use a window function when you need to keep every input row and *annotate* it with a group-level or neighbor-relative value (each order alongside its region's total, or its rank within the region). If you find yourself joining a `GROUP BY` summary back to the detail table to attach the aggregate, that's the signal a window function is the cleaner tool. They can also be combined — window functions can operate over the results of a grouped query.

**17. RANK vs DENSE_RANK vs ROW_NUMBER use cases.**
`ROW_NUMBER`: deduplication and pagination — "keep exactly one row per user" or "rows 21–30 of the result", where you need a strict unique sequence with no ties. `RANK`: competition/standings where gaps are meaningful — "two people tied for 2nd place means the next is 4th", such as leaderboard positions. `DENSE_RANK`: "the Nth distinct value" queries where you don't want holes — "find employees earning the 3rd-highest distinct salary", or bucketing by distinct rank levels. The deciding questions are: do you need uniqueness (ROW_NUMBER), gap semantics on ties (RANK), or contiguous distinct levels (DENSE_RANK)?

## Common Mistakes

- **Filtering on a window function in `WHERE`** — it's not computed yet; wrap in a subquery/CTE.
- **The `LAST_VALUE` default-frame trap** — returns the current row's value; fix the frame or flip `ORDER BY` with `FIRST_VALUE`.
- **Forgetting `ORDER BY` for a running total** — `SUM() OVER ()` gives a grand total, not cumulative.
- **Using `ROW_NUMBER` when ties should share a rank** (or `RANK`/`DENSE_RANK` when you need exactly one row) — pick the function by tie semantics.
- **Non-deterministic `ROW_NUMBER`** — without a tiebreaker in `ORDER BY`, tied rows get arbitrary, unstable numbers across runs.
- **Confusing `RANK` gaps with `DENSE_RANK`** — remember RANK skips, DENSE_RANK doesn't.
- **Assuming `RANGE` and `ROWS` are interchangeable** — they differ whenever ties exist in the `ORDER BY` key.
- **Ignoring partition resets** — `LAG`/running totals restart at each `PARTITION BY` boundary, which is usually what you want but bites if you forgot to partition.
- **Expecting NULLs from `LAG`/`LEAD` at boundaries to be zero** — supply the default argument.
- **Heavy unindexed sorts** — window functions over large tables without supporting indexes can be slow.

## Related Concepts

- **`GROUP BY` aggregation** — the collapsing counterpart; windows are the non-collapsing version.
- **Frame clause / `ROWS` / `RANGE` / `GROUPS`** — controlling which neighbors the function sees.
- **CTEs and subqueries** — needed to filter on window results (top-N-per-group pattern).
- **Logical query processing order** — explains why windows can't appear in `WHERE`/`GROUP BY`.
- **`WINDOW` clause** — naming and reusing a window spec across multiple functions.
- **Ordered-set & hypothetical-set aggregates** — `PERCENTILE_CONT`, `PERCENTILE_DISC`, `MODE`.
- **Gap-and-island analysis** — a classic pattern built from ranking-function differences.
- **Indexes on partition/order keys** — the main lever for window-function performance.

---

# Views

## What is it?

A **view** is a named, stored `SELECT` query that behaves like a virtual table. It has a name and columns and you can query it exactly like a table — but by default it stores **no data**. Each time you query the view, the database substitutes the view's underlying `SELECT` and runs it against the live base tables. So a view is essentially a *saved query given a table-like name*.

```sql
CREATE VIEW active_customers AS
SELECT customer_id, name, email, signup_date
FROM customers
WHERE status = 'active';

-- Now query it like a table:
SELECT * FROM active_customers WHERE signup_date > '2026-01-01';
```

The view holds no rows of its own; `SELECT * FROM active_customers` re-runs the underlying query against `customers` every time, always reflecting current data.

Two important variants:

- **(Standard/logical) view** — virtual, stores no data, always current, recomputed on each access. The default `CREATE VIEW`.
- **Materialized view** — physically **stores** the query's result set on disk like a cached table. Fast to read (no recomputation), but **stale**: it only updates when you `REFRESH` it. PostgreSQL supports these natively (`CREATE MATERIALIZED VIEW`).

An **updatable view** is a (usually simple) view through which you can also run `INSERT`/`UPDATE`/`DELETE`, and the changes pass through to the base table.

## Why is it needed?

First principles: as schemas and queries grow, you need **abstraction, security, and reuse** at the database layer — the same needs that drive functions and modules in application code.

1. **Abstraction / simplification.** Hide a complex multi-join, aggregating query behind a simple name. Analysts query `SELECT * FROM monthly_revenue_by_region` without knowing the six joins beneath it.
2. **Reuse across statements and users.** Unlike a CTE (single-statement scope), a view is a persistent, shared object usable by any query, session, or user — a single source of truth for a business definition (e.g. what "active customer" means).
3. **Security / access control.** Grant users access to a view that exposes only certain columns/rows, without granting access to the base table. This is column- and row-level security: a view over `employees` that omits `salary`, or filters to the current user's own rows.
4. **Logical data independence.** Views decouple applications from the physical schema. If you refactor base tables, you can often preserve the view's shape so dependent queries keep working — the view is a stable contract.
5. **Consistency of business logic.** Encapsulate a computation (revenue = qty × price − discount) once in a view so every consumer computes it identically, instead of each team re-deriving it (and disagreeing).
6. **Performance (materialized views).** Precompute and cache expensive aggregations/joins so dashboards read instantly, accepting controlled staleness.

## How does it work?

### Creating and replacing

```sql
CREATE VIEW employee_summary AS
SELECT e.emp_id, e.emp_name, d.dept_name, e.salary
FROM employees e
JOIN departments d ON d.dept_id = e.dept_id;

-- Redefine (must keep the same output columns to just add):
CREATE OR REPLACE VIEW employee_summary AS
SELECT e.emp_id, e.emp_name, d.dept_name, e.salary, e.hire_date
FROM employees e
JOIN departments d ON d.dept_id = e.dept_id;

DROP VIEW employee_summary;
```

When you query the view, the planner **expands** its definition into your query and optimizes the whole thing together (view inlining), so a view over a filtered table can still use the base table's indexes and push your extra predicates down.

### Updatable views

A view is automatically updatable (you can `INSERT`/`UPDATE`/`DELETE` through it) only if it's **simple enough** that each result row maps unambiguously to exactly one base-table row. In PostgreSQL, that means the view:

- selects from exactly **one** base table (or another updatable view),
- has **no** `DISTINCT`, `GROUP BY`, `HAVING`, aggregates, window functions, `UNION`/`INTERSECT`/`EXCEPT`, `LIMIT`, or set-returning functions in the target list,
- and its selected columns are simple column references (not expressions) for the columns you want to modify.

```sql
CREATE VIEW active_customers AS
SELECT customer_id, name, email, status
FROM customers
WHERE status = 'active';

-- This passes through to the customers table:
UPDATE active_customers SET email = 'new@x.com' WHERE customer_id = 42;
INSERT INTO active_customers (customer_id, name, email, status)
VALUES (99, 'Zoe', 'zoe@x.com', 'active');
```

For views too complex to be auto-updatable (joins, aggregates), PostgreSQL lets you make them writable manually with **`INSTEAD OF` triggers** that translate the DML into the appropriate base-table operations.

### WITH CHECK OPTION

There's a subtle hole in the `INSERT`/`UPDATE` above: nothing stops you inserting or updating a row that **doesn't satisfy the view's `WHERE`**. You could `INSERT` a customer with `status = 'inactive'` through `active_customers` — the row lands in `customers`, but then *vanishes from the view* (it doesn't match `status = 'active'`), or worse you could `UPDATE ... SET status = 'inactive'` and move a row out of the view. `WITH CHECK OPTION` closes this: it enforces that any row inserted or updated through the view must still satisfy the view's defining condition.

```sql
CREATE VIEW active_customers AS
SELECT customer_id, name, email, status
FROM customers
WHERE status = 'active'
WITH CHECK OPTION;

-- Now this is REJECTED, because the new row wouldn't be visible in the view:
INSERT INTO active_customers (customer_id, name, email, status)
VALUES (100, 'Ivan', 'ivan@x.com', 'inactive');   -- ERROR

UPDATE active_customers SET status = 'inactive' WHERE customer_id = 42;  -- ERROR
```

`LOCAL` vs `CASCADED` check option controls whether the check applies only to this view's condition (`LOCAL`) or also to all underlying views' conditions (`CASCADED`, the SQL default).

### Materialized views (PostgreSQL) and REFRESH

A materialized view stores the query result physically:

```sql
CREATE MATERIALIZED VIEW monthly_sales AS
SELECT date_trunc('month', order_date) AS month,
       region,
       SUM(amount) AS total
FROM orders
GROUP BY 1, 2;

-- Reads are fast (no recompute), but data is a snapshot as of creation/refresh.
SELECT * FROM monthly_sales WHERE region = 'EU';
```

To bring it up to date you must refresh it:

```sql
REFRESH MATERIALIZED VIEW monthly_sales;
```

Plain `REFRESH` takes an `ACCESS EXCLUSIVE` lock — the view is unreadable while refreshing. To avoid blocking readers, use the concurrent form (which requires a **unique index** on the materialized view):

```sql
CREATE UNIQUE INDEX ON monthly_sales (month, region);
REFRESH MATERIALIZED VIEW CONCURRENTLY monthly_sales;
```

`CONCURRENTLY` recomputes into a temporary result and applies the diff, keeping the view readable throughout (at the cost of being slower and needing that unique index). Note: PostgreSQL materialized views are **not** auto-refreshing — you schedule the refresh (cron, `pg_cron`, a trigger, or a job). Some other databases offer incrementally/auto-maintained materialized views.

### Views vs CTEs vs temp tables

| | View | CTE | Temp table |
|---|---|---|---|
| Persistence | Permanent schema object | Single statement only | Session/transaction |
| Stores data? | No (materialized: yes) | No | Yes |
| Reuse | Across all queries/users | Within one statement | Within one session |
| Indexable | Base table indexes; matview yes | No | Yes |
| Always current? | Yes (matview: on refresh) | Yes | Snapshot when populated |
| Lifecycle | `CREATE`/`DROP`, managed | Automatic | `CREATE`/populate/drop |

## Internal Working

**Standard views are macro-substitution.** A view stores only its definition (the parsed query tree) in the catalog — no data. When you reference a view, PostgreSQL's **rule system** rewrites your query, expanding the view reference into its underlying query (an `ON SELECT DO INSTEAD` rule). The combined query is then planned and optimized as a whole. Consequently, filters you add on top of the view are pushed into the underlying query and can use the base tables' indexes — a view over `WHERE status='active'` plus your `WHERE signup_date > …` is optimized together, not as two separate passes. There is essentially **zero storage and zero inherent runtime cost** to a plain view beyond running its query; it's not a cache.

**A view is not automatically faster.** Because it re-executes each time, a view over an expensive join is exactly as expensive as writing that join out. Views organize and secure; they don't accelerate (that's what materialized views are for).

**Updatable views** are rewritten so that DML against the view is redirected to the base table via auto-generated (or `INSTEAD OF` trigger) rules. `WITH CHECK OPTION` adds a validation step: after computing the new/updated row, the engine verifies it satisfies the view's qualification and raises an error if not.

**Materialized views** allocate real storage (a heap, like a table). At `CREATE`/`REFRESH`, the defining query runs and its output is written to that heap; subsequent reads scan the stored heap directly — that's the speedup. You can build **indexes** on a materialized view (unlike a plain view), which is central to their performance value and required (a unique one) for `REFRESH ... CONCURRENTLY`. The cost model: cheap, fast reads; expensive periodic refreshes; and **staleness** between refreshes. Plain `REFRESH` fully recomputes and locks; `CONCURRENTLY` recomputes into a side result and merges the difference, trading speed for availability.

**Dependency tracking.** The catalog records view-on-table and view-on-view dependencies, so you can't drop a column/table a view needs without `CASCADE`. This is what gives views their "logical data independence" contract — the database knows what depends on what.

## Advantages

- **Abstraction** — hide complex joins/aggregations behind a simple name; simpler consumer queries.
- **Security** — expose only chosen rows/columns; grant on the view, not the base table (row/column-level security).
- **Reuse & single source of truth** — a business definition lives in one place, shared across users and queries.
- **Logical data independence** — insulate applications from base-schema changes; the view is a stable interface.
- **Consistency** — encapsulate a computation so everyone computes it identically.
- **Materialized views** — precompute expensive results for fast dashboard reads.
- **Maintainability** — change the logic once in the view; all consumers benefit.

## Limitations

- **No performance gain for plain views** — they re-execute every time; a view over a slow query is still slow.
- **Updatability restrictions** — complex views (joins, aggregates) aren't auto-updatable; need `INSTEAD OF` triggers.
- **Materialized views go stale** — require explicit `REFRESH`; no built-in auto-refresh in PostgreSQL.
- **Refresh cost & locking** — plain `REFRESH` locks the view; `CONCURRENTLY` is slower and needs a unique index.
- **Nested-view complexity** — views built on views built on views become hard to reason about and can produce surprising plans.
- **Dependency rigidity** — base-schema changes may require dropping/recreating dependent views (`CASCADE`).
- **Not indexable (plain views)** — only the base tables (or a materialized view) can be indexed.
- **Hidden cost** — a simple-looking `SELECT * FROM view` may mask a very expensive underlying query.

## Real-world Applications

- **Security views** — expose `employees_public` (no salary/SSN) or a per-tenant view filtered to the current user's org, granting access only to the view.
- **Reporting abstractions** — `monthly_revenue_by_region`, `customer_lifetime_value` as named, reusable business definitions for BI tools.
- **Backward-compatible schema migration** — after splitting/renaming tables, a view with the old name/shape keeps legacy queries working.
- **Simplifying APIs for analysts** — hide 8-table joins so non-experts can self-serve with simple `SELECT`s.
- **Materialized dashboards** — precompute heavy aggregates refreshed nightly/hourly for instant-loading dashboards.
- **Data masking / GDPR** — views that redact or hash PII for certain roles.
- **Denormalized read models** — a materialized view joining normalized tables into a wide, query-friendly shape for search/list screens.

## Interview Questions

**Beginner**

1. What is a view and does it store data?
2. How do you create and drop a view?
3. What's the difference between a view and a table?

**Intermediate**

4. What makes a view updatable? Give conditions that make one non-updatable.
5. What does `WITH CHECK OPTION` do, and what problem does it solve?
6. What is a materialized view and how does it differ from a regular view?

**Advanced**

7. How does a plain view execute internally, and does using a view improve performance?
8. Explain `REFRESH MATERIALIZED VIEW` vs `REFRESH ... CONCURRENTLY`, including the locking and index requirements.
9. How do you make a complex (multi-table) view updatable in PostgreSQL?

**Scenario-based**

10. You must let a reporting team read customer data but never see credit-card numbers. How do you use views?
11. A dashboard aggregates millions of rows and is too slow. Walk through using a materialized view, including keeping it reasonably fresh without blocking readers.
12. You have `CREATE VIEW active AS SELECT ... WHERE status='active'` and someone updates a row's status to 'inactive' through the view. What happens, and how do you prevent it?

**"Why" questions**

13. Why doesn't a regular view improve query performance?
14. Why would you use a view instead of just repeating the query, and why a view over a CTE?

**Comparison questions**

15. View vs materialized view vs table — compare storage, freshness, and cost.
16. View vs CTE vs temporary table — when to use each.

## Model Answers

**1. What is a view, does it store data?**
A view is a named, stored `SELECT` statement that behaves like a virtual table. By default it stores **no data** — only its query definition lives in the catalog. Each time you query the view, the database re-runs the underlying query against the live base tables, so a view always reflects current data. You interact with it exactly like a table (`SELECT ... FROM view`), which is why it's often described as "a saved query with a table-like name." The exception is a *materialized* view, which does physically store its result set.

**2. Create and drop.**
`CREATE VIEW name AS SELECT ...;` defines it; `CREATE OR REPLACE VIEW name AS SELECT ...;` redefines an existing one (in PostgreSQL you may add trailing columns but not remove or reorder existing ones); `DROP VIEW name;` removes it (`DROP VIEW ... CASCADE` also drops objects that depend on it). For materialized views the analogous statements are `CREATE MATERIALIZED VIEW`, `REFRESH MATERIALIZED VIEW`, and `DROP MATERIALIZED VIEW`. You can also `GRANT`/`REVOKE` privileges on a view independently of its base tables, which is central to using views for access control.

**3. View vs table.**
A table physically stores rows on disk and owns its data; a (plain) view stores no rows — it's a stored query that computes its rows on demand from base tables, so it's always in sync with them and consumes negligible storage. You can index and constrain a table directly; a plain view relies on the base tables' indexes and can't have its own. Tables have independent lifecycles; views depend on their base tables and break/refuse to drop underlying columns without `CASCADE`. A materialized view sits in between: it stores data like a table but derives it from a query and must be refreshed to stay current.

**4. What makes a view updatable.**
A view is auto-updatable when each of its rows maps unambiguously to exactly one base-table row, so the engine knows precisely which base row an `INSERT`/`UPDATE`/`DELETE` should affect. In PostgreSQL that requires the view to draw from a single base table (or another updatable view), with its modifiable columns being plain column references, and with **no** `DISTINCT`, `GROUP BY`, `HAVING`, aggregate or window functions, `UNION`/`INTERSECT`/`EXCEPT`, `LIMIT`, or set-returning functions in the select list. Anything that combines, summarizes, or de-duplicates rows (a join producing ambiguous mappings, an aggregate collapsing many rows into one) makes the view non-updatable automatically, because there's no single base row to write back to. Such views can still be made writable via `INSTEAD OF` triggers.

**5. WITH CHECK OPTION.**
`WITH CHECK OPTION` enforces that any row you `INSERT` or `UPDATE` through the view must still satisfy the view's own `WHERE` condition. The problem it solves: without it, you can insert or update a row through a filtered view such that the row does **not** match the filter — it lands in the base table but is invisible through the view (or an update pushes an existing row *out* of the view). For an `active_customers` view (`WHERE status='active'`), that would let you insert an inactive customer or flip one to inactive through the very view meant to contain only active ones. With the check option, such operations are rejected. The `LOCAL`/`CASCADED` qualifier decides whether only this view's predicate is checked or also those of underlying views.

**6. Materialized view vs regular view.**
A regular view stores only a query definition and recomputes results on every access, so it's always current but pays the query cost each time. A materialized view physically stores the query's result set on disk (like a cached table), so reads are fast because there's no recomputation — but the stored data is a **snapshot** that becomes stale as the base tables change, and you must explicitly `REFRESH` it to update. Materialized views can also have their own indexes, which regular views cannot. The trade-off is freshness vs read speed: use a regular view when you need always-current data and the underlying query is cheap enough; use a materialized view when the query is expensive and you can tolerate controlled staleness for much faster reads.

**7. Plain view internals and performance.**
Internally a plain view is macro-substitution: PostgreSQL's rule system rewrites your query by expanding the view reference into its underlying query, then plans and optimizes the whole thing together. Crucially, predicates you add on top of the view are pushed into the underlying query and can use the base tables' indexes — the view doesn't create an optimization barrier for simple cases. Because it re-executes the underlying query every time, a plain view provides **no** inherent performance improvement: a view over an expensive six-way join costs the same as writing that join out. Views are for abstraction, security, and reuse — not speed. When you need speed for an expensive query, you use a *materialized* view, which stores the result and can be indexed.

**8. REFRESH vs REFRESH CONCURRENTLY.**
`REFRESH MATERIALIZED VIEW mv` fully recomputes the defining query and replaces the stored contents, but it takes an `ACCESS EXCLUSIVE` lock, so the view cannot be read while the refresh runs — fine for off-hours batch jobs, bad for a 24/7 dashboard. `REFRESH MATERIALIZED VIEW CONCURRENTLY mv` recomputes into a temporary result and then applies only the differences to the existing view, holding a weaker lock that lets readers keep querying throughout. The trade-offs: `CONCURRENTLY` is slower overall (it computes and then diffs) and it **requires a unique index** on the materialized view so it can identify rows to update — without one, PostgreSQL rejects the concurrent refresh. In both cases the refresh is manual/scheduled; PostgreSQL doesn't auto-refresh materialized views.

**9. Making a complex view updatable.**
PostgreSQL won't auto-update a view that joins tables or aggregates, because a result row doesn't map to a single base row. You make it writable with **`INSTEAD OF` triggers**: you create a trigger `INSTEAD OF INSERT OR UPDATE OR DELETE ON the_view` whose function contains the logic to translate the operation into concrete DML on the appropriate base table(s) — e.g. an `INSERT` into a two-table view might insert into the primary table and upsert into the secondary, using the trigger's `NEW`/`OLD` row values. The trigger takes full responsibility for deciding what "modifying this view row" means, which is why it's required: only you know the intended write semantics for an ambiguous multi-table mapping. (Older PostgreSQL/other engines used rewrite `RULE`s for this; `INSTEAD OF` triggers are the modern, clearer approach.)

**10. Hide credit-card numbers.**
Create a view that selects only the non-sensitive columns (omitting the credit-card column entirely, or exposing a masked/last-4 version), then grant the reporting role `SELECT` on that view while **revoking** its access to the underlying `customers` table:
```sql
CREATE VIEW customers_reporting AS
SELECT customer_id, name, email, city,
       'xxxx-xxxx-xxxx-' || right(card_number, 4) AS card_masked
FROM customers;
GRANT SELECT ON customers_reporting TO reporting_role;
REVOKE ALL ON customers FROM reporting_role;
```
Because the role can't touch the base table and the view never exposes the raw card number, the sensitive data is unreachable. This is column-level security via views; the same technique with a `WHERE` clause (e.g. `WHERE tenant_id = current_setting('app.tenant')::int`) gives row-level security. For robust protection you'd combine this with revoking base-table privileges and, ideally, PostgreSQL Row-Level Security policies.

**11. Speeding up a slow dashboard.**
Replace the live aggregate query with a **materialized view** that precomputes it:
```sql
CREATE MATERIALIZED VIEW dash_metrics AS
SELECT region, date_trunc('day', ts) AS day, count(*) AS n, sum(amount) AS total
FROM events GROUP BY 1, 2;
CREATE UNIQUE INDEX ON dash_metrics (region, day);   -- needed for CONCURRENTLY
```
The dashboard now reads pre-aggregated rows instantly instead of scanning millions of raw rows each load, and I can add indexes on the materialized view for its filter/sort columns. To keep it fresh without blocking the dashboard, schedule `REFRESH MATERIALIZED VIEW CONCURRENTLY dash_metrics` on an interval appropriate to the business (say every 15 minutes or hourly) using `pg_cron` or an external scheduler — `CONCURRENTLY` keeps the view readable during refresh, which the unique index enables. The trade-off I'd make explicit to stakeholders is bounded staleness (data is up to N minutes old) in exchange for fast, predictable read latency.

**12. Update that moves a row out of the view.**
Without `WITH CHECK OPTION`, the `UPDATE ... SET status='inactive'` succeeds: the change passes through to the base `customers` table, the customer becomes inactive, and the row simply disappears from the `active` view (it no longer matches `WHERE status='active'`). That's often not what you want — the view meant to represent "active customers" just let you deactivate one through itself. To prevent it, define the view `WITH CHECK OPTION`; then any `INSERT` or `UPDATE` through the view that would produce a row not satisfying `status='active'` is rejected with an error, so the view can only ever contain — and only ever be used to write — genuinely active rows.

**13. Why no performance gain from a plain view.**
Because a plain view stores no results — it's just a saved query that the database expands and re-executes against the base tables on every access. There's no caching, no precomputation, no stored rows; a `SELECT` from a view runs the same work as writing the underlying query inline. If that underlying query is a slow six-table join, the view is slow every single time. Views optimize for humans (abstraction, security, reuse), not for the machine. When you actually need to make an expensive query fast, you materialize it — a materialized view stores the computed result so reads skip the recomputation, which is precisely the performance tool a plain view is not.

**14. View vs repeating the query; view vs CTE.**
Versus repeating the query: a view gives you a single source of truth. The logic lives in one place, so every consumer computes the business definition identically, and when the definition changes you edit one object instead of hunting down copies that may have drifted apart — this is DRY applied to SQL, plus it enables granting security on the abstraction. Versus a CTE: a CTE is scoped to a single statement and vanishes afterward, so it can't be shared across queries, sessions, or users; a view is a persistent, catalog-level object reusable everywhere and grantable to roles. So you use a CTE to decompose *one* complex query for readability, and a view when the same logic must be reused across *many* queries or exposed as a stable, secured interface to other people and tools.

**15. View vs materialized view vs table.**
A **table** owns its data physically, is always the authoritative current state, is fully indexable and writable, and costs storage plus write maintenance. A **plain view** stores no data — it derives rows on demand from base tables, so it's always current, costs almost no storage, but pays full query cost on every read and can't be indexed itself. A **materialized view** stores a derived snapshot physically (like a table) and can be indexed, giving fast reads, but it's stale between refreshes and costs a periodic (sometimes locking) `REFRESH`. Freshness ranking: table/plain view are always current; materialized view is as-of last refresh. Cost ranking: plain view is cheap to maintain but expensive to read for heavy queries; materialized view is cheap to read but costs refreshes and storage; a table costs storage and write upkeep. Choose plain view for always-current abstraction over cheap queries, materialized view for fast reads of expensive derived data you can let go stale, and a table for authoritative source data you write to.

**16. View vs CTE vs temp table.**
A **CTE** is scoped to a single statement, stores no data, isn't indexable, and needs no lifecycle management — ideal for decomposing one complex query into readable named steps. A **temporary table** persists for the session or transaction, physically stores rows, can be indexed and analyzed, and is visible to multiple statements in that session — ideal when you build a heavy intermediate result once and reuse it across several queries, or when you need an index on it, at the cost of explicit create/populate/drop. A **view** is a permanent, shared catalog object reusable across all queries, sessions, and users, and grantable for security — ideal as a durable named abstraction or security boundary, though (unless materialized) it recomputes each time. Rule of thumb: CTE for in-statement readability, temp table for multi-statement/session reuse and indexing, view for a durable, shared, secured abstraction.

## Common Mistakes

- **Expecting a plain view to be faster** than its underlying query — it isn't; it re-executes each time.
- **Forgetting `WITH CHECK OPTION`** on a filtered updatable view, allowing rows to be inserted/updated out of the view's scope.
- **Assuming materialized views auto-refresh** in PostgreSQL — they don't; you must schedule `REFRESH`.
- **Using plain `REFRESH` on a live-read materialized view** — it locks out readers; use `CONCURRENTLY` (with a unique index).
- **Forgetting the unique index** required by `REFRESH ... CONCURRENTLY`.
- **Trying to `UPDATE` a non-updatable view** (join/aggregate) without an `INSTEAD OF` trigger.
- **Deeply nested views** (views on views on views) that hide runaway cost and become unmaintainable.
- **Granting on base tables instead of the view**, defeating the security purpose.
- **Treating a materialized view as always-current** and reporting stale numbers.
- **`SELECT *` in a view definition** — column changes in the base table can silently alter or break the view.

## Related Concepts

- **CTEs & subqueries** — the in-statement counterparts to a persistent view.
- **Temporary tables** — session-scoped materialization with indexing.
- **Materialized views & `REFRESH`** — cached, indexable derived data (PostgreSQL, Oracle, SQL Server indexed views).
- **`INSTEAD OF` triggers & rules** — making complex views writable.
- **Row-Level Security (RLS) & `GRANT`/`REVOKE`** — the security model views participate in.
- **Logical data independence** — the architectural principle views provide.
- **Query rewrite / view inlining / predicate pushdown** — how the planner expands and optimizes views.
- **Indexed views (SQL Server) / `pg_cron`** — engine-specific materialization and scheduling.


---

# Indexing

## What is it?

An index is an auxiliary, on-disk data structure that the database maintains **alongside** your table so it can find rows without scanning every one of them. The mental model I give juniors: a table is a book, and an index is the alphabetical index at the back. If you want every page mentioning "B-Tree", you do not read the whole book cover to cover — you jump to the "B" section, read the page numbers, and flip directly there.

Formally, an index is a mapping from **key values** (one or more columns) to **row locations** (in PostgreSQL, a physical tuple pointer called a `ctid` — a `(page, offset)` pair). The index stores the keys in a structure that supports fast lookup — usually a balanced tree — so that a query filtering on those keys touches a tiny fraction of the data.

```sql
-- The table has millions of rows
CREATE TABLE users (
    id         BIGINT PRIMARY KEY,   -- PK implicitly creates a unique B-Tree index
    email      TEXT,
    created_at TIMESTAMPTZ,
    country    TEXT
);

-- Without an index, this must read every row (a "sequential scan")
SELECT * FROM users WHERE email = 'ada@example.com';

-- Create an index so the lookup becomes O(log n)
CREATE INDEX idx_users_email ON users (email);
```

A crucial first-principles point: **an index is redundant, derived data.** Every fact it holds already exists in the table. You are trading disk space and write cost for read speed. That trade is the entire subject of this topic.

## Why is it needed?

Consider the physics. A table with 10 million rows, at ~200 bytes/row, is ~2 GB. To answer `WHERE email = ?` without an index, the engine must read all ~2 GB from disk and compare every row — a **full table scan**, O(n). Even on fast NVMe that is hundreds of milliseconds to seconds, and it competes for I/O and buffer cache with every other query.

With a B-Tree index, the same lookup is **O(log n)**. For 10 million rows, log base a few hundred (the tree fan-out) is 3–4 levels. The engine reads 3–4 index pages to find the pointer, then 1 page to fetch the row. That is ~5 page reads instead of ~250,000. This is not a 2x improvement; it is often a 10,000x improvement. That gap is why indexing is the single highest-leverage skill in query performance.

Indexes are needed wherever the database must **locate**, **order**, or **verify uniqueness** of data efficiently:

- **Point lookups & range scans** — `WHERE id = 42`, `WHERE created_at BETWEEN ...`.
- **Joins** — the join column on at least one side almost always needs an index, or the planner falls back to hashing/sorting entire tables.
- **Sorting & grouping** — a B-Tree stores keys in order, so `ORDER BY indexed_col` can skip the sort entirely.
- **Uniqueness constraints** — `PRIMARY KEY` and `UNIQUE` are *enforced* by an index; checking "does this value already exist?" on every insert would otherwise be a full scan.

## How does it work?

When you `CREATE INDEX`, PostgreSQL reads the table once, extracts the key column(s) plus each row's `ctid`, sorts them, and builds a tree whose leaves point back at the heap (the table's raw storage). From then on:

- **On write** (`INSERT`/`UPDATE`/`DELETE`), the engine updates the index too, keeping it in sync. This is the write cost.
- **On read**, the planner *considers* the index. It is not forced to use it — the query optimizer estimates the cost of an index scan versus a sequential scan and picks the cheaper plan (covered in Topic 3).

**Creation and management:**

```sql
-- Basic index
CREATE INDEX idx_users_country ON users (country);

-- Build without locking out writes (essential in production; slower, two passes)
CREATE INDEX CONCURRENTLY idx_users_country ON users (country);

-- Choose the access method explicitly
CREATE INDEX idx_users_email_hash ON users USING HASH (email);

-- Name, inspect, drop
\d users                              -- psql: lists indexes on the table
DROP INDEX idx_users_country;
DROP INDEX CONCURRENTLY idx_users_country;

-- Rebuild a bloated index (e.g. after heavy churn)
REINDEX INDEX CONCURRENTLY idx_users_email;

-- See index sizes and usage
SELECT indexrelname, idx_scan, pg_size_pretty(pg_relation_size(indexrelid))
FROM pg_stat_user_indexes
WHERE relname = 'users';
```

**Query optimization via indexes** — the payoff. Given `SELECT * FROM users WHERE email = 'ada@example.com'`:

1. Planner sees a predicate on `email`.
2. It finds `idx_users_email` covers that column.
3. It estimates: index scan costs ~5 page reads; seq scan costs ~250k. Index wins.
4. It descends the B-Tree to the matching key, reads the `ctid`, fetches that heap page, returns the row.

You *enable* this by putting indexes on the columns you filter, join, and sort on — and, just as importantly, by writing predicates the index can use (SARGability, Topic 3).

### Types of indexes

**B-Tree (default).** Balanced tree, keeps keys sorted. Supports `=`, `<`, `<=`, `>`, `>=`, `BETWEEN`, `IN`, `IS NULL`, prefix `LIKE 'abc%'`, and `ORDER BY`. This is the right answer ~95% of the time.

```sql
CREATE INDEX idx_orders_created ON orders (created_at);  -- great for range + sort
```

**Hash.** Stores a hash of the key. Supports **only equality** (`=`) — no ranges, no ordering. Lookups are O(1) average instead of O(log n), and the index can be smaller for long keys. In modern PostgreSQL (10+) hash indexes are WAL-logged and crash-safe, but they remain niche because B-Tree equality is already fast and far more flexible.

```sql
CREATE INDEX idx_sessions_token ON sessions USING HASH (token);  -- equality-only lookups
```

**Composite (multi-column).** Indexes several columns as an ordered tuple. Governed by the **leftmost-prefix rule**: an index on `(a, b, c)` can serve predicates on `a`, `a,b`, or `a,b,c` — a *prefix* of the column list — but **not** `b` alone or `c` alone, because the tree is sorted by `a` first, then `b` within equal `a`, then `c`.

```sql
CREATE INDEX idx_orders_cust_date ON orders (customer_id, created_at);

-- USES the index (prefix = customer_id, then range on created_at)
SELECT * FROM orders WHERE customer_id = 7 AND created_at >= '2026-01-01';

-- USES the index (leftmost column alone)
SELECT * FROM orders WHERE customer_id = 7;

-- Does NOT use it efficiently (skips the leftmost column)
SELECT * FROM orders WHERE created_at >= '2026-01-01';
```

Rule of thumb for column order: put **equality** columns first, then the **range/sort** column last. Also put the most **selective** equality column first when several are equality-filtered.

**Unique.** Enforces no duplicate key values *and* provides a fast lookup. Backs `PRIMARY KEY` and `UNIQUE` constraints.

```sql
CREATE UNIQUE INDEX idx_users_email_uniq ON users (email);
```

**Partial.** Indexes only the rows matching a `WHERE` clause. Smaller, cheaper to maintain, and perfect when queries always target a subset (e.g. active rows, or non-null values).

```sql
-- Only index unfulfilled orders; the index stays tiny even as history grows
CREATE INDEX idx_orders_pending ON orders (created_at)
WHERE status = 'pending';

-- Enforce "one primary address per user" only among primaries
CREATE UNIQUE INDEX one_primary_addr ON addresses (user_id)
WHERE is_primary;
```

**Covering indexes.** If an index contains *every column a query needs*, the engine answers from the index alone and never touches the heap — an **index-only scan**. PostgreSQL supports `INCLUDE` to bolt non-key payload columns onto the leaf entries without making them part of the search key.

```sql
-- Query needs email (filter) and full_name (output)
CREATE INDEX idx_users_email_cover ON users (email) INCLUDE (full_name);

SELECT email, full_name FROM users WHERE email = 'ada@example.com';
-- Can be an Index Only Scan: no heap fetch needed
```

(One caveat: index-only scans still consult the *visibility map* to confirm a row is visible to your transaction; heavy write churn can force heap fetches until `VACUUM` runs.)

**Selectivity** — the fraction of rows a predicate keeps. High selectivity = few rows match = index is very worthwhile. Low selectivity (e.g. a `boolean` or `gender` column where 50% match) means the index returns half the table; the planner will often (correctly) prefer a sequential scan because random index+heap I/O for half the rows is slower than one sequential sweep.

**When NOT to index:**

- **Low-selectivity columns** queried alone (booleans, small enums) — the planner won't use it, and you pay write cost for nothing. A partial index may still help.
- **Small tables** — a few hundred rows fit in one or two pages; a seq scan is already optimal.
- **Write-heavy, read-rarely tables** — every index multiplies write cost.
- **Columns you never filter/join/sort on** — an index nobody queries is pure overhead.
- **Wide, rapidly-updated columns** — churn causes index bloat.

**Write-amplification** — the core cost. Every index on a table must be updated on every relevant write. A table with 8 indexes turns one `INSERT` into 1 heap write + 8 index writes. `UPDATE`s that change indexed columns must delete-and-reinsert the index entry (PostgreSQL's HOT optimization avoids this *only* when no indexed column changed and the new tuple fits on the same page). This is why "just add an index" is not free advice: indexes accelerate reads and **tax every write**, plus consume disk and cache.

## Internal Working

The default index is a **B-Tree** — specifically a B+Tree variant (Lehman & Yao's high-concurrency design). "Balanced" means every leaf is at the same depth, guaranteeing the *same* number of page reads for any key. That is the property that turns O(n) into O(log n).

Structure:

- **Root page** — one page at the top; entry point for every search.
- **Internal (branch) pages** — hold *separator keys* and child pointers. They route searches; they do **not** point at table rows.
- **Leaf pages** — hold the actual keys **in sorted order**, each paired with a `ctid` (heap pointer). Leaves are chained in a **doubly-linked list**, so once you find the start of a range you walk sideways leaf-to-leaf without re-descending — this is what makes range scans and `ORDER BY` cheap.

Each page holds many keys (high **fan-out** — often hundreds), so the tree is very shallow. Depth ≈ log_fanout(N). With fan-out 300, one billion rows is only ~4 levels. Root and upper levels stay pinned in the buffer cache, so a lookup is effectively a couple of physical reads.

```
                              ROOT (internal page)
                     +----------------------------------+
                     |   [ *|50| *|200| * ]             |   separator keys route the search
                     +----+------+-------+--------------+
                          |      |       |
             <50          |  50..199     |  >=200
            +-------------+       |       +--------------+
            v                     v                      v
     INTERNAL PAGE          INTERNAL PAGE          INTERNAL PAGE
     [ *|17| *|33| * ]      [ *|90|*|140|* ]       [ *|260|*|400|* ]
        |    |    |            |    |   |              |     |    |
        v    v    v            v    v   v              v     v    v
   +-------------------  LEAF PAGES (sorted keys + ctid, doubly linked) -------------------+
   | [10→c][17→c]  <->  [33→c][41→c]  <->  [90→c][121→c] <-> [140→c][155→c] <-> [260→c]... |
   +---------------------------------------------------------------------------------------+
     ^ each leaf entry: key -> ctid(page,offset) into the HEAP (the table)
     <-> sibling links let a range scan walk leaves without touching internal pages again
```

**A search for key = 140:**
1. Start at ROOT. 140 is ≥ 50 and < 200 → follow the middle child pointer.
2. In the internal page `[*|90|*|140|*]`, 140 ≥ 140 → follow the right child.
3. Reach the leaf, find `140→ctid`. 3 page reads.
4. Use the `ctid` to fetch the heap page and read the row (unless it's an index-only scan).

**A range scan `key BETWEEN 33 AND 140`:** descend once to the leaf containing 33, then follow **sibling links** rightward reading 33, 41, 90, 121, 140 — no more root/internal traversals. This sequential leaf walk is why B-Trees dominate range and sort workloads.

**Inserts and balance:** a new key goes into the correct leaf. If the leaf is full, it **splits** into two, and the median key is pushed up to the parent. If that cascades to the root, the tree grows one level taller — *at the top*, which is how every leaf stays at equal depth. Deletes may merge or leave pages partially empty (PostgreSQL reclaims space lazily via `VACUUM`, which is a source of **index bloat**).

The key takeaways to state in an interview: balanced ⇒ uniform O(log n); high fan-out ⇒ shallow tree ⇒ few I/Os; sorted leaves + sibling links ⇒ cheap ranges and ordering; splits propagate upward ⇒ balance is maintained automatically.

## Advantages

- **Dramatically faster reads** — O(log n) lookups and range scans instead of O(n) scans; frequently orders of magnitude.
- **Efficient ordering & grouping** — sorted structure lets `ORDER BY`/`GROUP BY`/`DISTINCT`/`MIN`/`MAX` skip explicit sorts.
- **Fast joins** — an index on the join key enables index-nested-loop joins instead of full hashes/sorts.
- **Constraint enforcement** — unique and primary-key checks are O(log n) instead of O(n).
- **Index-only scans** — covering indexes answer queries without heap access, cutting I/O further.
- **Selective maintenance** — partial indexes keep only the hot subset, staying small and cheap.

## Limitations

- **Write amplification** — every write must maintain every relevant index; more indexes = slower `INSERT`/`UPDATE`/`DELETE`.
- **Disk & memory cost** — indexes can rival or exceed table size; they compete for buffer cache.
- **Not always used** — low selectivity, functions on the column, or type mismatches make the planner ignore them.
- **Bloat** — churn leaves dead entries; needs `VACUUM`/`REINDEX` to reclaim.
- **Maintenance overhead** — someone must choose, monitor, and prune them; unused indexes are silent tax.
- **Leftmost-prefix constraint** — a composite index does not help predicates that skip its leading column(s).
- **No help for non-SARGable predicates** — `WHERE LOWER(email) = ...` won't use a plain index on `email`.

## Real-world Applications

- **OLTP primary/foreign keys** — every PK is a unique index; FK columns get indexes so joins and cascades are fast.
- **Login/lookup by natural key** — unique index on `email`/`username` for O(log n) authentication.
- **Time-series & logs** — B-Tree on `created_at` (or `(device_id, ts)` composite) for range queries and retention scans; partial indexes for "recent/active" subsets.
- **E-commerce filtering** — composite indexes matching common filter+sort combos (`(category_id, price)`).
- **Multi-tenant SaaS** — leading `tenant_id` column in composite indexes so every query is scoped and selective.
- **Soft-delete systems** — partial indexes `WHERE deleted_at IS NULL` so the index ignores tombstoned rows.
- **Covering dashboards** — `INCLUDE` columns so hot aggregate/lookup queries run as index-only scans.

## Interview Questions

**Beginner**
1. What is a database index and what problem does it solve?
2. Does adding an index speed up reads, writes, or both? Explain.
3. What kind of index does a `PRIMARY KEY` create?

**Intermediate**
4. Explain the leftmost-prefix rule for composite indexes with an example.
5. What is a covering index / index-only scan, and when does it help?
6. What is selectivity and why does the planner care about it?

**Advanced**
7. Walk through how a B-Tree lookup achieves O(log n), including page splits and balance.
8. What is write amplification and how do multiple indexes affect write throughput?
9. When would a hash index beat a B-Tree, and why are they rarely used?

**Scenario-based**
10. A query `WHERE status = 'pending' ORDER BY created_at` is slow on a 50M-row table that is 99% non-pending. What index do you build and why?
11. You added an index on `email` but `WHERE LOWER(email) = 'x'` still does a seq scan. Why, and how do you fix it?

**"Why" questions**
12. Why is a B-Tree preferred over a hash index as the default?
13. Why does an index sometimes make a query slower (or get ignored)?

**Comparison questions**
14. B-Tree vs Hash index — trade-offs.
15. Composite index `(a, b)` vs two separate single-column indexes on `a` and `b`.

## Model Answers

**1. What is a database index and what problem does it solve?**
An index is a secondary, sorted data structure that maps key values to the physical locations of the rows containing them. It solves the problem of *finding rows without reading all of them*. Without an index, a filter like `WHERE email = ?` forces a sequential scan — the engine reads every row and compares, which is O(n) and scales linearly with table size. An index (usually a B-Tree) stores the keys in a balanced tree so the engine can descend to the matching value in O(log n) — a handful of page reads regardless of table size. The cost is that the index is redundant derived data: it consumes disk, competes for cache, and must be updated on every write. So an index is fundamentally a trade of write cost and space for read speed.

**2. Does adding an index speed up reads, writes, or both?**
Reads only — and it actively *slows* writes. Reads that filter, join, or sort on the indexed column(s) can use the index to avoid a full scan. But every `INSERT`, and every `UPDATE`/`DELETE` that touches an indexed column, must also maintain the index: locate and modify the corresponding tree entry, possibly splitting a page. This is write amplification. A table with eight indexes turns one insert into one heap write plus eight index writes. So the correct framing in an interview is: indexes accelerate the read paths you specifically design them for, and tax *all* writes plus disk/cache. You add them where read benefit outweighs write cost.

**3. What kind of index does a PRIMARY KEY create?**
A unique B-Tree index. Declaring `PRIMARY KEY` (or `UNIQUE`) creates a backing unique index automatically, because the only efficient way to guarantee "no duplicate value" on every insert is to look the value up in O(log n) — a sequential existence check would be O(n) per insert. A primary key additionally implies `NOT NULL`. That index then also serves lookups and joins on the key for free.

**4. Explain the leftmost-prefix rule.**
A composite index on `(a, b, c)` sorts entries by `a`, then by `b` within equal `a`, then by `c` within equal `(a,b)`. Because the ordering is hierarchical, the index can efficiently serve any query that constrains a *leading prefix* of the columns: `a`; `a` and `b`; or `a`, `b`, and `c`. It cannot efficiently serve `b` alone or `c` alone, because for a given `b` the matching entries are scattered across every value of `a` — there's no contiguous run to scan. Analogy: a phone book sorted by (last name, first name) lets you find everyone named "Smith", or "Smith, John", but is useless for finding everyone named "John". Practical consequence: order composite columns as equality-first, then range/sort; and lead with the column your queries always constrain (often `tenant_id`).

**5. What is a covering index / index-only scan?**
A covering index contains every column a query references, so the engine can satisfy the query from the index leaves alone without visiting the heap — an *index-only scan*. In PostgreSQL you build one either by putting all needed columns in the key, or by adding payload columns via `INCLUDE(...)` (which live only in the leaves and aren't part of the search key). It helps most for hot queries that select a few columns and are I/O-bound, because it eliminates the random heap fetch after the index lookup. Caveat: PostgreSQL still checks the visibility map to confirm each row is visible; on heavily-updated tables it may fall back to heap fetches until `VACUUM` updates that map.

**6. What is selectivity and why does the planner care?**
Selectivity is the fraction of rows a predicate keeps (sometimes framed inversely as how "distinct" a column is). A predicate matching 0.001% of rows is highly selective; one matching 50% is not. The planner uses estimated selectivity, from column statistics, to choose a plan. An index scan pays a per-matched-row cost of random I/O (descend tree + fetch heap page). If a predicate matches half the table, doing that random work for millions of rows is *slower* than one sequential sweep — so the planner correctly prefers a seq scan. Indexes pay off precisely when selectivity is high. This is why indexing a boolean column and querying it alone is usually pointless.

**7. Walk through a B-Tree lookup, splits, and balance.**
A B-Tree (B+Tree) has a root, internal pages, and leaf pages, all at equal depth. Internal pages hold separator keys and child pointers; leaves hold the sorted keys with heap pointers and are linked to their siblings. To find a key you start at the root and, at each level, binary-search the separators to pick the correct child, descending until you hit the leaf holding the key — that's log_fanout(N) page reads, e.g. ~4 for a billion rows at fan-out ~300. Inserts place the key in the correct leaf; if the leaf is full it splits into two and the median key is promoted to the parent. If the parent is also full the split cascades upward, and if it reaches the root the tree grows one level — always *at the top*, which is exactly why all leaves remain at equal depth (balance is maintained by growing upward, never by lengthening one branch). Deletes may merge or under-fill pages, reclaimed lazily by VACUUM. Balance guarantees a uniform O(log n); high fan-out keeps that log tiny; sorted, sibling-linked leaves make range scans and ORDER BY cheap.

**8. What is write amplification?**
Write amplification is the multiplication of physical writes caused by secondary structures. A single logical row insert must also insert an entry into every index on the table; each index insert may trigger a page split, and each is separately WAL-logged for durability. With N indexes, one insert is roughly 1 + N index maintenance operations plus their WAL. `UPDATE`s are worse: if an indexed column changes, the old index entry is removed and a new one added (PostgreSQL's HOT optimization avoids index churn only when no indexed column changed *and* the new tuple fits on the same heap page). The consequence: write throughput degrades roughly with the number of indexes, and index maintenance also causes bloat and more VACUUM work. This is the concrete cost that must be weighed against read speedups.

**9. When would a hash index beat a B-Tree?**
Only for pure equality lookups on large keys, where the O(1) average hash probe and smaller stored footprint (a fixed-size hash instead of a long key) can edge out the O(log n) B-Tree, and where you never need ranges or ordering. In practice they're rare because: B-Tree equality is already only a few page reads; B-Trees also serve ranges, sorts, and prefix matches, so one B-Tree covers more query shapes; and historically (pre-PG10) hash indexes weren't WAL-logged or crash-safe, which killed their reputation. So the honest answer is "almost never worth choosing over a B-Tree unless you've measured a specific equality-only hot path on a long key."

**10. Scenario: `WHERE status='pending' ORDER BY created_at`, 99% non-pending.**
Build a **partial** index: `CREATE INDEX idx_pending ON orders (created_at) WHERE status = 'pending';`. Two wins. First, it's tiny — it only holds the 1% of rows that are pending, so it stays small and cache-resident even as the table grows into hundreds of millions of historical rows. Second, because the key is `created_at`, the index is already sorted the way the query wants, so the `ORDER BY` is free (no sort node) and the planner can even stop early with a `LIMIT`. A full index on `(status, created_at)` would work too but would index all 50M rows including the 99% you never query in this path, wasting space and write cost. The partial index encodes the business fact that "pending is a small hot subset."

**11. Scenario: index on `email` unused for `WHERE LOWER(email)='x'`.**
The predicate is *non-SARGable*: you're not comparing `email`, you're comparing `LOWER(email)`, a computed value the plain index on `email` doesn't store. The index is sorted by the raw value, so it can't locate rows by the lowercased value. Two fixes: (a) build an **expression index** matching the predicate — `CREATE INDEX ON users (LOWER(email));` — so the index stores exactly the computed key the query filters on; or (b) normalize at write time (store `email` already-lowercased, or use `citext`) and query the column directly. Option (a) is the general pattern: index the expression you filter on.

**12. Why is a B-Tree the default over a hash index?**
Because it's the most *versatile* structure for the cost. A B-Tree serves equality, ranges (`<`, `>`, `BETWEEN`), prefix `LIKE`, `IS NULL`, `ORDER BY`, `MIN`/`MAX`, and uniqueness enforcement — one structure for nearly every access pattern. Its O(log n) with high fan-out is only a few page reads in practice, so the theoretical O(1) of a hash rarely matters. A hash index does *only* equality and nothing else. Given that the default should handle the widest range of queries well, the B-Tree is the obvious choice; hashes are a narrow specialization you opt into deliberately.

**13. Why does an index sometimes make a query slower or get ignored?**
Two separate phenomena. "Ignored": the planner estimates that using the index costs more than a seq scan — typically because the predicate has low selectivity (matches too many rows, so random index+heap I/O beats one sequential sweep), or the predicate is non-SARGable, or statistics are stale and mis-estimate row counts. "Slower": on a write-heavy table, an index you rarely read still taxes every write and consumes cache, degrading overall throughput; and for a low-selectivity read, forcing an index scan (e.g. via a hint or a bad plan) does more random I/O than a scan would. The fix is usually to match indexes to actual selective query patterns, keep statistics fresh with `ANALYZE`, and write SARGable predicates.

**14. B-Tree vs Hash index.**
B-Tree: sorted, balanced, O(log n); supports equality, ranges, ordering, prefix matches, uniqueness; the default. Hash: stores hashed keys, O(1) average, but equality-only — no ranges, no ordering, no sorting benefit. Hash can be smaller for very long keys and slightly faster on a pure equality hot path. But B-Tree's few-page-read equality is already fast and it covers vastly more query shapes, so it wins as the general choice. Choose hash only for a measured, equality-only workload on large keys. Also note hash indexes became crash-safe only in PostgreSQL 10+.

**15. Composite `(a, b)` vs two singles on `a` and `b`.**
A composite `(a, b)` is ideal when queries filter on `a` *and* `b` together (or on `a` alone via leftmost prefix), and it enables index-only scans and sorted `b`-within-`a` ordering. But it does nothing for `b` alone. Two single-column indexes each serve their own column and can be combined by the planner via a **bitmap AND** for `a AND b` queries — flexible, but the bitmap combine is less efficient than a purpose-built composite, and neither single index gives you the sorted `(a,b)` order. Decision: if you have a known hot query on both columns, build the composite (columns ordered equality-then-range/sort); if the columns are queried independently, prefer separate indexes. You can have both if the workloads justify the write cost.

## Common Mistakes

- **Indexing every column "just in case."** Each index taxes writes and cache; unused indexes are pure overhead. Index measured query patterns.
- **Wrong composite column order.** Leading with a range column, or putting the rarely-filtered column first, breaks the leftmost-prefix benefit.
- **Non-SARGable predicates.** `WHERE LOWER(col)=...`, `WHERE col + 0 = ...`, `WHERE col::text = ...`, or leading-wildcard `LIKE '%x'` prevent index use. Use expression indexes or rewrite.
- **Indexing low-selectivity columns alone.** Booleans/enums queried by themselves won't use the index; consider partial indexes instead.
- **Forgetting `CONCURRENTLY` in production.** A plain `CREATE INDEX` takes a lock that blocks writes for the whole build on a large table.
- **Ignoring bloat.** Heavy churn without `VACUUM`/`REINDEX` grows indexes and slowly erodes the speedup.
- **Assuming an index is always used.** The planner may correctly skip it; verify with `EXPLAIN`, don't assume.
- **Type mismatches.** Comparing a `bigint` column to a `text` literal (or across differing types) can defeat the index via implicit casts.

## Related Concepts

- **Query planner / optimizer & execution plans** (Topic 3) — decides *whether* to use an index.
- **Planner statistics & `ANALYZE`** — selectivity estimates depend on fresh stats.
- **VACUUM / autovacuum & the visibility map** — reclaim bloat; enable index-only scans.
- **SARGability** — writing predicates an index can use.
- **Other access methods** — GIN (arrays, JSONB, full-text), GiST/SP-GiST (geometry, ranges), BRIN (huge naturally-ordered tables).
- **Clustered vs non-clustered / heap storage** — PostgreSQL is heap-organized; `CLUSTER` physically reorders by an index once.
- **WAL (write-ahead log)** — durability mechanism that indexes also write to, part of write amplification.
- **Foreign keys** — usually want a supporting index on the referencing column.

# Stored Procedures & Functions

## What is it?

A **stored procedure** and a **user-defined function (UDF)** are named blocks of code that live *inside* the database and execute *on the server*, next to the data. Instead of pulling rows to the application, running logic, and pushing results back, you send one command and the database does the work where the data already sits.

In PostgreSQL both are typically written in **PL/pgSQL** (a procedural language layered over SQL with variables, `IF`/`LOOP`, exception handling), though functions can also be plain `SQL`, or `PL/Python`, `PL/v8`, `C`, etc.

- A **function** computes and **returns a value** — a scalar, a row, or a whole result set (`RETURNS TABLE`/`SETOF`). It is invoked *inside* a query: `SELECT my_func(x)`, `... WHERE id = other_func()`. Historically in PostgreSQL a function could not manage its own transactions.
- A **procedure** (added in PostgreSQL 11) is invoked with **`CALL`**, may **return nothing** (or use `INOUT` params), and crucially **can manage transactions** — it can `COMMIT` and `ROLLBACK` mid-body. It's meant for *doing things* (multi-step operations, batch jobs, ETL), not for producing a value inside a query.

```sql
-- A scalar FUNCTION: returns one value, usable inside a query
CREATE FUNCTION order_total(p_order_id BIGINT)
RETURNS NUMERIC
LANGUAGE sql
AS $$
    SELECT COALESCE(SUM(quantity * unit_price), 0)
    FROM order_items
    WHERE order_id = p_order_id;
$$;

SELECT id, order_total(id) FROM orders;   -- called inside a SELECT

-- A PROCEDURE: does work, controls its own transactions, called with CALL
CREATE PROCEDURE archive_old_orders(p_before DATE)
LANGUAGE plpgsql
AS $$
BEGIN
    INSERT INTO orders_archive SELECT * FROM orders WHERE created_at < p_before;
    DELETE FROM orders WHERE created_at < p_before;
    COMMIT;                              -- allowed in a procedure, not in a function
END;
$$;

CALL archive_old_orders('2025-01-01');
```

## Why is it needed?

The motivating first principle is **data locality**: moving computation to the data is cheaper than moving data to the computation. Several concrete needs follow.

- **Fewer round trips.** A workflow that is "read, decide, update, read again, update" is five network round trips from an app. As one stored routine it is a **single** call. On a chatty workload across a WAN, this is the difference between milliseconds and seconds.
- **Atomicity and correctness.** Multi-step logic (transfer money: debit A, credit B) must be all-or-nothing. Encapsulating it server-side, in one transaction, guarantees no partial state even if the app crashes mid-way.
- **Reuse and single source of truth.** Business rules (tax, discount, validation) coded once in the database are shared by every client — web app, mobile backend, cron job, BI tool — with no duplication or drift.
- **Security & least privilege.** You can `REVOKE` direct table access and grant only `EXECUTE` on a procedure. With `SECURITY DEFINER`, the routine runs with the *owner's* privileges, letting untrusted callers perform tightly-scoped privileged actions without touching tables directly. This also shrinks SQL-injection surface (logic is parameterized and fixed).
- **Performance.** Query plans can be cached/reused; set-based logic runs close to storage.
- **Composability in SQL.** A function can be used *inside* queries, views, indexes (expression indexes), and `CHECK` constraints — extending the language itself.

## How does it work?

**Defining and calling.** You `CREATE` the routine once; it's parsed and stored in the catalog. Bodies are quoted, conventionally with **dollar-quoting** (`$$ ... $$`) so you don't have to escape inner single quotes.

```sql
-- PL/pgSQL function with variables, control flow, and a declared return
CREATE OR REPLACE FUNCTION apply_discount(p_customer BIGINT, p_amount NUMERIC)
RETURNS NUMERIC
LANGUAGE plpgsql
AS $$
DECLARE
    v_tier   TEXT;
    v_rate   NUMERIC := 0;
BEGIN
    SELECT tier INTO v_tier FROM customers WHERE id = p_customer;

    IF v_tier = 'gold' THEN
        v_rate := 0.15;
    ELSIF v_tier = 'silver' THEN
        v_rate := 0.07;
    END IF;

    RETURN p_amount * (1 - v_rate);
END;
$$;

SELECT apply_discount(42, 100.00);   -- => 85.00 for a gold customer
```

**Parameter modes:** `IN` (default, input), `OUT` (returned back), `INOUT` (both). Functions usually use `IN` and a `RETURNS` clause; procedures often use `INOUT` to send values back (a procedure has no `RETURNS`).

**Calling a procedure vs a function:**

```sql
CALL archive_old_orders('2025-01-01');        -- procedure: statement, not in a query
SELECT order_total(7);                          -- function: expression, inside a query
```

**Set-returning functions** produce rows you can query like a table:

```sql
-- RETURNS TABLE: a named-column result set
CREATE FUNCTION top_customers(p_limit INT)
RETURNS TABLE (customer_id BIGINT, total NUMERIC)
LANGUAGE sql
AS $$
    SELECT customer_id, SUM(quantity * unit_price) AS total
    FROM order_items JOIN orders USING (id)     -- illustrative join
    GROUP BY customer_id
    ORDER BY total DESC
    LIMIT p_limit;
$$;

SELECT * FROM top_customers(10);   -- used in the FROM clause like a table

-- SETOF <type> is the older equivalent, returning a set of a known row type
CREATE FUNCTION active_users() RETURNS SETOF users
LANGUAGE sql AS $$
    SELECT * FROM users WHERE last_login > now() - interval '30 days';
$$;
```

**Transaction control (procedures only):**

```sql
CREATE PROCEDURE reindex_in_batches()
LANGUAGE plpgsql
AS $$
DECLARE r RECORD;
BEGIN
    FOR r IN SELECT tablename FROM pg_tables WHERE schemaname = 'public' LOOP
        EXECUTE format('REINDEX TABLE %I', r.tablename);
        COMMIT;             -- commit after each table; a function could not do this
    END LOOP;
END;
$$;
```

**Error handling** uses exception blocks:

```sql
BEGIN
    -- risky work
EXCEPTION
    WHEN unique_violation THEN
        RAISE NOTICE 'duplicate, skipping';
    WHEN OTHERS THEN
        RAISE;              -- re-raise anything unexpected
END;
```

## Internal Working

When you `CREATE FUNCTION`/`PROCEDURE`, PostgreSQL stores the routine's definition (source text, argument types, return type, volatility, language) in the system catalog `pg_proc`. It is **not** compiled to machine code; the body is kept as text tied to a language handler (e.g. the PL/pgSQL handler).

On **first execution in a session**, the PL/pgSQL handler *parses* the body into an internal parse tree and caches it for the life of the session (or backend). Each SQL statement *inside* the body is prepared as needed and its plan is cached too — so repeated calls in the same session skip re-planning. This is why PL/pgSQL routines get faster after their first call in a connection, and why a routine can occasionally suffer from a **cached "generic" plan** that isn't ideal for a specific parameter (PostgreSQL mitigates this by choosing between custom and generic plans over the first several executions).

**Volatility classification** (`IMMUTABLE`, `STABLE`, `VOLATILE`) is metadata that tells the planner how the function behaves, enabling big optimizations:

- `IMMUTABLE` — same inputs always give same output, no DB access (e.g. pure math). The planner may **pre-evaluate** it once (constant folding) and it can be used in **expression indexes**.
- `STABLE` — consistent within a single statement (e.g. reads tables but doesn't change within the query, like `now()`-style consistency). Usable for index scans within a query.
- `VOLATILE` (default) — may return different results anytime or have side effects (`random()`, `INSERT`). Re-evaluated every call; never folded.

```sql
CREATE FUNCTION celsius_to_f(c NUMERIC) RETURNS NUMERIC
LANGUAGE sql IMMUTABLE AS $$ SELECT c * 9/5 + 32 $$;   -- eligible for an expression index
```

A function call executes inside the **calling transaction** — it cannot `COMMIT`. A procedure invoked by `CALL` runs in its own top-level context where transaction control statements are permitted (unless it's itself called inside an explicit transaction block). `SECURITY DEFINER` swaps the effective role for the routine's owner; PostgreSQL recommends pinning `search_path` on such routines to prevent hijacking via schema tricks.

## Advantages

- **Reduced network traffic** — one call replaces many round trips; logic runs where the data lives.
- **Encapsulated, reusable business logic** — one definition shared by all clients; single source of truth.
- **Atomicity** — multi-step operations wrapped in one transaction (procedures can even control commit boundaries).
- **Security** — grant `EXECUTE` instead of table access; `SECURITY DEFINER` for scoped privilege elevation; smaller injection surface.
- **Performance** — cached parse trees/plans, set-based server-side processing, less data marshalling.
- **Composability** — functions plug into queries, views, indexes, constraints, triggers.

## Limitations

- **Harder to version, test, and debug** — logic in the DB is outside typical app CI; debuggers are weaker than app-side ones.
- **Portability** — PL/pgSQL is PostgreSQL-specific; migrating to another engine means rewriting.
- **Hidden business logic** — rules buried in the DB can surprise app developers ("where does this value come from?").
- **Scaling coupling** — heavy procedural work runs on the (often single primary) database server, which is the hardest tier to scale horizontally.
- **Plan caching pitfalls** — a cached generic plan can be suboptimal for skewed parameters.
- **Volatility mislabeling** — marking a volatile function `IMMUTABLE` causes silently wrong results (stale constant folding / bad index entries).
- **Deployment friction** — schema-migration tooling for routines is clunkier than deploying app code.

## Real-world Applications

- **Financial transfers / ledgers** — debit+credit+audit in one atomic procedure so partial transfers are impossible.
- **Batch/ETL jobs** — procedures that process millions of rows in committed batches (`COMMIT` per chunk) to bound lock/WAL growth.
- **Complex validation & derived values** — functions computing tax, pricing tiers, scores, reused across the app and in `CHECK` constraints.
- **Reporting/analytics** — `RETURNS TABLE` functions exposing parameterized, reusable result sets to BI tools and dashboards.
- **Security boundary** — `SECURITY DEFINER` procedures let low-privilege services perform narrowly-scoped privileged operations.
- **Triggers** — trigger functions enforce invariants, maintain audit tables, and denormalize on write.
- **Expression indexes** — `IMMUTABLE` functions (e.g. `LOWER(email)`, normalization) indexed for fast lookups.

## Interview Questions

**Beginner**
1. What is a stored procedure, and how does a function differ from it in PostgreSQL?
2. How do you call a procedure versus a function?
3. What does `RETURNS TABLE` mean?

**Intermediate**
4. What are `IN`, `OUT`, and `INOUT` parameters?
5. Why can a procedure run `COMMIT`/`ROLLBACK` but a function generally cannot?
6. What is `SETOF` and how does it differ from `RETURNS TABLE`?

**Advanced**
7. Explain function volatility (`IMMUTABLE`/`STABLE`/`VOLATILE`) and how the planner uses it.
8. What is `SECURITY DEFINER` and what's the risk if you misuse it?
9. How does PostgreSQL cache and plan the SQL inside a PL/pgSQL routine, and how can that hurt you?

**Scenario-based**
10. You must move money between two accounts and record an audit row, all-or-nothing. How do you implement it and why server-side?
11. A nightly job deletes 50M rows and blows up WAL/locks in one giant transaction. How do you restructure it?

**"Why" questions**
12. Why put business logic in the database at all instead of the application?
13. Why might marking a function `IMMUTABLE` be dangerous?

**Comparison questions**
14. Stored procedure vs function — when do you choose each?
15. Server-side stored routine vs application-side logic — trade-offs.

## Model Answers

**1. Stored procedure vs function in PostgreSQL.**
Both are named server-side code blocks. A **function** returns a value (scalar, row, or set) and is invoked *inside* a query — `SELECT f(x)`, `WHERE id = g()` — and traditionally runs within the caller's transaction without controlling it. A **procedure** (PostgreSQL 11+) is invoked with `CALL` as a standalone statement, need not return a value, and can **control transactions** (`COMMIT`/`ROLLBACK` inside its body). The rule of thumb: use a *function* to **compute a value** you consume in SQL; use a *procedure* to **perform an action/workflow**, especially multi-step operations that need their own commit boundaries.

**2. Calling a procedure vs a function.**
A procedure is called with the `CALL` statement: `CALL archive_old_orders('2025-01-01');`. It cannot appear inside a `SELECT`. A function is called as an expression *within* a query: `SELECT order_total(7);` or `SELECT * FROM top_customers(10);` for a set-returning function, or in a `WHERE`/`ON`/`CHECK` clause. The distinction reflects intent: functions are values in the query language; procedures are commands.

**3. What does `RETURNS TABLE` mean?**
It declares that the function returns a *result set* with named, typed columns, so it can be queried like a table in a `FROM` clause: `SELECT * FROM top_customers(10)`. Inside, you produce rows either by a final `SELECT` (in a `LANGUAGE sql` function) or via `RETURN QUERY`/`RETURN NEXT` in PL/pgSQL. It turns a function into a reusable, parameterized, virtual table — great for exposing curated result sets to reports and BI tools.

**4. `IN`, `OUT`, `INOUT` parameters.**
`IN` (the default) is an input the caller supplies. `OUT` is an output the routine sets and returns to the caller — a function with `OUT` params effectively returns them as its result, and a procedure uses `OUT`/`INOUT` since it has no `RETURNS` clause. `INOUT` is both: the caller passes a value in and receives a (possibly modified) value back. `OUT`/`INOUT` let a single routine return multiple values without defining a composite return type.

**5. Why procedures can COMMIT but functions can't.**
A function is evaluated as part of a surrounding SQL statement, which is itself inside a transaction — you cannot commit *while producing a value for* a statement that is mid-execution, because that would break the atomicity and snapshot of the enclosing statement/transaction. A procedure invoked by `CALL` is a top-level statement, not embedded in another query's evaluation, so PostgreSQL can let it end one transaction and begin another between its steps. That's exactly why procedures were added: to enable batching and workflows that commit incrementally. (If a procedure is itself called inside an explicit `BEGIN...COMMIT` block, transaction control inside it is disallowed too.)

**6. `SETOF` vs `RETURNS TABLE`.**
Both return multiple rows. `RETURNS SETOF sometype` returns a set of an *existing* row type (a table's rowtype or a composite type or a scalar), so the column names/types come from that type. `RETURNS TABLE(col1 t1, col2 t2, ...)` declares the output columns *inline* with their own names and types. `RETURNS TABLE` is essentially syntactic sugar equivalent to `SETOF` with `OUT` parameters, and is clearer when the result shape is bespoke rather than matching an existing table. Functionally they're queried the same way, in a `FROM` clause.

**7. Function volatility.**
Volatility tells the planner how stable a function's output is, which unlocks optimizations. `IMMUTABLE`: same inputs → same output forever, no DB reads (pure math, string normalization). The planner can evaluate it once and fold it to a constant, and it may be used in **expression indexes**. `STABLE`: results don't change *within a single statement* (may read tables, respects the statement snapshot); safe to use for index scans within a query but not folded across the whole query. `VOLATILE` (default): may change at any call or have side effects (`random()`, `nextval()`, any `INSERT`/`UPDATE`); re-evaluated every time and never folded. Labeling correctly both improves performance (folding, index eligibility) and preserves correctness — the label is a *promise* the planner trusts.

**8. `SECURITY DEFINER` and its risk.**
By default a routine runs with the *caller's* privileges (`SECURITY INVOKER`). `SECURITY DEFINER` makes it run with the *owner's* privileges, so a low-privilege user can invoke it to perform a tightly-scoped action they otherwise couldn't — the canonical way to expose privileged operations safely. The risk: because it elevates privilege, a poorly-written definer routine is an attack vector. If `search_path` isn't pinned, an attacker can create a malicious object (function/table) in a schema that shadows the one your routine references, and your elevated routine will execute *their* code. Mitigation: always `SET search_path` explicitly on `SECURITY DEFINER` routines, grant `EXECUTE` narrowly, and keep the body minimal and parameterized.

**9. Plan caching inside PL/pgSQL.**
The first time a PL/pgSQL routine runs in a session, its body is parsed into an internal tree and cached; each embedded SQL statement is prepared and its plan cached for reuse in that session. This makes repeated calls fast (no re-parse/re-plan). The downside is the **generic plan** problem: a cached plan built without knowledge of specific parameter values may be poor for skewed data (e.g. a parameter that's sometimes very selective, sometimes not). PostgreSQL adaptively compares custom vs generic plans over the first several executions and picks a strategy, but you can still get stuck with a bad generic plan. Mitigations include using `EXECUTE` with dynamic SQL to force re-planning for volatile-selectivity queries, or restructuring the query.

**10. Scenario: atomic money transfer + audit.**
Implement it as a single server-side routine wrapped in one transaction so it's all-or-nothing. A function or procedure both work; a procedure is natural if you want explicit control. Inside: check/lock the source balance (`SELECT ... FOR UPDATE`), `UPDATE accounts SET balance = balance - amt WHERE id = src`, `UPDATE ... + amt WHERE id = dst`, `INSERT` the audit row — all in one transaction, with a `CHECK`/exception guarding against overdraft. Server-side is essential because: (a) atomicity is guaranteed even if the app crashes between steps; (b) locking is handled close to the data with minimal window; (c) it's one round trip instead of several; and (d) you can grant only `EXECUTE`, so no client can perform a raw partial update. Doing this from the app across multiple round trips risks partial state on failure and race conditions.

**11. Scenario: giant nightly delete blows up WAL/locks.**
The problem is doing 50M deletes in one transaction: WAL grows unbounded, locks and dead tuples pile up, and rollback risk is huge. Restructure into a **procedure that deletes in batches and commits each batch**: loop `DELETE FROM t WHERE ... AND id IN (SELECT id FROM t WHERE ... LIMIT 10000)`, then `COMMIT`, until no rows remain. Because a procedure can `COMMIT` mid-body, each batch releases locks and lets WAL be recycled and autovacuum reclaim space, keeping the operation bounded and interruptible. This is a textbook use of procedure transaction control that a function could not express.

**12. Why put logic in the database?**
Chiefly data locality, atomicity, reuse, and security. Running logic next to the data avoids shipping rows over the network and back — fewer round trips, less latency, especially for chatty multi-step workflows. Multi-statement operations get true atomicity in one transaction. Encoding a rule once in the DB gives every client a single source of truth with no duplication. And you can lock down tables and expose only `EXECUTE`, shrinking the attack/injection surface. The counterweight is testability, portability, and scaling — so the pragmatic stance is to put *data-intensive, integrity-critical, widely-shared* logic server-side, and keep app-specific orchestration in the application.

**13. Why is `IMMUTABLE` dangerous if misused.**
`IMMUTABLE` is a promise that the function's output depends *only* on its arguments and never changes. The planner acts on that promise: it may evaluate the function once and cache the result as a constant, and it may store its outputs in an expression index. If you mark a function `IMMUTABLE` but it actually reads a table or depends on `now()`/session settings, then folded constants and index entries become **stale and wrong** — queries silently return incorrect results, and the index no longer matches reality. There's no error; it's a correctness landmine. Only mark truly pure functions `IMMUTABLE`; use `STABLE` for ones that read data consistently within a statement.

**14. Procedure vs function — when to choose.**
Choose a **function** when you need a value inside SQL: a computed column, a filter expression, a reusable result set (`RETURNS TABLE`), an expression index, or a trigger function. Choose a **procedure** when you're performing an action/workflow — especially multi-step operations needing their own transaction control (batch jobs, ETL, incremental commits, admin tasks) — and you don't need to embed the call in a query. Shorthand: functions *compute and return*; procedures *do and control transactions*.

**15. Server-side routine vs application-side logic.**
Server-side wins on data locality (fewer round trips), atomicity, shared single-source-of-truth logic, and security (grant EXECUTE, `SECURITY DEFINER`). Application-side wins on testability (normal CI, unit tests, debuggers), version control and deployment (ships with app code), portability (not tied to PostgreSQL dialect), and horizontal scalability (app tier scales out; the DB primary doesn't). The engineering judgment: keep integrity-critical, data-heavy, cross-client operations in the DB; keep orchestration, presentation, and app-specific business flow in the application. Avoid burying surprising business rules in the DB where app developers won't find them.

## Common Mistakes

- **Using a function where you need transaction control.** Functions can't `COMMIT`; batching/ETL that must commit incrementally needs a procedure.
- **Mislabeling volatility.** Marking a data-reading or time-dependent function `IMMUTABLE` produces silently wrong results via constant folding and stale index entries.
- **`SECURITY DEFINER` without a pinned `search_path`.** Opens a privilege-escalation hole via schema shadowing.
- **Forgetting dollar-quoting.** Bodies with single quotes become an escaping nightmare; use `$$ ... $$`.
- **Row-by-row loops instead of set-based SQL.** Looping and issuing per-row statements is orders of magnitude slower than one set-based statement; prefer set logic unless truly necessary.
- **Swallowing exceptions with `WHEN OTHERS THEN NULL`.** Hides real failures; catch specific conditions and re-`RAISE` the unexpected.
- **Putting too much business logic in the DB.** Hurts testability, portability, and scalability; and hides logic from app developers.
- **Ignoring the generic-plan trap.** A cached plan can be poor for skewed parameters; use dynamic `EXECUTE` where selectivity varies wildly.

## Related Concepts

- **Triggers & trigger functions** — functions fired automatically on `INSERT`/`UPDATE`/`DELETE` to enforce invariants and audit.
- **Transactions, isolation, and locking** — the correctness foundation procedures build on (`FOR UPDATE`, `COMMIT`/`ROLLBACK`).
- **PL/pgSQL and other procedural languages** — `plpgsql`, `sql`, `plpython3u`, `plv8`, `C`.
- **Prepared statements & plan caching** — the mechanism behind in-routine SQL performance and the generic-plan issue.
- **Expression indexes** — enabled by `IMMUTABLE` functions.
- **Views & materialized views** — the declarative counterpart for encapsulating reusable queries.
- **Privileges/roles & `GRANT EXECUTE`** — the security model behind `SECURITY DEFINER`.
- **Cursors** — server-side row-by-row processing when set-based isn't possible.

---

# Performance Optimization

## What is it?

Performance optimization is the discipline of making queries return correct results using the **least work** — fewest disk reads, least CPU, least memory, least locking. The core insight I hammer with juniors: the database is a *cost-based machine*. For any query there are many possible execution strategies; the **query planner** estimates the cost of each and picks the cheapest. Optimization is (a) giving the planner the access paths (indexes) and accurate statistics it needs, and (b) writing SQL the planner can turn into a cheap plan.

Two pillars:

1. **Writing optimizable SQL** — SARGable predicates, avoiding `SELECT *`, not wrapping indexed columns in functions, sensible join order and data types.
2. **Reading what the database actually did** — execution plans via `EXPLAIN`/`EXPLAIN ANALYZE`, so you optimize with evidence instead of superstition.

```sql
-- The single most important habit: measure, don't guess
EXPLAIN ANALYZE
SELECT * FROM orders WHERE customer_id = 42 AND created_at >= '2026-01-01';
```

## Why is it needed?

Query cost does not scale gently — it scales with **how much data you touch**, and bad choices multiply. A missing index turns a millisecond lookup into a full-table scan. A function on an indexed column silently disables that index. A wrong join order can make an intermediate result 1,000× larger than necessary. An N+1 pattern turns one screen render into 500 queries. At small scale you never notice; at production scale the same query melts the database.

You need optimization because:

- **Latency is user-facing.** Slow queries are slow pages, timeouts, and abandoned carts.
- **Throughput is finite.** The database (especially the primary) is the hardest tier to scale horizontally; wasted work steals capacity from every other query.
- **Cost is real money.** More I/O and CPU means bigger instances and higher cloud bills.
- **Problems hide until scale.** Code that's fine on 10k rows can be catastrophic on 10M. Reading plans is how you catch it *before* production.

The professional mindset: **you cannot optimize what you cannot measure.** `EXPLAIN ANALYZE` is the microscope.

## How does it work?

### Writing optimizable SQL

**SARGable predicates.** SARGable = "Search ARGument-able" = a predicate the engine can satisfy by *seeking* through an index rather than evaluating a function on every row. The rule: **keep the indexed column bare on one side, operators the index understands, constant on the other side.**

```sql
-- NON-SARGable: function wraps the indexed column -> index unusable, seq scan
SELECT * FROM users WHERE LOWER(email) = 'ada@example.com';
SELECT * FROM orders WHERE date_trunc('day', created_at) = '2026-07-09';
SELECT * FROM t WHERE amount * 100 > 5000;
SELECT * FROM users WHERE email LIKE '%@example.com';   -- leading wildcard

-- SARGable rewrites: column stays bare, index can seek
SELECT * FROM users WHERE email = 'ada@example.com';              -- or index LOWER(email)
SELECT * FROM orders
WHERE created_at >= '2026-07-09' AND created_at < '2026-07-10';   -- range, not date_trunc
SELECT * FROM t WHERE amount > 50;
SELECT * FROM users WHERE email LIKE 'ada%';                       -- trailing wildcard OK
```

**Avoid `SELECT *`.** Fetch only the columns you use. Reasons: less network and memory; it enables **index-only scans** (impossible if you drag in columns the index doesn't cover); it avoids pulling large `TEXT`/`BYTEA`/TOAST columns you don't need; and it makes queries resilient to schema changes and clearer about intent.

```sql
-- Bad: forces heap access, ships every column
SELECT * FROM users WHERE email = 'ada@example.com';
-- Good: can be an index-only scan with INCLUDE(full_name)
SELECT email, full_name FROM users WHERE email = 'ada@example.com';
```

**Don't wrap indexed columns in functions** (the SARGability corollary). If you must transform, build a matching **expression index** so the stored key equals the filtered value.

```sql
CREATE INDEX idx_users_lower_email ON users (LOWER(email));   -- now LOWER(email)=... seeks
```

**Join order & join columns.** Ensure join keys are indexed (usually on the "many" side / the side being probed). The planner *reorders* joins itself, but you help it by indexing keys, filtering early (push selective `WHERE`s so intermediate results stay small), and keeping join keys the **same data type** on both sides.

**Data types matter.** Type mismatches force implicit casts that break index use (comparing `bigint` to `text`, or `varchar` to `int`). Right-sized types (`int` vs `bigint`, `timestamptz` vs storing timestamps as text) mean narrower rows, more rows per page, fewer I/Os, and comparisons the index can use. Store dates as date/timestamp types, not strings.

### Execution plans

`EXPLAIN` shows the plan the planner *chose*, with **estimates**. `EXPLAIN ANALYZE` **actually runs** the query and adds real timings and row counts, so you can compare estimate vs reality.

```sql
EXPLAIN              SELECT ...;   -- estimated plan only, does NOT run
EXPLAIN ANALYZE      SELECT ...;   -- runs it; shows actual time & rows (careful: executes writes too!)
EXPLAIN (ANALYZE, BUFFERS, VERBOSE) SELECT ...;   -- also shows cache/disk block usage
```

**Scan node types:**
- **Seq Scan** — read every row. Fine for small tables or low-selectivity predicates; a red flag on a large table with a selective filter.
- **Index Scan** — descend a B-Tree to matching keys, fetch each row from the heap. Best for high selectivity (few rows).
- **Index Only Scan** — answered from the index alone (covering index); no heap fetch.
- **Bitmap Index Scan + Bitmap Heap Scan** — build a bitmap of matching row locations from the index, then read the heap in **physical order**. Sits between the two: good for *medium* selectivity where an index scan's random I/O would hurt but a full seq scan is wasteful.

**Join algorithms:**
- **Nested Loop** — for each outer row, probe the inner (ideally via an index). Great when the outer side is small; O(n·m) if the inner isn't indexed.
- **Hash Join** — build a hash table on the smaller input, probe with the larger. Great for large, unsorted, equality joins; needs memory (`work_mem`).
- **Merge Join** — sort both inputs (or use already-sorted indexes) and merge. Good for large inputs already ordered on the join key, or range-ish joins.

**Reading the numbers:**
- `cost=0.29..8.31` — estimated *startup*..*total* cost in arbitrary planner units (roughly page/CPU work), not milliseconds.
- `rows=…` — estimated rows the node emits.
- `actual time=0.015..0.042 rows=1 loops=1` (ANALYZE only) — real startup..total ms **per loop**, real rows, and loop count. Multiply per-loop time by `loops` for total.
- **Estimate vs actual mismatch is the #1 diagnostic.** If `rows` estimate is 5 but `actual rows` is 500,000, the planner mis-estimated selectivity — usually **stale statistics** — and likely chose a bad plan.

### Planner statistics (`ANALYZE`)

The planner's cost estimates depend on **statistics**: per-column row counts, most-common-values, histograms, and null fractions stored in `pg_statistic`. These are gathered by the `ANALYZE` command (and automatically by **autovacuum**). Stale stats → wrong selectivity estimates → wrong plans (e.g. seq scan when an index scan would win, or a nested loop over half a million rows).

```sql
ANALYZE orders;                 -- refresh stats for one table
ANALYZE;                        -- whole database
-- Improve estimates on a skewed column by collecting more detail:
ALTER TABLE orders ALTER COLUMN status SET STATISTICS 1000;
ANALYZE orders;
```

### The N+1 problem

An application anti-pattern: run **1** query to fetch N parent rows, then **N** more queries — one per parent — to fetch each parent's children. 1 + N round trips instead of 1 or 2. It's the classic ORM trap (lazy loading in a loop). Each query may be individually fast, but the round-trip overhead and planning cost dominate, and it scales linearly with result size.

```sql
-- N+1 (app pseudocode):
--   orders = SELECT * FROM orders WHERE customer_id = 42;      -- 1 query
--   for each order: SELECT * FROM order_items WHERE order_id = order.id;  -- N queries

-- Fix 1: a single JOIN
SELECT o.id, i.product_id, i.quantity
FROM orders o
JOIN order_items i ON i.order_id = o.id
WHERE o.customer_id = 42;

-- Fix 2: one batched IN / ANY query (ORM "eager loading")
SELECT * FROM order_items WHERE order_id = ANY(ARRAY[1,2,3, ...]);
```

## Internal Working

The planner/optimizer is **cost-based**. Given a parsed query, it:

1. **Enumerates access paths** per table — seq scan, each usable index scan, bitmap scan — and estimates each one's cost from statistics and cost parameters (`seq_page_cost`, `random_page_cost`, `cpu_tuple_cost`, etc.).
2. **Enumerates join orders and join methods** (nested loop / hash / merge), building up the cheapest way to combine tables. Because join-order search is exponential, it uses dynamic programming for few tables and a genetic algorithm (`geqo`) beyond a threshold.
3. **Estimates cardinality** at each step — how many rows each node emits — using the column statistics (MCVs, histograms, `n_distinct`, null fraction). This is the make-or-break step: **cardinality errors compound** up the plan tree. A 10× underestimate at a leaf can cascade into a catastrophically wrong join method at the top (e.g. a nested loop chosen because it thought the outer side had 5 rows, but it had 500,000).
4. **Chooses the lowest-estimated-cost plan** and hands it to the executor.

`random_page_cost` (default 4.0 vs `seq_page_cost` 1.0) encodes the assumption that random I/O is ~4× a sequential read — this is why the planner favors seq scans as selectivity drops (many random index+heap fetches lose to one sequential sweep). On SSDs people often lower it (e.g. 1.1) to reflect cheap random I/O, which shifts the planner toward index scans.

`EXPLAIN ANALYZE` runs the plan and instruments each node with real time and row counts, letting you spot exactly where estimate diverged from reality. That divergence points at the fix: stale stats (run `ANALYZE`), a missing index (add one), a non-SARGable predicate (rewrite), or a correlation the single-column stats can't see (create **extended statistics**, `CREATE STATISTICS`).

### Sample `EXPLAIN ANALYZE` and walkthrough

```sql
EXPLAIN ANALYZE
SELECT o.id, o.total, c.name
FROM orders o
JOIN customers c ON c.id = o.customer_id
WHERE o.created_at >= '2026-07-01'
ORDER BY o.total DESC
LIMIT 10;
```

```
Limit  (cost=1543.22..1543.25 rows=10 width=48)
       (actual time=12.481..12.484 rows=10 loops=1)
  ->  Sort  (cost=1543.22..1550.68 rows=2982 width=48)
            (actual time=12.480..12.481 rows=10 loops=1)
        Sort Key: o.total DESC
        Sort Method: top-N heapsort  Memory: 27kB
        ->  Hash Join  (cost=68.50..1478.79 rows=2982 width=48)
                       (actual time=1.204..10.932 rows=3011 loops=1)
              Hash Cond: (o.customer_id = c.id)
              ->  Bitmap Heap Scan on orders o
                        (cost=34.10..1401.55 rows=2982 width=20)
                        (actual time=0.412..7.617 rows=3011 loops=1)
                    Recheck Cond: (created_at >= '2026-07-01')
                    Heap Blocks: exact=812
                    ->  Bitmap Index Scan on idx_orders_created
                              (cost=0.00..33.36 rows=2982 width=0)
                              (actual time=0.293..0.293 rows=3011 loops=1)
                          Index Cond: (created_at >= '2026-07-01')
              ->  Hash  (cost=22.00..22.00 rows=1000 width=36)
                        (actual time=0.760..0.761 rows=1000 loops=1)
                    Buckets: 1024  Batches: 1  Memory Usage: 72kB
                    ->  Seq Scan on customers c
                              (cost=0.00..22.00 rows=1000 width=36)
                              (actual time=0.008..0.377 rows=1000 loops=1)
Planning Time: 0.284 ms
Execution Time: 12.560 ms
```

**How to read it — bottom-up is how it executes, top-down is how it's presented:**

- **Bitmap Index Scan on `idx_orders_created`** — uses the index on `created_at` to find matching rows, producing a bitmap of their locations. Estimated 2982, actual 3011 — estimate is excellent, so stats are fresh. Took 0.293 ms.
- **Bitmap Heap Scan on `orders`** — reads those `Heap Blocks: exact=812` in physical order (avoiding random-per-row I/O). The `Recheck Cond` re-verifies the predicate. This bitmap approach was chosen over a plain index scan because ~3000 rows is *medium* selectivity — reading 812 blocks sequentially beats 3000 random fetches.
- **Seq Scan on `customers`** (1000 rows) — the whole small table is scanned to **build the hash**; for a 1000-row table a seq scan is cheaper than any index, correctly.
- **Hash / Hash Join** — `customers` is hashed (72kB, `Batches: 1` means it fit in `work_mem` — no spill to disk), then each `orders` row probes the hash on `customer_id = c.id`. 3011 rows out in ~11 ms.
- **Sort** — `top-N heapsort` on `o.total DESC`, and because there's a `LIMIT 10` it keeps only the top 10 in 27kB rather than sorting all 3011. Nice optimization the planner applied.
- **Limit** — stops after 10 rows.
- **Planning 0.284 ms, Execution 12.56 ms.** Healthy: estimates match actuals, memory operations stayed in RAM (`Batches: 1`), no seq scan on a big table.

**What would signal a problem:** if `Bitmap Index Scan` estimated `rows=2982` but `actual rows=900000`, the stats are stale → `ANALYZE orders`. If `customers` were 5M rows and still `Seq Scan`, we'd want an index on `customers.id` (there should be a PK) and possibly a nested-loop plan. If `Hash` showed `Batches: 8`, the hash spilled to disk — raise `work_mem`. If Sort showed `Sort Method: external merge  Disk: …`, the sort spilled — add an index on `total` or raise `work_mem`.

## Advantages

- **Lower latency** — right plans turn seconds into milliseconds; users feel it directly.
- **Higher throughput & lower cost** — less wasted I/O/CPU frees the database for more concurrent work and smaller instances.
- **Evidence-based tuning** — `EXPLAIN ANALYZE` replaces guesswork with measured facts.
- **Early problem detection** — plans expose scaling landmines (seq scans, hash spills, cardinality blowups) before production load hits them.
- **Predictable scaling** — SARGable, well-indexed, N+1-free queries scale with result size, not table size.

## Limitations

- **Estimates can be wrong** — the planner is only as good as its statistics; skew and correlation defeat simple stats.
- **`EXPLAIN ANALYZE` runs the query** — it executes side effects; wrap writes in a transaction you roll back.
- **Cost units aren't milliseconds** — cost numbers are relative; only `ANALYZE`'s actual times are real.
- **Optimization is workload-specific** — an index that speeds reads taxes writes; tuning is a trade-off, not a free win.
- **Plan instability** — the same query can flip plans as data grows or stats shift, causing surprise regressions.
- **Diminishing returns / premature optimization** — micro-tuning a query that isn't the bottleneck wastes effort; profile first.

## Real-world Applications

- **Slow-query triage** — pull the slow query, run `EXPLAIN (ANALYZE, BUFFERS)`, find the seq scan / bad estimate / spill, fix index or stats.
- **API latency reduction** — eliminate ORM N+1 with eager loading/joins; often the single biggest win in web backends.
- **Reporting/analytics tuning** — ensure hash/merge joins and index scans over big tables; bump `work_mem` for sorts/hashes; use covering indexes.
- **Statistics maintenance** — schedule/verify autovacuum and `ANALYZE`, raise per-column stats targets on skewed columns, add extended stats for correlated columns.
- **Search features** — rewrite non-SARGable text filters into expression indexes or trigram/GIN indexes.
- **Cost-parameter tuning for SSDs** — lower `random_page_cost` so the planner favors index scans on flash storage.
- **Capacity planning** — reading plans reveals which queries will break at 10× data before they do.

## Interview Questions

**Beginner**
1. What is the difference between `EXPLAIN` and `EXPLAIN ANALYZE`?
2. What does a "Seq Scan" in a plan mean, and is it always bad?
3. Why is `SELECT *` discouraged in performance-sensitive queries?

**Intermediate**
4. What does SARGable mean? Give a non-SARGable predicate and rewrite it.
5. Explain the three join algorithms and when each is chosen.
6. What is a Bitmap Index Scan and when is it preferred over a plain Index Scan?

**Advanced**
7. In a plan, `rows` estimate is 10 but `actual rows` is 900,000. What does that tell you and how do you fix it?
8. How does the cost-based optimizer choose a plan, and why do cardinality errors matter so much?
9. What is `random_page_cost` and why might you lower it on SSDs?

**Scenario-based**
10. An endpoint got 20× slower after data grew; the plan shows a nested loop with a large inner seq scan. Diagnose and fix.
11. A dashboard page fires 500 near-identical small queries. What's happening and how do you fix it?

**"Why" questions**
12. Why does wrapping an indexed column in a function disable the index?
13. Why can stale statistics cause a fast query to suddenly become slow?

**Comparison questions**
14. Index Scan vs Bitmap Heap Scan vs Seq Scan — trade-offs.
15. Nested Loop vs Hash Join vs Merge Join.

## Model Answers

**1. `EXPLAIN` vs `EXPLAIN ANALYZE`.**
`EXPLAIN` shows the plan the optimizer *chose* along with its *estimates* — estimated cost (in arbitrary units) and estimated row counts — **without running** the query. `EXPLAIN ANALYZE` actually **executes** the query and adds *real* measurements: actual time per node (startup..total, in ms, per loop), actual rows, and loop counts. You use plain `EXPLAIN` for a quick, side-effect-free look, and `EXPLAIN ANALYZE` when you need to compare estimate-vs-reality and get true timings. Caution: `EXPLAIN ANALYZE` runs the statement, including writes, so wrap `INSERT`/`UPDATE`/`DELETE` in a transaction you roll back. Adding `BUFFERS` shows how many blocks came from cache vs disk, which is invaluable for I/O diagnosis.

**2. What is a Seq Scan and is it always bad?**
A Seq Scan reads every row of the table sequentially. It is *not* always bad — it's actually optimal when you need most of the rows (low selectivity), or when the table is small enough that reading it all is cheaper than index bookkeeping, because sequential I/O is far cheaper per page than random I/O. It's a red flag only when you have a *selective* predicate on a *large* table and expected an index scan — that usually means a missing index, a non-SARGable predicate, stale statistics, or a type mismatch. So the correct interview answer is "it depends on selectivity and table size; judge it against expectation, not in isolation."

**3. Why avoid `SELECT *`?**
Several reasons. It ships and buffers columns you don't need, wasting network and memory. It defeats **index-only scans**: if the index doesn't cover every selected column, the engine must do heap fetches, so pulling all columns forces heap access. It can drag in large TOASTed `TEXT`/`BYTEA` values you didn't want, causing extra I/O. It makes the query fragile to schema changes and obscures intent. Selecting only the needed columns keeps rows narrow, enables covering-index plans, and documents exactly what the query depends on.

**4. SARGable, with a rewrite.**
SARGable ("Search ARGument-able") describes a predicate the engine can satisfy by *seeking* through an index rather than computing something on every row. The requirement is the indexed column stays *bare* on one side with an index-friendly operator and a constant on the other. A non-SARGable example: `WHERE date_trunc('day', created_at) = '2026-07-09'` — the function wraps the column, so the B-Tree on `created_at` (sorted by raw values) can't seek; it degrades to a seq scan evaluating `date_trunc` per row. SARGable rewrite: `WHERE created_at >= '2026-07-09' AND created_at < '2026-07-10'`, which the index can serve as a range seek. General fixes: keep the column bare, or build an expression index matching the transformation.

**5. The three join algorithms.**
**Nested Loop**: for each row of the outer input, look up matches in the inner input — extremely efficient when the outer side is small and the inner side is indexed on the join key (each probe is O(log n)); disastrous (O(n·m)) if the inner has no index and both are large. **Hash Join**: build an in-memory hash table on the smaller input keyed by the join column, then scan the larger input probing the hash. Best for large, unsorted, *equality* joins; needs `work_mem` and spills to disk in "batches" if the hash doesn't fit. **Merge Join**: ensure both inputs are sorted on the join key (via indexes or explicit sorts), then walk them in lockstep merging matches. Best when inputs are already sorted (e.g. index order) or very large. The planner picks based on input sizes, available indexes, sort orders, and memory.

**6. Bitmap Index Scan vs plain Index Scan.**
A plain Index Scan walks the index and, for each match, immediately fetches that row from the heap — causing potentially *random* heap I/O, one seek per row. A Bitmap Index Scan instead reads the index to build an in-memory **bitmap** of all matching row locations, then the Bitmap Heap Scan reads the heap pages **in physical order**, each page once. It's preferred for **medium selectivity**: when matches are too many for per-row random fetches to be cheap, but too few to justify scanning the whole table. It also naturally combines multiple indexes (bitmap AND/OR) for multi-column filters. Plain index scans win for very few rows (and enable early termination for `LIMIT`); bitmap wins in the middle; seq scan wins when nearly everything matches.

**7. Estimate 10, actual 900,000 — diagnosis.**
This is a massive **cardinality under-estimate**: the planner thought this node would emit ~10 rows and it emitted 900k. That's the classic root cause of a bad plan, because the planner likely chose a strategy that's only good for tiny inputs — e.g. a nested loop expecting 10 outer rows now loops 900k times, or it skipped an index it should have used. Causes: **stale statistics** (run `ANALYZE table`), **column correlation** the single-column stats can't capture (create extended statistics: `CREATE STATISTICS ... (dependencies) ON a, b FROM t`), a skewed column needing a higher stats target (`ALTER TABLE ... SET STATISTICS 1000`), or a non-SARGable/opaque predicate the planner can't estimate. Fix the estimate first — often the plan then corrects itself — and re-check with `EXPLAIN ANALYZE`.

**8. How the cost-based optimizer chooses, and why cardinality matters.**
The optimizer enumerates candidate plans — access paths per table (seq/index/bitmap), then join orders and join methods — and assigns each an estimated **cost** derived from statistics (row counts, MCVs, histograms) and cost constants (`seq_page_cost`, `random_page_cost`, CPU costs). It picks the lowest-cost plan. The linchpin is **cardinality estimation**: how many rows each node emits. Errors compound multiplicatively up the tree — a 10× underestimate at a leaf can make a join look 10× cheaper than reality and flip the chosen join method (e.g. nested loop instead of hash), turning a good plan catastrophic. That's why "the planner picked a bad plan" almost always traces back to a bad row estimate, and why keeping statistics fresh and adding extended statistics for correlated columns is the highest-leverage tuning after indexing.

**9. `random_page_cost` and SSDs.**
`random_page_cost` is the planner's assumed cost of a *random* (non-sequential) page read relative to `seq_page_cost` (1.0); its default is 4.0, encoding spinning-disk physics where seeking is far slower than streaming. Index scans incur random I/O (jumping around the heap per matched row), so a high `random_page_cost` biases the planner toward seq scans as selectivity drops. On SSDs/NVMe, random reads are nearly as cheap as sequential, so the default over-penalizes index scans; lowering it (commonly ~1.1) tells the planner random access is cheap, shifting it to favor index scans where they genuinely win. It's a storage-aware calibration, ideally set from measured hardware behavior, not guessed.

**10. Scenario: 20× slower, nested loop with big inner seq scan.**
A nested loop with a large inner **sequential** scan means: for every outer row, the database re-scans the whole inner table — O(n·m), which explodes as data grows (hence the 20× regression). Two root causes to check. First, a **missing index on the inner join key**: add it so each probe becomes an index seek instead of a full scan, turning the nested loop efficient (or letting the planner switch to a hash join). Second, a **cardinality misestimate**: the planner chose nested loop because it thought the outer side was tiny; if `EXPLAIN ANALYZE` shows outer `actual rows` far above estimate, run `ANALYZE` (and consider extended stats) so the planner picks a hash join. Verify the fix with `EXPLAIN ANALYZE` and confirm the plan changed and actual time dropped.

**11. Scenario: 500 near-identical small queries — N+1.**
This is the N+1 problem: the code ran one query to load N parents, then looped issuing one child query per parent. Even though each is fast, 1 + N round trips dominate via network latency and per-query planning overhead, and it scales linearly with rows. The fix is to collapse it into one or two queries: either a single `JOIN` fetching parents and children together, or a batched `WHERE child.parent_id = ANY(ARRAY[...])`/`IN (...)` (what ORMs call eager loading / `includes`/`prefetch_related`). Ensure the child's `parent_id` is indexed. This typically cuts hundreds of queries to two and is often the single biggest latency win in an application backend.

**12. Why does a function on an indexed column disable the index?**
A B-Tree index stores and sorts the *raw* column values. A predicate like `WHERE LOWER(email) = 'x'` doesn't ask "where is `email` equal to something" — it asks "where does the *computed* value `LOWER(email)` equal 'x'", and the index has no entries for `LOWER(email)`; it only knows the original values, in original order. So the engine can't seek — it must compute `LOWER(email)` for every row and compare, i.e. a full scan. The fixes reflect this exactly: either don't transform the column (keep the predicate on the bare column), or create an **expression index** on `LOWER(email)` so the index stores precisely the computed key the query filters on.

**13. Why can stale statistics slow a fast query?**
The planner chooses plans from estimated row counts derived from column statistics. As data changes — bulk loads, growth, shifting distributions — those stats drift from reality if `ANALYZE`/autovacuum hasn't refreshed them. With wrong selectivity estimates, the planner can flip to a bad plan: choose a seq scan believing a predicate matches many rows when it now matches few, or pick a nested loop believing an input is tiny when it's huge. Nothing about the SQL changed, but the *plan* silently regressed. That's why a query "suddenly" gets slow after a data load, and why refreshing statistics (`ANALYZE`) is the first thing to check — often it restores the good plan instantly.

**14. Index Scan vs Bitmap Heap Scan vs Seq Scan.**
They sit on a **selectivity spectrum**. Index Scan: descend the index and fetch each matching row from the heap individually — random I/O per row; best for *high* selectivity (few rows) and enables early stop for `LIMIT`/ordered output. Bitmap Heap Scan: read the index to build a bitmap of locations, then read heap pages once each in physical order — amortizes random I/O; best for *medium* selectivity and combining multiple indexes. Seq Scan: read the whole table sequentially — cheapest per page; best for *low* selectivity (most rows match) or small tables. As the fraction of matching rows grows, the optimal choice moves Index → Bitmap → Seq, which is exactly the progression the cost-based planner walks as estimated selectivity drops.

**15. Nested Loop vs Hash Join vs Merge Join.**
Nested Loop: probe the inner for each outer row; wins when the outer input is small and the inner is indexed on the join key (cheap repeated seeks); terrible for two large unindexed inputs. Hash Join: hash the smaller input, probe with the larger; wins for large, unsorted **equality** joins; costs memory and spills to disk (batches) if it exceeds `work_mem`. Merge Join: both inputs sorted on the join key, then merged in one pass; wins when inputs are already sorted (index order) or too big to hash comfortably, and supports the sorted output for free. The planner chooses on input sizes, indexes/sort orders available, join operator (hash needs equality), and memory. Rule of thumb: small+indexed → nested loop; big+unsorted+equality → hash; big+already-sorted → merge.

## Common Mistakes

- **Trusting `EXPLAIN` cost as milliseconds.** Costs are relative planner units; only `EXPLAIN ANALYZE` gives real time.
- **Reading only the top of the plan.** Execution starts at the leaves; the bottleneck is usually a deep node, not the root.
- **Ignoring estimate-vs-actual divergence.** The biggest clue to a bad plan is a large `rows` vs `actual rows` gap — check statistics.
- **Non-SARGable predicates.** Functions/expressions/leading wildcards on indexed columns silently force seq scans.
- **`SELECT *` everywhere.** Blocks index-only scans, drags TOAST columns, wastes I/O and memory.
- **Forgetting `EXPLAIN ANALYZE` executes writes.** Run DML variants inside a rolled-back transaction.
- **Never running `ANALYZE` after bulk loads.** Stale stats produce bad plans on freshly loaded data.
- **N+1 in ORMs.** Lazy loading in loops; use eager loading/joins and index the foreign key.
- **Premature micro-optimization.** Tuning queries that aren't the bottleneck; profile to find the real slow ones first.
- **Ignoring spills.** `Batches > 1` (hash) or `external merge Disk:` (sort) mean you spilled; raise `work_mem` or add an index.

## Related Concepts

- **Indexing** (Topic 1) — the primary access-path lever the planner needs; SARGability exists to use it.
- **Planner statistics, `ANALYZE`, extended statistics (`CREATE STATISTICS`)** — the accuracy of cardinality estimates.
- **VACUUM / autovacuum & the visibility map** — keeps stats fresh and enables index-only scans.
- **`work_mem` / `shared_buffers` / cost constants** — memory and cost calibration that shape plan choice.
- **`pg_stat_statements`** — find the actual slow/expensive queries to optimize first.
- **Prepared statements & generic plans** — plan caching effects on repeated queries.
- **Partitioning** — pruning to touch fewer rows on huge tables.
- **Materialized views & denormalization** — precompute expensive results.
- **ORM eager loading** — the application-side cure for N+1.


---

# Transactions & ACID

## What is it?

A **transaction** is a logical unit of work that groups one or more SQL statements into a single, indivisible operation. Either *all* of the statements succeed and their effects become permanent, or *none* of them do and the database is left exactly as it was before the transaction started.

Think of it from first principles: a database is shared, concurrent, and can crash at any instant. Real-world business operations rarely map to a single row change — transferring money touches two accounts, placing an order touches inventory, an orders table, and a payments table. If the system fails halfway through such a multi-step operation, you must never be left in a state where money left one account but never arrived in the other. The transaction is the abstraction that guarantees this "all-or-nothing" behaviour.

The properties that make transactions trustworthy are captured by the acronym **ACID**: **A**tomicity, **C**onsistency, **I**solation, **D**urability. These four properties are the contract the database engine promises to uphold.

Let me define each precisely, because interviewers love to probe whether you actually understand them versus reciting the acronym.

### Atomicity — "all or nothing"

Atomicity means a transaction is treated as a single indivisible unit. Every statement inside it either commits together or the whole thing is rolled back, leaving no partial effects. If step 3 of a 5-step transaction fails, steps 1 and 2 are undone.

```sql
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;  -- debit
UPDATE accounts SET balance = balance + 100 WHERE id = 2;  -- credit
COMMIT;
```

If the database crashes after the debit but before the credit, atomicity guarantees that on restart the debit is rolled back. You never lose the $100 into the void.

### Consistency — "valid state to valid state"

Consistency means a transaction moves the database from one *valid* state to another *valid* state, respecting all defined rules: primary keys, foreign keys, `CHECK` constraints, `NOT NULL`, unique constraints, and triggers. If a transaction would violate any constraint, it is aborted.

Crucially, consistency is a *shared responsibility*. The database enforces the declared constraints, but the application is responsible for the business-level invariants that aren't expressible as constraints (e.g., "the sum of all account balances in a transfer must remain constant"). ACID's "C" guarantees the DB won't let you commit a state that breaks its declared rules; it does not magically know your business logic.

```sql
-- A CHECK constraint enforcing a business rule
ALTER TABLE accounts ADD CONSTRAINT balance_non_negative CHECK (balance >= 0);

BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;  -- if this makes balance < 0
-- the CHECK fails, the statement errors, and we can ROLLBACK — consistency preserved
ROLLBACK;
```

### Isolation — "concurrent transactions don't step on each other"

Isolation means concurrently executing transactions do not interfere with one another. The *ideal* is that the result of running transactions concurrently is identical to running them one after another in some serial order (this ideal is called **serializability**).

In practice, databases offer several **isolation levels** that trade strictness for performance, allowing certain anomalies (dirty reads, non-repeatable reads, phantom reads) in exchange for higher concurrency. Isolation is the deepest and most examined of the four — the entire next topic (Concurrency Control) is essentially about how isolation is implemented and tuned.

```sql
-- Two clients running simultaneously; isolation decides what each can see
-- of the other's uncommitted or in-flight changes.
BEGIN ISOLATION LEVEL REPEATABLE READ;
SELECT balance FROM accounts WHERE id = 1;  -- reads a stable snapshot
-- ... other transaction modifies id=1 and commits ...
SELECT balance FROM accounts WHERE id = 1;  -- still sees the original value
COMMIT;
```

### Durability — "committed means permanent"

Durability means once a transaction has been committed, its changes survive *any* subsequent failure — power loss, OS crash, process kill. The committed data is safely persisted, typically by flushing a **write-ahead log (WAL)** record to non-volatile storage before acknowledging the commit to the client.

```sql
BEGIN;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;  -- once this returns success, the change is durable.
         -- Pull the power cord one nanosecond later; on restart the +100 is still there.
```

## Why is it needed?

Without transactions, every statement is its own independent, immediately-persisted change, and you have no way to bundle related changes safely. Three fundamental problems arise:

1. **Partial failure corrupts data.** A multi-step business operation (transfer, order, booking) that fails midway leaves the database in an impossible state — money debited but not credited, an order row without a payment row, a seat marked sold but no ticket issued. Atomicity solves this.

2. **Concurrent access produces wrong answers.** Thousands of users hit the same rows at once. Without isolation, one user's half-finished work becomes visible to another, two users overwrite each other's updates, and aggregate reports read inconsistent snapshots. Isolation solves this.

3. **Crashes lose acknowledged work.** A user sees "payment successful," the server crashes, and the payment vanishes because it was only in memory. Durability solves this.

From a first-principles standpoint: databases exist to be the **single source of truth** for a system. A source of truth that can be left half-written, that returns different answers depending on timing races, or that forgets acknowledged writes is not trustworthy. ACID transactions are what let you *reason* about your data. They turn "the database probably reflects reality" into "the database provably reflects a valid, agreed-upon state."

This is also why ACID is the historical dividing line between traditional RDBMSs (PostgreSQL, Oracle, SQL Server, MySQL/InnoDB) and early NoSQL systems that traded ACID for scale. Modern engineering has largely swung back toward wanting strong transactional guarantees wherever correctness matters (finance, inventory, identity).

## How does it work?

You control transaction boundaries explicitly with a small set of commands.

### BEGIN — start a transaction

```sql
BEGIN;              -- or BEGIN TRANSACTION; or START TRANSACTION;
```

Everything after `BEGIN` is provisional — visible to your own session but not yet permanent and (depending on isolation level) not visible to others.

### COMMIT — make it permanent

```sql
COMMIT;             -- durably persist every change since BEGIN
```

`COMMIT` is the moment durability kicks in. The engine flushes the WAL, and only then acknowledges success.

### ROLLBACK — undo everything

```sql
ROLLBACK;           -- discard every change since BEGIN; DB returns to pre-transaction state
```

`ROLLBACK` is the enforcement mechanism of atomicity. Any error, any change of mind, and you throw away the whole unit of work.

### SAVEPOINT / ROLLBACK TO SAVEPOINT — partial rollback

Sometimes you want to undo *part* of a transaction without abandoning the whole thing. A **savepoint** is a named marker inside a transaction you can roll back to.

```sql
BEGIN;
INSERT INTO orders (id, customer_id) VALUES (1001, 42);

SAVEPOINT after_order;

INSERT INTO order_items (order_id, sku, qty) VALUES (1001, 'ABC', 5);
-- oops, SKU 'ABC' is discontinued; undo just the item, keep the order
ROLLBACK TO SAVEPOINT after_order;

INSERT INTO order_items (order_id, sku, qty) VALUES (1001, 'XYZ', 5);
COMMIT;   -- order 1001 with the XYZ item is committed; the ABC attempt vanished
```

`ROLLBACK TO SAVEPOINT` rewinds to the marker but keeps the transaction open, so you can continue. You can also `RELEASE SAVEPOINT after_order;` to discard the marker once you no longer need it.

### The classic bank-transfer example

This is the canonical illustration and a near-guaranteed interview prompt. Transfer $100 from account 1 to account 2:

```sql
BEGIN;

-- Step 1: debit the sender
UPDATE accounts SET balance = balance - 100 WHERE id = 1;

-- Step 2: verify the sender didn't go negative (business rule)
-- (a CHECK constraint or an explicit guard)
SELECT balance FROM accounts WHERE id = 1;   -- must be >= 0

-- Step 3: credit the receiver
UPDATE accounts SET balance = balance + 100 WHERE id = 2;

COMMIT;   -- both updates land together, or neither does
```

Now map the ACID properties onto it:

- **Atomicity**: if the credit (step 3) fails — say account 2 was deleted — the debit (step 1) is rolled back. No money disappears.
- **Consistency**: the invariant "total money across both accounts is unchanged" holds after commit, and the `balance >= 0` CHECK constraint prevents an overdraft from ever committing.
- **Isolation**: if two transfers touch account 1 at the same time, isolation prevents them from both reading the old balance and double-spending (a *lost update*).
- **Durability**: once `COMMIT` returns, the transfer survives a crash; the customer's app can safely show "transfer complete."

## Internal Working

Understanding *how* the engine actually delivers ACID separates a senior candidate from a memorizer. Here's what PostgreSQL (and, broadly, most RDBMSs) do under the hood.

### Write-Ahead Logging (WAL) — the engine of atomicity and durability

The golden rule: **the log record describing a change is flushed to durable storage *before* the change is considered committed, and before dirty data pages are necessarily written back.** This is Write-Ahead Logging.

- When you modify a row, the engine (a) writes a WAL record describing the change and (b) modifies the in-memory page in the buffer pool. The data page itself may sit dirty in memory for a while.
- On `COMMIT`, the engine forces the WAL up to and including the commit record to disk (`fsync`). Only then does it acknowledge success. This gives **durability**: even if the dirty data pages were never written, replaying the WAL after a crash reconstructs the committed state.
- On crash **recovery**, the engine replays WAL (redo) for committed transactions and undoes/omits effects of uncommitted ones. This gives **atomicity** across crashes.

### MVCC — Multi-Version Concurrency Control (the engine of isolation in PostgreSQL)

PostgreSQL does not overwrite a row in place when you `UPDATE` it. Instead it writes a *new version* of the row (a new tuple) and marks the old version as expired. Each tuple carries hidden system columns:

- `xmin` — the transaction ID that created (inserted) this tuple version.
- `xmax` — the transaction ID that deleted/superseded this tuple version (0 if still live).

Every transaction runs against a **snapshot**: a consistent view of which transaction IDs are considered committed-and-visible at the moment the snapshot was taken. Visibility rules use `xmin`/`xmax` plus the snapshot to decide which version of each row *this* transaction should see. Consequences:

- **Readers never block writers and writers never block readers**, because a reader just looks at the appropriate older version while a writer creates a newer one.
- Old versions that are no longer visible to any live snapshot become "dead tuples," reclaimed later by **VACUUM**.

```sql
-- Conceptually, after an UPDATE, two versions coexist:
-- version A: xmin=100, xmax=205  (the old balance, now superseded)
-- version B: xmin=205, xmax=0    (the new balance, live)
-- A transaction whose snapshot predates 205 still sees version A.
```

### Transaction IDs and commit status

Each transaction gets a monotonic **transaction ID (XID)**. A commit-log structure (the `pg_xact`/`clog`) records whether each XID committed, aborted, or is in progress. Visibility checks consult this to interpret `xmin`/`xmax`.

### Locks — the engine of write serialization and stricter isolation

While reads use MVCC snapshots, *writes* still need locks so two transactions can't corrupt the same tuple simultaneously. Row-level locks are acquired on update/delete; a second writer to the same row waits for the first to commit or roll back. Higher isolation levels (SERIALIZABLE) add predicate/serialization tracking on top. This is the bridge into the next topic, Concurrency Control.

### Putting it together for the transfer

`BEGIN` acquires a snapshot lazily; the first `UPDATE` writes WAL + a new tuple version for account 1 and takes a row lock on it; the second does the same for account 2; `COMMIT` writes and flushes a commit WAL record and marks the XID committed in the commit log — at which point the new tuple versions become visible to future snapshots and the change is durable.

## Advantages

- **Correctness under failure.** Atomicity + durability mean crashes and errors never leave half-applied, unrecoverable states. This is the single biggest reason to use an RDBMS for money, inventory, and identity.
- **Correctness under concurrency.** Isolation lets you write business logic *as if* you were the only user, dramatically simplifying application code. You don't have to hand-roll locking protocols for every operation.
- **Declarative integrity.** Consistency via constraints (PK/FK/CHECK/UNIQUE) means invariants are enforced centrally by the engine, not scattered and re-implemented in every application that touches the DB.
- **Simple mental model for developers.** "Wrap related changes in BEGIN/COMMIT and they're safe" is a powerful, easy-to-teach guarantee.
- **Recoverability.** The WAL doubles as the basis for point-in-time recovery, replication, and standby servers.

## Limitations

- **Performance cost.** Enforcing ACID isn't free: WAL flushing (`fsync`) adds latency to commits, locks and snapshots add bookkeeping, and stricter isolation reduces concurrency. High-throughput systems must tune (group commit, batching, appropriate isolation levels).
- **Long transactions are toxic.** A transaction held open for minutes holds locks and pins old row versions, bloating tables (VACUUM can't reclaim tuples still visible to a long-running snapshot) and blocking other writers. Keep transactions short.
- **Distributed ACID is hard.** Across multiple databases/services, single-node ACID doesn't apply; you need two-phase commit (2PC) or saga patterns, both with real trade-offs. ACID guarantees are strongest *within* one database.
- **Consistency is only as good as your constraints.** ACID's "C" won't protect an invariant you never declared. If you forget the `balance >= 0` CHECK and don't guard in code, the DB will happily commit a negative balance.
- **Isolation anomalies at weaker levels.** To get performance, most databases *default* to READ COMMITTED, which still permits non-repeatable and phantom reads. "ACID" does not automatically mean "fully serializable" — you must opt in.

## Real-world Applications

- **Banking and payments.** The archetypal case: fund transfers, ledger postings, double-entry bookkeeping. Every posting is a transaction; the debit and credit must be atomic.
- **E-commerce order placement.** Decrement inventory, create the order, create the payment record, create shipment — all one transaction so you never sell stock you can't fulfill or charge for an order that wasn't created.
- **Ticketing / seat & inventory booking.** Reserve a seat, mark it unavailable, and record the booking atomically to prevent double-booking the same seat.
- **Inventory management.** Stock adjustments across warehouses where partial application would corrupt counts.
- **Identity and account management.** Creating a user plus their default settings plus an audit record together, so you never have a user without required associated rows.
- **Any "move something from A to B" operation.** Transfers of money, points, credits, ownership — all need atomic debit/credit pairs.

## Interview Questions

**Beginner**
1. What is a database transaction?
2. What does ACID stand for, and what does each letter mean?
3. What is the difference between `COMMIT` and `ROLLBACK`?

**Intermediate**
4. Explain the classic bank-transfer example and how each ACID property applies to it.
5. What is a `SAVEPOINT`, and when would you use `ROLLBACK TO SAVEPOINT` instead of a full `ROLLBACK`?
6. Who is responsible for the "Consistency" property — the database or the application?

**Advanced**
7. How does a database actually guarantee Durability? Explain write-ahead logging.
8. How does PostgreSQL implement Isolation using MVCC? What are `xmin` and `xmax`?
9. How is Atomicity preserved across a mid-transaction crash?

**Scenario-based**
10. Your application shows "payment successful" but after a server crash the payment is gone. Which ACID property was violated and how would you fix it?
11. During a batch import, row 5,000 of 10,000 fails a constraint. You want to keep the first 4,999. How would you structure the transaction?

**"Why" questions**
12. Why are long-running transactions considered harmful?
13. Why does ACID's "Consistency" not automatically protect all your business rules?

**Comparison questions**
14. What is the difference between Atomicity and Consistency? (They're commonly confused.)
15. What is the difference between Atomicity and Isolation?

## Model Answers

**1. What is a database transaction?**
A transaction is a logical unit of work composed of one or more SQL statements that the database treats as a single indivisible operation. The defining promise is "all or nothing": either every statement's effect is made permanent together (on `COMMIT`), or none of them take effect (on `ROLLBACK` or failure). Transactions exist because real business operations almost always require multiple related changes — a transfer touches two accounts, an order touches several tables — and the system can fail or be interrupted by concurrent users at any instant. The transaction is the abstraction that lets you bundle those changes so the database is never left in a partial, inconsistent state, and so you can reason about correctness as though you were operating alone and nothing could crash.

**2. What does ACID stand for?**
ACID is Atomicity, Consistency, Isolation, Durability. *Atomicity* is all-or-nothing: the whole transaction commits or the whole thing is undone, with no partial effects. *Consistency* means every committed transaction moves the database from one valid state to another, honoring all declared constraints (PK, FK, CHECK, UNIQUE, triggers); a transaction that would violate a constraint is aborted. *Isolation* means concurrent transactions don't interfere — ideally the concurrent result equals some serial execution; in practice, tunable isolation levels trade strictness for concurrency. *Durability* means once a transaction commits, its effects survive any subsequent crash, because the commit is persisted (typically by flushing a WAL record) before success is acknowledged. Together they make the database a trustworthy single source of truth: correct under failure, correct under concurrency, and never forgetting acknowledged work.

**3. Difference between COMMIT and ROLLBACK?**
`COMMIT` finalizes a transaction: it makes every change since `BEGIN` permanent and durable, flushing the write-ahead log so the changes survive a crash, and makes them visible to other transactions. `ROLLBACK` does the opposite: it discards every change since `BEGIN`, returning the database to exactly the state it was in before the transaction started. `COMMIT` is how you assert "this unit of work is complete and valid"; `ROLLBACK` is the enforcement mechanism of atomicity — triggered either explicitly (you changed your mind or detected an error) or implicitly (a statement errored, or the connection dropped). A key subtlety: after certain errors, a transaction enters an aborted state where the only legal action is `ROLLBACK` (or `ROLLBACK TO SAVEPOINT`); further statements are rejected until you do.

**4. Explain the classic bank-transfer example.**
Transferring $100 from account 1 to account 2 requires two updates: debit account 1 and credit account 2. These must be wrapped in a single transaction. *Atomicity*: if the credit fails after the debit succeeded — say account 2 no longer exists — the debit is rolled back, so money never vanishes. *Consistency*: the invariant that total money is conserved holds after commit, and a `CHECK (balance >= 0)` constraint prevents committing an overdraft. *Isolation*: if two transfers hit account 1 concurrently, isolation stops them both from reading the same starting balance and double-spending (a lost update); one waits for the other via row locks or fails and retries. *Durability*: once `COMMIT` returns, the transfer is persisted and survives a crash, so it's safe for the app to tell the customer the transfer succeeded. The example is powerful because it exercises all four properties in a handful of lines and maps to countless "move value from A to B" operations.

**5. What is a SAVEPOINT?**
A savepoint is a named marker placed inside an open transaction that lets you roll back *part* of the transaction without abandoning all of it. You create one with `SAVEPOINT name;`, and later `ROLLBACK TO SAVEPOINT name;` rewinds the database to that marker while keeping the transaction open so you can continue. You'd use it instead of a full `ROLLBACK` when most of the transaction's work is still good and you only need to undo a recent sub-step — for example, a batch loader that wants to skip an individual bad record without discarding the thousands of good ones already staged, or a procedure that tries an operation, and on failure falls back to an alternative while preserving earlier work. Savepoints also matter for error handling: in PostgreSQL, once a statement errors the transaction is aborted, but if you had wrapped the risky statement between a savepoint and a `ROLLBACK TO SAVEPOINT`, you can recover and keep going. `RELEASE SAVEPOINT name;` discards a marker you no longer need.

**6. Who is responsible for Consistency?**
It's a shared responsibility, and this trips people up. The database guarantees that no transaction can commit a state that violates the constraints you have *declared* — primary keys, foreign keys, `CHECK`s, `UNIQUE`s, `NOT NULL`s, and triggers. That's the part ACID's "C" gives you for free. But the database has no knowledge of business invariants you didn't encode. If your rule is "an account balance may never go negative" and you never added `CHECK (balance >= 0)` nor guarded it in code, the engine will happily commit a negative balance — it's perfectly consistent with the *declared* schema. So: the DB enforces declared integrity; the application (and the DBA who designs the schema) is responsible for translating business rules into constraints and/or correct transaction logic. The practical lesson is to push as many invariants as possible into declarative constraints, because those are enforced centrally and can't be bypassed by a buggy application code path.

**7. How is Durability guaranteed? Explain WAL.**
Durability is delivered primarily through Write-Ahead Logging. The core rule is that a log record describing a change is forced to stable storage *before* the transaction is acknowledged as committed — and before the actual data pages need to be written back. When you modify data, the engine appends a WAL record (a compact description of the change) and modifies the page in the in-memory buffer pool, which may stay "dirty" in memory for a while. On `COMMIT`, the engine issues an `fsync` to flush the WAL up through the commit record to disk, and only then returns success to the client. If the machine loses power immediately after, the data pages might never have been written — but that's fine, because on restart the recovery process replays the WAL (redo) to reconstruct every committed change. This decoupling is also a performance win: sequential WAL writes are far cheaper than random data-page writes, so you get durability without paying random-I/O latency on every commit. Techniques like group commit batch multiple transactions' flushes together to amortize the `fsync` cost.

**8. How does PostgreSQL implement Isolation with MVCC?**
PostgreSQL uses Multi-Version Concurrency Control. Instead of overwriting a row in place, an `UPDATE` writes a new *version* (tuple) of the row and marks the old version as expired; a `DELETE` just marks the version expired. Every tuple carries hidden system columns, chiefly `xmin` (the transaction ID that created this version) and `xmax` (the transaction ID that expired it, or 0 if still live). Each transaction operates against a *snapshot* — a record of which transaction IDs are committed and visible as of a particular moment. Visibility rules combine `xmin`, `xmax`, and the snapshot to decide which version of each row this transaction should see: a transaction sees the version whose creator is committed-and-visible and whose expirer is not visible to it. The huge benefit is that readers never block writers and writers never block readers: a reader simply reads the appropriate older version while a writer creates a newer one. Old versions no longer visible to any snapshot become dead tuples, reclaimed later by `VACUUM`. READ COMMITTED takes a fresh snapshot per statement; REPEATABLE READ takes one snapshot at the start and keeps it for the whole transaction, which is why it sees a stable view.

**9. How is Atomicity preserved across a crash?**
Through the combination of WAL and recovery. As a transaction runs, its changes are described in WAL records and applied to in-memory pages, but nothing is "final" until a commit record is durably written. If the system crashes mid-transaction, on restart the recovery process scans the WAL: it *redoes* changes belonging to transactions that have a durable commit record, and it ensures transactions *without* a commit record leave no lasting effect — either because their dirty pages never reached disk, or, where they did, by undoing them (in engines with undo logs) or by MVCC visibility simply never treating that XID as committed. In PostgreSQL specifically, an uncommitted transaction's tuple versions carry an XID that the commit log marks as aborted, so visibility rules ignore them and `VACUUM` later reclaims them. The net effect: a half-finished transfer that crashed after the debit but before commit is as if it never happened — atomicity holds even across power loss.

**10. "Payment successful" but gone after a crash — which property, and the fix?**
This is a Durability violation: the system acknowledged a commit whose effects did not actually survive the crash. The usual root causes are (a) the write never really reached durable storage — e.g., `fsync` was disabled, or the change was only in an application-side cache / in-memory buffer and never committed to the database at all; or (b) the "success" was reported to the user *before* the database `COMMIT` returned. The fixes: ensure the payment is written inside a real DB transaction and that you only tell the user "successful" *after* `COMMIT` returns; make sure the database's durability settings are honest (don't disable `fsync`/`synchronous_commit` on data you can't afford to lose); and for critical systems, use replication with synchronous standbys so a committed transaction exists on more than one machine before acknowledgment. Architecturally, never treat an in-memory or asynchronously-queued write as durable; the durability boundary is the successful return of `COMMIT`.

**11. Batch import, row 5,000 fails, keep the first 4,999.**
Use savepoints to isolate each risky row (or batch of rows) so one failure doesn't poison the whole transaction. Wrap the import in a single transaction, and before each insert set a savepoint; if the insert fails, `ROLLBACK TO SAVEPOINT` to discard just that row and continue, then `COMMIT` at the end to keep everything that succeeded. Conceptually:

```sql
BEGIN;
-- loop per row:
SAVEPOINT row_sp;
INSERT INTO target (...) VALUES (...);      -- if this errors...
-- ...on error in application code:
ROLLBACK TO SAVEPOINT row_sp;               -- discard just this row, transaction stays alive
RELEASE SAVEPOINT row_sp;                    -- on success, drop the marker
-- end loop
COMMIT;
```

The alternative — committing each row in its own transaction — also "keeps the good ones" but sacrifices the all-or-nothing option and multiplies commit overhead (an `fsync` per row). Savepoints give you fine-grained recovery inside one atomic-until-commit unit. For very large imports, a common middle ground is batching: commit every N rows so a failure only rolls back the current batch, balancing durability overhead against how much work you're willing to redo.

**12. Why are long-running transactions harmful?**
Because an open transaction holds resources that hurt everyone else. It keeps any locks it has acquired, so other transactions that need those rows or tables block behind it, tanking concurrency and sometimes causing cascading waits or deadlocks. In an MVCC engine like PostgreSQL it's arguably worse: a long-running transaction pins an old snapshot, which means `VACUUM` cannot reclaim any row versions that were still visible when that snapshot began — even across the whole database. The result is table and index *bloat*: dead tuples pile up, tables grow, scans slow down, and disk fills. Long transactions also enlarge the blast radius of a rollback and the amount of WAL that must be retained. The senior-engineer guidance is to keep transactions as short as possible: do slow work (network calls, user think-time, heavy computation) *outside* the transaction, open it only to perform the necessary writes, and commit promptly. Never hold a transaction open across a user interaction.

**13. Why doesn't Consistency protect all business rules?**
Because "Consistency" in ACID specifically means "no committed state violates the constraints the database has been *told* about." The engine enforces exactly the rules you declared — keys, `CHECK`s, uniqueness, foreign keys, triggers — and nothing more. It has no semantic understanding of your domain. If your business rule isn't expressed as a constraint or enforced by your transaction logic, the database can't uphold it, and it will consider a state that breaks that unstated rule perfectly "consistent." So an invariant like "a customer's total open orders can't exceed their credit limit" is only protected if you encode it (via a constraint, trigger, or carefully serialized transaction logic). The practical takeaway is to push invariants into declarative constraints wherever possible — they're enforced centrally, atomically, and can't be skipped by an application bug — and to recognize that ACID's "C" is a guarantee about *declared* integrity, not omniscient business-rule enforcement.

**14. Difference between Atomicity and Consistency?**
They're frequently conflated but answer different questions. *Atomicity* is about the *bundling* of a transaction's statements: all of them take effect or none do, with no partial application, even across crashes. It says nothing about whether the resulting state is *valid* — a fully-applied transaction could still, in principle, produce a nonsensical state if the rules allowed it. *Consistency* is about *validity*: every committed transaction must leave the database satisfying all declared constraints, and any transaction that would break them is rejected. Put differently, atomicity guarantees you never see *half* a transaction; consistency guarantees you never see an *invalid* whole. They cooperate — atomicity's rollback is often the mechanism that *achieves* consistency when a statement violates a constraint (the violation triggers a rollback of the whole unit) — but the properties themselves are distinct: one is about indivisibility, the other about rule-conformance.

**15. Difference between Atomicity and Isolation?**
Atomicity concerns a *single* transaction's relationship to failure: its statements are indivisible — all-or-nothing — regardless of what else is happening. Isolation concerns *multiple concurrent* transactions' relationship to each other: it governs what one in-flight transaction can observe of another's uncommitted or concurrent work. You can violate one without the other conceptually: a system could apply a transaction atomically yet expose its uncommitted changes to a concurrent reader (an isolation failure — a dirty read), and conversely a perfectly isolated system still needs atomicity so that a crash mid-transaction doesn't leave a partial result. In the bank transfer, atomicity ensures the debit and credit land together or not at all; isolation ensures a *second* concurrent transfer against the same account doesn't read a stale balance and double-spend. Atomicity is about "the whole unit or nothing"; isolation is about "as if transactions ran one at a time."

## Common Mistakes

- **Confusing Atomicity and Consistency.** Atomicity = all-or-nothing bundling; Consistency = validity against declared constraints. Interviewers deliberately test this.
- **Assuming ACID means fully serializable by default.** Most databases default to READ COMMITTED, which still allows non-repeatable and phantom reads. You must explicitly raise the isolation level if you need stronger guarantees.
- **Reporting success before COMMIT returns.** Telling the user "done" based on an in-memory change or before the commit is acknowledged breaks the durability contract from the user's perspective.
- **Holding transactions open across slow work.** Wrapping user input, network calls, or heavy computation inside a transaction causes lock contention and MVCC bloat. Open late, commit early.
- **Relying on the DB to enforce undeclared business rules.** Forgetting a `CHECK` or foreign key and assuming "consistency" will save you.
- **Forgetting that an errored transaction must be rolled back.** In PostgreSQL, after a statement error the transaction is aborted and rejects further statements until `ROLLBACK` (or `ROLLBACK TO SAVEPOINT`).
- **Using autocommit unknowingly.** Many drivers/clients run in autocommit mode where each statement is its own transaction; multi-statement atomicity requires an explicit `BEGIN`.
- **Catching an exception mid-transaction and continuing without a savepoint.** The continuation silently fails because the transaction is already aborted.

## Related Concepts

- **Concurrency Control & Isolation Levels** (next topic) — the machinery that implements the "I" in ACID.
- **Deadlocks** (topic after) — a direct consequence of the locking used to enforce isolation.
- **Write-Ahead Logging (WAL)** — the mechanism behind atomicity and durability, and the basis of replication and point-in-time recovery.
- **MVCC** — PostgreSQL's approach to isolation without read locks.
- **Two-Phase Commit (2PC) & Sagas** — extending atomicity across multiple databases/services in distributed systems.
- **CAP theorem & BASE** — the distributed-systems trade-offs that contrast with single-node ACID.
- **Constraints (PK, FK, CHECK, UNIQUE)** — the declarative half of the "C" in ACID.
- **VACUUM / autovacuum** — reclaims the dead tuples MVCC produces; tightly linked to transaction lifetime.

# Concurrency Control

## What is it?

**Concurrency control** is the set of techniques a database uses to let many transactions execute *at the same time* while still preserving the "I" (Isolation) in ACID — that is, while making the concurrent result equivalent to *some* valid serial ordering, or at least controlling exactly which deviations from that ideal are permitted.

From first principles: a database with one user needs no concurrency control at all — run each transaction to completion, then the next. But real databases serve thousands of simultaneous connections hammering overlapping rows. If you naively let them all read and write shared data with no coordination, you get *anomalies*: one transaction reads another's half-finished work, two transactions overwrite each other, an aggregate report sees an inconsistent mix of old and new values. Concurrency control is how the engine prevents (or deliberately, and knowingly, permits) these anomalies while extracting as much parallelism as possible.

There are two big families of mechanism, and every database blends them:

- **Locking** — a transaction acquires locks on the data it touches, forcing conflicting transactions to wait. This is *pessimistic*: assume conflict will happen and prevent it up front.
- **Multi-Version Concurrency Control (MVCC)** — keep multiple versions of each row so readers see a consistent snapshot without blocking writers. PostgreSQL is fundamentally MVCC-based; readers don't block writers and writers don't block readers.

On top of these mechanisms sits the user-facing dial: **isolation levels** (READ UNCOMMITTED, READ COMMITTED, REPEATABLE READ, SERIALIZABLE), which let you choose how much isolation you want in exchange for how much concurrency you're willing to give up.

## Why is it needed?

Because uncontrolled concurrency produces *wrong answers*, and wrong answers in a database are often silent and catastrophic. Concretely, without concurrency control you hit classic anomalies:

- **Lost update.** Two transactions read balance = 100, both add 50, both write 150. One update is lost; the correct answer was 200.
- **Dirty read.** Transaction B reads a value Transaction A wrote but hasn't committed; A then rolls back, so B acted on data that never really existed.
- **Non-repeatable read.** Transaction A reads a row, Transaction B updates and commits it, A reads the same row again and gets a different value — within a single transaction, the "same" query changed.
- **Phantom read.** Transaction A runs a range query (`WHERE amount > 1000`), Transaction B inserts a new matching row and commits, A re-runs the query and a *new* row appears that wasn't there before.

At the same time, you can't just serialize everything (run transactions strictly one at a time) — that would reduce a 64-core server handling 50,000 transactions/second to a crawl. So concurrency control exists to thread the needle: **maximize throughput while bounding the anomalies that can occur to a level the application has agreed to tolerate.** The isolation level is that agreement, made explicit.

The deeper reason is *developer sanity*. Good concurrency control lets you write each transaction as if it were the only thing running. Without it, every developer would have to reason about every possible interleaving of every other transaction — an intractable, bug-ridden proposition.

## How does it work?

### Locking: shared vs exclusive

The foundational locking model has two lock modes:

- **Shared lock (S)** — a "read" lock. Multiple transactions can hold a shared lock on the same item simultaneously (many readers coexist). Compatible with other shared locks.
- **Exclusive lock (X)** — a "write" lock. Only one transaction can hold it, and it excludes *all* other locks (shared or exclusive). Needed to modify data.

The compatibility rule in one sentence: **reads can share; writes are exclusive.**

| Requested \ Held | Shared (S) | Exclusive (X) |
|---|---|---|
| **Shared (S)** | Compatible | Conflict (wait) |
| **Exclusive (X)** | Conflict (wait) | Conflict (wait) |

### Locking granularity: row vs table level

Locks can be taken at different *granularities*, trading concurrency against overhead:

- **Row-level locks** — lock just the affected rows. Maximum concurrency (other rows remain free), but more locks to track. This is what you want for OLTP. PostgreSQL locks rows for `UPDATE`/`DELETE`/`SELECT ... FOR UPDATE`.
- **Table-level locks** — lock the whole table. Cheap to track (one lock) but murders concurrency (nobody else can conflictingly touch the table). Used by DDL (`ALTER TABLE`), `LOCK TABLE`, `TRUNCATE`, and as an escalation strategy in some engines.

```sql
-- Explicit row lock: reserve rows for update, blocking other writers of these rows
BEGIN;
SELECT * FROM accounts WHERE id = 1 FOR UPDATE;   -- exclusive row lock on id=1
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
COMMIT;

-- Explicit table lock (rarely needed; blunt instrument)
BEGIN;
LOCK TABLE accounts IN EXCLUSIVE MODE;
-- ...
COMMIT;
```

### Optimistic vs Pessimistic concurrency control

Two philosophies for handling conflict:

- **Pessimistic** — assume conflicts *will* happen, so acquire locks *before* touching data and hold them until commit. Conflicting transactions wait. Great when contention is high (many transactions fight over the same rows); the cost is blocking and potential deadlocks. `SELECT ... FOR UPDATE` is pessimistic.
- **Optimistic** — assume conflicts are *rare*, so don't lock up front; instead read freely, and at commit time *check* whether anyone else changed the data you relied on. If they did, abort and retry. Great when contention is low; the cost is wasted work on the occasional retry. Typically implemented with a **version column** or timestamp:

```sql
-- Optimistic concurrency with a version column
-- Read:
SELECT id, balance, version FROM accounts WHERE id = 1;   -- got version = 7

-- Write only if nobody else changed it since we read (version still 7):
UPDATE accounts
   SET balance = balance - 100, version = version + 1
 WHERE id = 1 AND version = 7;
-- If 0 rows were affected, someone else won the race -> abort and retry.
```

### MVCC: readers don't block writers

PostgreSQL's core strategy. Instead of locking rows for reads, it keeps *multiple versions* of each row (see the Internal Working section of Topic 1: `xmin`/`xmax`). A reading transaction sees a consistent **snapshot** and simply reads the version appropriate to that snapshot. Consequences that are the single most important thing to say in an interview about PostgreSQL:

- **Readers never block writers, and writers never block readers.** A `SELECT` doesn't take shared locks that a writer must wait on; it reads an older version while the writer creates a newer one.
- **Writers still block writers on the same row.** Two transactions updating the *same* row serialize: the second waits for the first to commit or roll back (exclusive row lock).

### Isolation levels: the user-facing dial

SQL defines four standard isolation levels, from weakest to strongest:

```sql
SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;    -- PostgreSQL default
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;

-- or per-transaction:
BEGIN ISOLATION LEVEL SERIALIZABLE;
-- ...
COMMIT;
```

- **READ UNCOMMITTED** — weakest. Permits dirty reads (in the standard). *PostgreSQL note:* because of MVCC, PostgreSQL never actually shows uncommitted data, so it treats READ UNCOMMITTED identically to READ COMMITTED — dirty reads simply don't occur.
- **READ COMMITTED** — you only ever see committed data, but each *statement* gets a fresh snapshot. Prevents dirty reads; still allows non-repeatable reads and phantoms. PostgreSQL's default and the workhorse for most OLTP.
- **REPEATABLE READ** — the whole transaction sees one snapshot taken at its start, so re-reading a row gives the same value. Prevents dirty and non-repeatable reads. *In the SQL standard* it still allows phantoms; *in PostgreSQL* the snapshot-based implementation also prevents phantom reads (though it can still fail a serialization interaction and raise a serialization error).
- **SERIALIZABLE** — strongest. Guarantees the outcome is equivalent to *some* serial execution of the transactions. Prevents all three anomalies plus subtler *serialization anomalies* (like write skew). PostgreSQL implements this as **Serializable Snapshot Isolation (SSI)**, which monitors for dangerous read/write dependency patterns and aborts one transaction with a serialization failure rather than blocking.

## Internal Working

### The anomalies, precisely (with Txn A | Txn B timelines)

**Dirty Read** — reading another transaction's *uncommitted* change.

| Txn A | Txn B |
|---|---|
| `BEGIN;` | |
| `UPDATE accounts SET balance = 0 WHERE id = 1;` (not committed) | |
| | `BEGIN;` |
| | `SELECT balance FROM accounts WHERE id = 1;` → reads **0** (dirty!) |
| `ROLLBACK;` (balance was never really 0) | |
| | acts on a value that never existed |

Prevented at READ COMMITTED and above. PostgreSQL never permits it.

**Non-repeatable Read** — re-reading the *same row* yields a different value because another transaction committed an update in between.

| Txn A | Txn B |
|---|---|
| `BEGIN;` | |
| `SELECT balance FROM accounts WHERE id = 1;` → **100** | |
| | `BEGIN;` |
| | `UPDATE accounts SET balance = 500 WHERE id = 1;` |
| | `COMMIT;` |
| `SELECT balance FROM accounts WHERE id = 1;` → **500** (changed within my txn!) | |
| `COMMIT;` | |

Prevented at REPEATABLE READ and above.

**Phantom Read** — re-running the *same range query* returns a different *set of rows* because another transaction inserted (or deleted) matching rows.

| Txn A | Txn B |
|---|---|
| `BEGIN;` | |
| `SELECT count(*) FROM accounts WHERE balance > 1000;` → **3** | |
| | `BEGIN;` |
| | `INSERT INTO accounts(id,balance) VALUES (99, 5000);` |
| | `COMMIT;` |
| `SELECT count(*) FROM accounts WHERE balance > 1000;` → **4** (a phantom appeared!) | |
| `COMMIT;` | |

Prevented at SERIALIZABLE in the standard; PostgreSQL's REPEATABLE READ (snapshot) already prevents it.

### Isolation-level vs anomaly matrix (SQL standard)

| Isolation Level | Dirty Read | Non-repeatable Read | Phantom Read |
|---|---|---|---|
| READ UNCOMMITTED | Possible | Possible | Possible |
| READ COMMITTED | Prevented | Possible | Possible |
| REPEATABLE READ | Prevented | Prevented | Possible |
| SERIALIZABLE | Prevented | Prevented | Prevented |

### The same matrix as actually implemented by PostgreSQL

| Isolation Level (PostgreSQL) | Dirty Read | Non-repeatable Read | Phantom Read | Serialization Anomaly (e.g. write skew) |
|---|---|---|---|---|
| READ UNCOMMITTED (= READ COMMITTED) | Prevented | Possible | Possible | Possible |
| READ COMMITTED | Prevented | Possible | Possible | Possible |
| REPEATABLE READ | Prevented | Prevented | Prevented | Possible |
| SERIALIZABLE | Prevented | Prevented | Prevented | Prevented |

The key PostgreSQL-specific insight for interviews: PostgreSQL is *stricter than the standard requires*. It never shows dirty reads (so READ UNCOMMITTED collapses into READ COMMITTED), and its REPEATABLE READ already blocks phantoms because it's snapshot-based. The gap that remains at REPEATABLE READ is *serialization anomalies* like **write skew**, which only SERIALIZABLE (via SSI) closes.

### Write skew (why SERIALIZABLE exists beyond phantom prevention)

Classic example: a hospital requires *at least one* doctor on call. Two doctors, both on call, each independently decide to go off call. Each transaction reads "there are 2 on call, so it's safe for me to leave," then updates its own row.

| Txn A (Dr. Alice) | Txn B (Dr. Bob) |
|---|---|
| `BEGIN ISOLATION LEVEL REPEATABLE READ;` | `BEGIN ISOLATION LEVEL REPEATABLE READ;` |
| `SELECT count(*) FROM doctors WHERE on_call = true;` → **2** (OK to leave) | `SELECT count(*) FROM doctors WHERE on_call = true;` → **2** (OK to leave) |
| `UPDATE doctors SET on_call=false WHERE name='Alice';` | `UPDATE doctors SET on_call=false WHERE name='Bob';` |
| `COMMIT;` | `COMMIT;` |
| Result: **0 doctors on call** — invariant violated! | |

Both transactions individually saw a valid state, touched *different rows*, so no write-write conflict and no lock contention — yet together they broke the invariant. This is **write skew**. REPEATABLE READ permits it; only SERIALIZABLE detects the dangerous read/write dependency and aborts one transaction with a serialization error, forcing a retry that then sees the truth.

### How PostgreSQL SERIALIZABLE (SSI) works

Rather than taking heavy read locks, SSI runs at snapshot isolation but tracks read/write dependencies between concurrent transactions using lightweight predicate locks (SIReadLocks). When it detects a "dangerous structure" — a pattern of read-write dependencies that could produce a non-serializable outcome — it aborts one of the transactions with a `could not serialize access` error. Your application is expected to catch that error and **retry the transaction**. This keeps the common case fast (no blocking) while guaranteeing serializable results.

## Advantages

- **Correct results under heavy concurrency.** The whole point: thousands of simultaneous transactions without lost updates, dirty reads, or inconsistent aggregates.
- **Tunable trade-off.** Isolation levels let each transaction pick the right balance of safety vs. throughput. A reporting query can run at READ COMMITTED; a critical financial invariant can run at SERIALIZABLE.
- **MVCC's non-blocking reads (PostgreSQL).** Read-heavy workloads scale beautifully because `SELECT`s never wait for writers and never take locks that block writers — a massive win over pure lock-based systems where reads and writes contend.
- **Developer simplicity.** You can largely write transactions as if isolated, delegating interleaving correctness to the engine.
- **Choice of strategy.** Optimistic concurrency shines under low contention (no lock overhead), pessimistic under high contention (no wasted retries) — you can pick per use case.

## Limitations

- **Higher isolation costs throughput.** SERIALIZABLE adds tracking overhead and causes serialization failures that force retries; REPEATABLE READ pins snapshots. Stronger isolation = less concurrency and more aborts.
- **Locks cause blocking and deadlocks.** Pessimistic locking can stall transactions behind one another and produce deadlocks (next topic). Lock escalation (row → table) can suddenly collapse concurrency.
- **MVCC produces bloat.** Keeping old versions means dead tuples accumulate; `VACUUM` must reclaim them, and long transactions prevent that reclamation.
- **Weak levels silently allow anomalies.** READ COMMITTED (the default) still permits non-repeatable reads, phantoms, and lost updates in read-modify-write patterns. Developers often don't realize their "obvious" code has a race.
- **Optimistic concurrency wastes work under contention.** If conflicts are frequent, constant abort-and-retry can perform worse than just locking.
- **SERIALIZABLE requires retry logic.** Applications *must* handle serialization-failure errors by retrying, or they'll surface as spurious errors to users.

## Real-world Applications

- **Inventory / stock decrement.** `SELECT ... FOR UPDATE` on the stock row (pessimistic) to prevent overselling under a flash sale.
- **Optimistic UI edits.** Collaborative document/record editing using a version column so the second saver is told "someone else changed this, reload" rather than silently clobbering.
- **Banking invariants.** SERIALIZABLE (or explicit locking) for cross-account rules where write skew could otherwise violate a balance or limit invariant.
- **Reporting and analytics.** Long read-only queries rely on MVCC snapshots to see a consistent point-in-time view without blocking the live OLTP writers.
- **Booking systems.** Seat/room reservations use row locks or serializable transactions to prevent double-booking.
- **Job queues.** `SELECT ... FOR UPDATE SKIP LOCKED` lets many workers pull distinct jobs concurrently without stepping on each other — a beautiful use of row-level locking.

## Interview Questions

**Beginner**
1. What is concurrency control and why does a database need it?
2. What is the difference between a shared lock and an exclusive lock?
3. Name the four standard isolation levels from weakest to strongest.

**Intermediate**
4. Explain dirty read, non-repeatable read, and phantom read with an example of each.
5. What is the difference between optimistic and pessimistic concurrency control, and when would you choose each?
6. What does row-level vs table-level locking mean, and what's the trade-off?

**Advanced**
7. Explain MVCC and the claim that in PostgreSQL "readers don't block writers."
8. What is write skew, and why does REPEATABLE READ permit it while SERIALIZABLE prevents it?
9. How does PostgreSQL implement SERIALIZABLE, and what must the application do differently?

**Scenario-based**
10. Under a flash sale, two customers buy the last unit at the same time and you oversell. Diagnose and fix.
11. A nightly report sometimes shows totals that don't reconcile with any point in time. What's happening and how do you fix it?

**"Why" questions**
12. Why is PostgreSQL's REPEATABLE READ stronger than the SQL standard requires?
13. Why doesn't higher isolation come for free — what do you give up?

**Comparison questions**
14. Compare READ COMMITTED and REPEATABLE READ — what anomaly does moving up prevent, and what's the cost?
15. Compare lock-based concurrency control with MVCC.

## Model Answers

**1. What is concurrency control and why needed?**
Concurrency control is the collection of techniques a database uses to allow many transactions to run simultaneously while preserving isolation — keeping the result equivalent to some valid serial ordering, or precisely controlling which deviations are allowed. It's needed because a real database serves thousands of concurrent connections operating on overlapping data, and without coordination you get anomalies: lost updates (two transactions overwrite each other), dirty reads (seeing uncommitted data), non-repeatable reads, and phantoms. But you also can't just run everything serially, because that would destroy throughput on modern multi-core hardware. So concurrency control exists to maximize parallelism while bounding anomalies to a level the application has explicitly agreed to via its chosen isolation level. The deeper payoff is that it lets developers write each transaction as if it were alone, instead of reasoning about every possible interleaving.

**2. Shared vs exclusive lock?**
A shared lock is a read lock: multiple transactions can hold shared locks on the same item at once, because concurrent reads don't conflict. An exclusive lock is a write lock: only one transaction can hold it, and it's incompatible with every other lock, shared or exclusive, because a write must not proceed while anyone is reading or writing the same item. The one-line rule is "reads can share, writes are exclusive." Compatibility: S is compatible with S but conflicts with X; X conflicts with everything. Note that in an MVCC engine like PostgreSQL, ordinary `SELECT`s don't take shared row locks at all — they read a snapshot — so shared/exclusive conflicts mostly arise around explicit locking (`FOR SHARE`/`FOR UPDATE`) and writes, which is a big part of why PostgreSQL reads don't block writes.

**3. Four isolation levels weakest to strongest?**
READ UNCOMMITTED, READ COMMITTED, REPEATABLE READ, SERIALIZABLE. READ UNCOMMITTED allows dirty reads (though PostgreSQL never actually shows uncommitted data, so it behaves like READ COMMITTED). READ COMMITTED shows only committed data but each statement sees a fresh snapshot, so non-repeatable reads and phantoms are still possible — it's PostgreSQL's default. REPEATABLE READ pins a single snapshot for the whole transaction, preventing dirty and non-repeatable reads (and, in PostgreSQL, phantoms too). SERIALIZABLE guarantees the outcome equals some serial execution, additionally preventing serialization anomalies like write skew. As you go up, you trade concurrency for safety.

**4. Explain the three read anomalies.**
A *dirty read* is reading another transaction's uncommitted change: B reads a value A wrote before A commits; if A rolls back, B acted on data that never existed. A *non-repeatable read* is re-reading the *same row* within one transaction and getting a different value because another transaction updated and committed it in between — the row you read twice changed under you. A *phantom read* is re-running the *same range query* and getting a different *set of rows* because another transaction inserted or deleted matching rows — new rows "appear" or existing ones vanish between two identical queries. The distinction between non-repeatable and phantom matters: non-repeatable is about an existing row's *value* changing; phantom is about the *set membership* of a predicate changing. READ COMMITTED stops dirty reads; REPEATABLE READ additionally stops non-repeatable reads (and phantoms in PostgreSQL); SERIALIZABLE stops all of them plus write skew.

**5. Optimistic vs pessimistic — when to use each?**
Pessimistic concurrency assumes conflicts will happen, so it locks data before touching it (e.g., `SELECT ... FOR UPDATE`) and holds the lock until commit, forcing conflicting transactions to wait. Optimistic concurrency assumes conflicts are rare, so it doesn't lock; it reads freely and, at write/commit time, checks whether the data it relied on changed (typically via a version column or timestamp), aborting and retrying if so. Choose pessimistic under *high contention* — when many transactions genuinely fight over the same rows, locking avoids repeated wasted retries and gives predictable serialization; the cost is blocking and possible deadlocks. Choose optimistic under *low contention* — when conflicts are unlikely, you avoid all lock overhead and blocking, paying only the occasional retry; it also suits stateless web patterns where you don't want to hold a database lock across a user's think-time. A common web pattern: read a record with its version, let the user edit for minutes, then update `WHERE version = :read_version`; zero rows affected means someone else won, so you tell the user to reload — no lock was held during the long think-time.

**6. Row-level vs table-level locking trade-off?**
Row-level locking locks only the specific rows a transaction modifies, so other transactions can freely work on other rows of the same table — maximum concurrency, at the cost of tracking many individual locks. Table-level locking locks the entire table, which is cheap to track (a single lock) but devastating for concurrency because no one else can conflictingly touch the table until it's released. OLTP systems want row-level locking to let independent transactions proceed in parallel; PostgreSQL locks rows for `UPDATE`/`DELETE`/`SELECT FOR UPDATE`. Table locks show up for DDL (`ALTER TABLE`), `TRUNCATE`, explicit `LOCK TABLE`, and — in some engines (not typically PostgreSQL) — via *lock escalation*, where a transaction holding too many row locks is promoted to a table lock to save memory, which can abruptly serialize the whole table. The trade-off is granularity vs overhead: finer locks mean more parallelism but more bookkeeping.

**7. Explain MVCC and "readers don't block writers."**
MVCC, Multi-Version Concurrency Control, keeps multiple versions of each row rather than overwriting in place. In PostgreSQL, an `UPDATE` writes a new tuple version and marks the old one expired, with hidden `xmin`/`xmax` columns recording the creating and expiring transaction IDs. Each transaction runs against a snapshot that determines which committed versions are visible to it. Because a reader just consults the version appropriate to its snapshot, it never needs a lock that a writer would have to wait on — so readers don't block writers. Symmetrically, a writer creating a new version doesn't block readers, who continue reading the old version. This is transformative for read-heavy and mixed workloads: long analytical `SELECT`s see a stable, consistent point-in-time view without freezing the OLTP writers, and writers proceed without waiting for readers. The one contention that remains is writer-vs-writer on the *same row*: the second updater takes an exclusive row lock and waits for the first to commit or roll back. The cost of MVCC is that expired versions become dead tuples that `VACUUM` must reclaim, and long-lived snapshots delay that reclamation, causing bloat.

**8. What is write skew and why does the level matter?**
Write skew is a serialization anomaly where two concurrent transactions each read an overlapping dataset, each make a decision that is valid *given what it read*, and each write to *different* rows — so there's no write-write conflict — yet together they violate an invariant that spans the rows. The canonical example is two on-call doctors who each read "two doctors are on call, so it's safe for me to go off call," then each set their own row to off-call; individually valid, jointly they leave zero doctors on call. REPEATABLE READ permits this because it only guarantees each transaction sees a stable snapshot and prevents dirty/non-repeatable/phantom reads — but it doesn't detect that the *combination* of two snapshot transactions writing disjoint rows is non-serializable. SERIALIZABLE prevents it: PostgreSQL's SSI tracks the read/write dependencies between the two transactions, recognizes the dangerous pattern, and aborts one with a serialization error, so on retry it sees the other's committed change and correctly refuses to leave. The lesson is that phantom-prevention alone isn't full serializability; write skew is exactly the gap that SERIALIZABLE closes.

**9. How does PostgreSQL implement SERIALIZABLE and what changes for the app?**
PostgreSQL implements SERIALIZABLE with Serializable Snapshot Isolation (SSI). It doesn't take heavy read locks; instead it runs transactions at snapshot isolation while tracking the read/write dependencies among concurrent transactions using lightweight predicate locks (SIReadLocks). When it detects a "dangerous structure" — a cycle of read-write dependencies that could yield a result not equivalent to any serial order — it aborts one participating transaction with a `could not serialize access due to read/write dependencies` error rather than blocking it. This keeps the common, non-conflicting case as fast as snapshot isolation while still guaranteeing serializable outcomes. The crucial application-side consequence: your code must be prepared to catch serialization-failure errors (SQLSTATE 40001) and *retry the whole transaction*, ideally with a small backoff. If you don't implement retry, SERIALIZABLE will surface as intermittent, confusing errors to users. So adopting SERIALIZABLE is a joint decision: the database gives you correctness, and you accept responsibility for idempotent, retry-safe transaction logic.

**10. Flash-sale oversell — diagnose and fix.**
This is a lost update / read-modify-write race. Both customers' transactions read `stock = 1`, both decide "one left, I can buy," both decrement to 0 and insert an order — two units sold, but only one existed. Under READ COMMITTED (the default), nothing prevents this because each transaction's read saw a committed value and the writes touched the count independently. Fixes, in order of preference: (a) *Pessimistic lock the stock row*: `SELECT stock FROM products WHERE id=:id FOR UPDATE;` at the start, so the second transaction blocks until the first commits and then sees `stock = 0` and refuses. (b) *Atomic conditional update*: `UPDATE products SET stock = stock - 1 WHERE id=:id AND stock > 0;` and treat zero rows affected as "sold out" — this collapses the read-modify-write into a single atomic statement so there's no window. (c) *Optimistic version check* if you prefer, retrying on conflict. (d) *SERIALIZABLE* isolation with retry, which would detect the anomaly and abort one. For a hot single-row counter, the atomic conditional `UPDATE` (b) is usually the cleanest and most scalable.

**11. Nightly report totals don't reconcile — why and fix?**
The report is almost certainly running at READ COMMITTED, where *each statement* gets a fresh snapshot. Over a long multi-query report (or even a single query that scans a large table while writers commit), different parts of the computation observe the database at *different points in time*, so the totals reflect a mix of pre- and post-transaction states that never existed as a single consistent snapshot — for example, it counts a transfer's debit but not its credit because the credit committed between two reads. The fix is to run the report in a single transaction at REPEATABLE READ (or SERIALIZABLE), so the entire report sees one consistent snapshot taken at its start; every query within it observes the same point-in-time view and the totals reconcile. In PostgreSQL you can also use `BEGIN ISOLATION LEVEL REPEATABLE READ` or even a repeatable-read read-only deferrable transaction for reporting. The general principle: any computation that must be internally consistent across multiple reads belongs in one snapshot-stable transaction.

**12. Why is PostgreSQL REPEATABLE READ stronger than the standard?**
The SQL standard defines isolation levels by the anomalies they must *at minimum* prevent, and it explicitly allows REPEATABLE READ to still permit phantom reads. But PostgreSQL implements REPEATABLE READ using snapshot isolation: the transaction takes one snapshot at its start and sees exactly that consistent version of the database for its entire duration. A phantom — a new row appearing in a re-run range query — can't happen, because the transaction's snapshot simply doesn't include rows committed by other transactions after the snapshot was taken. So PostgreSQL's REPEATABLE READ prevents dirty, non-repeatable, *and* phantom reads, exceeding the standard's requirement. The standard only mandates a floor, not a ceiling, so an implementation is free to be stricter. What PostgreSQL's REPEATABLE READ still does *not* prevent is serialization anomalies like write skew, which require SERIALIZABLE. This is a favorite interview point because candidates who memorized the textbook matrix ("REPEATABLE READ allows phantoms") get caught out by PostgreSQL's actual, stronger behavior.

**13. Why isn't higher isolation free?**
Because stronger isolation means the database must do more to hide concurrency, and that costs throughput and introduces failures. REPEATABLE READ pins a snapshot for the whole transaction, which extends how long old row versions must be retained (delaying VACUUM and risking bloat) and can cause serialization errors when a transaction's snapshot conflicts with committed changes. SERIALIZABLE adds dependency tracking (SSI predicate locks and monitoring) and, crucially, *aborts* transactions when it detects potential non-serializable schedules — so you get serialization-failure errors that force retries, consuming CPU on wasted work and requiring retry logic in the app. Lock-based stricter isolation (in other engines) increases lock duration and scope, causing more blocking and deadlocks and reducing parallelism. So the trade-off is fundamental: the closer you get to "as if serial," the more you constrain the concurrency that made the database fast in the first place. The engineering discipline is to use the *lowest* isolation level that is still correct for a given transaction, not blanket-maximize isolation.

**14. Compare READ COMMITTED and REPEATABLE READ.**
READ COMMITTED gives each *statement* its own fresh snapshot: you never see uncommitted data, but two identical queries in the same transaction can return different results if another transaction committed in between (non-repeatable reads) and new matching rows can appear (phantoms). REPEATABLE READ takes *one* snapshot at the transaction's start and uses it for every statement, so re-reads are stable — no non-repeatable reads, and in PostgreSQL no phantoms either. Moving up prevents those read anomalies, which matters for any transaction that reads the same data more than once and needs consistency across those reads (reports, multi-step validations). The cost: REPEATABLE READ holds an older snapshot longer (more version retention / potential bloat) and can *fail* with a serialization error if it detects a conflicting concurrent update, which READ COMMITTED would have silently absorbed by re-reading the latest committed row. So READ COMMITTED favors throughput and never fails for isolation reasons; REPEATABLE READ favors intra-transaction consistency at the price of occasional serialization failures you must retry.

**15. Lock-based CC vs MVCC.**
In pure lock-based concurrency control, reads take shared locks and writes take exclusive locks, and the compatibility rules force conflicting transactions to wait — critically, a writer blocks readers and readers block writers, because they contend on the same locks. This guarantees isolation but serializes read/write access to hot data and is prone to blocking and deadlocks. MVCC instead keeps multiple versions of each row so that reads consult a snapshot rather than taking locks that writers must respect; readers don't block writers and writers don't block readers, which dramatically improves concurrency for mixed and read-heavy workloads. The trade-offs: MVCC must store and later garbage-collect old versions (VACUUM, bloat, and sensitivity to long transactions), and it still needs locks for writer-vs-writer conflicts and for the extra machinery of SERIALIZABLE. Lock-based systems avoid version storage overhead but pay in reduced read/write concurrency. Most modern databases, PostgreSQL included, are MVCC-based precisely because non-blocking reads are such a large practical win, layering locking on top only where writes genuinely conflict.

## Common Mistakes

- **Assuming the default level is safe for read-modify-write.** READ COMMITTED does *not* prevent lost updates; `SELECT` then `UPDATE` on the value you read is a race. Use `FOR UPDATE`, an atomic conditional `UPDATE`, or a version check.
- **Reciting the textbook matrix for PostgreSQL.** "REPEATABLE READ allows phantoms" is true for the *standard* but false for PostgreSQL, which prevents them. Know both.
- **Believing SERIALIZABLE never errors.** It deliberately aborts transactions with serialization failures; forgetting to implement retry logic turns correctness into user-facing errors.
- **Confusing non-repeatable read with phantom read.** Non-repeatable = an existing row's value changed; phantom = the set of rows matching a predicate changed.
- **Thinking MVCC means no locks at all.** Writers still lock rows against other writers; MVCC only removes read-vs-write blocking.
- **Holding pessimistic locks across user think-time or network calls.** This is the classic way to create massive contention and deadlocks; use optimistic concurrency for long human-in-the-loop edits.
- **Ignoring MVCC bloat.** Long transactions plus heavy update churn without adequate VACUUM leads to table/index bloat and degraded performance.
- **Setting SERIALIZABLE globally "to be safe."** Blanket-maximizing isolation needlessly cuts throughput and multiplies retries; pick the lowest correct level per transaction.

## Related Concepts

- **Transactions & ACID** (previous topic) — isolation levels are the tunable expression of the "I."
- **Deadlocks** (next topic) — a direct consequence of the locking used by pessimistic concurrency control.
- **MVCC internals: `xmin`/`xmax`, snapshots, VACUUM** — the mechanism behind non-blocking reads.
- **`SELECT ... FOR UPDATE` / `FOR SHARE` / `SKIP LOCKED` / `NOWAIT`** — explicit locking clauses for pessimistic control and job queues.
- **Serializable Snapshot Isolation (SSI)** — PostgreSQL's serializable implementation.
- **Lost update, write skew, serialization anomalies** — the concurrency bugs isolation levels do or don't prevent.
- **Optimistic concurrency / version columns / ETags** — the application-level pattern for low-contention conflict handling.
- **Two-phase locking (2PL)** — the classical locking protocol underpinning lock-based serializability.

---

# Deadlocks

## What is it?

A **deadlock** is a situation where two or more transactions are each waiting for a lock that another of them holds, forming a *cycle of waiting* that can never resolve on its own. Every transaction in the cycle is blocked, holding resources the next one needs, and none can proceed — they would wait forever unless the database intervenes.

From first principles: pessimistic concurrency control makes transactions *wait* for locks. Waiting is normal and fine when it's linear — A waits for B, B finishes, A proceeds. A deadlock is what happens when the wait graph becomes *circular*: A waits for B, and B waits for A. Now the chain has no end. Neither will ever release its lock because each is blocked acquiring the other's. This is the concurrency-control equivalent of two people in a narrow hallway, each stepping the same direction to let the other pass, forever.

The formal way to see it: model transactions and the locks they wait for as a directed **wait-for graph** — an edge from T1 to T2 means "T1 is waiting for a lock held by T2." A deadlock exists precisely when this graph contains a *cycle*. Databases detect deadlocks by finding such cycles.

Deadlocks are not bugs in the database — they're an inherent possibility whenever multiple transactions acquire multiple locks in different orders. A well-engineered system minimizes them and handles the ones that occur gracefully.

## Why is it needed?

This heading is really "why do deadlocks *matter* and why must we understand them" — deadlocks aren't something you *want*, they're something the concurrency system inevitably produces and must manage.

They matter because:

1. **They're unavoidable in general.** Any system that lets transactions hold one lock while requesting another can, in principle, deadlock. You cannot design them completely out of a rich transactional system without unacceptable restrictions; you can only reduce their frequency and handle them well.

2. **Unhandled, they'd hang the system.** Without detection, deadlocked transactions would block indefinitely, and every other transaction waiting behind *them* would pile up, cascading into a stalled database. That's why every serious RDBMS has automatic deadlock detection and resolution.

3. **They surface as runtime errors your application must handle.** When PostgreSQL breaks a deadlock, it aborts one transaction with `ERROR: deadlock detected` (SQLSTATE 40P01). If your app doesn't catch and retry that, users see failures. So understanding deadlocks is essential to writing robust transactional code.

4. **Their frequency is a design signal.** Frequent deadlocks usually indicate a *design* problem — inconsistent lock ordering, over-broad locking, or transactions that are too long. Understanding them lets you fix the root cause rather than just retrying.

So we study deadlocks not because they're desirable but because they're an emergent, inevitable property of concurrency control that a competent engineer must anticipate, minimize, detect, and recover from.

## How does it work?

### The four Coffman conditions

A deadlock requires *all four* of these to hold simultaneously (classic operating-systems theory that maps directly onto databases):

1. **Mutual exclusion** — a resource (lock) can be held by only one transaction at a time (true of exclusive locks).
2. **Hold and wait** — a transaction holds at least one lock while requesting another.
3. **No preemption** — a lock can't be forcibly taken away; it's released only voluntarily (by commit/rollback).
4. **Circular wait** — a cycle of transactions exists, each waiting for a lock held by the next.

Break *any one* of these and deadlock becomes impossible. Prevention strategies (below) each target one of these conditions — usually circular wait (via lock ordering) or hold-and-wait (via lock timeouts / acquiring all locks up front).

### The classic two-transaction deadlock

The textbook case: two transactions update the same two rows in *opposite order*. This is a near-certain interview question, so internalize the timeline.

| Txn A | Txn B |
|---|---|
| `BEGIN;` | `BEGIN;` |
| `UPDATE accounts SET balance = balance - 100 WHERE id = 1;` | |
| *(A now holds exclusive lock on row 1)* | |
| | `UPDATE accounts SET balance = balance - 50 WHERE id = 2;` |
| | *(B now holds exclusive lock on row 2)* |
| `UPDATE accounts SET balance = balance + 100 WHERE id = 2;` | |
| *(A wants row 2 — held by B — **A waits**)* | |
| | `UPDATE accounts SET balance = balance + 50 WHERE id = 1;` |
| | *(B wants row 1 — held by A — **B waits**)* |
| **DEADLOCK**: A waits for B, B waits for A | |

The wait-for graph now has a cycle: A → B (A waits for row 2 held by B) and B → A (B waits for row 1 held by A). Neither will ever release, because releasing requires committing, and committing requires acquiring the lock it's blocked on.

### What the database does about it

The engine runs a **deadlock detector**. Periodically (in PostgreSQL, after a transaction has waited for `deadlock_timeout`, default 1 second), it builds the wait-for graph and searches for a cycle. When it finds one, it picks a **victim** transaction, aborts it (rolls it back, releasing its locks), and returns a deadlock error to that client. The surviving transaction(s) can then acquire the freed locks and proceed.

```sql
-- The victim receives:
-- ERROR:  deadlock detected
-- DETAIL: Process 1234 waits for ShareLock on transaction 5678; blocked by process 4321.
--         Process 4321 waits for ShareLock on transaction 5679; blocked by process 1234.
-- HINT:   See server log for query details.
-- SQLSTATE: 40P01
```

The correct application response is to catch SQLSTATE 40P01 and **retry the whole transaction**. On retry, the contention has usually cleared and it succeeds.

### The fix for the classic case: consistent lock ordering

If both transactions always lock rows in the *same* order (say, ascending by `id`), the cycle can't form. Whoever grabs row 1 first also gets row 2 next; the other simply waits linearly for both — no cycle:

```sql
-- Both transactions lock in ascending id order -> no circular wait possible
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;   -- lock lower id first
UPDATE accounts SET balance = balance + 100 WHERE id = 2;   -- then higher id
COMMIT;
```

## Internal Working

### Detection: the wait-for graph and `deadlock_timeout`

PostgreSQL doesn't check for deadlocks on every lock request — that would be wasteful, since the vast majority of waits are transient and resolve quickly. Instead, when a transaction blocks on a lock, it sleeps. If it's still waiting after `deadlock_timeout` (default 1s), the engine runs the deadlock-detection algorithm: it constructs the wait-for graph among the currently waiting transactions and performs a cycle search (a depth-first traversal looking for a back-edge). If no cycle, the transaction keeps waiting (the timeout was just a check). If a cycle is found, resolution kicks in.

The `deadlock_timeout` is a deliberate trade-off: too low and you waste CPU running detection on waits that would have cleared naturally; too high and genuinely deadlocked transactions stay stuck longer before being broken. One second is a sensible default for OLTP.

### Victim selection

Once a cycle is confirmed, the engine must break it by aborting one participant. PostgreSQL aborts the transaction *whose lock request would complete the cycle* — essentially the one that detected the deadlock while trying to acquire the final edge. Other database engines use cost heuristics: SQL Server, for instance, chooses the transaction estimated to be *cheapest to roll back* (least log/work done) as the "deadlock victim." The aborted transaction is rolled back entirely, its locks released, and it receives the deadlock error; the survivor(s) proceed.

### Distinguishing deadlock from a plain lock wait / lock timeout

It's important not to conflate two different things:

- A **deadlock** is a *cycle* — it will never resolve, so the engine must actively break it. Detection is mandatory and resolution is an abort.
- A **long lock wait** (or **lock timeout**) is *linear* — A is simply waiting for B, which will eventually commit. This is not a deadlock; it resolves on its own. You can cap how long you're willing to wait with `lock_timeout` or `SELECT ... FOR UPDATE NOWAIT` / `SKIP LOCKED`, but that's a separate mechanism from deadlock detection.

```sql
SET lock_timeout = '3s';   -- abort MY statement if I wait > 3s for any lock
SELECT * FROM accounts WHERE id = 1 FOR UPDATE NOWAIT;  -- error immediately if locked
SELECT * FROM jobs WHERE state='ready' FOR UPDATE SKIP LOCKED LIMIT 1;  -- skip locked rows
```

### Lock upgrade deadlocks (a subtler internal cause)

A common non-obvious deadlock source is *lock upgrades*: two transactions both take a shared lock on the same row (compatible, so both succeed), then both try to upgrade to exclusive to write. Each needs the other to drop its shared lock first — a two-transaction deadlock that didn't involve two different rows at all. This is why patterns like "`SELECT` then later `UPDATE` the same row" under explicit shared locking can deadlock, and why `SELECT ... FOR UPDATE` (taking the exclusive lock up front) is often safer than `SELECT ... FOR SHARE` followed by an update.

## Advantages

Deadlocks themselves have no advantages — but *automatic deadlock detection and resolution* is a genuine feature, and that's what this heading is really about:

- **Guaranteed liveness.** Automatic detection means the database never permanently hangs on a deadlock; it always breaks the cycle and keeps making progress, rather than leaving transactions stuck forever.
- **No application-side deadlock avoidance required for correctness.** You don't have to prove your lock-ordering is globally acyclic to avoid a frozen system; the engine is a safety net that catches any cycle you missed.
- **Fast, bounded resolution.** With `deadlock_timeout`, a deadlock is detected and broken within a small, predictable time window rather than lingering.
- **Clear, catchable signal.** The deadlock error (40P01) is explicit, so applications can implement a clean retry policy, and the server log's detail helps diagnose the offending queries for a permanent fix.
- **It enables aggressive concurrency.** Because deadlocks are handled, the system can allow fine-grained locking and high parallelism without designers having to be paranoid that any missed ordering will hang production.

## Limitations

- **A victim transaction is aborted.** Deadlock resolution always sacrifices someone's work; that transaction is rolled back and must be retried, wasting the effort it had done.
- **Requires retry logic in the app.** If you don't catch and retry 40P01, deadlocks become user-visible errors. This is real engineering work that's easy to forget.
- **Detection has a latency and CPU cost.** The `deadlock_timeout` wait means a real deadlock isn't broken instantly, and running the detector consumes CPU. Tuning is a trade-off.
- **Frequent deadlocks signal deeper problems and degrade throughput.** High deadlock rates mean lots of rolled-back work and retries, hurting performance — a symptom of bad lock ordering, over-broad locks, or long transactions.
- **Non-determinism.** Which transaction becomes the victim can be hard to predict, making failures feel random and complicating debugging.
- **Retries can starve or livelock.** Naive immediate retries of two transactions can repeatedly re-deadlock; you need randomized backoff to avoid a livelock where they keep colliding.

## Real-world Applications

Deadlocks aren't "applied" — but understanding and mitigating them is applied constantly:

- **Fund transfers / double-entry ledgers.** The archetypal deadlock generator: transfers touching pairs of accounts in varying orders. Solved by always locking accounts in a canonical order (e.g., by account id).
- **Inventory and order systems.** Orders that lock multiple product rows; without consistent ordering, concurrent multi-item orders deadlock.
- **Batch jobs vs OLTP.** A long batch update taking many locks colliding with live transactions — mitigated by short transactions, off-peak scheduling, and `SKIP LOCKED`.
- **Job queues / worker pools.** Deliberately *avoiding* deadlocks and lock contention with `SELECT ... FOR UPDATE SKIP LOCKED` so workers grab disjoint jobs.
- **Idempotent, retry-safe service design.** Any service issuing multi-row transactions is built to catch 40P01 (and serialization failures) and retry with backoff — a standard resilience pattern.
- **Migrations / schema changes.** DDL taking strong locks can deadlock with live traffic; run with `lock_timeout` and during low-traffic windows to avoid taking down production.

## Interview Questions

**Beginner**
1. What is a deadlock in a database?
2. What does PostgreSQL do when it detects a deadlock?
3. What is a wait-for graph?

**Intermediate**
4. Walk through the classic two-transaction deadlock with a timeline.
5. What are the four conditions required for a deadlock, and how does breaking one prevent it?
6. What's the difference between a deadlock and a transaction that's simply waiting a long time for a lock?

**Advanced**
7. How does PostgreSQL detect deadlocks, and what role does `deadlock_timeout` play?
8. How is the victim transaction chosen, and what happens to it?
9. Explain a lock-upgrade deadlock and how to avoid it.

**Scenario-based**
10. Your logs show frequent `deadlock detected` errors on a money-transfer endpoint under load. How do you diagnose and permanently fix it?
11. A worker pool of 20 processes polling a jobs table keeps deadlocking as they grab work. How do you fix it?

**"Why" questions**
12. Why can't you simply prevent all deadlocks by design and remove detection?
13. Why does keeping transactions short reduce deadlocks?

**Comparison questions**
14. Compare deadlock *prevention* and deadlock *detection/resolution*.
15. Compare optimistic concurrency and pessimistic locking with respect to deadlocks.

## Model Answers

**1. What is a deadlock?**
A deadlock is a state in which two or more transactions are each waiting for a lock held by another, forming a cycle of waiting that cannot resolve on its own. Every transaction in the cycle is blocked while holding a resource the next one needs, so none can proceed. The simplest form is two transactions: A holds a lock on row 1 and wants row 2, while B holds row 2 and wants row 1 — A waits for B and B waits for A, forever. It arises inherently from pessimistic locking whenever transactions hold one lock while requesting another and do so in conflicting orders. Databases model this as a wait-for graph and define a deadlock precisely as a *cycle* in that graph. It's not a database bug; it's an emergent possibility of concurrency that the engine must detect and break.

**2. What does PostgreSQL do on detecting a deadlock?**
It resolves the deadlock by choosing a *victim* transaction, aborting and rolling it back to release its locks, and returning an error to that client — `ERROR: deadlock detected`, SQLSTATE 40P01, with server-log detail identifying the blocked processes and queries. The other transaction(s) in the cycle can then acquire the freed locks and continue. This guarantees the system never permanently hangs. From the application's perspective, the aborted transaction's work is lost and the correct response is to catch the 40P01 error and retry the whole transaction, usually with a short randomized backoff; on retry the contention has typically cleared and it succeeds. The detail in the server log is also your primary tool for finding the offending queries so you can apply a permanent fix such as consistent lock ordering.

**3. What is a wait-for graph?**
A wait-for graph is a directed graph the database uses to reason about lock waits: each node is a transaction, and a directed edge from T1 to T2 means "T1 is waiting to acquire a lock currently held by T2." As long as this graph is acyclic, every wait is linear and will eventually resolve as transactions commit and release their locks. A deadlock exists precisely when the wait-for graph contains a *cycle* — a set of transactions each waiting on the next, closing back on itself, so no one can ever proceed. Deadlock detection is therefore implemented as cycle detection over this graph (a depth-first search looking for a back edge). When a cycle is found, the engine breaks it by aborting one node (the victim), which removes its edges and unblocks the rest.

**4. Walk through the classic two-transaction deadlock.**
Two transactions update the same two rows in opposite order. Txn A begins and updates row 1 (`id=1`), acquiring an exclusive lock on it. Concurrently Txn B begins and updates row 2 (`id=2`), acquiring an exclusive lock on that row. Now A proceeds to update row 2 — but row 2 is locked by B, so A blocks and waits. Meanwhile B proceeds to update row 1 — but row 1 is locked by A, so B blocks and waits. The wait-for graph now has A → B and B → A: a cycle. Neither can release its lock because releasing requires committing, and committing requires first acquiring the lock it's blocked on. The database's deadlock detector notices the cycle (after `deadlock_timeout`), aborts one of them (say A) with a deadlock error, releasing A's lock on row 1 so B can complete; A then retries. The root-cause fix is consistent lock ordering: if both transactions always touch the rows in ascending id order, whoever gets row 1 first also gets row 2, and the other just waits linearly — no cycle can form.

**5. What are the four conditions for deadlock?**
The four Coffman conditions, all of which must hold simultaneously: (1) *Mutual exclusion* — a lock can be held by only one transaction at a time; (2) *Hold and wait* — a transaction holds at least one lock while requesting another; (3) *No preemption* — locks aren't forcibly taken away, only released voluntarily by commit or rollback; (4) *Circular wait* — a cycle of transactions each waits for a lock held by the next. Because all four are necessary, breaking any single one makes deadlock impossible, and every prevention strategy targets one of them. Consistent lock ordering eliminates *circular wait* (the most common practical approach). Acquiring all needed locks atomically up front eliminates *hold and wait*. Lock timeouts introduce a form of *preemption* by aborting a waiter. Mutual exclusion is essentially inherent to write locking, so it's rarely the one you attack. Framing prevention as "which Coffman condition am I breaking" is a clean way to reason about it in an interview.

**6. Deadlock vs long lock wait?**
A long lock wait is *linear and self-resolving*: transaction A is waiting for a lock held by B, and B is actively working and will eventually commit or roll back, at which point A proceeds. Nothing is fundamentally stuck; A is just slow because of contention. A deadlock is *circular and permanent*: A waits for B and B waits for A (possibly through a longer cycle), so no one will ever release, and the wait would last forever without intervention. The database treats them differently: a plain wait is allowed to continue (optionally bounded by your `lock_timeout` or `NOWAIT`/`SKIP LOCKED` choices), whereas a deadlock is a cycle the engine *must* actively detect and break by aborting a victim. Practically, a lock wait shows up as latency; a deadlock shows up as a 40P01 error on one transaction. Confusing the two leads to misdiagnosis — e.g., raising `deadlock_timeout` won't fix slow waits, and adding `lock_timeout` won't fix true deadlocks (it just converts them into timeout errors sooner).

**7. How does PostgreSQL detect deadlocks and what is `deadlock_timeout`?**
PostgreSQL uses lazy, on-demand detection rather than checking every lock acquisition. When a transaction requests a lock that's unavailable, it blocks and sleeps. If it's still blocked after `deadlock_timeout` (default one second), the engine runs its deadlock-detection routine: it builds the wait-for graph over currently waiting transactions and searches for a cycle via a depth-first traversal. If there's no cycle, the transaction simply keeps waiting — the timeout was just a periodic check, and the wait continues (this correctly handles the common case of a slow-but-not-deadlocked wait). If a cycle is found, PostgreSQL resolves it by aborting a victim. The `deadlock_timeout` is a tuning knob balancing two costs: set it too low and you burn CPU running detection on waits that would have cleared on their own; set it too high and genuine deadlocks stay stuck longer before being broken. One second is a good OLTP default because most legitimate waits clear well under it, so detection rarely runs needlessly, yet real deadlocks are broken promptly.

**8. How is the victim chosen and what happens to it?**
When a cycle is confirmed, the engine must abort one participant to break it. In PostgreSQL, the victim is essentially the transaction whose lock acquisition would close the cycle — the one that runs the detection upon timing out and finds itself completing a loop. It is rolled back entirely: all its changes are undone and all its locks released, which removes its edges from the wait-for graph and lets the remaining transactions acquire the freed locks and proceed. The victim receives the deadlock error (40P01) and is expected to retry. Other engines use different heuristics — SQL Server, for example, picks the transaction it estimates is *cheapest to roll back* (least log generated / work done) as the deadlock victim, minimizing wasted effort, and lets you bias this with `DEADLOCK_PRIORITY`. Either way, the defining outcome is that exactly one transaction is sacrificed and fully rolled back so the rest can live; deadlock resolution is inherently destructive to someone's work, which is why minimizing deadlock frequency matters and why retry logic is mandatory.

**9. Explain a lock-upgrade deadlock.**
A lock-upgrade deadlock happens without two different rows being involved in opposite order. Two transactions each take a *shared* lock on the *same* row — shared locks are mutually compatible, so both succeed. Then each transaction tries to *upgrade* its shared lock to an *exclusive* lock in order to write that row. But an exclusive lock is incompatible with any other lock, including the other transaction's shared lock, so each upgrade must wait for the other to release its shared lock first — and neither will, because each is blocked trying to upgrade. That's a cycle: a two-transaction deadlock arising purely from lock upgrades. It commonly appears with a `SELECT ... FOR SHARE` (or a read that takes a shared lock) followed later by an `UPDATE` of the same row. The fix is to acquire the *exclusive* lock up front instead of upgrading: use `SELECT ... FOR UPDATE` at the point of read so the very first lock is already exclusive, meaning the second transaction blocks immediately (a plain linear wait) rather than both sneaking in with shared locks and then deadlocking on the upgrade. More generally, avoid the read-shared-then-write pattern on rows you intend to modify.

**10. Frequent deadlocks on a money-transfer endpoint — diagnose and fix.**
First, diagnose using the server logs: the 40P01 detail lists the two blocked processes and their queries, which almost always reveals the pattern — concurrent transfers locking the *same pair* of account rows in *opposite order* (transfer X→Y locks X then Y; transfer Y→X locks Y then X). That opposite ordering is the circular-wait condition. The permanent fix is *consistent lock ordering*: make every transfer lock the involved accounts in a canonical order — for example, always lock the lower account id first, regardless of which is sender and which is receiver. In code, sort the two account ids and `SELECT ... FOR UPDATE` them in ascending order before doing the debit/credit, so no cycle can ever form. Complementary measures: keep the transaction as short as possible (do no network calls or user interaction while holding the locks), and add retry-on-40P01 with randomized backoff as a safety net for any residual deadlocks. Together, consistent ordering eliminates the structural cause and short transactions plus retries handle the rare edge cases. Raising `deadlock_timeout` is *not* a fix — it only delays detection.

**11. Worker pool deadlocking while grabbing jobs.**
The workers are contending on the jobs table — likely each doing something like `SELECT ... WHERE state='ready' ... FOR UPDATE` and colliding on the same candidate rows, blocking and sometimes deadlocking as they lock rows in overlapping orders or upgrade locks. The idiomatic fix is `SELECT ... FOR UPDATE SKIP LOCKED`: each worker asks for the next ready job but *skips* any row another worker has already locked, so twenty workers grab twenty distinct jobs with no blocking and no deadlocks. A typical pattern is `SELECT id FROM jobs WHERE state='ready' ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1`, then update that job's state to 'processing' and commit quickly. `SKIP LOCKED` sidesteps the whole problem by eliminating contention on the same rows rather than fighting over them. Keep the claiming transaction tiny (claim and commit, then process the job outside the transaction) so locks are held for microseconds. This turns a deadlock-prone hotspot into a clean, highly concurrent queue.

**12. Why can't you just prevent all deadlocks and drop detection?**
Because guaranteeing a globally acyclic lock order across an entire non-trivial application is effectively impossible in practice. Lock acquisition order often depends on runtime data (which rows a query touches, which depends on user input), on many independent code paths written by different people over time, on the query planner's choices, and on implicit locks taken by foreign keys, triggers, and indexes that you don't directly control. Enforcing one canonical order everywhere would require either draconian restrictions (e.g., take all locks up front in a fixed order, which kills concurrency and is often infeasible when you don't know the lock set in advance) or perfect discipline no real codebase sustains. So databases treat prevention as *best-effort mitigation* — you reduce deadlock frequency with consistent ordering, short transactions, and appropriate isolation — and rely on *detection and resolution* as the guaranteed safety net that keeps the system live no matter what ordering slips through. Removing detection would mean any single missed ordering could permanently hang production. Detection is cheap insurance against an unavoidable residual risk.

**13. Why do short transactions reduce deadlocks?**
Because the probability of a deadlock is tied to how long transactions hold locks and how many they accumulate while waiting. A deadlock needs two (or more) transactions to be simultaneously holding locks the others want — that's a *window of overlap*. The longer a transaction holds its locks (the longer it runs before committing), the wider that window, and the more likely a concurrent transaction will request a conflicting lock during it, forming a cycle. Short transactions acquire their locks, do their work, and commit quickly, releasing locks fast and shrinking the overlap window toward zero, so cycles rarely have time to form. Short transactions also tend to hold *fewer* locks at once, reducing the number of edges that could close a cycle. This is why the standard guidance is to do slow work — network calls, user think-time, heavy computation, external API calls — *outside* the transaction, opening it only to perform the necessary writes and committing immediately. It's the single most effective and universally applicable deadlock-reduction (and general contention-reduction) technique, and it costs nothing in correctness.

**14. Compare deadlock prevention vs detection/resolution.**
Prevention tries to make deadlocks *impossible* by construction, by ensuring one of the four Coffman conditions can never hold. In databases this is mostly *consistent lock ordering* (kills circular wait), acquiring all locks atomically up front (kills hold-and-wait), and lock timeouts (a form of preemption). Prevention is proactive and, where achievable, avoids the wasted work of aborted transactions entirely — but perfect prevention is often impractical because lock sets and orders depend on runtime data and sprawling code paths. Detection/resolution is *reactive*: it lets deadlocks happen, then finds them (cycle search in the wait-for graph after `deadlock_timeout`) and breaks them by aborting a victim. It guarantees liveness regardless of application discipline, but pays the cost of rolled-back work and requires app-side retry logic, and it introduces detection latency. Real systems use *both*: design for prevention to keep deadlock frequency low (consistent ordering, short transactions, minimal lock scope), and rely on the engine's detection/resolution as the guaranteed backstop for whatever slips through. Prevention optimizes the common case; detection guarantees the system never hangs.

**15. Optimistic concurrency vs pessimistic locking re: deadlocks.**
Pessimistic locking acquires locks before touching data and holds them until commit, which is exactly what creates the possibility of deadlock — multiple transactions holding locks while waiting for others' locks can form a cycle. So deadlocks are an inherent risk of pessimistic concurrency, mitigated by consistent ordering and short transactions and handled by detection. Optimistic concurrency, by contrast, *doesn't hold locks* during its read/compute phase; it reads freely and only checks for conflicts at commit time via a version column or timestamp, aborting and retrying on conflict. Because it isn't holding locks and waiting on others' locks, classic lock-cycle deadlocks essentially don't arise. The trade-off is that optimistic concurrency converts the problem from "deadlock" into "conflict-and-retry": under high contention it can suffer many aborts and retries (and, if not careful with backoff, livelock where transactions keep colliding), which can be less efficient than just locking. So pessimistic locking risks deadlocks but avoids wasted retries under high contention; optimistic concurrency avoids lock deadlocks but risks retry churn under contention. The choice follows contention level: high contention favors pessimistic (with good lock ordering), low contention favors optimistic.

## Common Mistakes

- **Not implementing retry on 40P01.** The single most common mistake. Deadlocks *will* happen occasionally even in good designs; without catch-and-retry (with randomized backoff), they surface as user-facing errors.
- **Inconsistent lock ordering.** Acquiring multi-row locks in data-dependent order (e.g., sender-then-receiver) is the classic cause. Always lock in a canonical order (e.g., by primary key).
- **Confusing deadlock with lock wait.** Misdiagnosing a slow linear wait as a deadlock (or vice versa) leads to the wrong fix; `lock_timeout` and `deadlock_timeout` address different problems.
- **Raising `deadlock_timeout` to "fix" deadlocks.** It only delays detection, leaving the deadlock stuck longer; it doesn't remove the cycle.
- **Long transactions.** Holding locks across user interaction, network calls, or heavy computation widens the deadlock window dramatically.
- **The read-shared-then-write pattern.** `SELECT FOR SHARE` (or a shared-locking read) followed by an `UPDATE` invites lock-upgrade deadlocks; take `FOR UPDATE` up front instead.
- **Retrying immediately without backoff.** Two transactions that deadlocked can immediately re-collide on retry (a livelock); add randomized backoff.
- **Over-broad locking.** Locking whole tables or more rows than needed multiplies the chances of a cycle; lock the minimum necessary.

## Related Concepts

- **Concurrency Control & Locking** (previous topic) — deadlocks are the failure mode of pessimistic locking; shared/exclusive lock compatibility is what makes cycles possible.
- **Transactions & ACID** (first topic) — deadlock resolution rolls back a transaction, relying on atomicity; retry restores correctness.
- **Wait-for graph & the four Coffman conditions** — the theoretical model of deadlock and its prevention.
- **`lock_timeout`, `NOWAIT`, `SKIP LOCKED`** — statement-level lock-wait controls; `SKIP LOCKED` is the go-to for deadlock-free job queues.
- **`deadlock_timeout`** — the PostgreSQL setting governing detection latency.
- **Serialization failures (40001) vs deadlocks (40P01)** — two distinct retryable error classes that robust apps handle together.
- **Consistent lock ordering / short transactions** — the two most important practical prevention techniques.
- **Livelock and retry backoff** — the failure mode of naive retry loops and its remedy.


---

# Database Normalization

## What is it?

Normalization is a systematic process of organizing the columns and tables of a relational database to minimize data redundancy and eliminate undesirable characteristics like insertion, update, and deletion anomalies. It was formalized by Edgar F. Codd (the father of the relational model) in the early 1970s, and it rests on a body of theory called **functional dependencies**.

At its core, normalization asks one deceptively simple question repeatedly: *"Does every non-key column in this table depend on the key, the whole key, and nothing but the key?"* If the answer is ever "no", the table is a candidate for further decomposition into smaller, well-behaved tables joined back together by keys.

A **functional dependency** (written `X → Y`) means: if you know the value of `X`, you can determine exactly one value of `Y`. For example, in a table of employees, `employee_id → employee_name` because a given employee ID maps to exactly one name. The whole edifice of normal forms is built on reasoning about which columns functionally determine which others.

The **Normal Forms** are a graduated ladder of increasingly strict conditions:

- **1NF** — atomic values, no repeating groups.
- **2NF** — 1NF plus no partial dependency on a composite key.
- **3NF** — 2NF plus no transitive dependency.
- **BCNF** — a stricter version of 3NF where every determinant is a candidate key.

There are higher forms (4NF for multivalued dependencies, 5NF for join dependencies, 6NF), but in practice the vast majority of production schemas target 3NF or BCNF, so those are the ones assessors care most about.

## Why is it needed?

Imagine a naive "one big table" design for a company's order system:

**`orders_flat` (unnormalized)**

| order_id | cust_name | cust_city | product      | unit_price | qty |
|----------|-----------|-----------|--------------|-----------|-----|
| 1        | Alice     | Boston    | Keyboard     | 40        | 2   |
| 1        | Alice     | Boston    | Mouse        | 25        | 1   |
| 2        | Bob       | Denver    | Keyboard     | 40        | 1   |

This design creates three classic **anomalies**:

1. **Update anomaly** — If Alice moves from Boston to Chicago, you must update *every* row where she appears. Miss one, and the database now holds two contradictory truths. Consistency has been lost.
2. **Insertion anomaly** — You cannot record a new product (say "Monitor", price 200) until someone actually orders it, because product data only exists inside order rows. The key (order_id, product) forces a phantom order.
3. **Deletion anomaly** — If Bob's only order (order 2) is deleted, you lose the fact that "Keyboard costs 40" if that were his only line — data that should be independent vanishes as a side-effect.

The deeper reason normalization matters is the **single source of truth** principle. Every fact should be stored exactly once. When a fact lives in exactly one place, updates are atomic and consistency is structurally guaranteed rather than hoped for. Redundancy is not just a storage-cost concern (though it is that too) — redundancy is the enemy of correctness, because redundant copies drift.

## How does it work?

You walk a table up the ladder of normal forms, decomposing it at each step. Let's take a single realistic example and carry it all the way through. Start with a university course-enrollment record kept "flat":

**Starting table — `enrollment_raw`**

| student_id | student_name | courses                        | instructor | instructor_office | dept   |
|-----------|--------------|--------------------------------|-----------|-------------------|--------|
| S1        | Alice        | Math101, Phys201               | Dr. Kaur  | B-12              | Science|
| S2        | Bob          | Math101                        | Dr. Kaur  | B-12              | Science|
| S3        | Carol        | Eng150                         | Dr. Lin   | A-05              | Arts   |

Assume the relevant functional dependencies (the "business rules"):

```
student_id                -> student_name
course_id                 -> instructor, dept
instructor                -> instructor_office
(student_id, course_id)   -> (nothing extra; it's the enrollment fact)
```

### Step 1 — First Normal Form (1NF): atomic values, no repeating groups

The `courses` column packs multiple values into one cell (`Math101, Phys201`). 1NF forbids this — every cell must hold a single atomic value, and there must be no repeating groups. We split the multi-valued cell into multiple rows:

**After 1NF — `enrollment_1nf`**

| student_id | student_name | course_id | instructor | instructor_office | dept    |
|-----------|--------------|-----------|-----------|-------------------|---------|
| S1        | Alice        | Math101   | Dr. Kaur  | B-12              | Science |
| S1        | Alice        | Phys201   | Dr. Sen   | C-09              | Science |
| S2        | Bob          | Math101   | Dr. Kaur  | B-12              | Science |
| S3        | Carol        | Eng150    | Dr. Lin   | A-05              | Arts    |

The primary key is now the composite `(student_id, course_id)` — that pair uniquely identifies each enrollment row.

### Step 2 — Second Normal Form (2NF): remove partial dependencies

2NF applies only when you have a **composite** key. A **partial dependency** exists when a non-key column depends on *only part* of the composite key rather than the whole key.

Look at `student_name`: it depends on `student_id` alone, not on `(student_id, course_id)`. That's partial. Likewise `instructor`, `dept`, and `instructor_office` depend on `course_id` alone. We split these out:

**`students` (student_id → student_name)**

| student_id | student_name |
|-----------|--------------|
| S1        | Alice        |
| S2        | Bob          |
| S3        | Carol        |

**`courses` (course_id → instructor, dept, instructor_office)**

| course_id | instructor | instructor_office | dept    |
|-----------|-----------|-------------------|---------|
| Math101   | Dr. Kaur  | B-12              | Science |
| Phys201   | Dr. Sen   | C-09              | Science |
| Eng150    | Dr. Lin   | A-05              | Arts    |

**`enrollments` (the pure many-to-many fact)**

| student_id | course_id |
|-----------|-----------|
| S1        | Math101   |
| S1        | Phys201   |
| S2        | Math101   |
| S3        | Eng150    |

Now every non-key attribute depends on the whole key of its table.

### Step 3 — Third Normal Form (3NF): remove transitive dependencies

A **transitive dependency** is when a non-key column depends on another non-key column instead of directly on the key. In `courses`, we have `course_id → instructor → instructor_office`. The office depends on the *instructor*, not directly on the course. That's transitive: `course_id` determines `instructor`, and `instructor` determines `instructor_office`.

We factor the instructor out:

**`courses` (course_id → instructor, dept)**

| course_id | instructor | dept    |
|-----------|-----------|---------|
| Math101   | Dr. Kaur  | Science |
| Phys201   | Dr. Sen   | Science |
| Eng150    | Dr. Lin   | Arts    |

**`instructors` (instructor → instructor_office)**

| instructor | instructor_office |
|-----------|-------------------|
| Dr. Kaur  | B-12              |
| Dr. Sen   | C-09              |
| Dr. Lin   | A-05              |

Now if Dr. Kaur changes office, we update exactly one row. The transitive dependency is gone.

### Step 4 — Boyce-Codd Normal Form (BCNF): every determinant is a candidate key

BCNF is a tightening of 3NF. The rule: **for every functional dependency `X → Y`, `X` must be a candidate key** (a superkey). 3NF permits a narrow exception BCNF removes — when a non-prime attribute is a determinant of a prime attribute.

The classic teaching example: suppose a rule that *each course in a given room is taught by exactly one instructor*, and *each instructor teaches in exactly one room*. Table `teaching(student, course, instructor)` with dependencies:

```
(student, course) -> instructor      (candidate key -> ok)
instructor        -> course          (instructor determines course, but instructor is NOT a candidate key)
```

This satisfies 3NF (because `course` is a prime attribute, part of a candidate key) but violates BCNF because `instructor` is a determinant yet not a candidate key. Decompose into:

**`instructor_course` (instructor → course)**

| instructor | course  |
|-----------|---------|
| Dr. Kaur  | Math101 |
| Dr. Sen   | Phys201 |

**`student_instructor` (student, instructor)**

| student | instructor |
|---------|-----------|
| S1      | Dr. Kaur  |
| S1      | Dr. Sen   |

Now every determinant is a key. Note the tradeoff: BCNF decomposition is not always dependency-preserving — sometimes you cannot enforce every original FD with simple key constraints after splitting, which is precisely why 3NF (which guarantees both lossless-join *and* dependency preservation) is often the practical stopping point.

Here is the final normalized schema expressed as PostgreSQL DDL:

```sql
CREATE TABLE students (
    student_id   TEXT PRIMARY KEY,
    student_name TEXT NOT NULL
);

CREATE TABLE instructors (
    instructor        TEXT PRIMARY KEY,
    instructor_office TEXT NOT NULL
);

CREATE TABLE courses (
    course_id  TEXT PRIMARY KEY,
    instructor TEXT NOT NULL REFERENCES instructors (instructor),
    dept       TEXT NOT NULL
);

CREATE TABLE enrollments (
    student_id TEXT NOT NULL REFERENCES students (student_id),
    course_id  TEXT NOT NULL REFERENCES courses  (course_id),
    PRIMARY KEY (student_id, course_id)
);
```

## Internal Working

Normalization is really an exercise in **functional-dependency algebra**, and understanding what happens "under the hood" clarifies why the rules are what they are.

Given a set of functional dependencies `F` over a set of attributes, you compute the **closure** `X+` of an attribute set `X` — the set of all attributes functionally determined by `X` using Armstrong's axioms (reflexivity, augmentation, transitivity). If `X+` equals all attributes, then `X` is a **superkey**. A minimal superkey is a **candidate key**. All of the normal-form definitions are phrased in terms of candidate keys and prime attributes (attributes that are part of *some* candidate key).

Decomposition must satisfy two properties to be "good":

1. **Lossless-join decomposition** — when you split table `R` into `R1` and `R2`, joining them back on their shared attributes must reproduce exactly the original rows, with no spurious extra rows. Formally, the decomposition is lossless if `(R1 ∩ R2) → R1` or `(R1 ∩ R2) → R2` holds. This is why we always split along a determinant: the shared column is a key in at least one of the resulting tables.
2. **Dependency preservation** — the union of FDs enforceable within each decomposed table should imply the original FD set, so you never need a join just to validate a constraint. 3NF can always be achieved with both properties; BCNF guarantees only lossless-join.

The query engine "pays back" normalization at read time via **joins**. A normalized read reconstructs the wide logical row by joining along foreign keys, using indexes on those keys (typically B-tree indexes). This is why normalization and indexing are inseparable topics: a normalized schema without indexes on its join columns performs poorly. The optimizer relies on primary-key/foreign-key metadata and index statistics to pick hash joins, merge joins, or nested-loop joins.

## Advantages

- **Eliminates redundancy** — each fact stored once, drastically reducing storage and, more importantly, the surface area for inconsistency.
- **Prevents update/insert/delete anomalies** — structural guarantees rather than application-level discipline.
- **Improves data integrity and consistency** — a single source of truth per fact.
- **Smaller, faster writes** — updating one row instead of thousands; write amplification drops.
- **Clearer schema semantics** — each table models exactly one entity or relationship, which makes the schema self-documenting and easier to evolve.
- **Better concurrency** — narrow, targeted updates lock fewer rows and reduce contention.

## Limitations

- **Read performance cost** — reconstructing a logical view may require many joins, which can be expensive for read-heavy analytical workloads.
- **Query complexity** — developers must write and reason about multi-table joins; a "simple" report can touch six tables.
- **Over-normalization** — pushing to 5NF/6NF everywhere yields diminishing returns and a schema that is painful to query.
- **Not aligned with analytical access patterns** — OLAP/data-warehouse workloads often deliberately denormalize into star/snowflake schemas for scan efficiency.
- **BCNF can lose dependency preservation** — sometimes you cannot enforce all business rules with local constraints after decomposition.

### Denormalization — and when it's appropriate

Denormalization is the *deliberate, informed* reintroduction of redundancy to optimize read performance. It is not a license to be sloppy; it is a conscious trade of write-cost and consistency-risk for read-speed. Appropriate when:

- **Read-heavy, write-light workloads** — e.g., reporting dashboards, product catalogs, where reads vastly outnumber writes.
- **Analytical/OLAP systems** — star and snowflake schemas denormalize dimensions for fast aggregation scans.
- **Expensive, repeated joins** — caching a computed `order_total` or a `customer_name` copy on the order avoids a hot join.
- **Aggregation/materialized views** — precomputed roll-ups (via `MATERIALIZED VIEW`) trade freshness for speed.

The golden rule: **normalize until it hurts, then denormalize until it works.** Always denormalize *from* a normalized model so you understand exactly which invariant you're relaxing, and add mechanisms (triggers, materialized views, application logic, or scheduled jobs) to keep the redundant copies in sync.

## Real-world Applications

- **OLTP systems** (banking, e-commerce checkout, inventory) target 3NF/BCNF because transactional correctness is paramount and writes are frequent.
- **Data warehouses** (analytics, BI) deliberately denormalize into star schemas — fact tables surrounded by denormalized dimension tables — for scan and aggregation speed.
- **SaaS multi-tenant apps** normalize core entities but often denormalize tenant-scoped read models.
- **Master Data Management** relies on strict normalization to maintain a single golden record per customer/product.
- **Event sourcing / CQRS** splits a normalized write model from denormalized read projections — a formal architectural embrace of both sides of this trade.

## Interview Questions

**Beginner**
1. What is database normalization and why do we do it?
2. What is a functional dependency? Give an example.
3. What does First Normal Form require?

**Intermediate**
4. Explain the difference between a partial dependency and a transitive dependency, and which normal form each addresses.
5. Walk me through normalizing a table from 1NF to 3NF.

**Advanced**
6. What is BCNF, and how does it differ from 3NF? Give a table that is in 3NF but not in BCNF.
7. What does it mean for a decomposition to be lossless-join and dependency-preserving? Can BCNF always guarantee both?

**Scenario-based**
8. You inherit a `customers` table where each row stores up to three phone numbers in `phone1`, `phone2`, `phone3`. What normalization problem is this and how do you fix it?

**"Why" questions**
9. Why does redundancy threaten correctness and not merely waste storage?

**Comparison questions**
10. Normalization vs denormalization — when would you choose each?

## Model Answers

**1. What is database normalization and why do we do it?**

Normalization is the process of structuring a relational schema so that each fact is stored exactly once, by decomposing tables according to their functional dependencies. We do it primarily to eliminate redundancy and the three anomalies redundancy causes: update anomalies (having to change the same fact in many places, risking inconsistency), insertion anomalies (being unable to record a fact because it has nowhere to live without other unrelated data), and deletion anomalies (losing a fact as an unintended side effect of deleting a row). The deeper motivation is the single-source-of-truth principle: when a fact lives in one place, consistency is structurally guaranteed rather than dependent on application discipline. We progress through a ladder of normal forms — 1NF, 2NF, 3NF, BCNF — each imposing a stricter condition, and in practice most transactional schemas target 3NF or BCNF.

**2. What is a functional dependency? Give an example.**

A functional dependency `X → Y` states that the value of attribute set `X` uniquely determines the value of attribute set `Y` — for any two rows with the same `X`, their `Y` values must be identical. It encodes a business rule about the data. For example, `employee_id → employee_name` means a given employee ID always maps to exactly one name; you cannot have two rows with the same ID but different names. Functional dependencies are the raw material of normalization: every normal form is defined in terms of which dependencies are permitted. A determinant (the left side, `X`) that determines every attribute in the table is a superkey, and the minimal such determinant is a candidate key.

**3. What does First Normal Form require?**

1NF requires that every column hold a single atomic value and that there are no repeating groups or arrays within a cell, and that each row is uniquely identifiable (a key exists). The classic violation is a comma-separated list like `"Math101, Phys201"` in one cell, or repeating columns like `phone1, phone2, phone3`. To reach 1NF you split multi-valued cells into separate rows (or separate related tables). The reason 1NF matters is that non-atomic values make the data un-queryable by the relational engine — you cannot efficiently `WHERE course_id = 'Phys201'` if courses are buried in a delimited string, and you cannot index or join on them properly.

**4. Explain the difference between a partial dependency and a transitive dependency, and which normal form each addresses.**

A partial dependency occurs when a non-key attribute depends on only *part* of a composite primary key rather than the whole key; 2NF removes these. For example, in `enrollments(student_id, course_id, student_name)` with key `(student_id, course_id)`, `student_name` depends only on `student_id` — a partial dependency — so we move it to a `students` table. A transitive dependency occurs when a non-key attribute depends on *another non-key* attribute, which in turn depends on the key; 3NF removes these. For example, `course_id → instructor → instructor_office`: the office depends on the instructor, not directly on the course. Both are forms of "the attribute doesn't depend directly and wholly on the key", and both are solved by factoring the offending attributes into their own table keyed by their true determinant.

**5. Walk me through normalizing a table from 1NF to 3NF.**

Start with a flat table, say `enrollment(student_id, student_name, course_id, instructor, instructor_office, dept)` with a comma-separated `courses` column. First, 1NF: split the multi-valued `courses` into one row per course, giving a composite key `(student_id, course_id)`. Second, 2NF: identify partial dependencies — `student_name` depends only on `student_id`, and `instructor/dept/instructor_office` depend only on `course_id`. Move each group into its own table (`students`, `courses`) and leave `enrollments(student_id, course_id)` as the pure relationship. Third, 3NF: within `courses`, notice `course_id → instructor → instructor_office` is transitive, so extract `instructors(instructor, instructor_office)`. The end result is four clean tables — `students`, `courses`, `instructors`, `enrollments` — each modeling exactly one thing, joined by foreign keys, with every fact stored once.

**6. What is BCNF, and how does it differ from 3NF? Give a table that is in 3NF but not in BCNF.**

BCNF (Boyce-Codd Normal Form) requires that for *every* non-trivial functional dependency `X → Y`, `X` must be a superkey. 3NF is slightly more permissive: it allows a dependency `X → Y` where `X` is not a superkey, provided `Y` is a prime attribute (part of some candidate key). The difference bites when a non-prime attribute determines a prime attribute. Classic example: `teaching(student, course, instructor)` where `(student, course) → instructor` and also `instructor → course` (each instructor teaches exactly one course). This is in 3NF because `course` is prime (part of the candidate key `(student, course)`), but it violates BCNF because `instructor` is a determinant yet not a candidate key. We decompose into `instructor_course(instructor, course)` and `student_instructor(student, instructor)`. The catch: this BCNF decomposition is not dependency-preserving — the FD `(student, course) → instructor` can no longer be enforced without a join — which is exactly why 3NF is often the practical target.

**7. What does it mean for a decomposition to be lossless-join and dependency-preserving? Can BCNF always guarantee both?**

A decomposition of table `R` into `R1` and `R2` is lossless-join if joining `R1` and `R2` on their common attributes reproduces exactly `R` — no rows lost, no spurious rows added. Formally this holds when the shared attributes form a key of at least one of the pieces: `(R1 ∩ R2) → R1` or `→ R2`. Dependency preservation means every original functional dependency can be enforced by checking constraints within individual decomposed tables, without needing a join. 3NF decomposition can always achieve *both* lossless-join and dependency preservation simultaneously. BCNF guarantees lossless-join but *not* always dependency preservation — some FDs may become un-enforceable locally after decomposition. This is the fundamental trade-off: BCNF gives stronger redundancy elimination but may cost you the ability to cheaply enforce a business rule, so engineers sometimes deliberately stop at 3NF.

**8. Scenario: `customers` table with `phone1`, `phone2`, `phone3`. What's the problem and fix?**

This is a repeating-group violation of 1NF disguised as fixed columns. The problems are numerous: it caps customers at three phones arbitrarily, wastes space with NULLs for customers with one phone, and makes queries like "find the customer who owns this number" require checking three columns with `OR`. The fix is to extract phones into a separate `customer_phones(customer_id, phone, phone_type)` table with a foreign key back to `customers`, one row per phone. This removes the arbitrary limit, eliminates the NULL columns, lets you index the phone column for fast lookups, and lets you attach metadata like `phone_type` ('mobile'/'work') cleanly. The `customers` table keeps only attributes that genuinely depend on `customer_id` alone.

**9. Why does redundancy threaten correctness and not merely waste storage?**

Because redundant copies drift. When the same fact — Alice's city, a product's price — is physically stored in many rows, every update must touch every copy atomically to keep them consistent. In practice some update path will eventually miss a copy: a bug, a partial transaction, a manual data fix, a race condition. The moment two copies disagree, the database contains a contradiction, and there is no principled way to know which copy is "true". Storage is cheap and getting cheaper; correctness is not. Normalization removes the *possibility* of contradiction by ensuring each fact has exactly one home, so an update is inherently atomic and consistency is a structural property of the schema rather than a runtime hope.

**10. Normalization vs denormalization — when would you choose each?**

Choose normalization when write correctness and integrity dominate — OLTP systems like banking, order processing, and inventory, where the same data changes frequently and anomalies are unacceptable. Normalization makes writes cheap and safe and guarantees consistency structurally. Choose denormalization when reads dominate and joins become the bottleneck — reporting dashboards, product catalogs, and especially OLAP/data-warehouse star schemas where you scan and aggregate huge volumes and rarely update. Denormalization trades write cost and consistency risk for read speed, so you must add machinery (triggers, materialized views, scheduled refreshes, or application logic) to keep the redundant copies in sync. The disciplined approach is to always design a normalized model first, then selectively denormalize the proven hot paths, knowing exactly which invariant you are relaxing and why.

## Common Mistakes

- **Confusing 2NF and 3NF triggers** — 2NF is about partial dependency on *part of a composite key*; 3NF is about transitive dependency via a *non-key* attribute. If the key is a single column, 2NF is automatically satisfied.
- **Storing CSV/JSON blobs to "avoid a join"** — reintroduces 1NF violations and makes the data un-queryable and un-indexable.
- **Normalizing blindly to the highest form** — 5NF everywhere is over-engineering; most systems want 3NF/BCNF.
- **Denormalizing without a sync strategy** — adding a redundant copy but no trigger/job to maintain it guarantees eventual inconsistency.
- **Forgetting indexes on foreign keys** — a normalized schema without join-column indexes performs terribly; PostgreSQL does *not* auto-index foreign keys.
- **Treating normalization as a storage optimization** — its primary purpose is correctness, not saving disk.
- **Ignoring dependency preservation when pushing to BCNF** — silently losing the ability to enforce a business rule.

## Related Concepts

- **Functional dependencies, candidate keys, superkeys, prime attributes** — the theoretical vocabulary.
- **Armstrong's axioms and attribute closure** — the algebra used to derive keys and verify normal forms.
- **Higher normal forms (4NF multivalued dependencies, 5NF join dependencies, 6NF)** — for completeness.
- **Star and snowflake schemas** — the canonical denormalized designs for data warehousing.
- **Materialized views** — the standard mechanism for managed, refreshable denormalization.
- **Indexing** — inseparable from normalization because joins need indexed keys.
- **ACID and transactions** — the guarantees that make normalized writes safe.

# Database Design Principles

## What is it?

Database design is the discipline of translating real-world information requirements into a well-structured relational schema — deciding what tables exist, what columns they hold, what data types those columns use, how tables reference one another, and what rules (constraints) the data must always obey. Where normalization gives you the *theory* of avoiding redundancy, database design is the *engineering craft* of applying that theory alongside a dozen other concerns: correctness, performance, evolvability, and human readability.

Good design rests on a handful of load-bearing concepts:

- **Primary keys** — the one column (or set of columns) that uniquely identifies each row. The choice between a **natural key** (a real-world identifier like an email or ISBN) and a **surrogate key** (a system-generated meaningless identifier like an auto-incrementing integer or UUID) is one of the most consequential decisions you make.
- **Foreign keys** — a column in one table that references the primary key of another, encoding a relationship and enforcing **referential integrity**.
- **Constraints** — declarative rules (`NOT NULL`, `UNIQUE`, `CHECK`, `DEFAULT`, primary/foreign key) that the database engine enforces on every write, so invalid data can never enter regardless of which application or human writes it.
- **Referential integrity actions** — what happens to child rows when a parent is deleted or updated: `CASCADE`, `RESTRICT`/`NO ACTION`, `SET NULL`, `SET DEFAULT`.
- **Indexing strategy** — deciding which columns get indexes to accelerate reads, balanced against the write and storage cost indexes impose.
- **Naming conventions** — a consistent, predictable vocabulary for tables, columns, keys, and constraints that makes the schema self-documenting.

## Why is it needed?

A schema is the single most permanent artifact in most systems. Application code gets rewritten, frameworks come and go, but the database schema — and the data inside it — often outlives all of them, sometimes by decades. A design mistake in code is a refactor; a design mistake in a schema is a migration over billions of live rows, usually under load, usually irreversible. This asymmetry is *the* reason database design deserves disproportionate care up front.

Concretely, good design buys you:

- **Data integrity by construction** — constraints make invalid states unrepresentable. If a rule ("an order must belong to a real customer", "a price cannot be negative") lives in the schema, no buggy service, no ad-hoc script, no midnight manual `UPDATE` can violate it. Push integrity as close to the data as possible, because the data outlives every application that touches it.
- **Predictable performance** — the right keys and indexes turn full-table scans into millisecond lookups.
- **Evolvability** — a clean schema with good keys and constraints can absorb new requirements; a tangled one calcifies.
- **Team velocity** — consistent naming and clear relationships let a new engineer read the schema and understand the domain without a tribal-knowledge briefing.

## How does it work?

Let's design a small e-commerce schema and make each decision explicit.

**Primary keys: natural vs surrogate.** Suppose a `customers` table. A natural key candidate is `email`. But emails change, they're large (bad for indexes and foreign-key copies), and reusing/merging accounts becomes painful. So we use a **surrogate** primary key and keep `email` as a `UNIQUE` natural key for lookups. This is the mainstream default: surrogate PK for identity, natural `UNIQUE` constraint for the real-world uniqueness rule.

```sql
CREATE TABLE customers (
    customer_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,  -- surrogate
    email       TEXT NOT NULL UNIQUE,                             -- natural key
    full_name   TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

**Foreign keys and referential integrity.** An order belongs to exactly one customer. The `REFERENCES` clause makes it impossible to insert an order for a non-existent customer, and the `ON DELETE` action decides what happens if the customer is removed:

```sql
CREATE TABLE orders (
    order_id    BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    customer_id BIGINT NOT NULL
                REFERENCES customers (customer_id)
                ON DELETE RESTRICT       -- cannot delete a customer with orders
                ON UPDATE CASCADE,
    status      TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'paid', 'shipped', 'cancelled')),
    total_cents INTEGER NOT NULL CHECK (total_cents >= 0),
    placed_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- PostgreSQL does NOT auto-index foreign keys; do it yourself.
CREATE INDEX idx_orders_customer_id ON orders (customer_id);
```

**The referential-integrity actions**, chosen per relationship by what the *business* means:

- `ON DELETE CASCADE` — deleting the parent deletes the children. Right for *owned* data: delete an order, its `order_items` should vanish. Dangerous for shared data — a cascade can silently wipe huge subtrees.
- `ON DELETE RESTRICT` / `NO ACTION` — refuse to delete the parent while children exist. The safe default for financially or legally significant links (don't delete a customer who has orders). (`RESTRICT` checks immediately; `NO ACTION` can be deferred to end of transaction.)
- `ON DELETE SET NULL` — orphan the children by nulling the FK. Right when the relationship is optional: delete a `sales_rep`, and their customers keep existing with `sales_rep_id = NULL`. Requires the FK column to be nullable.
- `ON DELETE SET DEFAULT` — set the FK to a default value (e.g., a sentinel "unassigned" row).

For *owned* child data — order line items — cascade is exactly right:

```sql
CREATE TABLE order_items (
    order_id   BIGINT NOT NULL
               REFERENCES orders (order_id) ON DELETE CASCADE,
    product_id BIGINT NOT NULL
               REFERENCES products (product_id) ON DELETE RESTRICT,
    quantity   INTEGER NOT NULL CHECK (quantity > 0),
    unit_cents INTEGER NOT NULL CHECK (unit_cents >= 0),
    PRIMARY KEY (order_id, product_id)   -- composite natural key here is fine
);
```

Notice the deliberate mix: an item's link to its *order* cascades (items are owned by the order), but its link to a *product* restricts (you must not delete a product that historical orders reference).

**Indexing strategy.** Index the columns you filter, join, and sort on — foreign keys, and columns in `WHERE`/`ORDER BY`. Every index speeds reads but slows writes (each `INSERT`/`UPDATE` must maintain it) and consumes storage, so index deliberately, not reflexively. Use composite indexes for multi-column predicates (order matters — leftmost-prefix rule), partial indexes for filtered subsets, and unique indexes to enforce natural-key uniqueness.

```sql
-- Composite index for a common query: a customer's recent orders
CREATE INDEX idx_orders_customer_placed ON orders (customer_id, placed_at DESC);

-- Partial index: only index the small set of unshipped orders
CREATE INDEX idx_orders_pending ON orders (placed_at)
    WHERE status = 'pending';
```

**Naming conventions.** Pick one convention and never deviate: `snake_case` identifiers, plural table names (`customers`, `orders`), singular column names, `<table>_id` for primary/foreign keys, `idx_<table>_<cols>` for indexes, `fk_/chk_/uq_` prefixes for named constraints. Consistency is the whole point: when every foreign key is `<referenced_table_singular>_id`, an engineer can infer relationships without reading DDL.

## Internal Working

Under the hood, these design choices map onto concrete engine mechanics:

- **Primary key** is implemented as a `UNIQUE NOT NULL` constraint backed by a **B-tree index**. In PostgreSQL the table is a heap and the PK is just a unique index pointing into it (unlike, say, InnoDB where the PK *is* the clustered storage order). This is why surrogate integer keys are attractive: small, monotonic integers make dense, cache-friendly B-trees and cheap foreign-key copies, whereas random UUIDv4 keys scatter inserts across the index and hurt cache locality (UUIDv7/time-ordered UUIDs mitigate this).
- **Foreign key** enforcement means that on every child `INSERT`/`UPDATE`, the engine performs a lookup against the parent's unique index to confirm the referenced row exists; and on every parent `DELETE`/`UPDATE`, it must find referencing children — which is precisely why an *un-indexed* foreign key makes parent deletes catastrophically slow (a full scan of the child table per delete). The referential action (`CASCADE`, `RESTRICT`, `SET NULL`) is executed as a system-generated trigger fired within the same transaction, preserving atomicity.
- **CHECK constraints** are evaluated inline on each write; they're cheap boolean expressions and never require I/O.
- **Indexes** are most commonly B-trees (ordered, good for equality and range), with specialized types (hash, GiST, GIN for full-text/JSONB/arrays, BRIN for huge naturally-ordered tables). The query planner consults index statistics (`ANALYZE`) to decide whether using an index is cheaper than a sequential scan.
- **Deferred constraints** — foreign keys declared `DEFERRABLE INITIALLY DEFERRED` are checked at `COMMIT` rather than per-statement, which enables inserting mutually-referencing rows in any order within a transaction.

## Advantages

- **Integrity guaranteed at the lowest layer** — the database rejects invalid data regardless of application bugs; correctness doesn't depend on every writer behaving.
- **Self-documenting schema** — good keys, foreign keys, and naming let the schema explain the domain.
- **Performance through targeted indexing** — millisecond lookups instead of scans.
- **Safe cascading semantics** — owned data is cleaned up automatically; critical links are protected.
- **Evolvability** — clean relationships and surrogate keys make refactors and merges tractable.
- **Cross-application consistency** — multiple services sharing a database all inherit the same enforced rules.

## Limitations

- **Constraints add write overhead** — every FK check and index maintenance costs time on the hot write path.
- **Schema rigidity** — strongly typed, constrained schemas resist ad-hoc change; migrations on large tables are operationally hard.
- **Over-indexing degrades writes** — each redundant index is pure write-time and storage tax.
- **Cascade footguns** — a misjudged `ON DELETE CASCADE` can silently destroy large data subtrees.
- **Natural-key volatility** — choosing a natural key that later changes forces painful updates propagated through every foreign key.
- **Surrogate keys hide duplicates** — a surrogate PK will happily let two rows that are semantically identical coexist unless you also add the right `UNIQUE` natural-key constraint.

## Real-world Applications

- **E-commerce / ERP** — customers, orders, items, products with carefully chosen cascade vs restrict semantics for financial correctness.
- **Multi-tenant SaaS** — a `tenant_id` foreign key on every table, indexed first in composite indexes, enforcing tenant isolation.
- **Financial systems** — heavy use of `RESTRICT` and `CHECK` constraints; deletes are often forbidden entirely in favor of soft-delete flags to preserve audit trails.
- **Content platforms** — `ON DELETE CASCADE` for owned content (delete a user, delete their draft posts) alongside `SET NULL` for authored-but-shared artifacts.
- **Reference-data management** — `RESTRICT` on lookup tables (currencies, countries) so you can't delete a code still in use.

## Interview Questions

**Beginner**
1. What is a primary key, and what is a foreign key?
2. What is referential integrity?
3. Name three types of constraints and what each enforces.

**Intermediate**
4. Explain the difference between a natural key and a surrogate key, and when you'd pick each.
5. Walk through the `ON DELETE` options and give a real scenario for each.

**Advanced**
6. Why does an unindexed foreign key hurt performance, and on which operation specifically?
7. How does the choice of primary key (auto-increment integer vs UUIDv4 vs UUIDv7) affect index and write performance?

**Scenario-based**
8. You must delete a customer who has 10 years of orders, but finance requires that historical orders never disappear. How do you design for this?

**"Why" questions**
9. Why should integrity rules live in the database rather than only in application code?

**Comparison questions**
10. `ON DELETE CASCADE` vs `ON DELETE RESTRICT` vs `ON DELETE SET NULL` — compare and contrast.

## Model Answers

**1. What is a primary key, and what is a foreign key?**

A primary key is the column or set of columns that uniquely and non-nullably identifies every row in a table — no two rows may share it, and it may never be NULL. It's the row's identity, and the engine backs it with a unique index. A foreign key is a column in one table whose values must match an existing primary-key (or unique) value in another table; it encodes a relationship ("this order belongs to that customer") and enforces referential integrity, meaning the database refuses to store a foreign-key value that points to a nonexistent parent. Together they are the mechanism by which normalized tables, decomposed to avoid redundancy, are reconnected into a coherent whole at query time via joins.

**2. What is referential integrity?**

Referential integrity is the guarantee that every foreign-key value actually references an existing row in the parent table — there are no "dangling pointers" in the data. If `orders.customer_id` is a foreign key into `customers`, referential integrity ensures you can never have an order pointing at a customer that doesn't exist, and (depending on the configured action) you can't delete a customer out from under their orders without a defined consequence. The database enforces this on every relevant write within the transaction, so it holds even under concurrency and application bugs. It's the structural property that makes a relational schema trustworthy: relationships expressed by foreign keys are always valid.

**3. Name three types of constraints and what each enforces.**

`NOT NULL` enforces that a column must always have a value, encoding the rule "this fact is mandatory". `UNIQUE` enforces that no two rows share a value (or combination of values) in the constrained column(s), encoding real-world uniqueness like one account per email; it's backed by a unique index. `CHECK` enforces an arbitrary boolean predicate on each row, like `total_cents >= 0` or `status IN (...)`, encoding domain rules the type system alone can't express. Beyond these, `PRIMARY KEY` combines `UNIQUE` + `NOT NULL` for identity, `FOREIGN KEY` enforces referential integrity, and `DEFAULT` supplies a value when none is given. The common thread: constraints move rules from fallible application code into the engine, where they're enforced universally.

**4. Explain the difference between a natural key and a surrogate key, and when you'd pick each.**

A natural key is an identifier that has real-world meaning and already uniquely identifies the entity — an email, an ISBN, a country code, a Social Security Number. A surrogate key is a system-generated, semantically meaningless identifier — an auto-incrementing integer or a UUID — whose only job is to be unique and stable. You pick a surrogate as the primary key in the common case because natural keys tend to change (people change email), can be large (bad for the foreign-key copies scattered across child tables and their indexes), and sometimes turn out not to be as unique or permanent as assumed. The best-practice pattern is both: a surrogate `id` as the primary key for stable identity and cheap joins, plus a `UNIQUE` constraint on the natural key to still enforce the real-world uniqueness rule. Natural keys shine as the primary key mainly for stable, standardized reference data (an ISO currency code) or in junction tables where the composite of two foreign keys is naturally the key.

**5. Walk through the `ON DELETE` options and give a real scenario for each.**

`ON DELETE CASCADE` deletes child rows when the parent is deleted — correct for owned data, like deleting an `order` and having its `order_items` automatically removed, since items have no meaning without their order. `ON DELETE RESTRICT` (and the similar `NO ACTION`) refuses to delete a parent that still has children — correct for protected relationships, like refusing to delete a `customer` who has `orders`, or a `product` still referenced by historical line items; it forces the application to deal with the children first. `ON DELETE SET NULL` nulls the child's foreign key, orphaning it while keeping it alive — correct for optional relationships, like deleting a `sales_rep` and leaving their customers assigned to nobody (`sales_rep_id = NULL`); this requires the FK column to be nullable. `ON DELETE SET DEFAULT` sets the FK to a predefined default, such as reassigning to an "unassigned" sentinel row. The choice is never technical alone — it's a direct encoding of what the deletion *means* in the business domain.

**6. Why does an unindexed foreign key hurt performance, and on which operation specifically?**

The pain shows up on operations against the *parent* table — deletes and primary-key updates — not on the child inserts most people expect. When you delete or update a parent row, the engine must find every child row that references it to apply the referential action (or to verify none exist for `RESTRICT`). Without an index on the child's foreign-key column, that search is a full sequential scan of the entire child table, performed once per affected parent row. On a large child table this turns a routine parent delete into a table-scan-per-row disaster, and it can also cause lock contention. This is a notorious gotcha because PostgreSQL automatically indexes the parent's primary key but does *not* automatically index the child's foreign-key column — you must create that index yourself, and it should be near-reflexive for any FK you'll delete or join across.

**7. How does the choice of primary key affect index and write performance?**

Auto-incrementing integers are monotonic and small: new rows append to the "right edge" of the B-tree, keeping it dense and cache-friendly, and the small key size keeps both the index and every foreign-key copy compact. Random UUIDv4 keys are the opposite — each insert lands at a random position in the index, causing page splits scattered across the structure, poor cache locality, and index bloat, plus they're 16 bytes and propagate that bulk to every referencing table. This is why high-write systems historically favored bigints. UUIDv7 (and other time-ordered UUID schemes) restore locality by prefixing the value with a timestamp, so inserts are again roughly monotonic while retaining the UUID advantages of client-side generation and non-guessability. The trade: integers leak scale and are guessable/enumerable (a problem for public identifiers); UUIDs are globally unique and safe to generate anywhere and expose, at some storage and, for v4, write-locality cost. A common pattern is a bigint surrogate PK for internal joins plus a separate UUID/public token for external exposure.

**8. Scenario: delete a customer with 10 years of orders that must never disappear.**

The core insight is that "delete the customer" and "destroy the orders" are two different requirements, and the design must separate them. I would not use a hard `DELETE` at all here; I'd use a **soft delete** — a `deleted_at TIMESTAMPTZ` (or `is_active BOOLEAN`) column on `customers` — so "deleting" a customer sets that flag, hiding them from active queries while preserving the row and thus keeping every order's foreign key valid and its history intact. The `orders.customer_id` foreign key would be `ON DELETE RESTRICT` precisely so that an accidental hard delete is impossible. Active-customer queries filter `WHERE deleted_at IS NULL` (ideally through a view or partial index so it's not forgotten). This satisfies finance (orders and their customer linkage are permanent and auditable), satisfies operations (the customer disappears from day-to-day use), and keeps referential integrity intact. If regulation later demands actual PII erasure, you anonymize the customer row in place (null out name/email) rather than deleting it, again preserving the order relationships.

**9. Why should integrity rules live in the database rather than only in application code?**

Because the database is the single point through which all data must pass, and it long outlives any particular application. Rules in application code are enforced only by that code — but data is typically touched by multiple services, background jobs, admin scripts, data migrations, and the occasional human running manual SQL at 2 a.m. during an incident. Any of those paths can bypass application-layer validation and write garbage. A `CHECK`, `NOT NULL`, `UNIQUE`, or `FOREIGN KEY` constraint, by contrast, is enforced by the engine on *every* write from *every* source, atomically and under concurrency, with no way around it. This is the "defense in depth, closest to the data" principle: validate in the app for good UX and fast feedback, but make the invariants that must *never* be violated structural properties of the schema, because that's the only layer guaranteed to be in the path of every mutation for the entire life of the data.

**10. `ON DELETE CASCADE` vs `RESTRICT` vs `SET NULL` — compare and contrast.**

All three define what happens to child rows when their parent is deleted, and the right choice is a statement about ownership and business meaning. `CASCADE` says the children are *owned* by and meaningless without the parent, so destroy them together (order to order-items). It's convenient but dangerous — a single delete can silently cascade through a large subtree. `RESTRICT` says the relationship is *protective*: the parent may not be deleted while children reference it, forcing an explicit decision (don't delete a customer with orders, or a product referenced by history). It's the safest default for significant links. `SET NULL` says the relationship is *optional*: the child survives independently with its link severed (delete a sales rep, keep the customers unassigned) — it requires a nullable FK and is wrong when the relationship is mandatory. In short: `CASCADE` for owned data, `RESTRICT` to protect critical references, `SET NULL` for optional associations. Choosing among them is domain modeling, not a technical toss-up, and getting it wrong either loses data (cascade) or blocks legitimate operations (restrict).

## Common Mistakes

- **Not indexing foreign keys** — the number-one cause of mysteriously slow parent deletes and joins in PostgreSQL, which does not auto-create these indexes.
- **Blanket `ON DELETE CASCADE`** — convenient until it silently deletes a subtree you didn't intend; use it only for genuinely owned data.
- **Choosing a volatile natural key as the primary key** — when it changes, the update ripples through every foreign key that copied it.
- **Relying solely on application code for integrity** — leaves the door open to every non-app write path.
- **Over-indexing** — adding an index per column "just in case", taxing every write and wasting storage.
- **Inconsistent naming** — mixing `userId`, `user_id`, and `uid` across tables destroys the schema's self-documenting quality.
- **Random UUIDv4 as a high-write clustered/primary key** — index fragmentation and cache-thrash; prefer bigint or UUIDv7.
- **Storing money as floats** — always use integer minor units or `NUMERIC`; floats introduce rounding errors in financial data.

## Related Concepts

- **Normalization** — the theory that dictates which tables and keys should exist in the first place.
- **Indexing (B-tree, hash, GiST, GIN, BRIN, partial, composite)** — the performance half of design.
- **ACID transactions and isolation levels** — the guarantees within which constraints are enforced.
- **Surrogate vs natural key debate; UUIDv4 vs UUIDv7** — key-selection nuance.
- **Soft deletes and audit trails** — patterns for preserving history under deletion requirements.
- **Deferrable constraints** — for handling circular references within a transaction.
- **Schema migrations (online DDL, `pg_repack`)** — the operational reality of evolving a live schema.

---

# Entity Relationship (ER) Modeling

## What is it?

Entity-Relationship modeling is a conceptual, technology-agnostic way of describing the *things* in a domain and the *relationships* between them, before you commit to any particular table layout. Introduced by Peter Chen in 1976, it sits one level of abstraction above the relational schema: you draw an **ER diagram** that captures *what the business is about* — customers, products, orders, the fact that customers place orders — and only then mechanically translate that diagram into concrete tables. It's the blueprint you draw before you pour the concrete.

The vocabulary of ER modeling:

- **Entity** — a distinct thing the business cares about and wants to store data about: a `Customer`, a `Product`, an `Order`. An **entity set** is the collection of all such things; an **entity instance** is one specific member. A **weak entity** is one that cannot be identified by its own attributes alone and depends on another (owner) entity — e.g., an `OrderLine` only makes sense in the context of its `Order`.
- **Attribute** — a property of an entity. Attributes come in flavors:
  - **Simple (atomic)** — indivisible, like `age`.
  - **Composite** — decomposable into sub-parts, like `address` → `{street, city, zip}`.
  - **Derived** — computable from other data rather than stored, like `age` derived from `birth_date`.
  - **Multivalued** — can hold several values for one entity, like a person's multiple `phone_numbers`.
  - **Key attribute** — uniquely identifies an instance, like `customer_id`.
- **Relationship** — an association between entities, like `Customer *places* Order`. Relationships have a **degree** (binary, ternary) and a **cardinality**.
- **Cardinality** — how many instances of one entity relate to how many of another: **1:1** (one-to-one), **1:N** (one-to-many), and **M:N** (many-to-many).

## Why is it needed?

ER modeling exists to separate *thinking about the domain* from *thinking about the implementation*. If you jump straight to `CREATE TABLE`, you conflate two hard problems — understanding the business and encoding it efficiently — and you tend to get both wrong. By modeling entities and relationships first, in a notation a non-technical stakeholder can read, you can validate your understanding of the domain ("wait, can an order have more than one customer?") *before* it's expensive to change.

The concrete payoffs:

- **Shared language with stakeholders** — an ER diagram is understandable by product managers and domain experts, so requirements bugs are caught at whiteboard cost, not production-migration cost.
- **A disciplined path to a normalized schema** — a well-drawn ER model translates, by a mechanical set of rules, into tables that are already close to 3NF. The modeling forces you to identify each entity's key and each relationship's cardinality, which is most of what normalization needs.
- **Correct handling of many-to-many** — the single most common beginner schema error is trying to cram a many-to-many relationship into two tables; ER modeling makes the need for a **junction table** obvious and explicit.
- **Documentation that outlives the code** — the diagram is a durable map of the system.

## How does it work?

You interview the domain, extract entities (usually the nouns), attach attributes, connect entities with relationships, and annotate each relationship with its cardinality. Consider a bookstore. The nouns are `Author`, `Book`, `Customer`, `Order`. The rules:

- An author writes many books, and (to keep it interesting) a book can have many authors → **M:N** between Author and Book.
- A customer places many orders, but an order belongs to exactly one customer → **1:N** between Customer and Order.
- An order contains many books, and a book appears in many orders → **M:N** between Order and Book, and that relationship *itself* carries data (quantity, price at time of sale).

Here is the ER diagram in Chen-style ASCII, using `[Entity]`, `(attribute)`, and `<Relationship>` with cardinality on the connecting lines:

```
                 (name)        (title)         (price)
                    |             |               |
                    |             |               |
   ______________   |    M     ______    N     ___|________
  |   AUTHOR     |--( )------< WRITES >------( )--|   BOOK   |
  |______________|          \________/           |__________|
    |        |                                      |     |
 (author_id) (bio)                            (book_id)  (isbn)
                                                     |
                                                     | M
                                                 ____v____
                                                <CONTAINS >   --- attributes: (quantity)(unit_price)
                                                 \_______/
                                                     | N
   ________________       1        __________        |
  |   CUSTOMER     |-----------< PLACES >-----( )-----| 
  |________________|    1        \________/    N    __v_______
    |          |                                   |  ORDER   |
(customer_id)(email)                               |__________|
                                                    |        |
                                                (order_id)(placed_at)

Cardinality legend:  1 --- exactly one     N/M --- many
Relationship <WRITES> is M:N   |   <PLACES> is 1:N   |   <CONTAINS> is M:N with attributes
```

### Converting the ER diagram to relational tables

There is a deterministic recipe:

1. **Each strong entity becomes a table**, with its key attribute as the primary key. Composite attributes are flattened into their sub-columns; multivalued attributes are pulled out into their own child table; derived attributes are typically *not* stored (computed on read).
2. **A 1:N relationship** is represented by putting a **foreign key on the "many" side** pointing at the "one" side. No new table needed. (`Order` gets a `customer_id` FK.)
3. **A 1:1 relationship** puts a foreign key (with a `UNIQUE` constraint) on either side — usually the optional side.
4. **An M:N relationship becomes its own table** — a **junction (associative) table** — whose primary key is the composite of the two foreign keys, plus any attributes the relationship itself carries.

Applying the recipe to the bookstore:

```sql
-- Strong entities become tables
CREATE TABLE authors (
    author_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name      TEXT NOT NULL,
    bio       TEXT
);

CREATE TABLE books (
    book_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    isbn    TEXT NOT NULL UNIQUE,
    title   TEXT NOT NULL,
    price   NUMERIC(10,2) NOT NULL CHECK (price >= 0)
);

CREATE TABLE customers (
    customer_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    email       TEXT NOT NULL UNIQUE
);

-- 1:N (Customer places Order): FK on the "many" side (orders)
CREATE TABLE orders (
    order_id    BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    customer_id BIGINT NOT NULL REFERENCES customers (customer_id) ON DELETE RESTRICT,
    placed_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_orders_customer_id ON orders (customer_id);

-- M:N (Author writes Book): junction table, composite PK of the two FKs
CREATE TABLE book_authors (
    book_id   BIGINT NOT NULL REFERENCES books   (book_id)   ON DELETE CASCADE,
    author_id BIGINT NOT NULL REFERENCES authors (author_id) ON DELETE CASCADE,
    PRIMARY KEY (book_id, author_id)
);
CREATE INDEX idx_book_authors_author ON book_authors (author_id);

-- M:N WITH attributes (Order contains Book): junction table carrying relationship data
CREATE TABLE order_items (
    order_id   BIGINT NOT NULL REFERENCES orders (order_id) ON DELETE CASCADE,
    book_id    BIGINT NOT NULL REFERENCES books  (book_id)  ON DELETE RESTRICT,
    quantity   INTEGER NOT NULL CHECK (quantity > 0),
    unit_price NUMERIC(10,2) NOT NULL CHECK (unit_price >= 0),  -- price at sale time
    PRIMARY KEY (order_id, book_id)
);
CREATE INDEX idx_order_items_book ON order_items (book_id);

-- Multivalued attribute (a customer's many phone numbers) becomes its own table
CREATE TABLE customer_phones (
    customer_id BIGINT NOT NULL REFERENCES customers (customer_id) ON DELETE CASCADE,
    phone       TEXT NOT NULL,
    phone_type  TEXT,
    PRIMARY KEY (customer_id, phone)
);
```

Note the pedagogically important choice of storing `unit_price` on `order_items`: the relationship carries the price *at the time of sale*, which must not change when the book's catalog price later changes. That's a relationship attribute, and the junction table is exactly where it belongs.

## Internal Working

ER modeling is a *conceptual* layer, so its "internal working" is really the theory of the mapping to the *logical* (relational) and then *physical* layers — the classic three-schema architecture (conceptual → logical → physical).

- **Entities map to relations (tables); attributes map to columns; entity instances map to rows.** The key attribute becomes the primary key, implemented as a unique B-tree index as discussed under design principles.
- **A binary M:N relationship is provably impossible to represent in two tables without redundancy** — you'd have to repeat one side's rows for each related instance of the other, reintroducing the very anomalies normalization forbids. The junction table is the *only* faithful representation, and it's simply the relational encoding of the relationship set itself: each row is one (left, right) pair that actually exists. Its composite primary key `(left_id, right_id)` structurally forbids duplicate pairings.
- **Cardinality constraints** map onto keys and nullability: 1:N puts a (possibly `NOT NULL`) FK on the many side; a mandatory "one" side means `NOT NULL` on that FK; 1:1 is a FK plus a `UNIQUE` constraint; participation constraints (total vs partial — must every customer have at least one order?) map onto `NOT NULL` and sometimes onto deferred checks or application logic, since SQL can't natively enforce "every parent must have at least one child".
- **Weak entities** map to tables whose primary key includes the owner's key (a composite key), and whose FK to the owner uses `ON DELETE CASCADE`, reflecting existence dependence.
- **Higher-degree relationships** (ternary) generalize to junction tables with three foreign keys — though ternary relationships are often better decomposed into multiple binary ones after analysis.

## Advantages

- **Bridges business and technical worlds** — a diagram stakeholders and engineers can both read, catching requirement errors early.
- **Produces near-normalized schemas mechanically** — following the mapping rules yields tables that are typically already in 3NF.
- **Makes many-to-many explicit** — the model forces recognition of junction tables, preventing the most common design error.
- **Implementation-independent** — the same conceptual model can target PostgreSQL, MySQL, or even a document store, deferring technology commitment.
- **Durable documentation** — a living map of the domain that aids onboarding and future changes.
- **Cleanly handles relationship attributes** — data that belongs to a relationship (sale price, enrollment date) has an obvious home.

## Limitations

- **No standard for behavior or constraints beyond structure** — ER captures data shape well but not complex business rules, temporal logic, or process.
- **Notation fragmentation** — Chen, Crow's Foot, UML, IDEF1X all differ, causing communication friction across teams and tools.
- **Ternary and higher relationships are hard to model and read** — they often need refinement into binary relationships.
- **Cardinality can be ambiguous at the whiteboard** — "a customer has an order" hides whether it's mandatory, and minimum-cardinality/participation is easy to omit.
- **Doesn't dictate physical performance choices** — indexing, partitioning, and denormalization are outside the conceptual model and must be layered on later.
- **Poor fit for non-relational paradigms** — graph, hierarchical, and heavily document-oriented domains don't map cleanly to entity/relationship/table.

## Real-world Applications

- **Greenfield application design** — the first artifact produced before writing any DDL, agreed with product before build.
- **Database documentation and onboarding** — reverse-engineered ER diagrams (via tools like dbdiagram, pgModeler, or DataGrip) explain a legacy schema to new engineers.
- **Requirements workshops** — a shared canvas for analysts and domain experts to validate what data the system must hold.
- **Data warehouse dimensional modeling** — ER thinking underlies identifying facts and dimensions before building star schemas.
- **API and microservice boundaries** — entities and their relationships inform how to partition data ownership across services.

## Interview Questions

**Beginner**
1. What is an entity, and what is an attribute? Give examples of the attribute types.
2. What is cardinality in an ER model? Name the three types.

**Intermediate**
3. How do you convert a 1:N relationship into relational tables? Where does the foreign key go and why?
4. What is a composite attribute and a multivalued attribute, and how does each map to tables?

**Advanced**
5. Why can't a many-to-many relationship be represented with just the two entity tables, and what is the correct representation?
6. What is a weak entity, and how does it map to a relational schema?

**Scenario-based**
7. Model a university where students enroll in courses and each enrollment has a grade. Show the tables.

**"Why" questions**
8. Why model conceptually with an ER diagram before writing `CREATE TABLE` statements at all?

**Comparison questions**
9. Compare 1:1, 1:N, and M:N relationships in terms of how each is implemented in a relational schema.

## Model Answers

**1. What is an entity, and what is an attribute? Give examples of the attribute types.**

An entity is a distinct, identifiable thing in the domain that the business wants to store data about — a `Customer`, a `Book`, an `Order`. The set of all such things is the entity set, and it typically becomes a table. An attribute is a property of an entity. Attributes come in several types: a *simple/atomic* attribute is indivisible, like `age`; a *composite* attribute decomposes into sub-parts, like `address` splitting into `street`, `city`, and `zip`; a *derived* attribute is computed rather than stored, like `age` derived from `birth_date`; a *multivalued* attribute can hold several values for one instance, like a customer's multiple phone numbers; and a *key* attribute uniquely identifies an instance, like `customer_id`. Recognizing the attribute type matters because it dictates the mapping: composites flatten into multiple columns, multivalued attributes become their own child table, and derived attributes are usually not stored at all.

**2. What is cardinality in an ER model? Name the three types.**

Cardinality describes how many instances of one entity can be associated with instances of another through a relationship. The three types are one-to-one (1:1), where each instance on each side relates to at most one on the other — for example, each employee has one assigned parking spot and vice versa; one-to-many (1:N), where one instance on the "one" side relates to many on the "many" side but not the reverse — one customer places many orders, but each order belongs to exactly one customer; and many-to-many (M:N), where instances on both sides can relate to many on the other — a book can have many authors and an author can write many books. Cardinality is the single most important annotation on a relationship because it directly determines the table mapping: 1:N places a foreign key on the many side, while M:N requires an entirely separate junction table.

**3. How do you convert a 1:N relationship into relational tables? Where does the foreign key go and why?**

You keep the two entities as two tables and place the foreign key on the "many" side, pointing at the primary key of the "one" side. For `Customer places Order` (one customer, many orders), the `orders` table gets a `customer_id` column referencing `customers(customer_id)`. The reason the FK goes on the many side is that each row there relates to exactly *one* row on the other side, so a single FK column can hold that one reference without any repetition. If you tried to put the reference on the "one" side instead, you'd need to store many order references in a single customer row — which either violates 1NF (a list in a cell) or forces repeating columns, both of which are exactly the redundancy we're avoiding. You also decide the FK's nullability from the participation constraint (mandatory relationship → `NOT NULL`) and choose an `ON DELETE` action from the business meaning, then index the FK for join and delete performance.

**4. What is a composite attribute and a multivalued attribute, and how does each map to tables?**

A composite attribute is one that naturally decomposes into smaller sub-attributes — `address` breaking into `street`, `city`, `state`, and `zip`. It maps by flattening: you store each sub-part as its own column (`street`, `city`, ...) in the entity's table, which keeps the data atomic and queryable (you can filter by city). A multivalued attribute can hold several values for a single entity instance — a customer with three phone numbers, or a product with several tags. It cannot be flattened into one row without violating 1NF (you'd need a list in a cell or arbitrary repeating columns like `phone1/phone2/phone3`), so it maps to a *separate child table* with a foreign key back to the owning entity, one row per value — `customer_phones(customer_id, phone)`. This removes the arbitrary cap on how many values are allowed, eliminates NULL padding, and makes each value independently indexable and queryable.

**5. Why can't a many-to-many relationship be represented with just the two entity tables, and what is the correct representation?**

Because neither side can hold the reference to the other without repetition. In `Author writes Book`, if you put a `book_id` on the author table, an author writing five books needs five book references in one row — impossible atomically, forcing either a list (1NF violation) or duplicated author rows. Put an `author_id` on the book table and you have the symmetric problem for co-authored books. Duplicating rows to make it fit reintroduces exactly the update, insertion, and deletion anomalies normalization exists to prevent. The correct and only faithful representation is a third table — a *junction* (or associative/bridge) table — whose rows are the actual (author, book) pairs that exist, with a composite primary key `(author_id, book_id)` of the two foreign keys. That composite key structurally forbids duplicate pairings, and crucially, the junction table is also the natural home for any attribute that belongs to the *relationship itself* rather than to either entity — like the quantity and sale price in an `order_items` table linking orders and products.

**6. What is a weak entity, and how does it map to a relational schema?**

A weak entity is one that cannot be uniquely identified by its own attributes alone; it depends on a stronger "owner" entity for its identity and existence. The classic example is an `OrderLine` (or `OrderItem`): a line number like "line 1" is meaningless on its own — "line 1 *of order 5000*" is what identifies it. It maps to a table whose primary key *includes* the owner's primary key, forming a composite key (`order_id` plus a discriminator, or the composite of two FKs as in `order_items(order_id, book_id)`). Its foreign key to the owner is typically declared `ON DELETE CASCADE`, because a weak entity's existence depends on its owner — delete the order, and its lines must go too, since they have no independent meaning. This existence dependence is the defining trait that distinguishes a weak entity from an ordinary entity that merely happens to have a foreign key.

**7. Scenario: university students enroll in courses, each enrollment has a grade. Show the tables.**

This is a textbook M:N relationship with a relationship attribute. A student enrolls in many courses and a course has many students, so the relationship is many-to-many — and the `grade` belongs to the enrollment (the pairing), not to the student or the course alone, so it lives on the junction table:

```sql
CREATE TABLE students (
    student_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    full_name  TEXT NOT NULL
);

CREATE TABLE courses (
    course_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    title     TEXT NOT NULL
);

CREATE TABLE enrollments (
    student_id  BIGINT NOT NULL REFERENCES students (student_id) ON DELETE CASCADE,
    course_id   BIGINT NOT NULL REFERENCES courses  (course_id) ON DELETE RESTRICT,
    grade       TEXT CHECK (grade IN ('A','B','C','D','F') OR grade IS NULL),
    enrolled_at DATE NOT NULL DEFAULT CURRENT_DATE,
    PRIMARY KEY (student_id, course_id)
);
CREATE INDEX idx_enrollments_course ON enrollments (course_id);
```

The composite primary key `(student_id, course_id)` enforces that a student can't enroll in the same course twice, `grade` is nullable because it doesn't exist until the course is completed, and I index `course_id` so "list everyone in this course" is fast (the PK's leftmost column already covers "list this student's courses").

**8. Why model conceptually with an ER diagram before writing `CREATE TABLE` at all?**

Because jumping straight to DDL forces you to solve two hard problems at once — understanding the business domain and encoding it efficiently in tables — and you tend to botch both. An ER diagram lets you reason about the domain in a notation that non-technical stakeholders can read and validate, so you catch fundamental misunderstandings ("can an order actually have two customers?", "must every book have an author?") at whiteboard cost rather than after you've built tables, written application code against them, and loaded production data. It also gives you a disciplined, mechanical path to a good schema: once entities, keys, and relationship cardinalities are pinned down, the translation to normalized tables is almost automatic, and cardinality analysis surfaces the junction tables that many-to-many relationships require — the exact thing people get wrong when they improvise DDL. Finally, the diagram is durable documentation that outlives the code and speeds onboarding. In short, ER modeling front-loads the cheap-to-fix decisions before they become expensive-to-fix schema migrations.

**9. Compare 1:1, 1:N, and M:N relationships in terms of implementation.**

All three are implemented with foreign keys, but the arrangement differs. A 1:1 relationship is implemented by placing a foreign key on one side (usually the optional side) with a `UNIQUE` constraint on it, so that at most one row on each side can be paired — for example, `user_profiles(user_id UNIQUE REFERENCES users)`; sometimes 1:1 entities are even merged into one table if they're always accessed together. A 1:N relationship places a plain (non-unique) foreign key on the "many" side pointing at the "one" side — `orders.customer_id` — because each many-side row references exactly one one-side row; no extra table is needed. An M:N relationship cannot be done with foreign keys on the entity tables at all; it requires a separate junction table whose composite primary key is the two foreign keys, and which optionally carries relationship attributes — `book_authors(book_id, author_id)`. So the progression is: 1:1 is a FK plus `UNIQUE`, 1:N is a bare FK on the many side, and M:N is a whole new table. Recognizing which one you have — driven entirely by the cardinality you determined during modeling — is what tells you the correct physical structure.

## Common Mistakes

- **Modeling M:N with foreign keys on the entity tables** — the classic error; it forces list-in-a-cell or duplicated rows. Always use a junction table.
- **Putting the 1:N foreign key on the "one" side** — you can't store many child references in one parent row without violating 1NF.
- **Storing multivalued attributes as CSV or repeating columns** (`phone1/phone2/phone3`) — violates 1NF; extract into a child table.
- **Storing derived attributes and letting them drift** — a stored `age` or `order_total` that isn't maintained goes stale; compute on read or maintain deliberately.
- **Forgetting relationship attributes** — putting `grade` or sale-time `unit_price` on an entity instead of the junction table, causing wrong data when the entity's own values change.
- **Ignoring participation/minimum cardinality** — capturing "one-to-many" but not whether the relationship is mandatory, losing `NOT NULL` information.
- **Overusing ternary relationships** — often clearer and more correct as multiple binary relationships.

## Related Concepts

- **Normalization** — ER modeling and normalization are two routes to the same well-structured schema; a good ER model lands near 3NF automatically.
- **Junction / associative / bridge tables** — the relational encoding of M:N relationships.
- **Crow's Foot, Chen, UML, and IDEF1X notations** — alternative ER diagram styles.
- **Cardinality and participation (total vs partial) constraints** — the precise semantics of relationships.
- **Weak entities and identifying relationships** — existence-dependent modeling.
- **Dimensional modeling (facts and dimensions, star schemas)** — the analytics-oriented descendant of ER thinking.
- **Relational algebra and joins** — how the decomposed entities are recombined at query time.

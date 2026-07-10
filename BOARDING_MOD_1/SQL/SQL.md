# SQL Revision Roadmap

## Focus Areas
- SQL Fundamentals
- CRUD Operations
- Queries
- Joins
- Aggregations
- Window Functions
- Subqueries
- Indexing
- Transactions
- Database Design
- Normalization
- Performance Optimization

---

# Week 1: SQL Fundamentals

## Introduction to SQL

### Topics
- What is SQL?
- Importance of SQL in data manipulation and retrieval
- SQL syntax and structure
- Setting up a SQL environment
  - PostgreSQL
  - MySQL

---

## Database Fundamentals

### Concepts
- Database
- Table
- Row
- Column
- Primary Key
- Foreign Key
- Constraints
- Schema

---

## SQL Data Types

### Numeric
- INTEGER
- BIGINT
- DECIMAL
- FLOAT

### String
- CHAR
- VARCHAR
- TEXT

### Date & Time
- DATE
- TIME
- TIMESTAMP

### Boolean
- BOOLEAN

---

## CRUD Operations

### CREATE

#### Topics
- INSERT statement
- Insert single row
- Insert multiple rows

---

### READ

#### SELECT Statements
- Select all columns
- Select specific columns
- DISTINCT
- Aliases (`AS`)

#### Filtering
- WHERE
- Comparison operators
- Logical operators (`AND`, `OR`, `NOT`)
- BETWEEN
- IN
- LIKE
- ILIKE (PostgreSQL)
- IS NULL
- IS NOT NULL

#### Sorting
- ORDER BY
- ASC
- DESC

#### Limiting Results
- LIMIT
- OFFSET

---

### UPDATE

#### Topics
- UPDATE statement
- Updating multiple columns
- Conditional updates

---

### DELETE

#### Topics
- DELETE statement
- Conditional delete
- DELETE vs TRUNCATE vs DROP

---

# SQL Functions

## String Functions

- CONCAT()
- LENGTH()
- LOWER()
- UPPER()
- TRIM()
- SUBSTRING()
- REPLACE()

---

## Numeric Functions

- ROUND()
- CEIL()
- FLOOR()
- ABS()

---

## Date Functions

- CURRENT_DATE
- CURRENT_TIMESTAMP
- AGE()
- EXTRACT()

---

## NULL Handling

- COALESCE()
- NULLIF()

---

# Week 2: Querying & Data Analysis

## Aggregate Functions

### Topics
- COUNT()
- SUM()
- AVG()
- MIN()
- MAX()

---

## GROUP BY

### Topics
- Grouping records
- Aggregate calculations

---

## HAVING

### Topics
- Filtering grouped results
- HAVING vs WHERE

---

## Joins

### Introduction
Learn how multiple tables are related using keys.

### Types of Joins

#### INNER JOIN
Returns matching records.

#### LEFT JOIN
Returns all rows from the left table.

#### RIGHT JOIN
Returns all rows from the right table.

#### FULL OUTER JOIN
Returns all matching and non-matching rows.

#### SELF JOIN
Joining a table with itself.

#### CROSS JOIN
Cartesian product of two tables.

---

## Subqueries

### Topics
- Introduction to subqueries
- Single-row subqueries
- Multiple-row subqueries
- Nested subqueries
- Correlated subqueries

---

## Common Table Expressions (CTEs)

### Topics
- WITH clause
- Recursive CTEs
- Improving query readability

---

# Week 3: Advanced SQL

## Window Functions

### Introduction
Learn how window functions perform calculations across a set of rows without grouping them.

### Ranking Functions
- ROW_NUMBER()
- RANK()
- DENSE_RANK()
- NTILE()

### Analytical Functions
- LEAD()
- LAG()
- FIRST_VALUE()
- LAST_VALUE()

### Aggregate Window Functions
- SUM() OVER()
- AVG() OVER()
- COUNT() OVER()

### Window Clauses
- PARTITION BY
- ORDER BY
- Frame clauses (`ROWS BETWEEN ...`)

---

## Views

### Topics
- Creating Views
- Updating Views
- Materialized Views (PostgreSQL)

---

## Indexing

### Topics
- What is an Index?
- Why indexing is important
- Index creation
- Managing indexes
- Query optimization using indexes

### Types of Indexes
- B-Tree Index
- Hash Index
- Composite Index
- Unique Index
- Partial Index

---

## Stored Procedures & Functions

### Stored Procedures
- Introduction
- Creating procedures
- Executing procedures

### Functions
- User-defined functions
- Returning scalar values
- Returning tables

---

# Week 4: Database Design & Transactions

## Transactions

### ACID Properties

- Atomicity
- Consistency
- Isolation
- Durability

---

## Transaction Commands

- BEGIN
- COMMIT
- ROLLBACK
- SAVEPOINT

---

## Concurrency Control

### Topics
- Locking mechanisms
- Isolation levels
- Shared locks
- Exclusive locks

### Common Problems
- Dirty Reads
- Non-repeatable Reads
- Phantom Reads

---

## Deadlocks

### Topics
- Identifying deadlocks
- Preventing deadlocks
- Resolving deadlocks

---

## Database Normalization

### Importance
- Reduce redundancy
- Improve consistency

### Normal Forms
- First Normal Form (1NF)
- Second Normal Form (2NF)
- Third Normal Form (3NF)
- Boyce-Codd Normal Form (BCNF)

### Practice
- Normalize sample databases
- Identify normalization issues

---

## Database Design Principles

### Topics
- Designing efficient databases
- Choosing primary keys
- Foreign key relationships
- Constraints
- Referential integrity

---

## Entity Relationship (ER) Modeling

### Topics
- Entities
- Attributes
- Relationships
- Cardinality

### Practice
- Draw ER diagrams
- Convert ER diagrams into relational tables

---

# Practical Applications

## SQL Assignments

### CRUD Operations
- Practice INSERT
- Practice SELECT
- Practice UPDATE
- Practice DELETE

### Query Writing
- Multi-table joins
- Nested subqueries
- Aggregate queries
- Window functions

### Hands-on Practice
- Use sample databases
- Write real-world SQL queries

---

## Real-World Case Study

### Tasks
- Analyze business problems
- Write optimized SQL queries
- Extract business insights
- Present findings

---

# Topics to Revise

## SQL Fundamentals
- SQL Syntax
- Data Types
- CRUD Operations
- Filtering
- Sorting

---

## Querying
- SELECT
- WHERE
- ORDER BY
- LIMIT
- DISTINCT

---

## Aggregation
- COUNT
- SUM
- AVG
- MIN
- MAX
- GROUP BY
- HAVING

---

## Joins
- INNER JOIN
- LEFT JOIN
- RIGHT JOIN
- FULL OUTER JOIN
- SELF JOIN
- CROSS JOIN

---

## Subqueries
- Single-row
- Multiple-row
- Nested
- Correlated
- CTEs

---

## Window Functions
- ROW_NUMBER
- RANK
- DENSE_RANK
- LEAD
- LAG
- FIRST_VALUE
- LAST_VALUE
- PARTITION BY

---

## Performance Optimization
- Indexing
- Query Optimization
- Execution Plans (`EXPLAIN`, `EXPLAIN ANALYZE`)

---

## Transactions
- ACID
- BEGIN
- COMMIT
- ROLLBACK
- SAVEPOINT
- Isolation Levels
- Locking
- Deadlocks

---

## Database Design
- ER Modeling
- Normalization
- Constraints
- Relationships

---

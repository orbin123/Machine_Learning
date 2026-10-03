## Question 1 — Student Course Management System (SQL)

### 1. Aggregate Functions + GROUP BY + HAVING

### 1. Number of students enrolled in each course

```sql
SELECT 
    c.title,
    COUNT(e.student_id) AS total_students
FROM courses c
JOIN enrollments e
ON c.course_id = e.course_id
GROUP BY c.course_id, c.title;
```

---

### 2. Average price of all courses

```sql
SELECT 
    AVG(price) AS average_course_price
FROM courses;
```

---

### 3. Courses having enrollment count greater than 2

```sql
SELECT 
    c.title,
    COUNT(e.student_id) AS enrollment_count
FROM courses c
JOIN enrollments e
ON c.course_id = e.course_id
GROUP BY c.course_id, c.title
HAVING COUNT(e.student_id) > 2;
```

---

# 2. Joins + Subqueries

### 4. Students with their enrolled courses using INNER JOIN

```sql
SELECT 
    s.name AS student_name,
    c.title AS course_name,
    e.enrolled_on
FROM students s
INNER JOIN enrollments e
ON s.student_id = e.student_id
INNER JOIN courses c
ON e.course_id = c.course_id;
```

---

### 5. All courses with enrollment count including zero enrollments

```sql
SELECT
    c.title,
    COUNT(e.student_id) AS enrollment_count
FROM courses c
LEFT JOIN enrollments e
ON c.course_id = e.course_id
GROUP BY c.course_id, c.title;
```

---

### 6(a). Students enrolled in the most expensive course

```sql
SELECT 
    s.name,
    c.title,
    c.price
FROM students s
JOIN enrollments e
ON s.student_id = e.student_id
JOIN courses c
ON e.course_id = c.course_id
WHERE c.price = (
    SELECT MAX(price)
    FROM courses
);
```

---

### 6(b). Increase price of courses below average price by 10%

```sql
UPDATE courses
SET price = price * 1.10
WHERE price < (
    SELECT AVG(price)
    FROM courses
);
```

---

# 3. Advanced SQL

### 7. CROSS JOIN — All student-course combinations

```sql
SELECT
    s.name AS student_name,
    c.title AS course_name
FROM students s
CROSS JOIN courses c;
```

---

### 8. SELF JOIN — Students from the same country

```sql
SELECT
    s1.name AS student1,
    s2.name AS student2,
    s1.country
FROM students s1
JOIN students s2
ON s1.country = s2.country
AND s1.student_id < s2.student_id;
```

---

### 9(a). UNION

```sql
SELECT name, country
FROM students
WHERE country = 'India'

UNION

SELECT name, country
FROM students
WHERE country = 'USA';
```

---

### 9(b). UNION ALL

```sql
SELECT name, country
FROM students
WHERE country = 'India'

UNION ALL

SELECT name, country
FROM students
WHERE country = 'USA';
```

---

# 4. Database Optimization and Constraints

### 10. Create index on students(country)

```sql
CREATE INDEX idx_student_country
ON students(country);
```

---

### 11. Add unique constraint on course titles

```sql
ALTER TABLE courses
ADD CONSTRAINT unique_course_title
UNIQUE(title);
```

---

### 12. Foreign Key Constraints

```sql
ALTER TABLE enrollments
ADD CONSTRAINT fk_student
FOREIGN KEY(student_id)
REFERENCES students(student_id);


ALTER TABLE enrollments
ADD CONSTRAINT fk_course
FOREIGN KEY(course_id)
REFERENCES courses(course_id);
```

---

# 5. Transactions + Views

### 13-15. Transaction with rollback

```sql
START TRANSACTION;

UPDATE courses
SET price = 100
WHERE course_id = 101;

ROLLBACK;
```

---

### 16. Update and save using commit

```sql
BEGIN TRANSACTION;

UPDATE courses
SET price = 120
WHERE course_id = 101;

COMMIT;
```

---

### 17. Create View

```sql
CREATE VIEW student_course_details AS

SELECT
    s.name AS student_name,
    c.title AS course_title,
    e.enrolled_on AS enrollment_date

FROM students s

JOIN enrollments e
ON s.student_id = e.student_id

JOIN courses c
ON e.course_id = c.course_id;
```

To view:

```sql
SELECT * 
FROM student_course_details;
```

---

### Topics Covered:
- GROUP BY + HAVING
- Aggregate functions
- INNER JOIN / LEFT JOIN
- Subqueries
- CROSS JOIN
- SELF JOIN
- UNION / UNION ALL
- Indexing
- Constraints
- Transactions (COMMIT / ROLLBACK)
- Views

(These topics align with the SQL section of your roadmap. )

#### Q2: Stored procedure for updating salary

create or replace procedure update_salary(
  emp_id int,
  new_salary decimal
)
language plpgsql
as $$
begin
update employees
set salary = new_salary
where employee_id = emp_id;
end;
$$;

#### Q3: Function for getiing employee salary

create or replace function get_employee_salry(
  emp_id int
)
returns decimal
language plpgsql
as $$
declare
emp_salary decimal;
begin
select salary
into emp_salary
from employees
where employee_id = emp_id;
return emp_salary;
end;
$$;
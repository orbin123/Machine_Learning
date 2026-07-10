# Python — Theory & Interview Preparation Guide

> A first-principles study guide covering the entire Python revision syllabus (Fundamentals → OOP → Exception Handling → Advanced Python).
> Written for technical interviews, viva, written assessments, coding rounds, and practical lab exams.
>
> **How to use this document:** Read each topic top-to-bottom once for understanding, then revisit the *Interview Questions*, *Model Answers*, and *Common Mistakes* sections the night before an assessment. Every "why" here matters more than memorized definitions — interviewers probe reasoning.

## Table of Contents

**Week 1 — Fundamentals**
1. [Python Basics & Syntax](#1-python-basics--syntax)
2. [Data Types & Type Conversion](#2-data-types--type-conversion)
3. [Operators](#3-operators)
4. [Control Flow](#4-control-flow)
5. [Functions](#5-functions)
6. [Lists](#6-lists)
7. [Tuples](#7-tuples)
8. [Dictionaries](#8-dictionaries)
9. [Sets](#9-sets)
10. [Strings & Regular Expressions](#10-strings--regular-expressions)
11. [File Handling](#11-file-handling)
12. [Modules & Packages](#12-modules--packages)

**Week 2 — Object-Oriented Programming**
13. [OOP Fundamentals](#13-oop-fundamentals)
14. [Encapsulation](#14-encapsulation)
15. [Abstraction](#15-abstraction)
16. [Inheritance](#16-inheritance)
17. [Polymorphism](#17-polymorphism)
18. [Magic / Dunder Methods](#18-magic--dunder-methods)
19. [Class Relationships (Composition & Aggregation)](#19-class-relationships-composition--aggregation)
20. [Properties](#20-properties)
21. [Dataclasses](#21-dataclasses)

**Week 3 — Exception Handling**
22. [Exception Handling](#22-exception-handling)
23. [Raising & Custom Exceptions](#23-raising--custom-exceptions)
24. [Assertions](#24-assertions)
25. [Logging](#25-logging)

**Week 4 — Advanced Python**
26. [Iterators](#26-iterators)
27. [Generators](#27-generators)
28. [Closures & Decorators](#28-closures--decorators)
29. [Context Managers](#29-context-managers)
30. [Functional Programming](#30-functional-programming)
31. [Comprehensions](#31-comprehensions)
32. [Type Hints](#32-type-hints)
33. [Memory Management](#33-memory-management)
34. [Concurrency](#34-concurrency)

---

# 1. Python Basics & Syntax

## What is it?

Python is a **high-level, interpreted, dynamically-typed, garbage-collected** programming language. "Syntax" is simply the set of rules that decide whether the text you write is a valid Python program. The defining syntactic feature of Python is that **indentation (whitespace) is part of the grammar** — it replaces the curly braces `{}` used by C, Java, and JavaScript to mark blocks of code.

The building blocks you must be fluent in are: **variables** (names bound to objects), **comments** (notes ignored by the interpreter), **indentation** (block structure), **keywords** (reserved words), **input/output** (talking to the user/terminal), and **dynamic typing** (types are attached to values, not to names).

## Why is it needed?

Every language needs a grammar so that both humans and the interpreter agree on what a program means. Python's designers made a deliberate bet: **code is read far more often than it is written**, so the syntax should optimize for readability. Enforced indentation means every Python program from every developer looks structurally similar — you cannot write "ugly" nesting the way you can in C. This reduces the cognitive cost of reading other people's code, which is where most engineering time actually goes.

Dynamic typing exists to make the language fast to write and flexible: you don't declare types up front, so prototyping and scripting are quick.

## How does it work?

When you run `python script.py`:

```
Source code (script.py)
        │  1. Lexing / tokenizing
        ▼
Tokens  (NAME, NUMBER, INDENT, NEWLINE ...)
        │  2. Parsing
        ▼
AST (Abstract Syntax Tree)
        │  3. Compilation
        ▼
Bytecode (.pyc, stored in __pycache__)
        │  4. Execution
        ▼
CPython Virtual Machine (evaluation loop)
```

Key syntactic rules:

- **Statements** usually end at a newline (no semicolons needed; `;` is legal but discouraged).
- **Blocks** are introduced by a colon `:` and defined by consistent indentation (PEP 8 recommends **4 spaces**, never tabs mixed with spaces).
- **Comments** start with `#` and run to end of line. There is no dedicated multi-line comment; triple-quoted strings `"""..."""` are string literals often *used* as docstrings.
- **Keywords** (e.g. `if`, `for`, `def`, `class`, `return`, `None`, `True`, `False`, `and`, `or`, `not`, `import`, `lambda`, `yield`, `with`, `async`, `await`) are reserved and cannot be used as identifiers.

Input/output:

```python
name = input("Enter your name: ")   # input() ALWAYS returns a str
print("Hello,", name)               # print separates args with a space by default
print("a", "b", sep="-", end="!\n") # -> a-b!
```

## Internal Working

- **Dynamic typing:** In Python a variable is just a **name in a namespace (a dict) bound to an object**. `x = 5` creates an `int` object `5` and binds the name `x` to it. Reassigning `x = "hi"` rebinds `x` to a `str` object — the name has no fixed type. The *object* carries the type; you can inspect it with `type(x)`.
- **Everything is an object.** Integers, functions, classes, and modules are all first-class objects with attributes and a type.
- **Indentation** is converted by the tokenizer into synthetic `INDENT` and `DEDENT` tokens, which the parser uses exactly like braces. That is why inconsistent indentation raises `IndentationError` / `TabError`.
- **Name binding vs mutation:** `x = x + 1` does not mutate the integer `5`; it creates a new integer `6` and rebinds `x`. Integers are immutable.

## Advantages

- **Readability** — enforced structure means uniform-looking code.
- **Fast to write** — no type declarations, no compile step visible to the user.
- **Batteries included** — huge standard library.
- **Portable** — same code runs on Windows/macOS/Linux.

## Limitations

- **Runtime type errors** — because types aren't checked ahead of time, a `TypeError` may only surface when a specific line runs.
- **Whitespace sensitivity** — copy-pasting code from web pages can silently mix tabs and spaces.
- **Speed** — interpreted and dynamically typed, so raw CPU-bound loops are slower than C/Java.

## Real-world Applications

Every Python program relies on these fundamentals, but they're most visible in **scripting and automation** (DevOps scripts, data pipelines), **teaching** (Python is the world's most common first language largely because of its clean syntax), and **rapid prototyping** at companies like Instagram, Dropbox, and Spotify where quick iteration matters.

## Interview Questions

- **Beginner:** What does `input()` return and why is that a common source of bugs?
- **Beginner:** How does Python know where a block of code ends?
- **Intermediate:** Explain "dynamic typing." Is Python strongly or weakly typed?
- **Intermediate:** What is the difference between a statement and an expression?
- **Advanced:** Walk me through what happens between typing `python app.py` and the code executing.
- **"Why":** Why did Python's designers choose significant whitespace over braces?
- **Comparison:** How is Python's typing different from Java's?
- **Scenario:** Your teammate's file raises `TabError`. What happened and how do you prevent it team-wide?

## Model Answers

**Q: What does `input()` return and why is that a bug source?**
`input()` **always returns a string**, even if the user types digits. Beginners write `age = input("age: ")` then do `age + 1` expecting arithmetic and get a `TypeError`, or worse compare `"10" < "9"` (string comparison, which is `True` lexicographically). The fix is explicit conversion: `age = int(input("age: "))`, ideally wrapped in a `try/except ValueError` for invalid input.

**Q: Is Python strongly or weakly typed?**
Python is **strongly typed but dynamically typed**. *Dynamically* typed means the type is checked at runtime and names can be rebound to any type. *Strongly* typed means Python does not silently coerce unrelated types: `"5" + 5` raises `TypeError` rather than guessing (unlike JavaScript, which is weakly typed and returns `"55"`). So the two axes are independent: dynamic ≠ weak.

**Q: How does Python know where a block ends?**
By indentation. A colon `:` starts a block and every line indented one level deeper belongs to it; the first line dedented back closes it. Internally the tokenizer emits `INDENT`/`DEDENT` tokens that the parser treats like braces. Consistency matters — mixing tabs and spaces triggers `TabError`.

**Q (Why): Why significant whitespace?**
Because indentation already communicates structure to human readers, and in brace languages the indentation and the braces can *disagree* (misleading code). By making indentation authoritative, Python removes that class of bug and forces uniform formatting, which lowers the cost of reading unfamiliar code — the dominant activity in real engineering.

## Common Mistakes

- Forgetting `input()` returns `str`.
- Mixing tabs and spaces (configure your editor to insert 4 spaces).
- Using `=` (assignment) where `==` (comparison) is meant.
- Believing a variable "is" an `int`; it's a name bound to an int *object*.
- Using triple-quoted strings as "comments" — they're actually evaluated string objects (fine as docstrings, wasteful elsewhere).

## Related Concepts

[Data Types](#2-data-types--type-conversion), [Memory Management](#33-memory-management) (name binding), [PEP 8 naming](#12-modules--packages), [Type Hints](#32-type-hints).

# 2. Data Types & Type Conversion

## What is it?

A **data type** is the classification that tells Python (a) what kind of value an object holds and (b) what operations are valid on it. The **primitive/built-in scalar types** are:

| Type | Example | Mutable? | Notes |
|------|---------|----------|-------|
| `int` | `42`, `-7`, `10**100` | No | Arbitrary precision — no overflow |
| `float` | `3.14`, `2.0`, `1e-9` | No | 64-bit IEEE-754 double |
| `bool` | `True`, `False` | No | Subclass of `int` (`True == 1`) |
| `str` | `"hello"` | No | Unicode sequence of characters |
| `NoneType` | `None` | No | The single "absence of value" object |

**Type conversion (casting)** is turning a value of one type into another, either **implicit** (Python does it automatically) or **explicit** (you call `int()`, `float()`, `str()`, `bool()`, etc.).

## Why is it needed?

Types give meaning to bits. The same bytes could be an integer, a float, or text — the type decides how they're interpreted and what `+` means (numeric addition vs string concatenation). Conversion is needed because data crosses boundaries: `input()` gives you `str`, a CSV file gives you `str`, an API gives you `str`, but your logic needs `int`/`float`. Without conversion you cannot compute on external data.

## How does it work?

**Implicit conversion (coercion)** happens only between compatible numeric types to avoid data loss:

```python
result = 3 + 4.0     # int is promoted to float -> 7.0 (float)
flag   = True + 1    # bool promoted to int -> 2
```

Python will **never** implicitly convert `str` ↔ number — that would be error-prone, so it raises `TypeError`.

**Explicit conversion** uses constructor functions:

```python
int("42")        # 42
int("42", 2)     # 2  -> parse "42"? No: base 2 -> ValueError; int("101", 2) -> 5
int(3.99)        # 3   -> truncates toward zero, does NOT round
float("3.14")    # 3.14
str(42)          # "42"
bool(0)          # False;  bool("")-> False;  bool([]) -> False
list("abc")      # ['a','b','c']
```

**Truthiness** — every object has a boolean value. Falsy values are: `False`, `None`, `0`, `0.0`, `""`, `[]`, `{}`, `()`, `set()`. Everything else is truthy.

## Internal Working

- `int` is **arbitrary precision**: CPython stores big integers as an array of 30-bit "digits", so `2**1000` just works — no overflow like in C.
- `float` is a C `double` (64-bit). This is why `0.1 + 0.2 == 0.30000000000000004` — decimals can't always be represented exactly in binary. Use `math.isclose()` or the `decimal` module for money.
- `bool` **subclasses `int`**, so `True + True == 2` and `isinstance(True, int)` is `True`. This is an intentional design choice.
- `str` is immutable and stored as Unicode code points; CPython optimizes storage (Latin-1, UCS-2, or UCS-4) depending on the characters present.
- **Small-integer caching:** CPython pre-creates integer objects from `-5` to `256`, so `a = 100; b = 100; a is b` is `True` — but this is an implementation detail, never rely on it. Use `==` for value equality.

## Advantages

- Arbitrary-precision `int` removes a whole class of overflow bugs.
- A single unified `None` makes "no value" explicit and checkable with `is None`.
- Truthiness makes conditionals concise (`if items:` instead of `if len(items) > 0:`).

## Limitations

- Float imprecision surprises beginners (money math must use `decimal`).
- Implicit `bool`-is-`int` can hide bugs (`True + True` silently equals `2`).
- Silent truncation in `int(3.99) == 3` (not rounding) trips people up.

## Real-world Applications

Type conversion is everywhere data enters a system: parsing web-form fields, reading config files, deserializing JSON (where numbers vs strings matter), and ETL/data-cleaning pipelines that coerce messy columns into consistent types before analysis.

## Interview Questions

- **Beginner:** Name the immutable built-in types in Python.
- **Beginner:** What does `int("3.5")` return?
- **Intermediate:** Why is `0.1 + 0.2 != 0.3`? How do you handle money?
- **Intermediate:** What values are "falsy" in Python?
- **Advanced:** `a = 256; b = 256; a is b` is often `True` but `a = 257; b = 257; a is b` may be `False`. Explain.
- **"Why":** Why is `bool` a subclass of `int`?
- **Comparison:** Difference between `is` and `==`?
- **Scenario:** A user submits `"007"` as an ID; you store it as `int` and it becomes `7`, breaking a lookup. What went wrong and how do you fix it?

## Model Answers

**Q: Why is `0.1 + 0.2 != 0.3`?**
Floats are IEEE-754 doubles stored in binary. `0.1` and `0.2` have no exact finite binary representation, so they're stored as the nearest representable value; their sum rounds to `0.30000000000000004`. For exact decimal arithmetic (currency), use `decimal.Decimal("0.1") + decimal.Decimal("0.2")`, or for tolerance comparisons use `math.isclose(a, b)`. Never compare floats with `==`.

**Q: `int("3.5")`?**
It raises `ValueError`. `int()` parsing a *string* expects an integer literal; `"3.5"` isn't one. To get `3` you must go through float first: `int(float("3.5"))`. Note `int(3.5)` (a float, not a string) returns `3` by truncation.

**Q: `a is b` for 256 vs 257?**
`is` compares **object identity** (same memory), `==` compares **value**. CPython caches small integers `-5..256` as singletons, so both names point to the same cached `256` object → `is` is `True`. `257` is outside the cache, so two literals may create two distinct objects → `is` can be `False`. The lesson: use `==` for equality; `is` only for singletons like `None`.

**Q (Why): Why is `bool` a subclass of `int`?**
Historically Python had no `bool` type; comparisons returned `1`/`0`. When `bool` was added (2.3), backward compatibility required `True`/`False` to still behave as `1`/`0` in arithmetic and indexing. Subclassing `int` preserved that. It's occasionally useful (`sum(x > 0 for x in data)` counts positives) but can mask bugs.

**Q (Scenario): `"007"` → `7`:**
Converting an identifier that has semantic leading zeros to `int` destroys information. IDs, phone numbers, and ZIP codes are *identifiers*, not quantities — keep them as `str`. Rule of thumb: only convert to a number if you'll do arithmetic on it.

## Common Mistakes

- Comparing floats with `==`.
- Using `is` for value comparison (works by luck for small ints, fails for big ones).
- Assuming `int()` rounds — it truncates.
- Storing IDs/ZIP codes as `int` and losing leading zeros.
- Forgetting empty containers are falsy, causing `if data is not None` vs `if data` confusion.

## Related Concepts

[Operators](#3-operators), [Strings](#10-strings--regular-expressions), [Memory Management](#33-memory-management) (mutability, identity), [Type Hints](#32-type-hints).

---

# 3. Operators

## What is it?

Operators are symbols that perform operations on operands (values). Python groups them into: **arithmetic** (`+ - * / // % **`), **comparison/relational** (`== != > < >= <=`), **logical** (`and or not`), **assignment** (`= += -= *= /=` …), **membership** (`in`, `not in`), **identity** (`is`, `is not`), plus bitwise (`& | ^ ~ << >>`).

## Why is it needed?

Operators are concise, readable notation for the most common computations. Writing `a + b` is clearer than a method call `a.__add__(b)` — but crucially, that's exactly what Python does under the hood, which is why the same `+` works on ints, floats, strings, and lists.

## How does it work?

```python
7 / 2     # 3.5   true division -> always float
7 // 2    # 3     floor division -> rounds toward -infinity
-7 // 2   # -4    (NOT -3!) floor rounds down
7 % 3     # 1     modulo/remainder
2 ** 10   # 1024  exponent

# Logical operators SHORT-CIRCUIT and return an OPERAND, not a bool:
0 or "default"    # "default"  (returns first truthy, or last)
"a" and "b"       # "b"        (returns first falsy, or last)
x = user_input or "guest"   # common idiom for defaults

# Membership & identity:
3 in [1, 2, 3]       # True
"ell" in "hello"     # True (substring)
x is None            # identity check

# Chained comparison (Python-specific):
1 < x < 10           # True if x is between 1 and 10; each operand evaluated once
```

## Internal Working

- Every operator is **syntactic sugar for a dunder method**: `a + b` calls `a.__add__(b)`; if that returns `NotImplemented`, Python tries `b.__radd__(a)`. This is *operator overloading* and is how your own classes can support `+` (see [Magic Methods](#18-magic--dunder-methods)).
- **Short-circuit evaluation:** `and`/`or` stop as soon as the result is known and **return the operand itself**, not a coerced boolean. `a and b` = "if `a` is falsy return `a`, else return `b`". This enables the `x or default` idiom and guards like `obj and obj.method()`.
- **Chained comparisons** `a < b < c` are compiled to `a < b and b < c` with `b` evaluated only once — a genuine Python feature, not available in C/Java.
- `is` compiles to an identity (pointer) comparison; `==` calls `__eq__`.

## Advantages

- Overloadable — your classes read like built-ins.
- Short-circuiting avoids unnecessary/expensive evaluation and enables safe guards.
- Chained comparisons match mathematical notation.

## Limitations

- `and`/`or` returning operands (not `bool`) surprises people expecting `True`/`False`.
- Floor division rounding toward negative infinity (`-7 // 2 == -4`) is unintuitive.
- Overloading can be abused to make code cryptic.

## Real-world Applications

Operator overloading powers NumPy/Pandas (`array_a + array_b` does element-wise math), SQLAlchemy (`User.age > 18` builds a SQL query), and pathlib (`Path("/") / "etc" / "hosts"`). Short-circuiting is used constantly for defaults and null-guards.

## Interview Questions

- **Beginner:** Difference between `/` and `//`?
- **Beginner:** What does `5 % 3` return, and what is modulo used for?
- **Intermediate:** What does `0 or "x" or "y"` evaluate to and why?
- **Intermediate:** Difference between `is` and `==` (again — it's a favorite).
- **Advanced:** How would you make `Vector(1,2) + Vector(3,4)` work?
- **"Why":** Why do `and`/`or` return operands instead of booleans?
- **Comparison:** `-7 // 2` vs `int(-7 / 2)` — do they match?
- **Scenario:** You need a default when a config value is missing or empty. Which operator and why?

## Model Answers

**Q: `0 or "x" or "y"`?**
Evaluates left to right, returning the **first truthy operand**. `0` is falsy, `"x"` is truthy → returns `"x"` and never evaluates `"y"` (short-circuit). If all were falsy, `or` returns the last operand.

**Q: `-7 // 2` vs `int(-7 / 2)`?**
`-7 // 2` is **floor division** → `-4` (rounds toward −∞). `int(-7 / 2)` computes `-3.5` then `int()` **truncates toward zero** → `-3`. They differ for negatives. Knowing this prevents off-by-one bugs in indexing/pagination math.

**Q: Make `Vector + Vector` work?**
Define `__add__`:
```python
class Vector:
    def __init__(self, x, y): self.x, self.y = x, y
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    def __repr__(self): return f"Vector({self.x}, {self.y})"
```
Now `Vector(1,2) + Vector(3,4)` → `Vector(4, 6)`. Python translated `+` into `__add__`.

**Q (Why): `and`/`or` return operands?**
It's strictly more powerful. Returning the operand lets you write `name = provided or "Anonymous"` and `record and record.id`. Since any object has truthiness, returning the object preserves information a plain `True`/`False` would throw away. When you specifically want a bool, wrap it: `bool(a and b)`.

## Common Mistakes

- Expecting `and`/`or` to return `True`/`False`.
- Using `is` instead of `==` for numbers/strings.
- Forgetting `/` yields a float even for `4 / 2` (→ `2.0`).
- Confusing floor division and truncation for negatives.
- Writing `if x == None` instead of the idiomatic `if x is None`.

## Related Concepts

[Data Types](#2-data-types--type-conversion), [Magic Methods](#18-magic--dunder-methods) (overloading), [Control Flow](#4-control-flow) (truthiness in conditions).

---

# 4. Control Flow

## What is it?

Control flow is how a program **decides which statements run and how many times**. It has two pillars: **conditionals** (`if` / `elif` / `else`) that branch based on truth, and **loops** (`for`, `while`) that repeat. **Loop control** statements — `break`, `continue`, `pass` — alter the flow inside loops. Python also has the `for...else` / `while...else` construct.

## Why is it needed?

Without control flow a program is just a straight line of statements. Real logic requires decisions ("if the user is an admin…") and repetition ("for every row in the file…"). Control flow is what turns a script into an algorithm.

## How does it work?

```python
# Conditional
if score >= 90:
    grade = "A"
elif score >= 80:      # only checked if previous conditions were False
    grade = "B"
else:
    grade = "C"

# for loop iterates over any ITERABLE
for i, item in enumerate(items):   # enumerate gives index + value
    print(i, item)

# while loop repeats while condition is truthy
n = 5
while n > 0:
    print(n); n -= 1

# Loop control
for x in data:
    if x is None:
        continue        # skip to next iteration
    if x == "STOP":
        break           # exit the loop entirely
    process(x)

# for...else: else runs ONLY if the loop finished without break
for x in items:
    if x == target:
        print("found"); break
else:
    print("not found")   # runs when target never found
```

- `pass` is a **no-op placeholder** — it does nothing, used where a statement is syntactically required but you have no logic yet (empty function/class/branch).

## Internal Working

- `for` does **not** use a counter. It calls `iter(iterable)` to get an **iterator**, then repeatedly calls `next()` until `StopIteration` is raised, which Python catches to end the loop (see [Iterators](#26-iterators)). This is why `for` works on lists, strings, files, generators, and dict keys uniformly.
- The `else` clause on loops is tied to `break`: the loop keeps an internal "did we break?" state; `else` runs only if the loop exhausted the iterator normally. Think of it as "no-break".
- `if`/`elif` conditions are evaluated **lazily top-to-bottom**; the first truthy one wins and the rest are skipped.

## Advantages

- `for...else` elegantly expresses search-without-a-flag-variable.
- `for`-over-iterables unifies iteration across all container types.
- `enumerate`/`zip` remove manual index bookkeeping (fewer off-by-one bugs).

## Limitations

- `for...else` is famously confusing — many teams avoid it because readers misread `else` as "if loop didn't run".
- Python has **no `switch` statement** historically (3.10+ adds `match`/`case` structural pattern matching, but that's beyond this basic syllabus).
- Deeply nested `if` pyramids hurt readability — prefer early returns / guard clauses.

## Real-world Applications

Control flow is universal: request routing (`if method == "POST"`), retry loops (`while attempts < max`), data validation pipelines, game loops, and batch processing every record in a dataset.

## Interview Questions

- **Beginner:** Difference between `break` and `continue`?
- **Beginner:** What is `pass` used for?
- **Intermediate:** Explain `for...else`. When does the `else` run?
- **Intermediate:** How does a `for` loop iterate internally?
- **Advanced:** Rewrite a nested-loop linear search using `for...else` and explain the benefit.
- **"Why":** Why does Python's `for` not have a C-style index counter?
- **Comparison:** When would you choose `while` over `for`?
- **Scenario:** You must retry a network call up to 3 times, stopping on first success. Which construct?

## Model Answers

**Q: When does `for...else`'s `else` run?**
The `else` runs **only if the loop completed without hitting `break`**. It's designed for search loops: you `break` when you find the target; if you never break, the `else` handles the "not found" case — removing the need for a `found = False` flag variable. Mnemonic: read it as `for...nobreak`.

**Q: How does a `for` loop iterate internally?**
Python calls `iter(iterable)` once to obtain an iterator object, then calls `next()` on it repeatedly, binding each returned value to the loop variable. When `next()` raises `StopIteration`, Python catches it silently and ends the loop. Because the protocol is `iter()` + `next()`, `for` works identically on lists, tuples, strings, dicts, sets, files, and lazy generators.

**Q: `while` vs `for`?**
Use `for` when you're iterating over a **known collection or a fixed number of steps** (it's the default and clearer). Use `while` when the number of iterations is **unknown ahead of time and depends on a condition** — retry-until-success, reading until EOF, game loops, or "keep going until converged". If you find yourself manually managing an index with `while`, a `for` with `enumerate`/`range` is usually cleaner.

**Q (Scenario): Retry a network call 3× stopping on success:**
```python
for attempt in range(3):
    if call_api():          # returns True on success
        break
else:
    raise RuntimeError("all 3 attempts failed")
```
`for range(3)` bounds the retries, `break` stops early on success, and `for...else` cleanly expresses the all-failed case.

## Common Mistakes

- Misreading `for...else` as "else if loop body didn't run".
- Using `break`/`continue` outside a loop (SyntaxError).
- Modifying a list while iterating over it (skips elements) — iterate over a copy or build a new list.
- Off-by-one errors from manual indexing instead of `enumerate`.
- Infinite `while` loops from forgetting to update the condition variable.

## Related Concepts

[Iterators](#26-iterators), [Operators](#3-operators) (truthiness), [Comprehensions](#31-comprehensions) (loop expressions), [Functions](#5-functions) (early returns).

# 5. Functions

## What is it?

A **function** is a named, reusable block of code that takes **inputs (parameters)**, performs work, and optionally **returns an output**. In Python functions are **first-class objects**: you can assign them to variables, pass them as arguments, return them from other functions, and store them in data structures. This section covers definition, the full argument system, **scope** (LEGB), **lambda** (anonymous functions), and **recursion**.

## Why is it needed?

Functions are the primary tool for **abstraction and reuse** — the "DRY" (Don't Repeat Yourself) principle. They let you name a piece of behavior, test it in isolation, and reason about a program in small pieces instead of one giant script. First-class function support is what makes decorators, callbacks, and functional-style code possible.

## How does it work?

```python
def greet(name, greeting="Hello", *args, **kwargs):
    return f"{greeting}, {name}!"
```

**The argument system (order matters):**

```python
def f(pos_only, /, standard, *args, kw_only, **kwargs):
    ...
#        ^ positional-only   ^ collects extra positionals
#                     ^ normal      ^ keyword-only (after *)   ^ collects extra keywords
```

- **Positional arguments** — matched by order: `f(1, 2)`.
- **Keyword arguments** — matched by name: `f(name="Sam")`; order-independent, self-documenting.
- **Default arguments** — `greeting="Hello"`; make a parameter optional.
- **`*args`** — collects extra positional args into a **tuple**.
- **`**kwargs`** — collects extra keyword args into a **dict**.
- **Positional-only** (before `/`) and **keyword-only** (after `*`) — restrict how callers may pass arguments.

**Scope — the LEGB rule** (name lookup order):

```
Local  -> Enclosing -> Global -> Built-in
```

```python
x = "global"
def outer():
    x = "enclosing"
    def inner():
        # reads x from Enclosing via LEGB
        print(x)
    inner()

def counter():
    n = 0
    def inc():
        nonlocal n     # rebind the enclosing n
        n += 1
        return n
    return inc

g = 0
def bump():
    global g           # rebind the module-level g
    g += 1
```

**Lambda** — a single-expression anonymous function:

```python
square = lambda x: x * x
sorted(people, key=lambda p: p.age)        # sort by age
list(map(lambda x: x*2, nums))
list(filter(lambda x: x % 2 == 0, nums))
```

**Recursion** — a function that calls itself, with a **base case** (stops recursion) and a **recursive case** (reduces the problem):

```python
def factorial(n):
    if n <= 1:          # base case
        return 1
    return n * factorial(n - 1)   # recursive case
```

## Internal Working

- Calling a function pushes a **stack frame** onto the call stack holding its local namespace, arguments, and return address. Returning pops the frame. Python's default recursion limit is ~1000 frames (`sys.setrecursionlimit`) to avoid a C-stack overflow — deep recursion raises `RecursionError`.
- Default argument values are **evaluated once, at function-definition time**, and stored on the function object (`func.__defaults__`). This is the source of the infamous **mutable default argument** bug.
- `*args`/`**kwargs` pack extra arguments into a fresh tuple/dict per call.
- `global` and `nonlocal` don't create scope — they change *where a name is bound*. Without them, assigning to a name inside a function creates a **new local**, which is why reading-then-assigning a global raises `UnboundLocalError`.
- A `def` with no `return` returns `None`.
- **Closures:** an inner function that references enclosing variables keeps them alive via `__closure__` cells (see [Decorators](#28-closures--decorators)).

## Advantages

- Reuse, testability, and readability (named intent).
- First-class functions enable higher-order patterns (map/filter, decorators, callbacks).
- Flexible calling conventions (`*args`/`**kwargs`) enable wrappers and APIs.

## Limitations

- Recursion is limited by the stack and is usually slower than iteration in CPython (no tail-call optimization).
- The mutable-default-argument pitfall causes subtle shared-state bugs.
- Lambdas are restricted to a single expression — no statements, no annotations, harder to debug (they're anonymous in tracebacks).

## Real-world Applications

Everything is built from functions. Higher-order functions power sorting keys, event callbacks, Flask/Django view functions, `map`/`filter` data transforms, and decorators used for routing, caching, and auth. Recursion shines on tree/graph traversal (file systems, JSON, DOM), divide-and-conquer algorithms, and parsers.

## Interview Questions

- **Beginner:** Difference between a parameter and an argument?
- **Beginner:** What does a function return if there's no `return` statement?
- **Intermediate:** Difference between `*args` and `**kwargs`?
- **Intermediate:** Explain the LEGB rule.
- **Intermediate:** What's the mutable-default-argument bug? Show the fix.
- **Advanced:** Difference between `global` and `nonlocal`.
- **Advanced:** Convert a recursive factorial to iterative and discuss trade-offs.
- **"Why":** Why does `def f(x, lst=[])` misbehave?
- **Comparison:** `lambda` vs `def` — when to use each?
- **Scenario:** Design a function that accepts any number of tags plus arbitrary metadata.

## Model Answers

**Q: Mutable default argument bug + fix.**
```python
def add(item, bucket=[]):      # BUG: bucket is created ONCE at def time
    bucket.append(item)
    return bucket
add(1)   # [1]
add(2)   # [1, 2]  <- surprise! same list reused
```
The default `[]` is evaluated once when the function is defined and stored on the function object, so every call that omits `bucket` shares that same list. **Fix** with the sentinel `None` pattern:
```python
def add(item, bucket=None):
    if bucket is None:
        bucket = []            # fresh list per call
    bucket.append(item)
    return bucket
```

**Q: `global` vs `nonlocal`.**
Both let you *rebind* a name that isn't local. `global` targets the **module-level** namespace; `nonlocal` targets the **nearest enclosing function** scope (used in closures/decorators). Without either, an assignment inside a function always creates a new local, and referencing that name before assignment raises `UnboundLocalError`. Prefer returning values over mutating globals — global mutation makes code hard to test.

**Q: LEGB rule.**
When Python resolves a name it searches four scopes in order: **L**ocal (inside the current function), **E**nclosing (any outer functions), **G**lobal (module level), **B**uilt-in (`len`, `print`, etc.). The first match wins. This is why an inner function can read an outer variable, and why shadowing a built-in (`list = [1,2]`) breaks `list()` calls afterward.

**Q: `lambda` vs `def`.**
Use `lambda` for a **short, throwaway, single-expression** function passed inline — typically as a `key=` to `sorted`/`max`, or a quick `map`/`filter` transform. Use `def` when the function needs a name (for tracebacks/reuse), multiple statements, documentation, default args, or type hints. Lambdas can't contain statements (`return`, loops, assignments) and appear as `<lambda>` in stack traces, hurting debuggability. PEP 8 even discourages assigning a lambda to a name — use `def` there.

**Q (Scenario): Any number of tags + arbitrary metadata:**
```python
def create_post(title, *tags, **metadata):
    return {"title": title, "tags": tags, "meta": metadata}

create_post("Hi", "python", "oop", author="Sam", views=0)
# {'title': 'Hi', 'tags': ('python','oop'), 'meta': {'author':'Sam','views':0}}
```
`*tags` captures variable positional tags as a tuple; `**metadata` captures arbitrary named fields as a dict.

## Common Mistakes

- Mutable default arguments (`def f(x=[])`).
- Assigning to a global/enclosing name without declaring `global`/`nonlocal` → `UnboundLocalError`.
- Shadowing built-ins (`list`, `dict`, `id`, `sum`).
- Deep recursion without a base case → `RecursionError`.
- Confusing return value `None` (no `return`) with an intended result.

## Related Concepts

[Closures & Decorators](#28-closures--decorators), [Functional Programming](#30-functional-programming), [Recursion in DSA], [Scope & Memory](#33-memory-management).

---

# 6. Lists

## What is it?

A `list` is Python's built-in **ordered, mutable, heterogeneous sequence** — a resizable array. It preserves insertion order, allows duplicates, supports indexing/slicing, and can hold values of any (mixed) types.

## Why is it needed?

You constantly need an ordered, growable collection: a to-do list, rows read from a file, results of a computation. Fixed-size arrays (as in C) are painful; lists grow and shrink automatically and come with rich methods.

## How does it work?

```python
nums = [3, 1, 4, 1, 5]
nums[0]        # 3     indexing (0-based)
nums[-1]       # 5     negative index from the end
nums[1:4]      # [1, 4, 1]   slice [start:stop:step], stop exclusive
nums[::-1]     # reversed copy
nums.append(9)         # add to end (amortized O(1))
nums.insert(0, 99)     # insert at index (O(n) shift)
nums.pop()             # remove & return last (O(1))
nums.pop(0)            # remove & return first (O(n))
nums.remove(4)         # remove first matching VALUE
nums.sort()            # in-place sort; sorted(nums) returns a new list
squares = [x*x for x in range(5)]      # list comprehension
```

Nested lists model matrices: `grid = [[0]*3 for _ in range(3)]` (use the comprehension, not `[[0]*3]*3`, which aliases rows).

## Internal Working

- CPython's `list` is a **dynamic array of pointers** to objects (not the objects themselves), so it can hold mixed types.
- It **over-allocates** capacity (growth factor ~1.125×) so `append` is **amortized O(1)** — most appends don't reallocate; occasionally the backing array is copied to a bigger block.
- Indexing is **O(1)** (pointer arithmetic). Insert/delete at the front is **O(n)** because all later elements shift.
- `in`, `index`, `remove` are **O(n)** (linear scan).
- `sort()` uses **Timsort** — a stable, adaptive merge/insertion hybrid, O(n log n) worst case, O(n) on already-sorted data.

## Advantages

- O(1) index access and append.
- Mutable and dynamically sized.
- Rich API (slicing, comprehensions, sort with `key`).

## Limitations

- O(n) membership tests and front insert/delete (use `set`/`deque` when those dominate).
- Mutability + aliasing causes shared-state bugs (`a = b` shares the same list).
- Higher memory overhead than arrays of primitives (stores pointers); use `array`/NumPy for large numeric data.

## Real-world Applications

Collecting results, buffering stream data, representing stacks (`append`/`pop`), matrices/grids, and any ordered dataset before it's turned into a set/dict/DataFrame.

## Interview Questions

- **Beginner:** How do you copy a list without aliasing the original?
- **Beginner:** Difference between `append` and `extend`?
- **Intermediate:** Time complexity of `append`, `insert(0,...)`, `x in list`?
- **Intermediate:** `sort()` vs `sorted()`?
- **Advanced:** Why is `[[0]*3]*3` dangerous?
- **"Why":** Why is `append` O(1) but `insert(0, x)` O(n)?
- **Comparison:** When would you use a list vs a set vs a deque?
- **Scenario:** You append 1M items in a loop — any performance concern?

## Model Answers

**Q: Copy without aliasing.**
`b = a` copies the *reference*, so mutating `b` mutates `a`. To copy: `b = a[:]`, `b = a.copy()`, or `b = list(a)` — all **shallow** copies (nested objects are still shared). For nested structures use `copy.deepcopy(a)`.

**Q: `append` vs `extend`.**
`append(x)` adds `x` as a **single element** (a list argument becomes one nested element). `extend(iterable)` adds **each element** of the iterable. `[1,2].append([3,4])` → `[1,2,[3,4]]`; `[1,2].extend([3,4])` → `[1,2,3,4]`.

**Q: `sort()` vs `sorted()`.**
`list.sort()` sorts **in place** and returns `None` (mutates the original). `sorted(iterable)` returns a **new sorted list** and works on any iterable, leaving the input unchanged. Both accept `key=` and `reverse=`. Use `sorted` when you need the original preserved or the input isn't a list.

**Q: Why is `[[0]*3]*3` dangerous?**
`[0]*3` makes one inner list; `*3` then makes a list of **three references to that same inner list**. Mutating `grid[0][0] = 1` changes all rows because they're the same object. Use `[[0]*3 for _ in range(3)]` to create independent rows.

**Q (Why): `append` O(1) vs `insert(0)` O(n)?**
Lists are contiguous arrays with spare capacity at the *end*, so appending just writes to the next slot (amortized O(1)). Inserting at the front has no spare room there, so every existing element must shift one position right — O(n). If you need fast front operations, use `collections.deque` (O(1) both ends).

## Common Mistakes

- `b = a` then mutating and being surprised both change (aliasing).
- `[[0]*n]*m` shared-row bug.
- Removing items while iterating (index skipping).
- Using a list for frequent membership tests (O(n)) instead of a set.
- Expecting `list.sort()` to return the sorted list (it returns `None`).

## Related Concepts

[Tuples](#7-tuples), [Comprehensions](#31-comprehensions), [Memory Management](#33-memory-management) (aliasing, deep copy), Stacks/Queues in DSA.

---

# 7. Tuples

## What is it?

A `tuple` is an **ordered, immutable sequence**. Like a list but you cannot change it after creation (no append/remove/reassign of elements). Written with parentheses `(1, 2, 3)` — but it's the **commas** that make a tuple, not the parentheses.

## Why is it needed?

Tuples represent **fixed collections of related values** — a coordinate `(x, y)`, an RGB color `(255, 0, 0)`, a database row. Immutability signals "this shouldn't change," makes the object **hashable** (usable as a dict key or set member), and can be slightly faster/lighter than a list.

## How does it work?

```python
point = (3, 4)
x, y = point            # unpacking
a, *rest = (1, 2, 3, 4) # a=1, rest=[2,3,4]  (extended unpacking)
single = (5,)           # ONE-element tuple needs a trailing comma
not_tuple = (5)         # this is just int 5!
packed = 1, 2, 3        # packing without parentheses
a, b = b, a             # swap via tuple packing/unpacking

point.count(3)          # 1
point.index(4)          # 1  (only two non-mutating methods)
```

## Internal Working

- A tuple stores a **fixed-length array of object pointers**. Because the length and contents can't change, CPython can allocate it exactly once and skip the over-allocation lists do — lower memory, and small tuples are cached/reused.
- **Immutability enables hashability**: `hash(tuple)` combines the hashes of its elements — but only if *all* elements are themselves hashable. `(1, [2])` is unhashable because it contains a list.
- Immutability is **shallow**: a tuple's *references* are fixed, but if it holds a mutable object (`([1,2], 3)`), that inner list can still be mutated.

## Advantages

- Hashable → usable as dict keys / set elements (if elements are hashable).
- Immutability communicates intent and prevents accidental modification.
- Slightly less memory and faster to construct than lists.
- Natural for multiple return values and unpacking.

## Limitations

- Cannot grow/shrink/modify — wrong choice for evolving data.
- Fixed positional meaning is less readable than named fields (use `namedtuple`/`dataclass` for clarity).
- Shallow immutability can mislead (a tuple containing a list isn't fully frozen).

## Real-world Applications

Returning multiple values from a function, dictionary keys that are composite (`grid[(row, col)]`), fixed records/rows, and coordinates. `namedtuple` is used for lightweight immutable records throughout the standard library (e.g., `os.stat` results, `time.struct_time`).

## Interview Questions

- **Beginner:** How do you create a one-element tuple?
- **Beginner:** Can you change an element of a tuple?
- **Intermediate:** Why can a tuple be a dict key but a list cannot?
- **Intermediate:** What is tuple unpacking? Show a swap.
- **Advanced:** Is `([1,2], 3)` hashable? Why/why not?
- **"Why":** Why choose a tuple over a list?
- **Comparison:** Tuple vs list vs namedtuple.
- **Scenario:** You need to use a (latitude, longitude) pair as a cache key. Which type?

## Model Answers

**Q: Why can a tuple be a dict key but not a list?**
Dict keys must be **hashable** and hashability requires the object's hash to never change over its lifetime. Lists are mutable, so their contents (and any hash) could change, breaking the dict's internal bucket placement — Python therefore makes lists unhashable. Tuples are immutable, so (provided their elements are also hashable) their hash is stable and they're valid keys.

**Q: Is `([1,2], 3)` hashable?**
No. Tuple hashing combines the hashes of all elements; the first element is a **list**, which is unhashable, so `hash(([1,2], 3))` raises `TypeError`. A tuple is only hashable if *every* element is hashable. `((1,2), 3)` would be fine.

**Q: Tuple vs list vs namedtuple.**
- **List:** mutable, for homogeneous, changing sequences.
- **Tuple:** immutable, for fixed heterogeneous records; hashable; positional access `t[0]`.
- **namedtuple:** an immutable tuple subclass with **named fields** (`p.x` instead of `p[0]`) — same memory profile as a tuple but far more readable for records. Prefer it (or a frozen dataclass) when the positions have meaning.

**Q (Why): Tuple over list?**
Choose a tuple when the collection is **conceptually fixed** and you want (a) to prevent accidental mutation, (b) to use it as a dict key / set element, or (c) a small performance/memory edge. The immutability is a correctness feature: it documents that the data is a constant record.

## Common Mistakes

- Writing `(5)` and expecting a tuple (need `(5,)`).
- Trying to mutate a tuple element.
- Assuming full immutability when the tuple contains a mutable object.
- Overusing positional tuples where a namedtuple/dataclass would be far clearer.

## Related Concepts

[Lists](#6-lists), [Dictionaries](#8-dictionaries) (hashable keys), [Dataclasses](#21-dataclasses), namedtuple, [Sets](#9-sets).

---

# 8. Dictionaries

## What is it?

A `dict` is Python's **hash map / associative array**: an unordered-by-concept (insertion-ordered since 3.7) collection of **key → value** pairs with **average O(1)** lookup, insert, and delete. Keys must be **hashable and unique**; values can be anything.

## Why is it needed?

When you need to look something up **by a key** rather than by position — a phone book, a cache, counting word frequencies, JSON-like nested data — a dict gives near-instant access. Doing the same with a list would require O(n) scanning.

## How does it work?

```python
user = {"name": "Sam", "age": 30}
user["name"]              # "Sam"   KeyError if missing
user.get("email")         # None    (safe; no KeyError)
user.get("email", "n/a")  # default
user["email"] = "s@x.com" # insert/update
del user["age"]           # delete
"name" in user            # True    membership tests KEYS, O(1)

for k, v in user.items():        # iterate pairs
    print(k, v)
user.keys(); user.values(); user.items()   # dynamic views

squares = {x: x*x for x in range(5)}        # dict comprehension
user.setdefault("roles", []).append("admin")
from collections import Counter, defaultdict
Counter("banana")               # {'a':3,'n':2,'b':1}
```

## Internal Working

- Backed by a **hash table**. Storing a key: compute `hash(key)`, map it to a bucket index, and place the entry. Lookup repeats the hash to jump straight to the bucket — hence **O(1) average**.
- **Collisions** (two keys hashing to the same bucket) are resolved by open addressing/probing. Worst-case lookup is O(n) if many keys collide, but good hash distribution keeps it O(1) in practice.
- Since **Python 3.7**, dicts **preserve insertion order** as a language guarantee (CPython 3.6 implemented a compact, order-preserving layout that also cut memory ~20–25%).
- **Key equality:** two keys are "the same" if `hash(a) == hash(b)` **and** `a == b`. This is why custom classes used as keys must implement both `__hash__` and `__eq__` consistently.

## Advantages

- Average O(1) lookup/insert/delete.
- Extremely flexible for structured/nested data (mirrors JSON).
- Insertion-ordered, with rich helpers (`get`, `setdefault`, `Counter`, `defaultdict`).

## Limitations

- Keys must be hashable (no lists/dicts as keys).
- More memory than a list for the same number of values (stores hashes + keys + values).
- Worst-case O(n) under pathological collisions.

## Real-world Applications

Caches and memoization, counting/frequency analysis (`Counter`), configuration, JSON/API payloads, database rows as records, grouping data (`defaultdict(list)`), and any "look up by identifier" need.

## Interview Questions

- **Beginner:** Difference between `d["k"]` and `d.get("k")`?
- **Beginner:** Can a list be a dict key?
- **Intermediate:** Average and worst-case time complexity of dict lookup?
- **Intermediate:** How do you count occurrences of items?
- **Advanced:** What two methods must a custom object implement to be a dict key, and why must they agree?
- **"Why":** Why are dict lookups O(1) on average?
- **Comparison:** `dict` vs `list` for lookups; `setdefault` vs `defaultdict`.
- **Scenario:** Build a word-frequency counter for a large text file efficiently.

## Model Answers

**Q: `d["k"]` vs `d.get("k")`.**
`d["k"]` returns the value but raises `KeyError` if the key is absent. `d.get("k")` returns `None` (or a supplied default `d.get("k", 0)`) instead of raising. Use `[]` when a missing key is a bug you want to surface; use `.get()` when absence is normal and you have a sensible default.

**Q: Why must a dict-key object implement `__hash__` and `__eq__` consistently?**
The dict finds a key by first hashing it to a bucket, then confirming with `==`. If two objects are equal (`a == b`) they **must** have the same hash, otherwise the dict could store them in different buckets and fail to find an existing key. Rule: **equal objects → equal hashes** (the reverse need not hold). If you define `__eq__` you should define `__hash__` too, or the object becomes unhashable.

**Q: Why O(1) average lookup?**
The hash function converts a key directly into a bucket index, so finding a key doesn't depend on how many items are stored — it's a constant number of steps. Only when many keys collide into the same bucket does it degrade toward O(n), which good hashing and the table's automatic resizing keep rare.

**Q (Scenario): Word-frequency counter.**
```python
from collections import Counter
with open("big.txt") as f:
    counts = Counter(word.lower() for line in f for word in line.split())
counts.most_common(10)
```
`Counter` is a dict subclass purpose-built for this: each word is O(1) to tally, the generator streams the file lazily (low memory), and `most_common` ranks results. Manually, `defaultdict(int)` with `counts[w] += 1` is equivalent.

## Common Mistakes

- Using `[]` access and getting surprise `KeyError` (use `.get`/`setdefault`/`defaultdict`).
- Trying to use a list/dict as a key.
- Relying on ordering pre-3.7 (or assuming "sorted" — it's *insertion* order, not sorted order).
- Defining `__eq__` without `__hash__` on key objects.
- Mutating a dict while iterating its keys (`RuntimeError`).

## Related Concepts

[Sets](#9-sets) (hash tables), [Tuples](#7-tuples) (hashable keys), [Magic Methods](#18-magic--dunder-methods) (`__hash__`/`__eq__`), `collections`, [Comprehensions](#31-comprehensions).

---

# 9. Sets

## What is it?

A `set` is an **unordered collection of unique, hashable elements**. It's essentially a dict with keys but no values, giving **O(1) membership tests** and native **mathematical set operations** (union, intersection, difference). `frozenset` is its immutable, hashable variant.

## Why is it needed?

Two big use-cases: **deduplication** (remove duplicates instantly) and **fast membership testing** (`x in big_set` is O(1) vs O(n) for a list). Plus, when your problem is naturally about set algebra — "which users are in both groups?", "what tags are new?" — sets express it directly.

## How does it work?

```python
s = {1, 2, 3, 3}      # {1, 2, 3} — duplicates dropped
empty = set()          # {} is an empty DICT, not a set!
s.add(4); s.discard(9) # discard: no error if absent; remove: raises KeyError
3 in s                 # O(1) membership

a, b = {1,2,3}, {2,3,4}
a | b     # union            {1,2,3,4}
a & b     # intersection     {2,3}
a - b     # difference       {1}
a ^ b     # symmetric diff   {1,4}   (in one but not both)
a <= b    # subset test
unique = set(my_list)               # dedupe a list
frozen = frozenset([1,2,3])         # hashable set (can be a dict key)
```

## Internal Working

- Implemented as a **hash table** (like dict, values omitted). Add/lookup/remove are **O(1) average** by hashing the element to a bucket.
- Elements must be **hashable** — you can't put a list in a set, but you can put a tuple or frozenset.
- **Unordered:** there's no index; iteration order is an implementation artifact of the hash layout, not insertion order (unlike dict).
- Set operations iterate the smaller set and probe the larger, giving roughly O(min(len(a), len(b))) for intersection.

## Advantages

- O(1) membership and deduplication.
- Expressive, readable set-algebra operators.
- `frozenset` is hashable → usable as dict keys / nested in other sets.

## Limitations

- Unordered and unindexable (no `s[0]`).
- Elements must be hashable (no lists/dicts inside).
- Higher per-element memory than a list; no duplicates allowed (sometimes you *need* counts → use `Counter`).

## Real-world Applications

De-duplicating records, tracking "seen" items in algorithms (visited nodes in BFS/DFS), permission/tag/role comparisons, finding common or unique elements between datasets, and fast blocklist/allowlist lookups.

## Interview Questions

- **Beginner:** How do you remove duplicates from a list?
- **Beginner:** How do you create an empty set?
- **Intermediate:** Why is `x in a_set` faster than `x in a_list`?
- **Intermediate:** Explain union, intersection, difference, symmetric difference.
- **Advanced:** What's a `frozenset` and when do you need it?
- **"Why":** Why can't a set contain a list?
- **Comparison:** set vs list vs dict — when to pick each.
- **Scenario:** You must find users present in list A but not list B among millions of records.

## Model Answers

**Q: Why is set membership faster than list membership?**
A list checks membership by **scanning element by element** — O(n). A set hashes the element to compute a bucket and checks just that bucket — **O(1) average**, independent of size. For large collections and repeated lookups, converting a list to a set first is a massive speedup.

**Q: Why can't a set contain a list?**
Sets store elements in a hash table keyed by `hash(element)`. Lists are mutable and therefore unhashable (their contents/hash could change, corrupting bucket placement). Use an **immutable** equivalent — a tuple or `frozenset` — instead.

**Q: `frozenset` and when needed.**
A `frozenset` is an immutable, hashable set. Because regular sets are mutable (unhashable), you can't use a set as a dict key or put it inside another set — but you *can* with a frozenset. Example: caching results keyed by a *set* of parameters, or building a "set of sets".

**Q (Scenario): Users in A but not B among millions.**
```python
only_in_a = set(A) - set(B)
```
Build sets once (O(n) total), then the difference is roughly O(len(A)) with O(1) lookups into B's set — far better than the O(n·m) nested-loop approach on lists. If order/duplicates matter, post-process, but for pure "present in A not B", set difference is optimal and reads like the requirement.

## Common Mistakes

- Using `{}` for an empty set (that's an empty dict) — use `set()`.
- Expecting sets to preserve order or support indexing.
- Trying to add a list/dict to a set.
- Using a set when you need counts or duplicates (use `Counter`/list).
- Forgetting `remove` raises `KeyError` while `discard` doesn't.

## Related Concepts

[Dictionaries](#8-dictionaries) (hash tables), [Tuples](#7-tuples)/frozenset (hashable), Graph traversal "visited" sets, [Comprehensions](#31-comprehensions).

---

# 10. Strings & Regular Expressions

## What is it?

A `str` is an **immutable sequence of Unicode characters**. Strings come with a large method API for searching, splitting, joining, formatting, and case handling. **Regular expressions** (`re` module) are a mini-language for **pattern matching** in text.

## Why is it needed?

Text is everywhere — user input, files, logs, HTML, APIs. String methods handle the common 90% (split a CSV line, strip whitespace, format a message). Regex handles the structured-pattern 10% that methods can't: "find all emails", "validate a phone number", "extract dates".

## How does it work?

```python
s = "  Hello, World  "
s.strip()                 # "Hello, World"  (also lstrip/rstrip)
s.lower(); s.upper(); s.title()
s.replace("l", "L")
s.split(",")              # ['  Hello', ' World  ']
",".join(["a", "b"])      # "a,b"   (join is a str method!)
s.startswith("  He")      # True
s.find("World")           # index or -1;  .index raises if absent
"abc"[::-1]               # "cba"  reverse via slicing

name, n = "Sam", 3
f"{name} has {n} items, {n/7:.2f}"   # f-string: fast, readable formatting
"{} {}".format(a, b)                  # .format style

import re
re.findall(r"\d+", "a12b345")         # ['12', '345']
re.search(r"(\w+)@(\w+)", "x@y")      # Match; .group(1)='x', .group(2)='y'
re.sub(r"\s+", " ", messy)            # collapse whitespace
pattern = re.compile(r"^\d{3}-\d{4}$")# compile once, reuse many times
```

## Internal Working

- **Immutability:** every "modifying" operation returns a **new string**; the original is untouched. That's why building a string with `+=` in a loop is O(n²) — each concatenation copies everything. Use `"".join(list_of_parts)` instead (O(n)).
- CPython **interns** short/identifier-like strings (caches them) so identical literals may share one object — an optimization for fast `==`/dict-key comparison, not something to rely on with `is`.
- Strings are stored as Unicode code points with an adaptive internal width (1/2/4 bytes per char) depending on the largest code point.
- **Regex engine:** `re` compiles a pattern into a small state machine/bytecode, then runs it against the text. `re.compile` does this once so repeated matching skips recompilation. Backtracking can make poorly-written patterns catastrophically slow ("ReDoS").

## Advantages

- Rich, readable method API; f-strings are fast and clean.
- Regex is extremely powerful for pattern extraction/validation.
- Immutability makes strings hashable (dict keys) and safe to share.

## Limitations

- Immutability makes repeated concatenation expensive (`join` fixes it).
- Regex is write-only/hard to read and easy to get wrong; overusing it for simple splits is a smell.
- Unicode edge-cases (emoji, combining characters) can surprise length/indexing assumptions.

## Real-world Applications

Log parsing, input validation (emails, phone numbers), template/message generation (f-strings), tokenizing/cleaning text for NLP, scraping, search-and-replace tooling, and config/format conversion.

## Interview Questions

- **Beginner:** Are strings mutable? What happens when you "change" one?
- **Beginner:** How do you reverse a string?
- **Intermediate:** Why is building a string with `+=` in a loop slow? Fix it.
- **Intermediate:** Difference between `find` and `index`; `split` and `join`.
- **Advanced:** When would you `re.compile`? What is catastrophic backtracking?
- **"Why":** Why are strings immutable in Python?
- **Comparison:** f-strings vs `.format()` vs `%` formatting.
- **Scenario:** Validate that a string is a valid email-like token and extract the domain.

## Model Answers

**Q: Are strings mutable?**
No — strings are immutable. `s.replace(...)`, `s.upper()`, `s += "x"` all **return new string objects**; the original is unchanged. To "mutate" you must rebind the name. This immutability is why strings can be dict keys and are safe to share across code.

**Q: Why is `+=` in a loop slow, and the fix?**
Because each `+=` creates a brand-new string and copies all existing characters into it — doing that n times is **O(n²)**. Collect the pieces in a list and join once:
```python
parts = []
for chunk in data:
    parts.append(chunk)
result = "".join(parts)   # O(n)
```

**Q: `re.compile` and catastrophic backtracking.**
`re.compile(pattern)` builds the pattern's matcher **once**; reusing the compiled object in a loop avoids recompiling every call — a real speedup for hot paths. **Catastrophic backtracking** happens when nested/ambiguous quantifiers (e.g. `(a+)+$`) force the engine to try exponentially many ways to match on failing input, freezing your program (a ReDoS denial-of-service risk). Mitigate by writing specific patterns, avoiding nested quantifiers, and anchoring.

**Q: f-strings vs `.format()` vs `%`.**
- **f-strings** (3.6+): `f"{x:.2f}"` — most readable, fastest, evaluated inline. Default choice.
- **`.format()`**: `"{:.2f}".format(x)` — works when the template is separate from the data (e.g. stored in a config) or for older code.
- **`%`**: `"%.2f" % x` — legacy C-style; still seen in logging (`logger.info("%s", x)` defers formatting). Prefer f-strings for new code.

**Q (Scenario): Validate email-ish and extract domain.**
```python
import re
m = re.match(r"^([\w.+-]+)@([\w-]+\.[\w.-]+)$", token)
domain = m.group(2) if m else None
```
`match` anchors at the start; the two groups capture local part and domain. (For production, real email validation is notoriously tricky — often better to send a confirmation email than to over-engineer the regex.)

## Common Mistakes

- Trying to assign `s[0] = "x"` (strings are immutable → `TypeError`).
- Quadratic string building with `+=` in loops.
- Using regex for trivial jobs `str.split`/`in` handle better.
- Confusing `find` (returns -1) with `index` (raises).
- Forgetting `join` is called on the *separator* (`",".join(list)`).

## Related Concepts

[Data Types](#2-data-types--type-conversion), [File Handling](#11-file-handling) (parsing text), [Memory Management](#33-memory-management) (immutability/interning), `re`, f-strings.

---

# 11. File Handling

## What is it?

File handling is reading from and writing to files on disk. Python's `open()` returns a **file object** you interact with via `read`/`readline`/`readlines`/`write`/`writelines`, controlled by a **mode** (`r`, `w`, `a`, `r+`, plus `b`/`t` for binary/text). The idiomatic way is the **`with` statement (context manager)**, which guarantees the file is closed.

## Why is it needed?

Programs must **persist** data beyond a single run and **consume** external data. Reading a config, processing a CSV of records, writing logs or reports, exporting JSON — all require file I/O. Doing it safely (closing handles, handling encodings, not loading gigabytes into memory) is a core skill.

## How does it work?

```python
# BEST PRACTICE: context manager auto-closes even on exceptions
with open("data.txt", "r", encoding="utf-8") as f:
    content = f.read()          # entire file as one string
    # f.readline()  -> one line
    # f.readlines() -> list of lines (loads all into memory)

# Memory-efficient: iterate line by line (lazy, streams)
with open("big.log") as f:
    for line in f:              # reads one line at a time
        process(line.rstrip("\n"))

with open("out.txt", "w") as f:  # 'w' truncates! 'a' appends
    f.write("hello\n")
    f.writelines(["a\n", "b\n"])

import csv, json
with open("data.csv", newline="") as f:
    for row in csv.DictReader(f):   # row is a dict per line
        print(row["name"])

with open("config.json") as f:
    config = json.load(f)           # parse JSON -> dict
with open("out.json", "w") as f:
    json.dump(config, f, indent=2)  # write dict -> JSON
```

**Modes:** `r` read (default, error if missing), `w` write (creates/**truncates**), `a` append, `x` exclusive-create (error if exists), `+` read+write; add `b` for binary (`rb`, `wb`), `t` for text (default).

## Internal Working

- `open()` returns a file object backed by an OS file descriptor and an **in-memory buffer**. Writes accumulate in the buffer and are **flushed** to disk in blocks (for performance); `close()`/context-exit flushes remaining data. Forgetting to close can lose buffered writes.
- Iterating a file object (`for line in f`) is **lazy** — it reads a chunk and yields lines on demand, so you can process a file larger than RAM. `read()`/`readlines()` load everything at once.
- **Text vs binary:** in text mode Python decodes bytes → `str` using an encoding (default is platform-dependent, so **always pass `encoding="utf-8"`**) and translates newlines. Binary mode gives raw `bytes`.
- The `with` statement calls the file's `__enter__`/`__exit__`; `__exit__` closes the file **even if an exception is raised** inside the block (see [Context Managers](#29-context-managers)).

## Advantages

- `with` guarantees cleanup — no leaked file descriptors.
- Lazy line iteration handles arbitrarily large files.
- Standard library covers text, CSV, and JSON out of the box.

## Limitations

- Encoding bugs if you don't specify `encoding` (mojibake, `UnicodeDecodeError`).
- `w` mode silently truncates existing files — data loss if you meant `a`.
- Manual `open`/`close` without `with` leaks handles on exceptions.

## Real-world Applications

Config loading, ETL/data pipelines (reading CSVs, writing processed output), log processing, report/export generation, serializing app state to JSON, and reading model/data files in ML workflows.

## Interview Questions

- **Beginner:** Why use `with open(...)` instead of `open()` + `close()`?
- **Beginner:** Difference between `read()`, `readline()`, `readlines()`?
- **Intermediate:** Difference between modes `w`, `a`, `x`, `r+`?
- **Intermediate:** How do you read a 10 GB file without running out of memory?
- **Advanced:** What does the `with` statement guarantee if an exception occurs mid-write?
- **"Why":** Why should you always pass `encoding="utf-8"`?
- **Comparison:** `read()` vs iterating the file object.
- **Scenario:** Append audit entries to a log file safely under errors.

## Model Answers

**Q: Why `with open(...)`?**
Because `with` uses the file as a **context manager**: it guarantees `close()` is called when the block exits — **even if an exception is raised** — flushing buffered writes and releasing the OS file descriptor. Manual `open()`/`close()` leaks the handle if an error occurs between them (unless you write a verbose `try/finally`, which is exactly what `with` does for you).

**Q: `read` vs `readline` vs `readlines`.**
`read()` returns the **whole file** as a single string (or `read(n)` for n chars). `readline()` returns the **next single line** (including `\n`). `readlines()` returns a **list of all lines**. `read` and `readlines` load everything into memory; for large files iterate the file object (`for line in f`) which is lazy.

**Q: Read a 10 GB file safely.**
Don't call `read()`/`readlines()`. Iterate the file object line by line, which streams one line at a time:
```python
with open("huge.txt", encoding="utf-8") as f:
    for line in f:
        process(line)
```
Memory stays roughly constant regardless of file size. For fixed-size chunks (binary), use `iter(lambda: f.read(8192), b"")`.

**Q (Why): Always pass `encoding="utf-8"`.**
Without it, Python uses the platform's *default* encoding (UTF-8 on Linux/macOS, often cp1252 on Windows). The same code then reads/writes different bytes on different machines, producing garbled text ("mojibake") or `UnicodeDecodeError`. Explicit `encoding="utf-8"` makes behavior deterministic and portable.

**Q (Scenario): Safe append log.**
```python
with open("audit.log", "a", encoding="utf-8") as f:
    f.write(f"{event}\n")
```
`a` mode appends without truncating existing entries and creates the file if absent; `with` guarantees the write is flushed and the handle closed even if the process errors afterward.

## Common Mistakes

- Using `w` when you meant `a`, wiping the file.
- Forgetting to close (no `with`) → lost buffered data / leaked handles.
- Not specifying `encoding` → cross-platform bugs.
- Loading a huge file with `read()`/`readlines()` and running out of memory.
- Forgetting `newline=""` with the `csv` module (can double newlines on Windows).

## Related Concepts

[Context Managers](#29-context-managers), [Exception Handling](#22-exception-handling) (I/O errors), [Strings](#10-strings--regular-expressions) (parsing), `csv`/`json`, [Generators](#27-generators) (streaming).

---

# 12. Modules & Packages

## What is it?

A **module** is a single `.py` file containing reusable code (functions, classes, variables). A **package** is a directory of modules (traditionally with an `__init__.py`). The **Standard Library** is the huge set of modules shipped with Python. **`pip`** installs third-party packages from PyPI, and a **virtual environment (`venv`)** isolates a project's dependencies. **PEP 8** is the style guide covering naming conventions.

## Why is it needed?

Beyond a few hundred lines, one file becomes unmanageable. Modules let you **split code by responsibility** and reuse it across projects via `import`. Packages group related modules. Virtual environments prevent "dependency hell" — project A needing `requests 2.20` and project B needing `2.31` can coexist because each has its own isolated environment.

## How does it work?

```python
import math                  # whole module; use math.sqrt
from math import sqrt, pi    # import specific names
from math import sqrt as s   # alias
import numpy as np           # conventional alias
from mypackage.utils import helper   # from a package

# A module can be run OR imported:
def main():
    ...
if __name__ == "__main__":   # runs only when executed directly, not when imported
    main()
```

```
myproject/
├── mypackage/
│   ├── __init__.py     # marks it a package; runs on import
│   ├── core.py
│   └── utils.py
└── main.py
```

Environment workflow:
```bash
python -m venv .venv          # create isolated env
source .venv/bin/activate     # activate (Windows: .venv\Scripts\activate)
pip install requests          # installs into the venv only
pip freeze > requirements.txt # snapshot deps
pip install -r requirements.txt
```

**PEP 8 naming:** `snake_case` for functions/variables/modules, `PascalCase` for classes, `UPPER_SNAKE` for constants, leading underscore `_private` for internal, and 4-space indentation.

**Key standard-library modules:** `os`/`pathlib` (files/paths), `sys` (interpreter), `math`/`random`, `datetime`/`time`, `collections` (Counter, defaultdict, deque, namedtuple), `itertools`, `functools`, `re`, `json`/`csv`, `logging`, `typing`.

## Internal Working

- On `import x`, Python: (1) checks `sys.modules` cache — if already imported, reuses it (modules are **singletons**, imported **once** per process); (2) else searches `sys.path` (current dir, `PYTHONPATH`, site-packages) for the module; (3) executes the module top-to-bottom, binding its names into a new module object; (4) caches compiled bytecode in `__pycache__/*.pyc` to skip recompilation next time.
- `__name__` is set to `"__main__"` when a file is run directly, and to the module's name when imported — enabling the `if __name__ == "__main__"` guard so import doesn't trigger a script's side effects.
- `venv` creates a directory with its own `python` and `site-packages`; activating it prepends that to `PATH` so `pip`/`python` resolve to the isolated copy.

## Advantages

- Organizes and reuses code; clear namespacing avoids name clashes.
- Modules imported once and cached → efficient.
- venv gives reproducible, isolated dependencies per project.
- Massive standard library ("batteries included").

## Limitations

- **Circular imports** (A imports B, B imports A) cause errors and need restructuring.
- `from module import *` pollutes the namespace and hides origins (discouraged).
- Global module state can cause hidden coupling.
- Dependency/version management still requires discipline (lockfiles).

## Real-world Applications

Every non-trivial codebase is organized into packages. venv/requirements power reproducible deployments and CI. The standard library alone covers dates, JSON, HTTP, subprocess, and more without third-party code — critical for scripting and automation.

## Interview Questions

- **Beginner:** Difference between a module and a package?
- **Beginner:** What does `import` actually do?
- **Intermediate:** What is `if __name__ == "__main__"` for?
- **Intermediate:** Why use a virtual environment?
- **Advanced:** What happens on a circular import and how do you fix it?
- **"Why":** Why is a module imported only once even if imported in many files?
- **Comparison:** `import x` vs `from x import y` vs `from x import *`.
- **Scenario:** Two projects need different versions of the same library. Solve it.

## Model Answers

**Q: What does `import` do?**
It (1) looks in `sys.modules` and reuses the module if already loaded; (2) otherwise finds the file on `sys.path`, (3) **executes it once** creating a module object whose top-level names become attributes, (4) caches it in `sys.modules` and its bytecode in `__pycache__`, and (5) binds the chosen name(s) in your namespace. Because of the cache, re-importing the same module elsewhere doesn't re-run it.

**Q: `if __name__ == "__main__"` purpose?**
`__name__` equals `"__main__"` only when the file is **run directly**, and equals the module's own name when it is **imported**. Guarding your script's entry point with this check means the code runs when you execute the file but **not** when another module imports it — so you can both reuse a module's functions and give it a runnable CLI without side effects on import.

**Q: Why is a module imported only once?**
The first import executes the module and stores the resulting object in `sys.modules`. Every subsequent `import` (anywhere in the process) finds it in that cache and reuses the same object. This makes imports cheap and gives modules **singleton** semantics — module-level state is shared process-wide, which is why module globals act like a namespace-scoped singleton.

**Q: Circular imports.**
If `a.py` does `import b` at the top and `b.py` does `import a`, then while `a` is still executing its import of `b`, `b` tries to import the not-yet-finished `a` and finds it partially initialized — often an `ImportError` or `AttributeError`. Fixes: restructure to remove the cycle (extract shared code into a third module), import **inside the function** where it's needed (deferred import), or import the module object (`import a`) rather than names (`from a import x`).

**Q (Scenario): Different library versions per project.**
Give each project its own virtual environment: `python -m venv .venv` in each, activate it, and `pip install` the required version there. Each venv has an isolated `site-packages`, so Project A's `requests==2.20` and Project B's `requests==2.31` never collide. Pin them in each project's `requirements.txt` for reproducibility.

## Common Mistakes

- Forgetting the `__main__` guard, so imports trigger script execution.
- `from module import *` causing name clashes and unclear origins.
- Installing packages globally instead of into a venv.
- Naming your file the same as a stdlib module (`random.py`, `email.py`) — it shadows the real one.
- Not pinning versions, leading to "works on my machine".

## Related Concepts

[Functions](#5-functions), [Standard Library modules], PEP 8 (naming), packaging/`pip`, [Type Hints](#32-type-hints) (`typing` module).

# 13. OOP Fundamentals

## What is it?

Object-Oriented Programming models a program as **objects** — bundles of **data (attributes)** and **behavior (methods)** — created from blueprints called **classes**. Core vocabulary: a **class** defines structure; an **object/instance** is a concrete thing made from it; the **constructor `__init__`** initializes a new instance; `self` is the reference to the current instance; **instance variables** are per-object; **class variables** are shared by all instances.

## Why is it needed?

As programs grow, loose functions and global data become tangled. OOP **groups related state and behavior together** ("a `BankAccount` knows its balance and how to deposit"), enabling **modeling of real-world entities**, **encapsulation** of internal details, and **reuse via inheritance**. It's the dominant paradigm for large, maintainable systems.

## How does it work?

```python
class BankAccount:
    bank_name = "PyBank"          # CLASS variable — shared by all instances

    def __init__(self, owner, balance=0):   # constructor
        self.owner = owner        # INSTANCE variable — unique per object
        self.balance = balance

    def deposit(self, amount):    # instance method — 'self' is the instance
        self.balance += amount
        return self.balance

    @classmethod
    def from_dict(cls, d):        # alternative constructor; 'cls' is the class
        return cls(d["owner"], d["balance"])

    @staticmethod
    def is_valid_amount(x):       # no self/cls — just namespaced utility
        return x > 0

acct = BankAccount("Sam", 100)    # __init__ runs
acct.deposit(50)                  # acct.balance -> 150
BankAccount.bank_name             # "PyBank"
```

- **Instance method** (`self`): operates on a specific object.
- **`@classmethod`** (`cls`): operates on the class; common for alternative constructors.
- **`@staticmethod`**: a plain function grouped inside the class; no access to instance/class.

## Internal Working

- A class is itself an **object** (an instance of `type`). Defining a class runs its body once, building a namespace (`__dict__`) of its methods and class variables.
- `acct = BankAccount("Sam")` triggers `__new__` (allocates the object) then `__init__` (initializes it). `self` is just the first positional argument Python auto-passes — `acct.deposit(50)` is sugar for `BankAccount.deposit(acct, 50)`.
- **Attribute lookup order:** `acct.x` checks the **instance `__dict__` first**, then the **class**, then base classes (the MRO). This is why an instance variable *shadows* a class variable of the same name.
- Assigning `acct.bank_name = "X"` creates a **new instance variable** on `acct`, leaving the class variable and other instances untouched — a classic source of confusion.

## Advantages

- Bundles state + behavior → models the problem domain naturally.
- Encapsulation hides internals; inheritance/polymorphism enable reuse and extension.
- Easier to maintain and reason about large systems in terms of collaborating objects.

## Limitations

- Overkill for small scripts; adds ceremony.
- Poorly designed hierarchies become rigid ("inheritance hell").
- Shared mutable class variables and `self` mistakes cause subtle bugs.

## Real-world Applications

Django/Flask models, ORM entities, GUI widgets, game entities, API client classes, ML estimators (`model.fit()`), and virtually every framework you'll use exposes an object-oriented API.

## Interview Questions

- **Beginner:** Difference between a class and an object?
- **Beginner:** What is `self`?
- **Intermediate:** Difference between instance variables and class variables?
- **Intermediate:** `@classmethod` vs `@staticmethod` vs instance method?
- **Advanced:** What's the difference between `__new__` and `__init__`?
- **"Why":** Why does modifying `self.x` not affect other instances but modifying a class variable can?
- **Comparison:** When would you use a classmethod as an alternative constructor?
- **Scenario:** You set a class variable to `[]` and every instance shares it. Explain and fix.

## Model Answers

**Q: Class vs object.**
A **class** is a blueprint/template describing what attributes and methods instances will have (`BankAccount`). An **object** is a concrete instance created from that blueprint, with its own data (`acct = BankAccount("Sam")`). One class → many independent objects.

**Q: Instance vs class variables.**
**Instance variables** (`self.balance`) are stored per-object in the instance's `__dict__` and differ between objects. **Class variables** (`bank_name`) live on the class and are **shared by all instances**; changing them via the class affects everyone. Reads fall back to the class if the instance has no such attribute, but assigning through an instance creates a shadowing instance variable rather than changing the class.

**Q: classmethod vs staticmethod vs instance method.**
- **Instance method** takes `self`; needs a specific object's data.
- **classmethod** takes `cls`; operates on the class — ideal for **alternative constructors** (`Date.from_string(...)`) and for correctly returning subclass instances.
- **staticmethod** takes neither; it's a utility logically grouped in the class but independent of instance/class state (e.g., a validation helper).

**Q: `__new__` vs `__init__`.**
`__new__(cls, ...)` **creates and returns** the new (uninitialized) instance — it's the actual constructor and runs first. `__init__(self, ...)` **initializes** that already-created instance (sets attributes) and returns `None`. You rarely override `__new__` — only for immutable types (subclassing `int`/`tuple`/`str`), singletons, or metaclass tricks. Everyday initialization goes in `__init__`.

**Q (Scenario): Shared mutable class variable.**
```python
class Cart:
    items = []          # BUG: one list shared by ALL carts
c1, c2 = Cart(), Cart()
c1.items.append("apple")
c2.items              # ['apple']  <- shared!
```
Because `items` is a class variable, both instances reference the same list. Fix by making it an **instance** variable in `__init__`:
```python
class Cart:
    def __init__(self):
        self.items = []   # fresh list per instance
```
Rule: mutable per-object state belongs in `__init__`, not as a class variable.

## Common Mistakes

- Forgetting `self` in method definitions or when accessing attributes.
- Using a mutable class variable for per-instance state (shared-list bug).
- Assigning through an instance and unexpectedly shadowing a class variable.
- Confusing `__init__` with a "constructor that creates" (it initializes; `__new__` creates).
- Treating classmethods/staticmethods interchangeably.

## Related Concepts

[Encapsulation](#14-encapsulation), [Inheritance](#16-inheritance), [Magic Methods](#18-magic--dunder-methods), [Dataclasses](#21-dataclasses), [Properties](#20-properties).

---

# 14. Encapsulation

## What is it?

Encapsulation is **bundling data and the methods that operate on it inside a class, and controlling access to the internal state**. Python signals intended visibility by **naming convention**: `public` (normal names), `_protected` (single leading underscore — "internal, don't touch"), and `__private` (double leading underscore — triggers **name mangling**). Python has **no true access modifiers** like Java's `private`; it relies on convention plus one mechanism (name mangling).

## Why is it needed?

To protect **invariants** and hide implementation details. If any code can set `account.balance = -999`, you can't guarantee the balance is never negative. By exposing methods (`deposit`, `withdraw`) and treating the raw field as internal, you keep control and can change the internal representation later without breaking callers.

## How does it work?

```python
class Account:
    def __init__(self, balance):
        self.owner = "Sam"        # public
        self._internal_id = 42    # protected: convention only ("please don't")
        self.__balance = balance  # private: name-mangled

    def get_balance(self):        # controlled access
        return self.__balance

    def deposit(self, amt):
        if amt <= 0:
            raise ValueError("amount must be positive")   # enforce invariant
        self.__balance += amt

a = Account(100)
a.owner            # OK, public
a._internal_id     # works, but you're breaking convention
# a.__balance      # AttributeError!
a._Account__balance  # 100  <- name mangling: __balance -> _ClassName__balance
```

## Internal Working

- **Single underscore `_x`** is *pure convention* — the interpreter does nothing special; it just tells humans "internal".
- **Double underscore `__x`** triggers **name mangling**: inside class `Account`, `self.__balance` is rewritten by the compiler to `self._Account__balance`. This is **not security** — you can still access `a._Account__balance` — its real purpose is to **avoid accidental name clashes** in inheritance (a subclass's `__balance` won't collide with the parent's).
- Encapsulation is often paired with **`@property`** (see [Properties](#20-properties)) to expose managed, validated access that *looks* like a plain attribute.

## Advantages

- Protects invariants and hides implementation, allowing internal changes without breaking callers.
- Centralizes validation (one place enforces the rules).
- Name mangling prevents subclass attribute collisions.

## Limitations

- **Not enforced** — "private" is honor-system; determined code can bypass it. Python's philosophy is "we're all consenting adults."
- Name mangling can confuse newcomers and complicate debugging/serialization.
- Over-encapsulation (getters/setters for everything) is un-Pythonic; use `@property` only when you need logic.

## Real-world Applications

Library/API design (public surface vs `_internal` helpers), ORM models protecting persistence fields, financial/domain classes enforcing invariants (non-negative balances), and framework internals prefixed with `_` to discourage external use.

## Interview Questions

- **Beginner:** What do single and double leading underscores mean?
- **Beginner:** Does Python have true private variables?
- **Intermediate:** What is name mangling and what problem does it solve?
- **Intermediate:** How is encapsulation typically combined with `@property`?
- **Advanced:** Can you access a "private" attribute from outside the class? Show how.
- **"Why":** Why doesn't Python enforce access control like Java?
- **Comparison:** `_protected` vs `__private` — practical difference.
- **Scenario:** Ensure a temperature object can never hold a value below absolute zero.

## Model Answers

**Q: Single vs double underscore.**
`_name` is a **convention** meaning "protected/internal — use at your own risk"; Python does nothing to enforce it. `__name` triggers **name mangling** to `_ClassName__name`, mainly to avoid attribute-name clashes across an inheritance hierarchy. Neither makes an attribute truly inaccessible.

**Q: Does Python have true private variables?**
No. There is no hard access control. `__x` is mangled (mildly obscured) and `_x` is convention. You *can* always reach an attribute if you know the mangled name (`obj._Class__x`). Python trusts developers rather than enforcing privacy — "we're all consenting adults."

**Q: Name mangling — what problem?**
Inside a class, `self.__x` becomes `self._ClassName__x`. Its purpose is **not** security but **collision avoidance**: if a base class and a subclass both use `__value` internally, mangling keeps them distinct (`_Base__value` vs `_Sub__value`), so a subclass can't accidentally clobber the parent's private attribute.

**Q (Why): Why no enforced access control?**
Python's design philosophy favors flexibility and trusts the programmer. Rigid access modifiers add complexity and are routinely worked around anyway (reflection in Java). Instead Python uses conventions plus tools like `@property` to provide *managed* access where it matters, keeping the language simple while still communicating intent.

**Q (Scenario): Temperature ≥ absolute zero.**
Encapsulate the field and validate on write, typically via a property:
```python
class Temperature:
    def __init__(self, celsius):
        self.celsius = celsius          # goes through the setter
    @property
    def celsius(self):
        return self._celsius
    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("below absolute zero")
        self._celsius = value
```
Now every assignment — even in `__init__` — is validated, and the invariant can never be violated through the public interface.

## Common Mistakes

- Believing `__x` makes an attribute truly private/secure.
- Overusing Java-style getters/setters instead of plain attributes + `@property` when needed.
- Being surprised that `__x` "disappears" (it's mangled) when debugging or serializing.
- Accessing others' `_internal` attributes and coupling to implementation details.

## Related Concepts

[Properties](#20-properties), [OOP Fundamentals](#13-oop-fundamentals), [Abstraction](#15-abstraction), name mangling.

---

# 15. Abstraction

## What is it?

Abstraction means **exposing only the essential interface and hiding the complex implementation**. In Python it's formalized with **abstract base classes (ABCs)** via the `abc` module: an **abstract class** cannot be instantiated and declares **abstract methods** that subclasses *must* implement. It answers "*what* an object does" while deferring "*how*".

## Why is it needed?

To define **contracts**. If every payment processor must provide a `pay(amount)` method, you declare an abstract `PaymentProcessor` with an abstract `pay`. Any concrete processor (`StripeProcessor`, `PayPalProcessor`) must implement it, and the rest of your code can rely on the interface without caring about the concrete class — enabling **polymorphism** and **loose coupling**.

## How does it work?

```python
from abc import ABC, abstractmethod

class Shape(ABC):                 # abstract base class
    @abstractmethod
    def area(self):               # contract: subclasses MUST implement
        ...

    def describe(self):           # concrete method — shared behavior
        return f"A shape with area {self.area()}"

class Circle(Shape):
    def __init__(self, r): self.r = r
    def area(self):               # required implementation
        return 3.14159 * self.r ** 2

# Shape()          -> TypeError: Can't instantiate abstract class
c = Circle(2)
c.area()           # 12.566...
c.describe()       # inherited concrete method uses the subclass's area()
```

## Internal Working

- `ABC` uses the `ABCMeta` **metaclass**. When you subclass and try to instantiate, `ABCMeta` checks whether **all** `@abstractmethod`-decorated names have been overridden. If any remain abstract, instantiation raises `TypeError`.
- `@abstractmethod` simply flags a method by setting `__isabstractmethod__ = True`; the metaclass collects these into `__abstractmethods__`.
- ABCs also enable **virtual subclassing** (`register`) and back `isinstance` checks against interfaces like `collections.abc.Iterable` — that's how `isinstance(x, Iterable)` works without explicit inheritance.

## Advantages

- Enforces a consistent interface/contract across implementations.
- Enables polymorphism and dependency inversion (code depends on the abstraction, not concretes).
- Can mix abstract (must-implement) and concrete (shared) methods.

## Limitations

- Adds boilerplate; for simple cases Python's **duck typing** already provides informal abstraction without ABCs.
- Overusing deep abstract hierarchies can over-engineer a small program.
- `Protocol` (structural typing, 3.8+) is often a lighter alternative to nominal ABCs.

## Real-world Applications

Plugin architectures (each plugin implements a base interface), payment/notification/storage backends behind a common ABC, the `collections.abc` hierarchy (`Sequence`, `Mapping`, `Iterable`) that your own containers can plug into, and framework base classes like Django's `BaseCommand`.

## Interview Questions

- **Beginner:** What is an abstract class? Can you instantiate one?
- **Beginner:** What does `@abstractmethod` do?
- **Intermediate:** Difference between abstraction and encapsulation?
- **Intermediate:** Can an abstract class have concrete (implemented) methods?
- **Advanced:** How does Python enforce that abstract methods are implemented?
- **"Why":** Why use an ABC instead of just relying on duck typing?
- **Comparison:** Abstract class vs interface (and Python's `Protocol`).
- **Scenario:** Design a plugin system where every plugin must implement `run()`.

## Model Answers

**Q: What is an abstract class; can you instantiate it?**
An abstract class (subclass of `ABC` with ≥1 `@abstractmethod`) defines a **contract** but is **incomplete**, so it cannot be instantiated — attempting `Shape()` raises `TypeError`. Its purpose is to be subclassed; the subclass becomes instantiable only after implementing every abstract method.

**Q: Abstraction vs encapsulation.**
They're complementary. **Encapsulation** is about **hiding internal state** and bundling data with methods (the *how* is protected). **Abstraction** is about **exposing a simplified interface** and hiding complexity (the *what* is emphasized). Encapsulation is a mechanism (access control/bundling); abstraction is a design goal (define a clean contract). ABCs implement abstraction; underscores/`@property` implement encapsulation.

**Q: How does Python enforce abstract-method implementation?**
Through the `ABCMeta` metaclass. It records all `@abstractmethod` names in `__abstractmethods__`. At instantiation time it checks that set is empty for the concrete class; if any abstract method wasn't overridden, it raises `TypeError`. So enforcement happens **at object-creation time**, not at class-definition time.

**Q (Why): ABC vs duck typing.**
Duck typing ("if it has `.run()`, call it") is informal and only fails **at call time** with an `AttributeError`, possibly deep in execution. An ABC makes the contract **explicit and eager**: a class missing `run()` can't even be instantiated, catching the error at the boundary and documenting the required interface. Use ABCs for stable, public contracts; duck typing/`Protocol` for lightweight or third-party-friendly designs.

**Q (Scenario): Plugin system requiring `run()`.**
```python
from abc import ABC, abstractmethod
class Plugin(ABC):
    @abstractmethod
    def run(self): ...

class Backup(Plugin):
    def run(self): print("backing up")

def execute(plugins: list[Plugin]):
    for p in plugins:
        p.run()               # guaranteed to exist
```
Any plugin that forgets `run()` fails to instantiate, so the registry can never contain a broken plugin — the contract is enforced up front.

## Common Mistakes

- Forgetting to subclass `ABC` (then `@abstractmethod` doesn't prevent instantiation).
- Thinking the error occurs at class definition (it's at instantiation).
- Overusing abstract hierarchies where duck typing/`Protocol` suffices.
- Not implementing *all* abstract methods and being surprised the subclass is still abstract.

## Related Concepts

[Inheritance](#16-inheritance), [Polymorphism](#17-polymorphism), [Encapsulation](#14-encapsulation), `abc`, [Type Hints](#32-type-hints) (`Protocol`), duck typing.

---

# 16. Inheritance

## What is it?

Inheritance lets a **child (derived) class reuse and extend** a **parent (base) class**'s attributes and methods, modeling an **"is-a"** relationship (`Dog` **is an** `Animal`). Python supports **single**, **multiple**, **multilevel**, and **hierarchical** inheritance, and resolves method lookups via the **MRO (Method Resolution Order)**.

## Why is it needed?

To avoid duplicating shared behavior. Common logic goes in the base class; specializations override or add. It enables polymorphism (treat all subclasses uniformly through the base type) and organizes code into taxonomies.

## How does it work?

```python
class Animal:
    def __init__(self, name): self.name = name
    def speak(self): return "..."

class Dog(Animal):                 # single inheritance
    def speak(self):               # override
        return "Woof"

class Puppy(Dog):                  # multilevel: Puppy -> Dog -> Animal
    def speak(self):
        return super().speak() + " (tiny)"   # call parent version

# Multiple inheritance
class Swimmer:  def swim(self): return "swim"
class Flyer:    def fly(self):  return "fly"
class Duck(Swimmer, Flyer):        # inherits from both
    pass

Duck.__mro__      # (Duck, Swimmer, Flyer, object) — lookup order
```

- **Single:** one parent. **Multilevel:** chain (grandparent→parent→child). **Hierarchical:** many children share one parent. **Multiple:** one child, several parents.
- `super()` delegates to the **next class in the MRO**, not literally "the parent" — crucial in multiple inheritance.

## Internal Working

- Every class has an **MRO**, computed by the **C3 linearization** algorithm, listing the order Python searches classes for an attribute/method. `Duck.__mro__` (or `Duck.mro()`) shows it. Lookup walks the MRO left-to-right and stops at the first match.
- `super()` returns a proxy that dispatches to the **next class in the MRO after the current one**, enabling **cooperative multiple inheritance** (each class's method calls `super()` so every ancestor runs exactly once — the "diamond problem" solution).
- All classes ultimately inherit from **`object`**, the root of the hierarchy.

## Advantages

- Code reuse and a clear "is-a" taxonomy.
- Polymorphism: uniform handling of subclasses via the base interface.
- `super()` + MRO handle complex multiple inheritance predictably.

## Limitations

- **Tight coupling:** subclasses depend on the parent's implementation; base-class changes can break children ("fragile base class").
- Deep hierarchies are hard to follow.
- Multiple inheritance can be confusing (MRO surprises, name clashes) — often **composition is preferable** ("favor composition over inheritance").

## Real-world Applications

Framework base classes (Django `Model`, `View`; exception hierarchies), GUI widget trees, ML estimator base classes, and the built-in exception hierarchy (`Exception` → `ArithmeticError` → `ZeroDivisionError`).

## Interview Questions

- **Beginner:** What is inheritance and the "is-a" relationship?
- **Beginner:** What does `super()` do?
- **Intermediate:** Explain single vs multiple vs multilevel vs hierarchical inheritance.
- **Intermediate:** What is the diamond problem and how does Python solve it?
- **Advanced:** What is the MRO and which algorithm computes it?
- **"Why":** Why is "composition over inheritance" often recommended?
- **Comparison:** Method overriding vs overloading.
- **Scenario:** A `Duck` should be both a `Swimmer` and a `Flyer`. Design it and explain lookup order.

## Model Answers

**Q: What does `super()` do?**
`super()` returns a proxy that calls the method of the **next class in the MRO**, letting a subclass extend rather than fully replace parent behavior (`super().__init__(...)` to run the parent's initializer, then add to it). In multiple inheritance it's not "call my parent" but "call the next class in the linearized order," which is what makes cooperative inheritance work.

**Q: Diamond problem and Python's solution.**
The diamond problem: `D` inherits from `B` and `C`, both of which inherit from `A`. Which `A` is used, and does `A.__init__` run twice? Python solves it with the **C3 linearization MRO**, producing a single consistent order (`D, B, C, A, object`) where each class appears **once**. If every class calls `super().__init__()`, each ancestor's method runs exactly once, in MRO order — no duplication, no ambiguity.

**Q: What is the MRO?**
The Method Resolution Order is the deterministic sequence of classes Python searches when resolving an attribute/method, computed by **C3 linearization**. It guarantees a child precedes its parents and preserves the order parents are listed, while keeping each class once. Inspect it via `Cls.__mro__`.

**Q (Why): Composition over inheritance.**
Inheritance creates tight coupling — the subclass is bound to the parent's internals, and changes ripple down (fragile base class). It also forces an "is-a" model even when the real relationship is "has-a." **Composition** (holding another object and delegating) is more flexible: you can swap components, mix behaviors, and change them at runtime without reworking a class tree. Prefer inheritance for genuine is-a specialization; prefer composition for reuse of behavior.

**Q: Overriding vs overloading.**
**Overriding**: a subclass redefines a method inherited from the parent (same name, same object) — Python fully supports this. **Overloading**: multiple methods with the same name but different signatures — Python does **not** support it natively (a later `def` replaces an earlier one). You emulate it with default/`*args` parameters or `functools.singledispatch`.

## Common Mistakes

- Forgetting `super().__init__()`, leaving the parent's state uninitialized.
- Assuming `super()` means "the direct parent" in multiple inheritance.
- Building deep/fragile hierarchies where composition fits better.
- Name clashes across multiple parents resolved unexpectedly by the MRO.
- Confusing overriding with (nonexistent) overloading.

## Related Concepts

[Polymorphism](#17-polymorphism), [Abstraction](#15-abstraction), [Composition & Aggregation](#19-class-relationships-composition--aggregation), MRO/`super`, [Magic Methods](#18-magic--dunder-methods).

---

# 17. Polymorphism

## What is it?

Polymorphism ("many forms") means a **single interface works with different types**, each responding in its own way. In Python it shows up as **method overriding** (subclasses redefine a base method), **duck typing** (any object with the right method fits, regardless of class), and **operator/dunder polymorphism** (`+`, `len()`, `str()` behave per type). Python emphasizes duck typing over signature-based overloading.

## Why is it needed?

To write **generic, extensible code**. A function `render(shapes)` can call `s.area()` on circles, squares, and triangles uniformly — and still work for a new shape you add tomorrow, without modifying `render`. This is the **Open/Closed Principle**: open to extension, closed to modification.

## How does it work?

```python
# Method overriding + polymorphic use
class Cat:  def speak(self): return "Meow"
class Dog:  def speak(self): return "Woof"

for animal in [Cat(), Dog()]:
    print(animal.speak())      # same call, different behavior

# Duck typing: no shared base class needed
def make_it_speak(thing):
    return thing.speak()       # works for ANYTHING with .speak()

# Built-in polymorphism
len("abc"); len([1,2]); len({1:2})   # len() works on many types
"a" + "b"; [1] + [2]; 3 + 4          # + adapts per type
```

## Internal Working

- Method calls are resolved **dynamically at runtime** via the object's type and MRO — Python looks up `speak` on the actual object, so the correct override runs (late binding).
- **Duck typing** works because Python never checks a declared type; it just attempts the attribute/method access. "If it walks like a duck and quacks like a duck, it's a duck." Failure surfaces as `AttributeError` only if the method is missing.
- **Built-in polymorphism** is powered by dunder methods: `len(x)` calls `x.__len__()`, `x + y` calls `x.__add__(y)`, `str(x)` calls `x.__str__()`. Any class implementing these participates (see [Magic Methods](#18-magic--dunder-methods)).

## Advantages

- Generic, reusable, extensible code (add new types without changing callers).
- Duck typing reduces boilerplate — no forced inheritance/interfaces.
- Uniform APIs (`len`, iteration, `+`) across built-ins and custom types.

## Limitations

- Duck typing errors appear only **at runtime** (no compile-time interface check) — mitigated by tests and `Protocol`/type hints.
- Can obscure what types a function actually accepts (documentation matters).
- Ad-hoc overloading isn't native (need `singledispatch`).

## Real-world Applications

File-like objects (anything with `.read()`/`.write()` works with code expecting a file), the iterator protocol (any `__iter__` works in `for`), serialization frameworks, and plugin systems where diverse classes share a method name.

## Interview Questions

- **Beginner:** What is polymorphism? Give a built-in example.
- **Beginner:** What is method overriding?
- **Intermediate:** What is duck typing?
- **Intermediate:** Does Python support method overloading? How do you emulate it?
- **Advanced:** How do `len()` and `+` achieve polymorphism internally?
- **"Why":** Why does duck typing suit a dynamic language like Python?
- **Comparison:** Overriding vs overloading vs duck typing.
- **Scenario:** Write a function that processes any object that can be "drawn" without a shared base class.

## Model Answers

**Q: What is duck typing?**
Duck typing is Python's style of polymorphism where an object's **suitability is determined by the presence of the required methods/attributes, not by its class or any declared interface**. If an object has a `.speak()` method, `make_it_speak()` works on it — whether it's a `Dog`, a `Robot`, or a mock. "If it quacks like a duck, treat it as a duck." The trade-off: mismatches are caught at runtime, not compile time.

**Q: Does Python support overloading?**
Not by signature — you can't define two methods with the same name and different parameter lists; the second definition simply replaces the first. You emulate overloading with **default arguments**, **`*args`/`**kwargs`** and internal type checks, or **`functools.singledispatch`** (dispatch on the first argument's type). Operator "overloading," by contrast, *is* supported via dunder methods.

**Q: How do `len()`/`+` achieve polymorphism?**
They delegate to dunder methods on the operand. `len(x)` calls `type(x).__len__(x)`; `x + y` calls `type(x).__add__(x, y)` (falling back to `y.__radd__`). Because the behavior lives on each type, the same syntax produces type-appropriate results, and your own classes join in simply by defining `__len__`/`__add__`. This is polymorphism through a shared protocol.

**Q (Scenario): Draw anything drawable.**
```python
def render(objects):
    for obj in objects:
        obj.draw()        # duck typing: any object with draw() works

class Circle:  def draw(self): print("O")
class Square:  def draw(self): print("[]")
render([Circle(), Square()])
```
No base class is required — any object exposing `draw()` participates. To document/enforce the contract you could add a `Protocol` type hint (`class Drawable(Protocol): def draw(self)->None: ...`), keeping the flexibility while enabling static checks.

## Common Mistakes

- Assuming Python supports signature-based overloading.
- Relying on duck typing without tests, so missing methods blow up in production.
- Forgetting that overriding requires the **same method name** to take effect.
- Not implementing the needed dunder to plug into built-in polymorphism.

## Related Concepts

[Inheritance](#16-inheritance), [Abstraction](#15-abstraction), [Magic Methods](#18-magic--dunder-methods), [Type Hints](#32-type-hints) (`Protocol`), `functools.singledispatch`.

---

# 18. Magic / Dunder Methods

## What is it?

**Dunder** ("double underscore") or **magic methods** are specially named methods like `__init__`, `__str__`, `__len__`, `__eq__`, `__add__` that Python calls **implicitly** in response to syntax and built-in functions. They let your objects integrate with the language's operators and protocols — the mechanism behind operator overloading and duck typing.

## Why is it needed?

So custom objects behave like built-ins. Implement `__len__` and `len(obj)` works; implement `__getitem__` and `obj[i]` and iteration work; implement `__eq__`/`__lt__` and comparisons/sorting work. This makes your classes intuitive and lets them plug into the standard-library machinery.

## How does it work?

```python
class Money:
    def __init__(self, amount, currency="USD"):
        self.amount, self.currency = amount, currency
    def __repr__(self):                    # unambiguous, for developers/debugging
        return f"Money({self.amount!r}, {self.currency!r})"
    def __str__(self):                     # readable, for end users / print()
        return f"{self.amount:.2f} {self.currency}"
    def __eq__(self, other):               # ==
        return (self.amount, self.currency) == (other.amount, other.currency)
    def __lt__(self, other):               # <  (enables sorting)
        return self.amount < other.amount
    def __add__(self, other):              # +
        return Money(self.amount + other.amount, self.currency)
    def __len__(self):                     # len()
        return int(self.amount)
    def __getitem__(self, key): ...        # obj[key]
    def __call__(self, *a): ...            # makes the instance callable: obj()
```

Common dunders: **construction** `__new__`, `__init__`; **representation** `__str__`, `__repr__`; **sizing** `__len__`; **comparison** `__eq__`, `__lt__`, `__gt__`, `__le__`, `__ge__`, `__ne__`; **arithmetic** `__add__`, `__sub__`, `__mul__`; **container** `__getitem__`, `__setitem__`, `__contains__`; **iteration** `__iter__`, `__next__`; **callable** `__call__`; **context manager** `__enter__`, `__exit__`; **hashing** `__hash__`.

## Internal Working

- Python maps syntax/builtins to dunders: `a + b` → `type(a).__add__(a, b)`; `len(a)` → `type(a).__len__(a)`; `a == b` → `__eq__`; `for x in a` → `__iter__`/`__next__`; `a[i]` → `__getitem__`; `a()` → `__call__`; `with a` → `__enter__`/`__exit__`.
- Dunder lookups use the **type**, not the instance (`type(a).__add__`), which is why they must be defined on the class.
- `__repr__` is the fallback for `__str__` (if `__str__` is absent, `print` uses `__repr__`). `__repr__` should ideally be valid Python that recreates the object.
- Defining `__eq__` sets `__hash__` to `None` (object becomes unhashable) unless you also define `__hash__` — because equal objects must hash equally.

## Advantages

- Objects feel native — operators, `len`, iteration, printing all "just work".
- Enables powerful DSLs and library ergonomics (NumPy, pathlib, SQLAlchemy).
- Integrates custom classes with standard-library protocols (sorting, sets, `with`).

## Limitations

- Overloading operators unintuitively hurts readability (`+` should mean something like addition).
- Easy to introduce inconsistencies (`__eq__` without `__hash__`; `__lt__` disagreeing with `__eq__`).
- Too many dunders can obscure a simple class — consider `dataclass` which generates several for you.

## Real-world Applications

`__repr__`/`__str__` in every debuggable class; `__eq__`/`__hash__` for value objects used in sets/dicts; `__enter__`/`__exit__` for resource managers; `__iter__`/`__next__` for custom iterables; `__call__` for function-like objects (decorators-as-classes, stateful callbacks); operator dunders in numeric/vector libraries.

## Interview Questions

- **Beginner:** Difference between `__str__` and `__repr__`?
- **Beginner:** What does `__init__` do — is it the constructor?
- **Intermediate:** How do you make your object usable with `len()` and `in`?
- **Intermediate:** What happens if you define `__eq__` but not `__hash__`?
- **Advanced:** What does `__call__` enable? Give a use case.
- **"Why":** Why does `a + b` work for both ints and your custom class?
- **Comparison:** `__new__` vs `__init__`.
- **Scenario:** Make instances of your `Vector` class sortable and printable.

## Model Answers

**Q: `__str__` vs `__repr__`.**
`__str__` produces a **readable, user-facing** string (used by `print()`/`str()`); `__repr__` produces an **unambiguous, developer-facing** string (used by the REPL, `repr()`, and containers), ideally valid Python that could recreate the object: `Money(5.0, 'USD')`. If `__str__` is missing, Python falls back to `__repr__` — so **always define `__repr__`** for debuggability, and add `__str__` only when a friendlier display is needed.

**Q: `__eq__` without `__hash__`.**
Defining `__eq__` makes Python set `__hash__` to `None`, so the object becomes **unhashable** and can't be used in a set or as a dict key (`TypeError: unhashable type`). This is deliberate: if two objects compare equal they must hash equally, and Python can't infer a correct hash from your custom `__eq__`. Provide a matching `__hash__` (e.g. `hash((self.a, self.b))`) for value objects, or use `@dataclass(frozen=True)` which generates both.

**Q: What does `__call__` enable?**
It makes an **instance callable like a function** — `obj()` invokes `obj.__call__()`. Use cases: stateful callbacks/accumulators, function objects that carry configuration (e.g. a `Multiplier(3)` you can call as `triple(x)`), memoizers, and class-based decorators. It blurs the line between objects and functions while keeping per-instance state.

**Q (Scenario): Sortable + printable Vector.**
```python
from functools import total_ordering
@total_ordering
class Vector:
    def __init__(self, x, y): self.x, self.y = x, y
    def __repr__(self): return f"Vector({self.x}, {self.y})"
    def _mag(self): return (self.x**2 + self.y**2) ** 0.5
    def __eq__(self, o): return self._mag() == o._mag()
    def __lt__(self, o): return self._mag() < o._mag()

sorted([Vector(3,4), Vector(1,1)])   # sorts by magnitude; repr prints nicely
```
`__lt__` + `__eq__` (plus `@total_ordering` to fill in `>`, `<=`, `>=`) make it sortable; `__repr__` makes it printable.

## Common Mistakes

- Defining only `__str__` and getting ugly `<object at 0x...>` in lists (define `__repr__`).
- Defining `__eq__` without `__hash__` and losing hashability.
- Overloading operators in surprising ways.
- Forgetting `__eq__`/`__lt__` must be **consistent** (else sorting/sets misbehave).
- Putting creation logic in `__init__` thinking it's the allocator (that's `__new__`).

## Related Concepts

[OOP Fundamentals](#13-oop-fundamentals), [Operators](#3-operators) (overloading), [Iterators](#26-iterators) (`__iter__`/`__next__`), [Context Managers](#29-context-managers), [Dataclasses](#21-dataclasses), [Dictionaries](#8-dictionaries) (`__hash__`).

---

# 19. Class Relationships (Composition & Aggregation)

## What is it?

Besides inheritance ("is-a"), objects relate through **"has-a"** relationships where one object **contains** others:
- **Composition** — a *strong* "owns-a": the part **cannot exist independently** of the whole and shares its lifetime (a `Car` **has-a** `Engine`; destroy the car, the engine goes with it).
- **Aggregation** — a *weak* "has-a": the part **can exist independently**; the whole just references it (a `Team` **has** `Player`s, but players outlive any one team).

## Why is it needed?

Because "favor composition over inheritance": building objects from smaller collaborating objects is more **flexible** than deep inheritance trees. Composition lets you swap parts, reuse them, and change behavior at runtime without rigid hierarchies — and it models real containment relationships accurately.

## How does it work?

```python
# COMPOSITION: Engine is created and owned by Car; its lifetime is tied to Car
class Engine:
    def start(self): return "Engine started"

class Car:
    def __init__(self):
        self.engine = Engine()          # created INSIDE — owned by the Car
    def start(self):
        return self.engine.start()      # delegation

# AGGREGATION: players exist independently and are passed IN
class Player:
    def __init__(self, name): self.name = name

class Team:
    def __init__(self, players):
        self.players = players          # references EXTERNAL objects

p1, p2 = Player("A"), Player("B")        # exist on their own
team = Team([p1, p2])                    # team merely references them
```

**Rule of thumb:** if the container **creates** the part (composition, owned) vs **receives** it from outside (aggregation, referenced).

## Internal Working

- Both are implemented simply by **storing references** to other objects as attributes; the difference is **conceptual/lifecycle**, not syntactic. Composition typically creates the part in `__init__`; aggregation accepts it as a parameter.
- **Delegation**: the container forwards calls to its parts (`Car.start` calls `engine.start`), exposing a simple interface while reusing the part's behavior.
- Because Python uses reference counting/GC, "lifetime tied to the whole" means when the `Car` is collected and nothing else references the `Engine`, the engine is collected too (composition). In aggregation, external references keep the part alive.

## Advantages

- Flexible and loosely coupled; parts are swappable and independently testable.
- Avoids fragile deep inheritance; combine behaviors freely ("has-a" over "is-a").
- Models real-world containment and ownership accurately.

## Limitations

- More explicit wiring/delegation code than inheritance.
- Can produce many small classes and boilerplate delegation methods.
- The composition/aggregation distinction is conceptual — easy to blur.

## Real-world Applications

A `Car` composed of `Engine`/`Wheels`; a `House` composed of `Room`s; a `Playlist` aggregating `Song`s; a web `Service` composed of a `Logger`, `DBConnection`, and `Cache` injected via constructor (dependency injection is aggregation/composition in practice).

## Interview Questions

- **Beginner:** Difference between "is-a" and "has-a"?
- **Beginner:** Give an example of composition.
- **Intermediate:** Difference between composition and aggregation?
- **Intermediate:** What is delegation?
- **Advanced:** Why is "composition over inheritance" recommended?
- **"Why":** When would you choose aggregation instead of composition?
- **Comparison:** Inheritance vs composition for code reuse.
- **Scenario:** Model a `Computer` and its `CPU`, `RAM` — which relationship, and why?

## Model Answers

**Q: Composition vs aggregation.**
Both are "has-a", differing in **ownership and lifecycle**. **Composition** is a strong ownership: the whole creates and owns the part, and the part's lifetime is bound to the whole (a `Car`'s `Engine`). **Aggregation** is a weak association: the part exists independently and is merely referenced by the whole (a `Team` referencing `Player`s who outlive the team). Implementation-wise, composition usually instantiates the part internally; aggregation receives it from outside.

**Q: What is delegation?**
Delegation is when an object handles a request by **forwarding it to one of its component objects** rather than implementing it itself — `Car.start()` calls `self.engine.start()`. It's the practical glue of composition: the container presents a simple interface and reuses its parts' behavior, achieving reuse **without inheritance**.

**Q (Why): Composition over inheritance.**
Inheritance tightly binds a subclass to a parent's implementation (fragile base class) and locks you into a single "is-a" taxonomy chosen at design time. Composition assembles behavior from interchangeable parts you can swap or reconfigure at runtime, keeps classes small and testable, and avoids MRO/diamond complications. Use inheritance for genuine specialization; use composition for "reuse this capability."

**Q (Scenario): Computer with CPU and RAM.**
Typically **composition**: a `Computer` creates and owns its `CPU` and `RAM`, whose lifecycle is tied to the machine.
```python
class CPU: ...
class RAM: ...
class Computer:
    def __init__(self):
        self.cpu = CPU()      # owned
        self.ram = RAM()
```
If instead components were pooled/shared and moved between machines, you'd model it as **aggregation** (pass them in). The deciding question is whether the parts are owned by and lifetime-bound to the whole.

## Common Mistakes

- Reaching for inheritance when the relationship is really "has-a".
- Confusing composition (owned, created inside) with aggregation (referenced, passed in).
- Overusing delegation boilerplate instead of exposing the component directly when appropriate.
- Treating the distinction as syntactic rather than about lifecycle/ownership.

## Related Concepts

[Inheritance](#16-inheritance) (is-a), [OOP Fundamentals](#13-oop-fundamentals), dependency injection, [Dataclasses](#21-dataclasses), [Polymorphism](#17-polymorphism).

---

# 20. Properties

## What is it?

A **property** turns a method into a **managed attribute**: `obj.x` looks like plain attribute access but actually runs getter/setter/deleter code behind the scenes. Implemented with the `@property` decorator (getter) plus `@x.setter` and `@x.deleter`. It's Python's Pythonic answer to Java-style getters/setters.

## Why is it needed?

You want the **clean syntax of attribute access** (`temp.celsius`) but with **validation, computation, or read-only behavior** underneath. Properties let you start with a plain public attribute and later add logic **without changing the public interface** — callers keep writing `obj.x`, so no code breaks. This is the "uniform access principle."

## How does it work?

```python
class Temperature:
    def __init__(self, celsius=0):
        self.celsius = celsius            # invokes the setter (validation)

    @property
    def celsius(self):                    # getter: obj.celsius
        return self._celsius

    @celsius.setter
    def celsius(self, value):             # setter: obj.celsius = v
        if value < -273.15:
            raise ValueError("below absolute zero")
        self._celsius = value

    @celsius.deleter
    def celsius(self):                    # del obj.celsius
        del self._celsius

    @property
    def fahrenheit(self):                 # computed, read-only (no setter)
        return self._celsius * 9/5 + 32

t = Temperature(25)
t.celsius            # 25   (getter)
t.celsius = 30       # setter validates
t.fahrenheit         # 86.0 (computed on the fly)
# t.fahrenheit = 100 -> AttributeError: can't set attribute (no setter)
```

## Internal Working

- `property` is a **descriptor** — an object implementing `__get__`, `__set__`, `__delete__`. When you access `t.celsius`, Python finds the `celsius` property on the class and calls its `__get__`, which runs your getter. Assignment triggers `__set__` → your setter.
- `@property` creates the descriptor with only a getter; `@celsius.setter` returns a **new** property object with the setter added (that's why the setter method reuses the name `celsius`).
- Because it's a **class-level** descriptor, the logic applies to every instance, and the actual data is conventionally stored in a differently-named instance attribute (`self._celsius`) to avoid infinite recursion.

## Advantages

- Clean attribute syntax with hidden logic — no `get_x()`/`set_x()` clutter.
- Add validation/computation later **without breaking the public API** (backward compatible).
- Create read-only or computed attributes easily.

## Limitations

- Hidden computation behind `obj.x` can surprise readers (an "attribute" that's actually expensive).
- Slight overhead vs a raw attribute (a method call each access) — usually negligible.
- Overusing properties for trivial fields is un-Pythonic; only add them when logic is needed.

## Real-world Applications

Validation (non-negative balance, valid email), computed values (`full_name` from `first`+`last`, `area` from `radius`), read-only/derived fields in models and ORMs, lazy-loaded/cached attributes, and unit conversions.

## Interview Questions

- **Beginner:** What does `@property` do?
- **Beginner:** How do you make a read-only attribute?
- **Intermediate:** Why use `@property` instead of a plain attribute or `get_x()`?
- **Intermediate:** Why store data in `self._x` inside a property for `x`?
- **Advanced:** How does `property` work under the hood (descriptors)?
- **"Why":** Why is the "uniform access principle" valuable?
- **Comparison:** `@property` vs a getter method vs a public attribute.
- **Scenario:** Add validation to a class that already exposes `self.price` publicly, without breaking callers.

## Model Answers

**Q: Why `@property` over a plain attribute or `get_x()`?**
Over a **getter method** because callers write `obj.x` (natural attribute syntax) instead of `obj.get_x()`. Over a **plain attribute** because a property can validate, compute, or make the value read-only. The killer feature is evolution: you can *ship* a plain attribute and later convert it to a property with validation, and **no caller code changes** — the access syntax is identical. That backward compatibility is why Python doesn't need Java-style getters everywhere.

**Q: Why store data in `self._celsius`?**
Because the property *is named* `celsius`. If the setter did `self.celsius = value`, that assignment would invoke the setter again → infinite recursion. Storing the actual value in a different backing attribute (`self._celsius`) breaks the cycle: the property manages access; `_celsius` holds the raw data.

**Q: How does `property` work under the hood?**
`property` is a **descriptor**: an object defining `__get__`/`__set__`/`__delete__`, placed on the class. Because data descriptors take priority in attribute lookup, accessing `instance.celsius` calls the descriptor's `__get__` (your getter) and assigning calls `__set__` (your setter). The `@x.setter`/`@x.deleter` decorators return augmented property objects with those methods filled in.

**Q (Scenario): Add validation to public `self.price`.**
Convert `price` into a property backed by `_price`:
```python
class Product:
    def __init__(self, price):
        self.price = price          # now routed through the setter
    @property
    def price(self):
        return self._price
    @price.setter
    def price(self, value):
        if value < 0:
            raise ValueError("price cannot be negative")
        self._price = value
```
Existing code doing `p.price` or `p.price = 10` keeps working unchanged, but negative prices are now rejected — validation added with zero API breakage.

## Common Mistakes

- Infinite recursion by storing to `self.x` instead of `self._x` inside the property.
- Adding a getter but forgetting the setter, unintentionally making it read-only.
- Sprinkling properties on trivial fields (premature; just use attributes).
- Hiding expensive work behind a property so callers don't realize the cost.

## Related Concepts

[Encapsulation](#14-encapsulation), [OOP Fundamentals](#13-oop-fundamentals), descriptors, [Dataclasses](#21-dataclasses), [Magic Methods](#18-magic--dunder-methods).

---

# 21. Dataclasses

## What is it?

A **dataclass** (`@dataclass`, from the `dataclasses` module, Python 3.7+) is a decorator that **auto-generates boilerplate methods** — `__init__`, `__repr__`, `__eq__`, and optionally ordering/hashing — from **class-level annotated fields**. It's built for classes that primarily **hold data**.

## Why is it needed?

Writing a simple data-holding class means repetitive boilerplate: an `__init__` that assigns every field, a `__repr__`, an `__eq__` comparing all fields. Dataclasses eliminate that — you declare fields once with type hints and get all the plumbing for free, reducing errors and noise.

## How does it work?

```python
from dataclasses import dataclass, field

@dataclass
class Point:
    x: int                       # required field
    y: int = 0                   # field with default

p = Point(3, 4)
p                                # Point(x=3, y=4)   <- auto __repr__
Point(3, 4) == Point(3, 4)       # True              <- auto __eq__

@dataclass(order=True, frozen=True)   # sortable AND immutable
class Version:
    major: int
    minor: int = 0

@dataclass
class Team:
    name: str
    members: list = field(default_factory=list)   # mutable default done right!
```

Key parameters: `frozen=True` (immutable, hashable), `order=True` (generates `<`,`>`,`<=`,`>=`), `eq=True` (default), `field(default_factory=...)` for mutable defaults, `field(compare=False)` to exclude a field from equality.

## Internal Working

- `@dataclass` **inspects the class's `__annotations__`** (the `name: type` declarations) and **synthesizes methods by generating source code and `exec`-ing it** at class-creation time. So `@dataclass` produces a normal class — there's no runtime magic per instance; it's code generation once.
- `frozen=True` generates `__setattr__`/`__delattr__` that raise `FrozenInstanceError`, making instances immutable, and enables `__hash__`.
- `field(default_factory=list)` sidesteps the **mutable default argument** trap: the factory is **called per instance**, so each object gets its own list instead of sharing one.
- Type hints here are **not enforced** at runtime (Python ignores them for execution) — they're documentation/tooling hints; the field list is what drives generation.

## Advantages

- Massive boilerplate reduction (`__init__`/`__repr__`/`__eq__` free).
- Readable, self-documenting field declarations with types and defaults.
- `frozen`/`order` give immutable, hashable, sortable value objects trivially.
- `default_factory` handles mutable defaults correctly.

## Limitations

- Type hints aren't validated at runtime (bad data still gets in — use `pydantic` if you need validation).
- Slightly less control than a hand-written class for complex init logic (though `__post_init__` helps).
- Inheritance with defaults has ordering rules (non-default fields can't follow default ones).
- For huge numbers of instances, `__slots__`/namedtuple may be leaner (3.10+ supports `@dataclass(slots=True)`).

## Real-world Applications

DTOs / API request-response models, configuration objects, records in data pipelines, domain entities, and anywhere you'd otherwise write a class whose main job is holding a handful of typed fields. `frozen=True` dataclasses make excellent dict keys and value objects.

## Interview Questions

- **Beginner:** What does `@dataclass` generate for you?
- **Beginner:** How do you give a dataclass field a default value?
- **Intermediate:** What is `frozen=True` and why would you use it?
- **Intermediate:** Why must mutable defaults use `field(default_factory=...)`?
- **Advanced:** How does `@dataclass` generate its methods internally?
- **"Why":** When would you choose a dataclass over a plain class or a namedtuple?
- **Comparison:** dataclass vs namedtuple vs dict vs plain class.
- **Scenario:** Model an immutable, hashable, sortable `SemanticVersion`.

## Model Answers

**Q: What does `@dataclass` generate?**
By default it generates `__init__` (assigning each declared field), `__repr__` (a readable `ClassName(field=value, ...)`), and `__eq__` (tuple-wise field comparison). With options it also generates ordering methods (`order=True`) and immutability/hashing (`frozen=True`). You just declare annotated fields; the decorator writes the plumbing.

**Q: Why `field(default_factory=list)` for mutable defaults?**
A bare `members: list = []` would share **one list across all instances** (same class-body evaluation issue as mutable default arguments), and dataclasses actually forbid it with an error for list/dict/set. `default_factory=list` tells the generated `__init__` to **call `list()` fresh for each new instance**, giving every object its own independent collection.

**Q: dataclass vs namedtuple vs dict vs plain class.**
- **dict:** flexible but no fixed schema, no type hints, string-key access, no methods.
- **namedtuple:** immutable, lightweight, tuple-based, iterable/unpackable — great for small fixed records, but awkward to add methods/defaults.
- **dataclass:** mutable-or-frozen, typed fields, defaults, methods, auto `__eq__`/`__repr__` — the modern default for record-like classes.
- **plain class:** full control, but you hand-write all boilerplate. Choose a dataclass when the class is mostly typed data; a plain class when behavior/complex init dominates.

**Q (Scenario): Immutable, hashable, sortable SemanticVersion.**
```python
from dataclasses import dataclass
@dataclass(frozen=True, order=True)
class SemanticVersion:
    major: int
    minor: int = 0
    patch: int = 0

v = SemanticVersion(1, 2, 0)
sorted([SemanticVersion(1,0), SemanticVersion(1,2)])   # order=True
{v}                                                    # frozen=True -> hashable
```
`frozen=True` makes it immutable and hashable (usable in sets/dict keys); `order=True` generates comparison operators so versions sort field-by-field (major, then minor, then patch).

## Common Mistakes

- Using a bare mutable default (`x: list = []`) instead of `default_factory`.
- Expecting type hints to validate data at runtime (they don't).
- Putting a field with a default before one without it (init-ordering error).
- Forgetting `frozen=True` is required for hashability/immutability.
- Reaching for a dataclass when a namedtuple or plain dict is simpler.

## Related Concepts

[OOP Fundamentals](#13-oop-fundamentals), [Type Hints](#32-type-hints), namedtuple, [Tuples](#7-tuples), [Magic Methods](#18-magic--dunder-methods), [Properties](#20-properties).

# 22. Exception Handling

## What is it?

An **exception** is an event that disrupts normal program flow when an error occurs at runtime. **Exception handling** is the structured mechanism — `try` / `except` / `else` / `finally` — that lets you **catch** these errors and respond gracefully instead of crashing. First distinguish the three error categories: **syntax errors** (invalid code, caught before running), **runtime errors** (exceptions raised while running: `ZeroDivisionError`, `FileNotFoundError`), and **logical errors** (code runs but produces wrong results — no exception at all).

## Why is it needed?

Real programs face conditions the code can't prevent: a file is missing, the network drops, the user types letters where a number is expected. Without handling, any such event terminates the program with a traceback. Exception handling lets you **anticipate, isolate, and recover** — retry, use a default, log and continue, or fail cleanly with a helpful message — making software robust.

## How does it work?

```python
try:
    value = int(input("Enter a number: "))   # may raise ValueError
    result = 10 / value                       # may raise ZeroDivisionError
except ValueError:
    print("Not a valid number")               # handle a specific error
except ZeroDivisionError as e:
    print(f"Cannot divide by zero: {e}")      # bind the exception object
except (TypeError, KeyError):
    print("Type or key problem")              # handle multiple in one block
else:
    print(f"Success: {result}")               # runs ONLY if no exception
finally:
    print("Always runs — cleanup goes here")  # runs no matter what
```

- **`try`** — wraps risky code.
- **`except`** — catches a matching exception; can target a specific class, a tuple of classes, or (dangerously) everything.
- **`else`** — runs only if the `try` block raised **nothing** (keeps the "success path" separate from the risky call).
- **`finally`** — runs **always** (exception or not, even on `return`/`break`), for cleanup (closing files, releasing locks).

## Internal Working

- Exceptions are **objects** in a class hierarchy rooted at `BaseException` → `Exception` → specific errors (`ArithmeticError` → `ZeroDivisionError`, `LookupError` → `KeyError`/`IndexError`). An `except SomeError` clause matches that class **and all its subclasses**.
- When an exception is raised, Python **unwinds the call stack** looking for the nearest enclosing `try` with a matching `except`. If none is found anywhere up the stack, the program terminates and prints a **traceback**.
- `except` clauses are tested **top-to-bottom**; the first match wins — so order **specific → general** (a broad `except Exception` first would swallow everything below it).
- `finally` is guaranteed by the interpreter even if the `try`/`except` executes a `return`, `break`, or re-raises.

## Advantages

- Separates error-handling code from normal logic (cleaner than status-code checks).
- Precise, targeted recovery per error type.
- `finally` guarantees resource cleanup; `else` clarifies the success path.
- Exceptions propagate automatically until handled — no manual error passing.

## Limitations

- Overly broad `except:`/`except Exception:` hides bugs and makes debugging hard.
- Exceptions for ordinary control flow can hurt readability (though Python uses them for iteration internally).
- Swallowing exceptions silently (empty `except: pass`) is a notorious anti-pattern.

## Real-world Applications

File/network I/O (`FileNotFoundError`, `ConnectionError`), input validation (`ValueError`), API request retries, database transactions (rollback in `except`, commit in `else`, close in `finally`), and web frameworks converting exceptions into HTTP error responses.

## Interview Questions

- **Beginner:** Difference between syntax, runtime, and logical errors?
- **Beginner:** What does `finally` guarantee?
- **Intermediate:** When does `else` run in a try/except? Why use it?
- **Intermediate:** Why is a bare `except:` a bad idea?
- **Advanced:** In what order should multiple `except` clauses appear, and why?
- **"Why":** Why does `except ValueError` also catch subclasses?
- **Comparison:** `try/except` vs checking conditions beforehand (LBYL vs EAFP).
- **Scenario:** Open a file, process it, and guarantee it's closed even if processing fails (without `with`).

## Model Answers

**Q: What does `finally` guarantee?**
`finally` runs **no matter how the `try` block exits** — whether it completes normally, raises an exception (handled or not), or executes `return`/`break`/`continue`. It's for cleanup that must happen unconditionally: closing files, releasing locks, restoring state. Even if an exception isn't caught, `finally` executes before the exception propagates onward.

**Q: When does `else` run and why use it?**
The `else` block runs **only when the `try` block raised no exception**. Its value is separating the "code that might fail" (in `try`) from "code that should run on success" (in `else`), so the `try` stays minimal and you don't accidentally catch exceptions from the success-path code. Compare: putting success code inside `try` risks an unrelated exception being caught by your `except`.

**Q: Why is bare `except:` bad?**
A bare `except:` (or `except Exception:`) catches **everything**, including bugs you didn't anticipate and even `KeyboardInterrupt`/`SystemExit` (for the truly bare form). It hides the real error, makes debugging painful, and can leave the program in a bad state while pretending all is well. Always catch the **most specific** exception you can actually handle, and let unexpected ones propagate.

**Q (Comparison): LBYL vs EAFP.**
**LBYL** ("Look Before You Leap") checks preconditions first: `if key in d: use(d[key])`. **EAFP** ("Easier to Ask Forgiveness than Permission") just tries and catches failure: `try: use(d[key]) except KeyError: ...`. Python idiomatically prefers **EAFP** — it avoids race conditions (the state can change between check and use) and is often cleaner. LBYL suits cheap checks or when the "exceptional" case is common.

**Q (Scenario): Guaranteed close without `with`.**
```python
f = open("data.txt")
try:
    process(f)
finally:
    f.close()      # runs even if process() raises
```
`finally` guarantees `close()` regardless of what `process` does. (This is exactly the pattern `with open(...)` automates via context managers.)

## Common Mistakes

- Bare `except:` / `except Exception: pass` swallowing errors silently.
- Ordering `except Exception` before specific clauses (it catches everything first).
- Putting cleanup in `except` instead of `finally` (skipped when no error, or on other errors).
- Catching an exception you can't actually handle (better to let it propagate).
- Confusing logical errors (no exception) with runtime errors (they need tests, not `try`).

## Related Concepts

[Raising & Custom Exceptions](#23-raising--custom-exceptions), [Context Managers](#29-context-managers) (`with`), [Assertions](#24-assertions), [Logging](#25-logging), [File Handling](#11-file-handling).

---

# 23. Raising & Custom Exceptions

## What is it?

**Raising** is deliberately signaling an error with the `raise` statement (`raise ValueError("bad input")`). **Custom exceptions** are your own exception classes — subclasses of `Exception` — that represent domain-specific error conditions (`InsufficientFundsError`, `InvalidConfigError`). Python ships a rich set of **built-in exceptions** you should reuse when they fit.

## Why is it needed?

You raise to **enforce contracts** ("this function requires a positive amount — otherwise raise") and to **convert bad states into catchable events** rather than letting corrupt data flow onward. Custom exceptions give errors **meaningful names** and let callers catch *exactly* your error type (`except InsufficientFundsError`) rather than a generic one, making error handling precise and self-documenting.

## How does it work?

```python
def withdraw(balance, amount):
    if amount <= 0:
        raise ValueError("amount must be positive")     # reuse a built-in
    if amount > balance:
        raise InsufficientFundsError(balance, amount)   # domain-specific
    return balance - amount

# Custom exception — subclass Exception, add useful context
class InsufficientFundsError(Exception):
    def __init__(self, balance, amount):
        self.balance, self.amount = balance, amount
        super().__init__(f"Need {amount}, only {balance} available")

# Re-raising and exception chaining
try:
    withdraw(100, 500)
except InsufficientFundsError as e:
    log(e)
    raise                       # re-raise the SAME exception, preserving traceback

try:
    int("abc")
except ValueError as e:
    raise ConfigError("bad config") from e   # chain: "direct cause"
```

- `raise` with no argument inside an `except` **re-raises** the current exception (preserving the original traceback).
- `raise NewError(...) from original` sets `__cause__` for **exception chaining**, showing the root cause in the traceback.

## Internal Working

- `raise X` (a class) instantiates `X()`; `raise X(args)` raises that instance. `raise` alone re-raises the exception currently being handled.
- A custom exception is just a class inheriting from `Exception` (or a more specific built-in). It participates in the same class-based matching — `except Exception` will catch it because it's a subclass.
- **Chaining:** when an exception is raised inside an `except` block, Python auto-sets `__context__` ("During handling of the above, another occurred"). Using `from e` sets `__cause__` explicitly ("The above was the direct cause"). `from None` suppresses the chain.
- Design tip: create a **base class for your app's exceptions** (`class AppError(Exception)`) and subclass from it, so callers can `except AppError` to catch any of your errors.

## Advantages

- Raising enforces invariants early ("fail fast") near the source of the problem.
- Custom exceptions are self-documenting and let callers catch specific cases.
- Chaining preserves root-cause context for debugging.
- A base app-exception enables catching all domain errors uniformly.

## Limitations

- Over-creating exception classes for every tiny case adds clutter — reuse built-ins when they fit.
- Poorly designed hierarchies make `except` clauses awkward.
- Raising for normal control flow can be misused.

## Real-world Applications

Libraries define exception hierarchies (`requests.exceptions.RequestException` → `ConnectionError`, `Timeout`); web frameworks raise `Http404`; validation layers raise domain errors that the API layer maps to HTTP status codes; business logic raises `InsufficientFundsError`, `AccountLockedError`, etc.

## Interview Questions

- **Beginner:** How do you raise an exception? What's the difference between `raise ValueError` and `raise ValueError("msg")`?
- **Beginner:** How do you create a custom exception?
- **Intermediate:** What does a bare `raise` (no argument) do?
- **Intermediate:** Why subclass `Exception` and not `BaseException`?
- **Advanced:** Explain exception chaining (`raise ... from ...`).
- **"Why":** Why create custom exceptions instead of raising generic `Exception`?
- **Comparison:** Reuse a built-in exception vs define a custom one.
- **Scenario:** A validation layer needs callers to distinguish "field missing" from "field invalid". Design it.

## Model Answers

**Q: What does bare `raise` do?**
Inside an `except` block, `raise` with no argument **re-raises the exception currently being handled**, preserving its original traceback. It's used when you want to do something (log, clean up, partially handle) and then let the same error propagate to a higher level, without losing where it originally occurred. Re-raising via `raise e` also works but can reset the traceback context in some cases; bare `raise` is the clean idiom.

**Q: Why subclass `Exception`, not `BaseException`?**
`BaseException` is the root and includes system-exiting exceptions like `KeyboardInterrupt` and `SystemExit`, which you generally **don't** want caught by broad `except Exception` handlers. User-defined errors should inherit from `Exception` so they're caught by normal handlers and treated as ordinary, recoverable application errors — not confused with interpreter control signals.

**Q: Explain exception chaining.**
Chaining links a new exception to the one that caused it. Writing `raise HighLevelError("failed") from low_level_err` sets `__cause__`, and the traceback shows "The above exception was the direct cause of the following," preserving the full story. If you raise inside an `except` without `from`, Python still records the original as `__context__` ("During handling…"). Use `from e` to make the causal link explicit, or `from None` to hide an irrelevant internal error from callers.

**Q (Why): Custom exceptions over generic `Exception`.**
A custom class gives the error a **meaningful name** and lets callers catch precisely that condition: `except InsufficientFundsError` handles just that case while letting other errors propagate. Raising a generic `Exception` forces callers to catch everything or inspect message strings (fragile). Custom exceptions also carry structured context (balance, amount) and, under a shared base class, allow catching an entire category of your app's errors.

**Q (Scenario): Distinguish "missing" vs "invalid".**
```python
class ValidationError(Exception): ...          # base
class MissingFieldError(ValidationError): ...
class InvalidFieldError(ValidationError): ...

def validate(data, field):
    if field not in data:
        raise MissingFieldError(field)
    if not data[field]:
        raise InvalidFieldError(field)
```
Callers can handle each specifically (`except MissingFieldError`) **or** catch the whole category (`except ValidationError`) thanks to the shared base — precise handling with a clean hierarchy.

## Common Mistakes

- Raising a bare string (`raise "error"`) — must raise an exception instance/class.
- Subclassing `BaseException` for app errors (interferes with `KeyboardInterrupt`).
- Losing the original traceback by catching and raising a new error without `from e`.
- Creating one giant custom exception with no hierarchy (hard to catch selectively).
- Using exceptions for expected, routine outcomes where a return value is clearer.

## Related Concepts

[Exception Handling](#22-exception-handling), [Assertions](#24-assertions), [Logging](#25-logging), [Inheritance](#16-inheritance) (exception hierarchy), [OOP Fundamentals](#13-oop-fundamentals).

---

# 24. Assertions

## What is it?

An **assertion** (`assert condition, message`) is a **sanity check** that states "this condition must be true at this point; if not, something is fundamentally broken." If the condition is false, Python raises `AssertionError`. Assertions are a **debugging/development** aid to catch programmer mistakes and verify internal invariants — **not** a mechanism for validating external input or handling expected errors.

## Why is it needed?

To catch bugs **early and close to their cause** by documenting and enforcing assumptions your code relies on: "the list is sorted here", "this index is non-negative", "the function never returns None here". They make invariants explicit and fail loudly during development, rather than letting a corrupt assumption silently produce wrong results downstream.

## How does it work?

```python
def average(numbers):
    assert len(numbers) > 0, "numbers must not be empty"   # precondition
    total = sum(numbers)
    result = total / len(numbers)
    assert result >= min(numbers), "avg can't be below the minimum"  # invariant
    return result

average([])   # AssertionError: numbers must not be empty
```

`assert X, msg` is equivalent to `if not X: raise AssertionError(msg)`.

**Critical caveat:** running Python with the `-O` (optimize) flag **removes all assert statements**. So assertions must **never** be used for logic your program depends on at runtime — only for checks that are safe to strip in production.

## Internal Working

- `assert expr, msg` compiles to roughly: `if __debug__: if not expr: raise AssertionError(msg)`. `__debug__` is `True` normally and `False` under `python -O`, so the entire assertion is skipped (or even removed at compile time) in optimized mode.
- Because they can vanish, assertions carry **no guarantee** of executing — this is why they're unsuitable for validating user input or enforcing security/permissions.

## Advantages

- Catch bugs early, near their source, with a clear message.
- Self-documenting: they state the assumptions the code makes.
- Zero runtime cost in production (stripped with `-O`).

## Limitations

- **Stripped by `-O`** — never rely on them for required behavior or input validation.
- Not for expected/recoverable errors (use exceptions instead).
- Overusing them for user-facing checks is a security/correctness hazard.

## Real-world Applications

Internal invariant checks in algorithms and data structures, verifying preconditions/postconditions during development, guarding "this should never happen" branches, and as lightweight checks in **unit tests** (`assert result == expected` — though test frameworks like `pytest` use `assert` heavily and don't run with `-O`).

## Interview Questions

- **Beginner:** What does `assert` do?
- **Beginner:** What exception does a failed assertion raise?
- **Intermediate:** Why should you not use assertions to validate user input?
- **Intermediate:** What happens to asserts when Python runs with `-O`?
- **Advanced:** Difference between assertions and exceptions — when to use each?
- **"Why":** Why does `assert` exist if it can be disabled?
- **Comparison:** `assert x > 0` vs `if x <= 0: raise ValueError(...)`.
- **Scenario:** You want to guarantee a bank withdrawal never accepts a negative amount in production. Assert or raise?

## Model Answers

**Q: Why not use assertions for user input?**
Because assertions **can be disabled** — running with `python -O` strips every `assert`, so any validation you put there simply **won't execute** in optimized production. User input is *expected* to sometimes be wrong and must be validated with real, always-on code (`if ...: raise ValueError`). Assertions are for **programmer errors** (broken internal assumptions), not for handling untrusted external data.

**Q: Assertions vs exceptions — when to use each?**
Use an **exception** for *expected, recoverable* error conditions that can occur in normal operation (missing file, invalid user input, network failure) — these must always be handled. Use an **assertion** for *should-never-happen* internal invariants that indicate a **bug** if violated (a sorting function returning unsorted data). Rule of thumb: if a user could trigger it, raise an exception; if only a programmer mistake could trigger it, assert.

**Q (Why): Why have `assert` if it can be disabled?**
Precisely because it's a **development-time** tool. Assertions let you liberally document and verify internal assumptions during development and testing (catching bugs early) without paying any performance cost in optimized production, where they're stripped. Their disable-ability is a feature: expensive sanity checks stay in dev but disappear in prod.

**Q (Scenario): Negative-amount withdrawal in production.**
Use a **`raise`**, not an assert:
```python
if amount < 0:
    raise ValueError("amount cannot be negative")
```
This is a validation of (potentially user-driven) input that must **always** run. An `assert amount >= 0` would silently vanish under `-O`, allowing negative withdrawals in production — a serious bug. Assertions are only appropriate for internal invariants you're confident about, never for enforcing runtime business rules.

## Common Mistakes

- Using `assert` to validate user input or enforce security (stripped by `-O`).
- `assert (x, msg)` with parentheses — that asserts a **non-empty tuple**, which is always truthy (a silent no-op bug!). Correct: `assert x, msg`.
- Relying on assertion side effects (they may not run).
- Using assertions where a clear exception with recovery is warranted.

## Related Concepts

[Exception Handling](#22-exception-handling), [Raising & Custom Exceptions](#23-raising--custom-exceptions), [Logging](#25-logging), unit testing, `__debug__`.

---

# 25. Logging

## What is it?

**Logging** is recording events that happen while a program runs — for diagnostics, monitoring, and auditing. Python's `logging` module provides a flexible, industrial-strength framework with **severity levels** (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`), configurable **handlers** (console, file, network), **formatters**, and per-module **loggers**. It's the professional replacement for scattered `print()` statements.

## Why is it needed?

`print()` is fine for a quick script but terrible for real applications: you can't easily turn it off, filter by importance, add timestamps/context, or send output to files/servers. Logging gives you **controllable, leveled, routable** output so you can run verbose diagnostics in development and only warnings/errors in production — without changing code — and keep a durable record for debugging incidents.

## How does it work?

```python
import logging

logging.basicConfig(
    level=logging.INFO,                       # minimum level to emit
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    filename="app.log",                       # omit -> logs to console
)
logger = logging.getLogger(__name__)          # per-module logger (best practice)

logger.debug("detailed diagnostic")           # hidden (below INFO)
logger.info("user logged in")
logger.warning("disk 85%% full")
logger.error("payment failed")
try:
    1 / 0
except ZeroDivisionError:
    logger.exception("math error")            # ERROR + full traceback
```

**Levels (low → high):** `DEBUG` (10) → `INFO` (20) → `WARNING` (30, the default threshold) → `ERROR` (40) → `CRITICAL` (50). The logger emits a message only if its level ≥ the configured threshold.

## Internal Working

- The framework has four core pieces: **Loggers** (the entry point you call; organized in a hierarchy by dotted name), **Handlers** (decide *where* records go — `StreamHandler`, `FileHandler`, `RotatingFileHandler`, `SMTPHandler`), **Formatters** (decide *how* records look), and **Filters** (fine-grained inclusion rules).
- Each `logger.info(...)` creates a **LogRecord**; if the record's level passes the logger's threshold, it flows to the logger's handlers (and, by **propagation**, up to ancestor loggers' handlers, ultimately the root). Each handler applies its own level and formatter.
- `getLogger(__name__)` gives each module its own logger named after the module, so you can tune verbosity per component and see *where* a message came from. Using `%`-style lazy args (`logger.info("x=%s", x)`) means the string is only formatted if the message will actually be emitted.

## Advantages

- Adjustable verbosity via levels — no code changes to switch dev/prod detail.
- Routes output anywhere (files, rotating files, syslog, email) via handlers.
- Structured, timestamped, contextual records; per-module control.
- `logger.exception()` captures full tracebacks automatically.

## Limitations

- More setup/concepts than `print` (loggers/handlers/formatters).
- Misconfiguration (e.g., double handlers) causes duplicate lines.
- Excessive `DEBUG` logging can hurt performance and bloat files.
- Care needed not to log sensitive data (passwords, PII, tokens).

## Real-world Applications

Every production service: request/response logging in web apps, error tracking, audit trails, background-job monitoring, and feeding log aggregators (ELK/Datadog/Splunk). Rotating file handlers cap disk usage; different levels feed dashboards vs alerts.

## Interview Questions

- **Beginner:** Why use `logging` instead of `print()`?
- **Beginner:** Name the logging levels in order.
- **Intermediate:** What's the default logging level and what does it mean?
- **Intermediate:** Difference between `logger.error()` and `logger.exception()`?
- **Advanced:** Explain loggers, handlers, and formatters.
- **"Why":** Why use `getLogger(__name__)` per module?
- **Comparison:** `print` debugging vs logging.
- **Scenario:** Log INFO+ to a rotating file but show only ERRORs on the console.

## Model Answers

**Q: Why logging over print?**
`print` always writes to stdout, can't be filtered by importance, has no timestamps/levels/context, and must be manually removed. `logging` lets you assign **severity levels** and change the threshold (verbose in dev, quiet in prod) **without touching code**, **route** output to files/servers via handlers, add structured formatting (time, module, level), and capture tracebacks. It's controllable, persistent, and production-grade — print is not.

**Q: `error()` vs `exception()`.**
Both log at `ERROR` level, but `logger.exception(msg)` **must be called from within an `except` block** and automatically **attaches the current exception's traceback** to the log record. `logger.error(msg)` logs just the message with no traceback. Use `exception()` when handling a caught error so you capture *where* and *why* it failed; use `error()` for error conditions where there's no active exception.

**Q: Loggers, handlers, formatters.**
- **Logger:** the object you call (`logger.info`); named hierarchically, holds a level threshold, and dispatches records.
- **Handler:** determines the **destination** (console, file, rotating file, email) and has its own level/formatter — a single logger can have multiple handlers.
- **Formatter:** defines the **layout** of each line (timestamp, level, logger name, message).
A record passes the logger's threshold, then each handler filters and formats it independently. This separation lets the same log stream go to several places in different formats.

**Q (Why): `getLogger(__name__)` per module?**
It creates a logger named after the module (e.g., `myapp.services.payment`), so log lines show **exactly where they originated**, and you can **tune verbosity per component** (silence a chatty library, keep DEBUG on your module). Because loggers form a hierarchy by dotted name, configuration set on a parent (or root) applies to children via propagation, giving both granular control and central defaults.

**Q (Scenario): INFO+ to rotating file, ERROR+ to console.**
```python
import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

fh = RotatingFileHandler("app.log", maxBytes=1_000_000, backupCount=3)
fh.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)

fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
fh.setFormatter(fmt); ch.setFormatter(fmt)
logger.addHandler(fh); logger.addHandler(ch)
```
Two handlers on one logger: the rotating file captures INFO and above (with size-capped rotation), while the console only surfaces ERROR and above — different destinations, different thresholds, same records.

## Common Mistakes

- Using `print` for real application diagnostics.
- Calling `basicConfig` after handlers are set (it's a no-op then) or adding handlers twice → duplicate log lines.
- Logging sensitive data (passwords, tokens, PII).
- Using `error()` inside an `except` instead of `exception()` and losing the traceback.
- Eagerly formatting messages (`f"...{x}"`) instead of lazy `%s` args for expensive values.

## Related Concepts

[Exception Handling](#22-exception-handling), [Raising & Custom Exceptions](#23-raising--custom-exceptions), [Assertions](#24-assertions), [Modules & Packages](#12-modules--packages), monitoring/observability.

# 26. Iterators

## What is it?

An **iterable** is any object you can loop over (`list`, `str`, `dict`, `file`) — it implements `__iter__()` which returns an **iterator**. An **iterator** is the object that actually produces values one at a time via `__next__()`, raising `StopIteration` when exhausted. The pair is the **iterator protocol**, and it's the machinery behind every `for` loop.

## Why is it needed?

To provide a **uniform way to traverse any collection** without exposing its internal structure, and to enable **lazy, memory-efficient** iteration — you can iterate a 10 GB file or an infinite sequence one element at a time instead of materializing everything. It decouples "how to loop" from "what you're looping over".

## How does it work?

```python
nums = [10, 20, 30]
it = iter(nums)          # __iter__ -> an iterator
next(it)                 # 10  (__next__)
next(it)                 # 20
next(it)                 # 30
next(it)                 # StopIteration raised

# What `for` really does under the hood:
it = iter(nums)
while True:
    try:
        x = next(it)
    except StopIteration:
        break
    print(x)

# A custom iterator:
class Countdown:
    def __init__(self, start): self.current = start
    def __iter__(self):        return self          # iterable: returns an iterator
    def __next__(self):
        if self.current <= 0:
            raise StopIteration
        self.current -= 1
        return self.current + 1

for n in Countdown(3):        # 3, 2, 1
    print(n)
```

## Internal Working

- `for x in obj` calls `iter(obj)` (→ `obj.__iter__()`) **once** to get an iterator, then calls `next()` repeatedly, catching `StopIteration` to stop.
- An **iterator is also an iterable** (its `__iter__` returns `self`), which is why you can use it directly in a `for`. But it's **one-shot** — once exhausted it stays exhausted; you must get a fresh iterator to loop again.
- A **list is iterable but not its own iterator**: each `iter(list)` returns a **new, independent** iterator, so you can loop over a list many times. That separation is intentional.
- Iterators hold only their **current position/state**, not all elements — hence the memory efficiency.

## Advantages

- Uniform traversal across all container types (`for` "just works").
- Lazy: constant memory regardless of size; supports infinite streams.
- Decouples iteration from the container's internal representation.

## Limitations

- Iterators are **single-use/consumable** — surprising when reused.
- No random access, no `len()`, can't go backwards.
- Writing a correct custom iterator is verbose (generators are usually easier — see next section).

## Real-world Applications

Reading files/streams line by line, database cursors, paginated API results, `range`/`zip`/`enumerate`/`map` (all lazy iterators in Python 3), and any pipeline that processes data larger than memory.

## Interview Questions

- **Beginner:** Difference between an iterable and an iterator?
- **Beginner:** What ends a `for` loop internally?
- **Intermediate:** Why can you loop over a list twice but not an iterator twice?
- **Intermediate:** What two methods define the iterator protocol?
- **Advanced:** Implement a custom iterator for a range-like object.
- **"Why":** Why are iterators memory-efficient?
- **Comparison:** Iterator vs generator.
- **Scenario:** You need to process a file too big for RAM. How does the iterator protocol help?

## Model Answers

**Q: Iterable vs iterator.**
An **iterable** is anything you can loop over — it implements `__iter__()` which **returns an iterator** (lists, strings, dicts, files). An **iterator** is the worker that produces values: it implements `__next__()` (and `__iter__` returning itself), yielding one item per call and raising `StopIteration` when done. In short: the iterable is the *source*; the iterator is the *cursor* that walks it and holds the position.

**Q: Why loop a list twice but not an iterator?**
A **list is iterable, not an iterator**: each `for`/`iter()` on it creates a **fresh iterator** starting at the beginning, so you can loop any number of times. An **iterator is its own iterator** and carries consumable state — once `next()` reaches the end and raises `StopIteration`, it's exhausted permanently. Looping it again immediately hits `StopIteration` and yields nothing. To re-iterate, obtain a new iterator from the original iterable.

**Q (Why): Why are iterators memory-efficient?**
Because an iterator stores only its **current position and enough state to compute the next value**, not the entire sequence. It produces items **on demand** ("lazily"), so iterating a million-line file or an infinite counter uses roughly constant memory — nothing is pre-materialized. Contrast with building a full list, which allocates memory proportional to the number of elements.

**Q (Scenario): Process a file too big for RAM.**
File objects implement the iterator protocol: `for line in f` calls `next()` to read **one line at a time**, so memory stays flat regardless of file size. You never load the whole file; you pull and process each line lazily. This is the same protocol that lets `for` iterate custom iterators, generators, and infinite sequences — a single uniform, memory-safe mechanism.

## Common Mistakes

- Trying to reuse an exhausted iterator and getting no output.
- Calling `len()` or indexing an iterator (unsupported).
- Forgetting to raise `StopIteration` in a custom `__next__` (infinite loop).
- Confusing `__iter__` (returns iterator) with `__next__` (returns next value).

## Related Concepts

[Generators](#27-generators), [Control Flow](#4-control-flow) (`for`), [Comprehensions](#31-comprehensions), [Magic Methods](#18-magic--dunder-methods), [File Handling](#11-file-handling).

---

# 27. Generators

## What is it?

A **generator** is the easiest way to create an iterator: a function that uses **`yield`** instead of `return`. Calling it doesn't run the body — it returns a **generator object** that produces values lazily, one per `yield`, **remembering its state** (local variables, execution position) between calls. There are also **generator expressions**: `(x*x for x in range(10))` — like a list comprehension but lazy.

## Why is it needed?

To get all the benefits of iterators (lazy, memory-efficient, infinite sequences) **without the boilerplate** of a class with `__iter__`/`__next__`/`StopIteration`. Generators express streaming and pipelines concisely and are the standard tool for processing large or infinite data.

## How does it work?

```python
def countdown(n):
    while n > 0:
        yield n            # pauses here, returns n, resumes on next()
        n -= 1

gen = countdown(3)         # nothing runs yet — returns a generator object
next(gen)                  # 3   (runs until first yield)
next(gen)                  # 2   (resumes after yield)
for x in countdown(3): ... # 3, 2, 1

# Infinite generator — impossible with a list
def naturals():
    n = 1
    while True:
        yield n
        n += 1

# Generator expression (lazy) vs list comprehension (eager)
squares_lazy  = (x*x for x in range(1_000_000))   # ~0 memory now
squares_eager = [x*x for x in range(1_000_000)]   # allocates 1M ints

# Pipeline: chain generators — each stage is lazy
def read(f):    return (line.strip() for line in f)
def only_errors(lines): return (l for l in lines if "ERROR" in l)
```

## Internal Working

- A function containing `yield` is compiled into a **generator function**; calling it constructs a **generator object** (an iterator) without executing the body.
- Each `next()` runs the body until it hits a `yield`, **suspends** — saving the entire frame (locals, instruction pointer) — and returns the yielded value. The next `next()` **resumes exactly where it left off**. When the function returns/ends, `StopIteration` is raised automatically.
- This suspend/resume is why generators are **stateful and lazy**: state lives in the paused frame, values are computed only when pulled.
- `yield from iterable` delegates to a sub-generator/iterable. Generators can also **receive** values via `.send()` and be closed with `.close()` (coroutine features).

## Advantages

- Concise (no iterator-class boilerplate) and readable.
- Lazy and memory-efficient; supports infinite sequences and streaming pipelines.
- Composable — chain generators into data-processing pipelines with flat memory.

## Limitations

- **Single-use/consumable** (like all iterators) — can't reuse or index.
- No `len()`, no random access, can't rewind.
- Harder to debug (state is implicit in the suspended frame).
- Slight per-item overhead vs a pre-built list if you need the whole thing repeatedly.

## Real-world Applications

Streaming file/log processing, reading large datasets in chunks/batches, paginated API consumption, ETL pipelines, `range`/`map`/`filter`-style lazy transforms, producing infinite ID/sequence streams, and building data pipelines in ML preprocessing.

## Interview Questions

- **Beginner:** What does `yield` do? How is it different from `return`?
- **Beginner:** What is returned when you call a generator function?
- **Intermediate:** Difference between a list comprehension and a generator expression?
- **Intermediate:** How is a generator more memory-efficient than a list?
- **Advanced:** How does a generator "remember" its state between calls?
- **"Why":** Why would you use a generator for reading a large file?
- **Comparison:** Generator function vs custom iterator class.
- **Scenario:** Produce the Fibonacci sequence lazily and take the first 10.

## Model Answers

**Q: `yield` vs `return`.**
`return` **ends** the function and hands back a single value; the function's local state is discarded. `yield` **pauses** the function, hands back a value, and **preserves all local state and position** so execution resumes right after the `yield` on the next `next()` call. A function with `yield` becomes a generator that can produce a whole *stream* of values over time, not just one.

**Q: List comprehension vs generator expression.**
Syntax differs only by brackets: `[x for x in it]` (list) vs `(x for x in it)` (generator). The list comprehension is **eager** — it computes and stores **all** elements immediately (memory ∝ size). The generator expression is **lazy** — it computes each value **on demand** and holds essentially no memory, ideal for large/infinite data or when you'll consume it once. Use a list when you need random access, `len`, or to iterate multiple times; a generator when streaming.

**Q: How does a generator remember state?**
When a generator hits `yield`, the interpreter **suspends the function's frame** — its local variables and the exact instruction pointer are saved on the generator object. The frame isn't destroyed (unlike a normal return); it's parked. The next `next()` **reactivates that same frame**, so execution continues with all locals intact. This frame suspension/resumption is the core mechanism enabling laziness and statefulness.

**Q (Why): Generator for a large file.**
Because it reads and yields **one piece at a time** instead of loading the whole file into memory. A generator pipeline (`read → filter → transform`) processes a multi-gigabyte file in roughly constant memory, since each stage pulls the next item lazily. A list-based approach would try to hold the entire file (and each intermediate result) in RAM and likely crash.

**Q (Scenario): Lazy Fibonacci, first 10.**
```python
def fib():
    a, b = 0, 1
    while True:               # infinite — fine, it's lazy
        yield a
        a, b = b, a + b

from itertools import islice
list(islice(fib(), 10))       # [0,1,1,2,3,5,8,13,21,34]
```
The generator produces Fibonacci numbers indefinitely without storing them; `islice` pulls exactly 10 and stops, so it never runs forever.

## Common Mistakes

- Expecting a generator to be reusable/reiterable (it's consumed once).
- Calling `len()` or indexing a generator.
- Using an eager list comprehension for huge data (memory blowup) where a generator fits.
- Forgetting the generator body doesn't run until iterated (side effects deferred).
- Materializing a generator with `list()` unnecessarily, defeating laziness.

## Related Concepts

[Iterators](#26-iterators), [Comprehensions](#31-comprehensions), [Functional Programming](#30-functional-programming), `itertools`, [Concurrency](#34-concurrency) (`async` generators), [File Handling](#11-file-handling).

---

# 28. Closures & Decorators

## What is it?

A **closure** is a nested function that **captures and remembers variables from its enclosing scope**, even after the outer function has returned. A **decorator** is a function that **takes another function (or class) and returns a modified/wrapped version**, letting you add behavior (logging, timing, caching, auth) **without changing the original code** — applied with the `@decorator` syntax. Decorators are built on closures.

## Why is it needed?

Closures let functions carry private state without a class. Decorators implement **cross-cutting concerns** — behavior needed across many functions (logging every call, timing, access control, caching) — in **one reusable place**, keeping the decorated functions clean and honoring DRY. They're the mechanism behind `@property`, `@staticmethod`, Flask's `@app.route`, and `@functools.lru_cache`.

## How does it work?

```python
# CLOSURE: inner() remembers `factor` after make_multiplier returns
def make_multiplier(factor):
    def inner(x):
        return x * factor          # captures `factor`
    return inner
double = make_multiplier(2)
double(5)                          # 10 — `factor=2` still remembered

# DECORATOR: wrap a function to add behavior
import functools, time
def timer(func):
    @functools.wraps(func)         # preserves func's __name__, __doc__
    def wrapper(*args, **kwargs):  # accept ANY signature
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.perf_counter()-start:.4f}s")
        return result
    return wrapper

@timer                             # slow = timer(slow)
def slow(n):
    return sum(range(n))
slow(1_000_000)

# PARAMETERIZED decorator (a decorator factory — 3 levels deep)
def repeat(times):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*a, **k):
            for _ in range(times):
                result = func(*a, **k)
            return result
        return wrapper
    return decorator

@repeat(3)
def greet(): print("hi")
```

## Internal Working

- `@decorator` above `def f` is pure syntactic sugar for `f = decorator(f)` — it rebinds the name `f` to the wrapper.
- A closure keeps captured variables alive via **cell objects** stored in `func.__closure__`; the free variable names are in `func.__code__.co_freevars`. This is why `double` still knows `factor=2` after `make_multiplier` returned.
- `wrapper(*args, **kwargs)` forwards **any** arguments so the decorator works on functions of any signature.
- `functools.wraps(func)` copies the original's `__name__`, `__doc__`, `__module__`, etc. onto the wrapper — without it, the decorated function would masquerade as `wrapper`, breaking introspection, docs, and debugging.
- A **parameterized decorator** is a function returning a decorator (three nested levels): `repeat(3)` returns `decorator`, which wraps the function.

## Advantages

- DRY reuse of cross-cutting behavior across many functions.
- Non-invasive: the original function's code is untouched.
- Composable — stack multiple decorators.
- Closures give lightweight stateful functions without classes.

## Limitations

- Extra indirection can obscure control flow and complicate stack traces (mitigated by `functools.wraps`).
- Forgetting `wraps` breaks introspection.
- Debugging stacked decorators can be tricky; order matters (bottom-up application).
- Closures capturing loop variables cause the classic **late-binding** bug.

## Real-world Applications

`@app.route` (Flask routing), `@login_required` (auth), `@lru_cache`/`@cache` (memoization), `@property`/`@staticmethod`/`@classmethod`, retry/rate-limit decorators, logging/metrics wrappers, and `@pytest.fixture`. Closures back callbacks, function factories, and stateful counters.

## Interview Questions

- **Beginner:** What is a decorator? What does `@timer` translate to?
- **Beginner:** What is a closure?
- **Intermediate:** Why do decorators use `*args, **kwargs` in the wrapper?
- **Intermediate:** What does `functools.wraps` do and why does it matter?
- **Advanced:** Write a decorator that accepts an argument (parameterized decorator).
- **"Why":** Why are decorators built on closures?
- **Comparison:** Decorator vs subclassing vs manually editing a function.
- **Scenario:** Add caching to an expensive pure function without touching its body.

## Model Answers

**Q: What is a closure?**
A closure is a function that **captures variables from its enclosing (non-global) scope and keeps them alive after that outer scope has exited**. In `make_multiplier`, the returned `inner` still references `factor` even though `make_multiplier` has returned — Python stores `factor` in a **cell** attached to `inner.__closure__`. Closures let a function carry private, persistent state without needing a class.

**Q: Why `*args, **kwargs` in the wrapper?**
So the decorator is **generic** — it can wrap a function with *any* signature. The wrapper accepts arbitrary positional and keyword arguments and forwards them unchanged to the original (`func(*args, **kwargs)`). Without this, a decorator would only work for functions matching the wrapper's fixed parameter list.

**Q: What does `functools.wraps` do?**
It's a decorator applied to the wrapper that **copies the wrapped function's metadata** (`__name__`, `__doc__`, `__qualname__`, `__wrapped__`, etc.) onto the wrapper. Without it, `decorated_func.__name__` would be `"wrapper"` and its docstring lost, breaking introspection, help text, logging that uses `__name__`, and some frameworks that inspect signatures. Always use `@functools.wraps(func)` when writing decorators.

**Q (Why): Why are decorators built on closures?**
Because a decorator's wrapper needs to **remember the original function** (`func`) to call it later. `func` is a free variable captured from the decorator's scope — that capture *is* a closure. The closure keeps `func` (and any decorator arguments, like `times` in `repeat`) alive inside the wrapper, so when the wrapper eventually runs, it can invoke the original with the remembered configuration.

**Q (Scenario): Add caching without touching the body.**
```python
import functools
@functools.lru_cache(maxsize=None)     # memoizes results by arguments
def fib(n):
    return n if n < 2 else fib(n-1) + fib(n-2)
```
`lru_cache` wraps `fib`, storing results keyed by arguments so repeated calls return instantly and the exponential recursion collapses to linear — all without editing `fib`'s logic. (For a hand-rolled version, a closure holding a `cache` dict does the same.) The function must be **pure** (same input → same output) for caching to be correct.

## Common Mistakes

- Omitting `functools.wraps`, corrupting the wrapped function's identity/docs.
- Wrapper with a fixed signature instead of `*args, **kwargs`.
- Closure late-binding: `funcs = [lambda: i for i in range(3)]` all return `2` — fix with `lambda i=i: i`.
- Wrong decorator order when stacking (they apply bottom-to-top).
- Caching an impure function and getting stale/incorrect results.

## Related Concepts

[Functions](#5-functions) (scope, first-class), [Functional Programming](#30-functional-programming), `functools`, [Properties](#20-properties), [Context Managers](#29-context-managers), [Magic Methods](#18-magic--dunder-methods) (`__call__`).

---

# 29. Context Managers

## What is it?

A **context manager** is an object that defines **setup and teardown** logic run around a block of code, used with the **`with` statement**. It implements the **context-manager protocol**: `__enter__()` (runs on entry) and `__exit__()` (runs on exit — *guaranteed*, even if an exception occurs). The canonical example is `with open(...) as f:` which guarantees the file is closed.

## Why is it needed?

To **reliably acquire and release resources** — files, network connections, database sessions, locks — so cleanup happens **automatically and even on errors**, without verbose `try/finally` everywhere. It encapsulates the "always clean up" pattern in a reusable object, preventing leaked file handles, unreleased locks, and unclosed connections.

## How does it work?

```python
with open("data.txt") as f:      # __enter__ returns f
    data = f.read()
# __exit__ called here — file closed even if read() raised

# Custom context manager via a class
class Timer:
    def __enter__(self):
        import time; self.start = time.perf_counter()
        return self                       # bound to the `as` variable
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        print(f"Elapsed: {time.perf_counter()-self.start:.4f}s")
        return False                      # False -> propagate any exception

with Timer():
    do_work()

# Easier: contextlib.contextmanager turns a generator into a CM
from contextlib import contextmanager
@contextmanager
def managed_resource():
    r = acquire()                         # setup (before yield)
    try:
        yield r                           # value bound to `as`
    finally:
        release(r)                        # teardown (guaranteed)

with managed_resource() as res:
    use(res)
```

## Internal Working

- `with cm as x:` compiles to: call `cm.__enter__()`, bind its return to `x`, run the body, then **always** call `cm.__exit__(exc_type, exc_val, exc_tb)` — with the exception info if one occurred, or three `None`s if not.
- `__exit__`'s **return value controls exception propagation**: returning a **truthy** value **suppresses** the exception (swallows it); returning **falsy/None** lets it propagate (the normal case). This is how `contextlib.suppress` works.
- `@contextmanager` wraps a **generator**: everything before `yield` is `__enter__`, the yielded value is what `as` binds, and everything after `yield` (ideally in a `finally`) is `__exit__`. The generator is resumed once on exit.
- `contextlib.ExitStack` manages a dynamic number of context managers; `with a, b:` nests multiple in one statement.

## Advantages

- Guaranteed, exception-safe cleanup — no leaked resources.
- Removes repetitive `try/finally` boilerplate; intent is explicit.
- Reusable and composable; `@contextmanager` makes custom ones concise.
- Can suppress or transform exceptions where appropriate.

## Limitations

- Slightly more concepts than a plain `try/finally`.
- Accidentally suppressing exceptions (returning truthy from `__exit__`) hides bugs.
- Not every resource maps cleanly to enter/exit semantics.

## Real-world Applications

File handling (`open`), database sessions/transactions (commit on success, rollback on error, close always), thread locks (`with lock:`), temporary directories (`tempfile.TemporaryDirectory`), `threading.Lock`, changing/restoring global state (working directory, settings), and timing/profiling blocks.

## Interview Questions

- **Beginner:** What does the `with` statement do?
- **Beginner:** Which two methods define a context manager?
- **Intermediate:** What is guaranteed if an exception occurs inside a `with` block?
- **Intermediate:** How do you create a context manager with `contextlib`?
- **Advanced:** How does `__exit__`'s return value affect exceptions?
- **"Why":** Why prefer `with open(...)` over manual open/close?
- **Comparison:** Class-based vs `@contextmanager` generator-based CM.
- **Scenario:** Ensure a database transaction commits on success and rolls back on error.

## Model Answers

**Q: Which two methods, and what's guaranteed on exception?**
`__enter__` (setup, returns the value bound by `as`) and `__exit__(exc_type, exc_val, exc_tb)` (teardown). Even if the `with` body raises, Python **guarantees `__exit__` is called** with the exception details — so cleanup (closing files, releasing locks) always happens. That guarantee is the whole point: resource release is decoupled from whether the body succeeded.

**Q: How does `__exit__`'s return value affect exceptions?**
If the body raised, Python passes the exception info to `__exit__`. If `__exit__` returns a **truthy** value, Python treats the exception as **handled and suppresses it** (execution continues after the `with`). If it returns **falsy/`None`**, the exception **propagates** normally after cleanup. Most `__exit__` implementations return `None`/`False` — you only return truthy when you deliberately want to swallow specific errors.

**Q (Why): `with open` over manual open/close.**
`with` uses the file as a context manager, guaranteeing `close()` via `__exit__` **even if an exception occurs** mid-block — flushing buffers and releasing the OS handle. Manual `open`/`close` leaks the descriptor if code between them raises, unless you write an explicit `try/finally` (which is exactly, and more verbosely, what `with` does). It's safer and clearer.

**Q (Scenario): Transaction commit/rollback.**
```python
from contextlib import contextmanager
@contextmanager
def transaction(conn):
    try:
        yield conn
        conn.commit()          # success path
    except Exception:
        conn.rollback()        # error path
        raise                  # re-raise so caller sees it
    finally:
        conn.close()           # always

with transaction(conn) as c:
    c.execute("UPDATE ...")
```
The generator-based context manager commits when the block finishes cleanly, rolls back and re-raises on any exception, and closes the connection unconditionally — encapsulating correct transaction handling in one reusable place.

## Common Mistakes

- Forgetting to put teardown in a `finally` inside `@contextmanager` (skipped on exception).
- Accidentally returning truthy from `__exit__` and silently swallowing errors.
- Not using `with` and leaking resources on exceptions.
- Assuming `__enter__`'s return is the manager itself (it's whatever you `return`; `open` returns the file).

## Related Concepts

[File Handling](#11-file-handling), [Exception Handling](#22-exception-handling), [Generators](#27-generators) (`@contextmanager`), `contextlib`, [Concurrency](#34-concurrency) (locks), [Magic Methods](#18-magic--dunder-methods).

---

# 30. Functional Programming

## What is it?

Functional programming (FP) is a style that treats **functions as first-class values** and favors **pure functions** (output depends only on input, no side effects) and **immutability**. Python isn't purely functional but supports FP tools: **`map`**, **`filter`**, **`reduce`**, plus helpers **`zip`**, **`enumerate`**, **`any`**, **`all`**, and higher-order functions/lambdas.

## Why is it needed?

FP produces **concise, composable, testable** code. Pure functions are trivial to test (no hidden state) and reason about, and functional transforms (`map`/`filter`) express "what to compute" declaratively instead of manual loops with mutable accumulators. It underpins data pipelines and parallelizable transforms.

## How does it work?

```python
nums = [1, 2, 3, 4, 5]

list(map(lambda x: x*x, nums))              # [1,4,9,16,25]  apply fn to each
list(filter(lambda x: x % 2 == 0, nums))    # [2,4]          keep where True

from functools import reduce
reduce(lambda acc, x: acc + x, nums, 0)     # 15  fold to a single value

list(zip(["a","b"], [1,2]))                 # [('a',1),('b',2)]  pair up
list(enumerate(["a","b"], start=1))         # [(1,'a'),(2,'b')]  index+value

any(x > 4 for x in nums)                    # True  — at least one truthy
all(x > 0 for x in nums)                    # True  — every element truthy
```

- **`map(fn, it)`** applies `fn` to every element (lazy iterator).
- **`filter(pred, it)`** keeps elements where `pred` is truthy (lazy).
- **`reduce(fn, it, init)`** folds the iterable into one value (in `functools`).
- **`zip`** transposes multiple iterables; **`enumerate`** pairs index with value.
- **`any`/`all`** short-circuit boolean aggregations.

## Internal Working

- `map`/`filter`/`zip`/`enumerate` return **lazy iterators** in Python 3 — they compute values on demand (memory-efficient, composable) and are consumed once. Wrap in `list()` to materialize.
- `reduce` walks left-to-right, threading an accumulator: `reduce(f, [a,b,c], init)` = `f(f(f(init,a),b),c)`.
- `any`/`all` **short-circuit**: `any` stops at the first truthy, `all` stops at the first falsy — efficient on large/lazy inputs.
- These accept **any iterable** (including generators), so they chain into pipelines: `sum(map(f, filter(p, data)))`.

## Advantages

- Concise, declarative, composable transforms.
- Pure functions are easy to test and reason about; no shared-state bugs.
- Lazy iterators keep memory low and enable pipelines.
- `any`/`all` short-circuit for efficiency.

## Limitations

- Overuse (deeply nested `map`/`filter`/`lambda`) hurts readability — **comprehensions are often more Pythonic**.
- Python lacks tail-call optimization and true immutability enforcement.
- `reduce` can be less readable than an explicit loop/`sum`.
- Lambdas are limited to one expression and anonymous in tracebacks.

## Real-world Applications

Data cleaning/transformation pipelines, functional stream processing, `sorted(..., key=lambda)` everywhere, aggregations, and the conceptual basis of Spark/Pandas `apply`, MapReduce, and reactive/event pipelines.

## Interview Questions

- **Beginner:** What do `map` and `filter` do?
- **Beginner:** Difference between `any` and `all`?
- **Intermediate:** What is a pure function and why is it desirable?
- **Intermediate:** How does `reduce` work? Give an example.
- **Advanced:** Why do `map`/`filter` return iterators in Python 3, and what's the implication?
- **"Why":** Why might a comprehension be preferred over `map`/`filter`?
- **Comparison:** `map`/`filter` vs list comprehensions.
- **Scenario:** Compute the total price of in-stock items from a list of product dicts, functionally.

## Model Answers

**Q: `map` vs `filter`; `any` vs `all`.**
`map(fn, it)` **transforms** each element by applying `fn`; `filter(pred, it)` **selects** elements where `pred` is truthy. `any(iterable)` returns `True` if **at least one** element is truthy (short-circuiting on the first); `all(iterable)` returns `True` only if **every** element is truthy (short-circuiting on the first falsy). `any([])` is `False`, `all([])` is `True` (vacuous truth).

**Q: What is a pure function?**
A pure function's return value depends **only on its arguments** and it has **no side effects** (doesn't mutate external state, do I/O, or depend on globals). Same inputs always give the same output. Purity makes functions **trivially testable** (no setup/mocking of hidden state), **cacheable/memoizable**, **thread-safe**, and easy to reason about — the opposite of functions that secretly read/modify global state.

**Q (Advanced): Why do `map`/`filter` return iterators?**
For **laziness and memory efficiency**. In Python 3 they yield values on demand rather than building a full list, so they use near-constant memory and compose into pipelines that stream large data. The implication: the result is **consumed once** and has no `len()`/indexing — you must wrap in `list()`/`tuple()` if you need a reusable, indexable collection or want to iterate it twice.

**Q (Why): Comprehension over `map`/`filter`.**
A comprehension is often **more readable and Pythonic**, especially when combined with a condition: `[x*x for x in nums if x % 2 == 0]` reads more clearly than `list(map(lambda x: x*x, filter(lambda x: x%2==0, nums)))`, and avoids two lambdas. Comprehensions also don't need `list()` wrapping. Reserve `map`/`filter` for when you already have a named function (`map(str.upper, words)`) or want the lazy iterator directly.

**Q (Scenario): Total price of in-stock items.**
```python
products = [{"price": 10, "stock": 5}, {"price": 20, "stock": 0}, {"price": 7, "stock": 3}]
total = sum(p["price"] for p in products if p["stock"] > 0)   # 17
```
A generator expression filters in-stock items and projects their price, and `sum` folds them — concise, lazy, and pure. Equivalent functional form: `sum(map(lambda p: p["price"], filter(lambda p: p["stock"] > 0, products)))`, but the comprehension is clearer.

## Common Mistakes

- Forgetting `map`/`filter`/`zip` are lazy and iterating them twice (empty the second time).
- Overusing nested lambdas where a comprehension is clearer.
- Using `reduce` for sums (just use `sum`).
- Confusing `any([])`/`all([])` edge cases.
- Writing "pure" functions that secretly mutate a passed-in list (aliasing).

## Related Concepts

[Comprehensions](#31-comprehensions), [Functions](#5-functions) (lambda, first-class), [Generators](#27-generators), [Iterators](#26-iterators), `functools`, [Closures & Decorators](#28-closures--decorators).

---

# 31. Comprehensions

## What is it?

Comprehensions are **concise expressions that build a collection from an iterable in a single readable line**. Python has four: **list** `[...]`, **dict** `{k: v ...}`, **set** `{...}`, and **generator** `(...)`. They combine a transform expression, a `for` clause, and optional `if` filters.

## Why is it needed?

They replace the verbose "create empty container → loop → append/insert" pattern with one declarative line that states *what* you want. They're more readable, often faster (optimized C-level looping), and idiomatic Python. The generator variant additionally gives laziness.

## How does it work?

```python
# LIST comprehension  [expr for item in iterable if condition]
squares  = [x*x for x in range(10)]
evens    = [x for x in range(20) if x % 2 == 0]
labels   = ["even" if x%2==0 else "odd" for x in range(5)]   # ternary in expr
pairs    = [(x, y) for x in range(3) for y in range(3)]       # nested loops
flat     = [n for row in matrix for n in row]                 # flatten

# DICT comprehension
sq_map   = {x: x*x for x in range(5)}         # {0:0, 1:1, ...}
inverted = {v: k for k, v in original.items()}

# SET comprehension (unique results)
unique_lengths = {len(w) for w in words}

# GENERATOR comprehension (lazy — parentheses)
gen = (x*x for x in range(1_000_000))         # no memory upfront
total = sum(x*x for x in range(100))          # parens optional as sole arg
```

## Internal Working

- A comprehension compiles to an **optimized loop** that builds the container. List/set/dict comprehensions run in their **own scope** (a nested function), so the loop variable **doesn't leak** into the surrounding namespace (a Python 3 change from Python 2).
- Multiple `for` clauses read **left-to-right as nested loops**: `[... for x in A for y in B]` ≡ outer `x`, inner `y`.
- The **generator** version produces an iterator, computing items lazily — same syntax with `()`.
- Order in the syntax: **output expression** first, then the `for`/`if` clauses that feed it.

## Advantages

- Concise and readable; states intent declaratively.
- Often faster than an explicit append loop (C-level iteration).
- Loop variable is scoped (no leakage) in Python 3.
- Generator version adds laziness for free.

## Limitations

- Nested/multi-clause comprehensions with conditionals become **unreadable** — use a loop instead.
- Can't include statements (only expressions), so no `try`/complex logic inside.
- Eager list/dict/set comprehensions on huge data waste memory (use a generator).
- Overly clever one-liners hurt maintainability.

## Real-world Applications

Data transformation and filtering, building lookup maps (dict comprehensions), deduplicating (set comprehensions), flattening nested data, quick projections from lists of records/dicts, and constructing test data — ubiquitous in data-processing and everyday Python.

## Interview Questions

- **Beginner:** Write a list comprehension for squares of even numbers 0–20.
- **Beginner:** What are the four kinds of comprehension?
- **Intermediate:** Difference between a list comprehension and a generator expression?
- **Intermediate:** How do you filter and transform in one comprehension?
- **Advanced:** How do nested `for` clauses in a comprehension execute?
- **"Why":** Why doesn't the loop variable leak in Python 3 comprehensions?
- **Comparison:** Comprehension vs `map`/`filter` vs explicit loop.
- **Scenario:** Build a dict mapping each word to its length, only for words longer than 3 letters.

## Model Answers

**Q: List comprehension vs generator expression.**
Same syntax, different brackets: `[...]` builds a **full list eagerly** (all elements stored, indexable, reusable, memory ∝ size); `(...)` builds a **lazy generator** (computes on demand, one-time use, near-zero memory). Choose the list when you need random access, `len`, or repeated iteration; choose the generator for large/streamed data consumed once (e.g., feeding `sum`/`any`/`max`).

**Q (Advanced): How do nested `for` clauses execute?**
They execute as **nested loops in written order, left = outer**. `[(x,y) for x in A for y in B]` iterates every `x` in `A`, and for each `x` iterates every `y` in `B` — equivalent to:
```python
result = []
for x in A:
    for y in B:
        result.append((x, y))
```
A common use is flattening: `[n for row in matrix for n in row]` (outer over rows, inner over elements).

**Q (Why): Why no loop-variable leak in Python 3?**
Python 3 evaluates each comprehension in its **own implicit function scope**, so the loop variable lives only inside that scope and is discarded afterward. In Python 2, comprehensions shared the enclosing scope and would overwrite an outer variable of the same name — a bug source. The Python 3 change prevents accidental clobbering and keeps comprehensions self-contained.

**Q (Scenario): Word→length dict for words >3 letters.**
```python
words = ["hi", "hello", "yo", "python"]
lengths = {w: len(w) for w in words if len(w) > 3}
# {'hello': 5, 'python': 6}
```
A dict comprehension with an `if` filter: the `for` iterates words, the `if` keeps only those longer than 3, and `w: len(w)` builds each key-value pair — one readable line replacing an init-loop-assign block.

## Common Mistakes

- Writing deeply nested comprehensions that are unreadable (prefer a loop).
- Using a list comprehension where a generator would save memory.
- Forgetting dict comprehensions need `key: value`, not just an expression.
- Trying to put statements (`print`, `try`) inside a comprehension.
- Confusing `{...}` set vs dict comprehension (dict needs a colon).

## Related Concepts

[Functional Programming](#30-functional-programming), [Generators](#27-generators), [Lists](#6-lists)/[Dictionaries](#8-dictionaries)/[Sets](#9-sets), [Iterators](#26-iterators), [Control Flow](#4-control-flow).

---

# 32. Type Hints

## What is it?

**Type hints** (PEP 484) are optional **annotations** declaring the expected types of variables, parameters, and return values (`def add(x: int, y: int) -> int:`). They are **not enforced at runtime** — Python ignores them for execution — but power **static type checkers** (mypy, pyright), IDE autocomplete, and documentation. The `typing` module provides tools: `Optional`, `Union`, `Literal`, `TypedDict`, `Protocol`, `Generic`, and more.

## Why is it needed?

In large, dynamically-typed codebases, "what type is this?" becomes a real cost. Type hints add **machine-checkable documentation**: tools catch type errors *before* running the code, IDEs give better autocomplete and refactoring, and readers understand interfaces at a glance — recovering some of static typing's safety without losing Python's flexibility.

## How does it work?

```python
from typing import Optional, Union, Literal, TypedDict, Protocol
def greet(name: str, times: int = 1) -> str:
    return f"Hi {name} " * times

age: int = 30                       # variable annotation
scores: list[int] = [90, 85]        # 3.9+ builtin generics
mapping: dict[str, int] = {}

def find(key: str) -> Optional[int]: # Optional[int] == Union[int, None]
    ...                              # may return int or None

def parse(x: Union[int, str]) -> int:  # accepts int OR str  (3.10+: int | str)
    return int(x)

Mode = Literal["r", "w", "a"]        # only these exact values allowed
def open_file(mode: Mode): ...

class Movie(TypedDict):              # a dict with a fixed typed shape
    title: str
    year: int

class Drawable(Protocol):           # structural typing (duck typing, checked)
    def draw(self) -> None: ...
def render(obj: Drawable) -> None:  # anything with draw() satisfies this
    obj.draw()
```

## Internal Working

- Annotations are stored in `__annotations__` (on functions/modules/classes) but **not checked by the interpreter** — `def f(x: int)` happily accepts a string at runtime. They're metadata.
- **Static checkers** (mypy/pyright) parse the code and annotations *without running it* and report inconsistencies (`error: Argument 1 has incompatible type "str"; expected "int"`).
- **`Optional[X]`** is shorthand for `Union[X, None]` — it means "X or None", **not** "optional argument". **`Union[A, B]`** means "A or B" (3.10+: `A | B`).
- **`Protocol`** enables **structural subtyping**: a class satisfies `Drawable` if it *has* a `draw()` method, without explicitly inheriting — formalizing duck typing for the type checker.
- **`Generic`/`TypeVar`** let you write type-parameterized classes/functions (`list[T]`), and **`TypedDict`** describes the exact key/value types of a dict.

## Advantages

- Catches type bugs statically, before runtime.
- Self-documenting signatures; better IDE autocomplete/refactoring.
- Scales large codebases and team collaboration.
- No runtime cost (ignored during execution).

## Limitations

- **Not enforced at runtime** — bad types still get through unless you add explicit checks or use `pydantic`.
- Adds verbosity; complex generics can be hard to read.
- Requires running a separate checker in CI to get value.
- Some dynamic patterns are hard to type precisely.

## Real-world Applications

Public library APIs (typed signatures for users), large applications with mypy/pyright in CI, `pydantic`/FastAPI (which *do* use hints for runtime validation and serialization), dataclasses (fields declared via hints), and editor tooling for autocomplete and inline error detection.

## Interview Questions

- **Beginner:** Are type hints enforced at runtime?
- **Beginner:** What does `-> int` mean in a function signature?
- **Intermediate:** Difference between `Optional[int]` and `Union[int, str]`?
- **Intermediate:** What tool checks type hints and when?
- **Advanced:** What is a `Protocol` and how does it relate to duck typing?
- **"Why":** Why add type hints to a dynamically-typed language?
- **Comparison:** `TypedDict` vs `dataclass` vs `dict`.
- **Scenario:** A function may return a `User` or `None`. How do you annotate it and what does that force callers to do?

## Model Answers

**Q: Are type hints enforced at runtime?**
No. Python **ignores them during execution** — they're stored in `__annotations__` as metadata but don't cause runtime type checks. `def f(x: int)` will run fine with `f("hello")`. Their value comes from **static analysis tools** (mypy, pyright) and IDEs that read the hints *without running the code* to flag mismatches and power autocomplete. For runtime enforcement you need explicit checks or a library like `pydantic`.

**Q: `Optional[int]` vs `Union[int, str]`.**
`Optional[int]` is exactly `Union[int, None]` — the value is **either an `int` or `None`**. It does **not** mean "optional parameter" (that's about defaults). `Union[int, str]` means the value is **either an `int` or a `str`**. In modern Python (3.10+) you can write these as `int | None` and `int | str`. A common gotcha: `Optional` is about *None-ability*, not argument optionality.

**Q (Advanced): What is a `Protocol`?**
A `Protocol` (PEP 544) defines an interface by **structure**: any class that has the required methods/attributes **satisfies it automatically**, without explicitly subclassing. It's **static duck typing** — `render(obj: Drawable)` accepts anything with a `draw()` method, and the type checker verifies that at analysis time. Protocols give you the flexibility of duck typing *plus* static safety, and work with third-party classes you can't modify.

**Q (Why): Why type-hint a dynamic language?**
Because at scale, dynamic typing's "figure out the type by reading everything" becomes expensive and error-prone. Hints add **checkable documentation**: tools catch a whole class of bugs (wrong argument types, `None` misuse) before the code runs, IDEs offer accurate autocomplete/refactoring, and interfaces become self-explanatory — all **without sacrificing** Python's dynamic execution, since hints are optional and ignored at runtime.

**Q (Scenario): Return `User` or `None`.**
```python
from typing import Optional
def find_user(uid: int) -> Optional[User]:   # or: User | None
    ...
```
The `Optional[User]` return type tells the checker (and readers) the result **might be `None`**, so a type checker will **flag any code that uses the result without first handling `None`** (e.g., `find_user(1).name` errors until you guard with `if user is not None:`). This prevents the classic `AttributeError: 'NoneType' object has no attribute ...` bug.

## Common Mistakes

- Believing hints are enforced at runtime.
- Misreading `Optional` as "optional argument" instead of "or None".
- Not running a type checker in CI, so hints drift out of sync with code.
- Over-engineering complex generics that hurt readability.
- Using `list`/`dict` bare without element types when precision matters.

## Related Concepts

[Functions](#5-functions) (annotations), [Dataclasses](#21-dataclasses), [Abstraction](#15-abstraction) (`Protocol`), [Modules & Packages](#12-modules--packages) (`typing`), mypy/pyright, pydantic.

---

# 33. Memory Management

## What is it?

Memory management is how Python **allocates and frees memory for objects** automatically. Its pillars: **reference counting** (the primary, immediate reclamation mechanism), a **cyclic garbage collector** (`gc`) for reference cycles, and the semantic distinctions of **mutable vs immutable** objects and **shallow vs deep copy** (`copy` module). You never `malloc`/`free` manually — but you must understand these to avoid aliasing bugs and leaks.

## Why is it needed?

Manual memory management (C) is error-prone (leaks, dangling pointers, double frees). Python automates it so you focus on logic. But automation isn't magic: understanding reference counting explains **aliasing bugs** (`b = a` shares an object), the mutable-default trap, why `is` vs `==` differ, and how to correctly **copy** nested structures — all common interview and real-world pitfalls.

## How does it work?

```python
import sys, copy

a = [1, 2, 3]
b = a                     # NO copy — both names reference the SAME list
b.append(4)
a                         # [1, 2, 3, 4]  <- aliasing!

sys.getrefcount([1])      # shows reference count (temporarily inflated by the call)

# Mutable vs immutable
x = 5; y = x; y += 1      # ints immutable -> x still 5 (rebinding, not mutation)
lst = [1]; m = lst; m.append(2)   # lists mutable -> lst is [1, 2]

# Shallow vs deep copy
original = [[1, 2], [3, 4]]
shallow  = copy.copy(original)     # new outer list, SAME inner lists
deep     = copy.deepcopy(original) # fully independent copy at all levels
shallow[0].append(99)              # also changes original[0]!  (shared inner)
deep[0].append(99)                 # original untouched
```

## Internal Working

- **Reference counting:** every object has a counter of how many references point to it. Binding a name, appending to a list, passing as an argument → count up; rebinding, `del`, going out of scope → count down. When the count hits **zero**, the object is **freed immediately**. This is deterministic and prompt.
- **Reference cycles:** counting alone can't free `a → b → a` (each keeps the other's count ≥1 even when unreachable). CPython's **cyclic garbage collector** (`gc`) periodically finds and collects such unreachable cycles, using a generational algorithm (young objects checked more often).
- **Interning/caching:** small ints (−5..256) and some short strings are cached and reused — an optimization that explains surprising `is` results.
- **Mutable vs immutable:** immutable objects (`int`, `str`, `tuple`, `frozenset`) can't change in place — "modifying" rebinds to a new object. Mutable objects (`list`, `dict`, `set`) change in place, so multiple references see the change.
- **Assignment never copies** — it binds a name to the *same* object. Copies require `[:]`, `.copy()`, `copy.copy` (shallow), or `copy.deepcopy` (recursive).

## Advantages

- Automatic — no manual free/leak/dangling-pointer bugs.
- Reference counting reclaims memory **promptly and deterministically**.
- `gc` handles cycles the counter can't.
- `copy`/`deepcopy` give explicit control when you need independence.

## Limitations

- Reference counting adds per-operation overhead and (historically) motivates the **GIL** (see Concurrency).
- Cycles need the separate `gc`, whose runs are non-deterministic.
- Aliasing/mutability bugs are easy to hit and hard to spot.
- `deepcopy` can be slow and can choke on non-copyable objects.

## Real-world Applications

Understanding this prevents shared-state bugs in caches/config, correct copying of nested data before mutation, tuning `gc` for latency-sensitive apps, diagnosing memory leaks (lingering references), and knowing why default-mutable-argument and aliasing bugs occur.

## Interview Questions

- **Beginner:** Does `b = a` copy the list?
- **Beginner:** Difference between mutable and immutable objects?
- **Intermediate:** How does Python free memory? What is reference counting?
- **Intermediate:** Difference between shallow and deep copy?
- **Advanced:** Why does Python need a cyclic garbage collector in addition to reference counting?
- **"Why":** Why does modifying a list inside a function affect the caller's list?
- **Comparison:** `copy.copy` vs `copy.deepcopy` vs slicing.
- **Scenario:** You pass a nested config dict into a function that modifies it and the caller's copy changes unexpectedly. Diagnose and fix.

## Model Answers

**Q: How does Python free memory / what is reference counting?**
Every object tracks a **reference count** — how many references point to it. Operations that add a reference (assignment, insertion into a container, argument passing) increment it; removing references (`del`, reassignment, scope exit) decrement it. When the count reaches **zero**, the object is unreachable and CPython **immediately deallocates it**. This makes most cleanup prompt and deterministic. A supplementary **cyclic GC** handles objects trapped in reference cycles that never reach zero.

**Q: Shallow vs deep copy.**
A **shallow copy** (`copy.copy`, `list[:]`, `.copy()`) creates a **new outer container but reuses references to the same inner objects** — so mutating a nested element affects both copies. A **deep copy** (`copy.deepcopy`) **recursively copies everything**, producing a fully independent structure. Use shallow when the elements are immutable or you *want* sharing; use deep when you need to mutate nested data without touching the original.

**Q (Advanced): Why a cyclic GC on top of reference counting?**
Reference counting can't reclaim **reference cycles**: if object A references B and B references A, each keeps the other's count at ≥1 even when nothing else can reach them — a leak. CPython adds a **generational cyclic garbage collector** that periodically detects groups of mutually-referencing, otherwise-unreachable objects and frees them. So counting handles the common acyclic case immediately, and `gc` cleans up the cycles it can't.

**Q (Why): Modifying a list in a function affects the caller.**
Python passes arguments by **object reference** (sometimes called "pass by object reference/assignment"): the parameter and the caller's variable point to the **same object**. If the function **mutates** that object in place (`lst.append(x)`), the caller sees the change because there's only one list. If the function **rebinds** the parameter (`lst = [...]`), that only changes the local name and the caller is unaffected. Mutation propagates; rebinding doesn't.

**Q (Scenario): Nested config mutated unexpectedly.**
The function received the **same** dict (and same nested dicts) the caller holds; mutating it in place changed the caller's data through aliasing. Fixes: (1) don't mutate inputs — build and return a new structure; or (2) make an independent copy first with `copy.deepcopy(config)` (a shallow copy is insufficient because the config is *nested* — inner dicts would still be shared). Prefer treating inputs as immutable to avoid the whole class of bug.

## Common Mistakes

- Assuming `b = a` copies (it aliases).
- Using a shallow copy on nested data and being surprised inner objects are shared.
- Mutating function arguments and affecting the caller.
- Mutable default arguments (shared object across calls).
- Relying on `is`/interning for value equality.

## Related Concepts

[Data Types](#2-data-types--type-conversion) (mutability), [Lists](#6-lists) (aliasing), [Functions](#5-functions) (default args, pass semantics), `copy`, `gc`, [Concurrency](#34-concurrency) (GIL).

---

# 34. Concurrency

## What is it?

Concurrency is **structuring a program to make progress on multiple tasks**. Python offers three models: **`threading`** (multiple threads in one process, sharing memory), **`multiprocessing`** (multiple processes, separate memory, true parallelism), and **`asyncio`** (single-threaded cooperative multitasking with `async`/`await`). The crucial constraint is the **GIL (Global Interpreter Lock)**, which lets only **one thread execute Python bytecode at a time** in CPython.

## Why is it needed?

Programs often wait — for network responses, disk, databases (**I/O-bound**) — or need to crunch numbers on many cores (**CPU-bound**). Concurrency lets you overlap I/O waits (do useful work while waiting) and, via multiprocessing, use multiple CPU cores. Choosing the right model for the workload is the key skill.

## How does it work?

```python
# THREADING — great for I/O-bound (GIL released during I/O waits)
import threading
def download(url): ...            # spends time waiting on network
threads = [threading.Thread(target=download, args=(u,)) for u in urls]
for t in threads: t.start()
for t in threads: t.join()

# MULTIPROCESSING — great for CPU-bound (bypasses the GIL, true parallelism)
from multiprocessing import Pool
def heavy(n): return sum(i*i for i in range(n))
with Pool(4) as p:
    results = p.map(heavy, [10**6]*4)   # runs on 4 cores in parallel

# ASYNCIO — great for MANY concurrent I/O tasks, single thread
import asyncio
async def fetch(url):
    await asyncio.sleep(1)         # non-blocking wait; yields control
    return url
async def main():
    return await asyncio.gather(*(fetch(u) for u in urls))  # concurrent
asyncio.run(main())
```

**Decision rule:** I/O-bound with a few tasks → **threading**; I/O-bound with many tasks → **asyncio**; CPU-bound → **multiprocessing**.

## Internal Working

- **The GIL** is a mutex in CPython ensuring only one thread runs Python bytecode at a time, protecting the interpreter's internal state (including reference counts). Consequence: threads **do not** give CPU parallelism for pure-Python compute — CPU-bound threaded code runs about as fast as single-threaded (plus overhead).
- **Threads still help I/O-bound work** because the GIL is **released during blocking I/O** (and by many C extensions like NumPy), so while one thread waits on the network, another runs.
- **Multiprocessing** spawns **separate processes**, each with its **own interpreter and GIL**, achieving true multi-core parallelism — at the cost of higher memory and **inter-process communication** (data is pickled between processes).
- **asyncio** is **single-threaded cooperative** concurrency: an **event loop** runs coroutines that voluntarily yield control at `await` points (usually awaiting I/O). No GIL contention, no thread overhead — but a blocking call (or heavy CPU work) inside a coroutine **freezes the whole loop**.

## Advantages

- **threading:** simple shared-memory model; effective for I/O concurrency.
- **multiprocessing:** true parallelism across cores; sidesteps the GIL for CPU work.
- **asyncio:** massive I/O concurrency (thousands of tasks) with low overhead and no locks in single-threaded code.

## Limitations

- **GIL** blocks CPU parallelism for threads in CPython.
- **threading:** shared mutable state → race conditions, needs locks; deadlock risk.
- **multiprocessing:** heavy (process startup, memory), IPC overhead, data must be picklable.
- **asyncio:** requires async-aware libraries; one blocking/CPU-heavy call stalls everything; steeper mental model.

## Real-world Applications

Web servers handling many simultaneous connections (asyncio — FastAPI/aiohttp), concurrent API/web scraping (threading or asyncio), parallel data processing / numeric batch jobs (multiprocessing), background task queues, and pipelines overlapping I/O with computation.

## Interview Questions

- **Beginner:** What is the GIL?
- **Beginner:** Difference between a thread and a process?
- **Intermediate:** When would you use threading vs multiprocessing vs asyncio?
- **Intermediate:** Why doesn't threading speed up CPU-bound Python code?
- **Advanced:** How does asyncio achieve concurrency without threads?
- **"Why":** Why does the GIL exist if it limits parallelism?
- **Comparison:** Concurrency vs parallelism.
- **Scenario:** You must scrape 10,000 URLs quickly, then run a heavy CPU aggregation. Which tools for each phase?

## Model Answers

**Q: What is the GIL?**
The **Global Interpreter Lock** is a mutex in CPython that permits only **one thread to execute Python bytecode at a time**. It simplifies memory management (protecting reference counts) and makes single-threaded code fast and C-extension integration easier. The trade-off: multithreaded **CPU-bound** Python can't use multiple cores simultaneously. It's released during I/O, so I/O-bound threading still benefits — and multiprocessing bypasses it entirely by using separate processes.

**Q: threading vs multiprocessing vs asyncio.**
- **threading:** best for **I/O-bound** tasks with a **modest** number of concurrent operations; shares memory; limited by the GIL for CPU work.
- **multiprocessing:** best for **CPU-bound** work — separate processes give **true parallelism** across cores, at the cost of memory and IPC.
- **asyncio:** best for **I/O-bound** work with **many** (thousands of) concurrent tasks; single-threaded cooperative scheduling, very low overhead, but needs async libraries and must never block the loop.

**Q: Why doesn't threading speed up CPU-bound code?**
Because CPU-bound code holds the **GIL** while executing bytecode, and the GIL allows only one thread to run Python at a time. Multiple CPU-bound threads therefore **take turns** rather than running in parallel, so you get no speedup (and even a slight slowdown from lock contention and context switching). For CPU parallelism you need **multiprocessing** (separate interpreters/GILs) or a C extension that releases the GIL.

**Q (Advanced): How does asyncio achieve concurrency without threads?**
Via an **event loop** running **coroutines** that use **cooperative multitasking**. When a coroutine hits `await` on an I/O operation, it **yields control back to the loop** instead of blocking; the loop runs other ready coroutines meanwhile, and resumes the first when its I/O completes. It's all on **one thread**, so there's no GIL contention or locking for shared state — but the model relies on tasks voluntarily yielding, so any blocking or CPU-heavy call inside a coroutine stalls the entire loop.

**Q (Comparison): Concurrency vs parallelism.**
**Concurrency** is *dealing with* many tasks by interleaving their progress (structure) — one core rapidly switching between tasks counts. **Parallelism** is *doing* many tasks **at literally the same instant** on multiple cores (execution). asyncio and threading (under the GIL) give concurrency; multiprocessing gives true parallelism. Concurrency is about program structure; parallelism is about simultaneous execution.

**Q (Scenario): Scrape 10k URLs, then heavy aggregation.**
Two phases, two tools. **Phase 1 (I/O-bound, 10k tasks):** use **asyncio** (e.g., `aiohttp` + `asyncio.gather`) to fire thousands of requests concurrently on one thread with minimal overhead — the GIL is irrelevant since the work is waiting on the network. **Phase 2 (CPU-bound aggregation):** use **multiprocessing** (`Pool.map`) to spread the computation across cores for true parallelism, bypassing the GIL. Matching the model to the bottleneck (I/O vs CPU) is the whole point.

## Common Mistakes

- Using threading for CPU-bound work and expecting a speedup (GIL blocks it).
- Blocking (or heavy CPU) inside an `async` coroutine, freezing the event loop.
- Ignoring race conditions on shared state in threads (missing locks).
- Sharing non-picklable objects or huge data across processes (IPC/pickle errors).
- Confusing concurrency with parallelism.

## Related Concepts

[Memory Management](#33-memory-management) (GIL, refcounts), [Generators](#27-generators) (async generators), [Context Managers](#29-context-managers) (locks), [Iterators](#26-iterators), I/O and networking.

---

*End of theory guide. Pair this with `practical.md` for coding exercises, notebook workflows, and hands-on interview questions covering the same syllabus.*


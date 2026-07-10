# Python — Practical & Coding Assessment Guide

> Companion to `theory.md`. This document is the **hands-on** half: predicted coding questions, production-quality solutions, complexity analysis, alternatives, interview variations, and follow-ups — organized by the same syllabus (Fundamentals → OOP → Exceptions → Advanced).
>
> **How to use:** Try each problem yourself **before** reading the solution. In an assessment, always (1) restate the problem, (2) state your approach and complexity, (3) code cleanly, (4) test with edge cases. The "Approach" sections model exactly that narration.

## Table of Contents

1. [Fundamentals & Strings](#part-1--fundamentals--strings)
2. [Collections (List / Dict / Set / Tuple)](#part-2--collections)
3. [Functions, Recursion & Functional Programming](#part-3--functions-recursion--functional-programming)
4. [File Handling (Text / CSV / JSON)](#part-4--file-handling)
5. [Object-Oriented Programming](#part-5--object-oriented-programming)
6. [Exception Handling & Logging](#part-6--exception-handling--logging)
7. [Iterators, Generators, Decorators, Context Managers](#part-7--advanced-python)
8. [Data-Structure Implementations (DSA)](#part-8--data-structure-implementations)
9. [Coding Question Bank (Easy / Medium / Hard)](#part-9--coding-question-bank)
10. [Notebook-Style Workflow (Data Processing)](#part-10--notebook-style-workflow)

---

# Part 1 — Fundamentals & Strings

## Practical Question 1

**Difficulty:** Easy
**Estimated Time:** 10 min
**Concepts Tested:** strings, immutability, two-pointer, slicing

**Problem Statement:** Check whether a string is a **palindrome**, ignoring case, spaces, and punctuation.

**Example Input:** `"A man, a plan, a canal: Panama"`
**Example Output:** `True`

**Approach**
1. Normalize: keep only alphanumeric characters, lowercase them.
2. Compare the cleaned string to its reverse — or use two pointers from both ends moving inward.
3. Two pointers avoid building a reversed copy (O(1) extra space).

## Python Implementation

```python
def is_palindrome(s: str) -> bool:
    # Two-pointer, O(1) extra space (besides the filtered view)
    left, right = 0, len(s) - 1
    while left < right:
        while left < right and not s[left].isalnum():   # skip non-alphanumerics
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        if s[left].lower() != s[right].lower():          # case-insensitive compare
            return False
        left += 1
        right -= 1
    return True
```

- `isalnum()` filters punctuation/spaces without regex.
- Comparing `lower()` per char handles case.
- **Time:** O(n) — each character visited once. **Space:** O(1) extra.

## Alternative Solution

```python
def is_palindrome_simple(s: str) -> bool:
    cleaned = [c.lower() for c in s if c.isalnum()]   # O(n) space
    return cleaned == cleaned[::-1]                    # slice-reverse compare
```
Cleaner to read, but uses **O(n)** extra space for the cleaned list and its reverse. Prefer this in interviews for clarity unless asked to optimize space.

## Interview Variations

- Return the **longest palindromic substring** (expand-around-center, O(n²)).
- Check if the string can become a palindrome by removing **at most one** character.
- Palindrome for **numbers** without converting to string.

## Common Follow-up Questions

- "Why are strings immutable — how does that affect reversing?" (Reverse creates a new object; you can't reverse in place.)
- "What's the complexity of `s[::-1]`?" (O(n) time and space.)
- "How would you handle Unicode / accented characters?" (Normalize with `unicodedata`.)

---

## Practical Question 2

**Difficulty:** Easy–Medium
**Estimated Time:** 15 min
**Concepts Tested:** dictionaries, counting, strings, sorting

**Problem Statement:** Return the **first non-repeating character** in a string; if none, return `None`.

**Example Input:** `"aabbcdd"`
**Example Output:** `"c"`

**Approach**
1. Count each character's frequency in one pass with a dict.
2. Iterate the string again in order; return the first character whose count is 1.
3. Two passes = O(n); relies on dict insertion order / a second scan for "first".

## Python Implementation

```python
from collections import Counter

def first_unique_char(s: str) -> str | None:
    counts = Counter(s)                 # pass 1: O(n) frequency map
    for ch in s:                        # pass 2: preserve original order
        if counts[ch] == 1:
            return ch
    return None
```

- `Counter` builds the frequency dict in one line.
- Second loop over `s` guarantees we return the **first** by position.
- **Time:** O(n). **Space:** O(k) where k = distinct characters (≤ alphabet size).

## Alternative Solution

Single structured pass storing first-seen index, then pick min index with count 1 — but two clean passes are simpler and equally O(n). For streaming data, an `OrderedDict` of counts also works.

## Interview Variations

- Return the **index** instead of the character.
- Find the first character that **repeats**.
- Do it for a **stream** where characters arrive one at a time (maintain a queue of candidates).

## Common Follow-up Questions

- "Why `Counter` over a plain dict?" (Purpose-built, `most_common`, cleaner.)
- "What if the string is huge and can't fit in memory?" (Stream + counts; two passes over a file.)
- "How does dict ordering help here?" (Insertion order lets the second pass find the earliest.)

---

## Practical Question 3

**Difficulty:** Medium
**Estimated Time:** 15 min
**Concepts Tested:** anagrams, hashing, sorting, dict comparison

**Problem Statement:** Given a list of words, **group anagrams** together.

**Example Input:** `["eat", "tea", "tan", "ate", "nat", "bat"]`
**Example Output:** `[["eat","tea","ate"], ["tan","nat"], ["bat"]]`

**Approach**
1. Anagrams share the same multiset of letters → their **sorted letters** form a canonical key.
2. Use a `defaultdict(list)` mapping key → list of words.
3. Return the dict's values.

## Python Implementation

```python
from collections import defaultdict

def group_anagrams(words: list[str]) -> list[list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for word in words:
        key = "".join(sorted(word))     # canonical signature; anagrams collide here
        groups[key].append(word)
    return list(groups.values())
```

- Sorting each word gives an anagram-invariant key.
- `defaultdict(list)` avoids manual "key exists?" checks.
- **Time:** O(n · k log k) for n words of length k (sorting each). **Space:** O(n · k).

## Alternative Solution

Use a **character-count tuple** as the key (`tuple(count of each letter)`), giving **O(n · k)** — faster than sorting for long words:
```python
key = tuple(sorted(Counter(word).items()))   # or a 26-length count tuple
```

## Interview Variations

- Return only the **largest** anagram group.
- **Count** how many anagram groups exist.
- Check if two strings are anagrams (single comparison of `Counter`s).

## Common Follow-up Questions

- "Sorted-key vs count-key complexity trade-off?"
- "Why must the key be immutable?" (Dict keys must be hashable → tuple/str, not list.)
- "How would you handle Unicode or case-insensitivity?"

---

# Part 2 — Collections

## Practical Question 4

**Difficulty:** Easy–Medium
**Estimated Time:** 15 min
**Concepts Tested:** lists, hash sets, two-pointer, complexity trade-offs

**Problem Statement:** Given an array of integers and a target, return the **indices of two numbers** that add up to the target (classic "Two Sum").

**Example Input:** `nums = [2, 7, 11, 15], target = 9`
**Example Output:** `[0, 1]` (because 2 + 7 = 9)

**Approach**
1. Brute force checks all pairs → O(n²).
2. Better: one pass with a hash map `value → index`. For each `x`, check if `target - x` was already seen.
3. Trades O(n) space for O(n) time.

## Python Implementation

```python
def two_sum(nums: list[int], target: int) -> list[int]:
    seen: dict[int, int] = {}                 # value -> index
    for i, x in enumerate(nums):
        complement = target - x
        if complement in seen:                # O(1) lookup
            return [seen[complement], i]
        seen[x] = i
    return []                                  # no pair found
```

- The hash map turns the "have I seen the complement?" question into O(1).
- We store **after** checking, so we never pair an element with itself.
- **Time:** O(n). **Space:** O(n).

## Alternative Solution

If the array is **sorted** (or you sort it, losing original indices), a **two-pointer** approach uses O(1) space:
```python
def two_sum_sorted(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        s = nums[lo] + nums[hi]
        if s == target: return [lo, hi]
        if s < target:  lo += 1
        else:           hi -= 1
    return []
```

## Interview Variations

- **Three Sum** (find triplets summing to 0).
- Return **all** pairs, not just the first.
- Count pairs with a given **difference**.

## Common Follow-up Questions

- "Why is the hash-map version O(n) not O(n²)?"
- "What if there are duplicates?"
- "Space-time trade-off: when would you prefer two-pointer?"

---

## Practical Question 5

**Difficulty:** Medium
**Estimated Time:** 20 min
**Concepts Tested:** dict, ordering, LRU semantics, OrderedDict / doubly linked list

**Problem Statement:** Implement an **LRU (Least Recently Used) cache** with O(1) `get` and `put`.

**Example**
```
cache = LRUCache(2)
cache.put(1, "a"); cache.put(2, "b")
cache.get(1)          # "a"  (1 is now most-recently-used)
cache.put(3, "c")     # evicts key 2 (least recently used)
cache.get(2)          # None
```

**Approach**
1. Need fast lookup (dict) **and** fast "move to most-recent" / "evict oldest" ordering.
2. `collections.OrderedDict` gives both: `move_to_end` and `popitem(last=False)` are O(1).
3. On `get`, mark used; on `put`, insert/update and evict if over capacity.

## Python Implementation

```python
from collections import OrderedDict
from typing import Any

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: OrderedDict[int, Any] = OrderedDict()

    def get(self, key: int) -> Any:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)          # mark as most-recently-used (O(1))
        return self.cache[key]

    def put(self, key: int, value: Any) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)      # refresh recency
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)   # evict least-recently-used (front)
```

- `OrderedDict` maintains insertion/access order; front = oldest, back = newest.
- Every access moves the key to the back; eviction pops the front.
- **Time:** O(1) for both operations. **Space:** O(capacity).

## Alternative Solution

Implement manually with a **hash map + doubly linked list** (what `OrderedDict` does internally): the map gives O(1) node lookup, the linked list gives O(1) reordering/eviction. This is the "prove you understand it" version interviewers sometimes require.

## Interview Variations

- **LFU** (Least *Frequently* Used) cache.
- Add **TTL** (time-based expiry).
- Make it **thread-safe** (wrap operations in a lock).

## Common Follow-up Questions

- "Why not a plain dict?" (No O(1) 'oldest' tracking pre-ordering logic.)
- "How does `functools.lru_cache` relate?" (Same idea, decorator form.)
- "Where's the doubly linked list hiding?" (Inside `OrderedDict`.)

---

## Practical Question 6

**Difficulty:** Easy
**Estimated Time:** 10 min
**Concepts Tested:** sets, membership, deduplication, order preservation

**Problem Statement:** Remove duplicates from a list **while preserving original order**.

**Example Input:** `[3, 1, 3, 2, 1, 5]`
**Example Output:** `[3, 1, 2, 5]`

**Approach**
1. A `set(list)` dedupes but **loses order**.
2. Track a `seen` set while iterating; append each element the first time only.

## Python Implementation

```python
def dedupe_ordered(items: list) -> list:
    seen = set()
    result = []
    for x in items:
        if x not in seen:          # O(1) membership
            seen.add(x)
            result.append(x)
    return result
```

- `seen` gives O(1) membership; `result` preserves first-seen order.
- **Time:** O(n). **Space:** O(n).
- One-liner (3.7+): `list(dict.fromkeys(items))` — dict keys are unique and ordered.

## Alternative Solution

`list(dict.fromkeys(items))` is the idiomatic, fastest-to-write version and preserves order because dicts keep insertion order.

## Interview Variations

- Dedupe but keep the **last** occurrence's position.
- Dedupe a list of **dicts** by a key field.
- Dedupe **unhashable** items (fall back to O(n²) with a list of seen).

## Common Follow-up Questions

- "Why does `set()` lose order?" (Hash-table layout, not insertion order.)
- "What if elements aren't hashable?" (Can't use a set; O(n²) or make them hashable.)

---

# Part 3 — Functions, Recursion & Functional Programming

## Practical Question 7

**Difficulty:** Easy–Medium
**Estimated Time:** 15 min
**Concepts Tested:** recursion, base case, memoization, iterative conversion

**Problem Statement:** Compute the **nth Fibonacci number**. Discuss naive recursion, memoized, and iterative.

**Example Input:** `n = 10`
**Example Output:** `55`

**Approach**
1. Naive recursion mirrors the math but recomputes subproblems → **O(2ⁿ)**.
2. **Memoize** to cache results → **O(n)**.
3. **Iterative** with two rolling variables → **O(n)** time, **O(1)** space (best).

## Python Implementation

```python
from functools import lru_cache

# 1. Memoized recursion — O(n) time, O(n) space
@lru_cache(maxsize=None)
def fib_memo(n: int) -> int:
    if n < 2:
        return n                       # base cases: fib(0)=0, fib(1)=1
    return fib_memo(n - 1) + fib_memo(n - 2)

# 2. Iterative — O(n) time, O(1) space (production choice)
def fib_iter(n: int) -> int:
    if n < 2:
        return n
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr  # roll forward
    return curr
```

- `@lru_cache` transparently caches, collapsing exponential recursion to linear.
- The iterative version keeps only the last two numbers — constant space.
- **Naive** (shown in theory) is O(2ⁿ) — never use for large n.

## Alternative Solution

Matrix exponentiation or Binet's formula gives **O(log n)** — worth mentioning for "can you do better than O(n)?" follow-ups.

## Interview Variations

- Return the **whole sequence** up to n (generator!).
- **Tribonacci** (sum of last three).
- Detect if a number **is** a Fibonacci number.

## Common Follow-up Questions

- "Why is naive recursion exponential?" (Overlapping subproblems recomputed.)
- "What does `lru_cache` do internally?" (Memo dict keyed by args.)
- "Convert recursion to iteration — why does it save space?" (No call stack of n frames.)

---

## Practical Question 8

**Difficulty:** Medium
**Estimated Time:** 15 min
**Concepts Tested:** recursion, trees/nested structures, accumulation

**Problem Statement:** Given an arbitrarily **nested list** of integers, return the **flat sum** of all integers.

**Example Input:** `[1, [2, [3, 4], 5], [6]]`
**Example Output:** `21`

**Approach**
1. Recursion naturally handles unknown depth.
2. For each element: if it's a list, recurse; else add it.
3. Base case is implicit — an empty list contributes 0.

## Python Implementation

```python
def nested_sum(data: list) -> int:
    total = 0
    for item in data:
        if isinstance(item, list):
            total += nested_sum(item)    # recurse into sublists
        else:
            total += item                # base: a plain number
    return total
```

- `isinstance(item, list)` decides recurse-vs-add.
- **Time:** O(n) over total elements. **Space:** O(d) recursion depth (d = nesting depth).

## Alternative Solution

Iterative with an explicit **stack** (avoids recursion-limit issues on very deep nesting):
```python
def nested_sum_iter(data):
    stack, total = list(data), 0
    while stack:
        item = stack.pop()
        if isinstance(item, list): stack.extend(item)
        else:                       total += item
    return total
```

## Interview Variations

- **Flatten** the nested list instead of summing (generator with `yield from`).
- Handle nested **dicts** too.
- Sum only integers at a **given depth**.

## Common Follow-up Questions

- "What's the recursion depth limit and how to avoid `RecursionError`?"
- "Recursive vs iterative trade-offs here?"
- "How would `yield from` help flatten lazily?"

---

## Practical Question 9

**Difficulty:** Easy–Medium
**Estimated Time:** 12 min
**Concepts Tested:** map/filter/reduce, lambdas, comprehensions, pure functions

**Problem Statement:** Given a list of transaction dicts, compute the **total amount** of all `"completed"` transactions above 100, functionally.

**Example Input:**
```python
[{"amount": 150, "status": "completed"},
 {"amount": 50,  "status": "completed"},
 {"amount": 300, "status": "pending"}]
```
**Example Output:** `150`

**Approach**
1. Filter to completed transactions with amount > 100.
2. Project each to its amount.
3. Sum. Express as a generator expression (Pythonic) and as `map`/`filter` (functional).

## Python Implementation

```python
def total_large_completed(txns: list[dict]) -> int:
    # Comprehension form — clearest, lazy, pure
    return sum(
        t["amount"]
        for t in txns
        if t["status"] == "completed" and t["amount"] > 100
    )
```

- Generator expression filters and projects in one pass; `sum` folds it.
- **Time:** O(n). **Space:** O(1) (lazy — nothing materialized).

## Alternative Solution

```python
from functools import reduce
total = reduce(
    lambda acc, t: acc + t["amount"],
    filter(lambda t: t["status"] == "completed" and t["amount"] > 100, txns),
    0,
)
```
Functionally equivalent but less readable — a good "show you know `map`/`filter`/`reduce`" answer, while noting the comprehension is more Pythonic.

## Interview Variations

- Group totals **by status** (use `defaultdict`).
- Return the **average**, handling empty input safely.
- Apply a **currency conversion** with `map` before summing.

## Common Follow-up Questions

- "Comprehension vs `map`/`filter` — which and why?"
- "What makes this a pure function?"
- "Why is the generator memory-efficient for a huge list?"

---

# Part 4 — File Handling

## Practical Question 10

**Difficulty:** Easy–Medium
**Estimated Time:** 15 min
**Concepts Tested:** file reading, generators, memory efficiency, context managers

**Problem Statement:** Count the frequency of each **word** in a potentially large text file, memory-efficiently.

**Example:** file contains `"the cat sat on the mat"` → `{"the": 2, "cat": 1, ...}`

**Approach**
1. Never load the whole file — iterate line by line (lazy).
2. Tokenize each line, normalize case, tally with `Counter`.
3. `with` guarantees the file closes.

## Python Implementation

```python
from collections import Counter

def word_frequencies(path: str) -> Counter:
    counts: Counter = Counter()
    with open(path, encoding="utf-8") as f:      # explicit encoding = portable
        for line in f:                            # lazy: one line at a time
            counts.update(line.lower().split())   # tokenize + tally
    return counts

# Usage:
# word_frequencies("book.txt").most_common(5)
```

- Iterating `f` streams the file → constant memory regardless of size.
- `Counter.update` tallies each line's words.
- **Time:** O(N) total words. **Space:** O(V) distinct words (vocabulary).

## Alternative Solution

For punctuation-aware tokenizing, use `re.findall(r"[a-z']+", line.lower())` instead of `split()`. For truly massive files across machines, a MapReduce/`multiprocessing` split-and-merge approach scales further.

## Interview Variations

- Return the **top-k** words (`most_common(k)`).
- Count **lines** or **characters** instead.
- Ignore a **stop-word** set.

## Common Follow-up Questions

- "Why iterate the file instead of `read()`/`readlines()`?" (Memory.)
- "Why pass `encoding='utf-8'`?" (Cross-platform determinism.)
- "How would you parallelize for a 100 GB file?"

---

## Practical Question 11

**Difficulty:** Medium
**Estimated Time:** 20 min
**Concepts Tested:** CSV, JSON, dicts, data transformation, `with`

**Problem Statement:** Read a CSV of employees, filter those in a given department, and write the result to a JSON file.

**Example:** `employees.csv` → `engineering.json` containing only Engineering staff.

**Approach**
1. `csv.DictReader` yields each row as a dict keyed by header.
2. Filter rows by department; coerce numeric fields.
3. `json.dump` writes the list with indentation.

## Python Implementation

```python
import csv
import json

def export_department(csv_path: str, json_path: str, dept: str) -> int:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)                    # row -> dict per header
        people = [
            {**row, "salary": int(row["salary"])}     # coerce salary to int
            for row in reader
            if row["department"] == dept
        ]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(people, f, indent=2)                # pretty-printed JSON
    return len(people)
```

- `DictReader` removes manual header/index handling.
- `newline=""` prevents blank-line issues on Windows.
- Two `with` blocks guarantee both files close.
- **Time:** O(n) rows. **Space:** O(m) filtered rows held before writing.

## Alternative Solution

For huge files, stream row-by-row and write JSON Lines (one JSON object per line) so you never hold all rows in memory. Use `csv.reader` (index-based) if there's no header.

## Interview Variations

- Aggregate: **average salary per department** (group with `defaultdict`).
- Handle **missing/malformed** rows gracefully (try/except per row).
- Convert JSON **back** to CSV.

## Common Follow-up Questions

- "`DictReader` vs `reader`?"
- "Why `newline=''` with csv?"
- "How to handle a column with commas/quotes?" (The csv module handles quoting.)

---

# Part 5 — Object-Oriented Programming

## Practical Question 12

**Difficulty:** Medium
**Estimated Time:** 20 min
**Concepts Tested:** classes, encapsulation, properties, exceptions, invariants

**Problem Statement:** Design a `BankAccount` class with deposit/withdraw that **enforces a non-negative balance** and validates inputs.

**Approach**
1. Keep balance in a protected attribute exposed via a **read-only property**.
2. Validate amounts in `deposit`/`withdraw`; raise domain errors for bad states.
3. Add `__repr__` for debuggability.

## Python Implementation

```python
class InsufficientFundsError(Exception):
    """Raised when a withdrawal exceeds the available balance."""

class BankAccount:
    def __init__(self, owner: str, balance: float = 0.0):
        self.owner = owner
        self._balance = 0.0
        if balance:
            self.deposit(balance)          # reuse validation for the initial deposit

    @property
    def balance(self) -> float:            # read-only: no setter exposed
        return self._balance

    def deposit(self, amount: float) -> float:
        if amount <= 0:
            raise ValueError("deposit amount must be positive")
        self._balance += amount
        return self._balance

    def withdraw(self, amount: float) -> float:
        if amount <= 0:
            raise ValueError("withdrawal amount must be positive")
        if amount > self._balance:
            raise InsufficientFundsError(
                f"tried to withdraw {amount}, balance is {self._balance}"
            )
        self._balance -= amount
        return self._balance

    def __repr__(self) -> str:
        return f"BankAccount(owner={self.owner!r}, balance={self._balance})"
```

- `balance` is read-only — the only way to change it is through validated methods (encapsulation).
- A **custom exception** lets callers catch overdrafts specifically.
- **Invariant** (`balance ≥ 0`) can never be violated through the public API.

## Alternative Solution

Model money with `decimal.Decimal` (avoids float rounding for currency), and/or make an **immutable** transaction log so history is auditable. For a data-only variant, a `@dataclass` with `__post_init__` validation.

## Interview Variations

- Add **transfer** between two accounts (atomic — both succeed or neither).
- Add **transaction history**.
- Add **overdraft limit** or **interest** subclass (inheritance).

## Common Follow-up Questions

- "Why a read-only property instead of a public attribute?"
- "Where would you use `Decimal` and why?"
- "How do you make `transfer` safe against partial failure?" (try/except + rollback.)

---

## Practical Question 13

**Difficulty:** Medium
**Estimated Time:** 20 min
**Concepts Tested:** inheritance, abstraction (ABC), polymorphism, `super()`

**Problem Statement:** Design a shape hierarchy where every shape must implement `area()` and `perimeter()`, then compute the total area of a mixed list of shapes.

**Approach**
1. An **abstract base class** `Shape` declares `area`/`perimeter` as abstract — enforcing the contract.
2. Concrete `Circle`, `Rectangle` implement them.
3. Polymorphism lets a single loop total any shapes.

## Python Implementation

```python
from abc import ABC, abstractmethod
import math

class Shape(ABC):
    @abstractmethod
    def area(self) -> float: ...

    @abstractmethod
    def perimeter(self) -> float: ...

    def describe(self) -> str:                     # shared concrete method
        return f"{type(self).__name__}: area={self.area():.2f}"

class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius
    def area(self) -> float:
        return math.pi * self.radius ** 2
    def perimeter(self) -> float:
        return 2 * math.pi * self.radius

class Rectangle(Shape):
    def __init__(self, w: float, h: float):
        self.w, self.h = w, h
    def area(self) -> float:
        return self.w * self.h
    def perimeter(self) -> float:
        return 2 * (self.w + self.h)

def total_area(shapes: list[Shape]) -> float:
    return sum(s.area() for s in shapes)           # polymorphic call

# Shape()  -> TypeError: can't instantiate abstract class
shapes = [Circle(2), Rectangle(3, 4)]
total_area(shapes)                                 # 12.566... + 12 = 24.566...
```

- The ABC guarantees every concrete shape implements the interface — a missing method makes the subclass non-instantiable.
- `total_area` doesn't care about concrete types — pure polymorphism.

## Alternative Solution

Use `typing.Protocol` for **structural** typing (no forced inheritance) if shapes come from third-party code you can't subclass. Or `@dataclass` shapes with methods for less boilerplate.

## Interview Variations

- Add a `Square` reusing `Rectangle` (inheritance vs composition discussion).
- Sort shapes by area (`__lt__` / `sorted(key=...)`).
- Add a `Triangle` and validate its sides.

## Common Follow-up Questions

- "What happens if a subclass forgets `perimeter()`?"
- "ABC vs Protocol vs duck typing — trade-offs?"
- "Where does `super()` fit if `Circle` extended another shape?"

---

## Practical Question 14

**Difficulty:** Easy–Medium
**Estimated Time:** 12 min
**Concepts Tested:** dunder methods, `__eq__`/`__hash__`, `__repr__`, dataclasses

**Problem Statement:** Create a `Point` value object that prints nicely, compares by value, and can be used in a `set`.

**Approach**
1. Value equality + hashability requires consistent `__eq__` and `__hash__`.
2. A **frozen dataclass** generates all of this correctly and immutably.
3. Show the manual version too, since interviewers ask for it.

## Python Implementation

```python
from dataclasses import dataclass

@dataclass(frozen=True)          # frozen -> immutable + auto __hash__
class Point:
    x: int
    y: int

p1, p2 = Point(1, 2), Point(1, 2)
p1 == p2          # True  (auto __eq__)
{p1, p2}          # {Point(x=1, y=2)}  -> deduped, since equal + hashable
print(p1)         # Point(x=1, y=2)  (auto __repr__)
```

Manual equivalent (what the dataclass generates):
```python
class Point:
    def __init__(self, x, y): self.x, self.y = x, y
    def __repr__(self):  return f"Point(x={self.x}, y={self.y})"
    def __eq__(self, o):  return isinstance(o, Point) and (self.x, self.y) == (o.x, o.y)
    def __hash__(self):   return hash((self.x, self.y))   # MUST match __eq__
```

- `frozen=True` gives immutability and a valid `__hash__` for free.
- Rule: **equal objects must have equal hashes** — the dataclass enforces this.

## Alternative Solution

`namedtuple("Point", "x y")` — immutable, hashable, iterable/unpackable, minimal memory. Choose it for pure positional records; choose a frozen dataclass when you want type hints/methods/defaults.

## Interview Variations

- Make points **sortable** (`order=True`).
- Add a `distance_to(other)` method.
- Support `p1 + p2` (`__add__`).

## Common Follow-up Questions

- "What breaks if you define `__eq__` but not `__hash__`?"
- "Why does the object become unhashable then?"
- "dataclass vs namedtuple vs plain class here?"

---

# Part 6 — Exception Handling & Logging

## Practical Question 15

**Difficulty:** Easy–Medium
**Estimated Time:** 15 min
**Concepts Tested:** try/except/else/finally, custom exceptions, validation, retry

**Problem Statement:** Write a `safe_divide` and a robust `parse_int` that validate input and handle errors cleanly; then implement a **retry** wrapper for a flaky operation.

**Approach**
1. Catch the **specific** exceptions you can handle; don't swallow everything.
2. Use `else` for the success path and `finally` for cleanup.
3. Retry: loop N times, catch, back off, re-raise on final failure.

## Python Implementation

```python
import logging, time

logger = logging.getLogger(__name__)

def parse_int(value: str, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (ValueError, TypeError):          # specific, expected failures
        logger.warning("could not parse %r as int", value)
        return default

def retry(operation, attempts: int = 3, delay: float = 0.5):
    """Call operation(); retry on failure up to `attempts` times."""
    for i in range(1, attempts + 1):
        try:
            return operation()
        except Exception as e:               # broad here is OK: we re-raise on last
            logger.warning("attempt %d/%d failed: %s", i, attempts, e)
            if i == attempts:
                raise                        # exhausted -> propagate the real error
            time.sleep(delay * i)            # simple linear backoff
```

- `parse_int` catches only `ValueError`/`TypeError` — real bugs still surface.
- `retry` logs each failure and **re-raises** the last one (never hides permanent errors).
- Lazy `%s` logging args avoid formatting cost when not logged.

## Alternative Solution

Use a decorator form of `retry` (`@retry(attempts=3)`) built on closures/`functools.wraps`, or a library like `tenacity` for production. For divide, prefer letting `ZeroDivisionError` propagate unless a sentinel is explicitly desired.

## Interview Variations

- **Exponential** backoff with jitter.
- Retry only on **specific** exception types (`retry_on=(ConnectionError,)`).
- Add a **timeout** budget across attempts.

## Common Follow-up Questions

- "Why not `except Exception: pass`?"
- "When is a broad `except` acceptable?" (Log-and-reraise, top-level boundaries.)
- "`error()` vs `exception()` for logging inside except?"

---

# Part 7 — Advanced Python

## Practical Question 16

**Difficulty:** Medium
**Estimated Time:** 15 min
**Concepts Tested:** generators, `yield`, lazy evaluation, memory efficiency

**Problem Statement:** Write a generator that yields fixed-size **batches** (chunks) from any iterable — useful for batching API calls or DB inserts.

**Example Input:** `chunk([1,2,3,4,5], 2)`
**Example Output:** `[1,2] [3,4] [5]`

**Approach**
1. Accumulate items into a buffer; `yield` when it reaches `size`.
2. `yield` the final partial batch after the loop.
3. Lazy — works on infinite/streamed iterables.

## Python Implementation

```python
from typing import Iterable, Iterator

def chunk(iterable: Iterable, size: int) -> Iterator[list]:
    if size < 1:
        raise ValueError("size must be >= 1")
    batch: list = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch                # emit a full batch, then reset
            batch = []
    if batch:                          # leftover partial batch
        yield batch
```

- Uses a rolling buffer — **O(size)** memory, independent of total length.
- Works on generators/files (lazy source), not just lists.
- **Time:** O(n). **Space:** O(size).

## Alternative Solution

`itertools.islice`-based version pulls slices from an iterator without a manual buffer:
```python
from itertools import islice
def chunk_islice(iterable, size):
    it = iter(iterable)
    while batch := list(islice(it, size)):
        yield batch
```
(Python 3.12+: `itertools.batched` does this natively.)

## Interview Variations

- **Sliding window** of size k (overlapping) instead of disjoint chunks.
- Chunk by **byte size** rather than count.
- Parallel-process each chunk.

## Common Follow-up Questions

- "Why a generator instead of returning a list of lists?" (Memory/laziness.)
- "How does this help batching DB inserts?"
- "What's `itertools.islice` doing?"

---

## Practical Question 17

**Difficulty:** Medium
**Estimated Time:** 15 min
**Concepts Tested:** decorators, closures, `functools.wraps`, cross-cutting concerns

**Problem Statement:** Write a `@timer` decorator that logs how long any function takes, preserving the function's identity, plus a parameterized `@retry(n)` decorator.

**Approach**
1. Wrapper accepts `*args, **kwargs` to work on any signature.
2. `functools.wraps` preserves `__name__`/`__doc__`.
3. Parameterized decorator = a function returning a decorator (three levels).

## Python Implementation

```python
import functools, time, logging

logger = logging.getLogger(__name__)

def timer(func):
    @functools.wraps(func)                       # preserve identity/metadata
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:                                 # log even if func raises
            elapsed = time.perf_counter() - start
            logger.info("%s took %.4fs", func.__name__, elapsed)
    return wrapper

def retry(times: int):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if attempt == times:
                        raise
        return wrapper
    return decorator

@timer
@retry(3)                                        # applies bottom-up: retry, then timer
def flaky_fetch(url): ...
```

- `@functools.wraps` keeps `flaky_fetch.__name__` correct (not `"wrapper"`).
- Decorator **order** matters: `@retry(3)` wraps first, `@timer` wraps that.
- `finally` ensures timing logs even on exceptions.

## Alternative Solution

Class-based decorator using `__call__` (carries state as instance attributes) — useful for decorators that accumulate metrics across calls. Or use `contextlib`/`time.perf_counter` inline for one-offs.

## Interview Variations

- A `@cache`/memoize decorator (dict keyed by args).
- A `@rate_limit(calls, period)` decorator.
- Preserve and expose call **count** on the wrapped function.

## Common Follow-up Questions

- "What does `@timer` desugar to?" (`fn = timer(fn)`.)
- "Why `functools.wraps`?"
- "Why does the wrapper use `*args, **kwargs`?"
- "Explain the three nesting levels of a parameterized decorator."

---

## Practical Question 18

**Difficulty:** Medium
**Estimated Time:** 15 min
**Concepts Tested:** context managers, `__enter__`/`__exit__`, `contextlib`, cleanup

**Problem Statement:** Build a context manager that **times a code block** and another that **temporarily changes the working directory** and restores it, guaranteeing restoration even on error.

**Approach**
1. Class-based: implement `__enter__` (setup) and `__exit__` (teardown).
2. Generator-based with `@contextmanager`: setup before `yield`, teardown in `finally`.
3. `__exit__`/`finally` guarantee cleanup on exceptions.

## Python Implementation

```python
import os, time
from contextlib import contextmanager

# Class-based: a timing block
class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self                              # bound to `as t`
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start
        print(f"Block took {self.elapsed:.4f}s")
        return False                             # don't suppress exceptions

# Generator-based: temporary chdir with guaranteed restore
@contextmanager
def working_directory(path: str):
    original = os.getcwd()
    os.chdir(path)                               # setup
    try:
        yield path                               # body runs here
    finally:
        os.chdir(original)                       # ALWAYS restore, even on error

with Timer() as t:
    do_work()

with working_directory("/tmp"):
    ...                                          # cwd is /tmp here; restored after
```

- `__exit__` runs even if the block raises; returning `False` lets the exception propagate.
- The generator's `finally` guarantees the directory is restored no matter what.

## Alternative Solution

`contextlib.suppress(FileNotFoundError)` for ignoring specific errors, `contextlib.ExitStack` to manage a dynamic number of context managers, or `tempfile.TemporaryDirectory()` for auto-cleaned temp dirs.

## Interview Variations

- A **database transaction** CM (commit/rollback — see theory §29).
- A CM that **suppresses** a specific exception (return truthy from `__exit__`).
- A reentrant **lock** context manager.

## Common Follow-up Questions

- "What does returning `True` from `__exit__` do?"
- "Why put teardown in `finally` in the generator version?"
- "How does `with` guarantee cleanup vs a plain `try/finally`?"

---

# Part 8 — Data-Structure Implementations

> The syllabus's "Coding Practice → Advanced" and DSA overlap expects you to implement core structures from scratch, with complexity and follow-ups. Each below is production-quality and interview-ready.

## Practical Question 19 — Stack & Queue

**Difficulty:** Easy
**Concepts Tested:** lists, `deque`, LIFO/FIFO, amortized complexity

**Problem Statement:** Implement a **Stack** (LIFO) and a **Queue** (FIFO) with O(1) operations.

```python
from collections import deque

class Stack:
    """LIFO. Python list is a natural stack (append/pop at the end)."""
    def __init__(self):
        self._items: list = []
    def push(self, x):  self._items.append(x)      # O(1) amortized
    def pop(self):
        if not self._items:
            raise IndexError("pop from empty stack")
        return self._items.pop()                    # O(1)
    def peek(self):     return self._items[-1]
    def is_empty(self): return not self._items
    def __len__(self):  return len(self._items)

class Queue:
    """FIFO. Use deque — list.pop(0) is O(n)!"""
    def __init__(self):
        self._items: deque = deque()
    def enqueue(self, x): self._items.append(x)     # O(1)
    def dequeue(self):
        if not self._items:
            raise IndexError("dequeue from empty queue")
        return self._items.popleft()                # O(1) — the key point
    def __len__(self): return len(self._items)
```

- **Stack:** a list is ideal — `append`/`pop` at the end are O(1) amortized.
- **Queue:** use `collections.deque`; a list's `pop(0)` is **O(n)** (shifts all elements). This distinction is a classic interview "gotcha".
- **Complexity:** all operations O(1). **Space:** O(n).

**Follow-ups:** "Why deque over list for a queue?" · "Implement a queue using two stacks." · "Design a stack with `get_min()` in O(1)" (auxiliary min-stack).

---

## Practical Question 20 — Singly Linked List

**Difficulty:** Medium
**Concepts Tested:** nodes, pointers, traversal, reversal

**Problem Statement:** Implement a singly linked list with `append`, `prepend`, `delete(value)`, and `reverse()`.

```python
class Node:
    __slots__ = ("value", "next")            # memory optimization
    def __init__(self, value, nxt=None):
        self.value = value
        self.next = nxt

class LinkedList:
    def __init__(self):
        self.head: Node | None = None

    def append(self, value):                 # O(n) — walk to the tail
        node = Node(value)
        if not self.head:
            self.head = node
            return
        cur = self.head
        while cur.next:
            cur = cur.next
        cur.next = node

    def prepend(self, value):                # O(1) — new head
        self.head = Node(value, self.head)

    def delete(self, value):                 # O(n) — relink around the node
        cur, prev = self.head, None
        while cur:
            if cur.value == value:
                if prev:
                    prev.next = cur.next
                else:
                    self.head = cur.next
                return True
            prev, cur = cur, cur.next
        return False

    def reverse(self):                       # O(n) time, O(1) space
        prev, cur = None, self.head
        while cur:
            cur.next, prev, cur = prev, cur, cur.next   # flip the pointer
        self.head = prev

    def to_list(self):
        out, cur = [], self.head
        while cur:
            out.append(cur.value); cur = cur.next
        return out
```

- `reverse()` is the star: three-pointer pointer-flipping in place, **O(n)/O(1)**.
- `prepend` is O(1); `append`/`delete` are O(n) (no tail pointer).
- `__slots__` cuts per-node memory by avoiding a `__dict__`.

**Follow-ups:** "Reverse recursively." · "Detect a cycle (Floyd's tortoise & hare)." · "Find the middle node in one pass." · "Why a linked list over an array here?" (O(1) insert/delete at known position, no shifting.)

---

## Practical Question 21 — Binary Search Tree

**Difficulty:** Medium–Hard
**Concepts Tested:** trees, recursion, BST property, traversals

**Problem Statement:** Implement a BST supporting `insert`, `search`, and **in-order traversal** (which yields sorted order).

```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    def insert(self, value):
        self.root = self._insert(self.root, value)

    def _insert(self, node, value):
        if node is None:
            return TreeNode(value)              # found the spot
        if value < node.value:
            node.left = self._insert(node.left, value)
        elif value > node.value:
            node.right = self._insert(node.right, value)
        # equal -> ignore duplicates
        return node

    def search(self, value) -> bool:
        node = self.root
        while node:                             # iterative: O(h)
            if value == node.value:
                return True
            node = node.left if value < node.value else node.right
        return False

    def inorder(self) -> list:                  # left, root, right -> sorted
        result = []
        def walk(node):
            if node:
                walk(node.left)
                result.append(node.value)
                walk(node.right)
        walk(self.root)
        return result
```

- BST property: left subtree < node < right subtree.
- **insert/search:** O(h) where h = height — **O(log n)** if balanced, **O(n)** if degenerate (sorted input → a linked list).
- **In-order traversal** yields elements in sorted order — a defining BST property.

**Follow-ups:** "Why can a BST degrade to O(n)?" (Unbalanced.) · "How do self-balancing trees (AVL/Red-Black) fix it?" · "Delete a node with two children." · "Validate whether a tree is a valid BST." · "BFS vs DFS traversal."

---

## Practical Question 22 — Graph with BFS & DFS

**Difficulty:** Medium–Hard
**Concepts Tested:** graphs, adjacency list, BFS (queue), DFS (stack/recursion), visited set

**Problem Statement:** Represent a graph as an adjacency list and implement **BFS** and **DFS** traversals.

```python
from collections import deque, defaultdict

class Graph:
    def __init__(self):
        self.adj: dict[int, list[int]] = defaultdict(list)

    def add_edge(self, u, v, directed=False):
        self.adj[u].append(v)
        if not directed:
            self.adj[v].append(u)

    def bfs(self, start) -> list:               # level-by-level; uses a QUEUE
        visited = {start}
        order, q = [], deque([start])
        while q:
            node = q.popleft()                  # FIFO
            order.append(node)
            for nbr in self.adj[node]:
                if nbr not in visited:
                    visited.add(nbr)            # mark on enqueue (avoids dupes)
                    q.append(nbr)
        return order

    def dfs(self, start) -> list:               # deep-first; recursion = implicit stack
        visited, order = set(), []
        def visit(node):
            visited.add(node)
            order.append(node)
            for nbr in self.adj[node]:
                if nbr not in visited:
                    visit(nbr)
        visit(start)
        return order
```

- **BFS** uses a **queue** (`deque`) → explores neighbors level by level; finds shortest path in unweighted graphs.
- **DFS** uses recursion (or an explicit **stack**) → goes deep first.
- The **`visited` set** (O(1) membership) prevents infinite loops on cycles.
- **Complexity (both):** O(V + E) time, O(V) space.

**Follow-ups:** "BFS vs DFS — when to use each?" (BFS: shortest path/levels; DFS: cycle detection, topological sort, connectivity.) · "Convert recursive DFS to iterative." · "Detect a cycle." · "Why an adjacency list over a matrix?" (Sparse graphs: O(V+E) space vs O(V²).)

---

## Practical Question 23 — Hash Table (from scratch)

**Difficulty:** Hard
**Concepts Tested:** hashing, collisions, chaining, load factor

**Problem Statement:** Implement a simple hash table with `put`/`get`/`delete` using **separate chaining**.

```python
class HashTable:
    def __init__(self, capacity: int = 8):
        self.capacity = capacity
        self.size = 0
        self.buckets: list[list] = [[] for _ in range(capacity)]  # chaining

    def _index(self, key) -> int:
        return hash(key) % self.capacity        # map hash -> bucket

    def put(self, key, value):
        bucket = self.buckets[self._index(key)]
        for i, (k, _) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)         # update existing
                return
        bucket.append((key, value))              # new key
        self.size += 1
        if self.size / self.capacity > 0.7:      # load factor threshold
            self._resize()

    def get(self, key):
        for k, v in self.buckets[self._index(key)]:
            if k == key:
                return v
        raise KeyError(key)

    def delete(self, key):
        bucket = self.buckets[self._index(key)]
        for i, (k, _) in enumerate(bucket):
            if k == key:
                del bucket[i]; self.size -= 1; return
        raise KeyError(key)

    def _resize(self):                           # keep operations ~O(1)
        old = [pair for b in self.buckets for pair in b]
        self.capacity *= 2
        self.buckets = [[] for _ in range(self.capacity)]
        self.size = 0
        for k, v in old:
            self.put(k, v)
```

- `hash(key) % capacity` picks a bucket; collisions live in a per-bucket list (**chaining**).
- **Resizing** when the **load factor** exceeds ~0.7 keeps chains short, preserving **O(1) average**.
- **Complexity:** O(1) average, O(n) worst case (all keys collide).

**Follow-ups:** "Chaining vs open addressing?" · "Why resize on load factor?" · "What makes a good hash function?" · "Why must keys be hashable/immutable?" (This is exactly how Python's `dict` works conceptually.)

---

## Practical Question 24 — Searching & Sorting

**Difficulty:** Easy–Medium
**Concepts Tested:** binary search, merge sort, quicksort, complexity

**Problem Statement:** Implement **binary search** and **merge sort**; explain complexity.

```python
def binary_search(arr: list[int], target: int) -> int:
    """Requires a SORTED array. Returns index or -1."""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2                     # avoids overflow in other langs
        if arr[mid] == target:
            return mid
        if arr[mid] < target:
            lo = mid + 1                         # search right half
        else:
            hi = mid - 1                         # search left half
    return -1

def merge_sort(arr: list[int]) -> list[int]:
    if len(arr) <= 1:                            # base case
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])                 # divide
    right = merge_sort(arr[mid:])
    return _merge(left, right)                   # conquer/combine

def _merge(a: list[int], b: list[int]) -> list[int]:
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:                         # <= keeps it STABLE
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    out.extend(a[i:]); out.extend(b[j:])
    return out
```

- **Binary search:** halves the search space each step → **O(log n)**; requires sorted input.
- **Merge sort:** divide-and-conquer, **O(n log n)** guaranteed, **stable**, but **O(n)** extra space.
- Python's built-in `sorted`/`list.sort` use **Timsort** (adaptive, stable, O(n log n)).

**Follow-ups:** "Merge sort vs quicksort trade-offs?" (Quicksort: in-place O(log n) space but O(n²) worst case; merge sort: stable, guaranteed O(n log n), O(n) space.) · "Why must binary search input be sorted?" · "Find the first/last occurrence of a duplicate (binary search variants)."

---

# Part 9 — Coding Question Bank

> Rapid-fire predicted questions grouped by difficulty, with the **interviewer's intent** (why they ask) and a compact solution/hint. Practice narrating approach + complexity for each.

## Easy

**E1. Reverse a string / integer.**
*Why asked:* tests immutability understanding and slicing. `s[::-1]`; for an int, handle sign and use `//`/`%` or `int(str(n)[::-1])`.

**E2. FizzBuzz.**
*Why asked:* the universal "can they code at all" filter; watch order of conditions (check `% 15` first, or build a string).
```python
for n in range(1, 101):
    print("FizzBuzz" if n%15==0 else "Fizz" if n%3==0 else "Buzz" if n%5==0 else n)
```

**E3. Count vowels / character frequency.** *Why asked:* dict/`Counter` fluency. `Counter(c for c in s if c in "aeiou")`.

**E4. Find max/min without built-ins.** *Why asked:* loop + comparison basics; discuss empty-input edge case.

**E5. Check if two strings are anagrams.** *Why asked:* hashing vs sorting trade-off. `Counter(a) == Counter(b)`.

**E6. Sum of digits / factorial (iterative & recursive).** *Why asked:* recursion base-case reasoning.

**E7. Swap two variables without a temp.** *Why asked:* tuple unpacking. `a, b = b, a`.

## Medium

**M1. Two Sum / Three Sum.** *Why asked:* hash-map optimization from O(n²)→O(n). (See Q4.)

**M2. Valid parentheses.** *Why asked:* stack application.
```python
def valid(s):
    pairs, stack = {")":"(","]":"[","}":"{"}, []
    for c in s:
        if c in "([{": stack.append(c)
        elif not stack or stack.pop() != pairs[c]: return False
    return not stack
```

**M3. Group anagrams.** *Why asked:* canonical-key hashing. (See Q3.)

**M4. Merge two sorted lists.** *Why asked:* two-pointer merge (core of merge sort). (See Q24 `_merge`.)

**M5. Move zeros to end / in-place partition.** *Why asked:* two-pointer, in-place mutation.

**M6. Find duplicates / first missing positive.** *Why asked:* set vs sort vs index-marking.

**M7. Implement `@cache`/memoize decorator.** *Why asked:* closures + decorators + dict. (See Q17.)

**M8. Flatten a nested list.** *Why asked:* recursion / `yield from`. (See Q8.)

**M9. LRU cache.** *Why asked:* data-structure composition. (See Q5.)

**M10. Word frequency from a file.** *Why asked:* I/O + generators + Counter. (See Q10.)

## Hard

**H1. Longest substring without repeating characters.** *Why asked:* sliding-window + set.
```python
def longest_unique(s):
    seen, left, best = {}, 0, 0
    for right, c in enumerate(s):
        if c in seen and seen[c] >= left:
            left = seen[c] + 1
        seen[c] = right
        best = max(best, right - left + 1)
    return best
```
*Complexity:* O(n) time, O(k) space.

**H2. Merge k sorted lists.** *Why asked:* heap usage (`heapq`), O(N log k).

**H3. Detect a cycle in a linked list (Floyd's).** *Why asked:* two-pointer trick, O(1) space.

**H4. Serialize/deserialize a binary tree.** *Why asked:* traversal + reconstruction.

**H5. Implement a hash table from scratch.** *Why asked:* deep understanding of dicts. (See Q23.)

**H6. Producer/consumer with threads or asyncio.** *Why asked:* concurrency + synchronization; discuss GIL, `queue.Queue`, locks.

**H7. Top-k frequent elements.** *Why asked:* `Counter.most_common` vs heap; O(n log k).

---

# Part 10 — Notebook-Style Workflow

> Some assessments (especially data-processing tasks) run in a **Jupyter Notebook**. Even for pure-Python this pattern shows structured thinking: split the task into labeled cells with explanations. Below is a complete, runnable **standard-library-only** data-processing workflow (no ML libraries needed — matches this syllabus) analyzing a CSV of sales records.

### Cell 1 — Imports

```python
# All standard library — no external dependencies.
import csv
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
```
*Explanation:* Group all imports in the first cell so dependencies are visible up front. This workflow needs `csv`/`json` for I/O, `collections` for aggregation, `statistics` for summary metrics, and `datetime` for parsing dates.

### Cell 2 — Create / Load the dataset

```python
# In a real exam you'd read a provided file. Here we generate a small sample
# so the notebook is self-contained and reproducible.
sample = """date,product,category,quantity,price
2026-01-05,Widget,Hardware,3,19.99
2026-01-05,Gadget,Hardware,1,49.99
2026-01-06,eBook,Digital,5,9.99
2026-01-06,Widget,Hardware,2,19.99
2026-01-07,Course,Digital,1,199.00
"""
with open("sales.csv", "w", encoding="utf-8") as f:
    f.write(sample)

with open("sales.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))          # each row -> dict keyed by header
rows[:2]
```
*Explanation:* `csv.DictReader` turns each line into a dict, so we reference fields by name (`row["price"]`) rather than fragile indices. We materialize to a list here because the dataset is small; for large files we'd stream.

### Cell 3 — Explore the data

```python
print(f"Rows: {len(rows)}")
print(f"Columns: {list(rows[0].keys())}")
print(f"Categories: {set(r['category'] for r in rows)}")
print(f"Date range: {min(r['date'] for r in rows)} to {max(r['date'] for r in rows)}")
```
*Explanation:* First understand shape and content — row count, columns, distinct categories, and the date span. This is the "sanity check" step before any transformation, catching surprises early.

### Cell 4 — Data cleaning & type coercion

```python
def clean(row: dict) -> dict:
    return {
        "date": datetime.strptime(row["date"], "%Y-%m-%d").date(),
        "product": row["product"].strip(),
        "category": row["category"].strip(),
        "quantity": int(row["quantity"]),      # CSV gives strings — coerce
        "price": float(row["price"]),
    }

clean_rows = [clean(r) for r in rows]
clean_rows[0]
```
*Explanation:* CSV values are **all strings**. We coerce `quantity`→int, `price`→float, and parse `date` into a real `date` object, and strip whitespace. Doing this once, up front, prevents type bugs downstream. In production you'd wrap `clean` in try/except to quarantine malformed rows.

### Cell 5 — Feature engineering

```python
for r in clean_rows:
    r["revenue"] = r["quantity"] * r["price"]   # derived field
    r["weekday"] = r["date"].strftime("%A")
clean_rows[0]
```
*Explanation:* Derive new fields useful for analysis — `revenue` (quantity × price) and `weekday`. Feature engineering is creating informative columns from raw ones; here it enables revenue aggregation and day-of-week patterns.

### Cell 6 — Aggregation / "model" (compute the metrics)

```python
revenue_by_category = defaultdict(float)
units_by_product = Counter()

for r in clean_rows:
    revenue_by_category[r["category"]] += r["revenue"]
    units_by_product[r["product"]] += r["quantity"]

revenue_by_category = dict(revenue_by_category)
revenue_by_category, units_by_product.most_common(3)
```
*Explanation:* The analytical core. `defaultdict(float)` sums revenue per category without key-existence checks; `Counter` tallies units per product and gives `most_common`. These grouping idioms are the "training" step of a data task — turning rows into insights.

### Cell 7 — Evaluation / summary statistics

```python
revenues = [r["revenue"] for r in clean_rows]
summary = {
    "total_revenue": round(sum(revenues), 2),
    "mean_order_value": round(statistics.mean(revenues), 2),
    "median_order_value": round(statistics.median(revenues), 2),
    "max_order": round(max(revenues), 2),
    "num_orders": len(revenues),
}
summary
```
*Explanation:* Summarize with descriptive statistics — total, mean, median, max. Reporting both mean and median reveals skew (a few large orders pull the mean above the median). `statistics` covers this without NumPy.

### Cell 8 — Visualization (text-based, no libraries)

```python
def bar_chart(data: dict, width: int = 40):
    peak = max(data.values())
    for label, value in sorted(data.items(), key=lambda kv: -kv[1]):
        bar = "█" * int(value / peak * width)
        print(f"{label:<10} | {bar} {value:.2f}")

bar_chart(revenue_by_category)
```
*Explanation:* Even without matplotlib you can visualize proportions with a text bar chart — scaling each bar to the maximum value. In a notebook with libraries you'd swap this for `matplotlib`/`seaborn`, but the principle (compare magnitudes visually) is the same.

### Cell 9 — Interpretation & export

```python
# Interpretation (write findings as markdown/comments in the real notebook):
# - Hardware drives most revenue by volume; Digital has higher per-order value
#   (e.g., the Course), pulling the mean order value above the median.
# - Widget is the top unit-seller; consider bundling with Gadget.

# Persist results for downstream use / grading:
with open("summary.json", "w", encoding="utf-8") as f:
    json.dump({"summary": summary, "revenue_by_category": revenue_by_category},
              f, indent=2, default=str)
print("Saved summary.json")
```
*Explanation:* Close the loop: **interpret** the numbers in plain language (the point of the analysis, not just the code) and **export** results to JSON for reuse or grading. `default=str` safely serializes any non-JSON-native types. A strong submission always ends with "what does this mean?", not just a dump of figures.

---

## How to Approach Any Coding Assessment (checklist)

1. **Restate** the problem and confirm inputs/outputs and constraints.
2. **Clarify edge cases:** empty input, single element, duplicates, negatives, huge input, invalid types.
3. **State a brute-force** solution and its complexity first (shows you can start).
4. **Optimize:** identify the bottleneck; reach for the right structure (hash map for lookups, set for membership, deque for queues, heap for top-k).
5. **State target complexity** before coding.
6. **Write clean code:** meaningful names, small functions, type hints, no premature cleverness.
7. **Test out loud:** walk through the example, then an edge case.
8. **Discuss trade-offs & follow-ups:** time/space, alternative approaches, how it scales.

*Pair this with `theory.md` for the conceptual depth behind each of these problems.*

# Data Structures & Algorithms — Theory & Interview Preparation Guide

> This document teaches every topic in the Data Structures syllabus from first
> principles. It is written to prepare you for **technical interviews, viva,
> written assessments, coding rounds, and practical lab exams**. Each topic
> follows the same structure so you can revise predictably. Read it slowly —
> depth beats memorization.

## How to use this guide

- Read **"What is it?"** and **"Why is it needed?"** first to build intuition.
- Study **"How does it work?"** and **"Internal Working"** to be able to *implement* from scratch.
- Rehearse the **Interview Questions** out loud. Cover the model answer, answer yourself, then compare.
- Skim **Common Mistakes** the night before an exam — these are the cheap marks people lose.

## Table of Contents

**Week 1 — Fundamentals**
1. [Introduction to Data Structures & Algorithms](#1-introduction-to-data-structures--algorithms)
2. [Memory Management](#2-memory-management)
3. [Complexity Analysis](#3-complexity-analysis)
4. [Arrays](#4-arrays)
5. [Linked Lists](#5-linked-lists)
6. [Strings](#6-strings)
7. [Searching Algorithms](#7-searching-algorithms)
8. [Recursion](#8-recursion)

**Week 2 — Intermediate**
9. [Sorting Algorithms](#9-sorting-algorithms)
10. [Hash Tables](#10-hash-tables)
11. [Stacks](#11-stacks)
12. [Queues](#12-queues)
13. [Trees & Binary Search Trees](#13-trees--binary-search-trees)
14. [Graphs (BFS & DFS)](#14-graphs-bfs--dfs)

---

# 1. Introduction to Data Structures & Algorithms

## What is it?

A **data structure** is a way of organizing and storing data in a computer so that it can be used efficiently. An **algorithm** is a finite, well-defined sequence of steps that transforms an input into a desired output.

Think of a data structure as *how you arrange your tools in a toolbox* and an algorithm as *the procedure you follow to build something with those tools*. The two are inseparable: the way you store data determines which algorithms are cheap and which are expensive. For example, finding a name in a sorted phone book is fast because the data is organized (sorted); finding a name in a random pile of papers is slow because it is not.

Formally, a data structure is a triple: a **collection of values**, the **relationships** among those values, and the **operations** that can be applied to them. A stack, for example, is a collection of items where the relationship is "last in, first out" and the operations are `push` and `pop`.

## Why is it needed?

Computers have finite memory and finite time. Two programs can produce the *same correct answer*, but one may run in a fraction of a second while the other takes hours or crashes from running out of memory. The difference is almost always the choice of data structure and algorithm.

Real motivations:

- **Scale.** A naive approach that works on 100 records may be unusable on 100 million. Google searches billions of pages in milliseconds only because of carefully chosen data structures (inverted indexes, tries, hash maps).
- **Cost.** Efficient code uses fewer servers, less electricity, and cheaper hardware.
- **Responsiveness.** Interactive systems (games, trading platforms, autocomplete) must respond within milliseconds.
- **Correctness under constraints.** Some problems are only *solvable at all* with the right structure (e.g., shortest-path routing needs graphs).

## How does it work?

You pick a data structure by asking: *what operations does my problem perform most often, and how large is the data?* Then you choose the structure whose common operations are cheap.

```
Problem  ──▶  Identify dominant operations (search? insert? sorted order?)
          ──▶  Estimate data size (N)
          ──▶  Choose data structure with cheap dominant operations
          ──▶  Choose/design algorithm over that structure
          ──▶  Analyze complexity, verify it meets constraints
```

Example decision:
- Need fast lookup by key → **Hash Table** (O(1) average).
- Need sorted order + fast lookup → **Balanced BST** (O(log n)).
- Need fast insert/delete at ends only → **Linked List / Deque**.
- Need index-based random access → **Array**.

## Internal Working

Under the hood, every data structure ultimately maps to **contiguous or linked blocks of memory** and CPU instructions that manipulate them. Arrays map directly to a contiguous block; linked structures use pointers (memory addresses) that connect scattered blocks. Algorithms are compiled into loops, comparisons, and arithmetic that the CPU executes. The "efficiency" of a structure is really about **how many memory accesses and CPU operations** a task requires and **how cache-friendly** those accesses are.

## Advantages

- Correct choice → dramatic speed and memory improvements.
- Makes previously infeasible problems feasible.
- Provides a shared vocabulary for engineers ("just use a heap here").

## Limitations

- No single structure is best for everything — every choice is a trade-off (fast search often costs slower insert, and vice versa).
- Premature optimization can add complexity for little gain; sometimes a simple list is fine.

## Real-world Applications

- **Databases** use B-trees and hash indexes for fast lookups.
- **Operating systems** use queues for scheduling and stacks for function calls.
- **Networking** uses graphs for routing (Dijkstra, BGP).
- **Compilers** use trees (AST) and stacks for parsing.
- **Search engines** use tries, inverted indexes, and hash maps.

## Interview Questions

**Beginner**
- What is the difference between a data structure and an algorithm?
- Give two everyday examples of data structures.

**Intermediate**
- How do you decide which data structure to use for a given problem?
- What does it mean for an algorithm to be "correct"?

**Advanced**
- Explain a situation where a theoretically slower algorithm is faster in practice.

**Scenario-based**
- You must build an autocomplete feature for a search box. Which data structure would you choose and why?

**"Why" questions**
- Why does the choice of data structure affect the algorithm's complexity?

**Comparison**
- Compare "abstract data type" vs "data structure".

## Model Answers

**Q: What is the difference between a data structure and an algorithm?**
A data structure is *how data is organized and stored* together with the operations allowed on it; an algorithm is *the step-by-step procedure* that operates on that data to solve a problem. They are complementary: the data structure defines the cost of each operation, and the algorithm defines the sequence of operations. For example, binary search is an *algorithm*; it only works efficiently when the data lives in a *sorted array* (a data structure). Change the structure to an unsorted linked list and the same algorithm no longer applies.

**Q: How do you decide which data structure to use?**
I start by identifying the operations the program performs most frequently and the size of the data. If lookups by key dominate, I lean toward a hash table for O(1) average access. If I also need ordering or range queries, I choose a balanced tree for O(log n). If insertions/deletions happen mostly at the ends, a linked list or deque is appropriate. I also consider memory overhead, cache behavior, and whether the data is static or changing. The decision is a trade-off analysis, not a lookup of a "best" structure.

**Q: Explain a situation where a theoretically slower algorithm is faster in practice.**
Insertion sort is O(n²) and quicksort is O(n log n), yet for very small arrays (say fewer than ~16 elements) insertion sort is often faster because it has low constant factors, no recursion overhead, and excellent cache locality. This is why real library sorts (like Timsort) switch to insertion sort for small subarrays. Big-O hides constant factors and hardware effects such as caching and branch prediction, which dominate at small sizes.

**Q (Scenario): Autocomplete data structure?**
I would use a **trie (prefix tree)**. Each node represents a character, and paths from the root spell out prefixes, so retrieving all words with a given prefix is proportional to the length of the prefix plus the number of results — independent of the total dictionary size. A hash map of full words cannot answer prefix queries efficiently. In production I might augment the trie with frequency scores at nodes to rank suggestions.

## Common Mistakes

- Confusing an **abstract data type** (the interface, e.g., "a stack") with its **implementation** (array-based vs linked-list-based).
- Jumping to code before analyzing which operations dominate.
- Assuming "faster Big-O = always faster" — constants and data size matter.
- Ignoring memory/space cost and focusing only on time.

## Related Concepts

- [Complexity Analysis](#3-complexity-analysis) — the language for comparing choices.
- Abstract Data Types (ADTs).
- Trade-offs between time and space.

---

# 2. Memory Management

## What is it?

**Memory management** is how a program obtains, uses, and releases memory during its lifetime. When a program runs, the operating system gives it a chunk of memory divided into regions — most importantly the **stack** and the **heap**. Memory management is the set of rules and mechanisms that decide *where* a piece of data lives, *how long* it lives, and *who is responsible* for freeing it.

## Why is it needed?

Memory is a finite, shared resource. If a program keeps requesting memory and never releases it, it eventually exhausts what is available and crashes — or starves other programs. If it releases memory too early and then uses it, the program corrupts data or crashes. Good memory management keeps programs **fast** (allocation is a cost), **correct** (no use-after-free), and **stable over time** (no leaks that grow until the process dies). This is critical for long-running systems like servers that must run for months.

## How does it work?

A running process's address space is typically laid out like this:

```
 High addresses
 ┌──────────────────────┐
 │        Stack         │  grows downward ↓ (function calls, locals)
 │          │           │
 │          ▼           │
 │                      │
 │          ▲           │
 │          │           │
 │        Heap          │  grows upward ↑ (dynamic allocation)
 ├──────────────────────┤
 │   BSS / Data segment │  global & static variables
 ├──────────────────────┤
 │    Text / Code       │  the program instructions
 └──────────────────────┘
 Low addresses
```

- **Stack:** Automatically managed. Each function call pushes a *stack frame* holding its local variables and return address; when the function returns, the frame is popped and its memory is instantly reclaimed. Fast (just move a pointer) but limited in size — deep recursion causes a **stack overflow**.
- **Heap:** Manually or automatically managed dynamic memory. You request a block (`malloc` in C, `new` in Java/C++, object creation in Python), use it, and it must eventually be freed. Flexible and large, but slower and prone to leaks/fragmentation.

**Stack allocation example (conceptual):**
```
def f():
    x = 10      # x lives on the stack frame of f
    return x    # frame destroyed on return; x gone automatically
```

**Heap allocation example:**
```
data = [0] * 1_000_000   # the list's backing storage lives on the heap
```

## Internal Working

- **Manual management (C/C++):** `malloc`/`free`, `new`/`delete`. The programmer is fully responsible. Forgetting to free → **leak**; freeing twice → **double free**; using after free → **dangling pointer** bug.
- **Automatic management (Python, Java, Go, C#):** A **garbage collector (GC)** reclaims memory that is no longer reachable. Python primarily uses **reference counting**: every object holds a count of how many references point to it; when the count drops to zero, the object is freed immediately. Python adds a **cyclic garbage collector** to catch reference *cycles* (objects that reference each other but are unreachable from the program), which pure reference counting cannot free.
- **Allocators** maintain free lists / size classes to serve requests quickly and reduce **fragmentation** (unusable gaps between allocated blocks).

**What is a memory leak?** A memory leak occurs when a program allocates memory but loses all references to it *without freeing it*, or keeps references it no longer needs. The memory is neither usable nor reclaimable, so the process's memory footprint grows over time. In managed languages leaks happen not from forgetting `free`, but from **unintentionally keeping objects reachable** — e.g., a global cache/list that grows forever, or an event listener never removed.

```
Leak pattern (managed language):
cache = {}
def handle(request):
    cache[request.id] = request   # never removed → grows unbounded → leak
```

## Advantages

- **Stack:** extremely fast, automatic cleanup, great cache locality.
- **Heap:** flexible size and lifetime; data can outlive the function that created it.
- **Automatic GC:** removes a whole class of bugs (use-after-free, double free) and speeds development.

## Limitations

- **Stack:** limited size; cannot hold data that must outlive its function; deep recursion overflows.
- **Heap:** slower allocation, fragmentation, and (without GC) leak-prone.
- **GC:** introduces pauses and CPU overhead; reference counting cannot handle cycles alone.

## Real-world Applications

- **Servers/databases** run for months, so even tiny leaks are fatal — memory profiling is routine.
- **Embedded/real-time systems** often avoid the heap entirely for predictability.
- **Game engines** use custom allocators (memory pools) to avoid GC pauses during frames.
- **Browsers** aggressively manage memory to keep many tabs alive.

## Interview Questions

**Beginner**
- What is the difference between stack and heap memory?
- What is a memory leak?

**Intermediate**
- How does Python free memory that is no longer used?
- Why can deep recursion cause a crash?

**Advanced**
- Reference counting cannot reclaim cyclic references. Explain why, and how Python handles it.
- What is memory fragmentation and why does it matter?

**Scenario-based**
- A long-running web service slowly consumes more RAM until it is restarted daily. How would you diagnose and fix it?

**"Why" questions**
- Why is stack allocation faster than heap allocation?

**Comparison**
- Compare manual memory management (C) vs garbage collection (Python/Java).

## Model Answers

**Q: Stack vs heap?**
The stack stores function call frames — local variables and return addresses — and is managed automatically in Last-In-First-Out order: when a function returns, its frame is popped and memory reclaimed instantly. Allocation is just moving the stack pointer, so it is very fast, but the stack is small and its data cannot outlive the function. The heap is a large region for dynamically allocated memory whose lifetime you control; data on the heap can live as long as needed and be shared across functions, but allocation is slower and must be freed (manually or by a garbage collector), and it can fragment.

**Q: What is a memory leak and how does it happen in Python?**
A memory leak is memory that the program has allocated but can no longer use *and* will not release. In C it typically comes from forgetting to `free`. In Python, which frees objects automatically, leaks come from unintentionally keeping objects reachable — for example a module-level dictionary or list that you keep appending to, a cache without eviction, or registered callbacks that are never removed. Because a live reference still exists, the garbage collector correctly refuses to reclaim the object, and memory grows without bound.

**Q: Why can't reference counting alone reclaim cycles?**
Reference counting frees an object the moment its count hits zero. But consider two objects A and B that reference each other: even if nothing else in the program points to them, A's count is at least 1 (because B points to it) and B's count is at least 1 (because A points to it). Their counts never reach zero, so they are never freed despite being unreachable. Python solves this with a separate **cyclic garbage collector** that periodically finds groups of objects reachable only from each other and collects them.

**Q (Scenario): Slowly growing RAM in a web service?**
I'd treat it as a leak. First I'd confirm the trend with monitoring (RSS over time). Then I'd take heap snapshots at intervals under load and diff them to see which object types grow — using tools like `tracemalloc`, `objgraph`, or a profiler. Common culprits: an unbounded cache, accumulating log/session objects, or listeners not deregistered. The fix is usually to bound the growth: add cache eviction (LRU with a max size), remove references when done, or use weak references so the GC can reclaim cached objects under pressure.

## Common Mistakes

- Believing garbage-collected languages "cannot leak" — they can, via lingering references.
- Confusing stack overflow (too-deep recursion) with heap exhaustion (out of memory).
- Storing large or long-lived data on the stack.
- Forgetting that closures and default mutable arguments can silently retain memory.

## Related Concepts

- [Recursion](#8-recursion) — uses the call stack; deep recursion → stack overflow.
- [Complexity Analysis](#3-complexity-analysis) — space complexity measures heap/stack usage.
- Garbage collection, reference counting, pointers.

---

# 3. Complexity Analysis

## What is it?

**Complexity analysis** measures how the resources an algorithm uses — **time** (number of basic operations) and **space** (amount of memory) — grow as the input size `n` grows. Instead of timing code with a stopwatch (which depends on hardware, language, and load), we count operations *as a function of input size* and describe the growth rate. **Asymptotic analysis** focuses on behavior as `n` becomes large, ignoring constant factors and lower-order terms.

**Big-O notation** expresses an **upper bound** on growth — the worst case. We also have:
- **Big-Ω (Omega):** lower bound (best case).
- **Big-Θ (Theta):** tight bound (when upper and lower match).

## Why is it needed?

We need a way to compare algorithms that is **independent of hardware and input**. A stopwatch tells you how fast *this* code ran on *this* machine with *this* input today; complexity tells you how the algorithm will **scale**. If algorithm A is O(n log n) and B is O(n²), then no matter the machine, B will eventually lose badly as data grows. Interviewers ask for complexity because it predicts whether your solution survives at scale.

## How does it work?

You count the number of **elementary operations** (comparisons, assignments, arithmetic) as a function of `n`, then keep only the **dominant term** and drop constants.

```
T(n) = 3n² + 5n + 100
       └─ drop constants and lower-order terms
     ⇒ O(n²)
```

**Rules of thumb:**
- A single loop over n items → **O(n)**.
- Nested loop, each over n → **O(n²)**.
- Halving the problem each step (binary search) → **O(log n)**.
- Divide in half + linear work per level (merge sort) → **O(n log n)**.
- Recursion that branches into 2 each call, depth n → **O(2ⁿ)**.

**Growth comparison (small → large, best → worst):**

```
O(1) < O(log n) < O(n) < O(n log n) < O(n²) < O(2ⁿ) < O(n!)

 n        n log n     n²          2ⁿ
 10       ~33         100         1,024
 100      ~664        10,000      ~1.3e30   (astronomically large)
 1000     ~9,966      1,000,000   don't ask
```

## Internal Working

Big-O has a precise definition: `f(n) = O(g(n))` means there exist constants `c > 0` and `n₀` such that for all `n ≥ n₀`, `f(n) ≤ c·g(n)`. In words: beyond some input size, `f` never grows faster than a constant multiple of `g`. This is why we drop constants (they fold into `c`) and lower-order terms (they are dominated for large `n`).

We analyze three cases:
- **Best case (Ω):** luckiest input (e.g., target is the first element).
- **Average case:** expected over typical inputs.
- **Worst case (O):** unluckiest input; usually what we quote, because it guarantees an upper bound.

**Space complexity** counts extra memory beyond the input: variables, recursion stack frames, and auxiliary data structures. An **in-place** algorithm uses O(1) extra space.

## Advantages

- Hardware- and language-independent comparison.
- Predicts scalability before you run anything.
- Concise shared vocabulary for engineers.

## Limitations

- Hides constant factors and lower-order terms that matter at small `n`.
- Worst case may be rare; average case may be more relevant.
- Ignores cache effects, memory bandwidth, and real hardware behavior.

## Real-world Applications

- Choosing algorithms/data structures for systems handling millions of records.
- Database query planners estimate cost using complexity.
- Interview screening — nearly every coding interview asks "what's the time and space complexity?"

## Common complexities of core operations (memorize this table)

| Structure | Access | Search | Insert | Delete |
|---|---|---|---|---|
| Array (dynamic) | O(1) | O(n) | O(n) (O(1) amortized at end) | O(n) |
| Linked List | O(n) | O(n) | O(1) (at known position) | O(1) (at known position) |
| Stack | O(n) | O(n) | O(1) (push) | O(1) (pop) |
| Queue | O(n) | O(n) | O(1) (enqueue) | O(1) (dequeue) |
| Hash Table | — | O(1) avg / O(n) worst | O(1) avg | O(1) avg |
| Binary Search Tree (balanced) | O(log n) | O(log n) | O(log n) | O(log n) |
| Binary Search Tree (worst/skewed) | O(n) | O(n) | O(n) | O(n) |

## Interview Questions

**Beginner**
- What is Big-O notation?
- What is the time complexity of accessing an array element by index?

**Intermediate**
- Explain the difference between best, average, and worst case with an example.
- What is amortized time complexity?

**Advanced**
- Prove or explain why binary search is O(log n).
- Why do we drop constants and lower-order terms in Big-O?

**Scenario-based**
- Your function passes on small test data but times out on large inputs. How do you reason about it?

**"Why" questions**
- Why is O(n log n) considered the practical lower bound for comparison-based sorting?

**Comparison**
- Compare O(n) space vs O(1) space (in-place) algorithms.
- Big-O vs Big-Θ vs Big-Ω.

## Model Answers

**Q: What is Big-O and why drop constants?**
Big-O describes the asymptotic upper bound on how an algorithm's running time (or space) grows with input size `n`. Formally `f(n)=O(g(n))` if beyond some input size, `f(n)` stays within a constant multiple of `g(n)`. We drop constants and lower-order terms because as `n` grows, the highest-order term dominates and constant factors depend on hardware, not the algorithm. For example, `3n² + 100n + 5` is O(n²): for large `n`, the `n²` term swamps everything else, and whether the coefficient is 3 or 300 doesn't change the fundamental scaling.

**Q: Best vs average vs worst case?**
Consider linear search for a value in an unsorted array of size n. Best case is O(1) — the value is the first element. Worst case is O(n) — the value is last or absent, so we check every element. Average case, assuming the value is equally likely at any position, is about n/2 comparisons, still O(n). We usually quote the worst case because it gives a guarantee, but for structures like hash tables the *average* case (O(1)) is what matters in practice even though the worst case is O(n).

**Q: What is amortized complexity?**
Amortized complexity is the average cost per operation over a sequence of operations, when occasional expensive operations are "paid for" by many cheap ones. The classic example is appending to a dynamic array: most appends are O(1), but when the array is full it must resize by copying all n elements (O(n)). Because resizing doubles capacity, it happens rarely enough that the *average* cost per append across many appends is O(1) — we say append is **O(1) amortized**. It differs from average case: amortized is about a worst-case *sequence*, not probabilistic inputs.

**Q: Why is binary search O(log n)?**
Binary search works on a sorted array by comparing the target to the middle element and discarding half the remaining range each step. Starting with n elements, after one step we have n/2, then n/4, then n/8, and so on. The number of times you can halve n before reaching 1 is log₂(n). Since we do constant work per step, the total is O(log n). That's why doubling the data adds only one extra step.

**Q: Why is O(n log n) the bound for comparison sorting?**
Any sort that only compares elements must distinguish among all n! possible orderings. A comparison yields one bit of information (yes/no), and a decision tree distinguishing n! outcomes must have height at least log₂(n!), which by Stirling's approximation is Θ(n log n). Therefore no comparison-based sort can do better than O(n log n) in the worst case. Non-comparison sorts like counting/radix sort can beat this because they don't compare — they use the values' structure directly.

## Common Mistakes

- Saying "Big-O is the running time" — it's a bound on *growth*, not actual time.
- Reporting best case as if it were the general complexity.
- Forgetting recursion's **space** cost (call-stack frames).
- Confusing O(log n) base — the base is irrelevant in Big-O (constant factor).
- Thinking lower Big-O always means faster in practice.

## Related Concepts

- Every other topic — complexity is how we compare them all.
- [Recursion](#8-recursion) — recurrence relations, space on the stack.
- [Sorting](#9-sorting-algorithms) — canonical complexity examples.

---

# 4. Arrays

## What is it?

An **array** is a collection of elements stored in **contiguous memory locations**, accessed by an integer **index**. All elements are usually of the same type (in low-level languages), and the array has a fixed base address. Because the elements sit next to each other in memory and each occupies the same number of bytes, the computer can compute the exact address of any element instantly.

There are two flavors:
- **Static array** (fixed size, e.g., C `int arr[10]`): size chosen at creation, cannot grow.
- **Dynamic array** (resizable, e.g., Python `list`, Java `ArrayList`, C++ `vector`): grows automatically by allocating a bigger block and copying.

## Why is it needed?

Arrays give you **O(1) random access**: reading or writing element `i` takes the same tiny amount of time regardless of the array's size. This is the fastest possible access and the reason arrays underpin almost everything — strings, matrices, hash tables, heaps, and dynamic lists are all built on arrays. When you know you'll index into data frequently, arrays are the natural choice.

## How does it work?

Given a base address `B`, element size `s` bytes, and index `i`, the address of element `i` is computed by simple arithmetic:

```
address(i) = B + i * s
```

No searching, no traversal — one multiplication and one addition. That constant-time formula is the whole magic.

```
Index:     0     1     2     3     4
        ┌─────┬─────┬─────┬─────┬─────┐
Value:  │ 10  │ 20  │ 30  │ 40  │ 50  │
        └─────┴─────┴─────┴─────┴─────┘
Addr:   100   104   108   112   116     (4 bytes each)

address(3) = 100 + 3*4 = 112  → value 40, in O(1)
```

**Operations and costs:**
- **Access by index:** O(1).
- **Search (unsorted):** O(n) — must scan.
- **Search (sorted):** O(log n) with binary search.
- **Insert/delete at end (dynamic):** O(1) amortized.
- **Insert/delete at middle/front:** O(n) — must shift elements to keep contiguity.

```
Insert 99 at index 1 → shift 20,30,40,50 right:
Before: [10, 20, 30, 40, 50]
After:  [10, 99, 20, 30, 40, 50]   (4 shifts → O(n))
```

## Internal Working

A **dynamic array** keeps three things: a pointer to a heap block, the current **size** (used slots), and the **capacity** (allocated slots). When you append and `size == capacity`, it allocates a new block (typically **double** the capacity), copies existing elements over, and frees the old block. Doubling makes the *amortized* cost of append O(1): although a single resize is O(n), resizes happen rarely (after 1, 2, 4, 8, … appends), so the total copying work across `n` appends is at most ~2n, i.e., O(1) per append on average.

Because elements are contiguous, arrays are extremely **cache-friendly**: when the CPU loads one element, it pulls neighboring elements into cache too, making sequential scans very fast in practice — often faster than "theoretically equal" linked structures.

## Advantages

- O(1) random access by index.
- Excellent cache locality → fast iteration.
- Low memory overhead per element (no pointers).
- Simple and universally supported.

## Limitations

- Insert/delete in the middle is O(n) (shifting).
- Static arrays have fixed size; dynamic arrays pay occasional O(n) resize + copy.
- Resizing may temporarily need extra memory (old + new block).

## Real-world Applications

- Backing store for dynamic lists, stacks, heaps, hash tables.
- Image/matrix data (pixels, tensors) — 2D/ND arrays.
- Lookup tables and buffers (audio, network packets).
- Any time-critical indexed access (game state, simulation grids).

## Interview Questions

**Beginner**
- Why is array access O(1)?
- What is the difference between a static and a dynamic array?

**Intermediate**
- Why is inserting at the beginning of an array O(n)?
- Explain how a dynamic array grows and why append is O(1) amortized.

**Advanced**
- Why are arrays often faster than linked lists in practice even for the "same" complexity?
- How would you rotate an array by k positions in-place with O(1) extra space?

**Scenario-based**
- You need a structure with frequent random reads but rare inserts. Array or linked list? Why?

**"Why" questions**
- Why does doubling (not adding a fixed amount) on resize give amortized O(1)?

**Comparison**
- Array vs linked list: compare access, insert, delete, memory.

## Model Answers

**Q: Why is array access O(1)?**
Because array elements are stored contiguously and are equally sized, the address of element `i` is computed directly as `base + i * element_size`. That is a single multiplication and addition regardless of `i` or the array length, so it takes constant time. There is no need to walk through preceding elements as you would in a linked list.

**Q: Why is inserting at the beginning O(n)?**
An array must stay contiguous. To insert at index 0, every existing element has to move one slot to the right to make room, which is `n` moves for `n` elements — O(n). Inserting at the end (in a dynamic array with spare capacity) needs no shifting, so it's O(1). This asymmetry is why arrays are great for end operations but poor for front/middle insertions.

**Q: Why is append amortized O(1) with doubling?**
When capacity is full, the array doubles and copies all `n` elements — an O(n) operation. But because capacity doubles, these expensive copies occur at sizes 1, 2, 4, 8, …, n. The total copying across `n` appends sums to `1 + 2 + 4 + … + n ≈ 2n`, which is O(n) total, or **O(1) per append averaged**. If instead we grew by a fixed amount (say +1 each time), every append would copy, giving O(n) per append and O(n²) overall — which is why doubling (geometric growth) is essential.

**Q: Why are arrays often faster than linked lists in practice?**
Even when both are O(n) for a scan, arrays win because of **cache locality**. Contiguous elements are loaded into CPU cache lines together, so iterating an array mostly hits fast cache. A linked list's nodes are scattered across the heap, so each `next` pointer likely causes a cache miss and a slow main-memory fetch. Arrays also avoid the per-node pointer overhead. So constant factors and memory behavior favor arrays for traversal-heavy work.

## Common Mistakes

- Thinking arrays can grow for free — dynamic arrays occasionally pay O(n) to resize.
- Off-by-one errors on indices and bounds (`< n` vs `<= n`).
- Assuming middle insert/delete is cheap — it's O(n) due to shifting.
- Forgetting that Python lists are dynamic arrays, not linked lists.

## Related Concepts

- [Linked Lists](#5-linked-lists) — the contrasting structure.
- [Strings](#6-strings) — arrays of characters.
- [Hash Tables](#10-hash-tables), [Stacks](#11-stacks), [Queues](#12-queues) — often array-backed.

---

# 5. Linked Lists

## What is it?

A **linked list** is a linear data structure where elements (called **nodes**) are stored at scattered memory locations and connected by **pointers**. Each node holds two things: the **data** and a **reference (pointer)** to the next node. Unlike an array, nodes are *not* contiguous — you follow pointers to move through the list.

Variants:
- **Singly Linked List (SLL):** each node points only to the next node. One-directional.
- **Doubly Linked List (DLL):** each node points to both the next and the previous node. Two-directional.
- **Circular Linked List:** the last node points back to the head (not in this syllabus but worth knowing).

## Why is it needed?

Arrays are painful when you insert or delete in the middle because everything must shift. A linked list solves this: to insert or delete a node, you just **re-wire a couple of pointers** — O(1) if you already hold the position. Linked lists also grow and shrink one node at a time without resizing or copying, so memory is allocated exactly as needed. They're ideal when the data changes shape frequently and you don't need random indexed access.

## How does it work?

**Singly Linked List:**

```
head
 │
 ▼
┌────┬───┐   ┌────┬───┐   ┌────┬───┐
│ 10 │ ●─┼──▶│ 20 │ ●─┼──▶│ 30 │ ╱ │──▶ None
└────┴───┘   └────┴───┘   └────┴───┘
 data next    data next    data next
```

**Doubly Linked List:**

```
       ┌────┬────┬───┐   ┌────┬────┬───┐   ┌────┬────┬───┐
None ◀─┼─╱  │ 10 │ ●─┼──▶│ ●  │ 20 │ ●─┼──▶│ ●  │ 30 │ ╱ │──▶ None
       └────┴────┴───┘   └────┴────┴───┘   └────┴────┴───┘
        prev data next    prev data next    prev data next
```

**Core operations:**
- **Traverse:** start at `head`, follow `next` until `None` — O(n).
- **Access by index:** O(n) — no formula, you must walk.
- **Insert/delete at head:** O(1) — re-point head.
- **Insert/delete after a known node:** O(1) — re-wire pointers.
- **Insert/delete by value:** O(n) — must search first, then O(1) to rewire.
- **Search:** O(n).

**Insert at head (SLL):**
```
new_node.next = head
head = new_node
```

**Delete a node with value x (SLL):** walk with a `prev` pointer, then `prev.next = current.next`.

```
Before:  A → X → B
Set A.next = B:
After:   A → B     (X unlinked)
```

## Internal Working

Each node is a small heap-allocated object containing the data and one (SLL) or two (DLL) pointers. The list itself is just a reference to the `head` node (and often a `tail` for O(1) end-append). Because nodes live wherever the allocator places them, traversal jumps around memory, causing cache misses — this is the hidden cost that makes linked lists slower in practice than their Big-O suggests.

A **doubly linked list** stores a `prev` pointer too, enabling O(1) deletion of a node you already hold (no need to find its predecessor) and backward traversal — at the cost of one extra pointer per node and more bookkeeping.

To **convert an array to a linked list**, iterate the array and append each element as a new node, keeping a `tail` pointer so each append is O(1); total O(n).

To **remove duplicates from a sorted SLL**, walk once comparing each node to its next; if equal, unlink the next — O(n), because duplicates are adjacent when sorted.

## Advantages

- O(1) insert/delete at a known position (especially head).
- Grows/shrinks dynamically, no resizing or copying.
- Memory allocated per node — no wasted pre-allocated capacity.
- DLL allows backward traversal and O(1) delete of a held node.

## Limitations

- No O(1) random access — indexing is O(n).
- Extra memory per node for pointer(s).
- Poor cache locality → slower traversal than arrays in practice.
- More complex to implement; pointer bugs (losing the list, cycles) are common.

## Real-world Applications

- Implementation of **stacks and queues** (esp. when size is unknown).
- **LRU cache** uses a doubly linked list + hash map for O(1) eviction.
- **Undo/redo** and browser history (doubly linked).
- Music/media playlists (next/previous).
- The free list inside memory allocators.

## Interview Questions

**Beginner**
- What is a linked list and how is a node structured?
- What is the difference between a singly and doubly linked list?

**Intermediate**
- Why is inserting at the head O(1) but accessing the k-th element O(n)?
- How do you detect a cycle in a linked list?

**Advanced**
- Reverse a singly linked list in-place. Explain the pointer manipulation.
- Find the middle of a linked list in one pass.

**Scenario-based**
- You must implement an LRU cache with O(1) get and put. What structure do you use and why?

**"Why" questions**
- Why would you choose a linked list instead of an array for a queue?

**Comparison**
- Compare singly vs doubly linked list in memory and capability.
- Compare linked list vs dynamic array for frequent middle insertions.

## Model Answers

**Q: Singly vs doubly linked list?**
A singly linked list node stores data and a single `next` pointer, so you can only traverse forward and deleting a node requires knowing its predecessor. A doubly linked list node additionally stores a `prev` pointer, allowing backward traversal and O(1) deletion of a node you already hold, because you can reach its neighbors directly. The trade-off is extra memory (one more pointer per node) and more pointers to maintain correctly on every insert/delete.

**Q: Why is head insert O(1) but indexing O(n)?**
Inserting at the head only requires creating a node, pointing its `next` to the current head, and moving the head reference — a constant number of pointer operations independent of list length. Indexing, however, has no address formula like an array; to reach element k you must start at the head and follow `next` k times, which is O(n). This is the fundamental trade-off: linked lists trade fast random access for fast structural edits.

**Q: Reverse a singly linked list in-place.**
I use three pointers: `prev = None`, `curr = head`, and a temporary `next`. I walk the list, and for each node I save `next = curr.next`, reverse the link with `curr.next = prev`, then advance `prev = curr` and `curr = next`. When `curr` becomes `None`, `prev` is the new head. This runs in O(n) time and O(1) space because I reverse the existing pointers rather than building a new list. The key insight is saving `next` *before* overwriting `curr.next`, otherwise I'd lose the rest of the list.

**Q (Scenario): LRU cache with O(1) get/put?**
I combine a **hash map** and a **doubly linked list**. The hash map maps keys to nodes for O(1) lookup. The doubly linked list maintains usage order: most-recently-used at the front, least-recently-used at the back. On `get`, I look up the node in the map and move it to the front. On `put`, I insert at the front; if capacity is exceeded, I remove the tail node (LRU) and delete its key from the map. The doubly linked list gives O(1) removal of any node (because each node knows its neighbors), and the map gives O(1) access — together they achieve O(1) for both operations.

**Q (Why): Linked list instead of array for a queue?**
A queue needs enqueue at one end and dequeue at the other. With a plain array, dequeuing from the front is O(n) because remaining elements must shift. A linked list (with head and tail pointers) does both ends in O(1) with no shifting and no resizing. It also grows exactly to the number of elements. (In practice a ring buffer / deque can also give O(1), but conceptually the linked list cleanly avoids the shifting problem.)

## Common Mistakes

- Losing the rest of the list by overwriting `next` before saving it (in reversal).
- Forgetting to update `head`/`tail` on insert/delete at the ends.
- Not handling edge cases: empty list, single node, deleting the head.
- In a DLL, updating only one direction's pointer, corrupting the list.
- Creating an accidental cycle, causing infinite traversal.

## Related Concepts

- [Arrays](#4-arrays) — the contrasting contiguous structure.
- [Stacks](#11-stacks) and [Queues](#12-queues) — often built on linked lists.
- [Hash Tables](#10-hash-tables) — chaining uses linked lists; LRU combines both.

---

# 6. Strings

## What is it?

A **string** is a sequence of characters. Internally it is essentially an **array of characters** (bytes or code units), often with encoding rules (ASCII, UTF-8, UTF-16). In many languages strings are **immutable** — once created they cannot be changed; any "modification" creates a new string. Python, Java, JavaScript, and C# all use immutable strings; C uses mutable null-terminated character arrays.

## Why is it needed?

Text is everywhere: names, messages, files, URLs, DNA sequences, source code. Strings give us a first-class way to store and manipulate text, with a rich set of operations (search, replace, split, compare) built on top of the array-of-characters foundation. Understanding strings deeply matters because string problems are among the most common in interviews and because subtle issues (immutability cost, encoding) cause real bugs.

## How does it work?

A string stores characters contiguously, like an array, plus length information (or a terminator). Indexing a character is O(1), just like an array.

```
 "HELLO"
 Index:  0    1    2    3    4
        ┌────┬────┬────┬────┬────┐
        │ H  │ E  │ L  │ L  │ O  │
        └────┴────┴────┴────┴────┘
```

**Common operations:**
- **Access char by index:** O(1).
- **Length:** O(1) (usually stored).
- **Concatenation:** O(n + m) — must copy both into a new string (immutability).
- **Substring:** O(k) for length-k slice.
- **Search (substring):** O(n·m) naive; O(n + m) with KMP.
- **Compare:** O(min(n, m)).

**The syllabus example** — *replace each alphabet with the letter n positions from it (a Caesar cipher / shift)*: for each character, compute its new position with modular arithmetic so it wraps around the alphabet.

```
shift 'a' by 2 → 'c'
shift 'y' by 2 → 'a'  (wraps: y→z→a)
Formula:  new = (ord(ch) - ord('a') + n) % 26 + ord('a')
```

## Internal Working

Because most languages make strings **immutable**, operations that appear to modify a string actually allocate a new one and copy characters. This is why building a string by repeated concatenation in a loop is O(n²): each `+=` copies the whole accumulated string. The fix is to collect pieces in a list and `join` once (O(n)), or use a mutable buffer (`StringBuilder` in Java, `bytearray`/list in Python).

Immutability has benefits: strings can be **safely shared** between threads and used as **hash keys** (their hash never changes), and interning (deduplicating identical strings) saves memory.

**Encoding:** ASCII uses 1 byte per character (128 symbols). **UTF-8** is variable-width (1–4 bytes), so the number of bytes ≠ number of characters for non-ASCII text — a common source of bugs when you assume `len(bytes) == len(characters)`.

## Advantages

- O(1) indexed character access (array-backed).
- Immutability enables safe sharing, hashing, and caching.
- Rich, well-optimized library operations.

## Limitations

- Immutability makes in-place edits impossible; naive concatenation is O(n²).
- Encoding complexity (UTF-8) breaks naive byte/char assumptions.
- Substring search can be expensive without the right algorithm.

## Real-world Applications

- Parsing and validation (emails, URLs, config files).
- Search engines, text editors, autocomplete.
- Cryptography (ciphers, hashing).
- Bioinformatics (DNA/protein sequences).
- Natural language processing and tokenization.

## Interview Questions

**Beginner**
- Are strings mutable in Python? What does that imply?
- How do you reverse a string?

**Intermediate**
- Why is building a string with repeated `+=` in a loop slow, and how do you fix it?
- How would you check if two strings are anagrams?

**Advanced**
- Explain an efficient substring-search algorithm (e.g., KMP) at a high level.
- How do you find the longest palindromic substring?

**Scenario-based**
- You must implement a Caesar cipher that shifts each letter by n and wraps around. How?

**"Why" questions**
- Why are immutable strings useful as dictionary keys?

**Comparison**
- Compare string concatenation via `+` vs `join`.

## Model Answers

**Q: Are Python strings mutable? Implications?**
No, Python strings are immutable — once created they cannot be changed in place. Any operation that "changes" a string, like `s.replace()` or `s += x`, actually creates and returns a new string. The implication is twofold: (1) building strings by repeated concatenation in a loop is inefficient — O(n²) — because each step copies the whole string, so you should accumulate pieces in a list and use `''.join(pieces)` once; and (2) immutability makes strings safe to use as dictionary keys and to share across threads, since their value and hash never change.

**Q: Why is `+=` in a loop slow, and the fix?**
Because strings are immutable, each `result += chunk` creates a brand-new string containing all previous characters plus the new chunk, copying everything each time. Over `n` appends this is 1 + 2 + … + n ≈ O(n²) character copies. The fix is to append the chunks to a list (O(1) each) and join them once at the end with `''.join(list)`, which is a single O(n) pass. Java's equivalent is using `StringBuilder` instead of `+`.

**Q: Check if two strings are anagrams?**
Two strings are anagrams if they contain the same characters with the same frequencies. The clean approach is to count character frequencies with a hash map (or `collections.Counter`) for both strings and compare the counts — O(n) time, O(1) extra space for a fixed alphabet. An alternative is to sort both strings and compare — O(n log n), simpler but slower. I'd first check that lengths match to reject early. The counting approach is preferred because it's linear.

**Q (Scenario): Caesar cipher with wrap-around?**
For each character I map it into a 0–25 range by subtracting the base (`ord('a')`), add the shift `n`, take modulo 26 so it wraps past 'z' back to 'a', then add the base back: `chr((ord(ch) - ord('a') + n) % 26 + ord('a'))`. I handle uppercase and lowercase with their own bases and leave non-letters unchanged. The modulo is the key trick that makes the alphabet circular. Decryption is the same with shift `26 - n` (or `-n`).

## Common Mistakes

- Building strings with `+=` in loops (quadratic).
- Assuming one byte equals one character (breaks for UTF-8).
- Trying to mutate a string in place in immutable-string languages.
- Off-by-one errors in substring indices.
- Forgetting case and non-alphabetic characters in cipher problems.

## Related Concepts

- [Arrays](#4-arrays) — strings are character arrays.
- [Hash Tables](#10-hash-tables) — frequency counting, anagrams.
- [Searching Algorithms](#7-searching-algorithms) — substring search.

---

# 7. Searching Algorithms

## What is it?

**Searching** means finding whether (and where) a target value exists in a collection. The two fundamental algorithms are:
- **Linear Search:** check each element one by one until you find the target or reach the end.
- **Binary Search:** on a **sorted** collection, repeatedly halve the search range by comparing the target to the middle element.

## Why is it needed?

Retrieval is one of the most common operations in computing — "does this exist, and where?" Linear search is universal but slow on large data. Binary search is dramatically faster (logarithmic) but requires sorted data. Knowing when each applies — and the sorted precondition of binary search — is a staple of interviews and a foundation for many other algorithms (search trees, `bisect`, lower/upper bound).

## How does it work?

**Linear Search — O(n):**
```
Look for 30 in [10, 40, 30, 20]:
 check 10 ✗ → check 40 ✗ → check 30 ✓ (found at index 2)
```
Works on any collection, sorted or not. Best case O(1) (first element), worst case O(n).

**Binary Search — O(log n):** requires a **sorted** array.
```
Find 23 in [4, 8, 15, 16, 23, 42], indices 0..5
low=0, high=5 → mid=2 (value 15) → 23 > 15 → search right half
low=3, high=5 → mid=4 (value 23) → found at index 4
```
Each comparison discards half the remaining elements, so it takes at most log₂(n) steps.

```
Step 1: [4  8  15 |16  23  42]   mid=15, target bigger → go right
Step 2:            [16  23 |42]  mid=23 → found
```

## Internal Working

**Binary search** maintains two pointers, `low` and `high`, bounding the current search range. Each iteration computes `mid = low + (high - low) // 2` (this form avoids integer overflow that `(low + high) // 2` can cause in fixed-width languages). It compares `arr[mid]` to the target:
- Equal → found.
- Target greater → discard left half: `low = mid + 1`.
- Target smaller → discard right half: `high = mid - 1`.

The loop continues while `low <= high`. When the range empties, the target is absent. It can be written iteratively (O(1) space) or recursively (O(log n) space for the call stack). Variants like **lower_bound / upper_bound** find the first/last position and power range queries.

## Advantages

- **Linear:** works on any data, no preconditions, minimal code, cache-friendly.
- **Binary:** logarithmic — 1 million elements in ~20 steps, 1 billion in ~30.

## Limitations

- **Linear:** O(n), too slow for large data with frequent searches.
- **Binary:** requires sorted data (sorting costs O(n log n)); needs random access (arrays, not linked lists); careful boundary handling or off-by-one/infinite-loop bugs.

## Real-world Applications

- Binary search underlies database indexes, `bisect` insertions, dictionary lookups by key range, version-control **git bisect** (finding the commit that introduced a bug), and "search in rotated array" style interview problems.
- Linear search is used for small or unsorted data, or when a single pass is acceptable.

## Interview Questions

**Beginner**
- What is the time complexity of linear vs binary search?
- What precondition must hold for binary search?

**Intermediate**
- Why is binary search O(log n)? Explain the halving.
- Write binary search and handle the empty/not-found case.

**Advanced**
- How would you search in a sorted array that has been rotated?
- Find the first occurrence of a value in a sorted array with duplicates.

**Scenario-based**
- You search a 10-million-element sorted list thousands of times per second. Which algorithm, and what's the cost per search?

**"Why" questions**
- Why can't you binary search a linked list efficiently?

**Comparison**
- Linear vs binary search: when is linear actually the better choice?

## Model Answers

**Q: Linear vs binary complexity and preconditions?**
Linear search is O(n) and works on any collection because it simply checks each element in turn. Binary search is O(log n) but requires the data to be **sorted** and to support **random access** (like an array), because it jumps to the middle each step. If the data is unsorted, you either use linear search or pay O(n log n) to sort first — which only pays off if you'll search many times.

**Q: Why is binary search O(log n)?**
Each comparison eliminates half of the remaining candidates. Starting from `n` elements, the range shrinks n → n/2 → n/4 → … → 1. The number of halvings needed to reduce `n` to 1 is log₂(n), and each step does constant work, giving O(log n). Concretely, a billion elements need only about 30 comparisons.

**Q: Search in a rotated sorted array?**
A rotated sorted array (e.g., `[15, 23, 42, 4, 8]`) is still "half-sorted": at any midpoint, at least one side is properly sorted. I do a modified binary search: compute `mid`, then determine which half is sorted by comparing `arr[low]` and `arr[mid]`. If the left half is sorted and the target lies within its range, I search left; otherwise right. Symmetrically for the right half. This keeps it O(log n) without sorting. The key insight is using the sorted half to decide direction.

**Q (Why): Binary search on a linked list?**
Binary search needs to jump to the middle element in O(1), which requires random access via `base + index`. A linked list has no such formula — reaching the middle means walking from the head, which is O(n). So each "jump" costs O(n), destroying the O(log n) benefit; overall it degrades to O(n log n) or worse. That's why binary search is an array algorithm; on linked structures we use balanced trees or skip lists instead.

**Q (Scenario): 10M sorted list, thousands of searches/sec?**
Binary search. Ten million elements need at most about 24 comparisons per search (log₂(10⁷) ≈ 23.3), which is trivially fast even thousands of times per second. Linear search would average 5 million comparisons per query — many orders of magnitude worse. Since the list is already sorted, there's no sorting cost, so binary search is clearly correct here.

## Common Mistakes

- Running binary search on unsorted data (silently wrong results).
- Off-by-one errors in `low`/`high`/`mid`, causing infinite loops or misses.
- Using `(low + high) // 2` in languages where it overflows.
- Forgetting the not-found case.
- Not moving `low`/`high` past `mid`, causing an infinite loop.

## Related Concepts

- [Sorting Algorithms](#9-sorting-algorithms) — binary search's precondition.
- [Complexity Analysis](#3-complexity-analysis) — O(log n) derivation.
- [Trees & BST](#13-trees--binary-search-trees) — binary search as a tree structure.
- [Recursion](#8-recursion) — recursive binary search.

---

# 8. Recursion

## What is it?

**Recursion** is when a function solves a problem by **calling itself** on a smaller version of the same problem, until it reaches a case simple enough to answer directly. Every recursive function has two parts:
- **Base case:** the smallest input that is answered directly, without further recursion. It stops the recursion.
- **Recursive case:** the function reduces the problem and calls itself, trusting that the smaller call returns the correct answer.

## Why is it needed?

Some problems are naturally self-similar: a structure is defined in terms of smaller copies of itself. Trees, nested lists, divide-and-conquer algorithms (merge sort, quicksort, binary search), backtracking (permutations, mazes), and mathematical definitions (factorial, Fibonacci) are all far cleaner expressed recursively than with loops. Recursion lets you write code that mirrors the structure of the problem, which is easier to reason about and prove correct.

## How does it work?

Consider factorial: `n! = n × (n-1)!`, with base case `0! = 1`.

```
factorial(4)
= 4 * factorial(3)
      = 3 * factorial(2)
            = 2 * factorial(1)
                  = 1 * factorial(0)
                        = 1            ← base case
                  = 1 * 1 = 1
            = 2 * 1 = 2
      = 3 * 2 = 6
= 4 * 6 = 24
```

The calls "wind up" (each waits for the smaller call), hit the base case, then "unwind" (results return back up). Each pending call is stored as a **stack frame** on the call stack.

## Internal Working

Every function call pushes a **stack frame** holding its parameters, local variables, and the return address. In recursion, frames stack up until the base case is reached, then they pop one by one as each returns. This is why:
- The **maximum recursion depth** determines **space complexity** — recursion uses O(depth) stack memory even if it returns a single number.
- Too-deep recursion causes a **stack overflow** (Python caps recursion at ~1000 by default).

**Recursion vs iteration:** any recursion can be rewritten with an explicit stack/loop. Some languages optimize **tail recursion** (where the recursive call is the last action) into a loop, using O(1) stack — but Python does **not** do tail-call optimization.

**Beware naive recursion cost:** naive Fibonacci recomputes the same subproblems exponentially:
```
fib(5) calls fib(4) and fib(3); fib(4) calls fib(3) again... → O(2ⁿ)
```
**Memoization** (caching results) collapses this to O(n) — the bridge to dynamic programming.

## Advantages

- Clean, readable code for self-similar/recursive structures (trees, divide-and-conquer).
- Mirrors mathematical definitions closely; easier correctness reasoning.
- Natural fit for backtracking and traversal.

## Limitations

- Uses O(depth) call-stack memory; deep recursion overflows the stack.
- Function-call overhead makes it slower than an equivalent loop.
- Naive recursion can recompute subproblems (exponential) without memoization.
- Harder to debug (many stacked frames).

## Real-world Applications

- Tree/graph traversals (DFS), directory walking, JSON/XML parsing.
- Divide-and-conquer: merge sort, quicksort, binary search.
- Backtracking: permutations, N-Queens, Sudoku, maze solving.
- Parsers and interpreters (recursive descent).
- Dynamic programming (recursion + memoization).

## Interview Questions

**Beginner**
- What are the two essential parts of any recursive function?
- Write a recursive factorial.

**Intermediate**
- What is the space complexity of a recursive function and why?
- Why is naive recursive Fibonacci exponential, and how do you fix it?

**Advanced**
- Convert a recursive function to an iterative one using an explicit stack.
- What is tail recursion and does Python optimize it?

**Scenario-based**
- Your recursive tree traversal crashes with "maximum recursion depth exceeded" on deep trees. What do you do?

**"Why" questions**
- Why does every recursion need a base case?

**Comparison**
- Compare recursion vs iteration in readability, speed, and memory.

## Model Answers

**Q: Two essential parts of recursion?**
Every recursive function needs a **base case** and a **recursive case**. The base case is the simplest input that can be answered directly without recursing — it's what stops the recursion and prevents infinite calls. The recursive case reduces the problem toward the base case and calls the function on that smaller input. For factorial, the base case is `factorial(0) = 1`, and the recursive case is `factorial(n) = n * factorial(n-1)`. Without a correct base case (or if the recursive case doesn't move toward it), the recursion never terminates and overflows the stack.

**Q: Space complexity of recursion?**
A recursive function uses O(depth) space on the **call stack**, because each unfinished call keeps a stack frame — holding its parameters, locals, and return address — until it returns. Even if the function returns a single number, the frames pile up to the maximum recursion depth. For example, recursive factorial(n) uses O(n) stack space, and a balanced-tree traversal uses O(log n) (the height). This is why deep recursion can overflow the stack even when it does little work per call.

**Q: Why is naive Fibonacci exponential and how to fix it?**
Naive `fib(n) = fib(n-1) + fib(n-2)` recomputes the same subproblems repeatedly — `fib(n-2)` is computed by both `fib(n)` (indirectly) and `fib(n-1)`, and this duplication compounds, producing about 2ⁿ calls, i.e., O(2ⁿ). The fix is **memoization**: cache each `fib(k)` the first time it's computed and reuse it, so each value is computed once — O(n) time and O(n) space. Alternatively, an iterative bottom-up version uses O(n) time and O(1) space by keeping only the last two values.

**Q (Scenario): "Maximum recursion depth exceeded" on deep trees?**
This means the recursion is deeper than the interpreter's stack limit (Python defaults to ~1000). I have a few options: (1) convert the traversal to an **iterative** version using an explicit stack, which uses heap memory instead of the limited call stack — the robust fix; (2) if the depth is bounded and just slightly over, raise the limit with `sys.setrecursionlimit`, though that risks a real crash; (3) if the recursion is tail-recursive, restructure it into a loop. For production code on arbitrary-depth input, I prefer the explicit-stack iterative approach.

**Q (Why): Why a base case?**
The base case is the termination condition. Recursion works by reducing a problem to a smaller one repeatedly; without a base case, the reductions never stop, the function keeps calling itself, and the call stack grows until it overflows and the program crashes. The base case is the point where the answer is known directly, letting the chain of calls start returning and unwinding.

## Common Mistakes

- Missing or wrong base case → infinite recursion / stack overflow.
- Recursive case that doesn't actually shrink the input toward the base.
- Forgetting recursion's stack-space cost.
- Naive recursion without memoization (exponential blowup).
- Assuming Python optimizes tail calls (it does not).

## Related Concepts

- [Memory Management](#2-memory-management) — the call stack and stack overflow.
- [Trees & BST](#13-trees--binary-search-trees) and [Graphs](#14-graphs-bfs--dfs) — recursive traversals (DFS).
- [Sorting](#9-sorting-algorithms) — merge sort/quicksort are recursive.
- Dynamic programming — recursion + memoization.

---

# 9. Sorting Algorithms

## What is it?

**Sorting** arranges elements into a defined order (ascending or descending). This syllabus focuses on three simple **comparison-based** algorithms:
- **Bubble Sort:** repeatedly swap adjacent out-of-order pairs; large values "bubble" to the end.
- **Selection Sort:** repeatedly select the minimum of the unsorted portion and place it next.
- **Insertion Sort:** grow a sorted prefix by inserting each new element into its correct place, like sorting a hand of cards.

All three are O(n²) in the worst case but teach the core mechanics of sorting and are stepping stones to efficient sorts (merge, quick, heap).

## Why is it needed?

Sorted data unlocks fast operations: binary search (O(log n)), efficient merging, deduplication, range queries, and meaningful presentation (leaderboards, price lists). Sorting is one of the most-studied problems because it appears everywhere and because its analysis teaches complexity, stability, and in-place techniques. Interviews use these simple sorts to test whether you can reason about swaps, invariants, and complexity — even if production code calls a library sort.

## How does it work?

**Bubble Sort:** pass through the list swapping adjacent elements if they're in the wrong order; after each full pass the largest unsorted element is in place. Repeat until a pass makes no swaps.
```
[5, 1, 4, 2]  → compare pairs
5>1 swap → [1,5,4,2]
5>4 swap → [1,4,5,2]
5>2 swap → [1,4,2,5]  (5 bubbled to end)
next pass → [1,2,4,5] ...
```

**Selection Sort:** find the minimum in the unsorted part and swap it to the front of that part.
```
[5, 1, 4, 2]
min=1 → swap → [1, 5, 4, 2]
min of rest=2 → swap → [1, 2, 4, 5]
min of rest=4 → already → [1, 2, 4, 5]
```

**Insertion Sort:** take each element and shift larger sorted elements right to insert it.
```
[5, 1, 4, 2]
insert 1 → [1, 5, 4, 2]
insert 4 → [1, 4, 5, 2]
insert 2 → [1, 2, 4, 5]
```

## Internal Working

| Algorithm | Best | Average | Worst | Space | Stable? | Notes |
|---|---|---|---|---|---|---|
| Bubble Sort | O(n) | O(n²) | O(n²) | O(1) | Yes | Best case O(n) with early-exit flag on a sorted array |
| Selection Sort | O(n²) | O(n²) | O(n²) | O(1) | No | Always O(n²); minimizes number of swaps (n−1) |
| Insertion Sort | O(n) | O(n²) | O(n²) | O(1) | Yes | Excellent on nearly-sorted / small data; used by Timsort |

Key internal ideas:
- **In-place:** all three use O(1) extra memory — they rearrange within the array.
- **Stability:** a stable sort preserves the relative order of equal elements. Bubble and insertion are stable; selection is not (a long-distance swap can reorder equals). Stability matters when sorting by multiple keys.
- **Adaptivity:** insertion and bubble (with early exit) run in O(n) on already-sorted data; selection does not adapt.
- **Comparison lower bound:** no comparison sort beats O(n log n) worst case — so these simple O(n²) sorts are for learning, small inputs, or nearly-sorted data.

## Advantages

- Simple to understand and implement (great for learning and interviews).
- In-place, O(1) extra memory.
- Insertion sort is genuinely fast for small or nearly-sorted arrays.
- Bubble/insertion are stable.

## Limitations

- O(n²) — impractical for large datasets.
- Selection sort never adapts to existing order.
- Real systems use O(n log n) sorts (merge/quick/heap/Timsort).

## Real-world Applications

- Teaching fundamentals of algorithm analysis and invariants.
- **Insertion sort** is used inside hybrid sorts (Timsort in Python/Java, introsort in C++) for small subarrays.
- Nearly-sorted streaming data (online insertion).
- Small fixed-size arrays where simplicity beats overhead.

## Interview Questions

**Beginner**
- Explain how bubble sort works.
- Which of these three sorts is best for a nearly-sorted array?

**Intermediate**
- What does "stable sort" mean and which of these are stable?
- Why is selection sort always O(n²) even on sorted input?

**Advanced**
- Why can't any comparison-based sort beat O(n log n)?
- How does insertion sort achieve O(n) best case?

**Scenario-based**
- You continuously receive small batches of nearly-sorted data. Which simple sort fits and why?

**"Why" questions**
- Why does selection sort do the fewest swaps but isn't the fastest?

**Comparison**
- Compare bubble, selection, and insertion sort on time, stability, and adaptivity.

## Model Answers

**Q: How does bubble sort work?**
Bubble sort repeatedly steps through the list, comparing each adjacent pair and swapping them if they're in the wrong order. After the first full pass, the largest element has "bubbled" to the last position; after the second pass, the second-largest is in place, and so on. With an optimization — a flag that detects a pass with no swaps — it stops early on an already-sorted array, giving O(n) best case. Its average and worst case are O(n²) because of the nested passes, and it's stable and in-place.

**Q: What is a stable sort; which of these are stable?**
A stable sort preserves the relative order of elements that compare equal. For example, if you sort a list of (name, age) pairs by age, a stable sort keeps people of the same age in their original name order. Bubble sort and insertion sort are stable because they only swap adjacent elements and never jump an equal element past another. Selection sort is not stable, because it swaps the minimum into place across long distances, which can move an equal element past its duplicate. Stability matters for multi-key sorting.

**Q: Why is selection sort always O(n²)?**
Selection sort finds the minimum of the unsorted portion on every pass, and finding that minimum always scans the entire remaining unsorted region regardless of whether the data is already sorted. So the number of comparisons is fixed at roughly n²/2 no matter the input — it does not adapt. This contrasts with insertion or bubble sort, which can finish early or do less work when the input is already ordered. Selection sort's only redeeming trait is that it performs at most n−1 swaps.

**Q: Why can't comparison sorts beat O(n log n)?**
A comparison sort learns about the order only by asking yes/no questions ("is a < b?"). To sort n distinct items, it must be able to produce any of the n! possible orderings, and a sequence of yes/no comparisons forms a decision tree whose leaves are those orderings. A binary tree with n! leaves has height at least log₂(n!) ≈ n log n. Since the height is the worst-case number of comparisons, no comparison sort can do better than Θ(n log n) worst case. Non-comparison sorts (counting, radix) bypass this by exploiting the values directly.

**Q (Scenario): small nearly-sorted batches?**
Insertion sort. It's adaptive: on nearly-sorted data each element is already close to its final position, so few shifts are needed, approaching O(n). It's also stable, in-place, and has very low overhead, which makes it ideal for small batches — exactly why production hybrid sorts fall back to insertion sort for small or nearly-sorted subarrays.

## Common Mistakes

- Assuming all three have the same best case — insertion/bubble can be O(n), selection cannot.
- Confusing stability; claiming selection sort is stable.
- Forgetting the early-exit optimization that gives bubble sort O(n) best case.
- Using O(n²) sorts on large datasets in real code instead of a library sort.
- Off-by-one errors in the inner loop bounds.

## Related Concepts

- [Searching Algorithms](#7-searching-algorithms) — binary search needs sorted data.
- [Complexity Analysis](#3-complexity-analysis) — best/worst case, the n log n bound.
- [Recursion](#8-recursion) — merge sort and quicksort (efficient sorts).

---

# 10. Hash Tables

## What is it?

A **hash table** (hash map, dictionary) stores **key–value pairs** and provides **average O(1)** insert, delete, and lookup. It uses a **hash function** to convert a key into an array index, so you can jump directly to where a value is stored instead of searching. Python's `dict` and `set`, Java's `HashMap`, and C++'s `unordered_map` are all hash tables.

## Why is it needed?

Searching a list for a key is O(n); a sorted structure gives O(log n); but many problems need *instant* lookup by key — checking membership, counting frequencies, caching, deduplication, indexing. Hash tables deliver that: on average, finding a value by its key takes constant time no matter how much data you store. This single property makes them one of the most used data structures in all of software.

## How does it work?

1. A **hash function** maps a key to an integer.
2. That integer is reduced (modulo the array size) to a **bucket index**.
3. The pair is stored in that bucket.

```
key "apple" ──hash──▶ 8493021 ──% 8──▶ bucket 5
key "banana" ─hash──▶ 1200544 ──% 8──▶ bucket 0

Buckets:
0: (banana, ...)
1:
2:
5: (apple, ...)
...
```

When two keys map to the **same bucket**, that's a **collision**, and the table must handle it.

**Collision handling strategies:**
- **Separate chaining:** each bucket holds a linked list (or dynamic array) of entries; collisions are appended. Lookup scans that short list.
  ```
  bucket 3: (cat,1) → (dog,2) → (owl,3)
  ```
- **Open addressing:** on collision, probe for the next empty slot in the array itself (linear probing, quadratic probing, or double hashing). No separate lists; the array holds everything.

## Internal Working

- **Good hash function:** distributes keys uniformly across buckets, is fast to compute, and is deterministic (same key → same hash). Poor hash functions cluster keys, causing many collisions and degrading to O(n).
- **Load factor** = (number of entries) / (number of buckets). As it rises, collisions increase. When it crosses a threshold (e.g., 0.7), the table **resizes** — allocates a larger bucket array and **rehashes** every existing key into the new array. Resizing is O(n) but rare, keeping insert **amortized O(1)**.
- **Worst case O(n):** if all keys collide into one bucket (bad hash or adversarial input), operations degrade to a linear scan. Modern languages randomize hashes to resist deliberate collision attacks.
- **Unordered:** classic hash tables don't maintain key order (Python 3.7+ `dict` preserves *insertion* order as an implementation detail, but not sorted order).

## Advantages

- Average O(1) insert, delete, lookup — extremely fast.
- Flexible keys (strings, tuples, any hashable type).
- Ideal for membership tests, counting, caching, indexing.

## Limitations

- Worst case O(n) under heavy collisions.
- No inherent ordering; can't do efficient range/sorted queries (use a tree instead).
- Extra memory overhead (empty buckets, pointers).
- Requires a good hash function; keys must be hashable (immutable).
- Occasional O(n) resize pauses.

## Real-world Applications

- Language dictionaries/maps and sets.
- **Database indexing** (hash indexes) and in-memory caches (Redis, Memcached).
- Deduplication, frequency counting (word counts), grouping.
- Symbol tables in compilers/interpreters.
- Detecting duplicates and two-sum-style problems in O(n).

## Interview Questions

**Beginner**
- What is a hash function and what is a hash table?
- What is the average time complexity of a dictionary lookup?

**Intermediate**
- What is a collision and how is it handled?
- What is a load factor and why does it trigger resizing?

**Advanced**
- Why is the worst-case complexity of a hash table O(n)?
- Compare separate chaining and open addressing.

**Scenario-based**
- Given an array, find whether any two numbers sum to a target in O(n). How?

**"Why" questions**
- Why must hash-table keys be immutable/hashable?

**Comparison**
- Hash table vs balanced BST for lookups — when would you pick the tree?

## Model Answers

**Q: Hash function and hash table?**
A hash function takes a key and returns an integer (a hash code) deterministically — the same key always yields the same value. A hash table uses that integer, reduced modulo the number of buckets, as an index into an array where the key–value pair is stored. This lets you compute *where* a key lives instead of searching for it, giving average O(1) access. The quality of the hash function is critical: it must spread keys evenly to avoid collisions.

**Q: What is a collision and how is it handled?**
A collision occurs when two different keys hash to the same bucket index — unavoidable because there are more possible keys than buckets. The two main strategies are **separate chaining**, where each bucket holds a small list of entries and colliding keys are appended (lookup scans that short list), and **open addressing**, where a collision triggers probing for the next free slot within the array itself. Both keep average operations O(1) as long as the load factor stays low.

**Q: Why is worst case O(n)?**
Average O(1) assumes keys spread evenly so each bucket holds a constant number of entries. But if the hash function is poor or an adversary crafts keys that all hash to the same bucket, every entry lands in one chain (or one probe sequence), and lookup must scan all n of them — O(n). Real implementations mitigate this by randomizing the hash seed and by resizing to keep chains short, but the theoretical worst case remains linear.

**Q: Load factor and resizing?**
The load factor is the ratio of stored entries to buckets. As it grows, buckets hold more entries on average, so collisions and chain lengths increase, slowing operations. When it exceeds a threshold (commonly around 0.7), the table allocates a bigger bucket array — usually doubling — and rehashes every existing key into the new buckets. This resize is O(n), but because it happens rarely (capacity grows geometrically), insertion stays amortized O(1) while keeping lookups fast.

**Q (Scenario): Two numbers summing to target in O(n)?**
I iterate once, and for each number `x` I check whether `target - x` is already in a hash set of numbers seen so far. If it is, I've found the pair; otherwise I add `x` to the set and continue. Each lookup and insert is O(1) average, so the whole scan is O(n) time and O(n) space. This beats the naive O(n²) double loop by trading memory for speed — a classic hash-table application.

**Q (Comparison): Hash table vs balanced BST?**
I'd pick a hash table when I only need key lookups, inserts, and deletes and don't care about order — it's O(1) average versus the tree's O(log n). I'd pick a balanced BST when I need **ordered** operations: sorted iteration, range queries (all keys between a and b), or finding the nearest/next key. Hash tables can't do those efficiently. So the deciding question is: do I need ordering? If yes, tree; if not, hash table.

## Common Mistakes

- Assuming hash tables are always O(1) — worst case is O(n).
- Using a mutable object as a key (its hash can change, breaking lookups).
- Believing hash tables keep keys sorted (they don't).
- Writing a poor hash function that clusters keys.
- Forgetting to handle collisions when implementing one from scratch.

## Related Concepts

- [Arrays](#4-arrays) — the underlying bucket storage.
- [Linked Lists](#5-linked-lists) — separate chaining.
- [Strings](#6-strings) — string hashing, anagram/frequency problems.
- [Trees & BST](#13-trees--binary-search-trees) — the ordered alternative.

---

# 11. Stacks

## What is it?

A **stack** is a linear data structure that follows **LIFO** — **Last In, First Out**. The last element added is the first removed, like a stack of plates: you add and remove from the **top** only. Its core operations are:
- **push** — add an element to the top.
- **pop** — remove and return the top element.
- **peek/top** — look at the top without removing.
- **isEmpty** — check if empty.

All are O(1).

## Why is it needed?

Many problems require processing items in **reverse order of arrival** or tracking **nested / most-recent** context: undo history, function calls, matching brackets, backtracking. A stack captures exactly this "deal with the most recent thing first" discipline in O(1) operations. It's simple, but it's the backbone of expression evaluation, recursion, and DFS.

## How does it work?

```
push(1), push(2), push(3):

        top ──▶ ┌───┐
                │ 3 │   ← last in
                ├───┤
                │ 2 │
                ├───┤
                │ 1 │   ← first in
                └───┘

pop() → returns 3 (last in, first out)
```

Only the top is accessible. To reach the bottom, you must pop everything above it.

## Internal Working

A stack is an **abstract data type** with two common implementations:
- **Array-backed:** keep a dynamic array and a `top` index. `push` appends (amortized O(1)); `pop` removes the last (O(1)). Cache-friendly; occasional resize. Python's `list` with `append`/`pop` is a stack.
- **Linked-list-backed:** push/pop at the head in O(1); no resizing, but per-node pointer overhead and worse cache locality.

The **call stack** that runs your programs is a literal stack: each function call pushes a frame, each return pops one — which is exactly why recursion depth is bounded and why any recursion can be simulated iteratively with an explicit stack.

## Advantages

- All core operations O(1).
- Simple, predictable, low overhead.
- Perfect model for nested/most-recent-first problems and backtracking.

## Limitations

- Access only at the top — no random access or searching (O(n) if you need it).
- Not suitable when you need FIFO order (use a queue) or arbitrary access.
- Array-backed version may resize; fixed-size stacks can overflow.

## Real-world Applications

- **Function call stack** and recursion.
- **Undo/redo** in editors.
- **Browser back button** history.
- **Expression evaluation** and syntax parsing (balanced brackets, infix→postfix).
- **DFS** (explicit stack) and backtracking algorithms.

## Interview Questions

**Beginner**
- What does LIFO mean? Name the stack operations.
- Give a real-world example modeled by a stack.

**Intermediate**
- How would you check if brackets in an expression are balanced using a stack?
- How can you implement a stack using arrays vs linked lists?

**Advanced**
- Design a stack that returns the minimum element in O(1) (min stack).
- How do you evaluate a postfix expression with a stack?

**Scenario-based**
- You must implement undo in a text editor. Which structure and why?

**"Why" questions**
- Why is recursion tied to the concept of a stack?

**Comparison**
- Stack vs queue — how do their orderings differ and when do you use each?

## Model Answers

**Q: What is LIFO and the operations?**
LIFO means Last In, First Out: the most recently pushed element is the first one popped. The core operations are `push` (add to top), `pop` (remove and return the top), `peek`/`top` (read the top without removing), and `isEmpty`. All are O(1) because they only touch the top. A stack of plates is the classic analogy — you take the top plate first.

**Q: Check balanced brackets with a stack?**
I scan the string left to right. Each time I see an opening bracket (`(`, `[`, `{`) I push it. Each time I see a closing bracket, I check the top of the stack: if it's the matching opening bracket, I pop it; otherwise the string is unbalanced. At the end, the string is balanced only if the stack is empty (every opener was closed). The stack naturally handles nesting because the most recently opened bracket must be the first one closed — exactly LIFO. This is O(n) time and O(n) space.

**Q: Design a min stack with O(1) getMin?**
I keep a second "min stack" alongside the main stack. Whenever I push a value, I also push the current minimum (the smaller of the new value and the current min) onto the min stack. On pop, I pop both. The top of the min stack always holds the minimum of the current contents, so `getMin` is O(1). This trades O(n) extra space for O(1) minimum queries. An optimization stores the min only when it changes.

**Q (Why): Why is recursion tied to a stack?**
Because each recursive call must be remembered until it finishes, and calls finish in reverse order of how they started — the most recent call returns first. That's LIFO, so the runtime uses a call stack: each call pushes a frame with its locals and return address, and returning pops it. This is why recursion depth is limited by stack size and why you can always rewrite recursion iteratively using your own explicit stack to hold the pending work.

**Q (Comparison): Stack vs queue?**
A stack is LIFO — the last element in is the first out, with access only at the top — ideal for backtracking, undo, and DFS. A queue is FIFO — the first element in is the first out, with insertion at the rear and removal from the front — ideal for fair, in-order processing like task scheduling and BFS. The choice depends on whether you want to process the most recent item first (stack) or the oldest item first (queue).

## Common Mistakes

- Popping from an empty stack (underflow) — always check `isEmpty`.
- Confusing LIFO with FIFO.
- Forgetting the final "stack must be empty" check in bracket matching.
- Using a stack where FIFO order is actually required.

## Related Concepts

- [Queues](#12-queues) — the FIFO counterpart.
- [Recursion](#8-recursion) — powered by the call stack.
- [Graphs](#14-graphs-bfs--dfs) — DFS uses a stack.
- [Arrays](#4-arrays) / [Linked Lists](#5-linked-lists) — implementations.

---

# 12. Queues

## What is it?

A **queue** is a linear data structure that follows **FIFO** — **First In, First Out**. The first element added is the first removed, like a line of people: you join at the **rear** and are served from the **front**. Core operations (all O(1)):
- **enqueue** — add to the rear.
- **dequeue** — remove and return from the front.
- **peek/front** — view the front without removing.
- **isEmpty** — check if empty.

Variants: **circular queue** (reuses freed slots in a fixed array), **deque** (double-ended, insert/remove at both ends), and **priority queue** (serves highest priority first, usually a heap).

## Why is it needed?

Many systems must process items **in the order they arrive** — fairness and ordering matter. Print jobs, web requests, messages, and BFS all need "first come, first served." A queue enforces this discipline with O(1) enqueue/dequeue. Without it, you'd either lose order or pay O(n) to remove from the front of a plain array.

## How does it work?

```
enqueue(1), enqueue(2), enqueue(3):

 front                     rear
   │                        │
   ▼                        ▼
 ┌───┐   ┌───┐   ┌───┐
 │ 1 │──▶│ 2 │──▶│ 3 │
 └───┘   └───┘   └───┘

dequeue() → returns 1 (first in, first out)
```

You add at the rear and remove from the front — opposite ends, unlike a stack.

## Internal Working

- **Naive array queue:** enqueue appends (O(1)), but dequeue removes index 0, forcing all elements to shift left — O(n). This is why a plain array is a poor queue.
- **Circular queue (ring buffer):** use a fixed array with `front` and `rear` indices that wrap around with modulo. Both operations are O(1) and no shifting occurs; memory is reused. The trade-off is a fixed capacity (or resize logic) and tracking full vs empty.
  ```
  capacity 5, indices wrap: rear = (rear + 1) % 5
  ```
- **Linked-list queue:** keep `head` (front) and `tail` (rear) pointers; enqueue at tail, dequeue at head, both O(1), grows dynamically.
- **Python:** use `collections.deque` (O(1) both ends) — **not** `list.pop(0)`, which is O(n).
- **Priority queue:** implemented with a binary **heap**, giving O(log n) insert and O(log n) remove-min, not FIFO but priority order.

## Advantages

- O(1) enqueue and dequeue with a proper implementation.
- Preserves arrival order (fairness).
- Natural fit for scheduling, buffering, and BFS.

## Limitations

- No random access or search (O(n)).
- Naive array implementation is O(n) to dequeue.
- Fixed-size circular queues can overflow; must track full/empty carefully.

## Real-world Applications

- **CPU / task scheduling**, print spoolers, request queues.
- **Message queues** (Kafka, RabbitMQ) and buffering (streaming, IO).
- **BFS** graph traversal and shortest path in unweighted graphs.
- **Producer–consumer** pipelines.
- **Priority queues** for Dijkstra, event simulation, and job priorities.

## Interview Questions

**Beginner**
- What does FIFO mean? Name the queue operations.
- Give a real-world example of a queue.

**Intermediate**
- Why is dequeuing from a plain array O(n), and how does a circular queue fix it?
- How do you implement a queue using two stacks?

**Advanced**
- What is a priority queue and how is it implemented?
- How does a circular queue distinguish full from empty?

**Scenario-based**
- You're building a print spooler serving jobs fairly. Which structure and why?

**"Why" questions**
- Why is a queue the right structure for BFS but not DFS?

**Comparison**
- Queue vs deque vs priority queue — how do they differ?

## Model Answers

**Q: What is FIFO and the operations?**
FIFO means First In, First Out: elements leave in the same order they arrived. The operations are `enqueue` (add to the rear), `dequeue` (remove from the front), `peek`/`front` (read the front), and `isEmpty`. A line at a checkout is the analogy — the first person in line is served first. With a proper implementation, all operations are O(1).

**Q: Why is array dequeue O(n) and how does a circular queue fix it?**
Removing from the front of a plain array (index 0) leaves a gap, so every remaining element must shift one position left to keep the array contiguous — that's O(n). A circular queue avoids shifting by keeping two indices, `front` and `rear`, that advance with modulo arithmetic and wrap around the fixed array. Dequeue just moves `front` forward; the vacated slot is reused later when `rear` wraps around. Both enqueue and dequeue become O(1) with no data movement.

**Q: Implement a queue with two stacks?**
I use an `in` stack and an `out` stack. Enqueue always pushes onto `in`. Dequeue pops from `out`; if `out` is empty, I first pour everything from `in` into `out`, which reverses the order so the oldest element ends up on top. Each element is moved between stacks at most once, so dequeue is amortized O(1) even though a single transfer is O(n). This cleverly turns two LIFO structures into one FIFO structure.

**Q: What is a priority queue?**
A priority queue serves elements by **priority** rather than arrival order — the highest (or lowest) priority element is dequeued first. It's typically implemented with a binary heap, giving O(log n) insertion and O(log n) removal of the top-priority element, with O(1) peek. It's used in Dijkstra's shortest path, event-driven simulations, and job scheduling where some tasks must jump the line. It is not FIFO; ties may or may not preserve arrival order depending on implementation.

**Q (Why): Why a queue for BFS but not DFS?**
BFS explores a graph level by level — it must visit all nodes at distance 1 before distance 2, and so on. A queue's FIFO order naturally delivers nodes in the order they were discovered, producing this level-by-level expansion. DFS instead dives as deep as possible before backtracking, which requires processing the most recently discovered node first — LIFO — so DFS uses a stack (or recursion). The traversal order you want dictates the structure: FIFO→BFS, LIFO→DFS.

## Common Mistakes

- Using `list.pop(0)` in Python (O(n)) instead of `collections.deque` (O(1)).
- Dequeuing from an empty queue (underflow).
- In a circular queue, confusing the full and empty conditions.
- Confusing FIFO (queue) with LIFO (stack).
- Assuming a priority queue is FIFO — it orders by priority.

## Related Concepts

- [Stacks](#11-stacks) — the LIFO counterpart; two stacks can build a queue.
- [Graphs](#14-graphs-bfs--dfs) — BFS uses a queue.
- [Trees & BST](#13-trees--binary-search-trees) — level-order traversal uses a queue; heaps back priority queues.
- [Arrays](#4-arrays) / [Linked Lists](#5-linked-lists) — implementations.

---

# 13. Trees & Binary Search Trees

## What is it?

A **tree** is a hierarchical (non-linear) data structure of **nodes** connected by edges, with one **root** at the top and no cycles. Each node has a value and links to **child** nodes. Key terms: **root** (top), **leaf** (no children), **parent/child**, **subtree**, **height** (longest root-to-leaf path), and **depth** (distance from root).

- **Binary Tree:** each node has at most **two** children (left and right).
- **Binary Search Tree (BST):** a binary tree with an **ordering invariant** — for every node, all values in its **left subtree are smaller** and all values in its **right subtree are larger**. This ordering enables O(log n) search, insert, and delete *when balanced*.

## Why is it needed?

Linear structures (arrays, lists) force a trade-off: fast search *or* fast insert, rarely both. A **balanced BST** gives you **both** in O(log n), plus **sorted order** for free (in-order traversal yields sorted output) and efficient **range queries** and **nearest-value** lookups — things hash tables can't do. Trees also naturally model hierarchy: file systems, org charts, HTML/DOM, decision trees. They're the backbone of databases (B-trees) and many algorithms.

## How does it work?

```
        50            ← root
       /  \
     30    70
    /  \   / \
   20  40 60  80      ← leaves at bottom
```
This is a BST: everything left of 50 is < 50, everything right is > 50, recursively.

**Search for 60:** start at root 50 → 60 > 50 go right → at 70, 60 < 70 go left → at 60, found. Each step discards a subtree → O(height).

**Insert 45:** search for 45's place → 45 < 50 left → 45 > 30 right → 45 > 40 right → empty, insert as right child of 40.

**Traversals** (ways to visit all nodes):
- **In-order (Left, Root, Right):** visits BST values in **sorted order** → `20 30 40 50 60 70 80`.
- **Pre-order (Root, Left, Right):** used to copy/serialize a tree → `50 30 20 40 70 60 80`.
- **Post-order (Left, Right, Root):** used to delete/free a tree → `20 40 30 60 80 70 50`.
- **Level-order (BFS):** visit level by level using a queue → `50 30 70 20 40 60 80`.

## Internal Working

- A BST node stores a value plus `left` and `right` pointers (like a doubly branching linked list).
- **Search/insert** walk down comparing values, costing **O(height)**. In a **balanced** tree height is O(log n); in a **skewed** tree (inserting sorted data) it degrades to a linked list with height O(n).
- **Deletion** has three cases:
  1. **Leaf:** just remove it.
  2. **One child:** replace the node with its child.
  3. **Two children:** replace the node's value with its **in-order successor** (smallest value in the right subtree) or in-order predecessor, then delete that successor.
- **Self-balancing BSTs** (AVL, Red-Black trees) perform **rotations** on insert/delete to keep height O(log n), guaranteeing worst-case O(log n). Databases use **B-trees/B+ trees** (many children per node) to minimize disk reads.
- **Validate a BST:** an in-order traversal must produce strictly increasing values; equivalently, recursively check each node lies within a valid (min, max) range that tightens as you descend. A common bug is checking only immediate children instead of the whole subtree range.
- **Closest value in a BST:** walk from the root toward the target, tracking the closest value seen; go left/right based on comparison — O(height).

## Advantages

- Balanced BST: O(log n) search, insert, delete.
- In-order traversal gives sorted data for free.
- Supports range queries, min/max, successor/predecessor, nearest value.
- Models hierarchy naturally.

## Limitations

- **Unbalanced/skewed** BSTs degrade to O(n) — self-balancing needed for guarantees.
- More complex than arrays/lists; pointer-based, poorer cache locality.
- Deletion (two-children case) is fiddly.
- Plain BSTs don't self-balance; sorted insertions are the worst case.

## Real-world Applications

- **Database indexes** (B-trees / B+ trees) for fast lookups and range scans.
- **File systems** and hierarchical data (directories, DOM, org charts).
- **Sorted maps/sets** (`TreeMap`, `std::map`) with ordered iteration.
- **Autocomplete / routing** (tries are specialized trees).
- **Expression trees**, decision trees, Huffman coding.

## Interview Questions

**Beginner**
- What is a binary tree vs a binary search tree?
- What are the leaf, root, height, and depth of a tree?

**Intermediate**
- What are the three DFS traversals, and what does in-order give for a BST?
- What is the time complexity of BST search and when does it degrade?

**Advanced**
- How do you delete a node with two children from a BST?
- How do you validate whether a binary tree is a BST?
- What is a balanced tree and why do AVL/Red-Black trees exist?

**Scenario-based**
- You insert already-sorted data into a BST and search becomes slow. Why, and how do you fix it?

**"Why" questions**
- Why does an in-order traversal of a BST produce sorted output?

**Comparison**
- BST vs hash table — when do you choose each?

## Model Answers

**Q: Binary tree vs BST?**
A binary tree is any tree where each node has at most two children, with no ordering rule. A binary search tree adds an **ordering invariant**: for every node, all values in its left subtree are smaller and all in its right subtree are larger. That invariant is what makes search efficient — at each node you can discard an entire subtree — giving O(log n) operations when the tree is balanced. Without the invariant, you'd have to check every node (O(n)).

**Q: The three DFS traversals; what does in-order give?**
Pre-order visits Root, then Left, then Right — useful for copying or serializing a tree. In-order visits Left, Root, Right — for a BST this yields the values in **sorted ascending order**. Post-order visits Left, Right, then Root — useful for deleting or evaluating expression trees, since children are processed before their parent. All three are O(n) and are naturally written recursively (or iteratively with an explicit stack).

**Q: Delete a node with two children?**
When a node has two children, I can't simply remove it without breaking the tree. Instead I find its **in-order successor** — the smallest value in its right subtree (go right once, then left as far as possible). I copy that successor's value into the node being "deleted," then delete the successor from the right subtree. The successor has at most one child (no left child by definition), so its deletion reduces to the easy leaf or one-child case. This preserves the BST ordering invariant. The in-order predecessor works symmetrically.

**Q: Validate a BST?**
The common mistake is checking only that each node's left child is smaller and right child is larger — that misses violations deeper in the tree. The correct approach passes down a valid **(min, max) range**: the root may be any value, its left subtree must lie in (−∞, root), its right subtree in (root, +∞), and these bounds tighten as you descend. If any node falls outside its allowed range, it's not a BST. Equivalently, perform an in-order traversal and verify the output is strictly increasing. Both are O(n).

**Q (Scenario): Sorted input makes BST slow?**
Inserting already-sorted data into a plain BST creates a **skewed** tree — each new value is larger than all previous, so it always attaches to the right, producing essentially a linked list of height n. Search then degrades from O(log n) to O(n). The fix is a **self-balancing BST** (AVL or Red-Black tree), which performs rotations during insertion to keep the height O(log n) regardless of input order, restoring guaranteed O(log n) operations. Alternatively, insert in randomized order or use a balanced structure from the start.

**Q (Why): Why does in-order traversal give sorted output?**
In-order traversal visits the left subtree, then the node, then the right subtree. By the BST invariant, everything in the left subtree is smaller than the node and everything in the right subtree is larger. So recursively, all smaller values are emitted before the node and all larger values after it — at every level. Applied throughout the tree, this produces values in strictly ascending order.

**Q (Comparison): BST vs hash table?**
A hash table gives average O(1) lookup but no ordering. A balanced BST gives O(log n) lookup but maintains **sorted order**, enabling range queries, ordered iteration, and nearest/successor lookups that a hash table can't do efficiently. So I choose a hash table for pure key lookups where order doesn't matter, and a BST (or ordered map) when I need sorted traversal, ranges, or min/max/successor operations.

## Common Mistakes

- Validating a BST by comparing only parent–child, ignoring subtree ranges.
- Forgetting the two-children deletion case (successor replacement).
- Assuming BSTs are always O(log n) — skewed trees are O(n).
- Confusing height (edges to deepest leaf) with depth (distance from root).
- Mixing up the traversal orders.

## Related Concepts

- [Recursion](#8-recursion) — traversals and most tree operations.
- [Queues](#12-queues) — level-order (BFS) traversal.
- [Stacks](#11-stacks) — iterative DFS traversal.
- [Hash Tables](#10-hash-tables) — the unordered alternative.
- [Graphs](#14-graphs-bfs--dfs) — a tree is a connected acyclic graph.

---

# 14. Graphs (BFS & DFS)

## What is it?

A **graph** is a collection of **vertices (nodes)** connected by **edges**. Unlike a tree, a graph can have **cycles**, disconnected parts, and any connection pattern. Graphs model relationships. Key types:
- **Directed vs Undirected:** edges have direction (Twitter follows) or not (Facebook friends).
- **Weighted vs Unweighted:** edges carry a cost/distance or not.
- **Cyclic vs Acyclic:** contains cycles or not (a DAG is a directed acyclic graph).

The two fundamental traversals are:
- **BFS (Breadth-First Search):** explore level by level, nearest first, using a **queue**.
- **DFS (Depth-First Search):** explore as deep as possible before backtracking, using a **stack** or recursion.

## Why is it needed?

Enormous classes of problems are naturally relationships between entities: social networks, maps and routing, web-page links, dependencies between tasks, network topology. Graphs are the model, and BFS/DFS are the workhorses for exploring them — finding connectivity, shortest paths (unweighted), cycles, reachability, ordering dependencies, and components. Almost any "who is connected to whom / how do I get from A to B" problem is a graph problem.

## How does it work?

**Representations:**
- **Adjacency List:** each vertex stores a list of its neighbors. Space O(V + E). Efficient for **sparse** graphs (most real graphs). Preferred default.
  ```
  A: [B, C]
  B: [A, D]
  C: [A, D]
  D: [B, C]
  ```
- **Adjacency Matrix:** a V×V grid where cell (i, j) is 1 if an edge exists. Space O(V²). O(1) edge lookup but wasteful for sparse graphs; good for **dense** graphs.
  ```
     A B C D
  A [0 1 1 0]
  B [1 0 0 1]
  C [1 0 0 1]
  D [0 1 1 0]
  ```

**BFS** from A (uses a queue, visits nearest first):
```
Queue: [A] → visit A, enqueue B,C
Queue: [B,C] → visit B, enqueue D
Queue: [C,D] → visit C
Queue: [D] → visit D
Order: A B C D   (level by level)
```

**DFS** from A (uses a stack/recursion, goes deep):
```
Visit A → go to B → go to D → go to C → back
Order: A B D C   (one deep path, then backtrack)
```

Both must track **visited** nodes to avoid infinite loops on cycles.

## Internal Working

- **BFS** maintains a **queue** and a **visited** set. Dequeue a node, visit it, enqueue all unvisited neighbors (marking them visited). Because it expands in order of distance, BFS finds the **shortest path in an unweighted graph**. Time O(V + E), space O(V).
- **DFS** uses **recursion** (implicit call stack) or an explicit **stack**. Visit a node, then recurse into an unvisited neighbor, going deep before backtracking. Time O(V + E), space O(V) (stack + visited). DFS is natural for **cycle detection, topological sort, connected components, and path finding**.
- The **visited set is essential** — without it, cycles cause infinite loops. (Trees don't need it because they have no cycles.)
- **Complexity** is O(V + E) with an adjacency list because each vertex and each edge is processed once; with an adjacency matrix it's O(V²) because scanning neighbors costs O(V) per vertex.

## Advantages

- Model almost any relationship or network.
- BFS gives shortest paths in unweighted graphs; DFS gives cycle detection, topological ordering, components.
- Both traverse in O(V + E) with adjacency lists — linear in the graph size.
- Flexible representations tuned to sparse or dense graphs.

## Limitations

- Can be memory-heavy (BFS queue can hold a whole level; matrices are O(V²)).
- DFS recursion can overflow the stack on deep/large graphs (use iterative DFS).
- Plain BFS/DFS don't handle **weighted** shortest paths — need Dijkstra/Bellman-Ford.
- Must manage visited state carefully to avoid infinite loops.

## Real-world Applications

- **Social networks:** friend suggestions, degrees of separation (BFS).
- **Maps / GPS routing:** shortest paths (BFS for unweighted, Dijkstra for weighted).
- **Web crawling** and PageRank (the web is a graph).
- **Dependency resolution / build systems / task scheduling** (topological sort via DFS).
- **Network broadcasting, garbage collection (reachability), recommendation engines, maze solving.**

## Interview Questions

**Beginner**
- What is a graph? How does it differ from a tree?
- What data structures do BFS and DFS use?

**Intermediate**
- Compare adjacency list vs adjacency matrix.
- Why do graph traversals need a visited set but tree traversals don't?

**Advanced**
- How do you detect a cycle in a directed graph?
- How does BFS find the shortest path in an unweighted graph?
- Implement DFS iteratively without recursion.

**Scenario-based**
- You need the fewest hops between two users in a social network. BFS or DFS? Why?
- You must order build tasks so dependencies come first. Which algorithm?

**"Why" questions**
- Why is BFS O(V + E) with an adjacency list but O(V²) with a matrix?

**Comparison**
- BFS vs DFS — traversal order, data structure, and typical uses.

## Model Answers

**Q: Graph vs tree; what do BFS/DFS use?**
A graph is a set of vertices connected by edges, allowing cycles, disconnected components, and arbitrary connections; edges may be directed and weighted. A tree is a special graph that is connected, acyclic, and has exactly one path between any two nodes (n−1 edges for n nodes). BFS uses a **queue** to explore level by level (nearest first); DFS uses a **stack** or recursion to explore as deep as possible before backtracking. Both need a visited set on graphs to handle cycles.

**Q: Adjacency list vs matrix?**
An adjacency list stores, for each vertex, a list of its neighbors, using O(V + E) space — efficient for sparse graphs (most real-world graphs) and iterating a vertex's neighbors is fast. An adjacency matrix is a V×V grid marking which pairs are connected, using O(V²) space; it gives O(1) edge-existence checks but wastes memory on sparse graphs and makes neighbor iteration O(V). I default to adjacency lists unless the graph is dense or I need constant-time edge lookups.

**Q: Why do graphs need a visited set but trees don't?**
Trees are acyclic with a single path between nodes, so a traversal never revisits a node — there's no way to loop back. Graphs can contain **cycles**, so following edges could return to an already-visited node and loop forever. The visited set records which nodes have been processed so the traversal skips them, guaranteeing each node is handled once and the algorithm terminates. It's also what keeps complexity at O(V + E).

**Q: How does BFS find the shortest path in an unweighted graph?**
BFS explores nodes in increasing order of distance from the source: it visits all nodes one edge away, then two edges away, and so on, because the FIFO queue always processes closer nodes before farther ones. The first time BFS reaches a node, it has arrived via the fewest possible edges, so recording each node's discovering parent lets you reconstruct a shortest path. This only holds for **unweighted** graphs (or equal weights); with varying weights you need Dijkstra, because fewer edges may not mean lower cost.

**Q: Detect a cycle in a directed graph?**
I run DFS while tracking each node's state in three colors: unvisited, "in the current recursion stack" (being explored), and fully finished. If during DFS I encounter an edge to a node that's currently *in the recursion stack*, that's a **back edge**, which means a cycle. Nodes that are merely finished don't indicate a cycle. This is O(V + E). For undirected graphs the rule differs slightly — a cycle exists if DFS reaches an already-visited node that isn't the immediate parent.

**Q (Scenario): Fewest hops between two users?**
BFS. "Fewest hops" is the shortest path in an unweighted graph, and BFS explores outward level by level, so the first time it reaches the target it has used the minimum number of edges. DFS could find *a* path but not necessarily the shortest, and might wander deep in the wrong direction first. So BFS from the source, stopping when the target is dequeued, gives the minimum number of connections.

**Q (Scenario): Order build tasks by dependency?**
This is a **topological sort** on a directed acyclic graph where an edge A→B means A must come before B. I can do it with DFS: run DFS, and after fully exploring a node (post-order), push it onto a stack; the reversed finish order is a valid topological ordering. Alternatively, Kahn's algorithm repeatedly removes nodes with no remaining incoming edges using a queue. Both are O(V + E). If the graph has a cycle, no valid ordering exists (a dependency deadlock).

**Q (Why): BFS O(V + E) vs O(V²)?**
With an adjacency list, BFS visits each vertex once (O(V)) and examines each edge once when scanning neighbors (O(E)), totaling O(V + E) — linear in the graph. With an adjacency matrix, finding a vertex's neighbors means scanning an entire row of length V, and doing that for all V vertices costs O(V²) regardless of how few edges exist. So the representation, not the algorithm, drives the difference; adjacency lists are faster for sparse graphs.

## Common Mistakes

- Forgetting the visited set → infinite loops on cyclic graphs.
- Using DFS for shortest path in an unweighted graph (BFS is correct).
- Using plain BFS/DFS for **weighted** shortest paths (need Dijkstra).
- Marking nodes visited at the wrong time in BFS (enqueue-time vs dequeue-time), causing duplicates.
- Deep recursive DFS overflowing the stack — use an explicit stack for large graphs.
- Choosing an adjacency matrix for a large sparse graph (O(V²) memory).

## Related Concepts

- [Queues](#12-queues) — BFS engine.
- [Stacks](#11-stacks) / [Recursion](#8-recursion) — DFS engine.
- [Trees & BST](#13-trees--binary-search-trees) — trees are acyclic connected graphs.
- [Hash Tables](#10-hash-tables) — visited sets and adjacency lookups.
- Dijkstra, Bellman-Ford, topological sort — beyond-syllabus next steps.

---

## Final Revision Checklist

- **Complexity table** (Topic 3) — be able to recite access/search/insert/delete for every structure.
- **Amortized O(1)** — dynamic array append and hash table insert; know *why*.
- **When to use what:** array (random access), linked list (end/middle edits), hash table (O(1) lookup), BST (ordered + range), stack (LIFO/backtracking), queue (FIFO/BFS), graph (relationships).
- **LIFO vs FIFO** and their traversals: stack→DFS, queue→BFS.
- **Recursion** needs a base case; costs O(depth) stack space.
- **Binary search** needs sorted data and random access.
- **BST** degrades to O(n) when skewed → self-balancing trees.
- **Graphs** need a visited set; BFS for shortest unweighted path.

> Practice implementing each structure from scratch (see `practical.md`). Being able to *code* them under time pressure is what separates a pass from a distinction.

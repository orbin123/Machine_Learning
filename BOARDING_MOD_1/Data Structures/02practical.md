# Data Structures & Algorithms — Practical & Coding Assessment Guide

> This is the *coding* companion to `theory.md`. It predicts the practical
> questions you are likely to face in coding rounds, lab exams, and viva
> practicals, and gives **production-quality Python solutions** with complexity
> analysis, alternatives, variations, and follow-ups. Type every solution out
> by hand at least once — reading is not the same as being able to produce it
> under a timer.

## How to use this guide

- For each problem: read the **Problem Statement**, attempt it *yourself first*, then compare with the solution.
- Study the **Approach** before the code — interviewers care more about your reasoning than syntax.
- Always state **time and space complexity** out loud; it's often an explicit rubric item.
- Rehearse the **Follow-up Questions** — they are where interviews actually get decided.

## Contents

- [Section A — Arrays](#section-a--arrays)
- [Section B — Linked Lists](#section-b--linked-lists)
- [Section C — Strings](#section-c--strings)
- [Section D — Searching](#section-d--searching)
- [Section E — Recursion](#section-e--recursion)
- [Section F — Sorting](#section-f--sorting)
- [Section G — Hash Tables](#section-g--hash-tables)
- [Section H — Stacks](#section-h--stacks)
- [Section I — Queues](#section-i--queues)
- [Section J — Trees & BST](#section-j--trees--bst)
- [Section K — Graphs (BFS & DFS)](#section-k--graphs-bfs--dfs)
- [Coding Questions Bank (Easy / Medium / Hard)](#coding-questions-bank)
- [Exam & Viva Survival Tips](#exam--viva-survival-tips)

> **Environment note:** All solutions are plain Python 3 and run in a script or a
> Jupyter cell. Where a topic is naturally notebook-friendly (e.g., benchmarking
> sorts, visualizing complexity), a **notebook cell workflow** is provided.

---

# Section A — Arrays

## Practical Question 1

**Difficulty:** Easy
**Estimated Time:** 10 minutes
**Concepts Tested:** array traversal, hash set, single-pass optimization

**Problem Statement**
Given an array of integers `nums` and an integer `target`, return the indices of the two numbers that add up to `target`. Assume exactly one solution exists and you may not use the same element twice.

**Example Input**
```
nums = [2, 7, 11, 15], target = 9
```

**Example Output**
```
[0, 1]        # because nums[0] + nums[1] = 2 + 7 = 9
```

**Approach (step-by-step)**
1. The brute force checks every pair — O(n²). We can do better.
2. As we scan, for each number `x` the partner we need is `target - x`.
3. Keep a hash map of `value → index` for numbers already seen.
4. For each `x`, check if `target - x` is already in the map. If yes, we found the pair. If not, store `x` and continue.
5. One pass, O(1) average lookups → O(n) total.

### Python Implementation

```python
def two_sum(nums: list[int], target: int) -> list[int]:
    """Return indices of the two numbers summing to target (assumes one exists)."""
    seen: dict[int, int] = {}          # value -> index of numbers seen so far
    for i, x in enumerate(nums):       # single pass over the array
        complement = target - x        # the partner we need for x
        if complement in seen:         # O(1) average membership test
            return [seen[complement], i]
        seen[x] = i                    # remember x for future complements
    return []                          # no pair (won't happen per constraints)
```

**Line notes**
- `seen` trades O(n) memory for O(1) lookups — the core speed/space trade-off.
- We check the complement *before* inserting `x`, which prevents using the same element twice.

**Complexity**
- **Time:** O(n) — each element visited once, O(1) average hash operations.
- **Space:** O(n) — the hash map may hold up to n entries.

### Alternative Solution

Brute-force double loop (no extra memory):
```python
def two_sum_brute(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
```
O(n²) time, O(1) space. Acceptable only for tiny inputs; the hash approach is strictly better for scale.

If the array were **sorted**, a **two-pointer** approach gives O(n) time and O(1) space (see Question 3).

### Interview Variations

- Return the **values** instead of indices.
- Count **how many pairs** sum to target (watch for duplicates).
- **Three-sum:** find triplets summing to zero (sort + two pointers → O(n²)).
- Numbers can be **reused** (return all valid pairs).

### Common Follow-up Questions

- *What if there are multiple valid answers?* Clarify whether to return any one or all; the hash approach returns the first found.
- *What if no solution exists?* Return an empty list or sentinel; don't assume.
- *Can you do it without extra space?* Only if sorting is allowed (two pointers), which costs O(n log n) and changes indices.

---

## Practical Question 2

**Difficulty:** Easy–Medium
**Estimated Time:** 15 minutes
**Concepts Tested:** in-place manipulation, reversal trick, O(1) space

**Problem Statement**
Rotate an array to the right by `k` steps, **in place**, using O(1) extra space.

**Example Input**
```
nums = [1, 2, 3, 4, 5, 6, 7], k = 3
```

**Example Output**
```
[5, 6, 7, 1, 2, 3, 4]
```

**Approach (step-by-step)**
1. Rotating right by `k` moves the last `k` elements to the front.
2. Naive: use an extra array — O(n) space. We want O(1).
3. **Reversal trick:** reverse the whole array, then reverse the first `k`, then reverse the rest.
4. Reduce `k` modulo `n` first (rotating by `n` does nothing).

```
Original:            [1 2 3 4 5 6 7], k=3
Reverse all:         [7 6 5 4 3 2 1]
Reverse first k=3:   [5 6 7 4 3 2 1]
Reverse last n-k:    [5 6 7 1 2 3 4]  ✓
```

### Python Implementation

```python
def rotate(nums: list[int], k: int) -> None:
    """Rotate nums right by k steps in place (modifies nums)."""
    n = len(nums)
    k %= n                              # k may exceed n; only remainder matters
    if k == 0:
        return

    def reverse(lo: int, hi: int) -> None:
        while lo < hi:                  # swap ends moving inward
            nums[lo], nums[hi] = nums[hi], nums[lo]
            lo += 1
            hi -= 1

    reverse(0, n - 1)                   # reverse entire array
    reverse(0, k - 1)                   # reverse first k
    reverse(k, n - 1)                   # reverse remaining n-k
```

**Complexity**
- **Time:** O(n) — three reversals, each linear.
- **Space:** O(1) — swaps happen in place.

### Alternative Solution

Slice-based (Pythonic, but O(n) extra space):
```python
def rotate_slice(nums, k):
    k %= len(nums)
    nums[:] = nums[-k:] + nums[:-k]     # note nums[:] to modify in place
```
Clean and correct, but allocates a new list. Use it when readability matters more than memory.

### Interview Variations

- Rotate **left** by `k` (reverse the segments in the opposite order, or rotate right by `n-k`).
- Rotate a **linked list** by k.
- Rotate a 2D matrix by 90° (a related in-place reversal/transpose problem).

### Common Follow-up Questions

- *Why take `k % n`?* Rotating by a full length returns the original; only the remainder changes the array, and it prevents index errors when `k > n`.
- *Can you do it with cyclic replacements instead?* Yes — a juggling algorithm moves each element directly to its target in one pass, also O(n)/O(1), but it's trickier to get right.

---

## Practical Question 3

**Difficulty:** Medium
**Estimated Time:** 15 minutes
**Concepts Tested:** two pointers, sorted-array invariants

**Problem Statement**
Given a **sorted** array, find two numbers that add up to a target and return their (1-indexed) positions, using O(1) extra space.

**Example Input**
```
numbers = [2, 7, 11, 15], target = 9
```

**Example Output**
```
[1, 2]
```

**Approach (step-by-step)**
1. Because the array is sorted, use two pointers: `left` at the start, `right` at the end.
2. Compute the sum. If it equals target → done.
3. If the sum is **too small**, move `left` right (increase the sum).
4. If **too large**, move `right` left (decrease the sum).
5. This converges in one pass — O(n) time, O(1) space.

### Python Implementation

```python
def two_sum_sorted(numbers: list[int], target: int) -> list[int]:
    """Two-pointer search on a sorted array; returns 1-indexed positions."""
    left, right = 0, len(numbers) - 1
    while left < right:
        current = numbers[left] + numbers[right]
        if current == target:
            return [left + 1, right + 1]   # 1-indexed as required
        elif current < target:
            left += 1                      # need a bigger sum
        else:
            right -= 1                     # need a smaller sum
    return []
```

**Complexity**
- **Time:** O(n) — each pointer moves at most n times total.
- **Space:** O(1) — no extra structures.

### Alternative Solution

The hash-map approach from Question 1 also works (O(n) time) but uses O(n) space and ignores the sorted property. When the array is already sorted, two pointers are preferred for O(1) space.

### Interview Variations

- **Three-sum / four-sum** (fix one element, two-pointer the rest).
- Find a pair with a given **difference**.
- Count pairs with sum **less than** target.

### Common Follow-up Questions

- *Why does moving a pointer never skip the answer?* Because the array is sorted, if the sum is too small the only way to increase it is to raise the left value; the current right can't pair with anything smaller than left to reach target, so discarding it is safe.
- *What if the array isn't sorted?* Either sort first (O(n log n)) or use the hash-map version.

---

# Section B — Linked Lists

> The syllabus lists many linked-list tasks. We build a reusable node/list
> foundation, then solve each required operation. **Memorize the singly linked
> list scaffold** — it appears in almost every linked-list question.

## Foundation: Node and Singly Linked List

```python
class Node:
    """A single node of a singly linked list."""
    def __init__(self, data):
        self.data = data
        self.next = None


class SinglyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None          # keep tail for O(1) append

    def append(self, data):
        """Add a node at the END. O(1) with a tail pointer."""
        node = Node(data)
        if self.head is None:      # empty list
            self.head = self.tail = node
            return
        self.tail.next = node
        self.tail = node

    def prepend(self, data):
        """Add a node at the BEGINNING. O(1)."""
        node = Node(data)
        node.next = self.head
        self.head = node
        if self.tail is None:      # list was empty
            self.tail = node

    def print_forward(self):
        """Print elements in forward order. O(n)."""
        values = []
        current = self.head
        while current:
            values.append(str(current.data))
            current = current.next
        print(" -> ".join(values) if values else "empty")
```

## Practical Question 1

**Difficulty:** Easy
**Estimated Time:** 10 minutes
**Concepts Tested:** array-to-list construction, appending

**Problem Statement**
Convert an array into a singly linked list, preserving order.

**Example Input**
```
[10, 20, 30]
```

**Example Output**
```
10 -> 20 -> 30
```

**Approach**
Iterate the array, appending each element as a new node. Using a tail pointer makes each append O(1), so the whole conversion is O(n).

### Python Implementation

```python
def array_to_linked_list(arr: list) -> SinglyLinkedList:
    """Build a singly linked list from an array in O(n)."""
    ll = SinglyLinkedList()
    for value in arr:              # order preserved by appending at the tail
        ll.append(value)
    return ll
```

**Complexity:** Time O(n), Space O(n) (n nodes).

### Common Follow-up Questions

- *Without a tail pointer, what's the cost?* Each append walks to the end → O(n) per append → O(n²) total. The tail pointer is what keeps it O(n).

---

## Practical Question 2

**Difficulty:** Medium
**Estimated Time:** 15 minutes
**Concepts Tested:** deletion by value, predecessor tracking, edge cases

**Problem Statement**
Delete the first node containing a specified value `x` from a singly linked list.

**Example Input**
```
List: 10 -> 20 -> 30 -> 40,  x = 30
```

**Example Output**
```
10 -> 20 -> 40
```

**Approach (step-by-step)**
1. Handle the empty-list case.
2. If the **head** holds `x`, move `head` forward and return.
3. Otherwise walk with a `prev` pointer until `current.data == x`.
4. Unlink by setting `prev.next = current.next`.
5. Fix the `tail` if we deleted the last node.

### Python Implementation

```python
def delete_value(ll: SinglyLinkedList, x) -> bool:
    """Delete the first node equal to x. Returns True if deleted."""
    if ll.head is None:                    # empty list
        return False

    if ll.head.data == x:                  # deleting the head
        ll.head = ll.head.next
        if ll.head is None:                # list became empty
            ll.tail = None
        return True

    prev, current = ll.head, ll.head.next
    while current:
        if current.data == x:
            prev.next = current.next       # unlink current
            if current is ll.tail:         # deleted the tail
                ll.tail = prev
            return True
        prev, current = current, current.next
    return False                           # value not found
```

**Complexity:** Time O(n) (search), Space O(1).

### Interview Variations

- Delete **all** nodes equal to `x` (keep scanning, don't return early).
- Delete the **n-th node from the end** (two-pointer gap technique).
- **Reverse the linked list** instead of deleting (see Question 4) — a very common swap.

### Common Follow-up Questions

- *Why track `prev`?* In a singly list you can't go backward; to unlink a node you must edit its predecessor's `next`.
- *Which edge cases break naive code?* Empty list, deleting the head, deleting the tail, value absent. Always test these.

---

## Practical Question 3

**Difficulty:** Medium
**Estimated Time:** 15 minutes
**Concepts Tested:** insertion relative to a value, before/after handling

**Problem Statement**
Insert a new node with value `v` **before** the first node whose value is `x`. Also support inserting **after**.

**Example Input**
```
List: 10 -> 20 -> 30,  insert 15 before 20
```

**Example Output**
```
10 -> 15 -> 20 -> 30
```

**Approach**
- **Insert after `x`:** find the node with value `x`, splice the new node between it and its successor — O(n) to find, O(1) to splice.
- **Insert before `x`:** you need the node *before* `x`, so track `prev`. Special-case inserting before the head.

### Python Implementation

```python
def insert_after(ll: SinglyLinkedList, x, v) -> bool:
    """Insert v immediately AFTER the first node equal to x."""
    current = ll.head
    while current:
        if current.data == x:
            node = Node(v)
            node.next = current.next
            current.next = node
            if current is ll.tail:         # inserted at the very end
                ll.tail = node
            return True
        current = current.next
    return False


def insert_before(ll: SinglyLinkedList, x, v) -> bool:
    """Insert v immediately BEFORE the first node equal to x."""
    if ll.head is None:
        return False
    if ll.head.data == x:                  # insert before head → prepend
        ll.prepend(v)
        return True
    prev, current = ll.head, ll.head.next
    while current:
        if current.data == x:
            node = Node(v)
            node.next = current
            prev.next = node
            return True
        prev, current = current, current.next
    return False
```

**Complexity:** Time O(n), Space O(1) for both.

### Common Follow-up Questions

- *Why is "insert before" harder than "insert after"?* A singly list only has forward links, so reaching the predecessor of `x` requires tracking `prev` or a special head case; "after" needs only the node itself.
- *How would a doubly linked list simplify this?* With a `prev` pointer you can insert before a node directly without scanning from the head.

---

## Practical Question 4

**Difficulty:** Medium
**Estimated Time:** 15 minutes
**Concepts Tested:** in-place pointer reversal (the classic)

**Problem Statement**
Reverse a singly linked list in place and return the new head.

**Example Input**
```
1 -> 2 -> 3 -> 4 -> None
```

**Example Output**
```
4 -> 3 -> 2 -> 1 -> None
```

**Approach (step-by-step)**
1. Use three pointers: `prev = None`, `curr = head`, and a temp `nxt`.
2. For each node: save `nxt = curr.next`, then reverse the link `curr.next = prev`.
3. Advance `prev = curr` and `curr = nxt`.
4. When `curr` is `None`, `prev` is the new head.

```
prev=None  curr=1 -> 2 -> 3 -> 4
Step: save nxt=2, point 1->None, prev=1, curr=2
...repeat... final: 4 -> 3 -> 2 -> 1 -> None
```

### Python Implementation

```python
def reverse_list(head: Node) -> Node:
    """Reverse a singly linked list in place; return new head. O(n)/O(1)."""
    prev = None
    curr = head
    while curr:
        nxt = curr.next        # save next BEFORE we overwrite it
        curr.next = prev       # reverse the pointer
        prev = curr            # advance prev
        curr = nxt             # advance curr
    return prev                # prev is the new head
```

**Complexity:** Time O(n), Space O(1).

### Alternative Solution

Recursive reversal (elegant but O(n) stack space):
```python
def reverse_recursive(head):
    if head is None or head.next is None:
        return head
    new_head = reverse_recursive(head.next)
    head.next.next = head       # make the next node point back
    head.next = None
    return new_head
```
Avoid on very long lists — it can overflow the call stack.

### Interview Variations

- Reverse **only between positions m and n**.
- Reverse in **groups of k** nodes.
- Check if a list is a **palindrome** (reverse the second half and compare).

### Common Follow-up Questions

- *Why save `nxt` first?* Once you set `curr.next = prev`, the original link to the rest of the list is gone; saving `nxt` preserves it.
- *Iterative vs recursive?* Iterative uses O(1) space and is preferred for long lists; recursive is cleaner but uses O(n) stack.

---

## Practical Question 5

**Difficulty:** Easy–Medium
**Estimated Time:** 10 minutes
**Concepts Tested:** sorted-list property, adjacent-duplicate removal

**Problem Statement**
Remove duplicates from a **sorted** singly linked list so each value appears once.

**Example Input**
```
1 -> 1 -> 2 -> 3 -> 3
```

**Example Output**
```
1 -> 2 -> 3
```

**Approach**
Because the list is sorted, duplicates are adjacent. Walk once; whenever the current node equals the next node, unlink the next. Otherwise advance.

### Python Implementation

```python
def remove_duplicates_sorted(head: Node) -> Node:
    """Remove adjacent duplicates from a sorted list. O(n)/O(1)."""
    current = head
    while current and current.next:
        if current.data == current.next.data:
            current.next = current.next.next   # skip the duplicate
        else:
            current = current.next             # advance only when distinct
    return head
```

**Complexity:** Time O(n), Space O(1).

### Interview Variations

- Remove duplicates from an **unsorted** list (use a hash set → O(n) time, O(n) space).
- Remove **all** nodes that have duplicates (leave only distinct values).

### Common Follow-up Questions

- *Why is O(1) space possible here but not for an unsorted list?* Sorting guarantees duplicates are neighbors, so a single comparison suffices; unsorted lists need a hash set to remember what was seen.

---

## Doubly Linked List (construction + reverse print)

**Problem Statement**
Construct a doubly linked list and print it in both forward and reverse order.

### Python Implementation

```python
class DNode:
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None


class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def append(self, data):
        """Append at the tail. O(1)."""
        node = DNode(data)
        if self.head is None:
            self.head = self.tail = node
            return
        node.prev = self.tail
        self.tail.next = node
        self.tail = node

    def print_forward(self):
        vals, cur = [], self.head
        while cur:
            vals.append(str(cur.data))
            cur = cur.next
        print(" <-> ".join(vals) if vals else "empty")

    def print_reverse(self):
        """Reverse print is O(n) using prev pointers — no extra structure."""
        vals, cur = [], self.tail
        while cur:
            vals.append(str(cur.data))
            cur = cur.prev
        print(" <-> ".join(vals) if vals else "empty")
```

**Why a DLL for reverse printing?** In a singly list, printing in reverse needs recursion or a stack (O(n) extra space). A doubly linked list walks backward from the tail using `prev` pointers in O(n) time and O(1) extra space — the extra pointer per node pays off.

### Common Follow-up Questions

- *Cost of the extra `prev` pointer?* More memory per node and two links to maintain on every insert/delete — easy to corrupt if you update only one direction.
- *When is a DLL worth it?* Backward traversal, O(1) deletion of a held node, and structures like LRU caches and deques.

---

# Section C — Strings

## Practical Question 1

**Difficulty:** Easy–Medium
**Estimated Time:** 12 minutes
**Concepts Tested:** character arithmetic, modular wrap-around (the syllabus cipher task)

**Problem Statement**
Write a function that replaces each alphabet in a string with the letter occurring at the **n-th position** from it in the alphabet, wrapping around (a Caesar shift). Non-letters stay unchanged; preserve case.

**Example Input**
```
text = "abcXYZ", n = 2
```

**Example Output**
```
"cdeZAB"
```

**Approach (step-by-step)**
1. For each character, decide if it's lowercase, uppercase, or other.
2. For a letter, map it to 0–25 by subtracting its base (`'a'` or `'A'`).
3. Add `n`, take modulo 26 so it wraps past 'z'/'Z', then add the base back.
4. Leave non-letters untouched. Build the result with a list + `join` (avoid O(n²) `+=`).

### Python Implementation

```python
def shift_cipher(text: str, n: int) -> str:
    """Shift each letter by n positions with wrap-around; keep case & non-letters."""
    result = []                                  # accumulate pieces, join once
    for ch in text:
        if ch.islower():
            base = ord('a')
            result.append(chr((ord(ch) - base + n) % 26 + base))
        elif ch.isupper():
            base = ord('A')
            result.append(chr((ord(ch) - base + n) % 26 + base))
        else:
            result.append(ch)                    # non-letters pass through
    return "".join(result)
```

**Line notes**
- `(ord(ch) - base + n) % 26` is the whole trick: it keeps the letter inside the 26-letter ring.
- Using a list and `"".join` keeps it O(n); repeated `+=` on a string would be O(n²).

**Complexity:** Time O(n), Space O(n) (the output string).

### Alternative Solution

`str.translate` with a prebuilt table is the fastest production approach:
```python
def shift_cipher_table(text, n):
    import string
    lower = string.ascii_lowercase
    upper = string.ascii_uppercase
    table = str.maketrans(
        lower + upper,
        lower[n % 26:] + lower[:n % 26] + upper[n % 26:] + upper[:n % 26],
    )
    return text.translate(table)
```
Builds the mapping once; ideal when applying the same shift to many strings.

### Interview Variations

- **Decrypt** a shifted string (shift by `26 - n`).
- Shift by a **keyword** (Vigenère cipher).
- Shift **digits** too, wrapping 0–9.

### Common Follow-up Questions

- *Why modulo 26?* It makes the alphabet circular so shifting 'y' by 2 gives 'a', not a non-letter character.
- *Why not use `+=` in the loop?* Strings are immutable, so each `+=` copies the whole accumulated string — quadratic. List + `join` is linear.

---

## Practical Question 2

**Difficulty:** Easy
**Estimated Time:** 10 minutes
**Concepts Tested:** frequency counting, hashing, anagram detection

**Problem Statement**
Determine whether two strings are anagrams (same characters, same counts).

**Example Input**
```
s = "listen", t = "silent"
```

**Example Output**
```
True
```

**Approach**
1. If lengths differ, they can't be anagrams — return early.
2. Count character frequencies of both and compare.
3. Using a hash map (`Counter`) this is O(n).

### Python Implementation

```python
from collections import Counter

def is_anagram(s: str, t: str) -> bool:
    """True if s and t are anagrams. O(n) time, O(1) space (fixed alphabet)."""
    if len(s) != len(t):          # quick reject
        return False
    return Counter(s) == Counter(t)
```

**Complexity:** Time O(n), Space O(1) for a fixed alphabet (at most 26/128 keys).

### Alternative Solution

Sort both and compare — O(n log n), no hash map:
```python
def is_anagram_sort(s, t):
    return sorted(s) == sorted(t)
```
Simpler but slower; use when you can't or don't want to build a counter.

### Interview Variations

- **Group anagrams** in a list of words (key each word by its sorted form).
- Check anagrams **ignoring spaces/case/punctuation**.
- Find the **minimum deletions** to make two strings anagrams.

### Common Follow-up Questions

- *Why check length first?* It's an O(1) reject that avoids counting when the answer is obviously false.
- *Counter vs sorting?* Counter is O(n) vs sorting's O(n log n); sorting wins only on simplicity or when hashing isn't available.

---

## Practical Question 3

**Difficulty:** Easy–Medium
**Estimated Time:** 10 minutes
**Concepts Tested:** two pointers, in-place reversal, palindrome check

**Problem Statement**
Check whether a string is a palindrome, considering only alphanumeric characters and ignoring case.

**Example Input**
```
"A man, a plan, a canal: Panama"
```

**Example Output**
```
True
```

**Approach**
Use two pointers from both ends, skipping non-alphanumeric characters, comparing lowercased characters. O(n) time, O(1) space.

### Python Implementation

```python
def is_palindrome(s: str) -> bool:
    """Two-pointer palindrome check ignoring case & non-alphanumerics."""
    left, right = 0, len(s) - 1
    while left < right:
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        if s[left].lower() != s[right].lower():
            return False
        left += 1
        right -= 1
    return True
```

**Complexity:** Time O(n), Space O(1).

### Interview Variations

- Return the **longest palindromic substring** (expand-around-center, O(n²)).
- Check if a string can become a palindrome by removing **at most one** character.
- Count palindromic substrings.

### Common Follow-up Questions

- *Why two pointers instead of reversing?* Reversing costs O(n) extra space; two pointers verify in place with O(1) space.
- *How to handle Unicode?* `isalnum` and `lower` handle many scripts, but full Unicode normalization may be needed for accented characters.

---

# Section D — Searching

## Practical Question 1

**Difficulty:** Easy
**Estimated Time:** 8 minutes
**Concepts Tested:** linear scan, early return

**Problem Statement**
Implement linear search: return the index of `target` in an array, or -1 if absent.

**Example Input**
```
arr = [4, 2, 7, 1, 9], target = 7
```

**Example Output**
```
2
```

### Python Implementation

```python
def linear_search(arr: list, target) -> int:
    """Return index of target, or -1. O(n) time, O(1) space."""
    for i, value in enumerate(arr):
        if value == target:
            return i           # early return on first match
    return -1
```

**Complexity:** Time O(n) worst/average, O(1) best; Space O(1). Works on unsorted data.

### Common Follow-up Questions

- *When is linear search the right choice?* Small or unsorted data, or when you'll search only once (sorting first wouldn't pay off).

---

## Practical Question 2

**Difficulty:** Medium
**Estimated Time:** 15 minutes
**Concepts Tested:** binary search, loop invariants, overflow-safe mid

**Problem Statement**
Implement binary search on a sorted array. Return the index of `target` or -1.

**Example Input**
```
arr = [1, 3, 5, 7, 9, 11], target = 7
```

**Example Output**
```
3
```

**Approach (step-by-step)**
1. Maintain `low` and `high` bounding the search range.
2. Compute `mid = low + (high - low) // 2` (overflow-safe form).
3. Compare `arr[mid]` to target; discard the half that can't contain it.
4. Loop while `low <= high`; if the range empties, return -1.

### Python Implementation

```python
def binary_search(arr: list, target) -> int:
    """Iterative binary search on a sorted array. O(log n) time, O(1) space."""
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = low + (high - low) // 2      # avoids overflow in fixed-width langs
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1                  # target in right half
        else:
            high = mid - 1                 # target in left half
    return -1
```

**Complexity:** Time O(log n), Space O(1). (Recursive version is O(log n) space.)

### Alternative Solution

Recursive binary search:
```python
def binary_search_rec(arr, target, low, high):
    if low > high:
        return -1
    mid = low + (high - low) // 2
    if arr[mid] == target:
        return mid
    if arr[mid] < target:
        return binary_search_rec(arr, target, mid + 1, high)
    return binary_search_rec(arr, target, low, mid - 1)
```

### Interview Variations

- **First/last occurrence** of a value with duplicates (bias the search left/right).
- **Search in a rotated sorted array** (decide which half is sorted).
- **Find the square root** / first bad version (binary search on the answer space).
- **Insertion point** (`bisect_left` behavior).

### Common Follow-up Questions

- *Why `low + (high - low)//2` instead of `(low+high)//2`?* In fixed-width integer languages `low+high` can overflow; this form can't. It's a good habit even in Python.
- *Off-by-one traps?* Ensure `low`/`high` actually move past `mid`, or the loop never terminates. Decide `<=` vs `<` deliberately.

---

## Notebook Workflow: Benchmarking Linear vs Binary Search

> Lab exams sometimes ask you to *demonstrate* the complexity difference. Here is
> a clean notebook-cell workflow.

**Cell 1 — Imports**
```python
import time
import random
import matplotlib.pyplot as plt
```

**Cell 2 — Implementations**
```python
# (paste linear_search and binary_search from above)
```

**Cell 3 — Timing helper**
```python
def time_search(func, arr, target):
    start = time.perf_counter()
    func(arr, target)
    return time.perf_counter() - start
```

**Cell 4 — Run across growing sizes**
```python
sizes = [10**3, 10**4, 10**5, 10**6]
linear_times, binary_times = [], []
for n in sizes:
    data = list(range(n))          # sorted 0..n-1
    target = n - 1                 # worst case for linear (last element)
    linear_times.append(time_search(linear_search, data, target))
    binary_times.append(time_search(binary_search, data, target))
```

**Cell 5 — Visualize**
```python
plt.plot(sizes, linear_times, label="Linear O(n)", marker="o")
plt.plot(sizes, binary_times, label="Binary O(log n)", marker="o")
plt.xlabel("Input size n"); plt.ylabel("Time (s)")
plt.xscale("log"); plt.legend(); plt.title("Linear vs Binary Search")
plt.show()
```

**Cell 6 — Interpretation (markdown)**
> Linear search time grows roughly proportionally with `n`, while binary search
> stays almost flat — visual proof of O(n) vs O(log n). Note binary search
> requires the sorted input we created with `range(n)`.

---

# Section E — Recursion

## Practical Question 1

**Difficulty:** Easy
**Estimated Time:** 8 minutes
**Concepts Tested:** base case, recursive case, factorial

**Problem Statement**
Compute `n!` recursively.

**Example Input/Output**
```
factorial(5) -> 120
```

### Python Implementation

```python
def factorial(n: int) -> int:
    """n! via recursion. Time O(n), Space O(n) call stack."""
    if n < 0:
        raise ValueError("factorial is undefined for negatives")
    if n <= 1:                 # base case: 0! = 1! = 1
        return 1
    return n * factorial(n - 1)  # recursive case
```

**Complexity:** Time O(n), Space O(n) (n stacked frames).

### Common Follow-up Questions

- *Iterative version?* A simple loop gives O(n) time and O(1) space — preferred when `n` is large enough to risk a stack overflow.

---

## Practical Question 2

**Difficulty:** Easy–Medium
**Estimated Time:** 12 minutes
**Concepts Tested:** overlapping subproblems, memoization

**Problem Statement**
Compute the n-th Fibonacci number efficiently.

**Example Input/Output**
```
fib(10) -> 55
```

**Approach**
Naive recursion is O(2ⁿ) because it recomputes subproblems. **Memoize** results so each value is computed once → O(n).

### Python Implementation

```python
from functools import lru_cache

@lru_cache(maxsize=None)          # caches results → each fib(k) computed once
def fib(n: int) -> int:
    """n-th Fibonacci with memoization. Time O(n), Space O(n)."""
    if n < 2:                     # base cases: fib(0)=0, fib(1)=1
        return n
    return fib(n - 1) + fib(n - 2)
```

### Alternative Solution

Bottom-up iterative — O(n) time, **O(1) space**:
```python
def fib_iter(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
```
Preferred in production: no recursion depth limit, constant memory.

### Interview Variations

- Count ways to **climb stairs** (1 or 2 steps) — same recurrence.
- **Tribonacci**, or Fibonacci **modulo m**.
- Return the whole sequence up to n.

### Common Follow-up Questions

- *Why is naive Fibonacci exponential?* It re-solves the same subproblems repeatedly; the call tree has ~2ⁿ nodes. Memoization collapses that to n unique computations.
- *Memoization vs tabulation?* Both O(n); memoization is top-down recursion + cache, tabulation is bottom-up iteration. Tabulation avoids stack limits.

---

## Practical Question 3

**Difficulty:** Medium
**Estimated Time:** 15 minutes
**Concepts Tested:** backtracking, recursion tree, permutations

**Problem Statement**
Generate all permutations of a list of distinct integers.

**Example Input**
```
[1, 2, 3]
```

**Example Output**
```
[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

**Approach**
Backtracking: build a permutation one element at a time; at each step choose an unused element, recurse, then **undo** the choice (backtrack) to explore alternatives.

### Python Implementation

```python
def permutations(nums: list[int]) -> list[list[int]]:
    """All permutations via backtracking. Time O(n * n!), Space O(n) recursion."""
    result = []

    def backtrack(current: list[int], remaining: list[int]) -> None:
        if not remaining:                 # base case: nothing left to place
            result.append(current[:])     # copy the completed permutation
            return
        for i in range(len(remaining)):
            current.append(remaining[i])                     # choose
            backtrack(current, remaining[:i] + remaining[i+1:])  # explore
            current.pop()                                    # un-choose (backtrack)

    backtrack([], nums)
    return result
```

**Complexity:** Time O(n · n!) (n! permutations, O(n) to build each), Space O(n) recursion depth plus output.

### Interview Variations

- **Subsets** (power set) — include/exclude each element.
- **Combinations** of size k.
- Permutations **with duplicates** (skip repeats to avoid duplicate outputs).
- **N-Queens**, Sudoku solver — classic backtracking.

### Common Follow-up Questions

- *What is backtracking?* A refined recursion that builds candidates incrementally and abandons a path as soon as it can't lead to a valid solution, undoing the last choice before trying the next.
- *Why copy `current[:]`?* `current` is mutated across the recursion; storing a reference would leave all results pointing at the same (eventually empty) list.

---

# Section F — Sorting

> The syllabus requires Bubble, Selection, and Insertion sort. Implement all
> three from memory; interviewers frequently ask you to code one and analyze it.

## Practical Question 1 — Bubble Sort

**Difficulty:** Easy
**Estimated Time:** 10 minutes
**Concepts Tested:** adjacent swaps, early-exit optimization, stability

**Problem Statement**
Sort an array in ascending order using bubble sort with the early-exit optimization.

**Example Input/Output**
```
[5, 1, 4, 2, 8] -> [1, 2, 4, 5, 8]
```

### Python Implementation

```python
def bubble_sort(arr: list) -> list:
    """In-place bubble sort with early exit. Best O(n), avg/worst O(n²)."""
    n = len(arr)
    for i in range(n - 1):
        swapped = False
        # after i passes, the last i elements are already in place
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:            # no swaps → already sorted → stop early
            break
    return arr
```

**Complexity:** Time O(n²) average/worst, O(n) best (sorted); Space O(1). Stable.

### Common Follow-up Questions

- *What does the `swapped` flag buy you?* On an already-sorted (or nearly-sorted) array the first clean pass exits early, giving O(n) best case instead of always O(n²).

---

## Practical Question 2 — Selection Sort

**Difficulty:** Easy
**Estimated Time:** 10 minutes
**Concepts Tested:** minimum selection, swap minimization, why it's not adaptive

### Python Implementation

```python
def selection_sort(arr: list) -> list:
    """In-place selection sort. Always O(n²); minimizes swaps (n-1). Not stable."""
    n = len(arr)
    for i in range(n - 1):
        min_idx = i
        for j in range(i + 1, n):      # find the minimum in the unsorted part
            if arr[j] < arr[min_idx]:
                min_idx = j
        if min_idx != i:               # swap it into position i
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr
```

**Complexity:** Time O(n²) in all cases; Space O(1). Not stable.

### Common Follow-up Questions

- *Why is it O(n²) even on sorted data?* Finding the minimum always scans the whole remaining region; it never detects that the data is already ordered.
- *One thing it's good at?* It performs at most n−1 swaps — useful when writes are far more expensive than comparisons (e.g., flash memory).

---

## Practical Question 3 — Insertion Sort

**Difficulty:** Easy
**Estimated Time:** 10 minutes
**Concepts Tested:** shifting, adaptivity, best case O(n)

### Python Implementation

```python
def insertion_sort(arr: list) -> list:
    """In-place insertion sort. Great on nearly-sorted data. Stable."""
    for i in range(1, len(arr)):
        key = arr[i]                   # element to insert into the sorted prefix
        j = i - 1
        while j >= 0 and arr[j] > key: # shift larger elements right
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key               # drop key into the gap
    return arr
```

**Complexity:** Time O(n²) average/worst, O(n) best (sorted); Space O(1). Stable.

### Interview Variations

- Sort in **descending** order (flip the comparison).
- Sort **strings** or objects by a key (`key=` style).
- Explain where insertion sort is used inside **Timsort** (small runs).

### Common Follow-up Questions

- *Why is insertion sort preferred for small/nearly-sorted arrays?* Each element is close to its final spot, so few shifts occur (approaching O(n)); it's stable, in-place, and has tiny constant factors — which is why library sorts fall back to it for small subarrays.
- *Bubble vs insertion?* Both O(n²) and stable, but insertion typically does fewer operations and is the practical choice.

## Notebook Workflow: Comparing the Three Sorts

**Cell 1 — Imports**
```python
import time, random
import matplotlib.pyplot as plt
```
**Cell 2 — paste the three sort functions**
**Cell 3 — Benchmark harness**
```python
def benchmark(sort_fn, data):
    arr = data[:]                      # copy so each sort gets the same input
    start = time.perf_counter()
    sort_fn(arr)
    return time.perf_counter() - start

sizes = [100, 500, 1000, 2000]
results = {"bubble": [], "selection": [], "insertion": []}
for n in sizes:
    data = [random.randint(0, 10000) for _ in range(n)]
    results["bubble"].append(benchmark(bubble_sort, data))
    results["selection"].append(benchmark(selection_sort, data))
    results["insertion"].append(benchmark(insertion_sort, data))
```
**Cell 4 — Plot**
```python
for name, times in results.items():
    plt.plot(sizes, times, marker="o", label=name)
plt.xlabel("n"); plt.ylabel("seconds"); plt.legend(); plt.title("O(n²) sorts")
plt.show()
```
**Cell 5 — Interpretation (markdown):** All three curves bend upward quadratically; insertion is usually fastest on random data among the three, and dramatically faster if you re-run on already-sorted input (its O(n) best case).

---

# Section G — Hash Tables

## Practical Question 1

**Difficulty:** Easy
**Estimated Time:** 10 minutes
**Concepts Tested:** frequency map, counting

**Problem Statement**
Given a list, return the first element that repeats (the first element whose duplicate appears later), or `None`.

**Example Input/Output**
```
[3, 5, 2, 5, 3] -> 5    # 5's duplicate appears before 3's does
```

### Python Implementation

```python
def first_repeating(arr: list):
    """First element that has a duplicate later. O(n) time, O(n) space."""
    seen = set()
    first = None
    # scan RIGHT to LEFT so the earliest repeater wins
    for i in range(len(arr) - 1, -1, -1):
        if arr[i] in seen:
            first = arr[i]
        else:
            seen.add(arr[i])
    return first
```

**Complexity:** Time O(n), Space O(n).

### Common Follow-up Questions

- *Why scan right-to-left?* It lets the leftmost repeating element overwrite later ones, so we end with the *first* repeater in one pass.

---

## Practical Question 2

**Difficulty:** Medium
**Estimated Time:** 20 minutes
**Concepts Tested:** implementing a hash table, collision handling (chaining)

**Problem Statement**
Implement a hash table (map) from scratch supporting `put`, `get`, and `remove`, using **separate chaining** for collisions.

### Python Implementation

```python
class HashTable:
    """A simple hash map using separate chaining."""

    def __init__(self, capacity: int = 8):
        self.capacity = capacity
        self.size = 0
        self.buckets = [[] for _ in range(capacity)]   # each bucket is a list

    def _index(self, key) -> int:
        return hash(key) % self.capacity               # map key to a bucket

    def put(self, key, value) -> None:
        idx = self._index(key)
        bucket = self.buckets[idx]
        for i, (k, _) in enumerate(bucket):
            if k == key:                               # update existing key
                bucket[i] = (key, value)
                return
        bucket.append((key, value))                    # new key
        self.size += 1
        if self.size / self.capacity > 0.7:            # load factor threshold
            self._resize()

    def get(self, key):
        bucket = self.buckets[self._index(key)]
        for k, v in bucket:
            if k == key:
                return v
        raise KeyError(key)

    def remove(self, key) -> None:
        bucket = self.buckets[self._index(key)]
        for i, (k, _) in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                self.size -= 1
                return
        raise KeyError(key)

    def _resize(self) -> None:
        """Double capacity and rehash everything. Amortized O(1) inserts."""
        old = [pair for bucket in self.buckets for pair in bucket]
        self.capacity *= 2
        self.buckets = [[] for _ in range(self.capacity)]
        self.size = 0
        for k, v in old:
            self.put(k, v)
```

**Complexity:** `put`/`get`/`remove` are O(1) average, O(n) worst (all keys in one bucket). Resize is O(n) but rare → amortized O(1) inserts.

### Interview Variations

- Implement with **open addressing / linear probing** instead of chaining.
- Add an **LRU eviction** policy (hash map + doubly linked list).
- Implement a **set** (keys only).

### Common Follow-up Questions

- *Why resize at load factor ~0.7?* Beyond that, chains lengthen and average lookup slows; resizing keeps buckets short and operations near O(1).
- *Chaining vs open addressing trade-offs?* Chaining is simple and degrades gracefully but uses extra memory for lists; open addressing is cache-friendlier but suffers clustering and needs careful deletion (tombstones).

---

# Section H — Stacks

## Foundation

```python
class Stack:
    """LIFO stack backed by a Python list."""
    def __init__(self):
        self._data = []

    def push(self, x):  self._data.append(x)          # O(1) amortized
    def pop(self):
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._data.pop()                       # O(1)
    def peek(self):
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self._data[-1]
    def is_empty(self): return len(self._data) == 0
    def __len__(self):  return len(self._data)
```

## Practical Question 1

**Difficulty:** Easy–Medium
**Estimated Time:** 12 minutes
**Concepts Tested:** stack for nested matching (the canonical stack problem)

**Problem Statement**
Given a string of brackets `()[]{}`, determine if it is **balanced** (every opener has a correct, correctly-ordered closer).

**Example Input/Output**
```
"{[()]}" -> True
"{[(])}" -> False
```

**Approach**
Push openers; on a closer, the top of the stack must be the matching opener. At the end the stack must be empty.

### Python Implementation

```python
def is_balanced(s: str) -> bool:
    """Check balanced brackets using a stack. O(n) time, O(n) space."""
    pairs = {')': '(', ']': '[', '}': '{'}   # closer -> opener
    stack = []
    for ch in s:
        if ch in '([{':
            stack.append(ch)                 # opener: push
        elif ch in ')]}':
            if not stack or stack[-1] != pairs[ch]:
                return False                 # mismatch or nothing to match
            stack.pop()                      # matched: pop the opener
    return not stack                         # balanced only if nothing left
```

**Complexity:** Time O(n), Space O(n).

### Interview Variations

- Return the **index** of the first unbalanced bracket.
- Support only one bracket type but also **other characters** (ignore them).
- **Min add to make valid** — count insertions needed.
- Evaluate a **postfix (RPN)** expression with a stack.

### Common Follow-up Questions

- *Why does a stack fit this problem?* Nesting is inherently LIFO — the most recently opened bracket must close first, which is exactly what a stack tracks.
- *Edge cases?* Empty string (balanced), a lone closer (unbalanced), unmatched openers left at the end.

---

## Practical Question 2

**Difficulty:** Medium
**Estimated Time:** 15 minutes
**Concepts Tested:** auxiliary stack, O(1) minimum

**Problem Statement**
Design a stack that supports `push`, `pop`, `top`, and `get_min`, all in O(1).

### Python Implementation

```python
class MinStack:
    """Stack with O(1) minimum via a parallel min-tracking stack."""
    def __init__(self):
        self._data = []
        self._mins = []                       # _mins[-1] is current minimum

    def push(self, x):
        self._data.append(x)
        # push the smaller of x and the current min
        self._mins.append(x if not self._mins else min(x, self._mins[-1]))

    def pop(self):
        self._mins.pop()
        return self._data.pop()

    def top(self):
        return self._data[-1]

    def get_min(self):
        return self._mins[-1]                 # O(1)
```

**Complexity:** All operations O(1) time; O(n) extra space for the min stack.

### Common Follow-up Questions

- *How is O(1) min possible?* By storing the running minimum alongside each element, so the current min is always at the top of the auxiliary stack.
- *Can you save space?* Store a min only when it changes, or encode deltas — trickier but reduces memory.

---

# Section I — Queues

## Foundation

```python
from collections import deque

class Queue:
    """FIFO queue backed by collections.deque (O(1) both ends)."""
    def __init__(self):
        self._data = deque()

    def enqueue(self, x): self._data.append(x)       # add at rear, O(1)
    def dequeue(self):
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        return self._data.popleft()                  # remove from front, O(1)
    def peek(self):       return self._data[0]
    def is_empty(self):   return len(self._data) == 0
    def __len__(self):    return len(self._data)
```

> **Exam trap:** Never implement a queue with `list.pop(0)` — that's O(n).
> Use `collections.deque`.

## Practical Question 1

**Difficulty:** Medium
**Estimated Time:** 18 minutes
**Concepts Tested:** building a queue from two stacks (classic)

**Problem Statement**
Implement a FIFO queue using only two stacks.

### Python Implementation

```python
class QueueFromStacks:
    """FIFO queue using two LIFO stacks. Amortized O(1) per operation."""
    def __init__(self):
        self._in = []      # push here
        self._out = []     # pop here

    def enqueue(self, x):
        self._in.append(x)                    # O(1)

    def dequeue(self):
        if not self._out:                     # refill only when out is empty
            while self._in:
                self._out.append(self._in.pop())   # reverses order → FIFO
        if not self._out:
            raise IndexError("dequeue from empty queue")
        return self._out.pop()                # O(1) amortized
```

**Complexity:** `enqueue` O(1); `dequeue` amortized O(1) (each element is moved between stacks at most once).

### Interview Variations

- Implement a **stack using two queues** (the mirror problem).
- Implement a **circular queue** with a fixed-size array (below).
- Build a queue that also supports **get_max** in O(1) amortized (monotonic deque).

### Common Follow-up Questions

- *Why is dequeue amortized O(1) if a transfer is O(n)?* Each element is transferred from `in` to `out` exactly once over its lifetime, so the total transfer work across n operations is O(n) → O(1) each on average.

---

## Practical Question 2 — Circular Queue

**Difficulty:** Medium
**Estimated Time:** 20 minutes
**Concepts Tested:** ring buffer, modular indices, full vs empty

**Problem Statement**
Implement a fixed-capacity circular queue with O(1) `enqueue` and `dequeue`.

### Python Implementation

```python
class CircularQueue:
    """Fixed-capacity ring buffer. All operations O(1)."""
    def __init__(self, capacity: int):
        self._data = [None] * capacity
        self._capacity = capacity
        self._front = 0
        self._size = 0

    def enqueue(self, x) -> bool:
        if self._size == self._capacity:      # full
            return False
        rear = (self._front + self._size) % self._capacity
        self._data[rear] = x
        self._size += 1
        return True

    def dequeue(self):
        if self._size == 0:                   # empty
            raise IndexError("dequeue from empty queue")
        x = self._data[self._front]
        self._data[self._front] = None
        self._front = (self._front + 1) % self._capacity   # wrap around
        self._size -= 1
        return x

    def is_full(self):  return self._size == self._capacity
    def is_empty(self): return self._size == 0
```

**Complexity:** All operations O(1); Space O(capacity).

### Common Follow-up Questions

- *How do you distinguish full from empty?* Track an explicit `size` (used here) or leave one slot empty; otherwise `front == rear` is ambiguous between full and empty.
- *Why modulo?* It wraps indices around the fixed array so freed front slots get reused — no shifting.

---

# Section J — Trees & BST

> The syllabus explicitly asks for: a BST with insertion, search, delete, and
> traversals; finding the closest value; and validating a BST. All are below.

## Foundation

```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
```

## Practical Question 1 — Build a BST (insert, search, delete, traversals)

**Difficulty:** Medium
**Estimated Time:** 30 minutes
**Concepts Tested:** BST invariant, recursion, deletion cases, traversals

**Problem Statement**
Implement a Binary Search Tree supporting insertion, search (`contains`), deletion, and in-order/pre-order/post-order/level-order traversals.

### Python Implementation

```python
from collections import deque

class BST:
    def __init__(self):
        self.root = None

    # ---------- INSERT ----------
    def insert(self, val):
        """Insert keeping the BST invariant. O(h) where h = height."""
        self.root = self._insert(self.root, val)

    def _insert(self, node, val):
        if node is None:
            return TreeNode(val)              # found the spot
        if val < node.val:
            node.left = self._insert(node.left, val)
        elif val > node.val:
            node.right = self._insert(node.right, val)
        # equal → ignore duplicates (or handle as needed)
        return node

    # ---------- SEARCH ----------
    def contains(self, val) -> bool:
        """Search. O(h)."""
        node = self.root
        while node:
            if val == node.val:
                return True
            node = node.left if val < node.val else node.right
        return False

    # ---------- DELETE ----------
    def delete(self, val):
        """Delete a value handling all three cases. O(h)."""
        self.root = self._delete(self.root, val)

    def _delete(self, node, val):
        if node is None:
            return None
        if val < node.val:
            node.left = self._delete(node.left, val)
        elif val > node.val:
            node.right = self._delete(node.right, val)
        else:
            # found the node to delete
            if node.left is None:            # 0 or 1 child (right)
                return node.right
            if node.right is None:           # 1 child (left)
                return node.left
            # 2 children: replace with in-order successor (min of right subtree)
            successor = self._min_node(node.right)
            node.val = successor.val
            node.right = self._delete(node.right, successor.val)
        return node

    def _min_node(self, node):
        while node.left:
            node = node.left
        return node

    # ---------- TRAVERSALS ----------
    def in_order(self) -> list:
        """Left, Root, Right → sorted order for a BST."""
        out = []
        def walk(n):
            if n:
                walk(n.left); out.append(n.val); walk(n.right)
        walk(self.root)
        return out

    def pre_order(self) -> list:
        out = []
        def walk(n):
            if n:
                out.append(n.val); walk(n.left); walk(n.right)
        walk(self.root)
        return out

    def post_order(self) -> list:
        out = []
        def walk(n):
            if n:
                walk(n.left); walk(n.right); out.append(n.val)
        walk(self.root)
        return out

    def level_order(self) -> list:
        """BFS traversal using a queue."""
        if not self.root:
            return []
        out, q = [], deque([self.root])
        while q:
            node = q.popleft()
            out.append(node.val)
            if node.left:  q.append(node.left)
            if node.right: q.append(node.right)
        return out
```

**Complexity:** insert/search/delete O(h) — O(log n) balanced, O(n) skewed. Traversals O(n) time, O(h) space (recursion/queue).

### Common Follow-up Questions

- *Delete with two children — why the in-order successor?* The successor (smallest in the right subtree) is the next-largest value, so replacing the node with it preserves ordering, and it has at most one child, making its removal easy.
- *How to guarantee O(log n)?* Use a self-balancing tree (AVL/Red-Black); a plain BST degrades to O(n) on sorted input.

---

## Practical Question 2 — Closest Value in a BST

**Difficulty:** Medium
**Estimated Time:** 12 minutes
**Concepts Tested:** BST navigation, tracking a running best

**Problem Statement**
Given a BST and a target value, return the value in the tree closest to the target.

**Example Input/Output**
```
Tree rooted at 10 with children 5 and 15..., target = 12  ->  closest = 13 (say)
```

### Python Implementation

```python
def closest_value(root: TreeNode, target: float):
    """Walk toward target, tracking closest. O(h) time, O(1) space."""
    closest = root.val
    node = root
    while node:
        if abs(node.val - target) < abs(closest - target):
            closest = node.val
        if target < node.val:
            node = node.left        # closer values are left
        elif target > node.val:
            node = node.right       # closer values are right
        else:
            return node.val         # exact match
    return closest
```

**Complexity:** Time O(h), Space O(1).

### Common Follow-up Questions

- *Why can we discard a whole subtree each step?* The BST property guarantees all smaller values are left and larger are right, so the target's closest neighbor lies along a single root-to-leaf path.

---

## Practical Question 3 — Validate a BST

**Difficulty:** Medium
**Estimated Time:** 15 minutes
**Concepts Tested:** subtree range constraints (the classic gotcha)

**Problem Statement**
Return `True` if a binary tree is a valid BST.

**Example Input/Output**
```
   5
  / \
 1   4        -> False  (4 is right of 5 but should be > 5)
    / \
   3   6
```

**Approach**
The common wrong approach only compares a node to its immediate children. The correct approach passes down a valid `(low, high)` range that tightens as you descend.

### Python Implementation

```python
def is_valid_bst(root: TreeNode) -> bool:
    """Validate using (low, high) bounds. O(n) time, O(h) space."""
    def valid(node, low, high):
        if node is None:
            return True
        if not (low < node.val < high):     # must lie strictly within bounds
            return False
        return (valid(node.left, low, node.val) and      # tighten upper bound
                valid(node.right, node.val, high))        # tighten lower bound
    return valid(root, float("-inf"), float("inf"))
```

**Complexity:** Time O(n), Space O(h).

### Alternative Solution

In-order traversal must be strictly increasing:
```python
def is_valid_bst_inorder(root):
    prev = float("-inf")
    stack, node = [], root
    while stack or node:
        while node:
            stack.append(node); node = node.left
        node = stack.pop()
        if node.val <= prev:        # not strictly increasing → invalid
            return False
        prev = node.val
        node = node.right
    return True
```

### Common Follow-up Questions

- *Why isn't comparing parent to child enough?* A node deep in the left subtree could be larger than an ancestor while still being smaller than its immediate parent — only the propagated range catches it.

---

# Section K — Graphs (BFS & DFS)

## Foundation

```python
from collections import deque, defaultdict

class Graph:
    """Undirected graph via adjacency list. Use directed by omitting the reverse edge."""
    def __init__(self):
        self.adj = defaultdict(list)

    def add_edge(self, u, v):
        self.adj[u].append(v)
        self.adj[v].append(u)        # remove this line for a directed graph
```

## Practical Question 1 — BFS Traversal

**Difficulty:** Medium
**Estimated Time:** 15 minutes
**Concepts Tested:** queue-based traversal, visited set

**Problem Statement**
Return the BFS traversal order of a graph starting from a given source.

### Python Implementation

```python
def bfs(graph: Graph, start):
    """Breadth-first traversal. O(V + E) time, O(V) space."""
    visited = set([start])
    order = []
    q = deque([start])
    while q:
        node = q.popleft()
        order.append(node)
        for neighbor in graph.adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)      # mark at enqueue time (avoids dupes)
                q.append(neighbor)
    return order
```

**Complexity:** Time O(V + E), Space O(V).

### Common Follow-up Questions

- *Why mark visited at enqueue, not dequeue?* Marking when enqueuing prevents the same node from being added multiple times by different neighbors, avoiding duplicate work.

---

## Practical Question 2 — DFS (recursive and iterative)

**Difficulty:** Medium
**Estimated Time:** 15 minutes
**Concepts Tested:** recursion vs explicit stack, visited set

### Python Implementation

```python
def dfs_recursive(graph: Graph, start, visited=None, order=None):
    """Recursive DFS. O(V + E) time, O(V) space (call stack)."""
    if visited is None:
        visited, order = set(), []
    visited.add(start)
    order.append(start)
    for neighbor in graph.adj[start]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited, order)
    return order


def dfs_iterative(graph: Graph, start):
    """Iterative DFS with an explicit stack — safe for deep graphs."""
    visited, order = set(), []
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for neighbor in graph.adj[node]:
            if neighbor not in visited:
                stack.append(neighbor)
    return order
```

**Complexity:** Both O(V + E) time, O(V) space.

### Interview Variations

- **Shortest path** in an unweighted graph (BFS storing parents).
- **Number of connected components** (DFS/BFS from each unvisited node).
- **Cycle detection** (directed: recursion-stack colors; undirected: parent check).
- **Topological sort** (DFS post-order or Kahn's algorithm).

### Common Follow-up Questions

- *Recursive vs iterative DFS?* Same complexity; recursion is concise but risks stack overflow on deep graphs, so use the explicit-stack version for large inputs.
- *Why does DFS give a different order than BFS?* DFS dives deep first (LIFO stack); BFS spreads level by level (FIFO queue).

---

## Practical Question 3 — Shortest Path (Unweighted) with BFS

**Difficulty:** Medium
**Estimated Time:** 18 minutes
**Concepts Tested:** BFS shortest path, parent reconstruction

**Problem Statement**
Find the shortest path (fewest edges) between two nodes in an unweighted graph.

### Python Implementation

```python
def shortest_path(graph: Graph, start, goal):
    """BFS shortest path in an unweighted graph. O(V + E)."""
    if start == goal:
        return [start]
    visited = set([start])
    parent = {start: None}              # to reconstruct the path
    q = deque([start])
    while q:
        node = q.popleft()
        for neighbor in graph.adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                if neighbor == goal:    # found target → rebuild path
                    path = [goal]
                    while parent[path[-1]] is not None:
                        path.append(parent[path[-1]])
                    return path[::-1]
                q.append(neighbor)
    return None                          # unreachable
```

**Complexity:** Time O(V + E), Space O(V).

### Common Follow-up Questions

- *Why does BFS give the shortest path here?* It expands nodes in order of distance, so the first time it reaches the goal it has used the fewest edges. This only holds for unweighted graphs; weighted graphs need Dijkstra.
- *How to handle weights?* Replace the queue with a min-priority queue (heap) — that's Dijkstra's algorithm.

---

# Coding Questions Bank

> A curated bank spanning the syllabus, tagged by difficulty, with *why an
> interviewer asks each*. Attempt them after studying the sections above.

## Easy

1. **Reverse an array in place.** *Why:* tests two-pointer basics and in-place O(1) thinking.
2. **Find the max/min in an array.** *Why:* baseline traversal and handling empty input.
3. **Linear search / count occurrences.** *Why:* confirms you can reason about O(n) and edge cases.
4. **Check if a string is a palindrome.** *Why:* two pointers, character handling.
5. **Reverse a string / a linked list.** *Why:* the single most common warm-up; linked-list pointer skills.
6. **FizzBuzz / count vowels.** *Why:* filters candidates who can't translate logic to clean code.
7. **Sum of digits (recursion).** *Why:* base case + recursive case fluency.
8. **Remove duplicates from a sorted list/array.** *Why:* exploiting sorted structure for O(1) space.

## Medium

1. **Two-sum (hash map).** *Why:* the canonical "trade space for time" insight.
2. **Valid parentheses (stack).** *Why:* recognizing LIFO problems.
3. **Binary search + variants (first/last occurrence, rotated array).** *Why:* boundary precision.
4. **Merge two sorted lists/arrays.** *Why:* the merge step behind merge sort; pointer coordination.
5. **BST insert/search/delete.** *Why:* recursion + the three deletion cases.
6. **Validate a BST.** *Why:* the subtree-range gotcha separates memorizers from understanders.
7. **BFS/DFS traversal + connected components.** *Why:* graph modeling and visited-set discipline.
8. **Implement a queue with two stacks (or vice versa).** *Why:* amortized analysis + data-structure composition.
9. **Group anagrams.** *Why:* hashing with a canonical key.
10. **Detect a cycle in a linked list (Floyd's).** *Why:* the fast/slow pointer pattern.

## Hard

1. **LRU cache (hash map + doubly linked list).** *Why:* combining structures for O(1) get/put — a design favorite.
2. **Shortest path in a weighted graph (Dijkstra).** *Why:* priority queue + greedy correctness.
3. **Serialize/deserialize a binary tree.** *Why:* traversal mastery and edge handling.
4. **Topological sort with cycle detection.** *Why:* dependency ordering; real build-system logic.
5. **Kth largest element (heap or quickselect).** *Why:* partial ordering, average-case analysis.
6. **Word ladder (BFS on implicit graph).** *Why:* modeling a problem as a graph you never explicitly build.
7. **Merge k sorted lists (heap).** *Why:* scaling the merge idea with a priority queue.

---

# Exam & Viva Survival Tips

**Before you code**
- Restate the problem and confirm constraints (input size, duplicates, empty input, negatives).
- Give 1–2 example inputs/outputs, including an edge case.
- State your approach and its complexity *before* typing — examiners grade reasoning.

**While coding**
- Name things clearly; add a short comment per non-obvious line.
- Handle edge cases explicitly: empty structure, single element, not-found, overflow.
- Prefer the idiomatic tool: `collections.deque` for queues, `dict`/`set` for O(1) lookup, `heapq` for priority queues.

**After coding**
- Dry-run your code on the example and one edge case, line by line.
- State time and space complexity out loud and justify them.
- Mention a trade-off or alternative ("a hash map trades O(n) space for O(1) lookup").

**Viva rapid-fire readiness** — be able to answer in one breath:
- Array vs linked list; when each wins.
- Why binary search needs sorted data.
- LIFO vs FIFO and their traversals (stack→DFS, queue→BFS).
- Average vs worst case of a hash table, and why.
- Why a skewed BST is O(n) and how balancing fixes it.
- Amortized O(1): dynamic-array append and hash-table insert.
- Recursion needs a base case and costs O(depth) stack space.

**Complexity cheat-sheet to memorize**

| Operation | Array | Linked List | Hash Table | Balanced BST |
|---|---|---|---|---|
| Access | O(1) | O(n) | — | O(log n) |
| Search | O(n) | O(n) | O(1) avg | O(log n) |
| Insert | O(n)* | O(1)** | O(1) avg | O(log n) |
| Delete | O(n)* | O(1)** | O(1) avg | O(log n) |

`*` O(1) amortized at the end of a dynamic array. `**` at a known position/head.

> Pair this file with `theory.md`: understand the *why* there, prove you can
> *build it* here. Good luck — you've got this.



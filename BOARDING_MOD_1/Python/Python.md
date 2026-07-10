# Revise Python (basics, OOP, exceptions, advanced concepts) and solve coding exercises.

# Python Revision Roadmap

## Focus Areas
- Python Fundamentals
- Object-Oriented Programming (OOP)
- Exception Handling
- Advanced Python Concepts
- File Handling
- Functional Programming
- Iterators & Generators
- Coding Practice

---

# Week 1: Python Fundamentals

## Python Basics

### Topics
- Python syntax
- Variables
- Comments
- Indentation
- Keywords
- Input and output
- Dynamic typing
- Naming conventions (PEP 8)

---

## Data Types

### Primitive Data Types
- Integer (`int`)
- Float (`float`)
- Boolean (`bool`)
- String (`str`)
- None (`NoneType`)

### Type Conversion
- Implicit conversion
- Explicit conversion (`int()`, `float()`, `str()`, etc.)

---

## Operators

### Arithmetic Operators
- `+`
- `-`
- `*`
- `/`
- `//`
- `%`
- `**`

### Comparison Operators
- `==`
- `!=`
- `>`
- `<`
- `>=`
- `<=`

### Logical Operators
- `and`
- `or`
- `not`

### Assignment Operators
- `=`
- `+=`
- `-=`
- `*=`
- `/=`

### Membership Operators
- `in`
- `not in`

### Identity Operators
- `is`
- `is not`

---

## Control Flow

### Conditional Statements
- `if`
- `elif`
- `else`
- Nested conditions

### Loops
- `for`
- `while`
- Nested loops

### Loop Control
- `break`
- `continue`
- `pass`

---

# Functions

## Function Basics

### Topics
- Defining functions
- Calling functions
- Parameters
- Arguments
- Return values

---

## Function Arguments

- Positional arguments
- Keyword arguments
- Default arguments
- Variable-length arguments (`*args`)
- Keyword variable arguments (`**kwargs`)
- Positional-only parameters
- Keyword-only parameters

---

## Scope

- Local scope
- Global scope
- `global` keyword
- `nonlocal` keyword

---

## Lambda Functions

- Anonymous functions
- Using lambda with:
  - `map()`
  - `filter()`
  - `sorted()`

---

## Recursion

- Recursive functions
- Base case
- Recursive case

---

# Python Collections

## Lists

### Topics
- Creating lists
- Indexing
- Slicing
- List methods
- Nested lists
- List comprehension

---

## Tuples

### Topics
- Immutable sequences
- Packing
- Unpacking
- Tuple methods

---

## Dictionaries

### Topics
- Creating dictionaries
- Accessing values
- Updating values
- Dictionary methods
- Dictionary comprehension

---

## Sets

### Topics
- Creating sets
- Set operations
- Union
- Intersection
- Difference
- Symmetric difference

---

# Strings

## Topics
- String methods
- Formatting
- f-strings
- Splitting
- Joining
- Searching
- Replacing
- Regular Expressions (`re` module)

---

# File Handling

## Reading Files

- `open()`
- `read()`
- `readline()`
- `readlines()`

---

## Writing Files

- `write()`
- `writelines()`
- Append mode
- Context managers (`with` statement)

---

## Working with File Formats

- Text files
- CSV files
- JSON files

---

# Modules & Packages

## Topics
- Importing modules
- Creating modules
- Packages
- Standard Library
- Virtual Environments (`venv`)
- `pip`

Useful modules:
- `os`
- `pathlib`
- `sys`
- `math`
- `random`
- `datetime`
- `collections`
- `itertools`
- `functools`

---

# Week 2: Object-Oriented Programming (OOP)

## OOP Fundamentals

### Topics
- Classes
- Objects
- Attributes
- Methods
- Constructors (`__init__`)
- Instance variables
- Class variables

---

## Four Pillars of OOP

### Encapsulation
- Public members
- Protected members
- Private members
- Name mangling

### Abstraction
- Abstract classes
- Abstract methods
- `abc` module

### Inheritance
- Single inheritance
- Multiple inheritance
- Multilevel inheritance
- Hierarchical inheritance

### Polymorphism
- Method overriding
- Method overloading (Python approach)
- Duck typing

---

## Special (Magic/Dunder) Methods

Learn commonly used dunder methods:

- `__init__`
- `__str__`
- `__repr__`
- `__len__`
- `__eq__`
- `__lt__`
- `__add__`
- `__getitem__`
- `__setitem__`
- `__iter__`
- `__next__`
- `__call__`

---

## Class Relationships

- Composition
- Aggregation

---

## Properties

- `@property`
- Getter
- Setter
- Deleter

---

## Dataclasses

- `@dataclass`
- Default values
- Frozen dataclasses

---

# Week 3: Exception Handling

## Basics

### Topics
- Syntax errors
- Runtime errors
- Logical errors

---

## Exception Handling

### Topics
- `try`
- `except`
- `else`
- `finally`

---

## Raising Exceptions

- `raise`
- Built-in exceptions
- Custom exceptions

---

## Custom Exceptions

Create your own exception classes.

---

## Assertions

- `assert`
- Debugging with assertions

---

## Logging

- `logging` module
- Log levels
- File logging

---

# Week 4: Advanced Python

## Iterators

### Topics
- Iterable vs Iterator
- `iter()`
- `next()`
- Creating custom iterators

---

## Generators

### Topics
- Generator functions
- `yield`
- Generator expressions
- Lazy evaluation

---

## Decorators

### Topics
- Function decorators
- Nested functions
- Closures
- Parameterized decorators
- `functools.wraps`

---

## Context Managers

### Topics
- `with` statement
- Context manager protocol
- `contextlib`

---

## Functional Programming

### Topics
- `map()`
- `filter()`
- `reduce()`
- `zip()`
- `enumerate()`
- `any()`
- `all()`

---

## Comprehensions

- List comprehensions
- Dictionary comprehensions
- Set comprehensions
- Generator comprehensions

---

## Type Hints

### Topics
- Function annotations
- `typing` module
- `Optional`
- `Union`
- `Literal`
- `TypedDict`
- `Protocol`
- `Generic`

---

## Memory Management

### Topics
- Reference counting
- Garbage Collection (`gc`)
- Mutable vs Immutable objects
- Shallow copy vs Deep copy (`copy` module)

---

## Concurrency (Basics)

### Topics
- Threads
- Processes
- `threading`
- `multiprocessing`
- Introduction to `asyncio`
- `async` / `await`

---

## Useful Built-in Functions

Learn how to use:

- `enumerate()`
- `zip()`
- `sorted()`
- `reversed()`
- `sum()`
- `min()`
- `max()`
- `any()`
- `all()`
- `isinstance()`
- `type()`

---

# Coding Practice

## Beginner Problems
- Number problems
- String problems
- List problems
- Dictionary problems
- Loop problems
- Function problems

---

## Intermediate Problems
- File handling
- OOP design
- Exception handling
- Pattern printing
- Recursion
- Generators

---

## Advanced Problems
- Custom iterators
- Decorators
- Context managers
- Data processing
- Mini automation scripts
- Algorithmic challenges

---

# Recommended Standard Library Modules

- `os`
- `pathlib`
- `sys`
- `math`
- `random`
- `datetime`
- `time`
- `collections`
- `itertools`
- `functools`
- `re`
- `json`
- `csv`
- `logging`
- `typing`
- `dataclasses`
- `abc`
- `copy`
- `gc`
- `asyncio`

---

# Interview Revision Checklist

## Python Fundamentals
- Variables & Data Types
- Operators
- Control Flow
- Functions
- Collections
- Strings
- File Handling
- Modules

### Object-Oriented Programming
- Classes & Objects
- Encapsulation
- Abstraction
- Inheritance
- Polymorphism
- Magic Methods
- Properties
- Dataclasses

### Exception Handling
- `try` / `except`
- `else`
- `finally`
- `raise`
- Custom Exceptions
- Assertions
- Logging

### Advanced Python
- Iterators
- Generators
- Decorators
- Closures
- Context Managers
- Comprehensions
- Functional Programming
- Type Hints
- Memory Management
- Concurrency
- Standard Library

---



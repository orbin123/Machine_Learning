# Web Development with Django — Theory & Interview Preparation Guide

> **How to use this document.** This is a first-principles study guide for the Django / Django REST Framework / AWS deployment syllabus. Every topic follows the same skeleton: *What → Why → How → Internal working → Advantages → Limitations → Real-world use → Interview questions with model answers → Common mistakes → Related concepts.* Read it slowly. The goal is not to memorise definitions but to be able to **explain and defend** every concept in a viva, a written test, or a technical interview.
>
> **Mental model to carry throughout.** Django is a server-side web framework written in Python. A browser (or a mobile app, or `curl`) sends an **HTTP request**; Django routes it to a **view**; the view talks to the **database through the ORM**, renders a **template** or serialises **JSON**, and returns an **HTTP response**. Almost everything below is a detail of that one sentence.

---

# Table of Contents

1. [Django Fundamentals](#1-django-fundamentals)
2. [Django ORM, Models & Migrations](#2-django-orm-models--migrations)
3. [Views, Templates & URL Routing](#3-views-templates--url-routing)
4. [Authentication, Forms & Sessions](#4-authentication-forms--sessions)
5. [REST APIs & Django REST Framework](#5-rest-apis--django-rest-framework)
6. [Deployment & AWS](#6-deployment--aws)
7. [Advanced Django](#7-advanced-django)

---

# 1. Django Fundamentals

## 1.1 What is Django?

### What is it?

Django is a **high-level, open-source web framework written in Python** that lets you build database-driven websites and APIs quickly and safely. "High-level" means it hands you large, ready-made building blocks — an ORM, an authentication system, an admin panel, a templating engine, form handling, security protections — instead of making you assemble them yourself from low-level pieces.

The phrase you will hear constantly is **"batteries included."** It comes from the Python world's description of its own standard library. Applied to Django it means: for the *common* things every web application needs (talking to a database, handling users logging in, protecting against attackers, generating HTML), Django already ships a solution. You are not gluing together ten third-party libraries just to get a login form working.

Django was created at a newspaper company (the *Lawrence Journal-World*) around 2003–2005, where developers had to ship news applications on tight deadlines. That origin explains its personality: it optimises for **developer speed and correctness under deadline pressure**, and it is opinionated about the "right" way to structure a project.

### Why is it needed?

Imagine building a web application *without* a framework, using only Python's raw tools. You would have to:

- Parse incoming HTTP requests by hand (read headers, cookies, query strings, POST bodies).
- Write raw SQL for every database read and write, and manually protect every query against SQL injection.
- Manually manage user sessions and hash passwords securely.
- Hand-roll protection against CSRF, XSS, and clickjacking.
- Build your own URL-to-function routing.
- Write boilerplate for an admin interface so non-programmers can edit data.

This is thousands of lines of security-critical, error-prone plumbing **before you write a single line of your actual product**. Django exists to eliminate that plumbing so you can focus on the business logic that is unique to your app. It also encodes years of hard-won security knowledge, so a junior developer using Django gets protections (against SQL injection, XSS, CSRF) *by default* that an expert might forget to add manually.

### How does it work? (the request/response cycle)

At the heart of Django is one loop: **an HTTP request comes in, a response goes out.** Here is the journey of a single request:

```
  Browser
    │  HTTP GET /articles/42/
    ▼
┌──────────────────────────────────────────────────────┐
│  WSGI/ASGI server (Gunicorn/Uvicorn) hands request    │
│  to Django                                            │
└──────────────────────────────────────────────────────┘
    ▼
  MIDDLEWARE (security, sessions, auth, CSRF) — inbound
    ▼
  URL ROUTER (urls.py)  ── matches /articles/42/ ──► view
    ▼
  VIEW (your Python function/class)
    │   ├─► MODEL / ORM  ──►  DATABASE  (SELECT/INSERT…)
    │   └─► TEMPLATE ENGINE (renders HTML)   or   SERIALIZER (JSON)
    ▼
  HttpResponse  (status code + headers + body)
    ▼
  MIDDLEWARE (outbound)
    ▼
  Browser renders page / app parses JSON
```

Every feature you learn — models, views, templates, forms, DRF — is a component that plugs into a specific stage of that pipeline.

### Internal working

Django is fundamentally a **WSGI application**. WSGI (Web Server Gateway Interface) is a Python standard that defines how a web server talks to a Python web app: the server calls a Python callable, passing an `environ` dictionary (the request) and a `start_response` function, and the app returns an iterable of bytes (the response body). Django's `wsgi.py` exposes exactly such a callable. Newer async-capable Django also speaks **ASGI** (the asynchronous successor to WSGI) via `asgi.py`.

Internally, Django builds an `HttpRequest` object from the WSGI `environ`, pushes it through the middleware stack, resolves the URL to a **view callable**, invokes it, and expects an `HttpResponse` object back, which it converts to the WSGI response format. This clean "request object in, response object out" contract is why views are so easy to test.

### Advantages

- **Speed of development** — batteries included means less code to write.
- **Security by default** — protections against the OWASP top vulnerabilities are on unless you disable them.
- **Scalability** — the stateless request/response design scales horizontally; Instagram and Pinterest run on Django.
- **Mature ecosystem** — DRF, Celery, Channels, and thousands of packages.
- **Excellent documentation** — widely considered among the best in open source.
- **The ORM** — write Python, not SQL, while keeping the option to drop to raw SQL.
- **The admin** — a free, production-usable CRUD interface for your data.

### Limitations

- **Monolithic and opinionated** — great when you follow "the Django way," friction when you fight it.
- **The ORM can hide performance problems** — the N+1 query problem (covered later) is easy to introduce.
- **Historically synchronous** — async support exists but is younger and less complete than the sync path.
- **Not ideal for tiny microservices** — for a single-endpoint service, a micro-framework like Flask or FastAPI is lighter.
- **Templating is server-side** — for rich, app-like frontends you typically pair Django (as an API) with React/Vue.

### Real-world applications

Instagram (one of the largest Django deployments in the world), Pinterest, Disqus, Mozilla, The Washington Post, Spotify (backend services), and countless SaaS products and internal tools. Django is especially common for **content-heavy sites, admin-heavy internal tools, and REST/JSON backends** for mobile and single-page apps.

### Interview Questions

**Beginner**
1. What is Django and what problem does it solve?
2. What does "batteries included" mean?
3. Name three features Django provides out of the box.

**Intermediate**
4. Walk me through what happens when a request hits a Django app.
5. What is WSGI and how does Django relate to it?

**Advanced**
6. When would you *not* choose Django?
7. How does Django's synchronous heritage affect high-concurrency workloads, and how does ASGI change that?

**Scenario-based**
8. Your team must ship a CRUD-heavy internal dashboard in two weeks. Argue for or against Django.

**"Why" questions**
9. Why does Django ship an ORM instead of expecting you to write SQL?

**Comparison**
10. Django vs Flask — when would you pick each?

### Model Answers

**Q1 — What is Django and what problem does it solve?**
Django is a high-level Python web framework for building database-backed websites and APIs. The core problem it solves is that every serious web application needs the same plumbing — request parsing, database access, user authentication, security protections, an admin interface, URL routing — and writing that plumbing by hand is slow, repetitive, and dangerous because so much of it is security-critical. Django provides all of that as reusable, well-tested components ("batteries included"), so developers spend their time on the business logic unique to their product rather than reinventing login forms and SQL-injection defences. It also encodes secure defaults, which means even inexperienced developers get protection against common attacks automatically.

**Q4 — Walk me through what happens when a request hits a Django app.**
A web server (Gunicorn in production) receives the HTTP request and passes it to Django through the WSGI interface. Django wraps the raw request data into an `HttpRequest` object and sends it through the **middleware** stack — components that handle cross-cutting concerns like security headers, session loading, and authentication. Then the **URL resolver** matches the request path against the patterns in `urls.py` and identifies the correct **view**. The view is the heart of the logic: it typically queries the database via the **ORM/models**, then either renders an HTML **template** or serialises data to JSON. The view returns an `HttpResponse`, which travels back out through the middleware (now applying outbound processing) and is finally handed back to the web server, which sends the bytes to the browser. The key mental model is "request object in, response object out."

**Q6 — When would you *not* choose Django?**
I'd avoid Django when the project is a tiny, single-purpose microservice where its size is overhead — a lightweight framework like Flask or FastAPI boots faster and has less ceremony. I'd also reconsider it for workloads that are heavily real-time or extremely high-concurrency I/O-bound (thousands of open websockets), where an async-first framework like FastAPI or Node may fit more naturally, although Django Channels and ASGI narrow that gap. Finally, if the app is essentially a rich frontend with almost no server logic, a static site or a serverless function might be simpler. The trade-off is always: Django's "batteries" are an asset when you use many of them and a cost when you use almost none.

**Q10 — Django vs Flask.**
Flask is a *micro*-framework: it gives you routing and request handling and lets you choose everything else (ORM, forms, auth) as separate libraries. Django is a *batteries-included* framework that provides all of those, tightly integrated. Choose Flask when you want maximum flexibility, a small footprint, or an unusual architecture, and you're comfortable assembling and maintaining the pieces. Choose Django when you want to move fast on a conventional database-backed app, want a built-in admin and auth, and value convention over configuration. A common rule of thumb: Flask for small/experimental services, Django for full products and teams.

### Common Mistakes

- Thinking Django *is* the web server. It isn't — Gunicorn/Nginx serve it in production; `runserver` is dev-only.
- Believing security is automatic no matter what. Defaults are safe, but disabling CSRF, using `DEBUG=True` in production, or building raw SQL yourself removes those protections.
- Conflating Django with Django REST Framework — DRF is a separate package layered on top.

### Related Concepts

Flask, FastAPI, WSGI/ASGI, MVC/MVT, Gunicorn, Nginx, the HTTP request/response cycle.

---

## 1.2 MVC vs MVT Architecture

### What is it?

**MVC (Model–View–Controller)** and **MVT (Model–View–Template)** are ways of organising a web application into three responsibilities so that data, logic, and presentation don't get tangled together. Django uses a variant it calls **MVT**.

The classic MVC pieces:
- **Model** — the data and business rules (talks to the database).
- **View** — the presentation (what the user sees).
- **Controller** — the glue that takes user input, calls the model, and picks a view.

Django's MVT relabels these:
- **Model** — same idea: your data layer (Django models / ORM).
- **Template** — the presentation layer (HTML with placeholders). This is what MVC calls the "View."
- **View** — the logic layer that receives the request, talks to models, and chooses a template. This is what MVC calls the "Controller."

So the confusing bit is purely naming: **Django's "View" ≈ MVC's "Controller"**, and **Django's "Template" ≈ MVC's "View."**

### Why is it needed?

Separation of concerns. If your database queries, business logic, and HTML are all mashed into one file, the app becomes impossible to test, reuse, or change safely. A designer can't touch the HTML without risking the logic; a backend developer can't change a query without reading through markup. MVT enforces boundaries: models don't know about HTML, templates don't run database queries, and views coordinate between them. This makes code testable (you can test a view without a browser), reusable (the same model powers a web page and a JSON API), and maintainable.

### How does it work?

```
  Request ──► URL Router ──► VIEW (Django)  ── asks ──►  MODEL ──► DB
                                │                          │
                                │◄──────── data ───────────┘
                                ▼
                          TEMPLATE (HTML + data)
                                │
                                ▼
                            Response
```

The **View** is the orchestrator. Django's twist versus MVC is that the framework itself acts as the "controller wiring": the URL dispatcher decides which view runs, so you don't write a separate controller class — Django provides that layer for you. That's why Django's authors say MVT is really "MVC where the framework is the controller."

### Internal working

There is no special "MVT engine" — MVT is a *convention*, not a runtime component. Concretely: the URL resolver (`django.urls`) maps a path to a view callable; the view is plain Python; models are classes backed by the ORM; templates are rendered by the template engine (`django.template`). The pattern is enforced by project structure and Django's APIs, not by magic.

### Advantages

- Clear separation → easier testing, teamwork, and maintenance.
- The same model can feed HTML templates *and* JSON APIs (reuse).
- Designers and backend devs can work in parallel.

### Limitations

- The naming mismatch confuses newcomers coming from Rails/Spring MVC.
- For very small apps the ceremony can feel heavy.
- Business logic sometimes leaks into views ("fat views"); Django doesn't force a dedicated service/business layer, so teams must impose that discipline themselves.

### Real-world applications

Every Django project uses MVT. In practice mature teams add a **service layer** (plain Python modules holding business logic) so views stay thin and models stay focused on data — a widely used refinement of the base pattern.

### Interview Questions

**Beginner:** What are the three parts of MVT?
**Intermediate:** How does Django's "View" differ from MVC's "View"?
**Advanced:** Where does business logic belong in a Django app, and why do "fat models / thin views" and "service layers" both exist as answers?
**Comparison:** MVC vs MVT — is the difference real or just naming?
**Why:** Why did Django rename Controller to View and View to Template?

### Model Answers

**"How does Django's View differ from MVC's View?"**
In classic MVC, the *View* is the presentation layer — the thing rendered to the user. In Django, that presentation role is played by the **Template**, and the word **View** is used for the component that contains request-handling logic — receiving the request, querying models, and choosing which template to render. So Django's View corresponds to MVC's Controller, and Django's Template corresponds to MVC's View. The functional separation is the same as MVC; only the vocabulary differs, which is why Django describes its pattern as MVT and half-jokingly says "the framework is the controller," because the URL dispatcher handles the routing that a controller would otherwise do.

**"Where does business logic belong?"**
The classic Django advice is "fat models, thin views": put data-related business logic as methods on the model so it lives next to the data it operates on, and keep views focused on request/response coordination. This works well up to a point. As logic grows and starts spanning multiple models or external services, many teams introduce a dedicated **service layer** — plain Python modules/functions that orchestrate models — to avoid bloated models and to keep logic reusable and testable independent of the web layer. Both approaches share the same goal: keep views thin and don't scatter business rules into templates.

### Common Mistakes

- Assuming Django "View" renders HTML directly — it usually delegates to a template or serializer.
- Putting database queries or heavy logic inside templates.
- Never refactoring, ending up with 300-line views full of business logic.

### Related Concepts

Separation of concerns, MVC, service layer pattern, fat models/thin views, URL dispatching.

---

## 1.3 Why Django? Key Features & Ecosystem

### What is it?

This topic is about the *value proposition*: the concrete features that make Django worth choosing, and the ecosystem of tools built around it.

**Key built-in features:**
- **ORM** — database access in Python.
- **Automatic admin** — instant CRUD UI for your models.
- **Authentication system** — users, groups, permissions, password hashing.
- **Forms framework** — HTML form rendering + validation.
- **Template engine** — safe HTML generation.
- **URL routing** — clean, human-readable URLs.
- **Migrations** — version-controlled database schema changes.
- **Security** — CSRF, XSS, SQL-injection, and clickjacking protection by default.
- **Internationalisation, caching, sessions, email, testing tools.**

**Ecosystem (third-party but near-standard):**
- **Django REST Framework (DRF)** — the de-facto way to build APIs.
- **Celery** — background/async task processing.
- **Channels** — websockets and async.
- **pytest-django, factory_boy** — testing.
- **django-filter, drf-spectacular** — API filtering and docs.
- **Gunicorn/uWSGI + Nginx** — production serving.

### Why is it needed?

The features answer a simple question every engineer asks: *"What do I get for free, and what must I build?"* Django's answer is "most of the common stuff is free," which shortens time-to-market and reduces bugs. The ecosystem matters because no framework can do everything; Django's strength is that the *most common* extensions (APIs, background jobs, real-time) have mature, well-maintained packages that integrate cleanly.

### How does it work? (feature ↔ problem mapping)

| Feature | Problem it removes |
|---|---|
| ORM | Writing/maintaining raw SQL, injection risk |
| Admin | Building an internal CRUD UI from scratch |
| Auth system | Password hashing, login/logout, permissions |
| Forms | Rendering + validating user input safely |
| Migrations | Manually altering DB schema, keeping envs in sync |
| Security middleware | CSRF/XSS/clickjacking defence |
| DRF (ecosystem) | Serialisation, API views, versioning, browsable API |
| Celery (ecosystem) | Long-running work blocking the request cycle |

### Internal working

Most features are implemented as **apps** and **middleware** that you enable in `settings.py` (`INSTALLED_APPS`, `MIDDLEWARE`). For example, the admin is just a Django app (`django.contrib.admin`); auth is `django.contrib.auth`. This "everything is a pluggable app" design is itself a feature — you compose your project from apps, including your own.

### Advantages / Limitations / Real-world

Covered in 1.1. The distinctive point for interviews: Django's advantage is **integration** — the auth system, ORM, admin, and forms are designed to work together (e.g., the admin uses forms, which use models, which use the ORM). The limitation is that this integration assumes you build a fairly conventional app.

### Interview Questions & Model Answers

**Q: Name Django's killer features and why each matters.**
The standouts are the ORM (Python database access with injection safety), the automatic admin (a free internal CRUD tool that alone can justify choosing Django), the authentication system (secure password handling, sessions, permissions out of the box), the forms framework (safe input validation and rendering), and migrations (version-controlled schema evolution so environments stay in sync). Each removes a category of repetitive, error-prone work. The reason they matter *together* is integration — they're designed to compose, so, for instance, the admin uses the forms framework which uses your models which use the ORM. That coherence is what makes Django productive.

**Q: What is DRF and why is it separate from Django?**
Django REST Framework is a third-party package that adds everything needed to build JSON/REST APIs: serializers (convert models ↔ JSON with validation), API-specific views and viewsets, authentication classes (token, JWT via plugins), permissions, pagination, filtering, and a browsable API for testing. It's separate because Django's core is oriented toward server-rendered HTML; APIs are a distinct concern with their own patterns. Keeping DRF separate lets it evolve independently and keeps core Django lean, while still integrating tightly (DRF serializers mirror Django forms and models).

### Common Mistakes

- Reaching for third-party packages before checking if Django core already does it (sessions, caching, signals are built in).
- Treating DRF as part of core and being confused when it isn't installed.

### Related Concepts

INSTALLED_APPS, middleware, contrib apps, DRF, Celery, Channels.

---

## 1.4 Setting Up a Django Project

### What is it?

The standardised sequence of steps to go from an empty folder to a running Django app: install Python, create an **isolated virtual environment**, install Django, create a **project**, and create one or more **apps** inside it.

Terminology that trips people up:
- **Project** — the whole website/configuration (settings, root URLs). Created with `django-admin startproject`.
- **App** — a self-contained module of functionality (e.g. `blog`, `accounts`, `payments`) inside the project. Created with `python manage.py startapp`. A project contains many apps; a well-designed app can be reused across projects.

### Why is it needed?

**Virtual environments** exist because different projects need different, often conflicting, versions of packages. Without isolation, installing Django 5 for one project could break another project that needs Django 3. A venv gives each project its own private set of dependencies, so projects don't interfere and deployments are reproducible.

The **project/app split** exists to keep large codebases organised and to make functionality reusable. Instead of one giant module, you break the site into focused apps, each owning its own models, views, templates, and URLs.

### How does it work? (step by step)

```bash
# 1. Create and activate an isolated environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install Django into that environment
pip install django

# 3. Create the project (note the trailing dot to avoid an extra nested folder)
django-admin startproject config .

# 4. Create an app
python manage.py startapp blog

# 5. Register the app in settings.py -> INSTALLED_APPS

# 6. Run the dev server
python manage.py runserver
```

- `python -m venv venv` builds a folder with a private Python and `pip`.
- `activate` changes your shell so `python`/`pip` point inside the venv.
- `startproject` scaffolds `manage.py` and the settings package.
- `startapp` scaffolds `models.py`, `views.py`, `admin.py`, etc.
- You must **add your app to `INSTALLED_APPS`** or Django ignores its models, templates, and migrations.

### Internal working

A virtual environment is essentially a directory containing a Python interpreter (or a symlink to one) and its own `site-packages`. Activation prepends the venv's `bin` to your `PATH` and sets `VIRTUAL_ENV`, so package installs and imports resolve locally. `pip freeze > requirements.txt` records exact versions so the environment can be recreated elsewhere — the foundation of reproducible deployments.

### Advantages

- Reproducible, conflict-free dependencies.
- Clean separation of concerns via apps.
- Easy onboarding: clone repo → create venv → `pip install -r requirements.txt` → run.

### Limitations

- Beginners forget to activate the venv and install packages globally.
- Managing many venvs manually is tedious (tools like `pipenv`, `poetry`, `uv`, or Docker help).

### Real-world applications

Every professional Python project uses environment isolation, whether venv, Poetry, or a Docker container. In production the "environment" is often a Docker image that pins the exact dependency set.

### Interview Questions & Model Answers

**Q: What is a virtual environment and why is it important?**
A virtual environment is an isolated Python installation with its own package directory, so each project can have its own dependency versions without affecting others. It matters because real machines run many projects, and those projects often need conflicting versions of the same library. Without isolation, upgrading a package for one project can silently break another. Virtual environments also make deployments reproducible: `pip freeze` captures exact versions, so the same environment can be recreated on a server or in CI, eliminating "works on my machine" problems.

**Q: What's the difference between a Django project and an app?**
A **project** is the entire web application and its configuration — settings, the root URL configuration, WSGI/ASGI entry points. An **app** is a focused, self-contained component of functionality within the project, such as `accounts` or `orders`, each with its own models, views, templates, and migrations. One project contains many apps. The split exists for organisation and reusability: a well-designed app (say, a generic `comments` app) can be dropped into another project. Django itself ships apps this way — `django.contrib.admin` and `django.contrib.auth` are apps.

**Q: Why `startproject config .` with a dot?**
The trailing dot tells Django to place `manage.py` and the settings package in the *current* directory instead of creating an extra nested folder. Without it you get `myproject/myproject/`, a redundant layer that many teams find annoying. Naming the settings package `config` (or `core`) rather than repeating the project name is a common convention that keeps import paths stable even if the project is renamed.

### Common Mistakes

- Installing packages without activating the venv → pollutes the global Python.
- Forgetting to add the app to `INSTALLED_APPS`.
- Committing the `venv/` folder to git (commit `requirements.txt` instead).
- Confusing "project" and "app."

### Related Concepts

pip, requirements.txt, Poetry/uv, Docker, INSTALLED_APPS, reproducible builds.

---

## 1.5 Django Project Structure

### What is it?

The set of files Django generates, and the specific job each one does. Understanding these files is fundamental — interviewers love asking "what does `wsgi.py` do?"

```
myproject/
├── manage.py            # CLI entry point for admin commands
├── config/              # the "project" package
│   ├── __init__.py      # marks it a Python package
│   ├── settings.py      # ALL configuration
│   ├── urls.py          # root URL routing table
│   ├── wsgi.py          # sync production entry point
│   └── asgi.py          # async production entry point
└── blog/                # an "app"
    ├── models.py        # database tables as Python classes
    ├── views.py         # request-handling logic
    ├── admin.py         # register models with the admin
    ├── apps.py          # app configuration
    ├── migrations/      # generated schema-change files
    └── tests.py         # tests
```

### Why is it needed?

A predictable structure means any Django developer can open any Django project and immediately know where things live. It also cleanly separates **configuration** (the project package) from **features** (the apps).

### How does it work? — file by file

- **`manage.py`** — a thin wrapper around `django-admin` that also sets the `DJANGO_SETTINGS_MODULE` environment variable so commands know which settings to use. You run everything through it: `runserver`, `makemigrations`, `migrate`, `createsuperuser`, `shell`, `test`. It is your command-line remote control for the project.

- **`settings.py`** — the single source of truth for configuration: installed apps, middleware, database credentials, `DEBUG`, `ALLOWED_HOSTS`, `SECRET_KEY`, static/media paths, templates, auth backends, time zone. Django reads it at startup. In production this is usually split by environment (dev/prod) and secrets are pulled from environment variables.

- **`urls.py`** — the root **URLconf**: a list of URL patterns mapping paths to views (or to other apps' URLconfs via `include()`). It's the routing table Django consults on every request.

- **`wsgi.py`** — exposes the WSGI callable that synchronous production servers like **Gunicorn** import to run your app. WSGI = Web Server Gateway Interface, the standard contract between a web server and a Python app.

- **`asgi.py`** — the asynchronous equivalent, exposing an ASGI application for async servers (Uvicorn/Daphne) — needed for websockets, long-lived connections, and async views.

- **`models.py` / `views.py` / `admin.py` / `apps.py` / `migrations/`** — per-app files (covered in their own sections).

### Internal working

When any command runs, `manage.py` sets `DJANGO_SETTINGS_MODULE=config.settings` and calls `django.setup()`, which loads settings, configures logging, and populates the **app registry** by importing each app in `INSTALLED_APPS`. From that point Django "knows" all your models and can route requests. `wsgi.py`/`asgi.py` do the same setup, then expose the callable the server needs.

### Advantages / Limitations

- **Advantage:** universal, predictable layout; clear config-vs-feature separation.
- **Limitation:** the default single `settings.py` doesn't scale to multiple environments; teams split it into `settings/base.py`, `settings/prod.py`, etc., or use `django-environ`.

### Interview Questions & Model Answers

**Q: What is the difference between `wsgi.py` and `asgi.py`?**
Both are entry points that a production server uses to run your Django application, but they implement different interfaces. `wsgi.py` exposes a **WSGI** application — the traditional synchronous standard where the server calls Django once per request and waits for a response. It's what Gunicorn or uWSGI use, and it's perfect for typical request/response web apps. `asgi.py` exposes an **ASGI** application — the asynchronous standard that supports long-lived connections like websockets, server-sent events, and Django's async views. You'd run it under an ASGI server like Uvicorn or Daphne. In short: WSGI for standard sync workloads, ASGI when you need real-time or async I/O. Django generates both so you can choose per deployment.

**Q: What does `manage.py` do?**
It's the command-line entry point for your project. Technically it's a small script that sets the `DJANGO_SETTINGS_MODULE` environment variable to point at your settings and then delegates to Django's command framework. Through it you run management commands: start the dev server (`runserver`), generate and apply migrations (`makemigrations`, `migrate`), create an admin user (`createsuperuser`), open a Python shell with Django loaded (`shell`), and run tests. It's functionally the same as `django-admin` but pre-wired to your specific project's settings.

**Q: What lives in `settings.py` and how do you handle it in production?**
`settings.py` holds all project configuration: `INSTALLED_APPS`, `MIDDLEWARE`, database connection, `DEBUG`, `ALLOWED_HOSTS`, `SECRET_KEY`, static/media file settings, templates, and auth configuration. In production you never hardcode secrets or run with `DEBUG=True`. Instead you read sensitive values (`SECRET_KEY`, database URL, API keys) from **environment variables**, set `DEBUG=False`, and lock down `ALLOWED_HOSTS`. Many teams split settings into a base file plus environment-specific overrides, or use a library like `django-environ` to load a `.env` file, keeping secrets out of version control.

### Common Mistakes

- Editing `wsgi.py`/`asgi.py` unnecessarily (they rarely need changes).
- Hardcoding secrets in `settings.py` and committing them.
- Not understanding that `manage.py` and `django-admin` are essentially the same tool.

### Related Concepts

WSGI, ASGI, Gunicorn, Uvicorn, environment variables, settings splitting, the app registry.

---

## 1.6 Running the Development Server & Core Commands

### What is it?

The set of `manage.py` commands you use constantly during development, and specifically the built-in **development server** (`runserver`).

Core commands:
- `runserver` — start the lightweight dev web server.
- `startapp <name>` — scaffold a new app.
- `makemigrations` — turn model changes into migration files.
- `migrate` — apply migrations to the database.
- `createsuperuser` — create an admin account.
- `shell` — interactive Python with Django loaded.

### Why is it needed?

You need instant feedback while coding. `runserver` gives you a working web server with **auto-reload** (it restarts when you save a file) and **detailed error pages**, so the develop-test loop is fast. The migration commands let you evolve the database safely and repeatably.

### How does it work?

```bash
python manage.py runserver          # http://127.0.0.1:8000
python manage.py runserver 8080     # custom port
python manage.py makemigrations     # detect model changes -> migration files
python manage.py migrate            # apply migrations to DB
python manage.py createsuperuser    # admin login
```

`runserver` starts a single-threaded (by default) WSGI server, watches your source files, and reloads on change. It also serves static files in development.

### Internal working

`runserver` uses Python's `wsgiref` simple server wrapped with Django's auto-reloader. The reloader runs your app in a child process and watches file modification times (or uses `watchman` if available); on change it restarts the child. Crucially it is **single-process, not production-grade** — no proper concurrency, no security hardening, no performance tuning.

### Advantages

- Zero-config, instant feedback, auto-reload, rich error pages, serves static files.

### Limitations — **the classic interview trap**

`runserver` is **strictly for development.** It is single-threaded by default, not optimised, and not security-hardened. In production you must use a real WSGI/ASGI server (Gunicorn/Uvicorn) behind a reverse proxy (Nginx). Using `runserver` in production is a serious mistake.

### Interview Questions & Model Answers

**Q: Can you use `runserver` in production? Why or why not?**
No. `runserver` is a lightweight development server built for convenience — it auto-reloads on code changes and shows detailed error pages — but it is explicitly not built for production. It's single-threaded by default so it handles concurrency poorly, it isn't performance-optimised, and it hasn't undergone security audits for public exposure. In production you run Django through a production-grade WSGI server such as **Gunicorn** (or an ASGI server like Uvicorn for async), typically behind **Nginx** acting as a reverse proxy that also serves static files and terminates TLS. The Django docs themselves warn against using `runserver` for anything but local development.

**Q: What's the difference between `makemigrations` and `migrate`?**
`makemigrations` inspects your models, compares them against the current migration history, and **writes migration files** describing the schema changes (e.g. "add column `email` to table `user`"). It does *not* touch the database. `migrate` takes those migration files and **applies them to the actual database**, running the SQL to create or alter tables, and records which migrations have been applied in a `django_migrations` table. The two-step design means schema changes are version-controlled files you can review, commit, and apply consistently across dev, staging, and production.

### Common Mistakes

- Deploying with `runserver`.
- Running `migrate` before `makemigrations` and wondering why the schema didn't change.
- Forgetting to run `migrate` after pulling new migrations from teammates.

### Related Concepts

Gunicorn, Nginx, migrations, auto-reload, `manage.py shell`.

---

# 2. Django ORM, Models & Migrations

## 2.1 What is an ORM?

### What is it?

**ORM** stands for **Object–Relational Mapping**. It's a layer that lets you interact with a relational database (tables, rows, columns) using **objects and methods in your programming language** instead of writing SQL by hand. In Django, you define a Python class; the ORM maps that class to a database table, each instance to a row, and each attribute to a column. You call Python methods (`Article.objects.filter(...)`) and the ORM translates them into SQL, runs them, and turns the results back into Python objects.

```
   Python world                ORM (translator)              Database world
 ┌───────────────┐          ┌──────────────────┐          ┌───────────────┐
 │ class Article │  ──────► │  generates SQL   │ ──────►  │  TABLE article│
 │ .objects      │          │  runs it         │          │  rows/columns │
 │ .filter(...)  │  ◄────── │  maps rows→objects│ ◄────── │  result set   │
 └───────────────┘          └──────────────────┘          └───────────────┘
```

### Why is it needed?

Three big reasons:

1. **Productivity & readability.** `Article.objects.filter(published=True)` is shorter and clearer than the equivalent SQL, and it's Python, so it composes with the rest of your code.
2. **Database portability.** The same ORM code runs on PostgreSQL, MySQL, SQLite, or Oracle. The ORM emits the right dialect. Switching databases (e.g., SQLite in dev, PostgreSQL in prod) needs almost no code change.
3. **Safety.** The ORM **parameterises queries automatically**, which prevents SQL injection — one of the most common and dangerous web vulnerabilities. Hand-written string-concatenated SQL is a classic injection hole; the ORM closes it by default.

It also centralises your schema as Python classes, which powers migrations and the admin.

### How does it work?

You define a model, then use its **manager** (`.objects`) to build a **QuerySet**:

```python
# Define
class Article(models.Model):
    title = models.CharField(max_length=200)
    published = models.BooleanField(default=False)

# Query (CRUD)
Article.objects.create(title="Hello")             # CREATE  -> INSERT
Article.objects.all()                              # READ    -> SELECT *
Article.objects.filter(published=True)             # READ    -> SELECT ... WHERE
a = Article.objects.get(id=1); a.title = "New"; a.save()   # UPDATE -> UPDATE
a.delete()                                         # DELETE  -> DELETE
```

The ORM builds the SQL lazily and only hits the database when the QuerySet is actually evaluated (iterated, sliced, `len()`, etc.).

### Internal working

- **Managers & QuerySets.** `.objects` is a *manager*; calling methods on it returns a *QuerySet*, a lazy, chainable object that represents a query but hasn't run yet. This laziness lets you build up complex queries (`.filter().exclude().order_by()`) with a single database round-trip.
- **Lazy evaluation.** SQL executes only when results are needed. This is powerful but is also the source of the N+1 problem (see performance section).
- **SQL compilation.** When evaluated, the QuerySet is compiled to SQL by a database-specific backend, executed via a parameterised cursor (values passed separately from the SQL string — the injection defence), and each row is mapped into a model instance.
- **The Active Record-ish pattern.** Django models mix data + persistence methods (`.save()`, `.delete()`) on the instance, similar to the Active Record pattern.

### Advantages

- Faster development, readable queries, database portability, automatic injection protection, integrates with migrations/admin/forms.

### Limitations

- **Abstraction hides cost** — easy to write code that triggers hundreds of queries (N+1).
- **Complex queries** (heavy analytics, exotic joins, window functions) can be awkward; sometimes raw SQL is clearer/faster.
- **Performance ceiling** — for extreme performance you may need `.raw()` or hand-tuned SQL.
- **Leaky abstraction** — you still need to understand SQL and indexes to use it well.

### Real-world applications

Every Django app uses the ORM for its data layer. Teams use raw SQL (`Model.objects.raw()` or `connection.cursor()`) only for the rare query the ORM can't express efficiently — reporting, analytics, or performance-critical hotspots.

### Interview Questions

**Beginner:** What is an ORM? What is a QuerySet?
**Intermediate:** Why are Django QuerySets lazy? What benefit does that give?
**Advanced:** How does the ORM prevent SQL injection? When would you drop to raw SQL?
**Scenario:** A page runs 500 queries and is slow. How does the ORM cause this and how do you fix it?
**Comparison:** ORM vs raw SQL — trade-offs.
**Why:** Why might a senior engineer *distrust* the ORM on a reporting-heavy screen?

### Model Answers

**Q: What is an ORM and why use one?**
An ORM (Object–Relational Mapping) is a translation layer between your object-oriented code and a relational database. You define classes; the ORM maps them to tables, instances to rows, and attributes to columns, and it converts method calls like `Article.objects.filter(published=True)` into SQL, runs them, and returns Python objects. You use one for three reasons: productivity (Python queries are more concise and composable than raw SQL), portability (the same code targets PostgreSQL, MySQL, or SQLite), and safety (queries are parameterised automatically, which prevents SQL injection). The trade-off is that the abstraction can hide the true cost of queries, so you still need to understand the SQL it generates.

**Q: Why are QuerySets lazy and why does that matter?**
A QuerySet is lazy because it represents a query but doesn't execute it until the results are actually needed — when you iterate it, slice it, call `len()`, or convert it to a list. This matters for two reasons. First, efficiency: you can chain filters (`.filter(...).exclude(...).order_by(...)`) and Django builds a single optimised SQL statement, so complex query construction still results in one database round-trip. Second, composability: you can pass QuerySets around, add conditions conditionally, and only pay for the query at the moment of evaluation. The downside is that laziness can surprise you — accessing a related object inside a loop can silently fire a new query each iteration (the N+1 problem), which is why `select_related`/`prefetch_related` exist.

**Q: How does the ORM prevent SQL injection, and when would you still use raw SQL?**
SQL injection happens when user input is concatenated directly into a SQL string, letting an attacker inject their own SQL. The Django ORM prevents this by **parameterising** every query: the SQL text and the values are sent to the database separately, so user input is always treated as data, never as executable SQL. As long as you use the ORM's methods (or pass params to `raw()` properly), you're protected by default. You'd drop to raw SQL when a query is hard or inefficient to express through the ORM — complex analytical queries, certain window functions, or a performance hotspot where you want full control. Even then you use parameterised raw queries (`Model.objects.raw(sql, params)`), never string formatting, to keep the injection protection.

### Common Mistakes

- Iterating related objects in a loop → N+1 queries.
- Using `.get()` when zero or many rows may match (raises `DoesNotExist`/`MultipleObjectsReturned`).
- Building raw SQL with f-strings → reintroduces injection risk.
- Assuming a QuerySet is a list (it's lazy; re-evaluating hits the DB again).

### Related Concepts

QuerySet, manager, lazy evaluation, SQL injection, `select_related`/`prefetch_related`, migrations.

---

## 2.2 Creating Models: Fields, Field Types & Relationships

### What is it?

A **model** is a Python class that defines the structure of one database table. Each **field** (a class attribute) becomes a column, with a type that determines the SQL column type and validation rules. **Relationships** are special fields that model how tables connect: one-to-one, one-to-many, and many-to-many.

```python
class Author(models.Model):
    name = models.CharField(max_length=100)
    bio = models.TextField(blank=True)

class Article(models.Model):
    title = models.CharField(max_length=200)
    body = models.TextField()
    created = models.DateTimeField(auto_now_add=True)
    author = models.ForeignKey(Author, on_delete=models.CASCADE, related_name="articles")
```

### Why is it needed?

The model is the **single source of truth** for your data. From one class definition Django derives: the database table (via migrations), form fields, admin UI, serializer fields (with DRF), and validation. Defining the schema in Python — rather than separately in SQL — keeps everything in sync and version-controlled.

### How does it work? — field types & relationships

**Common field types**
| Field | Stores | SQL-ish type |
|---|---|---|
| `CharField(max_length=n)` | short text | VARCHAR(n) |
| `TextField` | long text | TEXT |
| `IntegerField` / `BigIntegerField` | integers | INTEGER |
| `BooleanField` | true/false | BOOLEAN |
| `DateTimeField` / `DateField` | timestamps | TIMESTAMP/DATE |
| `EmailField` | validated email | VARCHAR |
| `DecimalField(max_digits, decimal_places)` | exact decimals (money!) | DECIMAL |
| `FloatField` | floating point | FLOAT |
| `SlugField` | URL-safe strings | VARCHAR |
| `FileField` / `ImageField` | file path (file stored on disk/S3) | VARCHAR |
| `JSONField` | JSON data | JSON/JSONB |

**Common field options:** `null` (DB-level NULL allowed), `blank` (form-level empty allowed — *not the same as null*), `default`, `unique`, `db_index`, `choices`, `verbose_name`.

**Relationships**
- **One-to-Many** → `ForeignKey`. Put it on the "many" side. An `Article` has one `Author`; an `Author` has many `Article`s. Requires `on_delete` (what to do when the referenced row is deleted).
- **One-to-One** → `OneToOneField`. Like a ForeignKey with a uniqueness constraint. Used to extend a model, e.g., a `Profile` per `User`.
- **Many-to-Many** → `ManyToManyField`. Django creates a hidden **join/through table** behind the scenes. E.g., an `Article` has many `Tag`s and a `Tag` labels many `Article`s.

### Internal working

- A `ForeignKey` creates an integer column (`author_id`) plus a foreign-key constraint in the database. Accessing `article.author` triggers a query (unless prefetched).
- `on_delete` is enforced by Django (and often the DB): `CASCADE` (delete children too), `PROTECT` (block deletion), `SET_NULL` (null the reference), `SET_DEFAULT`, `DO_NOTHING`.
- A `ManyToManyField` creates a separate **join table** with two foreign keys. You can customise it with a `through` model to store extra data on the relationship (e.g., the date a tag was applied).
- **`null` vs `blank`:** `null` is about the *database* (can the column be NULL?); `blank` is about *validation/forms* (can the field be left empty in a form?). For text fields the convention is to avoid `null=True` (use empty string `""`) and only set `blank=True`.

### Advantages / Limitations

- **Advantage:** one declarative definition powers schema, validation, admin, and forms.
- **Limitation:** must run migrations after every change; some exotic constraints need raw SQL or `Meta.constraints`; relationship queries can be costly if not optimised.

### Real-world applications

Every entity in an app is a model: users, orders, products, payments, comments. Relationships model the domain: `Order → Customer` (FK), `User ↔ Role` (M2M), `User → Profile` (O2O).

### Interview Questions & Model Answers

**Q: Explain the three relationship types with examples.**
A **one-to-many** relationship is modelled with a `ForeignKey` placed on the "many" side: for example, many `Article`s belong to one `Author`, so `Article` has `author = ForeignKey(Author)`. A **one-to-one** relationship uses `OneToOneField` — it's a foreign key with a uniqueness constraint, commonly used to extend a model, like a `Profile` that has exactly one `User`. A **many-to-many** relationship uses `ManyToManyField`, where each side can relate to many of the other: an `Article` can have many `Tag`s and each `Tag` applies to many `Article`s. Under the hood Django creates a hidden join table with two foreign keys to implement the many-to-many; if you need to store extra data about the relationship itself, you supply a custom `through` model.

**Q: What is the difference between `null=True` and `blank=True`?**
They operate at different layers. `null=True` is a **database** setting — it allows the column to store SQL NULL. `blank=True` is a **validation** setting — it allows the field to be left empty in forms and the admin. They're independent: a field can be `blank=True, null=False` (required at DB level but forms handle emptiness differently) or vice versa. The common convention is that for string-based fields you avoid `null=True`, because then "no value" could be represented two ways (NULL or empty string `""`), which is ambiguous. So for text you typically use only `blank=True` and store an empty string; for non-text fields like dates or numbers, `null=True` is appropriate when the value is genuinely optional.

**Q: What does `on_delete` do and what are the options?**
`on_delete` tells Django what to do to rows that point at a record when that record is deleted — it's mandatory on `ForeignKey` because deleting a referenced row must have defined behaviour. `CASCADE` deletes the dependent rows too (delete an author → delete their articles). `PROTECT` raises an error and prevents deletion while references exist (good for data you must not lose accidentally). `SET_NULL` sets the foreign key to NULL (requires `null=True`), keeping the child but detaching it. `SET_DEFAULT` sets it to a default value. `DO_NOTHING` leaves it to the database (rarely used, can violate integrity). Choosing correctly matters: `CASCADE` on a critical relationship can silently wipe large amounts of data, so many teams prefer `PROTECT` for important references.

**Q: Why use `DecimalField` instead of `FloatField` for money?**
Because floats are binary floating-point and can't represent many decimal fractions exactly — `0.1 + 0.2` isn't exactly `0.3`. For money that rounding error is unacceptable; it compounds and causes off-by-a-cent bugs and failed reconciliations. `DecimalField` stores exact base-10 values with a fixed number of digits and decimal places, so monetary arithmetic is precise. The rule is: use `DecimalField` for money and any value where exactness matters, and reserve `FloatField` for scientific/approximate quantities.

### Common Mistakes

- Using `FloatField` for currency.
- Setting `null=True` on `CharField`/`TextField`.
- Forgetting `on_delete`, or defaulting everything to `CASCADE` without thinking.
- Not adding `related_name`, then being stuck with the default reverse accessor names.

### Related Concepts

Migrations, `related_name`, `through` models, database constraints, normalization, `select_related`/`prefetch_related`.

---

## 2.3 Model Features: Methods, Meta, `__str__`

### What is it?

Beyond fields, models can carry **behaviour** and **configuration**:
- **Model methods** — regular Python methods that encapsulate business logic tied to a row (e.g., `order.total_price()`).
- **`__str__`** — the human-readable string representation of an instance (shown in the admin and shell).
- **`Meta` class** — an inner class holding model-level configuration: default ordering, table name, constraints, unique-together, verbose names, indexes.

```python
class Article(models.Model):
    title = models.CharField(max_length=200)
    published = models.BooleanField(default=False)
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created"]            # newest first by default
        verbose_name_plural = "articles"
        indexes = [models.Index(fields=["title"])]

    def __str__(self):
        return self.title

    def is_recent(self):                   # business logic on the model
        return (timezone.now() - self.created).days < 7
```

### Why is it needed?

- **`__str__`** makes objects readable everywhere (admin lists, debugging, logs). Without it you see unhelpful `<Article: Article object (1)>`.
- **Model methods** implement the "fat models, thin views" philosophy: business logic lives next to the data it uses, making it reusable and testable.
- **`Meta`** centralises table-level rules (ordering, uniqueness, indexes) so they're declared once and enforced consistently.

### How does it work / Internal working

- `__str__` is just Python's standard string dunder; Django calls it in the admin and shell.
- Model methods run in Python; if they only combine existing fields they're free, but if they trigger queries they can cause N+1 issues when called in loops.
- `Meta.ordering` adds an `ORDER BY` to *every* default query for that model (convenient, but a subtle performance cost if the column isn't indexed).
- `Meta.constraints` and `unique_together`/`UniqueConstraint` create real database constraints via migrations, enforcing integrity at the DB level.

### Advantages / Limitations

- **Advantage:** cleaner, DRY, self-documenting models; DB-enforced integrity.
- **Limitation:** `Meta.ordering` on unindexed columns hurts performance; heavy logic in model methods can hide query costs; over-stuffing models leads to "god models" (hence service layers for cross-model logic).

### Interview Questions & Model Answers

**Q: What is `__str__` and why define it?**
`__str__` defines the human-readable representation of a model instance. Django calls it in the admin interface, the shell, logs, and anywhere the object is coerced to a string. Defining it — e.g. returning `self.title` for an `Article` — means lists and dropdowns show meaningful labels like "Django Basics" instead of the default `Article object (1)`. It's a small thing that hugely improves the usability of the admin and the debuggability of your app, which is why it's considered essential on every model.

**Q: What goes in the `Meta` class?**
The `Meta` inner class holds model-level configuration that isn't a field. Common entries: `ordering` (default sort order applied to queries), `db_table` (override the table name), `verbose_name`/`verbose_name_plural` (human labels for the admin), `unique_together` or the newer `constraints` with `UniqueConstraint` (multi-column uniqueness), `indexes` (database indexes for performance), and `abstract = True` (make it a base class that doesn't create its own table). It's where you express table-wide rules once so they apply everywhere and get baked into migrations.

**Q: Where should business logic live — the model, the view, or somewhere else?**
The Django convention is "fat models, thin views": logic that operates on a single record's data belongs as a **method on the model**, close to the data, so it's reusable and testable without the web layer. Views should stay thin, just coordinating request and response. As logic grows to span multiple models or call external services, it's common to move it into a dedicated **service layer** — plain Python functions/modules — to avoid bloated models. What you should *not* do is put business logic in templates or duplicate it across views.

### Common Mistakes

- Not defining `__str__`.
- Putting `Meta.ordering` on an unindexed column and paying a hidden sort cost on every query.
- Cramming unrelated logic into one giant model.

### Related Concepts

Fat models, service layer, database constraints, indexes, `__repr__`.

---

## 2.4 Migrations & Schema Evolution

### What is it?

**Migrations** are Django's system for **version-controlling your database schema.** When you change a model (add a field, change a type, add an index), Django generates a **migration file** — a Python file describing the change — which you then **apply** to the database. Migrations are how the database schema stays in sync with your models across every developer's machine and every deployment.

Two commands do the work:
- `makemigrations` — compares your models to the last known state and **writes** migration files describing the difference.
- `migrate` — **executes** those migration files against the database and records what's been applied.

### Why is it needed?

Databases are stateful. When you add a `phone` field to `User`, the running database doesn't magically grow a column. Someone has to run `ALTER TABLE`. Doing that manually is error-prone and impossible to keep consistent across dev, staging, production, and every teammate. Migrations solve this by making schema changes **declarative, ordered, version-controlled files** that anyone can apply to reach the same schema. They also make changes reversible and reviewable in code review.

### How does it work?

```
Change model  ──►  python manage.py makemigrations  ──►  0002_add_phone.py (a file)
                                                               │
                                    git commit + push          │
                                                               ▼
On any machine ──►  python manage.py migrate  ──►  runs ALTER TABLE, records it
```

Each migration file lists **operations** (`AddField`, `CreateModel`, `AlterField`, `RunPython` for data migrations) and its **dependencies** (which migration must run first), forming an ordered chain per app.

### Internal working

- Django keeps a table called **`django_migrations`** recording every migration that has been applied. On `migrate`, it compares this record to the migration files on disk and runs only the unapplied ones, in dependency order.
- `makemigrations` computes the difference by building an in-memory "project state" from the existing migration files and diffing it against your current models — it does **not** read the live database.
- Migrations can be **schema migrations** (structure) or **data migrations** (using `RunPython` to transform existing rows, e.g., backfilling a new column).
- Migrations are per-app and can depend on migrations in other apps (e.g., a FK to another app's model), which Django orders automatically.

### Advantages

- Reproducible schema across all environments; reversible changes; changes reviewed in code review; supports data transformations; team-friendly.

### Limitations

- **Merge conflicts** when two people create migrations on the same app in parallel (resolved with `makemigrations --merge`).
- **Large-table migrations** can lock tables and cause downtime; production needs care (e.g., adding columns with defaults, or using tools/strategies for zero-downtime).
- Auto-generated migrations occasionally need manual editing (renames, complex data moves).

### Real-world applications

Every schema change in every Django project. In CI/CD pipelines, `migrate` runs automatically on deploy. Data migrations backfill values, split/merge tables, or fix historical data.

### Interview Questions & Model Answers

**Q: Explain the difference between `makemigrations` and `migrate` again, precisely.**
`makemigrations` detects changes in your models by diffing them against the recorded migration history (not the live database) and writes migration files describing those changes. It's a purely local, offline operation that produces files. `migrate` reads the migration files, determines which haven't yet been applied by consulting the `django_migrations` table in the database, and executes the corresponding SQL (creating/altering tables), then records them as applied. So `makemigrations` = "author the change as a file," `migrate` = "apply the file to the actual database." You commit the files so everyone applies identical changes.

**Q: What is a data migration and when do you need one?**
A data migration changes *data* rather than *structure*. You need one when a schema change requires transforming existing rows — for example, you add a new non-null `slug` field and must populate slugs for all existing articles, or you split a `full_name` column into `first_name`/`last_name`. You create it with an empty migration and a `RunPython` operation that contains a forward function (and ideally a reverse function) doing the transformation through the ORM. Data migrations run in the same ordered chain as schema migrations, so you can sequence "add column → backfill data → make column non-null" safely.

**Q: How do you handle a migration that would lock a huge table in production?**
The concern is that operations like adding a non-null column with a default, or adding an index, can lock the table and block writes, causing downtime. The strategy is to break the change into safe steps: add the column as nullable (fast, no rewrite), backfill values in batches via a data migration or background job, then add the not-null constraint and default separately. For indexes, PostgreSQL supports `CREATE INDEX CONCURRENTLY`, which Django can emit via `AddIndexConcurrently`. The general principle is zero-downtime migrations: make each step individually non-blocking and backwards-compatible so old and new code can both run during the deploy.

**Q: How do you resolve migration conflicts between two developers?**
When two developers branch off the same migration state and each adds a migration to the same app, you get two "leaf" migrations with the same parent — a conflict. Django detects this and you resolve it with `python manage.py makemigrations --merge`, which creates a **merge migration** that depends on both leaves, reconciling the branch. As long as the two migrations don't touch the same column in incompatible ways, this is safe. If they do conflict semantically, you edit the migrations manually. The habit that prevents pain is communicating about schema changes and rebasing/merging migrations promptly.

### Common Mistakes

- Editing the database manually instead of via migrations → drift.
- Forgetting to commit migration files.
- Running `migrate` in production without reviewing what a migration will do to large tables.
- Deleting/altering old applied migrations that others already ran.

### Related Concepts

`django_migrations` table, `RunPython` data migrations, zero-downtime deploys, `sqlmigrate` (see the SQL a migration will run), schema drift.

---

# 3. Views, Templates & URL Routing

## 3.1 Function-Based Views (FBVs)

### What is it?

A **view** in Django is the code that runs when a URL is requested. It receives an `HttpRequest` and returns an `HttpResponse`. A **Function-Based View (FBV)** is simply a Python *function* that does this:

```python
from django.shortcuts import render, get_object_or_404
from .models import Article

def article_detail(request, pk):
    article = get_object_or_404(Article, pk=pk)
    return render(request, "articles/detail.html", {"article": article})
```

The contract is dead simple: **function takes `request` (plus any URL parameters), returns a response.**

### Why is it needed?

Something has to turn a request into a response — fetch data, apply logic, decide what to send back. FBVs are the most explicit, readable way to do that. Everything the view does is written out in plain Python, top to bottom, so there's no hidden behaviour. For beginners and for simple or highly custom logic, FBVs are the clearest choice.

### How does it work?

1. The URL resolver matches the path and calls the view function, passing `request` and any captured URL kwargs (like `pk`).
2. The function inspects `request` (method, GET/POST data, user), does its work (usually querying models), and builds a response — often with `render()` (HTML) or `JsonResponse` (JSON).
3. It returns the response, which Django sends back through middleware to the client.

Handling different HTTP methods is done with `if request.method == "POST":` branches.

### Internal working

An FBV is literally the callable stored against a URL pattern. When matched, Django calls `view(request, *args, **kwargs)`. There's no class instantiation, no method resolution — just a function call, which is why FBVs are the most transparent view type. Decorators (`@login_required`, `@require_POST`) wrap the function to add behaviour.

### Advantages

- **Explicit and readable** — no hidden inheritance; what you see is what runs.
- **Easy to learn** and to debug.
- **Great for custom, one-off logic** that doesn't fit a standard pattern.
- Simple decorator-based composition.

### Limitations

- **Repetitive** — standard CRUD (list, detail, create, update, delete) means writing similar code again and again.
- **No inheritance** — you can't easily share behaviour across views except via decorators/helpers.
- More boilerplate for common patterns than CBVs.

### Interview Questions & Model Answers

**Q: What is a view in Django, fundamentally?**
Fundamentally, a view is a callable that takes an `HttpRequest` and returns an `HttpResponse`. It's the layer where request handling logic lives: it reads the request (method, parameters, authenticated user), typically queries the database through models, and then produces a response — rendered HTML via a template, a redirect, or JSON. In MVT terms, the view is the "controller" role: it coordinates between the data (models) and the presentation (templates). A Function-Based View expresses this as a plain function; a Class-Based View expresses it as a class, but the request-in/response-out contract is identical.

**Q: When would you prefer an FBV over a CBV?**
I'd prefer an FBV when the logic is simple, unusual, or highly custom — for example, a view with a lot of branching, a one-off endpoint, or logic that doesn't map onto Django's standard CRUD operations. FBVs are more explicit: everything executes top-to-bottom with no inheritance to trace, which makes them easier to read and debug. The trade-off is repetition — if I'm writing standard list/detail/create/update/delete views, CBVs (especially generic ones) remove a lot of boilerplate. So my rule is: FBVs for bespoke or simple logic, CBVs for conventional CRUD.

### Common Mistakes

- Forgetting to handle both GET and POST in a form view.
- Returning data instead of an `HttpResponse` object.
- Not using `get_object_or_404`, leading to unhandled `DoesNotExist` exceptions.

### Related Concepts

CBVs, decorators (`login_required`, `require_http_methods`), `render`, `HttpResponse`, middleware.

---

## 3.2 Class-Based Views (CBVs) & Generic Views

### What is it?

A **Class-Based View (CBV)** implements a view as a Python *class* instead of a function. Django provides a hierarchy of ready-made **generic CBVs** that implement common patterns so you write almost no code:

- `View` — the base class; you define `get()`, `post()` methods.
- `TemplateView` — render a template.
- `ListView` — list objects of a model.
- `DetailView` — show one object.
- `CreateView` / `UpdateView` / `DeleteView` — form handling for create/edit/delete.

```python
from django.views.generic import ListView, DetailView

class ArticleListView(ListView):
    model = Article
    template_name = "articles/list.html"
    context_object_name = "articles"
    paginate_by = 10

class ArticleDetailView(DetailView):
    model = Article
```

The `ListView` above replaces ~15 lines of FBV (query, paginate, render) with a few attributes.

### Why is it needed?

Web apps repeat the same CRUD patterns endlessly: "list these objects," "show this object," "handle this create form." Rewriting that logic per view is wasteful and bug-prone. Generic CBVs encapsulate these patterns as reusable, configurable classes. You customise via attributes (`model`, `template_name`, `paginate_by`) and by overriding methods (`get_queryset`, `get_context_data`, `form_valid`). Because they're classes, you can **inherit and mix** behaviour (via **mixins** like `LoginRequiredMixin`).

### How does it work? — request dispatch

CBVs use a class method `as_view()` in the URLconf, which returns a function Django can call. On a request, that function creates an instance and calls **`dispatch()`**, which looks at the HTTP method and routes to the matching handler — `get()` for GET, `post()` for POST, etc.

```
URL  ──► ArticleListView.as_view()  ──►  dispatch()  ──►  get()/post()  ──► response
```

### Internal working

- `as_view()` returns a closure that, per request, instantiates the class and calls `dispatch()`.
- `dispatch()` inspects `request.method` and calls `self.get`, `self.post`, etc.
- Generic views are built by **composing mixins**: e.g. `ListView` = `MultipleObjectTemplateResponseMixin` + `BaseListView` + others. Each mixin contributes one piece (getting the queryset, paginating, picking the template, building context). This is method-resolution-order (MRO) composition — powerful but the reason CBVs can be hard to follow.
- Customisation points are overridable methods: `get_queryset()` (which objects), `get_context_data()` (extra template variables), `form_valid()` (what to do on valid form submit).

### Advantages

- **DRY** — standard CRUD in a few lines.
- **Reusable & extensible** via inheritance and mixins (`LoginRequiredMixin`, `PermissionRequiredMixin`).
- Consistent structure across a codebase.

### Limitations

- **Steeper learning curve** — the mixin/MRO machinery is implicit; behaviour comes from parent classes you can't see in the file.
- **Harder to debug** — you often need the docs or `ccbv.co.uk` to know which method to override.
- **Over-abstraction risk** — heavily customised CBVs can become harder to read than an equivalent FBV.

### Real-world applications

CBVs (especially `ListView`/`DetailView`/`CreateView`) are the default for standard CRUD web pages. DRF's viewsets are the API-world analogue of generic CBVs.

### Interview Questions & Model Answers

**Q: FBV vs CBV — compare them.** *(A checklist favourite.)*
Both satisfy the same contract — request in, response out — but differ in structure and trade-offs. **FBVs** are plain functions: explicit, linear, easy to read and debug, ideal for custom or simple logic, but repetitive for standard CRUD and hard to share behaviour beyond decorators. **CBVs** are classes: they use inheritance and mixins to encapsulate common patterns, so Django's generic views (`ListView`, `CreateView`, etc.) let you implement standard CRUD with very little code, and you can compose behaviour like authentication via mixins. The cost is a steeper learning curve and less obvious control flow, because behaviour is inherited from parent classes rather than written in front of you. The practical guideline: use generic CBVs for conventional CRUD to stay DRY, and FBVs when logic is bespoke or when readability matters more than reuse.

**Q: How does a CBV actually get called for a request?**
In the URLconf you register `MyView.as_view()`, not the class itself. `as_view()` returns a view function. When a request matches, Django calls that function, which instantiates the class fresh for that request and calls its `dispatch()` method. `dispatch()` looks at the HTTP method — GET, POST, etc. — and calls the correspondingly named method (`get()`, `post()`). Those methods do the work and return a response. This is why a fresh instance is created per request (so it's safe to set instance attributes) and why the method-based dispatch lets you cleanly separate GET and POST handling into different methods instead of `if request.method` branches.

**Q: What are mixins and why do CBVs use them?**
A mixin is a small class that provides a focused piece of behaviour meant to be combined with others through multiple inheritance. CBVs use mixins to compose functionality: `ListView`, for instance, is assembled from mixins that handle fetching a queryset, paginating, selecting a template, and building context. This lets Django (and you) reuse behaviour across many views — a great example is `LoginRequiredMixin`, which you add to any CBV to require authentication without rewriting the check. The benefit is reuse and separation of concerns; the cost is that the real behaviour is spread across several parent classes, so understanding a view means understanding its method resolution order.

### Common Mistakes

- Registering the class (`MyView`) in urls instead of `MyView.as_view()`.
- Overriding the wrong method (e.g., putting query logic in `get()` instead of `get_queryset()`).
- Fighting a generic view with heavy overrides when an FBV would be clearer.
- Putting `LoginRequiredMixin` *after* the generic view in the inheritance list (order matters — mixin must come first).

### Related Concepts

`as_view()`, `dispatch()`, mixins, MRO, `get_queryset`/`get_context_data`/`form_valid`, DRF viewsets.

---

## 3.3 Request & Response Objects

### What is it?

Every view receives an **`HttpRequest`** object and must return an **`HttpResponse`** (or subclass). These objects are Django's Python representation of the raw HTTP message.

**`HttpRequest` key attributes:**
- `request.method` — `"GET"`, `"POST"`, etc.
- `request.GET` — query-string params (`?q=django` → `request.GET["q"]`).
- `request.POST` — submitted form data.
- `request.body` — raw request body (used for JSON APIs).
- `request.user` — the logged-in user (added by auth middleware).
- `request.FILES` — uploaded files.
- `request.headers`, `request.COOKIES`, `request.session`.

**Response types:**
- `HttpResponse("text")` — basic response.
- `render(request, template, context)` — render HTML.
- `JsonResponse({...})` — JSON (sets `Content-Type: application/json`).
- `redirect("url-name")` — 302 redirect.
- `HttpResponse(status=404)` and helpers like `HttpResponseNotFound`.

### Why is it needed?

HTTP is just text over a socket. Working with raw text is painful and unsafe. Django parses the raw request into a structured, convenient object (so you access `request.GET["q"]` instead of parsing the query string yourself) and lets you build responses with helpers that set the right status codes and headers. It's the ergonomic, safe boundary between raw HTTP and your Python logic.

### How does it work?

```python
def search(request):
    if request.method == "GET":
        query = request.GET.get("q", "")          # read query param safely
        results = Article.objects.filter(title__icontains=query)
        return render(request, "search.html", {"results": results})

def api_create(request):
    data = json.loads(request.body)               # raw JSON body
    article = Article.objects.create(**data)
    return JsonResponse({"id": article.id}, status=201)
```

### Internal working

- Django constructs `HttpRequest` from the WSGI/ASGI `environ`/scope, lazily parsing `GET`/`POST`/`body` on access.
- `request.user` and `request.session` are *added by middleware* (AuthenticationMiddleware, SessionMiddleware) — they aren't there by default, which is why middleware order matters.
- `HttpResponse` holds a status code, headers dict, and content; Django serialises it back to the HTTP wire format.
- `GET` and `POST` are `QueryDict` objects (immutable, multi-value capable).

### Advantages / Limitations

- **Advantage:** clean, safe, structured access to HTTP; convenient response builders.
- **Limitation:** `request.POST` only holds form-encoded data — JSON bodies must be read from `request.body` and parsed (DRF handles this for you). Confusing `GET`/`POST`/`body` is a common bug source.

### Interview Questions & Model Answers

**Q: How do you access query parameters vs form data vs a JSON body?**
Query-string parameters (from `?key=value` in the URL) are in `request.GET`, accessed like `request.GET.get("key")`. Form-encoded POST data (from an HTML form submit) is in `request.POST`. A raw JSON request body — common for APIs — is *not* in `request.POST`; it's in `request.body` as bytes, which you parse with `json.loads(request.body)`. This trips people up: submitting JSON and then reading `request.POST` returns nothing. In practice, when building APIs you use Django REST Framework, which unifies this by exposing `request.data` regardless of the content type, so you don't juggle `GET`/`POST`/`body` manually.

**Q: Where does `request.user` come from?**
`request.user` is not part of the raw HTTP request — it's attached by Django's **AuthenticationMiddleware**, which runs on every request, reads the session (loaded by SessionMiddleware) to find the user's ID, and looks up the user, setting `request.user` to that `User` instance or to an `AnonymousUser` if nobody is logged in. This is why middleware order matters: AuthenticationMiddleware must come after SessionMiddleware, because it depends on the session being loaded first. It also means in a plain WSGI context without those middlewares, `request.user` wouldn't exist.

**Q: What's the difference between `HttpResponse`, `JsonResponse`, and `render`?**
`HttpResponse` is the base response — you give it a body string and optionally a status code and headers. `render` is a shortcut that loads a template, fills it with a context dictionary, and returns an `HttpResponse` containing the generated HTML — it's what you use for server-rendered pages. `JsonResponse` is a specialised `HttpResponse` that serialises a Python dict to JSON and sets the `Content-Type` header to `application/json` — it's what you use for simple API endpoints without DRF. They all ultimately produce an `HttpResponse`; the differences are convenience and the content type they target.

### Common Mistakes

- Reading a JSON body from `request.POST` (it's empty; use `request.body`).
- Assuming `request.user` exists without auth middleware.
- Returning a bare string/dict from a view instead of a response object.
- Forgetting to set a proper status code (e.g., 201 for created).

### Related Concepts

QueryDict, middleware, `request.data` (DRF), status codes, CSRF token in POST.

---

## 3.4 Django Templates

### What is it?

The **Django Template Language (DTL)** is a small, deliberately restricted language for generating text output — almost always HTML — by mixing static markup with dynamic data. A template is an HTML file with **placeholders** and **logic tags** that Django fills in at render time.

Two kinds of markup:
- **Variables:** `{{ article.title }}` — output a value.
- **Tags:** `{% if %}`, `{% for %}`, `{% block %}`, `{% include %}`, `{% url %}` — logic/structure.
- **Filters:** `{{ name|upper }}`, `{{ price|floatformat:2 }}` — transform a value on output.

```django
<h1>{{ article.title }}</h1>
{% if article.published %}
  <p>Published on {{ article.created|date:"M d, Y" }}</p>
{% endif %}
<ul>
  {% for tag in article.tags.all %}
    <li>{{ tag.name }}</li>
  {% empty %}
    <li>No tags</li>
  {% endfor %}
</ul>
```

### Why is it needed?

You need to produce HTML that changes based on data (a product page differs per product). Doing that by concatenating strings in Python is ugly, unsafe (XSS), and mixes logic with markup. Templates give designers/frontend developers a clean HTML file they can edit, keep presentation separate from view logic, and — crucially — **auto-escape output to prevent XSS** by default. DTL is intentionally *limited* (you can't run arbitrary Python) to enforce the separation of concerns: heavy logic belongs in the view, not the template.

### How does it work / Internal working

- Django's template engine **compiles** a template into a tree of nodes once, then **renders** it with a `context` (the dict of variables the view passes). Compiled templates are cached.
- **Variable resolution** tries dictionary lookup, attribute access, and list-index lookup, in that order — that's why `{{ article.title }}` and `{{ article.tags.all }}` both work (attribute, then method call with no args).
- **Auto-escaping**: by default Django HTML-escapes variable output (`<` becomes `&lt;`), neutralising injected scripts. You opt out explicitly with `|safe` or `{% autoescape off %}` — which you should only do for content you fully trust.
- **Filters** are functions applied left-to-right; **tags** can contain logic and even render nested blocks.

### Advantages

- Clean separation of presentation and logic; auto-escaping (XSS protection) by default; designer-friendly; template inheritance reduces duplication; safe by design (limited language).

### Limitations

- Intentionally limited — no arbitrary Python, so complex presentation logic must be prepared in the view or a custom template tag/filter.
- Server-side rendering only; for rich interactivity you need JS/a frontend framework.
- Can be slower than pure string building for extreme cases (rarely relevant).

### Interview Questions & Model Answers

**Q: Why does Django auto-escape template output, and when would you turn it off?**
Auto-escaping is Django's built-in defence against **Cross-Site Scripting (XSS)**. If a user submits a comment containing `<script>steal()</script>` and you render it directly, the browser would execute that script for every visitor. Django prevents this by automatically converting HTML-special characters (`<`, `>`, `&`, `"`) into their harmless entity equivalents when a variable is output, so the script displays as text instead of running. You'd turn it off — with the `|safe` filter or `{% autoescape off %}` — only for content you've generated or sanitised yourself and know is safe HTML, such as rendered Markdown you've already cleaned. Turning it off on untrusted user input reopens the XSS hole, so it's done sparingly and deliberately.

**Q: Why is the Django template language intentionally limited?**
DTL deliberately restricts what you can do — you can't run arbitrary Python, call functions with arguments, or perform complex computation in a template. This is a design choice to enforce separation of concerns: templates should be about *presentation*, and business/query logic should live in the view (or a service layer). If templates allowed full Python, developers would inevitably put database queries and heavy logic in them, making the app untestable and tangling responsibilities. The limitation also improves safety (a template can't do arbitrary damage) and keeps templates approachable for frontend developers/designers who aren't Python experts. When you genuinely need custom logic in a template, the sanctioned escape hatch is to write a custom template tag or filter in Python.

### Common Mistakes

- Overusing `|safe` and reopening XSS vulnerabilities.
- Trying to put complex logic/queries in templates.
- Forgetting that method calls in templates take no arguments (`obj.method` works, `obj.method(x)` doesn't).
- Not using `{% for %}...{% empty %}` for empty states.

### Related Concepts

XSS, auto-escaping, custom template tags/filters, context, template inheritance (next), Jinja2 (alternative engine).

---

## 3.5 Template Features: Inheritance, Blocks, Includes, Filters, Tags

### What is it?

The features that make templates DRY and composable:
- **Template inheritance** — a `base.html` defines the page skeleton with named **`{% block %}`** regions; child templates `{% extends %}` it and fill/override those blocks.
- **Blocks** — placeholders a child template can override.
- **Includes** — `{% include "partial.html" %}` embeds a reusable snippet (navbar, card).
- **Filters** — transform values on output (`|date`, `|truncatewords`, `|default`).
- **Template tags** — logic/structure (`{% for %}`, `{% url %}`, `{% csrf_token %}`), plus custom tags you write.

```django
{# base.html #}
<html><head><title>{% block title %}Site{% endblock %}</title></head>
<body>
  {% include "navbar.html" %}
  <main>{% block content %}{% endblock %}</main>
</body></html>

{# article.html #}
{% extends "base.html" %}
{% block title %}{{ article.title }}{% endblock %}
{% block content %}<h1>{{ article.title }}</h1>{% endblock %}
```

### Why is it needed?

Every page on a site shares structure — the same header, footer, CSS/JS includes, navigation. Without inheritance you'd copy that boilerplate into every template, and changing the navbar would mean editing dozens of files. **Inheritance** defines the shared skeleton once; each page only supplies what's unique. **Includes** reuse smaller components. **Filters** keep formatting logic out of views. This is the DRY principle applied to presentation.

### How does it work / Internal working

- **`{% extends %}` must be the first tag** in a child template. At render time Django loads the parent, then replaces each `{% block name %}` in the parent with the child's block of the same name (falling back to the parent's default content if the child doesn't override it).
- Blocks can be nested and a child can call **`{{ block.super }}`** to include the parent block's content plus its own.
- `{% include %}` renders another template with the current context (or a restricted context you pass).
- Inheritance is resolved at render, and compiled templates are cached, so the overhead is minimal.

### Advantages / Limitations

- **Advantage:** DRY, consistent site-wide layout, single place to change shared structure, reusable components.
- **Limitation:** deep inheritance chains can be hard to trace; overusing includes with heavy context can hurt performance; logic still can't live in templates (by design).

### Interview Questions & Model Answers

**Q: Explain template inheritance and why it matters.**
Template inheritance lets you define a base template — `base.html` — that contains the common page structure (doctype, head, navigation, footer) with named `{% block %}` regions marking the parts that vary. Individual page templates then `{% extends "base.html" %}` and override only the blocks they care about, like `title` and `content`. It matters because it enforces DRY at the presentation layer: the shared layout exists in exactly one file, so changing the header or adding a global script is a one-line edit rather than a change across every page. It also guarantees visual consistency, since every page inherits the same skeleton. It's essentially the template equivalent of subclassing.

**Q: Difference between `{% include %}` and `{% extends %}`?**
`{% extends %}` is for **inheritance** — a child template declares that it's a specialised version of a base template and fills in the base's blocks; there's one base skeleton and the child provides the variable pieces. `{% include %}` is for **composition** — it embeds the fully-rendered output of another template at that spot, like dropping in a reusable navbar or product-card partial. Rule of thumb: use `extends` for the overall page structure (one per page, at the top), and `include` for repeated components you want to reuse across many pages. They're complementary: a base template often `include`s partials, and child templates `extend` the base.

**Q: What's `{{ block.super }}`?**
Inside a child template's overridden block, `{{ block.super }}` renders the content that the parent template had in that same block. It lets you *extend* rather than *replace* the parent's block — for example, a child page can add extra CSS while keeping the base's default styles by writing `{% block styles %}{{ block.super }}<link ...>{% endblock %}`. Without it, overriding a block completely discards the parent's version. It's the template analogue of calling `super()` in a subclass method.

### Common Mistakes

- Putting content before `{% extends %}` (it must be first).
- Forgetting `{% csrf_token %}` inside forms.
- Overriding a block and losing the parent content because `{{ block.super }}` wasn't used.
- Excessive `{% include %}` of query-heavy partials causing extra work per render.

### Related Concepts

DRY, `{% csrf_token %}`, custom template tags/filters, static files (`{% load static %}`), context processors.

---

## 3.6 URL Routing

### What is it?

**URL routing** (URLconf) is how Django maps an incoming request path to the view that should handle it. You define a list of **URL patterns** — each pairs a path pattern with a view — in `urls.py`.

```python
from django.urls import path, include
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("articles/", views.ArticleListView.as_view(), name="article-list"),
    path("articles/<int:pk>/", views.ArticleDetailView.as_view(), name="article-detail"),
    path("blog/", include("blog.urls")),        # delegate to an app's URLconf
]
```

Key concepts: **path converters** (`<int:pk>`, `<slug:slug>`, `<str:name>`), **`include()`** for splitting URLs per app, **URL names** for referring to URLs by name, **reversing** (generating a URL from its name), and **namespaces** for disambiguating names across apps.

### Why is it needed?

Users and clients request URLs; something must decide which code runs for `/articles/42/` versus `/login/`. Routing provides that mapping in a clean, centralised, readable place. **Named URLs and reversing** exist so you never hardcode URLs in code or templates — you refer to them by name, and if the URL structure changes, you update it in one place (the URLconf) and everything else keeps working. **`include()`** keeps large projects organised by letting each app own its routes. **Namespaces** prevent collisions when two apps both have a URL named `detail`.

### How does it work?

1. On each request, Django walks `urlpatterns` **top to bottom** and uses the **first pattern that matches** the path.
2. **Path converters** capture and type-convert parts of the URL: `<int:pk>` captures an integer and passes it to the view as `pk`.
3. **`include()`** hands off the remaining path to another app's URLconf, enabling modular routing.
4. **Reversing:** `reverse("article-detail", args=[42])` or `{% url 'article-detail' 42 %}` produces `/articles/42/` from the name — the inverse of matching.

```
Request /articles/42/
   │
   ▼  match top-to-bottom
path("articles/<int:pk>/", ArticleDetailView.as_view(), name="article-detail")
   │  captures pk=42 (int)
   ▼
ArticleDetailView(request, pk=42)
```

### Internal working

- Each `path()` is compiled into a regex-like matcher. Django's resolver tries them in order and stops at the first match, capturing converter values as view kwargs.
- **URL names** are stored in a reverse-lookup table. `reverse()`/`{% url %}` search this table (respecting namespaces) and rebuild the path, substituting arguments — so URLs are computed, never hardcoded.
- **Namespaces** (`app_name = "blog"` in the app's urls, plus `include("blog.urls", namespace="blog")`) qualify names as `blog:detail`, so identically-named URLs in different apps don't clash.

### Advantages

- Clean, human-readable URLs; centralised and modular routing; decoupling via named URLs (change structure without breaking links); namespaces prevent collisions; typed path converters reduce validation code.

### Limitations

- Order-sensitivity: a broad pattern placed early can shadow later ones.
- Overusing `re_path` (regex) hurts readability; prefer `path` with converters.
- Reverse lookups fail loudly if names/args are wrong (which is actually good — catches mistakes early).

### Real-world applications

Every route in every Django app. Named URLs are used pervasively in templates (`{% url %}`), redirects (`redirect("name")`), and `get_absolute_url()` on models. DRF routers auto-generate URL patterns for viewsets.

### Interview Questions & Model Answers

**Q: Why use named URLs and `reverse()` instead of hardcoding paths?**
Hardcoding a path like `/articles/42/` in templates and views scatters that string across the codebase. If the URL structure changes — say to `/posts/42/` — you'd have to hunt down and update every occurrence, and missing one silently breaks links. Named URLs decouple the *name* of a route from its *path*. You define the path once in the URLconf with a `name`, and everywhere else you refer to it by name via `{% url 'article-detail' 42 %}` in templates or `reverse('article-detail', args=[42])` in Python. Django computes the actual path from the current URLconf, so changing the URL pattern updates every link automatically. It's the routing equivalent of avoiding magic strings — it makes URLs maintainable and refactor-safe.

**Q: What are path converters?**
Path converters are typed placeholders in a URL pattern that capture part of the path and convert it to a Python type before passing it to the view. For example, `path("articles/<int:pk>/", ...)` captures the segment after `/articles/` as an **integer** named `pk`; if the segment isn't a valid integer, the pattern simply doesn't match, so you get automatic validation and type conversion. Built-in converters include `int` (integers), `str` (any non-slash text, the default), `slug` (letters, numbers, hyphens, underscores), `uuid`, and `path` (matches across slashes). They keep URLs clean and remove boilerplate validation from the view, since the view receives a correctly-typed argument.

**Q: What are URL namespaces and when do you need them?**
Namespaces qualify URL names so identical names in different apps don't collide. If both a `blog` app and a `shop` app define a URL named `detail`, then `reverse("detail")` would be ambiguous. By setting `app_name = "blog"` in the blog app's `urls.py` and including it, its URL becomes addressable as `blog:detail`, and the shop's as `shop:detail`. You need namespaces in any project with multiple apps that reuse common URL names (`list`, `detail`, `create`), which is very common. They let reusable apps define whatever names make sense internally without worrying about clashing with the rest of the project.

### Common Mistakes

- Placing a greedy/broad pattern before a specific one, so the specific one never matches.
- Hardcoding URLs in templates instead of `{% url %}`.
- Forgetting the trailing slash (Django's `APPEND_SLASH` usually redirects, but inconsistency causes confusion).
- Forgetting `app_name`/namespace and getting reverse collisions.

### Related Concepts

`include()`, path converters, `reverse()`/`{% url %}`, `get_absolute_url()`, DRF routers, `APPEND_SLASH`.

---

# 4. Authentication, Forms & Sessions

## 4.1 User Authentication (Login, Logout, Registration)

### What is it?

**Authentication** is the process of verifying *who a user is* — confirming they are who they claim to be, usually by checking a username/email and password. Django ships a complete authentication system in `django.contrib.auth`: a **`User` model**, password hashing, and helper functions/views for **login, logout, and registration**.

Core pieces:
- `User` model — stores username, email, hashed password, flags (`is_active`, `is_staff`, `is_superuser`).
- `authenticate(username, password)` — checks credentials, returns the user or `None`.
- `login(request, user)` — starts a logged-in session.
- `logout(request)` — ends the session.
- `UserCreationForm` — a ready-made registration form.

### Why is it needed?

Almost every app needs to know who the user is to show personalised data, restrict access, and attribute actions. Building authentication correctly is **hard and security-critical** — you must hash passwords properly (never store plaintext), defend against timing attacks, manage sessions securely, and handle edge cases. Django's auth system provides all of this, correctly, out of the box, so you don't reinvent (and get wrong) one of the most sensitive parts of an app.

### How does it work?

```python
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm

def login_view(request):
    if request.method == "POST":
        user = authenticate(request,
                            username=request.POST["username"],
                            password=request.POST["password"])
        if user is not None:
            login(request, user)              # sets the session
            return redirect("home")
    return render(request, "login.html")

def register_view(request):
    form = UserCreationForm(request.POST or None)
    if form.is_valid():
        form.save()                           # creates user, hashes password
        return redirect("login")
    return render(request, "register.html", {"form": form})

def logout_view(request):
    logout(request)                           # clears the session
    return redirect("home")
```

### Internal working

- **Password hashing:** Django never stores plaintext. On `set_password()`/registration it runs the password through a strong, slow hashing algorithm (default **PBKDF2** with many iterations, optionally Argon2/bcrypt) with a random **salt**, and stores `algorithm$iterations$salt$hash`. On login it re-hashes the submitted password with the same salt and compares — a constant-time comparison to resist timing attacks.
- **`authenticate()`** loops through configured **authentication backends** (default: the model backend) until one verifies the credentials.
- **`login()`** stores the user's ID in the **session** and issues a session cookie; subsequent requests are recognised via **AuthenticationMiddleware**, which sets `request.user`.
- **`logout()`** flushes the session data and cycles the session key.

### Advantages

- Secure password hashing by default; complete login/logout/registration flow; extensible (custom backends, custom user model); integrates with permissions, admin, and DRF.

### Limitations

- The default `User` uses **username** as the identifier; email-based login needs a custom user model or backend (best decided at project start).
- Session-based auth is stateful (server tracks sessions) — for stateless APIs you use token/JWT auth instead (covered in DRF).

### Real-world applications

Every app with accounts. The auth system underpins the admin, per-user data, and permission checks. It's commonly extended with a custom user model and social login (django-allauth).

### Interview Questions & Model Answers

**Q: How does Django store passwords, and why not store them encrypted?**
Django stores passwords as **salted hashes**, not as encrypted or plaintext values. When a password is set, Django runs it through a deliberately slow, one-way hashing algorithm (PBKDF2 by default, with Argon2/bcrypt available) combined with a random per-user salt, and stores the algorithm, iteration count, salt, and resulting hash. The key point is that hashing is **one-way**: you can't reverse a hash to recover the password. Encryption is the wrong tool because it's reversible — if the encryption key leaks, all passwords are exposed. With hashing, even if the database is stolen, attackers can't recover passwords directly; the salt prevents precomputed rainbow-table attacks, and the slowness makes brute-forcing expensive. On login, Django hashes the submitted password the same way and compares in constant time to avoid timing attacks.

**Q: Walk through what `login()` actually does.**
`login(request, user)` establishes an authenticated session. It stores the authenticated user's primary key (and the backend used) in the server-side **session**, then ensures a session cookie is sent to the browser holding the session's ID. From then on, every subsequent request carries that cookie; Django's SessionMiddleware loads the session, and AuthenticationMiddleware reads the stored user ID and populates `request.user` with the actual `User` object. It also cycles the session key on login to prevent session-fixation attacks. So `login()` is what turns a verified user (from `authenticate()`) into a persistent logged-in state across requests.

**Q: `authenticate()` vs `login()` — what's the difference?**
They're two distinct steps. `authenticate(username, password)` **verifies credentials**: it checks the supplied username/password against the configured authentication backends and returns the matching `User` object if they're valid, or `None` if not. It does *not* log anyone in — it just confirms identity. `login(request, user)` **establishes the session**: it takes an already-verified user and records them as logged in by writing to the session and setting the cookie. You almost always call them together — authenticate first to verify, then login to persist — but separating them lets you, for instance, authenticate a user in a context where you don't want to start a web session.

### Common Mistakes

- Storing or logging plaintext passwords.
- Not choosing a custom user model early (switching later is painful).
- Manually hashing passwords instead of using `set_password()`/forms.
- Forgetting that `authenticate` returns `None` (not raising) on bad credentials.

### Related Concepts

Sessions, AuthenticationMiddleware, custom user model, permissions, password validators, token/JWT auth.

---

## 4.2 Permissions & Authorization (Groups & Permissions)

### What is it?

**Authorization** is deciding *what an authenticated user is allowed to do* — as opposed to authentication (who they are). Django provides:
- **Permissions** — granular flags like `blog.add_article`, `blog.change_article`, `blog.delete_article`, `blog.view_article` (auto-created per model), plus custom permissions.
- **Groups** — named buckets of permissions you assign users to (e.g., "Editors"), so you manage permissions in bulk.
- Convenience flags: `is_staff` (can access admin), `is_superuser` (all permissions).
- Enforcement helpers: `@login_required`, `@permission_required`, `PermissionRequiredMixin`, and `user.has_perm(...)`.

### Why is it needed?

Not every logged-in user should do everything. An author can edit their own posts but not delete others'; a moderator can remove comments; an admin can do anything. Authorization enforces these rules so the app is secure and correct. **Groups** exist because assigning individual permissions to thousands of users is unmanageable — you assign permissions to a role (group) once and add users to it. This is **Role-Based Access Control (RBAC)** in practice.

### How does it work?

```python
# Enforce login
@login_required
def dashboard(request): ...

# Enforce a specific permission
@permission_required("blog.add_article", raise_exception=True)
def create_article(request): ...

# Programmatic checks
if request.user.has_perm("blog.delete_article"):
    ...

# Groups
editors = Group.objects.get(name="Editors")
user.groups.add(editors)          # user now inherits the group's permissions
```

### Internal working

- Django auto-creates four permissions per model (`add`, `change`, `delete`, `view`) at migration time, stored in the `auth_permission` table.
- A user's **effective permissions** = their directly-assigned permissions **∪** permissions from all their groups. Superusers implicitly have all permissions.
- `user.has_perm("app.codename")` checks this set (with caching per request). Custom permissions are declared in a model's `Meta.permissions` or via `Meta.constraints`-style config.
- Decorators/mixins call these checks and either allow the view or return 302→login / 403 Forbidden.

### Advantages / Limitations

- **Advantage:** granular, role-based, integrates with admin and DRF; groups make bulk management easy.
- **Limitation:** the built-in model permissions are **model-level**, not **object-level** (they don't natively express "can edit *this specific* article"). Object-level/row-level permissions need custom logic or a library like `django-guardian`.

### Interview Questions & Model Answers

**Q: Authentication vs Authorization — define both clearly.** *(Checklist item.)*
Authentication answers **"who are you?"** — it's the process of verifying a user's identity, typically by checking a username/email and password (or a token). Authorization answers **"what are you allowed to do?"** — once identity is established, it's the process of deciding whether that user may perform a particular action or access a particular resource. Authentication always comes first; authorization builds on it. A concrete example: logging in with your correct password is authentication; the system then checking whether your account has permission to delete another user's post is authorization. In Django, authentication is handled by `django.contrib.auth` (login, sessions, `authenticate`), and authorization is handled by the permissions and groups system (`has_perm`, `@permission_required`, `IsAuthenticated`/`IsAdminUser` in DRF).

**Q: What are groups and why use them?**
A group is a named collection of permissions that you can assign users to — effectively a **role**. Instead of granting each of the "add article," "change article," and "publish article" permissions to every editor individually, you create an "Editors" group with those permissions once and simply add users to that group; they inherit all of its permissions. This is role-based access control, and it matters for maintainability: when the role's responsibilities change, you update the group's permissions in one place and every member is affected. It also makes onboarding trivial — assign the new hire to the right groups. A user's effective permissions are the union of their personal permissions and all their groups' permissions.

**Q: What's the difference between model-level and object-level permissions?**
Django's built-in permissions are **model-level**: `blog.change_article` means "can change *articles* in general," not "can change *this particular* article." That's fine for coarse control (editors can edit articles) but insufficient when authorization depends on the specific object — e.g., authors may edit only their *own* articles. That's an **object-level** (row-level) permission, which Django doesn't provide natively. You implement it either with custom logic in the view/serializer (check `article.author == request.user`), a custom DRF permission's `has_object_permission`, or a library like `django-guardian` that stores per-object permissions. Interviewers ask this to see whether you know the built-in system's boundary.

### Common Mistakes

- Confusing authentication with authorization.
- Assuming model permissions handle "own object only" cases (they don't).
- Granting permissions to users individually instead of via groups.
- Forgetting `raise_exception=True` and silently redirecting instead of returning 403.

### Related Concepts

RBAC, `is_staff`/`is_superuser`, DRF permissions, object-level permissions, django-guardian.

---

## 4.3 Sessions, Cookies & Authentication Middleware

### What is it?

**HTTP is stateless** — each request is independent and the server doesn't inherently remember previous ones. **Sessions** are Django's mechanism for remembering data about a user *across* requests (like "this user is logged in"). A **cookie** is a small piece of data the browser stores and sends back with every request; Django uses a **session cookie** holding a random **session ID** to link a browser to its server-side session data.

- **Cookie:** `sessionid=abc123...` stored in the browser.
- **Session:** server-side storage (DB, cache, etc.) keyed by that ID, holding arbitrary data.
- **AuthenticationMiddleware:** on each request, reads the session, finds the user ID, and sets `request.user`.

### Why is it needed?

Because HTTP forgets everything between requests, you need a way to keep a user "logged in" as they navigate. You can't trust the client to just say "I'm user 5" — that's forgeable. The session pattern solves this: the server stores the sensitive data (who's logged in), and the browser only holds an **opaque random ID** that references it. The ID is meaningless to an attacker without the server-side data, and Django signs/secures the cookie so it can't be tampered with.

### How does it work?

```
1. User logs in.
2. Server creates a session row: {session_id: "abc123", data: {user_id: 5}}.
3. Server sends Set-Cookie: sessionid=abc123 (HttpOnly, Secure).
4. Browser stores it, and sends Cookie: sessionid=abc123 on every future request.
5. SessionMiddleware loads session "abc123"; AuthenticationMiddleware sets request.user = User(5).
6. On logout, the session is deleted/flushed.
```

You can also store custom data: `request.session["cart"] = [...]`.

### Internal working

- **SessionMiddleware** loads the session at request start (from the configured backend: database `django_session` table, cache, cached_db, or signed cookies) and saves it at the end if modified.
- **Session ID** is a long random string (high entropy) so it can't be guessed. On login Django **cycles** the key to prevent session fixation.
- The session **cookie** is set `HttpOnly` (JS can't read it → mitigates XSS token theft) and, in production, `Secure` (only sent over HTTPS) and ideally `SameSite=Lax/Strict` (mitigates CSRF).
- **AuthenticationMiddleware** depends on SessionMiddleware having run first — hence middleware ordering.

### Advantages

- Enables stateful features (login, carts, flash messages) over stateless HTTP; sensitive data stays server-side; secure cookie flags mitigate theft; pluggable storage backends (DB/cache/Redis).

### Limitations

- **Stateful** — the server must store and look up session data, which complicates horizontal scaling (need shared session storage like Redis, or sticky sessions).
- Not ideal for APIs consumed by mobile apps/SPAs across domains → token/JWT auth is preferred there (stateless).
- Session storage in the DB adds queries (use cache backend for performance).

### Real-world applications

Traditional server-rendered web apps (login, shopping carts, "remember this" preferences) use sessions. APIs typically use token/JWT auth instead. Large deployments store sessions in Redis for speed and to share across app servers.

### Interview Questions & Model Answers

**Q: HTTP is stateless — how does Django remember a logged-in user?**
Since HTTP treats each request independently, Django uses **sessions backed by a cookie**. When a user logs in, Django creates a server-side session containing their user ID and gives it a random session identifier. It sends that identifier to the browser as a **session cookie** (`sessionid`). The browser automatically returns the cookie on every subsequent request, and Django's SessionMiddleware uses it to load the corresponding session data, after which AuthenticationMiddleware sets `request.user`. The crucial design detail is that the browser only holds an opaque random ID — the actual "who is logged in" data lives on the server — so a client can't forge being a different user, and the ID reveals nothing on its own.

**Q: Why is the session cookie marked HttpOnly and Secure?**
These flags harden the cookie against theft. **HttpOnly** tells the browser not to expose the cookie to JavaScript (`document.cookie` can't read it). This matters because if an attacker manages an XSS injection, they still can't read the session cookie and hijack the session. **Secure** tells the browser to send the cookie only over HTTPS, so it's never transmitted in plaintext where a network attacker could sniff it. Together with **SameSite** (which restricts when the cookie is sent on cross-site requests, mitigating CSRF), these flags are essential production settings; Django lets you enable them via `SESSION_COOKIE_HTTPONLY`, `SESSION_COOKIE_SECURE`, and `SESSION_COOKIE_SAMESITE`.

**Q: Why does middleware order matter for authentication?**
Middleware runs in the order listed in `MIDDLEWARE`, and some middleware depends on the work of earlier ones. AuthenticationMiddleware sets `request.user` by reading the user ID from the **session** — but the session is only available because SessionMiddleware ran first and loaded it. If AuthenticationMiddleware were placed before SessionMiddleware, there'd be no session to read and `request.user` couldn't be populated correctly. This is the canonical example of why Django's default middleware order (SessionMiddleware, then AuthenticationMiddleware) exists, and why you shouldn't reorder middleware without understanding the dependencies.

### Common Mistakes

- Reordering middleware and breaking `request.user`.
- Not setting `Secure`/`HttpOnly`/`SameSite` in production.
- Storing large or sensitive data directly in cookies instead of the server session.
- Using DB session backend at scale without caching, adding query load.

### Related Concepts

Cookies, CSRF, `SameSite`, Redis session backend, token/JWT (stateless alternative), session fixation.

---

## 4.4 Django Forms & ModelForms

### What is it?

The **Forms framework** handles the full lifecycle of user input: **rendering** an HTML form, **validating** submitted data, **cleaning/converting** it to Python types, and reporting **errors** back to the user. Two flavours:
- **`Form`** — you declare fields manually; not tied to a model.
- **`ModelForm`** — automatically builds a form from a model's fields, and can `.save()` directly to the database.

```python
from django import forms
from .models import Article

class ContactForm(forms.Form):                 # plain Form
    email = forms.EmailField()
    message = forms.CharField(widget=forms.Textarea)

class ArticleForm(forms.ModelForm):            # ModelForm
    class Meta:
        model = Article
        fields = ["title", "body", "published"]
```

**Formsets** let you handle *multiple* instances of the same form on one page (e.g., editing 5 line-items at once).

### Why is it needed?

Handling user input correctly is surprisingly involved: you must render fields, preserve entered values on error, validate types and rules, sanitise against attacks, and show clear error messages. Doing this by hand for every form is tedious and error-prone. The forms framework automates it. **ModelForms** go further: since most forms map directly to a model (a "create article" form is just the Article fields), ModelForm derives the form from the model, eliminating duplication and keeping form and model in sync.

### How does it work?

```python
def create_article(request):
    form = ArticleForm(request.POST or None)
    if request.method == "POST" and form.is_valid():   # runs validation
        form.save()                                     # ModelForm: writes to DB
        return redirect("article-list")
    return render(request, "form.html", {"form": form}) # re-renders with errors
```

- `form.is_valid()` triggers validation; on success `form.cleaned_data` holds converted values.
- On failure, `form.errors` holds messages and the template re-renders with the user's input preserved.

### Internal working

- Binding data (`ArticleForm(request.POST)`) makes the form "bound." `is_valid()` runs each field's `clean_<field>()` and the form-wide `clean()`, populating `cleaned_data` or `errors`.
- Each field has a **widget** (how it renders) and validation logic (type + rules).
- **ModelForm** introspects the model's fields to auto-generate matching form fields and, in `save()`, maps `cleaned_data` back onto a model instance.
- Rendering escapes output (XSS-safe) and re-displays entered values and errors automatically.

### Advantages

- Automatic rendering, validation, type conversion, and error handling; ModelForms remove duplication; XSS/CSRF-safe when used with `{% csrf_token %}`; consistent, testable.

### Limitations

- Default HTML rendering is basic; production UIs often need widget/template customisation or crispy-forms.
- For pure JSON APIs, DRF **serializers** replace forms (forms are HTML-oriented).

### Interview Questions & Model Answers

**Q: What problem do Django forms solve?**
Django forms handle the entire input pipeline: rendering HTML form fields, receiving submitted data, validating and cleaning it into proper Python types, and reporting errors back with the user's values preserved. The problem they solve is that doing all this manually is repetitive and dangerous — you'd have to re-display entered values on validation failure, coerce strings into dates/numbers, enforce rules, and guard against injection, for every single form. The forms framework centralises that into a declarative class where you list fields (or, with ModelForm, derive them from a model), and it manages rendering, validation, and error handling for you. It also integrates with CSRF protection and auto-escaping, so forms are secure by default.

**Q: Form vs ModelForm — when to use each?**
Use a plain **`Form`** when the input doesn't correspond directly to a model — a contact form, a search box, a multi-step wizard, or a form that touches several models. You declare each field explicitly and handle what to do with `cleaned_data` yourself. Use a **`ModelForm`** when the form maps to a single model, which is the common case for create/edit pages: it auto-generates fields from the model, respects the model's field types and validators, and provides a `save()` that writes straight to the database. ModelForm keeps the form and model in sync and eliminates duplication, so it's the default for CRUD; drop to a plain Form when the mapping isn't one-to-one.

**Q: How do Django forms compare to DRF serializers?** *(Checklist item.)*
They play the same *role* — validating and converting incoming data — but in different worlds. **Forms** are built for HTML: they render `<input>` elements, work with form-encoded POST data, and produce error output meant for web pages. **Serializers** (DRF) are built for APIs: they convert between complex types (model instances/querysets) and JSON, validate incoming JSON, and produce JSON error responses; they also handle *serialisation* (object → JSON) which forms don't do. So in a server-rendered Django app you use forms; in a JSON API you use serializers. Conceptually they're siblings — a `ModelForm` and a `ModelSerializer` are analogous, one for HTML and one for JSON.

### Common Mistakes

- Forgetting `{% csrf_token %}` in the template (POST will be rejected).
- Not passing `request.POST` (form stays unbound, `is_valid()` fails silently).
- Using forms for JSON APIs instead of serializers.
- Not re-rendering the form on invalid submission (losing the user's input and errors).

### Related Concepts

ModelForm, formsets, widgets, CSRF, validators, DRF serializers.

---

## 4.5 Validation: Field Validation, Custom Validators, Clean Methods, Error Handling

### What is it?

Validation is checking that submitted data is correct and safe before you use it. Django validates at multiple levels:
- **Field validation** — built-in per field (`EmailField` checks email format, `max_length`, `required`).
- **Validators** — reusable functions/classes attached to a field (`validators=[MinValueValidator(0)]`), or custom ones you write.
- **`clean_<fieldname>()`** — a method to validate/normalise a single field with custom logic.
- **`clean()`** — a form-wide method to validate relationships *between* fields (e.g., "end date must be after start date").
- **Error handling** — raising `ValidationError`, which the framework collects and displays.

### Why is it needed?

**Never trust user input.** Data from the outside world can be malformed, malicious, or inconsistent. Validation ensures only correct data reaches your database and business logic, protecting data integrity and security. Different rules live at different levels: simple format checks are per-field; cross-field rules (password confirmation matching) need the whole form; reusable rules (a phone-number format used in many places) become custom validators. Django gives a clear place for each.

### How does it work?

```python
class SignupForm(forms.Form):
    password = forms.CharField(widget=forms.PasswordInput)
    confirm = forms.CharField(widget=forms.PasswordInput)

    def clean_password(self):                     # single-field custom rule
        pw = self.cleaned_data["password"]
        if len(pw) < 8:
            raise forms.ValidationError("Password too short.")
        return pw

    def clean(self):                              # cross-field rule
        data = super().clean()
        if data.get("password") != data.get("confirm"):
            raise forms.ValidationError("Passwords do not match.")
        return data
```

### Internal working

- On `is_valid()`, Django runs, per field: `to_python()` (type conversion) → field validators → `clean_<field>()`. Then it runs the form-wide `clean()`.
- Any `ValidationError` raised is caught and attached to `form.errors` (field-specific or `__all__` for form-wide), and the field's value stays available so the template can re-render it.
- **Model-level validation** (`full_clean()`) exists too, and model field validators run there; note that `Model.save()` does *not* automatically call `full_clean()` — ModelForms do call validation, which is one reason to prefer forms/serializers for input.

### Advantages / Limitations

- **Advantage:** layered, reusable, clear separation of single-field vs cross-field rules; secure-by-default.
- **Limitation:** validation logic can end up duplicated across forms, serializers, and models if not centralised (best practice: put invariants as model/validator logic so all paths enforce them).

### Interview Questions & Model Answers

**Q: What's the difference between `clean_<field>()` and `clean()`?**
`clean_<field>()` validates or normalises a **single field** with custom logic beyond the built-in checks — for example, `clean_username()` might enforce that a username isn't already taken. It runs after the field's type conversion and built-in validators, and it returns the cleaned value. `clean()` is the **form-wide** hook: it runs after all individual fields are cleaned and is the place for rules that depend on **multiple fields together**, such as confirming two password fields match or that an end date is after a start date. The distinction matters because a cross-field rule can't live in a single field's method — at that point the other field may not be validated yet — so Django gives you `clean()` where all `cleaned_data` is available.

**Q: What is a custom validator and when would you write one?**
A custom validator is a reusable callable that checks a value and raises `ValidationError` if it's invalid. You attach it to a field via `validators=[my_validator]`, and it can be shared across many fields, forms, and even model fields. You'd write one when the same validation rule appears in multiple places — say, a rule that a phone number matches a specific national format, or that an uploaded file is under a size limit. Rather than duplicating that check in every `clean_<field>()`, you encapsulate it once as a validator and reuse it. This keeps validation DRY and ensures the rule is enforced consistently everywhere the field is used.

**Q: Why should you never rely only on client-side (JavaScript/HTML) validation?**
Client-side validation improves user experience by giving instant feedback, but it can be trivially bypassed — a user can disable JavaScript, edit the HTML, or send requests directly with `curl` or Postman, skipping the browser entirely. Since the server can't trust that any client-side check ran, **all security- and integrity-critical validation must happen on the server** (in Django forms/serializers/models). Client-side validation is a convenience layer; server-side validation is the real gate. The correct approach is to do both: friendly client-side checks for UX, and authoritative server-side validation as the actual enforcement.

### Common Mistakes

- Relying on frontend validation alone.
- Putting cross-field logic in `clean_<field>()` where other fields aren't yet available.
- Forgetting to `return` the cleaned value from `clean_<field>()`.
- Assuming `Model.save()` validates (it doesn't call `full_clean()` automatically).

### Related Concepts

`ValidationError`, validators, `full_clean()`, DRF serializer validation, model constraints.

---

## 4.6 File & Image Uploads, Media Configuration

### What is it?

Handling files (documents, images) that users upload. Django provides `FileField` and `ImageField` model fields, `request.FILES` for accessing uploads, and **media configuration** (`MEDIA_URL`, `MEDIA_ROOT`) that governs where uploaded files are stored and served from.

```python
class Profile(models.Model):
    avatar = models.ImageField(upload_to="avatars/")   # needs Pillow
    resume = models.FileField(upload_to="resumes/")

# settings.py
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"
```

### Why is it needed?

Many apps need user-provided files: profile pictures, product images, PDF uploads. Storing files needs care — you don't put large binary blobs in the database; you store the **file on disk (or cloud storage) and keep only the path in the database**. Django standardises this: the field stores a path, the file goes to `MEDIA_ROOT`, and `MEDIA_URL` defines the public URL prefix. `ImageField` adds image-specific validation (and dimensions) via the Pillow library.

### How does it work?

```python
def upload(request):
    if request.method == "POST":
        form = ProfileForm(request.POST, request.FILES)   # note request.FILES!
        if form.is_valid():
            form.save()      # saves the file to MEDIA_ROOT, path to DB
```

- The form must be given **both** `request.POST` and `request.FILES`.
- The template `<form>` must set `enctype="multipart/form-data"` or the file won't be sent.
- In development, you wire up media serving in `urls.py` with `static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)`.

### Internal working

- On upload, Django streams the file (small files in memory, large ones to a temp file), then the storage backend writes it under `MEDIA_ROOT/upload_to/`. The model field stores the relative path string.
- **Static vs media distinction:** *static files* are your app's own assets (CSS/JS/images you ship); *media files* are user-uploaded content. They're configured and served differently.
- In production you typically offload media to **AWS S3** (via `django-storages`) so files aren't on the app server's ephemeral disk and can be served/cached via a CDN (CloudFront).

### Advantages / Limitations

- **Advantage:** clean abstraction; pluggable storage backends (local, S3); path-in-DB keeps the database lean.
- **Limitation:** serving media from the app server doesn't scale and is insecure for private files; needs cloud storage + signed URLs for production; uploads must be validated (type/size) to prevent abuse.

### Interview Questions & Model Answers

**Q: What's the difference between static files and media files?** *(Checklist item.)*
**Static files** are the assets that are part of your application itself — CSS, JavaScript, logos, icons — that you write and ship; they're the same for every user and don't change at runtime. They're managed via `STATIC_URL`/`STATIC_ROOT` and collected with `collectstatic` for production. **Media files** are content **uploaded by users** at runtime — profile pictures, document attachments, product photos — which vary per user and grow over time. They're managed via `MEDIA_URL`/`MEDIA_ROOT`. The distinction matters operationally: static files are deployed with your code and often served by WhiteNoise or a CDN, while media files need persistent, scalable storage (typically S3) since they're generated after deployment and can't live on an ephemeral app-server disk.

**Q: Where should uploaded files be stored in production and why not in the database?**
In production, uploaded files should be stored in a dedicated object store like **AWS S3** (via `django-storages`), not on the app server's local disk and not inside the database. Not the database because storing large binary blobs there bloats it, slows backups and queries, and databases aren't optimised for serving files — the standard pattern is to keep only the file's *path/URL* in the database and the bytes elsewhere. Not the local disk because modern deployments run multiple, often ephemeral, app servers (containers, autoscaled instances); a file uploaded to one server's disk wouldn't exist on the others and would vanish on redeploy. Object storage like S3 is durable, shared across all servers, scalable, and can be fronted by a CDN (CloudFront) for fast delivery, with signed URLs for private files.

**Q: What must the HTML form and view include for file uploads to work?**
Three things must line up. The HTML `<form>` must include `enctype="multipart/form-data"`, because file data is sent using multipart encoding, not the default URL encoding — without it the browser sends only the filename, not the file. The view must pass **both** `request.POST` and `request.FILES` when constructing the form (`Form(request.POST, request.FILES)`), since uploaded files arrive in `request.FILES`, separate from regular fields. And the model needs a `FileField` or `ImageField` with an `upload_to` path. Miss any of these — commonly the `enctype` or forgetting `request.FILES` — and the upload silently fails to save.

### Common Mistakes

- Omitting `enctype="multipart/form-data"`.
- Not passing `request.FILES` to the form.
- Storing files in the DB or on ephemeral local disk in production.
- Not validating file type/size (security and cost risk).
- Using `ImageField` without installing Pillow.

### Related Concepts

`request.FILES`, `MEDIA_ROOT`/`MEDIA_URL`, static files, `django-storages`, S3, CloudFront, `collectstatic`.

---

# 5. REST APIs & Django REST Framework

## 5.1 REST Fundamentals & HTTP Methods

### What is it?

**REST (Representational State Transfer)** is an architectural *style* for designing networked APIs. It's a set of conventions — not a protocol or library — for how a client and server communicate over HTTP. A **REST API** exposes your application's data as **resources** (nouns like `/articles/`, `/users/42/`) that clients manipulate using standard **HTTP methods** (verbs: GET, POST, PUT, PATCH, DELETE).

The core idea: **resources are nouns, HTTP methods are the verbs that act on them.**

| Method | Meaning | Example | Idempotent? |
|---|---|---|---|
| GET | Read a resource | `GET /articles/` (list), `GET /articles/5/` (one) | Yes |
| POST | Create a new resource | `POST /articles/` | No |
| PUT | Replace a resource entirely | `PUT /articles/5/` | Yes |
| PATCH | Partially update a resource | `PATCH /articles/5/` | No* |
| DELETE | Remove a resource | `DELETE /articles/5/` | Yes |

### Why is it needed?

Modern applications aren't just server-rendered web pages. A single backend often serves a web frontend (React/Vue), iOS and Android apps, and third-party integrations. All of them need a **common, language-agnostic way to read and write data.** REST provides that: it uses HTTP (which everything speaks), exchanges data as **JSON** (which every language parses), and follows predictable conventions so a developer can guess how an unfamiliar API works. It decouples frontend from backend — they can be built by different teams, in different languages, deployed independently.

### The 6 REST principles (constraints)

1. **Client–Server** — separation of concerns; UI and data storage evolve independently.
2. **Stateless** — each request contains all information needed; the server stores no client session between requests. (This is why token/JWT auth suits REST.)
3. **Cacheable** — responses declare whether they can be cached.
4. **Uniform Interface** — consistent resource URLs + standard methods.
5. **Layered System** — proxies, load balancers, gateways can sit between client and server transparently.
6. **Code on Demand** (optional) — server can send executable code (rarely used).

### How does it work?

```
Client (React app)                         Server (Django + DRF)
   │  GET /api/articles/          ───────►  returns JSON list, 200
   │  POST /api/articles/ {json}  ───────►  creates, returns new obj, 201
   │  PATCH /api/articles/5/ {..} ───────►  updates field, 200
   │  DELETE /api/articles/5/     ───────►  deletes, 204 No Content
```

Requests and responses carry **JSON** bodies and **HTTP status codes** indicating the outcome.

### Internal working / key concepts

- **Idempotency**: an operation is idempotent if doing it once or many times has the same effect. GET, PUT, DELETE are idempotent; POST is not (two POSTs create two resources). This matters for safe retries — a client can safely retry an idempotent request after a network failure.
- **Statelessness** means the server doesn't remember the client between requests, so authentication info (a token) must accompany *every* request. This enables horizontal scaling — any server can handle any request.

### Advantages

- Universal (HTTP + JSON), language-agnostic, decouples frontend/backend, cacheable, scalable (stateless), well-understood conventions, great tooling.

### Limitations

- **Over-/under-fetching** — a fixed endpoint may return too much or too little data (GraphQL addresses this).
- Multiple round-trips for related data.
- No built-in schema/typing (mitigated by OpenAPI).
- Not ideal for real-time (use websockets) or high-throughput internal RPC (use gRPC).

### Real-world applications

Virtually every public API (Stripe, Twitter/X, GitHub) and most SaaS backends expose REST APIs. Mobile apps and SPAs almost always talk to a REST (or GraphQL) backend.

### Interview Questions & Model Answers

**Q: What is REST and what makes an API "RESTful"?**
REST is an architectural style for building APIs over HTTP. An API is RESTful when it models the application's data as **resources** addressed by URLs (nouns like `/users/42/`), manipulates them using standard **HTTP methods** as verbs (GET to read, POST to create, PUT/PATCH to update, DELETE to remove), communicates outcomes via **HTTP status codes**, and typically exchanges **JSON**. Beyond that, it should honour REST's constraints — notably **statelessness** (each request is self-contained, the server keeps no per-client session) and a **uniform interface** (consistent, predictable URL and method conventions). The payoff is a standardised, language-agnostic contract that any client — browser, mobile app, or another service — can consume.

**Q: PUT vs PATCH — what's the difference?** *(Checklist item — very common.)*
Both update an existing resource, but differ in scope. **PUT replaces the entire resource** — the client sends the full representation, and the server overwrites the resource with it; any field omitted is typically treated as being set to empty/default. **PATCH applies a partial update** — the client sends only the fields that should change, and the rest are left as they are. So to change just an article's title, PATCH with `{"title": "New"}` is appropriate; PUT would require sending the whole article or risk wiping unspecified fields. There's also an idempotency nuance: PUT is idempotent (sending the same full representation repeatedly yields the same state), while PATCH isn't necessarily idempotent depending on the operation. In DRF, `PUT` maps to a full update and `PATCH` to `partial=True`.

**Q: What does "stateless" mean and why does REST require it?**
Stateless means the server does **not** retain any client context between requests — every request must carry everything needed to process it, including authentication. There's no server-side session tying a sequence of requests together. REST favours this because it makes the system **scalable and reliable**: since no request depends on server-held state, any server instance behind a load balancer can handle any request, you can add/remove servers freely, and a crashed server doesn't lose session data. The practical consequence is authentication design — instead of a server-side session cookie, RESTful APIs typically send a **token or JWT** with every request, so each request authenticates itself independently.

**Q: What is idempotency and why does it matter?**
An operation is idempotent if performing it multiple times has the same result as performing it once. GET (reading changes nothing), PUT (setting a resource to a specific state), and DELETE (deleting an already-deleted resource still leaves it deleted) are idempotent; POST is not, because each call creates a new resource. It matters for **safe retries**: networks fail, and a client that doesn't get a response can't tell whether the request succeeded. If the operation is idempotent, the client can safely retry without side effects. If it's not (like POST), retrying might create duplicates, which is why systems use techniques like idempotency keys for critical non-idempotent operations such as payments.

### Common Mistakes

- Using verbs in URLs (`/getArticles/`) instead of nouns + methods (`GET /articles/`).
- Using POST for everything.
- Returning 200 for errors instead of proper status codes.
- Making PUT behave like PATCH (partial) or vice versa.

### Related Concepts

HTTP methods, status codes, JSON, statelessness, idempotency, JWT, GraphQL, gRPC.

---

## 5.2 HTTP Status Codes

### What is it?

**HTTP status codes** are three-digit numbers the server returns to tell the client the *outcome* of a request. They're grouped by first digit:
- **2xx Success** — it worked.
- **3xx Redirection** — go elsewhere.
- **4xx Client Error** — the request was wrong (client's fault).
- **5xx Server Error** — the server failed (server's fault).

The syllabus's key codes:

| Code | Name | Meaning / when to use |
|---|---|---|
| **200** | OK | Successful GET/PUT/PATCH; request succeeded with a body |
| **201** | Created | Successful POST that created a resource |
| **204** | No Content | Success, but no body to return (typical for DELETE) |
| **400** | Bad Request | Malformed/invalid input (validation failed) |
| **401** | Unauthorized | Not authenticated (no/invalid credentials) |
| **403** | Forbidden | Authenticated but not allowed (no permission) |
| **404** | Not Found | Resource doesn't exist |
| **500** | Internal Server Error | Unhandled server-side bug |

### Why is it needed?

Clients need a **standard, machine-readable signal** of what happened, without parsing the body. A frontend checks the status code to decide whether to show data (200), redirect to login (401), show a "forbidden" message (403), display "not found" (404), or show a generic error (500). Using the correct codes makes APIs predictable, debuggable, and interoperable with tooling (monitoring, retries, caches all react to status codes).

### How does it work / Internal working

The status code is part of the HTTP response's first line (`HTTP/1.1 201 Created`). In DRF you return them via `Response(data, status=status.HTTP_201_CREATED)`. Middlewares, proxies, browsers, and monitoring tools all interpret them: caches store 200s, browsers follow 3xx, error trackers alert on 5xx.

### The critical 401 vs 403 distinction

- **401 Unauthorized** = "I don't know who you are" — authentication is missing or invalid. The fix is to log in / send valid credentials.
- **403 Forbidden** = "I know who you are, but you're not allowed" — authenticated but lacking permission. Logging in again won't help.

(The name "Unauthorized" for 401 is a historical misnomer — it's really about *authentication*.)

### Advantages / Limitations

- **Advantage:** standardised, tooling-friendly, self-documenting outcomes.
- **Limitation:** some situations are ambiguous (404 vs 403 to hide existence); teams sometimes misuse codes (returning 200 with an error body), which breaks the contract.

### Interview Questions & Model Answers

**Q: Difference between 401 and 403?**
Both are 4xx client errors about access, but they mean different things. **401 Unauthorized** means the request lacks valid **authentication** — the server doesn't know who you are because you sent no credentials or invalid ones. The remedy is to authenticate (log in, provide a valid token). **403 Forbidden** means you *are* authenticated — the server knows who you are — but you don't have **permission** to perform this action or access this resource. Re-authenticating won't help; you simply aren't allowed. In short: 401 is "who are you?", 403 is "you can't do that." This maps directly to authentication vs authorization.

**Q: Which status code for a successful POST that creates a resource, and for a DELETE?**
A successful **POST that creates a resource** should return **201 Created**, ideally with the newly created resource in the body and often a `Location` header pointing to its URL. Returning 200 works but 201 is more precise and signals that something new now exists. A successful **DELETE** typically returns **204 No Content** — the operation succeeded and there's no meaningful body to send back. Using these specific codes rather than a blanket 200 makes the API's behaviour explicit and lets clients and tooling respond correctly.

**Q: When would you return 400 vs 500?**
Return **400 Bad Request** when the *client* sent something wrong — invalid JSON, a missing required field, data that fails validation. It signals "fix your request." Return **500 Internal Server Error** when the *server* failed unexpectedly — an unhandled exception, a bug, a database that's down. It signals "the request was fine, but we broke." The distinction matters for debugging and responsibility: 4xx errors point the client to correct their input, while 5xx errors alert your team that something needs fixing on the backend. A well-behaved API never returns 500 for predictable validation problems — those should be 400 with a clear error message.

### Common Mistakes

- Returning 200 for everything, including errors.
- Confusing 401 and 403.
- Using 404 when 400 (bad input) is appropriate, or vice versa.
- Leaking stack traces in 500 responses (should be generic in production).

### Related Concepts

Authentication vs authorization, DRF `Response`, exception handling, idempotency, REST conventions.

---

## 5.3 Django REST Framework (DRF): Setup & Serializers

### What is it?

**Django REST Framework (DRF)** is the standard third-party toolkit for building REST APIs on top of Django. It adds serializers, API views, authentication classes, permissions, pagination, filtering, and a browsable API.

A **Serializer** is DRF's most fundamental concept: it **converts between complex data (Django model instances/querysets) and native Python types that can be rendered to JSON**, *and* validates incoming JSON before turning it into model data. Think of it as the API-world equivalent of a Django Form.

- **Serialization**: model instance → Python dict → JSON (for responses).
- **Deserialization**: JSON → validated Python data → model instance (for requests).

```python
from rest_framework import serializers
from .models import Article

class ArticleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Article
        fields = ["id", "title", "body", "author", "created"]
        read_only_fields = ["created"]
```

### Why is it needed?

Django models are Python objects; APIs speak JSON. Something must translate between them **in both directions**, and crucially, **validate** untrusted incoming JSON before it touches the database. Writing that translation and validation by hand for every model is tedious and error-prone. Serializers automate it: `ModelSerializer` derives fields from the model automatically, handles nested relationships, enforces validation rules, and produces clean JSON. They are the safe boundary between the outside world and your data — the API analogue of forms.

### How does it work?

```python
# Serialization (output)
article = Article.objects.get(pk=1)
ArticleSerializer(article).data          # -> {"id":1, "title":"...", ...}

# Serialization of many
ArticleSerializer(Article.objects.all(), many=True).data

# Deserialization (input) + validation
s = ArticleSerializer(data=request.data)
if s.is_valid():                          # runs validation
    s.save()                              # creates/updates the model
else:
    s.errors                              # validation error messages
```

### Internal working

- A **`Serializer`** declares fields explicitly (like a `Form`); a **`ModelSerializer`** introspects the model to auto-generate fields and provides default `create()`/`update()` implementations.
- On input, `is_valid()` runs field-level validation, `validate_<field>()` methods, and object-level `validate()`, populating `validated_data` or `errors`.
- **Nested serializers** represent relationships: embedding an `AuthorSerializer` inside `ArticleSerializer` outputs the author as a nested object instead of just an ID.
- `read_only`/`write_only` fields control direction (e.g., a password is write-only; a computed field is read-only).

### Advantages

- Automatic, bidirectional model↔JSON conversion; built-in validation; nested relationships; mirrors forms so it's familiar; integrates with the rest of DRF (views, permissions).

### Limitations

- Deeply nested serializers can be slow (N+1 queries) and complex to write for writes.
- Auto-generated fields sometimes need overriding for custom output.
- Learning curve around `create()`/`update()` for nested writes.

### Interview Questions & Model Answers

**Q: What is a serializer and what are its two jobs?**
A serializer is DRF's translation and validation layer between Django objects and JSON. Its two jobs are **serialization** — converting model instances or querysets into native Python data types that render to JSON for API responses — and **deserialization** — taking incoming JSON, **validating** it, and converting it into data you can save as a model. The validation part is critical: it's the gate that ensures untrusted client input is well-formed and safe before it reaches your database. A `ModelSerializer` automates most of this by generating fields from a model, much like a `ModelForm` does for HTML forms, which is why serializers are often described as "forms for APIs."

**Q: Serializer vs ModelSerializer?**
A plain **`Serializer`** requires you to declare every field explicitly and to write the `create()` and `update()` methods that turn validated data into saved objects — you use it when the data doesn't map neatly to one model, or you need full control. A **`ModelSerializer`** is a shortcut for the common case where the serializer corresponds to a model: it introspects the model and auto-generates the fields, and it provides default `create()`/`update()` implementations, so a typical CRUD serializer is just a `Meta` with `model` and `fields`. The relationship mirrors `Form` vs `ModelForm`: use `ModelSerializer` for straightforward model-backed endpoints and drop to `Serializer` for custom or multi-model payloads.

**Q: How do nested serializers work and what's the risk?**
A nested serializer embeds one serializer inside another to represent a relationship — for example, putting an `AuthorSerializer` as a field on `ArticleSerializer` so each article's response includes the full author object rather than just an `author_id`. This produces richer, more convenient JSON. The main risk is performance: if you serialize a list of articles and each one triggers a separate query to fetch its author, you get the **N+1 query problem**. The fix is to optimise the underlying queryset with `select_related` (for foreign keys) or `prefetch_related` (for many-to-many/reverse relations) so the related data is fetched efficiently. Nested *writes* are also non-trivial — you often must override `create()`/`update()` to handle saving related objects.

### Common Mistakes

- Not optimising querysets behind nested serializers → N+1 queries.
- Forgetting `many=True` when serializing a queryset.
- Exposing sensitive fields (e.g., password) instead of marking them `write_only`.
- Assuming `ModelSerializer` handles nested writes automatically (it doesn't).

### Related Concepts

Forms, validation, `select_related`/`prefetch_related`, `read_only`/`write_only`, DRF views.

---

## 5.4 API Views: APIView, Generic Views & Mixins

### What is it?

DRF offers a spectrum of ways to write API endpoints, trading control for brevity:
- **`APIView`** — the base class; like a Django CBV but API-aware (parses JSON, uses DRF `Request`/`Response`, applies authentication/permissions). You write `get()`, `post()` etc. manually.
- **Mixins** (`ListModelMixin`, `CreateModelMixin`, `RetrieveModelMixin`, `UpdateModelMixin`, `DestroyModelMixin`) — reusable pieces implementing one CRUD action each.
- **Generic views** (`ListAPIView`, `RetrieveUpdateDestroyAPIView`, etc.) — pre-combined mixins for common patterns; you just set `queryset` and `serializer_class`.

```python
# Most explicit: APIView
class ArticleList(APIView):
    def get(self, request):
        data = ArticleSerializer(Article.objects.all(), many=True).data
        return Response(data)

# Most concise: generic view
class ArticleList(generics.ListCreateAPIView):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
```

### Why is it needed?

APIs repeat the same CRUD patterns as web pages. `APIView` gives you full control for custom endpoints, but writing list/create/retrieve/update/delete by hand is repetitive. Mixins and generic views encapsulate those patterns, so a full CRUD endpoint becomes a couple of lines. The spectrum lets you pick the right level: full control (`APIView`) for unusual endpoints, maximum brevity (generic views) for standard CRUD.

### How does it work / Internal working

- **`APIView`** wraps Django's `View` but swaps in DRF's request/response, content negotiation (JSON in/out), and hooks for `authentication_classes` and `permission_classes`.
- **Mixins** each provide one method: `ListModelMixin.list()`, `CreateModelMixin.create()`, etc., operating on `self.get_queryset()` and `self.get_serializer()`.
- **Generic views** combine a `GenericAPIView` (which provides `queryset`/`serializer_class`/pagination/filtering plumbing) with the relevant mixins and wire them to HTTP methods. E.g. `ListCreateAPIView` = `GenericAPIView` + `ListModelMixin` + `CreateModelMixin`, mapping GET→list, POST→create.

### Advantages / Limitations

- **Advantage:** choose your abstraction level; generic views eliminate boilerplate; consistent structure.
- **Limitation:** like CBVs, generic views hide behaviour in parent classes; heavy customisation can be harder to follow than an `APIView`.

### Interview Questions & Model Answers

**Q: Explain the progression from APIView to generic views.**
DRF gives you a ladder of abstraction. At the bottom is **`APIView`**: an API-aware base class where you write each HTTP handler (`get`, `post`) yourself and control everything — ideal for custom endpoints. Next come **mixins**, each implementing a single CRUD action (`list`, `create`, `retrieve`, `update`, `destroy`) against a queryset and serializer; you combine the ones you need with `GenericAPIView`. At the top are **generic views** like `ListCreateAPIView` or `RetrieveUpdateDestroyAPIView`, which are pre-assembled combinations of those mixins — you just declare `queryset` and `serializer_class` and get a full endpoint in a couple of lines. The progression trades control for conciseness: use `APIView` when logic is bespoke, generic views when it's standard CRUD, and mixins when you want a specific subset.

**Q: What does GenericAPIView add over APIView?**
`GenericAPIView` extends `APIView` with the plumbing that CRUD endpoints commonly need: a `queryset` and `serializer_class` attribute, a `get_queryset()`/`get_serializer()` method pair, and built-in integration with **pagination**, **filtering**, and object lookup (`get_object()` using a lookup field). On its own it doesn't handle any HTTP method — you combine it with mixins (or use a ready-made generic view) to get `list`/`create`/etc. Essentially, `APIView` gives you the request/response and auth/permission framework, and `GenericAPIView` layers on the model-and-serializer conventions so mixins can implement CRUD generically.

### Common Mistakes

- Using `APIView` and manually reimplementing what a generic view already does.
- Overriding `get()`/`post()` on a generic view instead of the intended hooks (`get_queryset`, `perform_create`).
- Forgetting `serializer_class` or `queryset` on a generic view.

### Related Concepts

CBVs, mixins, viewsets, `get_queryset`/`perform_create`, pagination/filtering.

---

## 5.5 ViewSets & Routers

### What is it?

A **ViewSet** groups the logic for a set of related endpoints (all the CRUD actions for a resource) into a **single class**, instead of separate views for list/detail. A **Router** then automatically generates the URL patterns for a viewset, wiring HTTP methods to actions.

- **`ModelViewSet`** — full CRUD (list, create, retrieve, update, partial_update, destroy) in one class.
- **`ReadOnlyModelViewSet`** — only list + retrieve (read-only API).
- **Routers** (`DefaultRouter`, `SimpleRouter`) — generate the URLs.

```python
# views.py
class ArticleViewSet(viewsets.ModelViewSet):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer

# urls.py
router = DefaultRouter()
router.register("articles", ArticleViewSet)
urlpatterns = router.urls
# -> generates: GET/POST /articles/, GET/PUT/PATCH/DELETE /articles/{pk}/
```

### Why is it needed?

For a resource, list/create/retrieve/update/delete are all closely related and share a queryset and serializer. Writing five separate view classes and hand-registering their URLs is repetitive. A **ViewSet** consolidates them into one cohesive class, and a **Router** removes the URL boilerplate entirely by generating conventional RESTful routes automatically. This is the highest level of abstraction in DRF — a full REST resource in a handful of lines — and it keeps URL structure consistent across the API.

### How does it work / Internal working

- A viewset defines **actions** (`list`, `create`, `retrieve`, `update`, `partial_update`, `destroy`) rather than HTTP-method handlers. `ModelViewSet` inherits all of them from the generic mixins.
- When a router registers a viewset, it calls `as_view()` with a mapping like `{"get": "list", "post": "create"}` for the collection URL and `{"get": "retrieve", "put": "update", "patch": "partial_update", "delete": "destroy"}` for the detail URL — translating HTTP methods to viewset actions.
- **`DefaultRouter`** also adds a browsable API root and format suffixes; **`SimpleRouter`** is minimal.
- Custom actions are added with the `@action` decorator (e.g., `POST /articles/{pk}/publish/`).

### Advantages

- Maximum conciseness (full CRUD in a few lines); consistent, conventional URLs; less URL boilerplate; custom actions via `@action`.

### Limitations

- Most "magic" of all DRF options — routes and behaviour are implicit, which can confuse newcomers and complicate unusual URL schemes.
- Fine-grained per-endpoint customisation sometimes fights the abstraction.

### Interview Questions & Model Answers

**Q: What's the difference between APIView, generic views, and viewsets?** *(Checklist item.)*
They're increasing levels of abstraction for building API endpoints. **`APIView`** is the base — you handle each HTTP method yourself with full control, best for custom logic. **Generic views** (like `ListCreateAPIView`) combine mixins to implement standard CRUD for a *single* URL pattern with minimal code — you set a queryset and serializer. **ViewSets** go further: a single class (`ModelViewSet`) implements *all* CRUD actions for a resource at once, and paired with a **router** it auto-generates all the URL patterns. The trade-off is control vs conciseness: `APIView` gives the most control and least magic; viewsets give the least code and most convention. For a standard REST resource, a `ModelViewSet` + router is the idiomatic, DRY choice; for something bespoke, `APIView` is clearer.

**Q: How does a router know which method maps to which action?**
A router registers a viewset by calling its `as_view()` with an explicit mapping of HTTP methods to viewset action names. For the **collection** endpoint (`/articles/`) it maps `GET → list` and `POST → create`; for the **detail** endpoint (`/articles/{pk}/`) it maps `GET → retrieve`, `PUT → update`, `PATCH → partial_update`, and `DELETE → destroy`. So when a request arrives, the generated view dispatches to the corresponding action method on the viewset. This convention is why a `ModelViewSet` plus a router produces a complete, RESTful URL set automatically. Custom endpoints beyond CRUD are added with the `@action` decorator, and the router generates routes for those too.

**Q: When would you use ReadOnlyModelViewSet?**
`ReadOnlyModelViewSet` provides only the read actions — `list` and `retrieve` — and no create/update/delete. You use it for resources that clients should be able to **read but not modify** through the API: reference data like a list of countries or product categories, or any endpoint you want to expose safely without write access. It's more explicit and safer than a full `ModelViewSet` with permissions bolted on, because the write actions simply don't exist, so there's no risk of accidentally exposing them. If later you need write access for admins, you'd switch to a full viewset with appropriate permissions.

### Common Mistakes

- Forgetting to register the viewset with a router (no URLs generated).
- Using a `ModelViewSet` when you only need read access (use `ReadOnlyModelViewSet`).
- Trying to force non-standard URL schemes through routers instead of using explicit routes.
- Not using `@action` for custom endpoints, hacking them in awkwardly.

### Related Concepts

Routers, `@action`, generic views, mixins, nested routes (drf-nested-routers).

---

## 5.6 API Authentication: Session, Token & JWT

### What is it?

How an API verifies the identity of the client making each request. DRF supports several schemes:
- **Session Authentication** — reuses Django's session/cookie system. Good for browsers on the same domain (e.g., a React app served from the same site).
- **Token Authentication** — the server issues a random **token** stored in the DB; the client sends it in the `Authorization: Token <key>` header on each request. Stateless-ish (token looked up in DB).
- **JWT (JSON Web Token) Authentication** — the server issues a **signed, self-contained token** encoding the user's identity and expiry; the server verifies the signature without a DB lookup. Fully stateless. Provided by `djangorestframework-simplejwt`.

### Why is it needed?

REST is **stateless**, and APIs are consumed by clients that don't behave like browsers — mobile apps, other servers, SPAs on different domains — where cookies are awkward or unavailable. These clients need a way to prove identity on **every** request without a server-side session. Tokens (especially JWT) fit this: the client obtains a token once (by logging in) and attaches it to each subsequent request. JWT specifically avoids a database lookup per request by carrying the identity inside a cryptographically **signed** payload, which scales well.

### How does it work? — JWT flow

```
1. POST /api/token/  {username, password}   ──►  server verifies
2. Server returns   {access: "<jwt>", refresh: "<jwt>"}
3. Client stores tokens; on each request sends:
      Authorization: Bearer <access-jwt>
4. Server verifies the signature + expiry (no DB hit) -> request.user set
5. When access token expires, client POSTs the refresh token to get a new access token.
```

A JWT has three parts: `header.payload.signature` (base64url), e.g. `eyJ...` — header (algorithm), payload (claims like user id, exp), and a signature the server checks with its secret key.

### Internal working

- **Session auth:** relies on the session cookie + CSRF; stateful; server stores the session.
- **Token auth:** a random token row in the DB maps to a user; each request looks it up. Simple but requires a DB read and revocation is easy (delete the row).
- **JWT:** the token is **signed** (HMAC with the secret, or RSA). The server recomputes the signature to verify the token wasn't tampered with, and reads the embedded `exp` (expiry) and user id — **no DB lookup**. Because it's stateless, you can't easily revoke a JWT before it expires, which is why access tokens are **short-lived** and paired with a longer-lived **refresh token** used to get new access tokens.

### Advantages

- **Session:** simple for same-site web apps; leverages Django's mature system.
- **Token:** simple, works for any client, easy revocation.
- **JWT:** stateless, no per-request DB hit, scales horizontally, carries claims, works across services/domains.

### Limitations

- **Session:** stateful, cross-domain/mobile-unfriendly, needs CSRF handling.
- **Token:** DB lookup per request; single opaque token.
- **JWT:** **can't be revoked before expiry** (mitigated by short expiry + refresh + a blocklist); if the secret leaks, all tokens are compromised; storing JWTs in the browser has XSS/CSRF trade-offs (localStorage vs cookies).

### Real-world applications

JWT is the dominant scheme for SPAs and mobile apps talking to REST APIs. Token auth is common for simpler APIs and machine-to-machine access. Session auth is used when the frontend is served from the same Django site.

### Interview Questions & Model Answers

**Q: Session authentication vs JWT authentication — compare them.** *(Checklist item.)*
Both authenticate API requests but differ fundamentally in where state lives. **Session auth** is **stateful**: on login the server stores a session and gives the browser a cookie; each request sends the cookie and the server looks up the session to identify the user. It's simple and secure for a web frontend on the same domain but is awkward for mobile apps and cross-domain SPAs, and it requires CSRF protection because it's cookie-based. **JWT auth** is **stateless**: on login the server returns a cryptographically signed token that encodes the user's identity and an expiry; the client sends it in the `Authorization: Bearer` header on every request, and the server verifies the signature without any database or session lookup. JWT scales better (no shared session store needed), works naturally for mobile and cross-domain clients, but has weaker revocation — you can't easily invalidate a token before it expires, so you use short-lived access tokens plus refresh tokens. Rule of thumb: session auth for same-site web apps, JWT for APIs serving mobile/SPA/third-party clients.

**Q: Why do JWTs use a short-lived access token plus a refresh token?**
Because JWTs are stateless and self-contained, the server validates them purely by checking the signature and expiry — it doesn't consult a database, which is what makes them scalable but also means it **can't easily revoke one before it expires**. If access tokens were long-lived and one were stolen, the attacker would have prolonged access. The solution is to make the **access token short-lived** (minutes), so a stolen one is useful only briefly, and to issue a longer-lived **refresh token** that the client exchanges for new access tokens. The refresh token can be stored more securely and revoked (via a blocklist) if compromised. This design balances JWT's stateless performance with acceptable security, limiting the damage window of a leaked access token.

**Q: What are the security concerns with JWTs, and how do you mitigate them?**
Several. First, **revocation**: you can't invalidate a valid JWT before expiry, mitigated with short access-token lifetimes, refresh tokens, and a server-side blocklist for critical cases. Second, **secret compromise**: since JWTs are verified with the server's signing key, if that key leaks an attacker can mint valid tokens — so the key must be strongly protected and rotated. Third, **storage on the client**: storing JWTs in `localStorage` exposes them to XSS (any injected script can read them), while storing them in cookies exposes them to CSRF — you pick the lesser risk for your app (often HttpOnly cookies with SameSite, plus CSRF defences). Fourth, never put **sensitive data** in the payload, because it's only base64-encoded, not encrypted, so anyone can read it. Proper JWT use means short expiry, protected secrets, careful storage, and minimal, non-sensitive claims.

### Common Mistakes

- Storing sensitive data in the JWT payload (it's readable by anyone).
- Long-lived access tokens with no refresh strategy.
- Believing JWTs are encrypted (they're signed, not encrypted).
- Not planning revocation, then being unable to log users out.
- Using session auth for a cross-domain mobile client.

### Related Concepts

Statelessness, `djangorestframework-simplejwt`, CSRF, refresh tokens, OAuth2, token storage (localStorage vs cookies).

---

## 5.7 DRF Permissions

### What is it?

DRF **permissions** control *who can access an endpoint* — the API-layer authorization, checked after authentication. Built-in classes:
- **`IsAuthenticated`** — only logged-in users.
- **`IsAdminUser`** — only `is_staff` users.
- **`IsAuthenticatedOrReadOnly`** — anyone can read (GET), only authenticated can write.
- **`AllowAny`** — no restriction.
- **Custom permissions** — subclass `BasePermission` and implement `has_permission()` (endpoint-level) and/or `has_object_permission()` (object-level).

```python
class ArticleViewSet(viewsets.ModelViewSet):
    permission_classes = [IsAuthenticatedOrReadOnly]

class IsOwnerOrReadOnly(BasePermission):
    def has_object_permission(self, request, view, obj):
        if request.method in SAFE_METHODS:      # GET/HEAD/OPTIONS
            return True
        return obj.author == request.user       # only owner can edit
```

### Why is it needed?

Authentication tells you *who* the user is; permissions decide *what they may do*. Not every authenticated user should modify every resource — an author edits only their own articles, admins do more, anonymous users maybe only read. Permissions enforce these rules declaratively and consistently at the API boundary, keeping authorization logic out of every view body and preventing accidental exposure of data or actions.

### How does it work / Internal working

- DRF runs `permission_classes` on each request. `has_permission(request, view)` is checked first (endpoint-level: "can this user hit this endpoint at all?"). For detail views, `has_object_permission(request, view, obj)` is then checked (object-level: "can they act on *this* object?").
- If any permission returns `False`, DRF returns **403 Forbidden** (or **401** if unauthenticated).
- `SAFE_METHODS` (GET/HEAD/OPTIONS) let you allow reads while restricting writes.

### Advantages / Limitations

- **Advantage:** declarative, reusable, composable; separates object-level from endpoint-level checks; integrates with authentication.
- **Limitation:** `has_object_permission` is only invoked for object lookups (`get_object()`), not for list endpoints — filtering *which* objects a user can see in a list must be done in `get_queryset()`, a common source of data leaks.

### Interview Questions & Model Answers

**Q: How do DRF permissions work, and what's the difference between `has_permission` and `has_object_permission`?**
DRF checks a view's `permission_classes` on every request to decide access, running after authentication. There are two levels. **`has_permission(request, view)`** is the endpoint-level check — "is this user allowed to access this view at all?" — used for things like requiring authentication or admin status. **`has_object_permission(request, view, obj)`** is the object-level check — "is this user allowed to act on *this specific* object?" — used for rules like "only the author may edit their own article." DRF calls `has_permission` first; then, for views that retrieve a single object via `get_object()`, it also calls `has_object_permission`. The important caveat is that object-level permission is **not** automatically applied to list endpoints, so restricting which objects appear in a list must be done by filtering the queryset in `get_queryset()`.

**Q: How would you implement "users can only edit their own objects"?**
This is an object-level permission. I'd write a custom permission class subclassing `BasePermission` with a `has_object_permission` method: allow all safe methods (GET/HEAD/OPTIONS) for everyone, but for write methods return whether `obj.owner == request.user`. I'd attach it via `permission_classes`. Crucially, that only governs detail actions (retrieve/update/delete on a specific object). For the **list** endpoint I'd also override `get_queryset()` to return only the requesting user's objects, because object-level permissions aren't applied per-item in a list — without that, a user could *see* others' objects even if they can't edit them. So the complete answer combines a custom object permission for writes with queryset filtering for reads.

### Common Mistakes

- Relying on `has_object_permission` to filter list endpoints (it doesn't run per item).
- Forgetting to set `permission_classes` and leaving an endpoint open (or relying only on the global default).
- Confusing authentication failure (401) with permission failure (403).
- Not allowing `SAFE_METHODS` when you intend reads to be public.

### Related Concepts

Authentication, `get_queryset` filtering, object-level permissions, `SAFE_METHODS`, model vs object permissions.

---

## 5.8 Pagination, Filtering & API Documentation

### What is it?

Three concerns for making a list API usable and discoverable:

**Pagination** — splitting large result sets into pages instead of returning everything at once. DRF styles:
- **PageNumberPagination** — `?page=2` (classic page numbers).
- **LimitOffsetPagination** — `?limit=10&offset=20` (like SQL LIMIT/OFFSET).
- **CursorPagination** — opaque cursor pointing to a position; efficient and consistent for large, frequently-changing datasets.

**Filtering** — letting clients narrow results:
- **SearchFilter** — `?search=django` (text search across configured fields).
- **OrderingFilter** — `?ordering=-created` (sort).
- **DjangoFilterBackend** (django-filter) — `?author=5&published=true` (field-based filtering).

**API Documentation** — machine- and human-readable descriptions of the API:
- **OpenAPI** — the specification/standard (formerly Swagger spec).
- **Swagger UI** — interactive docs UI.
- **drf-spectacular** — generates an OpenAPI 3 schema (and Swagger UI) from your DRF code.

### Why is it needed?

- **Pagination:** returning a table of a million rows in one response would be slow, memory-heavy, and could crash clients. Pagination bounds response size, protects the server and DB, and enables infinite-scroll/next-page UIs.
- **Filtering:** clients rarely want *all* records — they want "published articles by author 5, newest first." Server-side filtering avoids sending huge payloads the client would discard, and pushes the work to the database where it's efficient (and indexable).
- **Documentation:** an API nobody understands is unusable. OpenAPI docs let frontend/mobile/third-party developers discover endpoints, parameters, and response shapes, and even generate client code and test requests interactively.

### How does it work?

```python
# settings.py — global pagination
REST_FRAMEWORK = {
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 20,
}

# a view enabling filtering/search/ordering
class ArticleViewSet(viewsets.ModelViewSet):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ["author", "published"]
    search_fields = ["title", "body"]
    ordering_fields = ["created", "title"]
# GET /articles/?author=5&search=django&ordering=-created&page=2
```

A paginated response typically looks like:
```json
{ "count": 137, "next": "…?page=3", "previous": "…?page=1", "results": [ ... ] }
```

### Internal working

- **Pagination** wraps the queryset: it slices it (`LIMIT/OFFSET` or a cursor `WHERE` clause) and builds `next`/`previous` links. **CursorPagination** uses a `WHERE ordering_field > last_seen` approach, which stays fast and consistent even as rows are inserted (unlike offset, which can skip/duplicate rows and gets slower deep into the data).
- **Filtering** backends translate query params into ORM `.filter()`/`.order_by()` calls before pagination runs.
- **drf-spectacular** introspects your serializers, views, and routers to emit an **OpenAPI 3** JSON/YAML schema; Swagger UI renders that schema into interactive docs.

### Advantages / Limitations

- **Advantage:** bounded responses, efficient DB-side filtering, discoverable self-documenting APIs.
- **Limitation:** offset pagination degrades on deep pages and can be inconsistent under concurrent writes (cursor fixes this but loses random page access); over-flexible filtering can enable expensive queries (need indexes/limits); auto-generated docs sometimes need annotations for accuracy.

### Interview Questions & Model Answers

**Q: Why paginate, and when would you choose cursor pagination over page-number pagination?**
You paginate to avoid returning unbounded result sets — sending thousands or millions of rows in one response is slow, memory-hungry, and can overwhelm both the database and the client. Pagination caps each response to a manageable size and provides links to fetch subsequent pages. **Page-number** (and limit/offset) pagination is simple and lets users jump to an arbitrary page, but it has two weaknesses on large datasets: deep offsets get slow because the database still scans and skips all preceding rows, and results can shift or duplicate if rows are inserted/deleted between requests. **Cursor pagination** encodes a pointer to the last-seen item and fetches "the next N after this cursor" using an indexed `WHERE` clause, so it stays fast regardless of depth and is consistent under concurrent writes. The trade-off is you lose random page access (no "jump to page 50"). So: page numbers for small/stable datasets or when users need to jump around; cursor pagination for large, high-write, or infinite-scroll feeds.

**Q: Why do filtering and searching server-side instead of fetching everything and filtering on the client?**
Because it's far more efficient and scalable. If the client fetched every record and filtered locally, you'd transfer huge payloads over the network that mostly get discarded, consume client memory, and put the filtering burden on a device that may be slow — and it simply doesn't work once the dataset is large. Server-side filtering pushes the work to the **database**, which is optimised for it and can use **indexes** to answer queries quickly, and then only the relevant, already-paginated rows travel over the wire. DRF's filter backends (DjangoFilterBackend, SearchFilter, OrderingFilter) translate query parameters into ORM filters so this is declarative. The one caution is to index the filtered/searched columns and constrain the allowed filters, so clients can't trigger arbitrarily expensive scans.

**Q: What is OpenAPI/Swagger and why document an API?**
OpenAPI is a standard, machine-readable format for describing a REST API — its endpoints, methods, parameters, request/response schemas, and auth. Swagger UI is a tool that renders an OpenAPI document into interactive, browsable documentation where developers can see every endpoint and even try requests live. In the DRF world, `drf-spectacular` generates the OpenAPI schema automatically from your serializers and views. Documenting an API matters because an API is a contract consumed by other developers — frontend, mobile, and third parties — who need to know exactly what endpoints exist, what to send, and what they'll get back. Good docs reduce integration time and errors, and a machine-readable schema additionally enables auto-generated client SDKs, contract testing, and mock servers. An undocumented API is effectively unusable by anyone who didn't write it.

### Common Mistakes

- Not paginating list endpoints → giant responses and slow queries.
- Using offset pagination on huge, fast-changing datasets (inconsistency, slow deep pages).
- Filtering/searching on unindexed columns → full table scans.
- Shipping an API with no documentation.
- Fetching all data to the client and filtering there.

### Related Concepts

QuerySet slicing, database indexes, django-filter, drf-spectacular, OpenAPI, cursor vs offset, N+1.

---

# 6. Deployment & AWS

## 6.1 Deployment Fundamentals (Env Vars, DEBUG, Secrets, Static/Media)

### What is it?

Deployment is the process of taking your app from your laptop to a server where real users can reach it. **Deployment fundamentals** are the settings and practices that must change between development and production so the app is secure, fast, and reliable:
- **Environment variables** — configuration (secrets, DB URLs) read from the environment, not hardcoded.
- **`DEBUG = False`** — never run production with debug on.
- **`SECRET_KEY`** — a secret cryptographic key kept out of source control.
- **`ALLOWED_HOSTS`** — the domains your app is allowed to serve.
- **Static files** — collected and served efficiently (WhiteNoise/CDN).
- **Media files** — user uploads stored durably (S3).

### Why is it needed?

Development settings are optimised for convenience and are **dangerous in production.** `DEBUG=True` shows detailed error pages that leak source code, settings, and even parts of your database — a huge security hole. Hardcoded secrets in source control get leaked the moment the repo is shared. The dev server can't handle real traffic. Deployment fundamentals exist to close these gaps: separate configuration from code (12-factor principle), hide secrets, harden security, and serve static/media correctly at scale.

### How does it work? — production checklist

```python
# settings.py (production essentials)
import os
DEBUG = False
SECRET_KEY = os.environ["SECRET_KEY"]                 # from env, not hardcoded
ALLOWED_HOSTS = os.environ["ALLOWED_HOSTS"].split(",")
DATABASES = {"default": dj_database_url.parse(os.environ["DATABASE_URL"])}

SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True

STATIC_ROOT = BASE_DIR / "staticfiles"                # collectstatic target
```
Then: `python manage.py collectstatic`, run under Gunicorn behind Nginx, with env vars supplied by the platform.

### Internal working

- **Environment variables** are read at process start (`os.environ`), so the same code image runs in dev/staging/prod with different config — this is the **12-factor "config in the environment"** principle, and it's what makes builds reproducible and secrets manageable.
- **`DEBUG=False`** disables the verbose error pages (Django then serves generic 404/500 pages) and enforces `ALLOWED_HOSTS` checking.
- **`collectstatic`** gathers static files from every app into one `STATIC_ROOT` directory so a single server/CDN can serve them.

### Advantages / Limitations

- **Advantage:** secure, reproducible, environment-agnostic deployments; secrets protected.
- **Limitation:** requires discipline and a config-management approach; misconfiguration (e.g., forgetting `collectstatic` or leaving `DEBUG=True`) causes outages or breaches.

### Interview Questions & Model Answers

**Q: Why must `DEBUG` be `False` in production?**
Because `DEBUG=True` makes Django return **detailed error pages** whenever something goes wrong, and those pages expose highly sensitive information: full stack traces, source code snippets, local variable values, settings, and sometimes database query contents. To an attacker, that's a roadmap of your application and potentially a direct leak of secrets. `DEBUG=True` also disables `ALLOWED_HOSTS` enforcement and serves static files in an insecure, inefficient way, and it keeps extra data in memory. In production you set `DEBUG=False` so Django shows generic error pages, enforces host validation, and behaves securely. Leaving debug on in production is one of the most common and serious Django misconfigurations.

**Q: How should secrets like `SECRET_KEY` and database passwords be managed?**
They should be kept **out of source code entirely** and injected through **environment variables** (or a secrets manager). Hardcoding them in `settings.py` means anyone with repo access — or anyone who obtains a leaked copy — gets your keys, and they end up permanently in git history. Instead, the code reads `os.environ["SECRET_KEY"]`, and the actual values are supplied by the deployment platform (env vars, a `.env` file that's git-ignored, AWS Secrets Manager, etc.). This follows the 12-factor principle of storing config in the environment, keeps secrets per-environment (dev vs prod differ), and lets you rotate a compromised key without changing code. The `SECRET_KEY` specifically underpins cryptographic signing (sessions, CSRF tokens, password resets), so if it leaks you must rotate it.

**Q: What is `collectstatic` and why is it needed?**
`collectstatic` is a management command that gathers all static files — CSS, JS, images — from every installed app and from your project into a single directory defined by `STATIC_ROOT`. It's needed because in development Django serves static files by searching each app's `static/` folder, which is convenient but slow and not how production works. In production you want static files in **one place** that a high-performance server (Nginx, WhiteNoise, or a CDN) can serve directly without involving Python. So the deployment flow is: run `collectstatic` to consolidate the files, then configure your web server or WhiteNoise to serve them from `STATIC_ROOT`. Forgetting to run it is a classic cause of "my CSS is missing in production."

### Common Mistakes

- Shipping with `DEBUG=True`.
- Committing `SECRET_KEY`/passwords to git.
- Forgetting `collectstatic`, leaving styles/JS broken.
- Empty or wildcard `ALLOWED_HOSTS`.
- Not enabling secure cookie / SSL redirect settings.

### Related Concepts

12-factor app, `django-environ`, Gunicorn, WhiteNoise, `ALLOWED_HOSTS`, `collectstatic`, secrets managers.

---

## 6.2 Gunicorn, Nginx & WhiteNoise (Serving Django in Production)

### What is it?

The production serving stack:
- **Gunicorn** — a production-grade **WSGI application server** that actually runs your Django code, managing multiple worker processes to handle concurrent requests.
- **Nginx** — a high-performance **web server / reverse proxy** that sits in front of Gunicorn: it terminates HTTPS, serves static files, load-balances, buffers slow clients, and forwards dynamic requests to Gunicorn.
- **WhiteNoise** — a library that lets Django/Gunicorn serve static files efficiently *without* Nginx (useful on platforms like Heroku/Render where you don't manage Nginx).
- **Procfile / Gunicorn** — how platforms know the command to start your app (`web: gunicorn config.wsgi`).

### Why is it needed?

Django's `runserver` is single-process, insecure, and slow — unusable for real traffic. **Gunicorn** runs your app across multiple worker processes so it handles many simultaneous requests. But an app server alone shouldn't face the raw internet: it's not optimised for serving static files, TLS, or slow/malicious clients. **Nginx** in front handles those efficiently, protects Gunicorn, and serves static assets far faster than Python can. This **Nginx → Gunicorn → Django** layering is the standard, battle-tested production architecture.

### How does it work?

```
Internet
   │  HTTPS
   ▼
┌─────────┐   /static/* ──────────────► served directly from disk
│  NGINX  │   (reverse proxy, TLS,
│         │    static, load balance)
└────┬────┘   dynamic requests
     │  proxy_pass -> 127.0.0.1:8000
     ▼
┌──────────┐   multiple worker processes
│ GUNICORN │───► Django app (WSGI)  ───► PostgreSQL
└──────────┘
```

### Internal working

- **Gunicorn** imports your `wsgi.py` callable and spawns N **worker processes** (rule of thumb: `2 × CPU cores + 1`). Each worker handles requests; the master monitors and restarts them. Because Python (with the GIL) doesn't parallelise CPU work in one process, multiple workers give real concurrency.
- **Nginx** accepts connections, handles TLS, and for dynamic paths **proxies** to Gunicorn over a local socket/port; for static paths it reads files directly. It also **buffers** requests/responses, shielding Gunicorn workers from slow clients (so a worker isn't tied up waiting on a slow connection).
- **WhiteNoise** hooks into Django's WSGI stack to serve compressed, cache-headed static files from within the app process — a simpler alternative when there's no separate Nginx.

### Advantages / Limitations

- **Advantage:** robust, scalable, secure production serving; clear separation (Nginx: I/O & static, Gunicorn: app).
- **Limitation:** more moving parts to configure; worker count/timeouts need tuning; long-lived/async workloads need different worker types (gevent/uvicorn workers) or ASGI.

### Interview Questions & Model Answers

**Q: What's the difference between Gunicorn and Nginx? Why use both?** *(Checklist item.)*
They do different jobs and complement each other. **Gunicorn** is a WSGI **application server** — it actually runs your Python/Django code, spawning multiple worker processes to execute your views and produce dynamic responses. **Nginx** is a **web server / reverse proxy** — it's extremely efficient at handling network I/O, terminating HTTPS, serving static files straight from disk, load-balancing, and protecting backend processes from slow or malicious clients. You use both because each is good at what the other isn't: Nginx shouldn't run Python, and Gunicorn shouldn't be exposed directly to the internet or bogged down serving static assets or waiting on slow connections. The standard architecture is Nginx in front as the public-facing layer, proxying dynamic requests to Gunicorn behind it, while serving static files itself. Gunicorn gives you concurrency for app code; Nginx gives you a fast, secure, static-serving front door.

**Q: Why can't you just use `runserver` in production?**
`runserver` is a development convenience: single-process (single-threaded by default), unoptimised, and never security-hardened for public exposure. It can't handle meaningful concurrency — a few simultaneous users would queue behind each other — and it lacks the robustness (worker management, timeouts, graceful restarts) that production needs. The Django docs explicitly warn against it. Production instead uses a proper WSGI server like **Gunicorn** (or an ASGI server for async), which runs multiple worker processes for true concurrency and is built to stay up under load, typically behind **Nginx** for TLS and static files. Using `runserver` in production would be slow, fragile, and insecure.

**Q: How do you decide how many Gunicorn workers to run?**
The common starting formula is **`(2 × number of CPU cores) + 1`** workers. The reasoning: Python's GIL means a single process can't use multiple cores for CPU-bound work, so you run multiple worker processes to actually parallelise across cores, and the "+1" plus the 2× factor accounts for workers being idle while waiting on I/O (database, network), so extra workers keep the CPUs busy. From there you tune based on real metrics — memory per worker (each worker is a full process consuming RAM), request latency, and whether the workload is I/O-bound or CPU-bound. For highly I/O-bound or async workloads you'd switch to async worker classes (gevent, or Uvicorn workers for ASGI) rather than simply adding sync workers.

### Common Mistakes

- Exposing Gunicorn directly to the internet with no reverse proxy.
- Setting far too many workers (memory exhaustion) or too few (poor concurrency).
- Expecting Gunicorn to serve static files well (use Nginx/WhiteNoise/CDN).
- Forgetting to configure timeouts, causing hung workers.

### Related Concepts

WSGI/ASGI, worker processes, GIL, reverse proxy, TLS termination, WhiteNoise, Uvicorn.

---

## 6.3 Deploying on AWS: EC2, Elastic Beanstalk, RDS, S3/CloudFront

### What is it?

Deploying a Django app on **Amazon Web Services**. The syllabus covers two deployment models and supporting services:
- **EC2 (Elastic Compute Cloud)** — a raw virtual server you fully control. You SSH in, install Python/Nginx/Gunicorn/PostgreSQL, and configure everything yourself. Maximum control, maximum responsibility.
- **Elastic Beanstalk** — a **Platform-as-a-Service (PaaS)** layer on top of EC2 that automates provisioning, deployment, load balancing, scaling, and monitoring. You push code; it handles the servers.
- **RDS (Relational Database Service)** — managed PostgreSQL (backups, patching, failover handled by AWS).
- **S3** — object storage for static/media files.
- **CloudFront** — a CDN that caches and serves your static/media globally with low latency.

### Why is it needed?

Real applications need to run somewhere reliable, scalable, and internet-reachable. AWS provides that infrastructure on demand. The **EC2 vs Beanstalk** choice reflects a fundamental trade-off: EC2 gives full control (any config, any software) but you manage everything (security patches, scaling, uptime); Beanstalk trades some control for automation (it wires up EC2, load balancers, auto-scaling, health checks for you). **RDS** exists because running your own production database — with backups, replication, failover, patching — is hard and risky; RDS manages it. **S3 + CloudFront** exist because app servers shouldn't store or serve user files (ephemeral disks, poor scaling); object storage + CDN do it durably and fast.

### How does it work?

**EC2 (manual) deployment steps:**
```
1. Launch an EC2 instance (choose OS, size, security group).
2. SSH in:  ssh -i key.pem ubuntu@<public-ip>
3. Install dependencies: python, pip, virtualenv, postgresql client, nginx.
4. Clone code, create venv, pip install -r requirements.txt.
5. Configure env vars, run migrate + collectstatic.
6. Run Gunicorn (as a systemd service) bound to a local socket.
7. Configure Nginx to proxy to Gunicorn and serve static files.
8. Point DNS at the instance; add HTTPS (Let's Encrypt/ACM).
```

**Elastic Beanstalk deployment:**
```
1. eb init (choose platform: Python).
2. eb create (provisions EC2, load balancer, auto-scaling group).
3. eb deploy (uploads code; Beanstalk runs it, handles scaling/health).
```

**Data/assets:** database on **RDS PostgreSQL**; static & media on **S3**, fronted by **CloudFront**.

### Internal working

- **EC2** is a virtual machine on AWS hardware; a **security group** is its firewall (open 80/443, restrict 22). You are responsible for the entire OS and stack.
- **Beanstalk** reads your code + config, provisions an environment (EC2 instances behind an Elastic Load Balancer in an auto-scaling group), deploys via its agent, runs health checks, and can add/remove instances based on load — abstracting the ops work while still using EC2 underneath.
- **RDS** runs the database on managed instances with automated backups, point-in-time recovery, and optional Multi-AZ failover.
- **S3** stores files as objects in buckets; **CloudFront** caches them at edge locations worldwide, so users download assets from a nearby edge rather than your origin.

### Advantages / Limitations

- **EC2:** full control, any stack; but you own patching, scaling, uptime, security — high operational burden.
- **Beanstalk:** fast, automated, handles scaling/monitoring; but less control and some "magic," and it can be harder to debug when the abstraction leaks.
- **RDS/S3/CloudFront:** managed, durable, scalable; but cost and vendor lock-in.

### Real-world applications

Startups often start on Beanstalk (or similar PaaS like Render/Railway) for speed, then move to EC2/ECS/EKS as they need more control. RDS + S3 + CloudFront is the near-universal data/asset stack for Django on AWS.

### Interview Questions & Model Answers

**Q: EC2 vs Elastic Beanstalk — when would you choose each?**
Both ultimately run your app on EC2 virtual machines, but at different levels of abstraction. **EC2 raw** gives you a bare server that you configure entirely — OS, Python, Nginx, Gunicorn, database client, security, scaling. You choose it when you need full control over the environment, have unusual requirements, or want to minimise cost by tuning everything yourself; the cost is operational responsibility (patching, monitoring, scaling, uptime are all on you). **Elastic Beanstalk** is a PaaS layer that automates provisioning, deployment, load balancing, auto-scaling, and health monitoring — you push code and it manages the infrastructure. You choose it when you want to ship quickly without becoming a full-time ops engineer, and you're comfortable trading some control and paying for the convenience. A common path: start on Beanstalk for speed, move toward managed EC2/containers as scale and customisation needs grow.

**Q: Walk me through deploying a Django app to an EC2 instance.**
First I launch an EC2 instance, choosing an OS (say Ubuntu), an instance size, and a security group that opens ports 80/443 to the world and restricts SSH (22) to my IP. I SSH in using the key pair. Then I install system dependencies — Python, pip, a virtualenv tool, the PostgreSQL client, and Nginx. I pull the code, create a virtual environment, and `pip install -r requirements.txt`. I set environment variables (SECRET_KEY, DATABASE_URL pointing at RDS, ALLOWED_HOSTS), then run `migrate` to set up the schema and `collectstatic` to gather static files. I run the app under **Gunicorn**, managed as a **systemd service** so it starts on boot and restarts on failure, bound to a local socket. I configure **Nginx** as a reverse proxy that forwards dynamic requests to Gunicorn and serves static files directly. Finally I point DNS at the instance and enable HTTPS with a certificate (Let's Encrypt or AWS ACM). The database lives on RDS and media on S3 rather than on the instance's ephemeral disk.

**Q: Why store static and media files on S3/CloudFront instead of on the EC2 instance?**
Because app-server disks are the wrong place for them at scale. EC2 instances are often **ephemeral** and horizontally scaled — files written to one instance don't exist on others and disappear when an instance is replaced or an auto-scaling event occurs, so user uploads would be lost or inconsistent. **S3** solves this: it's durable, effectively infinite object storage shared by all instances, so any server can reference the same files and they survive redeploys. **CloudFront** then sits in front as a CDN, caching those assets at edge locations worldwide so users download them from a nearby server with low latency, which also offloads traffic from your origin. The pattern keeps the app servers stateless (easy to scale and replace) and delivers assets fast and reliably. Django integrates this via `django-storages` with an S3 backend.

**Q: Why use RDS instead of running PostgreSQL on the same EC2 instance?**
Running your production database yourself is operationally risky and time-consuming: you'd be responsible for backups, patching, replication, failover, and monitoring, and a mistake can mean data loss. **RDS** is AWS's managed database service that handles all of that — automated backups, point-in-time recovery, security patching, and optional Multi-AZ failover to a standby in another availability zone. Separating the database onto RDS also decouples it from the app server, so you can scale, replace, or auto-scale app instances without touching the database, and the database isn't lost if an app instance dies. The trade-offs are cost and some loss of low-level control, but for production the reliability and reduced operational burden almost always justify it.

### Common Mistakes

- Storing media on the EC2 instance's local disk (lost on scaling/redeploy).
- Leaving SSH (port 22) open to the world in the security group.
- Running the DB on the app server for production.
- Not using systemd (Gunicorn dies and doesn't restart).
- Hardcoding AWS credentials instead of using IAM roles.

### Related Concepts

PaaS vs IaaS, security groups, IAM roles, systemd, load balancing, auto-scaling, `django-storages`, RDS Multi-AZ, CDN.

---

## 6.4 Database Deployment: PostgreSQL & Migrations in Production

### What is it?

Moving from the development database (usually SQLite) to a production-grade database (**PostgreSQL**), configuring it via environment variables, and running **migrations** safely against production data.

### Why is it needed?

**SQLite** is a file-based database perfect for development — zero setup — but it's **not suitable for production** web apps: it has limited concurrency (locking issues under simultaneous writes), no network access (can't be shared across multiple app servers), and fewer advanced features. **PostgreSQL** is a robust, concurrent, network-accessible relational database with rich features (JSONB, full-text search, advanced indexing, transactions) — the standard choice for production Django. Migrations must run in production to evolve the live schema, but doing so carelessly on large tables can cause **downtime**, so it needs a strategy.

### How does it work?

```python
# settings.py — read DB config from the environment
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ["DB_NAME"],
        "USER": os.environ["DB_USER"],
        "PASSWORD": os.environ["DB_PASSWORD"],
        "HOST": os.environ["DB_HOST"],   # e.g. RDS endpoint
        "PORT": "5432",
    }
}
```
Deploy flow: provision RDS PostgreSQL → set env vars → run `python manage.py migrate` (often automated in the deploy pipeline) → app connects.

### Internal working

- Django's PostgreSQL backend translates ORM queries to PostgreSQL SQL and manages a connection pool (with `CONN_MAX_AGE` for persistent connections).
- **Migrations in production** apply DDL (ALTER TABLE) to the live database. Some operations lock tables; the safe approach is **backwards-compatible, staged migrations** (add nullable column → deploy code that populates it → backfill → add constraint), so the app keeps working during the change.
- Running `migrate` is recorded in `django_migrations`, so it's idempotent — re-running applies only new migrations.

### Advantages / Limitations

- **Advantage:** concurrency, reliability, advanced features, network access, managed via RDS.
- **Limitation:** requires setup and connection management; risky migrations on big tables need care; dev/prod parity issues if you develop on SQLite but deploy on Postgres (subtle behaviour differences).

### Interview Questions & Model Answers

**Q: Why not use SQLite in production?**
SQLite is a lightweight, file-based database that's excellent for development and testing because it needs zero configuration, but it's a poor fit for production web applications for several reasons. It handles **concurrent writes** poorly — it locks the whole database file on writes, so under simultaneous traffic requests serialise and can time out. It's a **local file**, so it can't be shared across multiple app servers or containers, which breaks horizontal scaling. And it lacks many production features that PostgreSQL offers, like advanced indexing, JSONB, robust full-text search, and sophisticated concurrency control. PostgreSQL is designed for concurrent, networked, multi-client production workloads, which is why it's the standard production database for Django. A related best practice is to develop on the same database you deploy on (Postgres) to avoid subtle dev/prod behaviour differences.

**Q: How do you run migrations safely against a production database?**
The risk is that schema changes on large tables can lock them and cause downtime, and that new code and old code may briefly run at the same time during a deploy. The safe approach is **backwards-compatible, staged migrations**. For example, to add a required column: first add it as **nullable** (a fast, non-locking operation), deploy code that writes to it, **backfill** existing rows in batches (via a data migration or background job), and only then add the NOT NULL constraint in a later migration. For indexes on big tables, use non-blocking creation (PostgreSQL's `CREATE INDEX CONCURRENTLY`, exposed via `AddIndexConcurrently`). The overarching principle is **zero-downtime deploys**: each migration step is individually non-blocking and compatible with both the old and new versions of the code, and you always have a backup/rollback plan before applying schema changes to production data.

### Common Mistakes

- Developing on SQLite, deploying on Postgres, hitting behaviour differences.
- Running a table-locking migration on a huge table during peak traffic.
- Hardcoding DB credentials instead of env vars.
- Not backing up before risky migrations.

### Related Concepts

RDS, `CONN_MAX_AGE`, zero-downtime migrations, `dj-database-url`, dev/prod parity, connection pooling.

---

# 7. Advanced Django

## 7.1 Middleware & the Request Lifecycle

### What is it?

**Middleware** is a framework of hooks that wrap Django's request/response processing. Each middleware is a component that can inspect or modify **every** request on its way *in* to the view and **every** response on its way *out*. Middlewares form an ordered **stack**; a request passes down through them to the view, and the response passes back up.

Built-in examples: `SecurityMiddleware` (security headers/HTTPS), `SessionMiddleware` (loads the session), `AuthenticationMiddleware` (sets `request.user`), `CsrfViewMiddleware` (CSRF protection), `CommonMiddleware`.

```python
# a custom middleware
class TimingMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response       # runs once at startup

    def __call__(self, request):               # runs per request
        start = time.time()
        response = self.get_response(request)  # call the next layer / view
        response["X-Elapsed"] = time.time() - start
        return response
```

### Why is it needed?

Some concerns apply to **every** request — authentication, sessions, security headers, logging, rate limiting, CORS. Putting that logic in every view would be massive duplication. Middleware provides a single place to handle these **cross-cutting concerns**, running automatically for all requests. It's the aspect-oriented layer of Django: define once, apply everywhere.

### How does it work?

```
Request
  ▼
SecurityMiddleware        ── inbound ──►
SessionMiddleware         ── inbound ──►
AuthenticationMiddleware  ── inbound ──►
CsrfViewMiddleware        ── inbound ──►
        ▼
       VIEW  ──► produces response
        ▲
CsrfViewMiddleware        ◄── outbound ──
AuthenticationMiddleware  ◄── outbound ──
SessionMiddleware         ◄── outbound ── (saves session)
SecurityMiddleware        ◄── outbound ──
  ▲
Response
```

Each middleware calls `get_response(request)` to pass control to the next layer, then post-processes the returned response — like nested wrappers (an onion).

### Internal working

- At startup Django instantiates each middleware once, chaining them so each holds a reference to the next (`get_response`). This forms the nested call structure.
- **Order matters:** inbound processing runs top-to-bottom, outbound runs bottom-to-top. Dependencies dictate order — e.g., `AuthenticationMiddleware` must come *after* `SessionMiddleware` because it reads the session.
- A middleware can **short-circuit** by returning a response without calling `get_response` (e.g., a rate limiter returning 429), so the view never runs.
- Optional hooks: `process_view`, `process_exception`, `process_template_response` for finer control.

### Advantages / Limitations

- **Advantage:** DRY handling of cross-cutting concerns; composable; can short-circuit; global and automatic.
- **Limitation:** runs on *every* request, so heavy middleware slows everything; wrong ordering causes subtle bugs; over-using middleware for logic that belongs in views/services adds hidden complexity.

### Interview Questions & Model Answers

**Q: Explain the middleware request lifecycle.** *(Checklist item.)*
Middleware wraps Django's request/response handling as an ordered stack, like layers of an onion around the view. When a request arrives, it passes **inbound** through each middleware from top to bottom — `SecurityMiddleware`, `SessionMiddleware`, `AuthenticationMiddleware`, `CsrfViewMiddleware`, and so on — each able to inspect or modify it, or even short-circuit by returning a response early. It then reaches the **view**, which generates a response. The response travels back **outbound** through the same middlewares in **reverse** order (bottom to top), each able to modify it — for example `SessionMiddleware` saves any session changes on the way out. Structurally, each middleware calls `get_response(request)` to invoke the next layer and post-processes what comes back. The two key takeaways for interviews are that middleware handles cross-cutting concerns globally, and that **order matters** because inbound is top-down and outbound is bottom-up, and some middleware depends on earlier ones.

**Q: Give a real use case for custom middleware.**
Good candidates are concerns that must apply uniformly to every request. Examples: **request timing/logging** (record how long each request takes and log it), **rate limiting** (count requests per client and return 429 if they exceed a threshold — short-circuiting before the view), **adding security or CORS headers** to every response, **tenant resolution** in a multi-tenant app (read a subdomain and attach the tenant to the request), or **enforcing maintenance mode** (return a 503 for all requests when a flag is set). The common thread is that the logic is global and shouldn't be duplicated in every view. If a behaviour only applies to some views, a decorator or mixin is a better fit than middleware.

**Q: Why does middleware order matter? Give an example.**
Because middleware runs in sequence and later components often depend on the work of earlier ones, and because inbound order is top-to-bottom while outbound is bottom-to-top. The classic example is `SessionMiddleware` and `AuthenticationMiddleware`: authentication reads the logged-in user's ID from the session to set `request.user`, so the session must already be loaded — meaning `SessionMiddleware` must appear **before** `AuthenticationMiddleware`. Reverse them and `request.user` breaks. Similarly, security-related middleware is placed high so its protections apply to everything beneath it. Getting the order wrong doesn't always error loudly; it can cause subtle, hard-to-diagnose bugs, which is why you don't casually reorder the default `MIDDLEWARE` list.

### Common Mistakes

- Reordering middleware and breaking dependencies (auth/session).
- Doing expensive work in middleware that runs on every request.
- Forgetting to call `get_response` (request hangs / view never runs).
- Using middleware for view-specific logic (should be a decorator/mixin).

### Related Concepts

Request/response cycle, decorators, CORS, CSRF, rate limiting, `process_exception`.

---

## 7.2 Signals (pre_save, post_save, pre_delete, post_delete)

### What is it?

**Signals** are Django's implementation of the **observer pattern**: they let decoupled parts of your app get notified when certain events happen, without the code that triggers the event needing to know about the code that reacts. The ORM sends built-in signals around model operations:
- **`pre_save` / `post_save`** — before/after a model instance is saved.
- **`pre_delete` / `post_delete`** — before/after an instance is deleted.
- Also `m2m_changed`, `request_started`, etc.

```python
from django.db.models.signals import post_save
from django.dispatch import receiver

@receiver(post_save, sender=User)
def create_profile(sender, instance, created, **kwargs):
    if created:                       # only on new users
        Profile.objects.create(user=instance)
```

### Why is it needed?

Sometimes you want a side effect to happen whenever something occurs, but you don't want to couple the two pieces of code. Classic example: **every time a `User` is created, create a matching `Profile`.** You *could* put that in the user-creation view — but users get created in many places (admin, registration, shell, tests). A signal centralises the reaction so it fires no matter *where* the trigger happens. Signals decouple the "something happened" from the "react to it," which is useful for auditing, cache invalidation, notifications, and keeping related data in sync.

### How does it work / Internal working

- A **signal** is a dispatcher object. **Receivers** register with it (via `@receiver` or `.connect()`). When the sender emits the signal (`post_save.send(...)`, which the ORM does inside `Model.save()`), Django calls every connected receiver synchronously, passing context (`instance`, `created`, etc.).
- Signals run **in the same process and transaction** as the trigger (they're synchronous by default) — so a slow receiver slows the save, and an exception in a receiver can break the operation.
- Receivers are registered at app startup, typically in the app's `apps.py` `ready()` method, so they're wired up before requests are handled.

### Advantages

- Decoupling; centralised reactions to model events; great for cross-cutting side effects (profiles, audit logs, cache invalidation, notifications).

### Limitations

- **Hidden control flow** — the trigger and reaction are in different files, making behaviour hard to trace and debug ("why did a Profile get created?").
- **Synchronous** — heavy work in a signal blocks the request; offload to Celery.
- **Overuse** leads to spaghetti; for logic you control, an explicit method call or overriding `save()` is often clearer.

### Interview Questions & Model Answers

**Q: What are Django signals and when would you use them?**
Signals implement the observer pattern in Django: they let one part of the code broadcast that an event occurred (like a model being saved or deleted) and let other, decoupled parts react, without the two knowing about each other directly. The ORM emits built-in signals such as `pre_save`, `post_save`, `pre_delete`, and `post_delete`. You'd use them when you want a side effect to happen reliably whenever an event occurs regardless of where it's triggered — the canonical example is automatically creating a `Profile` whenever a `User` is created, since users can be created from registration, the admin, or the shell, and a signal ensures the profile is made in all cases. Other good uses are audit logging, cache invalidation, and sending notifications. The key benefit is decoupling.

**Q: What are the downsides of signals, and when would you avoid them?**
The main downside is **hidden, implicit control flow**: because the triggering code and the reacting receiver live in separate files, someone reading the save logic has no local indication that a signal fires and does extra work, which makes the system harder to understand and debug. They also run **synchronously** in the same transaction by default, so a slow or failing receiver can slow down or break the original operation. Because of this, many experienced developers avoid signals when the reaction is something they directly control and could just call explicitly — for instance, overriding the model's `save()` method or calling a service function is more transparent than a `post_save` signal. Signals shine for genuinely decoupled, cross-cutting concerns; for tightly related logic, explicit code is clearer. And heavy work in a signal should be pushed to a background task (Celery) so it doesn't block the request.

**Q: Difference between `pre_save` and `post_save`, and what is the `created` argument?**
`pre_save` fires **before** the instance is written to the database — useful for last-minute modifications like setting a slug or normalising a field, since you can still change the instance before it's saved. `post_save` fires **after** the database write succeeds — useful for side effects that need the saved record to exist, like creating a related object or invalidating a cache. `post_save` receivers get a boolean **`created`** argument that is `True` when the save inserted a brand-new row and `False` when it updated an existing one; you check it to run logic only for new objects (e.g., create a profile only for newly created users, not on every update).

### Common Mistakes

- Putting slow/blocking work in a signal (should use Celery).
- Forgetting to check `created` and re-running "on create" logic on every update.
- Not registering receivers in `apps.py ready()`, so they never fire.
- Overusing signals for logic that a `save()` override would express more clearly.

### Related Concepts

Observer pattern, `apps.py ready()`, overriding `save()`, Celery, transactions, `m2m_changed`.

---

## 7.3 Custom User Model

### What is it?

Replacing Django's default `User` model with your own, so you can customise how users are represented — most commonly to **log in with email instead of username**, or to add fields (phone, date of birth) directly to the user. Done by subclassing `AbstractUser` (keep Django's fields, add your own) or `AbstractBaseUser` (start from scratch) and setting `AUTH_USER_MODEL` in settings.

```python
class User(AbstractUser):
    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=20, blank=True)
    USERNAME_FIELD = "email"        # log in with email
    REQUIRED_FIELDS = []

# settings.py
AUTH_USER_MODEL = "accounts.User"
```

### Why is it needed?

The default `User` identifies people by **username**, but most modern apps use **email** as the login identifier. The default also can't easily hold app-specific fields. You *could* attach a separate `Profile` via a OneToOne, but for identity-level fields and email login it's cleaner to customise the user itself. Crucially, **Django strongly recommends setting a custom user model at the very start of a project**, even if you don't need changes yet, because switching later — once migrations and foreign keys reference the user table — is extremely painful.

### How does it work / Internal working

- `AbstractUser` provides all the default fields/behaviour (username, email, password, permissions) as an abstract base; you subclass it and add/override fields, then point `AUTH_USER_MODEL` at your model so the whole framework (auth, admin, permissions) uses it.
- `AbstractBaseUser` gives only the password/auth core, letting you define the identity fields entirely — more work, maximum control.
- `USERNAME_FIELD` tells Django which field is the unique login identifier; a **custom manager** (`create_user`/`create_superuser`) is needed when you change it.
- Everything that references the user should use `settings.AUTH_USER_MODEL` (or `get_user_model()`), never import `User` directly, so it stays correct.

### Advantages / Limitations

- **Advantage:** email login, custom identity fields, future-proofing; integrates with the whole auth system.
- **Limitation:** must be decided **early**; switching mid-project is very hard; `AbstractBaseUser` requires writing a manager and more boilerplate.

### Interview Questions & Model Answers

**Q: Why does Django recommend creating a custom user model at the start of every project?**
Because changing the user model **after** the database has been migrated is extremely difficult. The user table is referenced by foreign keys throughout the app — permissions, sessions, and every model that links to a user — and it's baked into the migration history. Swapping in a different user model at that point means rewriting migrations and untangling those references, which is error-prone and often requires wiping and rebuilding the database. By defining a custom user model up front (even a trivial subclass of `AbstractUser` that adds nothing yet), you reserve the ability to customise it later — add email login, extra fields — without a painful migration. It costs almost nothing to do at the start and saves enormous pain later, which is why it's considered a best practice on day one.

**Q: `AbstractUser` vs `AbstractBaseUser` — what's the difference?**
Both are bases for custom user models but at different starting points. **`AbstractUser`** is the full default user *without* being concrete — it already includes username, email, first/last name, password, and the permissions mixin. You subclass it when you want Django's standard user plus a few tweaks or extra fields; it's the common, low-effort choice. **`AbstractBaseUser`** provides only the core authentication machinery — password storage and the basics — and nothing else, so you define all the identity fields yourself and must write a custom manager with `create_user`/`create_superuser`. You choose it when you need a fundamentally different user model (e.g., no username at all, entirely email-based, unusual fields), accepting more boilerplate for full control. Rule of thumb: `AbstractUser` for "default user plus extras," `AbstractBaseUser` for "build the user from scratch."

**Q: How do you make users log in with email instead of username?**
The clean way is a custom user model. Subclass `AbstractUser`, make the `email` field unique, set `USERNAME_FIELD = "email"` (which tells Django the email is the login identifier) and adjust `REQUIRED_FIELDS`, and — because you've changed the identifier — provide a custom manager whose `create_user`/`create_superuser` use email. Then set `AUTH_USER_MODEL` to your model in settings, ideally before the first migration. Throughout the code you reference the user via `get_user_model()` or `settings.AUTH_USER_MODEL` rather than importing the default `User`. This makes the entire auth system, admin, and permissions treat email as the identity. Doing it via a custom model from the start is far easier than retrofitting email login later.

### Common Mistakes

- Not creating a custom user model early, then needing to switch painfully.
- Importing `django.contrib.auth.models.User` directly instead of `get_user_model()`.
- Changing `USERNAME_FIELD` without providing a custom manager.
- Forgetting to set `AUTH_USER_MODEL`.

### Related Concepts

`AbstractUser`/`AbstractBaseUser`, `get_user_model()`, `AUTH_USER_MODEL`, custom manager, authentication, Profile pattern.

---

## 7.4 Django Admin (Customisation, Filters, Search, Inlines)

### What is it?

The **Django admin** is an auto-generated, production-ready web interface for managing your models' data — create/read/update/delete records through a UI, with no code beyond registering your models. It's highly customisable via `ModelAdmin` classes:
- `list_display` — which columns show in the list.
- `list_filter` — sidebar filters.
- `search_fields` — a search box.
- `inlines` — edit related objects on the same page (e.g., order items inside an order).
- `readonly_fields`, `ordering`, `list_editable`, custom actions.

```python
@admin.register(Article)
class ArticleAdmin(admin.ModelAdmin):
    list_display = ["title", "author", "published", "created"]
    list_filter = ["published", "created"]
    search_fields = ["title", "body"]
    inlines = [CommentInline]
```

### Why is it needed?

Almost every app needs an internal tool for staff to manage data — moderate content, edit records, inspect the database. Building such a CRUD interface from scratch for every model is huge, repetitive work. The admin **generates it automatically** from your models, then lets you customise it. It's often cited as one of Django's biggest selling points: on day one you have a usable back-office. It saves building an internal dashboard and gives non-technical staff safe data access.

### How does it work / Internal working

- The admin is a Django app (`django.contrib.admin`) that, for each registered model, introspects its fields and builds list/detail/edit pages using the **forms framework** (which uses your models via the ORM). `list_filter`/`search_fields` translate to ORM `.filter()`/`Q` queries; `inlines` use formsets to edit related objects together.
- Registration (`admin.site.register` or `@admin.register`) tells the admin which models to expose and which `ModelAdmin` config to apply.
- Access is gated by permissions and `is_staff`.

### Advantages / Limitations

- **Advantage:** instant, customisable CRUD back-office; permission-aware; huge time saver.
- **Limitation:** it's an **internal admin tool, not a customer-facing UI** — not meant to be exposed to end users; heavy customisation can fight the framework; performance needs care (list pages can trigger N+1; use `list_select_related`).

### Interview Questions & Model Answers

**Q: What is the Django admin and why is it valuable?**
The Django admin is an automatically generated web interface for managing your application's data. Once you register a model, the admin gives you fully functional create, read, update, and delete pages for it — a complete back-office CRUD tool with essentially no code. It's valuable because virtually every application needs an internal interface for staff to manage records, and building one by hand for every model is enormous, repetitive work. The admin provides it for free and is deeply customisable: you control which columns appear (`list_display`), add filters (`list_filter`) and search (`search_fields`), edit related records inline (`inlines`), and define custom bulk actions. It's permission-aware, so you can control who sees what. It's one of Django's standout features — on the first day of a project you already have a usable administrative dashboard.

**Q: Is the Django admin meant to be your customer-facing UI?**
No. The admin is an **internal, staff-facing tool** for managing data — moderation, data entry, inspection — and it's designed around that use case, gated behind `is_staff` and permissions. It is not intended as the public interface your end users interact with. Exposing it to customers would be inappropriate both in terms of UX (it's a generic data-management UI, not a tailored product experience) and security/scope (it grants broad data access). For customer-facing features you build your own views/templates or a frontend consuming your API. The admin's job is to save you from building an internal back-office, not to replace your product's UI.

**Q: How do you customise the admin, and what performance pitfall should you watch for?**
You customise it by defining a `ModelAdmin` subclass and registering it: `list_display` sets the columns in the list view, `list_filter` adds a filter sidebar, `search_fields` adds a search box, `inlines` let you edit related objects on the same page (like line items within an order), and you can add `readonly_fields`, custom actions, and ordering. The performance pitfall is the **N+1 query problem** on list pages: if `list_display` includes a related field (like `article.author`), the admin may issue a separate query per row to fetch it. You fix this with `list_select_related` (or overriding `get_queryset` with `select_related`/`prefetch_related`) so the related data is fetched efficiently in fewer queries. Large admin list pages without this optimisation can become very slow.

### Common Mistakes

- Treating the admin as the end-user UI.
- N+1 queries on list pages (not using `list_select_related`).
- Exposing sensitive models without restricting permissions.
- Over-customising to the point of fighting the framework.

### Related Concepts

Forms framework, permissions/`is_staff`, `list_select_related`, formsets/inlines, N+1, custom admin actions.

---

## 7.5 Caching (Local Memory, Redis, Cache Decorators)

### What is it?

**Caching** stores the result of an expensive operation (a database query, a rendered page, a computed value) so subsequent requests can reuse it instead of recomputing — trading a little staleness for a lot of speed. Django has a flexible cache framework with pluggable backends:
- **Local Memory Cache** — per-process in-memory (default; fine for dev/single process).
- **Redis / Memcached** — external, shared in-memory stores (production; shared across all app servers).
- **Cache levels:** per-view (`@cache_page`), per-fragment (`{% cache %}` template tag), and low-level (`cache.get`/`cache.set`) for arbitrary values.

```python
from django.views.decorators.cache import cache_page

@cache_page(60 * 15)          # cache this view's output for 15 minutes
def article_list(request): ...

# low-level
from django.core.cache import cache
data = cache.get("stats")
if data is None:
    data = expensive_query()
    cache.set("stats", data, timeout=300)
```

### Why is it needed?

Some operations are expensive — a complex aggregation query, rendering a heavy page, calling a slow external API. If every request recomputes them, the app is slow and the database is overloaded. If the result doesn't change often, caching lets you compute it once and serve the stored copy to many requests, dramatically cutting latency and database load. It's one of the highest-impact performance tools: the fastest query is the one you never run.

### How does it work / Internal working

- Django's cache framework is a key–value API (`get`/`set`/`delete`) backed by the configured store. **Redis/Memcached** hold data in RAM in a **separate service**, so all app server processes/instances share the same cache (unlike local-memory, which each process has its own copy of).
- **`@cache_page`** stores the full rendered response keyed by URL (and varying headers); subsequent hits return the stored response without running the view.
- **Cache invalidation** — deciding when cached data is stale — is the hard part. Strategies: time-based expiry (`timeout`), explicit deletion on write (often via signals), or versioned keys.

### Advantages / Limitations

- **Advantage:** major latency and DB-load reduction; flexible granularity (view/fragment/value); Redis is shared and fast.
- **Limitation:** **stale data** risk; **cache invalidation is famously hard**; adds infrastructure (Redis) and complexity; local-memory cache isn't shared across processes so it's unsuitable for multi-server production.

### Interview Questions & Model Answers

**Q: What is caching and when should you use it?**
Caching stores the result of an expensive computation so it can be reused instead of recomputed. In Django that might be a whole rendered page, a template fragment, or the result of a costly database query or external API call. You should use it when an operation is expensive *and* its result doesn't change on every request — for example, a homepage showing "top articles this week" recalculated from a heavy aggregation. Instead of running that query for every visitor, you compute it once, store it (ideally in Redis), and serve the cached copy for, say, five minutes. The benefit is dramatically lower latency and reduced database load. The trade-off you accept is potential staleness — the cached data may be slightly out of date — so you cache things where that's acceptable and tune the expiry accordingly.

**Q: Why use Redis instead of local-memory caching in production?**
Local-memory cache lives inside a single Python process, so each Gunicorn worker (and each server) has its **own separate copy**. In production you run many workers across possibly many machines, which means the cache would be fragmented and inconsistent — one worker's cached value isn't visible to others, and invalidating an entry in one process doesn't clear it in the others. **Redis** (or Memcached) is a **separate, shared** in-memory service that all workers and servers talk to, so there's a single consistent cache across the whole deployment. It also persists across process restarts, supports data structures and atomic operations, and can be used for sessions and Celery too. So local-memory is fine for development or a single process, but production needs a shared external cache like Redis for correctness and scale.

**Q: Why is cache invalidation considered hard, and how do you handle it?**
The difficulty is knowing exactly *when* cached data has become stale and clearing it at the right moment — cache too long and users see outdated data, clear too eagerly and you lose the benefit. The classic saying is that cache invalidation is one of the two hard problems in computer science. Common strategies: **time-based expiry** (set a `timeout` and accept data can be up to that old — simplest and often good enough), **explicit invalidation on write** (when the underlying data changes, delete the relevant cache key — often wired up via `post_save`/`post_delete` signals), and **versioned or key-scoped caching** (include a version or identifier in the cache key so updating the data effectively points to a fresh key). The right approach depends on how fresh the data must be; many systems combine short expiries with explicit invalidation for critical updates.

### Common Mistakes

- Using local-memory cache in multi-process/multi-server production.
- Caching data that must always be fresh (leading to stale bugs).
- No invalidation strategy → users see outdated data indefinitely.
- Caching per-user data under a shared key (leaking one user's data to another).

### Related Concepts

Redis/Memcached, `@cache_page`, `{% cache %}`, cache invalidation, signals, CDN caching, database query optimisation.

---

## 7.6 Background Tasks (Celery, Redis, Scheduled Jobs)

### What is it?

**Background/asynchronous task processing** moves slow work *out* of the request/response cycle so the user gets a fast response while the heavy work runs separately. **Celery** is the standard distributed task queue for this; it uses a **message broker** (often **Redis** or RabbitMQ) to pass tasks to **worker** processes that execute them. It also supports **scheduled jobs** (periodic tasks) via Celery Beat.

```python
# tasks.py
from celery import shared_task

@shared_task
def send_welcome_email(user_id):
    user = User.objects.get(id=user_id)
    send_email(user.email, "Welcome!")     # slow I/O — done in background

# in a view
send_welcome_email.delay(user.id)          # returns immediately; task queued
```

### Why is it needed?

HTTP requests should return **quickly** — ideally in milliseconds. But some work is slow: sending emails, generating PDFs/reports, processing images/video, calling slow third-party APIs, running ML inference. If you do that work *inside* the request, the user waits (bad UX), the worker process is tied up (poor throughput), and a timeout may kill the request. Background tasks solve this: the view **queues** the task and returns immediately; a separate worker does the slow work later. This keeps the app responsive and lets heavy work scale independently. **Scheduled jobs** handle recurring work (nightly reports, cleanup, reminders) without a human triggering them.

### How does it work?

```
Web request ──► view calls task.delay()  ──► pushes message to BROKER (Redis)
                     │  (returns response immediately)
                     ▼
              user gets fast response

Separately:
  CELERY WORKER  ◄── pulls task from broker ──►  executes send_welcome_email()
                                                  (stores result in a backend)
```

- `.delay()` / `.apply_async()` serialise the task + args and put a message on the broker.
- One or more **workers** (separate processes, often separate machines) continuously pull messages and run the tasks.
- **Celery Beat** is a scheduler that periodically enqueues tasks on a cron-like schedule.

### Internal working

- The **broker** (Redis/RabbitMQ) is the queue holding pending task messages. Workers consume from it; you can scale by adding workers.
- A **result backend** (Redis/DB) optionally stores task results/status so you can check on them.
- Tasks should be **idempotent** where possible (they may be retried) and should not rely on request/session state (they run outside the web process).

### Advantages / Limitations

- **Advantage:** fast responses, better throughput, scalable heavy work, retries, scheduling, decoupling.
- **Limitation:** operational complexity (broker + workers to run/monitor); harder debugging (async, distributed); eventual consistency (the work isn't done when the response returns); needs care around retries/idempotency and failures.

### Real-world applications

Sending transactional emails/SMS, generating invoices/reports, thumbnailing images, transcoding video, syncing with external systems, nightly data cleanups, sending scheduled reminders/notifications.

### Interview Questions & Model Answers

**Q: Why use Celery / background tasks instead of doing the work in the view?**
Because slow work in the request cycle hurts both user experience and server capacity. HTTP requests should return quickly, but tasks like sending emails, generating reports, processing media, or calling slow external APIs can take seconds or minutes. If you do them in the view, the user stares at a spinner, the worker process handling that request is blocked and can't serve others (reducing throughput), and the request may hit a timeout and fail. Offloading to Celery fixes this: the view enqueues the task with `.delay()` and returns an immediate response, while a separate **worker** process performs the slow work in the background. This keeps the app responsive, lets you scale the heavy processing independently by adding workers, and provides retries for reliability. Scheduled/periodic work (via Celery Beat) is another driver — recurring jobs that no request triggers.

**Q: Explain Celery's architecture — broker, worker, result backend.**
Celery is a distributed task queue with three main pieces. The **broker** — typically Redis or RabbitMQ — is the message queue: when your app calls `task.delay()`, it serialises the task name and arguments into a message and pushes it onto the broker. The **worker** is a separate process (often on a separate machine) that continuously pulls messages off the broker and executes the corresponding task functions; you scale throughput by running more workers. The optional **result backend** — Redis or a database — stores each task's status and return value so you can query whether a task finished and what it produced. Separately, **Celery Beat** is a scheduler that enqueues tasks on a periodic schedule for recurring jobs. This decoupled design means the web app just drops messages on the broker and moves on, while workers handle execution independently and can fail, retry, and scale without affecting the web tier.

**Q: What should you keep in mind when writing a Celery task?**
Several things. First, **idempotency and retries** — tasks can fail and be retried, so a task should ideally produce the same correct result if run more than once (e.g., check before re-sending an email or re-charging a card, or use an idempotency key). Second, **don't depend on request/session state** — the task runs in a separate process outside the web request, so pass everything it needs as arguments (and prefer passing an ID like `user_id` rather than a whole object, so it fetches fresh data). Third, **keep tasks focused and serialisable** — arguments must be serialisable to go over the broker. Fourth, **handle failures and set retry/timeout policies** so a stuck task doesn't hang forever. Finally, remember the work is **eventually** done — the user's response returns before the task completes, so design the UX around that (e.g., "your report is being generated").

### Common Mistakes

- Passing whole model objects instead of IDs (stale data, serialisation issues).
- Non-idempotent tasks that duplicate side effects on retry.
- Relying on request/session state inside a task.
- Forgetting to run/monitor the worker and broker (tasks queue but never execute).
- Doing background-appropriate work synchronously in the view.

### Related Concepts

Message brokers (Redis/RabbitMQ), Celery Beat, idempotency, retries, async views, task queues, eventual consistency.

---

## 7.7 Security Best Practices (CSRF, XSS, SQL Injection, Hashing, Cookies, HTTPS)

### What is it?

The core web security threats and Django's defences against them:
- **CSRF (Cross-Site Request Forgery)** — tricking a logged-in user's browser into submitting an unwanted request. Django defends with a **CSRF token** required on state-changing requests.
- **XSS (Cross-Site Scripting)** — injecting malicious scripts into pages. Django defends via **template auto-escaping**.
- **SQL Injection** — injecting SQL through input. Django defends via the **ORM's parameterised queries**.
- **Password hashing** — storing passwords as salted hashes (PBKDF2/Argon2), never plaintext.
- **Secure cookies** — `HttpOnly`, `Secure`, `SameSite` flags.
- **HTTPS** — encrypting traffic (`SECURE_SSL_REDIRECT`, HSTS).

### Why is it needed?

Web apps are exposed to the entire internet, including attackers. These vulnerabilities are among the most common and damaging (they dominate the OWASP Top 10). A single XSS or SQL injection hole can compromise every user or the entire database. Security best practices exist to close these holes. Django's philosophy is **secure by default** — most defences are on unless you disable them — but you must understand them so you don't accidentally turn them off (e.g., overusing `|safe`, building raw SQL, or exempting CSRF).

### How does it work / Internal working — defence by defence

- **CSRF:** Django issues a random token tied to the session; forms include it via `{% csrf_token %}`, and `CsrfViewMiddleware` rejects any unsafe (POST/PUT/DELETE) request lacking a valid token. An attacker's forged request can't include the secret token, so it's blocked. `SameSite` cookies add another layer.
- **XSS:** the template engine **auto-escapes** variable output, converting `<`, `>`, `&`, `"` into entities so injected `<script>` renders as harmless text. You only bypass it with `|safe`, which you reserve for content you've sanitised.
- **SQL injection:** the ORM sends SQL and parameters **separately** to the database (parameterised queries), so user input is always data, never executable SQL. String-formatting SQL yourself defeats this.
- **Password hashing:** `set_password()` runs a slow, salted one-way hash (PBKDF2 default); the plaintext is never stored, and constant-time comparison resists timing attacks.
- **Secure cookies:** `HttpOnly` (JS can't read → limits XSS theft), `Secure` (HTTPS only), `SameSite` (limits cross-site sending → CSRF defence).
- **HTTPS:** encrypts data in transit; `SECURE_SSL_REDIRECT` forces HTTPS and HSTS tells browsers to always use it.

### Advantages / Limitations

- **Advantage:** Django ships strong, on-by-default protections against the most common attacks.
- **Limitation:** defaults can be undermined by developer error (disabling CSRF, `|safe` on untrusted input, raw SQL, `DEBUG=True`); some threats (business-logic flaws, auth misconfig, dependency vulns) aren't auto-handled.

### Interview Questions & Model Answers

**Q: What is CSRF and how does Django protect against it?** *(Checklist item.)*
CSRF, Cross-Site Request Forgery, is an attack where a malicious site tricks a victim's browser into sending an authenticated request to your app *without the user's intent* — for example, a hidden form on a bad site that POSTs to your "transfer money" endpoint. Because the browser automatically includes the user's session cookie, the request looks legitimate. Django defends against this with a **CSRF token**: it generates a secret, random token tied to the user's session, and legitimate forms must include it (via the `{% csrf_token %}` template tag). The `CsrfViewMiddleware` then rejects any state-changing request (POST/PUT/PATCH/DELETE) that lacks a valid token. Since a cross-site attacker can't read or guess the per-session token, their forged request is missing it and gets blocked. `SameSite` cookie settings reinforce this by limiting when the session cookie is sent on cross-site requests. For APIs using token/JWT auth in headers (not cookies), CSRF is less of a concern because there's no ambient cookie to exploit.

**Q: How does Django prevent XSS and SQL injection?**
For **XSS** (injecting malicious scripts into pages that then run in other users' browsers), Django's template engine **auto-escapes** all variable output by default — it converts HTML-special characters like `<` and `>` into safe entities, so if a user submits `<script>...</script>`, it displays as inert text instead of executing. You only lose this protection if you explicitly mark content `|safe`, which you must reserve for trusted/sanitised HTML. For **SQL injection** (injecting SQL through user input to manipulate queries), the Django ORM uses **parameterised queries**: it sends the SQL statement and the user-provided values to the database *separately*, so input is always treated as data and can never be executed as SQL. As long as you use the ORM (or parameterised `raw()` queries) rather than building SQL strings with user input via f-strings/concatenation, injection is prevented. Both defences are on by default — the risk comes from developers turning them off.

**Q: How does Django store passwords securely, and what cookie/HTTPS settings matter in production?**
Django stores passwords as **salted, hashed** values using a slow one-way algorithm (PBKDF2 by default, with Argon2/bcrypt available), never as plaintext or reversible encryption — so even a database breach doesn't directly expose passwords, the per-user salt defeats rainbow tables, and the deliberate slowness makes brute-forcing costly. For cookies and transport in production: set **`SESSION_COOKIE_SECURE`** and **`CSRF_COOKIE_SECURE`** so cookies are only sent over HTTPS, **`SESSION_COOKIE_HTTPONLY`** so JavaScript can't read the session cookie (limiting XSS-based theft), and **`SESSION_COOKIE_SAMESITE`** to restrict cross-site cookie sending (CSRF defence). Enforce **HTTPS** with `SECURE_SSL_REDIRECT = True` and enable **HSTS** (`SECURE_HSTS_SECONDS`) so browsers always use TLS. Combined with `DEBUG=False` and a protected `SECRET_KEY`, these settings form the baseline production security posture; Django's `manage.py check --deploy` audits many of them.

### Common Mistakes

- Disabling CSRF protection or `@csrf_exempt` without understanding the risk.
- Overusing `|safe` on user-generated content → XSS.
- Building SQL with f-strings/`.format()` → SQL injection.
- Running `DEBUG=True` in production (leaks data).
- Not enabling secure cookie/HTTPS settings.
- Storing passwords with weak/custom hashing.

### Related Concepts

OWASP Top 10, `{% csrf_token %}`, auto-escaping, parameterised queries, PBKDF2/Argon2, `SameSite`, HSTS, `check --deploy`.

---

## 7.8 Performance Optimization (N+1, select_related, prefetch_related, Indexing, Pagination, Caching)

### What is it?

Techniques to make a Django app fast, focused mostly on the **database**, which is usually the bottleneck:
- **The N+1 query problem** — accidentally running one query per item in a loop.
- **`select_related()`** — fetch related objects via a **SQL JOIN** in one query (for ForeignKey/OneToOne).
- **`prefetch_related()`** — fetch related objects in a **separate query** and join in Python (for ManyToMany/reverse FK).
- **Database indexing** — speed up lookups/filters/sorts on specific columns.
- **Pagination** — bound result-set size.
- **Caching** — avoid recomputation (covered in 7.5).
- Tools: `.only()`/`.defer()`, `.values()`, `django-debug-toolbar`, `QuerySet.explain()`.

### Why is it needed?

The ORM makes it easy to write code that's correct but **slow** — especially the N+1 problem, where iterating over objects and accessing a related field silently fires a query per object, turning one page load into hundreds of queries. As data grows, unindexed filters and unbounded queries make pages crawl or time out. Performance optimisation exists because a functionally correct app that's slow is still a failed app — users leave, servers fall over. Most Django performance work is about **reducing the number and cost of database queries.**

### How does it work?

**The N+1 problem and its fix:**
```python
# BAD: N+1 queries (1 for articles + 1 per article for its author)
for article in Article.objects.all():
    print(article.author.name)          # each access = a new query!

# GOOD: 1 query with a JOIN
for article in Article.objects.select_related("author"):
    print(article.author.name)          # author already fetched

# For many-to-many / reverse FK: prefetch_related
for article in Article.objects.prefetch_related("tags"):
    print([t.name for t in article.tags.all()])   # 2 queries total, not N+1
```

**Indexing:**
```python
class Article(models.Model):
    slug = models.SlugField(db_index=True)         # index a frequently-filtered column
    class Meta:
        indexes = [models.Index(fields=["author", "created"])]  # composite index
```

### Internal working

- **N+1** arises from **lazy loading**: accessing `article.author` triggers a query the first time. In a loop of N articles that's N extra queries plus the initial one = N+1.
- **`select_related`** adds a SQL `JOIN` so related rows come back in the *same* query — best for to-one relations. **`prefetch_related`** runs a second query for all related rows and stitches them in Python — needed for to-many relations where a JOIN would multiply rows.
- **Indexes** are separate data structures (usually B-trees) the database maintains so it can find matching rows without scanning the whole table — turning O(n) scans into O(log n) lookups. They speed reads but slow writes slightly and use storage, so you index the columns you actually filter/sort/join on.
- **`.explain()`** / EXPLAIN shows the database's query plan (index scan vs sequential scan), the tool for diagnosing slow queries.

### Advantages / Limitations

- **Advantage:** dramatic latency reductions; fewer queries; lower DB load; scales to large data.
- **Limitation:** requires understanding the generated SQL; over-indexing slows writes and wastes space; `select_related` on many relations can create huge JOINs; premature optimisation wastes effort (measure first).

### Interview Questions & Model Answers

**Q: What is the N+1 query problem and how do you fix it?** *(Checklist item.)*
The N+1 problem is a common performance bug where fetching a list of N objects and then accessing a related field on each one triggers **one additional query per object** — so a single conceptual operation becomes 1 (the list) + N (one per item) = N+1 queries. It happens because the ORM lazily loads related objects: the related data isn't fetched until you access it, so `for article in articles: print(article.author.name)` fires a fresh query for every article's author. With hundreds of rows, that's hundreds of round-trips and a very slow page. The fix is to tell the ORM to fetch the related data up front: use **`select_related`** for foreign-key/one-to-one relations (it adds a SQL JOIN so everything comes back in one query) and **`prefetch_related`** for many-to-many or reverse relations (it runs one extra query and joins in Python). Applied correctly, the N+1 collapses to a constant, small number of queries. Tools like django-debug-toolbar help you spot N+1 patterns.

**Q: `select_related` vs `prefetch_related` — when do you use each?** *(Checklist item.)*
Both eliminate the N+1 problem by fetching related objects efficiently, but they work differently and suit different relationship types. **`select_related`** performs a SQL **JOIN** and pulls the related data in the **same query** — it's for **to-one** relationships (`ForeignKey`, `OneToOneField`), like an article's author. Because it's a join, it's efficient when each row has exactly one related row. **`prefetch_related`** runs a **separate query** for the related objects and matches them up in **Python** — it's for **to-many** relationships (`ManyToManyField`, reverse foreign keys), like an article's tags or an author's articles. You use a separate query there because JOINing a to-many relation would multiply rows (one row per combination), which is wasteful; fetching the related set separately and stitching in Python is cleaner. Rule of thumb: `select_related` for "one related object" (follow the FK with a join), `prefetch_related` for "many related objects" (fetch the set in a second query).

**Q: What is a database index and what's the trade-off?**
A database index is an auxiliary data structure (typically a B-tree) that lets the database find rows matching a condition without scanning the entire table — analogous to the index at the back of a book. If you frequently filter, sort, or join on a column (say `slug` or `email`), indexing it turns a slow full-table scan into a fast lookup, which is essential as tables grow to millions of rows. The trade-off is that indexes aren't free: they consume storage, and they **slow down writes** because every insert/update/delete must also update the indexes. So you index the columns you actually query on — foreign keys (Django indexes these by default), and fields used in `filter`/`order_by`/joins — rather than indexing everything. You verify an index is being used with `EXPLAIN` / `QuerySet.explain()`, which shows whether the database chose an index scan or a sequential scan.

**Q: How would you approach a page that's loading slowly?**
I'd **measure first**, not guess. I'd use django-debug-toolbar (or query logging/APM) to see how many queries the page runs and how long each takes. The most common culprit is the **N+1 problem** — lots of near-identical queries — which I'd fix with `select_related`/`prefetch_related`. Next I'd look for **missing indexes** on columns used in filters/ordering, checking with `EXPLAIN` for sequential scans, and add indexes where appropriate. I'd ensure list endpoints are **paginated** so they're not loading huge result sets, and use `.only()`/`.values()` to avoid pulling unneeded columns. For expensive, rarely-changing computations I'd add **caching** (Redis). Finally, for genuinely heavy work that doesn't need to block the response, I'd move it to a **background task** (Celery). The theme is: profile to find the real bottleneck, then apply the targeted fix — usually reducing the number and cost of database queries.

### Common Mistakes

- Accessing related objects in loops without `select_related`/`prefetch_related` (N+1).
- Using `select_related` for many-to-many (wrong tool; use `prefetch_related`).
- Not paginating large lists.
- Over-indexing (slows writes) or not indexing filtered columns at all.
- Optimising without measuring (premature optimisation).
- Fetching full objects when `.values()`/`.only()` would do.

### Related Concepts

Lazy evaluation, QuerySets, B-tree indexes, `EXPLAIN`, django-debug-toolbar, pagination, caching, Celery, `.only()`/`.defer()`/`.values()`.

---

## Appendix: The Interview Preparation Checklist — Quick Answers

The syllabus lists these "be able to explain" items. Here's a one-paragraph anchor for each (details are in the sections above):

1. **Django MVT architecture** — Model (data/ORM), View (request logic = MVC's controller), Template (presentation = MVC's view); the framework wires routing. See 1.2.
2. **Django ORM vs raw SQL** — ORM: productive, portable, injection-safe, but can hide query cost; raw SQL: full control for complex/performance-critical queries. See 2.1.
3. **FBV vs CBV** — functions (explicit, simple, custom) vs classes (DRY, reusable via mixins, standard CRUD). See 3.1–3.2.
4. **`select_related` vs `prefetch_related`** — JOIN in one query for to-one vs separate query + Python join for to-many. See 7.8.
5. **Authentication vs Authorization** — who you are vs what you're allowed to do. See 4.2.
6. **Session vs JWT auth** — stateful cookie/session vs stateless signed token; web same-site vs API/mobile/SPA. See 5.6.
7. **Django Forms vs DRF Serializers** — HTML input validation/rendering vs JSON serialisation/validation. See 4.4, 5.3.
8. **APIView vs Generic Views vs ViewSets** — control vs conciseness; single-endpoint mixins vs full-resource classes + routers. See 5.4–5.5.
9. **PUT vs PATCH** — full replace (idempotent) vs partial update. See 5.1.
10. **Middleware request lifecycle** — ordered onion; inbound top-down, outbound bottom-up; order matters. See 7.1.
11. **CSRF protection** — per-session token required on state-changing requests; blocks forged cross-site requests. See 7.7.
12. **N+1 query problem** — one query per item from lazy loading; fixed with select/prefetch_related. See 7.8.
13. **Gunicorn vs Nginx** — WSGI app server (runs Python, concurrency) vs reverse proxy (TLS, static, protection). See 6.2.
14. **Static vs Media files** — your app's shipped assets vs user-uploaded content. See 4.6, 6.1.
15. **Deploying to AWS** — EC2 (manual: Nginx+Gunicorn+systemd) or Beanstalk (PaaS), with RDS + S3/CloudFront. See 6.3.

---

*End of theory.md — proceed to `practical.md` for hands-on coding, notebook-style, and API-building questions covering the same syllabus.*







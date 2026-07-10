# Web Development with Django — Practical & Coding Assessment Guide

> **How to use this document.** This is the hands-on companion to `theory.md`. It predicts the *practical* questions you'll face — the ones where you must **write code, design models, build endpoints, and reason about queries** — for the Django / DRF / AWS syllabus. Each question uses a consistent format: **Difficulty → Estimated Time → Concepts Tested → Problem → Example I/O → Approach → Implementation (with line-level explanations, time/space complexity) → Alternative Solution → Interview Variations → Follow-up Questions.**
>
> **The running project.** Many questions build one coherent app — a **"Blog / Article API"** — so you practise the way real assessments flow: model it, expose CRUD, secure it, optimise it, deploy it. Set it up once and reuse it.
>
> **Setup you'll assume throughout:**
> ```bash
> python -m venv venv && source venv/bin/activate
> pip install django djangorestframework djangorestframework-simplejwt django-filter drf-spectacular psycopg2-binary
> django-admin startproject config .
> python manage.py startapp blog
> # add 'rest_framework', 'blog', etc. to INSTALLED_APPS
> ```

---

# Table of Contents

- **Part A — Django Core (models, ORM, views, templates, URLs)**
- **Part B — Authentication, Forms & CRUD**
- **Part C — Building a REST API with DRF**
- **Part D — ORM Query Workbench (notebook-style shell exercises)**
- **Part E — Advanced (signals, caching, performance, middleware)**
- **Part F — Deployment tasks (AWS, Gunicorn, Nginx, PostgreSQL)**
- **Part G — Testing**
- **Part H — Rapid-fire coding questions (Easy / Medium / Hard) with interviewer intent**

---

# Part A — Django Core

## Practical Question 1 — Design the Data Model for a Blog

**Difficulty:** Easy–Medium
**Estimated Time:** 15–20 min
**Concepts Tested:** Models, field types, relationships (FK/M2M/O2O), `Meta`, `__str__`, `on_delete`

### Problem Statement

Design Django models for a blogging platform with the following requirements:
- An **Author** has a name and bio, and extends the built-in user (one profile per user).
- An **Article** has a title, slug, body, a publication status, timestamps, exactly one author, and many tags.
- A **Tag** has a unique name.
- A **Comment** belongs to one article and one user, with body text and a timestamp.
- Articles must default to newest-first ordering, and the slug must be unique and indexed.

### Example (what "good" looks like)

Given these models, `Article.objects.create(...)` should work, `article.tags.add(tag)` should attach tags, and `author.user.username` should reach the underlying user.

### Approach — step by step

1. **Author** → `OneToOneField(User)` (one profile per user) plus extra fields.
2. **Tag** → a simple model with a `unique=True` name.
3. **Article** → `ForeignKey(Author)` for one-to-many, `ManyToManyField(Tag)` for many-to-many, a `SlugField(unique=True, db_index=True)`, `auto_now_add`/`auto_now` timestamps, and `Meta.ordering`.
4. **Comment** → two `ForeignKey`s (article, user), both `on_delete=CASCADE`.
5. Add `__str__` to every model and think carefully about each `on_delete`.

### Python Implementation

```python
# blog/models.py
from django.conf import settings
from django.db import models


class Author(models.Model):
    # OneToOne = "extend" the user with profile data; one Author per User.
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    bio = models.TextField(blank=True)                 # blank (form-optional), not null

    def __str__(self):
        return self.user.get_username()                # readable in admin/shell


class Tag(models.Model):
    name = models.CharField(max_length=50, unique=True)  # no duplicate tags

    def __str__(self):
        return self.name


class Article(models.Model):
    class Status(models.TextChoices):                  # enumerated choices
        DRAFT = "draft", "Draft"
        PUBLISHED = "published", "Published"

    title = models.CharField(max_length=200)
    slug = models.SlugField(max_length=220, unique=True, db_index=True)  # indexed lookup
    body = models.TextField()
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.DRAFT)
    author = models.ForeignKey(                        # one-to-many: author -> articles
        Author, on_delete=models.CASCADE, related_name="articles"
    )
    tags = models.ManyToManyField(Tag, blank=True, related_name="articles")  # M2M
    created = models.DateTimeField(auto_now_add=True)  # set once, on insert
    updated = models.DateTimeField(auto_now=True)      # set on every save

    class Meta:
        ordering = ["-created"]                        # newest first by default
        indexes = [models.Index(fields=["status", "created"])]  # composite index

    def __str__(self):
        return self.title


class Comment(models.Model):
    article = models.ForeignKey(Article, on_delete=models.CASCADE, related_name="comments")
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    body = models.TextField()
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created"]                         # comments oldest-first

    def __str__(self):
        return f"Comment by {self.user} on {self.article}"
```

**Line-level notes**
- `settings.AUTH_USER_MODEL` (not a direct `User` import) keeps the code correct if you introduce a custom user model — a best-practice habit.
- `related_name="articles"` lets you write `author.articles.all()` instead of the default `author.article_set.all()`.
- `SlugField(unique=True, db_index=True)` — slugs are used in URLs (`/articles/my-post/`), so uniqueness + an index make lookups fast and safe.
- `auto_now_add` vs `auto_now`: created-once vs updated-every-save. A classic viva question.
- `TextChoices` gives you database-stored short values with human labels and IDE-friendly constants.

**Complexity:** Model definition is O(1) design work; the *index* on `slug` makes lookups O(log n) instead of O(n) full-table scans. The composite `(status, created)` index accelerates the common "published articles, newest first" query.

### Alternative Solution

Instead of a separate `Author` profile model, you could add fields directly to a **custom user model** (`AbstractUser`) — better when the fields are identity-level and you want email login. The `Author`-as-`OneToOne` approach is better when "author" is a *role* distinct from "user" and not every user is an author.

### Interview Variations

- "Add a `category` where each article has one category, and categories can be nested (self-referential FK)." → `ForeignKey("self", null=True, related_name="children")`.
- "A tag application should record *who* tagged it and *when*." → replace the plain M2M with a `through` model.
- "Prevent the same user from commenting twice with identical text on an article." → `Meta.constraints = [UniqueConstraint(fields=["article", "user", "body"], name="...")]`.

### Common Follow-up Questions

- *Why `on_delete=CASCADE` on comments but maybe `PROTECT` elsewhere?* — Comments are meaningless without their article, so cascade; but you might `PROTECT` an author who has published articles to avoid accidental mass deletion.
- *`null` vs `blank` on `bio`?* — `blank=True` only (form-optional); for a text field you store `""` rather than NULL.
- *How does `Meta.ordering` affect performance?* — it adds `ORDER BY` to every default query; index the ordering column or it's a cost on every fetch.

---

## Practical Question 2 — CRUD with the ORM in the Shell

**Difficulty:** Easy
**Estimated Time:** 10 min
**Concepts Tested:** ORM CRUD, QuerySets, `get` vs `filter`, `create`, `update`, `delete`

### Problem Statement

Using only the ORM (no SQL), demonstrate the full CRUD lifecycle on the `Article` model: create an article, read it back by slug, list all published articles, update its status, and delete it. Handle the "not found" case safely.

### Example Input / Output

```python
>>> Article.objects.create(title="Django 101", slug="django-101", body="...", author=a)
<Article: Django 101>
>>> Article.objects.filter(status="published").count()
1
```

### Approach

- **Create:** `.create()` (insert + return) or instantiate + `.save()`.
- **Read one:** `.get(slug=...)` but guard against `DoesNotExist`; prefer `get_object_or_404` in views.
- **Read many:** `.filter(...)` returns a lazy QuerySet.
- **Update:** either load-modify-`save()` (one row) or `.filter(...).update(...)` (bulk, no signals).
- **Delete:** `.delete()`.

### Python Implementation

```python
from blog.models import Article, Author
from django.core.exceptions import ObjectDoesNotExist

# CREATE
author = Author.objects.first()
article = Article.objects.create(
    title="Django 101", slug="django-101", body="Intro", author=author
)

# READ (single) — safely
try:
    a = Article.objects.get(slug="django-101")   # raises if 0 or >1 match
except ObjectDoesNotExist:
    a = None

# READ (many) — lazy QuerySet, only hits DB when evaluated
published = Article.objects.filter(status=Article.Status.PUBLISHED)
print(published.count())                         # -> SELECT COUNT(*)

# UPDATE — instance-level (fires signals, runs save())
a.status = Article.Status.PUBLISHED
a.save(update_fields=["status"])                 # only UPDATE the status column

# UPDATE — bulk (fast, but no save()/signals)
Article.objects.filter(author=author).update(status=Article.Status.PUBLISHED)

# DELETE
a.delete()
```

**Line-level notes**
- `.get()` raises `DoesNotExist` for zero matches and `MultipleObjectsReturned` for many — that's why you wrap it or use `get_object_or_404`.
- `update_fields=["status"]` writes only that column — a small performance and concurrency win.
- `.filter().update()` runs a single SQL `UPDATE` for many rows but **bypasses `save()` and signals** — know this trade-off.

**Complexity:** `get`/`filter` on an indexed column are O(log n); `.update()` bulk is one round-trip regardless of row count (O(1) queries), versus a Python loop of `save()`s which would be O(n) queries.

### Alternative Solution

`get_or_create()` and `update_or_create()` handle "create if missing" atomically, avoiding race conditions between a `filter().exists()` check and a `create()`.

### Interview Variations

- "Create 1000 articles efficiently." → `bulk_create()` (one query) instead of 1000 `.save()`s.
- "Increment a view counter without a race condition." → `F("views") + 1` in an `.update()`.

### Common Follow-up Questions

- *`get` vs `filter().first()`?* — `get` raises on missing/multiple; `first()` returns `None`/the first row. Use `get` when exactly one is expected.
- *Why is `.filter().update()` sometimes dangerous?* — it skips `save()`, `auto_now`, and signals; use instance `save()` when those matter.

---

## Practical Question 3 — Function-Based vs Class-Based Views for a List + Detail Page

**Difficulty:** Medium
**Estimated Time:** 20 min
**Concepts Tested:** FBVs, CBVs, `ListView`/`DetailView`, `render`, `get_object_or_404`, URL routing, templates

### Problem Statement

Build a public blog: a page listing published articles (paginated, 10 per page) and a detail page for a single article by slug. Implement it **twice** — once with function-based views, once with class-based generic views — and wire up the URLs.

### Approach

- **FBV:** query, paginate with `Paginator`, `render` a template.
- **CBV:** `ListView` with `paginate_by`, `DetailView` with `slug_field`; override `get_queryset` to show only published articles.
- URLs use a path converter for the slug and **named** routes.

### Python Implementation

```python
# blog/views.py  — FUNCTION-BASED
from django.core.paginator import Paginator
from django.shortcuts import render, get_object_or_404
from .models import Article

def article_list(request):
    qs = Article.objects.filter(status=Article.Status.PUBLISHED).select_related("author__user")
    page = Paginator(qs, 10).get_page(request.GET.get("page"))   # ?page=2
    return render(request, "blog/list.html", {"page": page})

def article_detail(request, slug):
    article = get_object_or_404(Article, slug=slug, status=Article.Status.PUBLISHED)
    return render(request, "blog/detail.html", {"article": article})


# blog/views.py  — CLASS-BASED (generic)
from django.views.generic import ListView, DetailView

class ArticleListView(ListView):
    template_name = "blog/list.html"
    context_object_name = "page"          # match template
    paginate_by = 10
    def get_queryset(self):
        return (Article.objects.filter(status=Article.Status.PUBLISHED)
                               .select_related("author__user"))

class ArticleDetailView(DetailView):
    model = Article
    slug_field = "slug"
    template_name = "blog/detail.html"
    def get_queryset(self):
        return Article.objects.filter(status=Article.Status.PUBLISHED)
```

```python
# blog/urls.py
from django.urls import path
from . import views

app_name = "blog"                         # namespace
urlpatterns = [
    path("", views.ArticleListView.as_view(), name="list"),
    path("<slug:slug>/", views.ArticleDetailView.as_view(), name="detail"),
]
```

```django
{# templates/blog/list.html #}
{% extends "base.html" %}
{% block content %}
  <ul>
    {% for article in page %}
      <li><a href="{% url 'blog:detail' article.slug %}">{{ article.title }}</a>
          — {{ article.author }}</li>
    {% empty %}
      <li>No articles yet.</li>
    {% endfor %}
  </ul>
  {% if page.has_next %}<a href="?page={{ page.next_page_number }}">Next</a>{% endif %}
{% endblock %}
```

**Line-level notes**
- `select_related("author__user")` avoids the N+1 problem — without it, `{{ article.author }}` in the loop fires a query per article (and another for the user).
- `get_object_or_404` returns a clean 404 instead of an unhandled exception.
- `{% url 'blog:detail' article.slug %}` uses the **named, namespaced** route — never hardcode `/articles/...`.
- The CBV version overrides `get_queryset` (not `get()`), the correct extension point.

**Complexity:** Both are O(1) in query count with `select_related` + pagination (2–3 queries per page regardless of list size); without those, the list page is O(n) queries (N+1) and unbounded rows.

### Alternative Solution

For a JSON API instead of HTML, you'd replace these with DRF views (Part C). For heavy customisation, the FBV is often clearer than fighting the CBV's hooks.

### Interview Variations

- "Only show the logged-in author their own drafts too." → filter `Q(status=PUBLISHED) | Q(author__user=request.user)`.
- "Add search by title via `?q=`." → extend `get_queryset` with `.filter(title__icontains=q)`.

### Common Follow-up Questions

- *Why override `get_queryset` instead of `queryset`?* — `get_queryset` runs per request, so it can use `request.user`/params; `queryset` is evaluated once at import.
- *Where would you add `select_related` in the CBV?* — inside `get_queryset`.

---

# Part B — Authentication, Forms & CRUD

## Practical Question 4 — Registration, Login & Logout (Session Auth)

**Difficulty:** Medium
**Estimated Time:** 20–25 min
**Concepts Tested:** Auth system, `UserCreationForm`, `authenticate`/`login`/`logout`, `@login_required`, CSRF, sessions

### Problem Statement

Implement full session-based authentication for the web app: a registration page that creates a user (with hashed password), a login page, a logout action, and a protected dashboard only logged-in users can see.

### Approach

1. **Register:** use `UserCreationForm` (handles password hashing + confirmation) — `form.save()` creates the user.
2. **Login:** `authenticate()` verifies credentials → `login()` starts the session.
3. **Logout:** `logout()` flushes the session.
4. **Protect:** `@login_required` decorator (FBV) or `LoginRequiredMixin` (CBV).
5. Every POST form includes `{% csrf_token %}`.

### Python Implementation

```python
# accounts/views.py
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect

def register_view(request):
    form = UserCreationForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user = form.save()            # creates user; password is hashed via set_password
        login(request, user)          # auto-login after signup
        return redirect("dashboard")
    return render(request, "accounts/register.html", {"form": form})

def login_view(request):
    form = AuthenticationForm(request, data=request.POST or None)
    if request.method == "POST" and form.is_valid():
        user = form.get_user()        # already authenticated by the form
        login(request, user)          # writes user id to session, sets cookie
        return redirect("dashboard")
    return render(request, "accounts/login.html", {"form": form})

def logout_view(request):
    logout(request)                   # clears session data + cycles session key
    return redirect("login")

@login_required                       # redirects anonymous users to LOGIN_URL
def dashboard(request):
    return render(request, "accounts/dashboard.html", {"user": request.user})
```

```django
{# accounts/login.html #}
{% extends "base.html" %}
{% block content %}
  <form method="post">
    {% csrf_token %}          {# REQUIRED — CsrfViewMiddleware rejects POST without it #}
    {{ form.as_p }}
    <button type="submit">Log in</button>
  </form>
{% endblock %}
```

**Line-level notes**
- `UserCreationForm.save()` calls `set_password()` internally — you **never** hash manually or store plaintext.
- `AuthenticationForm` runs `authenticate()` for you; `form.get_user()` returns the verified user.
- `login()` is what makes the user "stay logged in" — it stores the user id in the session and issues the `sessionid` cookie.
- `@login_required` sends anonymous users to `settings.LOGIN_URL` with a `?next=` back-link.
- Missing `{% csrf_token %}` → `403 Forbidden` on submit. This is a top exam gotcha.

**Complexity:** O(1) per request; the login lookup is an indexed username query (O(log n)); password verification is deliberately *slow* (hashing) — that's a security feature, not a bug.

### Alternative Solution

For APIs (mobile/SPA), replace session login with **JWT** (Part C, Q9) — stateless, no cookie/CSRF.

### Interview Variations

- "Redirect to the page the user originally wanted." → honour `request.GET['next']` / `next` hidden field.
- "Email-based login." → custom user model with `USERNAME_FIELD = "email"` + custom backend.

### Common Follow-up Questions

- *Where is the password stored and in what form?* — hashed (PBKDF2 + salt) in the user table, never plaintext.
- *`authenticate` vs `login`?* — verify credentials vs establish the session.
- *Why does logout cycle the session key?* — to prevent session fixation.

---

## Practical Question 5 — ModelForm CRUD with Validation & File Upload

**Difficulty:** Medium
**Estimated Time:** 25 min
**Concepts Tested:** ModelForm, custom validation (`clean_`/`clean`), `request.FILES`, `ImageField`, permission checks

### Problem Statement

Build a "create/edit article" form using a `ModelForm`. Requirements: only the article's author can edit it; the title must be at least 5 characters; a draft cannot be published without a body of at least 50 characters; and authors can upload a cover image.

### Approach

1. `ModelForm` from `Article` with the editable fields + an `ImageField`.
2. Field-level rule → `clean_title`. Cross-field rule (status vs body length) → `clean`.
3. View passes `request.POST, request.FILES`; template uses `enctype="multipart/form-data"`.
4. Ownership enforced in the view (object-level check).

### Python Implementation

```python
# blog/forms.py
from django import forms
from .models import Article

class ArticleForm(forms.ModelForm):
    class Meta:
        model = Article
        fields = ["title", "slug", "body", "status", "tags", "cover"]

    def clean_title(self):                              # single-field validation
        title = self.cleaned_data["title"]
        if len(title) < 5:
            raise forms.ValidationError("Title must be at least 5 characters.")
        return title                                    # MUST return the cleaned value

    def clean(self):                                    # cross-field validation
        data = super().clean()
        if data.get("status") == Article.Status.PUBLISHED and len(data.get("body", "")) < 50:
            raise forms.ValidationError("Published articles need a body of 50+ characters.")
        return data
```

```python
# blog/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from .forms import ArticleForm
from .models import Article

@login_required
def article_edit(request, slug):
    article = get_object_or_404(Article, slug=slug)
    if article.author.user != request.user:            # OBJECT-LEVEL ownership check
        raise PermissionDenied                          # -> 403
    form = ArticleForm(request.POST or None, request.FILES or None, instance=article)
    if request.method == "POST" and form.is_valid():
        form.save()                                     # writes fields + uploaded file
        return redirect("blog:detail", slug=article.slug)
    return render(request, "blog/edit.html", {"form": form})
```

```django
{# blog/edit.html #}
<form method="post" enctype="multipart/form-data">   {# enctype REQUIRED for uploads #}
  {% csrf_token %}
  {{ form.as_p }}
  <button type="submit">Save</button>
</form>
```

(Add `cover = models.ImageField(upload_to="covers/", blank=True)` to `Article`; install Pillow; set `MEDIA_URL`/`MEDIA_ROOT`.)

**Line-level notes**
- `clean_title` must **return** the value — forgetting this silently drops the field.
- `clean()` sees all fields, so it's the right place for the "status depends on body" rule.
- `ArticleForm(request.POST, request.FILES, instance=article)` — passing `instance` makes it an *edit* (update) rather than a create; passing `request.FILES` is mandatory for the upload.
- The ownership check is **object-level** authorization (model permissions alone can't express "own article only").
- `enctype="multipart/form-data"` — miss it and the file never arrives.

**Complexity:** O(1) per request; validation is linear in field count.

### Alternative Solution

In an API, this becomes a DRF `ModelSerializer` with `validate_title`/`validate` and an `IsOwnerOrReadOnly` permission (Part C).

### Interview Variations

- "Reject cover images over 2 MB or non-image files." → custom validator on the field checking `value.size` and content type.
- "Auto-generate the slug from the title." → override `save()` or use `slugify` in `clean`.

### Common Follow-up Questions

- *`clean_<field>` vs `clean`?* — single field vs cross-field.
- *Why not trust client-side validation?* — it's bypassable; server validation is authoritative.
- *Where should uploaded files live in production?* — S3, not the app server disk.

---

# Part C — Building a REST API with DRF

## Practical Question 6 — Serializers with Validation & Nested Data

**Difficulty:** Medium
**Estimated Time:** 20 min
**Concepts Tested:** `ModelSerializer`, `read_only`/`write_only`, nested serializers, `validate_`/`validate`, N+1 awareness

### Problem Statement

Write serializers for the `Article` API: output should include the author's username and the list of tag names (nested/read-friendly), plus `id` and `created` as read-only. On input, validate that the title is unique-ish (≥5 chars) and that a published article has a non-empty body.

### Approach

- `ModelSerializer` for `Article`.
- Author shown as a nested read-only serializer; tags shown by name via `SlugRelatedField` or a nested tag serializer.
- `read_only_fields` for `id`, `created`.
- Field validation via `validate_title`; object validation via `validate`.

### Python Implementation

```python
# blog/serializers.py
from rest_framework import serializers
from .models import Article, Tag

class TagSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tag
        fields = ["id", "name"]

class ArticleSerializer(serializers.ModelSerializer):
    author_name = serializers.CharField(source="author.user.username", read_only=True)
    tags = TagSerializer(many=True, read_only=True)               # nested read
    tag_ids = serializers.PrimaryKeyRelatedField(                 # write side
        queryset=Tag.objects.all(), many=True, write_only=True, source="tags"
    )

    class Meta:
        model = Article
        fields = ["id", "title", "slug", "body", "status",
                  "author_name", "tags", "tag_ids", "created"]
        read_only_fields = ["id", "created"]

    def validate_title(self, value):                              # field-level
        if len(value) < 5:
            raise serializers.ValidationError("Title too short (min 5).")
        return value

    def validate(self, data):                                     # object-level
        if data.get("status") == Article.Status.PUBLISHED and not data.get("body"):
            raise serializers.ValidationError("Published articles need a body.")
        return data
```

**Line-level notes**
- `author_name` uses `source="author.user.username"` — a read-only computed field pulling through relations.
- Two tag fields: `tags` (nested, read-only, human-friendly output) and `tag_ids` (write-only PKs, mapped to the same `tags` attribute via `source`). This is the idiomatic "different shape in vs out" pattern.
- `read_only_fields` prevents clients from setting server-controlled values.
- `validate_<field>` vs `validate` mirror Django forms' `clean_<field>`/`clean`.
- **N+1 warning:** serializing many articles with nested `author`/`tags` will N+1 unless the *view's* queryset uses `select_related("author__user").prefetch_related("tags")` (see Q7).

**Complexity:** Serialization is O(n) in the number of objects/fields; the query efficiency depends on the view's queryset, not the serializer.

### Alternative Solution

For fully nested *writes* (create tags inline), override `create()`/`update()` to handle the nested payload — more control, more code.

### Interview Variations

- "Add a computed `comment_count`." → `serializers.IntegerField(source="comments.count", read_only=True)` (or annotate in the queryset to avoid N+1).
- "Hide `body` in list view but show it in detail." → separate list/detail serializers via `get_serializer_class`.

### Common Follow-up Questions

- *Serializer vs ModelSerializer?* — explicit fields + manual create/update vs auto-generated.
- *How do you avoid N+1 with nested serializers?* — optimise the queryset in the view.
- *`read_only` vs `write_only`?* — output-only (computed) vs input-only (passwords).

---

## Practical Question 7 — Full CRUD API with ViewSet + Router (+ Permissions, Pagination, Filtering)

**Difficulty:** Medium–Hard
**Estimated Time:** 30 min
**Concepts Tested:** `ModelViewSet`, routers, `IsAuthenticatedOrReadOnly`, custom object permission, pagination, `DjangoFilterBackend`, `SearchFilter`, `OrderingFilter`, queryset optimisation

### Problem Statement

Expose the `Article` model as a complete REST API: `GET /api/articles/` (list, paginated), `POST` (create), `GET/PUT/PATCH/DELETE /api/articles/{id}/`. Anyone can read; only authenticated users can create; only the author can edit/delete their own article. Support filtering by `status`, search across `title`/`body`, and ordering by `created`.

### Approach

1. `ModelViewSet` + `DefaultRouter` for the CRUD + URLs.
2. `permission_classes` = `[IsAuthenticatedOrReadOnly, IsOwnerOrReadOnly]`.
3. `filter_backends` with django-filter, search, ordering.
4. Optimise `get_queryset` with `select_related`/`prefetch_related` to kill N+1.
5. `perform_create` sets the author from the request user.

### Python Implementation

```python
# blog/permissions.py
from rest_framework.permissions import BasePermission, SAFE_METHODS

class IsOwnerOrReadOnly(BasePermission):
    def has_object_permission(self, request, view, obj):
        if request.method in SAFE_METHODS:          # GET/HEAD/OPTIONS -> allow all
            return True
        return obj.author.user == request.user       # writes only for the owner
```

```python
# blog/api.py
from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticatedOrReadOnly
from rest_framework.filters import SearchFilter, OrderingFilter
from django_filters.rest_framework import DjangoFilterBackend
from .models import Article
from .serializers import ArticleSerializer
from .permissions import IsOwnerOrReadOnly

class ArticleViewSet(viewsets.ModelViewSet):
    serializer_class = ArticleSerializer
    permission_classes = [IsAuthenticatedOrReadOnly, IsOwnerOrReadOnly]
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ["status"]                     # ?status=published
    search_fields = ["title", "body"]                 # ?search=django
    ordering_fields = ["created", "title"]            # ?ordering=-created

    def get_queryset(self):
        # select_related for to-one (author->user); prefetch_related for to-many (tags)
        return (Article.objects
                .select_related("author__user")
                .prefetch_related("tags")
                .all())

    def perform_create(self, serializer):
        serializer.save(author=self.request.user.author)  # attach current user as author
```

```python
# config/urls.py
from rest_framework.routers import DefaultRouter
from blog.api import ArticleViewSet
from django.urls import path, include

router = DefaultRouter()
router.register("articles", ArticleViewSet, basename="article")

urlpatterns = [path("api/", include(router.urls))]     # auto-generates all CRUD routes
```

```python
# config/settings.py — global pagination
REST_FRAMEWORK = {
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 10,
}
```

**Line-level notes**
- The router turns one viewset into six endpoints and wires HTTP methods → actions (`GET`→list/retrieve, `POST`→create, `PUT`→update, `PATCH`→partial_update, `DELETE`→destroy).
- `IsOwnerOrReadOnly.has_object_permission` only runs on **detail** actions — the list is already read-open here; if you needed per-user *list* filtering you'd also filter `get_queryset`.
- `perform_create` injects the author server-side so clients can't spoof authorship.
- `select_related("author__user").prefetch_related("tags")` converts a potential N+1 into ~3 queries per page.
- Filtering/search/ordering are all driven by query params, handled before pagination.

**Example requests**
```
GET  /api/articles/?status=published&search=django&ordering=-created&page=2
POST /api/articles/     Authorization: Bearer <jwt>   {json body}
PATCH /api/articles/5/  {"title": "Updated"}          # partial update
DELETE /api/articles/5/                                # 204 if owner, 403 otherwise
```

**Complexity:** Per list request: ~3 queries (article page + prefetch tags + counts) regardless of page size → O(1) queries; DB-side filtering/ordering uses indexes → O(log n) lookups.

### Alternative Solution

Use separate **generic views** (`ListCreateAPIView` + `RetrieveUpdateDestroyAPIView`) instead of a viewset+router when you want explicit URL control. Use `APIView` when the logic is too custom for generics.

### Interview Variations

- "Add a custom `POST /api/articles/{id}/publish/` action." → `@action(detail=True, methods=["post"])`.
- "Cursor-paginate the feed." → swap pagination class to `CursorPagination` with `ordering = "-created"`.
- "Only return the requesting user's drafts in the list." → filter `get_queryset` by `request.user`.

### Common Follow-up Questions

- *Why does object permission not protect the list endpoint?* — `has_object_permission` runs per object on detail actions; list filtering must be done in `get_queryset`.
- *APIView vs generic vs viewset?* — control vs conciseness spectrum.
- *Where do you fix N+1 in DRF?* — the viewset's `get_queryset`.

---

## Practical Question 8 — JWT Authentication for the API

**Difficulty:** Medium
**Estimated Time:** 20 min
**Concepts Tested:** JWT auth, `simplejwt`, access/refresh tokens, stateless auth, `Authorization` header

### Problem Statement

Secure the API with JWT: expose endpoints to obtain and refresh tokens, require a valid access token for write operations, and explain the token lifecycle.

### Approach

1. Install `djangorestframework-simplejwt`.
2. Set the default authentication class to JWT.
3. Add `/api/token/` (obtain) and `/api/token/refresh/` routes.
4. Clients send `Authorization: Bearer <access>` on each request.

### Python Implementation

```python
# config/settings.py
from datetime import timedelta

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework_simplejwt.authentication.JWTAuthentication",
    ],
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 10,
}

SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(minutes=15),   # short-lived (limits theft window)
    "REFRESH_TOKEN_LIFETIME": timedelta(days=7),      # longer-lived, used to renew
    "ROTATE_REFRESH_TOKENS": True,
}
```

```python
# config/urls.py
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from django.urls import path

urlpatterns += [
    path("api/token/", TokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("api/token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
]
```

**Token lifecycle (the flow you must be able to narrate):**
```
POST /api/token/  {username, password}
   -> 200 {"access": "<jwt>", "refresh": "<jwt>"}

GET /api/articles/   Header: Authorization: Bearer <access>
   -> server verifies signature + expiry (NO DB lookup) -> request.user set

(access expires after 15 min)
POST /api/token/refresh/  {"refresh": "<jwt>"}
   -> 200 {"access": "<new jwt>"}
```

**Line-level notes**
- The access token is **signed** with `SECRET_KEY`; the server verifies it without hitting the DB — that's what makes JWT stateless and scalable.
- **Short access + long refresh** limits damage if an access token leaks; the refresh token renews it.
- `ROTATE_REFRESH_TOKENS` issues a new refresh token on each refresh (with blacklisting, old ones become invalid).
- The payload is only base64-encoded (readable!) — never put secrets in it.

**Complexity:** Verifying a JWT is O(1) (a signature check), versus token/session auth which needs a DB/session lookup per request.

### Alternative Solution

**Token auth** (`rest_framework.authtoken`) — a DB-stored opaque token; simpler and revocable, but requires a lookup per request. **Session auth** — for a same-site browser SPA.

### Interview Variations

- "Log a user out / revoke a token." → add the blacklist app and blacklist the refresh token (access tokens still valid until expiry — explain why).
- "Add the user's role to the token." → custom `TokenObtainPairSerializer` adding claims.

### Common Follow-up Questions

- *Session vs JWT — when each?* — same-site web vs mobile/SPA/cross-domain.
- *Why can't you instantly revoke a JWT?* — stateless; mitigate with short expiry + blacklist.
- *Is a JWT encrypted?* — no, signed; readable, so no secrets inside.

---

## Practical Question 9 — API Documentation with drf-spectacular

**Difficulty:** Easy
**Estimated Time:** 10 min
**Concepts Tested:** OpenAPI, Swagger UI, schema generation, `drf-spectacular`

### Problem Statement

Add interactive, auto-generated API documentation (OpenAPI 3 + Swagger UI) to the project.

### Python Implementation

```python
# settings.py
INSTALLED_APPS += ["drf_spectacular"]
REST_FRAMEWORK["DEFAULT_SCHEMA_CLASS"] = "drf_spectacular.openapi.AutoSchema"
SPECTACULAR_SETTINGS = {"TITLE": "Blog API", "VERSION": "1.0.0"}
```

```python
# urls.py
from drf_spectacular.views import (SpectacularAPIView,
                                   SpectacularSwaggerView, SpectacularRedocView)
urlpatterns += [
    path("api/schema/", SpectacularAPIView.as_view(), name="schema"),                 # raw OpenAPI
    path("api/docs/", SpectacularSwaggerView.as_view(url_name="schema")),             # Swagger UI
    path("api/redoc/", SpectacularRedocView.as_view(url_name="schema")),              # ReDoc
]
```

**Notes:** `drf-spectacular` introspects your serializers, viewsets, and routers to emit an OpenAPI 3 schema; Swagger UI renders it as browsable, try-it-live docs. Use `@extend_schema` to annotate endpoints where introspection needs help.

### Common Follow-up Questions

- *Why document an API?* — it's a contract for other developers; enables client generation and testing.
- *OpenAPI vs Swagger?* — OpenAPI is the spec; Swagger UI is the rendering tool.

---

# Part D — ORM Query Workbench (Notebook-Style Shell Exercises)

> Assessments often drop you into `python manage.py shell` (or a Jupyter kernel with `django_extensions`' `shell_plus`) and ask you to write **QuerySets**. Below is a "notebook" of progressively harder queries, split into cells with explanations — exactly how a practical lab exam is structured. Assume the models from Q1.

### Cell 1 — Setup (imports & Django bootstrap)

```python
# If in a Jupyter notebook, bootstrap Django first:
import os, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from django.db.models import Count, Q, F, Avg, Sum, Prefetch
from blog.models import Article, Author, Tag, Comment
```
*Explanation:* Notebooks aren't started through `manage.py`, so you must set the settings module and call `django.setup()` to load the app registry before importing models. In `manage.py shell` this is already done.

### Cell 2 — Basic filtering & field lookups

```python
Article.objects.filter(status="published")                     # equality
Article.objects.filter(title__icontains="django")              # case-insensitive LIKE
Article.objects.filter(created__year=2026)                     # date part
Article.objects.filter(title__startswith="How")                # prefix
Article.objects.exclude(status="draft")                        # NOT
```
*Explanation:* **Field lookups** (`__icontains`, `__year`, `__gte`) are the ORM's operator language, compiled to SQL `WHERE` clauses. `exclude` is the negation of `filter`.

### Cell 3 — Ordering, limiting, values

```python
Article.objects.order_by("-created")[:5]                       # latest 5 (LIMIT 5)
Article.objects.values("id", "title")                          # dicts, not model objects
Article.objects.values_list("title", flat=True)                # flat list of titles
Article.objects.count()                                        # SELECT COUNT(*)
Article.objects.filter(status="published").exists()            # cheap EXISTS check
```
*Explanation:* Slicing adds SQL `LIMIT/OFFSET` (lazily — no query until evaluated). `values()`/`values_list()` fetch only chosen columns (faster, less memory) — useful when you don't need full model instances. `exists()` is cheaper than `count()` or `len()` when you only need a yes/no.

### Cell 4 — Spanning relationships (joins via `__`)

```python
# Articles whose author's username is 'alice' (join Article->Author->User)
Article.objects.filter(author__user__username="alice")

# Authors who have at least one published article
Author.objects.filter(articles__status="published").distinct()

# Articles tagged 'python'
Article.objects.filter(tags__name="python")
```
*Explanation:* The double-underscore `__` traverses relationships, generating SQL JOINs. `distinct()` removes duplicate rows that many-to-many joins can produce.

### Cell 5 — Aggregation & annotation

```python
# Total published articles across the DB
Article.objects.filter(status="published").aggregate(total=Count("id"))

# Per-author article count (annotate = per-row aggregate)
Author.objects.annotate(n=Count("articles")).values("user__username", "n")

# Authors with more than 3 articles, most prolific first
(Author.objects.annotate(n=Count("articles"))
               .filter(n__gt=3)
               .order_by("-n"))
```
*Explanation:* `aggregate()` collapses a queryset to a **single** summary dict; `annotate()` attaches a computed value to **each** row (a `GROUP BY`). `filter` after `annotate` becomes a SQL `HAVING`. This is the ORM equivalent of SQL `GROUP BY ... HAVING`.

### Cell 6 — `Q` objects (complex OR/AND) and `F` expressions

```python
# published OR authored by the current user
Article.objects.filter(Q(status="published") | Q(author__user=some_user))

# NOT draft AND title contains 'api'
Article.objects.filter(~Q(status="draft") & Q(title__icontains="api"))

# Compare two columns / atomic increment (F avoids race conditions)
Article.objects.filter(updated__gt=F("created"))
Article.objects.filter(pk=1).update(views=F("views") + 1)      # atomic in SQL
```
*Explanation:* `Q` objects let you build `OR`/`NOT`/grouped conditions that plain kwargs can't. `F` references a column *inside* the database, enabling column-to-column comparisons and race-free atomic updates (the increment happens in SQL, not read-modify-write in Python).

### Cell 7 — The N+1 problem, demonstrated and fixed

```python
# BAD — N+1 (1 + one query per article for author, and per article for tags)
for a in Article.objects.all():
    print(a.author.user.username, [t.name for t in a.tags.all()])

# GOOD — select_related (JOIN, to-one) + prefetch_related (2nd query, to-many)
qs = Article.objects.select_related("author__user").prefetch_related("tags")
for a in qs:
    print(a.author.user.username, [t.name for t in a.tags.all()])   # ~3 queries total
```
*Explanation:* This is the single most-tested performance idea. `select_related` follows to-one relations with a JOIN in the same query; `prefetch_related` fetches to-many relations in a second query and joins in Python. Verify with `django-debug-toolbar` or `len(connection.queries)`.

### Cell 8 — Counting queries (proving your optimisation)

```python
from django.db import connection, reset_queries
from django.test.utils import override_settings

@override_settings(DEBUG=True)                 # query logging only happens when DEBUG
def count():
    reset_queries()
    list(Article.objects.select_related("author__user").prefetch_related("tags"))
    print(len(connection.queries), "queries")
count()
```
*Explanation:* `connection.queries` records executed SQL when `DEBUG=True`. This is how you *prove* an N+1 fix in an assessment — show the query count dropping from N+1 to a constant.

### Cell 9 — `Prefetch` with a filtered/ordered inner queryset

```python
recent_comments = Prefetch(
    "comments",
    queryset=Comment.objects.order_by("-created")[:5],
    to_attr="recent_comments",
)
for a in Article.objects.prefetch_related(recent_comments):
    print(a.title, a.recent_comments)          # only the 5 latest comments, no extra queries
```
*Explanation:* A `Prefetch` object customises *how* related objects are fetched (filter/order/limit them), stored under `to_attr`. This is the advanced tool for controlling prefetches — a strong signal in an interview.

### Notebook follow-up questions

- *Why is `values_list(..., flat=True)` faster than iterating model objects?* — it skips model instantiation and fetches one column.
- *When does a M2M query need `distinct()`?* — when a join produces duplicate parent rows.
- *`aggregate` vs `annotate`?* — whole-queryset summary vs per-row computed column.
- *How do you prove there's no N+1?* — count `connection.queries` before/after.

---

# Part E — Advanced (Signals, Caching, Middleware, Performance)

## Practical Question 10 — Auto-Create a Profile with a Signal

**Difficulty:** Easy–Medium
**Estimated Time:** 10 min
**Concepts Tested:** `post_save` signal, `created` flag, `apps.py ready()`, decoupling

### Problem Statement

Every time a `User` is created (from anywhere — admin, registration, shell), automatically create a matching `Author` profile.

### Python Implementation

```python
# blog/signals.py
from django.conf import settings
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Author

@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_author_profile(sender, instance, created, **kwargs):
    if created:                                   # only on INSERT, not every update
        Author.objects.create(user=instance)
```

```python
# blog/apps.py
from django.apps import AppConfig

class BlogConfig(AppConfig):
    name = "blog"
    def ready(self):
        from . import signals                     # register receivers at startup
```

**Line-level notes**
- `if created:` is essential — without it you'd try to create a duplicate profile on *every* user save. Top exam bug.
- Importing `signals` in `ready()` is what actually connects the receiver; forget it and the signal silently never fires.
- The reaction is decoupled from all the places users get created — the whole point of signals.

### Alternative Solution

Override the user creation flow, or use a `get_or_create` in a service function — more explicit, no "hidden" behaviour. Prefer this when *you* control creation and want traceability.

### Interview Variations

- "Send a welcome email on registration." → `post_save` + `send_welcome_email.delay(instance.id)` (offload to Celery).
- "Invalidate a cache when an article changes." → `post_save`/`post_delete` on `Article` calling `cache.delete(...)`.

### Follow-up Questions

- *Downside of signals?* — hidden control flow, synchronous; heavy work should go to Celery.
- *Why check `created`?* — to distinguish insert from update.

---

## Practical Question 11 — Cache an Expensive Endpoint & Invalidate It

**Difficulty:** Medium
**Estimated Time:** 15 min
**Concepts Tested:** low-level cache API, `cache_page`, Redis backend, invalidation

### Problem Statement

A "trending tags" computation is expensive (aggregation over all articles). Cache it for 10 minutes and invalidate it whenever an article's tags change.

### Python Implementation

```python
# settings.py — Redis cache backend (shared across workers)
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.redis.RedisCache",
        "LOCATION": os.environ.get("REDIS_URL", "redis://127.0.0.1:6379"),
    }
}
```

```python
# blog/services.py
from django.core.cache import cache
from django.db.models import Count
from .models import Tag

CACHE_KEY = "trending_tags"

def get_trending_tags():
    data = cache.get(CACHE_KEY)                       # 1) try cache
    if data is None:                                  # 2) miss -> compute
        data = list(Tag.objects.annotate(n=Count("articles"))
                                .order_by("-n")[:10]
                                .values("name", "n"))
        cache.set(CACHE_KEY, data, timeout=600)       # 3) store for 10 min
    return data

def invalidate_trending_tags():
    cache.delete(CACHE_KEY)                           # explicit invalidation on write
```

```python
# wire invalidation to writes (blog/signals.py)
from django.db.models.signals import m2m_changed
from .models import Article
@receiver(m2m_changed, sender=Article.tags.through)
def _bust(sender, **kwargs):
    invalidate_trending_tags()
```

**Line-level notes**
- The **cache-aside** pattern: check cache → on miss compute + store. This is the most common caching pattern.
- Redis (not local memory) so all Gunicorn workers/servers share one consistent cache.
- Time-based expiry (`timeout=600`) **plus** explicit invalidation on the relevant write — belt and braces for freshness.
- `m2m_changed` fires when tags are added/removed (the M2M through table changes).

### Alternative Solution

Whole-view caching with `@cache_page(600)` for a page whose entire output is cacheable — simpler but coarser (can't easily invalidate a single key).

### Follow-up Questions

- *Local-memory vs Redis?* — per-process vs shared; production needs shared.
- *Why is invalidation hard?* — knowing exactly when data is stale; combine expiry + explicit deletion.
- *Risk of caching per-user data under a shared key?* — data leakage between users.

---

## Practical Question 12 — Custom Middleware (Request Timing + Rate Limit sketch)

**Difficulty:** Medium
**Estimated Time:** 15 min
**Concepts Tested:** middleware structure, request lifecycle, short-circuiting, ordering

### Python Implementation

```python
# blog/middleware.py
import time
from django.core.cache import cache
from django.http import JsonResponse

class TimingMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response              # set up once at startup

    def __call__(self, request):
        start = time.perf_counter()
        response = self.get_response(request)         # -> next middleware / view (inbound)
        response["X-Response-Time-ms"] = f"{(time.perf_counter() - start) * 1000:.1f}"
        return response                               # outbound

class SimpleRateLimitMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        ip = request.META.get("REMOTE_ADDR", "anon")
        key = f"rl:{ip}"
        hits = cache.get(key, 0)
        if hits >= 100:                               # 100 req/min cap
            return JsonResponse({"detail": "Too many requests"}, status=429)  # short-circuit
        cache.set(key, hits + 1, timeout=60)
        return self.get_response(request)
```

```python
# settings.py — order matters
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",   # after session!
    "blog.middleware.SimpleRateLimitMiddleware",
    "blog.middleware.TimingMiddleware",
    # ...
]
```

**Line-level notes**
- The `__init__(get_response)` / `__call__(request)` shape is the modern middleware contract: init once, `__call__` per request.
- Everything **before** `get_response()` is inbound processing; **after** is outbound. The rate limiter **short-circuits** (returns without calling `get_response`) so the view never runs.
- Ordering: `AuthenticationMiddleware` must come after `SessionMiddleware` (it reads the session) — the canonical ordering question.

### Follow-up Questions

- *Middleware vs decorator?* — global (every request) vs per-view.
- *Why is production rate limiting not done this naively?* — needs atomic counters / a real limiter (django-ratelimit); this illustrates the concept.
- *What does short-circuiting mean?* — returning a response early so downstream layers/view don't execute.

---

# Part F — Deployment Tasks

## Practical Question 13 — Production-Ready Settings

**Difficulty:** Medium
**Estimated Time:** 15 min
**Concepts Tested:** env vars, `DEBUG`, `ALLOWED_HOSTS`, secure cookies, PostgreSQL, static files

### Python Implementation

```python
# config/settings.py (production-relevant excerpts)
import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ["SECRET_KEY"]                       # from env, never hardcoded
DEBUG = os.environ.get("DEBUG", "False") == "True"         # False in prod
ALLOWED_HOSTS = os.environ.get("ALLOWED_HOSTS", "").split(",")

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ["DB_NAME"],
        "USER": os.environ["DB_USER"],
        "PASSWORD": os.environ["DB_PASSWORD"],
        "HOST": os.environ["DB_HOST"],                     # RDS endpoint
        "PORT": "5432",
        "CONN_MAX_AGE": 60,                                # persistent connections
    }
}

# HTTPS / secure cookies (production)
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SESSION_COOKIE_HTTPONLY = True
SECURE_HSTS_SECONDS = 31536000

# Static files (WhiteNoise) + media (S3 in real prod)
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STORAGES = {"staticfiles": {"BACKEND":
    "whitenoise.storage.CompressedManifestStaticFilesStorage"}}
```

**Verification command:** `python manage.py check --deploy` audits many of these.
**Deploy steps:** `collectstatic` → `migrate` → run under Gunicorn.

### Follow-up Questions

- *Why `DEBUG=False`?* — debug pages leak source/settings/data.
- *Why env vars for secrets?* — keep them out of git; per-environment; rotatable.
- *What does `collectstatic` do?* — gathers static files into `STATIC_ROOT` for the web server/WhiteNoise/CDN.

---

## Practical Question 14 — Deploy to EC2 (Gunicorn + Nginx + systemd)

**Difficulty:** Hard
**Estimated Time:** 30–40 min
**Concepts Tested:** EC2, SSH, Gunicorn, Nginx reverse proxy, systemd, static files, RDS

### Problem Statement

Deploy the Django app to a fresh Ubuntu EC2 instance: run it under Gunicorn as a systemd service, put Nginx in front as a reverse proxy serving static files, and connect to an RDS PostgreSQL database.

### Approach & Commands

```bash
# 1. SSH in (security group: open 80/443; restrict 22 to your IP)
ssh -i key.pem ubuntu@<EC2_PUBLIC_IP>

# 2. System deps
sudo apt update && sudo apt install -y python3-venv python3-pip nginx postgresql-client

# 3. Code + venv + deps
git clone <repo> /home/ubuntu/app && cd /home/ubuntu/app
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt gunicorn

# 4. Env + migrate + static (DB_HOST points at the RDS endpoint)
export $(cat .env | xargs)
python manage.py migrate
python manage.py collectstatic --noinput
```

```ini
# 5. /etc/systemd/system/gunicorn.service  — keep Gunicorn running & auto-restart
[Unit]
Description=gunicorn daemon
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/app
EnvironmentFile=/home/ubuntu/app/.env
ExecStart=/home/ubuntu/app/venv/bin/gunicorn config.wsgi:application \
          --workers 3 --bind unix:/home/ubuntu/app/gunicorn.sock

[Install]
WantedBy=multi-user.target
```

```nginx
# 6. /etc/nginx/sites-available/app  — reverse proxy + static
server {
    listen 80;
    server_name example.com;

    location /static/ { alias /home/ubuntu/app/staticfiles/; }   # Nginx serves static

    location / {
        proxy_pass http://unix:/home/ubuntu/app/gunicorn.sock;   # dynamic -> Gunicorn
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
# 7. Enable everything
sudo systemctl enable --now gunicorn
sudo ln -s /etc/nginx/sites-available/app /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl restart nginx
# 8. HTTPS: sudo certbot --nginx  (Let's Encrypt) or use an ALB + ACM cert
```

**Line-level notes**
- **Gunicorn** runs the app (`config.wsgi:application`) with `--workers 3` (~`2×cores+1`), bound to a **Unix socket** (faster/safer than a TCP port for local proxying).
- **systemd** keeps Gunicorn alive across crashes/reboots — without it, a crash means downtime.
- **Nginx** serves `/static/` directly (fast) and proxies everything else to Gunicorn; the `X-Forwarded-*` headers let Django know the original scheme/host.
- The **database is on RDS**, not the instance — so scaling/replacing the instance doesn't lose data.

### Alternative Solution

**Elastic Beanstalk**: `eb init` → `eb create` → `eb deploy`. It provisions EC2 + load balancer + auto-scaling and handles the Nginx/Gunicorn wiring for you — less control, far less manual work.

### Interview Variations

- "Serve media from S3." → `django-storages` + S3 backend + CloudFront.
- "Zero-downtime deploys." → multiple instances behind a load balancer, rolling deploys, backwards-compatible migrations.

### Follow-up Questions

- *Gunicorn vs Nginx roles?* — app server vs reverse proxy/static/TLS.
- *Why systemd?* — process supervision + restart on failure/boot.
- *Why RDS, not DB on the instance?* — managed backups/failover, decoupled from ephemeral app servers.
- *EC2 vs Beanstalk?* — control vs automation.

---

# Part G — Testing

## Practical Question 15 — Test the Model, View, and API

**Difficulty:** Medium
**Estimated Time:** 25 min
**Concepts Tested:** `TestCase`, `Client`, DRF `APIClient`, fixtures, assertions, auth in tests

### Python Implementation

```python
# blog/tests.py
from django.test import TestCase
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient
from rest_framework import status
from .models import Article, Author

User = get_user_model()

class ArticleModelTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("alice", password="pw12345")
        self.author = Author.objects.create(user=self.user)

    def test_str_returns_title(self):                       # model behaviour
        a = Article.objects.create(title="Hello", slug="hello",
                                   body="x", author=self.author)
        self.assertEqual(str(a), "Hello")

    def test_default_ordering_newest_first(self):
        Article.objects.create(title="Old", slug="old", body="x", author=self.author)
        Article.objects.create(title="New", slug="new", body="x", author=self.author)
        titles = list(Article.objects.values_list("title", flat=True))
        self.assertEqual(titles[0], "New")                  # Meta.ordering = -created


class ArticleAPITest(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user("bob", password="pw12345")
        self.author = Author.objects.create(user=self.user)

    def test_list_is_public(self):
        res = self.client.get("/api/articles/")
        self.assertEqual(res.status_code, status.HTTP_200_OK)

    def test_create_requires_auth(self):                    # 401/403 when anonymous
        res = self.client.post("/api/articles/", {"title": "New API", "slug": "n",
                                                   "body": "b"*60, "status": "published"})
        self.assertIn(res.status_code, (401, 403))

    def test_authenticated_user_can_create(self):
        self.client.force_authenticate(self.user)           # bypass token in tests
        res = self.client.post("/api/articles/", {"title": "New API", "slug": "napi",
                                                  "body": "b"*60, "status": "published",
                                                  "tag_ids": []}, format="json")
        self.assertEqual(res.status_code, 201)
        self.assertEqual(Article.objects.count(), 1)
```

**Line-level notes**
- `setUp` builds fresh fixtures per test; each test runs in a **transaction that's rolled back**, so tests are isolated.
- `get_user_model()` (not `User` import) works with custom user models.
- DRF's `APIClient` + `force_authenticate` logs a user in without dealing with tokens — ideal for unit tests.
- Assert on **status codes** (`201`, `401/403`) and **side effects** (`Article.objects.count()`), the two things that matter for an endpoint.
- `User.objects.create_user` hashes the password (vs `create`, which wouldn't).

### Alternative Solution

`pytest-django` with fixtures and `factory_boy` for test data is the more scalable, industry-standard setup; the built-in `TestCase` is fine for assessments.

### Follow-up Questions

- *Why does each test start clean?* — transaction rollback per test.
- *`create` vs `create_user`?* — the latter hashes the password and sets defaults.
- *How do you test a permission (owner-only edit)?* — authenticate as a *different* user and assert 403.

---

# Part H — Rapid-Fire Coding Questions (with Interviewer Intent)

> Short, high-frequency questions. Each notes **why an interviewer asks it** — the signal they're looking for.

## Easy

**H1. Write a QuerySet for "the 5 most recent published articles."**
```python
Article.objects.filter(status="published").order_by("-created")[:5]
```
*Why asked:* checks you know filtering, ordering, and that slicing = `LIMIT` (and is lazy).

**H2. Return JSON `{"count": N}` of all users from a view (no DRF).**
```python
from django.http import JsonResponse
def user_count(request):
    return JsonResponse({"count": get_user_model().objects.count()})
```
*Why asked:* basic view + `JsonResponse` + `count()` without over-engineering.

**H3. Add a URL `/articles/<slug>/` mapping to a detail view, named.**
```python
path("articles/<slug:slug>/", views.detail, name="article-detail")
```
*Why asked:* path converters + named URLs (so you don't hardcode paths).

**H4. Give the model field for "money."**
`DecimalField(max_digits=10, decimal_places=2)` — *why asked:* floats can't represent currency exactly.

## Medium

**H5. Count articles per author in ONE query.**
```python
Author.objects.annotate(n=Count("articles")).values("user__username", "n")
```
*Why asked:* `annotate` vs `aggregate`, and doing it in the DB (not a Python loop → N+1).

**H6. "Published OR mine" in one filter.**
```python
Article.objects.filter(Q(status="published") | Q(author__user=request.user))
```
*Why asked:* `Q` objects for OR logic.

**H7. Atomically increment a counter (no race).**
```python
Article.objects.filter(pk=pk).update(views=F("views") + 1)
```
*Why asked:* `F` expressions and understanding read-modify-write races.

**H8. Custom DRF permission: only the owner may edit.**
```python
class IsOwner(BasePermission):
    def has_object_permission(self, request, view, obj):
        return request.method in SAFE_METHODS or obj.author.user == request.user
```
*Why asked:* object-level authorization and `SAFE_METHODS`.

**H9. Fix this N+1:**
```python
# before: for a in Article.objects.all(): a.author.user.username  (N+1)
Article.objects.select_related("author__user")   # after
```
*Why asked:* the single most important ORM performance skill.

## Hard

**H10. Bulk-create 10,000 articles efficiently.**
```python
Article.objects.bulk_create(
    [Article(title=f"A{i}", slug=f"a{i}", body="x", author=author) for i in range(10000)],
    batch_size=1000,
)
```
*Why asked:* one query vs 10,000 `save()`s; `batch_size` to avoid huge single statements. (Caveat: skips `save()`/signals/`auto_now`.)

**H11. Prefetch only each article's 3 latest comments.**
```python
Prefetch("comments", queryset=Comment.objects.order_by("-created")[:3], to_attr="latest")
```
*Why asked:* advanced `Prefetch` with a customised inner queryset.

**H12. Design a zero-downtime migration to add a required `slug` to a huge table.**
1. Add `slug` **nullable** (fast). 2. Deploy code that populates it. 3. Backfill existing rows in batches (data migration). 4. Add the `NOT NULL` + `unique` constraint in a later migration. *Why asked:* production-migration maturity — knowing table locks and backwards compatibility.

**H13. Two users hit "buy last item" simultaneously — prevent overselling.**
```python
from django.db import transaction
with transaction.atomic():
    item = Product.objects.select_for_update().get(pk=pk)   # row lock
    if item.stock > 0:
        item.stock = F("stock") - 1
        item.save(update_fields=["stock"])
```
*Why asked:* transactions + `select_for_update` (pessimistic locking) for concurrency correctness.

**H14. Explain and implement search + filter + ordering + cursor pagination on one endpoint.**
Combine `filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]` with a `CursorPagination` subclass (`ordering = "-created"`). *Why asked:* whether you can assemble the full "production list endpoint."

---

## Final Preparation Checklist (Practical)

Before the assessment, make sure you can, **from a blank editor**:

- [ ] Scaffold a project + app, register it, run migrations, create a superuser.
- [ ] Define models with FK/M2M/O2O, `Meta`, `__str__`, and correct `on_delete`.
- [ ] Write CRUD both as FBVs and CBVs, wired to named URLs.
- [ ] Build a registration/login/logout flow with `{% csrf_token %}` and `@login_required`.
- [ ] Write a `ModelForm`/`ModelSerializer` with `clean_`/`validate_` rules.
- [ ] Stand up a full DRF CRUD API: serializer → viewset → router → permissions → pagination → filtering.
- [ ] Add JWT auth and narrate the access/refresh lifecycle.
- [ ] Diagnose and fix an N+1 with `select_related`/`prefetch_related` and *prove* it via query count.
- [ ] Write `Q`/`F`/`annotate`/`aggregate` queries fluently in the shell.
- [ ] Wire a `post_save` signal (checking `created`) and register it in `ready()`.
- [ ] Cache an expensive computation (cache-aside) and invalidate it.
- [ ] Write a production `settings.py` (env vars, `DEBUG=False`, secure cookies, Postgres).
- [ ] Describe an EC2 deploy (Gunicorn + Nginx + systemd + RDS) end to end.
- [ ] Write `TestCase`/`APIClient` tests asserting status codes and side effects.

*Pair this with `theory.md` — practise saying the "why" out loud while you type the "how."*



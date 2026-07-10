# Web Development & API Creation Roadmap

## Focus Areas
- Django Fundamentals
- Django ORM
- Django Authentication
- Forms & Validation
- REST API Development (Django REST Framework)
- API Security
- Deployment
- AWS
- Full-Stack Best Practices

---

# Week 1: Django Fundamentals

## Introduction to Django

### Topics
- What is Django?
- MVC vs MVT Architecture
- Why Django?
- Key features of Django
- Django ecosystem

### Benefits
- Batteries Included
- Built-in Authentication
- ORM
- Security
- Scalability
- Admin Panel

---

## Setting Up a Django Project

### Installation

- Install Python
- Create a Virtual Environment
- Install Django
- Create a Django project
- Create a Django app

---

## Django Project Structure

Understand the purpose of each file.

```
project/
│
├── manage.py
├── settings.py
├── urls.py
├── asgi.py
├── wsgi.py
└── apps/
```

### Topics
- `manage.py`
- `settings.py`
- `urls.py`
- `wsgi.py`
- `asgi.py`

---

## Running the Development Server

### Commands

- Create project
- Create app
- Run server
- Apply migrations
- Create superuser

---

# Django Models

## Django ORM

### Topics
- What is an ORM?
- Benefits of ORM
- CRUD using ORM

---

## Creating Models

### Learn

- Fields
- Field types
- Relationships

### Relationships

- One-to-One
- One-to-Many
- Many-to-Many

---

## Model Features

- Model methods
- Meta class
- String representation (`__str__`)

---

## Migrations

Understand:

- makemigrations
- migrate
- Migration files
- Schema evolution

---

# Django Views

## Function-Based Views (FBVs)

- Creating FBVs
- Request object
- Response object

---

## Class-Based Views (CBVs)

- View
- TemplateView
- ListView
- DetailView
- CreateView
- UpdateView
- DeleteView

---

## Request & Response

Learn:

- GET
- POST
- Request object
- HttpResponse
- JsonResponse
- Redirects

---

# Django Templates

## Topics

- Template Language
- Variables
- Loops
- Conditions

---

## Template Features

- Template inheritance
- Base templates
- Blocks
- Includes
- Filters
- Template tags

---

# URL Routing

## Topics

- URL patterns
- Path converters
- Including app URLs
- URL names
- URL reversing
- Namespaces

---

# Practical Project

Build a Django web application including:

- Models
- Views
- Templates
- URL Routing
- Admin Panel

---

# Week 2: Authentication & Forms

## User Authentication

### Topics

- Authentication system
- User model
- Login
- Logout
- Registration

---

## Permissions

Learn:

- Authentication
- Authorization
- Groups
- Permissions

---

## Sessions

Topics:

- Session management
- Cookies
- Authentication middleware

---

## Django Forms

### Topics

- Forms
- ModelForms
- Form validation
- Formsets

---

## Validation

Learn:

- Field validation
- Custom validators
- Clean methods
- Error handling

---

## File Uploads

Topics:

- Image upload
- File upload
- Media configuration

---

# Week 3: REST API Development

## REST API Fundamentals

### Topics

- What is REST?
- REST principles
- HTTP methods

### HTTP Methods

- GET
- POST
- PUT
- PATCH
- DELETE

---

## HTTP Status Codes

Learn common status codes:

- 200
- 201
- 204
- 400
- 401
- 403
- 404
- 500

---

## Django REST Framework (DRF)

### Setup

- Install DRF
- Configure DRF

---

## Serializers

### Topics

- Serializer
- ModelSerializer
- Validation
- Nested serializers

---

## API Views

### Learn

- APIView
- Generic Views
- Mixins

---

## ViewSets

### Topics

- ModelViewSet
- ReadOnlyModelViewSet
- Routers

---

## API Routing

- Routers
- Nested routes

---

## Authentication

### Methods

- Session Authentication
- Token Authentication
- JWT Authentication

---

## Permissions

Learn:

- IsAuthenticated
- IsAdminUser
- Custom Permissions

---

## Pagination

Topics:

- PageNumberPagination
- LimitOffsetPagination
- CursorPagination

---

## Filtering

Learn:

- SearchFilter
- OrderingFilter
- DjangoFilterBackend

---

## API Documentation

### Tools

- Swagger
- OpenAPI
- drf-spectacular

---

# Week 4: Deployment & Production

## Deployment Fundamentals

### Topics

- Environment Variables
- Debug Mode
- Secret Keys
- Static Files
- Media Files

---

## Deployment to Heroku *(Legacy)*

> **Note:** Heroku no longer offers a free tier. Modern deployments are commonly done on platforms like Render, Railway, Fly.io, or AWS.

### Learn

- Procfile
- Gunicorn
- WhiteNoise
- Environment Variables

---

## Deployment on AWS

### EC2

Learn:

- Launch EC2 instance
- SSH access
- Install dependencies
- Configure Nginx
- Configure Gunicorn

---

## AWS Elastic Beanstalk

Topics:

- Environment creation
- Deployment
- Scaling
- Monitoring

---

## Database Deployment

Learn:

- PostgreSQL
- Environment configuration
- Database migrations

---

## Static & Media Files

Topics:

- WhiteNoise
- AWS S3
- CloudFront (Introduction)

---

# Advanced Django Concepts

## Middleware

Learn:

- Built-in middleware
- Custom middleware
- Request lifecycle

---

## Signals

Topics:

- pre_save
- post_save
- pre_delete
- post_delete

---

## Custom User Model

Learn:

- Extending AbstractUser
- Custom authentication

---

## Django Admin

Topics:

- Model registration
- Admin customization
- Filters
- Search
- Inlines

---

## Caching

Topics:

- Local Memory Cache
- Redis Cache
- Cache decorators

---

## Background Tasks

Introduction to:

- Celery
- Redis
- Scheduled jobs

---

## Security Best Practices

Learn:

- CSRF Protection
- XSS Prevention
- SQL Injection Prevention
- Password Hashing
- Secure Cookies
- HTTPS

---

## Performance Optimization

Topics:

- Query Optimization
- `select_related()`
- `prefetch_related()`
- Database Indexing
- Pagination
- Caching

---

# Practical Projects

## Django Web Application

Build an application with:

- User Authentication
- CRUD Operations
- Templates
- Forms
- File Uploads
- Admin Dashboard

---

## REST API Project

Build a production-ready API including:

- CRUD Endpoints
- JWT Authentication
- Permissions
- Pagination
- Filtering
- Search
- API Documentation
- PostgreSQL Database

---

## Deployment Project

Deploy the application to:

- AWS Elastic Beanstalk **or**
- EC2 with Nginx + Gunicorn

---

# Topics to Revise

## Django Fundamentals
- Project Structure
- Models
- Views
- Templates
- URL Routing

---

## ORM
- CRUD
- Relationships
- Queries
- Migrations

---

## Authentication
- Login
- Logout
- Registration
- Permissions
- Sessions

---

## Forms
- Forms
- ModelForms
- Validation
- File Uploads

---

## Django REST Framework
- Serializers
- Views
- ViewSets
- Routers
- Authentication
- Permissions
- Pagination
- Filtering

---

## Deployment
- Environment Variables
- Gunicorn
- WhiteNoise
- PostgreSQL
- AWS Deployment

---

## Advanced Django
- Middleware
- Signals
- Custom User Model
- Django Admin
- Caching
- Celery
- Security
- Performance Optimization

---

# Interview Preparation Checklist

Be able to explain:

- Django MVT architecture.
- Django ORM vs raw SQL.
- Function-Based Views vs Class-Based Views.
- `select_related()` vs `prefetch_related()`.
- Authentication vs Authorization.
- Session Authentication vs JWT Authentication.
- Django Forms vs DRF Serializers.
- APIView vs Generic Views vs ViewSets.
- PUT vs PATCH.
- Middleware request lifecycle.
- CSRF protection.
- N+1 query problem.
- Gunicorn vs Nginx.
- Static files vs Media files.
- How to deploy a Django application to AWS.

---

# Practice Goals

- Build **2–3 full-stack Django web applications** with authentication and CRUD functionality.
- Develop **2 production-ready REST APIs** using Django REST Framework, including JWT authentication, pagination, filtering, and API documentation.
- Deploy at least **one Django application** to AWS (EC2 or Elastic Beanstalk) and configure a production-ready environment with PostgreSQL, Gunicorn, and Nginx.
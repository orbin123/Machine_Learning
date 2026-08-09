# Student Course API

A Django REST Framework API for students, courses, and enrollments.

## Setup on macOS

From this `django-project` directory:

```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use `python -m pip` so package installation always targets the active Python.

## Run the project

```bash
cd student_api
python manage.py migrate
python manage.py runserver
```

Open the interactive API documentation at <http://127.0.0.1:8000/api/docs/>.

Main endpoints:

- `/api/students/`
- `/api/courses/`
- `/api/enrollments/`
- `/api/students/<student_id>/courses/`
- `/api/schema/`
- `/api/docs/`

Run the checks and tests with:

```bash
python manage.py check
python manage.py test
```

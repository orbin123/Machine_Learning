from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import (
    StudentViewSet,
    CourseViewSet,
    EnrollmentViewSet,
    StudentCourseViewSet,
    ProductViewSet
)

router = DefaultRouter()

router.register("students", StudentViewSet, basename="student")
router.register("courses", CourseViewSet, basename="course")
router.register("enrollments", EnrollmentViewSet, basename="enrollment")
router.register("products", ProductViewSet)

urlpatterns = [
    path("", include(router.urls)),
    path(
        "students/<int:student_id>/courses/",
        StudentCourseViewSet.as_view({"get": "list"}),
        name="student-courses",
    ),
]

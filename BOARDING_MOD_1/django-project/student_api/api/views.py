from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticatedOrReadOnly

from .models import Student, Course, Enrollment, Product
from .serializers import (
    StudentSerializer,
    CourseSerializer,
    EnrollmentSerializer,
    ProductSerializer,
)


class StudentViewSet(viewsets.ModelViewSet):
    queryset = Student.objects.all().order_by("id")
    serializer_class = StudentSerializer


class CourseViewSet(viewsets.ModelViewSet):
    queryset = Course.objects.all().order_by("id")
    serializer_class = CourseSerializer


class EnrollmentViewSet(viewsets.ModelViewSet):
    queryset = Enrollment.objects.select_related("student", "course")
    serializer_class = EnrollmentSerializer


class StudentCourseViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = CourseSerializer

    def get_queryset(self):
        student_id = self.kwargs["student_id"]
        return Course.objects.filter(enrollments__student_id=student_id).order_by("id")


class ProductViewSet(viewsets.ModelViewSet):
    queryset = Product.objects.all().order_by("id")
    serializer_class = ProductSerializer
    permission_classes = [IsAuthenticatedOrReadOnly]

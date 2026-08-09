from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from .models import Course, Enrollment, Student


class StudentCourseEnrollmentAPITests(APITestCase):
    def setUp(self):
        self.student = Student.objects.create(
            name="Orbin",
            email="orbin@example.com",
            country="India",
        )
        self.course = Course.objects.create(
            title="Django REST Framework",
            price="1499.00",
            duration=6,
        )

    def test_create_student(self):
        response = self.client.post(
            reverse("student-list"),
            {
                "name": "Asha",
                "email": "asha@example.com",
                "country": "India",
            },
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertTrue(Student.objects.filter(email="asha@example.com").exists())

    def test_create_enrollment(self):
        response = self.client.post(
            reverse("enrollment-list"),
            {"student": self.student.pk, "course": self.course.pk},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertTrue(
            Enrollment.objects.filter(
                student=self.student,
                course=self.course,
            ).exists()
        )

    def test_duplicate_enrollment_is_rejected(self):
        Enrollment.objects.create(student=self.student, course=self.course)

        response = self.client.post(
            reverse("enrollment-list"),
            {"student": self.student.pk, "course": self.course.pk},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_list_courses_for_student(self):
        Enrollment.objects.create(student=self.student, course=self.course)

        response = self.client.get(
            reverse("student-courses", kwargs={"student_id": self.student.pk})
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data[0]["title"], self.course.title)

    def test_schema_and_docs_are_available(self):
        schema_response = self.client.get(reverse("schema"))
        docs_response = self.client.get(reverse("swagger-ui"))

        self.assertEqual(schema_response.status_code, status.HTTP_200_OK)
        self.assertEqual(docs_response.status_code, status.HTTP_200_OK)

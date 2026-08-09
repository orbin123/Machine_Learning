from decimal import Decimal

from django.db import models
from django.core.validators import MinValueValidator


class Student(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    country = models.CharField(max_length=50)

    def __str__(self):
        return self.name


class Course(models.Model):
    title = models.CharField(max_length=100, unique=True)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    duration = models.PositiveIntegerField(help_text="Duration in weeks")

    def __str__(self):
        return self.title


class Enrollment(models.Model):
    student = models.ForeignKey(
        Student,
        on_delete=models.CASCADE,
        related_name="enrollments"
    )
    course = models.ForeignKey(
        Course,
        on_delete=models.CASCADE,
        related_name="enrollments"
    )
    enrolled_on = models.DateField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["student", "course"],
                name="unique_student_course",
            )
        ]
        ordering = ["-enrolled_on", "id"]

    def __str__(self):
        return f"{self.student} - {self.course}"


class Product(models.Model):
    name = models.CharField(max_length=100)
    category = models.CharField(max_length=100)
    price = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        validators=[MinValueValidator(Decimal("0.01"))],
    )
    stock = models.PositiveBigIntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

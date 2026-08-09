from rest_framework import serializers

from .models import Student, Course, Enrollment, Product


class StudentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Student 
        fields = "__all__"


class CourseSerializer(serializers.ModelSerializer):
    class Meta:
        model = Course 
        fields = "__all__"


class EnrollmentSerializer(serializers.ModelSerializer):
    class Meta: 
        model = Enrollment 
        fields = [
            "id",
            "student",
            "course", 
            "enrolled_on"
        ]
        read_only_fields = ["id", "enrolled_on"]

    def validate(self, data):
        student = data.get("student", getattr(self.instance, "student", None))
        course = data.get("course", getattr(self.instance, "course", None))

        matches = Enrollment.objects.filter(student=student, course=course)
        if self.instance is not None:
            matches = matches.exclude(pk=self.instance.pk)

        if matches.exists():
            raise serializers.ValidationError(
                "This student is already enrolled in this course."
            )

        return data


class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = "__all__"

    def validate_price(self, value):
        if value <= 0:
            raise serializers.ValidationError(
                "Price must be greater than zero."
            )
        return value

    def validate_stock(self, value):
        if value < 0:
            raise serializers.ValidationError(
                "Stock cannot be negative."
            )
        return value

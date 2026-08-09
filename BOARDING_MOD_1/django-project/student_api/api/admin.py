from django.contrib import admin

from .models import Course, Enrollment, Product, Student


admin.site.register([Student, Course, Enrollment, Product])

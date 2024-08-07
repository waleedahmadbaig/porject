from django.contrib import admin
from .models import BrainScans, Contact, NewDoctor
# from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
# from .models import User



admin.site.register(BrainScans)
admin.site.register(Contact)
admin.site.register(NewDoctor)


# class UserAdmin(BaseUserAdmin):
#     def save_model(self, request, obj, form, change):
#         if not change:  # If it's a newly created object 
#             obj.password_changed = False
#         super().save_model(request, obj, form, change)

# admin.site.register(User, UserAdmin)

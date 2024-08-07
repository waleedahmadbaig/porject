# middleware.py
from django.shortcuts import redirect
from django.urls import reverse_lazy
from django.contrib.auth.models import User  # Import User mode



# middleware.py

def first_login_check_middleware(get_response):
    def middleware(request):
        if request.user.is_authenticated and request.user.user:
            if not request.user.password_changed and not User.objects.filter(username=request.user.username).exclude(pk=request.user.pk).exists():
                # Only redirect to password change if password hasn't been changed AND no other user with same username exists
                return redirect(reverse_lazy('force_password_change'))
            else:
                return redirect(reverse_lazy('dashboard')) 
        return get_response(request)
    return middleware

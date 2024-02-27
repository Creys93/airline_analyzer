from django.shortcuts import redirect, render
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required

def user_login(request):
    """
    Handles user login.

    If the request method is POST, it validates the login form data.
    If the form is valid, it logs in the user and redirects to the home page.

    If the request method is GET, it renders the login form.

    Args:
    - request: The HTTP request object.

    Returns:
    - If the request method is POST and form is valid, redirects to 'home'.
    - If the request method is GET, renders 'login.html' template.
    """
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('home')  # Redirect to the home page after login
        else:
            return render(request, 'login.html', {'form': form, 'error': 'Invalid username or password. Please try again'})

    else:
        form = AuthenticationForm()
        return render(request, 'login.html', {'form': form})

def user_signup(request):
    """
    Handles user signup.

    If the request method is POST, it validates the signup form data.
    If the form is valid, it creates a new user, logs in the user, and redirects to the home page.

    If the request method is GET, it renders the signup form.

    Args:
    - request: The HTTP request object.

    Returns:
    - If the request method is POST and form is valid, redirects to 'home'.
    - If the request method is GET, renders 'signup.html' template.
    """
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('home')  # Redirect to the home page after signup
    else:
        form = UserCreationForm()
        return render(request, 'signup.html', {'form': form})

@login_required()
def user_logout(request):
    """
    Handles user logout.

    Logs out the current user and redirects to the home page.

    Args:
    - request: The HTTP request object.

    Returns:
    - Redirects to 'home'.
    """
    logout(request)
    return redirect('home')

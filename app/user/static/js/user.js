$(document).ready(function () {
    // Select all input elements within elements with the class 'user-container'
    var inputs = document.querySelectorAll('.user-container input');
    
    // Add the 'form-control' class to each selected input element
    inputs.forEach(function(input) {
        input.classList.add('form-control');
    });
});

// Function to retrieve the CSRF token from cookies
function getCSRFToken() {
  var csrfCookie = document.cookie.match(/csrftoken=([^ ;]+)/);
  return csrfCookie ? csrfCookie[1] : null;
}

// Execute code when the document is ready
$(document).ready(function () {
  // Handle file input change event to update the file label
  $(".custom-file-input").on("change", function () {
      var fileName = $(this).val().split("\\").pop();
      $(this).siblings(".custom-file-label").addClass("selected").html(fileName);
  });

  // Handle file uploads and validations calling a view
  $('#customFile').on('change', function () {
      $('.spinner-border').removeClass('d-none');
      var formData = new FormData();
      formData.append('filename', this.files[0]);

      $.ajax({
          url: '/validate/',
          type: 'POST',
          data: formData,
          processData: false,
          contentType: false,
          headers: {
              'X-CSRFToken': getCSRFToken()
          },
          success: function () {
              $(".custom-file-input").removeClass("error");
              $('.spinner-border').addClass('d-none');
              $('.alert-danger').addClass('d-none');
              $("#submit").removeAttr("disabled");
          },
          error: function () {
              $(".custom-file-input").addClass("error");
              $("#submit").prop("disabled", true);
              $('.spinner-border').addClass('d-none');
              $('.alert-danger').removeClass('d-none');
          }
      });
  });

  // Handle submit button click
  $('#submit').click(function () {
      $('.spinner-border').removeClass('d-none');
      $('.alert-info').removeClass('d-none');
  });

    $('.sorter').click(function (event) {
        // Prevent the default behavior of the anchor tag
        event.preventDefault();

        // Construct the updated URL with the added parameters
        appendCurrentParamsToUrl($(this).attr('href'));
    });

    $('.page-link').click(function (event) {
        // Prevent the default behavior of the anchor tag
        event.preventDefault();

        // Construct the updated URL with the added parameters
        appendCurrentParamsToUrl($(this).attr('href'));

    })

  // Set parameters from URL if filter element exists
  if (document.getElementById('filter')) {
      setParametersFromUrl();

      document.getElementById('filter-button').addEventListener('click', function () {
          applyFilter();
      });

      document.getElementById('filter').addEventListener('keyup', function (event) {
          if (event.key === 'Enter') {
              applyFilter();
          }
      });
  }

  // Handle reset button click if reset-button element exists
  if (document.getElementById('reset-button')) {
      document.getElementById('reset-button').addEventListener('click', function () {
          window.location.href = window.location.href.split('?')[0];
      });
  }
});

// Function to set parameters from URL
function setParametersFromUrl() {
  var urlParams = new URLSearchParams(window.location.search);
  var search_query = urlParams.get('search_query');
  if (search_query !== null) {
      document.getElementById('filter').value = decodeURIComponent(search_query);
  }

  // Set additional parameters from URL if corresponding elements exist
  if (document.getElementById('from')) {
      var from = urlParams.get('from');
      if (from !== null) {
          document.getElementById('from').value = decodeURIComponent(from);
      }
  }

  if (document.getElementById('extra-bag')) {
      var extraBagValue = urlParams.get('extra-bag');
      if (extraBagValue) {
          document.getElementById('extra-bag').checked = decodeURIComponent(extraBagValue);
      }
  }

  if (document.getElementById('pref-seat')) {
      var prefSeatValue = urlParams.get('pref-seat');
      if (prefSeatValue) {
          document.getElementById('pref-seat').checked = decodeURIComponent(prefSeatValue);
      }
  }

  if (document.getElementById('meals')) {
      var mealsValue = urlParams.get('meals');
      if (mealsValue) {
          document.getElementById('meals').checked = decodeURIComponent(mealsValue);
      }
  }

  if (document.getElementById('complete')) {
      var completeValue = urlParams.get('complete');
      if (completeValue) {
          document.getElementById('complete').checked = decodeURIComponent(completeValue);
      }
  }
}

// Function to apply filter based on input values
function applyFilter() {
  var filterValue = document.getElementById('filter').value;
  var currentUrl = window.location.href;
  var urlSearchParams = new URLSearchParams(window.location.search);
  var newFilterValue = encodeURIComponent(filterValue);

  urlSearchParams.delete('search_query');

  if (filterValue) {
      urlSearchParams.append('search_query', newFilterValue);
  }

  // Add additional parameters to URL search params if corresponding elements exist
  if (document.getElementById('from')) {
      var fromValue = document.getElementById('from').value;
      var newFromValue = encodeURIComponent(fromValue);
      urlSearchParams.delete('from');
      if (newFromValue) {
          urlSearchParams.append('from', newFromValue);
      }
  }

  if (document.getElementById('extra-bag')) {
      var extraBagValue = document.getElementById('extra-bag').checked;
      urlSearchParams.delete('extra-bag');
      if (extraBagValue) {
          var newExtraBagValue = extraBagValue ? 'true' : 'false';
          urlSearchParams.append('extra-bag', newExtraBagValue);
      }
  }

  if (document.getElementById('pref-seat')) {
      var prefSeatValue = document.getElementById('pref-seat').checked;
      urlSearchParams.delete('pref-seat');
      if (prefSeatValue) {
          var newPrefSeatValue = prefSeatValue ? 'true' : 'false';
          urlSearchParams.append('pref-seat', newPrefSeatValue);
      }
  }

  if (document.getElementById('meals')) {
      var mealsValue = document.getElementById('meals').checked;
      urlSearchParams.delete('meals');
      if (mealsValue) {
          var newMealsValue = mealsValue ? 'true' : 'false';
          urlSearchParams.append('meals', newMealsValue);
      }
  }

  if (document.getElementById('complete')) {
      var completeValue = document.getElementById('complete').checked;
      urlSearchParams.delete('complete');
      if (completeValue) {
          var newCompleteValue = completeValue ? 'true' : 'false';
          urlSearchParams.append('complete', newCompleteValue);
      }
  }

  var newUrl = currentUrl.split('?')[0] + (urlSearchParams.size > 0 ? '?' : '') + urlSearchParams.toString();
  window.location.href = newUrl;
}

function addParamsSorterPage(){
    var url = $(this).attr('href');
        
    var currentParams = new URLSearchParams(window.location.search);

    console.log(currentParams)

    var linkParams = new URLSearchParams($(this).attr('href').split('?')[1]);

    currentParams.forEach(function(value, key) {
        if (!linkParams.has(key)) {
            linkParams.append(key, value);
        }
    });

    var updatedUrl = url.split('?')[0] + '?' + linkParams.toString();

    window.location.href = updatedUrl;
}


function appendCurrentParamsToUrl(url) {
    // Extract the current parameters from the URL
    var currentParams = new URLSearchParams(window.location.search);

    // Extract the parameters from the provided URL
    var urlParams = new URLSearchParams(url.split('?')[1]);

    // Add the current parameters to the end of the provided URL's parameters
    currentParams.forEach(function(value, key) {
        if (!urlParams.has(key)) {
            urlParams.append(key, value);
        }
    });

    // Construct the updated URL with the added parameters
    var updatedUrl = url.split('?')[0] + '?' + urlParams.toString();

    window.location.href = updatedUrl;

    updatedUrl;
}
# airline_analyzer

# Master's Thesis Project
The main objective of this project is to develop a system that allows for accurately estimating flight reservations and detecting business opportunities in the airline sector. The specific objectives are:

Implement the XGBoost regression model for precise estimation of the number of reservations.
Utilize the K-Means algorithm to identify patterns and opportunities in reservation data.
Apply the SVM classifier for detection and classification of opportunities.
To achieve these objectives, a project will be developed using the Django framework in Python, which will consist of two main applications: "user" and "core". The "user" application will be responsible for managing user authentication and login, while the "core" application will handle all functionalities related to reservation estimation and opportunity detection in the airline sector.

In the "core" application, functionality will be implemented for users to upload CSV files with flight reservation data. Each uploaded CSV file will generate a new prediction in the system. These predictions will be processed by the XGBoost regression models and K-Means clustering algorithms to estimate the number of reservations and detect business opportunities, respectively.

Each user will have access to a personalized list of their predictions, where they can view the details of each one, including the results obtained from both the XGBoost model and the K-Means algorithm. Additionally, functionality will be provided for users to upload new CSV files, make new predictions, and query the results.

The Django project will follow best practices in web development and security, ensuring a robust, scalable, and easy-to-maintain system. The application will be designed in a modular and extensible manner, allowing for the incorporation of new functionalities and improvements in the future. This will ensure that the system can adapt to the changing needs of the airline industry and provide a high-quality service to end users.

# Running the Application
To run the app, you'll need to execute the following commands:

Download dependencies from requirements.txt
Generate the database: python manage.py migrate
Start the server: python manage.py runserver
Detect changes in models.py of the applications (in this project there are models only in core): python manage.py makemigrations
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os
import re
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from app.core.models import Prediction, PredictionResult, SVMResult
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import silhouette_score, accuracy_score, confusion_matrix, classification_report


def contain_special_characters(value):
    """
    Check if a string contains special characters.

    Args:
    - value: The string to check.

    Returns:
    - bool: True if the string contains special characters, False otherwise.
    """
    if isinstance(value, str):
        return bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', value))
    
    return False

def preprocessing(file, user):
    """
    Preprocesses the input file by removing rows with special characters.

    Args:
    - file: The input file.
    - user: The user associated with the user prediction.
    """

    df = pd.read_csv(file)

    row_special_characters = df.map(contain_special_characters)

    mask_filter = row_special_characters.any(axis=1)

    process_predictions(df[~mask_filter], file, user)


def process_predictions(df, file, user):
    """
    Processes the prediction by saving the Prediction and PredictionResult models, in addition to preparing the other necessary data.

    Args:
    - df: The DataFrame containing the predictions data.
    - file: The input file.
    - user: The user associated with the predictions.
    """

    random_seed = 42

    # Data split
    X = df[df.columns[:-1]]
    y = df['num_passengers']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define preprocessing using ColumnTransformer and Pipeline
    num_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, X_train.select_dtypes(include=['int64', 'float64']).columns),
            ('cat', cat_pipeline, X_train.select_dtypes(include=['object', 'category']).columns)
        ])

    # Fit the preprocessor to the training data
    preprocessor.fit(X_train)

    # Define the XGBoost pipeline
    xgb_pipeline = Pipeline([ 
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(random_state=random_seed))
    ])

    # Hyperparameters for XGBoost
    xgb_param_dist = {
        'regressor__n_estimators': [2000, 2100, 2200, 2300],
        'regressor__learning_rate': [0.05, 0.1, 0.2],
        'regressor__max_depth': [20, 25, 30],
        'regressor__min_child_weight': [1, 2, 3],
        'regressor__gamma': [0.05, 0.1, 0.15],
        'regressor__subsample': [0.7, 0.8, 0.9],
        'regressor__colsample_bytree': [0.7, 0.8, 0.9],
        'regressor__lambda': [15, 20, 25, 30, 35],
        'regressor__alpha': [0.05, 0.1, 0.15, 0.2]
    }

    print(f"Init process with XGBoost Model")
    # K-Fold Cross Validation
    kfold_cv = KFold(n_splits=10, shuffle=True, random_state=random_seed)
    
    # RandomizedSearchCV with K-Fold Cross Validation
    random_search = RandomizedSearchCV(xgb_pipeline, param_distributions=xgb_param_dist, n_iter=50, cv=kfold_cv, scoring='neg_mean_squared_error', n_jobs=-1, random_state=random_seed)
    
    # Train the model with the entire training set
    random_search.fit(X_train, y_train)

    # Best model
    best_model = random_search.best_estimator_

    # Save the trained model
    joblib.dump(best_model, 'xgboost_model.pkl')
        
    # Load the trained model
    loaded_model = joblib.load('xgboost_model.pkl')

    # Make predictions on the test set
    test_predictions = loaded_model.predict(X_test)

    # Show additional metrics
    mse = mean_squared_error(y_test, test_predictions)
    mae = mean_absolute_error(y_test, test_predictions)
    r2 = r2_score(y_test, test_predictions)

    # Make predictions with the loaded model
    predictions = loaded_model.predict(df)

    print('Predictions obtained, preparing inserts..')

    # Create Prediction object and save it to the database
    prediction = Prediction.objects.create(
        user=user,
        file_name=file.name,
        mse=round(mse,6),
        mae=round(mae,6),
        r2=round(r2,6)
    )

    prediction.save()

    prediction.inertia_path = f'inertia_{prediction.id}.png'
    prediction.silhouette_path = f'silhouette{prediction.id}.png'

    prediction.save()

    # Create PredictionResult objects and bulk insert them into the database
    prediction_results = []
    for i, row in enumerate(df.iterrows()):
        row_data = row[1].to_dict()
        prediction_result = PredictionResult(
            prediction=prediction,
            index=i+1,
            num_passengers=row_data['num_passengers'],
            sales_channel=row_data['sales_channel'],
            trip_type=row_data['trip_type'],
            purchase_lead=row_data['purchase_lead'],
            length_of_stay=row_data['length_of_stay'],
            flight_hour=row_data['flight_hour'],
            flight_day=row_data['flight_day'],
            route=row_data['route'],
            booking_origin=row_data['booking_origin'],
            wants_extra_baggage=row_data['wants_extra_baggage'],
            wants_preferred_seat=row_data['wants_preferred_seat'],
            wants_in_flight_meals=row_data['wants_in_flight_meals'],
            flight_duration=row_data['flight_duration'],
            booking_complete=row_data['booking_complete'],
            prediction_value=round(float(predictions[i]),6)
        )
        prediction_results.append(prediction_result)
    
    PredictionResult.objects.bulk_create(prediction_results)

    print('Process predictions finished')

    # Perform cluster analysis
    cluster_analysis(prediction, df)

def cluster_analysis(prediction, df):
    """
    Performs cluster analysis.

    Args:
    - prediction: The prediction object associated with the analysis.
    - df: The DataFrame containing the data for analysis.
    """
    # Create a OneHotEncoder object
    onehot_encoder = OneHotEncoder()
    
    # Get the columns containing categorical variables
    categorical_columns = ['flight_day']

    # Encode the categorical variables using one-hot encoding
    onehot_encoded = onehot_encoder.fit_transform(df[categorical_columns])

    # Convert the encoded output to a pandas DataFrame
    onehot_encoded_df = pd.DataFrame(onehot_encoded.toarray(), columns=onehot_encoder.get_feature_names_out(categorical_columns))

    # Remove the original columns and concatenate the encoded ones
    data_encoded = pd.concat([df.drop(columns=categorical_columns), onehot_encoded_df], axis=1)

    data_encoded = data_encoded.dropna()

    # Select features for cluster analysis
    features = ['length_of_stay', 'flight_hour', 'flight_duration'] + list(onehot_encoded_df.columns)

    # Normalize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_encoded[features])

    # Determine the optimal number of clusters using the elbow method
    inertia_list = []
    silhouette_list = []
    for n_clusters in range(3, 15):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(scaled_data)
        inertia_list.append(kmeans.inertia_)
        cluster_labels = kmeans.fit_predict(scaled_data)
        silhouette_avg = silhouette_score(scaled_data, cluster_labels)
        silhouette_list.append(silhouette_avg)

    # Save plots for inertia and silhouette scores
    inertia_path = os.path.join(os.getcwd(),'static', f'inertia_{prediction.id}.png')
    plt.figure()
    plt.plot(range(3, 15), inertia_list, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.savefig(inertia_path)

    silhouette_path = os.path.join(os.getcwd(),'static', f'silhouette{prediction.id}.png')
    plt.figure()
    plt.plot(range(3, 15), silhouette_list, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Average Silhouette Score')
    plt.title('Silhouette Score')
    plt.savefig(silhouette_path)

    print('Cluster analysis finished')

    classification(scaled_data, cluster_labels, prediction)


def classification(scaled_data, cluster_labels, prediction):
    """
    Performs classification using Support Vector Machine (SVM).

    Args:
    - scaled_data: The scaled feature data.
    - cluster_labels: The cluster labels.
    - prediction: The prediction object associated with the classification.

    Returns:
    None
    """

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(scaled_data, cluster_labels, test_size=0.3, random_state=42)

    # Initialize the SVM classifier
    svm_classifier = SVC(kernel='linear', random_state=42)

    # Fit the SVM classifier to the training data
    svm_classifier.fit(X_train, y_train)

    # Predict cluster labels for the test data using SVM
    svm_predictions = svm_classifier.predict(X_test)

    # Calculate SVM accuracy
    svm_accuracy = accuracy_score(y_test, svm_predictions)

    # Evaluate SVM classifier
    svm_conf_matrix = confusion_matrix(y_test, svm_predictions)
    svm_classification_report = classification_report(y_test, svm_predictions)

    # Create the heatmap using Seaborn
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(svm_conf_matrix, annot=True, cmap='coolwarm', fmt='d')

    # Save the plot to a BytesIO file
    buf = BytesIO()
    heatmap.figure.savefig(buf, format='png')
    buf.seek(0)

    # Convert the image to base64 to display it in the template
    heatmap_base64 = base64.b64encode(buf.read()).decode('utf-8')

    # Perform cross-validation for the SVM classifier
    svm_cv_scores = cross_val_score(svm_classifier, scaled_data, cluster_labels, cv=5)

    # Save the SVM classification result to the database
    svmResult = SVMResult(
        prediction=prediction,
        accuracy=svm_accuracy,
        confusion_matrix_base64=heatmap_base64,
        classification_report=svm_classification_report,
        cv_mean_accuracy=svm_cv_scores.mean()
    )

    svmResult.save()

    print('Classification finished')


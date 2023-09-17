import streamlit as st
import pandas as pd
import numpy as np

# Import necessary packages for machine learning models
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Import necessary packages for feature engineering methods
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, OneHotEncoder, LabelEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Import necessary packages for evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, silhouette_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title = "AnalytiXpress", page_icon = "./assets/logo.png")

st.markdown("# Machine Learning")
st.sidebar.header("Machine Learning")


# Define the functions for each section
def selectCols(df):
    st.header("Select Useful Columns")
    columns = st.multiselect("Select Columns", df.columns)
    df = df[columns]
    return df

def transformDF(selectedMethod):
    match selectedMethod:
        case "Imputation":
            st.sidebar.write("### Imputation Settings")
            strategy = st.sidebar.selectbox("Select imputation strategy", ["mean", "median", "most_frequent", "constant"])
            imputer = SimpleImputer(strategy = strategy)
            imputeCol = st.sidebar.multiselect("Select columns to impute", df.columns)
            df[imputeCol] = imputer.fit_transform(df[imputeCol])

        case "Scaling":
            st.sidebar.write("### Scaling Settings")
            scalerOption = st.sidebar.selectbox("Choose a scaler", ["Standard Scaler", "MinMax Scaler", "Normalizer"])
            if scalerOption == "Standard Scaler":
                scaler = StandardScaler()
            elif scalerOption == "MinMax Scaler":
                scaler = MinMaxScaler()
            elif scalerOption == "Normalize":
                norm = st.sidebar.selectbox("Select norm value", ["l2", "l1", "max"])
                scaler = Normalizer(norm = 'l2', axis = 0)
            scaleCol = st.sidebar.multiselect("Select columns to scale", df.columns)
            df[scaleCol] = scaler.fit_transform(df[scaleCol])

        case "One-Hot Encoding":
            st.sidebar.write("### One-Hot Encoding Settings")
            encoder = OneHotEncoder()
            encodeCol = st.sidebar.selectbox("Select column for one-hot encoding", df.columns)
            encodedData = encoder.fit_transform(df[[encodeCol]])
            df = pd.concat([df, pd.DataFrame(encodedData.toarray(), columns = encoder.get_feature_names_out([encodeCol]))], axis = 1)
            df.drop(encodeCol, axis = 1, inplace = True)

        case "Label Encoding":
            st.sidebar.write("### Label Encoding Settings")
            le = LabelEncoder()
            lencodeCol = st.sidebar.selectbox("Select column for label encoding", df.columns)
            df[lencodeCol] = le.fit_transform(df[lencodeCol])

        case "Polynomial Features":
            st.sidebar.write("### Polynomial Features Settings")
            degree = st.sidebar.slider("Select degree for polynomial features", 2, 5)
            poly = PolynomialFeatures(degree = degree)
            transformCol = st.multiselect("Select columns for polynomial transformation", df.columns)
            transformedData = poly.fit_transform(df[transformCol])
            df = pd.concat([df, pd.DataFrame(transformedData, columns = poly.get_feature_names_out(transformCol))], axis = 1)
            df.drop(transformCol, axis = 1, inplace = True)

        case "Feature Selection":
            st.sidebar.write("### Feature Selection")
            selectionMethod = st.selectbox("Choose a feature selection method", ["Correlation Matrix", "Select K-Best"])
            if selectionMethod == "Correlation Matrix":
                threshold = st.sidebar.slider("Select correlation threshold", 0.0, 1.0, 0.8)
                # Get the numberic columns
                dfDTypes = df.dtypes
                numColumns = []
                for col in df.columns:
                    if dfDTypes[col] in ["float64", "int64"]:
                        numColumns.append(col)
                corrMatrix = df[numColumns].corr().abs()
                upper = corrMatrix.where(pd.np.triu(pd.np.ones(corrMatrix.shape), k = 1).astype(bool))
                toDrop = [column for column in upper.columns if any(upper[column] > threshold)]
                df.drop(toDrop, axis = 1, inplace = True)
            
            elif selectionMethod == "Select K-Best":
                targetCol = st.sidebar.selectbox("Select target column", df.columns)
                k = st.slider("Select top K features", 1, len(df.columns) - 1, 5)
                selector = SelectKBest(f_classif, k = k)
                XNew = selector.fit_transform(df.drop(columns = [targetCol]), df[targetCol])
                df.drop(columns = [targetCol], inplace = True)
                df = df.columns[selector.get_support()]

        case "Feature Extraction":
            st.sidebar.write("### Feature Extraction Settings")
            nComponents = st.sidebar.slider("Select number of components for PCA", 1, min(df.shape[1], 10))
            pca = PCA(n_components = nComponents)
            transformedData = pca.fit_transform(df)
            df = pd.DataFrame(transformedData, columns=[f"PC{i+1}" for i in range(nComponents)])
            
        case "Time-Based Features":
            st.sidebar.write("### Time-Based Features Settings")
            # Assuming the time column is in datetime format
            timeCol = st.sidebar.selectbox("Select the time column", df.select_dtypes(include = ['datetime64']).columns)
            timeFeatures = st.sidebar.multiselect("Select time-based features", ["Year", "Month", "Day", "Hour", "Minute", "Second", "Day of Week", "Is Weekend"])
            
            if "Year" in timeFeatures:
                df[f"{timeCol}Year"] = df[timeCol].dt.year
            if "Month" in timeFeatures:
                df[f"{timeCol}Month"] = df[timeCol].dt.month
            if "Day" in timeFeatures:
                df[f"{timeCol}Day"] = df[timeCol].dt.day
            if "Hour" in timeFeatures:
                df[f"{timeCol}Hour"] = df[timeCol].dt.hour
            if "Minute" in timeFeatures:
                df[f"{timeCol}Minute"] = df[timeCol].dt.minute
            if "Second" in timeFeatures:
                df[f"{timeCol}Second"] = df[timeCol].dt.second
            if "Day of Week" in timeFeatures:
                df[f"{timeCol}Dayofweek"] = df[timeCol].dt.dayofweek
            if "Is Weekend" in timeFeatures:
                df[f"{timeCol}IsWeekend"] = df[timeCol].dt.weekday.apply(lambda x: 1 if x >= 5 else 0)

        case "Text Processing":
            st.sidebar.write("### Text Processing Settings")
            textCol = st.sidebar.selectbox("Select the text column", df.select_dtypes(include = ['object']).columns)
            maxN = st.sidebar.slider("Max number of features for TF-IDF", 10, 500, 100)
            vectorizer = TfidfVectorizer(max_features = maxN)
            TFIDFFeatures = vectorizer.fit_transform(df[textCol])
            TfidfDf = pd.DataFrame(TFIDFFeatures.toarray(), columns = vectorizer.get_feature_names_out())
            df = pd.concat([df, TfidfDf], axis = 1)
            df.drop(textCol, axis = 1, inplace = True)

        case "Feature Crosses":
            st.sidebar.write("### Feature Crosses Settings")
            crossCols = st.sidebar.multiselect("Select two columns for feature crossing", df.columns, default = df.columns[:2])
            if len(crossCols) == 2:
                df[f"{crossCols[0]}_x_{crossCols[1]}"] = df[crossCols[0]].astype(str) + "_" + df[crossCols[1]].astype(str)
            else:
                st.warning("Please select exactly two columns for feature crossing.")

        case _:
            st.write(f"No settings available for {selectedMethod}")
    return df

def FeatureEngineering(df):
    st.header("Feature Engineering")
    featureEngineeringOptions = ["Imputation", "Scaling", "One-Hot Encoding", "Label Encoding", "Binning/Discretization", 
                                 "Feature Scaling", "Polynomial Features", "Feature Selection", 
                                 "Feature Extraction", "Time-Based Features", "Text Processing", "Feature Crosses"]
    selectedMethod = st.selectbox("Select Feature Engineering Methods", featureEngineeringOptions)
    
    try:
        df = transformDF(selectedMethod)
    except Exception as e:
        st.warning(f"An error occurred: {e}")
        st.warning("Please adjust the feature engineering configurations.")
    return df

def Models(algorithmType, selectedAlgs, X_train, X_test, y_train, y_test):
    # Initialize the result dataframe
    resultDF = pd.DataFrame()
    # Fit the model
    for alg in selectedAlgs:
        # Progress bar for model fitting
        progressBar = st.progress(0)
        with st.spinner(f"Training the {alg} Model..."):
            try: 
                match alg:
                    case "Logistic Regression":
                        model = LogisticRegression()
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)

                    case "Decision Trees":
                        model = DecisionTreeClassifier()
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                
                    case "Random Forest":
                        model = RandomForestClassifier()
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                
                    case "Support Vector Machines (SVM)":
                        model = SVC()
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                
                    case "K-Nearest Neighbors (KNN)":
                        neighbors = st.sidebar.number_input("Number of neighbors")
                        model = KNeighborsClassifier(n_neighbors = neighbors)
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        
                    case "Naive Bayes":
                        model = GaussianNB()
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                    
                    case "AdaBoost Classifier":
                        estimators = st.sidebar.slider("Number of Estimators", 10, 200, 50)
                        learningRate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1)
                        model = AdaBoostClassifier(n_estimators = estimators, learning_rate = learningRate)
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                    
                    case "Gradient Boosting Classifier":
                        estimators = st.sidebar.slider("Number of Estimators", 10, 200, 100)
                        learningRate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1)
                        model = GradientBoostingClassifier(n_estimators = estimators, learning_rate = learningRate)
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                    
                    case "XGBoost Classifier":
                        model = XGBClassifier()
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        
                    case "LightGBM Classifier":
                        model = LGBMClassifier()
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                    
                    # Regression
                    case "Linear Regression":
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        
                    case "Ridge Regression":
                        model = Ridge()
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        
                    case "Lasso Regression":
                        alpha = st.sidebar.slider("Alpha Value", 0.01, 1.0, 0.1)
                        model = Lasso(alpha = alpha)
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        
                    case "Decision Tree Regressor":
                        model = DecisionTreeRegressor()
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)

                    case "Random Forest Regressor":
                        model = RandomForestRegressor()
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)

                    case "Support Vector Regression (SVR)":
                        model = SVR()
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        
                    case "K-Nearest Neighbors Regressor (KNN)":
                        neighbors = st.sidebar.slider("Number of Neighbors", 1, 50, 5)
                        model = KNeighborsRegressor(n_neighbors = neighbors)
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)

                    case "AdaBoost Regressor":
                        estimators = st.sidebar.slider("Number of Estimators", 10, 200, 50)
                        learningRate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1)
                        model = AdaBoostRegressor(n_estimators = estimators, learning_rate = learningRate)
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                    
                    case "Gradient Boosting Regressor":
                        estimators = st.sidebar.slider("Number of Estimators", 10, 200, 100)
                        learningRate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1)
                        model = GradientBoostingRegressor(n_estimators = estimators, learning_rate = learningRate)
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)

                    case "XGBoost Regressor":
                        model = XGBRegressor()
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        
                    case "LightGBM Regressor":
                        model = LGBMRegressor()
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        
                    # Clustering
                    case "K-Means Clustering":
                        clusters = st.sidebar.number_input("Number of Clusters")
                        model = KMeans(n_clusters = clusters)
                        model.fit(X_train)
                        labels = model.predict(X_test)
                        
                    case "Hierarchical Clustering":
                        clusters = st.slider("Number of Clusters", 2, 10, 3)
                        model = AgglomerativeClustering(n_clusters = clusters)
                        labels = model.fit_predict(X_train)
                        
                    case "DBSCAN":
                        eps = st.slider("Epsilon Value", 0.1, 5.0, 0.5)
                        min_samples = st.slider("Minimum Samples", 1, 10, 5)
                        model = DBSCAN(eps = eps, min_samples = min_samples)
                        labels = model.fit_predict(X_train)
                        
                    # Dimensionality Reduction
                    case "Principal Component Analysis (PCA)":
                        components = st.sidebar.number_input("Number of components")
                        model = PCA(n_components = components)
                        transformedData = model.fit_transform(X_train)
                        st.write("PCA Result: \n", transformedData)

                    # Time Series
                    case "ARIMA":
                        model = ARIMA(y_train, order = (5,1,0))
                        model_fit = model.fit(disp = 0)
                        predictions = model_fit.forecast(steps = len(y_test))[0]
                        st.write("ARIMA Prediction: ", predictions)
                        
                    case "Long Short-Term Memory (LSTM)":
                        X_train_reshaped = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))
                        X_test_reshaped = np.reshape(X_test.values, (X_test.shape[0], X_test.shape[1], 1))

                        model = Sequential()
                        model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train_reshaped.shape[1], 1)))
                        model.add(LSTM(units = 50))
                        model.add(Dense(units = 1))
                        
                        model.compile(optimizer = 'adam', loss = 'mean_squared_error')
                        model.fit(X_train_reshaped, y_train, epochs = 25, batch_size = 32)
                        
                        predictions = model.predict(X_test_reshaped)
                        
                        st.write("LSTM Prediction: ")
                        st.line_chart(predictions)

                    case _:
                        st.write(f"No settings available for {alg}")
                        
            except Exception as e:
                st.warning(f"An error occurred: {e}")
                st.warning("Please adjust the model configurations.")
            
            progressBar.progress(100)
            
        # Create the result dataframe content
        if algorithmType == "Classification" or algorithmType == "Regression":
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions)
            recall = recall_score(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            
            # Append the dictionary to the dataframe
            resultData = {
                'Model Name': alg,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'Mean Squared Error': mse,
                'Mean Absolute Error': mae,
                'R2 Score': r2,
                'F1 Score': f1
            }
            resultDF = resultDF.append(resultData, ignore_index = True)
            
            if algorithmType == "Classification":
                # Calculate the confusion matrix
                cm = confusion_matrix(y_test, predictions)
                st.write(f"Confusion Matrix for {alg} model: ", cm)
                # Calculate the classification report
                report = classification_report(y_test, predictions)
                st.write(f"Classification Report for {alg} model: ", report)

        elif algorithmType == "Clustering":
            silhouette = silhouette_score(X_train, labels)
            # Append the dictionary to the dataframe
            resultData = {
                'Model Name': alg,
                'Silhouette Score': silhouette,
            }
            resultDF = resultDF.append(resultData, ignore_index = True)

    return resultDF

def FitModels(df):
    st.header("Fit Model")
    # Select algorithm type
    mlTypes = ["Classification", "Regression", "Clustering", "Dimensionality Reduction", "Time Series Analysis"]
    algorithmType = st.sidebar.selectbox("Select Machine Learning Algorithm Type", mlTypes)
    # Select Model
    ClassificationAlgs = ["Logistic Regression", "Decision Trees", "Random Forest", "Support Vector Machines (SVM)", "K-Nearest Neighbors (KNN)", 
                          "Naive Bayes", "AdaBoost Classifier", "Gradient Boosting Classifier", "XGBoost Classifier", "LightGBM Classifier"]
    RegressionAlgs = ["Linear Regression", "Ridge Regression", "Lasso Regression", "Decision Tree Regressor", "Random Forest Regressor", "Support Vector Regression", 
                      "K-NN Regressor", "AdaBoost Regressor", "Gradient Boosting Regressor", "XGBoost Regressor", "LightGBM Regressor"]
    ClusteringAlgs = ["K-Means Clustering", "Hierarchical Clustering", "DBSCAN"]
    DReductionAlgs = ["Principal Component Analysis (PCA)"]
    TimeSeriesAlgs = ["ARIMA", "Long Short-Term Memory (LSTM)"]
    
    match algorithmType:
        case "Classification":
            selectedAlgs = st.sidebar.multiselect("Select one or more model(s) to fit", ClassificationAlgs)
        case "Regression":
            selectedAlgs = st.sidebar.multiselect("Select one or more model(s) to fit", RegressionAlgs)
        case "Clustering":
            selectedAlgs = st.sidebar.multiselect("Select one or more model(s) to fit", ClusteringAlgs)
        case "Dimensionality Reduction":
            selectedAlgs = st.sidebar.multiselect("Select one or more model(s) to fit", DReductionAlgs)
        case "Time Series Analysis":
            selectedAlgs = st.sidebar.multiselect("Select one or more model(s) to fit", TimeSeriesAlgs)
        case _:
            st.sidebar.write("Please select one algorithm type!")

    # Select X and Y data
    XData = st.multiselect("Select X data (features)", df.columns)
    yData = st.selectbox("Select Y data (target)", df.columns)
    # Train-test split and model fitting
    X = df[XData]
    y = df[yData]
    testSize = st.sidebar.number_input("Set the Test Size Proportion", 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize)

    resultDF = Models(algorithmType, selectedAlgs, X_train, X_test, y_train, y_test)
    return resultDF




'''
Main Page
'''
# Get the data from session_state
if 'df' in st.session_state:
    st.header("View the current dataframe")
    df = st.session_state.df
    st.write(df)
    
    # Sections Execution
    section = st.selectbox("Select a section", ["Select Useful Columns", "Feature Engineering", "Fit Models"])
    
    if section == "Select Useful Columns":
        df = selectCols(df)
        st.write(df)
    elif section == "Feature Engineering":
        df = FeatureEngineering(df)
        st.dataframe(df)
        st.session_state.df = df
    elif section == "Fit Models":
        resultDF = FitModels(df)
        st.header("Results")
        st.dataframe(resultDF)
    else:
        st.write("Please select one section!")
   
    
    

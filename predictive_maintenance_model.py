# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix
# import joblib
# import os
# import streamlit as st
# import plotly.express as px
# import plotly.graph_objects as go
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# class PredictiveMaintenanceModel:
#     def __init__(self):
#         self.model = RandomForestClassifier(n_estimators=100, random_state=42)
#         self.scaler = StandardScaler()
#         self.features = None
#         self.target = 'failure'
#         self.thresholds = {}  # Store normal operating ranges
        
#     def load_data(self, telemetry_path, failures_path):
#         # Load telemetry data
#         print("Loading telemetry data...")
#         telemetry = pd.read_csv(telemetry_path)
#         print("Telemetry columns:", telemetry.columns.tolist())
        
#         # Load failure data
#         print("Loading failure data...")
#         failures = pd.read_csv(failures_path)
#         print("Failures columns:", failures.columns.tolist())
        
#         # Convert datetime to pandas datetime
#         telemetry['datetime'] = pd.to_datetime(telemetry['datetime'])
#         failures['datetime'] = pd.to_datetime(failures['datetime'])
        
#         # Create binary failure indicator (1 if any component failed, 0 otherwise)
#         failures['failure'] = 1
        
#         # Merge the datasets
#         print("Merging datasets...")
#         data = pd.merge(telemetry, failures[['datetime', 'machineID', 'failure']], 
#                        on=['machineID', 'datetime'], 
#                        how='left')
        
#         # Fill NaN values in failure column with 0 (no failure)
#         data['failure'] = data['failure'].fillna(0)
        
#         # Set features
#         exclude_cols = ['datetime', 'machineID', 'failure']
#         self.features = [col for col in telemetry.columns if col not in exclude_cols]
#         print("Selected features:", self.features)
        
#         # Calculate and store normal operating ranges for each feature
#         for feature in self.features:
#             q25 = data[feature].quantile(0.25)
#             q75 = data[feature].quantile(0.75)
#             iqr = q75 - q25
#             self.thresholds[feature] = {
#                 'lower': q25 - 1.5 * iqr,
#                 'upper': q75 + 1.5 * iqr,
#                 'mean': data[feature].mean(),
#                 'std': data[feature].std()
#             }
            
#             # Mark as failure if outside normal range
#             outside_range = (data[feature] < self.thresholds[feature]['lower']) | \
#                           (data[feature] > self.thresholds[feature]['upper'])
#             data.loc[outside_range, 'failure'] = 1
        
#         print("Feature thresholds:", self.thresholds)
        
#         # Convert all feature columns to numeric
#         for feature in self.features:
#             if data[feature].dtype == 'object':
#                 try:
#                     data[feature] = pd.to_numeric(data[feature], errors='coerce')
#                 except Exception as e:
#                     print(f"Error converting {feature} to numeric: {e}")
#                     self.features.remove(feature)
        
#         # Fill any NaN values with mean of the column
#         for feature in self.features:
#             data[feature] = data[feature].fillna(data[feature].mean())
        
#         print(f"Data shape: {data.shape}")
#         print(f"Number of failures: {data['failure'].sum()}")
#         return data
    
#     def preprocess_data(self, data):
#         print("Preprocessing data...")
#         print("Available columns:", data.columns.tolist())
#         print("Using features:", self.features)
        
#         # Select features and target
#         X = data[self.features]
#         y = data[self.target]
        
#         # Split the data
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42, stratify=y
#         )
        
#         # Scale the features
#         print("Scaling features...")
#         X_train_scaled = self.scaler.fit_transform(X_train)
#         X_test_scaled = self.scaler.transform(X_test)
        
#         return X_train_scaled, X_test_scaled, y_train, y_test
    
#     def train(self, X_train, y_train):
#         print("Training model...")
#         self.model.fit(X_train, y_train)
#         print("Model training completed")
    
#     def evaluate(self, X_test, y_test):
#         print("Evaluating model...")
#         y_pred = self.model.predict(X_test)
#         print("\nClassification Report:")
#         print(classification_report(y_test, y_pred))
#         print("\nConfusion Matrix:")
#         print(confusion_matrix(y_test, y_pred))
    
#     def predict(self, new_data):
#         # Calculate anomaly scores for each feature
#         feature_scores = {}
#         weighted_scores = []
        
#         for feature in self.features:
#             value = new_data[feature].iloc[0]
#             threshold = self.thresholds[feature]
            
#             # Calculate how many standard deviations away from mean
#             z_score = abs((value - threshold['mean']) / threshold['std'])
            
#             # Calculate normalized severity (0 to 1)
#             if value < threshold['lower'] or value > threshold['upper']:
#                 # More gradual severity calculation
#                 deviation = max(
#                     abs(value - threshold['lower']) / abs(threshold['lower']),
#                     abs(value - threshold['upper']) / abs(threshold['upper'])
#                 )
#                 severity = min(1.0, deviation * 0.7)  # Scale factor to make it more gradual
#                 feature_scores[feature] = {
#                     'value': value,
#                     'severity': severity * 100,  # Convert to percentage
#                     'normal_range': (threshold['lower'], threshold['upper'])
#                 }
#                 weighted_scores.append(severity)
#             else:
#                 # Even within normal range, add small severity if close to bounds
#                 margin = 0.1  # 10% margin
#                 lower_margin = threshold['lower'] + (threshold['upper'] - threshold['lower']) * margin
#                 upper_margin = threshold['upper'] - (threshold['upper'] - threshold['lower']) * margin
                
#                 if value < lower_margin or value > upper_margin:
#                     severity = 0.2  # 20% severity for borderline values
#                     feature_scores[feature] = {
#                         'value': value,
#                         'severity': severity * 100,
#                         'normal_range': (threshold['lower'], threshold['upper'])
#                     }
#                     weighted_scores.append(severity)
#                 else:
#                     feature_scores[feature] = {
#                         'value': value,
#                         'severity': 0,
#                         'normal_range': (threshold['lower'], threshold['upper'])
#                     }
#                     weighted_scores.append(0)
        
#         # Calculate final probability using a more nuanced approach
#         if weighted_scores:
#             # Calculate base probability from anomaly scores
#             max_severity = max(weighted_scores)
#             avg_severity = sum(weighted_scores) / len(weighted_scores)
            
#             # Combine max and average severity for final probability
#             # This gives more weight to individual severe problems while still considering overall state
#             final_probability = (max_severity * 0.7 + avg_severity * 0.3) * 100
            
#             # Create probability array
#             probability = np.array([[1 - (final_probability/100), final_probability/100]])
            
#             # Determine prediction based on probability threshold
#             prediction = np.array([1 if final_probability > 50 else 0])
            
#             return prediction, probability, feature_scores
        
#         # If no anomalies, return low probability
#         return np.array([0]), np.array([[0.9, 0.1]]), feature_scores
    
#     def save_model(self, model_path='model'):
#         print("Saving model...")
#         os.makedirs(model_path, exist_ok=True)
#         model_data = {
#             'model': self.model,
#             'features': self.features,
#             'scaler': self.scaler,
#             'thresholds': self.thresholds
#         }
#         joblib.dump(model_data, os.path.join(model_path, 'predictive_maintenance_model.joblib'))
#         print("Model saved successfully")
    
#     def load_model(self, model_path='model'):
#         print("Loading model...")
#         model_data = joblib.load(os.path.join(model_path, 'predictive_maintenance_model.joblib'))
#         self.model = model_data['model']
#         self.features = model_data['features']
#         self.scaler = model_data['scaler']
#         self.thresholds = model_data['thresholds']
#         print("Model loaded successfully")

#     def predict_and_show_warning(self, new_data):
#         # Ensure new_data has all required features
#         missing_features = set(self.features) - set(new_data.columns)
#         if missing_features:
#             raise ValueError(f"Missing features in input data: {missing_features}")
        
#         # Select only the required features in the correct order
#         new_data = new_data[self.features]
        
#         # Scale the new data
#         new_data_scaled = self.scaler.transform(new_data)
        
#         # Make prediction
#         prediction, probability, warnings = self.predict(new_data)
        
#         # After making a prediction, add this code
#         if prediction[0] == 1:
#             st.error("⚠️ Failure Likely")
#             # Show which parameters are concerning
#             st.write("Concerning Parameters:")
#             for feature, details in warnings.items():
#                 severity_pct = details['severity'] * 100
#                 st.warning(
#                     f"{feature}:\n"
#                     f"Value: {details['value']:.2f}\n"
#                     f"Severity: {severity_pct:.1f}%\n"
#                     f"Normal range: {details['normal_range'][0]:.2f} to {details['normal_range'][1]:.2f}"
#                 )
#         else:
#             st.success("✅ Normal")

#     # Add Test Scenarios Section
#     st.header("Test Scenarios")
#     test_tab1, test_tab2 = st.tabs(["Predefined Scenarios", "Custom Test Cases"])
    
#     with test_tab1:
#         st.subheader("Test Predefined Scenarios")
        
#         try:
#             # Load historical data for scenarios
#             historical_data = self.load_data('PdM_telemetry.csv', 'PdM_failures.csv')
            
#             # Create example scenarios
#             scenarios = {
#                 "Normal Operation": {
#                     feature: historical_data[feature].median() 
#                     for feature in self.features
#                 },
#                 "Warning Signs": {
#                     feature: historical_data[feature].quantile(0.90) 
#                     for feature in self.features
#                 },
#                 "Critical Condition": {
#                     feature: historical_data[feature].quantile(0.95) * 1.2  # 20% above 95th percentile
#                     for feature in self.features
#                 },
#                 "Extreme Values": {
#                     feature: historical_data[feature].max() * 2  # Double the maximum observed value
#                     for feature in self.features
#                 },
#                 "Mixed Extreme": {  # Some values normal, some extreme
#                     feature: (
#                         historical_data[feature].median() if idx % 2 == 0 
#                         else historical_data[feature].max() * 2
#                     )
#                     for idx, feature in enumerate(self.features)
#                 }
#             }
            
#             # Allow user to select scenarios to compare
#             selected_scenarios = st.multiselect(
#                 "Select scenarios to test",
#                 list(scenarios.keys()),
#                 default=["Normal Operation"]
#             )
            
#             if st.button("Test Selected Scenarios"):
#                 # Create columns for each selected scenario
#                 cols = st.columns(len(selected_scenarios))
                
#                 for idx, (scenario_name, col) in enumerate(zip(selected_scenarios, cols)):
#                     with col:
#                         st.subheader(scenario_name)
                        
#                         # Create DataFrame for scenario
#                         scenario_data = pd.DataFrame([scenarios[scenario_name]])
                        
#                         # Make prediction
#                         prediction, probability, warnings = self.predict(scenario_data)
                        
#                         # Display prediction
#                         if prediction[0] == 1:
#                             st.error("⚠️ Failure Likely")
#                             if warnings:
#                                 st.write("Warning Details:")
#                                 for feature, details in warnings.items():
#                                     severity_pct = details['severity'] * 100
#                                     st.warning(
#                                         f"{feature}:\n"
#                                         f"Value: {details['value']:.2f}\n"
#                                         f"Severity: {severity_pct:.1f}%\n"
#                                         f"Normal range: {details['normal_range'][0]:.2f} to {details['normal_range'][1]:.2f}"
#                                     )
#                         else:
#                             st.success("✅ Normal")
                        
#                         # Display probability gauge
#                         fig = go.Figure(go.Indicator(
#                             mode="gauge+number",
#                             value=probability[0][1] * 100,
#                             domain={'x': [0, 1], 'y': [0, 1]},
#                             title={'text': "Failure Probability (%)"},
#                             gauge={
#                                 'axis': {'range': [0, 100]},
#                                 'bar': {'color': "darkred"},
#                                 'steps': [
#                                     {'range': [0, 20], 'color': "lightgreen"},
#                                     {'range': [20, 40], 'color': "lime"},
#                                     {'range': [40, 60], 'color': "yellow"},
#                                     {'range': [60, 80], 'color': "orange"},
#                                     {'range': [80, 100], 'color': "red"}
#                                 ],
#                                 'threshold': {
#                                     'line': {'color': "black", 'width': 4},
#                                     'thickness': 0.75,
#                                     'value': probability[0][1] * 100
#                                 }
#                             }
#                         ))
                        
#                         # Add more detailed gauge configuration
#                         fig.update_layout(
#                             height=250,
#                             margin=dict(l=10, r=10, t=50, b=10),
#                             font={'size': 16}
#                         )
                        
#                         st.plotly_chart(fig, use_container_width=True)
                        
#                         # Display severity breakdown
#                         if any(details['severity'] > 0 for details in feature_scores.values()):
#                             st.write("Parameter Status:")
#                             cols = st.columns(2)
#                             for idx, (feature, details) in enumerate(feature_scores.items()):
#                                 with cols[idx % 2]:
#                                     if details['severity'] > 0:
#                                         color = (
#                                             "🔴" if details['severity'] > 70 else
#                                             "🟡" if details['severity'] > 30 else
#                                             "🟢"
#                                         )
#                                         st.write(f"{color} {feature}:")
#                                         st.write(f"Current: {details['value']:.2f}")
#                                         st.write(f"Severity: {details['severity']:.1f}%")
#                                         st.write(f"Normal range: {details['normal_range'][0]:.2f} to {details['normal_range'][1]:.2f}")
                        
#                         # Display the values used
#                         st.write("Values used:")
#                         for feature, value in scenarios[scenario_name].items():
#                             st.write(f"{feature}: {value:.2f}")
                            
#         except Exception as e:
#             st.error(f"Error loading historical data: {str(e)}")
#             st.write("Please make sure your data files are available and properly formatted.")


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class PredictiveMaintenanceModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.features = None
        self.target = 'failure'
        self.thresholds = {}  # Store normal operating ranges
        
    def load_data(self, telemetry_path, failures_path):
        # Load telemetry data
        print("Loading telemetry data...")
        telemetry = pd.read_csv(telemetry_path)
        print("Telemetry columns:", telemetry.columns.tolist())
        
        # Load failure data
        print("Loading failure data...")
        failures = pd.read_csv(failures_path)
        print("Failures columns:", failures.columns.tolist())
        
        # Convert datetime to pandas datetime
        telemetry['datetime'] = pd.to_datetime(telemetry['datetime'])
        failures['datetime'] = pd.to_datetime(failures['datetime'])
        
        # Create binary failure indicator (1 if any component failed, 0 otherwise)
        failures['failure'] = 1
        
        # Merge the datasets
        print("Merging datasets...")
        data = pd.merge(telemetry, failures[['datetime', 'machineID', 'failure']], 
                       on=['machineID', 'datetime'], 
                       how='left')
        
        # Fill NaN values in failure column with 0 (no failure)
        data['failure'] = data['failure'].fillna(0)
        
        # Set features
        exclude_cols = ['datetime', 'machineID', 'failure']
        self.features = [col for col in telemetry.columns if col not in exclude_cols]
        print("Selected features:", self.features)
        
        # Calculate and store normal operating ranges for each feature
        for feature in self.features:
            q25 = data[feature].quantile(0.25)
            q75 = data[feature].quantile(0.75)
            iqr = q75 - q25
            self.thresholds[feature] = {
                'lower': q25 - 1.5 * iqr,
                'upper': q75 + 1.5 * iqr,
                'mean': data[feature].mean(),
                'std': data[feature].std()
            }
            
            # Mark as failure if outside normal range
            outside_range = (data[feature] < self.thresholds[feature]['lower']) | \
                          (data[feature] > self.thresholds[feature]['upper'])
            data.loc[outside_range, 'failure'] = 1
        
        print("Feature thresholds:", self.thresholds)
        
        # Convert all feature columns to numeric
        for feature in self.features:
            if data[feature].dtype == 'object':
                try:
                    data[feature] = pd.to_numeric(data[feature], errors='coerce')
                except Exception as e:
                    print(f"Error converting {feature} to numeric: {e}")
                    self.features.remove(feature)
        
        # Fill any NaN values with mean of the column
        for feature in self.features:
            data[feature] = data[feature].fillna(data[feature].mean())
        
        print(f"Data shape: {data.shape}")
        print(f"Number of failures: {data['failure'].sum()}")
        return data
    
    def preprocess_data(self, data):
        print("Preprocessing data...")
        print("Available columns:", data.columns.tolist())
        print("Using features:", self.features)
        
        # Select features and target
        X = data[self.features]
        y = data[self.target]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self, X_train, y_train):
        print("Training model...")
        self.model.fit(X_train, y_train)
        print("Model training completed")
    
    def evaluate(self, X_test, y_test):
        print("Evaluating model...")
        y_pred = self.model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
    
    def predict(self, new_data):
        # Calculate anomaly scores for each feature
        feature_scores = {}
        weighted_scores = []
        
        for feature in self.features:
            value = new_data[feature].iloc[0]
            threshold = self.thresholds[feature]
            
            # Calculate how many standard deviations away from mean
            z_score = abs((value - threshold['mean']) / threshold['std'])
            
            # Calculate normalized severity (0 to 1)
            if value < threshold['lower'] or value > threshold['upper']:
                # More gradual severity calculation
                deviation = max(
                    abs(value - threshold['lower']) / abs(threshold['lower']),
                    abs(value - threshold['upper']) / abs(threshold['upper'])
                )
                severity = min(1.0, deviation * 0.7)  # Scale factor to make it more gradual
                feature_scores[feature] = {
                    'value': value,
                    'severity': severity * 100,  # Convert to percentage
                    'normal_range': (threshold['lower'], threshold['upper'])
                }
                weighted_scores.append(severity)
            else:
                # Even within normal range, add small severity if close to bounds
                margin = 0.1  # 10% margin
                lower_margin = threshold['lower'] + (threshold['upper'] - threshold['lower']) * margin
                upper_margin = threshold['upper'] - (threshold['upper'] - threshold['lower']) * margin
                
                if value < lower_margin or value > upper_margin:
                    severity = 0.2  # 20% severity for borderline values
                    feature_scores[feature] = {
                        'value': value,
                        'severity': severity * 100,
                        'normal_range': (threshold['lower'], threshold['upper'])
                    }
                    weighted_scores.append(severity)
                else:
                    feature_scores[feature] = {
                        'value': value,
                        'severity': 0,
                        'normal_range': (threshold['lower'], threshold['upper'])
                    }
                    weighted_scores.append(0)
        
        # Calculate final probability using a more nuanced approach
        if weighted_scores:
            # Calculate base probability from anomaly scores
            max_severity = max(weighted_scores)
            avg_severity = sum(weighted_scores) / len(weighted_scores)
            
            # Combine max and average severity for final probability
            # This gives more weight to individual severe problems while still considering overall state
            final_probability = (max_severity * 0.7 + avg_severity * 0.3) * 100
            
            # Create probability array
            probability = np.array([[1 - (final_probability/100), final_probability/100]])
            
            # Determine prediction based on probability threshold
            prediction = np.array([1 if final_probability > 50 else 0])
            
            return prediction, probability, feature_scores
        
        # If no anomalies, return low probability
        return np.array([0]), np.array([[0.9, 0.1]]), feature_scores
    
    def save_model(self, model_path='model'):
        print("Saving model...")
        os.makedirs(model_path, exist_ok=True)
        model_data = {
            'model': self.model,
            'features': self.features,
            'scaler': self.scaler,
            'thresholds': self.thresholds
        }
        joblib.dump(model_data, os.path.join(model_path, 'predictive_maintenance_model.joblib'))
        print("Model saved successfully")
    
    def load_model(self, model_path='model'):
        print("Loading model...")
        model_data = joblib.load(os.path.join(model_path, 'predictive_maintenance_model.joblib'))
        self.model = model_data['model']
        self.features = model_data['features']
        self.scaler = model_data['scaler']
        self.thresholds = model_data['thresholds']
        print("Model loaded successfully")

    def predict_and_show_warning(self, new_data):
        # Ensure new_data has all required features
        missing_features = set(self.features) - set(new_data.columns)
        if missing_features:
            raise ValueError(f"Missing features in input data: {missing_features}")
        
        # Select only the required features in the correct order
        new_data = new_data[self.features]
        
        # Scale the new data
        new_data_scaled = self.scaler.transform(new_data)
        
        # Make prediction
        prediction, probability, warnings = self.predict(new_data)
        
        # After making a prediction, add this code
        if prediction[0] == 1:
            st.error("⚠️ Failure Likely")
            # Show which parameters are concerning
            st.write("Concerning Parameters:")
            for feature, details in warnings.items():
                severity_pct = details['severity'] * 100
                st.warning(
                    f"{feature}:\n"
                    f"Value: {details['value']:.2f}\n"
                    f"Severity: {severity_pct:.1f}%\n"
                    f"Normal range: {details['normal_range'][0]:.2f} to {details['normal_range'][1]:.2f}"
                )
        else:
            st.success("✅ Normal")
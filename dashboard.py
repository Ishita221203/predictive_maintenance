# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from predictive_maintenance_model import PredictiveMaintenanceModel
# import numpy as np
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
# from plotly.subplots import make_subplots
# import datetime
# import io

TELEMETRY_URL = "https://drive.google.com/uc?id=1duTtnefd7JafuQGiyCQ22Z-c4HM7gqmZ"
FAILURES_URL = "https://drive.google.com/uc?id=1sIZmHQDmCqjjJ6yuNTgm3FWrLToDEOwq"


# # Must be the first Streamlit command
# st.set_page_config(
#     page_title="Machine Health Monitoring Dashboard",
#     page_icon="⚙️",
#     layout="wide"
# )

# # Initialize the model
# @st.cache_resource
# def load_model():
#     model = PredictiveMaintenanceModel()
#     try:
#         model.load_model()
#         return model
#     except:
#         return None

# # Load historical data with fixed caching
# @st.cache_data
# def load_historical_data(_model):  # Added underscore to parameter
#     try:
#         return pd.read_csv(TELEMETRY_URL)
#     except Exception as e:
#         st.error(f"Error loading telemetry data: {str(e)}")
#         return None

# # Sidebar
# st.sidebar.title("Machine Health Monitoring")
# st.sidebar.markdown("---")

# # Main content
# st.title("Machine Health Monitoring Dashboard")
# st.markdown("---")

# # Load model and data
# model = load_model()
# historical_data = load_historical_data(model) if model is not None else None

# if model is None:
#     st.warning("Model not found. Please train the model first.")
#     if st.button("Train Model"):
#         with st.spinner("Training model..."):
#             model = PredictiveMaintenanceModel()
#             data = model.load_data(TELEMETRY_URL, 'PdM_failures.csv')
#             X_train, X_test, y_train, y_test = model.preprocess_data(data)
#             model.train(X_train, y_train)
            
#             # Calculate and display model metrics
#             st.header("Model Performance Metrics")
#             y_pred = model.model.predict(X_test)
            
#             accuracy = accuracy_score(y_test, y_pred)
#             precision = precision_score(y_test, y_pred)
#             recall = recall_score(y_test, y_pred)
#             f1 = f1_score(y_test, y_pred)
            
#             col1, col2, col3, col4 = st.columns(4)
#             col1.metric("Accuracy", f"{accuracy:.2%}")
#             col2.metric("Precision", f"{precision:.2%}")
#             col3.metric("Recall", f"{recall:.2%}")
#             col4.metric("F1 Score", f"{f1:.2%}")
            
#             model.save_model()
#             st.success("Model trained successfully!")
#             st.rerun()
# else:
#     # Real-time monitoring section
#     st.header("Real-time Monitoring")
    
#     # Create input fields for parameters
#     col1, col2, col3 = st.columns(3)
#     input_values = {}
    
#     # Distribute features across columns
#     features_per_column = len(model.features) // 3 + (len(model.features) % 3 > 0)
    
#     for i, feature in enumerate(model.features):
#         col_index = i // features_per_column
#         with [col1, col2, col3][col_index]:
#             input_values[feature] = st.number_input(
#                 f"{feature}", 
#                 min_value=float(-1e6),
#                 max_value=float(1e6),
#                 value=50.0
#             )
    
#     # Test Scenarios Section
#     st.header("Test Scenarios")
#     test_tab1, test_tab2 = st.tabs(["Predefined Scenarios", "Custom Test Cases"])
    
#     with test_tab1:
#         st.subheader("Test Predefined Scenarios")
        
#         if historical_data is not None:
#             # Create example scenarios
#             scenarios = {
#                 "Normal Operation": {
#                     feature: historical_data[feature].median() 
#                     for feature in model.features
#                 },
#                 "Warning Signs": {
#                     feature: historical_data[feature].quantile(0.90) 
#                     for feature in model.features
#                 },
#                 "Critical Condition": {
#                     feature: historical_data[feature].quantile(0.95) * 1.2  # 20% above 95th percentile
#                     for feature in model.features
#                 },
#                 "Extreme Values": {
#                     feature: historical_data[feature].max() * 2  # Double the maximum observed value
#                     for feature in model.features
#                 },
#                 "Mixed Extreme": {  # Some values normal, some extreme
#                     feature: (
#                         historical_data[feature].median() if idx % 2 == 0 
#                         else historical_data[feature].max() * 2
#                     )
#                     for idx, feature in enumerate(model.features)
#                 }
#             }
            
#             # Allow user to select scenarios to compare
#             selected_scenarios = st.multiselect(
#                 "Select scenarios to test",
#                 list(scenarios.keys()),
#                 default=["Normal Operation"]
#             )
            
#             if st.button("Test Selected Scenarios"):
#                 cols = st.columns(len(selected_scenarios))
                
#                 for idx, (scenario_name, col) in enumerate(zip(selected_scenarios, cols)):
#                     with col:
#                         st.subheader(scenario_name)
#                         scenario_data = pd.DataFrame([scenarios[scenario_name]])
#                         prediction, probability, warnings = model.predict(scenario_data)
                        
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
                        
#                         # Display probability gauge with more detailed steps
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
#                         if any(details['severity'] > 0 for details in warnings.values()):
#                             st.write("Parameter Status:")
#                             cols = st.columns(2)
#                             for idx, (feature, details) in enumerate(warnings.items()):
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
#         else:
#             st.error("Historical data not available. Please ensure data files are present.")
    
#     with test_tab2:
#         st.subheader("Custom Test Cases")
#         st.write("Enter custom values (one set per line) in CSV format:")
#         st.write("Format: " + ",".join(model.features))
        
#         if historical_data is not None:
#             example_normal = ",".join([str(historical_data[f].median()) for f in model.features])
#             example_critical = ",".join([str(historical_data[f].quantile(0.95)) for f in model.features])
            
#             st.caption("Example format (copy and modify these):")
#             st.code(f"Normal values:\n{example_normal}\n\nCritical values:\n{example_critical}")
        
#         batch_input = st.text_area("Enter test values (CSV format)", height=150)
        
#         if st.button("Test Custom Values"):
#             try:
#                 input_lines = [line.strip() for line in batch_input.split('\n') if line.strip()]
#                 test_cases = []
                
#                 for line in input_lines:
#                     values = [float(x.strip()) for x in line.split(',')]
#                     if len(values) == len(model.features):
#                         test_cases.append(dict(zip(model.features, values)))
#                     else:
#                         st.warning(f"Skipping invalid line: {line} (wrong number of values)")
                
#                 if test_cases:
#                     cols = st.columns(min(len(test_cases), 4))
                    
#                     for idx, (test_case, col) in enumerate(zip(test_cases, cols)):
#                         with col:
#                             st.subheader(f"Test Case {idx + 1}")
#                             test_data = pd.DataFrame([test_case])
#                             prediction, probability, feature_scores = model.predict(test_data)
                            
#                             if prediction[0] == 1:
#                                 st.error("⚠️ Failure Likely")
#                             else:
#                                 st.success("✅ Normal")
                            
#                             # Display probability gauge with more detailed steps
#                             fig = go.Figure(go.Indicator(
#                                 mode="gauge+number",
#                                 value=probability[0][1] * 100,
#                                 domain={'x': [0, 1], 'y': [0, 1]},
#                                 title={'text': "Failure Probability (%)"},
#                                 gauge={
#                                     'axis': {'range': [0, 100]},
#                                     'bar': {'color': "darkred"},
#                                     'steps': [
#                                         {'range': [0, 20], 'color': "lightgreen"},
#                                         {'range': [20, 40], 'color': "lime"},
#                                         {'range': [40, 60], 'color': "yellow"},
#                                         {'range': [60, 80], 'color': "orange"},
#                                         {'range': [80, 100], 'color': "red"}
#                                     ],
#                                     'threshold': {
#                                         'line': {'color': "black", 'width': 4},
#                                         'thickness': 0.75,
#                                         'value': probability[0][1] * 100
#                                     }
#                                 }
#                             ))
                            
#                             # Add more detailed gauge configuration
#                             fig.update_layout(
#                                 height=250,
#                                 margin=dict(l=10, r=10, t=50, b=10),
#                                 font={'size': 16}
#                             )
                            
#                             st.plotly_chart(fig, use_container_width=True)
                            
#                             # Display severity breakdown
#                             if any(details['severity'] > 0 for details in feature_scores.values()):
#                                 st.write("Parameter Status:")
#                                 cols = st.columns(2)
#                                 for idx, (feature, details) in enumerate(feature_scores.items()):
#                                     with cols[idx % 2]:
#                                         if details['severity'] > 0:
#                                             color = (
#                                                 "🔴" if details['severity'] > 70 else
#                                                 "🟡" if details['severity'] > 30 else
#                                                 "🟢"
#                                             )
#                                             st.write(f"{color} {feature}:")
#                                             st.write(f"Current: {details['value']:.2f}")
#                                             st.write(f"Severity: {details['severity']:.1f}%")
#                                             st.write(f"Normal range: {details['normal_range'][0]:.2f} to {details['normal_range'][1]:.2f}")
            
#             except Exception as e:
#                 st.error(f"Error processing input: {str(e)}")
#                 st.write("Please make sure your input follows the correct format.")

# # Footer
# st.markdown("---")
# st.markdown("Developed with ❤️ for Machine Health Monitoring")

# # Add a new section in your dashboard after the test scenarios
# st.header("Advanced Visualizations")
# viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Real-time Monitoring", "Pattern Analysis", "Correlation Analysis"])

# with viz_tab1:
#     st.subheader("Real-time Parameter Monitoring")
    
#     # Create multi-parameter line chart
#     try:
#         # Load historical data
#         historical_data = pd.read_csv(TELEMETRY_URL)
#         historical_data['datetime'] = pd.to_datetime(historical_data['datetime'])
        
#         # Create time series plot with multiple parameters
#         fig = make_subplots(rows=2, cols=2,
#                            subplot_titles=model.features,
#                            shared_xaxes=True)
        
#         row, col = 1, 1
#         for feature in model.features:
#             # Get threshold values for this feature
#             threshold = model.thresholds[feature]
            
#             # Add parameter line
#             fig.add_trace(
#                 go.Scatter(x=historical_data['datetime'], 
#                           y=historical_data[feature],
#                           name=feature,
#                           line=dict(width=2)),
#                 row=row, col=col
#             )
            
#             # Add threshold lines
#             fig.add_trace(
#                 go.Scatter(x=historical_data['datetime'],
#                           y=[threshold['upper']] * len(historical_data),
#                           name=f'{feature} Upper Threshold',
#                           line=dict(dash='dash', color='red', width=1)),
#                 row=row, col=col
#             )
            
#             fig.add_trace(
#                 go.Scatter(x=historical_data['datetime'],
#                           y=[threshold['lower']] * len(historical_data),
#                           name=f'{feature} Lower Threshold',
#                           line=dict(dash='dash', color='red', width=1)),
#                 row=row, col=col
#             )
            
#             col += 1
#             if col > 2:
#                 col = 1
#                 row += 1
        
#         fig.update_layout(height=800, showlegend=True,
#                          title_text="Parameter Trends with Thresholds")
#         st.plotly_chart(fig, use_container_width=True)
        
#     except Exception as e:
#         st.error(f"Error loading historical data: {str(e)}")

# with viz_tab2:
#     st.subheader("Pattern Analysis")
    
#     # Create distribution plots
#     if historical_data is not None:
#         # Select parameter for analysis
#         selected_param = st.selectbox("Select Parameter for Analysis", model.features)
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             # Create distribution plot
#             fig = go.Figure()
#             fig.add_trace(go.Histogram(
#                 x=historical_data[selected_param],
#                 name="Distribution",
#                 nbinsx=50,
#                 opacity=0.7
#             ))
            
#             # Add threshold lines
#             threshold = model.thresholds[selected_param]
#             fig.add_vline(x=threshold['lower'], 
#                          line_dash="dash", 
#                          line_color="red",
#                          annotation_text="Lower Threshold")
#             fig.add_vline(x=threshold['upper'], 
#                          line_dash="dash", 
#                          line_color="red",
#                          annotation_text="Upper Threshold")
            
#             fig.update_layout(
#                 title=f"{selected_param} Distribution",
#                 xaxis_title=selected_param,
#                 yaxis_title="Frequency"
#             )
#             st.plotly_chart(fig, use_container_width=True)
        
#         with col2:
#             # Create box plot
#             fig = go.Figure()
#             fig.add_trace(go.Box(
#                 y=historical_data[selected_param],
#                 name=selected_param,
#                 boxpoints='outliers'
#             ))
#             fig.update_layout(
#                 title=f"{selected_param} Box Plot",
#                 yaxis_title=selected_param
#             )
#             st.plotly_chart(fig, use_container_width=True)

# with viz_tab3:
#     st.subheader("Correlation Analysis")
    
#     if historical_data is not None:
#         # Calculate correlation matrix
#         corr_matrix = historical_data[model.features].corr()
        
#         # Create heatmap
#         fig = go.Figure(data=go.Heatmap(
#             z=corr_matrix,
#             x=model.features,
#             y=model.features,
#             colorscale='RdBu',
#             zmin=-1, zmax=1,
#             text=np.round(corr_matrix, 2),
#             texttemplate='%{text}',
#             textfont={"size": 10},
#             hoverongaps=False
#         ))
        
#         fig.update_layout(
#             title="Parameter Correlation Matrix",
#             height=600,
#             width=800
#         )
        
#         st.plotly_chart(fig, use_container_width=True)
        
#         # Add parameter relationship scatter plot
#         st.subheader("Parameter Relationships")
#         col1, col2 = st.columns(2)
#         with col1:
#             param1 = st.selectbox("Select First Parameter", model.features, key="param1")
#         with col2:
#             param2 = st.selectbox("Select Second Parameter", model.features, key="param2")
        
#         fig = go.Figure(data=go.Scatter(
#             x=historical_data[param1],
#             y=historical_data[param2],
#             mode='markers',
#             marker=dict(
#                 size=8,
#                 color=historical_data['datetime'].astype(np.int64),
#                 colorscale='Viridis',
#                 showscale=True,
#                 colorbar=dict(title="Time")
#             )
#         ))
        
#         fig.update_layout(
#             title=f"Relationship between {param1} and {param2}",
#             xaxis_title=param1,
#             yaxis_title=param2,
#             height=600
#         )
        
#         st.plotly_chart(fig, use_container_width=True)

# # Add a download section for the visualizations
# st.header("Export Data")
# if st.button("Generate Report"):
#     # Create a PDF or Excel report with all visualizations
#     buffer = io.BytesIO()
    
#     # Create Excel writer object
#     with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
#         # Write summary statistics
#         summary_stats = historical_data[model.features].describe()
#         summary_stats.to_excel(writer, sheet_name='Summary Statistics')
        
#         # Write correlation matrix
#         corr_matrix.to_excel(writer, sheet_name='Correlations')
        
#         # Write recent readings
#         recent_data = historical_data.tail(100)
#         recent_data.to_excel(writer, sheet_name='Recent Readings')
    
#     st.download_button(
#         label="Download Analysis Report",
#         data=buffer,
#         file_name="machine_health_report.xlsx",
#         mime="application/vnd.ms-excel"
#     )



import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from predictive_maintenance_model import PredictiveMaintenanceModel
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import datetime
import io

# Must be the first Streamlit command
st.set_page_config(
    page_title="Machine Health Monitoring Dashboard",
    page_icon="⚙️",
    layout="wide"
)

# Initialize the model
@st.cache_resource
def load_model():
    model = PredictiveMaintenanceModel()
    try:
        model.load_model()
        return model
    except:
        return None

# Load historical data with fixed caching
@st.cache_data
def load_historical_data(_model):
    try:
        return pd.read_csv(TELEMETRY_URL)
    except Exception as e:
        st.error(f"Error loading telemetry data: {str(e)}")
        return None

# Sidebar
st.sidebar.title("Machine Health Monitoring")
st.sidebar.markdown("---")

# Main content
st.title("Machine Health Monitoring Dashboard")
st.markdown("---")

# Load model and data
model = load_model()
historical_data = load_historical_data(model) if model is not None else None

if model is None:
    st.warning("Model not found. Please train the model first.")
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            model = PredictiveMaintenanceModel()
            data = model.load_data(TELEMETRY_URL, 'PdM_failures.csv')
            X_train, X_test, y_train, y_test = model.preprocess_data(data)
            model.train(X_train, y_train)
            
            # Calculate and display model metrics
            st.header("Model Performance Metrics")
            y_pred = model.model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{accuracy:.2%}")
            col2.metric("Precision", f"{precision:.2%}")
            col3.metric("Recall", f"{recall:.2%}")
            col4.metric("F1 Score", f"{f1:.2%}")
            
            model.save_model()
            st.success("Model trained successfully!")
            st.rerun()
else:
    # Real-time monitoring section
    st.header("Real-time Monitoring")
    
    # Create input fields for parameters
    col1, col2, col3 = st.columns(3)
    input_values = {}
    
    # Distribute features across columns
    features_per_column = len(model.features) // 3 + (len(model.features) % 3 > 0)
    
    for i, feature in enumerate(model.features):
        col_index = i // features_per_column
        with [col1, col2, col3][col_index]:
            input_values[feature] = st.number_input(
                f"{feature}", 
                min_value=float(-1e6),
                max_value=float(1e6),
                value=50.0
            )
    
    # Test Scenarios Section
    st.header("Test Scenarios")
    test_tab1, test_tab2 = st.tabs(["Predefined Scenarios", "Custom Test Cases"])
    
    with test_tab1:
        st.subheader("Test Predefined Scenarios")
        
        if historical_data is not None:
            # Create example scenarios
            scenarios = {
                "Normal Operation": {
                    feature: historical_data[feature].median() 
                    for feature in model.features
                },
                "Warning Signs": {
                    feature: historical_data[feature].quantile(0.90) 
                    for feature in model.features
                },
                "Critical Condition": {
                    feature: historical_data[feature].quantile(0.95) * 1.2  # 20% above 95th percentile
                    for feature in model.features
                },
                "Extreme Values": {
                    feature: historical_data[feature].max() * 2  # Double the maximum observed value
                    for feature in model.features
                },
                "Mixed Extreme": {  # Some values normal, some extreme
                    feature: (
                        historical_data[feature].median() if idx % 2 == 0 
                        else historical_data[feature].max() * 2
                    )
                    for idx, feature in enumerate(model.features)
                }
            }
            
            # Allow user to select scenarios to compare
            selected_scenarios = st.multiselect(
                "Select scenarios to test",
                list(scenarios.keys()),
                default=["Normal Operation"]
            )
            
            if st.button("Test Selected Scenarios"):
                cols = st.columns(len(selected_scenarios))
                
                for idx, (scenario_name, col) in enumerate(zip(selected_scenarios, cols)):
                    with col:
                        st.subheader(scenario_name)
                        scenario_data = pd.DataFrame([scenarios[scenario_name]])
                        prediction, probability, warnings = model.predict(scenario_data)
                        
                        # Display prediction
                        if prediction[0] == 1:
                            st.error("⚠️ Failure Likely")
                            if warnings:
                                st.write("Warning Details:")
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
                        
                        # Display probability gauge with unique key
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=probability[0][1] * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Failure Probability (%)"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkred"},
                                'steps': [
                                    {'range': [0, 20], 'color': "lightgreen"},
                                    {'range': [20, 40], 'color': "lime"},
                                    {'range': [40, 60], 'color': "yellow"},
                                    {'range': [60, 80], 'color': "orange"},
                                    {'range': [80, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "black", 'width': 4},
                                    'thickness': 0.75,
                                    'value': probability[0][1] * 100
                                }
                            }
                        ))
                        
                        fig.update_layout(
                            height=250,
                            margin=dict(l=10, r=10, t=50, b=10),
                            font={'size': 16}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, key=f"scenario_gauge_{scenario_name}")
                        
                        # Display severity breakdown
                        if any(details['severity'] > 0 for details in warnings.values()):
                            st.write("Parameter Status:")
                            cols = st.columns(2)
                            for idx, (feature, details) in enumerate(warnings.items()):
                                with cols[idx % 2]:
                                    if details['severity'] > 0:
                                        color = (
                                            "🔴" if details['severity'] > 70 else
                                            "🟡" if details['severity'] > 30 else
                                            "🟢"
                                        )
                                        st.write(f"{color} {feature}:")
                                        st.write(f"Current: {details['value']:.2f}")
                                        st.write(f"Severity: {details['severity']:.1f}%")
                                        st.write(f"Normal range: {details['normal_range'][0]:.2f} to {details['normal_range'][1]:.2f}")
        else:
            st.error("Historical data not available. Please ensure data files are present.")
    
    with test_tab2:
        st.subheader("Custom Test Cases")
        st.write("Enter custom values (one set per line) in CSV format:")
        st.write("Format: " + ",".join(model.features))
        
        if historical_data is not None:
            example_normal = ",".join([str(historical_data[f].median()) for f in model.features])
            example_critical = ",".join([str(historical_data[f].quantile(0.95)) for f in model.features])
            
            st.caption("Example format (copy and modify these):")
            st.code(f"Normal values:\n{example_normal}\n\nCritical values:\n{example_critical}")
        
        batch_input = st.text_area("Enter test values (CSV format)", height=150)
        
        if st.button("Test Custom Values"):
            try:
                input_lines = [line.strip() for line in batch_input.split('\n') if line.strip()]
                test_cases = []
                
                for line in input_lines:
                    values = [float(x.strip()) for x in line.split(',')]
                    if len(values) == len(model.features):
                        test_cases.append(dict(zip(model.features, values)))
                    else:
                        st.warning(f"Skipping invalid line: {line} (wrong number of values)")
                
                if test_cases:
                    cols = st.columns(min(len(test_cases), 4))
                    
                    for idx, (test_case, col) in enumerate(zip(test_cases, cols)):
                        with col:
                            st.subheader(f"Test Case {idx + 1}")
                            test_data = pd.DataFrame([test_case])
                            prediction, probability, feature_scores = model.predict(test_data)
                            
                            if prediction[0] == 1:
                                st.error("⚠️ Failure Likely")
                            else:
                                st.success("✅ Normal")
                            
                            # Display probability gauge with unique key
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=probability[0][1] * 100,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "Failure Probability (%)"},
                                gauge={
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': "darkred"},
                                    'steps': [
                                        {'range': [0, 20], 'color': "lightgreen"},
                                        {'range': [20, 40], 'color': "lime"},
                                        {'range': [40, 60], 'color': "yellow"},
                                        {'range': [60, 80], 'color': "orange"},
                                        {'range': [80, 100], 'color': "red"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "black", 'width': 4},
                                        'thickness': 0.75,
                                        'value': probability[0][1] * 100
                                    }
                                }
                            ))
                            
                            fig.update_layout(
                                height=250,
                                margin=dict(l=10, r=10, t=50, b=10),
                                font={'size': 16}
                            )
                            
                            st.plotly_chart(fig, use_container_width=True, key=f"custom_gauge_{idx}")
                            
                            # Display severity breakdown
                            if any(details['severity'] > 0 for details in feature_scores.values()):
                                st.write("Parameter Status:")
                                cols = st.columns(2)
                                for idx, (feature, details) in enumerate(feature_scores.items()):
                                    with cols[idx % 2]:
                                        if details['severity'] > 0:
                                            color = (
                                                "🔴" if details['severity'] > 70 else
                                                "🟡" if details['severity'] > 30 else
                                                "🟢"
                                            )
                                            st.write(f"{color} {feature}:")
                                            st.write(f"Current: {details['value']:.2f}")
                                            st.write(f"Severity: {details['severity']:.1f}%")
                                            st.write(f"Normal range: {details['normal_range'][0]:.2f} to {details['normal_range'][1]:.2f}")
            
            except Exception as e:
                st.error(f"Error processing input: {str(e)}")
                st.write("Please make sure your input follows the correct format.")

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ for Machine Health Monitoring")

# Add a new section in your dashboard after the test scenarios
st.header("Advanced Visualizations")
viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Real-time Monitoring", "Pattern Analysis", "Correlation Analysis"])

with viz_tab1:
    st.subheader("Real-time Parameter Monitoring")
    
    # Create multi-parameter line chart
    try:
        # Load historical data
        historical_data = pd.read_csv(TELEMETRY_URL)
        historical_data['datetime'] = pd.to_datetime(historical_data['datetime'])
        
        # Create time series plot with multiple parameters
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=model.features,
                           shared_xaxes=True)
        
        row, col = 1, 1
        for feature in model.features:
            # Get threshold values for this feature
            threshold = model.thresholds[feature]
            
            # Add parameter line
            fig.add_trace(
                go.Scatter(x=historical_data['datetime'], 
                          y=historical_data[feature],
                          name=feature,
                          line=dict(width=2)),
                row=row, col=col
            )
            
            # Add threshold lines
            fig.add_trace(
                go.Scatter(x=historical_data['datetime'],
                          y=[threshold['upper']] * len(historical_data),
                          name=f'{feature} Upper Threshold',
                          line=dict(dash='dash', color='red', width=1)),
                row=row, col=col
            )
            
            fig.add_trace(
                go.Scatter(x=historical_data['datetime'],
                          y=[threshold['lower']] * len(historical_data),
                          name=f'{feature} Lower Threshold',
                          line=dict(dash='dash', color='red', width=1)),
                row=row, col=col
            )
            
            col += 1
            if col > 2:
                col = 1
                row += 1
        
        fig.update_layout(height=800, showlegend=True,
                         title_text="Parameter Trends with Thresholds")
        st.plotly_chart(fig, use_container_width=True, key="param_monitoring")
        
    except Exception as e:
        st.error(f"Error loading historical data: {str(e)}")

with viz_tab2:
    st.subheader("Pattern Analysis")
    
    # Create distribution plots
    if historical_data is not None:
        # Select parameter for analysis
        selected_param = st.selectbox("Select Parameter for Analysis", model.features)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Create distribution plot with unique key
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=historical_data[selected_param],
                name="Distribution",
                nbinsx=50,
                opacity=0.7
            ))
            
            # Add threshold lines
            threshold = model.thresholds[selected_param]
            fig.add_vline(x=threshold['lower'], 
                         line_dash="dash", 
                         line_color="red",
                         annotation_text="Lower Threshold")
            fig.add_vline(x=threshold['upper'], 
                         line_dash="dash", 
                         line_color="red",
                         annotation_text="Upper Threshold")
            
            fig.update_layout(
                title=f"{selected_param} Distribution",
                xaxis_title=selected_param,
                yaxis_title="Frequency"
            )
            st.plotly_chart(fig, use_container_width=True, key=f"dist_plot_{selected_param}")
        
        with col2:
            # Create box plot with unique key
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=historical_data[selected_param],
                name=selected_param,
                boxpoints='outliers'
            ))
            fig.update_layout(
                title=f"{selected_param} Box Plot",
                yaxis_title=selected_param
            )
            st.plotly_chart(fig, use_container_width=True, key=f"box_plot_{selected_param}")

with viz_tab3:
    st.subheader("Correlation Analysis")
    
    if historical_data is not None:
        # Calculate correlation matrix
        corr_matrix = historical_data[model.features].corr()
        
        # Create heatmap with unique key
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=model.features,
            y=model.features,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            text=np.round(corr_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Parameter Correlation Matrix",
            height=600,
            width=800
        )
        
        st.plotly_chart(fig, use_container_width=True, key="correlation_heatmap")
        
        # Add parameter relationship scatter plot with unique key
        st.subheader("Parameter Relationships")
        col1, col2 = st.columns(2)
        with col1:
            param1 = st.selectbox("Select First Parameter", model.features, key="param1")
        with col2:
            param2 = st.selectbox("Select Second Parameter", model.features, key="param2")
        
        fig = go.Figure(data=go.Scatter(
            x=historical_data[param1],
            y=historical_data[param2],
            mode='markers',
            marker=dict(
                size=8,
                color=historical_data['datetime'].astype(np.int64),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Time")
            )
        ))
        
        fig.update_layout(
            title=f"Relationship between {param1} and {param2}",
            xaxis_title=param1,
            yaxis_title=param2,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True, key=f"scatter_{param1}_{param2}")

# Add a download section for the visualizations
st.header("Export Data")
if st.button("Generate Report"):
    # Create a PDF or Excel report with all visualizations
    buffer = io.BytesIO()
    
    # Create Excel writer object
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Write summary statistics
        summary_stats = historical_data[model.features].describe()
        summary_stats.to_excel(writer, sheet_name='Summary Statistics')
        
        # Write correlation matrix
        corr_matrix.to_excel(writer, sheet_name='Correlations')
        
        # Write recent readings
        recent_data = historical_data.tail(100)
        recent_data.to_excel(writer, sheet_name='Recent Readings')
    
    st.download_button(
        label="Download Analysis Report",
        data=buffer,
        file_name="machine_health_report.xlsx",
        mime="application/vnd.ms-excel"
    )

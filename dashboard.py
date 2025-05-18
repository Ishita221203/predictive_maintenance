# # import streamlit as st
# # import pandas as pd
# # import plotly.express as px
# # import plotly.graph_objects as go
# # from predictive_maintenance_model import PredictiveMaintenanceModel
# # import numpy as np
# # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# # import seaborn as sns
# # import matplotlib.pyplot as plt
# # from plotly.subplots import make_subplots
# # import datetime
# # import io

# TELEMETRY_URL = "https://drive.google.com/uc?id=1duTtnefd7JafuQGiyCQ22Z-c4HM7gqmZ"
# FAILURES_URL = "https://drive.google.com/uc?id=1sIZmHQDmCqjjJ6yuNTgm3FWrLToDEOwq"


# # # Must be the first Streamlit command
# # st.set_page_config(
# #     page_title="Machine Health Monitoring Dashboard",
# #     page_icon="⚙️",
# #     layout="wide"
# # )

# # # Initialize the model
# # @st.cache_resource
# # def load_model():
# #     model = PredictiveMaintenanceModel()
# #     try:
# #         model.load_model()
# #         return model
# #     except:
# #         return None

# # # Load historical data with fixed caching
# # @st.cache_data
# # def load_historical_data(_model):  # Added underscore to parameter
# #     try:
# #         return pd.read_csv(TELEMETRY_URL)
# #     except Exception as e:
# #         st.error(f"Error loading telemetry data: {str(e)}")
# #         return None

# # # Sidebar
# # st.sidebar.title("Machine Health Monitoring")
# # st.sidebar.markdown("---")

# # # Main content
# # st.title("Machine Health Monitoring Dashboard")
# # st.markdown("---")

# # # Load model and data
# # model = load_model()
# # historical_data = load_historical_data(model) if model is not None else None

# # if model is None:
# #     st.warning("Model not found. Please train the model first.")
# #     if st.button("Train Model"):
# #         with st.spinner("Training model..."):
# #             model = PredictiveMaintenanceModel()
# #             data = model.load_data(TELEMETRY_URL, FAILURES_URL)
# #             X_train, X_test, y_train, y_test = model.preprocess_data(data)
# #             model.train(X_train, y_train)
            
# #             # Calculate and display model metrics
# #             st.header("Model Performance Metrics")
# #             y_pred = model.model.predict(X_test)
            
# #             accuracy = accuracy_score(y_test, y_pred)
# #             precision = precision_score(y_test, y_pred)
# #             recall = recall_score(y_test, y_pred)
# #             f1 = f1_score(y_test, y_pred)
            
# #             col1, col2, col3, col4 = st.columns(4)
# #             col1.metric("Accuracy", f"{accuracy:.2%}")
# #             col2.metric("Precision", f"{precision:.2%}")
# #             col3.metric("Recall", f"{recall:.2%}")
# #             col4.metric("F1 Score", f"{f1:.2%}")
            
# #             model.save_model()
# #             st.success("Model trained successfully!")
# #             st.rerun()
# # else:
# #     # Real-time monitoring section
# #     st.header("Real-time Monitoring")
    
# #     # Create input fields for parameters
# #     col1, col2, col3 = st.columns(3)
# #     input_values = {}
    
# #     # Distribute features across columns
# #     features_per_column = len(model.features) // 3 + (len(model.features) % 3 > 0)
    
# #     for i, feature in enumerate(model.features):
# #         col_index = i // features_per_column
# #         with [col1, col2, col3][col_index]:
# #             input_values[feature] = st.number_input(
# #                 f"{feature}", 
# #                 min_value=float(-1e6),
# #                 max_value=float(1e6),
# #                 value=50.0
# #             )
    
# #     # Test Scenarios Section
# #     st.header("Test Scenarios")
# #     test_tab1, test_tab2 = st.tabs(["Predefined Scenarios", "Custom Test Cases"])
    
# #     with test_tab1:
# #         st.subheader("Test Predefined Scenarios")
        
# #         if historical_data is not None:
# #             # Create example scenarios
# #             scenarios = {
# #                 "Normal Operation": {
# #                     feature: historical_data[feature].median() 
# #                     for feature in model.features
# #                 },
# #                 "Warning Signs": {
# #                     feature: historical_data[feature].quantile(0.90) 
# #                     for feature in model.features
# #                 },
# #                 "Critical Condition": {
# #                     feature: historical_data[feature].quantile(0.95) * 1.2  # 20% above 95th percentile
# #                     for feature in model.features
# #                 },
# #                 "Extreme Values": {
# #                     feature: historical_data[feature].max() * 2  # Double the maximum observed value
# #                     for feature in model.features
# #                 },
# #                 "Mixed Extreme": {  # Some values normal, some extreme
# #                     feature: (
# #                         historical_data[feature].median() if idx % 2 == 0 
# #                         else historical_data[feature].max() * 2
# #                     )
# #                     for idx, feature in enumerate(model.features)
# #                 }
# #             }
            
# #             # Allow user to select scenarios to compare
# #             selected_scenarios = st.multiselect(
# #                 "Select scenarios to test",
# #                 list(scenarios.keys()),
# #                 default=["Normal Operation"]
# #             )
            
# #             if st.button("Test Selected Scenarios"):
# #                 cols = st.columns(len(selected_scenarios))
                
# #                 for idx, (scenario_name, col) in enumerate(zip(selected_scenarios, cols)):
# #                     with col:
# #                         st.subheader(scenario_name)
# #                         scenario_data = pd.DataFrame([scenarios[scenario_name]])
# #                         prediction, probability, warnings = model.predict(scenario_data)
                        
# #                         # Display prediction
# #                         if prediction[0] == 1:
# #                             st.error("⚠️ Failure Likely")
# #                             if warnings:
# #                                 st.write("Warning Details:")
# #                                 for feature, details in warnings.items():
# #                                     severity_pct = details['severity'] * 100
# #                                     st.warning(
# #                                         f"{feature}:\n"
# #                                         f"Value: {details['value']:.2f}\n"
# #                                         f"Severity: {severity_pct:.1f}%\n"
# #                                         f"Normal range: {details['normal_range'][0]:.2f} to {details['normal_range'][1]:.2f}"
# #                                     )
# #                         else:
# #                             st.success("✅ Normal")
                        
# #                         # Display probability gauge with more detailed steps
# #                         fig = go.Figure(go.Indicator(
# #                             mode="gauge+number",
# #                             value=probability[0][1] * 100,
# #                             domain={'x': [0, 1], 'y': [0, 1]},
# #                             title={'text': "Failure Probability (%)"},
# #                             gauge={
# #                                 'axis': {'range': [0, 100]},
# #                                 'bar': {'color': "darkred"},
# #                                 'steps': [
# #                                     {'range': [0, 20], 'color': "lightgreen"},
# #                                     {'range': [20, 40], 'color': "lime"},
# #                                     {'range': [40, 60], 'color': "yellow"},
# #                                     {'range': [60, 80], 'color': "orange"},
# #                                     {'range': [80, 100], 'color': "red"}
# #                                 ],
# #                                 'threshold': {
# #                                     'line': {'color': "black", 'width': 4},
# #                                     'thickness': 0.75,
# #                                     'value': probability[0][1] * 100
# #                                 }
# #                             }
# #                         ))
                        
# #                         # Add more detailed gauge configuration
# #                         fig.update_layout(
# #                             height=250,
# #                             margin=dict(l=10, r=10, t=50, b=10),
# #                             font={'size': 16}
# #                         )
                        
# #                         st.plotly_chart(fig, use_container_width=True)
                        
# #                         # Display severity breakdown
# #                         if any(details['severity'] > 0 for details in warnings.values()):
# #                             st.write("Parameter Status:")
# #                             cols = st.columns(2)
# #                             for idx, (feature, details) in enumerate(warnings.items()):
# #                                 with cols[idx % 2]:
# #                                     if details['severity'] > 0:
# #                                         color = (
# #                                             "🔴" if details['severity'] > 70 else
# #                                             "🟡" if details['severity'] > 30 else
# #                                             "🟢"
# #                                         )
# #                                         st.write(f"{color} {feature}:")
# #                                         st.write(f"Current: {details['value']:.2f}")
# #                                         st.write(f"Severity: {details['severity']:.1f}%")
# #                                         st.write(f"Normal range: {details['normal_range'][0]:.2f} to {details['normal_range'][1]:.2f}")
# #         else:
# #             st.error("Historical data not available. Please ensure data files are present.")
    
# #     with test_tab2:
# #         st.subheader("Custom Test Cases")
# #         st.write("Enter custom values (one set per line) in CSV format:")
# #         st.write("Format: " + ",".join(model.features))
        
# #         if historical_data is not None:
# #             example_normal = ",".join([str(historical_data[f].median()) for f in model.features])
# #             example_critical = ",".join([str(historical_data[f].quantile(0.95)) for f in model.features])
            
# #             st.caption("Example format (copy and modify these):")
# #             st.code(f"Normal values:\n{example_normal}\n\nCritical values:\n{example_critical}")
        
# #         batch_input = st.text_area("Enter test values (CSV format)", height=150)
        
# #         if st.button("Test Custom Values"):
# #             try:
# #                 input_lines = [line.strip() for line in batch_input.split('\n') if line.strip()]
# #                 test_cases = []
                
# #                 for line in input_lines:
# #                     values = [float(x.strip()) for x in line.split(',')]
# #                     if len(values) == len(model.features):
# #                         test_cases.append(dict(zip(model.features, values)))
# #                     else:
# #                         st.warning(f"Skipping invalid line: {line} (wrong number of values)")
                
# #                 if test_cases:
# #                     cols = st.columns(min(len(test_cases), 4))
                    
# #                     for idx, (test_case, col) in enumerate(zip(test_cases, cols)):
# #                         with col:
# #                             st.subheader(f"Test Case {idx + 1}")
# #                             test_data = pd.DataFrame([test_case])
# #                             prediction, probability, feature_scores = model.predict(test_data)
                            
# #                             if prediction[0] == 1:
# #                                 st.error("⚠️ Failure Likely")
# #                             else:
# #                                 st.success("✅ Normal")
                            
# #                             # Display probability gauge with more detailed steps
# #                             fig = go.Figure(go.Indicator(
# #                                 mode="gauge+number",
# #                                 value=probability[0][1] * 100,
# #                                 domain={'x': [0, 1], 'y': [0, 1]},
# #                                 title={'text': "Failure Probability (%)"},
# #                                 gauge={
# #                                     'axis': {'range': [0, 100]},
# #                                     'bar': {'color': "darkred"},
# #                                     'steps': [
# #                                         {'range': [0, 20], 'color': "lightgreen"},
# #                                         {'range': [20, 40], 'color': "lime"},
# #                                         {'range': [40, 60], 'color': "yellow"},
# #                                         {'range': [60, 80], 'color': "orange"},
# #                                         {'range': [80, 100], 'color': "red"}
# #                                     ],
# #                                     'threshold': {
# #                                         'line': {'color': "black", 'width': 4},
# #                                         'thickness': 0.75,
# #                                         'value': probability[0][1] * 100
# #                                     }
# #                                 }
# #                             ))
                            
# #                             # Add more detailed gauge configuration
# #                             fig.update_layout(
# #                                 height=250,
# #                                 margin=dict(l=10, r=10, t=50, b=10),
# #                                 font={'size': 16}
# #                             )
                            
# #                             st.plotly_chart(fig, use_container_width=True)
                            
# #                             # Display severity breakdown
# #                             if any(details['severity'] > 0 for details in feature_scores.values()):
# #                                 st.write("Parameter Status:")
# #                                 cols = st.columns(2)
# #                                 for idx, (feature, details) in enumerate(feature_scores.items()):
# #                                     with cols[idx % 2]:
# #                                         if details['severity'] > 0:
# #                                             color = (
# #                                                 "🔴" if details['severity'] > 70 else
# #                                                 "🟡" if details['severity'] > 30 else
# #                                                 "🟢"
# #                                             )
# #                                             st.write(f"{color} {feature}:")
# #                                             st.write(f"Current: {details['value']:.2f}")
# #                                             st.write(f"Severity: {details['severity']:.1f}%")
# #                                             st.write(f"Normal range: {details['normal_range'][0]:.2f} to {details['normal_range'][1]:.2f}")
            
# #             except Exception as e:
# #                 st.error(f"Error processing input: {str(e)}")
# #                 st.write("Please make sure your input follows the correct format.")

# # # Footer
# # st.markdown("---")
# # st.markdown("Developed with ❤️ for Machine Health Monitoring")

# # # Add a new section in your dashboard after the test scenarios
# # st.header("Advanced Visualizations")
# # viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Real-time Monitoring", "Pattern Analysis", "Correlation Analysis"])

# # with viz_tab1:
# #     st.subheader("Real-time Parameter Monitoring")
    
# #     # Create multi-parameter line chart
# #     try:
# #         # Load historical data
# #         historical_data = pd.read_csv(TELEMETRY_URL)
# #         historical_data['datetime'] = pd.to_datetime(historical_data['datetime'])
        
# #         # Create time series plot with multiple parameters
# #         fig = make_subplots(rows=2, cols=2,
# #                            subplot_titles=model.features,
# #                            shared_xaxes=True)
        
# #         row, col = 1, 1
# #         for feature in model.features:
# #             # Get threshold values for this feature
# #             threshold = model.thresholds[feature]
            
# #             # Add parameter line
# #             fig.add_trace(
# #                 go.Scatter(x=historical_data['datetime'], 
# #                           y=historical_data[feature],
# #                           name=feature,
# #                           line=dict(width=2)),
# #                 row=row, col=col
# #             )
            
# #             # Add threshold lines
# #             fig.add_trace(
# #                 go.Scatter(x=historical_data['datetime'],
# #                           y=[threshold['upper']] * len(historical_data),
# #                           name=f'{feature} Upper Threshold',
# #                           line=dict(dash='dash', color='red', width=1)),
# #                 row=row, col=col
# #             )
            
# #             fig.add_trace(
# #                 go.Scatter(x=historical_data['datetime'],
# #                           y=[threshold['lower']] * len(historical_data),
# #                           name=f'{feature} Lower Threshold',
# #                           line=dict(dash='dash', color='red', width=1)),
# #                 row=row, col=col
# #             )
            
# #             col += 1
# #             if col > 2:
# #                 col = 1
# #                 row += 1
        
# #         fig.update_layout(height=800, showlegend=True,
# #                          title_text="Parameter Trends with Thresholds")
# #         st.plotly_chart(fig, use_container_width=True)
        
# #     except Exception as e:
# #         st.error(f"Error loading historical data: {str(e)}")

# # with viz_tab2:
# #     st.subheader("Pattern Analysis")
    
# #     # Create distribution plots
# #     if historical_data is not None:
# #         # Select parameter for analysis
# #         selected_param = st.selectbox("Select Parameter for Analysis", model.features)
        
# #         col1, col2 = st.columns(2)
        
# #         with col1:
# #             # Create distribution plot
# #             fig = go.Figure()
# #             fig.add_trace(go.Histogram(
# #                 x=historical_data[selected_param],
# #                 name="Distribution",
# #                 nbinsx=50,
# #                 opacity=0.7
# #             ))
            
# #             # Add threshold lines
# #             threshold = model.thresholds[selected_param]
# #             fig.add_vline(x=threshold['lower'], 
# #                          line_dash="dash", 
# #                          line_color="red",
# #                          annotation_text="Lower Threshold")
# #             fig.add_vline(x=threshold['upper'], 
# #                          line_dash="dash", 
# #                          line_color="red",
# #                          annotation_text="Upper Threshold")
            
# #             fig.update_layout(
# #                 title=f"{selected_param} Distribution",
# #                 xaxis_title=selected_param,
# #                 yaxis_title="Frequency"
# #             )
# #             st.plotly_chart(fig, use_container_width=True)
        
# #         with col2:
# #             # Create box plot
# #             fig = go.Figure()
# #             fig.add_trace(go.Box(
# #                 y=historical_data[selected_param],
# #                 name=selected_param,
# #                 boxpoints='outliers'
# #             ))
# #             fig.update_layout(
# #                 title=f"{selected_param} Box Plot",
# #                 yaxis_title=selected_param
# #             )
# #             st.plotly_chart(fig, use_container_width=True)

# # with viz_tab3:
# #     st.subheader("Correlation Analysis")
    
# #     if historical_data is not None:
# #         # Calculate correlation matrix
# #         corr_matrix = historical_data[model.features].corr()
        
# #         # Create heatmap
# #         fig = go.Figure(data=go.Heatmap(
# #             z=corr_matrix,
# #             x=model.features,
# #             y=model.features,
# #             colorscale='RdBu',
# #             zmin=-1, zmax=1,
# #             text=np.round(corr_matrix, 2),
# #             texttemplate='%{text}',
# #             textfont={"size": 10},
# #             hoverongaps=False
# #         ))
        
# #         fig.update_layout(
# #             title="Parameter Correlation Matrix",
# #             height=600,
# #             width=800
# #         )
        
# #         st.plotly_chart(fig, use_container_width=True)
        
# #         # Add parameter relationship scatter plot
# #         st.subheader("Parameter Relationships")
# #         col1, col2 = st.columns(2)
# #         with col1:
# #             param1 = st.selectbox("Select First Parameter", model.features, key="param1")
# #         with col2:
# #             param2 = st.selectbox("Select Second Parameter", model.features, key="param2")
        
# #         fig = go.Figure(data=go.Scatter(
# #             x=historical_data[param1],
# #             y=historical_data[param2],
# #             mode='markers',
# #             marker=dict(
# #                 size=8,
# #                 color=historical_data['datetime'].astype(np.int64),
# #                 colorscale='Viridis',
# #                 showscale=True,
# #                 colorbar=dict(title="Time")
# #             )
# #         ))
        
# #         fig.update_layout(
# #             title=f"Relationship between {param1} and {param2}",
# #             xaxis_title=param1,
# #             yaxis_title=param2,
# #             height=600
# #         )
        
# #         st.plotly_chart(fig, use_container_width=True)

# # # Add a download section for the visualizations
# # st.header("Export Data")
# # if st.button("Generate Report"):
# #     # Create a PDF or Excel report with all visualizations
# #     buffer = io.BytesIO()
    
# #     # Create Excel writer object
# #     with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
# #         # Write summary statistics
# #         summary_stats = historical_data[model.features].describe()
# #         summary_stats.to_excel(writer, sheet_name='Summary Statistics')
        
# #         # Write correlation matrix
# #         corr_matrix.to_excel(writer, sheet_name='Correlations')
        
# #         # Write recent readings
# #         recent_data = historical_data.tail(100)
# #         recent_data.to_excel(writer, sheet_name='Recent Readings')
    
# #     st.download_button(
# #         label="Download Analysis Report",
# #         data=buffer,
# #         file_name="machine_health_report.xlsx",
# #         mime="application/vnd.ms-excel"
# #     )



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
# def load_historical_data(_model):
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
#             data = model.load_data(TELEMETRY_URL, FAILURES_URL)
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
                        
#                         # Display probability gauge with unique key
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
                        
#                         fig.update_layout(
#                             height=250,
#                             margin=dict(l=10, r=10, t=50, b=10),
#                             font={'size': 16}
#                         )
                        
#                         st.plotly_chart(fig, use_container_width=True, key=f"scenario_gauge_{scenario_name}")
                        
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
                            
#                             # Display probability gauge with unique key
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
                            
#                             fig.update_layout(
#                                 height=250,
#                                 margin=dict(l=10, r=10, t=50, b=10),
#                                 font={'size': 16}
#                             )
                            
#                             st.plotly_chart(fig, use_container_width=True, key=f"custom_gauge_{idx}")
                            
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
#         st.plotly_chart(fig, use_container_width=True, key="param_monitoring")
        
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
#             # Create distribution plot with unique key
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
#             st.plotly_chart(fig, use_container_width=True, key=f"dist_plot_{selected_param}")
        
#         with col2:
#             # Create box plot with unique key
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
#             st.plotly_chart(fig, use_container_width=True, key=f"box_plot_{selected_param}")

# with viz_tab3:
#     st.subheader("Correlation Analysis")
    
#     if historical_data is not None:
#         # Calculate correlation matrix
#         corr_matrix = historical_data[model.features].corr()
        
#         # Create heatmap with unique key
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
        
#         st.plotly_chart(fig, use_container_width=True, key="correlation_heatmap")
        
#         # Add parameter relationship scatter plot with unique key
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
        
#         st.plotly_chart(fig, use_container_width=True, key=f"scatter_{param1}_{param2}")

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
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# confusion_matrix, seaborn, matplotlib.pyplot are imported but not used in the main flow here
# Keep them if your full PredictiveMaintenanceModel or training includes them
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import datetime
import io
import joblib # For loading the .joblib model
from huggingface_hub import hf_hub_download # To download from Hugging Face
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # Assuming StandardScaler might be used
from sklearn.ensemble import RandomForestClassifier # Default model for local training fallback



file_id = "1duTtnefd7JafuQGiyCQ22Z-c4HM7gqmZ"
TELEMETRY_URL = f"https://drive.google.com/uc?id={file_id}"
# --- Configuration: User MUST verify/update these ---
# Option 1: Local file paths (ensure files are in the same directory or provide full path)
# TELEMETRY_URL = "https://drive.google.com/uc?id=1duTtnefd7JafuQGiyCQ22Z-c4HM7gqmZ"
FAILURES_URL = "https://drive.google.com/uc?id=1sIZmHQDmCqjjJ6yuNTgm3FWrLToDEOwq"
# ERRORS_URL = 'PdM_errors.csv' # Add if your model uses these
# MAINT_URL = 'PdM_maint.csv'  # Add if your model uses these

# Hugging Face Model Details
HF_REPO_ID = "ishitawarke/predictive_maintenance_model"
HF_FILENAME = "predictive_maintenance_model.joblib"
LOCAL_MODEL_SAVE_PATH = "predictive_maintenance_model_dashboard.joblib"

# --- PredictiveMaintenanceModel Class Definition ---
class PredictiveMaintenanceModel:
    def __init__(self):
        self.model = None  # The actual scikit-learn model
        self.scaler = None # Scaler if used
        # IMPORTANT: Default features. Update if your HF model uses different ones.
        # This list will be overridden if 'features' are found in the loaded joblib.
        self.features = ['volt', 'rotate', 'pressure', 'vibration']
        self.thresholds = {} # For feature warnings and visualizations
        self.target_variable = 'failure_label' # Or whatever your target column is named after processing

    def _load_joblib_object(self, path):
        """Loads object from joblib file and tries to set model, scaler, features."""
        try:
            loaded_object = joblib.load(path)
            if isinstance(loaded_object, dict):
                self.model = loaded_object.get('model')
                self.scaler = loaded_object.get('scaler')
                # If features were saved with the HF model, use them
                saved_features = loaded_object.get('features')
                if saved_features:
                    self.features = saved_features
            else: # Assuming it's just the scikit-learn model
                self.model = loaded_object
                # Scaler and features would need to be handled/defined separately if not in joblib
                st.info("Loaded model directly. Scaler and feature list assumed to be handled externally or by defaults.")

            if self.model is None:
                st.error(f"Failed to extract a valid model object from {path}.")
                return False
            return True
        except Exception as e:
            st.error(f"Error loading from joblib path {path}: {e}")
            return False

    def initialize_and_load_model(self, historical_data_df, hf_repo_id, hf_filename, local_fallback_path):
        """
        Tries to load model from Hugging Face, then local.
        Calculates thresholds if model loaded and historical_data_df is provided.
        """
        loaded_successfully = False
        # 1. Try Hugging Face
        try:
            st.info(f"Attempting to download model from Hugging Face: {hf_repo_id}/{hf_filename}")
            model_path_hf = hf_hub_download(repo_id=hf_repo_id, filename=hf_filename)
            if self._load_joblib_object(model_path_hf):
                st.success("Model successfully loaded from Hugging Face!")
                loaded_successfully = True
            else:
                st.warning("Could not load a valid model object from the Hugging Face file.")
        except Exception as e:
            st.warning(f"Could not load model from Hugging Face: {e}. Trying local fallback.")

        # 2. Try Local Fallback if HF failed
        if not loaded_successfully:
            if self._load_joblib_object(local_fallback_path):
                st.success(f"Model successfully loaded from local path: {local_fallback_path}")
                loaded_successfully = True
            else:
                st.info(f"Local model at {local_fallback_path} not found or failed to load.")

        if loaded_successfully and historical_data_df is not None and not historical_data_df.empty:
            self._calculate_thresholds(historical_data_df)
            if not self.features and self.model: # If features still not set, try to infer or warn
                st.warning("Model features not explicitly set or loaded. Ensure defaults are correct or visualizations/inputs might be affected.")
        elif loaded_successfully:
            st.warning("Historical data not available for threshold calculation. Threshold-based warnings/visuals may be limited.")
        return loaded_successfully

    def _calculate_thresholds(self, data_df):
        """Calculates lower and upper thresholds for each feature based on historical data."""
        if not self.features:
            st.error("Features not set. Cannot calculate thresholds.")
            return
        if data_df is None or data_df.empty:
            st.error("Historical data for threshold calculation is empty or None.")
            return

        for col in self.features:
            if col in data_df.columns:
                # Check if data for the column is all NaN or empty
                if data_df[col].isnull().all() or data_df[col].empty:
                    st.warning(f"No valid data for feature '{col}' to calculate thresholds. Using default [0, 100].")
                    self.thresholds[col] = {'lower': 0, 'upper': 100, 'median': 50}
                else:
                    self.thresholds[col] = {
                        'lower': data_df[col].quantile(0.01), # 1st percentile
                        'upper': data_df[col].quantile(0.99),  # 99th percentile
                        'median': data_df[col].median()
                    }
            else:
                st.warning(f"Feature '{col}' for threshold calculation not found in historical data. Using default [0,100].")
                self.thresholds[col] = {'lower': 0, 'upper': 100, 'median': 50}

    def load_data(self, telemetry_url, failures_url, errors_url=None, maint_url=None):
        """Loads and merges telemetry, failures, etc. for training."""
        try:
            telemetry = pd.read_csv(telemetry_url)
            failures = pd.read_csv(failures_url)
        except FileNotFoundError as e:
            st.error(f"Required data file not found: {e.filename}. Ensure '{telemetry_url}' and '{failures_url}' exist.")
            return None
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None


        telemetry['datetime'] = pd.to_datetime(telemetry['datetime'])
        failures['datetime'] = pd.to_datetime(failures['datetime'])

        # Label failures (predict 1 if failure occurs within next 24 hours for a machine)
        labeled_data = telemetry.copy()
        labeled_data[self.target_variable] = 0

        # Sort by machineID and datetime to correctly apply rolling windows or lookaheads if needed
        labeled_data = labeled_data.sort_values(by=['machineID', 'datetime'])

        # For each failure, mark preceding telemetry records for that machine
        for _, fail_row in failures.iterrows():
            machine_mask = labeled_data['machineID'] == fail_row['machineID']
            time_mask = (labeled_data['datetime'] >= fail_row['datetime'] - pd.Timedelta(hours=24)) & \
                        (labeled_data['datetime'] < fail_row['datetime'])
            labeled_data.loc[machine_mask & time_mask, self.target_variable] = 1
        return labeled_data

    def preprocess_data(self, data):
        """Preprocesses data: feature selection, scaling, splitting for training."""
        if data is None or data.empty:
            st.error("Data for preprocessing is empty or None.")
            return None, None, None, None

        # Drop unnecessary columns (if any) before selecting features
        cols_to_drop = ['datetime', 'machineID', 'model'] # 'model' is often categorical
        data_processed = data.drop(columns=[col for col in cols_to_drop if col in data.columns], errors='ignore')

        # Ensure all defined features are present, if not, it's an issue for training
        missing_features_for_training = [f for f in self.features if f not in data_processed.columns]
        if missing_features_for_training:
            st.error(f"Training data is missing required features: {', '.join(missing_features_for_training)}")
            # Fallback: use available numeric columns if features were not set well. Risky.
            # self.features = [col for col in data_processed.columns if data_processed[col].dtype in [np.int64, np.float64] and col != self.target_variable]
            # st.warning(f"Attempting to use features: {self.features}")
            return None, None, None, None


        if self.target_variable not in data_processed.columns:
            st.error(f"Target variable '{self.target_variable}' not found in the processed data.")
            return None, None, None, None

        X = data_processed[self.features]
        y = data_processed[self.target_variable]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Initialize and fit scaler ONLY during training
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Convert back to DataFrame to keep column names for model
        X_train_df = pd.DataFrame(X_train_scaled, columns=self.features, index=X_train.index)
        X_test_df = pd.DataFrame(X_test_scaled, columns=self.features, index=X_test.index)

        return X_train_df, X_test_df, y_train, y_test

    def train(self, X_train, y_train):
        """Trains a new RandomForestClassifier model."""
        if X_train is None or y_train is None:
            st.error("Training data (X_train or y_train) is None. Cannot train model.")
            return
        # Using RandomForestClassifier as a default, can be changed
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        self.model.fit(X_train, y_train)
        st.success("Model training complete.")

    def predict(self, input_data_df):
        """Makes predictions and calculates feature warnings."""
        if self.model is None:
            st.error("Model not loaded or trained.")
            return np.array([0]), np.array([[1.0, 0.0]]), {} # Default non-failure prediction
        if not self.features:
            st.error("Model features not defined. Cannot make predictions.")
            return np.array([0]), np.array([[1.0, 0.0]]), {}

        # Ensure input_data_df has all the necessary features
        missing_cols = [f for f in self.features if f not in input_data_df.columns]
        if missing_cols:
            st.error(f"Input data is missing features: {', '.join(missing_cols)}")
            return np.array([0]), np.array([[1.0, 0.0]]), {}

        input_features_df = input_data_df[self.features]

        if self.scaler:
            data_scaled = self.scaler.transform(input_features_df)
        else:
            st.warning("Scaler not available. Using raw input data for prediction. Ensure data is pre-scaled if necessary.")
            data_scaled = input_features_df.values # .values to ensure numpy array

        # Ensure data_scaled is a DataFrame with correct feature names if model expects it (some sklearn models do)
        data_scaled_df = pd.DataFrame(data_scaled, columns=self.features, index=input_features_df.index)

        predictions = self.model.predict(data_scaled_df)
        probabilities = self.model.predict_proba(data_scaled_df)

        # Generate warnings for the first row of input_data_df (common for single test case UI)
        warnings = {}
        first_row_values = input_features_df.iloc[0]
        for feature_name in self.features:
            value = first_row_values[feature_name]
            normal_range = (self.thresholds.get(feature_name, {}).get('lower', 0),
                            self.thresholds.get(feature_name, {}).get('upper', 100))
            severity = 0
            # Simplified severity calculation (0 to 100)
            if value < normal_range[0] and normal_range[0] != 0 : # Avoid division by zero if lower is 0
                 #severity = min(100, (normal_range[0] - value) / (normal_range[0] - self.thresholds.get(feature_name,{}).get('min_observed', normal_range[0]*0.5) ) * 100 )
                 severity = abs(value - normal_range[0]) / (abs(normal_range[0]) * 0.5 + 1e-6) * 50 # crude scale
                 severity = min(100, max(0, severity))
            elif value > normal_range[1] and normal_range[1] !=0:
                 #severity = min(100, (value - normal_range[1]) / (self.thresholds.get(feature_name,{}).get('max_observed', normal_range[1]*1.5) - normal_range[1]) * 100 )
                 severity = abs(value - normal_range[1]) / (abs(normal_range[1]) * 0.5 + 1e-6) * 50 # crude scale
                 severity = min(100, max(0, severity))


            warnings[feature_name] = {
                'value': value,
                'normal_range': normal_range,
                'severity': severity # Severity as a percentage
            }
        return predictions, probabilities, warnings


    def save_model(self, path=LOCAL_MODEL_SAVE_PATH):
        """Saves the current model, scaler, and features to a local joblib file."""
        if self.model:
            model_to_save = {'model': self.model, 'scaler': self.scaler, 'features': self.features}
            joblib.dump(model_to_save, path)
            st.success(f"Model (and scaler/features if available) saved to {path}")
        else:
            st.error("No model available to save.")

# --- Streamlit App Code ---

# Must be the first Streamlit command
st.set_page_config(
    page_title="Machine Health Monitoring Dashboard",
    page_icon="⚙️",
    layout="wide"
)

@st.cache_data # Caching for historical data loading
def load_historical_data_from_url(telemetry_url_param):
    try:
        df = pd.read_csv(telemetry_url_param)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    except FileNotFoundError:
        st.error(f"Telemetry data file not found at '{telemetry_url_param}'. Please check the path/URL and ensure the file exists.")
        return None
    except Exception as e:
        st.error(f"Error loading telemetry data from {telemetry_url_param}: {str(e)}")
        return None

# Load historical data (used for thresholds, visualizations, and training fallback)
# This is loaded once and passed around.
historical_data = load_historical_data_from_url(TELEMETRY_URL)


@st.cache_resource # Caching for the model object
def get_predictive_model(_historical_data_ref): # Pass historical data for threshold calculation
    pdm_model_instance = PredictiveMaintenanceModel()
    model_loaded = pdm_model_instance.initialize_and_load_model(
        historical_data_df=_historical_data_ref,
        hf_repo_id=HF_REPO_ID,
        hf_filename=HF_FILENAME,
        local_fallback_path=LOCAL_MODEL_SAVE_PATH
    )
    if model_loaded:
        return pdm_model_instance
    return None

# Load the model wrapper instance
# The 'model' variable will now be an instance of PredictiveMaintenanceModel
model = get_predictive_model(historical_data)


# Sidebar
st.sidebar.title("Machine Health Monitoring")
st.sidebar.markdown("---")
if model and model.model: # Check if the actual sklearn model is loaded inside the wrapper
    st.sidebar.success(f"Model loaded. Features: {', '.join(model.features)}")
    st.sidebar.json({f: f"{v['lower']:.2f}-{v['upper']:.2f}" for f,v in model.thresholds.items() if v},
                    expanded=False) # Show thresholds if calculated
else:
    st.sidebar.warning("Model not loaded.")


# Main content
st.title("Machine Health Monitoring Dashboard")
st.markdown("---")


if model is None or model.model is None: # If the wrapper or its internal model isn't loaded
    st.warning("Model not found or failed to load from Hugging Face / local. You can train a new one if data is available.")
    if historical_data is not None and not historical_data.empty:
        if st.button("Train New Model Locally"):
            with st.spinner("Training model..."):
                # Create a new instance for training to ensure fresh state
                training_model_instance = PredictiveMaintenanceModel()
                
                # Load data for training
                # Make sure FAILURES_URL is correctly defined at the top
                raw_training_data = training_model_instance.load_data(TELEMETRY_URL, FAILURES_URL)

                if raw_training_data is not None and not raw_training_data.empty:
                    X_train, X_test, y_train, y_test = training_model_instance.preprocess_data(raw_training_data)

                    if X_train is not None: # Preprocessing was successful
                        training_model_instance.train(X_train, y_train)
                        training_model_instance._calculate_thresholds(historical_data) # Calculate thresholds for the new model

                        # Calculate and display model metrics for the newly trained model
                        st.header("Newly Trained Model Performance Metrics")
                        # Use the internal scikit-learn model for direct prediction if needed for metrics
                        y_pred_train = training_model_instance.model.predict(X_test)

                        accuracy = accuracy_score(y_test, y_pred_train)
                        precision = precision_score(y_test, y_pred_train, zero_division=0)
                        recall = recall_score(y_test, y_pred_train, zero_division=0)
                        f1 = f1_score(y_test, y_pred_train, zero_division=0)

                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Accuracy", f"{accuracy:.2%}")
                        col2.metric("Precision", f"{precision:.2%}")
                        col3.metric("Recall", f"{recall:.2%}")
                        col4.metric("F1 Score", f"{f1:.2%}")

                        training_model_instance.save_model() # Save it locally
                        st.success("Model trained and saved locally successfully!")
                        st.info("Please RERUN the app to use the newly trained model through the standard loading process.")
                        st.rerun()
                    else:
                        st.error("Model training failed due to issues in data preprocessing.")
                else:
                    st.error("Could not load data for training. Please check TELEMETRY_URL and FAILURES_URL.")
    else:
        st.error(f"Historical data ('{TELEMETRY_URL}') is missing or empty. Cannot train a new model or calculate thresholds.")

else: # Model is loaded and ready
    # Real-time monitoring section
    st.header("Real-time Monitoring")
    if not model.features:
        st.warning("Model features are not defined. Real-time monitoring input cannot be generated.")
    else:
        col_rt1, col_rt2, col_rt3 = st.columns(3)
        input_values = {}
        features_per_column_rt = (len(model.features) + 2) // 3 # Ensure distribution
        
        current_col_list_rt = [col_rt1, col_rt2, col_rt3]
        for i, feature in enumerate(model.features):
            col_index_rt = i // features_per_column_rt
            with current_col_list_rt[col_index_rt]:
                default_val = model.thresholds.get(feature, {}).get('median', 50.0)
                input_values[feature] = st.number_input(
                    f"Input {feature}",
                    min_value=float(-1e6), # Consider dynamic min/max from historical_data if desired
                    max_value=float(1e6),
                    value=float(default_val),
                    key=f"rt_input_{feature}"
                )
        
        if st.button("Predict Machine Status", key="rt_predict_button"):
            input_df_rt = pd.DataFrame([input_values])
            prediction_rt, probability_rt, warnings_rt = model.predict(input_df_rt)

            if prediction_rt[0] == 1:
                st.error("⚠️ Failure Likely based on real-time input.")
            else:
                st.success("✅ Normal Operation Expected based on real-time input.")

            # Display probability gauge for real-time
            fig_rt_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability_rt[0][1] * 100, # Probability of class 1 (failure)
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Failure Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 20], 'color': "lightgreen"}, {'range': [20, 50], 'color': "yellow"},
                        {'range': [50, 80], 'color': "orange"}, {'range': [80, 100], 'color': "red"}
                    ],
                }
            ))
            fig_rt_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_rt_gauge, use_container_width=True)

            if warnings_rt and any(details['severity'] > 0 for details in warnings_rt.values()):
                st.subheader("Parameter Status & Warnings (Real-time Input):")
                warning_cols_rt = st.columns(min(len(model.features), 3)) # Max 3 cols for warnings
                warn_idx_rt = 0
                for feature_name_rt, details_rt in warnings_rt.items():
                     if details_rt['severity'] > 0: # Only show if there's some severity
                        with warning_cols_rt[warn_idx_rt % min(len(model.features), 3)]:
                            color_icon_rt = "🟢"
                            if details_rt['severity'] > 70: color_icon_rt = "🔴"
                            elif details_rt['severity'] > 30: color_icon_rt = "🟡"

                            st.markdown(f"**{color_icon_rt} {feature_name_rt}**")
                            st.write(f"Value: {details_rt['value']:.2f}")
                            st.write(f"Normal: {details_rt['normal_range'][0]:.2f} - {details_rt['normal_range'][1]:.2f}")
                            st.write(f"Severity: {details_rt['severity']:.1f}%")
                            warn_idx_rt += 1


    # Test Scenarios Section
    st.header("Test Scenarios")
    test_tab1, test_tab2 = st.tabs(["Predefined Scenarios", "Custom Test Cases"])

    with test_tab1:
        st.subheader("Test Predefined Scenarios")
        if historical_data is not None and not historical_data.empty and model.features:
            # Create example scenarios
            scenarios = {
                "Normal Operation": {
                    feature: model.thresholds.get(feature, {}).get('median', historical_data[feature].median() if feature in historical_data else 50.0)
                    for feature in model.features
                },
                "Warning Signs": {
                    feature: model.thresholds.get(feature, {}).get('upper', historical_data[feature].quantile(0.90) if feature in historical_data else 75.0) * 1.05 # Slightly above upper
                    for feature in model.features
                },
                "Critical Condition": {
                    feature: model.thresholds.get(feature, {}).get('upper', historical_data[feature].quantile(0.95) if feature in historical_data else 90.0) * 1.2
                    for feature in model.features
                }
            }
            # Add more scenarios from your original code if desired
            selected_scenarios_names = st.multiselect(
                "Select scenarios to test",
                list(scenarios.keys()),
                default=["Normal Operation"]
            )

            if st.button("Test Selected Scenarios"):
                if not selected_scenarios_names:
                    st.warning("Please select at least one scenario to test.")
                else:
                    num_selected_scenarios = len(selected_scenarios_names)
                    scenario_cols = st.columns(num_selected_scenarios)

                    for i, scenario_name in enumerate(selected_scenarios_names):
                        with scenario_cols[i]:
                            st.subheader(scenario_name)
                            scenario_input_values = scenarios[scenario_name]
                            scenario_df = pd.DataFrame([scenario_input_values])
                            prediction, probability, warnings = model.predict(scenario_df)

                            if prediction[0] == 1:
                                st.error("⚠️ Failure Likely")
                            else:
                                st.success("✅ Normal")

                            fig_gauge = go.Figure(go.Indicator(
                                mode="gauge+number", value=probability[0][1] * 100,
                                title={'text': "Failure Probability (%)"},
                                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "darkred"},
                                       'steps': [{'range': [0,20], 'color':'lightgreen'}, {'range': [20,40], 'color':'lime'},
                                                 {'range': [40,60], 'color':'yellow'}, {'range': [60,80], 'color':'orange'},
                                                 {'range': [80,100], 'color':'red'}]}))
                            fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10), font={'size': 12})
                            st.plotly_chart(fig_gauge, use_container_width=True, key=f"scenario_gauge_{scenario_name.replace(' ', '_')}")

                            if warnings and any(details['severity'] > 0 for details in warnings.values()):
                                st.write("Parameter Status:")
                                for feature, details in warnings.items():
                                    if details['severity'] > 0:
                                        st.warning(f"{feature}: Val={details['value']:.2f}, Sev={details['severity']:.1f}%, Range=({details['normal_range'][0]:.2f}-{details['normal_range'][1]:.2f})")
        else:
            st.error("Historical data or model features not available for predefined scenarios.")

    with test_tab2:
        st.subheader("Custom Test Cases")
        if model.features:
            st.write("Enter custom values (one set per line) in CSV format.")
            st.write(f"Format: {','.join(model.features)}")

            if historical_data is not None and not historical_data.empty:
                example_normal_values = ",".join([
                    f"{model.thresholds.get(f, {}).get('median', historical_data[f].median() if f in historical_data else 50.0):.2f}"
                    for f in model.features
                ])
                st.caption(f"Example (Normal): {example_normal_values}")

            batch_input_csv = st.text_area("Enter test values (CSV format)", height=150, key="custom_csv_input")

            if st.button("Test Custom Values"):
                if not batch_input_csv.strip():
                    st.warning("Please enter some CSV data to test.")
                else:
                    try:
                        input_lines = [line.strip() for line in batch_input_csv.split('\n') if line.strip()]
                        test_cases_list = []
                        valid_lines = 0
                        for line_idx, line_str in enumerate(input_lines):
                            values_str = [v.strip() for v in line_str.split(',')]
                            if len(values_str) == len(model.features):
                                try:
                                    values_float = [float(v) for v in values_str]
                                    test_cases_list.append(dict(zip(model.features, values_float)))
                                    valid_lines +=1
                                except ValueError:
                                    st.warning(f"Skipping line {line_idx+1}: Contains non-numeric values. ({line_str})")
                            else:
                                st.warning(f"Skipping line {line_idx+1}: Incorrect number of values. Expected {len(model.features)}, got {len(values_str)}. ({line_str})")
                        
                        if test_cases_list:
                            st.info(f"Processing {len(test_cases_list)} valid test case(s)...")
                            # For simplicity, process and display one by one, or adapt to batch if model.predict handles it well
                            # Here, we'll show results for each in columns
                            num_test_cases = len(test_cases_list)
                            custom_test_cols = st.columns(min(num_test_cases, 4)) # Max 4 columns

                            for idx, test_case_dict in enumerate(test_cases_list):
                                with custom_test_cols[idx % min(num_test_cases, 4)]:
                                    st.subheader(f"Test Case {idx + 1}")
                                    test_data_df = pd.DataFrame([test_case_dict])
                                    prediction, probability, feature_scores = model.predict(test_data_df)

                                    if prediction[0] == 1:
                                        st.error("⚠️ Failure Likely")
                                    else:
                                        st.success("✅ Normal")

                                    fig_custom_gauge = go.Figure(go.Indicator(
                                        mode="gauge+number", value=probability[0][1] * 100,
                                        title={'text': "Prob. (%)"},
                                        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "darkred"},
                                               'steps': [{'range': [0,20], 'color':'lightgreen'}, {'range': [20,40], 'color':'lime'},
                                                         {'range': [40,60], 'color':'yellow'}, {'range': [60,80], 'color':'orange'},
                                                         {'range': [80,100], 'color':'red'}]}))
                                    fig_custom_gauge.update_layout(height=200, margin=dict(l=5, r=5, t=30, b=5), font={'size': 10})
                                    st.plotly_chart(fig_custom_gauge, use_container_width=True, key=f"custom_gauge_{idx}")

                                    if feature_scores and any(details['severity'] > 0 for details in feature_scores.values()):
                                        st.write("Status:")
                                        for feature, details in feature_scores.items():
                                            if details['severity'] > 0:
                                                st.markdown(f"<small><b>{feature}:</b> V={details['value']:.1f}, S={details['severity']:.0f}%</small>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error processing custom input: {str(e)}")
        else:
            st.warning("Model features not defined. Cannot process custom CSV tests.")

    # --- Advanced Visualizations ---
    st.header("Advanced Visualizations")
    if historical_data is not None and not historical_data.empty and model and model.features and model.thresholds:
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Time Series Monitoring", "Pattern Analysis", "Correlation Analysis"])

        with viz_tab1:
            st.subheader("Parameter Time Series with Thresholds")
            # This section uses historical_data directly, which is already loaded
            # Create time series plot with multiple parameters
            num_features_viz = len(model.features)
            if num_features_viz > 0:
                n_cols_viz = 2 # Fixed 2 columns for subplots
                n_rows_viz = (num_features_viz + n_cols_viz - 1) // n_cols_viz

                # Select machineID for focused plotting
                available_machines = historical_data['machineID'].unique()
                selected_machine_id_viz = st.selectbox(
                    "Select MachineID for Time Series",
                    available_machines,
                    key="machine_select_viz_ts"
                )
                machine_data_viz = historical_data[historical_data['machineID'] == selected_machine_id_viz]


                fig_ts = make_subplots(rows=n_rows_viz, cols=n_cols_viz,
                                   subplot_titles=model.features,
                                   shared_xaxes=True, vertical_spacing=0.1)
                current_row, current_col = 1, 1
                for feature_name_viz in model.features:
                    if feature_name_viz in machine_data_viz.columns:
                        fig_ts.add_trace(
                            go.Scatter(x=machine_data_viz['datetime'], y=machine_data_viz[feature_name_viz],
                                      name=feature_name_viz, line=dict(width=2)),
                            row=current_row, col=current_col
                        )
                        # Add threshold lines
                        if feature_name_viz in model.thresholds:
                            thresh_lower = model.thresholds[feature_name_viz]['lower']
                            thresh_upper = model.thresholds[feature_name_viz]['upper']
                            fig_ts.add_hline(y=thresh_upper, line_dash="dash", line_color="red", row=current_row, col=current_col)
                            fig_ts.add_hline(y=thresh_lower, line_dash="dash", line_color="orange", row=current_row, col=current_col)
                    current_col += 1
                    if current_col > n_cols_viz:
                        current_col = 1
                        current_row += 1
                fig_ts.update_layout(height=max(250 * n_rows_viz, 400), showlegend=False,
                                     title_text=f"Parameter Trends for MachineID {selected_machine_id_viz} with Thresholds")
                st.plotly_chart(fig_ts, use_container_width=True, key="param_monitoring_ts_plot")
            else:
                st.info("No features available for time series plotting.")

        with viz_tab2:
            st.subheader("Distribution Analysis")
            if model.features:
                selected_param_dist = st.selectbox("Select Parameter for Distribution Analysis", model.features, key="param_dist_select_viz")
                if selected_param_dist and selected_param_dist in historical_data.columns:
                    col_hist, col_box = st.columns(2)
                    with col_hist:
                        fig_hist = px.histogram(historical_data, x=selected_param_dist, nbins=50, title=f"{selected_param_dist} Distribution")
                        if selected_param_dist in model.thresholds:
                            fig_hist.add_vline(x=model.thresholds[selected_param_dist]['lower'], line_dash="dash", line_color="orange", annotation_text="Lower")
                            fig_hist.add_vline(x=model.thresholds[selected_param_dist]['upper'], line_dash="dash", line_color="red", annotation_text="Upper")
                        st.plotly_chart(fig_hist, use_container_width=True)
                    with col_box:
                        fig_box = px.box(historical_data, y=selected_param_dist, title=f"{selected_param_dist} Box Plot")
                        st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.info("No features available for distribution analysis.")

        with viz_tab3:
            st.subheader("Correlation Analysis")
            if model.features and all(f in historical_data.columns for f in model.features):
                corr_matrix_viz = historical_data[model.features].corr()
                fig_corr = px.imshow(corr_matrix_viz, text_auto=".2f", aspect="auto",
                                     color_continuous_scale='RdBu_r', range_color=[-1,1],
                                     title="Feature Correlation Matrix")
                st.plotly_chart(fig_corr, use_container_width=True)

                st.subheader("Parameter Relationships (Scatter Plot)")
                scatter_col1, scatter_col2 = st.columns(2)
                with scatter_col1:
                    param1_scatter = st.selectbox("X-axis Parameter", model.features, index=0, key="param1_scatter_viz")
                with scatter_col2:
                    param2_options = [f for f in model.features if f != param1_scatter]
                    if not param2_options and model.features : param2_options = [model.features[0]] # if only one feature
                    
                    param2_scatter = st.selectbox("Y-axis Parameter", param2_options, index=min(1, len(param2_options)-1) if len(param2_options)>1 else 0, key="param2_scatter_viz")

                if param1_scatter and param2_scatter :
                    # Sample data for performance if dataset is large
                    sample_df_scatter = historical_data.sample(min(len(historical_data), 2000))
                    fig_scatter = px.scatter(sample_df_scatter, x=param1_scatter, y=param2_scatter,
                                            title=f"Relationship: {param1_scatter} vs {param2_scatter}",
                                            marginal_y="violin", marginal_x="box", trendline="ols",
                                            color_discrete_sequence=px.colors.qualitative.Plotly)
                    st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("Not enough features or data for correlation analysis.")
    else:
        st.info("Advanced visualizations require loaded historical data and a configured model with features and thresholds.")


    # Add a download section for the visualizations/report
    st.header("Export Data")
    if st.button("Generate Analysis Report (Excel)"):
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            if model.features and all(f in historical_data.columns for f in model.features):
                summary_stats = historical_data[model.features].describe()
                summary_stats.to_excel(writer, sheet_name='Summary Statistics')

                corr_matrix_export = historical_data[model.features].corr()
                corr_matrix_export.to_excel(writer, sheet_name='Correlations')
            else:
                 pd.DataFrame(["Features for report not fully available in historical data."]).to_excel(writer, sheet_name='Error')

            # Include a sample of recent data (if historical_data has 'datetime')
            if 'datetime' in historical_data.columns:
                recent_data = historical_data.sort_values(by='datetime', ascending=False).head(100)
            else: # Fallback if no datetime
                recent_data = historical_data.head(100)
            recent_data.to_excel(writer, sheet_name='Recent Readings Sample', index=False)

        buffer.seek(0) # Rewind the buffer
        st.download_button(
            label="Download Analysis Report (.xlsx)",
            data=buffer,
            file_name=f"machine_health_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ for Machine Health Monitoring | Model integration POC")

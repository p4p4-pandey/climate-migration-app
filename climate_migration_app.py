"""
Climate Migration Prediction System
This app predicts migration patterns based on climate data using AI/ML
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="Climate Migration Predictor",
    page_icon="🌍",
    layout="wide"
)

# Title and description
st.title("🌍 AI-Powered Climate Migration Predictor")
st.markdown("""
This system uses AI to predict potential migration patterns based on climate data.
It analyzes temperature, precipitation, and sea level data to estimate migration risk.
""")

# Sidebar for API key input
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("""
**Get your free API key from:**
[OpenWeatherMap](https://openweathermap.org/api)

1. Sign up for free
2. Go to API keys section
3. Copy your key and paste below
""")

# API key input (stored in session state for security)
api_key = st.sidebar.text_input("OpenWeather API Key", type="password", 
                                help="Your API key is not stored permanently")

# Sample data generator for training the AI model
@st.cache_data
def generate_training_data():
    """
    Generate synthetic training data for the AI model
    In a real project, this would come from historical climate and migration databases
    """
    np.random.seed(42)
    n_samples = 1000
    
    # Climate features
    temperature_change = np.random.uniform(-2, 5, n_samples)  # Temperature change in °C
    precipitation_change = np.random.uniform(-50, 50, n_samples)  # % change
    sea_level_rise = np.random.uniform(0, 30, n_samples)  # cm
    extreme_events = np.random.randint(0, 20, n_samples)  # Number of events per year
    
    # Migration risk calculation (this is our "AI" learning this pattern)
    # Higher temp change, more extreme events = higher migration risk
    migration_risk = (
        temperature_change * 15 +
        abs(precipitation_change) * 0.3 +
        sea_level_rise * 2 +
        extreme_events * 3 +
        np.random.normal(0, 5, n_samples)  # Some random variation
    )
    
    # Normalize to 0-100 scale
    migration_risk = np.clip(migration_risk, 0, 100)
    
    df = pd.DataFrame({
        'temperature_change': temperature_change,
        'precipitation_change': precipitation_change,
        'sea_level_rise': sea_level_rise,
        'extreme_events': extreme_events,
        'migration_risk': migration_risk
    })
    
    return df

# Train the AI model
@st.cache_resource
def train_migration_model():
    """
    Train a Random Forest model to predict migration risk
    This is our AI component
    """
    df = generate_training_data()
    
    X = df[['temperature_change', 'precipitation_change', 'sea_level_rise', 'extreme_events']]
    y = df['migration_risk']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_scaled, y)
    
    return model, scaler

# Fetch live climate data from API
def fetch_climate_data(city, api_key):
    """
    Fetch real-time climate data from OpenWeatherMap API
    """
    if not api_key:
        return None
    
    try:
        # Current weather data
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'description': data['weather'][0]['description'],
                'city': data['name'],
                'country': data['sys']['country']
            }
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Main app layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📊 Climate Input Parameters")
    
    # Option to use live data or manual input
    input_mode = st.radio("Input Mode:", ["Manual Entry", "Live Climate Data"])
    
    if input_mode == "Live Climate Data":
        city = st.text_input("Enter City Name", "London")
        
        if st.button("Fetch Live Data") and api_key:
            climate_data = fetch_climate_data(city, api_key)
            
            if climate_data:
                st.success(f"✅ Data fetched for {climate_data['city']}, {climate_data['country']}")
                st.info(f"Current: {climate_data['temperature']}°C, {climate_data['description']}")
                
                # Store in session state
                st.session_state['current_temp'] = climate_data['temperature']
            else:
                st.error("Could not fetch data. Check your API key and city name.")
        
        # Use fetched data or defaults
        current_temp = st.session_state.get('current_temp', 15)
        st.info(f"Using temperature: {current_temp}°C")
        
        # Simulate future changes based on current conditions
        temp_change = st.slider("Projected Temperature Increase (°C)", 0.0, 5.0, 2.0, 0.1)
        precip_change = st.slider("Projected Precipitation Change (%)", -50, 50, 0, 5)
        
    else:
        temp_change = st.slider("Temperature Change (°C)", -2.0, 5.0, 2.0, 0.1,
                               help="Expected change in average temperature")
        precip_change = st.slider("Precipitation Change (%)", -50, 50, -10, 5,
                                 help="Expected change in rainfall patterns")
    
    sea_level = st.slider("Sea Level Rise (cm)", 0, 30, 10, 1,
                         help="Expected rise in sea level")
    extreme_events = st.slider("Extreme Weather Events/Year", 0, 20, 5, 1,
                              help="Number of floods, droughts, storms, etc.")

with col2:
    st.header("🤖 AI Prediction Results")
    
    # Train model
    model, scaler = train_migration_model()
    
    # Prepare input for prediction
    input_data = np.array([[temp_change, precip_change, sea_level, extreme_events]])
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    prediction = np.clip(prediction, 0, 100)  # Ensure 0-100 range
    
    # Display prediction with color coding
    if prediction < 30:
        risk_level = "Low"
        color = "green"
    elif prediction < 60:
        risk_level = "Medium"
        color = "orange"
    else:
        risk_level = "High"
        color = "red"
    
    st.metric("Migration Risk Score", f"{prediction:.1f}/100", 
             delta=f"{risk_level} Risk", delta_color="inverse")
    
    # Visual gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Migration Risk Level"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "lightyellow"},
                {'range': [60, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    st.subheader("📋 Interpretation")
    
    if prediction < 30:
        st.success("""
        **Low Migration Risk**: Current climate conditions suggest stable population patterns.
        Minor adaptations may be needed but large-scale migration is unlikely.
        """)
    elif prediction < 60:
        st.warning("""
        **Medium Migration Risk**: Climate changes may cause gradual population shifts.
        Planning for infrastructure and resource management is recommended.
        """)
    else:
        st.error("""
        **High Migration Risk**: Significant climate stress predicted.
        Large-scale population displacement is possible. Urgent adaptation measures needed.
        """)

# Additional insights section
st.header("📈 Feature Importance Analysis")

col3, col4 = st.columns(2)

with col3:
    # Feature importance from the model
    feature_names = ['Temperature Change', 'Precipitation Change', 'Sea Level Rise', 'Extreme Events']
    importances = model.feature_importances_
    
    fig_importance = px.bar(
        x=importances,
        y=feature_names,
        orientation='h',
        title="Which Factors Matter Most?",
        labels={'x': 'Importance', 'y': 'Climate Factor'},
        color=importances,
        color_continuous_scale='Reds'
    )
    st.plotly_chart(fig_importance, use_container_width=True)

with col4:
    # Show comparison with different scenarios
    st.subheader("Scenario Comparison")
    
    scenarios = {
        'Best Case': [0.5, -10, 2, 1],
        'Current Path': [temp_change, precip_change, sea_level, extreme_events],
        'Worst Case': [4.0, -40, 25, 15]
    }
    
    scenario_predictions = {}
    for scenario_name, values in scenarios.items():
        input_scaled = scaler.transform([values])
        pred = model.predict(input_scaled)[0]
        scenario_predictions[scenario_name] = max(0, min(100, pred))
    
    fig_scenarios = px.bar(
        x=list(scenario_predictions.keys()),
        y=list(scenario_predictions.values()),
        title="Different Climate Scenarios",
        labels={'x': 'Scenario', 'y': 'Migration Risk'},
        color=list(scenario_predictions.values()),
        color_continuous_scale='RdYlGn_r'
    )
    st.plotly_chart(fig_scenarios, use_container_width=True)

# Information footer
st.markdown("---")
st.markdown("""
### ℹ️ About This System

**How it works:**
1. **Data Collection**: Gathers climate data (temperature, precipitation, sea level, extreme events)
2. **AI Processing**: Uses a Random Forest algorithm trained on climate-migration patterns
3. **Risk Prediction**: Calculates migration risk score based on climate stress indicators
4. **Visualization**: Presents results in an easy-to-understand format

**Technologies Used:**
- **Python**: Core programming language
- **Streamlit**: Interactive web interface
- **Scikit-learn**: AI/Machine Learning algorithms
- **OpenWeatherMap API**: Live climate data
- **Plotly**: Interactive visualizations

**Note**: This is a demonstration project. Real-world predictions require comprehensive datasets
including socioeconomic factors, political stability, and historical migration patterns.
""")

# Debug info (only if in development)
with st.expander("🔧 Technical Details"):
    st.write("**Model Information:**")
    st.write(f"- Algorithm: Random Forest Regressor")
    st.write(f"- Number of trees: 100")
    st.write(f"- Training samples: 1000")
    st.write(f"- Input features: 4 climate variables")
    st.write(f"- Output: Migration risk score (0-100)")

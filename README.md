# 🌍 Climate Migration Prediction System

An AI-powered web application that predicts climate-induced migration patterns using machine learning and live climate data.

## 📋 What This Project Does

This system analyzes climate factors to predict the risk of climate-induced migration from a region:
- **Temperature changes**
- **Precipitation patterns**
- **Sea level rise**
- **Extreme weather events**

It uses a **Random Forest AI model** to calculate a migration risk score (0-100) and visualizes the results.

## 🚀 How to Run the Project

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Get Your FREE API Key
1. Go to [OpenWeatherMap.org](https://openweathermap.org/api)
2. Click "Sign Up" (free account)
3. After login, go to "API keys" section
4. Copy your API key (looks like: `a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6`)

**Important**: Keep your API key private! Never share it publicly or commit it to GitHub.

### Step 3: Run the Application
```bash
streamlit run climate_migration_app.py
```

The app will open in your browser at `http://localhost:8501`

### Step 4: Using the App
1. Paste your API key in the sidebar (it won't be saved)
2. Choose "Live Climate Data" or "Manual Entry"
3. Adjust the climate parameters
4. See the AI prediction instantly!

## 🔐 Where to Keep Your API Key

**For Development (Recommended):**
- Enter it directly in the app sidebar each time
- The app doesn't save it permanently (secure!)

**For Production (Advanced):**
Create a file called `.env` in the same folder:
```
OPENWEATHER_API_KEY=your_api_key_here
```

Then add this to your code (top of the file):
```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENWEATHER_API_KEY')
```

And install: `pip install python-dotenv`

**⚠️ IMPORTANT**: Add `.env` to `.gitignore` so it's never uploaded to GitHub!

## 🎤 Presentation Guide

### Introduction (1 minute)
"Hello everyone! Today I'm presenting a Climate Migration Prediction System that uses AI to forecast population movements caused by climate change. This is becoming increasingly important as climate change accelerates globally."

### Problem Statement (30 seconds)
"Climate change is forcing millions to migrate. By 2050, an estimated 200 million people could be displaced. We need tools to predict where and when this might happen."

### Solution Overview (1 minute)
"My system uses:
1. **Real-time climate data** from OpenWeatherMap API
2. **Machine Learning** (Random Forest algorithm) to analyze patterns
3. **Interactive visualization** to make predictions accessible

The AI was trained on climate-migration correlations and can predict migration risk scores from 0-100."

### Live Demo (2-3 minutes)
1. Open the app
2. Show the live climate data feature
   - "Here I'm fetching real-time data for London"
3. Adjust parameters
   - "Let's see what happens if temperature rises by 3°C"
4. Explain the results
   - "The AI predicts a 'High Risk' scenario with score of 75/100"
5. Show different scenarios
   - "Compare best case vs worst case climate futures"

### Technical Highlights (1 minute)
"Key technologies:
- **Frontend**: Streamlit (Python-based web framework)
- **Backend**: Python with scikit-learn for ML
- **AI Model**: Random Forest with 100 decision trees
- **Data Source**: OpenWeatherMap API for live climate data
- **Visualization**: Plotly for interactive charts"

### Real-World Applications (30 seconds)
"This could help:
- Governments plan infrastructure for climate refugees
- NGOs allocate resources efficiently
- Researchers identify high-risk regions
- Policymakers make data-driven decisions"

### Challenges & Learnings (30 seconds)
"Challenges I faced:
- Integrating the API securely
- Training an accurate AI model
- Making complex data simple to understand

What I learned:
- How machine learning models work
- API integration best practices
- Data visualization techniques"

### Conclusion (30 seconds)
"This project demonstrates how AI can address real-world climate challenges. While simplified, it shows the potential of combining live data with machine learning for social good. Thank you!"

### Questions to Anticipate

**Q: How accurate is your model?**
A: "This is a demonstration using synthetic training data. A production system would need historical climate and migration data from sources like UN agencies and climate research centers. The Random Forest algorithm is proven for this type of prediction."

**Q: Why Random Forest?**
A: "Random Forest handles non-linear relationships well, which is perfect for climate data. It's also interpretable - we can see which features matter most."

**Q: Can this scale globally?**
A: "Yes! The API supports worldwide cities. We'd need to add more factors like economic conditions, political stability, and water availability for comprehensive predictions."

**Q: What about data privacy?**
A: "The app doesn't store any user data. API keys are entered per-session only. For production, we'd use secure environment variables."

**Q: How long did this take?**
A: "About [X hours/days]. Most time went into understanding the climate-migration relationship and making the UI intuitive."

## 📊 Project Structure

```
climate-migration-predictor/
│
├── climate_migration_app.py    # Main application
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🛠️ Technologies Explained

### Frontend: Streamlit
- Python-based web framework
- No HTML/CSS/JavaScript needed
- Perfect for data science projects
- Live updates as you change inputs

### Backend: Python
- **scikit-learn**: Machine learning library
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **requests**: API calls

### AI Model: Random Forest
- Ensemble learning method
- Combines 100 decision trees
- Robust to overfitting
- Shows feature importance

### API: OpenWeatherMap
- Free tier: 1000 calls/day
- Real-time weather data
- Covers 200,000+ cities worldwide
- Simple REST API

## 🎯 Key Features

1. ✅ **Live Climate Data Integration**
2. ✅ **AI-Powered Predictions**
3. ✅ **Interactive Visualizations**
4. ✅ **Multiple Input Modes**
5. ✅ **Scenario Comparison**
6. ✅ **Feature Importance Analysis**
7. ✅ **Risk Level Categorization**

## 🔮 Future Enhancements

- Add more data sources (precipitation, humidity, wind patterns)
- Include socioeconomic factors (GDP, unemployment, education)
- Historical migration data integration
- Map visualization showing migration routes
- Multi-city comparison feature
- Export reports as PDF

## 📚 Resources Used

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [OpenWeatherMap API Docs](https://openweathermap.org/api)
- [Climate Migration Research](https://www.worldbank.org/en/topic/climatechange/brief/groundswell-report)

## 🤝 Credits

Built as a demonstration of AI applications in climate science.

**Disclaimer**: This is an educational project. Real climate migration predictions require comprehensive datasets and expert validation.

---

Good luck with your presentation! 🚀

# 🌋 Landslide Prediction AI

An AI-based landslide prediction project that estimates the probability and intensity of a landslide for a selected location using geographical and environmental data.

I built this project to explore how machine learning can be used with real-world geographical data to identify areas that may be more vulnerable to landslides.


## About the Project

Landslides are affected by many different factors, including rainfall, slope, soil, terrain, land cover and elevation. Instead of looking at these factors individually, this project combines them and uses machine learning to estimate the landslide risk for a particular location.

The main idea is simple:

**Select a location on the map → collect the relevant data → process the features → predict landslide probability.**

The current system is mainly focused on locations in India.


## What Can It Do?

The current version of the project allows a user to:

- Select a location directly on the map
- Use the location's latitude and longitude
- Collect geographical and environmental information
- Analyse factors such as rainfall, soil, terrain and slope
- Generate a landslide probability
- Estimate the intensity/risk of a landslide
- View the result through a web interface


## How I Built It

The project started with collecting information about previous landslides and the environmental conditions associated with them.

I created a dataset using information collected from different publicly available sources and APIs, including data related to:

- ISRO
- USGS
- GSI
- NASA

The final dataset contains information related to approximately **13,000 previous landslide events**.

I then combined different geographical and environmental features and used them to train machine learning models.


## Features Used

Some of the major features used by the model include:

- Rainfall
- Soil characteristics
- Land cover
- Tree cover
- Terrain
- Slope
- Elevation
- Hydrology
- Latitude
- Longitude
- Historical landslide information

The idea was to combine as many meaningful geographical factors as possible rather than relying on a single variable.


## Machine Learning Models

I experimented with two main machine learning models:

### Random Forest

The initial Random Forest model achieved approximately:

**89% accuracy**

### XGBoost

I then developed an XGBoost model and achieved approximately:

**94% accuracy**

Based on the current testing results, XGBoost performs better than the Random Forest model and is the preferred model in the current version.


> These results are based on my current dataset and train/test split. Model performance can change with different data and testing conditions.


## How the System Works

The basic workflow of the project is:

User selects a location
        ↓
Latitude & Longitude
        ↓
Geographical / Environmental Data
        ↓
Feature Processing
        ↓
Machine Learning Model
        ↓
Landslide Probability
        ↓
Risk / Intensity Result





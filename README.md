# Delhi-AQI-Model: Analyzing the Impact of Punjab's Agricultural Fires on Delhi's Air Quality

## Overview

This project aims to analyze and model the relationship between agricultural fires in the state of Punjab and the subsequent impact on the Air Quality Index (AQI) in Delhi. During the post-monsoon season, large-scale stubble burning in Punjab is widely believed to be a major contributor to the severe air pollution experienced in the National Capital Region (NCR). 

This project uses data-driven methods to investigate and quantify this correlation by integrating air quality data, satellite-detected fire incidents, and meteorological data.

## Data Sources

The analysis is based on three primary datasets for the year 2024:

1.  **Delhi Air Quality**: Historical AQI data for 21 monitoring stations across Delhi. The raw data is processed to create a clean, time-series dataset for 2024.
    -   **File:** `data/delhi_air_quality_2024.csv`

2.  **Punjab Fire Incidents**: Satellite data of fire incidents from NASA's Fire Information for Resource Management System (FIRMS). The global dataset is filtered to isolate fires within the geographical boundaries of Punjab (Latitude: 29.5° to 32.5°, Longitude: 73.5° to 76.9°).
    -   **File:** `data/punjab_fires_2024.csv`

3.  **Punjab Wind Data**: Historical hourly wind speed data for the Punjab region. This is used to analyze the potential for wind to carry pollutants from Punjab towards Delhi.
    -   **File:** `data/wind_speed_punjab_2024.csv`

## Methodology

The initial data preparation and cleaning are performed in the `scratch.ipynb` notebook. The key preprocessing steps are:

1.  **Load Datasets**: The raw CSV files for Delhi's air quality, NASA's fire archives, and Punjab's wind speed are loaded into pandas DataFrames.
2.  **Temporal Filtering**: Each dataset is filtered to include only records from the year 2024, ensuring a consistent timeframe for the analysis.
3.  **Geospatial Filtering**: The NASA fire data is filtered to include only incidents within the geographical coordinates corresponding to the state of Punjab.
4.  **Export Processed Data**: The cleaned and filtered DataFrames are saved as new CSV files in the `data/` directory, ready for exploratory data analysis (EDA) and model building.

## Project Structure

```
.
├── data/
│   ├── delhi_air_quality_2024.csv          # Processed Delhi AQI data for 2024
│   ├── punjab_fires_2024.csv               # Processed Punjab fire data for 2024
│   └── wind_speed_punjab_2024.csv          # Processed Punjab wind speed data for 2024
│   ├── ... (other raw data files)
├── scratch.ipynb                           # Jupyter Notebook for data preprocessing and analysis
├── requirements.txt                        # Project dependencies
└── README.md                               # This file
```

## Getting Started

### Prerequisites

- Python 3.10+
- Jupyter Notebook or JupyterLab

### Installation

1.  Clone the repository to your local machine:
    ```sh
    git clone https://github.com/your-username/Delhi-AQI-Model.git
    cd Delhi-AQI-Model
    ```

2.  Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

### Usage

The data preprocessing steps are documented and can be executed by running the `scratch.ipynb` notebook in a Jupyter environment.

## Next Steps

With the data preprocessed, the following steps are planned:

-   **Exploratory Data Analysis (EDA)**: Create visualizations to identify trends, such as plotting daily fire counts in Punjab against daily AQI levels in Delhi.
-   **Feature Engineering**: Create new features, such as a "fire intensity" metric (e.g., combining fire count and brightness) and lag features to account for the time it takes for pollutants to travel.
-   **Correlation and Causation Analysis**: Use statistical methods to test the hypothesis that Punjab fires lead to increased AQI in Delhi, considering factors like wind direction.
-   **Predictive Modeling**: Build a machine learning model (e.g., Linear Regression, Gradient Boosting, or a time-series model like ARIMA/LSTM) to predict Delhi's AQI based on fire data and meteorological conditions.
-   **Visualization**: Develop a compelling visual narrative of the findings, potentially including maps and time-series plots.
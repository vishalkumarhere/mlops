# YouTube Analytics Dashboard 📊

A powerful analytics dashboard for YouTube channels that provides deep insights into video performance, content analysis, and trend prediction using machine learning.

## Table of Contents
- [Setup](#setup)
- [Features](#features)
- [Usage Guide](#usage-guide)
- [Understanding the Results](#understanding-the-results)
- [Technical Details](#technical-details)

## Setup

1. **Prerequisites**:
   - Python 3.8 or higher
   - A Google Cloud project with YouTube Data API v3 enabled
   - YouTube API key

2. **Installation**:
   ```bash
   # Clone the repository
   git clone <repository-url>
   cd youtube-analytics

   # Install dependencies
   pip install -r requirements.txt

   # Set up environment variables
   # Create a .env file and add your YouTube API key:
   YOUTUBE_API_KEY=your_api_key_here
   ```

## Features

### 1. Channel Overview
- Total subscribers, views, and video count
- Historical growth trends
- Channel performance metrics

### 2. Views Analysis
- Views over time visualization
- Peak performance periods
- Growth trajectory analysis

### 3. Engagement Analysis
- Engagement rate trends
- Like and comment rate analysis
- Correlation between views and engagement
- Regression analysis for performance prediction

### 4. Content Analysis
- Video title sentiment analysis
- Duration optimization insights
- Content categorization
- Topic clustering

### 5. Upload Pattern Analysis
- Best performing upload times
- Day-of-week performance analysis
- Seasonal trends

### 6. Predictions & Insights
- View count prediction
- Engagement rate prediction
- Content category analysis
- Optimal video parameters

### 7. Trend Analysis
- Regional trending topics
- Category performance analysis
- Popular tags and keywords
- Channel benchmarking

## Usage Guide

### Getting Started
1. Launch the application:
   ```bash
   streamlit run app.py
   ```

2. Enter a YouTube channel name in the search box
3. Select the channel from the suggestions
4. Wait for the analysis to complete

### Navigating the Dashboard

#### 1. Views Over Time Tab
- Shows the historical performance of videos
- X-axis: Publication date
- Y-axis: View count
- Use this to identify:
  - Growth trends
  - Viral videos
  - Seasonal patterns

#### 2. Engagement Analysis Tab
- Three scatter plots with regression lines:
  - Views vs Engagement Rate
  - Views vs Like Rate
  - Views vs Comment Rate
- R² values indicate correlation strength
- Use this to understand:
  - Audience interaction patterns
  - Content effectiveness
  - Viewer behavior trends

#### 3. Content Analysis Tab
- Sentiment Analysis:
  - Box plot shows sentiment distribution
  - Percentage breakdown of positive/negative/neutral titles
  - Green: Positive sentiment
  - Gray: Neutral sentiment
  - Red: Negative sentiment

- Duration Analysis:
  - Scatter plot with regression line
  - Statistics for optimal video length
  - Most successful duration ranges

#### 4. Upload Patterns Tab
- Best Upload Hours:
  - Bar chart showing average views by hour
  - Number of uploads per time slot
  - Optimal posting times

- Best Upload Days:
  - Performance by day of week
  - Upload frequency analysis
  - Weekly patterns

#### 5. Predictions & Insights Tab
- View Prediction Model:
  - R² score indicates model accuracy
  - Feature importance chart
  - Key factors affecting views

- Engagement Prediction:
  - Expected engagement rates
  - Important factors for engagement
  - Optimization recommendations

#### 6. Trend Analysis Tab
- Regional Trends:
  - Select different regions to analyze
  - Compare performance across markets
  - Identify regional preferences

- Category Performance:
  - Views by content category
  - Engagement rates per category
  - Top-performing content types

## Understanding the Results

### Metrics Interpretation

1. **Engagement Rate**
   - Formula: (Likes + Comments) / Views × 100
   - Good: > 5%
   - Average: 2-5%
   - Poor: < 2%

2. **Sentiment Scores**
   - Range: -1.0 to 1.0
   - Positive: > 0
   - Neutral: 0
   - Negative: < 0

3. **R² Scores**
   - Range: 0 to 1
   - > 0.7: Strong correlation
   - 0.4-0.7: Moderate correlation
   - < 0.4: Weak correlation

### Model Performance

1. **View Prediction Model**
   - R² score indicates prediction accuracy
   - Feature importance shows key factors
   - Use for content planning

2. **Engagement Prediction**
   - Helps optimize for interaction
   - Identifies engagement drivers
   - Guides content strategy

### Trend Analysis

1. **Regional Insights**
   - Market-specific performance
   - Cultural preferences
   - Time zone considerations

2. **Category Analysis**
   - Best performing content types
   - Audience preferences
   - Niche opportunities

## Technical Details

### Data Collection
- Uses YouTube Data API v3
- Fetches last 50 videos by default
- Updates in real-time

### Machine Learning Models
- Random Forest Regressors
- TF-IDF Vectorization
- K-means Clustering
- Sentiment Analysis using TextBlob

### Error Handling
- Graceful degradation for missing data
- Automatic retry for API failures
- Comprehensive logging

### Performance Optimization
- Batch processing for API calls
- Caching for frequently accessed data
- Efficient data structures

## Best Practices

1. **Content Strategy**
   - Use sentiment analysis for title optimization
   - Follow recommended video durations
   - Post at optimal times

2. **Engagement Optimization**
   - Focus on high-impact features
   - Balance quantity vs. quality
   - Monitor trend changes

3. **Growth Tactics**
   - Leverage regional insights
   - Target high-performing categories
   - Optimize for predicted metrics

## Troubleshooting

Common issues and solutions:
1. API Key errors: Verify .env file configuration
2. No data showing: Check channel name/ID
3. Slow performance: Reduce date range
4. Missing metrics: Ensure public video visibility 
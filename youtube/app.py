import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
from youtube_api import YouTubeAPI
from logging_config import logger
import time
import numpy as np
from scipy import stats
import mlflow

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db"))
if mlflow.active_run():
    mlflow.end_run()
# Set page config
st.set_page_config(
    page_title="YouTube Channel Analytics",
    page_icon="📊",
    layout="wide"
)

# Initialize YouTube API
try:
    youtube_api = YouTubeAPI()
except Exception as e:
    logger.error(f"Failed to initialize YouTube API: {str(e)}")
    st.error("Failed to initialize YouTube API. Please check your API key and try again.")
    st.stop()

def add_regression_line(fig, x, y):
    """Add regression line to scatter plot"""
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line_x = np.array([min(x), max(x)])
    line_y = slope * line_x + intercept
    
    fig.add_trace(go.Scatter(
        x=line_x,
        y=line_y,
        mode='lines',
        name=f'Trend Line (R² = {r_value**2:.3f})',
        line=dict(color='red', dash='dash')
    ))
    return fig

def analyze_channel(channel_name):
    try:
        # Get channel ID from name
        channel_id = youtube_api.get_channel_id_by_name(channel_name)
        if not channel_id:
            st.error("Channel not found. Please check the channel name and try again.")
            return None, None

        # Get channel statistics
        channel_stats = youtube_api.get_channel_stats(channel_id)
        if not channel_stats:
            st.error("Channel not found or inaccessible")
            return None, None

        # Get channel videos
        videos = youtube_api.get_channel_videos(channel_stats['playlist_id'])
        if not videos:
            st.error("No videos found for this channel")
            return None, None

        # Convert to DataFrame
        df = pd.DataFrame(videos)
        df['published_at'] = pd.to_datetime(df['published_at'])
        df['views'] = pd.to_numeric(df['views'])
        df['likes'] = pd.to_numeric(df['likes'])
        df['comments'] = pd.to_numeric(df['comments'])

        # Add additional analytics
        df = youtube_api.analyze_engagement(df)
        df['sentiment'] = youtube_api.analyze_sentiment(df)
        clusters, cluster_terms = youtube_api.analyze_content_clusters(df)
        df['content_cluster'] = clusters
        trending_topics = youtube_api.get_trending_topics(df)
        best_hours, best_days = youtube_api.analyze_upload_patterns(df)

        return channel_stats, df, cluster_terms, trending_topics, best_hours, best_days
    except Exception as e:
        logger.error(f"Error analyzing channel: {str(e)}")
        st.error(f"An error occurred while analyzing the channel: {str(e)}")
        return None, None, None, None, None, None

def display_channel_suggestions(channels):
    if not channels:
        return None
    
    st.write("### Suggested Channels:")
    for channel in channels:
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image(channel['thumbnail'], width=50)
        with col2:
            if st.button(f"Select: {channel['name']}", key=channel['channel_id']):
                return channel['name']
    return None

def main():
    st.title("YouTube Channel Analytics Dashboard 📊")
    
    # Initialize session state for channel suggestions
    if 'suggestions' not in st.session_state:
        st.session_state.suggestions = []
    if 'last_query' not in st.session_state:
        st.session_state.last_query = ""
    if 'last_suggestions_time' not in st.session_state:
        st.session_state.last_suggestions_time = 0
    
    # Input for channel name with debouncing
    channel_name = st.text_input("Enter YouTube Channel Name:", key="channel_input")
    
    # Update suggestions if the input has changed
    current_time = time.time()
    if channel_name != st.session_state.last_query and len(channel_name) >= 2:
        if current_time - st.session_state.last_suggestions_time > 1:  # Debounce for 1 second
            try:
                st.session_state.suggestions = youtube_api.search_channels(channel_name)
                st.session_state.last_suggestions_time = current_time
            except Exception as e:
                logger.error(f"Error fetching channel suggestions: {str(e)}")
                st.session_state.suggestions = []
    
    st.session_state.last_query = channel_name
    
    # Display suggestions
    if channel_name and len(channel_name) >= 2:
        selected_channel = display_channel_suggestions(st.session_state.suggestions)
        if selected_channel:
            channel_name = selected_channel
    
    if channel_name:
        with st.spinner("Analyzing channel data..."):
            channel_stats, df, cluster_terms, trending_topics, best_hours, best_days = analyze_channel(channel_name)
            
            if channel_stats and df is not None:
                # Display channel statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Channel Name", channel_stats['channel_name'])
                with col2:
                    st.metric("Subscribers", f"{int(channel_stats['subscribers']):,}")
                with col3:
                    st.metric("Total Views", f"{int(channel_stats['views']):,}")
                with col4:
                    st.metric("Total Videos", channel_stats['total_videos'])

                # Create tabs for different visualizations
                tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                    "Views Over Time", 
                    "Engagement Analysis", 
                    "Content Analysis",
                    "Upload Patterns",
                    "Predictions & Insights",
                    "Trend Analysis",
                    "Top Videos"
                ])

                with tab1:
                    # Views over time
                    fig_views = px.line(df, x='published_at', y='views',
                                      title='Views Over Time',
                                      labels={'published_at': 'Publication Date', 'views': 'Views'})
                    st.plotly_chart(fig_views, use_container_width=True)

                with tab2:
                    # Engagement metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        fig_engagement = px.scatter(df, x='views', y='engagement_rate',
                                                  title='Views vs Engagement Rate',
                                                  labels={'views': 'Views', 'engagement_rate': 'Engagement Rate (%)'})
                        fig_engagement = add_regression_line(fig_engagement, df['views'], df['engagement_rate'])
                        st.plotly_chart(fig_engagement, use_container_width=True)
                    with col2:
                        fig_likes = px.scatter(df, x='views', y='like_rate',
                                             title='Views vs Like Rate',
                                             labels={'views': 'Views', 'like_rate': 'Like Rate (%)'})
                        fig_likes = add_regression_line(fig_likes, df['views'], df['like_rate'])
                        st.plotly_chart(fig_likes, use_container_width=True)
                    with col3:
                        fig_comments = px.scatter(df, x='views', y='comment_rate',
                                                title='Views vs Comment Rate',
                                                labels={'views': 'Views', 'comment_rate': 'Comment Rate (%)'})
                        fig_comments = add_regression_line(fig_comments, df['views'], df['comment_rate'])
                        st.plotly_chart(fig_comments, use_container_width=True)

                with tab3:
                    # Content Analysis
                    col1, col2 = st.columns(2)
                    with col1:
                        # Sentiment Analysis
                        st.subheader("Video Title Sentiment Analysis")
                        fig_sentiment = px.box(df, y='sentiment',
                                             title='Video Title Sentiment Distribution',
                                             labels={'sentiment': 'Sentiment Score'})
                        fig_sentiment.update_layout(
                            yaxis_title="Sentiment Score",
                            showlegend=False
                        )
                        st.plotly_chart(fig_sentiment, use_container_width=True)
                        
                        # Add sentiment statistics
                        positive_titles = len(df[df['sentiment'] > 0])
                        negative_titles = len(df[df['sentiment'] < 0])
                        neutral_titles = len(df[df['sentiment'] == 0])
                        
                        st.write("### Sentiment Breakdown")
                        st.write(f"🟢 Positive Titles: {positive_titles} ({positive_titles/len(df)*100:.1f}%)")
                        st.write(f"🔴 Negative Titles: {negative_titles} ({negative_titles/len(df)*100:.1f}%)")
                        st.write(f"⚪ Neutral Titles: {neutral_titles} ({neutral_titles/len(df)*100:.1f}%)")
                    
                    with col2:
                        # Video Duration Analysis
                        st.subheader("Video Duration Analysis")
                        fig_duration = px.scatter(df, x='duration_minutes', y='views',
                                                title='Video Duration vs Views',
                                                labels={'duration_minutes': 'Duration (minutes)', 'views': 'Views'})
                        fig_duration = add_regression_line(fig_duration, df['duration_minutes'], df['views'])
                        st.plotly_chart(fig_duration, use_container_width=True)
                        
                        # Add duration statistics
                        avg_duration = df['duration_minutes'].mean()
                        median_duration = df['duration_minutes'].median()
                        most_viewed_duration = df.loc[df['views'].idxmax(), 'duration_minutes']
                        
                        st.write("### Duration Statistics")
                        st.write(f"⏱️ Average Duration: {avg_duration:.1f} minutes")
                        st.write(f"⌛ Median Duration: {median_duration:.1f} minutes")
                        st.write(f"🎯 Most Viewed Video Duration: {most_viewed_duration:.1f} minutes")

                with tab4:
                    # Upload Patterns
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Best Upload Hours")
                        fig_hours = go.Figure()
                        fig_hours.add_trace(go.Bar(
                            x=best_hours['hour'],
                            y=best_hours['mean'],
                            text=best_hours['count'].apply(lambda x: f"{x} uploads"),
                            textposition='auto',
                            hovertemplate="Hour: %{x}<br>Average Views: %{y:,.0f}<br>%{text}<extra></extra>"
                        ))
                        fig_hours.update_layout(
                            title="Average Views by Upload Hour",
                            xaxis_title="Hour of Day (24h)",
                            yaxis_title="Average Views",
                            showlegend=False,
                            plot_bgcolor='rgba(0,0,0,0)',
                            bargap=0.2
                        )
                        st.plotly_chart(fig_hours, use_container_width=True)
                    
                    with col2:
                        st.subheader("Best Upload Days")
                        fig_days = go.Figure()
                        fig_days.add_trace(go.Bar(
                            x=best_days['day_of_week'],
                            y=best_days['mean'],
                            text=best_days['count'].apply(lambda x: f"{x} uploads"),
                            textposition='auto',
                            hovertemplate="Day: %{x}<br>Average Views: %{y:,.0f}<br>%{text}<extra></extra>"
                        ))
                        fig_days.update_layout(
                            title="Average Views by Upload Day",
                            xaxis_title="Day of Week",
                            yaxis_title="Average Views",
                            showlegend=False,
                            plot_bgcolor='rgba(0,0,0,0)',
                            bargap=0.2
                        )
                        st.plotly_chart(fig_days, use_container_width=True)

                with tab5:
                    st.header("Predictions & Insights")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("View Prediction Model")
                        view_model = youtube_api.train_view_predictor(df)
                        if view_model:
                            st.write(f"Model Performance (R² Score): {view_model['r2_score']:.3f}")
                            
                            # Plot feature importance
                            fig_view_importance = px.bar(
                                view_model['feature_importance'],
                                x='importance',
                                y='feature',
                                title='Feature Importance for View Prediction',
                                orientation='h'
                            )
                            st.plotly_chart(fig_view_importance, use_container_width=True)
                    
                    with col2:
                        st.subheader("Engagement Prediction Model")
                        engagement_model = youtube_api.predict_engagement(df)
                        if engagement_model:
                            st.write(f"Model Performance (R² Score): {engagement_model['r2_score']:.3f}")
                            
                            # Plot feature importance
                            fig_eng_importance = px.bar(
                                engagement_model['feature_importance'],
                                x='importance',
                                y='feature',
                                title='Feature Importance for Engagement Prediction',
                                orientation='h'
                            )
                            st.plotly_chart(fig_eng_importance, use_container_width=True)
                    
                    # Video Categories Analysis
                    st.subheader("Video Categories Analysis")
                    categories = youtube_api.analyze_video_categories(df)
                    if categories is not None:
                        for _, cluster in categories.iterrows():
                            with st.expander(f"Category {cluster['cluster_id']+1} ({cluster['size']} videos)"):
                                st.write(f"Average Views: {cluster['avg_views']:,.0f}")
                                st.write(f"Average Engagement Rate: {cluster['avg_engagement']:.2f}%")
                                st.write(f"Average Duration: {cluster['avg_duration']:.1f} minutes")
                                st.write("Top Terms:", ", ".join(cluster['top_terms']))
                    
                    # Optimal Parameters
                    st.subheader("Optimal Video Parameters")
                    optimal_params = youtube_api.get_optimal_parameters(df)
                    if optimal_params:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write("### Duration")
                            st.write(f"Optimal Length: {optimal_params['duration']['mean']:.1f} minutes")
                            st.write(f"Range: {optimal_params['duration']['range'][0]:.1f} - {optimal_params['duration']['range'][1]:.1f} minutes")
                        
                        with col2:
                            st.write("### Title")
                            st.write(f"Optimal Length: {int(optimal_params['title_length']['mean'])} characters")
                            st.write(f"Range: {int(optimal_params['title_length']['range'][0])} - {int(optimal_params['title_length']['range'][1])} characters")
                        
                        with col3:
                            st.write("### Best Upload Times")
                            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                            best_days = [days[day] for day in optimal_params['best_days']]
                            st.write("Days:", ", ".join(best_days))
                            st.write("Hours:", ", ".join([f"{hour}:00" for hour in optimal_params['best_hours']]))
                        
                        st.write(f"Target Engagement Rate: {optimal_params['engagement_rate']['mean']:.2f}%")

                with tab6:
                    st.header("Trend Analysis")
                    
                    # Region selection
                    regions = {
                        'US': 'United States',
                        'GB': 'United Kingdom',
                        'CA': 'Canada',
                        'AU': 'Australia',
                        'IN': 'India',
                        'JP': 'Japan',
                        'KR': 'South Korea',
                        'DE': 'Germany',
                        'FR': 'France',
                        'BR': 'Brazil'
                    }
                    selected_region = st.selectbox("Select Region", options=list(regions.keys()), format_func=lambda x: regions[x])
                    
                    # Fetch trending videos
                    with st.spinner("Fetching trending videos..."):
                        trending_videos = youtube_api.get_trending_videos(region_code=selected_region)
                        
                        if trending_videos:
                            # Analyze trends
                            trend_analysis = youtube_api.analyze_trends(trending_videos)
                            
                            if trend_analysis:
                                # Basic metrics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Total Trending Videos", trend_analysis['total_videos'])
                                with col2:
                                    st.metric("Average Views", f"{int(trend_analysis['avg_views']):,}")
                                with col3:
                                    st.metric("Average Engagement", f"{trend_analysis['avg_engagement']:.2f}%")
                                with col4:
                                    st.metric("Average Duration", f"{trend_analysis['avg_duration']:.1f} min")
                                
                                # Time-based patterns
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.subheader("Popular Upload Times")
                                    st.write("Best Hours:", ", ".join([f"{hour}:00" for hour in trend_analysis['popular_hours']]))
                                    st.write("Best Days:", ", ".join(trend_analysis['popular_days']))
                                
                                with col2:
                                    st.subheader("Title Analysis")
                                    st.write(f"Average Title Length: {trend_analysis['avg_title_length']:.1f} characters")
                                    st.write("Common Words in Titles:")
                                    for word, count in trend_analysis['common_title_words'].items():
                                        st.write(f"- {word}: {count} times")
                                
                                # Tag analysis
                                st.subheader("Popular Tags")
                                tag_cols = st.columns(2)
                                tags_list = list(trend_analysis['common_tags'].items())
                                mid_point = len(tags_list) // 2
                                
                                with tag_cols[0]:
                                    for tag, count in tags_list[:mid_point]:
                                        st.write(f"- {tag}: {count} times")
                                with tag_cols[1]:
                                    for tag, count in tags_list[mid_point:]:
                                        st.write(f"- {tag}: {count} times")
                                
                                # Top channels
                                st.subheader("Top Trending Channels")
                                for channel, count in trend_analysis['top_channels'].items():
                                    st.write(f"- {channel}: {count} videos")
                                
                                # Category performance
                                st.subheader("Category Performance")
                                category_df = pd.DataFrame.from_dict(trend_analysis['category_performance'], orient='index')
                                fig_category = px.bar(
                                    category_df,
                                    x=category_df.index,
                                    y='views',
                                    title='Average Views by Category',
                                    labels={'index': 'Category ID', 'views': 'Average Views'}
                                )
                                st.plotly_chart(fig_category, use_container_width=True)
                                
                                # Trend prediction
                                st.subheader("Trend Prediction Model")
                                trend_model = youtube_api.predict_trend_performance(pd.DataFrame(trending_videos))
                                
                                if trend_model:
                                    st.write(f"Model Performance (R² Score): {trend_model['r2_score']:.3f}")
                                    
                                    # Plot feature importance
                                    fig_trend_importance = px.bar(
                                        trend_model['feature_importance'],
                                        x='importance',
                                        y='feature',
                                        title='Feature Importance for Trend Prediction',
                                        orientation='h'
                                    )
                                    st.plotly_chart(fig_trend_importance, use_container_width=True)
                        else:
                            st.error("Unable to fetch trending videos. Please try again later.")

                with tab7:
                    # Top performing videos
                    top_videos = df.nlargest(10, 'views')
                    st.subheader("Top 10 Videos by Views")
                    for idx, row in top_videos.iterrows():
                        with st.expander(f"{row['title']} ({int(row['views']):,} views)"):
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                # Embed video with proper video ID
                                video_url = youtube_api.get_video_url(row['video_id'])
                                st.markdown(
                                    f'<iframe width="100%" height="315" src="{video_url}" frameborder="0" allowfullscreen></iframe>',
                                    unsafe_allow_html=True
                                )
                            with col2:
                                st.write(f"Published: {row['published_at'].strftime('%Y-%m-%d')}")
                                st.write(f"Views: {int(row['views']):,}")
                                st.write(f"Likes: {int(row['likes']):,}")
                                st.write(f"Comments: {int(row['comments']):,}")
                                st.write(f"Engagement Rate: {row['engagement_rate']:.2f}%")
                                st.write(f"Duration: {row['duration_minutes']:.1f} minutes")
                                sentiment_color = 'green' if row['sentiment'] > 0 else ('gray' if row['sentiment'] == 0 else 'red')
                                st.write(f"Sentiment: <span style='color:{sentiment_color}'>{row['sentiment']:.2f}</span>", unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An unexpected error occurred. Please try again later.") 
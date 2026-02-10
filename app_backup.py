
import streamlit as st
import pandas as pd
import altair as alt
import folium
from streamlit_folium import st_folium
import os

# Page Configuration
st.set_page_config(layout="wide", page_title="Pixel Loss Dashboard")

# Constants
# Use absolute path relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "pixel_losses.csv")
WINDOWS = ['apr_may', 'sep_oct', 'oct_dec', 'dec_feb']

@st.cache_data
def load_data():
    """Load and preprocess data."""
    parquet_path = os.path.join(BASE_DIR, "data", "pixel_losses.parquet")
    csv_path = os.path.join(BASE_DIR, "data", "pixel_losses.csv")
    
    if os.path.exists(parquet_path):
        try:
            df = pd.read_parquet(parquet_path)
            return df
        except Exception as e:
            st.error(f"Error reading Parquet file: {e}. Trying CSV...")
    
    if os.path.exists(csv_path):
        try:
            # Enforce dtypes to fix Arrow serialization errors
            dtype_dict = {
                'pixel_id': 'int64',
                'dataset': 'str',
                'window': 'category',
                'year': 'int32',
                'payout': 'float64'
            }
            df = pd.read_csv(csv_path, dtype=dtype_dict)
            
            # Save as Parquet for future speedups
            try:
                df.to_parquet(parquet_path, index=False)
            except Exception as e:
                st.warning(f"Could not save optimization file (Parquet): {e}")
                
            return df
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return pd.DataFrame()
            
    st.error("Data file not found. Ensure 'data/pixel_losses.csv' exists.")
    return pd.DataFrame()

def main():
    st.title("Rio Grande do Sul - Historic Pixel Losses (Apr-Start)")
    
    # Load Data
    df = load_data()
    if df.empty:
        return

    # Ensure dataset is string to avoid Arrow errors
    if 'dataset' in df.columns:
        df['dataset'] = df['dataset'].astype(str)

    # Sidebar Filters
    st.sidebar.header("Filters")
    
    # 1. Map Visualization
    st.subheader("Total Historic Losses by Pixel")
    
    # Aggregate data for map
    map_data = df.groupby(['pixel_id', 'latitude', 'longitude'])['payout'].sum().reset_index()
    map_data.rename(columns={'payout': 'total_loss'}, inplace=True)
    
    # Create Map
    m = folium.Map(location=[-30.0, -53.0], zoom_start=7, tiles="CartoDB positron")
    
    # Add Heatmap/Circles
    # Using CircleMarkers for better interaction
    for _, row in map_data.iterrows():
        # Color based on loss magnitude (simple logic for now)
        color = "#e5f5e0" # Light green (low loss)
        if row['total_loss'] > 50000:
            color = "#a1d99b"
        if row['total_loss'] > 100000:
            color = "#31a354"
        if row['total_loss'] > 200000:
            color = "#006d2c" # Dark green (high loss)
            
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=f"Pixel {int(row['pixel_id'])}: ${row['total_loss']:,.2f}",
            tooltip=f"Pixel {int(row['pixel_id'])}"
        ).add_to(m)

    # Render Map and Capture Click
    # st_folium returns a dict with interaction data
    output = st_folium(m, width=None, height=500)
    
    # 2. Pixel Detail View
    if output['last_object_clicked']:
        # Extract pixel_id from clicked object? 
        # Folium circle markers don't easily return ID in last_object_clicked unless binding popups carefully.
        # Alternative: interactive checks.
        # For simplicity in V1, let's use the lat/lon to find the pixel, or a dropdown.
        # Getting lat/lon from click
        clicked_lat = output['last_object_clicked']['lat']
        clicked_lon = output['last_object_clicked']['lng']
        
        # Find closest pixel
        # Simple distance check
        map_data['dist'] = ((map_data['latitude'] - clicked_lat)**2 + (map_data['longitude'] - clicked_lon)**2)**0.5
        closest_pixel = map_data.loc[map_data['dist'].idxmin()]
        
        selected_pixel_id = closest_pixel['pixel_id']
        st.success(f"Selected Pixel: {int(selected_pixel_id)}")
        
    else:
        # Fallback: Selectbox
        pixel_ids = sorted(df['pixel_id'].unique())
        selected_pixel_id = st.sidebar.selectbox("Select Pixel ID", pixel_ids)

    # Filter data for selected pixel
    pixel_df = df[df['pixel_id'] == selected_pixel_id].copy()
    
    # Ensure all years and windows are present
    all_years = range(1996, 2026)
    full_index = pd.MultiIndex.from_product([all_years, WINDOWS], names=['year', 'window'])
    
    # Set index for reindexing
    if not pixel_df.empty:
        pixel_df = pixel_df.set_index(['year', 'window'])
        pixel_df = pixel_df.reindex(full_index, fill_value=0).reset_index()
    else:
        # Handle case where pixel has NO history (shouldn't happen if filtered correctly)
        pixel_df = pd.DataFrame(index=full_index).reset_index()
        pixel_df['payout'] = 0
        pixel_df['triggered'] = False

    # Restore pixel_id and other constant columns if lost during reindexing with 0 fill
    pixel_df['pixel_id'] = selected_pixel_id
    # triggered might be 0 (False) or True. 
    # payout might be 0.

    col1, col2, col3 = st.columns(3)
    total_payout = pixel_df['payout'].sum()
    max_payout = pixel_df['payout'].max()
    trigger_count = pixel_df[pixel_df['triggered'] == True].shape[0]
    
    col1.metric("Total Historic Payout", f"${total_payout:,.2f}")
    col2.metric("Max Single Year Payout", f"${max_payout:,.2f}")
    col3.metric("Total Trigger Events", trigger_count)

    # 3. Charts
    st.subheader("Loss History by Year and Window")
    
    # Bar Chart: Loss by Year
    chart = alt.Chart(pixel_df).mark_bar().encode(
        x='year:O',
        y='payout:Q',
        color='window:N',
        tooltip=['year', 'window', 'payout', 'triggered']
    ).properties(height=400)
    
    st.altair_chart(chart, use_container_width=True)
    
    # Table of Triggers
    st.subheader("Trigger Events")
    triggers = pixel_df[pixel_df['triggered'] == True].sort_values('year', ascending=False)
    if not triggers.empty:
        st.dataframe(
            triggers[['year', 'window', 'raw_percentile', 'coverage_mult', 'payout']].style.format({
                'raw_percentile': '{:.1f}',
                'payout': '${:,.2f}',
                'coverage_mult': '{:.0%}'
            })
        )
    else:
        st.info("No trigger events for this pixel.")

    # 4. Global Sanity Checks (Fixed)
    st.markdown("---")
    st.subheader("Region-Wide (Portfolio) Statistics")
    st.caption("These statistics allow for a sanity check of the entire portfolio and do not change with pixel selection.")
    
    import numpy as np
    
    # Calculate annual aggregation
    portfolio_annual = df.groupby('year')['payout'].sum()
    
    port_total_loss = portfolio_annual.sum()
    port_mean_loss = portfolio_annual.mean()
    port_max_loss = portfolio_annual.max()
    port_p95_loss = np.percentile(portfolio_annual, 95)
    port_p99_loss = np.percentile(portfolio_annual, 99)
    
    p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
    p_col1.metric("Total Region Loss", f"${port_total_loss:,.2f}")
    p_col2.metric("Mean Annual Loss", f"${port_mean_loss:,.2f}")
    p_col3.metric("Max Annual Loss", f"${port_max_loss:,.2f}")
    p_col4.metric("P95 Annual Loss", f"${port_p95_loss:,.2f}")
    p_col5.metric("P99 Annual Loss", f"${port_p99_loss:,.2f}")

if __name__ == "__main__":
    main()

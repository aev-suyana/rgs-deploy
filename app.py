
import streamlit as st
import pandas as pd
import altair as alt
import folium
from streamlit_folium import st_folium
import os
import json
import warnings

# Suppress FutureWarning for 'observed' parameter in groupby
warnings.simplefilter(action='ignore', category=FutureWarning)

# Page Configuration
st.set_page_config(layout="wide", page_title="Pixel Loss Dashboard")

# Constants
# Use absolute path relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "pixel_losses.csv")
MAPPING_PATH = os.path.join(BASE_DIR, "data", "pixel_municipality_assignment_v3.csv")
# Helper: Load Data
# Updated to use the FINAL aligned clustering (matching MD guide)
CLUSTERS_PATH = os.path.join(BASE_DIR, "data", "municipality_clusters_final.csv")
ERA5_PATH = os.path.join(BASE_DIR, "data", "valid_pixels_era5.csv")
WINDOWS = ['apr_may', 'sep_oct', 'oct_dec', 'dec_feb']

import geopandas as gpd
from shapely.geometry import Point, shape
import numpy as np
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
import altair as alt
from PIL import Image


@st.cache_data
def load_shapefiles():
    """Load and process municipality and cluster shapefiles."""
    # app.py is in .../filled_data/apr_start/web_app/
    # shapefile is in .../filled_data/apr_start/web_app/RS_Municipios_2021/
    
    shp_path = os.path.join(BASE_DIR, "RS_Municipios_2021", "RS_Municipios_2021.shp")
    
    if not os.path.exists(shp_path):
        st.error(f"Shapefile not found at: {shp_path}")
        return None, None

    try:
        # Load Munis
        gdf = gpd.read_file(shp_path)
        # Ensure lat/lon
        if gdf.crs.to_string() != 'EPSG:4326':
            gdf = gdf.to_crs("EPSG:4326")
            
        # Simplify geometry (try 0.001 for better resolution)
        gdf['geometry'] = gdf.simplify(0.001)
        
        # Load Clusters to merge
        if os.path.exists(CLUSTERS_PATH):
            cluster_df = pd.read_csv(CLUSTERS_PATH)
            # Normalize names for merge
            gdf['muni_norm'] = gdf['NM_MUN'].str.lower().str.strip()
            cluster_df['muni_norm'] = cluster_df['muni_name_first'].str.lower().str.strip()
            
            # Merge
            # Use 'cluster_aligned' which corresponds to the MD guide (2=South, 4=West)
            gdf = gdf.merge(cluster_df[['muni_norm', 'cluster_aligned']], on='muni_norm', how='left')
            gdf.rename(columns={'cluster_aligned': 'cluster'}, inplace=True)
            gdf['cluster'] = gdf['cluster'].fillna(-1).astype(int)
        else:
            gdf['cluster'] = -1
            
        # Create Dissolved Cluster Map
        # Filter out NaN clusters/unmatched
        valid_clusters = gdf[gdf['cluster'] != -1]
        
        # Debug: Check if we actually have clusters
        # st.write(f"Valid clusters found: {valid_clusters['cluster'].unique()}")

        if not valid_clusters.empty:
            clusters_gdf = valid_clusters.dissolve(by='cluster').reset_index()
        else:
            clusters_gdf = None
            
        return gdf, clusters_gdf
        
    except Exception as e:
        st.error(f"Error loading shapefiles: {e}")
        return None, None

@st.cache_data
def load_data():
    """Load and preprocess data."""
    parquet_path = os.path.join(BASE_DIR, "data", "pixel_losses.parquet")
    csv_path = os.path.join(BASE_DIR, "data", "pixel_losses.csv")
    
    if os.path.exists(parquet_path):
        try:
            df = pd.read_parquet(parquet_path)
            # Check if we have the merged columns. If not, reload from CSV.
            required_cols = ['muni_name', 'muni_lat', 'latitude', 'longitude']
            if any(col not in df.columns for col in required_cols):
                raise Exception("Missing merged columns in Parquet (stale cache).")
            
            # Enforce types again just in case
            df['dataset'] = df['dataset'].astype(str)
            return df
        except Exception:
            # Silently fail and fallback to CSV
            pass
            # st.error(f"Error reading Parquet file: {e}. Trying CSV...")
    
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
            # Ensure dataset is string (fix Arrow serialization errors)
            if 'dataset' in df.columns:
                df['dataset'] = df['dataset'].astype(str)
            else:
                df['dataset'] = 'chirps' # Default fallback
            
            # --- Load and Merge Mapping Data ---
            if os.path.exists(MAPPING_PATH) and os.path.exists(CLUSTERS_PATH):
                try:
                    # 1. Pixel -> Muni
                    loss_df = df
                    muni_map = pd.read_csv(MAPPING_PATH)
                    
                    # Force pixel_id to int to match loss_df (handle potential NaNs/floats)
                    muni_map['pixel_id'] = pd.to_numeric(muni_map['pixel_id'], errors='coerce').fillna(-1).astype(int)
                    
                    # Keep lat/lon for map plotting
                    muni_map = muni_map[['pixel_id', 'muni_name', 'muni_id', 'latitude', 'longitude']]
                    
                    # 2. Muni -> Cluster
                    cluster_map = pd.read_csv(CLUSTERS_PATH)
                    cluster_map = cluster_map[['muni_name_first', 'cluster_aligned', 'centroid_lon', 'centroid_lat']]
                    cluster_map.rename(columns={
                        'muni_name_first': 'muni_name', 
                        'cluster_aligned': 'cluster',
                        'centroid_lon': 'muni_lon',
                        'centroid_lat': 'muni_lat'
                    }, inplace=True)
                    
                    # 3. Merge
                    # Merge pixel map first
                    # If loss_df already has lat/lon, merge will suffix them.
                    # We prefer the lat/lon from MAPPING_PATH (muni_map) as it is verified 'clean'?
                    # Or valid_pixels_era5 has lat/lon?
                    # The error shows latitude_x and latitude_y.
                    # latitude_x comes from loss_df. latitude_y comes from muni_map.
                    # We'll use Coalesce or just Rename.
                    
                    df_merged = pd.merge(loss_df, muni_map, on='pixel_id', how='left')
                    
                    # Resolve suffixes if present
                    if 'latitude_y' in df_merged.columns:
                        df_merged['latitude'] = df_merged['latitude_y'].fillna(df_merged['latitude_x'])
                        df_merged['longitude'] = df_merged['longitude_y'].fillna(df_merged['longitude_x'])
                        df_merged.drop(columns=['latitude_x', 'longitude_x', 'latitude_y', 'longitude_y'], inplace=True, errors='ignore')
                    
                    # Merge cluster info
                    df_merged = pd.merge(df_merged, cluster_map, on='muni_name', how='left')
                    
                    # Fill NaNs for display
                    df_merged['muni_name'] = df_merged['muni_name'].fillna('Desconhecido')
                    df_merged['cluster'] = df_merged['cluster'].fillna(-1).astype(int)
                    
                    # Filter out summary/sentinel rows (pixel_id < 0) to fix portfolio stats
                    # The user file has pixel_id = -1 for global aggregates
                    # MOVED TO MAIN for clarity and correct df_portfolio creation
                    # df_merged = df_merged[df_merged['pixel_id'] >= 0]
                    
                    df = df_merged
                    
                except Exception as e:
                    st.error(f"Erro CRÍTICO ao carregar arquivos de mapeamento: {e}")
                    st.error(f"Verifique se {MAPPING_PATH} e {CLUSTERS_PATH} estão corretos e legíveis.")
                    st.stop() # Stop execution to prevent cascading errors
            else:
                 st.error(f"Arquivos de mapeamento não encontrados: {MAPPING_PATH} ou {CLUSTERS_PATH}")
                 st.stop()
                 
            # Final Validation
            if 'latitude' not in df.columns or 'longitude' not in df.columns:
                 st.error(f"Erro: Colunas de Latitude/Longitude não foram mescladas corretamente. Colunas atuais: {df.columns.tolist()}")
                 st.stop()

            # Save as Parquet for future speedups
            try:
                df.to_parquet(parquet_path, index=False)
            except Exception as e:
                #st.warning(f"Could not save optimization file (Parquet): {e}")
                pass
                
            return df
        except Exception as e:
            st.error(f"Erro ao ler arquivo CSV: {e}")
            return pd.DataFrame()
            
    st.error("Arquivo de dados não encontrado. Verifique se 'data/pixel_losses.csv' existe.")
    return pd.DataFrame()

@st.cache_data
def load_era5_pixels():
    """Load ERA5 pixel coordinates."""
    if os.path.exists(ERA5_PATH):
        try:
            return pd.read_csv(ERA5_PATH)
        except Exception as e:
            st.error(f"Erro ao carregar ERA5: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def filter_pixels_spatially(df, gdf, lat_col='latitude', lon_col='longitude'):
    """Strictly filter pixels to be within the municipality polygons."""
    if df.empty or gdf.empty:
        return df
    
    # Create GDF from pixels
    # Drop NaNs
    df = df.dropna(subset=[lat_col, lon_col])
    
    pixels_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326"
    )
    
    # Spatial Join (Inner)
    # This keeps only pixels that intersect with the shapes
    valid_pixels = gpd.sjoin(pixels_gdf, gdf[['geometry']], how='inner', predicate='intersects')
    
    # Return valid DF (drop geometry)
    return pd.DataFrame(valid_pixels.drop(columns='geometry'))


def main():
    st.title("Perdas Históricas por Pixel (Rio Grande do Sul)")
    
    # Load Data
    df = load_data()
    if df.empty:
        return
    
    # Load Shapes for Spatial Filtering
    gdf, _ = load_shapefiles()

    # Ensure dataset is string to avoid Arrow errors
    if 'dataset' in df.columns:
        df['dataset'] = df['dataset'].astype(str)
    
    # Filter out summary/sentinel rows (pixel_id < 0) globally
    # This ensures no checks downstream need to worry about -1
    if not df.empty:
        df = df[df['pixel_id'] >= 0]

    # Create a copy for Portfolio Stats (Static / Full Payout Structure)
    # This preserves pixels that might be spatially "stray" (outside shapefile) 
    # but are valid for the financial portfolio calculations.
    df_portfolio = df.copy()

    # Strict Spatial Filter to remove "Stray Pixels" for Visuals/Interaction
    # Filter CHIRPS (df)
    if not df.empty and gdf is not None:
         df = filter_pixels_spatially(df, gdf)
    
    # Load and Filter ERA5
    df_era5 = load_era5_pixels()
    if not df_era5.empty and gdf is not None:
         # Enrich ERA5 with Muni/Cluster info via Spatial Join
         # (We need muni/cluster to filter them for stats)
         era5_gdf = gpd.GeoDataFrame(
            df_era5,
            geometry=gpd.points_from_xy(df_era5.longitude, df_era5.latitude),
            crs="EPSG:4326"
         )
         era5_joined = gpd.sjoin(era5_gdf, gdf[['geometry', 'NM_MUN', 'cluster', 'muni_norm']], how='inner', predicate='intersects')
         df_era5 = pd.DataFrame(era5_joined.drop(columns='geometry'))
         # Rename columns to match df for filtering convenience
         df_era5.rename(columns={'NM_MUN': 'muni_name'}, inplace=True)
    
    # Sidebar Filters
    st.sidebar.header("Filtros")
    
    # Year Slider
    min_year = int(df['year'].min())
    max_year = int(df['year'].max())
    selected_years = st.sidebar.slider(
        "Selecione o Intervalo de Anos",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )
    
    # Municipality Filter
    available_munis = sorted(df['muni_name'].unique().astype(str)) if 'muni_name' in df.columns else []
    selected_munis = st.sidebar.multiselect(
        "Filtrar por Município",
        options=available_munis,
        default=[]
    )
    
    # Cluster Filter
    available_clusters = sorted(df['cluster'].unique()) if 'cluster' in df.columns else []
    # Remove -1 if present (unmapped) usually
    available_clusters = [c for c in available_clusters if c != -1]
    selected_clusters = st.sidebar.multiselect(
        "Filtrar por Cluster",
        options=available_clusters,
        default=[]
    )
    
    # Apply Filters
    df_filtered = df[
        (df['year'] >= selected_years[0]) & 
        (df['year'] <= selected_years[1])
    ]
    
    if selected_munis:
        df_filtered = df_filtered[df_filtered['muni_name'].isin(selected_munis)]
        
    if selected_clusters:
        df_filtered = df_filtered[df_filtered['cluster'].isin(selected_clusters)]
        
    # Recalculate metrics based on filtered data
    if df_filtered.empty:
        st.warning("Nenhum dado encontrado com os filtros selecionados.")
        return

    # Process Button
    if st.sidebar.button("Processar Filtros"):
        process_clicked = True
    else:
        process_clicked = False

    # 1. Map Visualization
    st.subheader("Perda Total Histórica por Pixel")
    
    # Aggregate data for map
    # Use filtered data for map aggregation!
    map_data = df_filtered.groupby(['pixel_id', 'latitude', 'longitude'])['payout'].sum().reset_index()
    map_data.rename(columns={'payout': 'total_loss'}, inplace=True)
    
    # Create GeoDataFrame for spatial operations
    gdf = gpd.GeoDataFrame(
        map_data,
        geometry=gpd.points_from_xy(map_data.longitude, map_data.latitude),
        crs="EPSG:4326"
    )
    
    # Create Map
    m = folium.Map(location=[-30.0, -53.0], zoom_start=7, tiles="CartoDB positron")
    
    # Add Draw Control
    draw = Draw(
        export=False,
        position='topleft',
        draw_options={'polyline': False, 'rectangle': True, 'polygon': True, 'circle': True, 'marker': False, 'circlemarker': False},
        edit_options={'edit': False}
    )
    draw.add_to(m)
    
    # Add Heatmap/Circles
    for _, row in map_data.iterrows():
        # Color based on loss magnitude
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

    # Render Map and Capture Interaction
    output = st_folium(m, width=None, height=500)
    
    # --- State Management & Selection Logic ---
    
    # Initialize session state if not present
    if 'selected_pixel_ids' not in st.session_state:
        st.session_state['selected_pixel_ids'] = []
    if 'last_map_output' not in st.session_state:
        st.session_state['last_map_output'] = None
    if 'selection_source' not in st.session_state:
        st.session_state['selection_source'] = 'default'

    # Check for Map Changes
    # specific keys to watch for changes
    map_keys = ['last_active_drawing', 'last_object_clicked']
    current_map_state = {k: output.get(k) for k in map_keys if output and k in output}
    
    # Determine if map interaction occurred this run
    # (Compare current relevant output with cached previous output)
    last_map_state = st.session_state['last_map_output']
    map_interaction_occurred = (current_map_state != last_map_state)
    
    # Update cache
    st.session_state['last_map_output'] = current_map_state
    
    selected_pixel_ids = []

    # Priority Logic:
    # 1. Button Click (Overrides everything)
    # 2. Map Interaction (Overrides previous state)
    # 3. Existing Session State (Persists previous choice)
    # 4. Default (First pixel)

    if process_clicked:
        # User explicitly requested to process the filtered pixels
        selected_pixel_ids = df_filtered['pixel_id'].unique().tolist()
        st.session_state['selected_pixel_ids'] = selected_pixel_ids
        st.session_state['selection_source'] = 'filter_button'
        st.success(f"Filtro Processado: {len(selected_pixel_ids)} pixels selecionados.")
        
    elif map_interaction_occurred and (output.get('last_active_drawing') or output.get('last_object_clicked')):
        # User interacted with map
        st.session_state['selection_source'] = 'map_interaction'
        
        if output.get('last_active_drawing'):
            # --- DRAWING MODE ---
            drawing = output['last_active_drawing']
            geom_type = drawing['geometry']['type']
            
            if geom_type == 'Polygon':
                # Polygon or Rectangle
                draw_shape = shape(drawing['geometry'])
                # Spatial filter using GeoPandas
                selected_gdf = gdf[gdf.geometry.within(draw_shape)]
                selected_pixel_ids = selected_gdf['pixel_id'].tolist()
                
            elif geom_type == 'Point' and 'properties' in drawing and 'radius' in drawing['properties']:
                # Circle (Point + Radius in meters)
                center = shape(drawing['geometry']) # Point(lon, lat)
                radius_m = drawing['properties']['radius']
                center_lon, center_lat = center.x, center.y
                
                # Vectorized Haversine Distance Calculation
                R = 6371000 # Earth radius in meters
                
                # Convert to radians
                lat1 = np.radians(center_lat)
                lon1 = np.radians(center_lon)
                lat2 = np.radians(gdf['latitude'])
                lon2 = np.radians(gdf['longitude'])
                
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                
                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
                distances = R * c
                
                selected_gdf = gdf[distances <= radius_m]
                selected_pixel_ids = selected_gdf['pixel_id'].tolist()
                
            if selected_pixel_ids:
                st.success(f"Região Selecionada (Mapa): {len(selected_pixel_ids)} pixels encontrados.")
            else:
                st.warning("Nenhum pixel encontrado na área desenhada.")
                
        elif output.get('last_object_clicked'):
            # --- CLICK MODE ---
            clicked_lat = output['last_object_clicked']['lat']
            clicked_lon = output['last_object_clicked']['lng']
            
            # Simple distance check to find closest pixel
            map_data['dist'] = ((map_data['latitude'] - clicked_lat)**2 + (map_data['longitude'] - clicked_lon)**2)**0.5
            closest_pixel = map_data.loc[map_data['dist'].idxmin()]
            selected_pixel_ids = [closest_pixel['pixel_id']]
            st.success(f"Pixel Selecionado (Clique): {int(selected_pixel_ids[0])}")
            
        # Update session state with map result
        st.session_state['selected_pixel_ids'] = selected_pixel_ids
        
    else:
        # --- PERSISTENCE / DEFAULT ---
        # If no new interaction, use what we have in session state
        if st.session_state['selected_pixel_ids']:
             selected_pixel_ids = st.session_state['selected_pixel_ids']
             curr_source = st.session_state['selection_source']
             if curr_source == 'filter_button':
                 st.info(f"Visualizando seleção via Filtro: {len(selected_pixel_ids)} pixels.")
             elif curr_source == 'map_interaction':
                 st.info(f"Visualizando seleção via Mapa: {len(selected_pixel_ids)} pixels.")
        else:
             # Default: First pixel if nothing selected ever
             if not df.empty:
                selected_pixel_ids = [sorted(df['pixel_id'].unique())[0]]
                st.info("Selecione um pixel, desenhe no mapa, ou use 'Processar Filtros'.")

    # --- Context Maps (Clusters & Municipalities) ---
    st.markdown("---")
    st.subheader("Mapas de Contexto dos Clusters")
    
    # Use pre-rendered cluster analysis plots
    # These are relative to the CWD (filled_data), not BASE_DIR (web_app)
    # UPDATED: Use local copies in 'data' folder for reliability
    chirps_map_path = os.path.join(BASE_DIR, "data", "map_chirps.png")
    era5_map_path = os.path.join(BASE_DIR, "data", "map_era5.png")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists(chirps_map_path):
            st.image(Image.open(chirps_map_path), caption="Rainfall Pixels (CHIRPS) by Cluster", use_column_width=True)
        else:
             st.warning(f"Mapa não encontrado: {chirps_map_path}")
             
    with col2:
        if os.path.exists(era5_map_path):
            st.image(Image.open(era5_map_path), caption="Soil Moisture Pixels (ERA5) by Cluster", use_column_width=True)
        else:
             st.warning(f"Mapa não encontrado: {era5_map_path}")

    if not selected_pixel_ids:
        st.stop()

    # Calculate Pixel Counts for Selection
    # CHIRPS: filtered by current selection
    n_chirps = df_filtered['pixel_id'].nunique()
    
    # ERA5: filtered by current selection (Muni/Cluster)
    # If selected_pixel_ids came from MAP CLICK, we can't easily filter ERA5 unless it's spatial.
    # But usually filters apply.
    era5_subset = df_era5.copy()
    if selected_munis:
        era5_subset = era5_subset[era5_subset['muni_name'].isin(selected_munis)]
    if selected_clusters:
        era5_subset = era5_subset[era5_subset['cluster'].isin(selected_clusters)]
    
    n_era5 = len(era5_subset)
    
    # Calculate Totals for comparison (unfiltered by spatial)
    total_chirps = df_portfolio[df_portfolio['dataset']=='chirps']['pixel_id'].nunique()
    total_era5 = df_portfolio[df_portfolio['dataset']=='era5']['pixel_id'].nunique()
    
    st.info(f"Pixels na Seleção (Filtro Espacial): **{n_chirps}** / {total_chirps} CHIRPS | **{n_era5}** / {total_era5} ERA5")

    # 3. Process Selected Data
    pixel_df = df[df['pixel_id'].isin(selected_pixel_ids)].copy()
    
    # Apply Time Filter to the detailed view as well
    pixel_df = pixel_df[
        (pixel_df['year'] >= selected_years[0]) & 
        (pixel_df['year'] <= selected_years[1])
    ]
    
    # Prepare data for charting (Aggregation vs Single)
    # Ensure aggregation is correct
    if len(selected_pixel_ids) > 1:
        # REGIONAL AGGREGATION
        # Sum payouts by Year/Window
        chart_df = pixel_df.groupby(['year', 'window'])[['payout']].sum().reset_index()
        
        # We need to ensure we account for triggers correctly
        # Summing 'triggered' boolean (True=1) gives count of triggers
        trigger_counts = pixel_df.groupby(['year', 'window'])['triggered'].sum().reset_index()
        chart_df = chart_df.merge(trigger_counts, on=['year', 'window'])
        
        # analysys_title = f"Análise Regional ({len(selected_pixel_ids)} pixels)"
        pass
        
    else:
        # SINGLE PIXEL
        chart_df = pixel_df.copy()
        # analysys_title = f"Análise do Pixel {int(selected_pixel_ids[0])}"
        pass

    # Ensure full timeline (Selected Years only)
    all_years = range(selected_years[0], selected_years[1] + 1)
    full_index = pd.MultiIndex.from_product([all_years, WINDOWS], names=['year', 'window'])
    
    if not chart_df.empty:
        chart_df = chart_df.set_index(['year', 'window'])
        chart_df = chart_df.reindex(full_index, fill_value=0).reset_index()
    else:
        chart_df = pd.DataFrame(index=full_index).reset_index()
        chart_df['payout'] = 0
        chart_df['triggered'] = 0

    # Calculate metrics based on Chart DF
    total_payout = chart_df['payout'].sum()
    # Max annual payout
    max_year_payout = chart_df.groupby('year')['payout'].sum().max()
    # Total trigger events
    total_triggers = chart_df['triggered'].sum()
    if len(selected_pixel_ids) == 1:
         # For single pixel, triggered might be boolean in original df, but sum works same (True=1)
         pass

    # Display Metrics
    # Display Metrics
    st.subheader("Análise da Área Selecionada")
    col1, col2, col3 = st.columns(3)
    col1.metric("Indenização Total Histórica", f"${total_payout:,.2f}")
    col2.metric("Máxima Indenização Anual", f"${max_year_payout:,.2f}")
    col3.metric("Eventos de Acionamento", f"{int(total_triggers)}")

    # 4. Charts
    st.markdown("---")
    st.subheader("Análise da Área Selecionada")
        
    st.markdown("#### Histórico de Perdas por Ano e Janela")
    
    chart_title = "Perda Regional Agregada" if len(selected_pixel_ids) > 1 else "Perda do Pixel"
    
    chart = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X('year:O', title='Ano'),
        y=alt.Y('payout:Q', title='Indenização ($)'),
        color=alt.Color('window:N', title='Janela'),
        tooltip=['year', 'window', alt.Tooltip('payout', format='$,.2f', title='Indenização'), alt.Tooltip('triggered', title='Acionamentos')]
    ).properties(height=400, title=chart_title)
    
    st.altair_chart(chart, use_container_width=True)
    
    # 5. Events Table
    if len(selected_pixel_ids) == 1:
        st.subheader("Eventos de Acionamento")
        # Filter from original pixel_df
        triggers = pixel_df[pixel_df['triggered'] == True].sort_values('year', ascending=False)
        if not triggers.empty:
            st.dataframe(
                triggers[['year', 'window', 'raw_percentile', 'coverage_mult', 'payout']].style.format({
                    'raw_percentile': '{:.1f}',
                    'payout': '${:,.2f}',
                    'coverage_mult': '{:.0%}'
                }),
                column_config={
                    "year": "Ano",
                    "window": "Janela",
                    "raw_percentile": "Percentil",
                    "coverage_mult": "Cobertura",
                    "payout": "Indenização"
                }
            )
        else:
            st.info("Nenhum evento de acionamento para este pixel.")
    else:
        st.subheader("Eventos Regionais Agregados")
        # Show aggregated stats per year/window (from chart_df)
        # Filter for rows with activity
        regional_events = chart_df[chart_df['payout'] > 0].sort_values(['year', 'payout'], ascending=[False, False])
        
        if not regional_events.empty:
            st.dataframe(
                regional_events[['year', 'window', 'payout', 'triggered']].style.format({
                    'year': '{:.0f}',
                    'payout': '${:,.2f}',
                    'triggered': '{:.0f}'
                }),
                column_config={
                    "year": "Ano",
                    "window": "Janela",
                    "payout": "Indenização Total",
                    "triggered": "Total de Acionamentos"
                }
            )
        else:
            st.info("Nenhuma perda nesta região.")

    # --- Portfolio Statistics (Fixed / Static) ---
    st.markdown("### Estatísticas de Toda a Região (Portfólio)")
    st.markdown("Estas estatísticas representam todo o portfólio (todos os pixels) para o período de tempo completo (1996-2025).")
    
    # We use 'df_portfolio' (the original loaded dataframe, cleaned of negative IDs but NOT spatially filtered) 
    # ensuring it represents the full 28M/40-20 structure.
    
    # Calculate Annual Losses with Reindexing to include 0-loss years
    # HARDCODED: The 28M/40-20 structure is strictly 30 years (1996-2025).
    # Dynamic min/max might miss years with 0 loss (e.g. 1996), skewing the mean.
    full_years = pd.Index(range(1996, 2026), name='year')
    
    portfolio_annual = df_portfolio.groupby('year')['payout'].sum()
    portfolio_annual = portfolio_annual.reindex(full_years, fill_value=0.0)
    
    port_total_loss = portfolio_annual.sum()
    port_mean_loss = portfolio_annual.mean()
    port_max_loss = portfolio_annual.max()
    port_p95_loss = portfolio_annual.quantile(0.95)
    port_p99_loss = portfolio_annual.quantile(0.99)
    
    col_stat1, col_stat2, col_stat3, col_stat4, col_stat5 = st.columns(5)
    col_stat1.metric("Perda Total do Portfólio", f"${port_total_loss:,.2f}")
    col_stat2.metric("Perda Média Anual", f"${port_mean_loss:,.2f}")
    col_stat3.metric("Perda Máxima Anual", f"${port_max_loss:,.2f}")
    col_stat4.metric("Perda P95 Anual", f"${port_p95_loss:,.2f}")
    col_stat5.metric("Perda P99 Anual", f"${port_p99_loss:,.2f}")
    
    st.markdown("#### Detalhamento por Perigo (Peril)")
    # Breakdown by Dataset (CHIRPS vs ERA5)
    # Use df_portfolio for consistency
    # 1. Pixel Counts
    chirps_pixels = df_portfolio[df_portfolio['dataset'] == 'chirps']['pixel_id'].nunique()
    era5_pixels = df_portfolio[df_portfolio['dataset'] == 'era5']['pixel_id'].nunique()
    
    # 2. Total Losses
    chirps_loss = df_portfolio[df_portfolio['dataset'] == 'chirps']['payout'].sum()
    era5_loss = df_portfolio[df_portfolio['dataset'] == 'era5']['payout'].sum()
    
    # 3. Events (Triggered)
    if 'triggered' in df_portfolio.columns:
        chirps_events = df_portfolio[(df_portfolio['dataset'] == 'chirps') & (df_portfolio['triggered'] == True)].shape[0]
        era5_events = df_portfolio[(df_portfolio['dataset'] == 'era5') & (df_portfolio['triggered'] == True)].shape[0]
    else:
         chirps_events = df_portfolio[(df_portfolio['dataset'] == 'chirps') & (df_portfolio['payout'] > 0)].shape[0]
         era5_events = df_portfolio[(df_portfolio['dataset'] == 'era5') & (df_portfolio['payout'] > 0)].shape[0]

    bd_col1, bd_col2, bd_col3 = st.columns(3)
    
    with bd_col1:
        st.markdown("**Contagem de Pixels**")
        st.write(f"CHIRPS (Chuva): **{chirps_pixels}**")
        st.write(f"ERA5 (Umidade): **{era5_pixels}**")
        
    with bd_col2:
        st.markdown("**Perda Total ($)**")
        st.write(f"CHIRPS: **${chirps_loss:,.2f}**")
        st.write(f"ERA5: **${era5_loss:,.2f}**")
        
    with bd_col3:
        st.markdown("**Eventos de Acionamento**")
        st.write(f"CHIRPS: **{chirps_events}**")
        st.write(f"ERA5: **{era5_events}**")

if __name__ == "__main__":
    main()

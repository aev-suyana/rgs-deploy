import streamlit as st
import pandas as pd
import altair as alt
import folium
from streamlit_folium import st_folium
import os
import json
import warnings
import io
import geopandas as gpd
from shapely.geometry import Point, shape
import numpy as np
from folium.plugins import Draw
from PIL import Image

# Suppress FutureWarning for 'observed' parameter in groupby
warnings.simplefilter(action='ignore', category=FutureWarning)

# Page Configuration
st.set_page_config(layout="wide", page_title="Pixel Loss Dashboard")

# Custom CSS for Aesthetics
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: white !important;
        }
        [data-testid="stSidebarNav"] {
            background-color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

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
            
            # ERA5 OFFSET FIX: Add 1B to ERA5 IDs to prevent collision with CHIRPS
            if 'pixel_id' in df.columns and 'dataset' in df.columns:
                df.loc[df['dataset'] == 'era5', 'pixel_id'] += 1000000000
                
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
            
            # ERA5 OFFSET FIX: Add 1B to ERA5 IDs to prevent collision with CHIRPS
            if 'pixel_id' in df.columns and 'dataset' in df.columns:
                df.loc[df['dataset'] == 'era5', 'pixel_id'] += 1000000000
            
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
            df = pd.read_csv(ERA5_PATH)
            # ERA5 OFFSET FIX: Match the offset in load_data
            if 'pixel_id' in df.columns:
                df['pixel_id'] += 1000000000
            return df
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
    # Header with Logo in Top-Right
    col_header1, col_header2 = st.columns([0.8, 0.2])
    with col_header1:
        st.title("Perdas Históricas por Pixel (Rio Grande do Sul)")
    with col_header2:
        logo_path = os.path.join(BASE_DIR, "logo.png")
        if os.path.exists(logo_path):
            st.image(Image.open(logo_path), use_column_width=True)
    
    # Load Data
    df = load_data()
    if df.empty:
        return
    
    # Load Shapes for Spatial Filtering
    gdf, _ = load_shapefiles()

    # --- Instruções de Uso ---
    with st.expander("📖 Guia de Uso - Primeiro Acesso", expanded=False):
        st.markdown("""
        Bem-vindo ao Dashboard de Perdas Climáticas. Aqui estão algumas orientações para sua análise:
        
        1.  **Filtros Laterais**: Utilize a barra lateral para filtrar os dados por **Intervalo de Anos**, **Município** ou **Cluster**.
        2.  **Melhores Práticas**: Recomendamos selecionar **apenas Municípios OU apenas Clusters** para evitar conflitos de seleção, embora o sistema suporte ambos.
        3.  **Seleção Múltipla**: Você pode selecionar vários itens em cada filtro.
        4.  **Mapa Interativo**: Use as ferramentas de desenho no mapa (polígono, retângulo, círculo) para selecionar áreas específicas.
        5.  **Análise de Seleção**: Os gráficos e métricas na parte inferior serão atualizados automaticamente com base na sua escolha (via filtros ou mapa).
        """)

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
    
    # --- Portfolio Statistics (Fixed / Static) ---
    st.markdown("### Estatísticas de Toda a Região (Portfólio)")
    st.markdown("Estas estatísticas representam todo o portfólio (todos os pixels) para o período de tempo completo (1996-2025).")
    
    # Calculate Annual Losses with Reindexing to include 0-loss years
    full_years = pd.Index(range(1996, 2026), name='year')
    
    portfolio_annual = df_portfolio.groupby('year')['payout'].sum()
    portfolio_annual = portfolio_annual.reindex(full_years, fill_value=0.0)
    
    port_total_loss = portfolio_annual.sum()
    port_mean_loss = portfolio_annual.mean()
    port_max_loss = portfolio_annual.max()
    
    # Calculate Counts
    chirps_pixels = df_portfolio[df_portfolio['dataset'] == 'chirps']['pixel_id'].nunique()
    era5_pixels = df_portfolio[df_portfolio['dataset'] == 'era5']['pixel_id'].nunique()
    muni_count = df_portfolio['muni_name'].nunique()
    
    # Calculate Active Years per Peril
    chirps_active_years = df_portfolio[(df_portfolio['dataset'] == 'chirps') & (df_portfolio['payout'] > 0)]['year'].nunique()
    era5_active_years = df_portfolio[(df_portfolio['dataset'] == 'era5') & (df_portfolio['payout'] > 0)]['year'].nunique()
    
    # Display Top Metrics
    # Use delta for "Active Years" or color
    col_stat1, col_stat2, col_stat3, col_stat4, col_stat5, col_stat6 = st.columns(6)
    
    col_stat1.metric("Cumulative Historical Loss", f"${port_total_loss:,.0f}")
    col_stat2.metric("Average Annual Loss", f"${port_mean_loss:,.0f}")
    col_stat3.metric("Max Annual Loss", f"${port_max_loss:,.0f}")
    col_stat4.metric("Municipalities", f"{muni_count}")
    
    # Show Pixels and Years (Hardcoded 30 years for Portfolio)
    col_stat5.metric("CHIRPS (Rainfall)", f"{chirps_pixels:,} px", f"{chirps_active_years}/30 years active")
    col_stat6.metric("ERA5 (Soil Moisture)", f"{era5_pixels:,} px", f"{era5_active_years}/30 years active")

    st.markdown("---")

    # Sidebar Filters
    logo_path = os.path.join(BASE_DIR, "logo.png")
    if os.path.exists(logo_path):
        st.sidebar.image(Image.open(logo_path), use_column_width=True)
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
    
    # Peril Filter
    peril_options = ["Chuva (CHIRPS)", "Seca (ERA5)"]
    selected_perils_friendly = st.sidebar.multiselect(
        "Filtrar por Perigo",
        options=peril_options,
        default=[]
    )
    # Map back to dataset values
    peril_map_rev = {"Chuva (CHIRPS)": "chirps", "Seca (ERA5)": "era5"}
    selected_perils = [peril_map_rev[p] for p in selected_perils_friendly]
    
    # Window Filter
    # Map Windows to Full Names (same as in charts for consistency)
    window_map_display = {
        'apr_may': 'April-May (Rainfall)',
        'sep_oct': 'September-October (Soil Moisture)',
        'oct_dec': 'October-December (Soil Moisture)',
        'dec_feb': 'December-February (Rainfall)'
    }
    available_windows_friendly = list(window_map_display.values())
    selected_windows_friendly = st.sidebar.multiselect(
        "Filtrar por Janelas Climáticas",
        options=available_windows_friendly,
        default=[]
    )
    # Map back to internal window keys
    window_map_rev = {v: k for k, v in window_map_display.items()}
    selected_windows = [window_map_rev[w] for w in selected_windows_friendly]
    
    # Apply Filters
    df_filtered = df[
        (df['year'] >= selected_years[0]) & 
        (df['year'] <= selected_years[1])
    ]
    
    if selected_munis:
        df_filtered = df_filtered[df_filtered['muni_name'].isin(selected_munis)]
        
    if selected_clusters:
        df_filtered = df_filtered[df_filtered['cluster'].isin(selected_clusters)]
        
    if selected_perils:
        df_filtered = df_filtered[df_filtered['dataset'].isin(selected_perils)]
        
    if selected_windows:
        df_filtered = df_filtered[df_filtered['window'].isin(selected_windows)]
        
    # Recalculate metrics based on filtered data
    if df_filtered.empty:
        st.warning("Nenhum dado encontrado com os filtros selecionados.")
        return

    # Process Button Removed - Auto-update implemented below

    # 1. Map Visualization
    st.subheader("Perda Total Histórica por Pixel")
    
    # Aggregate data for map
    # Use filtered data for map aggregation!
    map_data = df_filtered.groupby(['pixel_id', 'latitude', 'longitude'])['payout'].sum().reset_index()
    map_data.rename(columns={'payout': 'total_loss'}, inplace=True)
    
    # Create GeoDataFrame for spatial operations
    pixels_map_gdf = gpd.GeoDataFrame(
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
    # Add Heatmap/Circles
    for _, row in map_data.iterrows():
        # Color based on loss magnitude (Increased intensity)
        # Using brighter colors and higher opacity
        color = "#c7e9c0" # Pale green
        if row['total_loss'] > 50000:
            color = "#74c476"
        if row['total_loss'] > 100000:
            color = "#238b45"
        if row['total_loss'] > 200000:
            color = "#00441b" # Darkest green
            
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=4, # Slightly larger
            color="#333333", # Dark stroke for contrast
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.9, # More opaque/intense
            popup=f"Pixel {int(row['pixel_id'])}: ${row['total_loss']:,.0f}",
            tooltip=f"Pixel {int(row['pixel_id'])}"
        ).add_to(m)

    # Render Map and Capture Interaction
    output = st_folium(m, width=None, height=800)
    
    # --- State Management & Selection Logic ---
    
    # Initialize session state if not present
    if 'selected_pixel_ids' not in st.session_state:
        st.session_state['selected_pixel_ids'] = []
    if 'last_map_output' not in st.session_state:
        st.session_state['last_map_output'] = None
    if 'selection_source' not in st.session_state:
        st.session_state['selection_source'] = 'default'
        
    # Track filter state for auto-updates
    if 'last_selected_munis' not in st.session_state:
        st.session_state['last_selected_munis'] = []
    if 'last_selected_clusters' not in st.session_state:
        st.session_state['last_selected_clusters'] = []
    if 'last_selected_perils' not in st.session_state:
        st.session_state['last_selected_perils'] = []
    if 'last_selected_windows' not in st.session_state:
        st.session_state['last_selected_windows'] = []

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
    
    # Check for Filter Changes
    filter_changed = False
    if selected_munis != st.session_state['last_selected_munis']:
        filter_changed = True
        st.session_state['last_selected_munis'] = selected_munis
    
    if selected_clusters != st.session_state['last_selected_clusters']:
        filter_changed = True
        st.session_state['last_selected_clusters'] = selected_clusters
        
    if selected_perils != st.session_state['last_selected_perils']:
        filter_changed = True
        st.session_state['last_selected_perils'] = selected_perils
        
    if selected_windows != st.session_state['last_selected_windows']:
        filter_changed = True
        st.session_state['last_selected_windows'] = selected_windows
    
    selected_pixel_ids = []

    # Priority Logic:
    # 1. Filter Change (Auto-update via Sidebar)
    # 2. Map Interaction (Overrides previous state if new action)
    # 3. Existing Session State (Persists previous choice)
    # 4. Default (First pixel)

    if filter_changed:
        # User changed filters, update selection immediately
        chirps_ids = df_filtered['pixel_id'].unique().tolist()
        
        # Also grab valid ERA5 pixels for these filters
        # because df (CHIRPS) might lack muni mapping for ERA5, so filtering df drops them.
        # df_era5 has dynamically joined muni names.
        era5_ids = []
        if not df_era5.empty:
            subset_era5 = df_era5.copy()
            # Apply same filters
            if selected_munis:
                subset_era5 = subset_era5[subset_era5['muni_name'].isin(selected_munis)]
            if selected_clusters:
                subset_era5 = subset_era5[subset_era5['cluster'].isin(selected_clusters)]
            
            # Ensure we only pick pixels that assume exist in the main loss df for consistency,
            # Or just take all valid ERA5 pixels?
            # If main df has no ERA5 data for id=X, then adding X to selected_pixel_ids keeps it empty in pixel_df (metrics) anyway.
            # So it is safe to add.
            if 'pixel_id' in subset_era5.columns:
                era5_ids = subset_era5['pixel_id'].unique().tolist()

        selected_pixel_ids = list(set(chirps_ids + era5_ids))
        st.session_state['selected_pixel_ids'] = selected_pixel_ids
        st.session_state['selection_source'] = 'filter_change'
        # st.success(f"Filtros Atualizados: {len(selected_pixel_ids)} pixels.")
        
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
                selected_gdf = pixels_map_gdf[pixels_map_gdf.geometry.within(draw_shape)]
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
                lat2 = np.radians(pixels_map_gdf['latitude'])
                lon2 = np.radians(pixels_map_gdf['longitude'])
                
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                
                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
                distances = R * c
                
                selected_gdf = pixels_map_gdf[distances <= radius_m]
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
             if curr_source == 'filter_change' or curr_source == 'filter_button':
                 st.info(f"Visualizando seleção via Filtro: {len(selected_pixel_ids)} pixels.")
             elif curr_source == 'map_interaction':
                 st.info(f"Visualizando seleção via Mapa: {len(selected_pixel_ids)} pixels.")
        else:
             # Default: First pixel if nothing selected ever
             if not df.empty:
                selected_pixel_ids = [sorted(df['pixel_id'].unique())[0]]
                st.info("Selecione um pixel, desenhe no mapa, ou use os filtros laterais.")

    # --- Context Maps (Clusters & Municipalities) ---
    st.markdown("---")
    st.subheader("Referência de Clusters")
    st.info("Nota: Este mapa é apenas para referência. Utilize os filtros na barra lateral ou o mapa interativo acima para selecionar regiões para análise.")
    
    if gdf is not None:
         # Define specific colors to match the palette
         # Colors: 0-9
         colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
         
         # Create a small folium map for reference
         m_ref = folium.Map(location=[-30.0, -53.0], zoom_start=6, tiles="CartoDB positron", zoom_control=False, scrollWheelZoom=False, dragging=False)
         
         # Add Municipalities colored by Cluster
         folium.GeoJson(
             gdf,
             style_function=lambda x: {
                 'fillColor': colors[x['properties']['cluster'] % len(colors)] if x['properties']['cluster'] != -1 else '#cccccc',
                 'color': 'black',
                 'weight': 0.5,
                 'fillOpacity': 0.7
             },
             tooltip=folium.GeoJsonTooltip(
                 fields=['NM_MUN', 'cluster'],
                 aliases=['Município:', 'Cluster:'],
                 localize=True
             )
         ).add_to(m_ref)

         # Add Cluster Number Labels
         clusters_gdf = gdf.dissolve(by='cluster').centroid
         for cluster_id, centroid in clusters_gdf.items():
             if cluster_id != -1:
                 folium.Marker(
                     location=[centroid.y, centroid.x],
                     icon=folium.DivIcon(
                         html=f'<div style="font-size: 12pt; font-weight: bold; color: white; text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;">{int(cluster_id)}</div>',
                         icon_anchor=(5,5)
                     )
                 ).add_to(m_ref)
         
         st_folium(m_ref, width=None, height=500, key="ref_map") # width=None fills container
    else:
         st.info("Shapefile não carregado.")

    if not selected_pixel_ids:
        st.stop()

    # Selection breakdown by peril
    pixel_df = df[df['pixel_id'].isin(selected_pixel_ids)].copy()
    
    # Apply Filters to the selection
    pixel_df = pixel_df[
        (pixel_df['year'] >= selected_years[0]) & 
        (pixel_df['year'] <= selected_years[1])
    ]
    
    if selected_perils:
        pixel_df = pixel_df[pixel_df['dataset'].isin(selected_perils)]
        
    if selected_windows:
        pixel_df = pixel_df[pixel_df['window'].isin(selected_windows)]
    
    # Calculate unique IDs per dataset for the selection
    sel_chirps_px = pixel_df[pixel_df['dataset']=='chirps']['pixel_id'].nunique()
    sel_era5_px = pixel_df[pixel_df['dataset']=='era5']['pixel_id'].nunique()
    
    # Calculate active years per peril (Selection)
    sel_chirps_years = pixel_df[(pixel_df['dataset']=='chirps') & (pixel_df['payout']>0)]['year'].nunique()
    sel_era5_years = pixel_df[(pixel_df['dataset']=='era5') & (pixel_df['payout']>0)]['year'].nunique()
    
    all_years = range(selected_years[0], selected_years[1] + 1)
    n_years_sel = len(all_years)
    
    # Calculate Totals for comparison (unfiltered by spatial)
    total_chirps = df_portfolio[df_portfolio['dataset']=='chirps']['pixel_id'].nunique()
    total_era5 = df_portfolio[df_portfolio['dataset']=='era5']['pixel_id'].nunique()
    
    
    # Construct Description of Selection
    sel_desc_parts = []
    if selected_munis:
        sel_desc_parts.append(f"Municípios: {', '.join(selected_munis)}")
    if selected_clusters:
        # Cast to standard int to avoid np.int64 display in banner
        clean_clusters = [int(c) for c in selected_clusters]
        sel_desc_parts.append(f"Clusters: {sorted(clean_clusters)}")
        
    if selected_perils_friendly:
        sel_desc_parts.append(f"Perigos: {', '.join(selected_perils_friendly)}")
        
    if selected_windows_friendly:
        sel_desc_parts.append(f"Janelas: {', '.join(selected_windows_friendly)}")
    
    selection_description = " | ".join(sel_desc_parts) if sel_desc_parts else "Seleção Personalizada/Manual"
    
    st.info(f"**{selection_description}** — Pixels: **{sel_chirps_px}** / {total_chirps} CHIRPS | **{sel_era5_px}** / {total_era5} ERA5")

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
    else:
        # SINGLE PIXEL
        chart_df = pixel_df.copy()

    # Ensure full timeline (Selected Years only)
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
    max_year_payout = chart_df.groupby('year')['payout'].sum().max()
    avg_annual_payout = total_payout / len(all_years) if len(all_years) > 0 else 0

    # Display Metrics
    st.subheader("Selection Analysis")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("Cumulative Historical Loss", f"${total_payout:,.0f}")
    col2.metric("Average Annual Loss", f"${avg_annual_payout:,.0f}")
    col3.metric("Max Annual Loss", f"${max_year_payout:,.0f}")
    col4.metric("CHIRPS (Rainfall)", f"{sel_chirps_px:,} px", f"{sel_chirps_years}/{n_years_sel} years active")
    col5.metric("ERA5 (Soil Moisture)", f"{sel_era5_px:,} px", f"{sel_era5_years}/{n_years_sel} years active")

    # 4. Charts
    st.markdown("---")
        
    st.markdown("#### Histórico de Perdas por Ano e Janela")
    
    # Map Windows to Full Names
    window_map = {
        'apr_may': 'April-May (Rainfall)',
        'sep_oct': 'September-October (Soil Moisture)',
        'oct_dec': 'October-December (Soil Moisture)',
        'dec_feb': 'December-February (Rainfall)'
    }
    
    # Apply mapping
    chart_df['window_label'] = chart_df['window'].map(window_map).fillna(chart_df['window'])
    
    # Define Colors
    domain = ['April-May (Rainfall)', 'September-October (Soil Moisture)', 'October-December (Soil Moisture)', 'December-February (Rainfall)']
    range_ = ['#1f77b4', '#d8b365', '#a6611a', '#17becf'] # Blue (Apr-May), Browns (Sep-Dec), Cyan/Blue (Dec-Feb)

    chart_title = "Perda Regional Agregada" if len(selected_pixel_ids) > 1 else "Perda do Pixel"
    
    chart = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X('year:O', title='Year'),
        y=alt.Y('payout:Q', title='Payout ($)'),
        color=alt.Color('window_label:N', scale=alt.Scale(domain=domain, range=range_), title='Window'),
        tooltip=['year', 'window_label', alt.Tooltip('payout', format='$,.0f', title='Payout'), alt.Tooltip('triggered', title='Triggers')]
    ).properties(height=400, title=chart_title)
    
    st.altair_chart(chart, use_container_width=True)
    
            
    # --- Download Selection Data ---
    st.markdown("---")
    st.subheader("Exportar Dados da Seleção")
    st.warning("⚠️ **Aviso**: Os resultados obtidos através deste dashboard são preliminares e indicativos. Para fins de precificação final, propostas comerciais ou propósitos legais, os dados devem ser expressamente confirmados pela Suyana.")
    
    # Prepare download dataframe
    download_df = pixel_df.copy()
    
    # Attempt to enrich ERA5 metadata if 'muni_name' is missing/unknown for them in main df
    # (Since main df mapping might not cover ERA5 pixels)
    if not df_era5.empty:
        # subset df_era5 to relevant pixels to save memory/time
        relevant_era5_ids = download_df[download_df['dataset'] == 'era5']['pixel_id'].unique()
        if len(relevant_era5_ids) > 0:
            # FIX: Filter metadata by active sidebar selections to avoid bringing in "odd" regions.
            # Then de-duplicate to ensure one-to-one mapping for the export.
            era_meta = df_era5[df_era5['pixel_id'].isin(relevant_era5_ids)][['pixel_id', 'muni_name', 'cluster']]
            
            if selected_munis:
                era_meta = era_meta[era_meta['muni_name'].isin(selected_munis)]
            elif selected_clusters:
                era_meta = era_meta[era_meta['cluster'].isin(selected_clusters)]
            
            era5_subset_info = era_meta.drop_duplicates(subset='pixel_id')
            
            # Merge
            download_df = download_df.merge(era5_subset_info, on='pixel_id', how='left', suffixes=('', '_era5'))
            
            # Fill missing/Desconhecido values
            # Assuming 'Desconhecido' or NaN in main df
            mask_fill = (download_df['dataset'] == 'era5')
            if 'muni_name_era5' in download_df.columns:
                download_df.loc[mask_fill, 'muni_name'] = download_df.loc[mask_fill, 'muni_name_era5'].fillna(download_df.loc[mask_fill, 'muni_name'])
            if 'cluster_era5' in download_df.columns:
                 download_df.loc[mask_fill, 'cluster'] = download_df.loc[mask_fill, 'cluster_era5'].fillna(download_df.loc[mask_fill, 'cluster'])
            
            # Cleanup
            drop_cols = [c for c in download_df.columns if '_era5' in c]
            download_df.drop(columns=drop_cols, inplace=True)
            
    # Add 'Perigo' Column (Portuguese Mapping)
    download_df['peril'] = download_df['dataset'].map({'chirps': 'Chuva', 'era5': 'Seca'}).fillna('Desconhecido')

    # Select and Rename Columns
    cols_to_export = {
        'year': 'Ano',
        'window': 'Janela',
        'peril': 'Perigo',
        'dataset': 'Fonte de Dados',
        'muni_name': 'Município',
        'cluster': 'Cluster',
        'pixel_id': 'Pixel ID',
        'payout': 'Perda ($)',
        'triggered': 'Acionamento (Sim/Não)'
    }
    
    # Filter only existing columns
    existing_cols = [c for c in cols_to_export.keys() if c in download_df.columns]
    download_df = download_df[existing_cols].rename(columns=cols_to_export)
    
    # Sort
    if 'Ano' in download_df.columns and 'Janela' in download_df.columns:
        download_df.sort_values(by=['Ano', 'Janela'], inplace=True)

    # Convert to XLSX with Multiple Sheets
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Sheet 1: Raw Data
        download_df.to_excel(writer, index=False, sheet_name='Dados Brutos')
        
        # Sheet 2: Yearly Aggregate
        agg_annual = download_df.groupby('Ano')['Perda ($)'].sum().reset_index()
        agg_annual.to_excel(writer, index=False, sheet_name='Agregado Anual')
        
        # Sheet 3: Year-Peril Aggregate
        agg_peril = download_df.groupby(['Ano', 'Perigo', 'Fonte de Dados'])['Perda ($)'].sum().reset_index()
        agg_peril.to_excel(writer, index=False, sheet_name='Agregado Ano-Perigo')
        
        # Sheet 4: Year-Cluster-Peril Aggregate
        if 'Cluster' in download_df.columns:
            agg_cluster = download_df.groupby(['Ano', 'Cluster', 'Perigo', 'Fonte de Dados'])['Perda ($)'].sum().reset_index()
            agg_cluster.to_excel(writer, index=False, sheet_name='Agregado Ano-Cluster-Perigo')
            
        # Sheet 5: Year-Municipality-Peril Aggregate
        if 'Município' in download_df.columns:
            agg_muni = download_df.groupby(['Ano', 'Município', 'Perigo', 'Fonte de Dados'])['Perda ($)'].sum().reset_index()
            agg_muni.to_excel(writer, index=False, sheet_name='Agregado Ano-Mun-Perigo')
            
    excel_data = buffer.getvalue()
    
    st.download_button(
        label="📥 Baixar Dados da Seleção (Excel Multi-Abas)",
        data=excel_data,
        file_name='perdas_selecao_pixel_clima_completo.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )
    




if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import requests
import rasterio
import geopandas as gpd
import joblib
import math
import io
import os
import glob

from shapely.geometry import Point


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\workr\landslide Data"

MODEL_FILE = os.path.join(
    BASE_DIR,
    "Landslide_AI_GSI",
    "landslide_xgboost.pkl"
)

SRTM_FOLDER = os.path.join(
    BASE_DIR,
    "Slope data"
)

RAINFALL_FOLDER = os.path.join(
    BASE_DIR,
    "rainfall data"
)

RIVER_FILE = os.path.join(
    BASE_DIR,
    "river data",
    "HydroRIVERS",
    "HydroRIVERS_v10_as_shp",
    "HydroRIVERS_v10_as.shp"
)

TRAINING_FILE = os.path.join(
    BASE_DIR,
    "Landslide_AI_GSI",
    "landslide_training_data.csv"
)


# ============================================================
# LOAD MODEL
# ============================================================

xgb_model = joblib.load(MODEL_FILE)


# ============================================================
# LOAD TRAINING MEDIANS FOR SAFE DEM FALLBACK
# ============================================================

try:
    training_data = pd.read_csv(TRAINING_FILE)

    ELEVATION_FALLBACK = float(
        training_data["Elevation"].median()
    )

    SLOPE_FALLBACK = float(
        training_data["Slope"].median()
    )

except Exception:
    # Only used if the training CSV is unavailable.
    ELEVATION_FALLBACK = 1200.0
    SLOPE_FALLBACK = 20.0


# ============================================================
# RAINFALL FILE
# ============================================================

rainfall_files = glob.glob(
    os.path.join(
        RAINFALL_FOLDER,
        "IMERG-Final.CLIM.200006-202305.V07B*.tif"
    )
)

if len(rainfall_files) == 0:
    raise FileNotFoundError(
        "IMERG rainfall file not found."
    )

rainfall_file = rainfall_files[0]


# ============================================================
# LOAD RIVERS
# ============================================================

rivers = gpd.read_file(
    RIVER_FILE,
    engine="pyogrio"
)


# ============================================================
# HELPER: LOCAL DEM TILE
# ============================================================

def get_local_dem_file(latitude, longitude):

    lat_start = math.floor(latitude)
    lon_start = math.floor(longitude)

    lat_prefix = "N" if lat_start >= 0 else "S"
    lon_prefix = "E" if lon_start >= 0 else "W"

    lat_name = f"{lat_prefix}{abs(lat_start):02d}"
    lon_name = f"{lon_prefix}{abs(lon_start):03d}"

    tif_file = os.path.join(
        SRTM_FOLDER,
        f"{lat_name}{lon_name}.tif"
    )

    hgt_file = os.path.join(
        SRTM_FOLDER,
        f"{lat_name}{lon_name}.hgt"
    )

    if os.path.exists(tif_file):
        return tif_file

    if os.path.exists(hgt_file):
        return hgt_file

    return None


# ============================================================
# OPEN TOPO DATA FALLBACK
#
# This is used when the local SRTM tile is not available.
# It avoids requiring a local SRTM file for every map click.
# ============================================================

def get_online_elevations(latitude, longitude):

    spacing = 0.0003

    points = [
        (latitude + spacing, longitude - spacing),
        (latitude + spacing, longitude),
        (latitude + spacing, longitude + spacing),

        (latitude, longitude - spacing),
        (latitude, longitude),
        (latitude, longitude + spacing),

        (latitude - spacing, longitude - spacing),
        (latitude - spacing, longitude),
        (latitude - spacing, longitude + spacing)
    ]

    locations = "|".join(
        f"{lat},{lon}"
        for lat, lon in points
    )

    url = "https://api.opentopodata.org/v1/srtm90m"

    try:
        response = requests.get(
            url,
            params={"locations": locations},
            timeout=20
        )

        if response.status_code != 200:
            print(
                "OpenTopoData returned:",
                response.status_code
            )
            return None

        data = response.json()

        results = data.get("results", [])

        if len(results) != 9:
            return None

        elevations = []

        for item in results:
            elevation = item.get("elevation")

            if elevation is None:
                return None

            elevations.append(float(elevation))

        return elevations

    except Exception as e:
        print(
            "Online elevation service unavailable:",
            e
        )
        return None


# ============================================================
# ELEVATION
# ============================================================

def get_elevation(latitude, longitude):

    # --------------------------------------------------------
    # 1. Try local DEM first
    # --------------------------------------------------------

    tile_file = get_local_dem_file(
        latitude,
        longitude
    )

    if tile_file is not None:

        try:

            with rasterio.open(tile_file) as src:

                value = list(
                    src.sample([
                        (longitude, latitude)
                    ])
                )[0][0]

                if src.nodata is not None:

                    if value == src.nodata:
                        return None

                return float(value)

        except Exception as e:

            print(
                "Local DEM read error:",
                e
            )

    # --------------------------------------------------------
    # 2. Try online elevation service
    # --------------------------------------------------------

    online_values = get_online_elevations(
        latitude,
        longitude
    )

    if online_values is not None:
        return float(online_values[4])

    # --------------------------------------------------------
    # 3. Final safe fallback
    #
    # This prevents the API from failing when both DEM
    # sources are unavailable.
    # --------------------------------------------------------

    print(
        "DEM unavailable. Using training-data "
        "median elevation fallback."
    )

    return ELEVATION_FALLBACK


# ============================================================
# SLOPE
# ============================================================

def get_slope(latitude, longitude):

    # --------------------------------------------------------
    # 1. Try local DEM
    # --------------------------------------------------------

    tile_file = get_local_dem_file(
        latitude,
        longitude
    )

    if tile_file is not None:

        try:

            with rasterio.open(tile_file) as src:

                row, col = src.index(
                    longitude,
                    latitude
                )

                if row <= 0 or col <= 0:
                    raise ValueError(
                        "Point too close to DEM tile edge."
                    )

                if row >= src.height - 1:
                    raise ValueError(
                        "Point too close to DEM tile edge."
                    )

                if col >= src.width - 1:
                    raise ValueError(
                        "Point too close to DEM tile edge."
                    )

                window = rasterio.windows.Window(
                    col - 1,
                    row - 1,
                    3,
                    3
                )

                elevation = src.read(
                    1,
                    window=window
                )

                if elevation.shape != (3, 3):
                    raise ValueError(
                        "Invalid DEM neighborhood."
                    )

                if src.nodata is not None:

                    if np.any(
                        elevation == src.nodata
                    ):
                        raise ValueError(
                            "DEM neighborhood contains nodata."
                        )

                left = float(
                    elevation[1, 0]
                )

                right = float(
                    elevation[1, 2]
                )

                top = float(
                    elevation[0, 1]
                )

                bottom = float(
                    elevation[2, 1]
                )

                pixel_width = abs(
                    src.transform.a
                )

                pixel_height = abs(
                    src.transform.e
                )

                lat_rad = math.radians(
                    latitude
                )

                meter_x = (
                    pixel_width
                    * 111320
                    * math.cos(lat_rad)
                )

                meter_y = (
                    pixel_height
                    * 111320
                )

                dzdx = (
                    right - left
                ) / (
                    2 * meter_x
                )

                dzdy = (
                    bottom - top
                ) / (
                    2 * meter_y
                )

                slope = math.degrees(
                    math.atan(
                        math.sqrt(
                            dzdx ** 2 +
                            dzdy ** 2
                        )
                    )
                )

                return float(slope)

        except Exception as e:

            print(
                "Local slope calculation unavailable:",
                e
            )

    # --------------------------------------------------------
    # 2. Online DEM fallback
    # --------------------------------------------------------

    online_values = get_online_elevations(
        latitude,
        longitude
    )

    if online_values is not None:

        z1, z2, z3 = online_values[0:3]
        z4, z5, z6 = online_values[3:6]
        z7, z8, z9 = online_values[6:9]

        # Approximate spacing used above.
        meter_x = (
            0.0003
            * 111320
            * math.cos(
                math.radians(latitude)
            )
        )

        meter_y = (
            0.0003
            * 111320
        )

        dzdx = (
            (z3 + 2 * z6 + z9)
            -
            (z1 + 2 * z4 + z7)
        ) / (
            8 * meter_x
        )

        dzdy = (
            (z7 + 2 * z8 + z9)
            -
            (z1 + 2 * z2 + z3)
        ) / (
            8 * meter_y
        )

        slope = math.degrees(
            math.atan(
                math.sqrt(
                    dzdx ** 2 +
                    dzdy ** 2
                )
            )
        )

        return float(slope)

    # --------------------------------------------------------
    # 3. Final safe fallback
    # --------------------------------------------------------

    print(
        "Slope unavailable. Using training-data "
        "median slope fallback."
    )

    return SLOPE_FALLBACK


# ============================================================
# RAINFALL
# ============================================================

def get_rainfall(latitude, longitude):

    try:

        with rasterio.open(
            rainfall_file
        ) as src:

            value = list(
                src.sample([
                    (longitude, latitude)
                ])
            )[0][0]

            if src.nodata is not None:

                if value == src.nodata:
                    return None

            return float(value)

    except Exception as e:

        print(
            "Rainfall error:",
            e
        )

        return None


# ============================================================
# LAND COVER
# ============================================================

def get_land_cover(latitude, longitude):

    lat_start = math.floor(
        latitude / 3
    ) * 3

    lon_start = math.floor(
        longitude / 3
    ) * 3

    lat_name = (
        f"N{lat_start:02d}"
        if lat_start >= 0
        else f"S{abs(lat_start):02d}"
    )

    lon_name = (
        f"E{lon_start:03d}"
        if lon_start >= 0
        else f"W{abs(lon_start):03d}"
    )

    filename = (
        "ESA_WorldCover_10m_2021_v200_"
        f"{lat_name}{lon_name}_Map.tif"
    )

    url = (
        "https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
        f"v200/2021/map/{filename}"
    )

    try:

        with rasterio.Env(
            AWS_NO_SIGN_REQUEST="YES"
        ):

            with rasterio.open(url) as src:

                value = list(
                    src.sample([
                        (longitude, latitude)
                    ])
                )[0][0]

        return int(value)

    except Exception as e:

        print(
            "Land cover error:",
            e
        )

        return None


# ============================================================
# CLAY
# ============================================================

def get_clay(latitude, longitude):

    url = (
        "https://maps.isric.org/mapserv"
        "?map=/map/clay.map"
    )

    params = [

        (
            "SERVICE",
            "WCS"
        ),

        (
            "VERSION",
            "2.0.1"
        ),

        (
            "REQUEST",
            "GetCoverage"
        ),

        (
            "COVERAGEID",
            "clay_0-5cm_mean"
        ),

        (
            "FORMAT",
            "GEOTIFF_INT16"
        ),

        (
            "SUBSETTINGCRS",
            "http://www.opengis.net/def/crs/"
            "EPSG/0/4326"
        ),

        (
            "OUTPUTCRS",
            "http://www.opengis.net/def/crs/"
            "EPSG/0/4326"
        ),

        (
            "SUBSET",
            f"X({longitude - 0.002},"
            f"{longitude + 0.002})"
        ),

        (
            "SUBSET",
            f"Y({latitude - 0.002},"
            f"{latitude + 0.002})"
        )
    ]

    try:

        response = requests.get(
            url,
            params=params,
            timeout=30
        )

        if response.status_code != 200:
            return None

        with rasterio.open(
            io.BytesIO(
                response.content
            )
        ) as src:

            value = list(
                src.sample([
                    (longitude, latitude)
                ])
            )[0][0]

        if src.nodata is not None:

            if value == src.nodata:
                return None

        return float(value) / 10

    except Exception as e:

        print(
            "Clay error:",
            e
        )

        return None


# ============================================================
# RIVER DISTANCE
# ============================================================

def get_river_distance(
    latitude,
    longitude
):

    try:

        zone = int(
            (longitude + 180) / 6
        ) + 1

        # UTM EPSG
        if latitude >= 0:
            epsg = 32600 + zone
        else:
            epsg = 32700 + zone

        nearby_rivers = rivers.cx[
            longitude - 2:
            longitude + 2,
            latitude - 2:
            latitude + 2
        ].copy()

        if len(nearby_rivers) == 0:

            # If no river is found in the local subset,
            # use a conservative finite fallback.
            return 5.0

        point = gpd.GeoDataFrame(
            geometry=[
                Point(
                    longitude,
                    latitude
                )
            ],
            crs="EPSG:4326"
        )

        point = point.to_crs(
            epsg
        )

        nearby_rivers = nearby_rivers.to_crs(
            epsg
        )

        distances = (
            nearby_rivers.geometry.distance(
                point.geometry.iloc[0]
            )
        )

        return float(
            distances.min() / 1000
        )

    except Exception as e:

        print(
            "River distance error:",
            e
        )

        return 5.0


# ============================================================
# PREDICTION
# ============================================================

def predict_location(
    latitude,
    longitude
):

    print(
        f"\nAnalyzing location: "
        f"{latitude}, {longitude}"
    )

    elevation = get_elevation(
        latitude,
        longitude
    )

    slope = get_slope(
        latitude,
        longitude
    )

    rainfall = get_rainfall(
        latitude,
        longitude
    )

    land_cover = get_land_cover(
        latitude,
        longitude
    )

    clay = get_clay(
        latitude,
        longitude
    )

    river_distance = get_river_distance(
        latitude,
        longitude
    )

    print(
        "Environmental values:"
    )

    print(
        "Elevation:",
        elevation
    )

    print(
        "Slope:",
        slope
    )

    print(
        "Rainfall:",
        rainfall
    )

    print(
        "Land Cover:",
        land_cover
    )

    print(
        "Clay:",
        clay
    )

    print(
        "River Distance:",
        river_distance
    )

    # --------------------------------------------------------
    # Only non-DEM environmental data is required to be
    # available. DEM has its own safe fallback.
    # --------------------------------------------------------

    if rainfall is None:
        raise ValueError(
            "Rainfall data unavailable for this location."
        )

    if land_cover is None:
        raise ValueError(
            "Land cover data unavailable for this location."
        )

    if clay is None:
        raise ValueError(
            "Clay data unavailable for this location."
        )

    if elevation is None:
        elevation = ELEVATION_FALLBACK

    if slope is None:
        slope = SLOPE_FALLBACK

    if river_distance is None:
        river_distance = 5.0

    input_data = pd.DataFrame([
        {
            "Elevation": elevation,
            "Slope": slope,
            "Annual_Rainfall": rainfall,
            "Land_Cover": land_cover,
            "Clay_Content": clay,
            "Distance_to_River_km": river_distance
        }
    ])

    probability = (
        xgb_model.predict_proba(
            input_data
        )[0][1] * 100
    )

    if probability < 30:

        risk = "Low"

    elif probability < 60:

        risk = "Moderate"

    elif probability < 80:

        risk = "High"

    else:

        risk = "Very High"

    print(
        f"XGBoost susceptibility: "
        f"{probability:.2f}%"
    )

    return {

        "probability": round(
            float(probability),
            2
        ),

        "risk": risk,

        "elevation": round(
            float(elevation),
            2
        ),

        "slope": round(
            float(slope),
            2
        ),

        "rainfall": round(
            float(rainfall),
            2
        ),

        "land_cover": int(
            land_cover
        ),

        "clay": round(
            float(clay),
            2
        ),

        "river_distance": round(
            float(river_distance),
            3
        ),

        "dem_source": (
            "local SRTM"
            if get_local_dem_file(
                latitude,
                longitude
            ) is not None
            else "online/fallback"
        )
    }

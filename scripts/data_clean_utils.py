#data_clean_utils
# scripts/data_clean_utils.py

import numpy as np
import pandas as pd
import re
from pathlib import Path
import ast
import logging

# =====================================================
# LOGGER SETUP
# =====================================================
logger = logging.getLogger("property_data_cleaning")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)


# =====================================================
# DEBUG UTILITIES
# =====================================================
def debug_df_snapshot(df: pd.DataFrame, title: str, max_cols: int = 40):
    """
    Print one-row snapshot for debugging. Helps identify where None/NaN appears.
    """
    try:
        logger.info(f"\n--- DEBUG SNAPSHOT: {title} ---")

        if df is None:
            logger.info("DF is None")
            return

        logger.info(f"Shape: {df.shape}")

        if df.empty:
            logger.info("DF is empty")
            return

        row = df.iloc[0]

        cols = list(df.columns)[:max_cols]
        logger.info("Top columns preview:")
        for c in cols:
            logger.info(f"{c} = {row.get(c)}")

        # show top missing columns
        na_cnt = df.isna().sum().sort_values(ascending=False).head(15)
        logger.info("Top NA columns:")
        logger.info("\n" + na_cnt.to_string())

        logger.info("--- END SNAPSHOT ---\n")

    except Exception as e:
        logger.error(f"debug_df_snapshot failed: {repr(e)}")


def debug_value(df: pd.DataFrame, col: str, title: str = ""):
    """
    Print dtype + first-row value + python type for a column.
    """
    try:
        if df is None or df.empty or col not in df.columns:
            logger.info(f"[DEBUG VALUE] {title} | Column missing or DF empty: {col}")
            return

        val = df[col].iloc[0]
        logger.info(f"[DEBUG VALUE] {title} | {col}")
        logger.info(f"  dtype: {df[col].dtype}")
        logger.info(f"  value: {val}")
        logger.info(f"  python_type: {type(val)}")

    except Exception as e:
        logger.error(f"debug_value failed: {repr(e)}")

#------------------------------------------------------
def ensure_string_col(df, col, fill=""):
    if col not in df.columns:
        df[col] = fill
    df[col] = df[col].astype("string").fillna(fill)
    return df

# =====================================================
# LOAD DATA
# =====================================================
def load_data(data_path: Path) -> pd.DataFrame:
    try:
        logger.info(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully | Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error("Input data file not found.")
        raise


# =====================================================
# BASIC CLEANING
# =====================================================
def basic_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting basic property data cleaning")
    df = data.copy()

    if df.empty:
        raise ValueError("No input row provided.")

    debug_df_snapshot(df, "RAW INPUT (Before lowercase + drops)")

    try:
        # convert column names into lowercase
        df.columns = df.columns.str.lower()
        debug_df_snapshot(df, "AFTER LOWERCASE COLUMNS")

        # drop unwanted columns
        df.drop([
            '@id', '@type', 'bhk_type', 'locality_url',
            'md_booking amount', 'md_loan offered',
            'ap_price', 'ap_price per sqft', 'ap_configuration',
            'ap_pjt_url', 'ap_ratings', 'ap_reviews_by',
            'headings_with_ratings', 'aboutpjt_bhk',
            '2 bhk flat', '3 bhk flat', '1 bhk flat',
            'studio apartment', '4 bhk flat', '5 bhk flat',
            'multistorey apartment', '3 bhk villa', '4 bhk villa',
            'residential plot', '2 bhk builder', '3 bhk builder',
            '4 bhk penthouse', '5 bhk penthouse',
            '6 bhk flat', 'rent',
            'commercial office space', '3 bhk penthouse'
        ], axis=1, inplace=True, errors="ignore")

        # Rename columns for clarity
        df.rename(columns={
            'md_water availability': 'water_availability_hours',
            'locality_url_review': 'locality_review_count',
            'liv_environment': 'environment_rating',
            'liv_commuting': 'commuting_rating',
            'liv_places of interest': 'places_of_interest_rating',
            'md_status of electricity': 'electricity_availability_hours',
            'md_landmarks': 'nearby_landmarks',
            'md_authority approval': 'authority_approval',
            'md_rera id': 'rera_id',
            'aboutpjt_launch date': 'project_launch_date',
            'locality_rank': 'locality_rank',
            'locality_url_rating': 'locality_rating'
        }, inplace=True)

        df.drop_duplicates(inplace=True)

        # drop columns with all nan values
        df.drop([
            'bb_unfurnished', 'leftbb_unfurnished',
            'bb_semi-furnished', 'leftbb_semi-furnished',
            'bb_furnished', 'leftbb_furnished',
            '5 bhk villa', 'many_plot area'
        ], axis=1, inplace=True, errors="ignore")

        # make all string column values lowercase
        df[df.select_dtypes('object').columns] = df.select_dtypes('object').apply(
            lambda col: col.map(lambda x: x.lower() if isinstance(x, str) else x)
        )

        # ------------------------------------------------------------------------------------
        # Helper functions
        # ------------------------------------------------------------------------------------
        def check_more_than_one_value_in_column(df, cols, new_col_name, col_name):
            df[new_col_name] = df[cols].notna().sum(axis=1) > 1

            def combine_values():
                df[col_name] = [list(values) for values in zip(*[df[col] for col in cols])]
            combine_values()

            found_distinct = False

            if df[new_col_name].any():
                for index, row in df[col_name].items():
                    non_nan_vals = [val for val in row if pd.notna(val)]
                    if len(set(non_nan_vals)) > 1:
                        found_distinct = True

            if not found_distinct:
                df[col_name] = df[col_name].apply(
                    lambda row: next((val for val in row if pd.notna(val)), None)
                )

        def combine_first_valid(df, source_cols, new_col_name):
            df[new_col_name] = df[source_cols].apply(
                lambda row: next(
                    (
                        str(x) for x in row
                        if pd.notna(x)
                        and str(x).strip().lower() != 'nan'
                        and str(x).strip() != ''
                    ),
                    np.nan
                ),
                axis=1
            )
            df[new_col_name] = df[new_col_name].astype(str).str.strip().str.lower()
            df.drop(columns=source_cols, inplace=True, errors="ignore")
            return df

        # ------------------------------------------------------------------------------------
        # BED / BATH
        # ------------------------------------------------------------------------------------
        check_more_than_one_value_in_column(
            df,
            ['numberofrooms', 'bb_beds', 'leftbb_beds', 'bb_bed', 'leftbb_bed'],
            'multi_bed_filled',
            'bed'
        )

        check_more_than_one_value_in_column(
            df,
            ['bb_baths', 'leftbb_baths', 'bb_bath', 'leftbb_bath'],
            'multi_bath_filled',
            'bath'
        )

        # ------------------------------------------------------------------------------------
        # PARKING FIXED (None-safe)
        # ------------------------------------------------------------------------------------
        parking_cols = ['bb_covered-parking', 'leftbb_covered-parking',
                        'many_car parking', 'leftmany_car parking']

        # Make sure cols exist
        for c in parking_cols:
            if c not in df.columns:
                df[c] = np.nan

        # Convert string parking like "2 covered" etc already done for many_ cols above
        # Now force numeric + None-safe
        df[parking_cols] = df[parking_cols].replace({None: np.nan})
        df[parking_cols] = df[parking_cols].apply(pd.to_numeric, errors="coerce")

        df["parking"] = df[parking_cols].max(axis=1, skipna=True)


        # ------------------------------------------------------------------------------------
        # AREA (carpet + super built-up)
        # ------------------------------------------------------------------------------------
        df['area_work'] = df["many_carpet area"].combine_first(df["leftmany_carpet area"])

        df["carpet_area"] = df["area_work"].apply(
            lambda x: float(re.match(r'([\d,\.]+)', x).group(1).replace(',', ''))
            if pd.notna(x) and re.match(r'^[\d,\.]+', x)
            else None
        )

        df['cost_per_sqft'] = df['area_work'].str.extract(r'₹([\d,\.]+)')[0].str.replace(',', '').astype(float)
        df['area_unit'] = df['area_work'].str.extract(r'/([^/]+)$')

        # STRICT unit check (DO NOT CRASH -> DROP bad rows)
        if "area_unit" in df.columns and len(df) > 0:
            bad_units = df["area_unit"].astype(str).str.strip().str.lower().isin(["sqm", "kanal"])
            if bad_units.any():
                logger.warning(f"Unsupported area units found: {df.loc[bad_units, 'area_unit'].unique()}. Setting area fields to NaN instead of dropping.")
                df.loc[bad_units, ["carpet_area", "cost_per_sqft", "area_unit"]] = np.nan


        df['super_build_area_work'] = df["leftmany_super built-up area"].combine_first(df["many_super built-up area"])

        df['initial_unit'] = df['super_build_area_work'].apply(
            lambda x: ''.join([char for char in str(x)[re.match(r'\d+', str(x)).end():] if char.isalpha()])[:4]
            if isinstance(x, str) and re.match(r'\d+', str(x))
            else None
        )

        if "initial_unit" in df.columns and len(df) > 0:
            bad_units = df["initial_unit"].astype(str).str.strip().str.lower().isin(["sqms", "sqyr"])
            if bad_units.any():
                logger.warning(f"Unsupported initial units found: {df.loc[bad_units, 'initial_unit'].unique()}. Setting super-built-up fields to NaN instead of dropping.")
                df.loc[bad_units, ["super_build_up_area", "super_build_up_cost_per_sqft", "initial_unit"]] = np.nan



        df["super_build_up_area"] = df["super_build_area_work"].apply(
            lambda x: float(re.match(r'([\d,\.]+)', x).group(1).replace(',', ''))
            if pd.notna(x) and re.match(r'^[\d,\.]+', x)
            else None
        )

        df['super_build_up_cost_per_sqft'] = df['super_build_area_work'].str.extract(r'₹([\d,\.]+)')[0].str.replace(',', '').astype(float)
        df['super_built_up_area_unit'] = df['super_build_area_work'].str.extract(r'/([^/]+)$')

        df['f_area'] = df["carpet_area"].combine_first(df["super_build_up_area"])
        df['f_area_unit'] = df["super_built_up_area_unit"].combine_first(df["area_unit"])

        df['f_area'] = pd.to_numeric(df['f_area'], errors="coerce")

        df['dupli_f_area'] = np.where(
            pd.isna(df['f_area']) & pd.notna(df['area']),
            df['area'].str.extract(r'([\d,\.]+)')[0].str.replace(',', '').astype(float),
            None
        )

        df['dupli_f_area_unit'] = np.where(
            pd.isna(df['f_area']) & pd.notna(df['area']),
            df['area'].str.extract(r'([a-zA-Z\-]+)$')[0],
            None
        )

        def update_values(df, update_cols, using_cols):
            for update_col, using_col in zip(update_cols, using_cols):
                df[update_col] = np.where(
                    pd.isna(df['f_area']) & pd.notna(df['area']),
                    df[using_col],
                    df[update_col]
                )
            return df

        df = update_values(df, ['f_area_unit', 'f_area'], ['dupli_f_area_unit', 'dupli_f_area'])

        cols_to_drop = [
            'many_carpet area', 'leftmany_carpet area',
            'leftmany_super built-up area', 'many_super built-up area',
            'area', 'area_work', 'carpet_area', 'cost_per_sqft',
            'area_unit', 'initial_unit', 'super_build_area_work',
            'super_build_up_area', 'super_build_up_cost_per_sqft',
            'super_built_up_area_unit', 'dupli_f_area',
            'dupli_f_area_unit', 'f_area_unit'
        ]
        df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

        df = df.rename(columns={'f_area': 'area'})
        df['area'] = pd.to_numeric(df['area'], errors="coerce")

        # ------------------------------------------------------------------------------------
        # FLOOR
        # ------------------------------------------------------------------------------------
        df['floor_work_1'] = df['many_floor'].combine_first(df['leftmany_floor']).astype(str)

        df['flat_on_floor'] = df['floor_work_1'].apply(
            lambda x: x.split('(')[0].strip() if '(' in str(x) else None
        )

        df['total_floor'] = df['floor_work_1'].apply(
            lambda x: x.split('(')[1].strip() if '(' in str(x) else None
        )

        df['total_floor'] = df['total_floor'].str.extract(r'(\d+)').astype(float)
        df['flat_on_floor'] = df['flat_on_floor'].replace({'lower basement': -1, 'upper basement': -2, 'ground': 0})
        df['total_floor'] = np.where(
            pd.isna(df['total_floor']) & pd.notna(df['md_floors allowed for construction']),
            df['md_floors allowed for construction'],
            df['total_floor']
        )
        df['flat_on_floor'] = pd.to_numeric(df['flat_on_floor'], errors="coerce")

        # ------------------------------------------------------------------------------------
        # LIFT
        # ------------------------------------------------------------------------------------
        check_more_than_one_value_in_column(
            df,
            ['many_lifts', 'md_lift', 'leftmany_lifts', 'many_lift', 'leftmany_lift'],
            'multi_lift_filled',
            'lift'
        )

        # ------------------------------------------------------------------------------------
        # BALCONY
        # ------------------------------------------------------------------------------------
        df['balcony'] = (
            df['bb_balcony']
            .combine_first(df['leftbb_balcony'])
            .combine_first(df['bb_balconies'])
            .combine_first(df['leftbb_balconies'])
        )

        # ------------------------------------------------------------------------------------
        # GEO -> LAT/LON
        # ------------------------------------------------------------------------------------
        df['lattitude'] = df['geo'].str.split(',').str[1].str.split(':').str[1].str.strip(" '\"")
        df['longitude'] = df['geo'].str.split(',').str[2].str.split(':').str[1].str.strip(" '\"}")

        df["lattitude"] = pd.to_numeric(df["lattitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

        # ------------------------------------------------------------------------------------
        # PROJECT IN ACRES FIXED SAFE NUMERIC
        # ------------------------------------------------------------------------------------
        def convert_to_acres(value):
            if isinstance(value, str):
                if 'acre' in value:
                    return round(float(value.replace('acre', '').strip()), 4)
                elif 'sq-m' in value:
                    return round(float(value.replace('sq-m', '').strip()) * 0.000247105, 4)
                elif 'sq-ft' in value:
                    return round(float(value.replace('sq-ft', '').strip()) * 0.0000229568, 4)
                elif 'hectare' in value:
                    return round(float(value.replace('hectare', '').strip()) * 2.47105, 4)
                elif 'sq-yrd' in value:
                    return round(float(value.replace('sq-yrd', '').strip()) * 0.000836127, 4)
            elif isinstance(value, (int, float)):
                return round(value * 0.0000229568, 4)
            return np.nan

        df['project_in_acres'] = df['aboutpjt_project size'].apply(convert_to_acres)
        df['project_in_acres'] = pd.to_numeric(df['project_in_acres'], errors="coerce")

        debug_value(df, "project_in_acres", "AFTER project_in_acres conversion")

        # Safe filtering: DO NOT DROP rows in API prediction
        if "project_in_acres" in df.columns and len(df) > 0:
            too_large = df["project_in_acres"] > 1000
            too_small = (df["project_in_acres"] <= 0.005) & df["project_in_acres"].notna()

            if too_large.any() or too_small.any():
                logger.warning(
                    f"project_in_acres invalid -> setting NaN | too_large={too_large.sum()} too_small={too_small.sum()}"
                )
                df.loc[too_large | too_small, "project_in_acres"] = np.nan


        # ------------------------------------------------------------------------------------
        # WATER AVAILABILITY HOURS
        # ------------------------------------------------------------------------------------
        if "water_availability_hours" in df.columns:
            df['water_availability_hours'] = df['water_availability_hours'].astype(str).str.split(' ').str[0]
            df['water_availability_hours'] = pd.to_numeric(df['water_availability_hours'], errors='coerce')

        # ------------------------------------------------------------------------------------
        # ENVIRONMENT/COMMUTING/INTEREST
        # ------------------------------------------------------------------------------------
        for col in ['environment_rating', 'commuting_rating', 'places_of_interest_rating']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.split('/').str[0]
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # ------------------------------------------------------------------------------------
        # PROJECT AGE MONTHS
        # ------------------------------------------------------------------------------------
        if "project_launch_date" in df.columns:
            df['project_launch_date'] = pd.to_datetime(df['project_launch_date'], format='%b-%y', errors="coerce")

            extract_date = pd.to_datetime('2024-12-01')
            df['project_age_months'] = (extract_date.year - df['project_launch_date'].dt.year) * 12 + \
                                      (extract_date.month - df['project_launch_date'].dt.month)

            df.drop(columns='project_launch_date', inplace=True, errors="ignore")

    
        #--------------------------------------------------------------------------------------------------------------------------------------------------------------
        #categorical column
        #--------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        #builder
        df = combine_first_valid(
            df,
            source_cols=['many_developer','leftmany_developer','ap_buildr'],
            new_col_name='builder'
        )

        def standardize_property_name(name):
            """
            Standardize property names to a single consistent name.
            """
            # Define a mapping of possible variations to standardized names
            mapping = {
                "a&o realty / a and o realty / a & o realty ltd.": "a&o realty",
                "adhiraj constructions / adhiraj constructions pvt. ltd.": "adhiraj constructions",
                "arihant superstructures ltd / arihant superstructures ltd.": "arihant superstructures ltd",
                "bharat infrastructure & engineering pvt. ltd. / bharat infrastructure and engineering": "bharat infrastructure & engineering",
                "bhoomi group / bhoomi / bhoomi properties": "bhoomi group",
                "choice group of companies / choice group": "choice group",
                "darshan properties / darshan properties group": "darshan properties",
                "dev land housing / dev land housing ltd.": "dev land housing",
                "ecohomes / eco homes": "ecohomes",
                "gundecha developers / gundecha / gundecha developing milestone /gundecha group": "gundecha group",
                "hiranandani communities / hiranandani constructions / hiranandani developers / hiranandani group / house of hiranandani": "hiranandani group",
                "k raheja realty/ k. raheja realty": "k raheja realty",
                "krishna enterprise / krishna enterprises": "krishna enterprise",
                "l & t realty / l&t realty": "l&t realty",
                "lodha / lodha group": "lodha group",
                "lok housing group / lok group": "lok housing group",
                "lokhandwala builders / lokhandwala constructions / lokhandwala construction industries pvt. ltd. / lokhandwala group": "lokhandwala group",
                "lokhandwala infrastructure": "lokhandwala infrastructure",
                "lotus logistic and developers / lotus logistics & developer pvt ltd": "lotus logistics",
                "neelam realtors / neelam realtors pvt. ltd.": "neelam realtors",
                "neelsidhi group / neelsidhi": "neelsidhi group",
                "nirmal lifestyle / nirmal life style": "nirmal lifestyle",
                "omkar realtors and developers pvt. ltd. / omkar realtors": "omkar realtors",
                "parinee developers / parinee group": "parinee group",
                "platinum group / platinum group builders / platinum constructions": "platinum group",
                "prescon group / prescon": "prescon group",
                "puraniks builders / puranik builders ltd. / puranik group": "puraniks group",
                "r k builders / r k builders and developers": "r k builders",
                "qualcon properties llp / qualcon": "qualcon",
                "raheja universal (pvt.) ltd. / raheja universal pvt. ltd.": "raheja universal",
                "raheja developers / raheja developers ltd.": "raheja developers",
                "raj realty group / raj realty": "raj realty group",
                "rashmi housing pvt. ltd. / rashmi housing": "rashmi housing",
                "ravi group of builders and developers / ravi group": "ravi group",
                "rna / rna ng builders / rna corp / rna group": "rna group",
                "rohan lifescapes / rohan lifescapes ltd.": "rohan lifescapes",
                "romell group / romell real estate pvt. ltd.": "romell group",
                "rustomjee / rustomjee developers": "rustomjee",
                "sahajanand developers / sahajanand infrastructure pvt. ltd.": "sahajanand developers",
                "sainath developers / sainath group": "sainath developers",
                "sapphire group and builder / sapphire group": "sapphire group",
                "saptashree builders & developers / sapta shree builders & developers": "saptashree",
                "shapoorji pallonji real estate / shapoorji pallonji group": "shapoorji pallonji group",
                "sheth creators / sheth creators pvt. ltd.": "sheth creators",
                "shree ostwal builders ltd. / shree ostwal builders and developers": "shree ostwal builders",
                "shreedham builders and developers / shreedham group": "shreedham group",
                "shreeji construction / shreeji group / shreeji group builder and developer": "shreeji group",
                "smgk associates / smgk group": "smgk group",
                "space india / space india builders & developers": "space india",
                "spenta builders / spenta corp. pvt. ltd.": "spenta",
                "sugee realty & developers (india) pvt. ltd. / sugee group": "sugee group",
                "swastik realtors / swastik group builders & developers": "swastik",
                "tharwani realty / tharwani group": "tharwani group",
                "titanium group / titanium builders and developers": "titanium group",
                "today global homes / today global builders & developers": "today global",
                "transcon developers / transcon group": "transcon group",
                "tridhaatu realty / tridhaatu realty & infra pvt. ltd.": "tridhaatu realty",
                "vaibhavlaxmi builders & developers / vaibhavlaxmi builders and developers / vaibhav laxmi developers": "vaibhavlaxmi builders",
                "vbhc value homes pvt. ltd. / vbhc": "vbhc",
                "vihang infrastructure pvt ltd / vihang group": "vihang group",
                "vinay unique developers / vinay unique group": "vinay unique group"
            }
        
            # Handle null or missing values
            if not isinstance(name, str):
                return name
        
            # Normalize input name (e.g., lowercase, strip whitespace)
            normalized_name = name.strip().lower()
        
            # Standardize using the mapping
            for key, value in mapping.items():
                variations = key.split(" / ")
                if normalized_name in variations:
                    return value
        
            # Return the original name if no match is found
            return name
        
        # Apply the function to the 'builder' column of a DataFrame
        df['builder'] = df['builder'].apply(standardize_property_name)

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        #project_name
        df = combine_first_valid(
            df,
            source_cols=['ap_pjt_name', 'many_project', 'leftmany_project'],
            new_col_name='project_name'
        )

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        #furnish
        df = combine_first_valid(
            df,
            source_cols=['md_furnishing','many_furnished status','leftmany_furnished status'],
            new_col_name='furnish'
        )

        # ------------------------------------------------------------------------------------
        # CITY
        # ------------------------------------------------------------------------------------
        df = df.rename(columns={'address': 'wholeaddress'})
        df['addressregion'] = df['wholeaddress'].apply(
            lambda x: ast.literal_eval(x).get('addressregion') if isinstance(x, str) else x.get('addressregion')
        )
        df = df.rename(columns={'md_address': 'address'})
        df = df.rename(columns={'addressregion': 'city'})

        # ------------------------------------------------------------------------------------
        # LOCATION
        # ------------------------------------------------------------------------------------
        df["wholeaddress"] = df["wholeaddress"].apply(ast.literal_eval)
        df["location"] = df["wholeaddress"].apply(lambda x: x.get("addresslocality", ""))

        df['address'] = df['address'].astype(str).str.replace(r'\brd\b', 'road', regex=True)

        # if blank -> NaN
        df['location'] = df['location'].replace('', np.nan)
        
        #if below values match found in address column,then update location with the matched value
        
        lst = [
            "mira road east", "mira road west", "mira rd east", "mira rd west",
            "vile parle east", "vile parle west", "lower parel west", "lower parel east",
            "new panvel east", "new panvel west", "grand road east", "grand road west",
            "charni road east", "charni road west", "grand rd east", "grand rd west",
            "charni rd east", "charni rd west", "kanjur marg east", "kanjur marg west",
            "mira bhayandar east", "mira bhayandar west", "marine lines east", "marine lines west",
            "ram mandir west", "ram mandir east", "vasai road west", "vasai road east",
            "matunga road west", "matunga road east", "vasai rd west", "vasai rd east",
            "matunga rd west", "matunga rd east", "rajendra nagar west", "rajendra nagar east",
            "tilak nagar west", "tilak nagar east", "diva station east", "diva station west",
            "ville parla west", "ville parla east", "lower pare west", "lower pare east",
            "mumbai central east", "mumbai central west"
        ]
        # Step 1: Filter NaN rows
        df_nan1 = df[df['location'].isna()].copy()
        
        # Step 2 & 3: Match with lst and update location
        for index, row in df_nan1.iterrows():
            for loc in lst:
                if loc in row['address'].lower():  # Case insensitive match
                    df.at[index, 'location'] = loc
                    break  # Stop at first match

        # Function to extract "<name> east" or "<name> west" from 'address'
        def extract_location(address):
            # Use regex to find a word followed by 'east' or 'west'
            match = re.search(r'(\w+)\s+(east|west)', address, re.IGNORECASE)
            if match:
                return f"{match.group(1)} {match.group(2)}"
            return np.nan  # If no match, return NaN
        
        # Filter the rows where 'location' is NaN
        df_nan2 = df[df['location'].isna()]
        
        # Apply the extract_location function only to the 'address' column in the filtered rows
        df.loc[df_nan2.index, 'location'] = df_nan2['address'].apply(extract_location)
        
        mapping = {"near naupada police station, thane, maharashtra" : "thane west",
                "new suyash chs naupada, thane, maharashtra" : "thane west",
                "marine lines, mumbai, maharashtra" : "marine lines",
                "kalher, thane, maharashtra" : "bhiwandi",
                "kanikiya beverly park mira road, mumbai, maharashtra" : "mira road east",
                "204, 2nd flr, ramraj bldg, nr. ram mandir, rajanpada - sector-27, navi mumbai, maharashtra" : "sector 27 rajanpada",
                "202 dhrmasetu plot no 2225 sec 19 koperkhairane, navi mumbai, maharashtra" : "sector 19 koperkhairane",
                "kashimira near whestran express hyway, mumbai, maharashtra" : "mira road east",
                "near burhani college mazgaon mumbai 10, mumbai, maharashtra" : "mazgaon",
                "ulwe sector 21, navi mumbai, maharashtra" : "sector 21 ulwe",
                "lakeshore greens by lodha, thane, maharashtra" : "dombivli west",
                "santa cruz, mumbai, maharashtra" : "santacruz",
                "kopar khairane, navi mumbai, maharashtra" : "koparkhairane",
                "1801, 18th floor, chunam lane, lamington road, grantroad e, mumbai 400007, mumbai, maharashtra" : "grant road east",
                "charkop village near dingeshwar talao and jalaram temple, mumbai, maharashtra" : "kandivali west",
                "sarfaraz iqbal heights, ymca road 3, near maratha mandir, mumbai central, mumbai, maharashtra" : "mumbai central",
                "panchseel heights, mahavir nagar, mumbai, maharashtra" : "kandivali west",
                "sector 5 pushpak nagar, navi mumbai, maharashtra" : "sector 5 pushpak nagar",
                "poonam park view, global city, virar, thane, maharashtra" : "virar west",
                "om ekdant soc, sec-19, koperkharine, near jummy tower, navi mumbai, maharashtra" : "sector 19 koperkharine",
                "sai vrindhavan koparkhairne., navi mumbai, maharashtra" : "koparkhairne",
                "owale, ghodbunder road, thane, maharashtra" : "thane west",
                "sector 21 ulwe, navi mumbai, maharashtra" : "sector 21 ulwe",
                "dombivli, mumbai, maharashtra" : "dombivli",
                "amber enclave - 3rd floor thakurli e, mumbai, maharashtra" : "thakurli east",
                "anath sai apartment, thane, maharashtra" : "thane west",
                "willingdon heights 32nd flr near tardeo rto tulsiwadi, mumbai, maharashtra" : "tardeo",
                "12th floor c2 wing treetops lodha upper thane mankoli bhiwandi thane maharashtra 421302, mumbai, maharashtra" : "bhiwandi",
                "chincholi phatak, mumbai, maharashtra" : "malad west",
                "kanakiya, mumbai, maharashtra" : "kandivali east",
                "puranik hometown kasarvadavli, mumbai, maharashtra" : "thane west",
                "boraivali w 401, mumbai, maharashtra" : "borivali west",
                "prabhadevi, mumbai, maharashtra" : "prabhadevi",
                "green road, thane, maharashtra" : "thane west",
                "lagoona, thane, maharashtra" : "thane west",
                "kasarvadavli, thane, maharashtra" : "thane west",
                "kasarvadavli, thane, maharashtra" : "thane west",
                "dr annie besant road, worli, mumbai, maharashtra 400018, india, mumbai, maharashtra" : "worli",
                "gorai 2, mumbai, maharashtra" : "gorai",
                "lodha casa lakeshore green khoni dombivli, nilje gaon, maharashtra 421204, india, thane, maharashtra" : "dombivli east",
                "diamind garden chembur, mumbai, maharashtra" : "chembur",
                "sector 17 kamothe, navi mumbai, maharashtra" : "kamothe",
                "highland complex, mumbai, maharashtra" : "kandivali east",
                "jerbai wadia road, near tata hospital, parel, mumbai, maharashtra" : "parel",
                "gokhale road, naupada thane, thane, maharashtra" : "naupada",
                "taloja phase 2, navi mumbai, maharashtra" : "taloja",
                "ghansoli sector 11, navi mumbai, maharashtra" : "ghansoli",
                "ramnagar, thane, maharashtra" : "thane west",
                "ram maruti, thane, maharashtra" : "thane west",
                "marine lines, mumbai, maharashtra" : "marine lines",
                "sector 12 vashi., navi mumbai, maharashtra" : "sector 12 vashi",
                "just opposite of mansarovar railway station, navi mumbai, maharashtra" : "mansarovar",
                "bhaskar colony, thane, maharashtra" : "thane west",
                "taloja phase 2, navi mumbai, maharashtra" : "taloja",
                "charkop sector 3charkop gaon, mumbai, maharashtra" : "kandivali west",
                "157, pantnagar, 1st building naidu colony, mumbai, maharashtra" : "ghatkopar east",
                "godrej chandivali, mumbai, maharashtra" : "chandivali",
                "kalwa, thane, thane, maharashtra" : "kalwa",
                "ghansoli, navi mumbai, maharashtra" : "ghansoli",
                "suncity corner seawoodnerul, navi mumbai, maharashtra" : "nerul",
                "lagoona, thane, maharashtra" : "dombivli east",
                "satyam apartment, sector 19, kharghar, navi mumbai, maharashtra" : "kharghar",
                "tilak nagar chembur, mumbai 400089., mumbai, maharashtra" : "chembur",
                "401, sai aakash co op housing society, plot no.23, sector 18, ulwe, navi mumbai, maharashtra" : "sector 18 ulwe",
                "palava casa bella gold, mumbai, maharashtra" : "palava",
                "near vitthal mandir kharigaon kalwa, thane, maharashtra" : "kalwa",
                "kharghar, navi mumbai, maharashtra" : "kharghar",
                "neral karjat, mumbai, maharashtra" : "neral",
                "pahhal avenue, mumbai, maharashtra" : "goregaon west",
                "157, naidu colony, pantnagar, mumbai, maharashtra" : "ghatkopar east",
                "mangalmurthy complex, temghar, thane, maharashtra" : "bhiwandi",
                "plot no b1b, sector 9, airoli navimumbai, mumbai, maharashtra" : "sector 9 airoli",
                "chikhloli jambul phata, thane, maharashtra" : "chikhloli",
                "bapu nagar apartment., thane, maharashtra" : "bapu nagar",
                "crown taloja by lodha, taloja bypass phata, antarli, maharashtra 421204, mumbai, maharashtra" : "taloja",
                "morya garden residency vichumbe, navi mumbai, maharashtra" : "new panvel east",
                "sec-19, navi mumbai, maharashtra" : "sector 19 navi mumbai",
                "siddhivinayak appartment airoli diva koliwada near airoli mulund bridge diva goan gavthan, navi mumbai, maharashtra" : "airoli",  
                "kalher, thane, maharashtra" : "kalher",  
                "vinay nagar, mira road, mumbai, maharashtra" : "mira road east",
                "shree siddhivinayak tower vartaknagar, thane, maharashtra" : "vartaknagar",
                "kasarwadvali godbandar road thane, thane, maharashtra" : "kasarwadvali",  
                "panvel matheran road opp balaji symphony sukapur, navi mumbai, maharashtra" : "panvel",
                "sector 19, shahbaz gaon, cbd belapur, navi mumbai, navi mumbai, maharashtra" : "cbd belapur",
                "gamdevi grant road, mumbai, maharashtra" : "gamdevi",
                "dongri sandhurst road, mumbai, maharashtra" : "dongri",
                "casa rio arebiana, thane, maharashtra" : "thane",
                "lalani dreams residency, village dahivali turfe nid, taluka karjat, mumbai, maharashtra" : "karjat",
                "lodha crown akbar camp road kolshet mumbai maharashtra, mumbai, maharashtra" : "kolshet",  
                "202 sai shruti residency plot c 30 sector 4 khanda colony new panvel 410206, navi mumbai, maharashtra" : "new panvel",  
                "casa milano 12th floor - lodha palava phase 2 dombivali kalyan, navi mumbai, maharashtra" : "dombivli",  
                "203, sunrise glory shilphata near daighar police station, navi mumbai, maharashtra" : "shilphata", 
                "dronagiri navi mumbai., mumbai, maharashtra" : "dronagiri",  
                "muthaval, thane, maharashtra" : "muthaval",  
                "sector 5 koperkhairne navi mumbai, navi mumbai, maharashtra" : "koperkhairne",  
                "304, audumber chaya chsl, patilwadi, savarkar nagar, behind thakur college, thane, maharashtra" : "thane west",
                "old panvel near savarkar chowk., navi mumbai, maharashtra" : "old panvel",  
                "opposite j p international school haranwadi naka, mahim road, palghar, palghar, maharashtra" : "palghar",
                "tower 13 2003 runwal gardens dombivali, thane, maharashtra" : "dombivli",
                "village boisar, tal palghar, dist. thane, palghar, maharashtra" : "boisar",  
                "century bazar near chroma showroom, mumbai, maharashtra" : "century bazar",  
                "d/305., palghar, maharashtra" : "palghar",  
                "e 2 303 gaurav citymira road area, mumbai, maharashtra" : "mira road east",
                "umiya darshan chs, nerul sec 50 new, navi mumbaiseawoods, navi mumbai, maharashtra" : "seawoods",  
                "rambhau mhalgi marg, besides shrushti residency, khambalpada, thakurli e, dombivli e, thane, maharashtra" : "thakurli east",
                "ramabai paradise opp garden city tawor mira road thane, mumbai, maharashtra" : "mira road",  
                "siddhivinayak florentia garden citymira bhayandar, mumbai, maharashtra" : "mira bhayandar",  
                "bonkode sector 12, navi mumbai, maharashtra" : "sector 12 bonkode",  
                "vasant villa, padmavati devi marg, iit market, powai, mumbai 400076, mumbai, maharashtra" : "powai",  
                "novapark co opp housing society ltd flat no 303 plot no 68., navi mumbai, maharashtra" : "navi mumbai",
                "mira road area, mumbai, maharashtra" : "mira road",  
                "near divya heights in sector 26 navi mumbai, navi mumbai, maharashtra" : "sector 26 navi mumbai",
                "ganesh nagar, near boisar railway starion, palghar, maharashtra" : "boisar",  
                "c-001 nand dham building kashimira mira road, mumbai, maharashtra" : "mira road east",
                "om sankalp chs, kopar road, thane 421202, thane, maharashtra" : "dombivli west",
                "svarna kojagiri, mumbai, maharashtra" : "goregaon east",
                "unique aurum, poonam garden, thane, maharashtra" : "mira road east",
                "neelkanth darshan society b-203125a near hotel panvel palaceold panvel, mumbai, maharashtra" : "old panvel",
                "mira road kanakia, thane, maharashtra" : "mira road east",
                "panvel, navi mumbai, navi mumbai, maharashtra" : "navi mumbai",
                "chitalsar manpada, thane, maharashtra" : "manpada",  
                "near raj kamal studio, parel, mumbai, maharashtra" : "parel",  
                "nilje station road, nilje, thane, maharashtra" : "nilje", 
                "flat no-604, plot no-4, sector 14, taloja, navi mumbai, maharashtra" : "taloja",  
                "jethe tower, 701, ambawadi, opp. ambawadi bus stop, borivali e. mumbai-400068, mumbai, maharashtra" : "borivali east",  
                "lodha crown viva, flat 1006, 10th flr tower 5, majiwada, thane, mumbai, maharashtra" : "majiwada",  
                "sunbeam heritage hsg soc, sector 4c, khanda colony asudgoan panvel, navi mumbai, maharashtra" : "panvel",
                "lodha upper thane, treetops, thane, maharashtra" : "upper thane",  
                "aanandi park a101 behind ganapati mandir durgesh park kalher bhiwandi, thane, maharashtra" : "kalher",  
                "a-9/201 tejaswi apt, near st. thomas church, sai baba nagar, mira road., mumbai, maharashtra" : "mira road east",
                "sector 11, next to miraj cinema, navi mumbai, maharashtra" : "sector 11",
                "aster, regency anantham, dombivli, mumbai, maharashtra" : "dombivli",  
                "chand nagar, near baba medical, thane, maharashtra" : "thane", 
                "thane majiwada lodha complex opp-water tank, thane, maharashtra" : "majiwada",  
                "near kalidas natyamamdir, mumbai, maharashtra" : "mulund west",
                "badlapur, thane, maharashtra" : "badlapur",  
                "near mittal club, palghar, maharashtra" : "palghar",  
                "shree krupa apt flat no 102 plot144145 sector10 new panvel navi mumbai, navi mumbai, maharashtra" : "new panvel",  
                "sector 20, cbd belapur opp bank of india  park, adjacent to hansraj building, navi mumbai, maharashtra" : "sector 20 cbd belapur",  
                "brahmand patlipada link road, opp tulsi hotel, thane, maharashtra" : "thane",  
                "gurukiran socity airoli sector 30 gothavali, navi mumbai, maharashtra" : "sector 30 gothavali"}
        
        df = ensure_string_col(df, "location")
        # Fill "location" based on "address" matching mapping dictionary
        df.loc[df["location"].isna(), "location"] = df["address"].map(mapping)

        # Mapping dictionary
        replace_dict = {
            "bhayander": "bhayandar",
            "century Bazar": "century bazaar",
            "dombivali": "dombivli",
            "kasarwadvali": "kasarvadavali",
            "koparkhairane": "kopar khairane",
            "koparkhairne": "kopar khairane",
            "koperkhairne": "kopar khairane",
            "koperkhairane": "kopar khairane",
            "koperkharine": "kopar khairane",
            "mulund goregaon link road": "goregaon mulund link road",
            "naigoan": "naigaon",
            "nalasopara": "nala sopara",
            "nallasopara": "nala sopara",
            "palaspe phata": "palaspa",
            "palava": "palava city",
            "shil phata": "shilphata",
            "vartaknagar": "vartak nagar",
            "vileparle": "vile parle",
            "4 east": "ulhasnagar",
            "402borivali west": "borivali west",
            "adai": "adai navi mumbai"  # Careful with this if "adai" alone is meant to be corrected
        }
        
        # Function to apply mapping
        def correct_location(location):
            for wrong, correct in replace_dict.items():
                if pd.notnull(location) and wrong.lower() in location.lower():
                    # Replace wrong word with correct one (case-insensitive)
                    location = location.lower().replace(wrong.lower(), correct.lower())
            return location
        
        # Apply correction function
        df['location'] = df['location'].apply(correct_location)

        location_mapping = {
            "mulund airoli road": "navi mumbai",
            "taloja bypass road": "navi mumbai",
            "panvel": "navi mumbai",
            "sector 9 airoli": "navi mumbai",
            "taloja": "navi mumbai",
            "old panvel": "navi mumbai",
            "naigaon east vasai link road": "palghar",
            "naigaon palghar": "palghar",
            "vasai": "palghar",
            "vasai east": "palghar",
            "vasai road west": "palghar",
            "vasai west": "palghar",
            "virar": "palghar",
            "virar east": "palghar",
            "virar west": "palghar",
            "thane west": "thane",
            "kolshet": "thane",
            "majiwada": "thane",
            "kandivali east": "mumbai",
            "thane belapur road": "thane",
            "mahim": "mumbai",
            "bhayandar": "thane",
            "bhayandar east": "thane",
            "bhayandar west": "thane",
            "bhayandarpada": "thane",
            "mira bhayandar": "thane",
            "mira bhayandar road": "thane",
            "mira road": "thane",
            "mira road area": "thane",
            "mira road east": "thane",
            "nala sopara": "palghar",
            "naigaon east": "palghar",
            "naigaon west": "palghar",
            "nala sopara east": "palghar",
            "nala sopara west": "palghar",
            "kharghar": "navi mumbai"
        }

        
        # Update city based on location presence
        for key, value in location_mapping.items():
            df.loc[df["location"].str.contains(key, case=False, na=False), "city"] = value

        df['location'] = df['location'].replace('', np.nan)

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        # property_type : New property, Resale, Rent, Other
        #combine_first Update null elements with value in the same location in other.
        df['property_type'] = df["many_transaction type"].combine_first(df["leftmany_transaction type"])

        # df = df[~df['property_type'].isin(['other', 'rent'])]
        if "property_type" in df.columns and len(df) > 0:
            bad_types = df["property_type"].astype(str).str.strip().str.lower().isin(["rent", "other"])
            if bad_types.any():
                logger.warning("property_type is rent/other. For API prediction setting to NaN instead of dropping.")
                df.loc[bad_types, "property_type"] = np.nan

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        # ownership
        df = df.rename(columns={'md_type of ownership': 'ownership'})

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        #status
        #combine_first Update null elements with value in the same location in other.
        df['status'] = df['many_status'].combine_first(df['leftmany_status'])

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        #construction
        #combine_first Update null elements with value in the same location in other.
        df['construction_1'] = df['many_age of construction'].combine_first(df['leftmany_age of construction'])

        df = df.rename(columns={'md_age of construction': 'construction'})

        df['construction'] = df.apply(
            lambda row: 'under construction' if row['status'] == 'under construction' else row['construction'], axis=1
        )

        df['status'] = df.apply(
            lambda row: 'under construction' if row['construction'] == 'under construction' else row['status'], axis=1
        )

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        #extra rooms 
        #combine_first Update null elements with value in the same location in other.
        df['balcony1'] = df['leftmany_additional rooms'].combine_first(df['many_additional rooms'])
        
        df['extra_room'] = df['balcony1'].str.split(' ').str[1].str.strip()
        
        result = df['extra_room'].apply(
            lambda x: any(str(x) in str(room) for room in df['md_additional rooms']) if pd.notnull(x) else False
        )
        
        #sort value alphabetically 
        df['extra_rooms'] = df['md_additional rooms'].apply(
            lambda x: ', '.join(sorted(x.split(', '))) if pd.notna(x) else None
        )
        
        #remove none of these eg:from these 'none of these, store' and keep only store 
        #but if we have only 'none of these' then we keep that as it is 
        #also remove room word from all values 
        
        df['extra_rooms'] = df['md_additional rooms'].apply(
            lambda x: x if pd.isna(x) or str(x).strip() == 'none of these' else ', '.join(
                [item.replace(' room', '') for item in str(x).split(', ') if item != 'none of these']
            )
        )

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        #Facing
        #combine_first Update null elements with value in the same location in other.
        df['facing'] = df['leftmany_facing'].combine_first(df['many_facing'])

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        #towers and available_units
        df = df.rename(columns={'aboutpjt_total units': 'available_units', 
                            'aboutpjt_total towers': 'towers'})

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        #seller
        df['seller'] = df['potentialaction'].str.split(',').str[1].str.split(':').str[2].str.strip(" '\"")

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        # #price_category
        # # Define price bins and labels
        # price_bins = [0, 0.99, 1.99, 2.99, 3.99, 4.99, 5.99, 6.99, 7.99, 8.99, 9.99, 14.99, 20.00, float('inf')]
        # price_labels = [
        #     "0.00 - 0.99", "1.00 - 1.99", "2.00 - 2.99", "3.00 - 3.99", "4.00 - 4.99", 
        #     "5.00 - 5.99", "6.00 - 6.99", "7.00 - 7.99", "8.00 - 8.99", "9.00 - 9.99", 
        #     "10.00 - 14.99", "15.00 - 20.00", "20.00 and above"
        # ]
        
        # # Use pd.cut to categorize the prices
        # df['price_category'] = pd.cut(df['price'], bins=price_bins, labels=price_labels, right=True)

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        #overlooking
        df['overlooking'] = df['md_overlooking'].apply(
            lambda x: ', '.join(sorted(map(str.strip, x.split(',')))) if pd.notna(x) else np.nan
        )
        
        # Remove the phrase 'not available' from the 'overlooking' column
        #df['overlooking'] = df['overlooking'].str.replace(',? *not available', '', regex=True)
        df['overlooking'] = df['overlooking'].astype("string").fillna("").str.replace(',? *not available', '', regex=True)


        #--------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        #room_type
        df['room_type'] = df['name'].apply(lambda x: 'flat' if 'flat' in x else ('apartment' if 'apartment' in x else 'other'))

        # room_type
        df["room_type"] = df["name"].apply(
            lambda x: "flat" if "flat" in str(x).lower()
            else ("apartment" if "apartment" in str(x).lower() else "other")
        )

        # block apartments (drop for training dataset)
        if "room_type" in df.columns and len(df) > 0:
            bad_rooms = df["room_type"].astype(str).str.strip().str.lower().eq("apartment")
            if bad_rooms.any():
                logger.warning("room_type=apartment detected. For API prediction keeping row and setting room_type='other'.")
                df.loc[bad_rooms, "room_type"] = "other"


        # ------------------------------------------------------------------------------------
        # FORCE NUMERIC COLS TO NaN (fix NoneType math issues)
        # ------------------------------------------------------------------------------------
        numeric_cols = [
            "bed", "bath", "balcony", "parking", "lift",
            "area", "available_units", "project_in_acres",
            "flat_on_floor", "total_floor"
        ]

        for c in numeric_cols:
            if c in df.columns:
                df[c] = df[c].replace({None: np.nan})
                df[c] = pd.to_numeric(df[c], errors="coerce")



        #--------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        def add_engineered_features(df):
            df = df.copy()

            def safe_div(a, b):
                a = pd.to_numeric(a, errors="coerce")
                b = pd.to_numeric(b, errors="coerce")
                return a.div(b).replace([np.inf, -np.inf], np.nan)

            # --- Ratio Features ---
            df["bath_bed_ratio"] = safe_div(df["bath"], df["bed"])
            df["bed_area_ratio"] = safe_div(df["bed"], df["area"])
            df["bed_bath_ratio"] = safe_div(df["bed"], df["bath"])
            df["bed_balcony_ratio"] = safe_div(df["bed"], df["balcony"])

            # --- Density Features ---
            df["project_density"] = safe_div(df["available_units"], df["project_in_acres"])
            df["compactness_ratio"] = safe_div(df["area"], df["project_in_acres"])

            # --- Floor Features ---
            df["floor_ratio"] = safe_div(df["flat_on_floor"], df["total_floor"])
            df["remaining_floors"] = df["total_floor"] - df["flat_on_floor"]

            # --- Area per-unit Features ---
            df["area_per_bedroom"] = safe_div(df["area"], df["bed"])
            df["area_per_bathroom"] = safe_div(df["area"], df["bath"])
            df["area_per_balcony"] = safe_div(df["area"], df["balcony"])
            df["area_per_parking"] = safe_div(df["area"], df["parking"])

            # --- Amenity Ratios ---
            df["balcony_to_bed_ratio"] = safe_div(df["balcony"], df["bed"])
            df["parking_to_bed_ratio"] = safe_div(df["parking"], df["bed"])
            df["lift_to_total_floor_ratio"] = safe_div(df["lift"], df["total_floor"])

            return df

        
        
        # --- Apply before train-test split ---
        df = add_engineered_features(df)


        #--------------------------------------------------------------------------------------------------------------------------------------------------------------

        # For each property, calculate its distance (km) from the center of its city using the Haversine formula and store it in a new column distance_to_center_km

        # --- City center coordinates ---
        city_centers = {
            'mumbai': (18.9720, 72.8096),
            'navi mumbai': (19.0330, 73.0200),
            'thane': (19.2140, 72.9775),
            'palghar': (19.6967, 72.7699)
        }
        
        # --- Haversine function ---
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371  # km
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            return R * c
        
        dup_cols = df.columns[df.columns.duplicated()].tolist()
        print("DUPLICATE COLUMNS:", dup_cols)

        print(
            "distance_to_center_km count:",
            (df.columns == "distance_to_center_km").sum()
        )

        
        def add_distance_to_center(df, city_col='city', lat_col='lattitude', lon_col='longitude'):
            df = df.copy()

            # ✅ protect against duplicate columns (API single-row case)
            df = df.loc[:, ~df.columns.duplicated()].copy()

            def row_distance(row):
                city = str(row.get(city_col, "")).strip().lower()
                lat = row.get(lat_col, np.nan)
                lon = row.get(lon_col, np.nan)

                if city not in city_centers:
                    return np.nan
                if pd.isna(lat) or pd.isna(lon):
                    return np.nan

                lat2, lon2 = city_centers[city]
                return haversine(lat, lon, lat2, lon2)

            df["distance_to_center_km"] = df.apply(row_distance, axis=1)
            return df

        
        df = add_distance_to_center(df)

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------

        #nearby_location_km

        # Reusable function
        def combine_columns(df, cols, new_col):
            df[new_col] = df[cols].apply(lambda row: ', '.join(filter(pd.notna, row)), axis=1)
        
        # Education
        combine_columns(df, [
            'educational institute_1', 'educational institute_2', 
            'educational institute_3', 'educational institute_4', 
            'educational institute_5'
        ], 'education')
        
        # Transport
        combine_columns(df, [
            'transportation hub_1', 'transportation hub_2', 
            'transportation hub_3', 'transportation hub_4', 
            'transportation hub_5'
        ], 'transport')
        
        # Shopping Centre
        combine_columns(df, [
            'shopping centre_1', 'shopping centre_2', 
            'shopping centre_3', 'shopping centre_4', 
            'shopping centre_5'
        ], 'shopping_centre')
        
        # Commercial Hub
        combine_columns(df, [
            'commercial hub_1', 'commercial hub_2', 
            'commercial hub_3', 'commercial hub_4', 
            'commercial hub_5'
        ], 'commercial_hub')
        
        # Hospital
        combine_columns(df, [
            'hospital_1', 'hospital_2', 
            'hospital_3', 'hospital_4', 
            'hospital_5'
        ], 'hospital')
        
        # Tourist
        combine_columns(df, [
            'tourist spot_1', 'tourist spot_2', 
            'tourist spot_3', 'tourist spot_4'
        ], 'tourist')

        # Function to extract mean km from text
        # Initialize global zero counter
        # Function to extract mean km with zero replacement
        def extract_mean_km(text):
            if pd.isna(text):
                return np.nan
            km_values = [float(x) for x in re.findall(r'([\d.]+)\s*km', text)]
            km_values = [0.0001 if km == 0.0 else km for km in km_values]  #means something which is in zero km , like hospital in building, make that 0.0001 km
            return sum(km_values) / len(km_values) if km_values else np.nan
        
        # Function to extract min km with zero replacement
        def extract_min_km(text):
            if pd.isna(text):
                return np.nan
            km_values = [float(x) for x in re.findall(r'([\d.]+)\s*km', text)]
            km_values = [0.0001 if km == 0.0 else km for km in km_values]  
            return min(km_values) if km_values else np.nan
        
        # Apply to column
        df['education_mean_km'] = df['education'].apply(extract_mean_km)
        df['education_min_km'] = df['education'].apply(extract_min_km)
        
        
        # Function to count places within 2 km
        #so one row has so many values and from that how many are within 2km that we count here
        #eg: [1.0,3.0,1.9,4.8] so here it is 2
        def count_within_2km(text):
            if pd.isna(text):
                return np.nan
            km_values = [float(x) for x in re.findall(r'([\d.]+)\s*km', text)]
            return sum(1 for km in km_values if km <= 2.0)
        
        # List of combined location columns
        location_cols = ['education', 'transport', 'shopping_centre', 'commercial_hub', 'hospital', 'tourist']
        
        # Apply all 3 functions: mean, min, within_2km
        for col in location_cols:
            df[col + '_mean_km'] = df[col].apply(extract_mean_km)
            df[col + '_min_km'] = df[col].apply(extract_min_km)
            df[col + '_within_2km'] = df[col].apply(count_within_2km)
        
        # Show only mean and min columns
        mean_cols = [col + '_mean_km' for col in location_cols]
        min_cols = [col + '_min_km' for col in location_cols]
        within_2km_cols = [col + '_within_2km' for col in location_cols]
        
        # Add final 5 summary columns
        df['overall_min_mean_km'] = df[mean_cols].min(axis=1)
        df['overall_avg_mean_km'] = df[mean_cols].mean(axis=1)
        df['overall_min_min_km'] = df[min_cols].min(axis=1)
        df['overall_avg_min_km'] = df[min_cols].mean(axis=1)
        df['total_within_2km'] = df[within_2km_cols].sum(axis=1) #sum of all within_2km location_cols


        #--------------------------------------------------------------------------------------------------------------------------------------------------------------

        #flooring
        df = df.rename(columns={'md_flooring':'flooring'})

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------

        #amenities
        #this code is for creating separate amenities dataframe
        # Select columns that start with 'am_' and include 'id'
        #am_cols = ['id'] + [col for col in df.columns if col.startswith('am_')]
        
        # Create a separate DataFrame with those columns
        #am_df = df[am_cols].copy()
        
        # Drop 'am_' columns from the original DataFrame (keep 'id')
        #df = df.drop(columns=[col for col in df.columns if col.startswith('am_')])

        am_cols = [col for col in df.columns if col.startswith('am_')]

        # Step 2: Combine values row-wise into a comma-separated string (not a list)
        df['amenities'] = df[am_cols].apply(
            lambda row: ', '.join([str(val).strip() for val in row if isinstance(val, str) and val.strip() != ""]),
            axis=1
        )

        #final
        amenities_weightages = {
            "sea facing": 10,
            "private pool": 10,
            "private jaccuzi": 10,
            "sky villa": 10,
            "helipad": 10,
            "wrap around balcony": 7,
            "infinity swimming pool": 10,
            "high ceiling": 9,
            "located in the heart of city": 10,
            "large open space": 10,
            "skyline view": 10,
            "private terrace/garden": 10,
            "private garage": 10,
            "mansion": 10,
            "club house": 9,
            "large clubhouse": 9,
            "modular kitchen": 9,
            "central ac": 9,
            "banquet hall": 6,
            "premium branded fittings": 9,
            "private garden": 9,
            "full glass wall": 9,
            "garden view": 9,
            "theme based architectures": 9,
            "grand entrance lobby": 9,
            "smart home": 9,
            "library and business centre": 9,
            "recreational pool": 9,
            "projector": 8,
            "swimming pool": 8,
            "gymnasium": 8,
            "indoor squash & badminton courts": 8,
            "outdoor tennis courts": 8,
            "cycling & jogging track": 8,
            "kids play pool with water slides": 8,
            "guest lobby in each floor": 8,
            "aesthetically designed landscape garden": 8,
            "health club with steam / jacuzzi": 8,
            "meditation area": 8,
            "pet park": 8,
            "visitor parking": 8,
            "badminton court": 8,
            "kids play area": 7,
            "community hall": 7,
            "power back up": 7,
            "cctv camera": 7,
            "rain water harvesting": 7,
            "internet/wi-fi connectivity": 7,
            "cycling track": 7,
            "art center": 7,
            "library": 7,
            "fire sprinklers": 7,
            "multipurpose hall": 7,
            "event space & amphitheatre": 7,
            "flower gardens": 6,
            "curated garden": 6,
            "multipurpose courts": 7,
            "dth television facility": 5,
            "fire fighting equipment": 6,
            "provision for power backup": 7,
            "sand pit": 6,
            "sewage treatment plant": 6,
            "solar energy": 7,
            "piped gas": 6,
            "kids club": 6,
            "waste disposal": 6,
            "lift": 5,
            "security": 5,
            "maintenance staff": 5,
            "reserved parking": 5,
            "ro water system": 5,
            "wheelchair accessibility": 5,
            "shopping center": 5,
            "laundry service": 5,
            "bank & atm": 5,
            "community entrance gate": 5,
            "canopy walk": 4,
            "entry exit gate": 4,
            "early learning centre": 4,
            "earth quake resistant": 7,
            "waste water recycling": 6,
            "whiteboard": 3,
            "printer": 3,
            "tea/coffee": 3,
            "house help accommodation": 7,
            "study room": 5,
            "ground water recharging": 5,
            "unknown": 0,
            "3 tier security system": 8,
            "ac in each room": 9,
            "activity deck4": 7,
            "aerobics room": 7,
            "air conditioned": 9,
            "all wooden flooring": 8,
            "arts & craft studio": 6,
            "bar/lounge": 7,
            "barbeque pit": 6,
            "barbeque space": 6,
            "cafeteria/food court": 7,
            "coffee lounge & restaurants": 7,
            "concierge services": 9,
            "conference room": 8,
            "cricket net practice": 6,
            "dance studio": 7,
            "downtown": 10,
            "fingerprint access": 8,
            "fireplace": 6,
            "golf course": 10,
            "hilltop": 10,
            "horticulture": 6,
            "indoor games room": 7,
            "island kitchen layout": 8,
            "jogging and strolling track": 7,
            "kids splash pool": 7,
            "lawn with pathway": 6,
            "guest accommodation":8,
            "marble flooring": 9,
            "mini cinema theatre": 9,
            "half basketball court":7,
            "park": 8,
            "pool with temperature control": 10,
            "intercom facility":6,
            "rentable community space": 6,
            "retail boulevard (retail shops)": 8,
            "service/goods lift": 6,
            "skydeck": 9,
            "vaastu compliant": 7,
            "volleyball court": 6,
            "water front": 10,
            "water storage": 5,
            "water treatment plant": 7,
            "wine cellar": 8
        }

        # --- Function to assign amenities score BEFORE train-test split ---
        def assign_amenities_score(df, column="amenities", weightages=None, output_column="assigned_amenities_score"):
            if weightages is None:
                weightages = {}

            df = df.copy()

            def calculate_score(amenities_str):
                if not isinstance(amenities_str, str):
                    return np.nan  
                amenities_types = [f.strip().lower() for f in amenities_str.split(",")]
                total_weight = sum(weightages.get(f, 0) for f in amenities_types)
                return round(total_weight, 2) if total_weight > 0 else np.nan  

            # Apply calculation
            df[output_column] = df[column].apply(calculate_score)

            return df


        df = assign_amenities_score(df, column="amenities", weightages=amenities_weightages)

        # Count number of amenities by splitting the string on ','
        df['amenities_count'] = df['amenities'].apply(
            lambda x: len(str(x).split(',')) if isinstance(x, str) else 0
        )

        df.drop(columns="amenities", inplace=True)

        # Step 3: Drop original 'am_' columns
        df.drop(columns=am_cols, inplace=True)

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------

        #Custom data corrections and row-level cleaning

        # List of IDs to remove
        # ids_to_remove = [
        #     'cardid70421965',  
        #     'cardid71698587',
        #     'cardid41440251',
        #     'cardid70017925',  
        #     'cardid73050463',
        #     'cardid49131617',
        #     'cardid72273473',
        #     'cardid66762427',
        #     'cardid70615879',
        #     'cardid72819785',
        #     'cardid71143703',
        #     'cardid72821117',
        #     'cardid72884955',
        #     'cardid72803713',
        #     'cardid73037481',
        #     'cardid69783235',
        #     'cardid73144165',
        #     'cardid33966233',
        #     'cardid73046249',
        #     'cardid69702399',
        #     'cardid54078457',
        #     'cardid71697753'
        # ]
        
        # # Drop rows with matching IDs
        # df = df[~df['id'].isin(ids_to_remove)].reset_index(drop=True)
        
        # #after observation 
        # ids_to_update = ['cardid73059851', 'cardid72926775', 'cardid58806131']
        
        # df.loc[df['id'].isin(ids_to_update), 'city'] = 'palghar'
        # df.loc[df['id'].isin(ids_to_update), 'location'] = 'palghar'
        
        #assign 'thane' to the city for all rows where location is 'ulhasnagar'
        df.loc[df['location'] == 'ulhasnagar', 'city'] = 'thane'
        df.loc[df['location'] == 'agashi', 'city'] = 'palghar'
        df.loc[df['location'] == 'bhabola', 'city'] = 'palghar'
        df.loc[df['location'] == 'bolinj', 'city'] = 'palghar'
        df.loc[df['location'] == 'diwanman', 'city'] = 'palghar'
        df.loc[df['location'] == 'dongarpada road', 'city'] = 'palghar'
        df.loc[df['location'] == 'evershine city', 'city'] = 'palghar'
        df.loc[df['location'] == 'juchandra', 'city'] = 'thane'
        df.loc[df['location'] == 'morya nagar', 'city'] = 'palghar'
        df.loc[df['location'] == 'oswal nagari', 'city'] = 'thane'
        df.loc[df['location'] == 'padmavati nagar bolinj', 'city'] = 'palghar'
        df.loc[df['location'] == 'unique garden', 'city'] = 'thane'
        df.loc[df['location'] == 'rustomjee global city', 'city'] = 'palghar'
        df.loc[df['location'] == 'wagholi', 'city'] = 'thane'
        df.loc[df['location'] == 'vinay nagar', 'city'] = 'thane'
        df.loc[df['location'] == 'yashwanth nagar', 'city'] = 'palghar'
        df.loc[df['location'] == 'dongarpada', 'city'] = 'palghar'
        df.loc[df['location'] == 'beverly park', 'city'] = 'thane'
        df.loc[df['location'] == 'padrikhan wadi', 'city'] = 'palghar'
        df.loc[df['location'] == 'medetiya nagar', 'city'] = 'thane'
        df.loc[df['location'] == 'hatkesh udhog nagar', 'city'] = 'thane'
        df.loc[df['location'] == 'kashigaon', 'city'] = 'thane'
        df.loc[df['location'] == 'kashimira', 'city'] = 'thane'
        df.loc[df['location'] == 'sector 8 shanti nagar', 'city'] = 'thane'
        df.loc[df['location'] == 'shanti vihar', 'city'] = 'thane'
        df.loc[df['location'] == 'chulne', 'city'] = 'palghar'
        df.loc[df['location'] == 'mahajan wadi', 'city'] = 'thane'
        df.loc[df['location'] == 'sector 9 shanti nagar', 'city'] = 'thane'
        df.loc[df['location'] == 'chandan shanti', 'city'] = 'thane'
        df.loc[df['location'] == 'pleasant park', 'city'] = 'thane'
        df.loc[df['location'] == 'sector 3 shanti nagar', 'city'] = 'thane'
        df.loc[df['location'] == 'poonam sagar complex', 'city'] = 'thane'
        df.loc[df['location'] == 'stella', 'city'] = 'palghar'
        df.loc[df['location'] == 'ramdev park', 'city'] = 'thane'
        df.loc[df['location'] == 'golden nest phase 1', 'city'] = 'thane'
        df.loc[df['location'] == 'madhuban township', 'city'] = 'palghar'
        
        # Update city to 'thane' where address starts with 'mira' (case insensitive)
        df.loc[df['address'].str.lower().str.startswith('mira', na=False), 'city'] = 'thane'
        
        #make thane in city for all this ids
        # ids_to_update = [
        #     "cardid72703033",
        #     "cardid69846363",
        #     "cardid73257889",
        #     "cardid56191653",
        #     "cardid72796607",
        #     "cardid73026297",
        #     "cardid72794677",
        #     "cardid66964031",
        #     "cardid58541153",
        #     "cardid73076791",
        #     "cardid72794677",
        #     "cardid53323155",
        #     "cardid69812109",
        #     "cardid69665873",
        #     "cardid70673145",
        #     "cardid70120173",
        #     "cardid60101171",
        #     "cardid73012265",
        #     "cardid73028981",
        #     "cardid71481487",
        #     "cardid67617413",
        #     "cardid53977959"
            
        # ]
        
        # df.loc[df['id'].isin(ids_to_update), 'city'] = 'thane'
        
        # #make palghar in city for all this ids
        # ids_to_update = [
        #     "cardid72923721",
        #     "cardid61647785",
        #     "cardid70476757",
        #     "cardid72179863",
        #     "cardid72846389",
        #     "cardid73127129",
        #     "cardid61883771",
        #     "cardid72998493",
        #     "cardid73114181",
        #     "cardid71923233",
        #     "cardid63887703",
        #     "cardid72831163"
        # ]
        
        # df.loc[df['id'].isin(ids_to_update), 'city'] = 'palghar'
        
        # #make navi mumbai in city for all this ids
        # ids_to_update = [
        #     "cardid62724753"
        # ]
        
        # df.loc[df['id'].isin(ids_to_update), 'city'] = 'navi mumbai'


        #--------------------------------------------------------------------------------------------------------------------------------------------------------------



        # Create a mask for rows where 'lattitude' starts with 16, 12, or 9
        mask = (
            df['lattitude'].astype(str).str.startswith('16') |
            df['lattitude'].astype(str).str.startswith('12') |
            df['lattitude'].astype(str).str.startswith('9')
        )
        
        # Replace only 'lattitude' and 'longitude' with NaN for those rows
        df.loc[mask, ['lattitude', 'longitude']] = np.nan

















        
        





        #--------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        #drop columns
        df.drop(['numberofrooms','bb_beds','leftbb_beds','bb_bed','leftbb_bed','multi_bed_filled','bb_baths','leftbb_baths','bb_bath','leftbb_bath','multi_bath_filled',
                'bb_covered-parking','leftbb_covered-parking','many_car parking','leftmany_car parking','md_price breakup','property_loc','many_transaction type',
                'leftmany_transaction type','many_type of ownership','leftmany_type of ownership','many_status', 'leftmany_status','many_lifts','md_lift','leftmany_lifts',
                'many_lift','leftmany_lift','multi_lift_filled','aboutpjt_total floors','floor_work_1','many_floor','leftmany_floor','md_floors allowed for construction',
                'construction_1','many_age of construction','leftmany_age of construction','bb_balcony', 'leftbb_balcony', 'bb_balconies','leftbb_balconies',
                'leftmany_additional rooms', 'balcony1', 'many_additional rooms','extra_room', 'md_additional rooms','leftmany_facing','many_facing','ap_unit','ap_tower',
                'ap_tower & unit','geo','potentialaction','md_overlooking','room_type','aboutpjt_project size','educational institute_1','educational institute_2',
                'educational institute_3','educational institute_4','educational institute_5','transportation hub_1','transportation hub_2','transportation hub_3',
                'transportation hub_4','transportation hub_5','shopping centre_1','shopping centre_2','shopping centre_3','shopping centre_4','shopping centre_5',
                'commercial hub_1','commercial hub_2','commercial hub_3','commercial hub_4','commercial hub_5','hospital_1','hospital_2','hospital_3','hospital_4','hospital_5',
                'tourist spot_1','tourist spot_2','tourist spot_3','tourist spot_4','education', 'transport', 'shopping_centre', 'commercial_hub', 'hospital', 'tourist',
                'url','image','image_urls','name','wholeaddress','address','powercut_hours','id','costpersqft','emi', 'authority_approval_clean','rera_id_grouped',
                'nearby_landmarks'
                ],axis=1,inplace=True,errors="ignore") # 'locality_rank', 'locality_url_rating', price_category
        # print(df.shape)
        
        
        debug_df_snapshot(df, "AFTER BASIC CLEANING (END)")
        df = df.copy()
        return df

    except Exception as e:
        debug_df_snapshot(df, f"FAILED INSIDE basic_cleaning | ERROR: {repr(e)}")
        raise

# =====================================================
# MISSINGNESS HANDLING
# =====================================================
def property_missingness_identification(data: pd.DataFrame) -> pd.DataFrame:
    logger.info("Handling missingness")
    df = data.copy()
    debug_df_snapshot(df, "BEFORE MISSINGNESS")

    try:
        cols_to_drop = ['tourist_mean_km', 'tourist_min_km', 'hospital_mean_km', 'hospital_min_km']
        df = df.drop(columns=cols_to_drop, errors="ignore")

        debug_df_snapshot(df, "AFTER MISSINGNESS")
        return df

    except Exception as e:
        debug_df_snapshot(df, f"FAILED INSIDE missingness | ERROR: {repr(e)}")
        raise



# =====================================================
# FULL CLEANING PIPELINE
# =====================================================
def perform_property_data_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    cleaned_data = (
        data
        .pipe(basic_cleaning)
        .pipe(property_missingness_identification)
    )

    return cleaned_data


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    # data path for data
    DATA_PATH = "data/raw/f_original magicbricks cleaned 12022 data.csv"

    # read data
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print("Property data loaded successfully")

    # clean data
    cleaned_df = perform_property_data_cleaning(df)

    # optional save (if you want)
    cleaned_df.to_csv("data/cleaned/property_cleaned_for_utils.csv", index=False)
    print("Cleaned data saved successfully")

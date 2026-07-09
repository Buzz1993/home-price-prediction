# ===============================
# data_cleaning_rec.py
# ===============================

import numpy as np
import pandas as pd
import re
from pathlib import Path
import ast
import logging

# =====================================================
# LOGGER SETUP (same style as first script)
# =====================================================
logger = logging.getLogger("property_data_cleaning_rec")
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
def basic_cleaning(data: pd.DataFrame, webscraped_df: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    
    #convert column names into lowercase
    df.columns = df.columns.str.lower()

    #drop unwanted columns
    df.drop(['@type','locality_url','md_booking amount','md_loan offered','ap_price','ap_price per sqft','ap_configuration',
              'ap_ratings','ap_reviews_by','headings_with_ratings','aboutpjt_bhk','2 bhk flat',
              '3 bhk flat','1 bhk flat','studio apartment','4 bhk flat','5 bhk flat', 
              'multistorey apartment', '3 bhk villa', '4 bhk villa', 'residential plot', '2 bhk builder', '3 bhk builder','4 bhk penthouse','5 bhk penthouse', 
              '6 bhk flat','rent','commercial office space','3 bhk penthouse',
               
            ],axis=1,inplace=True) 
    #'md_water availability',  'locality_url_review', 'liv_environment','liv_commuting', 'liv_places of interest','md_status of electricity',
    #'md_landmarks', 'md_authority approval', 'md_rera id', 'aboutpjt_launch date' 
    #'ap_pjt_url','image_urls','bhk_type'

    # --- Rename columns for clarity ---
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
    
    #drop duplicate rows
    df.drop_duplicates(inplace=True)
    
    #delete column which have all nan values
    df.dropna(axis=1, how='all', inplace=True)

    df[df.select_dtypes('object').columns] = df.select_dtypes('object').apply(
        lambda col: col.map(lambda x: x.lower() if isinstance(x, str) else x)
    )

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------
    import pandas as pd
    #function 1
    def check_more_than_one_value_in_column(df, cols, new_col_name, col_name):
        # Step 1: Create a boolean column to check if more than one value (non-NaN) is filled in the specified columns
        df[new_col_name] = df[cols].notna().sum(axis=1) > 1
    
        # Step 2: Combine all values from the specified columns into a list for each row
        def combine_values():
            df[col_name] = [list(values) for values in zip(*[df[col] for col in cols])]
    
        combine_values()
    
        found_distinct = False
    
        # Step 3: If any row has more than one value
        if df[new_col_name].any():
            for index, row in df[col_name].items():
                non_nan_vals = [val for val in row if pd.notna(val)]
                # Step 4: If more than one unique value found in the row, print that row
                if len(set(non_nan_vals)) > 1: #take rows which have more than one unique value 
                    print(f"Row {index} has multiple distinct non-NaN values: {row}") 
                    found_distinct = True  #if get more than 1 distinct value then below thing wont run
    
        # Step 5: If no row has more than one unique value, safely pick the first non-NaN value
        if not found_distinct:
            df[col_name] = df[col_name].apply(lambda row: next((val for val in row if pd.notna(val)), None))
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    #function2
    def combine_first_valid(df, source_cols, new_col_name):
        """
        Create a new column with the first valid (non-null, non-'nan', non-empty) value 
        across the specified source columns.
        """
        # Combine columns into new_col_name using first valid value per row
        df[new_col_name] = df[source_cols].apply(
            lambda row: next(
                (str(x) for x in row if pd.notna(x) and str(x).strip().lower() != 'nan' and str(x).strip() != ''),
                np.nan
            ),
            axis=1
        )
        
        # Normalize the new column's casing and whitespace
        df[new_col_name] = df[new_col_name].str.strip().str.lower()
    
        # Drop the source columns
        df.drop(columns=source_cols, inplace=True)
    
        return df

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    #emi
    converted_emi = []

    for emi_n in df['emi']:
        if isinstance(emi_n, str):  # Check if emi_n is a string
            if 'k' in emi_n:
                # Convert from thousands to lakhs
                converted_emi.append(float(emi_n.replace('k', '')) / 100)
            elif 'l' in emi_n:
                # No change needed for lakhs
                converted_emi.append(float(emi_n.replace('l', '')))
            else:
                # Convert rupees to lakhs
                converted_emi.append(float(emi_n) / 100000)
        else:
            # If it's already a float, convert rupees to lakhs
            converted_emi.append(emi_n / 100000)
    
    # Add the converted values to the DataFrame
    df['converted_emi'] = converted_emi
    df = df.drop(['emi'], axis=1)
    df.rename(columns={'converted_emi': 'emi'}, inplace=True)
        
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------
    # numerical column
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------

    #price
    def convert_price_to_cr(val):
        if isinstance(val, str):  # check if value is a string
            parts = val.replace('₹', '').split()  # remove ₹ and split into amount and unit
            if len(parts) == 2:  # ensure both amount and unit exist
                amount, unit = parts
                amount = float(amount)  
                return amount / 100 if unit.lower() == 'lac' else amount  # convert lac to Cr
        return None  # return None if invalid
    
    # Apply conversion
    df['price'] = df['price'].apply(convert_price_to_cr)

    #datatype to float
    df['price'] = df['price'].astype('float64')
    
    # Drop rows with missing prices
    df = df.dropna(subset=['price'])

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    #bed
    check_more_than_one_value_in_column(df, ['numberofrooms', 'bb_beds', 'leftbb_beds', 'bb_bed','leftbb_bed'], 'multi_bed_filled','bed')  

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    #bath
    check_more_than_one_value_in_column(df, ['bb_baths', 'leftbb_baths', 'bb_bath', 'leftbb_bath'], 'multi_bath_filled','bath')  

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    import re
    #parking
    # Convert 'leftmany_car parking' into sum of all digits
    df['leftmany_car parking'] = df['leftmany_car parking'].apply(
        lambda x: sum(map(int, re.findall(r'\d+', str(x)))) if pd.notna(x) else np.nan
    )
    
    # Convert 'many_car parking' into sum of all digits
    df['many_car parking'] = df['many_car parking'].apply(
        lambda x: sum(map(int, re.findall(r'\d+', str(x)))) if pd.notna(x) else np.nan
    )
    
    # Take max across the four parking columns
    df['parking'] = df[
        ['bb_covered-parking', 'leftbb_covered-parking', 'many_car parking', 'leftmany_car parking']
    ].apply(
        lambda row: np.nanmax(row.values) if pd.notna(row).any() else np.nan,
        axis=1
    )

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    #area and costpersqft     

    #combine_first Update null elements with value in the same location in other.
    df['area_work'] = df["many_carpet area"].combine_first(df["leftmany_carpet area"])


    # Extract carpet area features
    df["carpet_area"] = df["area_work"].apply(lambda x: float(re.match(r'([\d,\.]+)', x).group(1).replace(',', '')) if pd.notna(x) and re.match(r'^[\d,\.]+', x) else None) 
    df['costpersqft'] = df['area_work'].str.extract(r'₹([\d,\.]+)')[0].str.replace(',', '').astype(float)
    df['area_unit'] = df['area_work'].str.extract(r'/([^/]+)$')
    
    # Remove unwanted area units
    df = df[~df['area_unit'].isin(['sqm', 'kanal'])]
    
    # Combine super built-up area
    df['super_build_area_work'] = df["leftmany_super built-up area"].combine_first(df["many_super built-up area"])
    
    
    # Extract super built-up features
    df['initial_unit'] = df['super_build_area_work'].apply(lambda x: ''.join([char for char in str(x)[re.match(r'\d+', str(x)).end():] if char.isalpha()])[:4] if isinstance(x, str) else None)
    df = df[~df['initial_unit'].isin(['sqms', 'sqyr'])]
    df["super_build_up_area"] = df["super_build_area_work"].apply(lambda x: float(re.match(r'([\d,\.]+)', x).group(1).replace(',', '')) if pd.notna(x) and re.match(r'^[\d,\.]+', x) else None)
    df['super_build_up_costpersqft'] = df['super_build_area_work'].str.extract(r'₹([\d,\.]+)')[0].str.replace(',', '').astype(float)
    df['super_built_up_area_unit'] = df['super_build_area_work'].str.extract(r'/([^/]+)$')
    
    # Final feature selection with combine_first
    df['f_area'] = df["carpet_area"].combine_first(df["super_build_up_area"])
    df['f_costpersqft'] = df["costpersqft"].combine_first(df["super_build_up_costpersqft"])
    df['f_area_unit'] = df["super_built_up_area_unit"].combine_first(df["area_unit"])
    df['f_area'] = df['f_area'].astype('float')
    df['f_costpersqft'] = df['f_costpersqft'].astype('float')
    
    
    # Fill missing values from 'area' column
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
    
    df['dupli_price'] = df.apply(
        lambda row: row['price'] * (10**7) if pd.isna(row['f_area']) and pd.notna(row['area'])
        else None,
        axis=1
    )
    
    df['dupli_costpersqft'] = np.round(df['dupli_price'].astype('float') / df['dupli_f_area'].astype('float'), 2)
    
    # Update final columns if missing
    def update_values(df, update_cols, using_cols):
        for update_col, using_col in zip(update_cols, using_cols):
            df[update_col] = np.where(
                pd.isna(df['f_area']) & pd.notna(df['area']),
                df[using_col],
                df[update_col]
            )
        return df
    
    # Define columns to update and corresponding columns to use
    columns_to_update = ['f_costpersqft', 'f_area_unit', 'f_area']
    using_columns = ['dupli_costpersqft', 'dupli_f_area_unit', 'dupli_f_area']
    
    # Update the DataFrame
    df = update_values(df, columns_to_update, using_columns)
    
    # Drop intermediate columns
    cols_to_drop = [
        'many_carpet area', 'leftmany_carpet area', 'leftmany_super built-up area', 'many_super built-up area', 'area',
        'area_work', 'carpet_area', 'costpersqft', 'area_unit', 'initial_unit',
        'super_build_area_work', 'super_build_up_area', 'super_build_up_costpersqft', 'super_built_up_area_unit',
        'dupli_f_area', 'dupli_f_area_unit', 'dupli_price', 'dupli_costpersqft', 'f_area_unit'
    ]
    df.drop(columns=cols_to_drop, inplace=True)

    df = df.rename(columns={'f_area':'area','f_costpersqft':'costpersqft'})

    df['area'] = df['area'].astype('float64')

    #After analyzing, found some errors in the area column; hence, dropped the rows with the below id's
    df = df[~df['id'].isin([
        'cardid13695470', 'cardid72545677', 'cardid71119645',
        'cardid46503375', 'cardid48667071', 'cardid72200975',
        'cardid70294971', 'cardid70608749', 'cardid72754063',
        'cardid72078141', 'cardid71460761', 'cardid45089373',
        'cardid71419541', 'cardid72848693', 'cardid73238137',
        'cardid70201319','cardid72749881','cardid60135579'
    ])]

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    #total_floor and flat_on_floor
    #combine_first Update null elements with value in the same location in other.
    df['floor_work_1'] = df['many_floor'].combine_first(df['leftmany_floor'])

    df['floor_work_1'] = df['floor_work_1'].astype('str') 

    df['flat_on_floor'] = df['floor_work_1'].apply(
        lambda x: x.split('(')[0].strip() if '(' in str(x) else None
    )

    df['total_floor'] = df['floor_work_1'].apply(
        lambda x: x.split('(')[1].strip() if '(' in str(x) else None
    )

    df['total_floor'] = df['total_floor'].str.extract(r'(\d+)').astype(float)
    
    df['flat_on_floor'] = df['flat_on_floor'].replace({'lower basement': -1, 'upper basement': -2,'ground':0})

    df['total_floor'] = np.where(
        pd.isna(df['total_floor']) & pd.notna(df['md_floors allowed for construction']),
        df['md_floors allowed for construction'],
        df['total_floor']
    )

    df['flat_on_floor'] = df['flat_on_floor'].astype('float64')

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    #lift
    check_more_than_one_value_in_column(df, ['many_lifts', 'md_lift', 'leftmany_lifts', 'many_lift','leftmany_lift'], 'multi_lift_filled','lift') 

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    #balcony
    #combine_first Update null elements with value in the same location in other.
    df['balcony'] = (
        df['bb_balcony']
        .combine_first(df['leftbb_balcony'])
        .combine_first(df['bb_balconies'])
        .combine_first(df['leftbb_balconies'])
    )
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    #longitude and lattitude
    df['latitude'] = df['geo'].str.split(',').str[1].str.split(':').str[1].str.strip(" '\"").astype('float')
    df['longitude'] = df['geo'].str.split(',').str[2].str.split(':').str[1].str.strip(" '\"}").astype('float')
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    #project_in_acres
    # Conversion function for different units to acres
    def convert_to_acres(value):
        if isinstance(value, str):  # Check if the value is a string
            if 'acre' in value:
                acres = float(value.replace('acre', '').strip())
                return round(acres, 4)  
            elif 'sq-m' in value:
                sqm = float(value.replace('sq-m', '').strip())
                return round(sqm * 0.000247105, 4)  
            elif 'sq-ft' in value:
                sqft = float(value.replace('sq-ft', '').strip())
                return round(sqft * 0.0000229568, 4)  
            elif 'hectare' in value:
                hectares = float(value.replace('hectare', '').strip())
                return round(hectares * 2.47105, 4)  
            elif 'sq-yrd' in value:
                sq_yrd = float(value.replace('sq-yrd', '').strip())
                return round(sq_yrd * 0.000836127, 4)  
        elif isinstance(value, (int, float)):  # If value is numeric
            return round(value * 0.0000229568, 4)  
        return np.nan

    # Apply the conversion to the column
    df['project_in_acres'] = df['aboutpjt_project size'].apply(lambda x: convert_to_acres(x))

    # drop more than 1000 acres values  
    df = df[(df["project_in_acres"] <= 1000) | (df["project_in_acres"].isna())].copy() 
    
    # drop less than 0.005 acres values
    df = df[(df["project_in_acres"] > 0.005) | (df["project_in_acres"].isna())].copy()

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------

    #water_availability_hours
    df['water_availability_hours'] = df['water_availability_hours'].str.split(' ').str[0]
    df['water_availability_hours'] = pd.to_numeric(df['water_availability_hours'], errors='coerce')
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    #environment_rating
    df['environment_rating'] = df['environment_rating'].str.split('/').str[0]
    df['environment_rating'] = pd.to_numeric(df['environment_rating'], errors='coerce')

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    #commuting_rating
    df['commuting_rating'] = df['commuting_rating'].str.split('/').str[0]
    df['commuting_rating'] = pd.to_numeric(df['commuting_rating'], errors='coerce')

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------

    #places_of_interest_rating
    df['places_of_interest_rating'] = df['places_of_interest_rating'].str.split('/').str[0]
    df['places_of_interest_rating'] = pd.to_numeric(df['places_of_interest_rating'], errors='coerce')

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------

    #electricity_availability_hours
    mapping = {
        "no/rare powercut": 0,         
        "less than 2 hour powercut": 2,
        "over 6 hours powercut": 6     
    }
    
    df['powercut_hours'] = df['electricity_availability_hours'].map(mapping)

    df.drop(columns='electricity_availability_hours', inplace=True)

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------

    #authority_approval
    threshold = 20  # keep only categories with at least 20 samples
    value_counts = df['authority_approval'].value_counts()
    rare_categories = value_counts[value_counts < threshold].index
    
    df['authority_approval_clean'] = df['authority_approval'].replace(rare_categories, "Other")
    df.drop(columns='authority_approval', inplace=True)



    #--------------------------------------------------------------------------------------------------------------------------------------------------------------

    #rera_id
    import pandas as pd
    import re
    # List of manually valid IDs
    manual_valid_ids = ["rera p51800049875", 'p51700034608, p51700046541','p52000053608.','p51700007024.','p52000045234.','p51800047539.','p51900006367, p51900003268','p52000006391,p52000017271','p51700018070.'
                       ,'p51700032262 ,p51700032205','p51700053017.','p51800052417 and p51800048862','p51700008172.']
    
    # Function to classify IDs
    def classify_rera(x):
        if pd.isna(x):
            return x
        x_lower = str(x).lower()
        
        if x_lower in [v for v in manual_valid_ids]:  # check manual valid IDs
            return "valid"
        if re.fullmatch(r"p\d{11}", x_lower):
            return "valid"
        if re.fullmatch(r"p\d{10}", x_lower):
            return "valid"    
        if re.fullmatch(r"a\d{11}", x_lower):
            return "valid"
        if re.fullmatch(r"\d{11}", x_lower):
            return "valid"
        if re.fullmatch(r"p\d{12}", x_lower):
            return "invalid"
    
        
        return np.nan
    
    # Apply classification
    df["rera_id_grouped"] = df["rera_id"].apply(classify_rera)
    
    # Value counts
    #print(df["rera_id_grouped"].value_counts(dropna=False))
    
    df.drop(columns='rera_id', inplace=True)

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------

    #project_launch_date
    # Convert to datetime
    df['project_launch_date'] = pd.to_datetime(df['project_launch_date'], format='%b-%y')
    
    # Extraction date
    extract_date = pd.to_datetime('2024-12-01')
    
    # Calculate age directly in months
    df['project_age_months'] = (extract_date.year - df['project_launch_date'].dt.year) * 12 + \
                               (extract_date.month - df['project_launch_date'].dt.month)

    df.drop(columns='project_launch_date', inplace=True)
    
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

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    #city
    df = df.rename(columns={'address':'wholeaddress'})

    df['addressregion'] = df['wholeaddress'].apply(
        lambda x: ast.literal_eval(x).get('addressregion') if isinstance(x, str) else x.get('addressregion')
    )

    
    #rename
    df = df.rename(columns={'md_address':'address'})

    df = df.rename(columns = {'addressregion':'city'})

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    #location
    # Convert string representation of dictionaries to actual dictionaries
    df["wholeaddress"] = df["wholeaddress"].apply(ast.literal_eval)
    
    # Extract 'addresslocality' into a new column
    df["location"] = df["wholeaddress"].apply(lambda x: x.get("addresslocality", ""))
    
    #make rd as road in address column
    df['address'] = df['address'].astype(str).str.replace(r'\brd\b', 'road', regex=True)
    
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

    df = df[~df['property_type'].isin(['other', 'rent'])]

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
    
    #price_category
    # Define price bins and labels
    price_bins = [0, 0.99, 1.99, 2.99, 3.99, 4.99, 5.99, 6.99, 7.99, 8.99, 9.99, 14.99, 20.00, float('inf')]
    price_labels = [
        "0.00 - 0.99", "1.00 - 1.99", "2.00 - 2.99", "3.00 - 3.99", "4.00 - 4.99", 
        "5.00 - 5.99", "6.00 - 6.99", "7.00 - 7.99", "8.00 - 8.99", "9.00 - 9.99", 
        "10.00 - 14.99", "15.00 - 20.00", "20.00 and above"
    ]
    
    # Use pd.cut to categorize the prices
    df['price_category'] = pd.cut(df['price'], bins=price_bins, labels=price_labels, right=True)

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    #overlooking
    df['overlooking'] = df['md_overlooking'].apply(
        lambda x: ', '.join(sorted(map(str.strip, x.split(',')))) if pd.notna(x) else np.nan
    )
    
    # Remove the phrase 'not available' from the 'overlooking' column
    df['overlooking'] = df['overlooking'].str.replace(',? *not available', '', regex=True)

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    #room_type
    df['room_type'] = df['name'].apply(lambda x: 'flat' if 'flat' in x else ('apartment' if 'apartment' in x else 'other'))

    #drop apartment rows
    df = df[df['room_type'] != 'apartment']

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    # adding extra columns
    def add_engineered_features(df):
        df = df.copy()
    
        # --- Ratio Features ---
        df["bath_bed_ratio"] = df["bath"].div(df["bed"], fill_value=np.nan).replace([np.inf, -np.inf], np.nan)
        df["bed_area_ratio"] = df["bed"].div(df["area"], fill_value=np.nan).replace([np.inf, -np.inf], np.nan)
        df["bed_bath_ratio"] = df["bed"].div(df["bath"], fill_value=np.nan).replace([np.inf, -np.inf], np.nan)
        df["bed_balcony_ratio"] = df["bed"].div(df["balcony"], fill_value=np.nan).replace([np.inf, -np.inf], np.nan)
    
        # --- Density Features ---
        df["project_density"] = df["available_units"].div(df["project_in_acres"], fill_value=np.nan).replace([np.inf, -np.inf], np.nan)
        df["compactness_ratio"] = df["area"].div(df["project_in_acres"], fill_value=np.nan).replace([np.inf, -np.inf], np.nan)
    
        # --- Floor Features ---
        df["floor_ratio"] = df["flat_on_floor"].div(df["total_floor"], fill_value=np.nan).replace([np.inf, -np.inf], np.nan)
        df["remaining_floors"] = df["total_floor"].sub(df["flat_on_floor"], fill_value=np.nan)
    
        # --- Area per-unit Features ---
        df["area_per_bedroom"] = df["area"].div(df["bed"], fill_value=np.nan).replace([np.inf, -np.inf], np.nan)
        df["area_per_bathroom"] = df["area"].div(df["bath"], fill_value=np.nan).replace([np.inf, -np.inf], np.nan)
        df["area_per_balcony"] = df["area"].div(df["balcony"], fill_value=np.nan).replace([np.inf, -np.inf], np.nan)
        df["area_per_parking"] = df["area"].div(df["parking"], fill_value=np.nan).replace([np.inf, -np.inf], np.nan)
    
        # --- Amenity Ratios ---
        df["balcony_to_bed_ratio"] = df["balcony"].div(df["bed"], fill_value=np.nan).replace([np.inf, -np.inf], np.nan)
        df["parking_to_bed_ratio"] = df["parking"].div(df["bed"], fill_value=np.nan).replace([np.inf, -np.inf], np.nan)
        df["lift_to_total_floor_ratio"] = df["lift"].div(df["total_floor"], fill_value=np.nan).replace([np.inf, -np.inf], np.nan)
    
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
    
    # --- Function to add distance column BEFORE split ---
    def add_distance_to_center(df, city_col='city', lat_col='latitude', lon_col='longitude'):
        df = df.copy()
        df['distance_to_center_km'] = df.apply(
            lambda row: haversine(
                row[lat_col],
                row[lon_col],
                city_centers[row[city_col]][0],
                city_centers[row[city_col]][1]
            ), axis=1
        )
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

    # Step 1: get all AM columns
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

    #df.drop(columns="amenities", inplace=True)

    # Step 3: Drop original 'am_' columns
    #df.drop(columns=am_cols, inplace=True)

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------

    #Custom data corrections and row-level cleaning

    # List of IDs to remove
    ids_to_remove = [
        'cardid70421965',  
        'cardid71698587',
        'cardid41440251',
        'cardid70017925',  
        'cardid73050463',
        'cardid49131617',
        'cardid72273473',
        'cardid66762427',
        'cardid70615879',
        'cardid72819785',
        'cardid71143703',
        'cardid72821117',
        'cardid72884955',
        'cardid72803713',
        'cardid73037481',
        'cardid69783235',
        'cardid73144165',
        'cardid33966233',
        'cardid73046249',
        'cardid69702399',
        'cardid54078457',
        'cardid71697753'
    ]
    
    # Drop rows with matching IDs
    df = df[~df['id'].isin(ids_to_remove)].reset_index(drop=True)
    
    #after observation 
    ids_to_update = ['cardid73059851', 'cardid72926775', 'cardid58806131']
    
    df.loc[df['id'].isin(ids_to_update), 'city'] = 'palghar'
    df.loc[df['id'].isin(ids_to_update), 'location'] = 'palghar'
    
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
    ids_to_update = [
        "cardid72703033",
        "cardid69846363",
        "cardid73257889",
        "cardid56191653",
        "cardid72796607",
        "cardid73026297",
        "cardid72794677",
        "cardid66964031",
        "cardid58541153",
        "cardid73076791",
        "cardid72794677",
        "cardid53323155",
        "cardid69812109",
        "cardid69665873",
        "cardid70673145",
        "cardid70120173",
        "cardid60101171",
        "cardid73012265",
        "cardid73028981",
        "cardid71481487",
        "cardid67617413",
        "cardid53977959"
        
    ]
    
    df.loc[df['id'].isin(ids_to_update), 'city'] = 'thane'
    
    #make palghar in city for all this ids
    ids_to_update = [
        "cardid72923721",
        "cardid61647785",
        "cardid70476757",
        "cardid72179863",
        "cardid72846389",
        "cardid73127129",
        "cardid61883771",
        "cardid72998493",
        "cardid73114181",
        "cardid71923233",
        "cardid63887703",
        "cardid72831163"
    ]
    
    df.loc[df['id'].isin(ids_to_update), 'city'] = 'palghar'
    
    #make navi mumbai in city for all this ids
    ids_to_update = [
        "cardid62724753"
    ]
    
    df.loc[df['id'].isin(ids_to_update), 'city'] = 'navi mumbai'


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------



    # Create a mask for rows where 'latitude' starts with 16, 12, or 9
    mask = (
        df['latitude'].astype(str).str.startswith('16') |
        df['latitude'].astype(str).str.startswith('12') |
        df['latitude'].astype(str).str.startswith('9')
    )
    
    # Replace only 'latitude' and 'longitude' with NaN for those rows
    df.loc[mask, ['latitude', 'longitude']] = np.nan


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------

    cols = [
        "transportation hub_1",
        "transportation hub_2",
        "transportation hub_3",
        "transportation hub_4",
        "transportation hub_5"
    ]
    
    # Combine into one column
    df["transportation_hubs"] = df[cols].astype(str).apply(
        lambda x: ", ".join([i for i in x if i != "nan"]),
        axis=1
    )
    
    # (Optional) Drop original columns
    df.drop(columns=cols, inplace=True)
    
    #print(df[["transportation_hubs"]].head())

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------


    import re

    def split_combined_stations(item):
        # Handle "A and B railway/monorail stations"
        match = re.search(r"(.+?) and (.+?) (railway|monorail) stations?", item)
        
        if match:
            part1 = match.group(1).strip()
            part2 = match.group(2).strip()
            station_type = match.group(3).strip()
            
            return [
                f"{part1} {station_type} station",
                f"{part2} {station_type} station"
            ]
        
        return [item]
    
    
    
    def clean_transportation_hubs(text):
        if pd.isna(text):
            return text
    
        # Replace entire problematic pattern BEFORE splitting
        text = text.replace(
            "chembur monorail station(46.4 km), tilak nagar railway station (harbour line)(45.7 km), bolinj naka bus stop(1.8 km)",
            "virar railway station"
        )
    
        # Step 2: Remove (x km) once here
        #text = re.sub(r"\(\s*\d+\.?\d*\s*km\s*\)", "", text)
        
        # Split by comma
        items = [i.strip().lower() for i in text.split(",")]
        
        cleaned = []
        
        for item in items:
            
            item = item.replace("nearest-", "") \
               .replace("upcoming-", "") \
               .replace("upcoming", "") \
               .replace("upcoming metro station kasardavali", "upcoming kasarvadavali metro station") \
               .replace("bhayander", "bhayandar") \
               .replace("rd", "road") \
               .replace("rasayani train station", "rasayani railway station") \
               .replace("thane railway station east bus stop", "thane railway station") \
               .replace("nr.panvel railway station.", "panvel railway station") \
               .strip()
    
            # REMOVE ONLY IF EXACT WORDS
            if item.strip().lower() in ["mumbai", "railway station"]:
                continue
                
            # Skip unwanted keywords
            unwanted_keywords = [
                "sector 14", "shilp chowk", "ganesh chowk",
                "shree saibaba mandir", "mansarovar railway station",
                "mangatram petrol pump bus stop","jangal mangal road",
                "(central and western lines)","(western and harbour line)",
                "(central and western line)","(western line)",
                "(harbour line)","(central line)","navda phata","gavhan phata",
                "mint colony","bolinj naka bus","vakola bridge","kolivery village bus stop",
                "various stations on the chembur-sant gadge maharaj monorail network",
                "trends - seawoods grand central mall","sector - 40",
                "koparkhairane railway station","hanuman road bus stop",
                "kasarvadavali naka","eastern freeway","new golden nest bus stop",
                "traction substation dahisar western railway","(east)","eastern freeway",
                "(central and harbour lines)","(vasai-diva-panvel memu line)",
                "vvmc headquater bus stop","(central and trans-harbour lines)"
                
            ]
            
            if any(kw in item for kw in unwanted_keywords):
                continue
        
            
            # Rename
            if "panvel railway junction" in item:
                item = item.replace("panvel railway junction", "panvel railway station")
    
            # Split combined stations
            split_items = split_combined_stations(item)
    
            cleaned.extend(split_items)
        
        # Remove duplicates while preserving order
        seen = set()
        final = []
        for i in cleaned:
            if i not in seen:
                seen.add(i)
                final.append(i)
    
        # FIX: return NaN instead of empty string
        if not final:
            return np.nan
        
        return ", ".join(final)
    
    
    # Apply
    df["transportation_hubs"] = df["transportation_hubs"].apply(clean_transportation_hubs)


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------

    df["transportation_hubs"] = df["transportation_hubs"].replace("", np.nan)

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------

    def clean_transportation(text):
        if not isinstance(text, str) or text.strip() == "":
            return ""
    
        text = text.lower().strip() #here
    
        # replace entire problematic pattern BEFORE splitting
        text = text.replace("malad railway station(3.1 km), kandivali railway station(1.8 km)", "kandivali rs") \
                .replace("malad railway station(10.0 km), kandivali railway station(10.1 km), mulund railway station(1.4 km), panch rasta(0.0 km)", "mulund rs") \
                .replace("ram mandir railway station(8.5 km), borivali railway station(0.7 km), borivali railway station(0.0 km)", "borivali rs") \
                .replace("pethpada metro station(1.9 km), raghunath metro station(2.9 km), kharghar railway station(2.3 km)", "kharghar rs") \
                .replace("dahisar railway station(9.8 km), borivali railway station(7.6 km)", "borivali rs") \
                .replace("panvel railway station(0.4 km), panvel railway station", "panvel rs") \
                .replace("teen hath naka(1.9 km), thane railway station(1.9 km), shivaji chowk / kalwa naka(1.9 km), bramand metro station(3.7 km)", "thane rs") \
                .replace("lower parel railway station(1.6 km), currey road station(1.9 km)", "lower parel rs") \
                .replace("kandivali railway station(10.3 km), nahur railway station(1.7 km)", "nahur rs") \
                .replace("ghatkopar railway station(0.8 km), kanjurmarg railway station(5.9 km)", "ghatkopar rs") \
                .replace("bandra worli sea link(4.8 km), lower parel monorail station(1.6 km), sant gadge maharaj monorail station(1.1 km), mahalakshmi railway station(0.7 km)", "mahalakshmi rs") \
                .replace("bandra railway station, khar road railway station, bandra railway station(1.6 km)", "bandra rs") \
                .replace("prabhadevi railway station(2.6 km), bandra worli sea link(5.1 km)", "prabhadevi rs") \
                .replace("taloja panchnand railway station(3.2 km), taloja metro station(3.8 km), taloja panchnand railway station(3.3 km)", "taloja panchnand rs") \
                .replace("bamandongri railway station, kharkopar railway station(4.4 km)", "kharkopar rs") \
                .replace("ghatkopar metro station(5.7 km), vidyavihar railway station(4.5 km), ghatkopar railway station(5.8 km)", "ghatkopar rs") \
                .replace("seawoods darawe railway station(1.0 km), seawoods metro station(1.0 km)", "seawoods darawe rs") \
                .replace("andheri railway station(10.3 km), versova metro station(13.2 km)", "andheri rs") \
                .replace("andheri railway station(2.9 km), versova metro station(0.9 km)", "andheri rs") \
                .replace("ghatkopar metro station(9.3 km), vidyavihar railway station(9.1 km)", "vidyavihar rs") \
                .replace("parel village bus stop(0.2 km), cotton green railway station(0.2 km)", "cotton green rs") \
                .replace("mahim junction railway station(0.5 km), bandra worli sea link(3.0 km), mahim railway station(0.5 km)", "mahim rs") \
                .replace("kanjurmarg railway station(39.3 km), virar railway station(0.1 km), sakinaka metro station(40.8 km)", "virar rs") \
                .replace("thane railway station(3.0 km), kasarvadavali metro station(7.1 km)", "thane rs") \
                .replace("bhayandar railway station(4.8 km), mira road railway station(7.8 km)", "bhayandar rs") \
                .replace("bandra worli sea link(8.0 km), bandra railway station(4.9 km), khar road railway station(3.3 km)", "khar road rs") \
                .replace("virar railway station(7.3 km), vasai road railway station(1.4 km)", "vasai road rs") \
                .replace("mira road railway station(13.2 km), bhayandar railway station(9.8 km)", "bhayandar rs") \
                .replace("grant road railway stations(0.7 km), mumbai central railway station(0.6 km)", "mumbai central rs") \
                .replace("kasarvadavali metro station(8.3 km), thane railway station(1.4 km)", "thane rs") \
                .replace("mira road railway station(0.4 km), dahisar railway station(3.2 km)", "mira road rs") \
                .replace("dockyaroad road railway station(38.1 km), bhayandar railway station(1.0 km)", "bhayandar rs") \
                .replace("naigaon monorail station, dadar east monorail station, dadar railway station(0.4 km)", "dadar rs") \
                .replace("sanpada railway station(1.2 km), juinagar railway station(0.3 km), sanpada railway station(1.1 km)", "juinagar rs") \
                .replace("belapur metro station(0.2 km), belapur railway station (e)(0.3 km), sector 7 metro station(1.2 km), kharghar sector 7 rbi colony metro station(1.1 km)", "belapur rs") \
                .replace("bhandup railway station, nahur railway station", "bhandup rs") \
                .replace("jogeshwari(1.1 km), and ram mandir railway stations(2.3 km)", "jogeshwari rs") \
                .replace("khar road railway station(4.4 km), bandra railway station(6.1 km), bandra worli sea link(9.3 km)", "khar road rs") \
                .replace("bandra railway station(2.2 km), bandra worli sea link(4.5 km)", "bandra rs") \
                .replace("kandivali railway station, malad railway station", "malad rs") \
                .replace("bhayandar railway station(4.6 km), mira road railway station(2.5 km)", "mira road rs") \
                .replace("cbd belapur metro terminal(4.2 km), vashi railway station(2.4 km)", "vashi rs") \
                .replace("asalpha metro station(2.4 km), mumbai(2.3 km)", "asalpha ms") \
                .replace("wadala monorail station(20.8 km), gtb nagar monorail station(19.1 km)", "wadala mono") \
                .replace("wadala bridge monorail station(0.5 km)", "wadala mono") \
                .replace("wadala bridge(5.2 km)", "wadala mono") \
                .replace("metro station kasaroadavali(0.7 km), kharbav railway station(4.0 km)", "kharbav rs") \
                .replace("bandra worli sea link(2.8 km), mahim junction railway station(0.7 km)", "mahim rs") \
                .replace("charni road railway station(2.1 km), mumbai central railway station(2.7 km)", "charni road rs") \
                .replace("kandivali railway station(12.6 km), mulund check naka depo(1.2 km), malad railway station(12.3 km)", "malad rs") \
                .replace("dombivali railway station(1.2 km), dombivali metro station(1.2 km)", "dombivali rs") \
                .replace("andheri and azad nagar metro stations(0.8 km), andheri railway station(1.0 km)", "andheri rs") \
                .replace("virar railway station(1.3 km), railway station(0.7 km)", "virar rs") \
                .replace("asangaon railway station(27.8 km), shahad railway station(2.0 km), ulhasnagar railway station(1.3 km), vithalwadi railway station(1.6 km)", "ulhasnagar rs") \
                .replace("khandeshwar metro station(2.8 km), panvel railway station(0.8 km)", "panvel rs") \
                .replace("mumbai cst mumbai central(12.7 km)", "mumbai central rs") \
                .replace("dombivali metro station(2.6 km), dombivali railway station(2.6 km)", "dombivali rs") \
                .replace("jogeshwari(2.9 km), and ram mandir railway stations(2.4 km)", "jogeshwari rs") \
                .replace("mahavir nagar metro station(3.3 km), borivali railway station(1.1 km)", "borivali rs") \
                .replace("khar road railway station(4.4 km), bandra railway station(6.0 km)", "khar road rs") \
                .replace("bhayandar railway station(4.6 km), mira road railway station(2.5 km)", "mira road rs" ) \
                .replace("lower parel(9.7 km)", "lower parel rs") \
                .replace("sant gadge maharaj chowk monorail station(7.6 km)", "mahalakshmi rs") \
                .replace("mith chowky bus stop(0.0 km)", "malad rs") \
                .replace("balkum naka(0.0 km)", "thane rs") \
                .replace("bandra worli sealink(6.5 km)", "mahim rs") \
                .replace("dahisar check naka(0.2 km)", "dahisar rs")
    
        # Step 2: Remove (x km)
        text = re.sub(r"\(\d+(\.\d+)?\s*km\)", "", text)
    
        # Step 3: Split by comma
        parts = [p.strip() for p in text.split(",") if p.strip()]  #here
    
        cleaned = []
    
        for item in parts:
    
            # Railway station → rs
            if "railway station" in item:
                item = item.replace("railway station", "").strip()
                item = f"{item} rs"
    
            # Metro station → ms
            elif "metro station" in item:
                item = item.replace("metro station", "").strip()
                item = f"{item} ms"
    
            # Monorail station → mono
            elif "monorail station" in item:
                item = item.replace("monorail station", "").strip()
                item = f"{item} mono"
    
            # Monorail → mono
            elif "monorail" in item:
                item = item.replace("monorail", "").strip()
                item = f"{item} mono"
    
            cleaned.append(item)
    
        # Step 4: Remove duplicates while preserving order
        seen = set()
        final = []
    
        for i in cleaned:
            if i not in seen:
                seen.add(i)
                final.append(i)
    
        # Convert list → string
        result = ", ".join(final)
    
        # If comma exists → replace with /
        if ", " in result:
            result = result.replace(", ", " / ")
    
        return result
    
    # Apply
    df["transportation_hubs"] = df["transportation_hubs"].apply(clean_transportation)
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------

    df["transportation_hubs"] = df["transportation_hubs"].replace("", np.nan)

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    # Step 1: Define bin size
    bin_size = 0.50
    
    # Step 2: Create bins
    max_price = df['price'].max()
    bins = np.arange(0, max_price + bin_size, bin_size)
    
    # Step 3: Create labels
    labels = [f"{bins[i]:.2f} - {bins[i+1]:.2f}" for i in range(len(bins)-1)]
    
    # Step 4: Create price_range column
    df['price_range'] = pd.cut(
        df['price'],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True   # ✅ ensures 0.50 → 0.00–0.50
    )
    
    # Step 5: Keep only existing ranges
    price_counts = (
        df['price_range']
        .value_counts()
        .sort_index()
    )
    
    # Remove zero-count bins (only real ranges remain)
    price_counts = price_counts[price_counts > 0]
    
    #print(price_counts)

 













    
    





    #--------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    #drop columns
    df.drop(['@id','numberofrooms','bb_beds','leftbb_beds','bb_bed','leftbb_bed','multi_bed_filled','bb_baths','leftbb_baths','bb_bath','leftbb_bath','multi_bath_filled',
             'bb_covered-parking','leftbb_covered-parking','many_car parking','leftmany_car parking','md_price breakup','property_loc','many_transaction type',
             'leftmany_transaction type','many_type of ownership','leftmany_type of ownership','many_status', 'leftmany_status','many_lifts','md_lift','leftmany_lifts',
             'many_lift','leftmany_lift','multi_lift_filled','aboutpjt_total floors','floor_work_1','many_floor','leftmany_floor','md_floors allowed for construction',
             'construction_1','many_age of construction','leftmany_age of construction','bb_balcony', 'leftbb_balcony', 'bb_balconies','leftbb_balconies',
             'leftmany_additional rooms', 'balcony1', 'many_additional rooms','extra_room', 'md_additional rooms','leftmany_facing','many_facing','ap_unit','ap_tower',
             'ap_tower & unit','geo','potentialaction','md_overlooking','room_type','aboutpjt_project size','educational institute_1','educational institute_2',
             'educational institute_3','educational institute_4','educational institute_5',
             'shopping centre_1','shopping centre_2','shopping centre_3','shopping centre_4','shopping centre_5',
             'commercial hub_1','commercial hub_2','commercial hub_3','commercial hub_4','commercial hub_5','hospital_1','hospital_2','hospital_3','hospital_4','hospital_5',
             'tourist spot_1','tourist spot_2','tourist spot_3','tourist spot_4',
             'url','image','name','wholeaddress','address','powercut_hours','price_category','emi', 'authority_approval_clean','rera_id_grouped',
             'nearby_landmarks',
             'water_availability_hours','available_units','towers','locality_review_count',
             'bath_bed_ratio','bed_area_ratio','bed_bath_ratio','bed_balcony_ratio',
             'project_density','compactness_ratio','floor_ratio','remaining_floors',
             'area_per_bedroom','area_per_bathroom','area_per_balcony','area_per_parking',
             'balcony_to_bed_ratio','parking_to_bed_ratio','lift_to_total_floor_ratio',
             'overall_min_mean_km','overall_avg_mean_km','overall_min_min_km','overall_avg_min_km',
             'seller','tourist_mean_km','tourist_min_km','hospital_mean_km','hospital_min_km','transport_mean_km','transport_min_km',
            ],axis=1,inplace=True) # 'locality_rank', 'locality_url_rating'
            #'id','transportation hub_1','transportation hub_2','transportation hub_3','transportation hub_4','transportation hub_5','price'
            #'education', 'transport', 'shopping_centre', 'commercial_hub', 'hospital', 'tourist'


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------

    # =========================
    # MERGE
    # =========================
    merged_df = pd.merge(df, webscraped_df, on="id", how="left")
    merged_df = merged_df.drop(columns=['AP_Pjt_URL'], errors='ignore')
    merged_df.columns = merged_df.columns.str.lower()
    
    # =========================
    # RANGE EXTRACTION
    # =========================
    def extract_range(value):
        if pd.isna(value):
            return None, None
        
        text = str(value).replace(',', '')  # ✅ remove commas
        
        nums = re.findall(r'\d+', text)
        
        if len(nums) >= 2:
            return int(nums[0]), int(nums[1])
        elif len(nums) == 1:
            return int(nums[0]), int(nums[0])
        else:
            return None, None
    
    merged_df[['rent_min', 'rent_max']] = merged_df['locality_avg_rent'].apply(lambda x: pd.Series(extract_range(x)))
    merged_df[['buy_min', 'buy_max']] = merged_df['locality_avg_buy'].apply(lambda x: pd.Series(extract_range(x)))
    
    # =========================
    # LIST FEATURES
    # =========================
    merged_df['tech_spec_list'] = merged_df['tech_spec'].str.split('|').apply(
        lambda lst: [i.strip().lower() for i in lst] if isinstance(lst, list) else lst
    )
    
    merged_df['usp_list'] = merged_df['usp'].str.split('|').apply(
        lambda lst: [i.strip().lower() for i in lst] if isinstance(lst, list) else lst
    )
    
    # =========================
    # CLEAN FUNCTIONS
    # =========================
    def clean_text(text):
        if pd.isna(text):
            return ""
        
        text = str(text)
        text = text.replace('\xa0', ' ')
        text = text.replace('|', ' ')
        text = text.lower()
        
        text = re.sub(r'\bmins?\b', 'minutes', text)
        text = re.sub(r'(\d)(km)', r'\1 km', text)
        
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def clean_review(text):
        if pd.isna(text):
            return ""
        
        text = str(text)
        text = text.replace('\xa0', ' ')
        text = re.sub(r'\.(?=[a-zA-Z])', '. ', text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def clean_overall(text):
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        remove_phrases = [
            "residents feel that",
            "people liked",
            "however",
            "there are some concerns",
            "some concerns",
            "buyers experienced",
            "while there is",
        ]
        
        for phrase in remove_phrases:
            text = text.replace(phrase, "")
        
        text = text.replace('\xa0', ' ')
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    # =========================
    # 🔹 STEP 1: CUSTOM FLUFF REMOVAL
    # =========================
    FLUFF_WORDS = [
        "luxurious", "luxury", "grand", "magnificent",
        "premium", "world class", "royal", "elite",
        "beautiful", "stunning", "amazing",
        "timeless appeal", "modern lifestyle",
        "exclusive", "high end", "ultra modern",
        "spacious living", "elegant lifestyle",
        "beautifully designed", "lavish", "iconic",
        "state of the art", "best in class"
    ]
    
    def remove_fluff(text):
        if not text:
            return ""
        
        text = str(text).lower()
        
        for word in FLUFF_WORDS:
            text = text.replace(word, "")
        
        return text
    
    # =========================
    # 🔹 STEP 2: TF-IDF BASED CLEANING
    # =========================
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    def apply_tfidf_filter(series, max_df=0.85, min_df=2, max_features=5000):
        """
        Remove overly common + weak words but KEEP important ones
        """
        vectorizer = TfidfVectorizer(
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            stop_words='english'
        )
        
        vectorizer.fit(series.fillna(''))
        vocab = set(vectorizer.get_feature_names_out())
        
        def filter_text(text):
            words = str(text).split()
            return " ".join([w for w in words if w in vocab])
        
        return series.apply(filter_text)
    
    
    
    # =========================
    # APPLY CLEANING
    # =========================
    merged_df['positive'] = merged_df['positive'].apply(clean_review)
    merged_df['needs_improvement'] = merged_df['needs_improvement'].apply(clean_review)
    merged_df['overall_clean'] = merged_df['overall'].apply(clean_overall)
    
    # =========================
    # ✅ AMENITIES HANDLING
    # =========================
    amenity_cols = [col for col in merged_df.columns if col.startswith('am_')]
    
    def get_amenities(row):
        amenities = []
        
        for col in amenity_cols:
            val = row[col]
            if pd.notna(val) and str(val).strip() != "":
                a = str(val).strip().lower()
                a = a.replace('/', ' ').replace('-', ' ')
                amenities.append(a)
        
        return list(set(amenities))  # remove duplicates
    
    merged_df['amenities_list'] = merged_df.apply(get_amenities, axis=1)
    
    # =========================
    # FUNCTION: normalize transport
    # =========================
    def clean_transport(text):
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
    
        # replace abbreviations
        text = re.sub(r'\brs\b', 'railway station', text)
        text = re.sub(r'\bms\b', 'metro station', text)
        text = re.sub(r'\bmo\b', 'monorail', text)
        text = re.sub(r'\bmono\b', 'monorail', text)
    
        # remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
    
        return text
    
    
    # =========================
    # APPLY ON COLUMN
    # =========================
    merged_df['transportation_hubs_clean'] = merged_df['transportation_hubs'].apply(clean_transport)
    
    # =========================
    # FUNCTION: distance → semantic text
    # =========================
    def parse_places(text):
        if pd.isna(text) or str(text).strip() == "":
            return []
        
        places = str(text).split(',')
        result = []
        
        for p in places:
            p = p.strip().lower()
            p = p.replace("nearest-", "")
            p = p.replace("nr.", "")
            
            match = re.match(r'(.*)\(([\d\.]+)\s*km\)', p)
            
            if match:
                name = match.group(1).strip()
                dist = float(match.group(2))
            else:
                name = p
                dist = None
            
            result.append((name, dist))
        
        return result
    
    
    def location_to_text(lst, prefix=""):
        if not lst:
            return ""
        
        texts = []
        seen = set()
        
        for name, dist in lst:
            if not name:
                continue
            
            name = name.strip().lower()
            
            # clean text
            name = name.replace("upcoming-", "upcoming ")
            name = re.sub(r'[^\w\s]', ' ', name)
            name = re.sub(r'\s+', ' ', name).strip()
            
            # 🔥 IMPORTANT: FILTER ONLY FOR TRANSPORT
            if prefix == "transport":
                if not any(k in name for k in ["station", "metro", "railway", "junction", "monorail"]):
                    continue
            
            # dedup
            if name in seen:
                continue
            seen.add(name)
            
            # distance tagging (ONLY for transport)
            if prefix == "transport":
                if dist is None:
                    text = f"nearby {name}"
                elif dist <= 2:
                    text = f"nearby {name}"
                elif dist <= 5:
                    text = f"close {name}"
                elif dist <= 10:
                    text = f"accessible {name}"
                else:
                    text = f"far {name}"
            else:
                text = name
            
            texts.append(text)
        
        if prefix:
            return prefix + " " + " ".join(texts)
        
        return " ".join(texts)
    
    
    # =========================
    # 🔹 LOCATION COLUMNS
    # =========================
    location_cols = [
        'education',
        'transport',
        'shopping_centre',
        'commercial_hub',
        'hospital',
        'tourist'
    ]
    
    # =========================
    # 🔹 APPLY PARSING + TEXT
    # =========================
    for col in location_cols:
        if col in merged_df.columns:
            
            # step 1: parse
            merged_df[f"{col}_parsed"] = merged_df[col].apply(parse_places)
            
            # step 2: convert to text
            if col == "transport":
                merged_df[f"{col}_text"] = merged_df[f"{col}_parsed"].apply(
                    lambda x: location_to_text(x, prefix="transport")
                )
            else:
                merged_df[f"{col}_text"] = merged_df[f"{col}_parsed"].apply(
                    location_to_text
                )
    
    
    # =========================
    # CREATE FEATURES TEXT
    # =========================
    merged_df['amenities_text'] = merged_df['amenities_list'].apply(lambda x: ' '.join(x))
    
    merged_df['nearest_text'] = (
        merged_df['education_text'].fillna('') + ' ' +
        merged_df['transport_text'].fillna('') + ' ' +
        merged_df['shopping_centre_text'].fillna('') + ' ' +
        merged_df['commercial_hub_text'].fillna('') + ' ' +
        merged_df['hospital_text'].fillna('') + ' ' +
        merged_df['tourist_text'].fillna('')
    )

    
    # =========================
    # 🔥 SMART MERGE (REMOVE DUPLICATES BETWEEN COLUMNS)
    # =========================
    def split_features(text):
        if pd.isna(text):
            return []
        
        return [i.strip().lower() for i in str(text).split('|') if i.strip()]
    
    merged_df['tech_spec_list'] = merged_df['tech_spec'].apply(split_features)
    merged_df['usp_list'] = merged_df['usp'].apply(split_features)
    
    # overall_clean is already sentence-based, not '|'
    merged_df['overall_list'] = merged_df['overall_clean'].apply(
        lambda x: re.split(r'[.]', x) if pd.notna(x) else []
    )
    
    
    def merge_dedup_lists(list1, list2, list3):
        
        combined = list1 + list2 + list3
        
        seen = set()
        result = []
        
        for item in combined:
            item = item.strip()
            
            if not item:
                continue
            
            if item not in seen:
                seen.add(item)
                result.append(item)
        
        return result
    
    
    merged_df['features_list'] = merged_df.apply(
        lambda row: merge_dedup_lists(
            row['tech_spec_list'],
            row['usp_list'],
            row['overall_list']
        ),
        axis=1
    )
    
    merged_df['features_text'] = merged_df['features_list'].apply(
        lambda x: ' '.join(x)
    )
    
    
    for col in ['nearest_text', 'features_text', 'amenities_text']:
        merged_df[col] = merged_df[col].apply(clean_text)
        merged_df[col] = merged_df[col].apply(remove_fluff)
    
    # Apply ONLY on embedding text columns
    merged_df['features_text'] = apply_tfidf_filter(merged_df['features_text'])
    merged_df['nearest_text'] = apply_tfidf_filter(merged_df['nearest_text'])
    
    # =========================
    # 🔹 FINAL TRIM
    # =========================
    def trim_text(text, max_words=250):
        return " ".join(str(text).split()[:max_words])
    
    merged_df['features_text'] = merged_df['features_text'].apply(trim_text)
    merged_df['nearest_text'] = merged_df['nearest_text'].apply(trim_text)

    # ==========================================
    # ✅ FIX: MCP METADATA DATASET (Copy from merged_df!)
    # ==========================================

    mcp_df = merged_df.copy()  # 🌟 Changed from df.copy() to merged_df.copy()

    mcp_df["amenities_mcp"] = (mcp_df["amenities"])

    mcp_df["features_mcp"] = mcp_df["features_list"].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")
    

    # Use the cleaned/combined strings from earlier in basic_cleaning
    mcp_df["nearest_mcp"] = (
        mcp_df["education"].fillna("") + ", " +
        mcp_df["transport"].fillna("") + ", " +
        mcp_df["shopping_centre"].fillna("") + ", " +
        mcp_df["commercial_hub"].fillna("") + ", " +
        mcp_df["hospital"].fillna("") + ", " +
        mcp_df["tourist"].fillna("")
    )

    # Filter out all other columns completely
    mcp_df = mcp_df[['id', 'nearest_mcp', 'amenities_mcp', 'features_mcp']]

    # Comprehensive list of standard and long-form suffix unit keywords
    unit_patterns = [
        r'sq\.?\s*ft\.?', r'acres?', r'mins?', r'minutes?', r'bhks?', r'ft', r'feets?', 
        r'towers?', r'apartments?', r'kilometers?', r'degrees?', r'mtrs?', r'meters?', 
        r'beds?', r'hours?'
    ]
    combined_units = "|".join(unit_patterns)

    # Apply Regex Cleaning to the 3 target text columns
    mcp_cols = ['nearest_mcp', 'amenities_mcp', 'features_mcp']
    
    for col in mcp_cols:
        # Cast to string safely
        mcp_df[col] = mcp_df[col].astype(str).fillna("")
        
        # 1. Remove bracketed metrics like (2.3 km), (30 m), (20m)
        mcp_df[col] = mcp_df[col].str.replace(r'\(\s*[\d.]+\s*(?:km|m)\s*\)', '', regex=True)
        
        # 2. Remove complex compound ranges like "2, 3 & 4 bhks" or "1, 2 bhk"
        mcp_df[col] = mcp_df[col].str.replace(rf'\b\d+(?:\s*,\s*\d+)*(?:\s*&\s*\d+)?\s*[-–—]?\s*(?:{combined_units})\b', '', regex=True)
        
        # 3. Remove dimensions/matrices like "3 x 3" or "24x7"
        mcp_df[col] = mcp_df[col].str.replace(r'\b\d+\s*[xX]\s*\d+\b', '', regex=True)
        
        # 4. Remove standard hyphenated or spaced measurements (e.g., 20 minutes, 1.6-acre, 950 meters)
        mcp_df[col] = mcp_df[col].str.replace(rf'\b[\d,.]+\s*[-–—]?\s*(?:{combined_units})\b', '', regex=True)
        
        # 5. Remove structural notation height layouts like "1b+g+35"
        mcp_df[col] = mcp_df[col].str.replace(r'\b(?:\d*[a-zA-Z]\+)+[\s\d]*\b', '', regex=True)
        mcp_df[col] = mcp_df[col].str.replace(r'\b[a-zA-Z]\+\d+\b', '', regex=True)
        
        # 6. Remove standalone hanging plus counters like "40+", "50+", or "60 +"
        mcp_df[col] = mcp_df[col].str.replace(r'\b\d+\s*\+(?=\s|,|$)', '', regex=True)
        
        # 7. Remove raw, trailing standalone distance sequences like 2.9 km or 20m
        mcp_df[col] = mcp_df[col].str.replace(r'\b[\d.]+\s*[-–—]?\s*(?:kms?|m)\b', '', regex=True)
        
        # 8. Clean up multiple repeating commas down to a single comma space
        mcp_df[col] = mcp_df[col].str.replace(r',\s*,+', ',', regex=True)
        
        # 9. Standardize spacing around remaining commas
        mcp_df[col] = mcp_df[col].str.replace(r'\s*,\s*', ', ', regex=True)
        
        # 10. Strip whitespace and any leading/trailing commas completely
        mcp_df[col] = mcp_df[col].str.strip().str.strip(',')
        
        # 11. If the entire row contains nothing but commas/whitespace, reset it to empty string
        mcp_df.loc[mcp_df[col].str.match(r'^[,\s]*$'), col] = ""

    mcp_df.to_csv(
        "data/cleaned/property_metadata_mcp.csv",
        index=False
    )

    print("Saved MCP dataset")
    
    # =========================
    # DROP UNUSED COLUMNS
    # =========================
    merged_df = merged_df.drop(
        columns=[
            'locality_avg_rent',
            'locality_avg_buy',
            'tech_spec',
            'usp',
            'overall',
            'overall_clean',
            'transportation_hubs',
            'education_parsed',
            'transport_parsed',
            'shopping_centre_parsed',
            'commercial_hub_parsed',
            'hospital_parsed',
            'tourist_parsed',
            'tech_spec_list',
            'usp_list',
            'amenities_list',
            'overall_list',
            'features_list',
            'education_text',
            'transport_text',
            'shopping_centre_text',
            'commercial_hub_text',
            'hospital_text',
            'tourist_text',
            "amenities"
        ] + location_cols + amenity_cols,
        errors='ignore'
    )

    # =========================================================================
    # 🤝 FINAL MERGE: Combine directly and save separately
    # =========================================================================
    # Direct merge since mcp_df only has id and the 3 clean text columns
    final_combined_df = pd.merge(
        merged_df, 
        mcp_df, 
        on="id", 
        how="left"
    )

    final_combined_df.drop(
        columns=[
            "amenities_text",
            "nearest_text",
            "features_text",
        ],
        errors="ignore",
        inplace=True
    )

    # Save final_combined_df separately as a new artifact
    combined_save_path = Path("data/cleaned/final_combined_mcp_data.csv")
    final_combined_df.to_csv(combined_save_path, index=False)
    print(f"Saved complete combined dataset separately to: {combined_save_path} | Shape: {final_combined_df.shape}")


    df= merged_df

            
    print(df.shape)
    
    return df


def perform_property_data_cleaning_rec(
    data: pd.DataFrame,
    webscraped_df: pd.DataFrame,   # ✅ added
    saved_data_path: Path
) -> None:
    
    logger.info("Starting recommendation data cleaning pipeline")
    
    cleaned_data = (
        data
        .pipe(basic_cleaning, webscraped_df=webscraped_df)   # ✅ pass here
    )
    
    cleaned_data.to_csv(saved_data_path, index=False)


if __name__ == "__main__":
    
    # root path
    root_path = Path(__file__).parent.parent.parent

    # data save directory
    cleaned_data_save_dir = root_path / "data" / "cleaned"
    cleaned_data_save_dir.mkdir(parents=True, exist_ok=True)

    # cleaned data file name
    cleaned_data_filename = "final_cleaned_rec_data.csv"

    # data save path
    cleaned_data_save_path = cleaned_data_save_dir / cleaned_data_filename

    # main data load path
    data_load_path = (
        root_path / "data" / "raw" / "f_original magicbricks cleaned 12022 data.csv"
    )

    # ✅ NEW: webscraped data path
    webscraped_data_path = (
        root_path / "data" / "raw" / "magic_text_webscrapping.csv"
    )

    # load the data
    df = load_data(data_load_path)
    logger.info("Main data read successfully")

    # ✅ load webscraped data
    webscraped_df = pd.read_csv(webscraped_data_path, encoding='latin1')
    logger.info("Webscraped data read successfully")

    # clean the data and save
    perform_property_data_cleaning_rec(
        data=df,
        webscraped_df=webscraped_df,   # ✅ added
        saved_data_path=cleaned_data_save_path
    )

    logger.info("Data cleaned and saved")


import pandas as pd
import numpy as np
import requests
import enum
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from src.ml import logger


class DATATYPE(enum.Enum):
    KEPLER = "KEPLER"
    TESS = "TESS"
    K2 = "K2"


class DataPreprocessor:
    """
    A class for preprocessing Kepler exoplanet data.
    """
    
    def __init__(
            self,
            data_dir: str,
            datatype: DATATYPE = DATATYPE.KEPLER):
        """
        Initialize the DataPreprocessor.
        
        Parameters:
        -----------
        data_dir : str, default='../data'
            Directory path for data files
        datatype : DATATYPE, default=DATATYPE.KEPLER
            Dataset typeto use
        """
        self.data_dir = data_dir
        self.datatype = datatype
        self.random_state = 42

    def download_data(self, url: str = None):
        """
        Download the data from the NASA hackathon.
        
        Parameters:
        -----------
        url : str, optional
            URL to download data from. If None, uses default Kepler URL.
        """
        if url is None:
            url = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative"
            
        response = requests.get(url)
        with open(f'{self.data_dir}/kepler.csv', 'wb') as file:
            file.write(response.content)

    def define_data(self, filename: str):
        """
        Define the data for the model.
        
        Parameters:
        -----------
        filename : str
            Name of the file to read
            
        Returns:
        --------
        tuple
            X (features) and y (labels) DataFrames
        """
        for delimiter in [',', ';']:
            logger.info(f"Reading file {filename} with delimiter {delimiter}")
            try:
                outcome_labels = ["koi_disposition", "koi_pdisposition"]
                df = pd.read_csv(f'{self.data_dir}/{filename}', 
                                 delimiter=delimiter)
                X, y = df.drop(columns=outcome_labels, axis=1), df[outcome_labels]
                logger.info(f"Defined data with shape: {X.shape} and "
                           f"{y.shape}")
                return X, y
            except Exception as e:
                logger.error(f"Error reading file {filename} with delimiter "
                             f"{delimiter}: {e}")

        raise ValueError(f"No data defined for file {filename}")


    def map_to_binary(self, disposition):
            if pd.isna(disposition):
                return np.nan
            elif disposition in ['CP', 'KP', 'PC']:  # Confirmed planets + candidates
                return 1
            elif disposition in ['APC', 'FP', 'FA']:  # False positives + alarms
                return 0
            else:
                return np.nan  # Unknown categories become NaN

    def prepare_tess_data(self, filename: str):

        for delimiter in [',', ';']:
            logger.info(f"Reading file {filename} with delimiter {delimiter}")
            try:
                dataset = pd.read_csv(
                    f'{self.data_dir}/{filename}',
                    delimiter=delimiter)
                logger.info(f"Read file {filename} with delimiter {delimiter}")
                break
            except Exception as e:
                logger.error(f"Error reading file {filename} with delimiter "
                             f"{delimiter}: {e}")

        unusables = ["str", "id", "date", "created", "update", "toi"]
        keep = [col for col in dataset.columns if not any(unus in col for unus in unusables)]
        dataset = dataset[keep]
        
        y = pd.DataFrame(dataset["tfopwg_disp"].apply(self.map_to_binary))

        drop_cols = [
            # Planet properties that depend on refined stellar params
            "pl_rade", "pl_radeerr1", "pl_radeerr2", "pl_radelim",
            "pl_insol", "pl_insolerr1", "pl_insolerr2", "pl_insollim",
            "pl_eqt",  "pl_eqterr1",  "pl_eqterr2",  "pl_eqtlim",

            # Stellar parameter uncertainties (strong follow-up proxies)
            "st_tefferr1", "st_tefferr2",
            "st_loggerr1", "st_loggerr2",
            "st_raderr1",  "st_raderr2",
            "st_disterr1", "st_disterr2",

            # Stellar “limit” flags tied to the above derived params
            "st_tefflim", "st_logglim", "st_radlim", "st_distlim",

            # (If present) non-predictive/admin or time-leaky
            "toi", "rastr", "decstr", "rowupdate", "toi_created",

            # (If present) use tid only for grouping in CV, not as a feature
            "tid",
        ]
        drop_cols = [col for col in drop_cols if col in dataset.columns]


        X = dataset.drop(columns=["tfopwg_disp"]+drop_cols)

        return X, y

    def process_disposition_labels(self, y_df: pd.DataFrame):
        """
        Process disposition labels according to the specified logic.
        
        Parameters:
        -----------
        y_df : pd.DataFrame
            DataFrame containing koi_disposition and koi_pdisposition columns
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with processed labels
        """
        y_processed = y_df.copy()
        
        # Initialize the new label column
        y_processed['processed_label'] = None
        
        for idx, row in y_processed.iterrows():
            koi_disp = row['koi_disposition']
            koi_pdisp = row['koi_pdisposition']
            
            if koi_disp == "CONFIRMED":
                y_processed.loc[idx, 'processed_label'] = "Confirmed"
            
            elif koi_disp == "FALSE POSITIVE":
                y_processed.loc[idx, 'processed_label'] = "False Positive"
            
            elif koi_disp == "CANDIDATE":
                if koi_pdisp == "FALSE POSITIVE":
                    y_processed.loc[idx, 'processed_label'] = (
                        "Inconsistent (Archive=CANDIDATE vs Kepler=FP)")
                else:
                    y_processed.loc[idx, 'processed_label'] = "Candidate"
            
            else:  # Archive is NOT DISPOSITIONED or missing
                if koi_pdisp in ["CANDIDATE", "FALSE POSITIVE"]:
                    if koi_pdisp == "CANDIDATE":
                        y_processed.loc[idx, 'processed_label'] = "Candidate"
                    else:  # koi_pdisp == "FALSE POSITIVE"
                        y_processed.loc[idx, 'processed_label'] = (
                            "False Positive")
                elif pd.isna(koi_disp) and pd.isna(koi_pdisp):
                    y_processed.loc[idx, 'processed_label'] = (
                        "Not dispositioned")
                else:
                    y_processed.loc[idx, 'processed_label'] = (
                        "Inconsistent/Unknown")

        y_processed.drop(columns=["koi_disposition", "koi_pdisposition"],
                         inplace=True)

        # Convert to binary outcome: 1 for Candidate/Confirmed, 0 for others (excluding NaN)
        y_processed['binary_label'] = y_processed['processed_label'].apply(
            lambda x: 1 if x in ['Candidate', 'Confirmed'] else (0 if pd.notna(x) else None)
        )
        
        # Drop the original processed_label column and rename binary_label
        y_processed.drop(columns=['processed_label'], inplace=True)
        y_processed.rename(columns={'binary_label': 'processed_label'}, inplace=True)

        return y_processed

    def drop_unwanted_columns(self, df: pd.DataFrame):
        """
        Drop unwanted columns from the data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with unwanted columns removed
        """
        identifiers = [col for col in df.columns 
                       if ("name" in col) or ("id" in col)]
        data_cols = [col for col in df.columns if "date" in col]

        dropped_cols = identifiers + data_cols

        df.drop(columns=dropped_cols, inplace=True)

        # Drop columns with only one unique value
        single_value_cols = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                single_value_cols.append(col)
        
        if single_value_cols:
            logger.info(f"Dropping columns with single values: "
                        f"{single_value_cols}")
            df = df.drop(columns=single_value_cols)
        else:
            logger.info("No columns with single values found")
        
        # Drop additional columns if they still exist
        drop_cols = [
            # Direct/near-label info
            "koi_disposition",         # final archive disposition
            "koi_pdisposition",        # disposition using Kepler data
            "kepler_name",             # only assigned for confirmed planets → label proxy

            # Robovetter major FP flags (encodes vetting outcome)
            "koi_fpflag_nt",           # not transit-like
            "koi_fpflag_ss",           # stellar eclipse
            "koi_fpflag_co",           # centroid offset
            "koi_fpflag_ec",           # ephemeris match / contamination

            # Planet properties that depend on refined stellar params
            "koi_prad", "koi_prad_err1", "koi_prad_err2",
            "koi_teq",  "koi_teq_err1",  "koi_teq_err2",
            "koi_insol","koi_insol_err1","koi_insol_err2",

            # Stellar parameter uncertainties (follow-up/derived)
            "koi_steff_err1","koi_steff_err2",
            "koi_slogg_err1","koi_slogg_err2",
            "koi_srad_err1", "koi_srad_err2",

            # Cohort/version info (can cause distribution shift across deliveries)
            "koi_tce_delivname",
        ]
        
        existing_cols_to_drop = [col for col in drop_cols if col in df.columns]
        
        if existing_cols_to_drop:
            logger.info(f"Dropping additional columns: {existing_cols_to_drop}")
            df.drop(columns=existing_cols_to_drop, inplace=True)

        return df

    def process_datalink_columns(self, df: pd.DataFrame):
        """
        Process the datalink columns.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with processed datalink columns
        """
        # Feature transformations
        # koi_fittype: convert 'none' to NaN
        df['koi_fittype'] = df['koi_fittype'].replace('none', pd.NA)

        # koi_datalink_dvr: convert to binary (1 if report exists, 0 if NaN)
        df['koi_datalink_dvr'] = df['koi_datalink_dvr'].apply(
            lambda x: 1 if x is not None else 0)

        # koi_datalink_dvs: convert to binary (1 if report exists, 0 if NaN)
        df['koi_datalink_dvs'] = df['koi_datalink_dvs'].apply(
            lambda x: 1 if x is not None else 0)

        return df

    def cap_outliers(self, df: pd.DataFrame, percentile: float = 0.01):
        """
        Cap outlier values for float and int columns at specified percentile.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        percentile : float, default=0.01
            Percentile threshold for capping (1% means cap at 1st and 99th 
            percentiles)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with capped outlier values
        """
        df_copy = df.copy()
        
        # Get numeric columns (int and float)
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df_copy[col].notna().sum() > 0:  # Only process if has non-NaN values
                lower_bound = df_copy[col].quantile(percentile)
                upper_bound = df_copy[col].quantile(1 - percentile)
                
                # Count outliers before capping
                outliers_count = ((df_copy[col] < lower_bound) | 
                                 (df_copy[col] > upper_bound)).sum()
                
                if outliers_count > 0:
                    logger.info(f"Capping {outliers_count} outliers in column "
                               f"'{col}' at {percentile*100}% percentile")
                    
                    # Cap the values
                    df_copy[col] = df_copy[col].clip(lower=lower_bound, 
                                                     upper=upper_bound)
        
        return df_copy

    def one_hot_encode_sklearn_with_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encode categorical columns while preserving NaN values.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with one-hot encoded categorical columns
        """
        categorical_cols = df.select_dtypes(
            include=['object', 'string']
        ).columns.tolist()
        if not categorical_cols:
            return df.copy()

        enc = OneHotEncoder(
            drop='first',         # drop-first like your code
            handle_unknown='ignore',
            sparse_output=False,
            dtype='int8'
        )
        # Ensure uniform string type per column; use sentinel for missing
        df_cat = df[categorical_cols].copy()
        for col in categorical_cols:
            col_obj = df_cat[col].astype('object')
            mask_na = col_obj.isna()
            col_str = col_obj.astype(str)
            # Temporarily replace missing with a sentinel to avoid mixed types
            col_str[mask_na] = "__MISSING__"
            df_cat[col] = col_str

        X_cat = enc.fit_transform(df_cat)
        cat_feature_names = enc.get_feature_names_out(categorical_cols)

        dummies = pd.DataFrame(
                    X_cat,
                    columns=cat_feature_names,
                    index=df.index
                    ).astype('Int8')

        # Propagate NaNs across each feature's dummy block
        start = 0
        for i, col in enumerate(categorical_cols):
            # number of output cols for this feature after drop='first'
            n_out = len(enc.categories_[i]) - (1 if enc.drop is not None else 0)
            if n_out > 0:
                cols_slice = dummies.columns[start:start + n_out]
                mask_na = df[col].isna()
                if mask_na.any():
                    dummies.loc[mask_na, cols_slice] = pd.NA
                start += n_out

        df_non_cat = df.drop(columns=categorical_cols)
        return pd.concat([df_non_cat, dummies], axis=1)

    def process_comment_columns(self, df: pd.DataFrame):
        """
        Process the comment columns.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with processed comment columns
        """
        X_copy = df.copy()
        # --- 1) Mapping from raw flags to categories
        CATMAP = {
            # Centroid/contamination
            "CENT_KIC_POS": "CENTROID", "CENT_RESOLVED_OFFSET": "CENTROID",
            "CENT_UNRESOLVED_OFFSET": "CENTROID", 
            "CENT_FEW_DIFFS": "DATA_QUALITY",
            "CENT_FEW_MEAS": "DATA_QUALITY", "CENT_NOFITS": "DATA_QUALITY",
            "CENT_SATURATED": "DATA_QUALITY", "CENT_UNCERTAIN": "DATA_QUALITY",
            "HALO_GHOST": "CENTROID",

            # Ephemeris match
            "EPHEM_MATCH": "EPHEMERIS_MATCH",

            # Secondaries / eclipsing-binary cues
            "MOD_SEC_DV": "SECONDARY/EB", "MOD_SEC_ALT": "SECONDARY/EB",
            "HAS_SEC_TCE": "SECONDARY/EB", "DEPTH_ODDEVEN_DV": "SECONDARY/EB",
            "DEPTH_ODDEVEN_ALT": "SECONDARY/EB", "SWEET_EB": "SECONDARY/EB",

            # Not-transit-like shape
            "LPP_DV": "NOT_TRANSIT_LIKE", "LPP_ALT": "NOT_TRANSIT_LIKE",
            "INCONSISTENT_TRANS": "NOT_TRANSIT_LIKE",
            "INDIV_TRANS_CHASES": "NOT_TRANSIT_LIKE",
            "INDIV_TRANS_MARSHALL": "NOT_TRANSIT_LIKE",
            "INDIV_TRANS_SKYE": "NOT_TRANSIT_LIKE",
            "INDIV_TRANS_ZUMA": "NOT_TRANSIT_LIKE",
            "INDIV_TRANS_TRACKER": "NOT_TRANSIT_LIKE",
            "INDIV_TRANS_RUBBLE": "NOT_TRANSIT_LIKE",
            "ALL_TRANS_CHASES": "NOT_TRANSIT_LIKE",
            "SWEET_NTL": "NOT_TRANSIT_LIKE",

            # Uniqueness / period aliases / contaminating geometry
            "MOD_NONUNIQ_DV": "UNIQUENESS/ALIASES",
            "MOD_NONUNIQ_ALT": "UNIQUENESS/ALIASES",
            "PERIOD_ALIAS_DV": "UNIQUENESS/ALIASES",
            "PERIOD_ALIAS_ALT": "UNIQUENESS/ALIASES",
            "PLANET_IN_STAR": "UNIQUENESS/ALIASES",

            # Overrides that permit PCs in special cases
            "PLANET_OCCULT_DV": "OVERRIDES/OK", 
            "PLANET_OCCULT_ALT": "OVERRIDES/OK",
            "PLANET_PERIOD_IS_HALF_DV": "OVERRIDES/OK",
            "PLANET_PERIOD_IS_HALF_ALT": "OVERRIDES/OK",

            # No comment
            "NO_COMMENT": "NO_COMMENT",
        }

        # --- 2) Helpers
        def split_flags(s: str) -> list[str]:
            if (not isinstance(s, str) or not s.strip() or 
                    s.strip() == "NO_COMMENT"):
                return [] if s != "NO_COMMENT" else ["NO_COMMENT"]
            return [p.strip() for p in s.split('---') if p.strip()]

        def to_category(flag: str) -> str:
            return CATMAP.get(flag, "OTHER")

        # --- 3) Parse flags
        # df must have a 'koi_comment' column
        X_copy["comment_flags"] = X_copy["koi_comment"].apply(split_flags)

        # --- 4) Count raw-flag frequencies across your dataset
        # explode -> count -> map back
        exploded = X_copy.explode("comment_flags", ignore_index=False)
        flag_counts = (exploded["comment_flags"]
                       .value_counts(dropna=False)
                       .rename_axis("flag")
                       .reset_index(name="flag_count"))

        # --- 5) Attach counts and categories
        exploded = exploded.merge(flag_counts, left_on="comment_flags", 
                                  right_on="flag", how="left")
        exploded["flag_category"] = exploded["comment_flags"].map(to_category)

        # --- 6) Collapse flags that occur only once
        # Option A: collapse to global OTHER
        # exploding singleton flags into "OTHER" category is already handled by 
        # CATMAP default, but you may want to *force* singleton flags to 
        # "OTHER_<Category>" for transparency:

        def collapse_singletons(row):
            if pd.isna(row["comment_flags"]) or row["comment_flags"] == "":
                return pd.NA
            if row["flag_count"] <= 1:
                # Use category-aware OTHER to keep some information:
                base = (row["flag_category"] if row["flag_category"] != "OTHER" 
                        else "MISC")
                return f"OTHER_{base}"
            return row["comment_flags"]

        exploded["flag_collapsed"] = exploded.apply(
            collapse_singletons,
            axis=1)

        # --- 7) Re-aggregate to rows
        agg = (exploded.groupby(exploded.index).agg({
                "flag_category": lambda x: sorted(set([c for c in x 
                                                      if isinstance(c, str)])),
                "flag_collapsed": lambda x: sorted(set([f for f in x 
                                                       if isinstance(f, str)]))
            }))

        X_copy["comment_categories"] = agg["flag_category"]
        X_copy["comment_flags_norm"] = agg["flag_collapsed"]

        # --- 8) (Optional) binary cues for curation
        DECISIVE_FP_CATS = {"EPHEMERIS_MATCH", "CENTROID", "SECONDARY/EB", 
                            "NOT_TRANSIT_LIKE", "UNIQUENESS/ALIASES"}

        def has_decisive_cue(cats: list[str]) -> bool:
            return any(c in DECISIVE_FP_CATS for c in cats)

        X_copy["has_decisive_fp_cue"] = X_copy["comment_categories"].apply(
                            has_decisive_cue)

        # --- 9) If you want a compact string form
        X_copy["comment_categories_str"] = X_copy["comment_categories"].apply(
            lambda xs: ",".join(xs) if xs else "")
        X_copy["comment_flags_norm_str"] = X_copy["comment_flags_norm"].apply(
            lambda xs: ",".join(xs) if xs else "")

        # --- 10) Apply and keep separate from training features
        comment_categories_str = X_copy["comment_categories"].apply(
            lambda xs: ",".join(xs) if xs else "")
        df.drop(
            columns=df.filter(like="comment", axis=1).columns,
            inplace=True)

        df["comment_categories_str"] = comment_categories_str

        return df

    def prepare_data(self, df: pd.DataFrame):
        """
        Prepare the data for the model.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
            
        Returns:
        --------
        pd.DataFrame
            Processed DataFrame ready for modeling
        """
        X_copy = df.copy()
        X = self.process_datalink_columns(X_copy)
        X = self.process_comment_columns(X)
        X = self.drop_unwanted_columns(X)
        # Note: Outlier capping is now handled in the pipeline (DataTypeTransformer)
        # X = self.cap_outliers(X)
        # Note: One-hot encoding is now handled in the pipeline (DataTypeTransformer)
        # X = self.one_hot_encode_sklearn_with_nan(X)

        return X

    def preprocessing_pipeline(self, filename: str):
        """
        Complete preprocessing pipeline that returns processed X and y.
        
        Parameters:
        -----------
        filename : str
            Name of the file to process
            
        Returns:
        --------
        tuple
            Processed X (features) and y (labels) DataFrames
        """
        # Load the data
        
        if self.datatype == DATATYPE.KEPLER:
            X, y = self.define_data(filename)
            # Process features
            X_processed = self.prepare_data(X)
            
            # Process labels
            y_processed = self.process_disposition_labels(y)

            # Check if X_processed and y_processed have the same length
            if len(X_processed) != len(y_processed):
                raise ValueError(f"X_processed and y_processed have different lengths: "
                            f"{len(X_processed)} vs {len(y_processed)}")
            
            # Check for missing values in y_processed and drop them along with corresponding X rows
            if y_processed.isnull().any().any():
                # Get indices of rows with missing values in y_processed
                missing_indices = y_processed.isnull().any(axis=1)
                
                # Drop rows with missing values from both X and y
                X_processed = X_processed[~missing_indices]
                y_processed = y_processed[~missing_indices]
                
                logger.info(f"Dropped {missing_indices.sum()} rows with missing labels")

                    # Drop koi_score column leakage
            X_processed.drop(columns=['koi_score'], inplace=True)
            flag_cols = [col for col in X_processed.columns if "koi_fpflag" in col]

            if flag_cols:
                logger.info(f"Dropping columns with koi_fpflag: {flag_cols}")
                X_processed.drop(columns=flag_cols, inplace=True)
                logger.info(f"Dropped columns with koi_fpflag: {flag_cols}")
            else:
                logger.info("No columns with koi_fpflag found")

        elif self.datatype == DATATYPE.TESS:
            # Process features
            X_processed, y_processed = self.prepare_tess_data(filename)
            
            # Process labels
            #y_processed = self.process_disposition_labels(y)
        elif self.datatype == DATATYPE.K2:
            # Process features
            X_processed = self.prepare_data(X)
            
            # Process labels
            y_processed = self.process_disposition_labels(y)
        else:
            raise ValueError(f"Invalid dataset: {self.datatype}")

        X_train, X_test, y_train, y_test = self.data_split(
            X_processed, y_processed)
        
        return X_train, X_test, y_train, y_test

    def data_split(self, X, y, test_size=0.2):
        return train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=self.random_state,
                    stratify=y
                )

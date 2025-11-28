import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler,MinMaxScaler,RobustScaler,PowerTransformer
from sklearn.impute import SimpleImputer



def load_and_clean_data(path):
    df = pd.read_csv(path)

    cols = [c for c in df.columns if not str(c).startswith("Unnamed")]
    df = df.loc[:, cols].copy()


    numeric_cols = ["Age", "Credit amount", "Duration"]
    for col in numeric_cols:
        if col in df.columns:
            df[col]=pd.to_numeric(df[col],errors="coerce")
        else:
            df[col]=np.nan
    return df



def feature_engineering(df,do_poly=False,poly_degree=2,drop_orig=False):
    df = df.copy()

    for c in ["Age","Credit amount","Duration"]:
        if c not in df.columns:
            df[c]=np.nan

    #basi engineered features:

    df["credit_per_duration"]=df["Credit amount"]/(df["Duration"].replace(0,np.nan))
    df["credit_per_age"]=df["Credit amount"]/(df["Age"].replace(0,np.nan)+1) #avoid div by 0
    df["age_sq"]=df["Age"]**2
    df["duration_sq"]=df["Duration"]**2
    df["credit_log"] = np.log1p(df["Credit amount"].clip(lower=0))
    df["age_credit_mul"]=df["Age"]*df["Credit amount"]
    df["age_duration_mul"]= df["Age"]*df["Duration"]
    df["credit_div_age_plus_dur"]=df["Credit amount"]/(df["Age"].fillna(0)+df["Duration"].fillna(0)+1)
    df["is_high_credit"] = (df["Credit amount"] > df["Credit amount"].quantile(0.75)).astype(int)

    #age bins(categorical )- useful for tree/grouping
    bins=[0,25,35,50,65,120]
    labels=["<25", "25-34", "35-49", "50-64", "65+"]
    df["age_group"]=pd.cut(df["Age"].fillna(-1),bins=bins,labels=labels,include_lowest=True)

    #flag: missing acounts
    if "Saving accounts" in df.columns:
        df["Saving_missing"]=df["Saving accounts"].isna().astype(int)
    if "Checking account" in df.columns:
        df["checking_missing"]=df["Checking account"].isna().astype(int)

    if do_poly:
        # chỉ áp dụng trên 3 cột numeric gốc
        from sklearn.preprocessing import PolynomialFeatures
        poly_cols = ["Age", "Credit amount", "Duration"]
        pf = PolynomialFeatures(degree=poly_degree, include_bias=False, interaction_only=False)
        poly_arr = pf.fit_transform(df[poly_cols].fillna(0))
        poly_names = pf.get_feature_names_out(poly_cols)
        poly_df = pd.DataFrame(poly_arr, columns=poly_names, index=df.index)
        df = pd.concat([df, poly_df], axis=1)
        if drop_orig:
            df = df.drop(columns=poly_cols)
    return df


def get_feature_types(df):
    df=df.copy()
    df=df.loc[:,[c for c in df.columns if not str(c).startswith("Unnamed")]]

    exclude ={"Risk","target"}

    numeric_cols=df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols=[c for c in numeric_cols if c not in exclude]

    categorical_cols=df.select_dtypes(include=["object","category","bool"]).columns.tolist()
    categorical_cols=[c for c in categorical_cols if c not in exclude]

    #also include the newly created, "age_group","duration_group" if present and not detected as numeric
    for special in ["age_group"]:
        if special in df.columns and special not in categorical_cols:
            categorical_cols.append(special)

    return numeric_cols,categorical_cols


def build_preprocessor(numeric_cols,categorical_cols,scaler="standard"):
    scaler_map = {
        "standard": StandardScaler(),
        "robust": RobustScaler(),
        "minmax": MinMaxScaler(),
        "power": PowerTransformer(method="yeo-johnson")
    }
    scaler_obj = scaler_map.get(scaler, StandardScaler())


    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",scaler_obj)
    ])

    categorical_pipeline=Pipeline([
        ("imputer",SimpleImputer(strategy="constant",fill_value="MISSING")),
        ("encoder",OneHotEncoder(handle_unknown="ignore",sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ], remainder="drop")  # drop các cột không được liệt kê

    return preprocessor
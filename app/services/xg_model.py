import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from math import atan2
import os
import requests
MODEL_PATH = "C:\\Users\\Harshita\\Documents\\FootBallProject\\app\\services\\models\\xg_model_statsbomb.pkl"

# === Feature Engineering ===
def calc_features(df):
    goal_x, goal_y = 105, 34  # Mid-goal position on a 105x68 pitch
    dx = goal_x - df['coordinates_x']
    dy = goal_y - df['coordinates_y']
    df['distance'] = np.sqrt(dx**2 + dy**2)
    df['angle'] = np.arctan2(7.32 * df['distance'], (df['distance']**2 - (7.32/2)**2))
    df['angle'] = df['angle'].fillna(0)
    return df


def train_xg_model():
    print("Downloading and training xG model from StatsBomb data...")
    
    import requests
    url = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/8655.json"
    raw = requests.get(url).json()  # Use requests instead of pandas here

    # Normalize list of dicts
    events = pd.json_normalize(raw)

    # Filter for shots only
    sb_shots = events[events['type.name'] == 'Shot'].copy()

    # Coordinate conversion from 120x80 to 105x68
    sb_shots['coordinates_x'] = sb_shots['location'].apply(lambda loc: loc[0] if isinstance(loc, list) else np.nan) * 105 / 120
    sb_shots['coordinates_y'] = sb_shots['location'].apply(lambda loc: loc[1] if isinstance(loc, list) else np.nan) * 68 / 80
    sb_shots['is_goal'] = sb_shots['shot.outcome.name'] == 'Goal'

    # Drop NaNs from bad locations
    sb_shots.dropna(subset=['coordinates_x', 'coordinates_y'], inplace=True)

    # Add features
    sb_shots = calc_features(sb_shots)

    # Train model
    model = LogisticRegression()
    model.fit(sb_shots[['distance', 'angle']], sb_shots['is_goal'])

    # Save model
    joblib.dump(model, MODEL_PATH)
    print("✅ xG model saved to", MODEL_PATH)




# === Load or train model ===
def get_xg_model():
    if not os.path.exists(MODEL_PATH):
        train_xg_model()
    return joblib.load(MODEL_PATH)


# === Predict xG on your shot DataFrame ===
def compute_xg(shots_df, x_col="coordinates_x", y_col="coordinates_y"):
    model = get_xg_model()

    shots = shots_df.copy()
    shots[x_col] = pd.to_numeric(shots[x_col], errors="coerce") * 105
    shots[y_col] = pd.to_numeric(shots[y_col], errors="coerce") * 68
    shots = shots.dropna(subset=[x_col, y_col])
    shots = shots.rename(columns={x_col: "coordinates_x", y_col: "coordinates_y"})
    shots = calc_features(shots)

    shots["xG"] = model.predict_proba(shots[["distance", "angle"]])[:, 1]
    return shots
if __name__ == "__main__":
    train_xg_model()

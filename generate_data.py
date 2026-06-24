"""
Synthetic real estate dataset generator for Nigerian property market.
Generates ~3,000 records aligned with the dataset schema in Table 3.2 (Chapter 3).
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    RAW_DATA_PATH, NEIGHBOURHOOD_COORDS, PROPERTY_TYPE_MULTIPLIERS,
    NEIGHBOURHOODS, PROPERTY_TYPES
)

np.random.seed(42)
N_SAMPLES = 3000
CURRENT_YEAR = 2025


def _property_age_factor(yr_built: int) -> float:
    age = CURRENT_YEAR - yr_built
    if age < 5:
        return 1.08
    elif age < 15:
        return 1.00
    elif age < 30:
        return 0.93
    elif age < 50:
        return 0.86
    return 0.78


def generate_dataset(n: int = N_SAMPLES) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    neighbourhoods = rng.choice(NEIGHBOURHOODS, size=n,
                                p=_neighbourhood_weights())
    property_types = rng.choice(PROPERTY_TYPES, size=n)

    # Structural features – distributions tied to neighbourhood tier
    sqft_living = []
    sqft_lot = []
    bedrooms = []
    bathrooms = []
    floors_list = []
    parking_spaces = []
    yr_built_list = []
    yr_renovated_list = []
    has_pool = []
    is_gated = []
    has_gym = []
    grade_list = []
    condition_list = []
    prices = []

    for i in range(n):
        nb = neighbourhoods[i]
        pt = property_types[i]
        info = NEIGHBOURHOOD_COORDS[nb]
        tier = info['tier']
        base_sqm = info['base_per_sqm']

        # Floor area varies by tier and property type
        if tier == 'ultra_premium':
            sqft = int(rng.normal(350, 80))
        elif tier == 'premium':
            sqft = int(rng.normal(250, 70))
        elif tier == 'mid_high':
            sqft = int(rng.normal(180, 50))
        else:
            sqft = int(rng.normal(120, 40))

        sqft = max(40, min(1500, sqft))

        if pt == 'Apartment':
            sqft = max(40, min(sqft, 300))
        elif pt == 'Bungalow':
            sqft = max(80, min(sqft, 400))

        lot = int(sqft * rng.uniform(1.5, 3.5))
        lot = max(sqft + 20, lot)

        # Rooms
        if tier in ('ultra_premium', 'premium'):
            bed = int(rng.choice([3, 4, 5, 6], p=[0.15, 0.35, 0.35, 0.15]))
            bath = int(rng.choice([2, 3, 4, 5], p=[0.10, 0.30, 0.40, 0.20]))
        elif tier == 'mid_high':
            bed = int(rng.choice([2, 3, 4, 5], p=[0.20, 0.40, 0.30, 0.10]))
            bath = int(rng.choice([2, 3, 4], p=[0.35, 0.45, 0.20]))
        else:
            bed = int(rng.choice([1, 2, 3, 4], p=[0.25, 0.40, 0.25, 0.10]))
            bath = int(rng.choice([1, 2, 3], p=[0.40, 0.45, 0.15]))

        bath = min(bath, bed)
        fl = 1 if pt in ('Apartment', 'Bungalow') else int(rng.choice([1, 2, 3], p=[0.30, 0.55, 0.15]))
        park = int(rng.choice([0, 1, 2, 3, 4], p=[0.10, 0.30, 0.35, 0.20, 0.05]))

        # Year built
        yr_b = int(rng.integers(1980, 2025))
        yr_r = 0
        if rng.random() < 0.25 and (CURRENT_YEAR - yr_b) > 10:
            yr_r = int(rng.integers(yr_b + 5, CURRENT_YEAR))

        # Amenities – more likely in premium tiers
        pool_prob = {'ultra_premium': 0.55, 'premium': 0.30, 'mid_high': 0.10, 'standard': 0.02}[tier]
        gated_prob = {'ultra_premium': 0.90, 'premium': 0.70, 'mid_high': 0.40, 'standard': 0.15}[tier]
        gym_prob = {'ultra_premium': 0.50, 'premium': 0.25, 'mid_high': 0.08, 'standard': 0.02}[tier]

        pool = int(rng.random() < pool_prob)
        gated = int(rng.random() < gated_prob)
        gym = int(rng.random() < gym_prob)

        # Construction grade (1–13) and condition (1–5)
        grade_mean = {'ultra_premium': 10, 'premium': 8, 'mid_high': 6, 'standard': 5}[tier]
        grd = int(np.clip(rng.normal(grade_mean, 1.2), 1, 13))
        cond = int(np.clip(rng.normal(3.5, 0.8), 1, 5))

        # Price model
        pt_mult = PROPERTY_TYPE_MULTIPLIERS[pt]
        age_factor = _property_age_factor(yr_b)
        grade_adj = 1 + 0.04 * (grd - 7)
        cond_adj = 1 + 0.02 * (cond - 3)
        room_adj = 1 + 0.03 * bed + 0.02 * bath + 0.01 * park
        amenity_adj = 1 + 0.06 * pool + 0.04 * gated + 0.03 * gym
        reno_adj = 1.05 if yr_r > 0 else 1.0

        base_price = base_sqm * sqft
        price = (base_price * pt_mult * age_factor * grade_adj
                 * cond_adj * room_adj * amenity_adj * reno_adj)
        noise = rng.normal(0, 0.08)
        price = price * (1 + noise)
        price = max(5_000_000, price)

        sqft_living.append(sqft)
        sqft_lot.append(lot)
        bedrooms.append(bed)
        bathrooms.append(bath)
        floors_list.append(fl)
        parking_spaces.append(park)
        yr_built_list.append(yr_b)
        yr_renovated_list.append(yr_r)
        has_pool.append(pool)
        is_gated.append(gated)
        has_gym.append(gym)
        grade_list.append(grd)
        condition_list.append(cond)
        prices.append(round(price, -4))

    lats = [NEIGHBOURHOOD_COORDS[nb]['lat'] + rng.normal(0, 0.003) for nb in neighbourhoods]
    longs = [NEIGHBOURHOOD_COORDS[nb]['long'] + rng.normal(0, 0.003) for nb in neighbourhoods]

    df = pd.DataFrame({
        'price': prices,
        'neighbourhood': neighbourhoods,
        'property_type': property_types,
        'sqft_living': sqft_living,
        'sqft_lot': sqft_lot,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'floors': floors_list,
        'parking_spaces': parking_spaces,
        'yr_built': yr_built_list,
        'yr_renovated': yr_renovated_list,
        'has_pool': has_pool,
        'is_gated': is_gated,
        'has_gym': has_gym,
        'grade': grade_list,
        'condition': condition_list,
        'lat': lats,
        'long': longs,
    })

    return df


def _neighbourhood_weights():
    weights = {
        'Victoria Island': 0.03,
        'Ikoyi': 0.03,
        'Banana Island': 0.01,
        'Lekki Phase 1': 0.07,
        'Lekki Phase 2': 0.05,
        'Ajah': 0.06,
        'Gbagada': 0.05,
        'Magodo': 0.05,
        'Surulere': 0.06,
        'Yaba': 0.05,
        'Ikeja': 0.06,
        'Ogba': 0.04,
        'Maitama': 0.03,
        'Asokoro': 0.03,
        'Wuse 2': 0.04,
        'Jabi': 0.04,
        'Garki': 0.04,
        'Gwarinpa': 0.05,
        'Kubwa': 0.04,
        'Lugbe': 0.03,
        'GRA Phase 1': 0.04,
        'GRA Phase 2': 0.04,
        'Trans Amadi': 0.04,
        'Rumuola': 0.03,
        'Diobu': 0.03,
        'Eleme': 0.03,
    }
    total = sum(weights.values())
    return [weights[nb] / total for nb in NEIGHBOURHOODS]


if __name__ == '__main__':
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    print("Generating synthetic real estate dataset...")
    df = generate_dataset(N_SAMPLES)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Dataset saved to {RAW_DATA_PATH}")
    print(f"Shape: {df.shape}")
    print(f"\nPrice summary (NGN):")
    print(df['price'].describe().apply(lambda x: f"₦{x:,.0f}"))
    print(f"\nNeighbourhood distribution:")
    print(df['neighbourhood'].value_counts().head(10))

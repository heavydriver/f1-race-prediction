# Features

One row per driver per race (2014–2025). All stats below use only data from **before** that race, except identifiers and the race’s own grid/qualifying.

See `data/driver_race_features.csv`.

## Identifiers

- `row_id` — `{raceId}_{driverId}`
- `raceId`, `year`, `round`, `date`, `race_name`
- `driverId`, `driver_ref`
- `constructorId`, `constructor_ref`
- `circuitId`, `circuit_ref`

## Grid and qualifying

- `grid` — starting grid position (pit lane / missing grid filled from quali when needed)
- `quali_position` — official qualifying result
- `last_quali_session` — last session the driver set a time in: `1` (Q1), `2` (Q2), or `3` (Q3)
- `quali_gap_to_pole_s` — last quali session time (Q3, else Q2, else Q1) minus the fastest time in that same session (seconds, never negative)
- `teammate_quali_gap_s` — this driver’s last-session time minus teammate’s (seconds; negative = faster). Empty if either driver has no quali time.
- `grid_minus_quali` — grid minus quali position (penalties / pit lane)

## Championship and team form

- `driver_champ_pos`, `driver_champ_points`, `driver_champ_wins` — standings after the previous race this season (0 at round 1)
- `constructor_champ_pos`, `constructor_champ_points`, `constructor_champ_wins` — same for the team
- `constructor_avg_finish_l3` — team mean finish over the last 3 races this season
- `races_with_constructor` — career races with this constructor before this race
- `stint_races_with_constructor` — consecutive races in the current stint with this team

## Driver history

- `career_races`, `career_wins`, `career_podiums`, `career_avg_finish`, `career_dnf_rate` — career totals / mean finish / DNF share before this race
- `circuit_races`, `circuit_avg_finish`, `circuit_wins`, `circuit_podiums` — this driver at this circuit before this race
- `avg_finish_l3` — mean finish over the last 3 races this season
- `season_dnf_rate` — DNF share this season before this race (`positionText` is letters)
- `circuit_dnf_rate` — DNF share at this circuit before this race
- `driver_age_years` — age on race day

## Circuit and race setting

- `circuit_type` — Permanent / Street / Hybrid
- `circuit_alt` — altitude (metres)
- `circuit_position_gain_avg` — historical mean of `(grid - finish)` at this circuit (higher = more place-changing)
- `circuit_safety_car_rate` — share of earlier races here with at least one safety car
- `circuit_safety_car_avg` — mean safety-car count per earlier race here
- `circuit_red_flag_rate` — share of earlier races here with at least one red flag
- `race_time_of_day` — `day` / `twilight` / `night` from local start hour

## Weather

Open-Meteo historical hourly values at the circuit (`lat`/`lng`) for the UTC hour of race start. Same values for every driver in a race.

- `air_temp` — 2 m air temperature (°C)
- `wind_speed` — 10 m wind speed (km/h)
- `humidity` — 2 m relative humidity (%)
- `rain` — `1` if that hour had rain (`rain` mm > 0), else `0`. Lights-out wet/dry, not rainfall over the whole race.

## Missing values

Empty is not the same as `0`.

- **`0` on purpose:** championship pos/points/wins at round 1 (and debutants with no prior standing). Counts like `career_races`, `circuit_wins` are also `0` when there is no history. `rain` is `0` when the start hour was dry.
- **Empty on purpose:** means and rates with no sample yet (`avg_finish_l3`, `season_dnf_rate`, `constructor_avg_finish_l3` at round 1; `circuit_avg_finish` / `circuit_dnf_rate` on a first visit; quali columns if there is no time). Do not fill those with `0` — that looks like a win or a perfect quali.
- **Career fallbacks:** `career_avg_finish` and `career_dnf_rate` are filled whenever the driver has raced before, including at round 1. True rookies stay empty.

When training:

1. Split years first (train 2014–2023, val 2024, test 2025).
2. Random Forest: `SimpleImputer(strategy="median")` fit on **train only**, then transform val/test. Trees do not need scaling.
3. XGBoost: can keep NaNs (`missing=np.nan`); it learns a default branch.
4. Optional: add a missing indicator for quali (`quali_position.isna()`) so the model can treat “did not set a time” as its own signal.

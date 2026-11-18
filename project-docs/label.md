# Labels

Stored in the same file as features: `data/driver_race_features.csv`.

Join key: `row_id` (`{raceId}_{driverId}`).

Do not use these columns as model inputs.

- `finish_position` — race finishing order (`positionOrder`). This is **y**.
- `dnf` — 1 if `positionText` is letters (`R`, `W`, `D`, `E`, `N`), else 0. Extra label for analysis, not the main target.

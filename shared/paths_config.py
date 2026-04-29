"""Centralized project paths and backward-compatible aliases."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAPSTONE_ROOT = PROJECT_ROOT.parent.parent.parent
RESEARCH_DATA = CAPSTONE_ROOT / "research_data"

POWER_MONTHLY_CSV = RESEARCH_DATA / "02_grid_operations/electric_power_operations_monthly_all_states.csv"
POWER_2010_2024_CSV = RESEARCH_DATA / "02_grid_operations/electric_power_operations_2010_2024.csv"
POWER_CSV = POWER_MONTHLY_CSV if POWER_MONTHLY_CSV.is_file() else POWER_2010_2024_CSV

SEDS_CSV = RESEARCH_DATA / "08_seds_state_energy/complete_seds_all_energy_1960_2023.csv"
RETAIL_PRICE_CSV = RESEARCH_DATA / "03_electricity_prices/retail_sales_price_revenue_monthly_all_states.csv"

GRID_STRESS_DIR = RESEARCH_DATA / "07_grid_stress"
CAISO_FUEL_MIX_CSV = GRID_STRESS_DIR / "caiso_fuel_mix.csv"
ERCOT_FUEL_MIX_HOURLY_CSV = GRID_STRESS_DIR / "erco_fuel_mix_hourly.csv"
ISONE_FUEL_MIX_CSV = GRID_STRESS_DIR / "isone_fuel_mix.csv"
NYISO_FUEL_MIX_CSV = GRID_STRESS_DIR / "nyiso_fuel_mix.csv"
PJM_FUEL_MIX_HOURLY_CSV = GRID_STRESS_DIR / "pjm_fuel_mix_hourly.csv"

ERCO_FUEL_MIX_HOURLY_CSV = ERCOT_FUEL_MIX_HOURLY_CSV

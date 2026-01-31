# Data Sources Reference

## Kazakhstan Bureau of National Statistics (BNS)

### Base URL
`https://stat.gov.kz`

### CPI Categories (COICOP)
The Consumer Price Index is disaggregated by COICOP (Classification of Individual Consumption According to Purpose) divisions:

| Code | Category | Tradable | Admin Price |
|------|----------|----------|-------------|
| 01 | Food and non-alcoholic beverages | Yes | No |
| 02 | Alcoholic beverages, tobacco | Yes | No |
| 03 | Clothing and footwear | Yes | No |
| 04 | Housing, water, electricity, gas | No | **Yes** |
| 05 | Furnishings, household equipment | Yes | No |
| 06 | Health | No | **Yes** |
| 07 | Transport | Yes | No |
| 08 | Communications | No | **Yes** |
| 09 | Recreation and culture | Yes | No |
| 10 | Education | No | **Yes** |
| 11 | Restaurants and hotels | No | No |
| 12 | Miscellaneous goods and services | Yes | No |

**Admin price categories** (04, 06, 08, 10) are regulated by the government and should NOT show significant FX pass-through. This is used as a falsification test.

### API Endpoints
- CSV data: `https://stat.gov.kz/api/iblock/element/{iblock_id}/csv/file/en/`
- Known iblock IDs:
  - 48953: Per capita monetary income (quarterly)
  - 48510: Per capita monetary income (annual)
  - 469805: Household expenditure structure
  - 49140: Consumer Price Index statistics

## National Bank of Kazakhstan (NBK)

### Base URL
`https://nationalbank.kz`

### Exchange Rate Endpoints
- RSS feed: `https://nationalbank.kz/rss/rates_all.xml`
- JSON API: `https://nationalbank.kz/ru/api/exchangerates/rates`
- Historical: `https://nationalbank.kz/rss/get_rates.cfm?fdate=YYYY.MM.DD`

### Response Format (JSON API)
```json
{
  "rates": [
    {"date": "2024-01-15", "value": 454.23},
    ...
  ]
}
```

## Federal Reserve Economic Data (FRED)

### Required Series
| Series ID | Description | Frequency |
|-----------|-------------|-----------|
| IGREA | Kilian Global Real Economic Activity Index | Monthly |
| VIXCLS | CBOE Volatility Index | Daily |
| DCOILBRENTEU | Brent Crude Oil Price | Daily |

### API
Requires FRED API key set in `.env` as `FRED_API_KEY`.

## Baumeister-Hamilton Oil Shocks

### Description
Structural oil supply shocks from Baumeister & Hamilton (2019, AER).

### Data Source
Original data hosted on Google Drive, downloaded via the BaumeisterLoader class.

### Shock Types
- Supply shocks (exogenous disruptions)
- Demand shocks (global activity driven)

## World Bank

### REER Indicator
- Code: `PX.REX.REER`
- Description: Real Effective Exchange Rate Index
- Frequency: Annual
- Note: Use for robustness checks only (annual frequency limits utility)

### API
```
https://api.worldbank.org/v2/country/KZ/indicator/PX.REX.REER?format=json
```

## Data Quality Indicators

### Real vs Synthetic Data

**Real CPI data characteristics:**
- Heterogeneous volatility across categories
- Food (01) typically more volatile than services
- Admin prices (04, 06, 08, 10) show low/no FX correlation
- Seasonal patterns in food categories

**Synthetic data red flags:**
- Uniform volatility across all categories
- Similar means and standard deviations
- Admin prices show same pass-through as tradables
- No seasonal variation

### Minimum Data Requirements

For FX pass-through analysis:
- CPI categories: At least 60 monthly observations (5 years)
- Exchange rates: Daily data matching CPI period
- Oil prices: Monthly, matching analysis period

## File Structure

```
data/
├── raw/
│   ├── fred/
│   │   ├── IGREA.parquet
│   │   ├── VIXCLS.parquet
│   │   └── DCOILBRENTEU.parquet
│   ├── baumeister_shocks/
│   │   └── shocks.parquet
│   ├── nbk/
│   │   └── usd_kzt.parquet
│   ├── kazakhstan_bns/
│   │   ├── cpi_categories.parquet
│   │   └── national_income.parquet
│   └── worldbank/
│       └── reer_kz.parquet
├── processed/
│   └── fx_passthrough/
│       ├── cpi_panel.parquet      # Built from raw data
│       ├── income_series.parquet  # Built from raw data
│       └── expenditure_series.parquet
└── metadata/
    └── download_status.json       # Tracks all downloads
```

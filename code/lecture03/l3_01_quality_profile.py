"""l3_01_quality_profile.py -- build a deliberately dirty sample and profile its quality.

The script manufactures an order table that contains the classic defects (mixed
date formats, duplicated keys, inconsistent category spellings, currency symbols,
missing values, impossible values) and then measures the five quality dimensions
used in the lecture: validity, accuracy, completeness, consistency, uniformity.
"""
import numpy as np                                                 # random draws and numeric helpers
import pandas as pd                                                # table construction and profiling
import matplotlib.pyplot as plt                                    # four-panel quality dashboard
from _common_l3 import RNG, save, dump                             # shared seed, figure and data writers

N = 600                                                            # number of order rows to synthesise
CITIES = ["Shenzhen", "shenzhen", "SHENZHEN ", " Shenzhen",        # the same city written five ways
          "Guangzhou", "Guangzhou City", "Guang Zhou"]             # more spelling variants on purpose

order_id = [f"SO{2026000 + i}" for i in range(1, N + 1)]           # clean sequential order ids
for i in RNG.choice(range(1, N), size=18, replace=False):          # pick 18 rows to duplicate later
    order_id[i] = order_id[i - 1]                                  # copy the previous id: duplicate key
dates = []                                                         # dates will arrive in three formats
for i in range(N):                                                 # build one raw date string per row
    day = pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(RNG.integers(0, 120)))   # random day
    fmt = RNG.integers(0, 3)                                       # choose one of three source formats
    dates.append(day.strftime("%Y-%m-%d") if fmt == 0 else         # ISO format, e.g. 2026-02-11
                 day.strftime("%d/%m/%Y") if fmt == 1 else         # day-first format, e.g. 11/02/2026
                 day.strftime("%Y%m%d"))                           # compact format, e.g. 20260211
city = [str(RNG.choice(CITIES)) for _ in range(N)]                 # city column with spelling variants
amount_raw = []                                                    # amounts as raw strings (never clean)
for i in range(N):                                                 # build each raw amount string
    value = float(RNG.lognormal(mean=6.2, sigma=0.7))              # log-normal spend: right skewed
    style = RNG.integers(0, 4)                                     # four different source conventions
    if style == 0:                                                 # style 1: yuan symbol + thousands
        amount_raw.append(f"¥{value:,.2f}")                        # e.g. ¥1,234.56
    elif style == 1:                                               # style 2: currency code suffix
        amount_raw.append(f"{value:.2f} RMB")                      # e.g. 1234.56 RMB
    elif style == 2:                                               # style 3: thousands in CNY
        amount_raw.append(f"{value / 1000:,.1f} thousand CNY")     # mixed unit: thousand yuan
    else:                                                          # style 4: plain number with spaces
        amount_raw.append(f" {value:,.2f} ")                       # leading/trailing spaces
qty = RNG.integers(1, 12, N).astype(float)                         # quantity, stored as float first
qty[RNG.choice(N, size=70, replace=False)] = np.nan                # 70 rows lose the quantity
amount_raw[3] = "-500.00"                                          # plant one impossible negative amount
amount_raw[7] = "0"                                                # plant one impossible zero amount
payment = RNG.choice(["paid", "Paid", "PAID", "unpaid", "N/A",      # payment column with synonyms
                      "Not Applicable", "-"], size=N)              # three spellings of "not applicable"

raw = pd.DataFrame({"order_id": order_id,                          # duplicated business key
                    "date_raw": dates,                             # mixed date formats
                    "city_raw": city,                              # inconsistent categories
                    "amount_raw": amount_raw,                      # currency noise and mixed units
                    "qty": qty,                                    # missing quantities
                    "payment_raw": payment})                       # synonym problem
dump(raw, "l3_orders_dirty.csv")                                   # write the dirty sample for later scripts

# ---------------------------------------------------------------- profiling: five quality dimensions
parsed_date = pd.to_datetime(raw["date_raw"], format="mixed", dayfirst=True, errors="coerce")   # best effort
amount = (raw["amount_raw"].str.replace(r"[^\d.\-]", "", regex=True)   # strip symbols for a quick check
          .pipe(pd.to_numeric, errors="coerce"))                       # coerce leftovers to NaN
dup_rows = int(raw.duplicated().sum())                             # fully duplicated rows (none planted)
dup_keys = int(raw["order_id"].duplicated().sum())                 # duplicated business keys (18 planted)
missing_rate = raw.isna().mean().mul(100).round(2)                 # completeness: share missing per column
bad_amount = int(((amount <= 0) | (amount > 100000)).sum())        # accuracy: impossible magnitudes
city_raw_n = raw["city_raw"].nunique()                             # consistency: variants seen in the raw column
city_clean_n = raw["city_raw"].str.strip().str.title().nunique()   # consistency: variants after normalisation
pay_raw_n = raw["payment_raw"].nunique()                           # uniformity: spellings of the same state
pay_clean_n = (raw["payment_raw"].str.strip().str.lower()          # normalise case and padding
               .replace({"n/a": "not_applicable", "-": "not_applicable",   # synonyms of "not applicable"
                         "not applicable": "not_applicable"}).nunique())   # then count again
valid_date_rate = 100 * parsed_date.notna().mean()                 # validity: share of parsable dates
valid_id_rate = 100 * raw["order_id"].str.match(r"^SO\d{7}$").mean()   # validity: id pattern compliance
score = {                                                          # scorecard: 0-100 per dimension
    "Validity": round(0.5 * valid_date_rate + 0.5 * valid_id_rate, 1),     # dates and ids both count
    "Accuracy": round(100 - 100 * bad_amount / N, 1),              # share of plausible amounts
    "Completeness": round(100 - float(missing_rate.mean()), 1),    # share of cells that are present
    "Consistency": round(100 * city_clean_n / max(city_raw_n, 1), 1),      # fewer variants after cleaning
    "Uniqueness": round(100 * (N - dup_keys) / N, 1),              # share of unique business keys
}                                                                  # end of the scorecard dictionary
report = pd.DataFrame({"dimension": list(score), "score": list(score.values())})   # tidy scorecard
dump(report, "l3_quality_scorecard.csv")                           # publish the scorecard as a sample
print(report.to_string(index=False))                               # show it in the console as well

# ---------------------------------------------------------------- figure: quality dashboard
fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.4))                 # one row: four panels for a 16:9 slide
mr = missing_rate.sort_values(ascending=False)                     # worst columns first
axes[0].bar(mr.index, mr.values, color="#8A0C3C", alpha=0.85)       # completeness panel
axes[0].set_title("Completeness:\nmissing rate (%)", fontsize=11)   # panel title
axes[0].tick_params(axis="x", rotation=45, labelsize=8)             # rotate labels to avoid overlap
axes[1].bar(["full-row\nduplicates", "duplicate\nkeys"], [dup_rows, dup_keys],   # two duplicate counts
            color=["#66717D", "#8A0C3C"])                           # uniqueness panel
axes[1].set_title("Uniqueness:\nduplicated records", fontsize=11)   # panel title
axes[1].tick_params(axis="x", labelsize=8)                          # compact tick labels
axes[2].bar(["raw", "normalised"], [city_raw_n, city_clean_n],      # variants before and after mapping
            color=["#C9A227", "#711426"])                           # consistency panel
axes[2].set_title("Consistency:\ncity spellings", fontsize=11)      # panel title
axes[2].tick_params(axis="x", labelsize=9)                          # tick labels
axes[3].barh(list(score)[::-1], list(score.values())[::-1], color="#8A0C3C", alpha=0.85)   # scorecard
axes[3].set_xlim(0, 100)                                            # scores are percentages
axes[3].tick_params(axis="y", labelsize=8.5)                        # dimension names
axes[3].set_title("Quality scorecard\n(0-100)", fontsize=11)        # panel title
plt.tight_layout()                                                 # balance the four panels
save(fig, "fig_quality_profile.png")                               # write the dashboard for the deck
print("payment spellings raw -> clean:", pay_raw_n, "->", pay_clean_n)   # uniformity evidence in the log
print("valid dates %:", round(valid_date_rate, 1), "| implausible amounts:", bad_amount)   # validity log

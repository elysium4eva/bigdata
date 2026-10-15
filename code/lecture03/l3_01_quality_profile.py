"""l3_01_quality_profile.py —— build a "dirty data" table and score its five quality dimensions.

The script deliberately creates the most common defects of an order table (mixed date formats, duplicate primary keys, inconsistent category spellings,
currency symbols inside amounts, missing values, impossible magnitudes), then quantifies them item by item through the five quality dimensions
from the slides (validity, accuracy, completeness, consistency, unit consistency).
"""
import numpy as np                                                 # random sampling and numeric work
import pandas as pd                                                # build tables and quality statistics
import matplotlib.pyplot as plt                                    # draw the four-panel quality dashboard
from _common_l3 import RNG, save, dump                             # reuse the shared seed, plot and dump helpers

N = 600                                                            # number of synthetic order rows
CITIES = ["Shenzhen", "shenzhen", "SHENZHEN ", " Shenzhen",        # the same city written five ways
          "Guangzhou", "Guangzhou City", "Guang Zhou"]             # a few more spellings for inconsistency
order_id = [f"SO{2026000 + i}" for i in range(1, N + 1)]           # start with a clean order id column
for i in RNG.choice(range(1, N), size=18, replace=False):          # pick 18 rows to duplicate keys
    order_id[i] = order_id[i - 1]                                  # copy the previous id to create duplicates
dates = []                                                         # dates arrive in three mixed formats
for i in range(N):                                                 # build raw date strings row by row
    day = pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(RNG.integers(0, 120)))   # pick a random day
    fmt = RNG.integers(0, 3)                                       # pick one source format at random
    dates.append(day.strftime("%Y-%m-%d") if fmt == 0 else         # format one: 2026-02-11
                 day.strftime("%d/%m/%Y") if fmt == 1 else         # format two: day/month/year
                 day.strftime("%Y%m%d"))                           # format three: compact form
city = [str(RNG.choice(CITIES)) for _ in range(N)]                 # city column keeps all spelling variants
amount_raw = []                                                    # amounts stored as raw strings
for i in range(N):                                                 # build raw amount strings row by row
    value = float(RNG.lognormal(mean=6.2, sigma=0.7))              # lognormal spend: right skewed
    style = RNG.integers(0, 4)                                     # four source conventions
    if style == 0:                                                 # style one: currency symbol + thousands
        amount_raw.append(f"¥{value:,.2f}")                        # e.g. ¥1,234.56
    elif style == 1:                                               # style two: currency code suffix
        amount_raw.append(f"{value:.2f} RMB")                      # e.g. 1234.56 RMB
    elif style == 2:                                               # style three: thousands CNY (unit mismatch)
        amount_raw.append(f"{value / 1000:,.1f} thousand CNY")     # e.g. 1.2 thousand CNY
    else:                                                          # style four: plain number with spaces
        amount_raw.append(f" {value:,.2f} ")                       # leading and trailing spaces
qty = RNG.integers(1, 12, N).astype(float)                         # quantity column starts as float
qty[RNG.choice(N, size=70, replace=False)] = np.nan                # 70 rows lose their quantity
amount_raw[3] = "-500.00"                                          # plant an impossible negative amount
amount_raw[7] = "0"                                                # plant an impossible zero amount
payment = RNG.choice(["paid", "Paid", "PAID", "unpaid", "N/A",      # payment status has synonyms
                      "Not Applicable", "-"], size=N)              # "not applicable" written three ways
raw = pd.DataFrame({"order_id": order_id,                          # order id (with duplicate keys)
                    "date_raw": dates,                             # date (mixed formats)
                    "city_raw": city,                              # city (inconsistent spellings)
                    "amount_raw": amount_raw,                      # amount (symbol and unit noise)
                    "qty": qty,                                    # quantity (with missing values)
                    "payment_raw": payment})                       # payment status (synonyms)
dump(raw, "l3_orders_dirty.csv")                                   # save for later scripts to read

# ---------------------------------------------------------------- quality profile: scoring five dimensions
parsed_date = pd.to_datetime(raw["date_raw"], format="mixed", dayfirst=True, errors="coerce")   # best-effort parsing
amount = (raw["amount_raw"].str.replace(r"[^\d.\-]", "", regex=True)   # strip symbols for a quick check
          .pipe(pd.to_numeric, errors="coerce"))                       # unparsable values become NaN
dup_rows = int(raw.duplicated().sum())                             # fully duplicated rows (0 here)
dup_keys = int(raw["order_id"].duplicated().sum())                 # duplicated business keys (18 planted)
missing_rate = raw.isna().mean().mul(100).round(2)                 # completeness: missing rate per column
bad_amount = int(((amount <= 0) | (amount > 100000)).sum())        # accuracy: impossible magnitudes
city_raw_n = raw["city_raw"].nunique()                             # consistency: raw variant count
city_clean_n = raw["city_raw"].str.strip().str.title().nunique()   # consistency: variants after normalising
pay_raw_n = raw["payment_raw"].nunique()                           # unit consistency: spelling count
pay_clean_n = (raw["payment_raw"].str.strip().str.lower()          # lowercase and trim first
               .replace({"n/a": "not_applicable", "-": "not_applicable",   # the "not applicable" synonyms
                         "not applicable": "not_applicable"}).nunique())   # merge them into one value
valid_date_rate = 100 * parsed_date.notna().mean()                 # validity: share of parsable dates
valid_id_rate = 100 * raw["order_id"].str.match(r"^SO\d{7}$").mean()   # validity: order ids matching format
score = {                                                          # quality scorecard: each item 0-100
    "Validity": round(0.5 * valid_date_rate + 0.5 * valid_id_rate, 1),   # dates and ids weigh half each
    "Accuracy": round(100 - 100 * bad_amount / N, 1),              # share of plausible amounts
    "Completeness": round(100 - float(missing_rate.mean()), 1),    # share of populated cells
    "Consistency": round(100 * city_clean_n / max(city_raw_n, 1), 1),    # fewer variants, higher score
    "Uniqueness": round(100 * (N - dup_keys) / N, 1),              # share of unique business keys
}                                                                  # end of the scorecard dictionary
report = pd.DataFrame({"dimension": list(score), "score": list(score.values())})   # tidy it into a table
dump(report, "l3_quality_scorecard.csv")                           # save the scorecard
print(report.to_string(index=False))                               # also print it in the terminal

# ---------------------------------------------------------------- figure: quality dashboard
fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.4))                 # four panels in a row for 16:9 slides
mr = missing_rate.sort_values(ascending=False)                     # worst-missing columns first
axes[0].bar(mr.index, mr.values, color="#8A0C3C", alpha=0.85)       # panel one: completeness
axes[0].set_title("Completeness:\nmissing rate (%)", fontsize=11)   # subplot title
axes[0].tick_params(axis="x", rotation=45, labelsize=8)             # rotate x labels to avoid overlap
axes[1].bar(["full-row\nduplicates", "duplicate\nkeys"], [dup_rows, dup_keys],   # two duplicate counts
            color=["#66717D", "#8A0C3C"])                           # panel two: uniqueness
axes[1].set_title("Uniqueness:\nduplicated records", fontsize=11)   # subplot title
axes[1].tick_params(axis="x", labelsize=8)                          # smaller x tick labels
axes[2].bar(["raw", "normalised"], [city_raw_n, city_clean_n],      # variants before and after
            color=["#C9A227", "#711426"])                           # panel three: consistency
axes[2].set_title("Consistency:\ncity spellings", fontsize=11)      # subplot title
axes[2].tick_params(axis="x", labelsize=9)                          # x tick label size
axes[3].barh(list(score)[::-1], list(score.values())[::-1], color="#8A0C3C", alpha=0.85)   # panel four: scorecard
axes[3].set_xlim(0, 100)                                            # scores on a 0-100 scale
axes[3].tick_params(axis="y", labelsize=8.5)                        # dimension label size
axes[3].set_title("Quality scorecard\n(0-100)", fontsize=11)        # subplot title
plt.tight_layout()                                                 # adjust spacing between panels
save(fig, "fig_quality_profile.png")                               # save for the slides
print("payment spellings raw -> clean:", pay_raw_n, "->", pay_clean_n)   # log: evidence of unit consistency
print("valid dates %:", round(valid_date_rate, 1), "| implausible amounts:", bad_amount)   # log: validity

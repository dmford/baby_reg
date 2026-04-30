# ==================================================
# Project: baby_reg.py
# Description: Regression + Visualization (tips dataset)
# Author: David Ford
# Date: 2026-04-24
# ==================================================


# ==================================================
# 0a. IMPORTS 
# ==================================================

# Standard libraries
import os
from pathlib import Path

# Data and analysis
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

# Tables and graphs
import matplotlib.pyplot as plt
from statsmodels.iolib.summary2 import summary_col


# ==================================================
# 0b. ENVIRONMENT SETUP 
# ==================================================
def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

clear_terminal()


# ==================================================
# 0c. PARAMETERS + FILE PATH SETUP
# ==================================================
script_path = Path(__file__).resolve()
script_dir = script_path.parent
script_name = script_path.stem

# Output directories
tables_dir = script_dir / "tables"
graphs_dir = script_dir / "graphs"

# creating output directories if they don't exist
tables_dir.mkdir(exist_ok=True)
graphs_dir.mkdir(exist_ok=True)

# Output counters for table/graph naming (start at 1)
table_counter = 1
graph_counter = 1

def save_table(content, name_prefix, table_counter, tables_dir):
    filename = tables_dir / f"{name_prefix}_table{table_counter}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Saved table: {filename}")
    return table_counter + 1


def save_graph(fig, name_prefix, graph_counter, graphs_dir):
    filename = graphs_dir / f"{name_prefix}_graph{graph_counter}.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved graph: {filename}")
    return graph_counter + 1


# ==================================================
# 1. DATA LOADING + EXPLORATION 
# ==================================================
# --- Load dataset ---
df = sns.load_dataset("tips")

# --- Quick inspection ---
print("\n--- DATA PREVIEW: FIRST 10 ROWS ---")
print(df.head(10))

print("\n--- DATA PREVIEW: LAST 10 ROWS ---")
print(df.tail(10))

print("\n--- DESCRIPTIVE STATISTICS ---")
print(df.describe())


# ==================================================
# 2. FEATURE ENGINEERING 
# ==================================================

# Create tip-share variables.
# tip_share is the decimal version: 0.15 = 15%.
# tip_pct is the percentage version: 15 = 15%.
df["tip_share"] = df["tip"] / df["total_bill"]
df["tip_pct"] = 100 * df["tip_share"]

# Basic sanity check.
print("\n--- TIP PERCENT CHECK ---")
print(f"Max tip %:  {df['tip_pct'].max():.2f}")
print(f"Min tip %:  {df['tip_pct'].min():.2f}")
print(f"Mean tip %: {df['tip_pct'].mean():.2f}")


# ==================================================
# 3. REGRESSION 
# ==================================================

print("\n--- RUNNING REGRESSIONS ---")

m1 = smf.ols("tip ~ total_bill", data=df).fit()
m2 = smf.ols("tip ~ total_bill + size", data=df).fit()
m3 = smf.ols("tip ~ total_bill + C(sex)", data=df).fit()
m4 = smf.ols("tip ~ total_bill + C(smoker)", data=df).fit()
m5 = smf.ols("tip ~ total_bill + C(day)", data=df).fit()
m6 = smf.ols("tip ~ total_bill + C(time)", data=df).fit()
m7 = smf.ols(
    "tip ~ total_bill + size + C(sex) + C(smoker) + C(day) + C(time)",
    data=df
).fit()

# --- Compile regression table ---
print("\n --- REGRESSION TABLE ---")
results_table = summary_col(
    [m1, m2, m3, m4, m5, m6, m7], 
    model_names=[
        "Base", 
        "Size", 
        "Gender", 
        "Smoker", 
        "Day", 
        "Time", 
        "Full"
    ], 
    stars=True
)

print(results_table)

# --- Save regression table ---
table_counter = save_table(
    results_table.as_text(),
    script_name,
    table_counter,
    tables_dir
)



# ==================================================
# 4. VISUALIZATION 
# ==================================================

# This section creates six simple visual checks:
#   G1: Bill size vs tip percentage
#   G2: Party size vs tip percentage
#   G3: Gender-specific fitted lines
#   G4: Smoker-specific fitted lines
#   G5: Day-of-week fitted lines
#   G6: Actual vs predicted tips from the full model

# --- Initialize 2x3 plotting grid ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

# --- G1: Baseline relationship ---
# Model: tip_pct ~ total_bill
# Purpose: visualize relationship between bill size and tip %
x = np.linspace(df["total_bill"].min(), df["total_bill"].max(), 100)

g1_model = smf.ols("tip_pct ~ total_bill", data=df).fit()

# prediction grid for smooth regression line
pred = pd.DataFrame({"total_bill": x})

# scatter + fitted line
axes[0].scatter(df["total_bill"], df["tip_pct"], alpha=0.3)
axes[0].plot(x, g1_model.predict(pred), color="red")

# giving the graph a title + axis labels 
axes[0].set_title("G1: Bill vs Tip %")
axes[0].set_xlabel("Total Bill")
axes[0].set_ylabel("Tip %")

# --- G2: Party Size Splits ---
# Model: tip_pct ~ size
# Purpose: visualize relationship between party size and tip %
x2 = np.linspace(df["size"].min(), df["size"].max(), 100)

g2_model = smf.ols("tip_pct ~ size", data=df).fit()

# prediction grid for smooth regression line
pred2 = pd.DataFrame({"size": x2})

# scatter + fitted line
axes[1].scatter(df["size"], df["tip_pct"], alpha=0.3)
axes[1].plot(x2, g2_model.predict(pred2), color="red")

# giving the graph a title + axis labels 
axes[1].set_title("G2: Size vs Tip %")
axes[1].set_xlabel("Party Size") 
axes[1].set_ylabel("Tip %")

# --- G3: Gender split ---
# Model: tip ~ total_bill (separate regressions by gender)
# Purpose: compare tipping behavior across genders
male = df[df["sex"] == "Male"]
female = df[df["sex"] == "Female"]

m_male = smf.ols("tip_pct ~ total_bill", data=male).fit()
m_female = smf.ols("tip_pct ~ total_bill", data=female).fit()

pred = pd.DataFrame({"total_bill": x})

# scatter + fitted line
axes[2].scatter(df["total_bill"], df["tip_pct"], alpha=0.3)

axes[2].plot(x, m_male.predict(pred), label="Male")
axes[2].plot(x, m_female.predict(pred), label="Female")

# giving the graph a legend + title + axis labels 
axes[2].legend()
axes[2].set_title("G3: Gender Split")
axes[2].set_xlabel("Total Bill")
axes[2].set_ylabel("Tip %")

# --- G4: Smoker split ---
# Model: tip ~ total_bill (separate regressions by smoking status)
# Purpose: compare tipping behavior for smokers vs non-smokers
yes = df[df["smoker"] == "Yes"]
no = df[df["smoker"] == "No"]

m_yes = smf.ols("tip_pct ~ total_bill", data=yes).fit()
m_no = smf.ols("tip_pct ~ total_bill", data=no).fit()

# scatter + fitted line
axes[3].scatter(df["total_bill"], df["tip_pct"], alpha=0.3)

axes[3].plot(x, m_yes.predict(pred), label="Smoker")
axes[3].plot(x, m_no.predict(pred), label="Non-Smoker")

# giving the graph a legend + title + axis labels 
axes[3].legend()
axes[3].set_title("G4: Smoker Split")
axes[3].set_xlabel("Total Bill")
axes[3].set_ylabel("Tip %")

# --- G5: Day-of-week split ---
# Model: tip ~ total_bill (separate regressions by day)
# Purpose: compare tipping patterns across days of the week
for day in df["day"].unique(): 
    sub = df[df["day"] == day]
    m = smf.ols("tip_pct ~ total_bill", data=sub).fit()
    axes[4].plot(x, m.predict(pred), label=day)

# scatter + fitted line
axes[4].scatter(df["total_bill"], df["tip_pct"], alpha=0.3)

# giving the graph a legend + title + axis labels 
axes[4].legend()
axes[4].set_title("G5: Day of Week Split")
axes[4].set_xlabel("Total Bill")
axes[4].set_ylabel("Tip %")

# --- G6: Kitchen sink model fit ---
# Model: predicted tip vs actual tip (from full regression m7)
# Purpose: assess overall model fit (perfect fit = 45-degree line)
pred_full = m7.predict(df)

# scatter: actual vs predicted
# 45-degree line = perfect predictions
axes[5].scatter(df["tip"], pred_full, alpha=0.4)

axes[5].plot(
    [df["tip"].min(), df["tip"].max()],
    [df["tip"].min(), df["tip"].max()],
    color="red"
)

# giving the graph a title + axis labels 
axes[5].set_title("G6: Kitchen Sink Fit")
axes[5].set_xlabel("Actual Tip ($)")
axes[5].set_ylabel("Predicted Tip ($)")

# --- Finalize and save figure ---
plt.tight_layout()
graph_counter = save_graph(
    fig,
    script_name,
    graph_counter,
    graphs_dir
)

# display figure
plt.show()

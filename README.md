# baby_reg

## Project Overview

`baby_reg.py` is an introductory regression + visualization project that uses the classic Seaborn `tips` dataset to demonstrate core empirical workflow in Python.

This script walks through:
- Data loading and inspection
- Feature engineering
- Multiple OLS regression specifications
- Regression table generation
- Automated table/graph saving
- Visualization of key relationships

The project is designed as a practical foundation for learning reproducible regression analysis and organized script structure. :contentReference[oaicite:0]{index=0}

---

# Main Research Question

**What factors explain tipping behavior, and how do variables like bill size, party size, gender, smoking status, day, and time affect tip outcomes?**

---

# Dataset

## Source:
- Seaborn built-in `tips` dataset

## Key Variables:
- `total_bill`
- `tip`
- `sex`
- `smoker`
- `day`
- `time`
- `size`

---

# Feature Engineering

The script creates:

## `tip_share`
```text
tip / total_bill

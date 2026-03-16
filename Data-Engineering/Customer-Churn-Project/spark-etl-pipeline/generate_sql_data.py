"""
generate_sql_data.py
Creates a SQLite database with 20 synthetic Telco Customer Churn records.
These records mirror the CSV schema exactly so they can be unioned later.

Place this file in the project root (spark-etl-pipeline/) and run:
    python generate_sql_data.py
"""

import sqlite3
import random
import string
import os

# Resolve DB path relative to this script's location (project root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "data", "raw", "supplementary_data.db")
TABLE_NAME = "supplementary_records"


def random_customer_id():
    """Generate a Telco-style customer ID like '1234-ABCDE'."""
    nums = "".join(random.choices(string.digits, k=4))
    chars = "".join(random.choices(string.ascii_uppercase, k=5))
    return f"{nums}-{chars}"


def generate_row():
    """
    Generate a single realistic synthetic row matching the CSV schema.
    Values are drawn from the actual distributions in the Telco dataset.
    """
    gender = random.choice(["Male", "Female"])
    senior = random.choices([0, 1], weights=[84, 16])[0]
    partner = random.choice(["Yes", "No"])
    dependents = random.choice(["Yes", "No"])
    tenure = random.randint(1, 72)
    phone_service = random.choices(["Yes", "No"], weights=[90, 10])[0]

    if phone_service == "No":
        multiple_lines = "No phone service"
    else:
        multiple_lines = random.choice(["Yes", "No"])

    internet = random.choices(
        ["DSL", "Fiber optic", "No"],
        weights=[34, 44, 22]
    )[0]

    def internet_dependent():
        if internet == "No":
            return "No internet service"
        return random.choice(["Yes", "No"])

    online_security = internet_dependent()
    online_backup = internet_dependent()
    device_protection = internet_dependent()
    tech_support = internet_dependent()
    streaming_tv = internet_dependent()
    streaming_movies = internet_dependent()

    contract = random.choices(
        ["Month-to-month", "One year", "Two year"],
        weights=[55, 21, 24]
    )[0]
    paperless = random.choice(["Yes", "No"])
    payment = random.choice([
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])

    if internet == "No":
        monthly = round(random.uniform(18.0, 30.0), 2)
    elif internet == "DSL":
        monthly = round(random.uniform(25.0, 70.0), 2)
    else:
        monthly = round(random.uniform(60.0, 110.0), 2)

    if tenure == 0:
        total = " "
    else:
        total = str(round(monthly * tenure * random.uniform(0.85, 1.05), 2))

    churn_weight = 0.25
    if contract == "Month-to-month":
        churn_weight += 0.20
    if internet == "Fiber optic":
        churn_weight += 0.10
    if tenure < 12:
        churn_weight += 0.15
    churn = "Yes" if random.random() < churn_weight else "No"

    return (
        random_customer_id(), gender, senior, partner, dependents,
        tenure, phone_service, multiple_lines, internet,
        online_security, online_backup, device_protection,
        tech_support, streaming_tv, streaming_movies,
        contract, paperless, payment,
        monthly, total, churn
    )


def main():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"[GENERATE] Removed existing {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(f"""
        CREATE TABLE {TABLE_NAME} (
            customerID       TEXT,
            gender           TEXT,
            SeniorCitizen    INTEGER,
            Partner          TEXT,
            Dependents       TEXT,
            tenure           INTEGER,
            PhoneService     TEXT,
            MultipleLines    TEXT,
            InternetService  TEXT,
            OnlineSecurity   TEXT,
            OnlineBackup     TEXT,
            DeviceProtection TEXT,
            TechSupport      TEXT,
            StreamingTV      TEXT,
            StreamingMovies  TEXT,
            Contract         TEXT,
            PaperlessBilling TEXT,
            PaymentMethod    TEXT,
            MonthlyCharges   REAL,
            TotalCharges     TEXT,
            Churn            TEXT
        )
    """)

    rows = [generate_row() for _ in range(20)]
    cursor.executemany(
        f"INSERT INTO {TABLE_NAME} VALUES ({','.join(['?'] * 21)})",
        rows
    )

    conn.commit()

    cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
    count = cursor.fetchone()[0]
    print(f"[GENERATE] Created {DB_PATH}")
    print(f"[GENERATE] Inserted {count} rows into '{TABLE_NAME}'")

    cursor.execute(f"SELECT * FROM {TABLE_NAME} LIMIT 2")
    for row in cursor.fetchall():
        print(f"[GENERATE] Sample: {row}")

    conn.close()


if __name__ == "__main__":
    main()
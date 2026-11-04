import sqlite3

import numpy as np


def create_db(db_path: str = "demo-db.sqlite") -> None:
    """
    Create a fresh demo SQLite database with tables for users, claims,
    sessions, messages, and knowledge base docs.

    Existing tables are dropped for reproducibility in demo runs.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Drop tables if they already exist
    cursor.executescript("""
    DROP TABLE IF EXISTS session;
    DROP TABLE IF EXISTS claim;
    DROP TABLE IF EXISTS message;
    DROP TABLE IF EXISTS doc;
    DROP TABLE IF EXISTS user;
    DROP TABLE IF EXISTS claim_history;
    """)

    # Create schema
    cursor.executescript("""
    CREATE TABLE user (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE,
        profile_data TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE claim (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        title TEXT NOT NULL,
        problem TEXT,
        solution TEXT,
        status TEXT CHECK(status IN ('Pending', 'Resolved', 'Rejected')) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES user(id)
    );

    CREATE TABLE session (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        ended_at TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES user(id)
    );
    
    CREATE TABLE message (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        sender TEXT CHECK(sender IN ('user','assistant', 'tool')) NOT NULL,
        content JSON NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(session_id) REFERENCES session(session_id)
    );
    
    CREATE TABLE doc (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        topic TEXT,
        question TEXT NOT NULL,
        answer TEXT NOT NULL
    );

    CREATE TABLE claim_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        incident_type TEXT NOT NULL,
        estimated_amount REAL NOT NULL,
        injuries INTEGER NOT NULL,
        police_report INTEGER NOT NULL,
        third_party INTEGER NOT NULL,
        days_to_report INTEGER NOT NULL,
        needed_review INTEGER NOT NULL
    );
    """)

    conn.commit()
    conn.close()


def generate_data(db_path: str = "demo-db.sqlite") -> None:
    """
    Populate the demo database with initial data:
    - 3 users (Alice, Bob, Charlie)
    - Claims for Charlie (one pending, one resolved)
    - Example FAQ document
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Insert users
    users = [
        ("Alice", "alice@example.com", '{"age": 30, "location": "Seattle"}'),
        ("Bob", "bob@example.com", '{"age": 25, "location": "New York"}'),
        ("Charlie", "charlie@example.com", '{"age": 40, "location": "Chicago"}'),
    ]

    cursor.executemany(
        "INSERT INTO user (name, email, profile_data) VALUES (?, ?, ?)", users
    )

    # Map user names to IDs
    cursor.execute("SELECT id, name FROM user")
    user_ids = {name: uid for uid, name in cursor.fetchall()}

    # Insert claims for Charlie
    claims = [
        (
            user_ids["Charlie"],
            "Car Accident Claim",
            "User was involved in a minor car accident and submitted documents.",
            None,
            "Pending",
        ),
        (
            user_ids["Charlie"],
            "Laptop Damage Claim",
            "Laptop stopped working after water damage.",
            "Claim approved. User received reimbursement of $800.",
            "Resolved",
        ),
    ]

    cursor.executemany(
        "INSERT INTO claim (user_id, title, problem, solution, status) VALUES (?, ?, ?, ?, ?)",
        claims,
    )

    # Insert demo docs
    docs = [
        (
            "general",
            "How do I submit a claim?",
            "To submit a claim, log in to the insurance portal, go to 'New Claim', "
            "and upload required documents (ID, photos, receipts).",
        ),
        (
            "general",
            "What documents are needed for car accident claims?",
            "For car accident claims, you need to provide a police report, "
            "photos of the damage, and repair estimates.",
        ),
        (
            "general",
            "How can I check the status of my claim?",
            "You can check claim status online via the portal or "
            "contact support with your claim ID.",
        ),
        (
            "general",
            "What is covered under home insurance?",
            "Home insurance typically covers fire, theft, and water damage. "
            "Flooding may require additional coverage.",
        ),
        (
            "general",
            "What should I do if I cannot submit a claim online?",
            "If you experience issues submitting a claim online, "
            "please contact support or visit a local branch.",
        ),
        (
            "general",
            "How long does it take to process a claim?",
            "Processing time is usually 5–10 business days depending on "
            "claim type and documentation completeness.",
        ),
    ]

    cursor.executemany(
        "INSERT INTO doc (topic, question, answer) VALUES (?, ?, ?)", docs
    )

    conn.commit()
    conn.close()


INCIDENT_EFFECT = {
    "auto_collision": 0.0,
    "theft": 1.0,
    "fire": 0.8,
    "water_damage": 0.3,
    "weather": -0.3,
    "other": 0.4,
}


def generate_claim_history(
    db_path: str = "demo-db.sqlite", n_rows: int = 600, seed: int = 42
) -> None:
    """
    Populate `claim_history` with synthetic past claims for the triage model.

    Each row has the structured features the LLM extracts from a free-text
    claim, plus `needed_review`: whether an adjuster had to review the claim
    manually. The label follows a simple logistic rule with noise, so the
    model has a real (if synthetic) signal to learn.
    """
    rng = np.random.default_rng(seed)
    types = list(INCIDENT_EFFECT)

    incident_type = rng.choice(types, size=n_rows, p=[0.35, 0.12, 0.06, 0.2, 0.15, 0.12])
    amount = np.round(rng.lognormal(mean=8.0, sigma=1.0, size=n_rows), -1)
    injuries = rng.random(n_rows) < 0.15
    police_report = rng.random(n_rows) < 0.45
    third_party = rng.random(n_rows) < 0.3
    days_to_report = rng.poisson(lam=6, size=n_rows) + (rng.random(n_rows) < 0.1) * 30

    z = (
        -2.4
        + 1.2 * (amount > 5_000)
        + 1.0 * (amount > 15_000)
        + 1.6 * injuries
        + 1.1 * (days_to_report > 14)
        + 0.8 * third_party
        - 0.8 * police_report
        + np.array([INCIDENT_EFFECT[t] for t in incident_type])
        + rng.normal(0, 0.5, size=n_rows)
    )
    needed_review = rng.random(n_rows) < 1 / (1 + np.exp(-z))

    rows = [
        (str(t), float(a), int(i), int(p), int(tp), int(d), int(r))
        for t, a, i, p, tp, d, r in zip(
            incident_type, amount, injuries, police_report,
            third_party, days_to_report, needed_review,
        )
    ]

    conn = sqlite3.connect(db_path)
    conn.executemany(
        "INSERT INTO claim_history (incident_type, estimated_amount, injuries, "
        "police_report, third_party, days_to_report, needed_review) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def main():
    """Recreate the demo DB with example data."""
    create_db(db_path="demo-db.sqlite")
    generate_data(db_path="demo-db.sqlite")
    generate_claim_history(db_path="demo-db.sqlite")


if __name__ == "__main__":
    main()

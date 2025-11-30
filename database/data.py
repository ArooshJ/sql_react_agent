"""
Database population script for SQL ReAct Agent
Creates and populates company.db with sample employee and department data.

Supports loading from CSV files (sample_data/) with hardcoded fallback.
"""

import sqlite3
import csv
import os

# Hardcoded fallback data (used if CSV files are not found)
DEPARTMENTS_FALLBACK = [
    (1, "Engineering", "San Francisco"),
    (2, "Marketing", "New York"),
    (3, "Sales", "Chicago"),
]

EMPLOYEES_FALLBACK = [
    (1, "Alice Johnson", "alice@company.com", 1, 95000.0, "2022-01-15"),
    (2, "Bob Smith", "bob@company.com", 1, 87000.0, "2022-03-20"),
    (3, "Carol White", "carol@company.com", 1, 110000.0, "2021-06-10"),
    (4, "David Brown", "david@company.com", 2, 72000.0, "2023-02-01"),
    (5, "Eve Davis", "eve@company.com", 2, 68000.0, "2023-05-15"),
    (6, "Frank Miller", "frank@company.com", 3, 65000.0, "2022-09-01"),
    (7, "Grace Lee", "grace@company.com", 3, 70000.0, "2022-11-20"),
    (8, "Henry Wilson", "henry@company.com", 3, 75000.0, "2021-12-01"),
]


def load_departments(script_dir: str) -> list:
    """Load departments from CSV or fallback to hardcoded data."""
    csv_path = os.path.join(script_dir, "sample_data", "departments.csv")
    
    if os.path.exists(csv_path):
        print("  Loading departments from CSV...")
        departments = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                departments.append((
                    int(row['id']),
                    row['name'],
                    row['location']
                ))
        return departments
    else:
        print("  Using hardcoded departments (CSV not found)")
        return DEPARTMENTS_FALLBACK


def load_employees(script_dir: str) -> list:
    """Load employees from CSV or fallback to hardcoded data."""
    csv_path = os.path.join(script_dir, "sample_data", "employees.csv")
    
    if os.path.exists(csv_path):
        print("  Loading employees from CSV...")
        employees = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                employees.append((
                    int(row['id']),
                    row['name'],
                    row['email'],
                    int(row['department_id']),
                    float(row['salary']),
                    row['hire_date']
                ))
        return employees
    else:
        print("  Using hardcoded employees (CSV not found)")
        return EMPLOYEES_FALLBACK


def create_database(db_path: str = "company.db"):
    """
    Creates the database and populates it with sample data.
    
    Args:
        db_path: Path to the database file (default: company.db)
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_db_path = os.path.join(script_dir, db_path)
    
    # Remove existing database if it exists
    if os.path.exists(full_db_path):
        os.remove(full_db_path)
        print(f"Removed existing database: {full_db_path}")
    
    # Create connection
    conn = sqlite3.connect(full_db_path)
    cursor = conn.cursor()
    
    # Read and execute schema
    schema_path = os.path.join(script_dir, "schema.sql")
    with open(schema_path, 'r') as f:
        schema_sql = f.read()
    cursor.executescript(schema_sql)
    print("✓ Schema created")
    
    # Load data (CSV or fallback)
    departments = load_departments(script_dir)
    employees = load_employees(script_dir)
    
    # Insert departments
    cursor.executemany(
        "INSERT INTO departments (id, name, location) VALUES (?, ?, ?)",
        departments
    )
    print(f"✓ Inserted {len(departments)} departments")
    
    # Insert employees
    cursor.executemany(
        "INSERT INTO employees (id, name, email, department_id, salary, hire_date) VALUES (?, ?, ?, ?, ?, ?)",
        employees
    )
    print(f"✓ Inserted {len(employees)} employees")
    
    # Commit and close
    conn.commit()
    conn.close()
    
    print(f"\n✓ Database created successfully at: {full_db_path}")
    print(f"  Tables: departments, employees")
    print(f"  Total employees: {len(employees)}")
    print(f"  Total departments: {len(departments)}")


if __name__ == "__main__":
    create_database()

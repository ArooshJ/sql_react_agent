-- SQL ReAct Agent - Database Schema
-- Simple Employee-Department Database

-- Table: departments
-- Stores company departments
CREATE TABLE IF NOT EXISTS departments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    location TEXT NOT NULL
);

-- Table: employees
-- Stores employee information
CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    department_id INTEGER NOT NULL,
    salary REAL NOT NULL,
    hire_date TEXT NOT NULL,  -- Format: YYYY-MM-DD
    FOREIGN KEY (department_id) REFERENCES departments(id)
);

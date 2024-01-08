CREATE TABLE IF NOT EXISTS html_files (
    id INTEGER PRIMARY KEY,
    category_id INTEGER NOT NULL,
    domain TEXT,
    html TEXT
);

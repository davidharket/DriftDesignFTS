import sqlite3

# Path to your SQLite database
db_path = 'C:/Users/bruker/Desktop/DbParser/html_files.db'

# Connect to the database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    # Query to find rows without '<!DOCTYPE' in the 'html' column
    cursor.execute("SELECT id FROM html_files WHERE html NOT LIKE '%<!DOCTYPE%'")

    # Fetch all matching rows
    rows_to_delete = cursor.fetchall()

    # Delete each row found
    for row in rows_to_delete:
        cursor.execute("DELETE FROM html_files WHERE id = ?", (row[0],))

    # Commit changes
    conn.commit()

except sqlite3.Error as e:
    print(f"An error occurred: {e}")

finally:
    # Close the connection to the database
    conn.close()

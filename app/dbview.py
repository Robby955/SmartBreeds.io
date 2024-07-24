import sqlite3

# Connect to your SQLite database
conn = sqlite3.connect('mydata.db')
cursor = conn.cursor()

# List all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in the database:", tables)


# Function to describe table
def describe_table(table_name):
    cursor.execute(f"PRAGMA table_info({table_name});")
    return cursor.fetchall()


# Function to query table
def query_table(table_name):
    cursor.execute(f"SELECT * FROM {table_name};")
    return cursor.fetchall()


# Example: Describe and query the 'users' table
if 'users' in [table[0] for table in tables]:
    print("Users Table Schema:")
    print(describe_table('users'))

    print("Users Table Data:")
    print(query_table('users'))

# Example: Describe and query the 'image_response' table
if 'image_response' in [table[0] for table in tables]:
    print("Image Response Table Schema:")
    print(describe_table('image_response'))

    print("Image Response Table Data:")
    print(query_table('image_response'))

# Example: Describe and query the 'game' table
if 'game' in [table[0] for table in tables]:
    print("Game Table Schema:")
    print(describe_table('game'))

    print("Game Table Data:")
    print(query_table('game'))

# Example: Describe and query the 'high_scores' table
if 'high_scores' in [table[0] for table in tables]:
    print("High Scores Table Schema:")
    print(describe_table('high_scores'))

    print("High Scores Table Data:")
    print(query_table('high_scores'))

# Close the connection
conn.close()

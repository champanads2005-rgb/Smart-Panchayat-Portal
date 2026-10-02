import mysql.connector


def get_db_connection():
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="champanads@14",
        database="smart_panchayat"
    )

    return connection


if __name__ == "__main__":

    connection = get_db_connection()

    if connection.is_connected():
        print("MySQL connection successful!")

    connection.close()
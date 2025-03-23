#! C:\Users\Sriya\AppData\Local\Programs\Python\Python311\python.exe

print("Content-Type: text/html\n")
print("Hello, World!")


import mysql.connector

print('Content Type: text/html\n')
print('<h1>Details!</h1>')

conn = mysql.connector.connect(host="localhost", user="root", password="", database="test")

cursor = conn.cursor()

query = "select * from users"
cursor.execute(query)

records = cursor.fetchall()

print("No ofo records: ", cursor.rowcount)

for record in records:
    print("id", record[0])
    print("username", record[1])
    print("password", record[2])
    print("age", record[4])
    print()

cursor.close()
conn.close()




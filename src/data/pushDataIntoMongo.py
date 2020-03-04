import csv
import pymongo

# Database informations
f = open("../../auth/auth.private","r")
lines = f.readlines()
dbUsername = lines[0].split("=")[1][:-1]
dbPassword = lines[1].split("=")[1][:-1]
dbURL = lines[2].split("=")[1][:-1]
f.close()

# Connect to database
try: 
    client = pymongo.MongoClient("mongodb+srv://"+dbUsername+":"+dbPassword+"@"+dbURL)
    print("Connected successfully!!!") 
    #TODO push data.
except Exception as e:
    print(e)   
    print("Could not connect to MongoDB") 

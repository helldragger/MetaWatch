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
    client = pymongo.MongoClient("mongodb://"+dbUsername+":"+dbPassword+"@"+dbURL)
    print("Connected successfully!!!") 
except:   
    print("Could not connect to MongoDB") 
#%%
db = client.Reports
collection = db.interest

# Datafile to push in database
vid_uuid = 31
request = {"vid_uuid":vid_uuid}
with open("../../data/processed/"+str(vid_uuid)+"_extracted.interest.csv") as interestfile:
    interest_reader = csv.reader(interestfile, delimiter=',')
    interest_header = next(interest_reader, None)
    with open("../../data/processed/"+str(vid_uuid)+"_extracted.ft.csv") as featurefile:
        feature_reader = csv.reader(featurefile, delimiter=',')
        feature_header = next(feature_reader, None)
        while True:
            interest_data = next(interest_reader, None)
            if interest_data == None:
                break
            feature_data = next(feature_reader, None)
            
            for i in range(len(interest_header)):
                request[interest_header[i]] = interest_data[i]
            for i in range(len(feature_header)):
                request[feature_header[i]] = feature_data[i]
            # Insert Data 
            request_id = collection.insert_one(request)
            print("Data inserted with record ids",request_id) 
print("All request done")



#%%
import csv
import pymongo
#%%
dbUsername= "JacquiotChristopher"
dbPassword= "UTlc3LxasWVyP4FA"
dbURL = "metawatch-cluster0-zucpo.mongodb.net"
# Connect to database
try: 
    client = pymongo.MongoClient("mongodb+srv://"+dbUsername+":"+dbPassword+"@"+dbURL)
    print("Connected successfully!!!") 
except Exception as e:
    print(e)   
    print("Could not connect to MongoDB") 
#%%

print(client.Reports.interest.find({})[0])
#%%
result = client.Reports.interest.insert_one({"frame":-1})
#%%
result.inserted_id
#%%
print(client.Reports.interest.find({"frame":-1})[0])
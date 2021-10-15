### WARNING-WARNING-WARNING-WARNING-WARNING-WARNING-WARNING-WARNING-WARNING ###
### DO NOT SUBMIT DIRECTLY ###
### DO NOT SUBMIT DIRECTLY ###
### DO NOT SUBMIT DIRECTLY ###
### DO NOT SUBMIT DIRECTLY ###
### DO NOT SUBMIT DIRECTLY ###





# List 1  3-7

UniversityList = ["Todai", "Tokodai", "Hokudai"]
print (UniversityList)

# list 1b
UniversityList = ["Todai", "Tokodai", "Hokudai"]
LengthUniversityList = len(UniversityList)
print(LengthUniversityList)

# list 2
DatatypeUniversityList = type(UniversityList)
print (DatatypeUniversityList)

# list 2b
print(UniversityList[0])
print(UniversityList[2])
print(UniversityList[1])

# list 2c
UniversityList = ["Todai", "Tokodai", "Tokodai", "Hokudai"]
print(UniversityList.index("Todai"))
print(UniversityList.index("Tokodai"))
print(UniversityList.index("Hokudai"))

# list 3
UniversityList = ["Todai", "Tokodai", "Hokudai"]
UniversityList = [" Hokudai ", "Tokodai", " Todai"]

# list 3a
UniversityList = ["Todai", "Tokodai", "Tokodai", "Hokudai"]
print(UniversityList)

# list 3b
UniversityList = ["Todai", "Tokodai", "Hokudai"]
UniversityList[0] = "Todai1"
print(UniversityList)

# list 3c
UniversityList = ["Todai", "Tokodai", "Hokudai"]
UniversityList.append("Handai")
print(UniversityList)
UniversityList.insert(1, "NII")
print(UniversityList)

# list 4a
UniversityList = ["Todai", "Tokodai", "Hokudai"]
UniversityList.clear()
print(UniversityList)

# list 4b
UniversityList = ["Todai", "Tokodai", "Hokudai"]
print(UniversityList.pop(1))
print(UniversityList)

# list 4c
UniversityList = ["Todai", "Tokodai", "Hokudai"]
UniversityList.remove("Todai")
print(UniversityList)

# Page 7-9 

# List 5a
UniversityList = ["Todai", "Tokodai", "Hokudai"]
del UniversityList[1]
print (UniversityList)

#List 5b
UniversityList = ["Todai", 1, "Hokudai", True, 9]
print (UniversityList)

#List 5c
print (type("Todai"))
print (type(1))
print (type("Hokudai"))
print (type(True))
print (type(9))

#Tuple 1a 8-9
UniversityList = ("Todai", "Tokodai", "Hokudai")
print (UniversityList)

#Tuple 1b
UniversityList = ("Todai", "Tokodai", "Hokudai")
l = len(UniversityList)
print(l)

#Tuple 1c
UniversityList = ("Todai","Tokodai","Hokudai")
t = type(UniversityList)
print(t)

#Tuple 1d
UniversityList = ("Todai","Tokodai","Hokudai")
print (UniversityList[1])

#Tuple 1f
UniversityList = ("Todai","Tokodai","Hokudai")
print (UniversityList.index("Hokudai"))

#Tuple 2a
UniversityList1 = ("Todai","Tokodai","Hokudai")
UniversityList2 = ("Tokodai","Todai","Hokudai")
UniversityList2 == UniversityList1

#Tuple 2b
UniversityList = ("Todai","Hokudai","Tokodai","Hokudai")
print (UniversityList)

#Tuple 2c
UniversityList = ("Todai",1,"Hokudai",True)
print (type("Todai"))
print (type(1))
print (type("Hokudai"))
print (type(9))


# Set 1a 10-11
UniversityList = {"Todai", "Tokodai", "Hokudai"}
print (UniversityList)

#Set 1b
UniversityList = {"Todai", "Tokodai", "Hokudai"}
print (len(UniversityList))

#Set 1c
UniversityList = {"Todai", "Tokodai", "Hokudai"}
print (type(UniversityList))

#Set 2a
UniversityList = {"Todai", "Tokodai", "Hokudai", "Tokodai"}
print (UniversityList)

#Set 2b
UniversityList = {"Todai", 1, True}
print (type("Todai"))
print (type(1))
print (type(True))

# String1  12-13
s = "The University of Tokyo"
print(s)

# string2a
a = """Tokyo is the capital of Japan, UTokyo is a university in Japan"""
print(a)

# string2b
a = '''Tokyo is the capital of Japan, UTokyo is a university in Japan'''
print(a)

# string3
s = "The University of Tokyo"
print(s[0])
print(s[1])
print(s[5])

# string4
s = "The University of Tokyo"
print(len(s))

# string5a
s = "The University of Tokyo"
print('h' in s)
print('H' in s)

# string5b
s = "The University of Tokyo"
print('The ' in s)
print('123' in s)

# string6
s = "The University of Tokyo"
print('The ' not in s)
print('123' not in s)


# Dictionary(14-15)
demographics={
"age":40,
"gender":"female",
"height":180,
"weight":70,
"smoke":False
}
print(demographics)
print(demographics["height"])
print(demographics["gender"])
print(len(demographics))
print(type(demographics))

# Page 18-20

a = 3
b = 5
if (a != b):
    print("Correct")

UnivList1 = [1, 3, 4]
UnivList2 = [1, 4, 3]
if (UnivList1 != UnivList2):
    print("Correct")

# Without indentation, it will raise an SyntaxError
# a = 3
# b = 5
# if (a != b):
# print("Correct")

a = 3
b = 5
if a > b:
    print("a is greater than b")
elif a < b:
    print("a is smaller than b")

a = 1
b = 1
if a > b:
    print("a is greater than b")
elif a < b:
    print("a is smaller than b")
else:
    print("a is equal to b")

a = 3
b = 5
if a >= b:
    print("a is greater than or equal to b")
else:
    print("a is smaller than b")

a = 1
b = 5
c = 0
if a < b and b > c:
    print("correct!!!")

a = 1
b = 5
c = 0
if a > b or b > c:
    print("0")

a = 10
if a > 5:
    print("Above 5")
    if a > 8:
        print("and above 8")
    else:
        print("but not above 8")

# Page 22-24

print("For loop")
univs = ['Todai', 'Tokodai', 'Hokudai']
for u in univs:
    print(u)
    
s = 'UTokyo'
for x in s:
    print(x)
    
print("Break Statement")
univs = ['Todai', 'Tokodai', 'Hokudai']
for x in univs:
    print(x)
    if x == "Tokodai":
        break

univs = ['Todai', 'Tokodai', 'Hokudai']
for x in univs:
    if x == "Tokodai":
        break
    print(x)

print("Continue Statement")
univs = ['Todai', 'Tokodai', 'Hokudai']
for x in univs:
    if x == "Tokodai":
        continue
    print(x)



# Range() 25-28
for i in range(5):
    print(i)
    
for i in range(1,5):
    print(i)
    
for i in range(2,10,3):
    print(i)


#else keyword
for x in range(5):
    print(x)
else:
    print("Done!")
    
for x in range(5):
    if x == 3:
        break
    print(x)
else:
    print("Done!")

    
#nested loop
univs = ["university1","university2","university3"]
places = ["Tokyo","Osaka","Hokkaido"]
for x in univs:
    for y in places:
        print(x+" is in "+y)

x = [1,3,5]
y = [2,5,9]
for i in range(0,len(x)):
    for j in range(0,len(y)):
        print(x[i]+y[j])
##--------------------------------------------end of nested loop



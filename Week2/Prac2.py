## Example 2
#Author: YANBO LIANG
#Python 3.6.3 - Unordered
SampleDict = {
    "a" : 40,
    "d" : 70,
    "f" : 25,
    "b" : 30,
    "c" : 80,
    "e" : 50
}

print("This is an unordered dictionary: ", SampleDict)
##
SampleDict["b"] = 100
print("New dict is ", SampleDict)
##
WrongDict = {
    "a" : 40,
    "d" : 70,
    "d" : 25,
    "d" : 65,
}
print("Only one d is shown: ", WrongDict)

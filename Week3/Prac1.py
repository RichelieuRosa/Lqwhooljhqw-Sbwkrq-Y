### Homework ex1

#Author: Yanbo Liang
#Date: 10.22

f = open("ex1.txt","w+")

for k in range(1,101):
    for i in range(1,6):
        if i == 5:
            f.write("%d^%d\r\n" % (k,i))
        else:
            f.write("%d^%d, " % (k, i))

# append the decending trend

for k in range(0,100):
    for i in range(1,6):
        if i == 5:
            f.write("%d^%d\r\n" % (100-k,6-i))
        else:
            f.write("%d^%d, " % (100-k, 6-i))

## HW3

#Author: Yanbo LIANG
#Date: 2021-10-16

listA = [1,2,3,4,5,6,7,8,9,10]
rlist = sorted(listA, reverse = True)
print("Original list", listA)
print("reversed list", rlist)

alpha = ['z', 'y', 'x', 'w', 'v', 'u', 't', 's', 'r', 'q', 'p', 'o', 'n', 'm', 'l', 'k', 'j', 'i', 'h', 'g', 'f', 'e', 'd', 'c', 'b',
 'a']
ind_n = alpha.index('n')
newAlpha = list()
for i in range(ind_n,-1,-1):
    newAlpha.append(alpha[i])

for k in range(len(alpha)-1, ind_n, -1):
    newAlpha.append(alpha[k])
print("Original alphabet is : ", alpha)
print("New alphabet list is: ", newAlpha)

l = [1,2,3,4,5,6,7]
for j in range(len(l)):
    print("="*50)
    print(l)
    for i in range(len(l) - 1, -1, -1):
        if i == j: del l[i]
    print(l)
    print("="*50)


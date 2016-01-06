#!/usr/bin/python3

from sys import argv, exit

if len(argv) != 2:
    print("Usage : %s <fichier_de_résultats" % argv[0])
    exit()

with open(argv[1], 'r') as f:
    data = []
    datanames = f.readline().split(';')[2:7] # Consumes the first line
    for line in f:
        data.append([ float(i) for i in line.split(';')[2:7] ])

    for i in range(5):
        my_sum = 0
        my_min = data[0][i]
        my_max = data[0][i]
        for j in data:
            my_sum += j[i]
            if j[i] > my_max:
                my_max = j[i]
            elif j[i] < my_min:
                my_min = j[i]

        my_moy = my_sum / len(data)

        print(datanames[i] + ' :')
        print("\tmin : %f\n\tmax : %f\n\tmoy : %f" % (my_min, my_max, my_moy))

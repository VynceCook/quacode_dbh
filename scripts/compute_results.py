#!/usr/bin/python3

from sys import argv, exit
from time import strftime, gmtime

if len(argv) != 3:
    print("Usage : %s <results_file> <algo_id>" % argv[0])
    exit()

with open(argv[1], 'r') as f:
    data = []
    datanames = f.readline().split(';')[2:7] # Consumes the first line
    for line in f:
        if line:
            data.append([ float(i) for i in line.split(';')[2:7] ])

    # min / max / moy calculation
    my_sum = [ 0 ] * 5
    my_min = list(data[0])
    my_max = list(data[0])
    my_moy = [ 0 ] * 5

    for i in range(5):
        for j in data:
            my_sum[i] += j[i]
            if j[i] > my_max[i]:
                my_max[i] = j[i]
            elif j[i] < my_min[i]:
                my_min[i] = j[i]

        my_moy[i] = my_sum[i] / len(data)

        print(datanames[i] + ' :')
        print("\tmin : %f\n\tmax : %f\n\tmoy : %f" % (my_min[i], my_max[i], my_moy[i]))
    
    # Saves the results in a dated file in csv format
    # First lines describes the minimal values, the second on describes the
    # maximal and the third the average
    with open(strftime("condensed_" + argv[2] + "_%y-%m-%d_%H:%M:%S", gmtime()) + ".txt", 'w') as res_file:
        res_file.write(';'.join(datanames) + '\n')
        res_file.write(';'.join([ str(i) for i in my_min ]) + '\n')
        res_file.write(';'.join([ str(i) for i in my_max ]) + '\n')
        res_file.write(';'.join([ str(i) for i in my_moy ]) + '\n')

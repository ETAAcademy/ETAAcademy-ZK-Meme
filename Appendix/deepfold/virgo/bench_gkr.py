import os
import random, string
import csv

SIZE = 18

def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))

tmp_file = randomword(20) + '.txt'
with open("fft_gkr.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    for i in range(10, SIZE):
        print('bench fft_gkr of size', i)
        round = 10
        prover_time = 0
        verifier_time = 0
        proof_size = 0
        for j in range(0, round):
            os.system('./fft_gkr ' + str(i) + ' ' + tmp_file)
            f = open(tmp_file)
            lines = f.readlines()
            v, ps, p = lines[0].split(' ')
            prover_time += float(p)
            proof_size += float(ps)
            verifier_time += float(v)
        writer.writerow([i, prover_time * 1000 / round, verifier_time * 1000 / round, proof_size / round])
os.remove(tmp_file)
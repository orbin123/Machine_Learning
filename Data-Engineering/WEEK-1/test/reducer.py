import sys 

cur_cat = None 
cur_amt = 0

for line in sys.stdin:
    cat , amt = line.strip().split('\t')

    if cur_cat == cat:
        cur_amt += amt 

    else:
        print(f"{cur_amt}\t{cur_cat}")


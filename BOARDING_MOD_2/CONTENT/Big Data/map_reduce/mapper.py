import sys 

for line in sys.stdin:
    for word in line.lower().split():
        print(f"{word}\t1")
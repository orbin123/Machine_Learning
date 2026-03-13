import sys

for line in sys.stdin:
    line = line.strip()

    fields = line.split(',')

    cat, amt = fields[3], fields[4]

    print(f'{cat}\t{amt}')
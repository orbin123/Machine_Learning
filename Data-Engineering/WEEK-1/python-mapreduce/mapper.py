import sys
import re

for line in sys.stdin:
    line = line.strip()
    words = re.split(r'\W+', line.lower())
    for word in words:
        if word: 
            print(f"{word}\t1")
import sys 

current_word = None 
count = 0

for line in sys.stdin:
    word, value = line.strip().split("\t")

    if word != current_word:
        if current_word is not None:
            print(f"{current_word}\t{count}")
        current_word = word 
        count = 0

    count += int(value)

if current_word is not None:
    print(f"{current_word}\t{count}")

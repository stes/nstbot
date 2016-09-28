import readchar
print("Reading a char:")
b_loop = True
ind = 0
while b_loop:
    char = readchar.readchar()
    char = readchar.readkey()
    ind += 1
    print char
    print type(char)
    if char.lower() == '8':
        b_loop = False

    if ind > 5:
        b_loop = False



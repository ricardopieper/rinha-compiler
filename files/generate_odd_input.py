bigsum = map(lambda x: str(x), list(range(0, 30)))
bigsum = " + ".join(bigsum)


buf = ""
for i in range(0, 4000):
    i_str = str(i)
    buf += "let x"+i_str+" = "+bigsum + ";\n"

last = i

buf += "print(x"+str(last)+")"

#write to file
f = open("odd_input.rinha", "w")
f.write(buf)
f.close()
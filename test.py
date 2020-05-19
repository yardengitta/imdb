import os
if not os.path.exists('output'):
    os.mkdir('output')

for x in range(10001,10010):
   name_file = os.path.join("output", "data"+str(x)+".txt")
   f = open(name_file, 'w')
   f.write(str(x) + "\n" + str(20000-x))
   f.close()

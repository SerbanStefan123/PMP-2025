import random

Red=3
Blue=4
Black=2
dice=0
count_Red=0

#Punctul A)
def draw():
   global Red,Blue,Black
   urna = ['Red'] * Red + ['Blue'] * Blue + ['Black'] * Black
   bila=random.choice(urna) 
   if bila=='Red':
    Red-=1
   elif bila=='Blue':
    Blue-=1
   else:
    Black -= 1
   print("Bila extrasa: " + bila)
   

def dice():
    global Red, Blue, Black
    dice=random.randint(1,6)
    if dice in [2,3,5]:
        Black+=1
    elif(dice==6):
        Red+=1
    else:
        Blue+=1
    print("Dupa aruncarea zarului: " + str(dice))

#Punctul B)
def prob_Red():
    global count_Red
    urna = ['Red'] * Red + ['Blue'] * Blue + ['Black'] * Black
    for i in range(1,10000):
        if(random.choice(urna)=='Red'):
            count_Red+=1
    print("Probabilitatea ca bila aleasa sa fie rosie este: " + str(count_Red/10000))


dice()
draw()
prob_Red()

#Punctul C (GRESIT)
print ("Probabilitatea teoretica ca bila aleasa sa fie rosie este: " + str(Red/(Red+Blue+Black)))
if count_Red/10000 > Red/(Red+Blue+Black):
    print("Probabilitatea teoretica este mai mica!")
else:
    print("Porbabilitatea teoretica este mai mare!")


import turtle


n_deg = 90
n_dist = 100
on = True

current = turtle.Turtle()

def DEG(degrees):
    global n_deg
    n_deg = degrees

def DIST(distance):
    global n_dist
    n_dist = distance

def L():
    current.left(n_deg)

def R():
    current.right(n_deg)

def F():
    current.forward(n_dist)

def BLUE():
    current.color("blue")

def BLACK():
    current.color("black")

def RED():
    current.color("red")

def GREEN():
    current.color("green")

def UP():
    current.penup()

def DOWN():
    current.pendown()

def Q():
    global on
    on = False

def main():

    while (on):

        user_command = str(input("enter command"))
        if user_command == "L":
            L()
        elif user_command == "F":
            F()
        elif user_command == "R":
            R()
        elif user_command == "UP":
            UP()
        elif user_command == "DOWN":
            DOWN()
        elif user_command == "BLUE":
            BLUE()
        elif user_command == "BLACK":
            BLACK()
        elif user_command == "RED":
            RED()
        elif user_command == "GREEN":
            GREEN()
        elif user_command == "DEG":
            degrees = (int(input("")))
            DEG(degrees)
        elif user_command == "DIST":
            distance = (int(input("")))
            DIST(distance)
        elif user_command == "Q":
            Q()
        else:
           print("Invalid Command -- Try Again.")

    print("END OF PROGRAM")

main()




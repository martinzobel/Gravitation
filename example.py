import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
fig, ax = plt.subplots()

ax.axis('equal')  # impose la même échelle sur les deux axes
ax.set(xlim=[-200, 200], ylim=[-200,200]) # limites du graphique en x et en y

# définition du nombre de corps massique
N = 10
# constante de gravitation 
G = 1
# définition de manières aléatoires (dans [10,100] x [10,100])des positions initiale des N corps à l’instant t 
# la forme de la matrice des positions est donc de la forme suivante : il y a N lignes pour 
# les N corps et la première colonne est la coordonnée selon x et la deuxième selon y
positionst = np.random.randint(10,100,(N,2)) 
print(positionst)
# arbitrairement on a choisit que la position du second instant (donc à t+dt) était la même qu’à l’instant initiale
# on utilise .copy() pour modifier les deux tableaux de manière indépendante 
positionstdt = positionst.copy() #On modifait les deux en même temps ducoup ca foirait tout
# définition de la masse des N corps de manière aléatoire entre 1 et 5 kg 
masses = np.random.randint(1,5,N) 
# définition du pas de temps dt
dt = 0.01

#la fonction calcule la position à t + 2dt à partir des positions à t et t+dt
def get_new_position(position):
# on récupère les coordonnées selon x des N corps à l’intant t
    Xt = positionst[:,0]
# on transforme notre “liste” en vecteur colonne pour pouvoir ensuite faire la transposée
    Xt = Xt[:,np.newaxis]
#idem
    Yt = positionst[:,1]
    Yt = Yt[:,np.newaxis]
#idem
    Xtdt = positionstdt[:,0][:,np.newaxis]#On avait un problème de shape
#idem
    Ytdt = positionstdt[:,1][:,np.newaxis]#idem
#on créer une matrice dans laquelle nous aurons les différences entre les différents x_i
#à l’aide du broadcasting on obtient donc un matrice Rx de la forme Rx [i,j] = x_i - x_j
    Rx = Xt - Xt.transpose()
#idem
    Ry = Yt - Yt.transpose()
#variable définie pour simplifier l’écriture
    R = (Rx*2 + Ry*2)*(3/2)
#on modifie les coefficients diagonaux qui sont des 0 car R[i,i] = x_i - x_i = 0 
#on les remplaces par l’infini pour éviter de diviser par 0 mais aussi pour que la de la force de gravitation d’un corps sur lui même soit nul 
    R[R==0] = np.inf
# calcule de x(t+2dt) à l’aide de la formule obtenu grâce au PFD mais aussi grâce à la méthode d’euler (pour les dérivées premières et secondes) 
# le produit avec masses se fait à l’aide du broadcasting
    Xt2dt = 2*Xtdt[:,0] - Xt[:,0] -(dt**2)*G*np.sum((Rx/R)*masses,axis = 1)#On a sommé sur i au lieu de sommer sur j 
    Yt2dt = 2*Ytdt[:,0] - Yt[:,0] - (dt**2)*G*np.sum((Ry/R)*masses,axis = 1)# idem
# la fonction np.column_stack permet d’assembler les deux vecteurs colonnes 
# le return renvoie donc les positions à t + dt et celle à t +2dt
    return positionstdt, np.column_stack((Xt2dt,Yt2dt))

scat = ax.scatter(positionst[:,0], positionst[:,1])

def animate(t):
    # une variable globale est une variable utilisée dans une fonction mais dont la modification de la valeur a une portée globale (donc extérieure à la fonction) 

    global positionst
    global positionstdt
# position à t devient position à t + dt et position à t + dt devient position à t +2dt
    positionst,positionstdt = get_new_position(positionst)

    # update the scatter plot:
    # le np.stack sert ici à mettre les positions dans la bonne shape
    data = np.stack([positionst[:,0],positionst[:,1]]).T
    scat.set_offsets(data)
    return scat

ani = animation.FuncAnimation(fig=fig, func=animate, interval=100)
plt.show()



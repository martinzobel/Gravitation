import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
fig, ax = plt.subplots()


ax.axis('equal')
ax.set(xlim=[-200, 200], ylim=[-200,200])

N = 10
G = 1
positionst = np.random.randint(10,100,(N,2))
print(positionst)
positionstdt = positionst.copy() #On modifait les deux en même temps ducoup ca foire tout
masses = np.random.randint(1,5,N)
dt = 1

def get_new_position(position):
    Xt = positionst[:,0]
    Xt = Xt[:,np.newaxis]
    Yt = positionst[:,1]
    Yt = Yt[:,np.newaxis]
    Xtdt = positionstdt[:,0][:,np.newaxis]#On avait un problème de shape
    Ytdt = positionstdt[:,1][:,np.newaxis]#idem
    Rx = Xt - Xt.transpose()
    Ry = Yt - Yt.transpose()
    R = (Rx**2 + Ry**2)**(3/2)
    R[R==0] = np.inf
    Xt2dt = 2*Xtdt[:,0] - Xt[:,0] -(dt**2)*G*np.sum((Rx/R)*masses,axis = 1)#On a sommé sur i au lieu de sommer sur j 
    Yt2dt = 2*Ytdt[:,0] - Yt[:,0] - (dt**2)*G*np.sum((Ry/R)*masses,axis = 1)# idem
    return positionstdt, np.column_stack((Xt2dt,Yt2dt))

scat = ax.scatter(positionst[:,0], positionst[:,1])


def animate(t):
    # une variable globale est une variable utilisée dans une fonction mais dont la modification de la valeur a une portée globale (donc extérieure à la fonction)

    global positionst
    global positionstdt
    positionst,positionstdt = get_new_position(positionst)

    # update the scatter plot:
    # le np.stack sert ici à mettre les positions dans la bonne shape
    data = np.stack([positionst[:,0],positionst[:,1]]).T
    scat.set_offsets(data)
    return scat

ani = animation.FuncAnimation(fig=fig, func=animate, interval=100)
plt.show()
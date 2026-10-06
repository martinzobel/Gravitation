import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
fig, ax = plt.subplots()


ax.axis('equal')
ax.set(xlim=[-200, 200], ylim=[-200,200])

N = 5
G = 10
positionst = np.random.randint(10,100,(N,2))
print(positionst)
positionstdt = positionst
masses = np.random.randint(1,5,N)
dt = 0.01

def get_new_position(position):
    Xt = positionst[:,0]
    Xt = Xt[:,np.newaxis]
    Yt = positionst[:,1]
    Yt = Yt[:,np.newaxis]
    Xtdt = positionstdt[:,0]
    Ytdt = positionstdt[:,1]
    Rx = Xt - Xt.transpose()
    Ry = Yt - Yt.transpose()
    R = (Rx**2 + Ry**2)**(3/2)
    R[R==0] = np.inf
    Xt2dt = 2*Xtdt - Xt -(dt**2)*G*np.sum((Rx/R)*masses,axis = 0)
    Yt2dt = 2*Ytdt - Yt - (dt**2)*G*np.sum((Ry/R)*masses,axis = 0)
    return positionstdt, np.column_stack((Xt2dt,Yt2dt))

scat = ax.scatter(positionst[0], positionst[1])


def animate(t):
    # une variable globale est une variable utilisée dans une fonction mais dont la modification de la valeur a une portée globale (donc extérieure à la fonction)

    global positionst,positionstdt
    positionst,positionstdt = get_new_position(positionst)

    # update the scatter plot:
    # le np.stack sert ici à mettre les positions dans la bonne shape
    data = np.stack([positionst[:,0],positionst[:,1]]).T
    scat.set_offsets(data)
    return scat

ani = animation.FuncAnimation(fig=fig, func=animate, interval=100)
plt.show()
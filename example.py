import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
fig, ax = plt.subplots()


ax.axis('equal')
#ax.set(xlim=[-1, 1000], ylim=[-1,1000])

N = 3
G = 1
positionst = np.random.randint(0,50,(N,2))
positionstdt = positionst
masses = np.random.randint(1,5,N)
dt = 1

def get_new_position(position):
    Xt = positions[:,0]
    Xt = Xt[:,np.newaxis]
    Yt = positions[:,1]
    Yt = Yt[:,np.newaxis]
    Xtdt = positionstdt[:,0]
    Ytdt = positionstdt[:,1]
    Rx = Xt - Xt.transpose()
    Ry = Yt - Yt.transpose()
    R = (Rx**2 + Ry**2)**(3/2)
    R[R==0] = inf
    Xt2dt = 2*Xtdt - Xt -(dt**2)*G*np.sum(masses*Rx/R,axis = 0)
    Yt2dt = 2*Ytdt - Yt - (dt**2)*G*np.sum(masses*Ry/R,axis = 0)
    global positionst
    global positionstdt
    positionst,positionstdt = positionstdt, np.array([Xt2dt,Yt2dt])
    return [[Xs[0]+1, Xs[1]-1], [Ys[0]+1, Ys[1]+1]]

scat = ax.scatter(positions[0], positions[1])


def animate(t):
    # une variable globale est une variable utilisée dans une fonction mais dont la modification de la valeur a une portée globale (donc extérieure à la fonction)

    global positions
    positions = get_new_position(positions)

    # update the scatter plot:
    # le np.stack sert ici à mettre les positions dans la bonne shape
    data = np.stack(positions).T
    scat.set_offsets(data)
    return scat

ani = animation.FuncAnimation(fig=fig, func=animate, interval=100)
plt.show()
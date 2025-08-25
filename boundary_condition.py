from celluloid import Camera
from matplotlib import pyplot as plt
import random
import numpy as np

random.seed(10)

numAtoms = 10
unitCellLength = 10

# positions of the particles
positions = np.random.rand(unitCellLength, 2)*unitCellLength+unitCellLength
velocities = (np.random.rand(unitCellLength, 2)-0.5)  # velocities of particles

iterations = 600
timestep = 0.1


def check_boundaries(positions, unitCellLength):
    # Function to check if an atom has crossed a cell boundary, if so, translate it back into the cell
    for posVal in range(len(positions)):
        for col in [0, 1]:

            if positions[posVal, col] > unitCellLength*2:
                positions[posVal, col] = positions[posVal, col] - \
                    unitCellLength

            if positions[posVal, col] < unitCellLength:
                positions[posVal, col] = positions[posVal, col] + \
                    unitCellLength


def update_positions(positions, velocities, timestep):
    # Function to continually progress the atom's positions based on their velocities
    positions += positions*velocities*timestep


def makeImages(positions, unitCellLength):
    # Function to make a complete array of all atoms, including the image atoms.
    numParts = len(positions)

    allParticles = np.zeros((numParts*9, 2))

    iter = -1

    # Add all the original particle positions
    while iter < numParts-1:
        iter += 1
        allParticles[iter, 0] = positions[iter, 0]
        allParticles[iter, 1] = positions[iter, 1]

    xmult = np.concatenate(
        (np.ones((numParts, 1)), np.zeros((numParts, 1))), axis=1)
    ymult = np.concatenate(
        (np.zeros((numParts, 1)), np.ones((numParts, 1))), axis=1)

    for xshift in range(1, -2, -1):
        for yshift in range(1, -2, -1):

            if xshift == 0 and yshift == 0:
                continue
            else:
                tempParticles = positions + xmult*xshift * \
                    unitCellLength + ymult*yshift*unitCellLength

                for i in range(numParts):
                    iter += 1
                    allParticles[iter, 0] = tempParticles[i, 0]
                    allParticles[iter, 1] = tempParticles[i, 1]

    return allParticles


def plotBoundaries(unitCellLength):
    # Function to plot the boundaries of the image cells
    for mult in range(1, 3):
        plt.plot([0, unitCellLength*3],
                 [unitCellLength*mult, unitCellLength*mult], 'k')
        plt.plot([unitCellLength*mult, unitCellLength*mult],
                 [0, unitCellLength*3], 'k')


fig = plt.figure()
plt.xlim([0, unitCellLength*3])
plt.ylim([0, unitCellLength*3])
camera = Camera(fig)

# Uncomment this section and remove the other coloring method to use the blue/green color scheme
# colorsUnit = ['b']*len(positions)
# colorsOther = ['g']*len(positions)*8
# colors =colorsUnit + colorsOther

colorsList = random.choices(range(1, 20), k=len(positions))
colors = colorsList*9

# Update the positions and save the plot
for i in range(iterations):

    allAtoms = makeImages(positions, unitCellLength)

    plt.scatter(allAtoms[:, 0], allAtoms[:, 1], s=30, c=colors)
    plotBoundaries(unitCellLength)

    update_positions(positions, velocities, timestep)
    check_boundaries(positions, unitCellLength)

    camera.snap()

animation = camera.animate()
animation.save('pbc_tile_color.gif', writer='pillow', fps=40)

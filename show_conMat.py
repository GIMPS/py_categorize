import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from conMat import matrix
import numpy as np
from trainLoss import loss
from mapping import name_mapping_list
plt.figure()
plt.plot(loss)
confusion=np.array(matrix)
# Set up plot
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111)
cax = ax.matshow(confusion)
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + name_mapping_list, rotation=90)
ax.set_yticklabels([''] + name_mapping_list)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()
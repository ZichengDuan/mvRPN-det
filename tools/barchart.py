import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages
n_groups = 9

means_men = (0.16505513* 100, 0.09253605* 100, 0.08481764* 100, 0.13324852* 100, 0.12383376* 100, 0.07862595* 100, 0.1014419* 100, 0.22044105* 100, 0)

means_women = (0.20166667* 100, 0.03416667* 100, 0.04666667* 100, 0.07666667* 100, 0.09583333* 100, 0.12* 100, 0.13833333* 100, 0.28666667* 100, 0)

occ_train = (36.9)
occ_test = (17.3)

pdf = PdfPages('distribution.pdf')
fig, ax = plt.subplots(1, 2, figsize=(40,11), gridspec_kw={
                           'width_ratios': [1, 3]})

index = np.arange(n_groups)
index2 = np.array([8])
bar_width = 0.35

opacity = 0.4
error_config = {'ecolor': '0.3'}
ax1 = ax[0]
ax2 = ax[1]
font1 = {'family' : 'Arial',
'weight' : 'normal',
'size'   : 22,
}
rects1 = ax1.bar(index, means_men, bar_width,
                alpha=opacity, color='b',
                 error_kw=error_config,
                label='Training set orientation distribution')
rects2 = ax1.bar(index + bar_width, means_women, bar_width,
                alpha=opacity, color='r',
                 error_kw=error_config,
                label='Testing set orientation distribution')

rects3 = ax1.bar(index2, occ_train, bar_width,
                alpha=opacity, color='g',
                 error_kw=error_config,
                label='Training set Inv./Occ. rate'
                )
rects4 = ax1.bar(index2 + bar_width, occ_test, bar_width,
                alpha=opacity, color='y',
                 error_kw=error_config,
                 label='Testing set Inv./Occ. rate')

# ax1.set_ylabel('%')
ax1.set_ylabel('%', fontdict=font1)
# ax.set_xlabel('Angle distribution')
# ax.set_title('Scores by group and gender')
# ax1.ticks(fontsize=100)
ax1.tick_params(axis='x', labelsize= 20)
ax1.tick_params(axis='y', labelsize= 20)
ax1.set_xticks(index + bar_width / 2)
ax1.set_xticklabels(('0', ' ', ' ', r'$\pi$', ' ', ' ', ' ',r'2$\pi$', 'Inv./Occ.'))
for a,b in zip(range(8),means_men):
    ax1.text(a, b+0.05, '%.1f' % b, ha='center', va= 'bottom',fontsize=20)

for a,b in zip(range(8),means_women):
    ax1.text(a+bar_width, b+0.05, '%.1f' % b, ha='center', va= 'bottom',fontsize=20)

ax1.text(8, occ_train+0.05, '%.1f' % occ_train, ha='center', va= 'bottom',fontsize=20)
ax1.text(8+bar_width, occ_test+0.05, '%.1f' % occ_test, ha='center', va= 'bottom',fontsize=20)

ax1.legend(prop=font1)
img= Image.open("/home/dzc/Desktop/CASIA/proj/mvRPN-det/tools/datasets.png")
ax2.imshow(img)
ax2.set_xticks([])
ax2.set_yticks([])
fig.tight_layout()

pdf.savefig()
# plt.show()
plt.close()
pdf.close()
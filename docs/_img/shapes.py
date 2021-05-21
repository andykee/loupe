import matplotlib.pyplot as plt
import loupe

c = loupe.circlemask(shape=(256, 256), radius=120)
plt.imshow(c)
plt.savefig('../_static/img/circlemask.png', transparent=True, bbox_inches='tight', dpi=300)

cm = loupe.circle(shape=(256, 256), radius=120)
plt.imshow(cm)
plt.savefig('../_static/img/circle.png', transparent=True, bbox_inches='tight', dpi=300)

cs = loupe.circle(shape=(256, 256), radius=50, shift=(50, -50))
plt.imshow(cs)
plt.savefig('../_static/img/circleshift.png', transparent=True, bbox_inches='tight', dpi=300)

h = loupe.hexagon(shape=(256, 256), radius=120)
plt.imshow(h)
plt.savefig('../_static/img/hexagon.png', transparent=True, bbox_inches='tight', dpi=300)

hr = loupe.hexagon(shape=(256, 256), radius=120, rotate=True)
plt.imshow(hr)
plt.savefig('../_static/img/hexagonrotate.png', transparent=True, bbox_inches='tight', dpi=300)

s = loupe.slit(shape=(256, 256), width=11)
plt.imshow(s)
plt.savefig('../_static/img/slit.png', transparent=True, bbox_inches='tight', dpi=300)

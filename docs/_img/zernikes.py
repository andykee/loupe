import numpy as np
import matplotlib.pyplot as plt
import loupe

mask = loupe.circlemask((256,256), 128)
z4 = 100 * loupe.zernike(mask, index=4)
plt.imshow(z4)
plt.savefig('../_static/img/zernike_focus.png', transparent=True, bbox_inches='tight', dpi=300)

mask = loupe.circlemask((256,256), 128)
coefficients = [0, 0, 0, 200, 0, -100]
z = loupe.zernike_compose(mask, coefficients)
plt.imshow(z)
plt.savefig('../_static/img/zernike_compose.png', transparent=True, bbox_inches='tight', dpi=300)

coeffs = np.random.rand(10)
opd = loupe.zernike_compose(mask, coeffs)
plt.imshow(opd)
plt.savefig('../_static/img/zernike_compose_random.png', transparent=True, bbox_inches='tight', dpi=300)

mask = loupe.circlemask((256,256), radius=50, shift=(0,60))
rho, theta = loupe.zernike_coordinates(mask, shift=(0,0))
z4 = loupe.zernike(mask, 4, rho=rho, theta=theta)
plt.imshow(z4)
plt.savefig('../_static/img/zernike_custom_coords.png', transparent=True, bbox_inches='tight', dpi=300)

mask = loupe.hexagon((256,256), radius=128)
rho, theta = loupe.zernike_coordinates(mask, shift=(0,0), rotate=60)
z2 = loupe.zernike(mask, 2, rho=rho, theta=theta)
plt.imshow(z2)
plt.savefig('../_static/img/zernike_custom_coords_hex.png', transparent=True, bbox_inches='tight', dpi=300)

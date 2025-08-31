import numpy as np
import matplotlib.pyplot as plt

def sph2cart(theta, phi, r=1.0):
    """Convert spherical (theta,phi) to Cartesian (x,y,z). 
    theta: polar angle from +Z (0..pi), phi: azimuth from +X (0..2pi)."""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

# --- Parameters for the highlighted region (in degrees) ---
theta0_deg = 50.0   # polar angle of region center
phi0_deg   = 40.0   # azimuth of region center
patch_radius_deg = 18.0  # angular half-size of the rectangular patch

# Convert to radians
theta0 = np.deg2rad(theta0_deg)
phi0   = np.deg2rad(phi0_deg)
patch_radius = np.deg2rad(patch_radius_deg)

# --- Create the figure (single 3D plot, no subplots) ---

fig = plt.figure(figsize=(7, 7))
fig.patch.set_facecolor("#d1d1d1")
#fig.patch.set_facecolor("#ADADAD")
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor("#d1d1d1")          # grigio chiaro per lo sfondo del grafico
#ax.set_facecolor("#ADADAD")
# --- Draw a light unit sphere for context ---
u = np.linspace(0, 2*np.pi, 60)
v = np.linspace(0, np.pi, 30)
X = np.outer(np.cos(u), np.sin(v))
Y = np.outer(np.sin(u), np.sin(v))
Z = np.outer(np.ones_like(u), np.cos(v))
ax.plot_surface(X, Y, Z, linewidth=0, alpha=0.15)

# --- Build a small spherical patch around (theta0, phi0) ---
theta_patch = np.linspace(theta0 - patch_radius, theta0 + patch_radius, 30)
phi_patch   = np.linspace(phi0   - patch_radius, phi0   + patch_radius, 30)
PH, TH = np.meshgrid(phi_patch, theta_patch)  
Xp, Yp, Zp = sph2cart(TH, PH, r=1.0)
ax.plot_surface(Xp, Yp, Zp, linewidth=0, alpha=0.6)

# --- Define a vector pointing close to the region, but not exactly into it ---
theta_vec = theta0
phi_vec   = phi0 + patch_radius * 1.3
vx, vy, vz = sph2cart(theta_vec, phi_vec, r=1.0)



# --- Add two more vectors farther from the region ---
theta_vec2 = theta0+patch_radius*2 # range 0...pi
print(theta_vec2)
phi_vec2   = phi0 + patch_radius * 3.0   # range 0...2pi
print(phi_vec2)

vx2, vy2, vz2 = sph2cart(theta_vec2, phi_vec2, r=1.0)
ax.quiver(0, 0, 0, vx2, vy2, vz2, length=1.0, normalize=True, color="green")
ax.text(vx2*1.1, vy2*1.1, vz2*1.1, "cat", color="green")

theta_vec3 = theta0
phi_vec3   = phi0 - patch_radius * 2.5   # dall'altro lato
vx3, vy3, vz3 = sph2cart(theta_vec3, phi_vec3, r=1.0)
ax.quiver(0, 0, 0, vx3, vy3, vz3, length=1.0, normalize=True, color="orange")
ax.text(vx3*1.1, vy3*1.1, vz3*1.1, "building", color="orange")





# Draw the arrow from the origin
ax.quiver(0, 0, 0, vx, vy, vz, length=1.0, normalize=True, color="red")

# --- Add axis lines through the origin ---
lim = 1.2
ax.plot([-lim, lim], [0, 0], [0, 0], color='k')  # X axis
ax.plot([0, 0], [-lim, lim], [0, 0], color='k')  # Y axis
ax.plot([0, 0], [0, 0], [-lim, lim], color='k')  # Z axis

# --- Labels and aspect ---
cx, cy, cz = sph2cart(theta0*0.8, phi0, r=1.15)
ax.text(cx, cy, cz, "birds (or things)\non top of buildings")
ax.text(vx*1.1, vy*1.1, vz*1.1, "cat on top of building", color="red")

#ax.set_title("Vector from origin near a highlighted region (unit sphere)")
#ax.set_xlabel("X")
#ax.set_ylabel("Y")
#ax.set_zlabel("Z")
ax.set_box_aspect([1, 1, 1])  
lim=lim-0.6
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim, lim)

# Remove the "cube" (background panes and edges)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.xaxis.pane.set_visible(False)
ax.yaxis.pane.set_visible(False)
ax.zaxis.pane.set_visible(False)

ax.xaxis._axinfo["grid"]['linewidth'] = 0
ax.yaxis._axinfo["grid"]['linewidth'] = 0
ax.zaxis._axinfo["grid"]['linewidth'] = 0

# Also hide the axis lines of the cube
ax.xaxis.line.set_color((1,1,1,0))  # transparent
ax.yaxis.line.set_color((1,1,1,0))
ax.zaxis.line.set_color((1,1,1,0))
#ax.xaxis.label.set_color((1,1,1,0))
#ax.yaxis.label.set_color((1,1,1,0))
#ax.zaxis.label.set_color((1,1,1,0))


vx_tip, vy_tip, vz_tip = vx, vy, vz
cx, cy, cz = sph2cart(theta0, phi0, r=1.0)
Tcx, Tcy, Tcz = sph2cart(theta0, phi0*1.15, r=1.0)

ax.plot([vx_tip, cx], [vy_tip, cy], [vz_tip, cz], color="blue", linestyle="--")
ax.text(Tcx,Tcy,Tcz,"distance",color="blue")
plt.show()

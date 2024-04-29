import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
  
# read the image 
img = cv2.imread('img000220_3.jpg') 
sh = img.shape
# print(sh)

center = np.array([((sh[0] - 1) / 2), ((sh[1] - 1) / 2)])

"""
# Total non-space pixels
total = 0
# cutoff val
k = 20
for i in range(sh[0]):
    for j in range(sh[1]):
        val = img[i][j]
        if val[0] > k or val[1] > k or val[2] > k:
            total += 1
            
print(total, sh[0]*sh[1])
"""

# Plotting assist
def get_rectangle(center, width, height, rot):
    tl = np.array([-width/2, height/2])
    tr = np.array([width/2, height/2])
    bl = np.array([-width/2, -height/2])
    br = np.array([width/2, -height/2])
    rot_mat = np.array([[np.cos(rot), -np.sin(rot)],
               [np.sin(rot), np.cos(rot)]])
    tl1 = rot_mat @ tl
    tr1 = rot_mat @ tr
    bl1 = rot_mat @ bl
    br1 = rot_mat @ br

    bl1[1] *= -1
    br1[1] *= -1
    tl1[1] *= -1
    tr1[1] *= -1

    xs = []
    ys = []
    # TL to TR
    xs1 = np.linspace(center[0] + tl1[0], center[0] + tr1[0])
    ys1 = np.linspace(center[1] + tl1[1], center[1] + tr1[1])

    # TR to BR
    xs2 = np.linspace(center[0] + tr1[0], center[0] + br1[0])
    ys2 = np.linspace(center[1] + tr1[1], center[1] + br1[1])

    # BR to BL
    xs3 = np.linspace(center[0] + br1[0], center[0] + bl1[0])
    ys3 = np.linspace(center[1] + br1[1], center[1] + bl1[1])

    # BL to TL
    xs4 = np.linspace(center[0] + bl1[0], center[0] + tl1[0])
    ys4 = np.linspace(center[0] + bl1[1], center[0] + tl1[1])

    return np.concatenate((xs1,xs2,xs3,xs4)), np.concatenate((ys1,ys2,ys3,ys4)), [tl1+center,tr1+center,bl1+center,br1+center]


img = mpimg.imread('img000220_3.jpg')


C = 1/2e8  # Tuning hyperparameter

# Objective function: largest sc/space ratio + rectangle size
def get_obj(corners, width, height):
    tl1 = corners[0]
    tr1 = corners[1]
    m1 = (tr1[1]-tl1[1])/(tr1[0]-tl1[0])
    bl1 = corners[2]
    br1 = corners[3]
    m2 = (br1[1]-bl1[1])/(br1[0]-bl1[0])

    # Total non-space pixels
    total = 0
    # cutoff val
    k = 20
    for i in range(sh[0]):
        for j in range(sh[1]):
            val = img[i][j]
            if j > tl1[0] and j < tr1[1] or j > bl1[0] and j < br1[0]:
                y_top = tl1[1] + m1 * (j - tl1[0]) # visual top, coordinate lower
                y_bot = bl1[1] + m2 * (j - bl1[0])
                if i > y_top and i < y_bot:
                    if val[0] > k or val[1] > k or val[2] > k:
                        total += 1
                        
    ratio = total / (width*height)
    size = width*height
    return ratio - C * (size - 26000) ** 2


plot = True

width = 400
height = 400

sf = 5

rot = 0*np.pi/180 # 0 radians

rf = 10 * np.pi/180

step = 100
rstep = 4

imgplot = plt.imshow(img)
xs, ys, corners = get_rectangle(center, width, height, rot)
obj = get_obj(corners, width, height)
plt.scatter(xs, ys, c='r')
plt.pause(0.05)
plt.clf()

for i in range(100):
    print()
    print("Iteration", i+1)
    xw, yw, cw = get_rectangle(center, width + sf, height, rot)
    objw = get_obj(cw, width + sf, height)
    xw2, yw2, cw2 = get_rectangle(center, width - sf, height, rot)
    objw2 = get_obj(cw2, width - sf, height)
    gw = (objw-objw2)/(2*sf)

    xh, yh, ch = get_rectangle(center, width, height + sf, rot)
    objh = get_obj(ch, width, height + sf)
    xh2, yh2, ch2 = get_rectangle(center, width, height - sf, rot)
    objh2 = get_obj(ch2, width, height - sf)
    gh = (objh-objh2)/(2*sf)

    xr, yr, cr = get_rectangle(center, width, height, rot + rf)
    objr = get_obj(cr, width, height)
    xr2, yr2, cr2 = get_rectangle(center, width, height, rot - rf)
    objr2 = get_obj(cr2, width, height)
    gr = (objr-objr2)/(2*rf)

    print(gw, gh, gr)
    
    width += step * gw

    height += step * gh

    rot += rstep * gr
    
    imgplot = plt.imshow(img)
    xs, ys, corners = get_rectangle(center, width, height, rot)
    obj = get_obj(corners, width, height)
    if plot:
        plt.scatter(xs, ys, c='r')
        plt.savefig('ml'+str(i+1)+'.png')
        plt.pause(0.05)
        plt.clf()
     
plt.show()

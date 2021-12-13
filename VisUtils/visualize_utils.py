import mayavi.mlab as mlab
import numpy as np

#from https://github.com/open-mmlab/OpenPCDet/tree/master/tools/visual_utils
def visualize_pts(pts, fig=None, bgcolor=(0, 0, 0), fgcolor=(1.0, 1.0, 1.0),
                  show_intensity=False, size=(600, 600), draw_origin=True):
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=fgcolor, engine=None, size=size)

    if show_intensity:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
    else:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
    if draw_origin:
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
        mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), tube_radius=0.1)

    return fig

#from Kitti.viz_util.py
def draw_lidar(
    pc,
    color=None,
    fig=None,
    bgcolor=(0, 0, 0),
    pts_scale=0.3,
    pts_mode="sphere",
    pts_color=None,
    color_by_intensity=False,
    pc_label=False,
    drawfov=False,
    drawregion=False,
    point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]
):
    """ Draw lidar points
    Args:
        pc: numpy array (n,3) of XYZ
        color: numpy array (n) of intensity or whatever
        fig: mayavi figure handler, if None create new one otherwise will use it
    Returns:
        fig: created or used fig
    """
    # ind = (pc[:,2]< -1.65)
    # pc = pc[ind]
    pts_mode = "point"
    print("====================", pc.shape)
    if fig is None:
        fig = mlab.figure(
            figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000)
        )
    if color is None:
        color = pc[:, 2] #Z height
    if pc_label:
        color = pc[:, 4]
    if color_by_intensity:
        #color = pc[:, 2]
        intensities=pc[:, 3]
        maxintensity=max(intensities)
        max_index = np.argmax(intensities, axis=0)
        print(intensities[max_index])
        print(pc[max_index,:])
        minintensity=min(intensities)
        color=np.sqrt(intensities)*10#(intensities-minintensity)

    mlab.points3d(
        pc[:, 0],
        pc[:, 1],
        pc[:, 2],
        color,
        color=pts_color,
        mode=pts_mode,
        colormap="gnuplot",
        scale_factor=pts_scale,
        figure=fig,
    )

    # draw origin
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode="sphere", scale_factor=0.2)

    # draw axis
    axes = np.array(
        [[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]],
        dtype=np.float64,
    )
    #plot3d: Draws lines between points, the positions of the successive points of the line
    mlab.plot3d(
        [0, axes[0, 0]],
        [0, axes[0, 1]],
        [0, axes[0, 2]],
        color=(1, 0, 0),#red, X (0,0,0)->(2,0,0)
        tube_radius=None,
        figure=fig,
    )
    mlab.text3d(axes[0, 0], axes[0, 1], axes[0, 2], "X", scale=(0.1, 0.1, 0.1)) #(2,0,0) position

    mlab.plot3d(
        [0, axes[1, 0]],
        [0, axes[1, 1]],
        [0, axes[1, 2]],
        color=(0, 1, 0),#green green, Y (0,2,0)
        tube_radius=None,
        figure=fig,
    )
    mlab.text3d(axes[1, 0], axes[1, 1], axes[1, 2], "Y", scale=(0.1, 0.1, 0.1)) #(0,2,0) position

    mlab.plot3d(
        [0, axes[2, 0]],
        [0, axes[2, 1]],
        [0, axes[2, 2]],
        color=(0, 0, 1),#blue Z (0,0,2)
        tube_radius=None,
        figure=fig,
    )
    mlab.text3d(axes[2, 0], axes[2, 1], axes[2, 2], "Z", scale=(0.1, 0.1, 0.1)) #(0,0,2) position

    if drawfov:
        # draw fov (todo: update to real sensor spec.)
        fov = np.array(
            [[20.0, 20.0, 0.0, 0.0], [20.0, -20.0, 0.0, 0.0]], dtype=np.float64  # 45 degree
        )

        mlab.plot3d(
            [0, fov[0, 0]],
            [0, fov[0, 1]],
            [0, fov[0, 2]],
            color=(1, 1, 1),
            tube_radius=None,
            line_width=1,
            figure=fig,
        )
        mlab.plot3d(
            [0, fov[1, 0]],
            [0, fov[1, 1]],
            [0, fov[1, 2]],
            color=(1, 1, 1),
            tube_radius=None,
            line_width=1,
            figure=fig,
        )

    if drawregion:
        #point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1] # 0:xmin, 1: ymin, 2: zmin, 3: xmax, 4: ymax, 5: zmax
        # draw square region
        x1 = point_cloud_range[0]#TOP_X_MIN
        x2 = point_cloud_range[3]#TOP_X_MAX
        y1 = point_cloud_range[1]#TOP_Y_MIN
        y2 = point_cloud_range[4]#TOP_Y_MAX
        linewidth=0.2
        tuberadius=0.01 #0.1
        mlab.plot3d(
            [x1, x1],
            [y1, y2],
            [0, 0],
            color=(0.5, 0.5, 0.5),
            tube_radius=tuberadius,
            line_width=linewidth,
            figure=fig,
        )
        mlab.plot3d(
            [x2, x2],
            [y1, y2],
            [0, 0],
            color=(0.5, 0.5, 0.5),
            tube_radius=tuberadius,
            line_width=linewidth,
            figure=fig,
        )
        mlab.plot3d(
            [x1, x2],
            [y1, y1],
            [0, 0],
            color=(0.5, 0.5, 0.5),
            tube_radius=tuberadius,
            line_width=linewidth,
            figure=fig,
        )
        mlab.plot3d(
            [x1, x2],
            [y2, y2],
            [0, 0],
            color=(0.5, 0.5, 0.5),
            tube_radius=tuberadius,
            line_width=linewidth,
            figure=fig,
        )

    # mlab.orientation_axes()
    mlab.view(
        azimuth=180,
        elevation=70,
        focalpoint=[12.0909996, -1.04700089, -2.03249991],
        distance=62.0,
        figure=fig,
    )
    return fig

def draw_gt_boxes3d(
    gt_boxes3d,
    fig,
    color=(1, 1, 1),
    line_width=1,
    draw_text=True,
    text_scale=(1, 1, 1),
    color_list=None,
    label=""
):
    """ Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    """
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n]
        if draw_text:
            mlab.text3d(
                b[4, 0],
                b[4, 1],
                b[4, 2],
                label,
                scale=text_scale,
                color=color,
                figure=fig,
            )
        for k in range(0, 4):
            # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i, j = k, (k + 1) % 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=color,
                tube_radius=None,
                line_width=line_width,
                figure=fig,
            )

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=color,
                tube_radius=None,
                line_width=line_width,
                figure=fig,
            )

            i, j = k, k + 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=color,
                tube_radius=None,
                line_width=line_width,
                figure=fig,
            )
    # mlab.show(1)
    # mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig
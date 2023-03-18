import itertools
import copy

import numpy as np
import networkx as nx

import io
import cv2

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.patches import Rectangle

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axisartist import floating_axes

from .colors import ripples_blue

def plot_with_colors(ax, x, y_values, color_values, cmap='jet'):
    fig = ax.figure
    
    vmin = color_values.min()
    vmax = color_values.max()
    
    cmap_obj = plt.get_cmap(cmap)
    
    for c, y in zip(color_values, y_values):
        color = (c - vmin)/(vmax - vmin)
        ax.plot(x, y, color=cmap_obj(color))
        
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, ax=ax)
    
def _create_colors(data, norm_values, cmap='RdBu_r'):
    vmin, vmax = norm_values
    norm_obj = mpl.colors.Normalize(vmin, vmax)
    cmap_obj = plt.get_cmap(cmap)
    
    res = {k:((0.5,0.5,0.5,1.0) if np.isnan(v) else cmap_obj(norm_obj(v))) for (k,v) in data.items()}
    
    return res

def draw_four_views(axes, data, parcel_names, surfaces, n_parcels=400, cmap='viridis', norm=None, title=None, norm_values=None, plot_data=None, 
                   title_kwargs=None, cbar_loc='left', cbar_ax_kwargs=None, significant_parcels=None, draw_borders=False):
    
    if cbar_ax_kwargs is None:
        cbar_ax_kwargs = dict()
        
    if title_kwargs is None:
        title_kwargs = dict()
    
    if not('fontsize' in title_kwargs):
        title_kwargs['fontsize'] = 7

    slices = [slice(0,n_parcels//2), slice(n_parcels//2,n_parcels)]
    
    if (norm_values is None):
        vmin, vmax =  np.nanpercentile(data, (1,99))
    else:
        vmin, vmax = norm_values
    
    if plot_data is None:
        plot_data = {n:v for (n,v) in zip(parcel_names, data)}

    if any([type(v) is tuple for v in data]):
        plot_data_colors = plot_data
    else:
        plot_data_colors = _create_colors(plot_data, norm_values, cmap=cmap)
                   
                   
    for hemi_idx, (bs, parcel_slice) in enumerate(zip(surfaces, slices)):
        # bs.set_data(plot_data_colors, significant_parcels)
        bs.set_data(plot_data_colors)

        zoom = 1.6
        
        for view_idx in range(2):
            camera_pos = (-1,0,0) if (hemi_idx == view_idx) else (1,0,0)
        
            bs.plot(show=False, camera_position=camera_pos, zoom=zoom, cmap=cmap, lightning='three lights', 
                    vmin=vmin, vmax=vmax, norm=norm, draw_borders=draw_borders)

            img = bs.plotter.screenshot(return_img=True)
            axes[view_idx, hemi_idx].imshow(img)
            axes[view_idx, hemi_idx].set_axis_off()
    
    if norm is None:
        cbar_norm = mpl.colors.Normalize(vmin, vmax)
    elif norm == 'log_discrete':
        bounds = np.geomspace(vmin, vmax, 10)
        cbar_norm = mpl.colors.BoundaryNorm(bounds, plt.get_cmap(cmap).N)
    elif norm == 'log':
        cbar_norm = mpl.colors.LogNorm(vmin, vmax)
    
    if cbar_loc == 'left':
        if not('bbox_to_anchor' in cbar_ax_kwargs):
            cbar_ax_kwargs['bbox_to_anchor'] = (-0.15,0.35,0.7,1.35)
        
        if not('height' in cbar_ax_kwargs):
            cbar_ax_kwargs['height'] = '60%'

        cbar_ax = inset_axes(axes[1,0], width="10%", loc=2, bbox_transform=axes[1,0].transAxes, **cbar_ax_kwargs) 
        orientation = 'vertical'
    elif cbar_loc == 'bottom':
        if not('bbox_to_anchor' in cbar_ax_kwargs):
            cbar_ax_kwargs['bbox_to_anchor'] = (0.3,-0.2,1.3,0.8)
            
        cbar_ax = inset_axes(axes[1,0], width="40%", height="10%", loc=8, bbox_transform=axes[1,0].transAxes, **cbar_ax_kwargs) 
        orientation = 'horizontal'
    
    
    cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=plt.get_cmap(cmap), orientation=orientation, norm=cbar_norm)
    cbar_ax.yaxis.set_ticks_position('left')
    
    if  norm == 'log_discrete':
        cbar.set_ticks(cbar.get_ticks()[::3])
    elif norm == 'log':
        cbar.set_ticks(np.geomspace(vmin, vmax, 4))
       
    cbar.outline.set_linewidth(0.25)
    axes[0,0].set_title(title,  x=1.15, **title_kwargs)


def plot_rectangles(ax, comm_to_nodes, colors=None):
    prev_size = 0
    for i in range(len(comm_to_nodes)):
        size = len(comm_to_nodes[i])
    
        edgecolor = 'black' if colors is None else colors[i]

        patch = Rectangle([prev_size - 0.5, prev_size- 0.5], size, size, linewidth=1, edgecolor=edgecolor, facecolor='none')

        ax.add_patch(patch)

        prev_size += size

def setup_diamond_axes(fig, rect, low, high):
    """
    A simple one.
    """
    tr = Affine2D().rotate_deg(45)

    grid_helper = floating_axes.GridHelperCurveLinear(
        tr, extremes=(low, high, low, high))

    ax1 = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
    fig.add_subplot(ax1)

    aux_ax = ax1.get_aux_axes(tr)

    grid_helper.grid_finder.grid_locator1._nbins = 4
    grid_helper.grid_finder.grid_locator2._nbins = 4

    return ax1, aux_ax


def create_contour_pixel(image):
    x = np.linspace(0, image.shape[1], image.shape[1]*10+1)[:-1]
    y = np.linspace(0, image.shape[0], image.shape[0]*10+1)[:-1]

    X, Y = np.meshgrid(x,y)

    f = lambda x,y: image[int(y),int(x) ]
    g = np.vectorize(f)
    Z = g(X,Y)
       
    return Z, x, y

def draw_contour_pixel(ax, x, y, Z):
    Z_px, x_img, y_img = create_contour_pixel(Z)
    
    x_px = np.geomspace(x[0], x[~0], len(x_img))
    y_px = np.geomspace(y[0], y[~0], len(y_img))
    
    ax.contour(x_px, y_px, Z_px, [0.5])

# do I need it?
def draw_contour(ax, X, Y, Z, levels, min_size):
    CS = ax.contour(X, Y, Z, levels)
    
    for level in CS.collections:
        for kp,path in reversed(list(enumerate(level.get_paths()))):
            # go in reversed order due to deletions!

            # include test for "smallness" of your choice here:
            # I'm using a simple estimation for the diameter based on the
            #    x and y diameter...
            verts = path.vertices # (N,2)-shape array of contour line coordinates
            diameter = np.max(verts.max(axis=0) - verts.min(axis=0))

            if diameter < min_size: # threshold to be refined for your actual dimensions!
                del(level.get_paths()[kp])  # no remove() for Path objects:(


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def circular_layout(G, scale=1, center=None, dim=2, theta_offset=0.0):
    if dim < 2:
        raise ValueError("cannot handle dimensions < 2")

    G, center = nx.layout._process_params(G, center, dim)

    paddims = max(0, (dim - 2))

    if len(G) == 0:
        pos = {}
    elif len(G) == 1:
        pos = {nx.utils.arbitrary_element(G): center}
    else:
        # Discard the extra angle since it matches 0 radians.
        theta = np.linspace(0, 1, len(G) + 1)[:-1] * 2 * np.pi + theta_offset
        theta = theta.astype(np.float32)
        
        pos = np.column_stack(
            [np.cos(theta), np.sin(theta), np.zeros((len(G), paddims))]
        )
        pos = nx.layout.rescale_layout(pos, scale=scale) + center
        pos = dict(zip(G, pos))

    return pos

def _compute_angle(pos_x, pos_y, ax=None, label_pos=0.5, fix_rotation=True):
    x1, y1 = pos_x
    x2, y2 = pos_y

    (x, y) = (
        x1 * label_pos + x2 * (1.0 - label_pos),
        y1 * label_pos + y2 * (1.0 - label_pos),
    )


    angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
    # make label orientation "right-side-up"
    if fix_rotation:
        if angle > 90:
            angle -= 180
        if angle < -90:
            angle += 180
    # transform data coordinate angle to screen coordinate angle
    
#     if ax is None:
    if True:
        trans_angle = angle
    else:
        xy = np.array((x, y))

        trans_angle = ax.transData.transform_angles(
            np.array((angle,)), xy.reshape((1, 2))
        )[0]

    return trans_angle

def plot_multigraph_edges(G, pos, features, R=0.2, node_size=1000, ax=None, angle_offset=0.5):
    ax_to_plot = plt.subplots()[1] if ax is None else ax
    
    center = np.mean(list(pos.values()), axis=0)
#     ax_to_plot.scatter([center[0]], [center[1]], color='red', s=50)
    
    for x, y in itertools.combinations(G.nodes, 2):
        pos_x = pos[x]
        pos_y = pos[y]
        
        label_pos = (pos_x + pos_y)/2
        
        label_angle = _compute_angle(pos_x, pos_y, ax=ax)
        angle = np.deg2rad(_compute_angle(pos_x, pos_y, ax=ax, fix_rotation=False))

        edge_data = G.get_edge_data(x, y)

        if edge_data is None:
            continue
        
        if len(edge_data) == 1:
            offsets = [0]
        else:
            offsets = np.linspace(-angle_offset, angle_offset, len(edge_data))
        
        if _compute_angle(pos_x, center) > label_angle:
            offset_coefficients = itertools.cycle([1,-1])
            line_styles = itertools.cycle(['solid', 'solid'])
#             line_colors = itertools.cycle(['black', 'gray'])
            line_colors = itertools.cycle(['silver', 'black'][::-1])
            z_orders = itertools.cycle([0, 1])
            
        else:
            offset_coefficients = itertools.cycle([1,-1])
            line_styles = itertools.cycle(['solid', 'solid'])
#             line_colors = itertools.cycle(['black', 'gray'])
            line_colors = itertools.cycle(['silver', 'black'][::-1])
            z_orders = itertools.cycle([0, 1])
            offsets = offsets[::-1]
            
        for fname, edge_offset, label_offset_coeff, ls, color, zo in zip(features, offsets, offset_coefficients, 
                                                                     line_styles, line_colors, z_orders):
            edge_value = edge_data[fname]
                
            pos_c1 = pos_x
            pos_c2 = pos_y

            lw = np.abs(edge_value)**2*15
            
            ax_to_plot.plot([pos_c1[0], pos_c2[0]], [pos_c1[1], pos_c2[1]], zorder=zo, lw=lw, color=color, ls=ls)
            
            label_pos = (pos_c1 + pos_c2)/2
            
            label_offset_angle = np.deg2rad(_compute_angle(center, label_pos, ax=ax, fix_rotation=False))
            label_offset_vector = np.array([np.cos(label_offset_angle), np.sin(label_offset_angle)])
            
            label_pos += label_offset_vector*label_offset_coeff*(0.25 + lw/50)
            
            ax.text(*label_pos, edge_value, rotation=label_angle, fontsize=6, 
                    horizontalalignment='center', verticalalignment='center')
            
def draw_correlation_polygon(values, values_lower, names, label, ax, x_offset=0, label_y=1.75, 
                             theta_offset=None, label_x=None, pos=None, node_color=ripples_blue, features=['corr', 'pcorr']):
    coords = [(0,0), (2,0), (1,1)]
    coords = [(x+x_offset,y) for (x,y) in coords]
    
    if label_x is None:
        label_x = x_offset
    
    G = nx.Graph()
    
    G.add_nodes_from(names)
    G.add_edge(names[0], names[1], corr=values[0].round(2), pcorr=values_lower[0].round(2))
    G.add_edge(names[0], names[2], corr=values[1].round(2), pcorr=values_lower[1].round(2))
    G.add_edge(names[1], names[2], corr=values[2].round(2), pcorr=values_lower[2].round(2))    

    colors = ['red' if G[u][v]['corr'] > 0 else 'blue' for u,v in G.edges()]

    if theta_offset is None:
        n_nodes = len(G)
        inner_deg = 180  / n_nodes
        circle_deg = inner_deg / 2
        
        theta_offset = -np.deg2rad(circle_deg + inner_deg*2)
    
    if pos is None:
        graph_pos = circular_layout(G, theta_offset=theta_offset)
    else:
        graph_pos = copy.deepcopy(pos)
    
    for key in graph_pos.keys():
        graph_pos[key][0] += x_offset
    
    label_coord = (label_x, label_y)
    ax.text(*label_coord, label, fontsize=7, clip_on=False)
    
    nx.draw_networkx_nodes(G, pos=graph_pos, node_color='white', edgecolors=node_color, linewidths=3.0, node_size=1000, ax=ax)
    nx.draw_networkx_labels(G, pos=graph_pos, font_size=7, ax=ax)
    
    plot_multigraph_edges(G, graph_pos, features=features, ax=ax, R=0.1)
    
    ax.set_axis_off()

def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img
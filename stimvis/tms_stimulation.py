#!/usr/bin/env python
# coding: utf-8


import numpy as np
import copy
import pickle

from dipy.io.streamline import load_trk
from dipy.io.image import load_nifti

from dipy.viz import actor, ui
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)

from dipy.tracking import streamline

import simnibs
from simnibs import sim_struct, run_simnibs

import vtkplotter

from fury.data import read_viz_icons, fetch_viz_icons
from fury.io import load_polydata
from fury import window, utils
from fury.utils import vtk

# a function to compute 1st-order numerical derivative using a 3-point schema
# t - a point (index in the 'line' array) at which derivative should be computed
# line - array representing a function
# h - step between the points on the line


def my_deriv(t, line, h=1):
    if t == 0:
        return (-3*line[t]+4*line[t+1]-line[t+2])/(2*h)
    elif t == len(line)-1:
        return (line[t-2]-4*line[t-1]+3*line[t])/(2*h)
    else:
        return (line[t+1]-line[t-1])/(2*h)


# This function is to run simulations of the induced magnetic field using simnibs software

def simulation(fnamehead, pathfem, pos_centre=[-74.296158, -10.213354, 28.307243], pos_ydir=[-74.217369, -37.293205, 20.05232], distance=4, current_change=1e6):
    # Initalize a session
    s = sim_struct.SESSION()
    # Name of head mesh
    s.fnamehead = fnamehead
    # Output folder
    s.pathfem = pathfem
    # Not to visualize results in gmsh when running simulations (else set to True)
    s.open_in_gmsh = False
    # Initialize a list of TMS simulations
    tmslist = s.add_tmslist()
    # Select coil. For full list of available coils, please see simnibs documentation
    tmslist.fnamecoil = 'Magstim_70mm_Fig8.nii.gz'

    # Initialize a coil position
    pos = tmslist.add_position()
    pos.centre = pos_centre  # Place the coil over
    pos.pos_ydir = pos_ydir  # Point the coil towards
    pos.distance = distance  # Distance between coil and head
    pos.didt = current_change  # Rate of change of current in the coil, in A/s.
    run_simnibs(s)


# a function to define new colors to display the calculated stimulation effect
def calculate_new_colors(colors, bundle_native, effective_field, effect_min, effect_max):
    for my_streamline in range(len(bundle_native)-1):
        my_stream = copy.deepcopy(bundle_native[my_streamline])
        for point in range(len(my_stream)):
            colors[my_streamline][point] = vtkplotter.colors.colorMap(effective_field[my_streamline][point, 2], name='jet', vmin=effect_min, vmax=effect_max)

    colors[my_streamline+1] = vtkplotter.colors.colorMap(effect_min, name='jet', vmin=effect_min, vmax=effect_max)
    return colors


def load_elems(nodes, elems):

    import numpy as np

    elems = elems[elems[:, 3] != -1, :]
    # Computing rectangles
    tmp = nodes[elems-1, :]
    elems_min = tmp.min(axis=1)
    elems_max = tmp.max(axis=1)
    tmp = 0
    sizes = (elems_max-elems_min).max(axis=0)
    # It is the index to reduce the elements to check
    order_min = np.argsort(elems_min[:, 0])
    return {"Nodes": nodes, "Elems": elems, "El_min": elems_min, "El_max": elems_max, "Sizes": sizes, "Order_min": order_min}


def get_ttrd(loaded_elems, point):
    import numpy as np

    nodes = loaded_elems["Nodes"]
    elems = loaded_elems["Elems"]
    elems_min = loaded_elems["El_min"]
    elems_max = loaded_elems["El_max"]
    sizes = loaded_elems["Sizes"]
    order_min = loaded_elems["Order_min"]

    # Binary search to reduce the iterating points from 4mln to around 200k.
    r = np.searchsorted(elems_min[:, 0], point[0], side='left', sorter=order_min)
    l = np.searchsorted(elems_min[:, 0], point[0] - sizes[0], side='right', sorter=order_min)
    # Projection the data to only these points
    e_max = elems_max[order_min[l:r]]
    e_min = elems_min[order_min[l:r]]

    # Checks which ttrds are possible to contain the point
    potential_ttrds = order_min[l:r][(point[0] <= e_max[:, 0]) & (e_min[:, 1] <= point[1]) & (point[1] <= e_max[:, 1]) & (e_min[:, 2] <= point[2]) & (point[2] <= e_max[:, 2])]

    # It checks if the ttrd contains the point
    def check_ttrd(ttrd, point):
        coord = np.column_stack((ttrd[1, :] - ttrd[0, :], ttrd[2, :] - ttrd[0, :], ttrd[3, :] - ttrd[0, :]))
        coord = np.linalg.inv(coord).dot(point - ttrd[0, :])
        return coord.min() >= 0 and coord.sum() <= 1
    # It checks if the ttrd with num ttrdNum contains the point

    def check_ttrd_byNum(ttrdNum, point):
        ttrd = nodes[elems[ttrdNum]-1]
        return check_ttrd(ttrd, point)

    # Just takes all ttrds that contain points
    nodeIndices = elems[[x for x in potential_ttrds if check_ttrd_byNum(x, point)]][0]
    ns = nodes[nodeIndices-1]

    norms = np.sum((ns-point)**2, axis=-1)**0.5
    weights = 1/(norms+1e-10)
    weights = weights / weights.sum()

    return {"Nodes": nodeIndices, "Weights": weights}


# In[15]:
# a function to get e-field vector at a given position [x,y,z]
def get_field(ttt, point, my_field):
    best_ttt = get_ttrd(ttt, point)
    return np.dot(my_field[best_ttt['Nodes']-1].T, best_ttt['Weights'])


# a function to calculate directional derivatives of the effective field at a given point [x,y,z]
def deriv_e_field(coordinates, e_field_nodes, LSD, ttt):

    step = 0.05

    x1 = coordinates[0]
    y1 = coordinates[1]
    z1 = coordinates[2]
    x0 = coordinates[0]-step
    x2 = coordinates[0]+step
    y0 = coordinates[1]-step
    y2 = coordinates[1]+step
    z0 = coordinates[2]-step
    z2 = coordinates[2]+step

    f_x0_y1_z1 = np.dot(get_field(ttt, np.asarray([x0, y1, z1]), e_field_nodes), LSD)
    f_x2_y1_z1 = np.dot(get_field(ttt, np.asarray([x2, y1, z1]), e_field_nodes), LSD)
    f_x1_y1_z1 = np.dot(get_field(ttt, np.asarray([x1, y1, z1]), e_field_nodes), LSD)
    f_x1_y0_z1 = np.dot(get_field(ttt, np.asarray([x1, y0, z1]), e_field_nodes), LSD)
    f_x1_y2_z1 = np.dot(get_field(ttt, np.asarray([x1, y2, z1]), e_field_nodes), LSD)
    f_x1_y1_z0 = np.dot(get_field(ttt, np.asarray([x1, y1, z0]), e_field_nodes), LSD)
    f_x1_y1_z2 = np.dot(get_field(ttt, np.asarray([x1, y1, z2]), e_field_nodes), LSD)

    gradx = my_deriv(1, [f_x0_y1_z1, f_x1_y1_z1, f_x2_y1_z1], step)
    grady = my_deriv(1, [f_x1_y0_z1, f_x1_y1_z1, f_x1_y2_z1], step)
    gradz = my_deriv(1, [f_x1_y1_z0, f_x1_y1_z1, f_x1_y1_z2], step)

    return np.dot([gradx, grady, gradz], LSD)


def change_TMS_effects(x, y, z):
    """
    Computes the TMS effects for a given coil position (x,y,z)
    according to the existing theoretical models
    see Silva et al. (2008) Elucidating the mechanisms and loci of neuronal excitation
    by transcranial magnetic stimulation using a finite element model of a cortical sulcus
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2693370/

    Parameters
    ----------
    x,y,z  :  float
        Coordinates of a stimulation coil.

    Returns
    -------
    my_lut  :  lookup_tablevtkLookupTable
        Lookup table for the colormap to be used when visualizing TMS effects over the streamlines

    my_colors : numpy array
        Contains colors encoing the TMS effects at each point of each of the streamlines

    effective_field  : numpy array, float
        Contains two components of the TMS effects and their sum for each point of each of the streamlines. Saved as a txt file.

    mesh_file  :  gmsh structure
        A gmsh structure containing information about the incuded electric field. Saved as a msh file.
    """

    l1 = 2  # membrane space constant 2mm
    l2 = l1**2
    effect_max = -1000000
    effect_min = 1000000
    position = [x-256/2, y-256/2, z-256/2]  # -256/2 because of a freesurfer RAS coordinate system
    current_out_dir = out_dir+str(x)+'_'+str(y)+'_'+str(z)
    simulation(mesh_path, current_out_dir, pos_centre=position)
    mesh_file = current_out_dir+'/'+subject_name+'_TMS_1-0001_Magstim_70mm_Fig8_nii_scalar.msh'
    field_mesh = simnibs.msh.read_msh(mesh_file)
    field_as_nodedata = field_mesh.elmdata[0].as_nodedata()
    field_at_nodes = field_as_nodedata.value
    ttt = load_elems(field_mesh.nodes.node_coord, field_mesh.elm.node_number_list)

    effective_field = copy.deepcopy(new_streams_T1_array)

    for stream in range(len(new_streams_T1_array)):
        my_steam = copy.deepcopy(new_streams_T1_array[stream])
        print('starting _'+str(stream)+' out of '+str(len(new_streams_T1_array)))
        for t in range(len(my_steam[:, 0])):
            # -256/2 because of a freesurfer RAS coordinate system
            x = my_steam[t, 0]-256/2
            y = my_steam[t, 1]-256/2
            z = my_steam[t, 2]-256/2
            xyz = np.asarray([x, y, z])

            field_vector_xyz = get_field(ttt, xyz, field_at_nodes)

            effective_field[stream][t, 0] = l1*np.dot(field_vector_xyz, streams_array_derivative[stream][t, :])
            effective_field[stream][t, 1] = l2*deriv_e_field(xyz, field_at_nodes, streams_array_derivative[stream][t, :], ttt)
            effective_field[stream][t, 2] = effective_field[stream][t, 0] + effective_field[stream][t, 1]
            if (effective_field[stream][t, 2] < effect_min):
                effect_min = effective_field[stream][t, 2]
            if effective_field[stream][t, 2] > effect_max:
                effect_max = effective_field[stream][t, 2]

    with open(current_out_dir+'/'+subject_name+'_effective_field.txt', 'wb') as f:
        pickle.dump(effective_field, f)
    f.close()

    my_lut = actor.colormap_lookup_table(scale_range=(effect_min, effect_max),
                                    hue_range=(0.4, 1.),
                                    saturation_range=(1, 1.))
    my_colors = calculate_new_colors(colors, bundle_native, effective_field, effect_min, effect_max)
    return my_lut, my_colors


def main():
    # reads the tractography data in trk format
    # extracts streamlines and the file header. Streamlines should be in the same coordinate system as the FA map (used later).
    # input example: '/home/Example_data/tracts.trk'
    tractography_file = input("Please, specify the file with tracts that you would like to analyse. File should be in the trk format. ")

    # streams, hdr = load_trk(tractography_file)  # for old DIPY version
    sft = load_trk(tractography_file, tractography_file)
    streams = sft.streamlines
    streams_array = np.asarray(streams)
    print('imported tractography data:'+tractography_file)

    # load T1fs_conform image that operates in the same coordinates as simnibs except for the fact the center of mesh
    # is located at the image center
    # T1fs_conform image should be generated in advance during the head meshing procedure
    # input example: fname_T1='/home/Example_data/T1fs_conform.nii.gz'

    fname_T1 = input("Please, specify the T1fs_conform image that has been generated during head meshing procedure. ")
    data_T1, affine_T1 = load_nifti(fname_T1)

    # load FA image in the same coordinates as tracts
    # input example:fname_FA='/home/Example_data/DTI_FA.nii'
    fname_FA = input("Please, specify the FA image. ")
    data_FA, affine_FA = load_nifti(fname_FA)

    print('loaded T1fs_conform.nii and FA images')

    # specify the head mesh file that is used later in simnibs to simulate induced electric field
    # input example:'/home/Example_data/SUBJECT_MESH.msh'
    mesh_path = input("Please, specify the head mesh file. ")

    last_slach = max([i for i, ltr in enumerate(mesh_path) if ltr == '/'])+1
    subject_name = mesh_path[last_slach:-4]

    # specify the directory where you would like to save your simulation results
    # input example:'/home/Example_data/Output'
    out_dir = input("Please, specify the directory where you would like to save your simulation results. ")
    out_dir = out_dir+'/simulation_at_pos_'

    # Co-registration of T1fs_conform and FA images. Performed in 4 steps.
    # Step 1. Calculation of the center of mass transform. Used later as starting transform.
    c_of_mass = transform_centers_of_mass(data_T1, affine_T1, data_FA, affine_FA)
    print('calculated c_of_mass transformation')

    # Step 2. Calculation of a 3D translation transform. Used in the next step as starting transform.
    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)
    level_iters = [10000, 1000, 100]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]
    affreg = AffineRegistration(metric=metric,
                        level_iters=level_iters,
                        sigmas=sigmas,
                        factors=factors)

    transform = TranslationTransform3D()
    params0 = None
    starting_affine = c_of_mass.affine
    translation = affreg.optimize(data_T1, data_FA, transform, params0,
                        affine_T1, affine_FA,
                        starting_affine=starting_affine)
    print('calculated 3D translation transform')

    # Step 3. Calculation of a Rigid 3D transform. Used in the next step as starting transform
    transform = RigidTransform3D()
    params0 = None
    starting_affine = translation.affine
    rigid = affreg.optimize(data_T1, data_FA, transform, params0,
                        affine_T1, affine_FA,
                        starting_affine=starting_affine)
    print('calculated Rigid 3D transform')

    # Step 4. Calculation of an affine transform. Used for co-registration of T1 and FA images.
    transform = AffineTransform3D()
    params0 = None
    starting_affine = rigid.affine
    affine = affreg.optimize(data_T1, data_FA, transform, params0,
                        affine_T1, affine_FA,
                        starting_affine=starting_affine)

    print('calculated Affine 3D transform')

    identity = np.eye(4)

    inv_affine_FA = np.linalg.inv(affine_FA)
    inv_affine_T1 = np.linalg.inv(affine_T1)
    inv_affine = np.linalg.inv(affine.affine)

    # transforming streamlines to FA space
    new_streams_FA = streamline.transform_streamlines(streams, inv_affine_FA)
    new_streams_FA_array = np.asarray(new_streams_FA)

    T1_to_FA = np.dot(inv_affine_FA, np.dot(affine.affine, affine_T1))
    FA_to_T1 = np.linalg.inv(T1_to_FA)

    # transforming streamlines from FA to T1 space
    new_streams_T1 = streamline.transform_streamlines(new_streams_FA, FA_to_T1)
    new_streams_T1_array = np.asarray(new_streams_T1)

    # calculating amline derivatives along the streamlines to get the local orientation of the streamlines

    streams_array_derivative = copy.deepcopy(new_streams_T1_array)

    print('calculating amline derivatives')
    for stream in range(len(new_streams_T1_array)):
        my_steam = new_streams_T1_array[stream]
        for t in range(len(my_steam[:, 0])):
            streams_array_derivative[stream][t, 0] = my_deriv(t, my_steam[:, 0])
            streams_array_derivative[stream][t, 1] = my_deriv(t, my_steam[:, 1])
            streams_array_derivative[stream][t, 2] = my_deriv(t, my_steam[:, 2])
            deriv_norm = np.linalg.norm(streams_array_derivative[stream][t, :])
            streams_array_derivative[stream][t, :] = streams_array_derivative[stream][t, :]/deriv_norm

    # to create a torus representing a coil in an interactive window

    torus = vtk.vtkParametricTorus()
    torus.SetRingRadius(5)
    torus.SetCrossSectionRadius(2)

    torusSource = vtk.vtkParametricFunctionSource()
    torusSource.SetParametricFunction(torus)
    torusSource.SetScalarModeToPhase()

    torusMapper = vtk.vtkPolyDataMapper()
    torusMapper.SetInputConnection(torusSource.GetOutputPort())
    torusMapper.SetScalarRange(0, 360)

    torusActor = vtk.vtkActor()
    torusActor.SetMapper(torusMapper)
    torusActor.SetPosition(30, 30, 27)

    list_streams_T1 = list(new_streams_T1)
    # adding one fictive bundle of length 1 with coordinates [0,0,0] to avoid some bugs with actor.line during visualization
    list_streams_T1.append(np.array([0, 0, 0]))

    bundle_native = list_streams_T1

    # generating a list of colors to visualize later the stimualtion effects
    effect_max = -1000000
    effect_min = 1000000
    colors = [np.random.rand(*current_streamline.shape) for current_streamline in bundle_native]

    for my_streamline in range(len(bundle_native)-1):
        my_stream = copy.deepcopy(bundle_native[my_streamline])
        for point in range(len(my_stream)):
            colors[my_streamline][point] = vtkplotter.colors.colorMap((effect_min+effect_max)/2, name='jet', vmin=effect_min, vmax=effect_max)

    colors[my_streamline+1] = vtkplotter.colors.colorMap(effect_min, name='jet', vmin=effect_min, vmax=effect_max)

    # Vizualization of fibers over T1
    i = 0
    j = 0
    k = 0
    number_of_stimulations = 0

    actor_line_list = []

    scene = window.Scene()
    scene.clear()
    scene.background((0.5, 0.5, 0.5))

    world_coords = False
    shape = data_T1.shape

    lut = actor.colormap_lookup_table(scale_range=(effect_min, effect_max),
                                        hue_range=(0.4, 1.),
                                        saturation_range=(1, 1.))

    actor_line_list.append(actor.line(bundle_native, colors, linewidth=5, fake_tube=True, lookup_colormap=lut))

    if not world_coords:
        image_actor_z = actor.slicer(data_T1, identity)
    else:
        image_actor_z = actor.slicer(data_T1, identity)

    slicer_opacity = 0.6
    image_actor_z.opacity(slicer_opacity)

    image_actor_x = image_actor_z.copy()
    x_midpoint = int(np.round(shape[0] / 2))
    image_actor_x.display_extent(x_midpoint,
                            x_midpoint, 0,
                            shape[1] - 1,
                            0,
                            shape[2] - 1)

    image_actor_y = image_actor_z.copy()
    y_midpoint = int(np.round(shape[1] / 2))
    image_actor_y.display_extent(0,
                            shape[0] - 1,
                            y_midpoint,
                            y_midpoint,
                            0,
                            shape[2] - 1)

    """
    Connect the actors with the scene.
    """

    scene.add(actor_line_list[0])
    scene.add(image_actor_z)
    scene.add(image_actor_x)
    scene.add(image_actor_y)

    show_m = window.ShowManager(scene, size=(1200, 900))
    show_m.initialize()

    """
    Create sliders to move the slices and change their opacity.
    """

    line_slider_z = ui.LineSlider2D(min_value=0,
                                max_value=shape[2] - 1,
                                initial_value=shape[2] / 2,
                                text_template="{value:.0f}",
                                length=140)

    line_slider_x = ui.LineSlider2D(min_value=0,
                                max_value=shape[0] - 1,
                                initial_value=shape[0] / 2,
                                text_template="{value:.0f}",
                                length=140)

    line_slider_y = ui.LineSlider2D(min_value=0,
                                max_value=shape[1] - 1,
                                initial_value=shape[1] / 2,
                                text_template="{value:.0f}",
                                length=140)

    opacity_slider = ui.LineSlider2D(min_value=0.0,
                                max_value=1.0,
                                initial_value=slicer_opacity,
                                length=140)

    """
    Сallbacks for the sliders.
    """
    def change_slice_z(slider):
        z = int(np.round(slider.value))
        image_actor_z.display_extent(0, shape[0] - 1, 0, shape[1] - 1, z, z)

    def change_slice_x(slider):
        x = int(np.round(slider.value))
        image_actor_x.display_extent(x, x, 0, shape[1] - 1, 0, shape[2] - 1)

    def change_slice_y(slider):
        y = int(np.round(slider.value))
        image_actor_y.display_extent(0, shape[0] - 1, y, y, 0, shape[2] - 1)

    def change_opacity(slider):
        slicer_opacity = slider.value
        image_actor_z.opacity(slicer_opacity)
        image_actor_x.opacity(slicer_opacity)
        image_actor_y.opacity(slicer_opacity)

    line_slider_z.on_change = change_slice_z
    line_slider_x.on_change = change_slice_x
    line_slider_y.on_change = change_slice_y
    opacity_slider.on_change = change_opacity

    """
    Сreate text labels to identify the sliders.
    """
    def build_label(text):
        label = ui.TextBlock2D()
        label.message = text
        label.font_size = 18
        label.font_family = 'Arial'
        label.justification = 'left'
        label.bold = False
        label.italic = False
        label.shadow = False
        label.background = (0, 0, 0)
        label.color = (1, 1, 1)
        return label

    line_slider_label_z = build_label(text="Z Slice")
    line_slider_label_x = build_label(text="X Slice")
    line_slider_label_y = build_label(text="Y Slice")
    opacity_slider_label = build_label(text="Opacity")

    """
    Create a ``panel`` to contain the sliders and labels.
    """

    panel = ui.Panel2D(size=(300, 200),
                    color=(1, 1, 1),
                    opacity=0.1,
                    align="right")
    panel.center = (1030, 120)

    panel.add_element(line_slider_label_x, (0.1, 0.75))
    panel.add_element(line_slider_x, (0.38, 0.75))
    panel.add_element(line_slider_label_y, (0.1, 0.55))
    panel.add_element(line_slider_y, (0.38, 0.55))
    panel.add_element(line_slider_label_z, (0.1, 0.35))
    panel.add_element(line_slider_z, (0.38, 0.35))
    panel.add_element(opacity_slider_label, (0.1, 0.15))
    panel.add_element(opacity_slider, (0.38, 0.15))

    scene.add(panel)

    """
    Create a ``panel`` to show the value of a picked voxel.
    """

    label_position = ui.TextBlock2D(text='Position:')
    label_value = ui.TextBlock2D(text='Value:')

    result_position = ui.TextBlock2D(text='')
    result_value = ui.TextBlock2D(text='')

    text2 = ui.TextBlock2D(text='Calculate')

    panel_picking = ui.Panel2D(size=(250, 125),
                            color=(1, 1, 1),
                            opacity=0.1,
                            align="left")
    panel_picking.center = (200, 120)

    panel_picking.add_element(label_position, (0.1, 0.75))
    panel_picking.add_element(label_value, (0.1, 0.45))

    panel_picking.add_element(result_position, (0.45, 0.75))
    panel_picking.add_element(result_value, (0.45, 0.45))

    panel_picking.add_element(text2, (0.1, 0.15))

    icon_files = []
    icon_files.append(('left', read_viz_icons(fname='circle-left.png')))
    button_example = ui.Button2D(icon_fnames=icon_files, size=(100, 30))
    panel_picking.add_element(button_example, (0.5, 0.1))

    def change_text_callback(i_ren, obj, button):
        text2.message = str(i)+' '+str(j)+' '+str(k)
        torusActor.SetPosition(i, j, k)
        lut, colors = change_TMS_effects(i, j, k)
        scene.rm(actor_line_list[0])
        actor_line_list.append(actor.line(bundle_native, colors, linewidth=5, fake_tube=True, lookup_colormap=lut))
        scene.add(actor_line_list[1])

        global number_of_stimulations
        global bar
        if number_of_stimulations > 0:
            scene.rm(bar)
        else:
            number_of_stimulations = number_of_stimulations + 1

        bar = actor.scalar_bar(lut)
        bar.SetTitle("TMS effect")

        bar.SetHeight(0.3)
        bar.SetWidth(0.10)  # the width is set first
        bar.SetPosition(0.85, 0.3)
        scene.add(bar)

        actor_line_list.pop(0)
        i_ren.force_render()

    button_example.on_left_mouse_button_clicked = change_text_callback

    scene.add(panel_picking)
    scene.add(torusActor)

    def left_click_callback(obj, ev):
        """Get the value of the clicked voxel and show it in the panel."""
        event_pos = show_m.iren.GetEventPosition()

        obj.picker.Pick(event_pos[0],
                    event_pos[1],
                    0,
                    scene)
        global i, j, k
        i, j, k = obj.picker.GetPointIJK()
        result_position.message = '({}, {}, {})'.format(str(i), str(j), str(k))
        result_value.message = '%.8f' % data_T1[i, j, k]
        torusActor.SetPosition(i, j, k)

    image_actor_z.AddObserver('LeftButtonPressEvent', left_click_callback, 1.0)

    global size
    size = scene.GetSize()

    def win_callback(obj, event):
        global size
        if size != obj.GetSize():
            size_old = size
            size = obj.GetSize()
            size_change = [size[0] - size_old[0], 0]
            panel.re_align(size_change)

    show_m.initialize()
    """
    Set the following variable to ``True`` to interact with the datasets in 3D.
    """
    interactive = True

    scene.zoom(2.0)
    scene.reset_clipping_range()

    if interactive:
        show_m.add_window_callback(win_callback)
        show_m.render()
        show_m.start()


if __name__ == "__main__":
    main()

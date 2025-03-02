"""nsd_mapdata
"""
import os
from os.path import join
import nibabel as nib
import numpy as np
import nibabel.freesurfer.mghformat as fsmgh
from scipy.ndimage import map_coordinates
from scipy import sparse
from tqdm import tqdm

def nsd_datalocation(base_path, dir0=None):
    """convenience function to find data on your system

    Args:
        dir0 ([str]): 'betas' | 'timeseries' | 'stimuli' | 'behaviour

    Returns: full path to the nsddata directories.

    """
    if dir0 is None:
        f = join(base_path, 'nsddata')
    elif dir0 == 'betas':
        f = join(base_path, 'nsddata_betas')
    elif dir0 == 'timeseries':
        f = join(base_path, 'nsddata_timeseries')
    elif dir0 == 'stimuli':
        f = join(base_path, 'nsddata_stimuli')
    elif dir0 == 'behaviour':
        f = join(base_path, 'nsddata', 'bdata', 'meadows')

    return f

def parse_case(sourcespace, targetspace, tdir):
    """parse_case

    Args:
        sourcespace (string): space in which the source data lies.
        targetspace (string): space to interpolate the source data to.
        tdir (string) : directory where the data lives.

    Returns:
        [int]: which case we are in.
    """
    hemi = None

    # figure out what case we are in
    if isinstance(sourcespace, list):
        casenum = 4
    elif sourcespace == 'fsaverage' or targetspace == 'fsaverage':
        casenum = 3
    elif targetspace[:3] == 'lh.' or targetspace[:3] == 'rh.':
        casenum = 2
    elif sourcespace[:2] == 'lh.' or sourcespace[:2] == 'rh.':
        casenum = 4
    else:
        casenum = 1

    if casenum == 4:
        if not isinstance(sourcespace, list):
            sourcespace = [sourcespace]

    # deal with basic setup
    if casenum == 1:
        tfile = os.path.join(f'{tdir}',
                             f'{sourcespace}-to-{targetspace}.nii.gz')
    elif casenum in (2, 3):
        if targetspace[:3] == 'lh.' or targetspace[:3] == 'rh.':
            hemi = targetspace[:3]
            tfile = os.path.join(
                f'{tdir}',
                f'{hemi}{sourcespace}-to-{targetspace[3:]}.mgz')
        else:
            # assert(ismember(sourcespace(1: 3), {'lh.' 'rh.'}))
            hemi = sourcespace[:3]

            tfile = os.path.join(
                f'{tdir}',
                f'{hemi}{sourcespace[3:]}-to-{targetspace}.mgz'
            )

    elif casenum == 4:
        tfile = []
        for c_space in sourcespace:
            hemi = c_space[:2]
            tfile.append(
                os.path.join(
                    f'{tdir}',
                    f'{hemi}.{targetspace}-to-{c_space[3:]}.mgz'
                )
            )

    return casenum, tfile

def load_transform(casenum, tfile):
    """load the transform file

    Args:
        casenum ([type]): [description]
        tfile ([type]): [description]

    Returns:
        [type]: [description]
    """
    # load transform
    if casenum == 1:
        a1_img = nib.load(tfile)
        a1_data = a1_img.get_fdata()  # X x Y x Z x 3
    elif casenum in (2, 3):
        # V x 3 (decimal coordinates) or V x 1 (index)
        a1_img = nib.load(tfile)
        a1_data = a1_img.get_fdata()
        # get rid of extra dims
        a1_data = a1_data.reshape([a1_data.shape[0], -1], order='F')
    elif casenum == 4:
        a1_data = []
        for p in tfile:
            a1_img = nib.load(p)
            a0_data = a1_img.get_fdata()
            a0_data = a0_data.reshape([a0_data.shape[0], -1], order='F')
            # V-across-differentsurfaces x 3 (decimal coordinates)
            a1_data.append(a0_data)
        # now we vertical stack
        a1_data = np.vstack(a1_data)

    return a1_data


def load_sourcedata(casenum, sourcedata):
    """load sourcedata if str filename is passed

    Args:
        casenum (int): data case
        sourcedata ([type]): str or ndarray

    Returns:
        [nd-array]: returns the data array if a str/path is passed

    """
    # load sourcedata
    if isinstance(sourcedata, list):
        sdatatemp = []
        # sourcedata here could already be a list of volumes, or a 
        # list of paths pointing to volumes
        for p in sourcedata:
            if isinstance(p, str):
                temp = nib.load(p).get_fdata()
                temp = temp.reshape([temp.shape[0], -1])
                sdatatemp.append(temp)
                # V-across-differentsurfaces x D
            else:
                sdatatemp.append(p)

            sourcedata = np.vstack(sdatatemp)

    elif isinstance(sourcedata, str):
        if casenum in (1, 2, 3):
            if sourcedata[-4:] == '.mgz':
                source_img = nib.load(sourcedata)
                sourcedata = source_img.get_fdata()
                sourcedata = sourcedata.reshape(
                    [sourcedata.shape[0], -1],
                    order='F')  # squish
            else:
                source_img = nib.load(sourcedata)
                sourcedata = source_img.get_fdata()
                # X x Y x Z x D

    else:
        print('data array passed')

    return sourcedata

def nsd_write_vol(data, res, outputfile, origin=None):
    """nsd_write_vol writes volumes to disk

    Args:
        data (nd-array): volumetric data to write
        res (float): data acquisition resolution (in mm)
        outputfile (filename/path): where to save
        origin (1d-array, optional): the origin point of the volume.
                                     Defaults to None.

    Raises:
        ValueError: [description]
    """

    data_class = data.dtype

    # create a default header
    header = nib.Nifti1Header()
    header.set_data_dtype(data_class)

    # affine
    affine = np.diag([res]*3 + [1])
    if origin is None:
        origin = (([1, 1, 1] + np.asarray(data.shape))/2)-1

    affine[0, -1] = -origin[0]*res
    affine[1, -1] = -origin[1]*res
    affine[2, -1] = -origin[2]*res

    # write the nifti volume
    img = nib.Nifti1Image(
        data,
        affine,
        header)

    img.to_filename(outputfile)


def nsd_write_fs(data, outputfile, fsdir):
    """similar to nsd_vrite_vol but for surface mgz

    Args:
        data (nd-array): the surface data
        outputfile (filename/path): where to save
        fsdir (path): we need to know where the fsdir is.

    Raises:
        ValueError: if wrong file name provided, e.g doesn't have
                    lh or rh in filename, error is raised.
    """

    # load template
    # load template
    if outputfile.find('lh.') != -1:
        hemi = 'lh'
    elif outputfile.find('rh.') != -1:
        hemi = 'rh'
    else:
        raise ValueError('wrong outpufile.')

    mgh0 = f'{fsdir}/surf/{hemi}.w-g.pct.mgh'

    if not os.path.exists(mgh0):
        mgh0 = f'{fsdir}/surf/{hemi}.orig.avg.area.mgh'

    img = fsmgh.load(mgh0)

    header = img.header
    affine = img.affine

    # Okay, make a new object now...
    vol_h = data[:, np.newaxis].astype(np.float64)
    v_img = fsmgh.MGHImage(vol_h, affine, header=header, extra={})

    v_img.to_filename(outputfile)

def isnotfinite(arr):
    """[utility function for finding non-finites]

    Args:
        arr (numpy array): array to find non-finites in

    Returns:
        [bool]: boolean indicating the non-finite elements
    """
    res = np.isfinite(arr)
    np.bitwise_not(res, out=res)  # in-place
    return res

def interp_wrapper(vol, coords, interptype='cubic'):
    """
     interp_wrapper(vol, coords, interptype)

     <vol> is a 3D matrix (can be complex-valued)
     <coords> is 3 x N with the matrix coordinates to interpolate at.
       one or more of the entries can be NaN.
     <interptype> (optional) is 'nearest' | 'linear' | 'cubic' | 'wta'.  
        default: 'cubic'.

     this is a convenient wrapper for ba_interp3.  the main problem with
     normal calls to ba_interp3 is that it assigns values to interpolation
     points that lie outside the original data range.  what we do is to
     ensure that coordinates that are outside the original field-of-view
     (i.e. if the value along a dimension is less than 1 or greater than
     the number of voxels in the original volume along that dimension)
     are returned as NaN and coordinates that have any NaNs are returned
     as NaN.

     another feature is 'wta' (winner-take-all). this involves the assumption
     that <vol> contains only discrete integers. each distinct integer is
     mapped as a binary volume (0s and 1s) using linear interpolation to each
     coordinate, the integer with the largest resulting value at that
     coordinate wins, and that coordinate is assigned the winning integer.

     for complex-valued data, we separately interpolate the real and imaginary
     parts.

     history:
     2019/09/01 - ported to python by ian charest

    """
    # input
    if interptype == 'cubic':
        order = 3
    elif interptype == 'linear':
        order = 1
    elif interptype == 'nearest':
        order = 0
    elif interptype == 'wta':
        order = 1  # linear
    else:
        raise ValueError('interpolation method not implemented.')

    # convert vol to float (needed)
    # vol = vol.astype(np.float32)

    # bad locations must get set to NaN
    bad = np.any(isnotfinite(coords), axis=0)
    coords[:, bad] = 1

    # out of range must become NaN, too
    bad = np.any(
        np.c_[
            bad,
            coords[0, :] < 1,
            coords[0, :] > vol.shape[0],
            coords[1, :] < 1,
            coords[1, :] > vol.shape[1],
            coords[2, :] < 1,
            coords[2, :] > vol.shape[2]], axis=1).astype(bool)

    # resample the volume
    if not np.any(np.isreal(vol)):
        # we interpolate the real and imaginary parts independently
        transformeddata = map_coordinates(
            np.nan_to_num(np.real(vol)).astype(np.float64),
            coords,
            order=order,
            mode='nearest') + 1j*map_coordinates(
                np.nan_to_num(np.imag(vol)).astype(np.float64),
                coords,
                order=order,
                mode='nearest')

    else:
        # this is the tricky 'wta' case
        if interptype == 'wta':

            # figure out the discrete integer labels
            alllabels = np.unique(vol.ravel())
            assert np.all(np.isfinite(alllabels))
            if len(alllabels) > 1000:
                print('warning: more than 1000 labels are present')

            # loop over each label
            allvols = []
            for c_label in alllabels:
                allvols.append(map_coordinates(
                    np.nan_to_num(vol == c_label).astype(np.float64),
                    coords,
                    order=order,
                    mode='nearest'
                    ))

            # make into a numpuy stack
            allvols = np.vstack(allvols)

            # which coordinates have no label contribution?
            realbad = np.sum(allvols, axis=0) == 0

            # perform winner-take-all (wta_is is the
            # index relative to alllabels!)
            wta_is = np.argmax(allvols, axis=0)

            # figure out the final labeling scheme
            transformeddata = alllabels[wta_is]

            # fill in NaNs for coordinates with no label
            # contribution and bad coordinates too
            transformeddata[realbad] = np.nan
            transformeddata[bad] = np.nan

        # this is the usual easy case
        else:
            # consider using mode constant with a cval.
            transformeddata = map_coordinates(
                np.nan_to_num(vol).astype(np.float64),
                coords,
                order=order,
                mode='nearest'
            )
            transformeddata[bad] = np.nan

    return transformeddata

def zerodiv(data1, data2, val=0, wantcaution=1):
    """zerodiv(data1,data2,val,wantcaution)
    Args:
        <data1>,<data2> are matrices of the same size or either
                        or both can be scalars.
        <val> (optional) is the value to use when <data2> is 0.
                        default: 0.
        <wantcaution> (optional) is whether to perform special
                        handling of weird cases (see below).
                        default: 1.
        calculate data1./data2 but use <val> when data2 is 0.
        if <wantcaution>, then if the absolute value of one or
                        more elements of data2 is less than 1e-5
                        (but not exactly 0), we issue a warning
                        and then treat these elements as if they
                        are exactly 0.
        if not <wantcaution>, then we do nothing special.

    note some weird cases:
    if either data1 or data2 is [], we return [].
    NaNs in data1 and data2 are handled in the usual way.

    """

    # handle special case of data2 being scalar
    if np.isscalar(data2):
        if data2 == 0:
            f = np.tile(val, data1.shape)
        else:
            if wantcaution and abs(data2) < 1e-5:
                print(
                    'warning: abs value of divisor is less than 1e-5.'
                    'treating the divisor as 0.')
                f = np.tile(val, data1.shape)
            else:
                f = data1/data2

    else:
        # do it
        bad = data2 == 0
        bad2 = abs(data2) < 1e-5
        if wantcaution and np.any(np.logical_and(bad2.ravel(), np.logical_not(bad.ravel()))):
            print(
                'warning: abs value of one or more divisors'
                'less than 1e-5.treating them as 0.')

        if wantcaution:
            data2[bad2] = 1
            f = data1/data2
            f[bad2] = val
        else:
            data2[bad] = 1
            f = data1/data2
            f[bad] = val

    return f

def mapsurfacetovolume(data, vertices, res, specialmode, emptyval):
    """mapsurfacetovolume(data, vertices, res, specialmode, emptyval)

    Args:
        data (nd-array): is the data with dimensionality n_datasets
                         (datasets) x V (vertices).
        vertices (nd-array): is 3 x V with the X-, Y-, and Z- coordinates
                         of the vertices.
        res (int): is the desired volume size. For example, 256 means
                         256 x 256 x 256.
        specialmode (bool): False means usual linear weighting. True
                         means treat each dataset as
            consisting of discrete integer labels and perform a
                         winner-take-all voting mechanism.
        emptyval ([type]): is the value to use when no vertices map
                         to a voxel

    Returns:
        targetdata [nd-array]: the data mapped to a volume in <targetdata>.
    """

    # calc/define
    n_vertices = vertices.shape[1]   # number of vertices
    n_voxels = res**3                # number of voxels
    n_datasets = data.shape[0]                # number of distinct datasets

    # prepare some sparse-related stuff
    vert_range = np.arange(n_vertices)

    # construct X [vertices x voxels,
    # each row has 8 entries with weights, the max for a weight is 3]
    x_old = sparse.coo_matrix((n_vertices, n_voxels))
    for x_n in [-1, 1]:
        for y_n in [-1, 1]:
            for z_n in [-1, 1]:

                # calc the voxel index and the distance
                # away from that voxel index
                if x_n == 1:
                    # ceil-val  (.1 means use weight of .9)
                    x_r = np.ceil(vertices[0, :]).astype(np.int)
                    x_d = x_r - vertices[0, :]
                else:
                    # val-floor (.1 means use weight of .9)
                    x_r = np.floor(vertices[0, :]).astype(np.int)
                    x_d = vertices[0, :] - x_r

                if y_n == 1:
                    y_r = np.ceil(vertices[1, :]).astype(np.int)
                    y_d = y_r - vertices[1, :]
                else:
                    y_r = np.floor(vertices[1, :]).astype(np.int)
                    y_d = vertices[1, :] - y_r

                if z_n == 1:
                    z_r = np.ceil(vertices[2, :]).astype(np.int)
                    z_d = z_r - vertices[2, :]
                else:
                    z_r = np.floor(vertices[2, :]).astype(np.int)
                    z_d = vertices[2, :] - z_r

                # calc # 1 x vertices with the voxel index to go to
                voxel_is = np.ravel_multi_index(
                    (x_r-1, y_r-1, z_r-1),
                    dims=(res, res, res),
                    order='F')
                # 1 x vertices with the weight to assign
                voxel_w = (1 - x_d) + (1 - y_d) + (1 - z_d)

                # construct the entries and add the old one in
                x_new = sparse.coo_matrix(
                    (voxel_w, (vert_range, voxel_is)),
                    shape=(n_vertices, n_voxels))
                x_new = x_old + x_new
                x_old = x_new

    # do it
    if specialmode == 0:

        # each voxel is assigned a weighted sum of vertex values.
        # this should be done as a weighted average.
        # thus, need to divide by sum of weights.
        # let's compute that now.
        wtssum = np.ones(n_vertices) * x_new   # 1 x voxels

        # take the vertex data and map to voxels
        transformeddata = data * x_new      # n_datasets x voxels

        # do the normalization
        # [if a voxel has no vertex contribution, it gets <emptyval>]
        transformeddata = zerodiv(
            transformeddata,
            np.tile(wtssum, n_datasets),
            emptyval)

        # prepare the results
        transformeddata = np.reshape(
            transformeddata.T,
            [res, res, res, n_datasets],
            order='F')

    else:

        # loop over datasets
        transformeddata = []
        for data_q in np.arange(n_datasets):

            # figure out discrete integer labels
            all_labels = np.unique(data[data_q, :]).astype(np.int).flatten()
            assert np.all(np.isfinite(all_labels))

            # expand data into separate channels
            # n_voxels x vertices
            data_new = np.zeros((len(all_labels), data.shape[1]))
            for c_label in all_labels:
                data_new[c_label, :] = data[data_q, :] == c_label

            # take the vertex data and map to voxels
            mapped = data_new*x_new      # n_voxels x voxels

            # which voxels have no vertex contribution?
            bad = np.sum(mapped, axis=0) == 0

            # perform winner-take-all
            # (mapped is the index relative to all_labels!)
            mapped = np.argmax(mapped, axis=0)

            # figure out the final labeling scheme
            finaldata = all_labels[mapped]

            # put in <emptyval>
            finaldata[bad] = emptyval

            # save
            transformeddata.append(
                np.reshape(
                    finaldata,
                    [res, res, res],
                    order='F')
                )

    return transformeddata

def transform_data(a1_data, sourcedata, tr_args):
    """transform_data

    Args:
        casenum (int): which case
        a1_data (nd-array): transformation map
        sourcedata (nd-array): data to be interpolated into target space
        tr_args (dict):
            casenum = tr_args['casenum']
            interptype = tr_args['interptype']
            targetspace = tr_args['targetspace']
            voxelsize = tr_args['voxelsize']
            res = tr_args['res']
            outputfile = tr_args['outputfile']
            outputclass = tr_args['outputclass']
            badval = tr_args['badval']
            fsdir = tr_args['fsdir']

    """
    # figure out if we have a 4d nifti as source
    n_dims = sourcedata.ndim

    # do it
    if tr_args['casenum'] == 1:    # volume-to-volume

        xdim, ydim, zdim, _ = a1_data.shape
        targetshape = (xdim, ydim, zdim)

        # construct coordinates
        coords = np.c_[a1_data[:, :, :, 0].ravel(order='F'),
                       a1_data[:, :, :, 1].ravel(order='F'),
                       a1_data[:, :, :, 2].ravel(order='F')].T

        # ensure that 9999 locations will propagate as NaN
        coords[coords == 9999] = np.nan
        coords = coords - 1  # coords is based on Kendrick's 1-based indexing.

        if n_dims == 4:
            # if a stack is passed
            transformeddata = []

            sourcedata = np.moveaxis(sourcedata, -1, 0)

            for sdata in tqdm(sourcedata, desc='volumes'):
                tmp = interp_wrapper(
                    sdata,
                    coords,
                    interptype=tr_args['interptype']).astype(
                        tr_args['outputclass'])

                tmp[np.isnan(tmp)] = tr_args['badval']
                tmp = np.reshape(tmp, targetshape, order='F')
                transformeddata.append(tmp)

            # reshape as a 4d volume
            transformeddata = np.moveaxis(np.asarray(transformeddata), 0, -1)
        else:

            transformeddata = interp_wrapper(
                sourcedata,
                coords,
                interptype=tr_args['interptype']).astype(
                    tr_args['outputclass'])

            transformeddata[np.isnan(transformeddata)] = tr_args['badval']
            transformeddata = np.reshape(
                transformeddata,
                targetshape,
                order='F')

        # if user wants a file, write it out
        if tr_args['outputfile'] is not None:
            if tr_args['targetspace'] == 'MNI':
                print('saving image in MNI space')

                transformeddata = np.flip(transformeddata, axis=0)
                origin = np.asarray([183-91, 127, 73]) - 1  # consider -1 here.

            else:
                origin = \
                    (([1, 1, 1] + np.asarray(transformeddata.shape[:3]))/2)-1

            nsd_write_vol(
                transformeddata,
                tr_args['voxelsize'],
                tr_args['outputfile'],
                origin=origin)

    elif tr_args['casenum'] == 2:    # volume-to-nativesurface

        # construct coordinates
        coords = np.c_[a1_data[:, 0].ravel(order='F'),
                       a1_data[:, 1].ravel(order='F'),
                       a1_data[:, 2].ravel(order='F')].T
        # ensure that 9999 locations will propagate as NaN
        coords[coords == 9999] = np.nan
        # coords is based on Kendrick's 1-based indexing.
        coords = coords - 1

        if n_dims == 4:
            transformeddata = []

            sourcedata = np.moveaxis(sourcedata, -1, 0)
            for sdata in tqdm(sourcedata, desc='volumes'):
                tmp = interp_wrapper(
                    sdata,
                    coords,
                    interptype=tr_args['interptype']).astype(
                        tr_args['outputclass'])

                tmp[np.isnan(tmp)] = tr_args['badval']
                transformeddata.append(tmp)

            # reshape as a n-dim volume
            transformeddata = np.moveaxis(np.asarray(transformeddata), 0, -1)
        else:
            transformeddata = interp_wrapper(
                sourcedata,
                coords,
                interptype=tr_args['interptype']).astype(
                    tr_args['outputclass'])

            transformeddata[np.isnan(transformeddata)] = tr_args['badval']

        # if user wants a file, write it out
        if tr_args['outputfile'] is not None:

            if tr_args['fsdir'] is None:
                raise ValueError('missing argument: fsdir')

            nsd_write_fs(
                transformeddata,
                tr_args['outputfile'],
                tr_args['fsdir'])

    # nativesurface-to-fsaverage  or  fsaverage-to-nativesurface
    elif tr_args['casenum'] == 3:

        # use nearest-neighbor and set the output class
        if n_dims == 1:
            transformeddata = \
                sourcedata[np.squeeze(a1_data.astype(int)) - 1].astype(
                    tr_args['outputclass'])
        elif n_dims > 1:
            transformeddata = \
                sourcedata[np.squeeze(a1_data.astype(int)) - 1, :].astype(
                    tr_args['outputclass'])
        # matlab based indexing in a1_data: 0-based in python

        # if user wants a file, write it out
        if tr_args['outputfile'] is not None:

            if tr_args['fsdir'] is None:
                raise ValueError('missing tr dict key: fsdir')

            nsd_write_fs(
                transformeddata,
                tr_args['outputfile'],
                tr_args['fsdir'])

    elif tr_args['casenum'] == 4:
        specialcase = 0
        if tr_args['interptype'] == 'surfacewta':
            specialcase = 1
        transformeddata = mapsurfacetovolume(
            sourcedata.T,
            a1_data.T,
            tr_args['res'],
            specialcase,
            tr_args['badval']
        )

        # reshape as a n-dim volume
        transformeddata = np.moveaxis(np.asarray(transformeddata), 0, -1).squeeze()

        # if user wants a file, write it out
        if tr_args['outputfile'] is not None:
            nsd_write_vol(
                transformeddata,
                tr_args['voxelsize'],
                tr_args['outputfile']
                )

    return transformeddata


class NSDmapdata():

    def __init__(self, base_dir):
        """[summary]

        Args:
            base_dir ([os.path]): directory where the nsd_data lives
        """
        self.base_dir = base_dir

    def fit(self,
            subjix,
            sourcespace,
            targetspace,
            sourcedata,
            interptype=None,
            badval=None,
            outputfile=None,
            outputclass=None,
            fsdir=None,
            ):
        """nsa_mapdata is used to map functional data between coordinate systems

        Arguments:
        __________

        subjix ([int]):  is the subject number 1-8

        sourcespace (['string']): is a string indicating the source space
                    (where the data currently are)

        targetspace (['string']): is a string indicating target space
                    (where the data need to go)

        sourcedata ([array or file]):
                    (1) one or more 3D volumes (X x Y x Z x D)
                    (2) a .nii or .nii.gz file with one or more 3D volumes
                    (3) one or more surface vectors (V x D)
                    (4) a .mgz file with one or more surface vectors

        interptype (['string', optional]): interpolation type. options are
                    'nearest' | 'linear' | 'cubic'. Default: 'cubic'.
                    Special cases are 'wta' and 'surfacewta'
                    (more details below).
        badval ([type], optional): is the value to use for invalid locations.
                    Defaults to None.

        outputfile (['string' or None]):
                    (1) a file.nii or file.nii.gz file to write to
                    (2) a [lh,rh].file.mgz file to write to
                    Default is None which means to not write out a file.

        outputclass ([string]): is the output format to use (e.g. 'single').
                    Default is to use the class of <sourcedata>. Note that
                    we always perform calculations in double format and then
                    convert at the end.

        fsdir (['path']):(optional) is the FreeSurfer subject directory for the
                    <targetspace>, like '/path/to/subj%02d' or
                    '/path/to/fsaverage'. We automatically sprintf the <subjix>
                    into <fsdir>. This input is needed only when writing .mgz
                    files.

        Returns:
        ________

        transformeddata: [array] data mapped to targetspace.


        There are four types of use-cases:

        (1) volume-to-volume:
        _____________________


        This includes [anat* | func* | MNI] -> [anat* | func* | MNI].
        Note that within-space transforms are not implemented
        (e.g. anat1pt0 to anat0pt8), but that is probably not very
        useful anyway.

        (2) volume-to-nativesurface:
        ____________________________


        This includes:

        [anat* | func* | MNI] -> [white | pial | layerB1 | layerB2 | layerB3].

        (3) nativesurface-to-fsaverage  or  fsaverage-to-nativesurface:
        _______________________________________________________________


        This includes [white] -> [fsaverage] and
                    [fsaverage] -> [white].
        In this case, note that nearest-neighbour
        is always used (<interptype> is ignored).

        (4) nativesurface-to-volume:
        ____________________________

        This includes [white | pial | layerB1 | layerB2 | layerB3] -> [anat*]
        In this case, a linear weighting scheme is always used, unless you
        specify <interptype> as 'surfacewta' which means to treat each
        dataset as containing discrete integers and perform a winner-take-all
        voting mechanism (this is useful for label data). Also, it is possible
        to supply data defined on multiple surfaces
        (e.g. layerB1 + layerB2 + layerB3) that are collectively mapped to
        volume. To do this, you should supply <sourcespace> as a cell vector
        of strings and supply <sourcedata> as a cell vector of things like
        cases (3) or (4) as described for <sourcedata> (see above). Note that
        it is okay to combine data defined on lh and rh surfaces!

        The valid strings for source and target spaces are:
        'anat0pt5'
        'anat0pt8'
        'anat1pt0'
        'func1pt0'
        'func1pt8'
        'MNI'
        '[lh,rh].white'
        '[lh,rh].pial'
        '[lh,rh].layerB1'
        '[lh,rh].layerB2'
        '[lh,rh].layerB3'
        'fsaverage'

        Map data from one space to another space. The data in the input
        variable <sourcedata> is mapped and returned in the output variable
        <transformeddata>.

        Details on the weighting scheme used for case (4) above:
            Each vertex contributes a linear kernel that has a size
            of exactly 2 x 2 x 2 voxels (at whatever the target
            anatomical resolution is). All of the linear kernels are added
            up, and values are obtained at the center of each volumetric
            voxel. In other words, the value associated with each voxel
            is simply a weighted average of vertices that are near that
            voxel (for example, within +/- 0.8 mm when targeting the
            anat0pt8 space). In the 'surfacewta' case, the integer labeling
            contributing the largest weight wins.

        Details on the 'wta' and 'surfacewta':
        These schemes are winner-take-all schemes. The sourcedata must
        consist of discrete integer labels. Each integer is separately
        mapped as a binary volume using linear interpolation, and the
        integer resulting in the largest value at a given location is
        assigned to that location.

        In the case of the target being MNI, we are going to write out LPI
        NIFTIs.
        so, we have to flip the first dimension so that the first voxel is
        indeed Left. also, in ITK-SNAP, the MNI template has world (ITK)
        coordinates at (0,0,0) which corresponds to voxel coordinates
        (91,127,73). These voxel coordinates are relative to RPI. So, for the
        origin of our LPI file that we will write, we need to make sure that we
        "flip" the first coordinate. The MNI volume has dimensions
        [182 218 182], so we subtract the first coordinate from 183.
        transformeddata = flipdim(transformeddata,1);  # now, it's in LPI
        origin = [183-91 127 73]

        """

        # setup
        nsd_path = nsd_datalocation(self.base_dir)
        tdir = os.path.join(f'{nsd_path}', 'ppdata',
                            f'subj{subjix:02d}', 'transforms')

        # set default interptype
        if interptype is None:
            interptype = 'cubic'

        # set default badval
        if badval is None:
            badval = 0

        # figure out which case
        casenum, tfile = parse_case(sourcespace, targetspace, tdir)

        # for writing target volumes, we need to know the voxel size
        if targetspace == 'anat0pt5':
            voxelsize = 0.5
            res = 512
        elif targetspace == 'anat0pt8':
            voxelsize = 0.8
            res = 320
        elif targetspace == 'anat1pt0':
            voxelsize = 1.0
            res = 256
        elif targetspace == 'func1pt0':
            voxelsize = 1.0
            res = None
        elif targetspace == 'func1pt8':
            voxelsize = 1.8
            res = None
        elif targetspace == 'MNI':
            voxelsize = 1
            res = None
        else:
            voxelsize = None
            res = None

        # load transform
        a1_data = load_transform(casenum, tfile)

        # load sourcedata
        sourcedata = load_sourcedata(casenum, sourcedata)

        sourceclass = sourcedata.dtype

        # deal with outputclass
        if outputclass is None:
            outputclass = sourceclass

        # collect arguments for transform_data
        transform_args = {
            'casenum': casenum,
            'sourcespace': sourcespace,
            'targetspace': targetspace,
            'interptype': interptype,
            'badval': badval,
            'outputfile': outputfile,
            'outputclass': outputclass,
            'voxelsize': voxelsize,
            'res': res,
            'fsdir': fsdir}

        # apply transform
        transformeddata = transform_data(
            a1_data,
            sourcedata,
            transform_args)

        return transformeddata

from argparse import ArgumentParser
from collections import defaultdict
from glob import glob
from operator import itemgetter
import h5py
import numpy as np; np.seterr(all='raise')
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
from paraview.simple import Calculator, Gradient, ResampleToImage, XMLUnstructuredGridReader, servermanager


def array2png(array, output_path='heatmap.png', dpi=100):

    vmin = array.min()
    vmax = array.max()
    if vmin >= 0.0:
        vmin = 0.0
        cmap = 'cool'
    else:
        vmax = max(vmax, -vmin)
        vmin = -vmax
        cmap = 'seismic'

    img = plt.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(img)
    plt.savefig(output_path, dpi=dpi)
    plt.clf()


def array2gif(array, output_path='animation.gif', fps=10, dpi=100):
    """
    Convert a 3D numpy array to a gif animation.

    Parameters:
    -----------
    array : numpy.ndarray
        3D array with shape (T, height, width) where T is the number of frames
    output_path : str, optional
        Path to save the output gif, default is 'animation.gif'
    fps : int, optional
        Frames per second in the output animation, default is 10
    dpi : int, optional
        Resolution of the output gif, default is 100

    Returns:
    --------
    None
    """
    # Validate input dimensions
    if len(array.shape) != 3:
        raise ValueError(f"Expected 3D array with shape (T, height, width), got shape {array.shape}")

    # Get number of frames
    num_frames = array.shape[0]

    # Determine global min and max for consistent colorbar across all frames
    vmin = array.min()
    vmax = array.max()
    if vmin >= 0.0:
        vmin = 0.0
        cmap = 'cool'
    else:
        vmax = max(vmax, -vmin)
        vmin = -vmax
        cmap = 'seismic'

    # Create the figure and axis
    fig, ax = plt.subplots()

    # Initial plot with fixed color scale based on entire array
    img = ax.imshow(array[0], cmap=cmap, vmin=vmin, vmax=vmax)

    # Add a colorbar
    plt.colorbar(img, ax=ax)

    # Turn off axis labels
    ax.set_axis_off()

    # Function to update the frame
    def update(frame):
        img.set_array(array[frame])
        return [img]

    # Create the animation
    animation = FuncAnimation(
        fig,
        update,
        frames=range(num_frames),
        interval=1000/fps,  # interval in milliseconds
        blit=True
    )

    # Save the animation as a gif
    animation.save(output_path, writer='pillow', fps=fps, dpi=dpi)

    # Close the figure to free memory
    plt.close(fig)


def vtu2D(vpath, n):
    """
    Interpolate 2D triangle mesh data from VTU file onto a regular grid.
    
    Parameters
    ----------
    vpath : str
        Path to the VTU file containing the triangle mesh
    n : int
        Target grid resolution for the longer dimension
    Returns
    -------
    dict
    """


    reader = XMLUnstructuredGridReader(FileName=vpath)
    reader.UpdatePipeline()

    bounds = reader.GetDataInformation().GetBounds()
    x_min, x_max, y_min, y_max = bounds[0], bounds[1], bounds[2], bounds[3]
    dx = x_max - x_min
    dy = y_max - y_min
    if dx > dy:
        nx = n
        ny = int(np.ceil(dy/dx * n))
    else:
        nx = int(np.ceil(dx/dy * n))
        ny = n

    resampleToImage = ResampleToImage(Input=reader)
    resampleToImage.SamplingDimensions = [nx, ny, 1]
    resampleToImage.SamplingBounds = [x_min, x_max, y_min, y_max, 0.0, 0.0]
    resampleToImage.UpdatePipeline()

    calculator = Calculator(Input=resampleToImage)
    calculator.ResultArrayName = 'Velocity3D'
    calculator.Function = 'Velocity_X*iHat + Velocity_Y*jHat + 0*kHat'
    calculator.UpdatePipeline()

    gradientFilter = Gradient(Input=calculator)
    gradientFilter.ScalarArray = ['POINTS', 'Velocity3D']
    gradientFilter.ComputeVorticity = 1
    gradientFilter.UpdatePipeline()

    output = {}
    for key, value in servermanager.Fetch(gradientFilter).GetPointData().items():
        if key in {'Partition', 'vtkGhostType', 'Gradient', 'Velocity3D'}:
            continue
        if key == 'vtkValidPointMask':
            key = 'Mask'
            dtype = np.bool
        else:
            dtype = np.float32
        if len(value.shape) == 2:
            if value.shape[1] == 3:
                absval = np.absolute(value[:,2])
                finfo = np.finfo(dtype)
                value[np.logical_or(absval > finfo.max, np.logical_and(absval > 0.0, absval < finfo.smallest_subnormal)),2] = 0.0
                output[key] = np.array(value[:,2], dtype=dtype).reshape(nx, ny)              #Vorticity
            else:
                for i, d in enumerate('xy'):
                    output[f'{key}_{d}'] = np.array(value[:,i], dtype=dtype).reshape(nx, ny) #Velocity
        else:
            output[key] = np.array(value, dtype=dtype).reshape(nx, ny)                       #Pressure

    return output


def get_structured_data(folder, resolution=64, dump_gif=False, verbose=False):

    data = defaultdict(lambda: [])
    output = {'errors': []}
    for vpath, t in sorted(((path, float(path.rsplit('.', 1)[0].rsplit('-', 1)[1])) for path in glob(f'{folder}/*.vtu')), key=itemgetter(1)):
        if verbose:
            print('Processing timestep', t, end='\r')
        tdata = {}
        for attempt in range(100): # rarely needs >3 attempts...
            err = ''
            try:
                tdata = vtu2D(vpath, resolution)
                if any(np.isnan(value).any() for value in tdata.values()):
                    err = 'NaNs'
                else:
                    break
            except FloatingPointError:
                err = 'ParaView'
            if err:
                if verbose:
                    print('Attempt', attempt+1, 'error:', err)
            else:
                break
        else:
            print('Processing', vpath, 'failed after', attempt+1, 'attempts')

        output['errors'].append(err)
        for key, value in tdata.items():
            if key == 'Mask':
                mask = value
            else:
                data[key.lower()].append(value)

    output['mask'] = mask
    for key, value in data.items():
        output[key] = np.stack(value)
        if dump_gif:
            array2gif(output[key]*mask, output_path=f'{folder}/{key}.gif')
            if verbose:
                print(f'Animation saved to {folder}/{key}.gif')

    with h5py.File(f'{folder}/flow.h5', 'w') as f:
        for key, value in output.items():
            f.create_dataset(key, data=value)
    if verbose:
        print(f"Flow data saved to flow.h5")

    return output


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('folder')
    parser.add_argument('--resolution', type=int)
    parser.add_argument('--dump-gif', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    get_structured_data(args.folder, resolution=args.resolution, dump_gif=args.dump_gif, verbose=args.verbose)

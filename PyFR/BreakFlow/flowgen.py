import sys
from collections import defaultdict
from glob import glob
from operator import itemgetter
import h5py
import numpy as np
import pyvista as pv
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
from paraview.simple import Calculator, Gradient, ResampleToImage, XMLUnstructuredGridReader, servermanager


def array2gif(array, output_path='animation.gif', fps=10, dpi=100, clip=False):
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
    clip : bool, optional
        Whether to clip values outside 3 standard deviations, default False

    Returns:
    --------
    None
    """
    # Validate input dimensions
    if len(array.shape) != 3:
        raise ValueError(f"Expected 3D array with shape (T, height, width), got shape {array.shape}")

    # Clip array
    if clip:
        std = array.std()
        nclip = (np.absolute(array) > 3.0*std).sum()
        print(f"Clipped {nclip} / {int(np.prod(array.shape))} values")
        array = np.clip(array, -3.0*std, 3.0*std)

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

    print(f"Animation saved to {output_path}")



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

    mesh = pv.read(vpath)
    points = np.array(mesh.points)
    x = points[:, 0]
    y = points[:, 1]
    
    x_min, x_max = x.min(), x.max()
    dx = x_max - x_min
    y_min, y_max = y.min(), y.max()
    dy = y_max - y_min
    
    if dx > dy:
        nx = n
        ny = int(np.ceil(dy/dx * n))
    else:
        nx = int(np.ceil(dx/dy * n))
        ny = n

    resampleToImage = ResampleToImage(Input=XMLUnstructuredGridReader(FileName=vpath))
    resampleToImage.SamplingDimensions = [nx, ny, 1]

    # resampleToImage.UseInputBounds = 1
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


if __name__ == '__main__':

    data = defaultdict(lambda: [])
    for vpath, t in sorted(((path, float(path.rsplit('.', 1)[0].rsplit('-', 1)[1])) for path in glob('*.vtu')), key=itemgetter(1)):
        print('Processing timestep', t, end='\r')
        # for key, value in vtu2D(vpath, int(sys.argv[1]) if sys.argv[1:] else 256).items():
        for key, value in vtu2D(vpath, int(sys.argv[1]) if sys.argv[1:] else 64).items():
            if key == 'Mask':
                mask = value
            else:
                data[key.lower()].append(value)

    output = {'mask': mask}
    with h5py.File(vpath.replace('.vtu', '.pyfrs'), 'r') as f:
        output['wallclock'] = float(str(f['stats'][()]).split('\\nwall-time = ', 1)[1].rsplit('\\nplugin-wall-time-common', 1)[0])
    for key, value in data.items():
        output[key] = np.stack(value)
        if sys.argv[2:]:
            array2gif(output[key], output_path=f'{key}.gif')

    with h5py.File('flow.h5', 'w') as f:
        for key, value in output.items():
            f.create_dataset(key, data=value)

    print(f"Flow data saved to flow.h5")

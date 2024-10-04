import cupy as cp


def save_cub(file_path, array):
    """
    Saves a 3D CuPy ndarray as a .cub file.

    Parameters:
    file_path (str): Path to the .cub file.
    array (cp.ndarray): 3D CuPy array to save.
    """
    if array.ndim != 3:
        raise ValueError("Array must be 3D.")

    # Get the shape of the array
    shape = array.shape

    # Save the shape and data to the file
    with open(file_path, 'wb') as f:
        # Save shape first (3 integers)
        f.write(cp.array(shape, dtype=cp.int32).tobytes())

        # Save the flattened array data
        f.write(array.astype(cp.float32).tobytes())


def load_cub(file_path):
    """
    Loads a .cub file into a 3D CuPy ndarray.

    Parameters:
    file_path (str): Path to the .cub file.

    Returns:
    cp.ndarray: The loaded 3D CuPy array.
    """
    with open(file_path, 'rb') as f:
        # Read the shape (3 integers)
        shape = cp.frombuffer(f.read(12), dtype=cp.int32)
        shape = tuple(shape.get())
        # Read the flattened array data
        array = cp.frombuffer(f.read(), dtype=cp.float32)

        # Reshape it into the original 3D shape
        array = array.reshape(shape)

    return array

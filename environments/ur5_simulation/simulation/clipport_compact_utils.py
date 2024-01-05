import numpy as np
import pybullet as p


# Cliport settings
TRAIN_COLORS = ['blue', 'red', 'green', 'yellow', 'brown', 'gray', 'cyan']
EVAL_COLORS = ['blue', 'red', 'green', 'orange', 'purple', 'pink', 'white']

def reconstruct_heightmaps(color, depth, configs, bounds, pixel_size):
        """Reconstruct top-down heightmap views from multiple 3D pointclouds."""
        heightmaps, colormaps = [], []
        for color, depth, config in zip(color, depth, configs):
                intrinsics = np.array(config['intrinsics']).reshape(3, 3)
                xyz = get_pointcloud(depth, intrinsics)
                position = np.array(config['position']).reshape(3, 1)
                rotation = p.getMatrixFromQuaternion(config['rotation'])
                rotation = np.array(rotation).reshape(3, 3)
                transform = np.eye(4)
                transform[:3, :] = np.hstack((rotation, position))
                xyz = transform_pointcloud(xyz, transform)
                heightmap, colormap = get_heightmap(xyz, color, bounds, pixel_size)
                heightmaps.append(heightmap)
                colormaps.append(colormap)
        return heightmaps, colormaps


def get_heightmap(points, colors, bounds, pixel_size):
        """Get top-down (z-axis) orthographic heightmap image from 3D pointcloud.
    
        Args:
            points: HxWx3 float array of 3D points in world coordinates.
            colors: HxWx3 uint8 array of values in range 0-255 aligned with points.
            bounds: 3x2 float array of values (rows: X,Y,Z; columns: min,max) defining
                region in 3D space to generate heightmap in world coordinates.
            pixel_size: float defining size of each pixel in meters.
    
        Returns:
            heightmap: HxW float array of height (from lower z-bound) in meters.
            colormap: HxWx3 uint8 array of backprojected color aligned with heightmap.
        """
        width = int(np.round((bounds[0, 1] - bounds[0, 0]) / pixel_size))
        height = int(np.round((bounds[1, 1] - bounds[1, 0]) / pixel_size))
        heightmap = np.zeros((height, width), dtype=np.float32)
        colormap = np.zeros((height, width, colors.shape[-1]), dtype=np.uint8)

        # Filter out 3D points that are outside of the predefined bounds.
        ix = (points[Ellipsis, 0] >= bounds[0, 0]) & (points[Ellipsis, 0] < bounds[0, 1])
        iy = (points[Ellipsis, 1] >= bounds[1, 0]) & (points[Ellipsis, 1] < bounds[1, 1])
        iz = (points[Ellipsis, 2] >= bounds[2, 0]) & (points[Ellipsis, 2] < bounds[2, 1])
        valid = ix & iy & iz
        points = points[valid]
        colors = colors[valid]

        # Sort 3D points by z-value, which works with array assignment to simulate
        # z-buffering for rendering the heightmap image.
        iz = np.argsort(points[:, -1])
        points, colors = points[iz], colors[iz]
        px = np.int32(np.floor((points[:, 0] - bounds[0, 0]) / pixel_size))
        py = np.int32(np.floor((points[:, 1] - bounds[1, 0]) / pixel_size))
        px = np.clip(px, 0, width - 1)
        py = np.clip(py, 0, height - 1)
        heightmap[py, px] = points[:, 2] - bounds[2, 0]
        for c in range(colors.shape[-1]):
                colormap[py, px, c] = colors[:, c]
        return heightmap, colormap


def get_pointcloud(depth, intrinsics):
    """Get 3D pointcloud from perspective depth image.

    Args:
        depth: HxW float array of perspective depth in meters.
        intrinsics: 3x3 float array of camera intrinsics matrix.

    Returns:
        points: HxWx3 float array of 3D points in camera coordinates.
    """
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    points = np.float32([px, py, depth]).transpose(1, 2, 0)
    return points


def transform_pointcloud(points, transform):
    """Apply rigid transformation to 3D pointcloud.

    Args:
        points: HxWx3 float array of 3D points in camera coordinates.
        transform: 4x4 float array representing a rigid transformation matrix.

    Returns:
        points: HxWx3 float array of transformed 3D points.
    """
    padding = ((0, 0), (0, 0), (0, 1))
    homogen_points = np.pad(points.copy(), padding,
                                                    'constant', constant_values=1)
    for i in range(3):
            points[Ellipsis, i] = np.sum(transform[i, :] * homogen_points, axis=-1)
    return points

def pix_to_xyz(pixel, height, bounds, pixel_size, skip_height=False):
    """Convert from pixel location on heightmap to 3D position."""
    u, v = pixel
    x = bounds[0, 0] + v * pixel_size
    y = bounds[1, 0] + u * pixel_size
    if not skip_height:
        z = bounds[2, 0] + height[u, v]
    else:
        z = 0.0
    return (x, y, z)


def xyz_to_pix(position, bounds, pixel_size):
    """Convert from 3D position to pixel location on heightmap."""
    u = int(np.round((position[1] - bounds[1, 0]) / pixel_size))
    v = int(np.round((position[0] - bounds[0, 0]) / pixel_size))
    return (u, v)


def invert(pose):
    return p.invertTransform(pose[0], pose[1])


def multiply(pose0, pose1):
    return p.multiplyTransforms(pose0[0], pose0[1], pose1[0], pose1[1])


def apply(pose, position):
    position = np.float32(position)
    position_shape = position.shape
    position = np.float32(position).reshape(3, -1)
    rotation = np.float32(p.getMatrixFromQuaternion(pose[1])).reshape(3, 3)
    translation = np.float32(pose[0]).reshape(3, 1)
    position = rotation @ position + translation
    return tuple(position.reshape(position_shape))


def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return (w, x, y, z)
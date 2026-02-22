"""
Modular Gmsh Mesh Generator with Rectangular Obstacles
Generates 2D meshes with adaptive refinement around obstacles
"""

import json
import os
import random
import sys
import gmsh
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle as MPLRectangle, Polygon
from matplotlib.collections import LineCollection
from shapely.geometry import Polygon as Poly
from shapely.affinity import rotate


class RectangleMeshGenerator:
    """Generate Gmsh meshes with rectangular obstacles and adaptive refinement."""
    
    def __init__(self, domain_bounds=None, refinement_zone=None):
        """
        Initialize the mesh generator.
        
        Parameters:
        -----------
        domain_bounds : dict, optional
            Domain boundaries {'x': [xmin, xmax], 'y': [ymin, ymax]}
            Default: {'x': [0, 50], 'y': [-25, 25]}
        refinement_zone : dict, optional
            Refinement zone boundaries {'x': [xmin, xmax], 'y': [ymin, ymax]}
            Default: {'x': [5, 45], 'y': [-20, 20]}
        """
        self.domain_bounds = domain_bounds or {'x': [0, 50], 'y': [-25, 25]}
        self.refinement_zone = refinement_zone or {'x': [5, 45], 'y': [-20, 20]}
        
        # Mesh parameters
        self.coarse_size = 2.0
        self.fine_size = 0.5
        self.obstacle_size = 0.2
        self.mesh_order = 1
        self.grading_dist_min = 0.5  # Distance where grading starts
        self.grading_dist_max = 5.0  # Distance where max size is reached
        
        # Physical group tags
        self.WALL = 1
        self.INLET = 2
        self.OUTLET = 3
        self.FLUID = 4
        
        self.rectangles = []
        
    def add_rectangle(self, center_x, center_y, width, height, rotation=0.0):
        """
        Add a rectangular obstacle.
        
        Parameters:
        -----------
        center_x : float
            X-coordinate of rectangle center
        center_y : float
            Y-coordinate of rectangle center
        width : float
            Rectangle width
        height : float
            Rectangle height
        rotation : float, optional
            Rotation angle in degrees (default: 0)
        """
        self.rectangles.append({
            'center': (center_x, center_y),
            'width': width,
            'height': height,
            'rotation': rotation
        })

    def set_mesh_order(self, mesh_order=1):
        self.mesh_order = mesh_order
        
    def set_mesh_resolution(self, coarse_size=None, fine_size=None, obstacle_size=None,
                           grading_dist_min=None, grading_dist_max=None):
        """
        Set mesh resolution and grading parameters.
        
        Parameters:
        -----------
        coarse_size : float, optional
            Element size in coarse region (near walls)
        fine_size : float, optional
            Element size in fine region (away from obstacles)
        obstacle_size : float, optional
            Element size near obstacles
        grading_dist_min : float, optional
            Distance from refinement zone where grading starts (default: 0.5)
        grading_dist_max : float, optional
            Distance from refinement zone where max coarse size is reached (default: 5.0)
            Smaller values create faster grading (larger cells sooner)
        """
        if coarse_size is not None:
            self.coarse_size = coarse_size
        if fine_size is not None:
            self.fine_size = fine_size
        if obstacle_size is not None:
            self.obstacle_size = obstacle_size
        if grading_dist_min is not None:
            self.grading_dist_min = grading_dist_min
        if grading_dist_max is not None:
            self.grading_dist_max = grading_dist_max
    
    def _get_rectangle_corners(self, rect):
        """Calculate the four corners of a rotated rectangle."""
        cx, cy = rect['center']
        w, h = rect['width'], rect['height']
        theta = np.radians(rect['rotation'])
        
        # Corners in local coordinate system (before rotation)
        local_corners = np.array([
            [-w/2, -h/2],
            [w/2, -h/2],
            [w/2, h/2],
            [-w/2, h/2]
        ])
        
        # Rotation matrix
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        
        # Rotate and translate corners
        rotated_corners = local_corners @ rotation_matrix.T
        corners = rotated_corners + np.array([cx, cy])
        
        return corners
    
    def generate_mesh(self, output_file='mesh.msh', verbosity=2):
        """
        Generate the mesh with Gmsh.
        
        Parameters:
        -----------
        output_file : str, optional
            Output mesh filename (default: 'mesh.msh')
        verbosity : int, optional
            gmsh verbosity (0: none, 1: most, 2: default)
        """
        gmsh.initialize()
        gmsh.model.add("rectangle_flow")
        
        # Domain boundaries
        x_min, x_max = self.domain_bounds['x']
        y_min, y_max = self.domain_bounds['y']
        
        # Refinement zone boundaries
        rx_min, rx_max = self.refinement_zone['x']
        ry_min, ry_max = self.refinement_zone['y']
        
        # Create outer domain boundary points with COARSE mesh size
        p1 = gmsh.model.geo.addPoint(x_min, y_min, 0, self.coarse_size)
        p2 = gmsh.model.geo.addPoint(x_max, y_min, 0, self.coarse_size)
        p3 = gmsh.model.geo.addPoint(x_max, y_max, 0, self.coarse_size)
        p4 = gmsh.model.geo.addPoint(x_min, y_max, 0, self.coarse_size)
        
        # Create outer domain boundary lines
        l1 = gmsh.model.geo.addLine(p1, p2)  # Bottom
        l2 = gmsh.model.geo.addLine(p2, p3)  # Right (outlet)
        l3 = gmsh.model.geo.addLine(p3, p4)  # Top
        l4 = gmsh.model.geo.addLine(p4, p1)  # Left (inlet)
        
        # Create refinement zone boundary points with FINE mesh size
        rp1 = gmsh.model.geo.addPoint(rx_min, ry_min, 0, self.fine_size)
        rp2 = gmsh.model.geo.addPoint(rx_max, ry_min, 0, self.fine_size)
        rp3 = gmsh.model.geo.addPoint(rx_max, ry_max, 0, self.fine_size)
        rp4 = gmsh.model.geo.addPoint(rx_min, ry_max, 0, self.fine_size)
        
        # Create refinement zone boundary lines
        rl1 = gmsh.model.geo.addLine(rp1, rp2)
        rl2 = gmsh.model.geo.addLine(rp2, rp3)
        rl3 = gmsh.model.geo.addLine(rp3, rp4)
        rl4 = gmsh.model.geo.addLine(rp4, rp1)
        
        # Create curve loops
        outer_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        refinement_loop = gmsh.model.geo.addCurveLoop([rl1, rl2, rl3, rl4])
        
        # Create rectangles (obstacles)
        obstacle_loops = []
        for rect in self.rectangles:
            corners = self._get_rectangle_corners(rect)
            
            # Add points for rectangle corners
            rect_points = []
            for corner in corners:
                pt = gmsh.model.geo.addPoint(corner[0], corner[1], 0, self.obstacle_size)
                rect_points.append(pt)
            
            # Add lines for rectangle edges
            rect_lines = []
            for i in range(4):
                line = gmsh.model.geo.addLine(rect_points[i], rect_points[(i+1)%4])
                rect_lines.append(line)
            
            # Create curve loop for rectangle
            rect_loop = gmsh.model.geo.addCurveLoop(rect_lines)
            obstacle_loops.append(rect_loop)
        
        # Create SINGLE plane surface for entire domain
        fluid_surface = gmsh.model.geo.addPlaneSurface([outer_loop] + obstacle_loops)
        
        # Synchronize
        gmsh.model.geo.synchronize()
        
        # Set Gmsh meshing options for better control
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)  # Allow boundary sizes to extend
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)  # Use point characteristic lengths
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
        gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1.0)  # Global scaling
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0)  # No min limit
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1e22)  # No max limit
        
        # Define physical groups for boundaries
        # Collect all obstacle edge curves
        all_obstacle_curves = []
        for i, rect in enumerate(self.rectangles):
            # Each obstacle has 4 curves starting at base_curve
            base_curve = 9 + i * 4  # Curves 9-12 for first obstacle, 13-16 for second, etc.
            obstacle_curves = [base_curve, base_curve+1, base_curve+2, base_curve+3]
            all_obstacle_curves.extend(obstacle_curves)
        
        # Adds physical groups
        gmsh.model.addPhysicalGroup(1, all_obstacle_curves, self.WALL, "wall")
        gmsh.model.addPhysicalGroup(1, [l1, l3, l4], self.INLET, "inlet")
        gmsh.model.addPhysicalGroup(1, [l2], self.OUTLET, "outlet")
        
        # Define physical group for fluid domain (SINGLE surface for PyFR)
        gmsh.model.addPhysicalGroup(2, [fluid_surface], self.FLUID, "fluid")
        
        # Set mesh size fields for gradual refinement
        # Use Box field to define refinement zone (no curve references needed)
        
        # Field 1: Box field for refinement zone
        # VIn = fine_size inside box, VOut = coarse_size outside
        gmsh.model.mesh.field.add("Box", 1)
        gmsh.model.mesh.field.setNumber(1, "VIn", self.fine_size)
        gmsh.model.mesh.field.setNumber(1, "VOut", self.coarse_size)
        gmsh.model.mesh.field.setNumber(1, "XMin", rx_min)
        gmsh.model.mesh.field.setNumber(1, "XMax", rx_max)
        gmsh.model.mesh.field.setNumber(1, "YMin", ry_min)
        gmsh.model.mesh.field.setNumber(1, "YMax", ry_max)
        gmsh.model.mesh.field.setNumber(1, "Thickness", self.grading_dist_max)  # Smooth transition
        
        # No Field 2 needed - Box handles grading directly
        
        if obstacle_loops:
            # Field 3: Distance from obstacles
            obstacle_curve_ids = []
            for i, rect in enumerate(self.rectangles):
                base_curve = 9 + i * 4
                obstacle_curve_ids.extend([base_curve, base_curve+1, base_curve+2, base_curve+3])
            
            gmsh.model.mesh.field.add("Distance", 3)
            gmsh.model.mesh.field.setNumbers(3, "CurvesList", obstacle_curve_ids)
            gmsh.model.mesh.field.setNumber(3, "Sampling", 100)
            
            # Field 4: Threshold for obstacle refinement
            gmsh.model.mesh.field.add("Threshold", 4)
            gmsh.model.mesh.field.setNumber(4, "InField", 3)
            gmsh.model.mesh.field.setNumber(4, "SizeMin", self.obstacle_size)
            gmsh.model.mesh.field.setNumber(4, "SizeMax", self.coarse_size)  # FIXED!
            gmsh.model.mesh.field.setNumber(4, "DistMin", 0.5)
            gmsh.model.mesh.field.setNumber(4, "DistMax", 5.0)  # Longer transition
            
            # Field 5: Min of Box field and obstacle fields
            gmsh.model.mesh.field.add("Min", 5)
            gmsh.model.mesh.field.setNumbers(5, "FieldsList", [1, 4])
            
            gmsh.model.mesh.field.setAsBackgroundMesh(5)
        else:
            gmsh.model.mesh.field.setAsBackgroundMesh(1)
        
        # Generate 2D mesh
        gmsh.option.setNumber("General.Terminal", verbosity)
        gmsh.model.mesh.generate(2)
        
        # Remove duplicate nodes to ensure clean mesh
        gmsh.model.mesh.removeDuplicateNodes()

        # Set mesh order
        gmsh.model.mesh.setOrder(self.mesh_order)

        # Set mesh format to 2.2 (more widely supported)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)  # ASCII format
        gmsh.option.setNumber("Mesh.SaveAll", 0)  # Only save mesh, not geometry
        
        # Write mesh file
        gmsh.write(output_file)
        
        # Get statistics before finalizing
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        element_types, element_tags, _ = gmsh.model.mesh.getElements()
        
        gmsh.finalize()
        
        if verbosity:
            print(f"Mesh generated successfully: {output_file}")
            print(f"  Domain: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
            print(f"  Refinement zone: x=[{rx_min}, {rx_max}], y=[{ry_min}, {ry_max}]")
            print(f"  Number of obstacles: {len(self.rectangles)}")
            print(f"  Nodes: {len(node_tags)}")
            print(f"  Elements: {sum(len(tags) for tags in element_tags)}")
        
    def validate_mesh(self, mesh_file):
        """
        Validate mesh integrity and check for common issues.
        
        Parameters:
        -----------
        mesh_file : str
            Mesh file to validate
            
        Returns:
        --------
        dict : Validation results
        """
        print(f"\nValidating mesh: {mesh_file}")
        print("=" * 60)
        
        nodes, elements = self._read_gmsh_mesh(mesh_file)
        
        # Check for duplicate nodes (nodes is numpy array of shape (N, 2))
        unique_nodes = set()
        duplicates = 0
        for i in range(len(nodes)):
            coord_tuple = (round(nodes[i, 0], 10), round(nodes[i, 1], 10))
            if coord_tuple in unique_nodes:
                duplicates += 1
            unique_nodes.add(coord_tuple)
        
        # Check element quality
        min_area = float('inf')
        degenerate_count = 0
        
        for elem in elements:
            if len(elem) == 3:  # Triangle
                p1, p2, p3 = nodes[elem[0]], nodes[elem[1]], nodes[elem[2]]
                
                # Calculate area using cross product
                v1 = (p2[0] - p1[0], p2[1] - p1[1])
                v2 = (p3[0] - p1[0], p3[1] - p1[1])
                area = abs(v1[0] * v2[1] - v1[1] * v2[0]) / 2
                
                min_area = min(min_area, area)
                if area < 1e-10:
                    degenerate_count += 1
        
        # Report results
        print(f"\nValidation Results:")
        print(f"  Total nodes: {len(nodes)}")
        print(f"  Total elements: {len(elements)}")
        print(f"  Duplicate nodes: {duplicates}")
        print(f"  Degenerate elements: {degenerate_count}")
        print(f"  Minimum element area: {min_area:.2e}")
        
        if duplicates == 0 and degenerate_count == 0:
            print(f"\n✅ Mesh validation PASSED - No issues found")
            return {'valid': True, 'duplicates': 0, 'degenerate': 0}
        else:
            print(f"\n⚠️  Mesh has issues that may cause problems")
            return {'valid': False, 'duplicates': duplicates, 'degenerate': degenerate_count}
        
    def plot_mesh(self, mesh_file='mesh.msh', output_file=None, show_refinement_zone=True, 
                  show_obstacles=True, figsize=(12, 8), dpi=300):
        """
        Plot the generated mesh and save to PNG.
        
        Parameters:
        -----------
        mesh_file : str, optional
            Mesh file to plot (default: 'mesh.msh')
        output_file : str, optional
            Output PNG filename (default: mesh_file with .png extension)
        show_refinement_zone : bool, optional
            Show refinement zone boundary (default: True)
        show_obstacles : bool, optional
            Show obstacle boundaries (default: True)
        figsize : tuple, optional
            Figure size (default: (12, 8))
        dpi : int, optional
            Resolution in dots per inch (default: 300)
        """
        # Read mesh file
        nodes, elements = self._read_gmsh_mesh(mesh_file)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot mesh elements
        for elem in elements:
            if len(elem) == 3:  # Triangle
                triangle = nodes[elem]
                edges = [
                    [triangle[0], triangle[1]],
                    [triangle[1], triangle[2]],
                    [triangle[2], triangle[0]]
                ]
                lc = LineCollection(edges, colors='gray', linewidths=0.3, alpha=0.5)
                ax.add_collection(lc)
        
        # Plot domain boundary
        x_min, x_max = self.domain_bounds['x']
        y_min, y_max = self.domain_bounds['y']
        
        domain_rect = MPLRectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                   fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(domain_rect)
        
        # Plot refinement zone
        if show_refinement_zone:
            rx_min, rx_max = self.refinement_zone['x']
            ry_min, ry_max = self.refinement_zone['y']
            
            refine_rect = MPLRectangle((rx_min, ry_min), rx_max - rx_min, ry_max - ry_min,
                                       fill=False, edgecolor='blue', linewidth=1.5,
                                       linestyle='--', label='Refinement zone')
            ax.add_patch(refine_rect)
        
        # Plot obstacles
        if show_obstacles:
            for rect in self.rectangles:
                corners = self._get_rectangle_corners(rect)
                polygon = Polygon(corners, fill=True, facecolor='red', 
                                 edgecolor='darkred', linewidth=2, alpha=0.3)
                ax.add_patch(polygon)
        
        # Labels for boundaries
        ax.text(x_min, (y_min + y_max)/2, 'INLET', fontsize=10, 
                ha='right', va='center', color='green', weight='bold')
        ax.text(x_max, (y_min + y_max)/2, 'OUTLET', fontsize=10, 
                ha='left', va='center', color='red', weight='bold')
        ax.text((x_min + x_max)/2, y_max, 'INLET', fontsize=10, 
                ha='center', va='bottom', color='black', weight='bold')
        ax.text((x_min + x_max)/2, y_min, 'INLET', fontsize=10, 
                ha='center', va='top', color='black', weight='bold')
        
        ax.set_xlim(x_min - 2, x_max + 2)
        ax.set_ylim(y_min - 2, y_max + 2)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Generated Mesh')
        ax.grid(True, alpha=0.3)
        
        if show_refinement_zone:
            ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        # Determine output filename
        if output_file is None:
            output_file = mesh_file.rsplit('.', 1)[0] + '.png'
        
        # Save to PNG
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Mesh visualization saved to: {output_file}")
        
        plt.close(fig)
        
    def _read_gmsh_mesh(self, mesh_file):
        """Read nodes and elements from Gmsh mesh file (supports format 2.2 and 4.1)."""

        with open(mesh_file, 'r') as f:
            lines = f.readlines()

        nodes = []
        elements = []
        
        # Parse nodes
        in_nodes = False
        node_count_line = False
        for i, line in enumerate(lines):
            if line.strip() == '$Nodes':
                in_nodes = True
                node_count_line = True
                continue
            elif line.strip() == '$EndNodes':
                in_nodes = False
                continue
            
            if in_nodes:
                if node_count_line:
                    node_count_line = False
                    continue
                parts = line.strip().split()
                if len(parts) >= 4:
                    nodes.append([float(parts[1]), float(parts[2])])
        
        nodes = np.array(nodes)
        
        # Parse elements (only triangles)
        in_elements = False
        elem_count_line = False
        for i, line in enumerate(lines):
            if line.strip() == '$Elements':
                in_elements = True
                elem_count_line = True
                continue
            elif line.strip() == '$EndElements':
                in_elements = False
                continue
            
            if in_elements:
                if elem_count_line:
                    elem_count_line = False
                    continue
                parts = line.strip().split()
                if len(parts) > 3:
                    elem_type = int(parts[1])
                    if elem_type == 2:  # Triangle
                        num_tags = int(parts[2])
                        node_start = 3 + num_tags
                        if node_start + 2 < len(parts):
                            elem_nodes = [int(parts[node_start]) - 1,
                                        int(parts[node_start + 1]) - 1,
                                        int(parts[node_start + 2]) - 1]
                            elements.append(elem_nodes)
        
        return nodes, elements


def create_polygon(center_x, center_y, width, height, rotation):
    """Create a polygon from rectangle parameters."""
    half_width = width / 2
    half_height = height / 2
    corners = [
        (-half_width, -half_height),
        (half_width, -half_height),
        (half_width, half_height),
        (-half_width, half_height)
    ]
    
    polygon = Poly(corners)
    polygon = rotate(polygon, rotation, origin='centroid', use_radians=False)
    polygon = Poly([(x + center_x, y + center_y) for x, y in polygon.exterior.coords])
    
    return polygon


def remove_overlaps_and_mirror(rectangles, max_area=60.0, min_dist=2.0, xlim=(10.0, 25.0), ylim=(-15.0, 15.0)):
    """
    Remove rectangles that overlap with previously seen rectangles and mirror across y-axis.
    
    Args:
        rectangles: List of tuples (center_x, center_y, width, height, rotation)
                   rotation is in degrees, counter-clockwise from horizontal
    Returns:
        List of non-overlapping rectangles in the same format
    """
    
    non_overlapping = []
    non_overlapping_polygons = []
    mirrored = []
    total_area = 0.0
    
    for i, rect in enumerate(rectangles):

        centerline = not rect[1]
        if centerline and rect[4] != 0.0 and (rect[2] != rect[3] or not rect[4] in {45.0, 90.0}):
            continue

        polygon = create_polygon(*rect)

        area = (2 - centerline) * polygon.area
        if total_area + area > max_area:
            continue

        if centerline:
            if any(x < xlim[0] or x > xlim[1] or y < ylim[0] or y > ylim[1] for x, y in polygon.exterior.coords):
                continue
        elif any(x < xlim[0] or x > xlim[1] or y < min_dist/2 or y > ylim[1] for x, y in polygon.exterior.coords):
            continue
        
        if not any(polygon.intersects(prev_poly) for prev_poly in non_overlapping_polygons):
            non_overlapping.append(rect)
            non_overlapping_polygons.append(polygon.buffer(min_dist))
            if not centerline:
                mirrored.append([rect[0], -rect[1], rect[2], rect[3], -rect[4]])
            total_area += area
    
    return non_overlapping + mirrored


def random_rectangle_mesh(output="mesh.msh", 
                          seed=None, 
                          centerline=False,
                          mesh_order=1, 
                          coarse_size=1.0, 
                          max_area=60.0, 
                          min_dist=2,
                          plot_mesh=False,
                          **kwargs):

    random.seed(seed)
    rectangles = []
    data = {'seed': seed, 'parameters': {}}

    for center_x in range(11, 25, min_dist):
        for center_y in range(0 if centerline else min_dist, 15, min_dist):
            on = random.randint(0, 1)
            area = random.randint(4, int(max_area / 2) if center_y else int(max_area))
            if random.randint(0, 1):
                width = random.randint(min_dist, int(area / min_dist))
                height = random.randint(min_dist, int(area / width))
            else:
                height = random.randint(min_dist, int(area / min_dist))
                width = random.randint(min_dist, int(area / height))
            if center_y:
                rotation = 15 * random.randint(0, 12)
            elif height == width:
                rotation = 45 * random.randint(0, 1)
            else:
                rotation = 0
            if on:
                rectangles.append([center_x, center_y, width, height, rotation])
            data['parameters'][f'{center_x},{center_y}'] = [on, width, height, rotation]

    rectangles = remove_overlaps_and_mirror(rectangles, max_area=max_area, min_dist=min_dist)
    data['area'] = sum(rect[2]*rect[3] for rect in rectangles)
    data['length'] = min(min(rect[2:4]) for rect in rectangles)
    data['rectangles'] = rectangles

    datafile = output.rsplit('.', 1)[0] + '.json'
    if os.path.isfile(output):
        with open(datafile, 'r') as f:
            if data == json.load(f):
                return
    with open(datafile, 'w') as f:
        json.dump(data, f)
    
    # Generate mesh
    generator = RectangleMeshGenerator()
    generator.set_mesh_resolution(coarse_size=coarse_size, fine_size=2.0*coarse_size/3.0, obstacle_size=coarse_size/3.0)
    generator.set_mesh_order(mesh_order=mesh_order)
    for rect in rectangles:
        generator.add_rectangle(*rect)
    generator.generate_mesh(output, **kwargs)
    
    # Plot mesh
    if plot_mesh:
        generator.plot_mesh(output)


def square_test_mesh(output="mesh.msh", 
                     mesh_order=1, 
                     coarse_size=1.0, 
                     size=3.0, 
                     plot_mesh=False,
                     **kwargs):

    generator = RectangleMeshGenerator()
    generator.set_mesh_resolution(coarse_size=coarse_size, fine_size=2.0*coarse_size/3.0, obstacle_size=coarse_size/3.0)
    generator.set_mesh_order(mesh_order=mesh_order)
    generator.add_rectangle(10, 0, size, size, 45)
    generator.generate_mesh(output, **kwargs)
    if plot_mesh:
        generator.plot_mesh(output)


if __name__ == "__main__":
    for i in range(int(sys.argv[1]) if sys.argv[1:] else 100):
        os.makedirs(str(i), exist_ok=True)
        random_rectangle_mesh(f'{i}/mesh.msh', i, plot_mesh=sys.argv[2:])

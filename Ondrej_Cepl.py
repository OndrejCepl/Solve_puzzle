"""
This script sorts sub-images into the correct order to reconstruct the original image.

@author: Ondrej Cepl

The process involves:
1. Reading the original image and its grayscale version.
2. Analyzing the color and shape of the sub-images.
3. Solving the puzzle to determine the correct order of sub-images.
4. Reconstructing the original image from the sorted sub-images.
5. Displaying the final solution and the step-by-step folding process.

Constants:
- IMG_PATH: Path to the input image.
- GRID_SIZE: Size of the grid for sub-images.
- GET_COLORS: Dictionary defining the colors and initial position for sorting.

Functions:
- main(): The main function that orchestrates the image sorting and reconstruction process.
- get_color_shape(): Analyzes the color and shape of the sub-images.
- solve_puzzle(): Solves the puzzle to find the correct order of sub-images.
- reconstruct_orig_image(): Reconstructs the original image from the sorted sub-images.

Usage:
Run the script directly to see the final solution and the step-by-step folding process.

The final solution will be displayed in a window, and you can press any key to see the folding process step-by-step.
"""

# standard libraries
import numpy as np
import copy

# type hints
from typing import Optional

# image processing
import cv2 as cv
from skimage import io
import imutils

# heuristic-based approach
import heapq
from itertools import combinations


## ======== initial variables and constants ========
# path to the image
IMG_PATH = r"./uloha.png"

# number of puzzle tiles (sub-images) in the grid
GRID_SIZE = 3

# define on which position in located image with all the colors and 
# describe where is each color located
GET_COLORS = {
    'top': 'red',
    'left': 'green',
    'bot': 'blue',
    'right': 'yellow',
    'position': (1,1), # starts from 0
}
#===================================================
## ========== functions to find the color ==========
def get_sub_image(img: np.ndarray, row: int, col: int) -> np.ndarray:
    """Extract appropriate sub image from the input image"""
    size_x = round((img.shape[0] - (GRID_SIZE + 1))/GRID_SIZE)
    size_y = round((img.shape[1] - (GRID_SIZE + 1))/GRID_SIZE)
    img_row_start = round(row * size_x + row)
    img_row_stop = round((row + 1) * size_x + (row + 2))
    
    img_col_start = round(col * size_y + col)
    img_col_stop = round((col + 1) * size_y + (col + 2))
    output_img = img[img_row_start : img_row_stop, 
                img_col_start : img_col_stop]

    return output_img

def get_intensity_at_position(img: np.ndarray, position: str, surr: int = 4) -> int:
    """Get the intensity from grayscale image at predefined positions

    Args:
        img (np.ndarray): The input grayscale image
        position (str): The predefined position to extract intensity from ('top', 'bot', 'left', 'right')
        surr (int, optional): The surrounding area size around the position. Defaults to 4.

    Returns:
        int: The average intensity value at the specified position
    """
    size = img.shape[0]
    size_half = round(size/2)
    border = 3
    match position:
        case 'top':
            return round(np.average(img[border : (surr + border), 
                                (size_half - surr) : (size_half + surr)]))
        case 'left':
            return round(np.average(img[(size_half - surr) : (size_half + surr), 
                                border : surr + border]))
        case 'bot':
            return round(np.average(img[(size - surr - border) : (size - border), 
                                (size_half - surr) : (size_half + surr)]))
        case 'right':
            return round(np.average(img[(size_half - surr) : (size_half + surr), 
                                (size - surr - border) : (size - border)]))
        case _:
            print('Position input is not valid!')
            
def get_intensities(img: np.ndarray) -> dict:
    """Create dictionary of intensities for all four colors"""
    intensities = {
        v : get_intensity_at_position(img, k, surr=4
                ) for k, v in GET_COLORS.items() if k != 'position'
        }
    return intensities

def find_intensity_thresholds(img: np.ndarray, row: int, col: int) -> dict:
    """Find color intensities thresholds in grayscale from one sub image"""

    sub_img_color = get_sub_image(img, row, col)

    if sub_img_color.shape[0] != sub_img_color.shape[1]:
        print("error at dimensions of sub image")

    intensities = get_intensities(sub_img_color)

    return intensities

def get_color_intensity_intervals(intensities: dict) -> dict:
    """Sorts the color intensity values into intervals."""
    sorted_intensities = dict(sorted(intensities.items(), key=lambda x:x[1]))
    values = list(sorted_intensities.values())

    for i, key in zip(range(len(values)), sorted_intensities.keys()):
        lower_edge = 0 if i == 0 else np.floor((values[i - 1] + values[i]) / 2) + 1
        upper_edge = 255 if i == len(values) - 1 else np.ceil((values[i] 
                                    + values[i + 1]) / 2) - 1

        sorted_intensities[key] = [lower_edge, upper_edge]
        
    return sorted_intensities
    

def get_color(color_intervals: dict, intensity: int) -> str:
    """Get color name according to given color intervals from given intensity"""
    for k, v in color_intervals.items():
        if all([v[0] <= intensity , intensity <= v[1]]):
            color = k 
    try:
        return color
    except:
        print('Color was not found!')

def get_colors(img: np.ndarray, color_intervals: dict) -> dict:
    """Find color category from given dictionary of positions and intensities"""
    found_colors = {}
    intensities:dict = get_intensities(img)
    for k, v in intensities.items():
        found_colors[k] = get_color(color_intervals, v)

    return found_colors
    
## ========== identify the shape ==========
def remove_background(img: np.ndarray) -> np.ndarray[int]:
    """Adjust constrast and separete foreground and background with otsu thresholding
    
    Contrast Limited Adaptive Histogram Equalization (CLAHE) is used to adjust 
    constrast followed by otsu thresholding to separete foreground from 
    the background and preserve sharp edges. 
    """
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(img)
    _, otsu = cv.threshold(equalized,100,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    return otsu

def get_sum_value(bw_img: np.ndarray) -> dict[str, int]:
    """
    Calculate the count of black pixels in each direction of the image.

    Args:
    bw_img (numpy.ndarray): Black and white image with black pixels represented as 0.

    Returns:
    dict: A dictionary containing the count of black pixels 
        in each direction ('top', 'left', 'bot', 'right').
    """
    # size of the image
    img_size = bw_img.shape[0]
    # normalize size of the black color to 1 for each non-zero pixel
    binary_img = bw_img == 0
    shape_count = {
        "top": 0,
        "left": 0,
        "bot": 0,
        "right": 0
    }
    
    conditions_map = {
        (True, True): 'top',
        (False, True): 'left',
        (False, False): 'bot',
        (True, False): 'right'
    }
    for row in range(binary_img.shape[0]):
        for col in range(binary_img.shape[1]):
            # if the value is true
            if binary_img[row, col]:
                # Calculate the actual conditions tuple 
                actual_cond = (col > row, img_size - row > col)
                shape_count[conditions_map[actual_cond]] += 1
                
    return shape_count

def find_shapes(bw_img: np.ndarray) -> dict:
    """Find shape of each side according to painted area"""
    shape_count = get_sum_value(bw_img)
    mid_value = np.mean(sorted(shape_count.values())[1:3])
    shape = {}
    for k, v in shape_count.items():
        shape[k] = 'smile' if v > mid_value else 'eyes'

    return shape

def get_color_shape(original_image: np.ndarray, 
                            original_image_bw: np.ndarray) -> list[dict]:
    """Containing color and shape information for sub-images in a grid.

    Args:
        original_image: The original RGB image.
        original_image_bw: The original black and white image.

    Returns:
        A list of dictionaries, each containing color, shape, location, 
        rotation, and image information for a sub-image in the grid.
    """
    mid_intensity_threshold = find_intensity_thresholds(original_image_bw, 
                                                        *GET_COLORS['position'])
    color_intervals = get_color_intensity_intervals(mid_intensity_threshold)
    img_descrition = []
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            sub_img_rgb = get_sub_image(original_image, row, col)
            sub_img = get_sub_image(original_image_bw, row, col)
            # get information about the color
            sub_colors = get_colors(sub_img, color_intervals)

            # get information about the shape 
            sub_img_bw = remove_background(sub_img)
            sub_shape = find_shapes(sub_img_bw)

            combine_values = dict(zip(sub_colors.keys(), 
                                    list(zip(sub_colors.values(), 
                                            sub_shape.values()))))
            combine_values['location'] = (row, col)
            combine_values['rotation'] = 0
            # information about the sub image
            combine_values['img'] = sub_img_rgb
            # combine_values['image'] = sub_img
            # write dict into list
            img_descrition.append(combine_values)

    return img_descrition

#===================================================
## ========== Solve algorithm ==========
class SubImage:
    """Hold information about color and shape in an object.

    Attributes:
        top (tuple[str, str]): Colors on the top edge.
        left (tuple[str, str]): Colors on the left edge.
        bot (tuple[str, str]): Colors on the bottom edge.
        right (tuple[str, str]): Colors on the right edge.
        location (tuple[int, int]): Current location of the subimage.
        rotation (int): Number of 90-degree rotations.
        orig_location (tuple[int, int], optional): Original location of the subimage.

    Methods:
        rotate(): Rotate the subimage counter-clockwise by 90 degrees.
    """
    def __init__(self, top: tuple[str, str], left: tuple[str, str], 
                        bot: tuple[str, str], right: tuple[str, str], 
                        location: tuple[int, int], rotation: int = 0, 
                        orig_location: tuple[int, int]=None) -> None:
        self.edges = {
            'top': top,
            'left': left,
            'bot': bot,
            'right': right
        }
        # original location
        self.location = location
        self.orig_location = orig_location if orig_location is not None else location
        # number of rotations
        self.rotation = rotation

    # Rotate the subimage counter-clock wise 
    # (top -> left -> bottom -> right -> top)
    def rotate(self):
        """Rotate image description by 90 deg counter clock-wise"""
        self.edges = {
            'top': self.edges['right'],
            'left': self.edges['top'],
            'bot': self.edges['left'],
            'right': self.edges['bot']
        }
        self.rotation += 1 # times by 90 deg

def initialize_grid(grid_values:list[dict]) -> list[list[SubImage]]:
    """Initialize the grid with original values"""
    grid = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    original_img_grid = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    for s_img in grid_values:
        grid[s_img['location'][0]][s_img['location'][1]] = \
            SubImage(*[(v) for k, v in s_img.items() if k != 'img'])
        original_img_grid[s_img['location'][0]][s_img['location'][1]] = s_img['img']
    
    return grid, original_img_grid

# Heuristic 
def count_mismatched_edges(grid: list[list[SubImage]]) -> int:
    """Count number of mismatched edges between adjacent sub images"""
    mismatches = 0
    rows, cols = len(grid), len(grid[0])

    # start at top left corner
    for i in range(rows):
        for j in range(cols):
            # Check right neighbor
            if j < cols - 1: # omit the right edge
                if any([grid[i][j].edges['right'][0] != grid[i][j + 1].edges['left'][0], 
                        grid[i][j].edges['right'][1] == grid[i][j + 1].edges['left'][1]]):
                    # print([i, j, grid[i][j].edges['right'], grid[i][j + 1].edges['left']])
                    mismatches += 1
            # Check bottom neighbor
            if i < rows - 1: # omit the bottom edge
                if any([grid[i][j].edges['bot'][0] != grid[i + 1][j].edges['top'][0],
                        grid[i][j].edges['bot'][1] == grid[i + 1][j].edges['top'][1]]):
                    # print([i, j, grid[i][j].edges['bot'], grid[i + 1][j].edges['top']])
                    mismatches += 1

    return mismatches

def swap_positions(new_grid: list[list[SubImage]], 
                    sp_1: tuple[int, int], sp_2: tuple[int, int]):
    """Swap the positions of two SubImage objects in a grid.

    Args:
        new_grid (list[list[SubImage]]): The grid containing SubImage objects.
        sp_1 (tuple[int, int]): The position of the first SubImage to swap.
        sp_2 (tuple[int, int]): The position of the second SubImage to swap.
    """
    # change the tiles properties
    new_grid[sp_1[0]][sp_1[1]], new_grid[sp_2[0]][sp_2[1]] = \
        (new_grid[sp_2[0]][sp_2[1]], new_grid[sp_1[0]][sp_1[1]])
    # swap the location back to keep track of original location
    new_grid[sp_1[0]][sp_1[1]].location, new_grid[sp_2[0]][sp_2[1]].location = \
        (new_grid[sp_2[0]][sp_2[1]].location, new_grid[sp_1[0]][sp_1[1]].location)

def get_neighbors(
    grid_state: list[list[SubImage]], grid_size: int
    ) -> list[list[list[SubImage]]]:
    """Get neighbors for a given grid state by swapping and rotating subimages.

    Args:
        grid_state (list[list[SubImage]]): The current state of the grid with subimages.
        grid_size (int): The size of the grid.

    Returns:
        list[list[list[SubImage]]]: A list of neighboring grid states after swapping and rotating subimages.
    """
    neighbors = []

    all_positions = [(i, j) for i in range(grid_size) for j in range(grid_size)] 

    for swap_pos_1, swap_pos_2 in combinations(all_positions, 2):
            new_grid_state = copy.deepcopy(grid_state)
            swap_positions(new_grid_state, swap_pos_1, swap_pos_2)

            # Generate neighbors by rotating each subimage
            for i in range(4): # First tile rotation
                for j in range(4): # Second tile rotation
                    rotated_grid_state = copy.deepcopy(new_grid_state)
                    for _ in range(i): # rotate i-times
                        rotated_grid_state[swap_pos_1[0]][swap_pos_1[1]].rotate()
                    for _ in range(j): # rotate j-times
                        rotated_grid_state[swap_pos_2[0]][swap_pos_2[1]].rotate()
                    neighbors.append(rotated_grid_state)

    return neighbors

def get_canonical_form(grid_state: list[list[SubImage]], 
                        include_extra_info: bool = False) -> tuple[tuple]:
    """
    Convert the grid into a canonical form (tuple of tuples) to track visited states.
    If include_extra_info is True, it will include location and rotation in the form.
    """
    if include_extra_info:
        return tuple(
            (tuple(tile.edges.items()), tile.location, tile.orig_location, tile.rotation)
            for row in grid_state for tile in row
            )
    else:
        return tuple(tuple(tile.edges.items()) for row in grid_state for tile in row)

def process_neighbors(
    current_grid: list[list[SubImage]], grid_size: int, 
    priority_queue: list[tuple[int, int, list[list[SubImage]]]], 
    unique_index: int, came_from: dict, explored_states: set) -> int:
    """
    Process each neighbor of the current grid state.

    Args:
        current_grid (list[list[SubImage]]): The current state of the grid with subimages.
        grid_size (int): The size of the grid.
        priority_queue (list[tuple[int, int, list[list[SubImage]]]]): The priority queue for processing neighbors.
        unique_index (int): The unique index for each grid state.
        came_from (dict): A dictionary to track the previous grid state for each grid state.
        explored_states (set): A set containing all explored grid states.

    Returns:
        int: The updated unique index after processing neighbors.
    """
    neighbors = get_neighbors(current_grid, grid_size)
    for next_grid_state in neighbors:
        next_explored = get_canonical_form(next_grid_state)
        if next_explored in explored_states:
            continue
        explored_states.add(next_explored)
        priority = count_mismatched_edges(next_grid_state)
        heapq.heappush(priority_queue, (priority, unique_index, next_grid_state))
        unique_index += 1
        came_from[
            get_canonical_form(next_grid_state, include_extra_info=True)
            ] = get_canonical_form(current_grid, include_extra_info=True)
        
    return unique_index, priority

def initialize_priority_queue(
    start_grid_state: list[list[SubImage]]
    ) -> list[tuple[int, int, list[list[SubImage]]]]:
    """Initialize the priority queue with the start grid state.
    """
    priority_queue: list[tuple[int, int, list[list[SubImage]]]] = []
    unique_index = 0
    heapq.heappush(priority_queue, (0, unique_index, start_grid_state))
    return priority_queue, unique_index + 1

def reconstruct_grid(detailed_state: tuple[tuple]) -> list[list[SubImage]]:
    """
    Reconstruct the grid from the detailed canonical form.
    The detailed form includes edges, location, and rotation for each tile.
    """
    grid = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE )]
    
    # Iterate over the detailed canonical form and reconstruct the grid
    for tile_data in detailed_state:
        tile_edges, location, orig_location, rotation = tile_data
        edges = {key: value for key, value in tile_edges}
        
        # Create a new SubImage object with the edges, location, and rotation
        tile = SubImage(*edges.items(), location, rotation, orig_location)
        
        # Place the tile in the correct position in the grid
        grid[location[0]][location[1]] = tile
    
    return grid

def reconstruct_path(
    came_from: dict, goal_state: list[list[SubImage]]
    ) -> list[list[list[SubImage]]]:
    """Reconstructs the path from the start to the goal state."""
    path = []
    current_state = get_canonical_form(goal_state, include_extra_info=True)
    
    while current_state is not None:  
        # Reconstruct the grid state using the detailed canonical form
        current_grid = reconstruct_grid(current_state)
        path.append(current_grid)
        current_state = came_from.get(current_state, None)
    
    return path[::-1]  # Reverse to get the path from start to goal

# A* search function
def a_star_search(start_grid_state: list[list[SubImage]], goal_mismatch: int,
                    grid_size: int) -> Optional[list[list[SubImage]]]:
    """
    Perform A* search algorithm to find a solution to the task.
    
    Args:
        start_grid_state: The initial state of the grid.
        goal_mismatch: The number of mismatches to achieve the goal state.
        grid_size: The size of the grid.
    
    Returns:
        Optional[list[list[SubImage]]]: The solved grid state or None if no solution is found.
    """
    priority_queue, unique_index = initialize_priority_queue(start_grid_state)
    
    # reconstruct the path
    came_from: dict[tuple, Optional[tuple]] = {}
    came_from[get_canonical_form(start_grid_state, include_extra_info=True)] = None

    # to track visited states 
    explored_states = set()
    explored_states.add(get_canonical_form(start_grid_state))

    while priority_queue:
        # pop the grid with the lowest priority
        _, _, current_grid = heapq.heappop(priority_queue)

        # Check if the current grid state has the goal mismatch (0 mismatches)
        if count_mismatched_edges(current_grid) == goal_mismatch:
            print(f'{unique_index} different variants were tested.')
            # return current_grid  # Puzzle solved!
            return reconstruct_path(came_from, current_grid)
            # return current_grid
        # process neighbors and get unique index and priority    
        unique_index, priority = process_neighbors(
            current_grid, grid_size, priority_queue, 
            unique_index, came_from, explored_states
            )
        print(
            f'Number of tested variants: {unique_index}. '
            f'Number of mismatches: {priority}'
            )
    return None  # No solution founded

# Main function to run the solver
def solve_puzzle(img_descrition: list[dict]) -> None:
    """Solve the puzzle using A* search algorithm."""
    start_grid_state, orig_grid_state = initialize_grid(img_descrition)

    goal_mismatch = 0  # We want zero mismatches at the end
    solution = a_star_search(start_grid_state, goal_mismatch, GRID_SIZE)

    if solution:
        print("Solution found!")
        for row in solution[-1]:
            for subimg in row:
                print(subimg.edges)
        return solution, orig_grid_state
    else:
        print("No solution found.")

def rotate_image(img: np.ndarray, number_of_rotations: int) -> np.ndarray:
    """Rotate inserted image by 0"""
    return imutils.rotate(img, (number_of_rotations % 4) * 90) 
    

def place_rotated_image(result_image: np.ndarray, rotated_image: np.ndarray, 
                        row_num: int, col_num: int, n_rows: int, n_cols: int) -> None:
    """Place the rotated sub-image onto the result image canvas."""
    img_row_start = round(row_num * (n_rows - 2) + row_num)
    img_row_stop = round((row_num + 1) * (n_rows - 2) + (row_num + 2))
    
    img_col_start = round(col_num * (n_cols - 2) + col_num)
    img_col_stop = round((col_num + 1) * (n_cols - 2) + (col_num + 2))
    
    result_image[img_row_start:img_row_stop, img_col_start:img_col_stop] = rotated_image
    
    return rotated_image

def reconstruct_orig_image(orig_grid_state: list[list[np.ndarray]], 
                            solution: list[list[SubImage]]) -> np.ndarray:
    """
    Reconstructs the original image from the solution grid.

    Args:
        orig_grid_state (list[list[np.ndarray]]): The original grid state containing sub-images.
        solution (list[list[SubImage]]): The solution grid with arranged sub-images.

    Returns:
        np.ndarray: The reconstructed original image.
    """
    # number of pixels in row and column
    n_rows = len(orig_grid_state[0][0])
    n_cols = len(orig_grid_state[0][0][0])
    result_image = np.ones((n_rows * GRID_SIZE, 
                             n_cols * GRID_SIZE, 3), dtype=np.uint8)
    for row_num, row in enumerate(solution):
        for col_num, sub_img in enumerate(row):
            # location of image in the original state
            x_pos, y_pos = sub_img.orig_location
            orig_sub_img = orig_grid_state[x_pos][y_pos]
            rotated_image = rotate_image(orig_sub_img, sub_img.rotation)
            place_rotated_image(result_image, rotated_image, 
                                row_num, col_num, n_rows, n_cols)
    return result_image
def main():
    original_image = cv.imread(IMG_PATH)
    original_image_bw = cv.imread(IMG_PATH, cv.IMREAD_GRAYSCALE)
    
    img_descrition = get_color_shape(original_image, original_image_bw)
    solution, orig_grid_state = solve_puzzle(img_descrition)

    result_img = reconstruct_orig_image(orig_grid_state, solution[-1])
    cv.imshow('Final solution', result_img)
    print(f'The final solution was dislayed. Press a key to see the folding process.')
    key = cv.waitKey(0)
    for num, img in enumerate(solution):
        res_img = reconstruct_orig_image(orig_grid_state, img)
        cv.imshow('Animated steps', res_img)
        key = cv.waitKey(0)
        print(f'Displayed img {num + 1}/{len(solution)}')
        
    cv.imwrite('solution.png', result_img)
    
if __name__ == "__main__":
    main()

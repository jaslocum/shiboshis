from multiprocessing.spawn import freeze_support
from PIL import Image, ImageOps
Image.MAX_IMAGE_PIXELS = None
from multiprocessing import Array, Process, Queue, cpu_count
import sys
import os
import random

# Change these 3 config parameters to suit your needs...
TILE_SIZE = 64  # height/width of mosaic tiles in pixels
# tile matching resolution (higher values give better fit but require more processing)
TILE_MATCH_RES = 16
# the mosaic image will be this many times wider and taller than the original
ENLARGEMENT = 16
# percentage of all potential tiles to sample per each get_best_fit_tile attempt
BEST_FIT_TILE_SAMPLE_PERCENT = .2
FINAL_FIT_TILE_SAMPLE_PERCENT = .1
# percent chance of tile changing between frames (0 to 100)
CHANGE_CHANCE = 25
# starting point in tile array
# grid of 80 * 125 = 10,000 (poster)
# grid of 100 * 100 = 10,000 (square)
# set radius's to 0 to disable an onion render (1-3)
START_SQUARES = 0
START_SQUARES_ARRAY = [[0]*START_SQUARES for i in range(3)]
# start square array:
#  element1 = start at x pos
#  element2 = start at y pos
#  element3 = layers to center of onion
START_SQUARES_ARRAY[0] = [40, 61, 33]
# START_SQUARES_ARRAY[1] = [22, 87, 16]
# START_SQUARES_ARRAY[2] = [55, 87, 16]
# type of fit for get_best_fit_tile selection mode
BEST_FIT = 1
FINAL_FIT = 3
RANDOM_FIT = 2
TILE_BLOCK_SIZE = TILE_SIZE / max(min(TILE_MATCH_RES, TILE_SIZE), 1)
# WORKER_COUNT = max(cpu_count() - 1, 1)
WORKER_COUNT = 1
OUT_FILE = 'mosaic'
EOQ_VALUE = None
# GLOBAL TILE DATA
TILES_DATA = None
TILES_USED = None
TILES_DIRECTORY = None
MOSAIC = None
MAX_FRAMES = 16

class TileProcessor:    
    def __process_tile(self, tile_path):
        try:
            img = Image.open(tile_path)
            img = ImageOps.exif_transpose(img)

            # tiles must be square, so get the largest square that fits inside the image
            w = img.size[0]
            h = img.size[1]
            min_dimension = min(w, h)
            w_crop = (w - min_dimension) / 2
            h_crop = (h - min_dimension) / 2
            img = img.crop((w_crop, h_crop, w - w_crop, h - h_crop))

            large_tile_img = img.resize(
                (TILE_SIZE, TILE_SIZE), Image.ANTIALIAS)
            small_tile_img = img.resize(
                (int(TILE_SIZE/TILE_BLOCK_SIZE), int(TILE_SIZE/TILE_BLOCK_SIZE)), Image.ANTIALIAS)

            return (large_tile_img.convert('RGB'), small_tile_img.convert('RGB'))
        except:
            return (None, None)

    def get_tiles(self, tiles_directory):
        large_tiles = []
        small_tiles = []
        # search the tiles directory recursively
        for root, subFolders, files in os.walk(tiles_directory):
            for tile_name in files:
                print('get_tiles: Reading {:40.40}'.format(tile_name), flush=True, end='\r')
                tile_path = os.path.join(root, tile_name)
                large_tile, small_tile = self.__process_tile(tile_path)
                if large_tile:
                    large_tiles.append(large_tile)
                    small_tiles.append(small_tile)
        # print('get_tiles: Processed {} tiles.'.format(len(large_tiles)))
        return (large_tiles, small_tiles)


class TargetImage1:
    def __init__(self, image1_path):
        self.image1_path = image1_path

    def get_data(self):
        print('Processing main image1...')
        img = Image.open(self.image1_path)
        w = img.size[0] * ENLARGEMENT
        h = img.size[1] * ENLARGEMENT
        large_img = img.resize((w, h), Image.ANTIALIAS)
        w_diff = (w % TILE_SIZE)/2
        h_diff = (h % TILE_SIZE)/2

        # if necessary, crop the image slightly so we use a whole number of tiles horizontally and vertically
        if w_diff or h_diff:
            large_img = large_img.crop(
                (w_diff, h_diff, w - w_diff, h - h_diff))

        small_img = large_img.resize(
            (int(w/TILE_BLOCK_SIZE), int(h/TILE_BLOCK_SIZE)), Image.ANTIALIAS)

        image_data = (large_img.convert('RGB'), small_img.convert('RGB'))

        print('Main image processed.')

        return image_data

class TargetImage2:
    def __init__(self, image2_path):
        self.image2_path = image2_path

    def get_data(self):
        print('Processing main image2...')
        img = Image.open(self.image2_path)
        w = img.size[0] * ENLARGEMENT
        h = img.size[1] * ENLARGEMENT
        large_img = img.resize((w, h), Image.ANTIALIAS)
        w_diff = (w % TILE_SIZE)/2
        h_diff = (h % TILE_SIZE)/2

        # if necessary, crop the image slightly so we use a whole number of tiles horizontally and vertically
        if w_diff or h_diff:
            large_img = large_img.crop(
                (w_diff, h_diff, w - w_diff, h - h_diff))

        small_img = large_img.resize(
            (int(w/TILE_BLOCK_SIZE), int(h/TILE_BLOCK_SIZE)), Image.ANTIALIAS)

        image_data = (large_img.convert('RGB'), small_img.convert('RGB'))

        print('Main image processed.')

        return image_data


class TileFitter:
    def __init__(self, tiles_data, tiles_used):
        self.tiles_used = tiles_used
        self.tiles_data = tiles_data

    def __lock_tile(self, tile_index, x, y):
        locked_tile = False
        if self.tiles_used[tile_index] == [-1, -1]:
            self.tiles_used[tile_index] = [x, y]
            locked_tile = True
        return locked_tile

    def __unlock_tile(self, tile_index):
        self.tiles_used[tile_index] = [-1, -1]

    def __get_tile_diff(self, t1, t2, bail_out_value):
        diff = 0
        for i in range(len(t1)):
            # diff += (abs(t1[i][0] - t2[i][0]) + abs(t1[i][1] - t2[i][1]) + abs(t1[i][2] - t2[i][2]))
            diff += ((t1[i][0] - t2[i][0])**2 + (t1[i][1] -
                     t2[i][1])**2 + (t1[i][2] - t2[i][2])**2)
            if diff > bail_out_value:
                # we know already that this isn't going to be the best fit, so no point continuing with this tile
                return diff
        return diff

    def get_best_fit_tile(self, img_data, fit_mode, x, y):
        best_fit_tile_index = None
        min_diff = sys.maxsize
        tile_index = 0
        max_tries = 1
        len_tiles_data = len(self.tiles_data)
        if fit_mode == RANDOM_FIT:
            # only try once so best fit will likely not be found, just a random tile will be selected
            max_tries = 2
        else:
            if fit_mode == BEST_FIT:
                tile_sample_percent = BEST_FIT_TILE_SAMPLE_PERCENT
            else:
                tile_sample_percent = FINAL_FIT_TILE_SAMPLE_PERCENT
            if len_tiles_data > (tile_sample_percent * len_tiles_data):
                max_tries = round(len_tiles_data*tile_sample_percent)
        # skip set of tiles examined to provide even error to matching of tiles
        trys = 1
        # go through each tile in turn looking for the best match for the part of the image represented by 'img_data'
        # if we are at the end of remaining tiles, the return best fit so far          
        while trys < max_tries:
            tile_index = random.randint(0, len_tiles_data - 1)
            # lock tile if not already locked
            if self.__lock_tile(tile_index, x, y):
                tile_data = self.tiles_data[tile_index]
                diff = self.__get_tile_diff(img_data, tile_data, min_diff)
                if diff < min_diff:
                    # unlock tile that was a contender to be used by other threads while searching for best fits
                    if best_fit_tile_index != None:
                        self.__unlock_tile(best_fit_tile_index)
                    min_diff = diff
                    best_fit_tile_index = tile_index
                else:
                    # unlock tile to be used by other threads while searching for best fits
                    self.__unlock_tile(tile_index)
            trys += 1
        if best_fit_tile_index == None:
            tile_index = 0
            while tile_index < len_tiles_data:
                # lock tile if not already locked
                if self.__lock_tile(tile_index, x, y):
                    tile_data = self.tiles_data[tile_index]
                    diff = self.__get_tile_diff(img_data, tile_data, min_diff)
                    if diff < min_diff:
                        # unlock tile that was a contender to be used by other threads while searching for best fits
                        if best_fit_tile_index != None:
                            self.__unlock_tile(best_fit_tile_index)
                        min_diff = diff
                        best_fit_tile_index = tile_index
                    else:
                        # unlock tile to be used by other threads while searching for best fits
                        self.__unlock_tile(tile_index)
                    if fit_mode == RANDOM_FIT:
                        tile_index = len_tiles_data
                tile_index += 1
        return best_fit_tile_index

def img_data_is_empty (img_data):
    image_data_is_empty = True
    for i in range(len(img_data)):
        for j in range(3):
            if img_data[i][j] != 0:
                image_data_is_empty = False
                break
        if image_data_is_empty == False:
            break
    return image_data_is_empty

class ProgressCounter:
    def __init__(self, total):
        self.total = total
        self.counter = 0

    def update(self, frame):
        self.counter += 1
        print("Progress: {:04.1f}%".format(
            100 * self.counter / self.total)+' frame #'+str(frame), flush=True, end='\r')


class MosaicImage:
    def __init__(self, original_img):
        self.image = Image.new(original_img.mode, original_img.size)
        self.x_tile_count = int(original_img.size[0] / TILE_SIZE)
        self.y_tile_count = int(original_img.size[1] / TILE_SIZE)
        self.total_tiles = self.x_tile_count * self.y_tile_count
        self.tiles_assigned = [[1 for y in range(self.y_tile_count)] for x in range(self.x_tile_count)]
        self.initialized = True

    def reset_tile_assigned(self, x, y):
        self.tiles_assigned[x][y] = 1
        return True

    def set_tile_assigned(self, x, y):
        self.tiles_assigned[x][y] = 0
        return True

    def get_tile_assigned(self, x, y):
        if self.tiles_assigned[x][y] == 1:
            return False
        else:
            return True

    def add_tile(self, tile_data, coords):
        img = Image.new('RGB', (TILE_SIZE, TILE_SIZE))
        img.putdata(tile_data)
        self.image.paste(img, coords)

    def save(self, path):
        self.image.save(path)


def fit_tiles(mosaic, tile_fitter, img_data, img_coords, x, y, fit_mode):
    # this function gets run by the worker processes, one on each CPU core
    # img_data, img_coords, x, y, fit_mode
    if not mosaic.get_tile_assigned(x, y):
        if not img_data_is_empty(img_data):
            tile_index = tile_fitter.get_best_fit_tile(img_data, fit_mode, x, y)
            if tile_index != None:
                mosaic.set_tile_assigned(x, y)
                build_mosaic(mosaic, tile_fitter, img_coords, tile_index)

def build_mosaic(mosaic, tile_fitter, img_coords, tile_index):
    tile_data = tile_fitter.tiles_data[tile_index]
    mosaic.add_tile(tile_data, img_coords)

def compose(original1_img, oringinal2_img, oringinal3_img, oringinal4_img, tiles):
    print('compose: press Ctrl-C to abort...')
    original_img1_large, original_img1_small = original1_img
    if oringinal2_img == None:
        original_img2_large = None
        original_img2_small = None
    else:
        original_img2_large, original_img2_small = oringinal2_img
    if oringinal3_img == None:
        original_img3_large = None
        original_img3_small = None
    else:
        original_img3_large, original_img3_small = oringinal3_img
    if oringinal4_img == None:
        original_img4_large = None
        original_img4_small = None
    else:
        original_img4_large, original_img4_small = oringinal4_img
    tiles_large, tiles_small = tiles

    frame = 0
    MOSAIC = MosaicImage(original_img1_large)
    x_tile_count = MOSAIC.x_tile_count
    y_tile_count = MOSAIC.y_tile_count
    TILES_DATA = [list(tile.getdata()) for tile in tiles_large]
    TILES_USED = list([[-1, -1] for tile in tiles_large])
    tile_fitter = TileFitter(TILES_DATA, TILES_USED)
    try:
        progress = ProgressCounter(MOSAIC.x_tile_count * MOSAIC.y_tile_count * 2 * MAX_FRAMES)

        # generate MAX_FRAMES frames
        while frame < MAX_FRAMES:

            # render best match images left to right and top to bottom
            # transition from image (1 and 2) to (3 and 4)
            # images 2 and 4 are the images to match first for best matching
            # before matching master images 1 and 3
            render_first_image_small = None
            if frame > 0:
                # frames 1 thru n
                if original_img4_small != None:
                    render_first_image_small = original_img4_small
                else:
                    if original_img2_small != None:
                        render_first_image_small = original_img2_small
            else:
                # render first frame (0)
                if original_img2_small != None:
                    render_first_image_small = original_img2_small

            if render_first_image_small != None:
                for y in range(MOSAIC.y_tile_count):
                    for x in range(MOSAIC.x_tile_count):
                        next_tile(MOSAIC, tile_fitter, progress, render_first_image_small,
                            x, y,  x_tile_count, y_tile_count, BEST_FIT, frame)

            # render best match images left to right and top to bottom
            # transition from image (1 and 2) to (3 and 4)
            # images 2 and 4 are the images to match first for best matching
            # before matching master images 1 and 3
            render_master_image_small = None
            if frame > 0:
                # frames 1 thru n
                if original_img3_small != None:
                    render_master_image_small = original_img3_small
                else:
                    if original_img1_small != None:
                        render_master_image_small = original_img1_small
            else:
                # render first frame (0)
                if original_img1_small != None:
                    render_master_image_small = original_img1_small

            # render defined start squares first
            current_start_square = 0
            while current_start_square < START_SQUARES and render_master_image_small != None:
                # square onion layer (mayber circular someday...)
                onion_layer = 0
                start_x = START_SQUARES_ARRAY[current_start_square][0]
                start_y = START_SQUARES_ARRAY[current_start_square][1]
                onion_layers = START_SQUARES_ARRAY[current_start_square][2]
                while onion_layer <= onion_layers:
                    if onion_layer > 0:
                        # north side of onion
                        y = start_y - onion_layer
                        for x in range((start_x - onion_layer), (start_x + onion_layer)):
                            next_tile(MOSAIC, tile_fitter, progress, render_master_image_small,
                                    x, y, x_tile_count, y_tile_count, BEST_FIT, frame)
                        # east side of onion
                        x = start_x + onion_layer
                        for y in range((start_y - onion_layer), (start_y + onion_layer + 1)):
                            next_tile(MOSAIC, tile_fitter, progress, render_master_image_small,
                                      x, y,  x_tile_count, y_tile_count, BEST_FIT, frame)
                        # south side of onion
                        y = start_y + onion_layer
                        for x in range((start_x - onion_layer), (start_x + onion_layer)):
                            next_tile(MOSAIC, tile_fitter, progress, render_master_image_small,
                                      x, y,  x_tile_count, y_tile_count, BEST_FIT, frame)
                        # west side of onion
                        x = start_x - onion_layer
                        for y in range((start_y - onion_layer), (start_y + onion_layer)):
                            next_tile(MOSAIC, tile_fitter, progress, render_master_image_small,
                                      x, y,  x_tile_count, y_tile_count, BEST_FIT, frame)
                    else:
                        next_tile(MOSAIC, tile_fitter, progress, render_master_image_small,
                                  start_x, start_y,  x_tile_count, y_tile_count, BEST_FIT, frame)
                    onion_layer += 1
                current_start_square += 1

            # render rest of image left to right and top to bottom
            for y in range(MOSAIC.y_tile_count):
                for x in range(MOSAIC.x_tile_count):
                    next_tile(MOSAIC, tile_fitter, progress, render_master_image_small,
                              x, y,  x_tile_count, y_tile_count, FINAL_FIT, frame)

            MOSAIC.save(OUT_FILE+str(frame)+'.jpg')
            # print('\nframe, output is in', OUT_FILE+str(FRAME)+'.jpg')
            frame = frame + 1
            release_tiles_used(TILES_USED, MOSAIC)

    except KeyboardInterrupt:
        print('\nHalting, please wait...')
        pass

def release_tiles_used(TILES_USED, MOSAIC):
    tile_index = 0
    for y in range(MOSAIC.y_tile_count):
        for x in range(MOSAIC.x_tile_count):
            if random.randint(0, 100) < CHANGE_CHANCE and tile_index < len(TILES_USED):
                MOSAIC.reset_tile_assigned(
                    TILES_USED[tile_index][0], TILES_USED[tile_index][1])
                TILES_USED[tile_index] = [-1, -1]
            tile_index += 1    

def next_tile(MOSAIC, tile_fitter, progress, original_img_small, x, y, x_tile_count, y_tile_count, fit_mode, frame):
    if  x >= 0 and x < x_tile_count:
        if y >= 0 and y < y_tile_count:
            large_box = (x * TILE_SIZE, y * TILE_SIZE, (x + 1)
                        * TILE_SIZE, (y + 1) * TILE_SIZE)
            small_box = (x * TILE_SIZE/TILE_BLOCK_SIZE, y * TILE_SIZE/TILE_BLOCK_SIZE,
                        (x + 1) * TILE_SIZE/TILE_BLOCK_SIZE, (y + 1) * TILE_SIZE/TILE_BLOCK_SIZE)
            img_data = list(original_img_small.crop(small_box).getdata())
            fit_tiles(MOSAIC, tile_fitter, img_data, large_box, x, y, fit_mode)
            progress.update(frame)

def mosaic(img1_path, img2_path, img3_path, img4_path, tiles_path):
    TILES_DIRECTORY = tiles_path
    image1_data = TargetImage1(img1_path).get_data()
    if img2_path == None:
        image2_data = None
    else:
        image2_data = TargetImage2(img2_path).get_data()
    if img3_path == None:
        image3_data = None
    else:
        image3_data = TargetImage2(img2_path).get_data()
    if img4_path == None:
        image4_data = None
    else:
        image4_data = TargetImage2(img2_path).get_data()
    data_tiles = TileProcessor().get_tiles(TILES_DIRECTORY)
    compose(image1_data, image2_data, image3_data, image4_data, data_tiles)

if __name__ == '__main__':
    freeze_support()
    if len(sys.argv) == 6:
        mosaic(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        if len(sys.argv) == 5:
            mosaic(sys.argv[1], sys.argv[2], sys.argv[3], None, sys.argv[4])
            if len(sys.argv) == 4:
                mosaic(sys.argv[1], sys.argv[2], None, None, sys.argv[4])
            else:
                mosaic(sys.argv[1], None, None, None, sys.argv[2])
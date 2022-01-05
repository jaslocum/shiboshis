from multiprocessing.spawn import freeze_support
from PIL import Image, ImageOps
Image.MAX_IMAGE_PIXELS = None
from multiprocessing import Process, Queue, cpu_count
import sys
import os
import random

# Change these 3 config parameters to suit your needs...
TILE_SIZE = 128  # height/width of mosaic tiles in pixels
# tile matching resolution (higher values give better fit but require more processing)
TILE_MATCH_RES = 4
# the mosaic image will be this many times wider and taller than the original
ENLARGEMENT = 32
# percentage of all potential tiles to sample per each get_best_fit_tile attempt
TILE_SAMPLE_PERCENT = .01
# surprise stop percentage chance
SURPRISE_STOP = .10

TILE_BLOCK_SIZE = TILE_SIZE / max(min(TILE_MATCH_RES, TILE_SIZE), 1)
# WORKER_COUNT = max(cpu_count() - 1, 1)
WORKER_COUNT = 1
OUT_FILE = 'mosaic.jpeg'
EOQ_VALUE = None


class TileProcessor:
    def __init__(self, tiles_directory):
        self.tiles_directory = tiles_directory

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

    def get_tiles(self):
        large_tiles = []
        small_tiles = []
        # search the tiles directory recursively
        for root, subFolders, files in os.walk(self.tiles_directory):
            for tile_name in files:
                print('get_tiles: Reading {:40.40}'.format(
                    tile_name), flush=True, end='\r')
                tile_path = os.path.join(root, tile_name)
                large_tile, small_tile = self.__process_tile(tile_path)
                if large_tile:
                    large_tiles.append(large_tile)
                    small_tiles.append(small_tile)
        # print('get_tiles: Processed {} tiles.'.format(len(large_tiles)))
        return (large_tiles, small_tiles)


class TargetImage:
    def __init__(self, image_path):
        self.image_path = image_path

    def get_data(self):
        print('Processing main image...')
        img = Image.open(self.image_path)
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
        self.tiles_data = tiles_data
        self.tiles_used = tiles_used

    def __lock_tile(self, tile_index):
        locked_tile = False
        if self.tiles_used[tile_index] == 1:
            self.tiles_used[tile_index] = 0
            locked_tile = True
        return locked_tile

    def __unlock_tile(self, tile_index):
        self.tiles_used[tile_index] = 1

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

    def get_best_fit_tile(self, img_data):
        best_fit_tile_index = None
        min_diff = sys.maxsize
        tile_index = 0
        max_tries = 1
        len_tiles_data = len(self.tiles_data)
        if len_tiles_data > TILE_SAMPLE_PERCENT:
            max_tries = round(len_tiles_data*TILE_SAMPLE_PERCENT)
        # skip set of tiles examined to provide even error to matching of tiles
        trys = 1
        # go through each tile in turn looking for the best match for the part of the image represented by 'img_data'
        # if we are at the end of remaining tiles, the return best fit so far
        while trys < max_tries:
            tile_index = random.randint(0, len_tiles_data - 1)
            if random.random() < SURPRISE_STOP:
                trys = max_tries
            # lock tile if not already locked
            if self.__lock_tile(tile_index):
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
                if self.__lock_tile(tile_index):
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
                tile_index += 1
        return best_fit_tile_index


def fit_tiles(work_queue, result_queue, tiles_data, tiles_used):
    # this function gets run by the worker processes, one on each CPU core
    tile_fitter = TileFitter(tiles_data, tiles_used)
    while True:
        try:
            img_data, img_coords = work_queue.get(True)
            if img_data == EOQ_VALUE:
                break
            tile_index = tile_fitter.get_best_fit_tile(img_data)
            if tile_index != None:
                result_queue.put((img_coords, tile_index))
            else:
                break
        except KeyboardInterrupt:
            pass
    # let the result handler know that this worker has finished everything
    result_queue.put((EOQ_VALUE, EOQ_VALUE))


class ProgressCounter:
    def __init__(self, total):
        self.total = total
        self.counter = 0

    def update(self):
        self.counter += 1
        print("Progress: {:04.1f}%".format(
            100 * self.counter / self.total), flush=True, end='\r')


class MosaicImage:
    def __init__(self, original_img):
        self.image = Image.new(original_img.mode, original_img.size)
        self.x_tile_count = int(original_img.size[0] / TILE_SIZE)
        self.y_tile_count = int(original_img.size[1] / TILE_SIZE)
        self.total_tiles = self.x_tile_count * self.y_tile_count

    def add_tile(self, tile_data, coords):
        img = Image.new('RGB', (TILE_SIZE, TILE_SIZE))
        img.putdata(tile_data)
        self.image.paste(img, coords)

    def save(self, path):
        self.image.save(path)


def build_mosaic(result_queue, original_img_large, tiles_data):
    mosaic = MosaicImage(original_img_large)
    active_workers = WORKER_COUNT
    while True:
        try:
            img_coords, best_fit_tile_index = result_queue.get()
            if img_coords == EOQ_VALUE:
                active_workers -= 1
                if not active_workers:
                    break
            else:
                tile_data = tiles_data[best_fit_tile_index]
                mosaic.add_tile(tile_data, img_coords)
        except KeyboardInterrupt:
            pass

    mosaic.save(OUT_FILE)
    print('\nFinished, output is in', OUT_FILE)

def compose(original_img, tiles):
    print('compose: press Ctrl-C to abort...')
    original_img_large, original_img_small = original_img
    tiles_large, tiles_small = tiles
    fit_process = []

    mosaic = MosaicImage(original_img_large)
    work_queue = Queue(WORKER_COUNT)
    result_queue = Queue()
    tiles_data = [list(tile.getdata()) for tile in tiles_large]
    all_tile_data_small = [list(tile.getdata()) for tile in tiles_small]
    TILES_USED = list([1 for tile in tiles_large])
    try:
        # start the worker processes that will build the mosaic image
        Process(target=build_mosaic, args=(
            result_queue, original_img_large, tiles_data)).start()

        # start the worker processes that will perform the tile fitting
        for n in range(WORKER_COUNT):
            Process(target=fit_tiles, args=(work_queue, result_queue, tiles_data, TILES_USED)).start()

        progress = ProgressCounter(mosaic.x_tile_count * mosaic.y_tile_count)
        for x in range(mosaic.x_tile_count):
            for y in range(mosaic.y_tile_count):
                large_box = (x * TILE_SIZE, y * TILE_SIZE, (x + 1)
                            * TILE_SIZE, (y + 1) * TILE_SIZE)
                small_box = (x * TILE_SIZE/TILE_BLOCK_SIZE, y * TILE_SIZE/TILE_BLOCK_SIZE,
                            (x + 1) * TILE_SIZE/TILE_BLOCK_SIZE, (y + 1) * TILE_SIZE/TILE_BLOCK_SIZE)
                work_queue.put(
                    (list(original_img_small.crop(small_box).getdata()), large_box))
                progress.update()

    except KeyboardInterrupt:
        print('\nHalting, please wait...')
        pass

    finally:
        # put these special values onto the queue to let the workers know they can terminate
        for n in range(WORKER_COUNT):
            work_queue.put((EOQ_VALUE, EOQ_VALUE))


def mosaic(img_path, tiles_path):
    global OUT_FILE
    image_data = TargetImage(img_path).get_data()
    data_tiles = TileProcessor(tiles_path).get_tiles()
    compose(image_data, data_tiles)


if __name__ == '__main__':
    freeze_support()
    if len(sys.argv) < 3:
        print('Usage: {} <image> <tiles directory>\r'.format(sys.argv[0]))
        mosaic('./shibaswap-icon.ee749b42(400x400).png',
               './imagesCopy/')
    else:
        mosaic(sys.argv[1], sys.argv[2])

# mosaic

This utility can be used to generate [photo-mosaic](http://en.wikipedia.org/wiki/Photographic_mosaic) images, to use it you must have Python installed, along with the [Pillow](http://pillow.readthedocs.org/en/latest/) imaging library.

Run the utility from the command line, as follows:

<pre>python mosaic.py &lt;image_finish&gt; &lt;[image_start]&gt; &lt;tiles directory&gt;
</pre>

*   The `image_finish` argument should contain the path to the image for which you want to build the mosaic
*   The `image_finish` argument should contain the path to the image for which you want render first with best fit (optional)
*   The `tiles directory` argument should contain the path to the directory containing the tile images (the directory will be searched recursively, so it doesn't matter if some of the images are contained in sub-directories)

Multi processing has been removed from original code for simplicity
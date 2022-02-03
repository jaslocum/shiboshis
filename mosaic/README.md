# mosaic

This utility can be used to generate [photo-mosaic](http://en.wikipedia.org/wiki/Photographic_mosaic) images, to use it you must have Python installed, along with the [Pillow](http://pillow.readthedocs.org/en/latest/) imaging library.

Run the utility from the command line, as follows:

<pre>python mosaic.py &lt;image_finish&gt; [&lt;image_start&gt;] &lt;tiles directory&gt;
</pre>

*   The `image_finish` argument should contain the path to the image for which you want to build the mosaic
*   The `image_start` argument should contain the path to the image for which you want render first with best fit (optional) - leave areas to not be fitted with null values (0,0,0)
*   The `tiles directory` argument should contain the path to the directory containing the tile images (the directory will be searched recursively, so it doesn't matter if some of the images are contained in sub-directories)

Multi processing has been removed from original code for simplicity

Happy Valentine's season #ShibArmy, and much love to our shiboshi community.  This is my final #Shiboshi #mosaic in a three part series. All 10K OG Shiboshis occur once in this mosaic. I think it looks pretty cool at a distance, or you can zoom in to make out each individual shiboshi.

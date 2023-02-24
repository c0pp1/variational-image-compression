import tensorflow as tf

if __name__ == '__main__':

    # Define the URL to download the dataset
    url = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'

    # Define the file name for the downloaded dataset
    filename = 'stl10_binary.tar.gz'

    # Download the dataset
    tf.keras.utils.get_file(
        fname=filename,
        origin=url,
        extract=True,
        cache_dir='.',
        cache_subdir='stl10'
    )
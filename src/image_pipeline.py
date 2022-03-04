class ImagePipeline:
    # This class handles image pipeline between .svs / .tif and Tensorflow
    ImagePipeline():
        batch = False
        training = False
        origin_directory = None
        destination_directory = None

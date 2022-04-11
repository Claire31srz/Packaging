# Part 0: Libraries
#import 'requirements.txt'
import argparse
import logging
import os

from LL.LLA import *

# Part 1: Parser
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int,
                    help="Number of epochs needed to train the model")
parser.add_argument("--batch_size", type=int,
                    help="Size of the batch needed to divid the epochs into it")
parser.add_argument("--visualization_size", type=int,
                    help="Square Root of the number of predicted images to display, to evaluate model efficiency")
parser.add_argument("--output", type=str,
                    help="path to output file")

args = parser.parse_args()

# Part 2: Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler("myscript_LLA.log")
c_handler.setLevel(logging.DEBUG)
f_handler.setLevel(logging.DEBUG)
c_format = logging.Formatter('%(name)s - [%(levelname)s] - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)
logger.addHandler(c_handler)
logger.addHandler(f_handler)

#Part 3: Errors
class Error(Exception):
    pass

class IncorrectOutputPath(Error):
    pass

class IncorrectCreatedParameters(Error):
    pass

class IncorrectModelParameters(Error):
    pass


# Part 4: Handling errors
    
def folder(inputfile, model):
    if os.path.exists(inputfile):
        model_filename = 'autoencoder_cifar10.json'
        weight_filename = 'autoencoder_cifar10_weights.h5'
        json_str = model.to_json()
        model_path = os.path.join(OUTDIR, model_filename)
        weight_path = os.path.join(OUTDIR, weight_filename)
        with open(model_path, "w") as txtfile:
            txtfile.write(json_str)
        model.save_weights(weight_path)
    else:
        raise IncorrectOutputPath ('Impossible to find the folder {}. Please enter a folder that exists.'.format(inputfile))
    
def parameter(n_epochs, batch_size, visualization_size):
    if isinstance(n_epochs, int):
        if isinstance(batch_size, int) & (batch_size > 0):
            n_images = visualization_size*visualization_size
            if isinstance(n_images, int) & (n_images > 0):
                model_autoencoder = fit_model_on_cifar10(n_epochs = n_epochs, batch_size = batch_size, visualization_size = visualization_size)
            else: raise IncorrectCreatedParameters('The parameter n_images = {} should be a positive integer. Please enter correct values for the visualization size.'.format(visualization_size))
        else: raise IncorrectModelParameters('The parameter Batch_size = {} should be positive integer. Please enter a correct value.'.format(batch_size))
    else: raise IncorrectModelParameters('The parameter n_epochs = {} should be positive integer. Please enter a correct value.'.format(n_epochs))
    return model_autoencoder

if __name__ == '__main__':
    logger.info("Execution starting. No error detected.")
    try:
        autoencoder = parameter(n_epochs = args.n_epochs, 
                                batch_size = args.batch_size, 
                                visualization_size = args.visualization_size)
        logger.info('The training is done. No error detected.')
    except (IncorrectModelParameters, IncorrectCreatedParameters) as error:
        logger.info('Error detected.')
        logger.warning(str(error))
    try:
        folder(model=model_autoencoder, intputfile = args.output)
        logger.info('Models and weights have been saved. No error detected.')
    except IncorrectOutputPath as error:
        logger.info('Error detected.')
        logger.warning(str(error))
    logger.info('Execution done. No error detected.')
                                                        

    
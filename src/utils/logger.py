import logging

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler to log to a file
    fh = logging.FileHandler('app.log')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

# # Example usage
# logger = setup_logger('my_logger')
# logger.info('This is an info message')
# logger.error('This is an error message')

# # Now you can read the log file content
# with open('app.log', 'r') as file:
#     print(file.read())  # This will print the content of the log file

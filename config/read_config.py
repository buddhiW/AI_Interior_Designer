import configparser

def read_config(config_file):

    # Create a ConfigParser object
    config = configparser.ConfigParser()
    
    # Read the configuration file
    config.read(config_file)
    
    # Accessing the parameters from the config file
    rag_path = config.get('Paths', 'rag_path')
    
    chunk_size = config.getint('RAG_params', 'chunk_size')
    chunk_overlap = config.getint('RAG_params', 'chunk_overlap')

    model_name = config.get('model', 'model_name')
    temperature = config.getfloat('model', 'temperature')
    
    # Return the parameters as a dictionary or assign them to variables
    return {
        'rag_path': rag_path,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'model_name': model_name,
        'temperature': temperature
    }
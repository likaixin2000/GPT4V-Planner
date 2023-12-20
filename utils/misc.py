def detect_environment():
    try:
        # This function is only available in Jupyter environment
        cfg = get_ipython().config 
        if 'IPKernelApp' in cfg.keys():
            return "Jupyter Notebook"
        else:
            return "Unknown"
    except NameError:
        return "Python Script"
def parse_config(config):
    if "tune_scheduler" not in config.keys():
        config["tune_scheduler"] = None
    else:
        config["tune_scheduler"] = eval(config["tune_scheduler"])
    if "tune_algo" not in config.keys():
        config["tune_algo"] = None
    else:
        config["tune_algo"] = eval(config["tune_algo"])
    if "tune_searcher" not in config.keys():
        config["tune_searcher"] = None
    else:
        config["tune_searcher"] = eval(config["tune_searcher"])
    return config


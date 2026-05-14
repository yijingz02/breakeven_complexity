import logging
from ruamel.yaml import YAML


class YParams():
    """ Yaml file parser """

    def __init__(self, yaml_filename=None, config_name=None, print_params=False):
        self._yaml_filename = yaml_filename
        self._config_name = config_name
        self.params = {}

        if print_params:
            print("------------------ Configuration ------------------")

        if yaml_filename:
            with open(yaml_filename) as _file:
                
                if config_name is None:
                    params = YAML().load(_file)
                else:
                    params = YAML().load(_file)[config_name]
                
                for key, val in params.items():
                    if print_params: 
                        print(key, val)
                    if val == 'None': 
                        val = None
                    self.params[key] = val
                    self.__setattr__(key, val)

        if print_params:
            print("---------------------------------------------------")

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, val):
        self.params[key] = val
        self.__setattr__(key, val)

    def __contains__(self, key):
        return (key in self.params)

    def update_params(self, config):
        for key, val in config.items():
            self.params[key] = val
            self.__setattr__(key, val)

    def log(self):
        logging.info("------------------ Configuration ------------------")
        logging.info("Configuration file: "+str(self._yaml_filename))
        logging.info("Configuration name: "+str(self._config_name))
        for key, val in self.params.items():
            logging.info(str(key) + ' ' + str(val))
        logging.info("---------------------------------------------------")
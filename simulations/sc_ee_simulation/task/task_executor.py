import json
from collections import OrderedDict
from simulations.sc_ee_simulation.task.image_id_generator import ImageIdGenerator
from simulations.sc_ee_simulation.common.strategy import Strategy


class TaskExecutor:
    def __init__(self, layer_data_path, output_data_path, ae_data_path, image_id_gen: ImageIdGenerator, strategy: Strategy):
        # with open(layer_data_path) as f:
        #     self.layer_data = json.load(f, object_pairs_hook=OrderedDict)
        # with open(output_data_path) as f:
        #     self.output_data = json.load(f, object_pairs_hook=OrderedDict)
        # with open(ae_data_path) as f:
        #     self.ae_data = json.load(f, object_pairs_hook=OrderedDict)
        self.layer_data = layer_data_path
        self.output_data = output_data_path
        self.ae_data = ae_data_path
        self.image_id_gen = image_id_gen
        self.strategy = strategy

    def execute_model(self):
        image_id = self.image_id_gen.get_id()
        on_device = True

        ed = 0
        es = 0
        size = 0
        prediction = -1
        cur_exit = 4

        for i in self.layer_data:
            if i["layer_type"] == 7:
                if i["split_idx"] == self.strategy.split:
                    on_device = False
                    if self.strategy.autoencoder != -1 and self.strategy.split != 0 and self.strategy.split != 21:
                        ed += self.ae_data[str(i["split_idx"])][str(self.strategy.autoencoder)]["encoder"]
                        es += self.ae_data[str(i["split_idx"])][str(self.strategy.autoencoder)]["decoder"]
                        size = self.ae_data[str(i["split_idx"])][str(self.strategy.autoencoder)]["size"]
            else:
                if self.strategy.ignore_ee and i["layer_type"] == 4 and not str(self.strategy.exit) in i["name"]:
                    continue
                if on_device:
                    ed += i["macs"]
                    if i["layer_type"] != 4 and self.strategy.autoencoder == -1:
                        size = i["size"]
                else:
                    es += i["macs"]

                if i["layer_type"] == 4:
                    exists = self.strategy.autoencoder != -1 and self.strategy.split != 0 and self.strategy.split != 21 and i["name"] in self.output_data[str(image_id)][str(self.strategy.split)][str(self.strategy.autoencoder)]
                    max_value = self.output_data[str(image_id)][i["name"]]["max"]
                    if exists:
                        max_value = self.output_data[str(image_id)][str(self.strategy.split)][str(self.strategy.autoencoder)][i["name"]]["max"]
                    if not self.strategy.ignore_ee and max_value > self.strategy.thresholds[i["name"]]:
                        cur_exit = 1 if i["name"] == "EE 1" else 2 if i["name"] == "EE 2" else 3
                        if exists:
                            prediction = self.output_data[str(image_id)][str(self.strategy.split)][str(self.strategy.autoencoder)][i["name"]]["argmax"]
                        else:
                            prediction = self.output_data[str(image_id)][i["name"]]["argmax"]
                        break
                    if self.strategy.ignore_ee and str(self.strategy.exit) in i["name"]:
                        cur_exit = self.strategy.exit
                        if on_device:
                            size = 0
                        if exists:
                            prediction = self.output_data[str(image_id)][str(self.strategy.split)][str(self.strategy.autoencoder)][i["name"]]["argmax"]
                        else:
                            prediction = self.output_data[str(image_id)][i["name"]]["argmax"]
                        break
        if prediction == -1:
            exists = self.strategy.autoencoder != -1 and self.strategy.split != 0 and self.strategy.split != 21 and "main" in self.output_data[str(image_id)][str(self.strategy.split)][str(self.strategy.autoencoder)]
            if exists:
                prediction = self.output_data[str(image_id)][str(self.strategy.split)][str(self.strategy.autoencoder)]["main"]["argmax"]
            else:
                prediction = self.output_data[str(image_id)]["main"]["argmax"]

        # print("ed: {}, es: {}, size: {}, prediction: {}".format(ed, es, size, prediction))
        return ed, es, size, prediction, cur_exit, self.output_data[str(image_id)]["target"]
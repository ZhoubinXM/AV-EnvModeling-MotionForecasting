import os, sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
import multiprocessing
import yaml
import argparse
from multiprocessing import Process
from data_processor import DataPorcessor
from tqdm import tqdm
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Config file with dataset parameters", default="./configs/tnt_config.yml")
args = parser.parse_args()

if __name__ == "__main__":
    with open(args.config, 'r') as yaml_file:
        cfg = yaml.safe_load(yaml_file)

    files = []
    for _, _, fs in os.walk(cfg['dataset']['datapath']):
        files.extend([
            os.path.join(cfg['dataset']['datapath'], file) for file in fs if file.endswith("csv") and not file.startswith('.')
        ])

    # # for test
    # files = files[:10]

    pbar = tqdm(total=len(files))
    queue = multiprocessing.Queue(cfg['dataset']['num_workers'])

    def exec(queue):
        data_processor = DataPorcessor(cfg)
        while True:
            file = queue.get()
            if file is None:
                break
            df = pd.read_csv(file, low_memory=False)
            for i in range(0, 4):
                df = df.rename(
                    columns={
                        "l_line_C0_" + str(i) + ".1": "l_line_C0_1" + str(i),
                        "l_line_C1_" + str(i) + ".1": "l_line_C1_1" + str(i),
                        "l_line_C2_" + str(i) + ".1": "l_line_C2_1" + str(i),
                        "l_line_C3_" + str(i) + ".1": "l_line_C3_1" + str(i)
                    })
            if "l_age_128" not in df.columns:  # file invalid
                continue
            for id, row in enumerate(df.itertuples()):
                if id % cfg['dataset']['down_sample_rate'] == 0:
                    data_processor.process(row)
            # data_processor.generate_sample()

    processes = [Process(target=exec, args=(queue, )) for _ in range(cfg['dataset']['num_workers'])]
    for each in processes:
        each.start()

    for file in files:
        assert file is not None
        queue.put(file)
        pbar.update(1)

    while not queue.empty():
        pass
    pbar.close()
    print("data process success!")

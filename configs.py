import yaml

with open("configs.yml", 'r') as ymlfile:
    cfg = yaml.full_load(ymlfile)

if __name__ == "__main__":
    print(cfg)
